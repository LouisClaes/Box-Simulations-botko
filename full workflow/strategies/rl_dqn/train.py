"""
DDQN training script for 3D online bin packing.

Complete training loop implementing:
  - Double DQN with candidate-based action space
  - Prioritised experience replay with n-step returns
  - Linear epsilon decay with configurable schedule
  - Periodic target network synchronisation
  - Evaluation with no exploration at regular intervals
  - Checkpointing and resume support
  - Comprehensive logging (CSV + TensorBoard + plots)

Usage:
    # Default Botko BV setup
    python train.py

    # Custom hyperparameters
    python train.py --episodes 50000 --batch_size 256 --lr 0.001 --gamma 0.95

    # Resume from checkpoint
    python train.py --resume outputs/rl_dqn/checkpoints/ep_10000.pt

    # Quick test run
    python train.py --episodes 100 --eval_interval 50 --log_interval 10

HPC deployment:
    Transfer the entire strategies/ folder + simulator/ + config.py.
    The script is self-contained with no external dependencies beyond
    PyTorch, NumPy, and the project's own modules.

References:
    - van Hasselt et al. (2016): Deep RL with Double Q-learning
    - Schaul et al. (2016): Prioritized Experience Replay
    - Tsang et al. (2025): DDQN for dual-bin packing
"""

from __future__ import annotations

import sys
import os
import argparse
import time
import json
import copy
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────
_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from config import Box, BinConfig, Orientation
from simulator.bin_state import BinState
from simulator.session import PackingSession, SessionConfig
from simulator.close_policy import HeightClosePolicy

from strategies.rl_common.environment import BinPackingEnv, EnvConfig, generate_random_boxes
from strategies.rl_common.rewards import RewardShaper, RewardConfig
from strategies.rl_common.logger import TrainingLogger
from strategies.rl_common.obs_utils import encode_heightmap, encode_box_features

from strategies.rl_dqn.config import DQNConfig
from strategies.rl_dqn.network import DQNNetwork
from strategies.rl_dqn.replay_buffer import (
    Transition,
    UniformReplayBuffer,
    PrioritisedReplayBuffer,
    NStepBuffer,
)
from strategies.rl_dqn.candidate_generator import CandidateGenerator, Candidate


# ─────────────────────────────────────────────────────────────────────────────
# Training Agent
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double DQN agent with candidate-based action selection.

    Manages:
      - Online and target networks
      - Epsilon-greedy exploration
      - Replay buffer (PER with n-step)
      - Gradient updates with DDQN targets
      - Target network synchronisation

    The agent operates on pre-generated candidate lists rather than
    the full grid action space, making training tractable.
    """

    def __init__(self, config: DQNConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

        # ── Networks ──────────────────────────────────────────────────────
        self.online_net = DQNNetwork(config).to(device)
        self.target_net = DQNNetwork(config).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # ── Optimiser ─────────────────────────────────────────────────────
        self.optimiser = optim.Adam(
            self.online_net.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # ── Replay buffer ─────────────────────────────────────────────────
        hm_shape = (config.num_bins, config.grid_l, config.grid_w)
        box_dim = config.box_feature_dim

        if config.buffer_alpha > 0:
            base_buffer = PrioritisedReplayBuffer(
                capacity=config.buffer_capacity,
                hm_shape=hm_shape,
                box_dim=box_dim,
                max_candidates=config.max_candidates,
                alpha=config.buffer_alpha,
                beta_start=config.buffer_beta_start,
                beta_end=config.buffer_beta_end,
                beta_frames=config.num_episodes * 50,  # ~steps
            )
        else:
            base_buffer = UniformReplayBuffer(
                capacity=config.buffer_capacity,
                hm_shape=hm_shape,
                box_dim=box_dim,
                max_candidates=config.max_candidates,
            )

        if config.n_step > 1:
            self.buffer = NStepBuffer(base_buffer, n=config.n_step, gamma=config.gamma)
        else:
            self.buffer = base_buffer

        # ── Candidate generator ───────────────────────────────────────────
        bin_config = BinConfig(
            length=config.bin_length,
            width=config.bin_width,
            height=config.bin_height,
            resolution=config.resolution,
        )
        self.candidate_gen = CandidateGenerator(
            bin_config=bin_config,
            num_bins=config.num_bins,
            max_candidates=config.max_candidates,
            use_corner_positions=config.use_corner_positions,
            use_extreme_points=config.use_extreme_points,
            use_ems_positions=config.use_ems_positions,
            use_grid_fallback=config.use_grid_fallback,
            grid_step=config.grid_fallback_step,
            num_orientations=config.num_orientations,
        )

        # ── Counters ──────────────────────────────────────────────────────
        self.global_step = 0
        self.episode = 0

    @property
    def epsilon(self) -> float:
        """Current epsilon (linearly decayed)."""
        cfg = self.config
        if self.episode >= cfg.eps_decay_episodes:
            return cfg.eps_end
        frac = self.episode / max(cfg.eps_decay_episodes, 1)
        return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

    def select_action(
        self,
        heightmaps: np.ndarray,
        box_features: np.ndarray,
        candidates: List[Candidate],
        candidate_features: np.ndarray,
        explore: bool = True,
    ) -> Tuple[int, Candidate]:
        """
        Select an action from the candidate list.

        With probability epsilon, selects a random candidate (exploration).
        Otherwise, evaluates Q-values for all candidates and picks the best.

        Args:
            heightmaps:         (num_bins, grid_l, grid_w) normalised.
            box_features:       (box_feature_dim,) normalised.
            candidates:         List of Candidate objects.
            candidate_features: (K, 7) feature array.
            explore:            Enable epsilon-greedy exploration.

        Returns:
            (index, candidate): Index into candidates list and the selected Candidate.
        """
        if not candidates:
            raise ValueError("No candidates available for action selection")

        # Epsilon-greedy exploration
        if explore and np.random.random() < self.epsilon:
            idx = np.random.randint(len(candidates))
            return idx, candidates[idx]

        # Greedy: evaluate Q-values for all candidates
        self.online_net.eval()
        with torch.no_grad():
            hm_t = torch.from_numpy(heightmaps).unsqueeze(0).to(self.device)
            box_t = torch.from_numpy(box_features).unsqueeze(0).to(self.device)
            act_t = torch.from_numpy(candidate_features).to(self.device)

            q_values = self.online_net.forward_batch_candidates(hm_t, box_t, act_t)
            idx = int(q_values.argmax(dim=0).item())

        return idx, candidates[idx]

    def store_transition(self, transition: Transition) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.add(transition)
        self.global_step += 1

    def train_step(self) -> Optional[float]:
        """
        Perform one gradient update from replay buffer.

        Implements Double DQN target computation:
          a* = argmax_a Q_online(s', a)
          target = r + gamma^n * Q_target(s', a*) * (1 - done)

        Returns:
            Loss value, or None if buffer too small.
        """
        cfg = self.config

        if len(self.buffer) < cfg.min_buffer_size:
            return None

        # Sample batch
        batch = self.buffer.sample(cfg.batch_size)

        # Convert to tensors
        state_hm = torch.from_numpy(batch.state_hm).to(self.device)
        state_box = torch.from_numpy(batch.state_box).to(self.device)
        action_feat = torch.from_numpy(batch.action_feat).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.device)
        next_state_hm = torch.from_numpy(batch.next_state_hm).to(self.device)
        next_state_box = torch.from_numpy(batch.next_state_box).to(self.device)
        dones = torch.from_numpy(batch.dones.astype(np.float32)).to(self.device)
        weights = torch.from_numpy(batch.weights).to(self.device)

        # ── Current Q-values ──────────────────────────────────────────────
        self.online_net.train()
        q_current = self.online_net(state_hm, state_box, action_feat).squeeze(1)

        # ── Double DQN target ─────────────────────────────────────────────
        # For each sample in the batch, find the best next action using the
        # ONLINE network, then evaluate it with the TARGET network.
        with torch.no_grad():
            next_q_values = torch.zeros(cfg.batch_size, device=self.device)

            for i in range(cfg.batch_size):
                if dones[i]:
                    continue

                n_cand = int(batch.next_num_candidates[i])
                if n_cand == 0:
                    continue

                # Get candidate features for this sample's next state
                cand_feats = torch.from_numpy(
                    batch.next_candidates[i, :n_cand]
                ).to(self.device)

                # Next state (expand for batch candidates)
                ns_hm = next_state_hm[i:i+1]
                ns_box = next_state_box[i:i+1]

                # Online network selects best action
                q_online = self.online_net.forward_batch_candidates(
                    ns_hm, ns_box, cand_feats,
                ).squeeze(1)
                best_idx = q_online.argmax().item()

                # Target network evaluates that action
                best_feat = cand_feats[best_idx:best_idx+1]
                q_target = self.target_net(ns_hm, ns_box, best_feat).squeeze(1)
                next_q_values[i] = q_target.item()

        # Compute discount factor (account for n-step)
        gamma_n = cfg.gamma ** cfg.n_step
        targets = rewards + gamma_n * next_q_values * (1.0 - dones)

        # ── Loss and update ───────────────────────────────────────────────
        td_errors = q_current - targets
        # Huber loss (smooth L1) with PER weights
        loss = (weights * nn.functional.smooth_l1_loss(
            q_current, targets, reduction="none"
        )).mean()

        self.optimiser.zero_grad()
        loss.backward()

        # Gradient clipping
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), cfg.grad_clip)

        self.optimiser.step()

        # ── Update priorities ─────────────────────────────────────────────
        td_np = td_errors.detach().cpu().numpy()
        self.buffer.update_priorities(batch.indices, np.abs(td_np))

        return float(loss.item())

    def sync_target(self) -> None:
        """Synchronise target network with online network."""
        cfg = self.config
        if cfg.tau >= 1.0:
            # Hard copy
            self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            # Soft (Polyak) update
            for target_param, online_param in zip(
                self.target_net.parameters(), self.online_net.parameters()
            ):
                target_param.data.copy_(
                    cfg.tau * online_param.data + (1.0 - cfg.tau) * target_param.data
                )

    def save_checkpoint(self, path: str) -> None:
        """Save full training state to a checkpoint file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "config": self.config.to_dict(),
            "episode": self.episode,
            "global_step": self.global_step,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Restore training state from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.episode = checkpoint.get("episode", 0)
        self.global_step = checkpoint.get("global_step", 0)


# ─────────────────────────────────────────────────────────────────────────────
# Environment helper: build observation tensors from session
# ─────────────────────────────────────────────────────────────────────────────

def build_state_tensors(
    bin_states: List[BinState],
    grippable: List[Box],
    bin_config: BinConfig,
    pick_window: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build heightmap and box feature arrays from session observation.

    Args:
        bin_states: Current bin states.
        grippable:  Grippable boxes.
        bin_config: Bin configuration.
        pick_window: Number of grippable box slots.

    Returns:
        heightmaps:  (num_bins, grid_l, grid_w) float32
        box_features: (pick_window * 5,) float32
    """
    num_bins = len(bin_states)

    # Heightmaps
    heightmaps = np.zeros(
        (num_bins, bin_config.grid_l, bin_config.grid_w), dtype=np.float32,
    )
    for i, bs in enumerate(bin_states):
        heightmaps[i] = bs.heightmap.astype(np.float32) / max(bin_config.height, 1.0)

    # Box features
    box_feats = np.zeros(pick_window * 5, dtype=np.float32)
    max_dim = max(bin_config.length, bin_config.width, bin_config.height)
    max_vol = max(bin_config.volume, 1.0)

    for i, box in enumerate(grippable[:pick_window]):
        offset = i * 5
        box_feats[offset + 0] = box.length / max_dim
        box_feats[offset + 1] = box.width / max_dim
        box_feats[offset + 2] = box.height / max_dim
        box_feats[offset + 3] = box.volume / max_vol
        box_feats[offset + 4] = box.weight / 50.0

    return heightmaps, box_feats


# ─────────────────────────────────────────────────────────────────────────────
# Training episode
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    agent: DQNAgent,
    session: PackingSession,
    boxes: List[Box],
    bin_config: BinConfig,
    reward_shaper: RewardShaper,
    explore: bool = True,
    train: bool = True,
) -> Dict[str, float]:
    """
    Run a single training/evaluation episode.

    Steps:
      1. Reset session with new boxes
      2. For each step:
         a. Build state tensors from observation
         b. Generate candidates for the current box
         c. Select action (epsilon-greedy or greedy)
         d. Execute action via session.step()
         e. Compute reward
         f. Store transition (if training)
         g. Train on batch from replay (if training)
      3. Return episode metrics

    Args:
        agent:         DQNAgent instance.
        session:       PackingSession instance.
        boxes:         List of boxes for this episode.
        bin_config:    Bin configuration.
        reward_shaper: Reward computation.
        explore:       Enable exploration.
        train:         Enable training updates.

    Returns:
        Dict with episode metrics (reward, fill, loss, placements, etc.)
    """
    cfg = agent.config
    obs = session.reset(boxes, strategy_name="rl_dqn")

    total_reward = 0.0
    total_loss = 0.0
    loss_count = 0
    placements = 0
    rejections = 0
    steps = 0
    prev_fill_rates = [0.0] * cfg.num_bins

    while not obs.done:
        grippable = obs.grippable
        if not grippable:
            # No grippable boxes — advance conveyor
            session.advance_conveyor()
            obs = session.observe()
            continue

        # Pick the first grippable box (FIFO)
        box = grippable[0]

        # Build state tensors
        heightmaps, box_features = build_state_tensors(
            obs.bin_states, grippable, bin_config, cfg.pick_window,
        )

        # Generate candidates
        candidates, candidate_features = agent.candidate_gen.generate(
            box, obs.bin_states,
        )

        if not candidates:
            # No valid candidates — advance conveyor
            session.advance_conveyor()
            rejections += 1
            total_reward += reward_shaper.rejection_penalty()
            obs = session.observe()
            continue

        # Select action
        action_idx, selected = agent.select_action(
            heightmaps, box_features, candidates, candidate_features,
            explore=explore,
        )

        # Execute placement
        step_result = session.step(
            box.id,
            selected.bin_idx,
            selected.x,
            selected.y,
            selected.orient_idx,
        )

        # Compute reward
        if step_result.placed:
            placements += 1
            new_obs = session.observe()
            new_fills = [bs.get_fill_rate() for bs in new_obs.bin_states]
            fill_delta = new_fills[selected.bin_idx] - prev_fill_rates[selected.bin_idx]

            reward = reward_shaper.placement_reward(
                box=box,
                bin_state=new_obs.bin_states[selected.bin_idx],
                bin_config=bin_config,
                fill_delta=fill_delta,
                pallet_closed=step_result.pallet_closed,
                closed_fill=(
                    step_result.closed_pallet_result.fill_rate
                    if step_result.pallet_closed else 0.0
                ),
            )
            prev_fill_rates = new_fills
        else:
            rejections += 1
            reward = reward_shaper.rejection_penalty()
            new_obs = session.observe()

        total_reward += reward

        # Build next-state tensors for transition storage
        next_hm, next_box_feats = build_state_tensors(
            new_obs.bin_states,
            new_obs.grippable if new_obs.grippable else [],
            bin_config,
            cfg.pick_window,
        )

        # Generate next-state candidates for Double DQN
        next_candidates_feat = np.zeros(
            (cfg.max_candidates, cfg.action_feature_dim), dtype=np.float32,
        )
        next_n_cand = 0
        if not new_obs.done and new_obs.grippable:
            next_box = new_obs.grippable[0]
            next_cands, next_cand_feats = agent.candidate_gen.generate(
                next_box, new_obs.bin_states,
            )
            next_n_cand = len(next_cands)
            if next_n_cand > 0:
                n = min(next_n_cand, cfg.max_candidates)
                next_candidates_feat[:n] = next_cand_feats[:n]
                next_n_cand = n

        # Store transition
        if train:
            transition = Transition(
                state_hm=heightmaps,
                state_box=box_features,
                action_feat=candidate_features[action_idx],
                reward=reward,
                next_state_hm=next_hm,
                next_state_box=next_box_feats,
                done=new_obs.done,
                next_candidates=next_candidates_feat,
                next_num_candidates=next_n_cand,
            )
            agent.store_transition(transition)

            # Train on batch
            loss = agent.train_step()
            if loss is not None:
                total_loss += loss
                loss_count += 1

            # Sync target network periodically
            if agent.global_step % cfg.target_update_freq == 0:
                agent.sync_target()

        obs = new_obs
        steps += 1

    # Flush n-step buffer at episode end
    if train and hasattr(agent.buffer, 'flush'):
        agent.buffer.flush()

    # Terminal reward
    result = session.result()
    terminal_reward = reward_shaper.terminal_reward(
        avg_fill=result.avg_closed_fill,
        pallets_closed=result.pallets_closed,
        placement_rate=result.placement_rate,
    )
    total_reward += terminal_reward

    avg_loss = total_loss / max(loss_count, 1)

    return {
        "reward": total_reward,
        "fill": result.avg_closed_fill,
        "pallets_closed": result.pallets_closed,
        "placement_rate": result.placement_rate,
        "placements": placements,
        "rejections": rejections,
        "steps": steps,
        "loss": avg_loss,
        "epsilon": agent.epsilon,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config: DQNConfig) -> None:
    """
    Main training loop.

    Runs for config.num_episodes, with periodic evaluation, checkpointing,
    and logging.  Supports resume from checkpoint.
    """
    # ── Setup ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training for {config.num_episodes} episodes")
    print(f"Output: {config.output_dir}")

    # Directories
    output_dir = os.path.abspath(config.output_dir)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Bin config
    bin_config = BinConfig(
        length=config.bin_length,
        width=config.bin_width,
        height=config.bin_height,
        resolution=config.resolution,
    )

    # Session config
    session_config = SessionConfig(
        bin_config=bin_config,
        num_bins=config.num_bins,
        buffer_size=config.buffer_size,
        pick_window=config.pick_window,
        close_policy=HeightClosePolicy(max_height=config.close_height),
        max_consecutive_rejects=10,
        enable_stability=False,
        allow_all_orientations=(config.num_orientations >= 6),
    )

    # Agent
    agent = DQNAgent(config, device)
    print(f"Network parameters: {agent.online_net.count_parameters():,}")

    # Resume from checkpoint
    start_episode = 0
    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
        print(f"Resuming from: {config.checkpoint_path}")
        agent.load_checkpoint(config.checkpoint_path)
        start_episode = agent.episode
        print(f"Resumed at episode {start_episode}")

    # Session
    session = PackingSession(session_config)

    # Reward shaper
    reward_shaper = RewardShaper(RewardConfig())

    # Logger
    logger = TrainingLogger(
        log_dir=os.path.join(output_dir, "logs"),
        strategy_name="rl_dqn",
        use_tensorboard=True,
    )
    logger.log_config(config.to_dict())

    # RNG
    rng = np.random.default_rng(42)

    # ── Best tracking ─────────────────────────────────────────────────────
    best_eval_fill = 0.0

    # ── Training loop ─────────────────────────────────────────────────────
    t_start = time.time()

    for ep in range(start_episode, config.num_episodes):
        agent.episode = ep

        # Generate random boxes for this episode
        boxes = generate_random_boxes(
            n=config.num_boxes_per_episode,
            size_range=config.box_size_range,
            weight_range=config.box_weight_range,
            rng=rng,
        )

        # Run training episode
        metrics = run_episode(
            agent=agent,
            session=session,
            boxes=boxes,
            bin_config=bin_config,
            reward_shaper=reward_shaper,
            explore=True,
            train=True,
        )

        # Log
        logger.log_episode(ep, **metrics)

        # Console progress
        if (ep + 1) % config.log_interval == 0:
            logger.print_progress(
                ep + 1, config.num_episodes,
                reward=metrics["reward"],
                fill=metrics["fill"],
                loss=metrics["loss"],
                eps=metrics["epsilon"],
                buf=len(agent.buffer),
            )

        # ── Evaluation ────────────────────────────────────────────────────
        if (ep + 1) % config.eval_interval == 0:
            eval_fills = []
            eval_rewards = []
            eval_rng = np.random.default_rng(12345)

            for eval_ep in range(config.eval_episodes):
                eval_boxes = generate_random_boxes(
                    n=config.num_boxes_per_episode,
                    size_range=config.box_size_range,
                    weight_range=config.box_weight_range,
                    rng=eval_rng,
                )
                eval_metrics = run_episode(
                    agent=agent,
                    session=session,
                    boxes=eval_boxes,
                    bin_config=bin_config,
                    reward_shaper=reward_shaper,
                    explore=False,
                    train=False,
                )
                eval_fills.append(eval_metrics["fill"])
                eval_rewards.append(eval_metrics["reward"])

            avg_eval_fill = float(np.mean(eval_fills))
            std_eval_fill = float(np.std(eval_fills))
            avg_eval_reward = float(np.mean(eval_rewards))

            logger.log_episode(
                ep,
                eval_fill=avg_eval_fill,
                eval_fill_std=std_eval_fill,
                eval_reward=avg_eval_reward,
            )

            elapsed = time.time() - t_start
            print(
                f"\n  [EVAL] Ep {ep+1} | "
                f"Fill: {avg_eval_fill:.4f} +/- {std_eval_fill:.4f} | "
                f"Reward: {avg_eval_reward:.2f} | "
                f"Best: {best_eval_fill:.4f} | "
                f"Time: {elapsed:.0f}s\n",
                flush=True,
            )

            # Save best model
            if avg_eval_fill > best_eval_fill:
                best_eval_fill = avg_eval_fill
                best_path = os.path.join(ckpt_dir, "best_model.pt")
                agent.save_checkpoint(best_path)
                # Also save just the network weights for inference
                agent.online_net.save(os.path.join(ckpt_dir, "best_network.pt"))
                print(f"  [BEST] New best model saved: fill={best_eval_fill:.4f}")

        # ── Checkpoint ────────────────────────────────────────────────────
        if (ep + 1) % config.checkpoint_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ep_{ep+1:06d}.pt")
            agent.save_checkpoint(ckpt_path)

    # ── Final ─────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s ({total_time/3600:.1f}h)")
    print(f"Best eval fill: {best_eval_fill:.4f}")

    # Save final model
    final_path = os.path.join(ckpt_dir, "final_model.pt")
    agent.save_checkpoint(final_path)
    agent.online_net.save(os.path.join(ckpt_dir, "final_network.pt"))

    # Generate training curves
    curves_path = logger.plot_training_curves(save=True, show=False)
    if curves_path:
        print(f"Training curves saved to: {curves_path}")

    logger.close()
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> DQNConfig:
    """Parse command-line arguments into a DQNConfig."""
    parser = argparse.ArgumentParser(
        description="Train Double DQN agent for 3D online bin packing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training
    parser.add_argument("--episodes", type=int, default=50_000, help="Total training episodes")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Max gradient norm")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 regularisation")

    # Exploration
    parser.add_argument("--eps_start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--eps_decay_fraction", type=float, default=0.8, help="Epsilon decay schedule")

    # Replay
    parser.add_argument("--buffer_capacity", type=int, default=500_000, help="Replay buffer size")
    parser.add_argument("--buffer_alpha", type=float, default=0.6, help="PER alpha (0=uniform)")
    parser.add_argument("--n_step", type=int, default=3, help="N-step returns")
    parser.add_argument("--min_buffer_size", type=int, default=1000, help="Min buffer before training")

    # Network
    parser.add_argument("--use_dueling", action="store_true", default=True, help="Dueling architecture")
    parser.add_argument("--no_dueling", dest="use_dueling", action="store_false")
    parser.add_argument("--use_batch_norm", action="store_true", default=True, help="Batch norm in CNN")

    # Target
    parser.add_argument("--target_update_freq", type=int, default=500, help="Target sync interval (steps)")
    parser.add_argument("--tau", type=float, default=1.0, help="Soft update coefficient (1=hard)")

    # Candidates
    parser.add_argument("--max_candidates", type=int, default=200, help="Max candidates per step")
    parser.add_argument("--grid_fallback_step", type=float, default=100.0, help="Grid fallback step (mm)")

    # Episode
    parser.add_argument("--num_boxes_per_episode", type=int, default=100, help="Boxes per episode")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Checkpoint interval")
    parser.add_argument("--log_interval", type=int, default=100, help="Console log interval")

    # Paths
    parser.add_argument("--output_dir", type=str, default="outputs/rl_dqn", help="Output directory")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")

    args = parser.parse_args()

    config = DQNConfig(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_fraction=args.eps_decay_fraction,
        buffer_capacity=args.buffer_capacity,
        buffer_alpha=args.buffer_alpha,
        n_step=args.n_step,
        min_buffer_size=args.min_buffer_size,
        use_dueling=args.use_dueling,
        use_batch_norm=args.use_batch_norm,
        target_update_freq=args.target_update_freq,
        tau=args.tau,
        max_candidates=args.max_candidates,
        grid_fallback_step=args.grid_fallback_step,
        num_boxes_per_episode=args.num_boxes_per_episode,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
        checkpoint_path=args.resume,
    )

    return config


if __name__ == "__main__":
    config = parse_args()
    train(config)
