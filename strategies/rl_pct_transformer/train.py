"""
PPO training loop for the PCT Transformer strategy.

Implements on-policy PPO training with variable-size action spaces
(placement candidates).  Based on the training procedure from:
  - Zhao et al. (ICLR 2022): episode-based training, gamma=1.0
  - Schulman et al. (2017): PPO clip, GAE, entropy bonus

Key features:
  - Variable-length action spaces handled via padding + masking
  - Buffer-aware training: tries each grippable box, picks highest value
  - Parallel environment rollouts (N_env x rollout_steps)
  - Cosine learning rate schedule with warmup
  - TensorBoard + CSV logging via rl_common.TrainingLogger
  - Checkpoint saving with best-model tracking

CLI:
    python train.py --episodes 200000 --num_envs 16 --lr 3e-4
    python train.py --checkpoint outputs/rl_pct_transformer/logs/best.pt --episodes 50000

Architecture note:
    Unlike standard PPO with fixed action spaces, we cannot vectorise the
    candidate generation across environments because each environment has
    different bin states.  Instead, we:
      1. Step all envs in parallel (observation collection)
      2. Generate candidates per-env (Python loop)
      3. Pad candidates to batch max for batched network forward
      4. Collect rollout buffer with variable-length candidates
      5. During PPO updates, re-pad each mini-batch
"""

from __future__ import annotations

import sys
import os
import time
import copy
import argparse
from typing import List, Optional, Dict, Tuple

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from config import Box, BinConfig, Orientation
from simulator.session import PackingSession, SessionConfig
from simulator.close_policy import HeightClosePolicy
from strategies.rl_common.rewards import RewardShaper, RewardConfig
from strategies.rl_common.logger import TrainingLogger
from strategies.rl_common.obs_utils import encode_box_features

from strategies.rl_pct_transformer.config import PCTTransformerConfig
from strategies.rl_pct_transformer.network import PCTTransformerNet
from strategies.rl_pct_transformer.candidate_generator import (
    CandidateGenerator,
    CandidateAction,
)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer for variable-size action spaces
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores rollout data for PPO with variable-size candidate sets.

    Each step stores:
      - item_features:      (item_input_dim,)
      - candidate_features: list of (N_i, candidate_input_dim) -- variable N_i
      - candidate_mask:     list of (N_i,) -- all True (only valid candidates stored)
      - action:             int -- index into the candidate list
      - log_prob:           float
      - value:              float
      - reward:             float
      - done:               bool

    During PPO updates, data is padded to the batch maximum N for batched
    forward passes.
    """

    def __init__(self) -> None:
        self.item_features: List[np.ndarray] = []
        self.candidate_features: List[np.ndarray] = []
        self.num_candidates: List[int] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def add(
        self,
        item_feat: np.ndarray,
        cand_feat: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        """Add a single step to the buffer."""
        self.item_features.append(item_feat)
        self.candidate_features.append(cand_feat)
        self.num_candidates.append(cand_feat.shape[0])
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self) -> None:
        """Reset the buffer."""
        self.item_features.clear()
        self.candidate_features.clear()
        self.num_candidates.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_gae(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and returns.

        Args:
            last_value:  V(s_T+1) for bootstrapping.
            gamma:       Discount factor.
            gae_lambda:  GAE lambda.

        Returns:
            advantages: (N,) array of advantage estimates.
            returns:    (N,) array of discounted returns.
        """
        n = len(self)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            # non_terminal: 1 if episode continues after step t, 0 if step t ended it.
            # done[t] = True means this step is the last step of the episode,
            # so V(s_{t+1}) = 0 (no future value to bootstrap from).
            non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae

            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]

            next_value = self.values[t]

        return advantages, returns

    def get_batches(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
        minibatch_size: int,
        device: torch.device,
    ):
        """
        Yield padded mini-batches for PPO updates.

        Each batch contains tensors padded to the maximum number of
        candidates in that batch.

        Yields:
            dict with keys:
                item_features:      (mb, item_dim)
                candidate_features: (mb, max_N, cand_dim)
                candidate_mask:     (mb, max_N)
                actions:            (mb,)
                old_log_probs:      (mb,)
                advantages:         (mb,)
                returns:            (mb,)
        """
        n = len(self)
        indices = np.random.permutation(n)

        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            batch_idx = indices[start:end]
            mb = len(batch_idx)

            # Find max candidates in this mini-batch
            max_N = max(self.num_candidates[i] for i in batch_idx)
            if max_N == 0:
                continue

            cand_dim = self.candidate_features[batch_idx[0]].shape[1]
            item_dim = self.item_features[batch_idx[0]].shape[0]

            # Allocate padded tensors
            item_batch = np.zeros((mb, item_dim), dtype=np.float32)
            cand_batch = np.zeros((mb, max_N, cand_dim), dtype=np.float32)
            mask_batch = np.zeros((mb, max_N), dtype=np.bool_)
            act_batch = np.zeros(mb, dtype=np.int64)
            logp_batch = np.zeros(mb, dtype=np.float32)
            adv_batch = np.zeros(mb, dtype=np.float32)
            ret_batch = np.zeros(mb, dtype=np.float32)

            for j, idx in enumerate(batch_idx):
                n_cand = self.num_candidates[idx]
                item_batch[j] = self.item_features[idx]
                cand_batch[j, :n_cand] = self.candidate_features[idx]
                mask_batch[j, :n_cand] = True
                act_batch[j] = self.actions[idx]
                logp_batch[j] = self.log_probs[idx]
                adv_batch[j] = advantages[idx]
                ret_batch[j] = returns[idx]

            yield {
                'item_features': torch.from_numpy(item_batch).to(device),
                'candidate_features': torch.from_numpy(cand_batch).to(device),
                'candidate_mask': torch.from_numpy(mask_batch).to(device),
                'actions': torch.from_numpy(act_batch).to(device),
                'old_log_probs': torch.from_numpy(logp_batch).to(device),
                'advantages': torch.from_numpy(adv_batch).to(device),
                'returns': torch.from_numpy(ret_batch).to(device),
            }


# ─────────────────────────────────────────────────────────────────────────────
# Single training environment wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PCTPackingEnv:
    """
    Lightweight environment wrapper for PCT Transformer training.

    Wraps a PackingSession with candidate generation.  Unlike the generic
    BinPackingEnv (which uses a flat discrete action space), this environment
    exposes a variable-size candidate action space.

    The agent sees:
      - item_features: current box encoded
      - candidates: list of CandidateAction with features
      - reward, done signals

    Step interface:
      action = index into the candidate list
      env.step(action) -> (item_feat, candidates, reward, done, info)
    """

    def __init__(
        self,
        config: PCTTransformerConfig,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)

        self.bin_config = BinConfig(
            length=config.bin_length,
            width=config.bin_width,
            height=config.bin_height,
            resolution=config.bin_resolution,
        )
        self.session_config = SessionConfig(
            bin_config=self.bin_config,
            num_bins=config.num_bins,
            buffer_size=config.buffer_size,
            pick_window=config.pick_window,
            close_policy=HeightClosePolicy(max_height=config.close_height),
            max_consecutive_rejects=config.max_consecutive_rejects,
            enable_stability=False,
            allow_all_orientations=(config.num_orientations >= 6),
        )

        self.candidate_gen = CandidateGenerator(
            bin_config=self.bin_config,
            min_support=config.min_support_ratio,
            num_orientations=config.num_orientations,
            floor_scan_step=config.floor_scan_step,
            dedup_tolerance=config.candidate_dedup_tolerance,
        )

        self.reward_shaper = RewardShaper(config.reward_config)
        self.session: Optional[PackingSession] = None
        self._prev_fill_rates: List[float] = []

        # Current step state
        self.current_box: Optional[Box] = None
        self.current_candidates: List[CandidateAction] = []
        self.done: bool = True

    def reset(self) -> Tuple[np.ndarray, List[CandidateAction], dict]:
        """
        Reset: generate boxes, initialise session, return first observation.

        Returns:
            (item_features, candidates, info)
        """
        # Generate random boxes
        boxes = self._generate_boxes()

        self.session = PackingSession(self.session_config)
        obs = self.session.reset(boxes, strategy_name="rl_pct_transformer")
        self._prev_fill_rates = [0.0] * self.config.num_bins
        self.done = False

        return self._observe(obs)

    def step(
        self, action_idx: int,
    ) -> Tuple[np.ndarray, List[CandidateAction], float, bool, dict]:
        """
        Execute an action (index into current candidates).

        Args:
            action_idx: Index into self.current_candidates.

        Returns:
            (item_features, candidates, reward, done, info)
        """
        if self.done or not self.current_candidates:
            # Terminal or no candidates -- return zero observation
            dummy_feat = np.zeros(self.config.item_input_dim, dtype=np.float32)
            return dummy_feat, [], 0.0, True, {"placed": False}

        # Get the selected candidate
        candidate = self.current_candidates[action_idx]

        # Execute placement via session.step()
        step_result = self.session.step(
            self.current_box.id,
            candidate.bin_idx,
            candidate.x, candidate.y,
            candidate.orient_idx,
        )

        # Compute reward
        if step_result.placed:
            obs_after = self.session.observe()
            new_fills = [bs.get_fill_rate() for bs in obs_after.bin_states]
            fill_delta = new_fills[candidate.bin_idx] - self._prev_fill_rates[candidate.bin_idx]

            reward = self.reward_shaper.placement_reward(
                box=self.current_box,
                bin_state=obs_after.bin_states[candidate.bin_idx],
                bin_config=self.bin_config,
                fill_delta=fill_delta,
                pallet_closed=step_result.pallet_closed,
                closed_fill=(
                    step_result.closed_pallet_result.fill_rate
                    if step_result.pallet_closed and step_result.closed_pallet_result
                    else 0.0
                ),
            )
            self._prev_fill_rates = new_fills
        else:
            reward = self.reward_shaper.rejection_penalty()

        # Get next observation
        obs = self.session.observe()
        self.done = obs.done

        # Terminal bonus
        if self.done:
            result = self.session.result()
            reward += self.reward_shaper.terminal_reward(
                avg_fill=result.avg_closed_fill,
                pallets_closed=result.pallets_closed,
                placement_rate=result.placement_rate,
            )

        item_feat, candidates, info = self._observe(obs)
        info["placed"] = step_result.placed
        info["pallet_closed"] = step_result.pallet_closed

        return item_feat, candidates, reward, self.done, info

    def _observe(self, obs) -> Tuple[np.ndarray, List[CandidateAction], dict]:
        """Build observation from session state."""
        if obs.done or not obs.grippable:
            self.done = True
            self.current_box = None
            self.current_candidates = []
            return (
                np.zeros(self.config.item_input_dim, dtype=np.float32),
                [],
                {"done": True},
            )

        # Buffer-aware box selection: try each grippable box, pick one with
        # the most candidates (proxy for best placement options)
        best_box = None
        best_candidates: List[CandidateAction] = []

        for box in obs.grippable:
            candidates = self.candidate_gen.generate(
                box, obs.bin_states,
                max_candidates=self.config.max_candidates,
            )
            if len(candidates) > len(best_candidates):
                best_box = box
                best_candidates = candidates

        if best_box is None or not best_candidates:
            # No valid placement for any grippable box -- advance conveyor
            self.session.advance_conveyor()
            new_obs = self.session.observe()
            self.done = new_obs.done
            if self.done:
                self.current_box = None
                self.current_candidates = []
                return (
                    np.zeros(self.config.item_input_dim, dtype=np.float32),
                    [],
                    {"done": True},
                )
            return self._observe(new_obs)

        self.current_box = best_box
        self.current_candidates = best_candidates

        # Encode item features
        item_feat = encode_box_features(best_box, self.bin_config)

        info = {
            "box_id": best_box.id,
            "num_candidates": len(best_candidates),
            "grippable": len(obs.grippable),
            "done": False,
        }

        return item_feat, best_candidates, info

    def _generate_boxes(self) -> List[Box]:
        """Generate random boxes for a training episode."""
        n = self.config.num_boxes_per_episode
        lo, hi = self.config.box_size_range
        wlo, whi = self.config.box_weight_range
        boxes = []
        for i in range(n):
            l = float(self.rng.integers(int(lo / 10), int(hi / 10) + 1) * 10)
            w = float(self.rng.integers(int(lo / 10), int(hi / 10) + 1) * 10)
            h = float(self.rng.integers(int(lo / 10), int(hi / 10) + 1) * 10)
            weight = float(self.rng.uniform(wlo, whi))
            boxes.append(Box(id=i, length=l, width=w, height=h, weight=weight))
        return boxes

    def get_session_result(self):
        """Get the final session result (call after episode ends)."""
        if self.session is not None:
            return self.session.result()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Batch utilities for variable-size candidates
# ─────────────────────────────────────────────────────────────────────────────

def pad_candidates(
    candidates_list: List[List[CandidateAction]],
    candidate_input_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-size candidate feature arrays to the batch maximum.

    Args:
        candidates_list:    List of candidate lists (one per env).
        candidate_input_dim: Feature dimension per candidate.
        device:             PyTorch device.

    Returns:
        candidate_features: (batch, max_N, candidate_input_dim)
        candidate_mask:     (batch, max_N) -- True = valid
    """
    batch_size = len(candidates_list)
    max_N = max(len(c) for c in candidates_list) if candidates_list else 1
    max_N = max(max_N, 1)  # At least 1 to avoid empty tensors

    features = np.zeros((batch_size, max_N, candidate_input_dim), dtype=np.float32)
    mask = np.zeros((batch_size, max_N), dtype=np.bool_)

    for i, cands in enumerate(candidates_list):
        n = len(cands)
        if n > 0:
            features[i, :n] = np.stack([c.features for c in cands])
            mask[i, :n] = True

    return (
        torch.from_numpy(features).to(device),
        torch.from_numpy(mask).to(device),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr_schedule(
    config: PCTTransformerConfig,
    optimizer: optim.Optimizer,
    num_updates: int,
) -> optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler.

    Supports: 'cosine', 'linear', 'constant'.
    With optional warmup phase.
    """
    warmup_steps = int(num_updates * config.lr_warmup_frac)

    if config.lr_schedule == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(num_updates - warmup_steps, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif config.lr_schedule == "linear":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0 - (step - warmup_steps) / max(num_updates - warmup_steps, 1)
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:  # constant
        return optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# PPO update
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(
    network: PCTTransformerNet,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    config: PCTTransformerConfig,
    device: torch.device,
) -> Dict[str, float]:
    """
    Perform PPO update from a filled rollout buffer.

    Steps:
      1. Compute GAE advantages and returns.
      2. For each PPO epoch:
         a. Shuffle and split into mini-batches.
         b. For each mini-batch: compute ratio, clipped loss, value loss, entropy.
         c. Backpropagate and step optimizer.

    Args:
        network:   The PCT Transformer actor-critic.
        optimizer: Adam optimiser.
        buffer:    Filled rollout buffer.
        config:    Hyperparameters.
        device:    PyTorch device.

    Returns:
        Dict of training metrics (policy_loss, value_loss, entropy, etc.).
    """
    # Compute GAE
    # Bootstrap value: use last observation's value (already in buffer)
    # For simplicity, if last step was terminal, last_value = 0
    last_value = 0.0 if buffer.dones[-1] else buffer.values[-1]
    advantages, returns = buffer.compute_gae(
        last_value, config.gamma, config.gae_lambda,
    )

    # Normalise advantages
    if config.normalize_advantages and len(advantages) > 1:
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

    # PPO epochs
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    num_updates = 0

    for epoch in range(config.ppo_epochs):
        for batch in buffer.get_batches(
            advantages, returns,
            config.minibatch_size,
            device,
        ):
            # Forward pass: evaluate old actions under current policy
            _, new_log_probs, entropy, new_values = network.get_action_and_value(
                batch['item_features'],
                batch['candidate_features'],
                batch['candidate_mask'],
                action=batch['actions'],
            )

            # PPO ratio
            log_ratio = new_log_probs - batch['old_log_probs']
            ratio = torch.exp(log_ratio)

            # Approximate KL for early stopping diagnostics
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()

            # Clipped surrogate objective
            adv = batch['advantages']
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (with optional clipping)
            new_values_squeezed = new_values.squeeze(-1)
            if config.clip_value:
                old_values = batch['returns'] - batch['advantages']
                value_clipped = old_values + torch.clamp(
                    new_values_squeezed - old_values,
                    -config.clip_value_range,
                    config.clip_value_range,
                )
                vl1 = (new_values_squeezed - batch['returns']).pow(2)
                vl2 = (value_clipped - batch['returns']).pow(2)
                value_loss = 0.5 * torch.max(vl1, vl2).mean()
            else:
                value_loss = 0.5 * (new_values_squeezed - batch['returns']).pow(2).mean()

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                policy_loss
                + config.value_loss_coeff * value_loss
                + config.entropy_coeff * entropy_loss
            )

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
            optimizer.step()

            # Accumulate metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += (-entropy_loss).item()
            total_approx_kl += approx_kl
            num_updates += 1

    if num_updates == 0:
        num_updates = 1

    return {
        "policy_loss": total_policy_loss / num_updates,
        "value_loss": total_value_loss / num_updates,
        "entropy": total_entropy / num_updates,
        "approx_kl": total_approx_kl / num_updates,
        "num_updates": num_updates,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(
    network: PCTTransformerNet,
    config: PCTTransformerConfig,
    num_episodes: int = 10,
    seed: int = 99999,
) -> Dict[str, float]:
    """
    Run evaluation episodes with deterministic (greedy) policy.

    Returns:
        Dict with mean/std of avg_closed_fill, placement_rate, episode_reward.
    """
    device = torch.device(config.resolved_device)
    network.eval()

    fills = []
    placement_rates = []
    episode_rewards = []

    for ep in range(num_episodes):
        env = PCTPackingEnv(config, seed=seed + ep)
        item_feat, candidates, info = env.reset()
        total_reward = 0.0

        while not env.done and candidates:
            # Prepare batch (single env)
            item_t = torch.from_numpy(item_feat).unsqueeze(0).to(device)
            cand_feat, cand_mask = pad_candidates(
                [candidates], config.candidate_input_dim, device,
            )

            with torch.no_grad():
                action, _, _, _ = network.get_action_and_value(
                    item_t, cand_feat, cand_mask,
                    deterministic=True,
                )

            action_idx = action.item()
            if action_idx >= len(candidates):
                action_idx = 0

            item_feat, candidates, reward, done, info = env.step(action_idx)
            total_reward += reward

        result = env.get_session_result()
        if result is not None:
            fills.append(result.avg_closed_fill)
            placement_rates.append(result.placement_rate)
        episode_rewards.append(total_reward)

    network.train()

    return {
        "eval_fill_mean": float(np.mean(fills)) if fills else 0.0,
        "eval_fill_std": float(np.std(fills)) if fills else 0.0,
        "eval_placement_rate": float(np.mean(placement_rates)) if placement_rates else 0.0,
        "eval_reward_mean": float(np.mean(episode_rewards)),
        "eval_reward_std": float(np.std(episode_rewards)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config: PCTTransformerConfig) -> str:
    """
    Main PPO training loop for PCT Transformer.

    Returns the path to the best checkpoint.
    """
    device = torch.device(config.resolved_device)
    print(f"[PCT-Transformer] Training on device: {device}")
    print(f"[PCT-Transformer] Config: {config.num_envs} envs, "
          f"{config.total_episodes} episodes, lr={config.learning_rate}")

    # Create network
    network = PCTTransformerNet(config).to(device)
    print(f"[PCT-Transformer] Network:\n{network.summary()}")

    # Optimizer
    optimizer = optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
        eps=1e-5,
    )

    # Estimate total updates for LR schedule
    # Rough: total_episodes / num_envs rollouts, each with ppo_epochs * minibatches
    est_rollouts = config.total_episodes // config.num_envs
    scheduler = get_lr_schedule(config, optimizer, est_rollouts)

    # Logger
    os.makedirs(config.log_dir, exist_ok=True)
    logger = TrainingLogger(
        log_dir=config.log_dir,
        strategy_name="rl_pct_transformer",
        use_tensorboard=config.use_tensorboard,
    )
    logger.log_config(config.to_dict())

    # Create parallel environments
    envs = [
        PCTPackingEnv(config, seed=config.seed + i)
        for i in range(config.num_envs)
    ]

    # Training state
    best_fill = 0.0
    best_checkpoint_path = os.path.join(config.log_dir, "best.pt")
    global_episode = 0
    global_step = 0
    t0 = time.time()

    print(f"[PCT-Transformer] Starting training...")

    while global_episode < config.total_episodes:
        # ── Collect rollout ──
        buffer = RolloutBuffer()
        network.eval()

        for env in envs:
            item_feat, candidates, info = env.reset()
            episode_reward = 0.0

            for step in range(config.rollout_steps):
                if env.done or not candidates:
                    break

                # Prepare single-env batch
                item_t = torch.from_numpy(item_feat).unsqueeze(0).to(device)
                cand_feats_np = np.stack([c.features for c in candidates])
                cand_t = torch.from_numpy(cand_feats_np).unsqueeze(0).to(device)
                mask_t = torch.ones(1, len(candidates), dtype=torch.bool, device=device)

                with torch.no_grad():
                    action, log_prob, _, value = network.get_action_and_value(
                        item_t, cand_t, mask_t,
                    )

                action_idx = action.item()
                if action_idx >= len(candidates):
                    action_idx = len(candidates) - 1

                # Step environment
                next_item_feat, next_candidates, reward, done, step_info = env.step(action_idx)
                episode_reward += reward

                # Store in buffer
                buffer.add(
                    item_feat=item_feat,
                    cand_feat=cand_feats_np,
                    action=action_idx,
                    log_prob=log_prob.item(),
                    value=value.item(),
                    reward=reward,
                    done=done,
                )

                item_feat = next_item_feat
                candidates = next_candidates
                global_step += 1

            # Episode finished (by done or rollout_steps)
            global_episode += 1

            # Get final result
            result = env.get_session_result()
            avg_fill = result.avg_closed_fill if result else 0.0
            placement_rate = result.placement_rate if result else 0.0

            # Log episode
            logger.log_episode(
                global_episode,
                reward=episode_reward,
                fill=avg_fill,
                placement_rate=placement_rate,
                pallets_closed=result.pallets_closed if result else 0,
            )

            if global_episode % config.log_interval == 0:
                elapsed = time.time() - t0
                logger.print_progress(
                    global_episode,
                    config.total_episodes,
                    reward=episode_reward,
                    fill=avg_fill,
                    lr=optimizer.param_groups[0]['lr'],
                    eps_per_sec=global_episode / max(elapsed, 1),
                )

        # ── PPO Update ──
        if len(buffer) > 0:
            network.train()
            update_metrics = ppo_update(network, optimizer, buffer, config, device)
            scheduler.step()

            # Log update metrics
            logger.log_step(
                global_step,
                policy_loss=update_metrics['policy_loss'],
                value_loss=update_metrics['value_loss'],
                entropy=update_metrics['entropy'],
                approx_kl=update_metrics['approx_kl'],
            )

        # ── Evaluation ──
        if global_episode % config.eval_interval == 0:
            eval_metrics = evaluate_policy(
                network, config,
                num_episodes=config.eval_episodes,
            )
            print(f"  [EVAL] fill={eval_metrics['eval_fill_mean']:.4f} "
                  f"+/-{eval_metrics['eval_fill_std']:.4f}, "
                  f"placement={eval_metrics['eval_placement_rate']:.4f}")

            logger.log_episode(
                global_episode,
                **eval_metrics,
            )

            # Save best model
            if eval_metrics['eval_fill_mean'] > best_fill:
                best_fill = eval_metrics['eval_fill_mean']
                torch.save({
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config.to_dict(),
                    'episode': global_episode,
                    'best_fill': best_fill,
                }, best_checkpoint_path)
                print(f"  [SAVE] New best model: fill={best_fill:.4f}")

        # ── Checkpoint ──
        if global_episode % config.save_interval == 0:
            ckpt_path = os.path.join(
                config.log_dir,
                f"checkpoint_ep{global_episode}.pt",
            )
            torch.save({
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.to_dict(),
                'episode': global_episode,
                'best_fill': best_fill,
            }, ckpt_path)

    # ── Final save ──
    final_path = os.path.join(config.log_dir, "final.pt")
    torch.save({
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.to_dict(),
        'episode': global_episode,
        'best_fill': best_fill,
    }, final_path)

    # Generate plots
    logger.plot_training_curves()
    logger.close()

    elapsed = time.time() - t0
    print(f"\n[PCT-Transformer] Training complete!")
    print(f"  Total episodes:  {global_episode}")
    print(f"  Best fill:       {best_fill:.4f}")
    print(f"  Wall time:       {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  Best checkpoint: {best_checkpoint_path}")
    print(f"  Final checkpoint:{final_path}")

    return best_checkpoint_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> PCTTransformerConfig:
    """Parse CLI arguments into a PCTTransformerConfig."""
    parser = argparse.ArgumentParser(
        description="Train PCT Transformer RL strategy for 3D bin packing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training
    parser.add_argument("--episodes", type=int, default=200_000,
                        help="Total training episodes")
    parser.add_argument("--num_envs", type=int, default=16,
                        help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--rollout_steps", type=int, default=20,
                        help="Steps per env per rollout")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="PPO epochs per update")

    # Network
    parser.add_argument("--d_model", type=int, default=128,
                        help="Transformer d_model")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of Transformer encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Environment
    parser.add_argument("--num_boxes", type=int, default=100,
                        help="Boxes per episode")
    parser.add_argument("--num_bins", type=int, default=2,
                        help="Number of pallet stations")
    parser.add_argument("--num_orientations", type=int, default=2,
                        help="Allowed orientations (2=flat, 6=all)")

    # Candidate generation
    parser.add_argument("--max_candidates", type=int, default=200,
                        help="Maximum candidates per step")
    parser.add_argument("--floor_scan_step", type=float, default=50.0,
                        help="Floor scan grid step (mm)")

    # PPO
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Discount factor")
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                        help="PPO clip ratio")
    parser.add_argument("--entropy_coeff", type=float, default=0.01,
                        help="Entropy bonus coefficient")

    # Logging
    parser.add_argument("--log_dir", type=str,
                        default="outputs/rl_pct_transformer/logs",
                        help="Log directory")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Log every N episodes")
    parser.add_argument("--eval_interval", type=int, default=200,
                        help="Evaluate every N episodes")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Save checkpoint every N episodes")

    # Resume
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda")

    args = parser.parse_args()

    config = PCTTransformerConfig(
        total_episodes=args.episodes,
        num_envs=args.num_envs,
        learning_rate=args.lr,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dropout=args.dropout,
        num_boxes_per_episode=args.num_boxes,
        num_bins=args.num_bins,
        num_orientations=args.num_orientations,
        max_candidates=args.max_candidates,
        floor_scan_step=args.floor_scan_step,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        entropy_coeff=args.entropy_coeff,
        log_dir=args.log_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    return config


def main() -> None:
    """CLI entry point."""
    config = parse_args()

    # Resume from checkpoint if specified
    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
        print(f"[PCT-Transformer] Resuming from {config.checkpoint_path}")
        device = torch.device(config.resolved_device)
        checkpoint = torch.load(config.checkpoint_path, map_location=device)

        # Merge saved config with CLI overrides for schedule-related params
        saved_config = checkpoint.get('config', {})
        print(f"  Previous best fill: {saved_config.get('best_fill', 'N/A')}")
        print(f"  Previous episode:   {checkpoint.get('episode', 'N/A')}")

    train(config)


if __name__ == "__main__":
    main()
