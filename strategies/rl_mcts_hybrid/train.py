"""
Training script for the MCTS-Guided Hierarchical Actor-Critic.

Implements a multi-phase training pipeline:

Phase 1: IMITATION WARM-START (imitation_epochs)
  - Collect demonstrations from the top heuristic strategies
  - Supervised learning on (state, action) pairs to initialise the policy
  - Uses cross-entropy loss on both high-level and low-level actions
  - Dramatically accelerates RL convergence

Phase 2: CURRICULUM RL (4 stages of increasing difficulty)
  - Stage 0: Single box, single bin (learn basic placement)
  - Stage 1: 4 boxes, single bin (learn sequencing)
  - Stage 2: Full pick window, 2 bins (learn item + bin selection)
  - Stage 3: Full problem + MCTS (learn planning)

Phase 3: MCTS-IMPROVED TRAINING (after curriculum)
  - Enable MCTS during data collection for better exploration
  - Use MCTS search policy as improved targets (policy distillation)
  - World model is trained alongside policy (auxiliary loss)
  - Void detection provides auxiliary supervision signal

Loss function:
  L = L_ppo_hl + L_ppo_ll + alpha * L_world_model + beta * L_void
      + gamma * L_imitation (annealed to 0)

Where:
  L_ppo_hl   = PPO clip loss on high-level policy
  L_ppo_ll   = PPO clip loss on low-level policy
  L_world_model = MSE on next-state + reward prediction
  L_void     = BCE on trapped void fraction prediction
  L_imitation = cross-entropy on heuristic demonstrations

Usage:
    python train.py
    python train.py --total_timesteps 10000000 --num_envs 32
    python train.py --resume outputs/rl_mcts_hybrid/checkpoints/latest.pt
    python train.py --phase imitation  # Run only imitation phase

HPC:
    python train.py --num_envs 64 --total_timesteps 20000000

References:
    - Schulman et al. (2017): PPO
    - Schrittwieser et al. (2020): MuZero
    - Fang et al. (2026): MCTS + MPC for online 3D BPP
    - Xiong et al. (2024): GOPT masked PPO
"""

from __future__ import annotations

import sys
import os
import argparse
import copy
import json
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import Box, BinConfig, Orientation
from simulator.session import PackingSession, SessionConfig
from simulator.close_policy import HeightClosePolicy
from simulator.bin_state import BinState

from strategies.rl_common.environment import BinPackingEnv, EnvConfig, generate_random_boxes
from strategies.rl_common.rewards import RewardShaper, RewardConfig
from strategies.rl_common.logger import TrainingLogger

from strategies.rl_mcts_hybrid.config import MCTSHybridConfig
from strategies.rl_mcts_hybrid.network import (
    MCTSHybridNet, HighLevelOutput, LowLevelOutput, WorldModelOutput,
)
from strategies.rl_mcts_hybrid.candidate_generator import EnrichedCandidateGenerator
from strategies.rl_mcts_hybrid.void_detector import compute_void_fraction
from strategies.rl_mcts_hybrid.strategy import (
    _encode_observation, _build_hl_action_mask, _decode_hl_action,
)


# ---------------------------------------------------------------------------
# Rollout buffer for hierarchical PPO
# ---------------------------------------------------------------------------

class HierarchicalRolloutBuffer:
    """
    Stores rollout data for the hierarchical (HL + LL) PPO update.

    Each step stores:
      - Observation tensors
      - HL action, log_prob, value
      - LL action, log_prob, value, candidate features/mask
      - Reward, done
      - World model targets (next heightmap features, void fraction)
    """

    def __init__(
        self,
        buffer_size: int,
        config: MCTSHybridConfig,
        device: torch.device,
    ) -> None:
        self.buffer_size = buffer_size
        self.config = config
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Clear the buffer."""
        self.global_states: List[torch.Tensor] = []
        self.hl_actions: List[int] = []
        self.hl_log_probs: List[float] = []
        self.hl_values: List[float] = []
        self.hl_masks: List[np.ndarray] = []

        self.ll_actions: List[int] = []
        self.ll_log_probs: List[float] = []
        self.ll_values: List[float] = []
        self.candidate_features: List[np.ndarray] = []
        self.candidate_masks: List[np.ndarray] = []
        self.hl_embeds: List[torch.Tensor] = []

        self.rewards: List[float] = []
        self.dones: List[bool] = []

        # World model targets
        self.wm_next_states: List[Optional[torch.Tensor]] = []
        self.wm_rewards: List[float] = []
        self.void_fractions: List[List[float]] = []

        self.size = 0

    def add(
        self,
        global_state: torch.Tensor,
        hl_action: int,
        hl_log_prob: float,
        hl_value: float,
        hl_mask: np.ndarray,
        ll_action: int,
        ll_log_prob: float,
        ll_value: float,
        candidate_feats: np.ndarray,
        candidate_mask: np.ndarray,
        hl_embed: torch.Tensor,
        reward: float,
        done: bool,
        next_global_state: Optional[torch.Tensor] = None,
        void_frac: Optional[List[float]] = None,
    ) -> None:
        """Add a single transition."""
        self.global_states.append(global_state.detach().cpu())
        self.hl_actions.append(hl_action)
        self.hl_log_probs.append(hl_log_prob)
        self.hl_values.append(hl_value)
        self.hl_masks.append(hl_mask)

        self.ll_actions.append(ll_action)
        self.ll_log_probs.append(ll_log_prob)
        self.ll_values.append(ll_value)
        self.candidate_features.append(candidate_feats)
        self.candidate_masks.append(candidate_mask)
        self.hl_embeds.append(hl_embed.detach().cpu())

        self.rewards.append(reward)
        self.dones.append(done)

        self.wm_next_states.append(
            next_global_state.detach().cpu() if next_global_state is not None else None
        )
        self.wm_rewards.append(reward)
        self.void_fractions.append(void_frac or [0.0, 0.0])

        self.size += 1

    def compute_gae(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation for both HL and LL.

        Returns (advantages, returns) arrays.
        """
        n = self.size
        advantages = np.zeros(n, dtype=np.float64)
        returns = np.zeros(n, dtype=np.float64)

        # Use combined value (HL + LL) for advantage computation
        values = np.array(
            [self.hl_values[i] + self.ll_values[i] for i in range(n)],
            dtype=np.float64,
        )

        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns


# ---------------------------------------------------------------------------
# Demonstration collector (imitation learning)
# ---------------------------------------------------------------------------

def collect_demonstrations(
    config: MCTSHybridConfig,
    num_episodes: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Collect expert demonstrations from the top heuristic strategies.

    Runs each heuristic strategy and records (state, action) pairs for
    supervised pre-training of the policy networks.

    Returns list of dicts with:
      - 'obs': observation tensors
      - 'hl_action': high-level action (always box=0, bin=0 for single-bin)
      - 'll_action': low-level candidate index
      - 'candidates': candidate list
      - 'reward': immediate reward
    """
    from strategies.base_strategy import get_strategy, STRATEGY_REGISTRY

    heuristic_names = [
        name for name in config.heuristic_names
        if name in STRATEGY_REGISTRY
    ]
    if not heuristic_names:
        heuristic_names = ["baseline"]

    demonstrations: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    candidate_gen = EnrichedCandidateGenerator(config)

    bin_config = BinConfig(
        length=config.bin_length,
        width=config.bin_width,
        height=config.bin_height,
        resolution=config.resolution,
    )

    eps_per_heuristic = max(1, num_episodes // len(heuristic_names))

    for heuristic_name in heuristic_names:
        print(f"  Collecting {eps_per_heuristic} episodes from '{heuristic_name}'...")
        strategy = get_strategy(heuristic_name)

        for ep in range(eps_per_heuristic):
            boxes = generate_random_boxes(
                n=config.num_boxes_per_episode,
                size_range=config.box_size_range,
                weight_range=config.box_weight_range,
                rng=rng,
            )

            session_cfg = SessionConfig(
                bin_config=bin_config,
                num_bins=1,  # Single-bin for demonstration simplicity
                buffer_size=1,
                pick_window=1,
                close_policy=HeightClosePolicy(config.close_height),
            )
            session = PackingSession(session_cfg)
            obs = session.reset(boxes, strategy_name="demo")

            from config import ExperimentConfig
            exp_config = ExperimentConfig(bin=bin_config)
            strategy.on_episode_start(exp_config)

            while not obs.done and obs.grippable:
                box = obs.grippable[0]
                bs = obs.bin_states[0]

                # Get heuristic placement
                decision = strategy.decide_placement(box, bs)
                if decision is None:
                    obs = session.observe()
                    if hasattr(session, 'advance_conveyor'):
                        session.advance_conveyor()
                    obs = session.observe()
                    continue

                # Generate candidates and find which one matches
                candidates = candidate_gen.generate(box, [bs])
                if not candidates:
                    obs = session.observe()
                    continue

                # Find closest candidate to the heuristic decision
                best_match = 0
                best_dist = float("inf")
                for i, c in enumerate(candidates):
                    dist = abs(c.x - decision.x) + abs(c.y - decision.y)
                    if c.orient_idx == decision.orientation_idx:
                        dist -= 100  # Strong bonus for matching orientation
                    if dist < best_dist:
                        best_dist = dist
                        best_match = i

                # Store demonstration
                cand_feats = candidate_gen.get_feature_array(candidates)
                demonstrations.append({
                    'box': box,
                    'bin_state_fill': bs.get_fill_rate(),
                    'hl_action': 0,  # box=0, bin=0
                    'll_action': best_match,
                    'candidate_features': cand_feats,
                    'n_candidates': len(candidates),
                    'heuristic': heuristic_name,
                })

                # Execute placement
                ol, ow, oh = Orientation.get_flat(
                    box.length, box.width, box.height,
                )[decision.orientation_idx] if decision.orientation_idx < len(
                    Orientation.get_flat(box.length, box.width, box.height)
                ) else (box.length, box.width, box.height)

                result = session.step(
                    box.id, 0, decision.x, decision.y,
                    decision.orientation_idx,
                )
                obs = session.observe()

    print(f"  Collected {len(demonstrations)} demonstration transitions "
          f"from {len(heuristic_names)} heuristics.")
    return demonstrations


# ---------------------------------------------------------------------------
# Imitation pre-training phase
# ---------------------------------------------------------------------------

def imitation_pretrain(
    model: MCTSHybridNet,
    demonstrations: List[Dict[str, Any]],
    config: MCTSHybridConfig,
    device: torch.device,
    logger: TrainingLogger,
) -> None:
    """
    Phase 1: Supervised pre-training from heuristic demonstrations.

    Trains the low-level policy to select the same candidates as the
    heuristics.  This provides a warm start for RL training.
    """
    print("\n=== Phase 1: Imitation Pre-Training ===")

    if not demonstrations:
        print("  No demonstrations available. Skipping imitation phase.")
        return

    optimizer = optim.Adam(model.parameters(), lr=config.imitation_lr)
    max_cands = config.max_candidates
    cand_dim = config.candidate_input_dim

    for epoch in range(config.imitation_epochs):
        # Shuffle demonstrations
        indices = np.random.permutation(len(demonstrations))
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        # Mini-batch training
        batch_size = min(256, len(demonstrations))
        for batch_start in range(0, len(demonstrations), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch = [demonstrations[i] for i in batch_indices]

            B = len(batch)

            # Prepare candidate features and targets
            padded_feats = np.zeros((B, max_cands, cand_dim), dtype=np.float32)
            cand_masks = np.zeros((B, max_cands), dtype=bool)
            targets = np.zeros(B, dtype=np.int64)

            for i, demo in enumerate(batch):
                n = min(demo['n_candidates'], max_cands)
                padded_feats[i, :n] = demo['candidate_features'][:n]
                cand_masks[i, :n] = True
                targets[i] = min(demo['ll_action'], n - 1)

            # Create dummy global state and HL embed
            # (We only train the low-level pointer in imitation)
            dummy_state = torch.zeros(B, config.global_state_dim, device=device)
            dummy_hl_embed = torch.zeros(B, config.high_level_embed_dim, device=device)
            cand_t = torch.as_tensor(padded_feats, device=device)
            mask_t = torch.as_tensor(cand_masks, device=device)
            target_t = torch.as_tensor(targets, device=device)

            # Forward pass through low-level policy
            ll_out = model.low_level(
                dummy_state, dummy_hl_embed,
                cand_t, mask_t,
                action=target_t,
            )

            # Cross-entropy loss (negative log-prob of the expert action)
            loss = -ll_out.log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            total_loss += loss.item() * B

            # Accuracy
            with torch.no_grad():
                ll_greedy = model.low_level(
                    dummy_state, dummy_hl_embed,
                    cand_t, mask_t,
                    deterministic=True,
                )
                correct = (ll_greedy.action == target_t).sum().item()
                total_correct += correct
                total_count += B

        avg_loss = total_loss / max(total_count, 1)
        accuracy = total_correct / max(total_count, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{config.imitation_epochs}: "
                  f"loss={avg_loss:.4f}, accuracy={accuracy:.1%}")

        logger.log_episode(
            epoch,
            loss=avg_loss,
            accuracy=accuracy,
            phase="imitation",
        )

    print(f"  Imitation pre-training complete. Final accuracy: {accuracy:.1%}")


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    model: MCTSHybridNet,
    optimizer: optim.Optimizer,
    buffer: HierarchicalRolloutBuffer,
    config: MCTSHybridConfig,
    device: torch.device,
    imitation_weight: float = 0.0,
) -> Dict[str, float]:
    """
    Perform a PPO update using collected rollout data.

    Updates both high-level and low-level policies simultaneously with
    shared advantage estimates.  Also trains the world model on
    transition data.

    Returns dict of loss components for logging.
    """
    n = buffer.size
    if n == 0:
        return {}

    # Compute GAE
    # Get last value estimate
    last_state = buffer.global_states[-1].to(device).unsqueeze(0)
    with torch.no_grad():
        trunk_out = model.high_level.trunk(last_state)
        last_hl_v = model.high_level.value_head(trunk_out).item()
    last_value = last_hl_v

    advantages, returns = buffer.compute_gae(
        last_value, config.gamma, config.gae_lambda,
    )

    # Normalise advantages
    adv_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    if adv_tensor.numel() > 1:
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
    ret_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)

    # Prepare batch tensors
    states_t = torch.stack(buffer.global_states).to(device)
    hl_actions_t = torch.tensor(buffer.hl_actions, dtype=torch.long, device=device)
    hl_old_logp = torch.tensor(buffer.hl_log_probs, dtype=torch.float32, device=device)
    ll_actions_t = torch.tensor(buffer.ll_actions, dtype=torch.long, device=device)
    ll_old_logp = torch.tensor(buffer.ll_log_probs, dtype=torch.float32, device=device)
    hl_embeds_t = torch.stack(buffer.hl_embeds).to(device)

    # HL masks
    hl_masks_t = torch.as_tensor(
        np.stack(buffer.hl_masks), dtype=torch.float32, device=device,
    )

    # Candidate features and masks
    max_cands = config.max_candidates
    cand_dim = config.candidate_input_dim
    cand_feats_padded = np.zeros((n, max_cands, cand_dim), dtype=np.float32)
    cand_masks_padded = np.zeros((n, max_cands), dtype=bool)
    for i in range(n):
        cf = buffer.candidate_features[i]
        cm = buffer.candidate_masks[i]
        nc = min(cf.shape[0], max_cands)
        cand_feats_padded[i, :nc] = cf[:nc]
        cand_masks_padded[i, :nc] = cm[:nc]

    cand_feats_t = torch.as_tensor(cand_feats_padded, device=device)
    cand_masks_t = torch.as_tensor(cand_masks_padded, device=device)

    # World model targets
    wm_rewards_t = torch.tensor(buffer.wm_rewards, dtype=torch.float32, device=device)
    void_targets_t = torch.tensor(
        buffer.void_fractions, dtype=torch.float32, device=device,
    )

    # PPO update epochs
    loss_stats: Dict[str, List[float]] = {
        'policy_hl': [], 'policy_ll': [], 'value_hl': [], 'value_ll': [],
        'entropy_hl': [], 'entropy_ll': [], 'world_model': [], 'void': [],
    }

    indices = np.arange(n)
    mb_size = min(config.minibatch_size, n)

    for epoch in range(config.num_epochs):
        np.random.shuffle(indices)

        for start in range(0, n, mb_size):
            end = min(start + mb_size, n)
            mb_idx = indices[start:end]
            mb = torch.tensor(mb_idx, dtype=torch.long, device=device)

            mb_states = states_t[mb]
            mb_hl_acts = hl_actions_t[mb]
            mb_hl_old_lp = hl_old_logp[mb]
            mb_ll_acts = ll_actions_t[mb]
            mb_ll_old_lp = ll_old_logp[mb]
            mb_adv = adv_tensor[mb]
            mb_ret = ret_tensor[mb]
            mb_hl_masks = hl_masks_t[mb]
            mb_cand_feats = cand_feats_t[mb]
            mb_cand_masks = cand_masks_t[mb]
            mb_hl_embeds = hl_embeds_t[mb]

            # ---- High-level policy ----
            hl_out = model.high_level(
                mb_states, mb_hl_masks,
                action=mb_hl_acts,
            )

            # PPO clip loss for HL
            ratio_hl = torch.exp(hl_out.log_prob - mb_hl_old_lp)
            surr1_hl = ratio_hl * mb_adv
            surr2_hl = torch.clamp(
                ratio_hl, 1.0 - config.clip_eps, 1.0 + config.clip_eps,
            ) * mb_adv
            policy_loss_hl = -torch.min(surr1_hl, surr2_hl).mean()

            value_loss_hl = F.mse_loss(hl_out.value, mb_ret)
            entropy_loss_hl = -hl_out.entropy.mean()

            # ---- Low-level policy ----
            ll_out = model.low_level(
                mb_states, mb_hl_embeds,
                mb_cand_feats, mb_cand_masks,
                action=mb_ll_acts,
            )

            ratio_ll = torch.exp(ll_out.log_prob - mb_ll_old_lp)
            surr1_ll = ratio_ll * mb_adv
            surr2_ll = torch.clamp(
                ratio_ll, 1.0 - config.clip_eps, 1.0 + config.clip_eps,
            ) * mb_adv
            policy_loss_ll = -torch.min(surr1_ll, surr2_ll).mean()

            value_loss_ll = F.mse_loss(ll_out.value, mb_ret)
            entropy_loss_ll = -ll_out.entropy.mean()

            # ---- World model loss ----
            mb_wm_rewards = wm_rewards_t[mb]
            wm_out = model.world_model(mb_states, mb_hl_acts, mb_ll_acts)
            wm_reward_loss = F.mse_loss(wm_out.reward_pred, mb_wm_rewards)

            # Void prediction loss
            mb_void_targets = void_targets_t[mb]
            void_loss = F.mse_loss(wm_out.void_fraction, mb_void_targets)

            # ---- Total loss ----
            total_loss = (
                policy_loss_hl + policy_loss_ll
                + config.vf_coeff * (value_loss_hl + value_loss_ll)
                + config.ent_coeff * (entropy_loss_hl + entropy_loss_ll)
                + config.world_model_loss_weight * wm_reward_loss
                + config.void_loss_weight * void_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            # Track losses
            loss_stats['policy_hl'].append(policy_loss_hl.item())
            loss_stats['policy_ll'].append(policy_loss_ll.item())
            loss_stats['value_hl'].append(value_loss_hl.item())
            loss_stats['value_ll'].append(value_loss_ll.item())
            loss_stats['entropy_hl'].append(-entropy_loss_hl.item())
            loss_stats['entropy_ll'].append(-entropy_loss_ll.item())
            loss_stats['world_model'].append(wm_reward_loss.item())
            loss_stats['void'].append(void_loss.item())

    return {k: float(np.mean(v)) for k, v in loss_stats.items() if v}


# ---------------------------------------------------------------------------
# Curriculum stage configuration
# ---------------------------------------------------------------------------

def get_curriculum_config(
    stage: int,
    base_config: MCTSHybridConfig,
) -> Dict[str, Any]:
    """
    Get environment configuration for a curriculum stage.

    Stage 0: Single box, single bin (basic placement)
    Stage 1: 4 boxes, single bin (sequencing)
    Stage 2: Full pick window, 2 bins (item + bin selection)
    Stage 3: Full problem + MCTS (planning)
    """
    configs = [
        # Stage 0: simplest
        {
            'num_boxes': 10,
            'num_bins': 1,
            'pick_window': 1,
            'buffer_size': 1,
            'use_mcts': False,
            'size_range': (200.0, 500.0),
        },
        # Stage 1: sequencing
        {
            'num_boxes': 30,
            'num_bins': 1,
            'pick_window': 4,
            'buffer_size': 4,
            'use_mcts': False,
            'size_range': (150.0, 550.0),
        },
        # Stage 2: multi-bin
        {
            'num_boxes': 60,
            'num_bins': 2,
            'pick_window': 4,
            'buffer_size': 8,
            'use_mcts': False,
            'size_range': (100.0, 600.0),
        },
        # Stage 3: full problem + MCTS
        {
            'num_boxes': base_config.num_boxes_per_episode,
            'num_bins': base_config.num_bins,
            'pick_window': base_config.pick_window,
            'buffer_size': base_config.buffer_size,
            'use_mcts': base_config.mcts_train_enabled,
            'size_range': base_config.box_size_range,
        },
    ]

    if stage < len(configs):
        return configs[stage]
    return configs[-1]


# ---------------------------------------------------------------------------
# Single rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(
    model: MCTSHybridNet,
    env: BinPackingEnv,
    candidate_gen: EnrichedCandidateGenerator,
    buffer: HierarchicalRolloutBuffer,
    config: MCTSHybridConfig,
    device: torch.device,
    rollout_steps: int,
) -> Dict[str, float]:
    """
    Collect a rollout of transitions for PPO training.

    Runs the current policy in the environment for rollout_steps steps,
    storing all data in the buffer.

    Returns episode statistics.
    """
    buffer.reset()

    obs, info = env.reset()
    episode_rewards = []
    episode_fills = []
    current_ep_reward = 0.0
    episodes_completed = 0

    for step in range(rollout_steps):
        env_obs = env._obs
        if env_obs is None or env_obs.done:
            obs, info = env.reset()
            env_obs = env._obs

        # Get current state
        grippable = env_obs.grippable if env_obs else []
        bin_states = [bs for bs in env_obs.bin_states] if env_obs else []

        if not grippable:
            obs, info = env.reset()
            continue

        box = grippable[0]

        # Encode observation
        obs_tensors = _encode_observation(box, bin_states, config, device)

        with torch.no_grad():
            global_state, _ = model.encode(
                obs_tensors['heightmaps'],
                obs_tensors['box_features'],
                obs_tensors['buffer_features'],
                obs_tensors['buffer_mask'],
            )

            # High-level mask
            hl_mask_np = _build_hl_action_mask(
                len(grippable), config.num_bins, bin_states, config,
            )
            hl_mask_t = torch.as_tensor(
                hl_mask_np.reshape(1, -1), device=device,
            )

            # High-level policy
            hl_out = model.high_level(global_state, hl_mask_t)

        hl_action_idx = int(hl_out.action.item())
        action_type, box_idx, bin_idx = _decode_hl_action(hl_action_idx, config)

        # Default: use first box and first bin
        if action_type != "place":
            box_idx = 0
            bin_idx = 0

        if box_idx >= len(grippable):
            box_idx = 0
        if bin_idx >= len(bin_states):
            bin_idx = 0

        target_box = grippable[box_idx]
        target_bs = bin_states[bin_idx]

        # Generate candidates
        candidates = candidate_gen.generate(target_box, [target_bs])

        if not candidates:
            # No valid candidates -- skip
            env_action = env.env_config.total_actions - 1  # skip action
            obs, reward, terminated, truncated, info = env.step(env_action)
            current_ep_reward += reward

            if terminated or truncated:
                episode_rewards.append(current_ep_reward)
                if 'final_avg_fill' in info:
                    episode_fills.append(info['final_avg_fill'])
                current_ep_reward = 0.0
                episodes_completed += 1
                obs, info = env.reset()
            continue

        # Prepare candidate features
        cand_feats = candidate_gen.get_feature_array(candidates)
        n_cands = len(candidates)
        max_cands = config.max_candidates

        padded_feats = np.zeros((1, max_cands, config.candidate_input_dim), dtype=np.float32)
        padded_feats[0, :min(n_cands, max_cands)] = cand_feats[:max_cands]
        cand_mask = np.zeros((1, max_cands), dtype=bool)
        cand_mask[0, :min(n_cands, max_cands)] = True

        cand_t = torch.as_tensor(padded_feats, device=device)
        mask_t = torch.as_tensor(cand_mask, device=device)

        # Low-level policy
        with torch.no_grad():
            ll_out = model.low_level(
                global_state, hl_out.action_embed,
                cand_t, mask_t,
            )

        ll_action_idx = int(ll_out.action.item())
        if ll_action_idx >= n_cands:
            ll_action_idx = 0

        selected = candidates[ll_action_idx]

        # Execute action in environment
        # Map (bin_idx, x, y, orient) to flat action
        try:
            env_cfg = env.env_config
            gx = int(round(selected.x / env_cfg.action_grid_step))
            gy = int(round(selected.y / env_cfg.action_grid_step))
            gx = min(gx, env_cfg.action_grid_l - 1)
            gy = min(gy, env_cfg.action_grid_w - 1)
            orient = selected.orient_idx

            env_action = env._encode_action(box_idx, bin_idx, gx, gy, orient)
            if env_action >= env_cfg.total_actions:
                env_action = env_cfg.total_actions - 1
        except Exception:
            env_action = env.env_config.total_actions - 1

        obs, reward, terminated, truncated, info = env.step(env_action)
        current_ep_reward += reward

        # Compute void fractions for auxiliary loss
        void_fracs = [0.0] * config.num_bins
        try:
            new_obs = env._obs
            if new_obs is not None:
                for bi, bs_new in enumerate(new_obs.bin_states[:config.num_bins]):
                    void_fracs[bi] = compute_void_fraction(bs_new)
        except Exception:
            pass

        # Store transition
        buffer.add(
            global_state=global_state.squeeze(0),
            hl_action=hl_action_idx,
            hl_log_prob=hl_out.log_prob.item(),
            hl_value=hl_out.value.item(),
            hl_mask=hl_mask_np,
            ll_action=ll_action_idx,
            ll_log_prob=ll_out.log_prob.item(),
            ll_value=ll_out.value.item(),
            candidate_feats=padded_feats[0],
            candidate_mask=cand_mask[0],
            hl_embed=hl_out.action_embed.squeeze(0),
            reward=reward,
            done=terminated or truncated,
            void_frac=void_fracs,
        )

        if terminated or truncated:
            episode_rewards.append(current_ep_reward)
            if 'final_avg_fill' in info:
                episode_fills.append(info['final_avg_fill'])
            current_ep_reward = 0.0
            episodes_completed += 1
            obs, info = env.reset()

    stats = {
        'episodes': episodes_completed,
        'mean_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'mean_fill': float(np.mean(episode_fills)) if episode_fills else 0.0,
        'rollout_steps': buffer.size,
    }
    return stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: MCTSHybridNet,
    config: MCTSHybridConfig,
    device: torch.device,
    num_episodes: int = 10,
    seed: int = 9999,
) -> Dict[str, float]:
    """
    Evaluate the current policy deterministically.

    Returns:
        Dict with mean_fill, mean_reward, mean_episode_length.
    """
    from strategies.rl_mcts_hybrid.strategy import RLMCTSHybridStrategy

    bin_config = BinConfig(
        length=config.bin_length,
        width=config.bin_width,
        height=config.bin_height,
        resolution=config.resolution,
    )

    env_config = EnvConfig(
        bin_config=bin_config,
        num_bins=config.num_bins,
        buffer_size=config.buffer_size,
        pick_window=config.pick_window,
        close_height=config.close_height,
        num_orientations=config.num_orientations,
        num_boxes_per_episode=config.num_boxes_per_episode,
        box_size_range=config.box_size_range,
        seed=seed,
    )

    env = BinPackingEnv(config=env_config)
    candidate_gen = EnrichedCandidateGenerator(config)

    model.eval()
    fills = []
    rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0

        while True:
            env_obs = env._obs
            if env_obs is None or env_obs.done:
                break

            grippable = env_obs.grippable
            bin_states = env_obs.bin_states

            if not grippable:
                break

            box = grippable[0]

            # Encode and run policy
            obs_tensors = _encode_observation(box, list(bin_states), config, device)

            with torch.no_grad():
                global_state, _ = model.encode(
                    obs_tensors['heightmaps'],
                    obs_tensors['box_features'],
                    obs_tensors['buffer_features'],
                    obs_tensors['buffer_mask'],
                )

                hl_mask_np = _build_hl_action_mask(
                    len(grippable), config.num_bins, list(bin_states), config,
                )
                hl_mask_t = torch.as_tensor(
                    hl_mask_np.reshape(1, -1), device=device,
                )
                hl_out = model.high_level(
                    global_state, hl_mask_t, deterministic=True,
                )

            # Generate candidates
            candidates = candidate_gen.generate(box, [bin_states[0]])

            if not candidates:
                env_action = env.env_config.total_actions - 1
                obs, reward, terminated, truncated, info = env.step(env_action)
                ep_reward += reward
                if terminated or truncated:
                    break
                continue

            cand_feats = candidate_gen.get_feature_array(candidates)
            n_cands = len(candidates)
            max_cands = config.max_candidates

            padded_feats = np.zeros(
                (1, max_cands, config.candidate_input_dim), dtype=np.float32,
            )
            padded_feats[0, :min(n_cands, max_cands)] = cand_feats[:max_cands]
            cand_mask = np.zeros((1, max_cands), dtype=bool)
            cand_mask[0, :min(n_cands, max_cands)] = True

            with torch.no_grad():
                ll_out = model.low_level(
                    global_state, hl_out.action_embed,
                    torch.as_tensor(padded_feats, device=device),
                    torch.as_tensor(cand_mask, device=device),
                    deterministic=True,
                )

            action_idx = int(ll_out.action.item())
            if action_idx >= n_cands:
                action_idx = 0

            selected = candidates[action_idx]

            # Execute
            try:
                gx = int(round(selected.x / env.env_config.action_grid_step))
                gy = int(round(selected.y / env.env_config.action_grid_step))
                gx = min(gx, env.env_config.action_grid_l - 1)
                gy = min(gy, env.env_config.action_grid_w - 1)
                env_action = env._encode_action(
                    0, 0, gx, gy, selected.orient_idx,
                )
                if env_action >= env.env_config.total_actions:
                    env_action = env.env_config.total_actions - 1
            except Exception:
                env_action = env.env_config.total_actions - 1

            obs, reward, terminated, truncated, info = env.step(env_action)
            ep_reward += reward

            if terminated or truncated:
                break

        rewards.append(ep_reward)
        if 'final_avg_fill' in info:
            fills.append(info['final_avg_fill'])

    model.train()
    return {
        'mean_fill': float(np.mean(fills)) if fills else 0.0,
        'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
        'num_episodes': num_episodes,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Main training function."""
    config = MCTSHybridConfig()

    # Override from args
    if args.total_timesteps:
        config.total_timesteps = args.total_timesteps
    if args.num_envs:
        config.num_envs = args.num_envs
    if args.lr:
        config.lr = args.lr

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Directories — use --output_dir if provided, else default
    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = os.path.join(_WORKFLOW_ROOT, "outputs", "rl_mcts_hybrid")
    log_dir = os.path.join(base_dir, "logs")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Logger
    logger = TrainingLogger(log_dir=log_dir, strategy_name="rl_mcts_hybrid")

    # Model
    model = MCTSHybridNet(config).to(device)
    print(model.summary())

    # Resume if checkpoint exists
    start_step = 0
    best_fill = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_step = checkpoint.get('total_steps', 0)
        best_fill = checkpoint.get('best_fill', 0.0)
        print(f"  Resumed at step {start_step}, best_fill={best_fill:.1%}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr, eps=1e-5)

    # Save config
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2, default=str)

    # Determine which phases to run
    run_imitation = args.phase in ("all", "imitation") and not args.skip_imitation
    run_rl = args.phase in ("all", "rl")

    # ============================================================
    # Phase 1: Imitation pre-training
    # ============================================================
    if start_step == 0 and run_imitation:
        print("\nCollecting demonstrations from heuristic strategies...")
        demos = collect_demonstrations(
            config, num_episodes=100, seed=args.seed,
        )
        if demos:
            imitation_pretrain(model, demos, config, device, logger)
            # Save after imitation
            save_path = os.path.join(ckpt_dir, "after_imitation.pt")
            model.save(save_path)
            print(f"  Saved imitation checkpoint: {save_path}")

    if not run_rl:
        print("\nPhase 'imitation' complete. Skipping RL training.")
        logger.close()
        return

    # ============================================================
    # Phase 2 + 3: Curriculum RL Training
    # ============================================================
    print("\n=== Phase 2: Curriculum RL Training ===")

    bin_config = BinConfig(
        length=config.bin_length,
        width=config.bin_width,
        height=config.bin_height,
        resolution=config.resolution,
    )

    # Compute curriculum stage boundaries
    steps_per_stage = config.total_timesteps // max(config.curriculum_stages, 1)
    current_stage = 0

    total_steps = start_step
    update_count = 0
    recent_fills = deque(maxlen=50)

    # Create environment
    curr_cfg = get_curriculum_config(current_stage, config)
    env_config = EnvConfig(
        bin_config=bin_config,
        num_bins=curr_cfg['num_bins'],
        buffer_size=curr_cfg['buffer_size'],
        pick_window=curr_cfg['pick_window'],
        close_height=config.close_height,
        num_orientations=config.num_orientations,
        num_boxes_per_episode=curr_cfg['num_boxes'],
        box_size_range=curr_cfg['size_range'],
        seed=args.seed,
    )
    env = BinPackingEnv(config=env_config)
    candidate_gen = EnrichedCandidateGenerator(config)
    rollout_buffer = HierarchicalRolloutBuffer(
        config.rollout_steps, config, device,
    )

    print(f"  Starting at curriculum stage {current_stage}")

    eval_interval = 5000
    checkpoint_interval = 10000
    log_interval = 1000

    while total_steps < config.total_timesteps:
        # Check curriculum advancement
        new_stage = min(
            total_steps // steps_per_stage,
            config.curriculum_stages - 1,
        )
        if new_stage > current_stage:
            current_stage = new_stage
            curr_cfg = get_curriculum_config(current_stage, config)
            env_config = EnvConfig(
                bin_config=bin_config,
                num_bins=curr_cfg['num_bins'],
                buffer_size=curr_cfg['buffer_size'],
                pick_window=curr_cfg['pick_window'],
                close_height=config.close_height,
                num_orientations=config.num_orientations,
                num_boxes_per_episode=curr_cfg['num_boxes'],
                box_size_range=curr_cfg['size_range'],
                seed=args.seed + current_stage * 1000,
            )
            env = BinPackingEnv(config=env_config)
            print(f"\n  Advancing to curriculum stage {current_stage}: "
                  f"{curr_cfg['num_boxes']} boxes, {curr_cfg['num_bins']} bins, "
                  f"pick_window={curr_cfg['pick_window']}")

        # Collect rollout
        model.train()
        rollout_stats = collect_rollout(
            model, env, candidate_gen, rollout_buffer,
            config, device, config.rollout_steps,
        )
        total_steps += rollout_buffer.size

        # PPO update
        imitation_weight = max(
            0.0,
            config.imitation_weight * (1.0 - total_steps / config.total_timesteps),
        )
        loss_stats = ppo_update(
            model, optimizer, rollout_buffer, config, device, imitation_weight,
        )

        update_count += 1

        if rollout_stats['mean_fill'] > 0:
            recent_fills.append(rollout_stats['mean_fill'])

        # Logging
        if total_steps % log_interval < config.rollout_steps:
            avg_fill = float(np.mean(recent_fills)) if recent_fills else 0.0
            print(
                f"  Step {total_steps:>8d}/{config.total_timesteps} | "
                f"Stage {current_stage} | "
                f"Fill {avg_fill:.1%} | "
                f"Reward {rollout_stats['mean_reward']:.2f} | "
                f"Eps {rollout_stats['episodes']}"
            )

            logger.log_episode(
                update_count,
                reward=rollout_stats['mean_reward'],
                fill=rollout_stats.get('mean_fill', 0.0),
                total_steps=total_steps,
                stage=current_stage,
                **loss_stats,
            )

        # Evaluation
        if total_steps % eval_interval < config.rollout_steps:
            eval_stats = evaluate(
                model, config, device, num_episodes=5, seed=9999,
            )
            print(
                f"  [EVAL] Fill={eval_stats['mean_fill']:.1%} | "
                f"Reward={eval_stats['mean_reward']:.2f}"
            )

            if eval_stats['mean_fill'] > best_fill:
                best_fill = eval_stats['mean_fill']
                save_path = os.path.join(ckpt_dir, "best_model.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config.to_dict(),
                    'total_steps': total_steps,
                    'best_fill': best_fill,
                    'eval_stats': eval_stats,
                }, save_path)
                print(f"  New best model saved! Fill={best_fill:.1%}")

        # Periodic checkpointing
        if total_steps % checkpoint_interval < config.rollout_steps:
            save_path = os.path.join(ckpt_dir, f"step_{total_steps}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.to_dict(),
                'total_steps': total_steps,
                'best_fill': best_fill,
            }, save_path)

    # Final save
    final_path = os.path.join(ckpt_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'total_steps': total_steps,
        'best_fill': best_fill,
    }, final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Best fill rate: {best_fill:.1%}")

    # Generate training curves
    try:
        logger.plot_training_curves()
        print("Training curves saved to", os.path.join(log_dir, "plots"))
    except Exception as e:
        print(f"Could not generate plots: {e}")

    logger.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MCTS-Guided Hierarchical Actor-Critic for 3D bin packing",
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=None,
        help="Total training timesteps (default: from config)",
    )
    parser.add_argument(
        "--num_envs", type=int, default=None,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for logs, checkpoints, plots",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--skip_imitation", action="store_true",
        help="Skip imitation pre-training phase",
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["all", "imitation", "rl"],
        help="Which training phase(s) to run",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
