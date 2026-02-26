"""
PPO training script for 3D online bin packing.

Implements the full Proximal Policy Optimization training loop with:
    - Vectorized environments (SyncVectorEnv or AsyncVectorEnv)
    - Decomposed action space with per-sub-action masking
    - Generalised Advantage Estimation (GAE, lambda=0.95)
    - PPO-clip objective with entropy bonus
    - Value function clipping
    - Gradient clipping (max_norm=0.5)
    - Learning rate scheduling (cosine annealing with warmup)
    - Periodic evaluation, checkpointing, and logging

Usage:
    python train.py
    python train.py --total_timesteps 10000000 --num_envs 32 --lr 1e-4
    python train.py --num_envs 64 --rollout_steps 512 --ppo_epochs 10

For HPC (e.g. 32-core node):
    python train.py --num_envs 32 --total_timesteps 20000000

References:
    - Schulman et al. (2017): Proximal Policy Optimization Algorithms
    - Zhao et al. (ICLR 2022): PCT decomposed action for online 3D-BPP
    - Xiong et al. (RA-L 2024): GOPT masked PPO for bin packing
    - Andrychowicz et al. (2021): What Matters in On-Policy RL
"""

from __future__ import annotations

import sys
import os
import argparse
import copy
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from config import BinConfig, Orientation
from strategies.rl_ppo.config import PPOConfig
from strategies.rl_ppo.network import ActorCritic
from strategies.rl_common.environment import BinPackingEnv, EnvConfig, make_env
from strategies.rl_common.rewards import RewardConfig
from strategies.rl_common.logger import TrainingLogger


# ─────────────────────────────────────────────────────────────────────────────
# Observation conversion utilities
# ─────────────────────────────────────────────────────────────────────────────

def obs_to_tensor(
    obs: Dict[str, np.ndarray],
    device: torch.device,
    ppo_config: PPOConfig,
) -> Dict[str, torch.Tensor]:
    """
    Convert a gymnasium observation dict to PyTorch tensors.

    Handles both single-env and vectorized-env observation shapes.
    Extracts the first grippable box as the 'current box' for the network.

    Args:
        obs:        Observation dict from BinPackingEnv.
        device:     Target PyTorch device.
        ppo_config: PPO configuration (for dimension info).

    Returns:
        Dict with:
            'heightmaps':      (batch, num_bins, grid_l, grid_w)
            'box_features':    (batch, 5) -- first grippable box
            'buffer_features': (batch, buffer_size, 5)
            'buffer_mask':     (batch, buffer_size) -- 1 where box exists
    """
    heightmaps = torch.as_tensor(obs['heightmaps'], dtype=torch.float32, device=device)
    box_features_raw = torch.as_tensor(obs['box_features'], dtype=torch.float32, device=device)
    buffer_features_raw = torch.as_tensor(obs['buffer_features'], dtype=torch.float32, device=device)

    # Handle single-env (no batch dim) vs vectorized (has batch dim)
    if heightmaps.dim() == 3:
        heightmaps = heightmaps.unsqueeze(0)
        box_features_raw = box_features_raw.unsqueeze(0)
        buffer_features_raw = buffer_features_raw.unsqueeze(0)

    batch = heightmaps.size(0)

    # Extract first grippable box as the "current box" (5 features)
    # box_features shape: (batch, pick_window, 4) -- we take first box and add weight=0.5
    current_box = box_features_raw[:, 0, :]  # (batch, 4)
    # Pad with a default weight feature (normalised)
    weight_feat = torch.full((batch, 1), 0.5, device=device, dtype=torch.float32)
    box_features = torch.cat([current_box, weight_feat], dim=-1)  # (batch, 5)

    # Buffer features: (batch, buffer_size, 4) -> pad to 5
    buf_weight = torch.full(
        (batch, buffer_features_raw.size(1), 1), 0.5,
        device=device, dtype=torch.float32,
    )
    buffer_features = torch.cat([buffer_features_raw, buf_weight], dim=-1)

    # Buffer mask: box is present if any feature > 0
    buffer_mask = (buffer_features_raw.abs().sum(dim=-1) > 1e-6).float()

    return {
        'heightmaps': heightmaps,
        'box_features': box_features,
        'buffer_features': buffer_features,
        'buffer_mask': buffer_mask,
    }


def compute_action_masks(
    obs: Dict[str, np.ndarray],
    ppo_config: PPOConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-sub-action masks from the flat action mask in the observation.

    The environment provides a flat action mask of shape (total_actions,).
    We decompose it into masks for each sub-action dimension:
        bin:    (batch, num_bins)
        x:      (batch, grid_l)
        y:      (batch, grid_w)
        orient: (batch, num_orientations)

    A sub-action value is valid if ANY combination involving it is valid.

    Args:
        obs:        Observation dict from BinPackingEnv.
        ppo_config: Configuration.
        device:     Target device.

    Returns:
        Dict with per-sub-action masks.
    """
    flat_mask = torch.as_tensor(obs['action_mask'], dtype=torch.float32, device=device)
    if flat_mask.dim() == 1:
        flat_mask = flat_mask.unsqueeze(0)

    batch = flat_mask.size(0)
    n_bins = ppo_config.num_bins
    gl = ppo_config.grid_l
    gw = ppo_config.grid_w
    n_orient = ppo_config.num_orientations
    pw = ppo_config.pick_window

    # Remove skip action (last element)
    placement_mask = flat_mask[:, :-1]  # (batch, pick_window * bins * gl * gw * orients)

    # We only use box_idx=0 (first grippable box) in our decomposed approach.
    # Extract the slice for box_idx=0.
    per_box_actions = n_bins * gl * gw * n_orient
    box0_mask = placement_mask[:, :per_box_actions]  # (batch, bins*gl*gw*orients)

    # Reshape to (batch, bins, gl, gw, orients)
    reshaped = box0_mask.view(batch, n_bins, gl, gw, n_orient)

    # Marginalise to per-dimension masks (any valid combination)
    bin_mask = reshaped.any(dim=-1).any(dim=-1).any(dim=-1).float()       # (batch, n_bins)
    x_mask = reshaped.any(dim=-1).any(dim=-1).any(dim=1).float()          # (batch, gl)
    y_mask = reshaped.any(dim=-1).any(dim=-2).any(dim=1).float()          # (batch, gw)
    orient_mask = reshaped.any(dim=-2).any(dim=-2).any(dim=1).float()     # (batch, n_orient)

    # Ensure at least one valid action per dimension (safety)
    for m in [bin_mask, x_mask, y_mask, orient_mask]:
        no_valid = (m.sum(dim=-1) == 0)
        if no_valid.any():
            m[no_valid] = 1.0  # Allow all if none valid (fallback)

    return {
        'bin': bin_mask,
        'x': x_mask,
        'y': y_mask,
        'orient': orient_mask,
    }


def decomposed_to_flat_action(
    actions: Dict[str, torch.Tensor],
    ppo_config: PPOConfig,
) -> np.ndarray:
    """
    Convert decomposed action dict to flat action indices for the environment.

    Maps (bin, x, y, orient) back to the flattened encoding used by
    BinPackingEnv, with box_idx=0 (always picking the first grippable box).

    Args:
        actions:    {'bin': (batch,), 'x': (batch,), 'y': (batch,), 'orient': (batch,)}
        ppo_config: Configuration.

    Returns:
        flat_actions: (batch,) numpy int array.
    """
    n_bins = ppo_config.num_bins
    gl = ppo_config.grid_l
    gw = ppo_config.grid_w
    n_orient = ppo_config.num_orientations

    bin_a = actions['bin'].cpu().numpy()
    x_a = actions['x'].cpu().numpy()
    y_a = actions['y'].cpu().numpy()
    orient_a = actions['orient'].cpu().numpy()

    # box_idx = 0 always (first grippable box)
    flat = (
        0 * (n_bins * gl * gw * n_orient)
        + bin_a * (gl * gw * n_orient)
        + x_a * (gw * n_orient)
        + y_a * n_orient
        + orient_a
    )
    return flat.astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores rollout data for PPO updates.

    Each slot stores one transition per environment per step.
    Total capacity: num_envs * rollout_steps.

    After collection, computes GAE advantages and normalises them.
    """

    def __init__(self, config: PPOConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.num_envs = config.num_envs
        self.rollout_steps = config.rollout_steps
        self.batch_size = config.batch_size
        self.pos = 0

        # Observation buffers (stored on CPU to save GPU memory, moved to device for updates)
        self.heightmaps = torch.zeros(
            (config.rollout_steps, config.num_envs, config.num_bins,
             config.grid_l, config.grid_w),
            dtype=torch.float32,
        )
        self.box_features = torch.zeros(
            (config.rollout_steps, config.num_envs, config.box_feat_dim),
            dtype=torch.float32,
        )
        self.buffer_features = torch.zeros(
            (config.rollout_steps, config.num_envs, config.buffer_size, config.box_feat_dim),
            dtype=torch.float32,
        )
        self.buffer_masks = torch.zeros(
            (config.rollout_steps, config.num_envs, config.buffer_size),
            dtype=torch.float32,
        )

        # Action buffers
        self.actions_bin = torch.zeros(
            (config.rollout_steps, config.num_envs), dtype=torch.long,
        )
        self.actions_x = torch.zeros(
            (config.rollout_steps, config.num_envs), dtype=torch.long,
        )
        self.actions_y = torch.zeros(
            (config.rollout_steps, config.num_envs), dtype=torch.long,
        )
        self.actions_orient = torch.zeros(
            (config.rollout_steps, config.num_envs), dtype=torch.long,
        )

        # Action masks
        self.masks_bin = torch.zeros(
            (config.rollout_steps, config.num_envs, config.num_bins),
            dtype=torch.float32,
        )
        self.masks_x = torch.zeros(
            (config.rollout_steps, config.num_envs, config.grid_l),
            dtype=torch.float32,
        )
        self.masks_y = torch.zeros(
            (config.rollout_steps, config.num_envs, config.grid_w),
            dtype=torch.float32,
        )
        self.masks_orient = torch.zeros(
            (config.rollout_steps, config.num_envs, config.num_orientations),
            dtype=torch.float32,
        )

        # Scalar buffers
        self.log_probs = torch.zeros(
            (config.rollout_steps, config.num_envs), dtype=torch.float32,
        )
        self.values = torch.zeros(
            (config.rollout_steps, config.num_envs), dtype=torch.float32,
        )
        self.rewards = torch.zeros(
            (config.rollout_steps, config.num_envs), dtype=torch.float32,
        )
        self.dones = torch.zeros(
            (config.rollout_steps, config.num_envs), dtype=torch.float32,
        )
        self.advantages = torch.zeros(
            (config.rollout_steps, config.num_envs), dtype=torch.float32,
        )
        self.returns = torch.zeros(
            (config.rollout_steps, config.num_envs), dtype=torch.float32,
        )

    def reset(self) -> None:
        """Reset buffer position for new rollout."""
        self.pos = 0

    def add(
        self,
        obs_tensor: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        action_masks: Dict[str, torch.Tensor],
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """
        Store one step of transitions for all environments.

        Args:
            obs_tensor:   Observation tensors from obs_to_tensor().
            actions:      Decomposed action dict.
            action_masks: Per-sub-action masks.
            log_prob:     (num_envs,) log-probabilities.
            value:        (num_envs,) value estimates.
            reward:       (num_envs,) rewards.
            done:         (num_envs,) done flags.
        """
        t = self.pos

        self.heightmaps[t] = obs_tensor['heightmaps'].cpu()
        self.box_features[t] = obs_tensor['box_features'].cpu()
        self.buffer_features[t] = obs_tensor['buffer_features'].cpu()
        self.buffer_masks[t] = obs_tensor['buffer_mask'].cpu()

        self.actions_bin[t] = actions['bin'].cpu()
        self.actions_x[t] = actions['x'].cpu()
        self.actions_y[t] = actions['y'].cpu()
        self.actions_orient[t] = actions['orient'].cpu()

        self.masks_bin[t] = action_masks['bin'].cpu()
        self.masks_x[t] = action_masks['x'].cpu()
        self.masks_y[t] = action_masks['y'].cpu()
        self.masks_orient[t] = action_masks['orient'].cpu()

        self.log_probs[t] = log_prob.cpu()
        self.values[t] = value.cpu()
        self.rewards[t] = torch.as_tensor(reward, dtype=torch.float32)
        self.dones[t] = torch.as_tensor(done, dtype=torch.float32)

        self.pos += 1

    def compute_gae(self, last_value: torch.Tensor) -> None:
        """
        Compute Generalised Advantage Estimation (GAE).

        GAE(gamma, lambda) = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        Also computes returns = advantages + values (for value loss).

        Args:
            last_value: (num_envs,) V(s_T) -- value of final observation.
        """
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda

        last_gae = torch.zeros(self.num_envs, dtype=torch.float32)
        last_val = last_value.cpu()

        for t in reversed(range(self.rollout_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + gamma * last_val * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
            last_val = self.values[t]

        self.returns = self.advantages + self.values

    def get_minibatch_indices(self) -> List[np.ndarray]:
        """
        Generate random mini-batch index arrays.

        Flattens (rollout_steps, num_envs) -> batch_size, then shuffles
        and splits into num_minibatches chunks.

        Returns:
            List of index arrays, one per mini-batch.
        """
        indices = np.arange(self.batch_size)
        np.random.shuffle(indices)
        mb_size = self.config.minibatch_size
        return [indices[i:i + mb_size] for i in range(0, self.batch_size, mb_size)]

    def get_minibatch(
        self,
        indices: np.ndarray,
        device: torch.device,
    ) -> Tuple[
        Dict[str, torch.Tensor],   # obs
        Dict[str, torch.Tensor],   # actions
        Dict[str, torch.Tensor],   # action_masks
        torch.Tensor,              # old_log_probs
        torch.Tensor,              # old_values
        torch.Tensor,              # advantages
        torch.Tensor,              # returns
    ]:
        """
        Retrieve a mini-batch by flattened indices.

        Flattens the (rollout_steps, num_envs) layout, indexes, and moves
        tensors to the target device.

        Args:
            indices: 1D index array into the flattened buffer.
            device:  Target device.

        Returns:
            Tuple of (obs, actions, masks, old_log_probs, old_values, advantages, returns).
        """
        # Flatten first two dims and index
        def _flat(t: torch.Tensor) -> torch.Tensor:
            shape = t.shape
            flat = t.view(shape[0] * shape[1], *shape[2:])
            return flat[indices].to(device)

        obs = {
            'heightmaps': _flat(self.heightmaps),
            'box_features': _flat(self.box_features),
            'buffer_features': _flat(self.buffer_features),
            'buffer_mask': _flat(self.buffer_masks),
        }

        actions_dict = {
            'bin': _flat(self.actions_bin),
            'x': _flat(self.actions_x),
            'y': _flat(self.actions_y),
            'orient': _flat(self.actions_orient),
        }

        action_masks = {
            'bin': _flat(self.masks_bin),
            'x': _flat(self.masks_x),
            'y': _flat(self.masks_y),
            'orient': _flat(self.masks_orient),
        }

        old_log_probs = _flat(self.log_probs)
        old_values = _flat(self.values)
        advantages = _flat(self.advantages)
        returns = _flat(self.returns)

        return obs, actions_dict, action_masks, old_log_probs, old_values, advantages, returns


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate scheduler
# ─────────────────────────────────────────────────────────────────────────────

def get_lr_scheduler(
    optimizer: optim.Optimizer,
    config: PPOConfig,
) -> optim.lr_scheduler._LRScheduler:
    """
    Create LR scheduler based on config.lr_schedule.

    Supports:
        'cosine':   CosineAnnealingLR with optional warmup
        'linear':   LinearLR from lr to 0
        'constant': No scheduling

    Args:
        optimizer: PyTorch optimiser.
        config:    PPO config.

    Returns:
        LR scheduler instance.
    """
    total_updates = config.num_updates

    if config.lr_schedule == 'cosine':
        warmup_steps = max(1, int(total_updates * config.lr_warmup_frac))
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates - warmup_steps, eta_min=1e-6,
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
        )
    elif config.lr_schedule == 'linear':
        return optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_updates,
        )
    else:
        return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_updates)


# ─────────────────────────────────────────────────────────────────────────────
# PPO update step
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    config: PPOConfig,
    device: torch.device,
) -> Dict[str, float]:
    """
    Perform one PPO update over the collected rollout.

    Iterates over ppo_epochs, each time splitting the buffer into
    mini-batches and computing the PPO-clip loss.

    Loss = policy_loss + value_loss_coeff * value_loss - entropy_coeff * entropy

    Args:
        model:     Actor-Critic network.
        optimizer: Optimiser.
        buffer:    Filled rollout buffer.
        config:    PPO hyperparameters.
        device:    Compute device.

    Returns:
        Dict with averaged loss components:
            'policy_loss', 'value_loss', 'entropy', 'total_loss',
            'approx_kl', 'clip_fraction'
    """
    model.train()
    metrics = {k: 0.0 for k in [
        'policy_loss', 'value_loss', 'entropy', 'total_loss',
        'approx_kl', 'clip_fraction',
    ]}
    n_updates = 0

    for epoch in range(config.ppo_epochs):
        for mb_indices in buffer.get_minibatch_indices():
            (obs, actions, masks, old_log_probs,
             old_values, advantages, returns) = buffer.get_minibatch(mb_indices, device)

            # Normalise advantages
            if config.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Forward pass: evaluate old actions under current policy
            _, new_log_probs, entropy, new_values = model(
                obs, action_masks=masks, actions=actions,
            )

            # ── Policy loss (PPO clip) ──
            log_ratio = new_log_probs - old_log_probs
            ratio = log_ratio.exp()

            # Approximate KL for early stopping diagnostics
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > config.clip_ratio).float().mean().item()

            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(
                ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio,
            )
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            # ── Value loss ──
            if config.clip_value:
                v_clipped = old_values + torch.clamp(
                    new_values - old_values,
                    -config.clip_value_range,
                    config.clip_value_range,
                )
                v_loss1 = (new_values - returns) ** 2
                v_loss2 = (v_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
            else:
                value_loss = 0.5 * ((new_values - returns) ** 2).mean()

            # ── Entropy bonus ──
            entropy_loss = entropy.mean()

            # ── Total loss ──
            total_loss = (
                policy_loss
                + config.value_loss_coeff * value_loss
                - config.entropy_coeff * entropy_loss
            )

            # ── Optimise ──
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            # ── Accumulate metrics ──
            metrics['policy_loss'] += policy_loss.item()
            metrics['value_loss'] += value_loss.item()
            metrics['entropy'] += entropy_loss.item()
            metrics['total_loss'] += total_loss.item()
            metrics['approx_kl'] += approx_kl
            metrics['clip_fraction'] += clip_frac
            n_updates += 1

    # Average
    for k in metrics:
        metrics[k] /= max(n_updates, 1)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: ActorCritic,
    config: PPOConfig,
    device: torch.device,
    num_episodes: int = 10,
    seed: int = 99999,
) -> Dict[str, float]:
    """
    Run evaluation episodes with greedy (deterministic) actions.

    Creates a fresh single-env and runs num_episodes episodes, collecting
    fill rate, placement rate, and episode return statistics.

    Args:
        model:        Trained actor-critic.
        config:       PPO config.
        device:       Compute device.
        num_episodes: Number of episodes to evaluate.
        seed:         RNG seed for evaluation.

    Returns:
        Dict with 'mean_fill', 'std_fill', 'mean_return', 'mean_placement_rate'.
    """
    model.eval()

    env_config = _make_env_config(config)
    env_config.seed = seed
    env = BinPackingEnv(config=env_config)

    fills = []
    returns = []
    placement_rates = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_return = 0.0
        done = False

        while not done:
            with torch.no_grad():
                obs_t = obs_to_tensor(obs, device, config)
                masks = compute_action_masks(obs, config, device)
                actions, _, _, _ = model(obs_t, action_masks=masks, deterministic=True)

            flat = decomposed_to_flat_action(actions, config)
            action = int(flat[0])

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

        fills.append(info.get('final_avg_fill', 0.0))
        returns.append(ep_return)
        placement_rates.append(info.get('placement_rate', 0.0))

    model.train()
    return {
        'mean_fill': float(np.mean(fills)),
        'std_fill': float(np.std(fills)),
        'mean_return': float(np.mean(returns)),
        'mean_placement_rate': float(np.mean(placement_rates)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_env_config(config: PPOConfig) -> EnvConfig:
    """Create an EnvConfig from a PPOConfig."""
    return EnvConfig(
        bin_config=BinConfig(
            length=config.bin_length,
            width=config.bin_width,
            height=config.bin_height,
            resolution=config.bin_resolution,
        ),
        num_bins=config.num_bins,
        buffer_size=config.buffer_size,
        pick_window=config.pick_window,
        close_height=config.close_height,
        max_consecutive_rejects=config.max_consecutive_rejects,
        num_orientations=config.num_orientations,
        num_boxes_per_episode=config.num_boxes_per_episode,
        box_size_range=config.box_size_range,
        box_weight_range=config.box_weight_range,
        reward_config=config.reward_config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config: PPOConfig) -> str:
    """
    Full PPO training loop.

    Steps:
        1. Create vectorized environments
        2. Initialise actor-critic network and optimiser
        3. For each rollout:
            a. Collect rollout_steps transitions per environment
            b. Compute GAE advantages
            c. Run ppo_epochs of mini-batch updates
            d. Log metrics and optionally evaluate
            e. Save checkpoints periodically
        4. Final evaluation and cleanup

    Args:
        config: Full PPO configuration.

    Returns:
        Path to the best checkpoint.
    """
    # ── Device setup ──
    if config.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.device)
    print(f"[PPO] Device: {device}")

    # ── Reproducibility ──
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ── Directories ──
    log_dir = os.path.abspath(config.log_dir)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Logger ──
    logger = TrainingLogger(
        log_dir=log_dir,
        strategy_name="rl_ppo",
        use_tensorboard=config.use_tensorboard,
    )
    logger.log_config(config.to_dict())

    # ── Environments ──
    print(f"[PPO] Creating {config.num_envs} parallel environments...")
    env_config = _make_env_config(config)
    envs = make_env(env_config, num_envs=config.num_envs, seed=config.seed)

    # ── Model ──
    model = ActorCritic(config).to(device)
    param_count = model.count_parameters()
    print(f"[PPO] Model parameters: {param_count:,}")

    # ── Optimiser + scheduler ──
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    lr_scheduler = get_lr_scheduler(optimizer, config)

    # ── Rollout buffer ──
    rollout_buffer = RolloutBuffer(config, device)

    # ── Training state ──
    global_step = 0
    best_fill = 0.0
    best_ckpt_path = ""
    num_updates = config.num_updates

    print(f"[PPO] Training for {config.total_timesteps:,} total timesteps "
          f"({num_updates} updates, batch_size={config.batch_size})")
    print(f"[PPO] Rollout: {config.rollout_steps} steps x {config.num_envs} envs, "
          f"{config.ppo_epochs} epochs x {config.num_minibatches} minibatches")

    t_start = time.time()

    # Initial observation
    obs, info = envs.reset()
    episode_returns = np.zeros(config.num_envs)
    episode_lengths = np.zeros(config.num_envs, dtype=np.int64)
    completed_episodes = 0
    recent_fills: List[float] = []
    recent_returns: List[float] = []

    for update in range(1, num_updates + 1):
        rollout_buffer.reset()
        model.eval()  # BN in eval mode during rollout (stable statistics)

        # ── Collect rollout ──
        for step in range(config.rollout_steps):
            with torch.no_grad():
                obs_t = obs_to_tensor(obs, device, config)
                masks = compute_action_masks(obs, config, device)
                actions, log_probs, entropy, values = model(
                    obs_t, action_masks=masks,
                )

            # Convert to flat actions for the environment
            flat_actions = decomposed_to_flat_action(actions, config)

            # Step environments
            next_obs, rewards, terminateds, truncateds, infos = envs.step(flat_actions)
            dones = np.logical_or(terminateds, truncateds)

            # Store transition
            rollout_buffer.add(
                obs_tensor=obs_t,
                actions=actions,
                action_masks=masks,
                log_prob=log_probs,
                value=values,
                reward=rewards,
                done=dones,
            )

            # Track episode stats
            episode_returns += rewards
            episode_lengths += 1
            global_step += config.num_envs

            # Handle completed episodes
            for i in range(config.num_envs):
                if dones[i]:
                    completed_episodes += 1
                    recent_returns.append(float(episode_returns[i]))
                    # Extract fill rate from info
                    if isinstance(infos, dict):
                        # Vectorized env: infos is a dict of arrays
                        fill = infos.get('final_avg_fill', np.zeros(config.num_envs))[i]
                    else:
                        fill = infos[i].get('final_avg_fill', 0.0) if isinstance(infos, list) else 0.0
                    recent_fills.append(float(fill))
                    episode_returns[i] = 0.0
                    episode_lengths[i] = 0

            obs = next_obs

        # ── Compute GAE ──
        with torch.no_grad():
            obs_t = obs_to_tensor(obs, device, config)
            last_values = model.get_value(obs_t)
        rollout_buffer.compute_gae(last_values)

        # ── PPO update ──
        model.train()
        update_metrics = ppo_update(model, optimizer, rollout_buffer, config, device)
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # ── Logging ──
        if update % config.log_interval == 0 and recent_returns:
            avg_return = float(np.mean(recent_returns[-100:]))
            avg_fill = float(np.mean(recent_fills[-100:])) if recent_fills else 0.0

            logger.log_episode(
                episode=completed_episodes,
                reward=avg_return,
                fill=avg_fill,
                loss=update_metrics['total_loss'],
                policy_loss=update_metrics['policy_loss'],
                value_loss=update_metrics['value_loss'],
                entropy=update_metrics['entropy'],
                approx_kl=update_metrics['approx_kl'],
                clip_fraction=update_metrics['clip_fraction'],
                lr=current_lr,
                global_step=global_step,
            )

            logger.print_progress(
                episode=completed_episodes,
                total_episodes=num_updates * 10,  # approximate
                reward=avg_return,
                fill=avg_fill,
                loss=update_metrics['total_loss'],
                entropy=update_metrics['entropy'],
                lr=current_lr,
                steps=global_step,
            )

        # ── Evaluation ──
        if update % config.eval_interval == 0:
            eval_metrics = evaluate(model, config, device, config.eval_episodes)
            print(f"  [EVAL] fill={eval_metrics['mean_fill']:.3f} +/- {eval_metrics['std_fill']:.3f} "
                  f"| return={eval_metrics['mean_return']:.2f} "
                  f"| placement={eval_metrics['mean_placement_rate']:.2%}")

            logger.log_episode(
                episode=completed_episodes,
                eval_fill=eval_metrics['mean_fill'],
                eval_return=eval_metrics['mean_return'],
                eval_placement_rate=eval_metrics['mean_placement_rate'],
            )

            # Track best model
            if eval_metrics['mean_fill'] > best_fill:
                best_fill = eval_metrics['mean_fill']
                best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
                _save_checkpoint(model, optimizer, config, update, global_step,
                                 best_fill, best_ckpt_path)
                print(f"  [BEST] New best fill: {best_fill:.4f} -> {best_ckpt_path}")

        # ── Periodic checkpoint ──
        if update % config.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{update:06d}.pt")
            _save_checkpoint(model, optimizer, config, update, global_step,
                             best_fill, ckpt_path)

    # ── Final cleanup ──
    elapsed = time.time() - t_start
    print(f"\n[PPO] Training complete in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"[PPO] Total steps: {global_step:,}, Episodes: {completed_episodes:,}")
    print(f"[PPO] Best eval fill: {best_fill:.4f}")

    # Final checkpoint
    final_path = os.path.join(ckpt_dir, "final_model.pt")
    _save_checkpoint(model, optimizer, config, num_updates, global_step,
                     best_fill, final_path)

    # Training curves
    logger.plot_training_curves()
    logger.close()

    # Close envs
    try:
        envs.close()
    except Exception:
        pass

    print(f"[PPO] Logs saved to: {log_dir}")
    resolved_best = best_ckpt_path or final_path
    print(f"[PPO] Best checkpoint: {resolved_best}")

    return resolved_best


def _save_checkpoint(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    config: PPOConfig,
    update: int,
    global_step: int,
    best_fill: float,
    path: str,
) -> None:
    """Save a training checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.to_dict(),
        'update': update,
        'global_step': global_step,
        'best_fill': best_fill,
    }, path)


def load_checkpoint(
    path: str,
    config: PPOConfig,
    device: torch.device,
) -> Tuple[ActorCritic, Dict]:
    """
    Load a trained model from checkpoint.

    Args:
        path:   Checkpoint file path.
        config: PPO config (must match architecture).
        device: Target device.

    Returns:
        (model, checkpoint_dict) tuple.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> PPOConfig:
    """Parse command-line arguments into a PPOConfig."""
    parser = argparse.ArgumentParser(
        description="PPO training for 3D online bin packing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Most important arguments exposed as CLI flags
    parser.add_argument("--total_timesteps", type=int, default=5_000_000,
                        help="Total environment timesteps")
    parser.add_argument("--num_envs", type=int, default=16,
                        help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--rollout_steps", type=int, default=256,
                        help="Steps per env per rollout")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="PPO epochs per update")
    parser.add_argument("--num_minibatches", type=int, default=8,
                        help="Mini-batches per epoch")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                        help="PPO clip epsilon")
    parser.add_argument("--entropy_coeff", type=float, default=0.01,
                        help="Entropy bonus coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Gradient clipping norm")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="PyTorch device")
    parser.add_argument("--log_dir", type=str, default="outputs/rl_ppo/logs",
                        help="Log directory")
    parser.add_argument("--num_boxes", type=int, default=100,
                        help="Boxes per episode")
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "linear", "constant"],
                        help="LR schedule")
    parser.add_argument("--no_tensorboard", action="store_true",
                        help="Disable TensorBoard logging")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Evaluation episodes per cycle")

    args = parser.parse_args()

    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        learning_rate=args.lr,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        num_minibatches=args.num_minibatches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coeff=args.entropy_coeff,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
        num_boxes_per_episode=args.num_boxes,
        lr_schedule=args.lr_schedule,
        use_tensorboard=not args.no_tensorboard,
        eval_episodes=args.eval_episodes,
    )

    return config


if __name__ == "__main__":
    ppo_config = parse_args()
    train(ppo_config)
