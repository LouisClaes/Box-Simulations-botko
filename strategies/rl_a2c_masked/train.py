"""
A2C training script with feasibility masking for 3D online bin packing.

On-policy training loop:
    1. Collect rollouts from N parallel environments (5 steps each)
    2. Compute N-step returns with GAE (Generalised Advantage Estimation)
    3. Compute ground-truth action masks at each step (for L_mask supervision)
    4. Update all three heads simultaneously with the 5-component loss

Key features:
    - 16 parallel environments (SyncVectorEnv / AsyncVectorEnv)
    - Curriculum learning: progressive difficulty over training
    - Linear LR decay from 1e-4 to 0
    - Gradient clipping (max_norm=0.5)
    - Checkpoint every 5000 updates
    - Full TrainingLogger integration (CSV, TensorBoard, matplotlib)
    - Evaluation episodes at regular intervals

Usage:
    python train.py
    python train.py --num_updates 200000 --num_envs 16 --lr 1e-4
    python train.py --device cuda --log_dir outputs/rl_a2c_masked/run1

References:
    - Zhao et al. (AAAI 2021): Feasibility masking for online 3D BPP
    - Mnih et al. (2016): A3C — asynchronous advantage actor-critic
    - Schulman et al. (2016): High-dimensional continuous control with GAE
"""

from __future__ import annotations

import sys
import os
import time
import copy
import argparse
import json
from typing import List, Optional, Dict, Any, Tuple

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

import numpy as np
import torch
import torch.optim as optim

from strategies.rl_a2c_masked.config import A2CMaskedConfig, CurriculumPhase
from strategies.rl_a2c_masked.network import A2CMaskedNetwork, A2CMaskedLoss, resolve_device
from strategies.rl_common.environment import BinPackingEnv, EnvConfig, generate_random_boxes
from strategies.rl_common.rewards import RewardConfig
from strategies.rl_common.logger import TrainingLogger
from config import BinConfig


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------

class RolloutStorage:
    """
    Fixed-size buffer for on-policy rollout data.

    Stores transitions for ``rollout_steps`` steps across ``num_envs``
    environments.  After collection, computes GAE advantages and discounted
    returns for the A2C update.

    Shape conventions (all torch tensors):
        heightmaps:    (T, N, num_bins, grid_l, grid_w)
        item_features: (T, N, 5)
        actions:       (T, N)
        log_probs:     (T, N)
        rewards:       (T, N)
        dones:         (T, N)
        values:        (T, N)
        true_masks:    (T, N, num_actions)
        mask_preds:    (T, N, num_actions)
        policies:      (T, N, num_actions)

    where T = rollout_steps, N = num_envs.
    """

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        config: A2CMaskedConfig,
        device: torch.device,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device
        self.config = config

        T, N = rollout_steps, num_envs
        num_bins = config.num_bins
        grid_l = config.grid_l
        grid_w = config.grid_w
        num_actions = config.num_actions

        # Observations
        self.heightmaps = torch.zeros(T, N, num_bins, grid_l, grid_w, device=device)
        self.item_features = torch.zeros(T, N, 5, device=device)

        # Actions and log probs
        self.actions = torch.zeros(T, N, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(T, N, device=device)

        # Rewards and dones
        self.rewards = torch.zeros(T, N, device=device)
        self.dones = torch.zeros(T, N, device=device)

        # Values
        self.values = torch.zeros(T, N, device=device)

        # Masks and policies
        self.true_masks = torch.zeros(T, N, num_actions, device=device)
        self.mask_preds = torch.zeros(T, N, num_actions, device=device)
        self.policies = torch.zeros(T, N, num_actions, device=device)

        # Entropy
        self.entropies = torch.zeros(T, N, device=device)

        # Computed after rollout
        self.advantages = torch.zeros(T, N, device=device)
        self.returns = torch.zeros(T, N, device=device)

        self.step_idx = 0

    def insert(
        self,
        heightmaps: torch.Tensor,
        item_features: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        true_masks: torch.Tensor,
        mask_preds: torch.Tensor,
        policies: torch.Tensor,
        entropies: torch.Tensor,
    ) -> None:
        """Insert one step of rollout data."""
        t = self.step_idx
        self.heightmaps[t] = heightmaps
        self.item_features[t] = item_features
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self.true_masks[t] = true_masks
        self.mask_preds[t] = mask_preds
        self.policies[t] = policies
        self.entropies[t] = entropies
        self.step_idx += 1

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """
        Compute GAE advantages and discounted returns.

        Uses the bootstrap value estimate V(s_{T+1}) from the last state.

        Args:
            last_value: (N,) value estimate for the state after the last step.
            gamma:      Discount factor.
            gae_lambda: GAE lambda parameter.
        """
        T = self.rollout_steps
        gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            # TD error: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]

            # GAE: A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        # Returns = advantages + values
        self.returns = self.advantages + self.values

    def flatten(self) -> Dict[str, torch.Tensor]:
        """
        Flatten (T, N, ...) tensors to (T*N, ...) for the update step.

        Returns:
            Dictionary of flattened tensors.
        """
        B = self.rollout_steps * self.num_envs

        return {
            "heightmaps": self.heightmaps.view(B, *self.heightmaps.shape[2:]),
            "item_features": self.item_features.view(B, -1),
            "actions": self.actions.view(B),
            "log_probs": self.log_probs.view(B),
            "rewards": self.rewards.view(B),
            "values": self.values.view(B),
            "true_masks": self.true_masks.view(B, -1),
            "mask_preds": self.mask_preds.view(B, -1),
            "policies": self.policies.view(B, -1),
            "entropies": self.entropies.view(B),
            "advantages": self.advantages.view(B),
            "returns": self.returns.view(B),
        }

    def reset(self) -> None:
        """Reset step index for next rollout."""
        self.step_idx = 0


# ---------------------------------------------------------------------------
# Environment wrappers for A2C
# ---------------------------------------------------------------------------

def _build_obs_tensors(
    obs_dict: Dict[str, np.ndarray],
    config: A2CMaskedConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert environment observation dict to tensors for the network.

    For the A2C network, we need:
        - heightmaps: (num_bins, grid_l, grid_w)
        - item_features: (5,) — first grippable box features
        - action_mask: (num_actions,)

    The environment's action space may differ from the network's (the env
    uses its own grid step).  We re-map the mask to the network's coarse grid.

    Args:
        obs_dict:  Observation from BinPackingEnv.
        config:    A2CMaskedConfig.
        device:    Target torch device.

    Returns:
        (heightmaps, item_features, action_mask) tensors.
    """
    heightmaps = torch.from_numpy(obs_dict["heightmaps"]).float().to(device)
    # item_features: first grippable box from box_features
    box_feats = obs_dict["box_features"]  # (pick_window, 4)
    # Extend to 5 features: (l, w, h, vol, weight=0.5 default)
    first_box = box_feats[0] if box_feats.shape[0] > 0 else np.zeros(4, dtype=np.float32)
    item_feat = np.array([first_box[0], first_box[1], first_box[2], first_box[3], 0.5],
                         dtype=np.float32)
    item_features = torch.from_numpy(item_feat).float().to(device)

    # Action mask: environment may have different action space shape
    # We need to recompute for the A2C coarse grid
    action_mask = torch.from_numpy(obs_dict["action_mask"]).float().to(device)

    return heightmaps, item_features, action_mask


def _remap_action_mask(
    env_mask: np.ndarray,
    env_config: EnvConfig,
    a2c_config: A2CMaskedConfig,
) -> np.ndarray:
    """
    Remap the environment's action mask to the A2C coarse action space.

    The environment uses action_grid_step (e.g. 10mm, total_actions including
    pick_window and skip action).  The A2C network uses a coarser grid
    (e.g. 50mm, num_actions without pick_window or skip).

    Strategy: For each coarse grid cell, check if ANY of the corresponding
    fine grid cells are valid.

    Args:
        env_mask:   Full environment action mask.
        env_config: Environment configuration.
        a2c_config: A2C network configuration.

    Returns:
        Coarse action mask of shape (a2c_config.num_actions,).
    """
    coarse_mask = np.zeros(a2c_config.num_actions, dtype=np.float32)
    fine_step = env_config.action_grid_step
    coarse_step = a2c_config.action_grid_step
    ratio = int(coarse_step / fine_step)

    for bin_idx in range(a2c_config.num_bins):
        for orient_idx in range(a2c_config.num_orientations):
            for cgx in range(a2c_config.action_grid_l):
                for cgy in range(a2c_config.action_grid_w):
                    # Check any fine grid cell in this coarse cell
                    valid = False
                    for dx in range(ratio):
                        if valid:
                            break
                        fgx = cgx * ratio + dx
                        if fgx >= env_config.action_grid_l:
                            continue
                        for dy in range(ratio):
                            fgy = cgy * ratio + dy
                            if fgy >= env_config.action_grid_w:
                                continue
                            # Encode for box_idx=0 (first grippable)
                            fine_idx = (
                                0 * (env_config.num_bins * env_config.action_grid_l
                                     * env_config.action_grid_w * env_config.num_orientations)
                                + bin_idx * (env_config.action_grid_l
                                             * env_config.action_grid_w
                                             * env_config.num_orientations)
                                + fgx * (env_config.action_grid_w * env_config.num_orientations)
                                + fgy * env_config.num_orientations
                                + orient_idx
                            )
                            if fine_idx < len(env_mask) and env_mask[fine_idx] > 0.5:
                                valid = True
                                break
                    if valid:
                        coarse_idx = (
                            bin_idx * (a2c_config.action_grid_l
                                       * a2c_config.action_grid_w
                                       * a2c_config.num_orientations)
                            + cgx * (a2c_config.action_grid_w * a2c_config.num_orientations)
                            + cgy * a2c_config.num_orientations
                            + orient_idx
                        )
                        coarse_mask[coarse_idx] = 1.0
    return coarse_mask


def _decode_a2c_action(
    action: int,
    config: A2CMaskedConfig,
) -> Tuple[int, int, int, int]:
    """
    Decode A2C network action index to (bin_idx, gx, gy, orient_idx).

    The A2C action space does NOT include pick_window or skip.
    It always acts on box_idx=0 (first grippable).

    Args:
        action: Flat action index in [0, num_actions).
        config: A2CMaskedConfig.

    Returns:
        (bin_idx, gx, gy, orient_idx) in the coarse grid.
    """
    n_orient = config.num_orientations
    n_gy = config.action_grid_w
    n_gx = config.action_grid_l

    orient_idx = action % n_orient
    action //= n_orient
    gy = action % n_gy
    action //= n_gy
    gx = action % n_gx
    action //= n_gx
    bin_idx = action

    return bin_idx, gx, gy, orient_idx


def _a2c_to_env_action(
    a2c_action: int,
    a2c_config: A2CMaskedConfig,
    env_config: EnvConfig,
) -> int:
    """
    Convert an A2C coarse-grid action to the environment's fine-grid action.

    Maps coarse (bin_idx, gx_coarse, gy_coarse, orient) to
    fine (box_idx=0, bin_idx, gx_fine, gy_fine, orient).

    Args:
        a2c_action: A2C action index.
        a2c_config: A2C configuration.
        env_config: Environment configuration.

    Returns:
        Environment action index.
    """
    bin_idx, cgx, cgy, orient_idx = _decode_a2c_action(a2c_action, a2c_config)

    ratio = int(a2c_config.action_grid_step / env_config.action_grid_step)
    fgx = cgx * ratio
    fgy = cgy * ratio

    # Clamp to valid range
    fgx = min(fgx, env_config.action_grid_l - 1)
    fgy = min(fgy, env_config.action_grid_w - 1)

    # Encode as environment action with box_idx=0
    env_action = (
        0 * (env_config.num_bins * env_config.action_grid_l
             * env_config.action_grid_w * env_config.num_orientations)
        + bin_idx * (env_config.action_grid_l * env_config.action_grid_w
                     * env_config.num_orientations)
        + fgx * (env_config.action_grid_w * env_config.num_orientations)
        + fgy * env_config.num_orientations
        + orient_idx
    )
    return env_action


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    network: A2CMaskedNetwork,
    config: A2CMaskedConfig,
    num_episodes: int = 10,
    seed: int = 9999,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Run deterministic evaluation episodes and return aggregate metrics.

    Args:
        network:      Trained A2C network (set to eval mode).
        config:       A2CMaskedConfig.
        num_episodes: Number of evaluation episodes.
        seed:         Random seed for reproducibility.
        device:       Torch device.

    Returns:
        Dictionary with avg_fill, avg_reward, avg_placements, etc.
    """
    if device is None:
        device = resolve_device(config.device)

    network.eval()

    env_config = _make_env_config(config, phase=None, seed=seed)
    env = BinPackingEnv(config=env_config)

    fills = []
    rewards_total = []
    placements = []
    pallets_closed_list = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            hm, item_feat, mask_raw = _build_obs_tensors(obs, config, device)
            coarse_mask = _remap_action_mask(
                obs["action_mask"], env_config, config,
            )
            coarse_mask_t = torch.from_numpy(coarse_mask).float().unsqueeze(0).to(device)

            action, _, _, _, _ = network.get_action_and_value(
                hm.unsqueeze(0),
                item_feat.unsqueeze(0),
                true_mask=coarse_mask_t,
                deterministic=True,
            )

            a2c_act = action.item()
            env_act = _a2c_to_env_action(a2c_act, config, env_config)

            # If environment mask says this action is invalid, fallback to skip
            if env_act >= len(obs["action_mask"]) or obs["action_mask"][env_act] < 0.5:
                env_act = env_config.total_actions - 1  # skip

            obs, reward, terminated, truncated, info = env.step(env_act)
            ep_reward += reward
            done = terminated or truncated

        fills.append(info.get("final_avg_fill", 0.0))
        rewards_total.append(ep_reward)
        placements.append(info.get("total_placed", 0))
        pallets_closed_list.append(info.get("pallets_closed", 0))

    network.train()

    return {
        "eval/avg_fill": float(np.mean(fills)),
        "eval/avg_reward": float(np.mean(rewards_total)),
        "eval/avg_placements": float(np.mean(placements)),
        "eval/avg_pallets_closed": float(np.mean(pallets_closed_list)),
        "eval/fill_std": float(np.std(fills)),
    }


# ---------------------------------------------------------------------------
# Environment config factory
# ---------------------------------------------------------------------------

def _make_env_config(
    config: A2CMaskedConfig,
    phase: Optional[CurriculumPhase] = None,
    seed: int = 42,
) -> EnvConfig:
    """
    Create an EnvConfig from A2CMaskedConfig, optionally applying curriculum phase.

    The environment still uses the fine grid (action_grid_step from env defaults).
    The A2C agent operates on a coarser grid and maps back.
    """
    num_boxes = config.num_boxes_per_episode
    size_range = config.box_size_range

    if phase is not None:
        num_boxes = phase.num_boxes
        size_range = phase.size_range

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
        action_grid_step=config.bin_resolution,  # Fine grid for environment
        num_boxes_per_episode=num_boxes,
        box_size_range=size_range,
        box_weight_range=config.box_weight_range,
        reward_config=config.reward_config,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def _get_lr(
    update: int,
    total_updates: int,
    initial_lr: float,
    schedule: str = "linear",
) -> float:
    """
    Compute learning rate for the given update.

    Args:
        update:        Current update number.
        total_updates: Total number of updates.
        initial_lr:    Starting learning rate.
        schedule:      'linear', 'cosine', or 'constant'.

    Returns:
        Learning rate for this update.
    """
    if schedule == "constant":
        return initial_lr
    frac = 1.0 - (update / max(total_updates, 1))
    if schedule == "cosine":
        import math
        frac = 0.5 * (1.0 + math.cos(math.pi * update / max(total_updates, 1)))
    return max(initial_lr * frac, 1e-7)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: A2CMaskedConfig) -> str:
    """
    Full A2C training with feasibility masking.

    Training loop:
        for update in range(num_updates):
            1. Get curriculum phase (box count, size range)
            2. Collect rollout (rollout_steps * num_envs transitions)
            3. Compute GAE advantages and returns
            4. Forward pass through network
            5. Compute 5-component loss
            6. Backprop with gradient clipping
            7. Log metrics
            8. Periodically evaluate and checkpoint

    Args:
        config: A2CMaskedConfig with all hyperparameters.

    Returns:
        Path to the best checkpoint.
    """
    device = resolve_device(config.device)
    print(f"[A2C-Masked] Training on device: {device}")
    print(f"[A2C-Masked] Action space: {config.num_actions} actions "
          f"({config.action_grid_l}x{config.action_grid_w}x"
          f"{config.num_orientations}x{config.num_bins})")
    print(f"[A2C-Masked] Total timesteps: {config.total_timesteps:,} "
          f"({config.num_updates:,} updates x {config.batch_size} batch)")

    # ── Setup ──────────────────────────────────────────────────────────────
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create network and loss
    network = A2CMaskedNetwork(config).to(device)
    loss_fn = A2CMaskedLoss(config)
    optimizer = optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
        eps=1e-5,
    )

    param_count = sum(p.numel() for p in network.parameters())
    trainable_count = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"[A2C-Masked] Network parameters: {param_count:,} "
          f"(trainable: {trainable_count:,})")

    # Logger
    os.makedirs(config.log_dir, exist_ok=True)
    logger = TrainingLogger(
        log_dir=config.log_dir,
        strategy_name="rl_a2c_masked",
        use_tensorboard=config.use_tensorboard,
    )
    logger.log_config(config.to_dict())

    # Checkpoint directory
    ckpt_dir = os.path.join(config.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Create environments (sequential for simplicity; parallel for speed)
    envs = []
    env_configs = []
    phase = config.get_curriculum_phase(0)
    for i in range(config.num_envs):
        ec = _make_env_config(config, phase=phase, seed=config.seed + i)
        env_configs.append(ec)
        envs.append(BinPackingEnv(config=ec))

    # Rollout storage
    storage = RolloutStorage(
        rollout_steps=config.rollout_steps,
        num_envs=config.num_envs,
        config=config,
        device=device,
    )

    # ── Initial reset ──────────────────────────────────────────────────────
    obs_list = []
    for env in envs:
        obs, _ = env.reset()
        obs_list.append(obs)

    # Tracking
    best_eval_fill = 0.0
    best_ckpt_path = ""
    global_step = 0
    episode_count = 0
    start_time = time.time()

    # Per-env episode tracking
    ep_rewards = [0.0] * config.num_envs
    ep_fills: List[float] = []
    ep_reward_history: List[float] = []

    print(f"\n[A2C-Masked] Starting training...")
    print(f"{'='*80}")

    # ── Training loop ──────────────────────────────────────────────────────
    for update in range(config.num_updates):
        # ── Curriculum phase ──
        new_phase = config.get_curriculum_phase(update)
        if (new_phase.num_boxes != phase.num_boxes or
                new_phase.size_range != phase.size_range):
            print(f"\n[Curriculum] Phase change at update {update}: "
                  f"{new_phase.num_boxes} boxes, size {new_phase.size_range}")
            phase = new_phase
            # Recreate environment configs
            for i in range(config.num_envs):
                ec = _make_env_config(config, phase=phase, seed=config.seed + i + update)
                env_configs[i] = ec
                envs[i] = BinPackingEnv(config=ec)
                obs, _ = envs[i].reset()
                obs_list[i] = obs
                ep_rewards[i] = 0.0

        # ── Learning rate schedule ──
        lr = _get_lr(update, config.num_updates, config.learning_rate, config.lr_schedule)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # ── Collect rollout ──
        storage.reset()
        network.eval()  # Use eval mode during data collection (no dropout etc.)

        with torch.no_grad():
            for step in range(config.rollout_steps):
                # Build batch tensors from all envs
                batch_hm = []
                batch_item = []
                batch_mask = []

                for i, obs in enumerate(obs_list):
                    hm, item_feat, _ = _build_obs_tensors(obs, config, device)
                    coarse_mask = _remap_action_mask(
                        obs["action_mask"], env_configs[i], config,
                    )
                    batch_hm.append(hm)
                    batch_item.append(item_feat)
                    batch_mask.append(torch.from_numpy(coarse_mask).float().to(device))

                batch_hm_t = torch.stack(batch_hm)         # (N, bins, H, W)
                batch_item_t = torch.stack(batch_item)      # (N, 5)
                batch_mask_t = torch.stack(batch_mask)      # (N, num_actions)

                # Forward pass
                output = network.forward(batch_hm_t, batch_item_t, true_mask=batch_mask_t)

                # Sample actions
                from torch.distributions import Categorical
                dist = Categorical(probs=output.policy)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                entropies = dist.entropy()

                # Execute actions in environments
                rewards_step = np.zeros(config.num_envs, dtype=np.float32)
                dones_step = np.zeros(config.num_envs, dtype=np.float32)

                for i in range(config.num_envs):
                    a2c_act = actions[i].item()
                    env_act = _a2c_to_env_action(a2c_act, config, env_configs[i])

                    # Validate against environment mask
                    if (env_act >= env_configs[i].total_actions or
                            obs_list[i]["action_mask"][env_act] < 0.5):
                        env_act = env_configs[i].total_actions - 1  # skip

                    obs_new, reward, terminated, truncated, info = envs[i].step(env_act)
                    rewards_step[i] = reward
                    ep_rewards[i] += reward
                    done = terminated or truncated
                    dones_step[i] = float(done)

                    if done:
                        # Log episode
                        episode_count += 1
                        ep_reward_history.append(ep_rewards[i])
                        final_fill = info.get("final_avg_fill", 0.0)
                        ep_fills.append(final_fill)
                        ep_rewards[i] = 0.0

                        # Reset environment
                        obs_new, _ = envs[i].reset()

                    obs_list[i] = obs_new
                    global_step += 1

                # Store transition
                storage.insert(
                    heightmaps=batch_hm_t,
                    item_features=batch_item_t,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=torch.from_numpy(rewards_step).float().to(device),
                    dones=torch.from_numpy(dones_step).float().to(device),
                    values=output.value.squeeze(-1),
                    true_masks=batch_mask_t,
                    mask_preds=output.mask_pred,
                    policies=output.policy,
                    entropies=entropies,
                )

            # ── Bootstrap value for last state ──
            batch_hm_last = []
            batch_item_last = []
            batch_mask_last = []
            for i, obs in enumerate(obs_list):
                hm, item_feat, _ = _build_obs_tensors(obs, config, device)
                coarse_mask = _remap_action_mask(
                    obs["action_mask"], env_configs[i], config,
                )
                batch_hm_last.append(hm)
                batch_item_last.append(item_feat)
                batch_mask_last.append(
                    torch.from_numpy(coarse_mask).float().to(device)
                )

            last_output = network.forward(
                torch.stack(batch_hm_last),
                torch.stack(batch_item_last),
                true_mask=torch.stack(batch_mask_last),
            )
            last_value = last_output.value.squeeze(-1)

        # ── Compute GAE ──
        storage.compute_returns_and_advantages(
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        # ── A2C update ──
        network.train()
        data = storage.flatten()

        # Normalise advantages
        advantages = data["advantages"]
        if config.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Re-evaluate actions through network to get fresh gradients
        actions_out, new_log_probs, new_entropy, new_values, new_mask_preds = \
            network.get_action_and_value(
                data["heightmaps"],
                data["item_features"],
                true_mask=data["true_masks"],
                action=data["actions"],
            )

        # Get full policy for infeasibility penalty
        full_output = network.forward(
            data["heightmaps"],
            data["item_features"],
            true_mask=data["true_masks"],
        )

        # Compute loss
        total_loss, loss_metrics = loss_fn(
            log_probs=new_log_probs,
            advantages=advantages,
            values=new_values,
            returns=data["returns"],
            entropies=new_entropy,
            mask_preds=new_mask_preds,
            mask_trues=data["true_masks"],
            policies=full_output.policy,
        )

        # Backprop
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            network.parameters(), config.max_grad_norm,
        )
        optimizer.step()

        # ── Mask accuracy ──
        with torch.no_grad():
            mask_acc = (
                (new_mask_preds > 0.5).float() == data["true_masks"]
            ).float().mean().item()

        # ── Logging ──
        loss_metrics["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        loss_metrics["lr"] = lr
        loss_metrics["mask_accuracy"] = mask_acc
        loss_metrics["global_step"] = global_step

        for k, v in loss_metrics.items():
            logger.log_step(global_step, **{k: v})

        if (update + 1) % config.log_interval == 0:
            elapsed = time.time() - start_time
            fps = global_step / max(elapsed, 1)
            avg_reward = float(np.mean(ep_reward_history[-50:])) if ep_reward_history else 0.0
            avg_fill = float(np.mean(ep_fills[-50:])) if ep_fills else 0.0

            logger.log_episode(
                episode=episode_count,
                reward=avg_reward,
                fill=avg_fill,
                loss=loss_metrics["loss/total"],
                entropy=loss_metrics["loss/entropy"],
                mask_accuracy=mask_acc,
            )

            logger.print_progress(
                episode=update + 1,
                total_episodes=config.num_updates,
                reward=avg_reward,
                fill=avg_fill,
                loss=loss_metrics["loss/total"],
                mask_acc=mask_acc,
                lr=lr,
                fps=int(fps),
                phase_boxes=phase.num_boxes,
            )

        # ── Evaluation ──
        if (update + 1) % config.eval_interval == 0:
            eval_metrics = evaluate(
                network, config,
                num_episodes=config.eval_episodes,
                seed=config.seed + 10000 + update,
                device=device,
            )
            for k, v in eval_metrics.items():
                logger.log_step(global_step, **{k: v})

            print(f"\n  [Eval] fill={eval_metrics['eval/avg_fill']:.4f} "
                  f"reward={eval_metrics['eval/avg_reward']:.2f} "
                  f"placements={eval_metrics['eval/avg_placements']:.1f} "
                  f"pallets={eval_metrics['eval/avg_pallets_closed']:.1f}\n")

            if eval_metrics["eval/avg_fill"] > best_eval_fill:
                best_eval_fill = eval_metrics["eval/avg_fill"]
                best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
                _save_checkpoint(network, optimizer, update, config,
                                 eval_metrics, best_ckpt_path)
                print(f"  [Best] New best fill: {best_eval_fill:.4f} "
                      f"-> saved to {best_ckpt_path}")

        # ── Periodic checkpoint ──
        if (update + 1) % config.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{update+1}.pt")
            _save_checkpoint(network, optimizer, update, config, {}, ckpt_path)
            print(f"  [Checkpoint] Saved to {ckpt_path}")

    # ── Final save ──
    final_path = os.path.join(ckpt_dir, "final_model.pt")
    _save_checkpoint(network, optimizer, config.num_updates, config, {}, final_path)

    # Training curves
    logger.plot_training_curves(save=True, show=False)
    logger.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"[A2C-Masked] Training complete!")
    print(f"  Total time:      {elapsed/3600:.1f}h")
    print(f"  Total steps:     {global_step:,}")
    print(f"  Total episodes:  {episode_count}")
    print(f"  Best eval fill:  {best_eval_fill:.4f}")
    print(f"  Best checkpoint: {best_ckpt_path}")
    print(f"  Final model:     {final_path}")

    return best_ckpt_path or final_path


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    network: A2CMaskedNetwork,
    optimizer: optim.Optimizer,
    update: int,
    config: A2CMaskedConfig,
    eval_metrics: Dict[str, float],
    path: str,
) -> None:
    """Save a training checkpoint."""
    torch.save({
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "update": update,
        "config": config.to_dict(),
        "eval_metrics": eval_metrics,
    }, path)


def load_checkpoint(
    path: str,
    config: Optional[A2CMaskedConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[A2CMaskedNetwork, A2CMaskedConfig, Dict]:
    """
    Load a training checkpoint.

    Args:
        path:   Path to checkpoint .pt file.
        config: Override config (if None, use saved config).
        device: Target device.

    Returns:
        (network, config, checkpoint_dict)
    """
    if device is None:
        device = resolve_device("auto")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if config is None:
        config = A2CMaskedConfig.from_dict(checkpoint["config"])

    network = A2CMaskedNetwork(config).to(device)
    network.load_state_dict(checkpoint["model_state_dict"])

    return network, config, checkpoint


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> A2CMaskedConfig:
    """Parse command-line arguments into A2CMaskedConfig."""
    parser = argparse.ArgumentParser(
        description="Train A2C with Feasibility Masking for 3D Bin Packing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training
    parser.add_argument("--num_updates", type=int, default=200_000,
                        help="Total number of parameter updates.")
    parser.add_argument("--num_envs", type=int, default=16,
                        help="Number of parallel environments.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate.")
    parser.add_argument("--rollout_steps", type=int, default=5,
                        help="Steps per environment per rollout.")

    # Loss weights
    parser.add_argument("--alpha_actor", type=float, default=1.0,
                        help="Weight for actor loss.")
    parser.add_argument("--beta_critic", type=float, default=0.5,
                        help="Weight for critic loss.")
    parser.add_argument("--lambda_mask", type=float, default=0.5,
                        help="Weight for mask BCE loss.")
    parser.add_argument("--omega_infeasibility", type=float, default=0.01,
                        help="Weight for infeasibility penalty.")
    parser.add_argument("--psi_entropy", type=float, default=0.01,
                        help="Weight for entropy bonus.")

    # A2C
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Gradient clipping max norm.")

    # Action space
    parser.add_argument("--action_grid_step", type=float, default=50.0,
                        help="Action grid step (mm).")

    # Curriculum
    parser.add_argument("--no_curriculum", action="store_true",
                        help="Disable curriculum learning.")

    # Logging
    parser.add_argument("--log_dir", type=str, default="outputs/rl_a2c_masked/logs",
                        help="Log directory.")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Print progress every N updates.")
    parser.add_argument("--save_interval", type=int, default=5000,
                        help="Save checkpoint every N updates.")
    parser.add_argument("--eval_interval", type=int, default=1000,
                        help="Evaluate every N updates.")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Evaluation episodes per cycle.")

    # Infrastructure
    parser.add_argument("--device", type=str, default="auto",
                        help="PyTorch device: auto, cpu, cuda.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--no_tensorboard", action="store_true",
                        help="Disable TensorBoard logging.")

    args = parser.parse_args()

    config = A2CMaskedConfig(
        num_updates=args.num_updates,
        num_envs=args.num_envs,
        learning_rate=args.lr,
        rollout_steps=args.rollout_steps,
        alpha_actor=args.alpha_actor,
        beta_critic=args.beta_critic,
        lambda_mask=args.lambda_mask,
        omega_infeasibility=args.omega_infeasibility,
        psi_entropy=args.psi_entropy,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        action_grid_step=args.action_grid_step,
        use_curriculum=not args.no_curriculum,
        log_dir=args.log_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        device=args.device,
        seed=args.seed,
        use_tensorboard=not args.no_tensorboard,
    )

    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = parse_args()

    print(f"\n{'='*80}")
    print(f"  A2C with Feasibility Masking — Training")
    print(f"{'='*80}")
    print(f"  Updates:      {config.num_updates:,}")
    print(f"  Environments: {config.num_envs}")
    print(f"  Rollout:      {config.rollout_steps} steps")
    print(f"  Batch size:   {config.batch_size}")
    print(f"  Total steps:  {config.total_timesteps:,}")
    print(f"  Actions:      {config.num_actions} "
          f"({config.action_grid_l}x{config.action_grid_w}x"
          f"{config.num_orientations}x{config.num_bins})")
    print(f"  LR:           {config.learning_rate}")
    print(f"  Curriculum:   {config.use_curriculum}")
    print(f"  Device:       {config.device}")
    print(f"  Log dir:      {config.log_dir}")
    print(f"{'='*80}\n")

    best_path = train(config)
    print(f"\nBest checkpoint: {best_path}")
