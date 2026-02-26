"""
PCT Transformer hyperparameter configuration.

All tuneable parameters are collected into PCTTransformerConfig.
Defaults are calibrated for the Botko BV dual-pallet setup with EUR pallets
(1200x800mm, height cap 2700mm, close at 1800mm).

References:
    - Zhao et al. (ICLR 2022): PCT network architecture and training procedure
    - Zhao et al. (IJRR 2025): Extended PCT with MCTS buffer search
    - Schulman et al. (2017): PPO clip ratio, entropy bonus
    - Vaswani et al. (2017): Transformer architecture parameters
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

from strategies.rl_common.rewards import RewardConfig


@dataclass
class PCTTransformerConfig:
    """
    Complete configuration for the PCT Transformer strategy.

    Grouped into logical sections:

    **Environment** -- physical setup (Botko BV defaults).
    **Candidate Generation** -- how placement candidates are produced.
    **Network** -- Transformer architecture dimensions.
    **PPO** -- core algorithm parameters.
    **Training** -- schedule, parallelism, logging.
    **Inference** -- checkpoint loading, fallback behaviour.

    All fields have sensible defaults.  Override via CLI or YAML.
    """

    # ── Environment ──────────────────────────────────────────────────────
    bin_length: float = 1200.0
    bin_width: float = 800.0
    bin_height: float = 2700.0
    bin_resolution: float = 10.0
    num_bins: int = 2
    buffer_size: int = 8
    pick_window: int = 4
    close_height: float = 1800.0
    max_consecutive_rejects: int = 10
    num_orientations: int = 2
    num_boxes_per_episode: int = 100
    box_size_range: Tuple[float, float] = (100.0, 600.0)
    box_weight_range: Tuple[float, float] = (1.0, 30.0)

    # ── Candidate Generation ─────────────────────────────────────────────
    min_support_ratio: float = 0.30
    """Minimum base support for a candidate to be valid."""

    floor_scan_step: float = 50.0
    """Grid step (mm) for floor-level candidate scan."""

    max_candidates: int = 200
    """Maximum number of candidates per step (cap for memory/speed)."""

    candidate_dedup_tolerance: float = 5.0
    """Merge candidates within this distance (mm) of each other."""

    # ── Network Architecture ─────────────────────────────────────────────
    # Item encoder: box features -> embedding
    item_input_dim: int = 5
    """Input features per box: (l_norm, w_norm, h_norm, vol_norm, weight_norm)."""

    item_hidden_dim: int = 64
    """Hidden layer size in item encoder MLP."""

    # Candidate encoder: placement features -> embedding
    candidate_input_dim: int = 12
    """Input features per candidate:
       bin_idx_onehot(2) + x_norm + y_norm + z_norm +
       support_ratio + height_after_norm + fill_after_norm +
       contact_ratio + gap_below_norm + adjacent_fill_norm + orient_idx_norm
       (last dim is orient_idx / num_orientations)
    """

    candidate_hidden_dim: int = 64
    """Hidden layer size in candidate encoder MLP."""

    # Transformer encoder
    d_model: int = 128
    """Embedding dimension throughout the Transformer."""

    nhead: int = 4
    """Number of attention heads."""

    num_encoder_layers: int = 3
    """Number of Transformer encoder layers."""

    dim_feedforward: int = 256
    """Hidden dimension in Transformer FFN sublayers."""

    dropout: float = 0.1
    """Dropout rate in Transformer."""

    # Value head
    value_hidden_dim: int = 64
    """Hidden layer size in value head MLP."""

    # Pointer decoder temperature
    pointer_temperature: float = 1.0
    """Temperature for pointer logits (lower = sharper, higher = more uniform).
    Set to sqrt(d_model) = ~11.3 for standard scaled dot-product attention."""

    use_scaled_attention: bool = True
    """If True, divide logits by sqrt(d_model) (standard Transformer scaling)."""

    # ── PPO Algorithm ────────────────────────────────────────────────────
    gamma: float = 1.0
    """Discount factor.  1.0 = undiscounted (following PCT paper for finite episodes)."""

    gae_lambda: float = 0.95
    """GAE lambda for advantage estimation."""

    clip_ratio: float = 0.2
    """PPO clipping parameter epsilon."""

    clip_value: bool = True
    """Whether to clip value function updates."""

    clip_value_range: float = 0.2
    """Value function clip range (if clip_value=True)."""

    entropy_coeff: float = 0.01
    """Entropy bonus coefficient (encourages exploration)."""

    value_loss_coeff: float = 0.5
    """Value function loss weight in total loss."""

    max_grad_norm: float = 0.5
    """Gradient clipping max norm."""

    normalize_advantages: bool = True
    """Whether to normalise advantages within mini-batch."""

    # ── Training Schedule ────────────────────────────────────────────────
    total_episodes: int = 200_000
    """Total training episodes."""

    num_envs: int = 16
    """Number of parallel environments for data collection."""

    rollout_steps: int = 20
    """Steps per environment per rollout collection."""

    ppo_epochs: int = 4
    """Number of optimisation epochs per rollout."""

    num_minibatches: int = 4
    """Number of mini-batches per epoch."""

    learning_rate: float = 3e-4
    """Initial learning rate."""

    lr_schedule: str = "cosine"
    """Learning rate schedule: 'cosine', 'linear', or 'constant'."""

    lr_warmup_frac: float = 0.02
    """Fraction of total updates for LR warmup (0 = no warmup)."""

    seed: int = 42
    """Random seed for reproducibility."""

    # ── Buffer-Aware Training ────────────────────────────────────────────
    enable_buffer_search: bool = True
    """Try each grippable box and pick the one with highest value estimate."""

    value_lookahead: bool = False
    """Use 1-step lookahead with value function for box selection (expensive)."""

    # ── Logging and Checkpointing ────────────────────────────────────────
    log_interval: int = 50
    """Log metrics every N episodes."""

    save_interval: int = 500
    """Save checkpoint every N episodes."""

    eval_interval: int = 200
    """Run evaluation episodes every N episodes."""

    eval_episodes: int = 10
    """Number of evaluation episodes per eval cycle."""

    log_dir: str = "outputs/rl_pct_transformer/logs"
    """Directory for logs, checkpoints, and plots."""

    use_tensorboard: bool = True
    """Enable TensorBoard logging."""

    # ── Reward Shaping ───────────────────────────────────────────────────
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    """Reward shaping weights (see rl_common.rewards)."""

    # ── Inference ────────────────────────────────────────────────────────
    checkpoint_path: Optional[str] = None
    """Path to trained model checkpoint for inference."""

    deterministic_inference: bool = True
    """Use argmax (greedy) actions at inference time."""

    fallback_strategy: str = "extreme_points"
    """Heuristic fallback when no valid RL action or no checkpoint."""

    device: str = "auto"
    """PyTorch device: 'auto', 'cpu', 'cuda', 'cuda:0', etc."""

    # ── Computed Properties ──────────────────────────────────────────────

    @property
    def grid_l(self) -> int:
        """Heightmap grid cells along length axis."""
        return int(self.bin_length / self.bin_resolution)

    @property
    def grid_w(self) -> int:
        """Heightmap grid cells along width axis."""
        return int(self.bin_width / self.bin_resolution)

    @property
    def batch_size(self) -> int:
        """Total rollout batch size: num_envs * rollout_steps."""
        return self.num_envs * self.rollout_steps

    @property
    def minibatch_size(self) -> int:
        """Samples per mini-batch."""
        return max(1, self.batch_size // self.num_minibatches)

    @property
    def resolved_device(self) -> str:
        """Resolve 'auto' to 'cuda' or 'cpu'."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    def to_dict(self) -> dict:
        """Serialise all parameters (for logging)."""
        d = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if hasattr(v, '__dict__') and not isinstance(v, type):
                    d[k] = {kk: vv for kk, vv in v.__dict__.items()
                            if not kk.startswith('_')}
                else:
                    d[k] = v
        # Add computed properties
        d['grid_l'] = self.grid_l
        d['grid_w'] = self.grid_w
        d['batch_size'] = self.batch_size
        d['minibatch_size'] = self.minibatch_size
        d['resolved_device'] = self.resolved_device
        return d

    @classmethod
    def from_dict(
        cls,
        data: dict,
        base: Optional["PCTTransformerConfig"] = None,
    ) -> "PCTTransformerConfig":
        """Rehydrate config from a checkpoint-safe dictionary."""
        config = base or cls()
        if not isinstance(data, dict):
            return config

        ignored = {"grid_l", "grid_w", "batch_size", "minibatch_size", "resolved_device"}
        for key, value in data.items():
            if key.startswith("_") or key in ignored or not hasattr(config, key):
                continue
            if key == "reward_config" and isinstance(value, dict):
                reward_cfg = RewardConfig()
                for rk, rv in value.items():
                    if hasattr(reward_cfg, rk):
                        setattr(reward_cfg, rk, rv)
                setattr(config, key, reward_cfg)
                continue
            try:
                setattr(config, key, value)
            except Exception:
                pass
        return config
