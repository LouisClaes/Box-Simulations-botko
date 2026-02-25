"""
PPO hyperparameter configuration.

All tuneable parameters are collected into PPOConfig, a frozen dataclass.
Defaults are calibrated for the Botko BV dual-pallet setup with EUR pallets
(1200x800mm, height cap 2700mm, close at 1800mm).

References:
    - Schulman et al. (2017): PPO clip ratio, entropy bonus
    - Zhao et al. (2022): Decomposed action space for 3D packing
    - Xiong et al. (2024): Masked PPO with attention for bin packing
    - Andrychowicz et al. (2021): PPO hyperparameter recommendations
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
class PPOConfig:
    """
    Complete PPO training and inference configuration.

    Grouped into logical sections:

    **Environment** -- physical setup (Botko BV defaults).
    **Network** -- architecture dimensions for the actor-critic.
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

    # ── Network architecture ─────────────────────────────────────────────
    # Heightmap CNN
    cnn_channels: Tuple[int, ...] = (32, 64, 128)
    """Conv2d channel progression: (32, 64, 128)."""

    cnn_pool_size: Tuple[int, int] = (4, 4)
    """AdaptiveAvgPool2d output spatial dimensions."""

    bin_embed_dim: int = 256
    """Per-bin embedding dimension after CNN + dense."""

    # Box encoder
    box_feat_dim: int = 5
    """Input features per box: (l, w, h, vol, weight)."""

    box_embed_dim: int = 128
    """Current box embedding dimension."""

    buffer_embed_dim: int = 64
    """Buffer (visible boxes) pooled embedding dimension."""

    # Attention
    attn_num_heads: int = 4
    """Number of heads in cross-attention module."""

    attn_dim: int = 128
    """Attention output dimension (= context vector size)."""

    # Actor / Critic heads
    head_hidden_dim: int = 256
    """Hidden dimension in actor and critic heads."""

    # Derived dimensions (computed, not user-set)
    @property
    def grid_l(self) -> int:
        """Heightmap grid cells along length axis."""
        return int(self.bin_length / self.bin_resolution)

    @property
    def grid_w(self) -> int:
        """Heightmap grid cells along width axis."""
        return int(self.bin_width / self.bin_resolution)

    @property
    def cnn_flat_dim(self) -> int:
        """Flattened CNN output: channels[-1] * pool_h * pool_w."""
        return self.cnn_channels[-1] * self.cnn_pool_size[0] * self.cnn_pool_size[1]

    @property
    def all_bins_dim(self) -> int:
        """Concatenated bin embeddings: num_bins * bin_embed_dim."""
        return self.num_bins * self.bin_embed_dim

    @property
    def actor_input_dim(self) -> int:
        """Actor/Critic input: context + box_embed + all_bins."""
        return self.attn_dim + self.box_embed_dim + self.all_bins_dim

    # ── PPO algorithm ────────────────────────────────────────────────────
    gamma: float = 0.99
    """Discount factor."""

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

    # ── Training schedule ────────────────────────────────────────────────
    total_timesteps: int = 5_000_000
    """Total environment steps across all parallel envs."""

    num_envs: int = 16
    """Number of parallel environments."""

    rollout_steps: int = 256
    """Steps per environment per rollout collection."""

    ppo_epochs: int = 4
    """Number of optimisation epochs per rollout."""

    num_minibatches: int = 8
    """Number of mini-batches per epoch."""

    learning_rate: float = 3e-4
    """Initial learning rate."""

    lr_schedule: str = "cosine"
    """Learning rate schedule: 'cosine', 'linear', or 'constant'."""

    lr_warmup_frac: float = 0.02
    """Fraction of total updates for LR warmup (0 = no warmup)."""

    seed: int = 42
    """Random seed for reproducibility."""

    # ── Logging and checkpointing ────────────────────────────────────────
    log_interval: int = 10
    """Log metrics every N rollouts."""

    save_interval: int = 50
    """Save checkpoint every N rollouts."""

    eval_interval: int = 25
    """Run evaluation episodes every N rollouts."""

    eval_episodes: int = 10
    """Number of evaluation episodes per eval cycle."""

    log_dir: str = "outputs/rl_ppo/logs"
    """Directory for logs, checkpoints, and plots."""

    use_tensorboard: bool = True
    """Enable TensorBoard logging."""

    # ── Reward shaping ───────────────────────────────────────────────────
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    """Reward shaping weights (see rl_common.rewards)."""

    # ── Inference ────────────────────────────────────────────────────────
    checkpoint_path: Optional[str] = None
    """Path to trained model checkpoint for inference."""

    deterministic_inference: bool = True
    """Use argmax (greedy) actions at inference time."""

    fallback_strategy: str = "baseline"
    """Heuristic fallback when no valid RL action exists."""

    device: str = "auto"
    """PyTorch device: 'auto', 'cpu', 'cuda', 'cuda:0', etc."""

    @property
    def batch_size(self) -> int:
        """Total rollout batch size: num_envs * rollout_steps."""
        return self.num_envs * self.rollout_steps

    @property
    def minibatch_size(self) -> int:
        """Samples per mini-batch."""
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        """Total number of PPO parameter updates."""
        return self.total_timesteps // self.batch_size

    def to_dict(self) -> dict:
        """Serialise all parameters (for logging)."""
        d = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if hasattr(v, 'to_dict'):
                    d[k] = v.to_dict() if callable(getattr(v, 'to_dict')) else str(v)
                elif hasattr(v, '__dict__'):
                    d[k] = {kk: vv for kk, vv in v.__dict__.items() if not kk.startswith('_')}
                else:
                    d[k] = v
        # Add computed properties
        d['grid_l'] = self.grid_l
        d['grid_w'] = self.grid_w
        d['batch_size'] = self.batch_size
        d['minibatch_size'] = self.minibatch_size
        d['num_updates'] = self.num_updates
        d['actor_input_dim'] = self.actor_input_dim
        return d
