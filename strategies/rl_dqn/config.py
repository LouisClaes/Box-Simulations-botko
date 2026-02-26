"""
Hyperparameter configuration for the Double DQN bin packing strategy.

All tuneable parameters are centralised here as a single dataclass with
documentation.  Default values are calibrated for the Botko BV thesis setup
(2 EUR pallets, 1200x800mm, close at 1800mm, resolution 10mm).

Usage:
    from strategies.rl_dqn.config import DQNConfig
    cfg = DQNConfig()
    cfg = DQNConfig(lr=0.0005, batch_size=512)

References:
    - Tsang et al. (2025): DDQN dual-bin architecture
    - Mnih et al. (2015): Nature DQN hyperparameters
    - Schaul et al. (2016): Prioritised experience replay
    - van Hasselt et al. (2016): Double DQN
    - Wang et al. (2016): Dueling network architectures
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class DQNConfig:
    """
    Complete hyperparameter set for the DDQN training pipeline.

    Grouped by function:

    Network Architecture
    --------------------
    cnn_channels : tuple
        Output channels for each Conv2d layer in the CNN branch.
    cnn_kernels : tuple
        Kernel sizes for each Conv2d layer.
    cnn_strides : tuple
        Strides for each Conv2d layer.
    box_hidden : int
        Hidden dimension for the box feature MLP branch.
    action_hidden : int
        Hidden dimension for the action feature MLP branch.
    merge_hidden : tuple
        Hidden dimensions for the merged MLP head.
    use_dueling : bool
        Enable dueling architecture (V + A - mean(A)).
    use_batch_norm : bool
        Enable batch normalisation in the CNN branch.

    Training
    --------
    lr : float
        Adam learning rate.
    batch_size : int
        Minibatch size for replay sampling.
    gamma : float
        Discount factor.
    tau : float
        Soft update coefficient for target network (1.0 = hard copy).
    target_update_freq : int
        Steps between target network updates (if tau == 1.0).
    grad_clip : float
        Maximum gradient norm for clipping (0 = disabled).
    weight_decay : float
        L2 regularisation for Adam.

    Exploration
    -----------
    eps_start : float
        Initial epsilon for epsilon-greedy.
    eps_end : float
        Final epsilon.
    eps_decay_fraction : float
        Fraction of total episodes over which epsilon decays linearly.
    noisy_nets : bool
        Use NoisyNet layers instead of epsilon-greedy (not implemented yet).

    Replay Buffer
    -------------
    buffer_capacity : int
        Maximum transitions in the replay buffer.
    buffer_alpha : float
        PER priority exponent (0 = uniform, 1 = full prioritisation).
    buffer_beta_start : float
        Initial importance-sampling exponent for PER.
    buffer_beta_end : float
        Final importance-sampling exponent.
    n_step : int
        N-step return horizon.
    min_buffer_size : int
        Minimum transitions before training starts.

    Candidate Generation
    --------------------
    max_candidates : int
        Maximum number of placement candidates to evaluate per step.
    use_corner_positions : bool
        Include corner-aligned candidate positions.
    use_extreme_points : bool
        Include extreme points from placed boxes.
    use_ems_positions : bool
        Include EMS-inspired candidate positions.
    use_grid_fallback : bool
        Include coarse grid positions as fallback.
    grid_fallback_step : float
        Grid step (mm) for fallback candidates.

    Episode & Environment
    ---------------------
    num_episodes : int
        Total training episodes.
    num_boxes_per_episode : int
        Boxes per training episode.
    eval_interval : int
        Episodes between evaluation runs.
    eval_episodes : int
        Number of episodes per evaluation.
    checkpoint_interval : int
        Episodes between model checkpoints.
    log_interval : int
        Episodes between console progress prints.

    Paths
    -----
    output_dir : str
        Base directory for checkpoints, logs, and plots.
    checkpoint_path : str
        Path to load a pre-trained checkpoint (empty = train from scratch).
    """

    # ── Network Architecture ──────────────────────────────────────────────
    cnn_channels: Tuple[int, ...] = (32, 64, 128, 256)
    cnn_kernels: Tuple[int, ...] = (5, 3, 3, 3)
    cnn_strides: Tuple[int, ...] = (2, 2, 1, 1)
    box_hidden: int = 128
    action_hidden: int = 64
    merge_hidden: Tuple[int, ...] = (256, 128)
    use_dueling: bool = True
    use_batch_norm: bool = True

    # ── Training ──────────────────────────────────────────────────────────
    lr: float = 0.001
    batch_size: int = 256
    gamma: float = 0.95
    tau: float = 1.0
    target_update_freq: int = 500
    grad_clip: float = 10.0
    weight_decay: float = 1e-5

    # ── Exploration ───────────────────────────────────────────────────────
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_fraction: float = 0.8
    noisy_nets: bool = False

    # ── Replay Buffer ─────────────────────────────────────────────────────
    buffer_capacity: int = 100_000
    buffer_alpha: float = 0.6
    buffer_beta_start: float = 0.4
    buffer_beta_end: float = 1.0
    n_step: int = 3
    min_buffer_size: int = 1000

    # ── Candidate Generation ──────────────────────────────────────────────
    max_candidates: int = 200
    use_corner_positions: bool = True
    use_extreme_points: bool = True
    use_ems_positions: bool = True
    use_grid_fallback: bool = True
    grid_fallback_step: float = 100.0

    # ── Episode & Environment ─────────────────────────────────────────────
    num_episodes: int = 50_000
    num_boxes_per_episode: int = 100
    eval_interval: int = 500
    eval_episodes: int = 10
    checkpoint_interval: int = 1000
    log_interval: int = 100

    # ── Physical Setup (Botko BV) ─────────────────────────────────────────
    bin_length: float = 1200.0
    bin_width: float = 800.0
    bin_height: float = 2700.0
    resolution: float = 10.0
    num_bins: int = 2
    buffer_size: int = 8
    pick_window: int = 4
    close_height: float = 1800.0
    num_orientations: int = 2
    box_size_range: Tuple[float, float] = (100.0, 600.0)
    box_weight_range: Tuple[float, float] = (1.0, 30.0)

    # ── Paths ─────────────────────────────────────────────────────────────
    output_dir: str = "outputs/rl_dqn"
    checkpoint_path: str = ""

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def grid_l(self) -> int:
        """Heightmap grid cells along length axis."""
        import math
        return math.ceil(self.bin_length / self.resolution)

    @property
    def grid_w(self) -> int:
        """Heightmap grid cells along width axis."""
        import math
        return math.ceil(self.bin_width / self.resolution)

    @property
    def heightmap_shape(self) -> Tuple[int, int]:
        """(grid_l, grid_w) — shape of a single bin's heightmap."""
        return (self.grid_l, self.grid_w)

    @property
    def cnn_input_channels(self) -> int:
        """Number of input channels to the CNN: one per bin."""
        return self.num_bins

    @property
    def box_feature_dim(self) -> int:
        """Input dimension for the box feature branch: pick_window * 5."""
        return self.pick_window * 5

    @property
    def action_feature_dim(self) -> int:
        """Input dimension for the action feature branch."""
        return 7  # (bin_idx, x_norm, y_norm, orient, z_norm, support_ratio, height_ratio)

    @property
    def eps_decay_episodes(self) -> int:
        """Number of episodes over which epsilon decays."""
        return int(self.num_episodes * self.eps_decay_fraction)

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary."""
        d = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                d[k] = v
        # Add derived properties
        d["grid_l"] = self.grid_l
        d["grid_w"] = self.grid_w
        d["eps_decay_episodes"] = self.eps_decay_episodes
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DQNConfig":
        """Create from a dictionary (ignores unknown keys)."""
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
