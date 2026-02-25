"""
Hyperparameter configuration for the RL Hybrid Hyper-Heuristic strategy.

All tuneable parameters are centralised here as a single dataclass.
Default values are calibrated for the Botko BV thesis setup
(2 EUR pallets, 1200x800mm, close at 1800mm, resolution 10mm).

Two training modes are supported:
  1. Tabular Q-learning (fast baseline, ~1 hour)
  2. DQN selector (better performance, ~4-8 hours)

Usage:
    from strategies.rl_hybrid_hh.config import HHConfig
    cfg = HHConfig()
    cfg = HHConfig(lr=0.0005, num_episodes=20000)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class HHConfig:
    """
    Complete hyperparameter set for the RL Hybrid Hyper-Heuristic.

    Grouped by function:

    Heuristic Portfolio
    -------------------
    heuristic_names : list
        Names of the low-level heuristics available to the selector.
        Must match registered strategy names in STRATEGY_REGISTRY.
    include_skip : bool
        Whether to include a SKIP action (advance conveyor without placing).

    State Features
    --------------
    state_dim : int
        Dimensionality of the handcrafted state feature vector.
    num_bins_physical : int
        Number of physical pallet stations (for feature extraction).
    features_per_bin : int
        Number of features extracted per bin.
    box_features_dim : int
        Number of features for the current box.
    buffer_features_dim : int
        Number of features for the buffer/conveyor state.
    progress_features_dim : int
        Number of episode progress features.
    history_features_dim : int
        Number of heuristic selection history features.
    phase_features_dim : int
        Number of packing phase indicator features.

    DQN Network Architecture
    ------------------------
    hidden_dims : tuple
        Hidden layer dimensions for the Q-network MLP.
    dropout : float
        Dropout rate for regularisation.

    Tabular Q-Learning
    ------------------
    fill_bins : int
        Number of discretisation bins for fill rate (tabular mode).
    height_bins : int
        Number of discretisation bins for height (tabular mode).
    box_size_bins : int
        Number of discretisation bins for box size (tabular mode).
    roughness_bins : int
        Number of discretisation bins for surface roughness (tabular mode).

    Training
    --------
    lr : float
        Learning rate (Adam for DQN, alpha for tabular).
    gamma : float
        Discount factor.
    eps_start : float
        Initial exploration rate.
    eps_end : float
        Final exploration rate.
    eps_decay_fraction : float
        Fraction of total episodes over which epsilon decays.
    num_episodes : int
        Total training episodes.
    num_boxes_per_episode : int
        Boxes per training episode.

    DQN-Specific
    -------------
    batch_size : int
        Minibatch size for replay sampling.
    buffer_capacity : int
        Maximum transitions in the experience replay buffer.
    min_buffer_size : int
        Minimum transitions before training starts.
    target_update_freq : int
        Episodes between target network synchronisation.
    grad_clip : float
        Maximum gradient norm for clipping.
    weight_decay : float
        L2 regularisation.

    Reward Shaping
    --------------
    reward_volume_weight : float
        Reward scaling for placed volume ratio.
    reward_fill_delta_weight : float
        Reward scaling for fill rate improvement.
    reward_failure_penalty : float
        Penalty when selected heuristic fails to place.
    reward_diversity_bonus : float
        Bonus for switching heuristics when beneficial.
    reward_terminal_weight : float
        Terminal reward scaling for avg_closed_fill.

    Evaluation & Logging
    --------------------
    eval_interval : int
        Episodes between evaluation runs.
    eval_episodes : int
        Episodes per evaluation.
    log_interval : int
        Episodes between console progress prints.
    checkpoint_interval : int
        Episodes between model checkpoints.

    Physical Setup
    --------------
    bin_length : float
    bin_width : float
    bin_height : float
    resolution : float
    num_bins : int
    buffer_size : int
    pick_window : int
    close_height : float
    box_size_range : tuple
    box_weight_range : tuple

    Paths
    -----
    output_dir : str
        Base directory for checkpoints, logs, and plots.
    checkpoint_path : str
        Path to load a pre-trained model (empty = train from scratch).
    """

    # ── Heuristic Portfolio ────────────────────────────────────────────────
    heuristic_names: List[str] = field(default_factory=lambda: [
        "baseline",              # 0: DBLF -- reliable, conservative
        "walle_scoring",         # 1: Best overall (68.3% fill)
        "surface_contact",       # 2: Second best (67.4%)
        "extreme_points",        # 3: Good for irregular boxes
        "skyline",               # 4: Good for similar-height boxes
        "layer_building",        # 5: Good for building layers
        "best_fit_decreasing",   # 6: Good for size-sorted arrivals
    ])
    include_skip: bool = True  # Action 7: advance conveyor

    # ── State Features ─────────────────────────────────────────────────────
    num_bins_physical: int = 2
    features_per_bin: int = 8
    box_features_dim: int = 5
    buffer_features_dim: int = 4
    progress_features_dim: int = 3
    history_features_dim: int = 8
    phase_features_dim: int = 3

    @property
    def state_dim(self) -> int:
        """Total dimensionality of the handcrafted state vector."""
        return (
            self.num_bins_physical * self.features_per_bin  # 16
            + self.box_features_dim                         # 5
            + self.buffer_features_dim                      # 4
            + self.progress_features_dim                    # 3
            + self.history_features_dim                     # 8
            + self.phase_features_dim                       # 3
        )  # Total: 39

    @property
    def num_actions(self) -> int:
        """Total number of actions (heuristics + optional skip)."""
        return len(self.heuristic_names) + (1 if self.include_skip else 0)

    # ── DQN Network Architecture ──────────────────────────────────────────
    hidden_dims: Tuple[int, ...] = (128, 128, 64)
    dropout: float = 0.1

    # ── Tabular Q-Learning ────────────────────────────────────────────────
    fill_bins: int = 5
    height_bins: int = 5
    box_size_bins: int = 3
    roughness_bins: int = 3

    @property
    def tabular_state_size(self) -> int:
        """Number of discrete states for tabular Q-learning.

        States are the Cartesian product of:
          - fill_bins (per bin)
          - height_bins (per bin)
          - box_size_bins
          - roughness_bins (for primary bin)
        """
        return (
            (self.fill_bins ** self.num_bins_physical)
            * (self.height_bins ** self.num_bins_physical)
            * self.box_size_bins
            * self.roughness_bins
        )

    # ── Training ──────────────────────────────────────────────────────────
    lr: float = 0.001
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_fraction: float = 0.8
    num_episodes: int = 10_000
    num_boxes_per_episode: int = 100

    # ── DQN-Specific ──────────────────────────────────────────────────────
    batch_size: int = 128
    buffer_capacity: int = 50_000
    min_buffer_size: int = 500
    target_update_freq: int = 500
    grad_clip: float = 10.0
    weight_decay: float = 1e-5

    # ── Reward Shaping ────────────────────────────────────────────────────
    reward_volume_weight: float = 10.0
    reward_fill_delta_weight: float = 5.0
    reward_failure_penalty: float = -0.5
    reward_skip_penalty: float = -0.3
    reward_diversity_bonus: float = 0.1
    reward_terminal_weight: float = 10.0

    # ── Evaluation & Logging ──────────────────────────────────────────────
    eval_interval: int = 500
    eval_episodes: int = 5
    log_interval: int = 100
    checkpoint_interval: int = 1000

    # ── Physical Setup (Botko BV) ─────────────────────────────────────────
    bin_length: float = 1200.0
    bin_width: float = 800.0
    bin_height: float = 2700.0
    resolution: float = 10.0
    num_bins: int = 2
    buffer_size: int = 8
    pick_window: int = 4
    close_height: float = 1800.0
    box_size_range: Tuple[float, float] = (100.0, 600.0)
    box_weight_range: Tuple[float, float] = (1.0, 30.0)

    # ── Paths ─────────────────────────────────────────────────────────────
    output_dir: str = "outputs/rl_hybrid_hh"
    checkpoint_path: str = ""

    # ── Derived Properties ────────────────────────────────────────────────

    @property
    def eps_decay_episodes(self) -> int:
        """Number of episodes over which epsilon decays linearly."""
        return int(self.num_episodes * self.eps_decay_fraction)

    def get_epsilon(self, episode: int) -> float:
        """Compute epsilon for a given episode (linear decay)."""
        if episode >= self.eps_decay_episodes:
            return self.eps_end
        frac = episode / max(self.eps_decay_episodes, 1)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary."""
        d = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                d[k] = v
        d["state_dim"] = self.state_dim
        d["num_actions"] = self.num_actions
        d["tabular_state_size"] = self.tabular_state_size
        d["eps_decay_episodes"] = self.eps_decay_episodes
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HHConfig":
        """Create from a dictionary (ignores unknown keys)."""
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
