"""
Configuration for the MCTS-Guided Hierarchical Actor-Critic strategy.

All hyperparameters for the three-level architecture:
  Level 1: World Model (state transition prediction + conveyor model)
  Level 2: High-Level Policy (item selection + bin assignment)
  Level 3: Low-Level Policy (Transformer pointer over placement candidates)
  MCTS: Monte Carlo Tree Search planner using the world model

Defaults are calibrated for the Botko BV setup:
  2 EUR pallets (1200x800mm), height cap 2700mm, resolution 10mm,
  conveyor with 8 visible boxes, pick window 4, close at 1800mm.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional


@dataclass
class MCTSHybridConfig:
    """
    Full configuration for the rl_mcts_hybrid strategy.

    Organised into sections matching the architecture components.
    """

    # ── Physical setup (Botko BV) ────────────────────────────────────────
    bin_length: float = 1200.0
    bin_width: float = 800.0
    bin_height: float = 2700.0
    resolution: float = 10.0
    num_bins: int = 2
    pick_window: int = 4
    buffer_size: int = 8
    close_height: float = 1800.0
    num_orientations: int = 2

    # Grid dimensions (derived, uses ceil to match BinConfig)
    @property
    def grid_l(self) -> int:
        import math
        return math.ceil(self.bin_length / self.resolution)  # 120

    @property
    def grid_w(self) -> int:
        import math
        return math.ceil(self.bin_width / self.resolution)    # 80

    # ── Shared Encoder ────────────────────────────────────────────────────
    # Per-bin CNN encoder (shared weights across bins)
    cnn_channels: Tuple[int, ...] = (32, 64, 128)
    cnn_pool_size: int = 4
    cnn_flat_dim: int = 2048       # 128 * 4 * 4
    bin_embed_dim: int = 256

    # Box feature dimensions
    box_feat_dim: int = 5          # (l, w, h, vol, weight) normalised
    box_embed_dim: int = 128
    buffer_embed_dim: int = 64

    # Global state embedding (after all encoding)
    global_state_dim: int = 768    # 2*256 (bins) + 128 (box) + 64 (buffer) + 64 (conveyor)
    conveyor_embed_dim: int = 64   # Conveyor state encoding

    # ── World Model ──────────────────────────────────────────────────────
    # Predicts: next heightmaps, next box features, reward estimate
    world_model_hidden: int = 512
    world_model_layers: int = 3
    world_model_action_dim: int = 32   # Action embedding for world model
    world_model_lr: float = 1e-4
    world_model_loss_weight: float = 1.0

    # Void prediction head (auxiliary)
    void_hidden: int = 256
    void_loss_weight: float = 0.5      # Weight for trapped void auxiliary loss

    # ── High-Level Policy (item + bin selection) ─────────────────────────
    # Selects: (box_index, bin_index) from grippable window
    high_level_actions: int = 10        # pick_window * num_bins + skip + reconsider
    high_level_hidden: int = 256
    high_level_embed_dim: int = 64      # Embedding for HL action -> LL conditioning

    # ── Low-Level Policy (placement via Transformer pointer) ─────────────
    # Selects: best candidate from generated set
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    pointer_temperature: float = 1.0
    max_candidates: int = 200

    # Candidate feature dimensions
    candidate_input_dim: int = 16      # Richer than PCT's 12
    candidate_hidden_dim: int = 64

    # ── MCTS Planner ─────────────────────────────────────────────────────
    mcts_simulations: int = 50         # Simulations per move at inference
    mcts_depth: int = 4                # Lookahead depth (matches pick_window)
    mcts_c_puct: float = 1.5          # Exploration constant
    mcts_temperature: float = 1.0      # Action selection temperature
    mcts_discount: float = 0.99
    mcts_enabled: bool = True          # Can disable for ablation

    # During training: fewer simulations for speed
    mcts_train_simulations: int = 10
    mcts_train_enabled: bool = False   # Start without MCTS, enable after warm-up

    # ── Heuristic Ensemble ───────────────────────────────────────────────
    # Which heuristics generate candidate positions
    heuristic_names: Tuple[str, ...] = (
        "walle_scoring",
        "surface_contact",
        "baseline",
        "extreme_points",
        "ems",
    )
    use_heuristic_candidates: bool = True
    use_corner_points: bool = True
    use_extreme_points: bool = True
    use_ems_positions: bool = True
    use_grid_fallback: bool = True
    grid_step: float = 50.0          # Coarser for speed (heuristics fill gaps)
    min_support: float = 0.30

    # ── Training ─────────────────────────────────────────────────────────
    total_timesteps: int = 5_000_000
    num_envs: int = 64                 # Parallel environments on HPC
    rollout_steps: int = 128           # Steps per rollout
    num_epochs: int = 4                # PPO update epochs
    minibatch_size: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2             # PPO clip
    vf_coeff: float = 0.5
    ent_coeff: float = 0.01           # Entropy bonus
    max_grad_norm: float = 0.5

    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_stages: int = 4
    # Stage 0: single box, single bin (learn placement)
    # Stage 1: 4 boxes, single bin (learn sequencing)
    # Stage 2: full pick window, 2 bins (learn item+bin selection)
    # Stage 3: full problem + MCTS (learn planning)

    # Warm-start from heuristic demonstrations
    imitation_epochs: int = 50
    imitation_lr: float = 1e-3
    imitation_weight: float = 0.5      # Auxiliary imitation loss during RL

    # ── Reward shaping ───────────────────────────────────────────────────
    # Base rewards (same as rl_common RewardConfig for consistency)
    volume_weight: float = 10.0
    fill_delta_weight: float = 5.0
    surface_contact_weight: float = 2.0
    height_penalty_weight: float = -1.0
    roughness_penalty_weight: float = -0.5
    close_bonus_weight: float = 5.0
    terminal_fill_weight: float = 10.0

    # Novel reward components
    void_penalty_weight: float = -3.0      # Penalise trapped voids
    item_selection_bonus: float = 1.0      # Reward for choosing best item
    multi_bin_balance_weight: float = 0.5  # Reward balanced bin utilisation
    lookahead_value_weight: float = 2.0    # MCTS value as reward shaping

    # ── Inference ─────────────────────────────────────────────────────────
    deterministic: bool = True             # Greedy at inference
    fallback_strategy: str = "walle_scoring"
    checkpoint_dir: str = "outputs/rl_mcts_hybrid/checkpoints"

    # ── Episode settings ─────────────────────────────────────────────────
    num_boxes_per_episode: int = 100
    box_size_range: Tuple[float, float] = (100.0, 600.0)
    box_weight_range: Tuple[float, float] = (1.0, 30.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise config to dict for checkpoint saving."""
        d = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                d[k] = v
        # Add derived properties
        d["grid_l"] = self.grid_l
        d["grid_w"] = self.grid_w
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MCTSHybridConfig":
        """Restore config from dict."""
        # Filter out derived properties
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# Convenience: default config instance
DEFAULT_CONFIG = MCTSHybridConfig()
