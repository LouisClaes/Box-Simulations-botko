"""
A2C with Feasibility Masking -- Hyperparameter configuration.

All tuneable parameters for the A2C agent with a learned feasibility mask
predictor, following Zhao et al. (AAAI 2021).  Defaults are calibrated for
the Botko BV thesis setup (2 EUR pallets, 1200x800mm, close at 1800mm).

Architecture summary:
    - Shared CNN encoder processes 4-channel heightmaps (per bin)
    - Item MLP embeds the current box features
    - Three output heads: Actor (policy), Critic (value), Mask (feasibility)
    - Coarse action grid (step=50mm) for tractable action space

Loss function (5 components):
    L = alpha * L_actor + beta * L_critic + lam * L_mask
        + omega * E_infeasibility - psi * E_entropy

    L_actor   = -log(pi(a|s)) * A(s,a)           [policy gradient]
    L_critic  = (V(s) - R_target)^2               [value regression]
    L_mask    = BCE(mask_pred, mask_true)           [mask supervision]
    E_inf     = sum pi(a|s) * log(M(a|s))          [infeasibility penalty]
    E_entropy = -sum pi(a|s) * log(pi(a|s))        [exploration bonus]

Curriculum learning phases:
    Phase 1 (0-30%):  30 boxes, 200-500mm
    Phase 2 (30-70%): 60 boxes, 150-550mm
    Phase 3 (70-100%): 100 boxes, 100-600mm

References:
    - Zhao et al. (AAAI 2021): Learning to pack with feasibility masking
    - Wu et al. (2017): A2C / ACKTR
    - Mnih et al. (2016): A3C — asynchronous advantage actor-critic
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

from strategies.rl_common.rewards import RewardConfig


# ---------------------------------------------------------------------------
# Curriculum phase definition
# ---------------------------------------------------------------------------

@dataclass
class CurriculumPhase:
    """
    A single curriculum learning phase.

    Attributes:
        start_frac:     Phase starts at this fraction of total updates.
        end_frac:       Phase ends at this fraction.
        num_boxes:      Number of boxes per episode in this phase.
        size_range:     (min_dim, max_dim) in mm for random box generation.
    """
    start_frac: float
    end_frac: float
    num_boxes: int
    size_range: Tuple[float, float]


# ---------------------------------------------------------------------------
# Main configuration
# ---------------------------------------------------------------------------

@dataclass
class A2CMaskedConfig:
    """
    Complete hyperparameter set for A2C with feasibility masking.

    Grouped into logical sections:

    **Environment** -- Physical setup (Botko BV defaults).
    **Action Space** -- Coarse grid discretisation.
    **Network Architecture** -- CNN encoder, item MLP, output heads.
    **A2C Algorithm** -- Core RL parameters.
    **Loss Weights** -- Multi-component loss coefficients.
    **Curriculum** -- Progressive difficulty schedule.
    **Training Schedule** -- Parallelism, updates, logging.
    **Inference** -- Checkpoint loading, fallback behaviour.
    """

    # ── Environment (Botko BV defaults) ────────────────────────────────────
    bin_length: float = 1200.0
    """Pallet X dimension (mm)."""

    bin_width: float = 800.0
    """Pallet Y dimension (mm)."""

    bin_height: float = 2700.0
    """Pallet Z height cap (mm)."""

    bin_resolution: float = 10.0
    """Heightmap grid cell size (mm)."""

    num_bins: int = 2
    """Number of simultaneous pallets."""

    buffer_size: int = 8
    """Conveyor belt capacity (visible boxes)."""

    pick_window: int = 4
    """Front N boxes the robot can reach."""

    close_height: float = 1800.0
    """Height at which pallets are closed (mm)."""

    max_consecutive_rejects: int = 10
    """Safety valve: stop after N consecutive rejects."""

    num_orientations: int = 2
    """Allowed orientations: 2 = flat only, 6 = all."""

    num_boxes_per_episode: int = 100
    """Default boxes per episode (overridden by curriculum)."""

    box_size_range: Tuple[float, float] = (100.0, 600.0)
    """Default (min, max) box dimension in mm (overridden by curriculum)."""

    box_weight_range: Tuple[float, float] = (1.0, 30.0)
    """(min, max) box weight in kg."""

    # ── Action space discretisation ────────────────────────────────────────
    action_grid_step: float = 50.0
    """Grid step for coarse action space (mm).
    Coarse grid: 50mm -> 24x16 positions per orientation per bin.
    Fine grid: 10mm -> 120x80 (too large for A2C)."""

    # ── Network architecture ───────────────────────────────────────────────
    # Shared CNN encoder (per-bin)
    cnn_channels: Tuple[int, ...] = (32, 64, 64, 128, 128)
    """Conv2d output channels for each layer."""

    cnn_kernels: Tuple[int, ...] = (3, 3, 3, 3, 3)
    """Kernel sizes for each Conv2d layer."""

    cnn_strides: Tuple[int, ...] = (1, 1, 2, 2, 2)
    """Strides for each Conv2d layer."""

    cnn_paddings: Tuple[int, ...] = (1, 1, 1, 1, 1)
    """Padding for each Conv2d layer."""

    cnn_pool_size: Tuple[int, int] = (4, 4)
    """AdaptiveAvgPool2d output spatial size (H, W)."""

    cnn_flat_to_embed: int = 512
    """Dense layer: flattened CNN -> per-bin embedding."""

    # Item (box) MLP
    item_input_dim: int = 5
    """Input features per box: (l, w, h, vol, weight) normalised."""

    item_hidden_dim: int = 64
    """Hidden dimension in item MLP."""

    item_embed_dim: int = 128
    """Output dimension of item embedding."""

    # Combined features
    @property
    def bin_embed_dim(self) -> int:
        """Per-bin CNN embedding dimension."""
        return self.cnn_flat_to_embed

    @property
    def total_bin_features(self) -> int:
        """Concatenated bin embeddings: num_bins * bin_embed_dim."""
        return self.num_bins * self.bin_embed_dim

    @property
    def combined_features(self) -> int:
        """Total feature vector: all bins + item embedding."""
        return self.total_bin_features + self.item_embed_dim

    # Actor head
    actor_hidden: int = 256
    """Hidden dimension in actor MLP."""

    # Critic head
    critic_hidden: int = 256
    """Hidden dimension in critic MLP."""

    # Mask predictor head
    mask_hidden: int = 256
    """Hidden dimension in mask predictor MLP."""

    mask_threshold: float = 0.5
    """Threshold for binarising mask predictions at inference."""

    # ── A2C algorithm parameters ──────────────────────────────────────────
    gamma: float = 0.99
    """Discount factor for future rewards."""

    gae_lambda: float = 0.95
    """Lambda for Generalised Advantage Estimation (GAE)."""

    max_grad_norm: float = 0.5
    """Gradient clipping max norm."""

    normalize_advantages: bool = True
    """Whether to normalise advantages within a rollout batch."""

    # ── Loss weights (from Zhao et al. AAAI 2021) ────────────────────────
    alpha_actor: float = 1.0
    """Weight for policy gradient loss L_actor."""

    beta_critic: float = 0.5
    """Weight for value regression loss L_critic."""

    lambda_mask: float = 0.5
    """Weight for mask BCE loss L_mask."""

    omega_infeasibility: float = 0.01
    """Weight for infeasibility penalty E_inf."""

    psi_entropy: float = 0.01
    """Weight for entropy bonus E_entropy."""

    # ── Curriculum learning ────────────────────────────────────────────────
    use_curriculum: bool = True
    """Enable curriculum learning (progressive difficulty)."""

    curriculum_phases: List[CurriculumPhase] = field(default_factory=lambda: [
        CurriculumPhase(start_frac=0.0,  end_frac=0.3,  num_boxes=30,  size_range=(200.0, 500.0)),
        CurriculumPhase(start_frac=0.3,  end_frac=0.7,  num_boxes=60,  size_range=(150.0, 550.0)),
        CurriculumPhase(start_frac=0.7,  end_frac=1.0,  num_boxes=100, size_range=(100.0, 600.0)),
    ])
    """Curriculum phases with progressive box count and size range."""

    # ── Training schedule ─────────────────────────────────────────────────
    num_updates: int = 200_000
    """Total number of parameter updates."""

    num_envs: int = 16
    """Number of parallel environments."""

    rollout_steps: int = 5
    """Steps per environment per rollout collection (N-step)."""

    learning_rate: float = 1e-4
    """Initial learning rate."""

    lr_schedule: str = "linear"
    """Learning rate schedule: 'linear', 'cosine', or 'constant'."""

    seed: int = 42
    """Random seed for reproducibility."""

    # ── Logging and checkpointing ─────────────────────────────────────────
    log_interval: int = 100
    """Print progress every N updates."""

    save_interval: int = 5000
    """Save checkpoint every N updates."""

    eval_interval: int = 1000
    """Run evaluation every N updates."""

    eval_episodes: int = 10
    """Number of evaluation episodes per eval cycle."""

    log_dir: str = "outputs/rl_a2c_masked/logs"
    """Directory for logs, checkpoints, and plots."""

    use_tensorboard: bool = True
    """Enable TensorBoard logging."""

    # ── Reward shaping ────────────────────────────────────────────────────
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    """Reward shaping weights (see rl_common.rewards)."""

    # ── Inference ─────────────────────────────────────────────────────────
    checkpoint_path: Optional[str] = None
    """Path to trained model checkpoint for inference."""

    deterministic_inference: bool = True
    """Use argmax (greedy) actions at inference time."""

    fallback_strategy: str = "baseline"
    """Heuristic fallback when no valid RL action or no model loaded."""

    device: str = "auto"
    """PyTorch device: 'auto', 'cpu', 'cuda', 'cuda:0', etc."""

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def grid_l(self) -> int:
        """Heightmap grid cells along length axis (full resolution)."""
        return int(self.bin_length / self.bin_resolution)

    @property
    def grid_w(self) -> int:
        """Heightmap grid cells along width axis (full resolution)."""
        return int(self.bin_width / self.bin_resolution)

    @property
    def action_grid_l(self) -> int:
        """Action grid cells along length axis (coarse)."""
        return max(1, int(self.bin_length / self.action_grid_step))

    @property
    def action_grid_w(self) -> int:
        """Action grid cells along width axis (coarse)."""
        return max(1, int(self.bin_width / self.action_grid_step))

    @property
    def num_actions(self) -> int:
        """Total discrete actions: grid_l * grid_w * orientations * bins.
        With step=50mm: 24 * 16 * 2 * 2 = 1536."""
        return self.action_grid_l * self.action_grid_w * self.num_orientations * self.num_bins

    @property
    def cnn_input_channels(self) -> int:
        """CNN input channels per bin: (height_norm, item_l, item_w, item_h)."""
        return 4

    @property
    def cnn_flat_dim(self) -> int:
        """Flattened CNN output: channels[-1] * pool_h * pool_w."""
        return self.cnn_channels[-1] * self.cnn_pool_size[0] * self.cnn_pool_size[1]

    @property
    def batch_size(self) -> int:
        """Total rollout batch: num_envs * rollout_steps."""
        return self.num_envs * self.rollout_steps

    @property
    def total_timesteps(self) -> int:
        """Total environment steps: num_updates * batch_size."""
        return self.num_updates * self.batch_size

    def get_curriculum_phase(self, update: int) -> CurriculumPhase:
        """
        Return the curriculum phase for the given update number.

        Falls back to the last phase if update exceeds total_updates.
        """
        if not self.use_curriculum or not self.curriculum_phases:
            return CurriculumPhase(
                start_frac=0.0, end_frac=1.0,
                num_boxes=self.num_boxes_per_episode,
                size_range=self.box_size_range,
            )
        frac = update / max(self.num_updates, 1)
        for phase in self.curriculum_phases:
            if phase.start_frac <= frac < phase.end_frac:
                return phase
        return self.curriculum_phases[-1]

    def to_dict(self) -> dict:
        """Serialise all parameters (for logging)."""
        d = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if hasattr(v, '__dict__') and not isinstance(v, type):
                    d[k] = {kk: vv for kk, vv in v.__dict__.items()
                            if not kk.startswith('_')}
                elif isinstance(v, list):
                    d[k] = [
                        {kk: vv for kk, vv in item.__dict__.items()
                         if not kk.startswith('_')}
                        if hasattr(item, '__dict__') else item
                        for item in v
                    ]
                else:
                    d[k] = v
        # Computed properties
        d['grid_l'] = self.grid_l
        d['grid_w'] = self.grid_w
        d['action_grid_l'] = self.action_grid_l
        d['action_grid_w'] = self.action_grid_w
        d['num_actions'] = self.num_actions
        d['combined_features'] = self.combined_features
        d['batch_size'] = self.batch_size
        d['total_timesteps'] = self.total_timesteps
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "A2CMaskedConfig":
        """Create from a dictionary (ignores unknown/computed keys)."""
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {}
        for k, v in d.items():
            if k in valid_fields:
                filtered[k] = v
        # Reconstruct curriculum phases if present
        if 'curriculum_phases' in filtered and isinstance(filtered['curriculum_phases'], list):
            phases = []
            for p in filtered['curriculum_phases']:
                if isinstance(p, dict):
                    phases.append(CurriculumPhase(**p))
                elif isinstance(p, CurriculumPhase):
                    phases.append(p)
            filtered['curriculum_phases'] = phases
        # Reconstruct reward config if present
        if 'reward_config' in filtered and isinstance(filtered['reward_config'], dict):
            filtered['reward_config'] = RewardConfig(**filtered['reward_config'])
        return cls(**filtered)
