"""
RewardShaper — Configurable reward computation for RL training.

Implements multi-component reward shaping based on insights from:
  - Zhao et al. (ICLR 2022): Volume-ratio reward with constraint penalties
  - Tsang et al. (2025): Pyramid + compactness composite reward
  - Xiong et al. (RA-L 2024): Terminal vs step-wise reward comparison
  - Verma et al. (AAAI 2020): Retroactive terminal reward

Key design decisions:
  1. Dense step rewards (not sparse terminal-only) — faster learning
  2. Multi-component: volume + surface contact + height penalty + stability
  3. Pallet-close bonus — incentivise high fill before close
  4. Terminal bonus — overall session quality signal
  5. Penalties for skips, invalid actions, and rejections

All weights are configurable via RewardConfig for easy ablation studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Forward reference — avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config import Box, BinConfig
    from simulator.bin_state import BinState


@dataclass
class RewardConfig:
    """
    Reward shaping weights.

    All weights can be tuned independently for ablation studies.
    The default values are calibrated for the Botko BV setup.

    Reference calibration:
      - A typical box volume ≈ 0.1-1.0% of bin volume
      - fill_delta per step ≈ 0.001-0.01
      - Episode length ≈ 50-200 steps
      - Good episode return ≈ 5-15
    """

    # ── Step rewards ──
    volume_weight: float = 10.0
    """Reward per unit of normalised volume placed (primary signal)."""

    fill_delta_weight: float = 5.0
    """Reward for fill rate improvement (encourages dense packing)."""

    surface_contact_weight: float = 2.0
    """Bonus for high surface contact ratio (Zhao et al. C2)."""

    height_penalty_weight: float = -1.0
    """Penalty for placing at high z (encourages filling bottom first)."""

    roughness_penalty_weight: float = -0.5
    """Penalty for increasing surface roughness."""

    # ── Pallet close rewards ──
    close_bonus_weight: float = 5.0
    """Bonus when a pallet is closed (scaled by fill rate)."""

    close_fill_threshold: float = 0.5
    """Minimum fill rate for close bonus (below = penalty)."""

    # ── Terminal rewards ──
    terminal_fill_weight: float = 10.0
    """Terminal bonus proportional to avg_closed_fill."""

    terminal_placement_bonus: float = 2.0
    """Terminal bonus proportional to placement_rate."""

    # ── Penalties ──
    rejection_penalty: float = -0.5
    """Penalty for attempted placement that fails validation."""

    skip_penalty: float = -0.3
    """Penalty for advancing conveyor without placing."""

    invalid_action_penalty: float = -1.0
    """Penalty for selecting an invalid action (box not in window)."""


class RewardShaper:
    """
    Computes shaped rewards from packing events.

    Stateless — all information comes from the arguments.
    Thread-safe for use in vectorized environments.

    Usage:
        shaper = RewardShaper(RewardConfig())
        reward = shaper.placement_reward(box, bin_state, bin_config, ...)
        reward += shaper.terminal_reward(avg_fill, pallets_closed, ...)
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def placement_reward(
        self,
        box,            # Box
        bin_state,      # BinState
        bin_config,     # BinConfig
        fill_delta: float,
        pallet_closed: bool = False,
        closed_fill: float = 0.0,
    ) -> float:
        """
        Compute reward for a successful placement.

        Components:
          1. Volume ratio: box.volume / bin.volume × weight
          2. Fill delta: improvement in fill rate × weight
          3. Surface contact: support_ratio at placement × weight
          4. Height penalty: z_placement / bin_height × weight
          5. Roughness: current roughness × weight
          6. Close bonus: if pallet closed, fill × weight

        Args:
            box:           The placed box.
            bin_state:     State AFTER placement.
            bin_config:    Bin dimensions.
            fill_delta:    Change in fill rate from this placement.
            pallet_closed: Whether this placement triggered a pallet close.
            closed_fill:   Fill rate of the closed pallet (if applicable).

        Returns:
            Scalar reward.
        """
        cfg = self.config
        reward = 0.0

        # 1. Volume ratio
        vol_ratio = box.volume / bin_config.volume
        reward += cfg.volume_weight * vol_ratio

        # 2. Fill delta
        reward += cfg.fill_delta_weight * fill_delta

        # 3. Surface contact (proxy: support ratio of the last placed box)
        if bin_state.placed_boxes:
            last = bin_state.placed_boxes[-1]
            support = bin_state.get_support_ratio(
                last.x, last.y, last.oriented_l, last.oriented_w, last.z,
            )
            reward += cfg.surface_contact_weight * support

        # 4. Height penalty
        max_h = bin_state.get_max_height() / bin_config.height
        reward += cfg.height_penalty_weight * max_h

        # 5. Roughness penalty
        roughness = min(bin_state.get_surface_roughness() / 100.0, 1.0)
        reward += cfg.roughness_penalty_weight * roughness

        # 6. Pallet close bonus
        if pallet_closed:
            if closed_fill >= cfg.close_fill_threshold:
                reward += cfg.close_bonus_weight * closed_fill
            else:
                # Closing too early is bad
                reward -= cfg.close_bonus_weight * (cfg.close_fill_threshold - closed_fill)

        return reward

    def rejection_penalty(self) -> float:
        """Reward for a rejected placement attempt."""
        return self.config.rejection_penalty

    def skip_penalty(self) -> float:
        """Reward for advancing conveyor without placing."""
        return self.config.skip_penalty

    def invalid_action_penalty(self) -> float:
        """Reward for selecting an action with an invalid box index."""
        return self.config.invalid_action_penalty

    def terminal_reward(
        self,
        avg_fill: float,
        pallets_closed: int,
        placement_rate: float,
    ) -> float:
        """
        Terminal reward at end of episode.

        Provides a global quality signal based on the full session result.

        Args:
            avg_fill:       Average fill rate of closed pallets.
            pallets_closed: Number of pallets completed.
            placement_rate: Fraction of total boxes that were placed.

        Returns:
            Scalar terminal reward.
        """
        cfg = self.config
        reward = 0.0

        # Fill quality
        reward += cfg.terminal_fill_weight * avg_fill

        # Placement efficiency
        reward += cfg.terminal_placement_bonus * placement_rate

        return reward
