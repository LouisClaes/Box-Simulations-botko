"""
Skyline strategy -- height-profile-based 3D bin packing.

This strategy builds a 1D "skyline profile" by projecting the heightmap onto
the x-axis (taking the minimum height per column).  It then sorts positions
by ascending height to identify the deepest valleys, and tries to fill them
first.

The intuition is simple: always fill the lowest gap.  This naturally produces
uniform layers and minimises wasted vertical space, because each box is pushed
into the deepest available depression before starting a new layer.

Scoring for each candidate position:
    score = -WEIGHT_Z * z
            + WEIGHT_VALLEY_FILL * valley_fill_bonus
            + WEIGHT_UNIFORMITY * uniformity_bonus

where:
  * z             -- the resting height (lower is better)
  * valley_fill   -- how much of the valley width the box covers (wider = better)
  * uniformity    -- negative variance of the footprint region after placement

The strategy does NOT modify bin_state.
"""

import numpy as np
from typing import Optional, List, Tuple

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Minimum support ratio -- matches the simulator's anti-float threshold.
MIN_SUPPORT: float = 0.30

# Scoring weights
WEIGHT_Z: float = 3.0           # Strongly prefer lower placements
WEIGHT_VALLEY_FILL: float = 1.5 # Reward covering more of the valley width
WEIGHT_UNIFORMITY: float = 0.5  # Reward uniform surface after placement

# Maximum number of valley x-positions to try before giving up on the
# skyline approach and falling back to a full grid scan.
MAX_VALLEY_CANDIDATES: int = 40

# Step size multiplier -- how coarsely to scan y within each valley column.
# 1.0 means step = max(1, resolution), matching the baseline.
Y_SCAN_STEP_MULT: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

@register_strategy
class SkylineStrategy(BaseStrategy):
    """
    Skyline (valley-first) placement strategy.

    1. Compute the skyline profile: for each x column, take the minimum
       height across all y values.  This represents the "reachable floor"
       at each x position.
    2. Sort x positions by ascending skyline height (deepest valleys first).
    3. For each valley x, try every allowed orientation and scan y positions
       to find valid placements.  Score each candidate by how low z is, how
       well the box fills the valley, and how uniform the surface will be
       after placement.
    4. Return the highest-scoring valid candidate, or None.
    """

    name = "skyline"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = 1.0

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Store config and derive scan step from resolution."""
        super().on_episode_start(config)
        self._scan_step = max(1.0, config.bin.resolution) * Y_SCAN_STEP_MULT

    # ── Main entry point ──────────────────────────────────────────────────

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Find the best valley-filling position for *box*.

        Args:
            box:       The box to place (original dimensions).
            bin_state: Current 3D bin state (read-only).

        Returns:
            PlacementDecision or None if the box cannot fit.
        """
        cfg = self.config
        bin_cfg = cfg.bin
        step = self._scan_step

        # Resolve allowed orientations
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        heightmap = bin_state.heightmap  # read-only reference

        # ── Step 1: Build the skyline profile ─────────────────────────────
        # For each x column, the skyline height is the minimum height across
        # all y cells in that column.  This tells us the "deepest reachable
        # point" at each x.
        skyline = np.min(heightmap, axis=1)  # shape: (grid_l,)

        # ── Step 2: Identify valley structure ─────────────────────────────
        # Sort x grid indices by ascending skyline height.
        valley_order = np.argsort(skyline)

        # Convert grid x indices to real-world x coordinates
        res = bin_cfg.resolution

        # ── Step 3: Search candidates ─────────────────────────────────────
        best_score: float = -np.inf
        best_candidate: Optional[Tuple[float, float, int]] = None

        # Track how many valley positions we have evaluated
        valleys_tried = 0

        for gx_start in valley_order:
            if valleys_tried >= MAX_VALLEY_CANDIDATES:
                break

            x_start = float(gx_start) * res
            valley_height = float(skyline[gx_start])

            # Compute the valley width: count contiguous columns at
            # approximately the same height as this valley floor.
            valley_width = self._measure_valley_width(
                skyline, gx_start, valley_height, res,
            )

            for oidx, (ol, ow, oh) in enumerate(orientations):
                # Quick reject: orientation can never fit in the bin
                if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                    continue

                # The box occupies x in [x_start, x_start + ol).
                # Check that the right edge fits inside the bin.
                if x_start + ol > bin_cfg.length + 1e-6:
                    continue

                # Scan y positions within this x column
                y = 0.0
                while y + ow <= bin_cfg.width + 1e-6:
                    z = bin_state.get_height_at(x_start, y, ol, ow)

                    # Height bounds
                    if z + oh > bin_cfg.height + 1e-6:
                        y += step
                        continue

                    # Anti-float support check
                    if z > 0.5:
                        sr = bin_state.get_support_ratio(x_start, y, ol, ow, z)
                        if sr < MIN_SUPPORT:
                            y += step
                            continue

                    # Stricter stability when enabled
                    if cfg.enable_stability:
                        sr = bin_state.get_support_ratio(x_start, y, ol, ow, z)
                        if sr < cfg.min_support_ratio:
                            y += step
                            continue

                    # ── Score this candidate ──────────────────────────
                    valley_fill = self._valley_fill_bonus(ol, valley_width)
                    uniformity = self._uniformity_bonus(
                        heightmap, x_start, y, ol, ow, oh, z, bin_cfg,
                    )

                    score = (
                        -WEIGHT_Z * z
                        + WEIGHT_VALLEY_FILL * valley_fill
                        + WEIGHT_UNIFORMITY * uniformity
                    )

                    if score > best_score:
                        best_score = score
                        best_candidate = (x_start, y, oidx)

                    y += step

            valleys_tried += 1

        # ── Step 4: Return result ─────────────────────────────────────────
        if best_candidate is None:
            return None

        bx, by, b_oidx = best_candidate
        return PlacementDecision(x=bx, y=by, orientation_idx=b_oidx)

    # ── Helper methods ────────────────────────────────────────────────────

    @staticmethod
    def _measure_valley_width(
        skyline: np.ndarray,
        gx_center: int,
        valley_floor: float,
        resolution: float,
    ) -> float:
        """
        Measure the width (in real-world units) of the valley around gx_center.

        A valley is defined as a contiguous run of x columns whose skyline
        height is within one resolution unit of valley_floor.  We expand
        left and right from gx_center.

        Args:
            skyline:      1D array of min-heights per x column.
            gx_center:    Grid x index at the valley's deepest point.
            valley_floor: Height at the valley floor.
            resolution:   Grid resolution (cm per cell).

        Returns:
            Valley width in real-world units (cm).
        """
        tolerance = resolution * 1.5
        n = len(skyline)

        # Expand left
        left = gx_center
        while left > 0 and abs(skyline[left - 1] - valley_floor) <= tolerance:
            left -= 1

        # Expand right
        right = gx_center
        while right < n - 1 and abs(skyline[right + 1] - valley_floor) <= tolerance:
            right += 1

        # Width = number of columns * resolution
        return float(right - left + 1) * resolution

    @staticmethod
    def _valley_fill_bonus(box_length: float, valley_width: float) -> float:
        """
        How much of the valley width the box covers.

        Returns a value in (0, 1].  A box that perfectly fills the valley
        scores 1.0.  A box wider than the valley is capped at 1.0.
        """
        if valley_width <= 0:
            return 0.0
        return min(box_length / valley_width, 1.0)

    @staticmethod
    def _uniformity_bonus(
        heightmap: np.ndarray,
        x: float, y: float,
        ol: float, ow: float, oh: float,
        z: float,
        bin_cfg,
    ) -> float:
        """
        Negative of the height variance in the box's footprint region after
        a virtual placement.  Lower variance = higher bonus.

        We copy the footprint region, paint the box top surface in, and
        compute variance.  The result is negated and normalised so that
        a perfectly flat surface yields bonus = 0 and rough surfaces
        yield negative values.

        Returns:
            A non-positive float (0.0 = perfectly uniform, negative = rough).
        """
        res = bin_cfg.resolution
        grid_l = bin_cfg.grid_l
        grid_w = bin_cfg.grid_w

        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), grid_l)
        gy_end = min(gy + int(round(ow / res)), grid_w)

        if gx >= gx_end or gy >= gy_end:
            return 0.0

        # Copy the region to avoid mutating the real heightmap
        region = heightmap[gx:gx_end, gy:gy_end].copy()

        # Paint in the new box top
        box_top = z + oh
        region = np.maximum(region, box_top)

        variance = float(np.var(region))

        # Normalise by bin_height^2 and negate (lower variance = higher bonus)
        max_var = bin_cfg.height ** 2
        if max_var <= 0:
            return 0.0
        return -(variance / max_var)
