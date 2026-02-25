"""
Skyline strategy -- height-profile-based 3D bin packing.

This strategy builds a 1D "skyline profile" by projecting the heightmap onto
the y-axis (taking the mean height per y-band across all x columns).  It then
sorts y-positions by ascending height to identify the lowest bands, and tries
to fill them first.

The intuition is simple: always fill the lowest horizontal band.  This
naturally produces uniform layers and minimises wasted vertical space,
because each box is pushed into the lowest available band before starting a
new layer.

Scoring for each candidate position:
    score = -WEIGHT_Z * z
            + WEIGHT_VALLEY_FILL * valley_fill_bonus
            + WEIGHT_UNIFORMITY * uniformity_bonus

where:
  * z             -- the resting height (lower is better)
  * valley_fill   -- how much of the valley depth the box covers (wider = better)
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
WEIGHT_VALLEY_FILL: float = 1.5 # Reward covering more of the valley depth
WEIGHT_UNIFORMITY: float = 0.5  # Reward uniform surface after placement

# Maximum number of valley y-positions to try before giving up on the
# skyline approach and falling back to a full grid scan.
MAX_VALLEY_CANDIDATES: int = 40

# Step size multiplier -- how coarsely to scan x within each valley band.
# 1.0 means step = max(1, resolution), matching the baseline.
X_SCAN_STEP_MULT: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

@register_strategy
class SkylineStrategy(BaseStrategy):
    """
    Skyline (valley-first) placement strategy.

    1. Compute the skyline profile: for each y-band, take the mean height
       across all x columns.  This represents the average surface level at
       each y position.
    2. Sort y positions by ascending skyline height (lowest bands first).
    3. For each valley y, try every allowed orientation and scan x positions
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
        self._scan_step = max(1.0, config.bin.resolution) * X_SCAN_STEP_MULT

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
        # For each y-band, the skyline height is the mean height across all
        # x cells in that band.  This tells us the average surface level at
        # each y, avoiding false valleys from single empty cells.
        skyline = np.mean(heightmap, axis=0)  # shape: (grid_w,)

        # ── Step 2: Identify valley structure ─────────────────────────────
        # Sort y grid indices by ascending skyline height.
        valley_order = np.argsort(skyline)

        # Convert grid y indices to real-world y coordinates
        res = bin_cfg.resolution

        # ── Step 3: Search candidates ─────────────────────────────────────
        best_score: float = -np.inf
        best_candidate: Optional[Tuple[float, float, int]] = None

        # Track how many valley positions we have evaluated
        valleys_tried = 0

        for gy_start in valley_order:
            if valleys_tried >= MAX_VALLEY_CANDIDATES:
                break

            y_start = float(gy_start) * res
            valley_height = float(skyline[gy_start])

            # Compute the valley depth (along y): count contiguous y-bands
            # at approximately the same height as this valley floor.
            valley_depth = self._measure_valley_depth(
                skyline, gy_start, valley_height, res,
            )

            for oidx, (ol, ow, oh) in enumerate(orientations):
                # Quick reject: orientation can never fit in the bin
                if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                    continue

                # The box occupies y in [y_start, y_start + ow).
                # Check that the far edge fits inside the bin.
                if y_start + ow > bin_cfg.width + 1e-6:
                    continue

                # Scan x positions within this y-band
                x = 0.0
                while x + ol <= bin_cfg.length + 1e-6:
                    z = bin_state.get_height_at(x, y_start, ol, ow)

                    # Height bounds
                    if z + oh > bin_cfg.height + 1e-6:
                        x += step
                        continue

                    # Anti-float support check
                    if z > 0.5:
                        sr = bin_state.get_support_ratio(x, y_start, ol, ow, z)
                        if sr < MIN_SUPPORT:
                            x += step
                            continue

                    # Stricter stability when enabled
                    if cfg.enable_stability:
                        sr = bin_state.get_support_ratio(x, y_start, ol, ow, z)
                        if sr < cfg.min_support_ratio:
                            x += step
                            continue

                    # Margin check (box-to-box gap enforcement)
                    if not bin_state.is_margin_clear(x, y_start, ol, ow, z, oh):
                        x += step
                        continue

                    # ── Score this candidate ──────────────────────────
                    valley_fill = self._valley_fill_bonus(ow, valley_depth)
                    uniformity = self._uniformity_bonus(
                        heightmap, x, y_start, ol, ow, oh, z, bin_cfg,
                    )

                    score = (
                        -WEIGHT_Z * z
                        + WEIGHT_VALLEY_FILL * valley_fill
                        + WEIGHT_UNIFORMITY * uniformity
                    )

                    if score > best_score:
                        best_score = score
                        best_candidate = (x, y_start, oidx)

                    x += step

            valleys_tried += 1

        # ── Step 4: Return result ─────────────────────────────────────────
        if best_candidate is None:
            return None

        bx, by, b_oidx = best_candidate
        return PlacementDecision(x=bx, y=by, orientation_idx=b_oidx)

    # ── Helper methods ────────────────────────────────────────────────────

    @staticmethod
    def _measure_valley_depth(
        skyline: np.ndarray,
        gy_center: int,
        valley_floor: float,
        resolution: float,
    ) -> float:
        """
        Measure the depth (in real-world units) of the valley around gy_center.

        A valley is defined as a contiguous run of y-bands whose skyline
        height is within one resolution unit of valley_floor.  We expand
        left and right from gy_center.

        Args:
            skyline:      1D array of mean-heights per y-band.
            gy_center:    Grid y index at the valley's lowest point.
            valley_floor: Height at the valley floor.
            resolution:   Grid resolution (cm per cell).

        Returns:
            Valley depth in real-world units (cm).
        """
        tolerance = resolution * 1.5
        n = len(skyline)

        # Expand backward (toward y=0)
        left = gy_center
        while left > 0 and abs(skyline[left - 1] - valley_floor) <= tolerance:
            left -= 1

        # Expand forward (toward y=width)
        right = gy_center
        while right < n - 1 and abs(skyline[right + 1] - valley_floor) <= tolerance:
            right += 1

        # Depth = number of bands * resolution
        return float(right - left + 1) * resolution

    @staticmethod
    def _valley_fill_bonus(box_width: float, valley_depth: float) -> float:
        """
        How much of the valley depth the box covers.

        Returns a value in (0, 1].  A box that perfectly fills the valley
        scores 1.0.  A box wider than the valley is capped at 1.0.
        """
        if valley_depth <= 0:
            return 0.0
        return min(box_width / valley_depth, 1.0)

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
