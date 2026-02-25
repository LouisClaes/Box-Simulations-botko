"""
Wall-E Scoring strategy -- multi-criteria scoring function for 3D bin packing.

Based on the approach described by Verma et al. (2020), this strategy evaluates
every feasible (x, y, orientation) candidate using a weighted combination of
sub-scores that capture different packing quality aspects:

  * G_var      -- surface variance after placement    (minimize)
  * G_high     -- how deep the box nestles in a valley (maximize)
  * G_flush    -- number of flush faces with walls/boxes (maximize)
  * pos_pen    -- distance from the origin corner      (minimize)
  * height_pen -- absolute placement height            (minimize)

The final composite score is:
    S = -alpha_var * G_var
        + alpha_high * G_high
        + alpha_flush * G_flush
        - alpha_pos * position_penalty
        - alpha_height * height_penalty

The candidate with the highest S is selected.

This is a deterministic, exhaustive strategy: it evaluates every legal position
on the grid (with a configurable step size) and every allowed orientation.  It
does NOT modify bin_state -- all heightmap operations are done on numpy copies.
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

# Scoring weights (from Verma et al. 2020, tuned for 120x80x150 bin)
ALPHA_VAR: float = 0.75       # Weight for surface variance (penalty)
ALPHA_HIGH: float = 1.0       # Weight for valley-nesting bonus
ALPHA_FLUSH: float = 1.0      # Weight for flush-face bonus
ALPHA_POS: float = 0.01       # Weight for position penalty (mild)
ALPHA_HEIGHT: float = 1.0     # Weight for height penalty

# Margin (in grid cells) around the box footprint for variance computation.
VARIANCE_MARGIN: int = 2

# Maximum number of flush faces a box can have (4 walls + bottom + top).
MAX_FLUSH_FACES: int = 6

# Tolerance for detecting flush contact (in real-world units, e.g. cm).
FLUSH_TOLERANCE: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

@register_strategy
class WallEScoringStrategy(BaseStrategy):
    """
    Wall-E Scoring strategy: deterministic multi-criteria placement.

    For every (x, y, orientation) candidate the strategy computes a composite
    score from five normalised sub-scores, then returns the single best one.
    The sub-scores capture surface smoothness, valley usage, wall contact,
    proximity to the origin, and absolute height.

    Hyper-parameters are defined as module-level constants and can be
    adjusted without changing any logic.
    """

    name = "walle_scoring"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = 1.0

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Store config and compute the grid scan step."""
        super().on_episode_start(config)
        self._scan_step = max(1.0, config.bin.resolution)

    # ── Main entry point ──────────────────────────────────────────────────

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Evaluate all feasible positions and return the highest-scoring one.

        Args:
            box:       The box to place (original dimensions).
            bin_state: Current 3D bin state (read-only).

        Returns:
            PlacementDecision with the best (x, y, orientation_idx), or None
            if the box cannot be placed anywhere.
        """
        cfg = self.config
        bin_cfg = cfg.bin
        step = self._scan_step

        # Resolve allowed orientations based on experiment config
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Take a read-only snapshot of the heightmap for scoring computations
        heightmap = bin_state.heightmap  # DO NOT modify -- used read-only below

        best_score: float = -np.inf
        best_candidate: Optional[Tuple[float, float, int]] = None  # (x, y, oidx)

        for oidx, (ol, ow, oh) in enumerate(orientations):
            # Skip orientations that can never fit in the bin
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue

            x = 0.0
            while x + ol <= bin_cfg.length + 1e-6:
                y = 0.0
                while y + ow <= bin_cfg.width + 1e-6:
                    # ── Resting height ────────────────────────────────
                    z = bin_state.get_height_at(x, y, ol, ow)

                    # Height bounds check
                    if z + oh > bin_cfg.height + 1e-6:
                        y += step
                        continue

                    # Anti-float support check
                    if z > 0.5:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < MIN_SUPPORT:
                            y += step
                            continue

                    # Stricter stability when enabled
                    if cfg.enable_stability:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < cfg.min_support_ratio:
                            y += step
                            continue

                    # Margin check (box-to-box gap enforcement)
                    if not bin_state.is_margin_clear(x, y, ol, ow, z, oh):
                        y += step
                        continue

                    # ── Compute sub-scores ────────────────────────────
                    g_var = self._compute_variance_score(
                        heightmap, x, y, ol, ow, oh, z, bin_cfg,
                    )
                    g_high = self._compute_valley_score(
                        heightmap, x, y, ol, ow, z, bin_cfg,
                    )
                    g_flush = self._compute_flush_score(
                        heightmap, x, y, ol, ow, oh, z, bin_cfg,
                    )
                    pos_pen = self._compute_position_penalty(x, y, bin_cfg)
                    h_pen = self._compute_height_penalty(z, bin_cfg)

                    # ── Composite score ───────────────────────────────
                    score = (
                        -ALPHA_VAR * g_var
                        + ALPHA_HIGH * g_high
                        + ALPHA_FLUSH * g_flush
                        - ALPHA_POS * pos_pen
                        - ALPHA_HEIGHT * h_pen
                    )

                    if score > best_score:
                        best_score = score
                        best_candidate = (x, y, oidx)

                    y += step
                x += step

        if best_candidate is None:
            return None

        bx, by, b_oidx = best_candidate
        return PlacementDecision(x=bx, y=by, orientation_idx=b_oidx)

    # ── Sub-score computations ────────────────────────────────────────────

    def _compute_variance_score(
        self,
        heightmap: np.ndarray,
        x: float, y: float,
        ol: float, ow: float, oh: float,
        z: float,
        bin_cfg,
    ) -> float:
        """
        G_var: height variance in a local neighbourhood after virtual placement.

        We take a region around the box footprint (with VARIANCE_MARGIN padding),
        copy it, paint the box's new top surface into the copy, and compute the
        variance.  Normalised by bin_height^2 so the score stays in [0, 1].

        Lower variance = more uniform surface = better.
        """
        res = bin_cfg.resolution
        grid_l = bin_cfg.grid_l
        grid_w = bin_cfg.grid_w

        # Grid indices for the box footprint
        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), grid_l)
        gy_end = min(gy + int(round(ow / res)), grid_w)

        # Expand with margin for neighbourhood context
        rx_start = max(0, gx - VARIANCE_MARGIN)
        ry_start = max(0, gy - VARIANCE_MARGIN)
        rx_end = min(grid_l, gx_end + VARIANCE_MARGIN)
        ry_end = min(grid_w, gy_end + VARIANCE_MARGIN)

        # Copy the neighbourhood region so we don't mutate the real heightmap
        region = heightmap[rx_start:rx_end, ry_start:ry_end].copy()

        # Paint the box footprint with the new top-of-box height
        box_top = z + oh
        local_gx = gx - rx_start
        local_gy = gy - ry_start
        local_gx_end = gx_end - rx_start
        local_gy_end = gy_end - ry_start
        region[local_gx:local_gx_end, local_gy:local_gy_end] = np.maximum(
            region[local_gx:local_gx_end, local_gy:local_gy_end],
            box_top,
        )

        if region.size == 0:
            return 0.0

        variance = float(np.var(region))
        # Normalise by bin_height^2
        max_var = bin_cfg.height ** 2
        return min(variance / max_var, 1.0) if max_var > 0 else 0.0

    def _compute_valley_score(
        self,
        heightmap: np.ndarray,
        x: float, y: float,
        ol: float, ow: float,
        z: float,
        bin_cfg,
    ) -> float:
        """
        G_high: how well the box nestles into a valley.

        Computed as (max_neighbour_height - z) / bin_height.
        If the surrounding terrain is higher than z, the box is "sunken"
        into a valley, which is desirable.  Score is clamped to [0, 1].
        """
        res = bin_cfg.resolution
        grid_l = bin_cfg.grid_l
        grid_w = bin_cfg.grid_w

        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), grid_l)
        gy_end = min(gy + int(round(ow / res)), grid_w)

        # Neighbourhood just outside the box footprint
        margin = VARIANCE_MARGIN
        nx_start = max(0, gx - margin)
        ny_start = max(0, gy - margin)
        nx_end = min(grid_l, gx_end + margin)
        ny_end = min(grid_w, gy_end + margin)

        # Extract full neighbourhood
        neighbourhood = heightmap[nx_start:nx_end, ny_start:ny_end]
        if neighbourhood.size == 0:
            return 0.0

        max_neighbour_h = float(np.max(neighbourhood))

        # Score: how much higher the neighbours are compared to placement z
        diff = max_neighbour_h - z
        if bin_cfg.height <= 0:
            return 0.0
        return max(0.0, min(diff / bin_cfg.height, 1.0))

    def _compute_flush_score(
        self,
        heightmap: np.ndarray,
        x: float, y: float,
        ol: float, ow: float, oh: float,
        z: float,
        bin_cfg,
    ) -> float:
        """
        G_flush: count of box faces flush with walls or adjacent boxes.

        Checks six faces:
          - Left   (x == 0):          flush with left bin wall
          - Right  (x + ol == length): flush with right bin wall
          - Back   (y == 0):          flush with back bin wall
          - Front  (y + ow == width):  flush with front bin wall
          - Bottom (z == 0 or matches heightmap): flush with floor or box below
          - Top:   bonus if height matches adjacent boxes

        Normalised by MAX_FLUSH_FACES (6).
        """
        flush_count = 0.0

        # Left wall
        if x <= FLUSH_TOLERANCE:
            flush_count += 1.0

        # Right wall
        if abs((x + ol) - bin_cfg.length) <= FLUSH_TOLERANCE:
            flush_count += 1.0

        # Back wall
        if y <= FLUSH_TOLERANCE:
            flush_count += 1.0

        # Front wall
        if abs((y + ow) - bin_cfg.width) <= FLUSH_TOLERANCE:
            flush_count += 1.0

        # Bottom face: flush with floor or with a supporting box surface
        if z < FLUSH_TOLERANCE:
            # Resting on the floor
            flush_count += 1.0
        else:
            # Check if the box base aligns well with the heightmap
            # (i.e. good support indicates flush contact with box below)
            res = bin_cfg.resolution
            gx = int(round(x / res))
            gy = int(round(y / res))
            gx_end = min(gx + int(round(ol / res)), bin_cfg.grid_l)
            gy_end = min(gy + int(round(ow / res)), bin_cfg.grid_w)
            footprint = heightmap[gx:gx_end, gy:gy_end]
            if footprint.size > 0:
                # Fraction of base cells that match z (i.e. support the box)
                matching = float(np.mean(np.abs(footprint - z) <= res * 0.5))
                flush_count += matching  # partial credit

        # Top face: bonus if box top matches height of adjacent columns
        box_top = z + oh
        res = bin_cfg.resolution
        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), bin_cfg.grid_l)
        gy_end = min(gy + int(round(ow / res)), bin_cfg.grid_w)

        # Sample adjacent columns (one cell outside the footprint on each side)
        adjacent_heights: List[float] = []
        if gx > 0:
            adjacent_heights.append(float(np.max(heightmap[gx - 1, gy:gy_end])))
        if gx_end < bin_cfg.grid_l:
            adjacent_heights.append(float(np.max(heightmap[gx_end, gy:gy_end])))
        if gy > 0:
            adjacent_heights.append(float(np.max(heightmap[gx:gx_end, gy - 1])))
        if gy_end < bin_cfg.grid_w:
            adjacent_heights.append(float(np.max(heightmap[gx:gx_end, gy_end])))

        if adjacent_heights:
            # Partial credit for how many adjacent edges match the box top
            matches = sum(1.0 for h in adjacent_heights if abs(h - box_top) <= FLUSH_TOLERANCE)
            flush_count += matches / len(adjacent_heights)

        return flush_count / MAX_FLUSH_FACES

    @staticmethod
    def _compute_position_penalty(x: float, y: float, bin_cfg) -> float:
        """
        Position penalty: penalises distance from the origin corner (0, 0).

        Normalised to [0, 1] by dividing by (bin_length + bin_width).
        """
        denom = bin_cfg.length + bin_cfg.width
        if denom <= 0:
            return 0.0
        return (x + y) / denom

    @staticmethod
    def _compute_height_penalty(z: float, bin_cfg) -> float:
        """
        Height penalty: penalises placing boxes high up.

        Normalised to [0, 1] by dividing by bin_height.
        """
        if bin_cfg.height <= 0:
            return 0.0
        return z / bin_cfg.height
