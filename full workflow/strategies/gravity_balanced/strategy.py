"""
Gravity-Balanced Packer strategy for 3D bin packing.

NOVEL STRATEGY -- not based on any published paper. This is an original
algorithm designed specifically for this simulator.

Core idea:
    Optimize placement to keep the center of gravity (CoG) low and centered
    in the bin.  A low, centered CoG creates physically stable stacking that
    prevents tipping during transport, while still maintaining high fill rates.

Key insight:
    In real-world palletizing, the most common cause of damage during
    transport is load tipping due to an off-center or high CoG.  By treating
    CoG optimization as a primary objective (rather than an afterthought),
    we produce stacking patterns that are inherently stable while still
    achieving competitive fill rates.

Algorithm:
    1. Compute the current center of gravity from all placed boxes.
       (Volume is used as a proxy for weight since all default weights are 1.0.)
    2. Generate candidate positions from a coarse grid (step=2cm) plus
       placed-box corners and bin corners.
    3. For each candidate (x, y, orientation):
       a. Compute resting z, check bounds and support.
       b. Compute the hypothetical NEW CoG after placing this box.
       c. Evaluate the CoG quality: lateral centering + height minimization.
       d. Compute a composite score from CoG quality, placement height,
          support ratio, and fill efficiency.
    4. Return the highest-scoring candidate.

Hyperparameters:
    WEIGHT_COG_HEIGHT   = 2.0  -- keep CoG low
    WEIGHT_COG_LATERAL  = 2.0  -- keep CoG centered in the x-y plane
    WEIGHT_LOW_Z        = 3.0  -- prefer low placements (floor first)
    WEIGHT_SUPPORT      = 1.5  -- prefer well-supported positions
    WEIGHT_FILL_EFF     = 1.0  -- don't sacrifice fill rate for CoG
    COARSE_STEP         = 2.0  -- grid scan step for candidate generation
    MIN_SUPPORT         = 0.30 -- anti-float threshold (matches simulator)
"""

import math
import numpy as np
from typing import Optional, List, Tuple, Set

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ---------------------------------------------------------------------------
# Module-level constants (hyperparameters)
# ---------------------------------------------------------------------------

# Anti-float threshold -- matches the simulator's rejection limit.
MIN_SUPPORT: float = 0.30

# Scoring weights
WEIGHT_COG_HEIGHT: float = 2.0    # Keep center of gravity low
WEIGHT_COG_LATERAL: float = 2.0   # Keep center of gravity centered in x-y
WEIGHT_LOW_Z: float = 3.0         # Prefer lower absolute placement height
WEIGHT_SUPPORT: float = 1.5       # Prefer well-supported positions
WEIGHT_FILL_EFF: float = 1.0      # Don't sacrifice fill rate

# Candidate generation: coarse grid step to keep runtime reasonable
# (120x80 / 2 = 2400 candidates vs 9600 at step=1).
COARSE_STEP: float = 2.0


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

@register_strategy
class GravityBalancedStrategy(BaseStrategy):
    """
    Gravity-Balanced Packer: placement strategy that optimizes the center
    of gravity to be low and centered.

    The strategy computes the hypothetical center of gravity after each
    candidate placement and scores candidates based on how well they keep
    the CoG low and centered. This produces physically stable stacking
    configurations that resist tipping during transport.

    The CoG is computed using box volume as a weight proxy (all boxes have
    weight 1.0 by default, so volume determines each box's contribution
    to the total CoG).

    Edge cases:
        - First box: placed at (0, 0) corner for maximum stability.
        - Empty bin: CoG computation is skipped; BLF logic is used.

    Attributes:
        name: Strategy identifier for the registry ("gravity_balanced").
    """

    name: str = "gravity_balanced"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = COARSE_STEP

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Store config and derive the grid scan step."""
        super().on_episode_start(config)
        self._scan_step = max(COARSE_STEP, config.bin.resolution)

    # -- Main entry point ---------------------------------------------------

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Find the best placement that optimizes center-of-gravity quality.

        Steps:
            1. Compute current CoG from placed boxes.
            2. Build candidate (x, y) positions from coarse grid + corners.
            3. For each candidate and orientation, check feasibility.
            4. Compute hypothetical new CoG and score the candidate.
            5. Return the highest-scoring feasible candidate.

        Args:
            box:       The box to place (original dimensions before rotation).
            bin_state: Current 3D bin state (read-only -- must NOT be modified).

        Returns:
            PlacementDecision(x, y, orientation_idx) or None if no valid
            placement exists.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        # Resolve allowed orientations
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Filter to orientations that can physically fit
        valid_orientations = [
            (oidx, ol, ow, oh)
            for oidx, (ol, ow, oh) in enumerate(orientations)
            if ol <= bin_cfg.length and ow <= bin_cfg.width and oh <= bin_cfg.height
        ]
        if not valid_orientations:
            return None

        # --- Edge case: empty bin (first box) ---
        # Place the first box at the back-left corner (0, 0) for stability.
        if len(bin_state.placed_boxes) == 0:
            return self._place_first_box(valid_orientations, bin_state, bin_cfg)

        # --- Compute current center of gravity ---
        cog_x, cog_y, cog_z, total_volume = self._compute_current_cog(
            bin_state
        )

        # --- Build candidates ---
        candidates = self._generate_candidates(bin_state, self._scan_step)

        # Bin center and max lateral distance (precompute for scoring)
        center_x = bin_cfg.length / 2.0
        center_y = bin_cfg.width / 2.0
        max_lateral = math.sqrt(center_x ** 2 + center_y ** 2)

        best_score: float = -np.inf
        best_candidate: Optional[Tuple[float, float, int]] = None

        for cx, cy in candidates:
            for oidx, ol, ow, oh in valid_orientations:
                # --- Bounds check ---
                if cx + ol > bin_cfg.length + 1e-6:
                    continue
                if cy + ow > bin_cfg.width + 1e-6:
                    continue

                # --- Resting height ---
                z = bin_state.get_height_at(cx, cy, ol, ow)

                # --- Height limit ---
                if z + oh > bin_cfg.height + 1e-6:
                    continue

                # --- Anti-float support check ---
                support_ratio = 1.0
                if z > 0.5:
                    support_ratio = bin_state.get_support_ratio(
                        cx, cy, ol, ow, z
                    )
                    if support_ratio < MIN_SUPPORT:
                        continue

                # --- Stricter stability when enabled ---
                if cfg.enable_stability and z > 0.5:
                    if support_ratio < cfg.min_support_ratio:
                        continue

                # --- Compute hypothetical new CoG ---
                box_vol = ol * ow * oh
                new_vol = total_volume + box_vol

                new_cog_x = (
                    (cog_x * total_volume + (cx + ol / 2.0) * box_vol) / new_vol
                )
                new_cog_y = (
                    (cog_y * total_volume + (cy + ow / 2.0) * box_vol) / new_vol
                )
                new_cog_z = (
                    (cog_z * total_volume + (z + oh / 2.0) * box_vol) / new_vol
                )

                # --- CoG quality scores ---
                # Lateral score: how centered is the new CoG in x-y?
                lateral_dist = math.sqrt(
                    (new_cog_x - center_x) ** 2
                    + (new_cog_y - center_y) ** 2
                )
                lateral_score = (
                    1.0 - lateral_dist / max_lateral
                    if max_lateral > 0
                    else 1.0
                )

                # Height score: how low is the new CoG?
                height_score = (
                    1.0 - new_cog_z / bin_cfg.height
                    if bin_cfg.height > 0
                    else 1.0
                )

                # --- Fill efficiency ---
                # How well does this box use the space it occupies?
                # Ratio of box volume to the "effective column" it occupies.
                effective_column_height = max(oh, z + oh)
                column_volume = ol * ow * effective_column_height
                fill_efficiency = (
                    box_vol / column_volume
                    if column_volume > 0
                    else 0.0
                )

                # --- Composite score ---
                z_norm = z / bin_cfg.height if bin_cfg.height > 0 else 0.0
                score = (
                    WEIGHT_COG_HEIGHT * height_score
                    + WEIGHT_COG_LATERAL * lateral_score
                    - WEIGHT_LOW_Z * z_norm
                    + WEIGHT_SUPPORT * support_ratio
                    + WEIGHT_FILL_EFF * fill_efficiency
                )

                if best_candidate is None or score > best_score:
                    best_score = score
                    best_candidate = (cx, cy, oidx)

        if best_candidate is None:
            return None

        return PlacementDecision(
            x=best_candidate[0],
            y=best_candidate[1],
            orientation_idx=best_candidate[2],
        )

    # -- First box placement ------------------------------------------------

    def _place_first_box(
        self,
        valid_orientations: List[Tuple[int, float, float, float]],
        bin_state: BinState,
        bin_cfg,
    ) -> Optional[PlacementDecision]:
        """
        Special handling for the first box: place at origin (0, 0).

        The first box is placed at the back-left corner for maximum wall
        contact and stability. Among valid orientations, we prefer the one
        that creates the flattest base (largest footprint area).

        Args:
            valid_orientations: List of (oidx, ol, ow, oh) tuples.
            bin_state:          Current bin state (empty).
            bin_cfg:            Bin configuration.

        Returns:
            PlacementDecision at (0, 0) with the best orientation, or None.
        """
        best_oidx: Optional[int] = None
        best_footprint: float = -1.0

        for oidx, ol, ow, oh in valid_orientations:
            # Must fit at origin
            if ol > bin_cfg.length + 1e-6 or ow > bin_cfg.width + 1e-6:
                continue
            if oh > bin_cfg.height + 1e-6:
                continue

            footprint = ol * ow
            if footprint > best_footprint:
                best_footprint = footprint
                best_oidx = oidx

        if best_oidx is None:
            return None

        return PlacementDecision(x=0.0, y=0.0, orientation_idx=best_oidx)

    # -- Center of gravity computation --------------------------------------

    @staticmethod
    def _compute_current_cog(
        bin_state: BinState,
    ) -> Tuple[float, float, float, float]:
        """
        Compute the current center of gravity from all placed boxes.

        Uses volume as a proxy for weight (all boxes default to weight 1.0,
        so a larger box contributes more to the CoG).

        Args:
            bin_state: Current bin state with placed boxes.

        Returns:
            Tuple of (cog_x, cog_y, cog_z, total_volume).
            If no boxes are placed, returns (0, 0, 0, 0).
        """
        placed = bin_state.placed_boxes
        if not placed:
            return (0.0, 0.0, 0.0, 0.0)

        total_volume = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        weighted_z = 0.0

        for p in placed:
            vol = p.volume
            total_volume += vol
            # Center of each placed box
            weighted_x += vol * (p.x + p.oriented_l / 2.0)
            weighted_y += vol * (p.y + p.oriented_w / 2.0)
            weighted_z += vol * (p.z + p.oriented_h / 2.0)

        if total_volume <= 0:
            return (0.0, 0.0, 0.0, 0.0)

        return (
            weighted_x / total_volume,
            weighted_y / total_volume,
            weighted_z / total_volume,
            total_volume,
        )

    # -- Candidate generation -----------------------------------------------

    def _generate_candidates(
        self,
        bin_state: BinState,
        step: float,
    ) -> List[Tuple[float, float]]:
        """
        Generate candidate (x, y) positions from three sources:

        1. Coarse grid scan at the given step size (default 2cm).
        2. Corners of all placed boxes (right edge, front edge, origin).
        3. Bin corners: (0,0), (length,0), (0,width), (length,width).

        The coarse grid keeps runtime manageable (O(L*W/step^2) candidates)
        while the box corners ensure we consider tight-packing positions
        that may fall between grid points.

        Candidates are sorted by (z, x, y) so low positions are evaluated
        first (since we heavily penalize high placements).

        Args:
            bin_state: Current bin state (read-only).
            step:      Grid scanning step size (cm).

        Returns:
            List of unique (x, y) candidate positions, sorted by estimated
            quality.
        """
        bin_cfg = bin_state.config
        seen: Set[Tuple[float, float]] = set()
        candidates: List[Tuple[float, float]] = []

        # Source 1: Coarse grid scan
        x = 0.0
        while x <= bin_cfg.length:
            y = 0.0
            while y <= bin_cfg.width:
                pt = (x, y)
                if pt not in seen:
                    seen.add(pt)
                    candidates.append(pt)
                y += step
            x += step

        # Source 2: Placed box corners
        for p in bin_state.placed_boxes:
            corner_points = [
                (p.x, p.y),
                (p.x_max, p.y),
                (p.x, p.y_max),
                (p.x_max, p.y_max),
            ]
            for pt in corner_points:
                if (pt not in seen
                        and 0 <= pt[0] <= bin_cfg.length
                        and 0 <= pt[1] <= bin_cfg.width):
                    seen.add(pt)
                    candidates.append(pt)

        # Source 3: Bin corners (ensure they are always candidates)
        bin_corners = [
            (0.0, 0.0),
            (bin_cfg.length, 0.0),
            (0.0, bin_cfg.width),
        ]
        for pt in bin_corners:
            if pt not in seen:
                seen.add(pt)
                candidates.append(pt)

        # Sort by estimated height then by position for efficient scanning
        def sort_key(pt: Tuple[float, float]) -> Tuple[float, float, float]:
            z_est = bin_state.get_height_at(pt[0], pt[1], 1.0, 1.0)
            return (z_est, pt[0], pt[1])

        candidates.sort(key=sort_key)
        return candidates
