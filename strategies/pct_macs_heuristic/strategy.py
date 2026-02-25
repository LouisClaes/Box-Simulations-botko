"""
MACS heuristic strategy for 3D bin packing — fast incremental implementation.

References:
    Zhao, H., Yu, Y., & Xu, K. (2022).
    "Deliberate Planning of 3D Bin Packing on Packing Configuration Trees."
    ICLR 2022.
    GitHub: https://github.com/alexfrom0815/Online-3D-BPP-PCT

    Zhao, H., et al. (2025).
    "PCT: Packing Configuration Trees for Online 3D Bin Packing."
    International Journal of Robotics Research (IJRR), 2025.

    Liu, B., Wang, H., Niu, B., Hao, J., & Zheng, C. (2020).
    "TAP-Net: Transport-and-Pack using Reinforcement Learning."
    ACM Transactions on Graphics (TOG), 39(6), 1-14.
    (MACS heuristic originates from the TAP-Net environment)

Algorithm:
    This module implements the MACS (Maximal Available Container Spaces)
    heuristic from the PCT repository, with a major performance improvement:
    the inner-loop heightmap copy is replaced by analytical delta computation.

    MACS score formula (higher is better):
        remaining_space    = sum(max(0, bin_height - heightmap_after[i,j]))
        roughness_penalty  = variance of heightmap_after
        height_penalty     = (z + oh) / bin_height

        score = remaining_space
                - MACS_ROUGHNESS_WEIGHT * roughness_penalty
                - MACS_HEIGHT_WEIGHT    * height_penalty

    Performance optimization (v2):
    The original implementation did O(grid_l × grid_w) NumPy operations
    (copy + sum + var) for EVERY (candidate, orientation) pair.  At 10cm
    resolution on a 120×80cm bin that is up to 576 full-array operations
    per box — causing ~1300ms/box.

    The new implementation precomputes:
        base_remaining:  total remaining space before placement
        base_sum:        sum of all heightmap values
        base_sum_sq:     sum of squares (for variance update)
        n_cells:         total grid cells

    Then for each candidate only the footprint cells are touched:
        delta_remaining: change in remaining space (footprint only)
        new variance:    updated via incremental formula using footprint delta

    Expected speedup: 5-15x → target <200ms/box.

Candidate generation:
    Identical to other strategies: full grid scan at bin resolution step,
    plus projected XY corners of all placed boxes.

Stability:
    MIN_SUPPORT = 0.30 is always enforced.  When ExperimentConfig.enable_stability
    is True, the stricter cfg.min_support_ratio is also enforced.
"""

import numpy as np
from typing import Optional, List, Tuple, Set, NamedTuple

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Anti-float threshold — must match the simulator's MIN_ANTI_FLOAT_RATIO.
MIN_SUPPORT: float = 0.30

# Floor-level threshold: placements at or below this height skip support check.
FLOOR_Z_THRESHOLD: float = 0.5

# MACS scoring weights
MACS_ROUGHNESS_WEIGHT: float = 0.1     # Penalty weight for heightmap variance
MACS_HEIGHT_WEIGHT: float = 100.0      # Penalty weight for normalised top height


# ---------------------------------------------------------------------------
# Precomputed bin statistics (computed once per box, before candidate loop)
# ---------------------------------------------------------------------------

class _BinStats(NamedTuple):
    """Precomputed statistics for fast incremental MACS scoring."""
    base_remaining: float   # sum(max(0, H - hm[i,j])) before placement
    base_sum: float         # sum(hm)   for variance update
    base_sum_sq: float      # sum(hm²)  for variance update
    n_cells: int            # total grid cells
    bin_height: float       # bin height (cached)


def _precompute_bin_stats(bin_state: BinState) -> _BinStats:
    """
    Compute heightmap aggregate statistics in one O(grid_l × grid_w) pass.

    Called once per box before the candidate evaluation loop.
    """
    hm = bin_state.heightmap
    bin_height = bin_state.config.height
    n_cells = hm.size

    base_remaining = float(np.sum(np.maximum(0.0, bin_height - hm)))
    base_sum = float(np.sum(hm))
    base_sum_sq = float(np.sum(hm * hm))

    return _BinStats(
        base_remaining=base_remaining,
        base_sum=base_sum,
        base_sum_sq=base_sum_sq,
        n_cells=n_cells,
        bin_height=bin_height,
    )


# ---------------------------------------------------------------------------
# Fast incremental MACS scoring
# ---------------------------------------------------------------------------

def _macs_score_fast(
    x: float,
    y: float,
    z: float,
    ol: float,
    ow: float,
    oh: float,
    bin_state: BinState,
    bin_cfg,
    stats: _BinStats,
) -> float:
    """
    Compute the MACS score for a candidate placement using incremental deltas.

    Instead of copying the full heightmap, only the footprint cells are read
    and the aggregate statistics are updated analytically:

        new_top = z + oh

        For each cell (i, j) in the footprint:
            old_h[i,j] = hm[i,j]
            new_h[i,j] = max(old_h[i,j], new_top)

        delta_remaining = sum(max(0, H-new_h) - max(0, H-old_h))
        remaining_space = base_remaining + delta_remaining

        new_sum    = base_sum    + sum(new_h - old_h)
        new_sum_sq = base_sum_sq + sum(new_h² - old_h²)
        new_mean   = new_sum / n_cells
        new_var    = new_sum_sq/n_cells - new_mean²

    Score = remaining_space - ROUGHNESS_WEIGHT * variance
                            - HEIGHT_WEIGHT    * (new_top / H)

    Args:
        x, y, z:    Candidate position.
        ol, ow, oh: Oriented box dimensions.
        bin_state:  Current bin state (read-only).
        bin_cfg:    Bin configuration.
        stats:      Precomputed bin statistics (_BinStats).

    Returns:
        Scalar score; higher is better.
    """
    res = bin_cfg.resolution

    # Footprint grid slice
    gx = int(round(x / res))
    gy = int(round(y / res))
    gx_end = min(gx + max(1, int(round(ol / res))), bin_cfg.grid_l)
    gy_end = min(gy + max(1, int(round(ow / res))), bin_cfg.grid_w)

    # Read footprint heights (no copy of full array)
    footprint = bin_state.heightmap[gx:gx_end, gy:gy_end]
    new_top = z + oh

    # New heights in footprint after placement
    new_footprint = np.maximum(footprint, new_top)

    # Incremental remaining-space update
    delta_remaining = float(
        np.sum(
            np.maximum(0.0, stats.bin_height - new_footprint)
            - np.maximum(0.0, stats.bin_height - footprint)
        )
    )
    remaining_space = stats.base_remaining + delta_remaining

    # Incremental variance update using online sum / sum-of-squares
    delta_sum = float(np.sum(new_footprint - footprint))
    delta_sum_sq = float(np.sum(new_footprint * new_footprint - footprint * footprint))

    new_sum = stats.base_sum + delta_sum
    new_sum_sq = stats.base_sum_sq + delta_sum_sq
    new_mean = new_sum / stats.n_cells
    # population variance = E[x²] - E[x]²
    new_var = max(0.0, new_sum_sq / stats.n_cells - new_mean * new_mean)

    # Height penalty (normalised)
    height_penalty = new_top / stats.bin_height if stats.bin_height > 0.0 else 0.0

    return (
        remaining_space
        - MACS_ROUGHNESS_WEIGHT * new_var
        - MACS_HEIGHT_WEIGHT    * height_penalty
    )


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

@register_strategy
class PCTMACSHeuristicStrategy(BaseStrategy):
    """
    Fast MACS (Maximal Available Container Spaces) heuristic from the
    PCT (Packing Configuration Tree) repository by Zhao et al.

    MACS scores a placement by how much usable vertical space remains in the
    bin after the box is placed.  The original heuristic (TAP-Net, Liu et al.
    ACM TOG 2020; adopted in PCT, Zhao et al. ICLR 2022) computes the sum of
    the largest-rectangle-in-histogram areas across all height layers.  This
    implementation uses a practical approximation that preserves the same core
    principle: prefer placements that keep the maximum amount of contiguous
    vertical space available for future boxes.

    Version 2 — Fast incremental scoring:
        Precomputes heightmap aggregate stats once per box.  Each candidate
        only touches its footprint cells and uses online delta formulas for
        remaining_space and variance.  This eliminates the full O(N) heightmap
        copy that caused ~1300ms/box latency.  Target: < 200ms/box.

    The three scoring components are:
        1. Remaining space:   sum(bin_height - heightmap[i, j]) over all cells.
                              More remaining space = more future packing potential.
        2. Roughness penalty: variance of the heightmap after placement (incremental).
                              Flat surfaces are easier to stack on.
        3. Height penalty:    normalised top-of-box height.
                              Low placements leave more room above.

    References:
        Zhao et al. (2022). "Deliberate Planning of 3D Bin Packing on Packing
        Configuration Trees." ICLR 2022.
        GitHub: https://github.com/alexfrom0815/Online-3D-BPP-PCT

        Zhao et al. (2025). "PCT: Packing Configuration Trees for Online 3D
        Bin Packing." IJRR 2025.

        Liu et al. (2020). "TAP-Net: Transport-and-Pack using Reinforcement
        Learning." ACM TOG, 39(6). (MACS heuristic origin)

    Attributes:
        name: Strategy identifier for the registry ("pct_macs_heuristic").
    """

    name: str = "pct_macs_heuristic"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = 1.0

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Store config and derive grid scan step from bin resolution."""
        super().on_episode_start(config)
        self._scan_step = max(1.0, config.bin.resolution)

    # -- Main entry point ---------------------------------------------------

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Propose a placement for *box* using the fast incremental MACS score.

        For each feasible (candidate position, orientation) pair the strategy:
            a. Checks bin boundary constraints.
            b. Computes resting z from the heightmap.
            c. Rejects if height limit exceeded.
            d. Enforces MIN_SUPPORT = 0.30 anti-float (always).
            e. Enforces cfg.min_support_ratio if enable_stability is True.
            f. Computes the MACS score incrementally using precomputed bin
               stats and footprint-only delta updates (no full array copy).

        Returns the candidate with the highest MACS score, or None if no
        valid placement exists.

        Args:
            box:       The box to place (original, un-rotated dimensions).
            bin_state: Current 3D bin state (read-only — do NOT mutate).

        Returns:
            PlacementDecision(x, y, orientation_idx) or None.
        """
        cfg = self.config
        bin_cfg = cfg.bin
        step = self._scan_step

        # --- Resolve allowed orientations ---
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Pre-filter orientations that could never fit in the bin.
        valid_orientations: List[Tuple[int, float, float, float]] = [
            (oidx, ol, ow, oh)
            for oidx, (ol, ow, oh) in enumerate(orientations)
            if (ol <= bin_cfg.length + 1e-6
                and ow <= bin_cfg.width + 1e-6
                and oh <= bin_cfg.height + 1e-6)
        ]
        if not valid_orientations:
            return None

        # --- Precompute bin statistics (ONE pass before the candidate loop) ---
        stats = _precompute_bin_stats(bin_state)

        # --- Generate candidates ---
        candidates = self._generate_candidates(bin_state, step)

        # --- Evaluate candidates ---
        best_score: float = -np.inf
        best_candidate: Optional[Tuple[float, float, int]] = None

        for cx, cy in candidates:
            for oidx, ol, ow, oh in valid_orientations:
                # Bin boundary checks
                if cx + ol > bin_cfg.length + 1e-6:
                    continue
                if cy + ow > bin_cfg.width + 1e-6:
                    continue

                # Resting z from the heightmap
                z = bin_state.get_height_at(cx, cy, ol, ow)

                # Height limit check
                if z + oh > bin_cfg.height + 1e-6:
                    continue

                # Anti-float: skip support check at floor level
                if z > FLOOR_Z_THRESHOLD:
                    support_ratio = bin_state.get_support_ratio(
                        cx, cy, ol, ow, z
                    )
                    # Hard minimum always enforced
                    if support_ratio < MIN_SUPPORT:
                        continue
                    # Stricter stability when configured
                    if cfg.enable_stability and support_ratio < cfg.min_support_ratio:
                        continue

                # Margin check (box-to-box gap enforcement)
                if not bin_state.is_margin_clear(cx, cy, ol, ow, z, oh):
                    continue

                # Fast incremental MACS score
                score = _macs_score_fast(
                    cx, cy, z, ol, ow, oh, bin_state, bin_cfg, stats
                )

                if score > best_score:
                    best_score = score
                    best_candidate = (cx, cy, oidx)

        if best_candidate is None:
            return None

        return PlacementDecision(
            x=best_candidate[0],
            y=best_candidate[1],
            orientation_idx=best_candidate[2],
        )

    # -- Candidate generation -----------------------------------------------

    def _generate_candidates(
        self,
        bin_state: BinState,
        step: float,
    ) -> List[Tuple[float, float]]:
        """
        Generate candidate (x, y) positions from two sources:

        1. Full grid scan at the given step size.
        2. Projected XY corners of all placed boxes.

        Duplicates are removed.  Returns a deduplicated list with grid
        positions first (top-left to bottom-right) then box corners.

        Args:
            bin_state: Current bin state (read-only).
            step:      Grid scanning step size (cm).

        Returns:
            List of unique (x, y) candidate positions.
        """
        bin_cfg = bin_state.config
        seen: Set[Tuple[float, float]] = set()
        candidates: List[Tuple[float, float]] = []

        # Source 1: full grid scan
        x = 0.0
        while x <= bin_cfg.length + 1e-9:
            y = 0.0
            while y <= bin_cfg.width + 1e-9:
                pt = (x, y)
                if pt not in seen:
                    seen.add(pt)
                    candidates.append(pt)
                y += step
            x += step

        # Source 2: projected corners of placed boxes
        for p in bin_state.placed_boxes:
            for pt in (
                (p.x, p.y),
                (p.x_max, p.y),
                (p.x, p.y_max),
                (p.x_max, p.y_max),
            ):
                if (pt not in seen
                        and 0.0 <= pt[0] <= bin_cfg.length + 1e-6
                        and 0.0 <= pt[1] <= bin_cfg.width + 1e-6):
                    seen.add(pt)
                    candidates.append(pt)

        return candidates
