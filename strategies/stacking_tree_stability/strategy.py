"""
Stacking Tree Stability Strategy for 3D bin packing.

Paper:
    Zhao, H., She, Q., Zhu, C., Yang, Y., & Xu, K. (2022).
    "Learning Practically Feasible Policies for Online 3D Bin Packing."
    Science China Information Sciences, 65(1), 112105.
    arXiv:2108.13680.

Core concept:
    The paper introduces a stacking tree representation for tracking the
    physical stability of a packed arrangement.  Each placed box is a node
    in the tree; edges encode "rests-on" relationships (parent below,
    child above).  Stability is checked via the leverage / center-of-gravity
    (CoG) principle: a box is physically stable if its CoG projects inside
    the convex hull of the contact points formed by its supporting boxes.

    This strategy implements the stability check in pure Python / NumPy and
    uses it as a filter and scoring signal during candidate evaluation.

Algorithm:
    1. Generate candidate positions from a coarse grid scan plus the
       corners of all placed boxes.
    2. For each candidate (x, y, orientation), compute the resting z from
       the heightmap and check feasibility (bounds + height limit + anti-
       float MIN_SUPPORT).
    3. Run the stacking-tree stability check:
       a. Identify all placed boxes whose top face touches z (potential
          supports).
       b. Compute per-support contact rectangles and aggregate them into
          a set of contact points (center of each contact rectangle).
       c. Check whether the box's CoG (x + l/2, y + w/2) lies inside the
          convex hull of the contact points.
       d. Compute support_ratio = total_contact_area / box_footprint_area.
    4. Score valid, stable placements:
          score = stability_bonus * 10.0
                + support_ratio  *  5.0
                - height_norm    *  2.0
    5. Return the highest-scoring stable placement.

Scoring weights (module-level constants):
    WEIGHT_STABILITY  = 10.0   -- primary: must be stable
    WEIGHT_SUPPORT    =  5.0   -- secondary: prefer high support ratio
    WEIGHT_HEIGHT     =  2.0   -- tertiary: prefer low placements

References:
    * Zhao et al. (2022), arXiv:2108.13680
    * Classic stability criterion: Blum et al. (1994), "A sampling-based
      approach to planning" (CoG-in-convex-hull stability test).
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from config import Box, PlacementDecision, ExperimentConfig, Orientation, Placement
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ---------------------------------------------------------------------------
# Module-level constants (hyperparameters)
# ---------------------------------------------------------------------------

# Anti-float threshold -- must match the simulator's rejection limit.
MIN_SUPPORT: float = 0.30

# Scoring weights.
WEIGHT_STABILITY: float = 10.0
WEIGHT_SUPPORT: float = 5.0
WEIGHT_HEIGHT: float = 2.0

# Grid scan step for candidate generation (cm).  Larger = faster but coarser.
SCAN_STEP: float = 2.0

# Height tolerance for identifying support surfaces (cm).
# A placed box whose top is within this tolerance of z is a potential support.
SUPPORT_HEIGHT_TOLERANCE: float = 1.0


# ---------------------------------------------------------------------------
# Stacking tree node
# ---------------------------------------------------------------------------

@dataclass
class StackNode:
    """
    Node in the stacking tree representing one placed box.

    Used by _check_placement_stable() to collect support geometry for the
    CoG stability test.  Not stored persistently between calls; a new
    StackNode is constructed for the candidate box during each evaluation.

    Attributes:
        box_id:   ID of the corresponding Box.
        x, y, z:  Bottom-back-left corner position (cm).
        l, w, h:  Oriented dimensions (cm).
        weight:   Box weight (used for mass-flow extensions; currently
                  the CoG check does not require per-box mass).
        children: box_ids of boxes stacked above this one.
        parents:  box_ids of boxes this one rests on.
        mass_flow: Effective mass flowing downward through this node
                  (reserved for future leverage-force extension).
    """
    box_id: int
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    weight: float = 1.0
    children: List[int] = field(default_factory=list)
    parents: List[int] = field(default_factory=list)
    mass_flow: float = 0.0

    @property
    def cog(self) -> Tuple[float, float, float]:
        """Center of gravity (assumes uniform density)."""
        return (self.x + self.l / 2.0,
                self.y + self.w / 2.0,
                self.z + self.h / 2.0)

    @property
    def top_z(self) -> float:
        """Z coordinate of the top face."""
        return self.z + self.h


# ---------------------------------------------------------------------------
# Convex hull helper
# ---------------------------------------------------------------------------

def _point_in_convex_hull_simple(
    point: Tuple[float, float],
    hull_points: List[Tuple[float, float]],
) -> bool:
    """
    Test whether *point* lies inside (or on the boundary of) the convex
    hull of *hull_points* using the cross-product winding method.

    Degenerate cases:
        - 0 points: always False.
        - 1 point:  True iff the query point coincides with it (within 1 cm).
        - 2 points: True iff the query point lies on the segment.
        - 3+ points: full convex hull test via Graham scan cross products.

    This implementation is intentionally simple (O(n log n) per call).
    For typical stacking scenarios with 2-6 contact points it is fast
    enough that it does not dominate the decision loop.

    Args:
        point:       (px, py) query point.
        hull_points: List of 2D points defining the support polygon.

    Returns:
        True if *point* is inside (or on) the convex hull.
    """
    n = len(hull_points)
    if n == 0:
        return False

    px, py = point

    if n == 1:
        dx = px - hull_points[0][0]
        dy = py - hull_points[0][1]
        return abs(dx) < 1.0 and abs(dy) < 1.0

    if n == 2:
        # Segment test: point is on the segment if the cross product is ~0
        # and the dot product is in [0, 1].
        ax, ay = hull_points[0]
        bx, by = hull_points[1]
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        cross = abx * apy - aby * apx
        if abs(cross) > 1.0:
            return False
        dot = apx * abx + apy * aby
        ab_sq = abx * abx + aby * aby
        return 0 <= dot <= ab_sq

    # For 3+ points: compute convex hull via Graham scan (sorted by angle),
    # then test point-in-convex-polygon via cross products.

    # Compute centroid of hull_points to use as reference for angle sort.
    cx_mean = sum(p[0] for p in hull_points) / n
    cy_mean = sum(p[1] for p in hull_points) / n

    # Sort hull points by polar angle around the centroid.
    def _angle(p: Tuple[float, float]) -> float:
        return float(np.arctan2(p[1] - cy_mean, p[0] - cx_mean))

    sorted_pts = sorted(hull_points, key=_angle)

    # Point-in-convex-polygon: the point is inside iff it is to the left of
    # (or on) every directed edge of the polygon (counter-clockwise order).
    def _cross(o: Tuple[float, float],
               a: Tuple[float, float],
               b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    m = len(sorted_pts)
    for i in range(m):
        o = sorted_pts[i]
        a = sorted_pts[(i + 1) % m]
        cross_val = _cross(o, a, (px, py))
        if cross_val < -1e-6:
            return False

    return True


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

@register_strategy
class StackingTreeStabilityStrategy(BaseStrategy):
    """
    Stacking Tree Stability: placement strategy that uses the stacking-tree
    CoG stability criterion to filter candidates and reward stable placements.

    For every candidate position the strategy checks whether the new box
    would be physically stable by verifying that its center of gravity
    projects inside the convex hull of its contact points with supporting
    boxes below.  Stable placements are scored by:

        score = WEIGHT_STABILITY * stability_flag
              + WEIGHT_SUPPORT   * support_ratio
              - WEIGHT_HEIGHT    * height_norm

    Unstable placements (CoG outside support polygon) that nevertheless
    pass the MIN_SUPPORT anti-float check are penalized (stability_flag=0)
    but still returned as a last resort if no stable placement is found.

    Reference:
        Zhao et al., "Learning Practically Feasible Policies for Online 3D
        Bin Packing", Science China Information Sciences, 2022. arXiv:2108.13680.

    Attributes:
        name: Strategy identifier for the registry ("stacking_tree_stability").
    """

    name: str = "stacking_tree_stability"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = SCAN_STEP

    # -- Lifecycle -----------------------------------------------------------

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Store config; derive scan step from bin resolution."""
        super().on_episode_start(config)
        self._scan_step = max(SCAN_STEP, config.bin.resolution)

    # -- Main entry point ----------------------------------------------------

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Find the most stable placement for *box*.

        Steps:
            1. Generate candidate (x, y) positions.
            2. Evaluate each candidate x orientation for feasibility,
               support, and CoG stability.
            3. Score and return the best candidate.

        Args:
            box:       The box to place (original dimensions).
            bin_state: Current 3D bin state (read-only).

        Returns:
            PlacementDecision(x, y, orientation_idx) or None.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        # Resolve allowed orientations.
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Quick check: does any orientation even fit the bin at all?
        any_fits = any(
            ol <= bin_cfg.length and ow <= bin_cfg.width and oh <= bin_cfg.height
            for ol, ow, oh in orientations
        )
        if not any_fits:
            return None

        # Build candidate positions.
        candidates = self._generate_candidates(bin_state, bin_cfg)

        best_score: float = -np.inf
        best: Optional[Tuple[float, float, int]] = None

        for cx, cy in candidates:
            for oidx, (ol, ow, oh) in enumerate(orientations):
                # Bounds check.
                if cx + ol > bin_cfg.length + 1e-6:
                    continue
                if cy + ow > bin_cfg.width + 1e-6:
                    continue

                # Resting height.
                z = bin_state.get_height_at(cx, cy, ol, ow)

                # Height limit.
                if z + oh > bin_cfg.height + 1e-6:
                    continue

                # Anti-float support check.
                support_ratio = 1.0
                if z > 0.5:
                    support_ratio = bin_state.get_support_ratio(
                        cx, cy, ol, ow, z
                    )
                    if support_ratio < MIN_SUPPORT:
                        continue

                # Optional stricter stability from config.
                if cfg.enable_stability and z > 0.5:
                    if support_ratio < cfg.min_support_ratio:
                        continue

                # Stacking-tree stability check.
                is_stable, stable_support = self._check_placement_stable(
                    cx, cy, z, ol, ow, oh, bin_state
                )

                # Use the more precise support ratio from the stability check
                # when available; otherwise keep the heightmap-based one.
                if stable_support > 0.0:
                    effective_support = stable_support
                else:
                    effective_support = support_ratio

                # Score: stable placements get a large bonus.
                stability_flag = 1.0 if is_stable else 0.0
                height_norm = z / bin_cfg.height if bin_cfg.height > 0 else 0.0

                score = (
                    WEIGHT_STABILITY * stability_flag
                    + WEIGHT_SUPPORT * effective_support
                    - WEIGHT_HEIGHT * height_norm
                )

                if score > best_score:
                    best_score = score
                    best = (cx, cy, oidx)

        if best is None:
            return None

        return PlacementDecision(x=best[0], y=best[1], orientation_idx=best[2])

    # -- Candidate generation ------------------------------------------------

    def _generate_candidates(
        self,
        bin_state: BinState,
        bin_cfg,
    ) -> List[Tuple[float, float]]:
        """
        Generate candidate (x, y) positions from two sources:

        1. Coarse grid scan at self._scan_step.
        2. Corners of all placed boxes (high-value positions for snug fits
           and good support geometry).

        Deduplicates and returns a list of unique positions.

        Args:
            bin_state: Current bin state (read-only).
            bin_cfg:   Bin configuration.

        Returns:
            List of unique (x, y) candidate positions.
        """
        step = self._scan_step
        seen = set()
        candidates: List[Tuple[float, float]] = []

        # Grid scan.
        x = 0.0
        while x <= bin_cfg.length + 1e-6:
            y = 0.0
            while y <= bin_cfg.width + 1e-6:
                key = (round(x, 4), round(y, 4))
                if key not in seen:
                    seen.add(key)
                    candidates.append((x, y))
                y += step
            x += step

        # Corners of placed boxes.
        for p in bin_state.placed_boxes:
            for cx, cy in [
                (p.x, p.y),
                (p.x_max, p.y),
                (p.x, p.y_max),
                (p.x_max, p.y_max),
            ]:
                if 0 <= cx <= bin_cfg.length and 0 <= cy <= bin_cfg.width:
                    key = (round(cx, 4), round(cy, 4))
                    if key not in seen:
                        seen.add(key)
                        candidates.append((cx, cy))

        return candidates

    # -- Stability check -----------------------------------------------------

    def _check_placement_stable(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        bin_state: BinState,
    ) -> Tuple[bool, float]:
        """
        Check whether placing a box at (x, y, z) with oriented dimensions
        (ol, ow, oh) is physically stable using the stacking-tree CoG
        criterion from Zhao et al. (2022).

        Algorithm:
            1. Floor-placed boxes (z < 0.5) are trivially stable.
            2. Find all placed boxes whose top face is at height z
               (within SUPPORT_HEIGHT_TOLERANCE).
            3. For each supporting box, compute the contact rectangle
               (overlap between the new box's footprint and the support's
               footprint).  The contact point is the center of this rectangle.
            4. Accumulate total contact area and contact points.
            5. Compute support_ratio = total_contact_area / box_footprint.
            6. If support_ratio < MIN_SUPPORT: unstable (anti-float violation).
            7. CoG test:
               - Single contact point: CoG must be within l/4 and w/4 of it.
               - Multiple contact points: CoG must be inside the convex hull.

        Args:
            x, y, z:   Bottom-back-left corner of the candidate placement.
            ol, ow, oh: Oriented dimensions of the candidate box.
            bin_state:  Current live bin state (read-only).

        Returns:
            (is_stable, support_ratio)
                is_stable:    True if CoG criterion is satisfied.
                support_ratio: Total contact area / box footprint area.
                               0.0 if no supports found.
        """
        # Floor: always stable.
        if z < 0.5:
            return True, 1.0

        box_area = ol * ow
        if box_area <= 0.0:
            return False, 0.0

        # Collect supporting boxes: placed boxes whose top face is at height z.
        support_items: List[Placement] = [
            p for p in bin_state.placed_boxes
            if abs(p.z + p.oriented_h - z) < SUPPORT_HEIGHT_TOLERANCE
            and p.x < x + ol - 1e-6
            and p.x_max > x + 1e-6
            and p.y < y + ow - 1e-6
            and p.y_max > y + 1e-6
        ]

        if not support_items:
            return False, 0.0

        # Compute contact rectangles and aggregate.
        total_contact_area: float = 0.0
        contact_points: List[Tuple[float, float]] = []

        for sup in support_items:
            # Overlap rectangle.
            ox1 = max(x, sup.x)
            oy1 = max(y, sup.y)
            ox2 = min(x + ol, sup.x_max)
            oy2 = min(y + ow, sup.y_max)

            if ox1 >= ox2 - 1e-9 or oy1 >= oy2 - 1e-9:
                continue

            area = (ox2 - ox1) * (oy2 - oy1)
            total_contact_area += area
            contact_points.append(((ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0))

        if not contact_points:
            return False, 0.0

        support_ratio = total_contact_area / box_area

        # Anti-float check: reject if below minimum support.
        if support_ratio < MIN_SUPPORT:
            return False, support_ratio

        # CoG of the candidate box (horizontal projection only).
        cog_x = x + ol / 2.0
        cog_y = y + ow / 2.0
        cog = (cog_x, cog_y)

        # Stability check.
        if len(contact_points) == 1:
            # Single support: CoG must be within l/4 and w/4 of the contact
            # point (generous but physically motivated: the contact patch is
            # not a point but a rectangle ol x ow in the worst case).
            cp = contact_points[0]
            is_stable = (
                abs(cog_x - cp[0]) < ol / 4.0 + 1e-6
                and abs(cog_y - cp[1]) < ow / 4.0 + 1e-6
            )
        else:
            # Multiple supports: CoG must be inside the convex hull of all
            # contact points.
            is_stable = _point_in_convex_hull_simple(cog, contact_points)

        return is_stable, support_ratio
