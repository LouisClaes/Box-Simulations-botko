"""
LBCP Stability strategy for 3D bin packing.

Based on:
    Gao, Wang, Kong, Chong (2025).
    "Online 3D Bin Packing with Fast Stability Validation and Stable
    Rearrangement Planning."  arXiv:2507.09123.  JAIST.

Core concept — Load-Bearable Convex Polygon (LBCP):
    The LBCP of a placed item is the region on its top face from which it
    can structurally support downward load.  Unlike a simple 2D overlap
    check, LBCP accounts for whether the support surface is itself stable
    and can transmit force all the way to the bin floor.

Key theorems:
    1. Item resting on the bin floor: entire top face is its LBCP.
    2. Item i supported by items j1, j2, ...: its LBCP is the convex hull
       of (top_face_i ∩ LBCP_j) for all j where z_max_j == z_i.
    3. Item i is stable iff its 2D centre-of-gravity (CoG) lies inside its
       support polygon (convex hull of the contact region below it).
    4. Placing a newly-stable item cannot destabilise any item already below
       it — so incremental re-checking is not required.

SSV (Stability Stability Validation) per candidate (x, y, orientation):
    a. z   = get_height_at(x, y, ol, ow)       — resting height
    b. If z < 0.5  →  floor placement, always valid, support_ratio = 1.0
    c. Find support items: placed boxes whose z_max ≈ z (within CONTACT_TOL)
       AND whose footprint overlaps the candidate's footprint.
    d. For each support item compute its LBCP rectangle (see _get_lbcp()).
    e. Collect all heightmap grid cells within the candidate footprint where
       the height equals z (within CONTACT_TOL) AND the cell centre falls
       inside at least one supporting item's LBCP.
    f. Build support polygon = convex hull of those cell centres.
    g. If no valid contact cells → unstable.
    h. CoG of the candidate box (uniform density) = (x + ol/2, y + ow/2).
    i. Stable iff CoG is inside support polygon.
    j. support_ratio = (# valid contact cells) / (# footprint cells).

LBCP approximation:
    - Floor items (z ≈ 0): LBCP = full top face (rectangle).
    - Stacked items:        LBCP = top face shrunk inward by LBCP_SHRINK_STACKED
                            on each edge (conservative: 15% per side).

Composite placement score:
    score  = 10.0 × support_ratio     (primary:    reward stable support)
           +  5.0 × contact_ratio     (secondary:  maximise bottom contact)
           -  2.0 × height_norm       (tertiary:   prefer low placement)
           -  1.0 × roughness_delta   (quaternary: keep surface smooth)

Candidate generation:
    Full grid scan at resolution step size PLUS corners of placed boxes,
    identical to the surface_contact strategy.

Hyperparameters:
    MIN_SUPPORT            = 0.30   — anti-float threshold (always enforced)
    LBCP_SHRINK_FLOOR      = 0.0    — no shrink for floor items
    LBCP_SHRINK_STACKED    = 0.15   — 15% per-edge shrink for stacked items
    CONTACT_TOL            = 0.5    — cm tolerance for height matching
    WEIGHT_STABILITY       = 10.0
    WEIGHT_CONTACT         = 5.0
    WEIGHT_HEIGHT          = 2.0
    WEIGHT_ROUGHNESS_DELTA = 1.0
"""

import numpy as np
from typing import Optional, List, Tuple, Set

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ---------------------------------------------------------------------------
# Module-level constants (hyperparameters)
# ---------------------------------------------------------------------------

MIN_SUPPORT: float = 0.30          # Anti-float threshold — always enforced.

# LBCP shrink factors (fraction of the dimension shrunk from each edge).
LBCP_SHRINK_FLOOR: float = 0.0     # Floor items: full top face is valid.
LBCP_SHRINK_STACKED: float = 0.15  # Stacked items: 15% per edge shrink.

# Height tolerance for matching "resting level" (in cm).
CONTACT_TOL: float = 0.5

# Scoring weights.
WEIGHT_STABILITY: float = 10.0
WEIGHT_CONTACT: float = 5.0
WEIGHT_HEIGHT: float = 2.0
WEIGHT_ROUGHNESS_DELTA: float = 1.0

# Try to import scipy ConvexHull; fall back to an internal Graham scan.
try:
    from scipy.spatial import ConvexHull as _ScipyConvexHull
    _SCIPY_AVAILABLE = True
except ImportError:
    _ScipyConvexHull = None
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

@register_strategy
class LBCPStabilityStrategy(BaseStrategy):
    """
    LBCP Stability Placement Strategy (Gao et al. 2025).

    Evaluates every candidate (x, y, orientation) using the Load-Bearable
    Convex Polygon (LBCP) stability criterion:

      1. Compute the support polygon below the candidate as the convex hull
         of all grid-cell centres that (a) lie in the candidate footprint,
         (b) are at the correct resting height, and (c) fall inside the LBCP
         of the item below them.
      2. Accept the placement only if the candidate's 2D centre-of-gravity
         lies inside the support polygon (or if z ≈ 0, i.e. floor placement).
      3. Score each accepted candidate with a weighted combination of
         stability, contact ratio, height penalty, and surface roughness.
      4. Return the highest-scoring candidate.

    The strategy always allows floor placements (z < 0.5).  The standard
    MIN_SUPPORT = 0.30 anti-float threshold is always applied in addition to
    the LBCP check.

    Attributes:
        name: Strategy identifier for the registry ("lbcp_stability").
    """

    name: str = "lbcp_stability"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = 1.0

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Store config and derive the grid scan step from bin resolution."""
        super().on_episode_start(config)
        # Use 2× resolution for grid scan; box corners supplement coverage.
        self._scan_step = max(1.0, config.bin.resolution * 2.0)

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Find the best LBCP-stable placement for *box*.

        Steps:
            1. Build candidate (x, y) positions: grid scan + box corners.
            2. For each candidate and each valid orientation:
               a. Compute resting z.
               b. Reject out-of-bounds and over-height candidates.
               c. Run LBCP stability check.
               d. Skip if not stable (unless floor placement) or support < MIN_SUPPORT.
               e. Compute contact ratio and roughness delta.
               f. Compute composite score.
            3. Return the highest-scoring candidate, or None.

        Args:
            box:       The box to place (original, un-rotated dimensions).
            bin_state: Current 3D bin state.  Read-only — do NOT modify.

        Returns:
            PlacementDecision(x, y, orientation_idx) or None if no valid
            placement exists.
        """
        cfg = self.config
        bin_cfg = cfg.bin
        step = self._scan_step

        # Resolve allowed orientations.
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Quick filter: drop orientations that can't fit inside the bin at all.
        valid_orientations = [
            (oidx, ol, ow, oh)
            for oidx, (ol, ow, oh) in enumerate(orientations)
            if ol <= bin_cfg.length and ow <= bin_cfg.width and oh <= bin_cfg.height
        ]
        if not valid_orientations:
            return None

        heightmap = bin_state.heightmap
        resolution = bin_cfg.resolution

        # Build candidate positions.
        candidates = self._generate_candidates(bin_state, step)

        # Pre-compute surface roughness for the delta calculation.
        current_roughness = bin_state.get_surface_roughness()

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

                # --- LBCP stability check ---
                # Floor placements are always valid; the check is skipped.
                if z < 0.5:
                    is_stable = True
                    support_ratio = 1.0
                else:
                    is_stable, support_ratio = self._validate_lbcp(
                        cx, cy, z, ol, ow, oh, bin_state
                    )

                # --- Anti-float: always enforce MIN_SUPPORT ---
                if support_ratio < MIN_SUPPORT:
                    continue

                # --- Stability gate (respects cfg.enable_stability too) ---
                if not is_stable:
                    continue
                if cfg.enable_stability and z > 0.5:
                    if support_ratio < cfg.min_support_ratio:
                        continue

                # --- Contact ratio (bottom face) ---
                contact_ratio = self._compute_contact_ratio(
                    cx, cy, z, ol, ow, heightmap, bin_cfg
                )

                # --- Roughness delta (skip for floor placements: cost > benefit) ---
                if z < resolution * 0.5:
                    roughness_delta = 0.0
                else:
                    roughness_delta = self._compute_roughness_delta(
                        cx, cy, z, ol, ow, oh,
                        heightmap, bin_cfg, resolution, current_roughness,
                    )

                # --- Composite score ---
                height_norm = z / bin_cfg.height if bin_cfg.height > 0 else 0.0
                score = (
                    WEIGHT_STABILITY * support_ratio
                    + WEIGHT_CONTACT * contact_ratio
                    - WEIGHT_HEIGHT * height_norm
                    - WEIGHT_ROUGHNESS_DELTA * roughness_delta
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

    # -------------------------------------------------------------------------
    # LBCP validation core
    # -------------------------------------------------------------------------

    def _validate_lbcp(
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
        Run the SSV (Structural Stability Validation) check for a candidate.

        The algorithm:
            1. Find all support items: placed boxes whose z_max ≈ z and whose
               footprint overlaps the candidate footprint.
            2. For each support item, compute its LBCP (rectangle, possibly
               shrunk).
            3. Walk every grid cell in the candidate footprint.  A cell is a
               "valid contact cell" if the heightmap value there matches z
               (within CONTACT_TOL) AND the cell centre lies inside at least
               one support item's LBCP.
            4. Build the support polygon = convex hull of valid contact cell
               centres.
            5. CoG of the candidate (uniform density) = (x + ol/2, y + ow/2).
            6. Stable iff CoG lies inside the support polygon.

        Args:
            x, y:      Bottom-left corner of the candidate footprint.
            z:         Resting height (z > 0 guaranteed by caller).
            ol, ow:    Oriented footprint dimensions.
            oh:        Oriented height (unused — passed for completeness).
            bin_state: Current bin state (read-only).

        Returns:
            (is_stable, support_ratio) where support_ratio is the fraction of
            footprint cells that are valid contact cells.
        """
        bin_cfg = bin_state.config
        resolution = bin_cfg.resolution
        heightmap = bin_state.heightmap

        # --- 1. Find support items ---
        support_items = self._get_support_items(x, y, z, ol, ow, bin_state)

        if not support_items:
            # Nothing below us at the right height — cannot be stable.
            return False, 0.0

        # --- 2. Compute LBCP for each support item ---
        # Each LBCP is stored as (lx_min, ly_min, lx_max, ly_max).
        lbcps = [self._get_lbcp(p) for p in support_items]

        # --- 3. Find valid contact cells ---
        contact_points = self._compute_contact_points(
            x, y, z, ol, ow, heightmap, bin_cfg, lbcps
        )

        gx = int(round(x / resolution))
        gy = int(round(y / resolution))
        gx_end = min(gx + int(round(ol / resolution)), bin_cfg.grid_l)
        gy_end = min(gy + int(round(ow / resolution)), bin_cfg.grid_w)
        total_cells = (gx_end - gx) * (gy_end - gy)

        if total_cells == 0:
            return False, 0.0

        support_ratio = len(contact_points) / total_cells

        if len(contact_points) == 0:
            return False, 0.0

        # --- 4. Build support polygon ---
        hull_pts = self._convex_hull_2d(contact_points)

        # --- 5. CoG of candidate ---
        cog = (x + ol / 2.0, y + ow / 2.0)

        # --- 6. CoG-in-polygon test ---
        is_stable = self._point_in_polygon(cog, hull_pts)

        return is_stable, support_ratio

    def _get_support_items(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        bin_state: BinState,
    ) -> list:
        """
        Return placed boxes that can potentially support the candidate.

        A box p is a support item iff:
          - p.z_max is within CONTACT_TOL of z (its top face is at the
            resting level of the candidate), AND
          - p's footprint overlaps the candidate's footprint (non-zero
            overlap in both x and y).

        Args:
            x, y:      Bottom-left corner of the candidate.
            z:         Resting height of the candidate.
            ol, ow:    Candidate footprint dimensions.
            bin_state: Current bin state.

        Returns:
            List of Placement objects that are candidate supports.
        """
        result = []
        x_max = x + ol
        y_max = y + ow

        for p in bin_state.placed_boxes:
            # Height check: top of support item must be at resting level.
            if abs(p.z_max - z) > CONTACT_TOL:
                continue
            # Footprint overlap check (AABB intersection).
            if p.x_max <= x or p.x >= x_max:
                continue
            if p.y_max <= y or p.y >= y_max:
                continue
            result.append(p)

        return result

    def _get_lbcp(self, placed_box) -> Tuple[float, float, float, float]:
        """
        Compute the LBCP rectangle for a placed box.

        For floor items (z ≈ 0): LBCP = full top face (no shrink).
        For stacked items:        LBCP = top face shrunk inward by
                                  LBCP_SHRINK_STACKED × dimension / 2
                                  from each edge (conservative).

        The physical rationale: the effective load-bearing region of a
        stacked item is smaller than its full footprint because load must
        be transmitted through a possibly smaller support region below it.
        The 15% per-side shrink is a conservative approximation.

        Args:
            placed_box: A Placement object from bin_state.placed_boxes.

        Returns:
            (lx_min, ly_min, lx_max, ly_max) — the LBCP bounding rectangle.
        """
        if placed_box.z < 0.5:
            shrink = LBCP_SHRINK_FLOOR
        else:
            shrink = LBCP_SHRINK_STACKED

        sl = placed_box.oriented_l * shrink / 2.0
        sw = placed_box.oriented_w * shrink / 2.0
        return (
            placed_box.x + sl,
            placed_box.y + sw,
            placed_box.x_max - sl,
            placed_box.y_max - sw,
        )

    def _compute_contact_points(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        heightmap: np.ndarray,
        bin_cfg,
        lbcps: List[Tuple[float, float, float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Find all grid-cell centres in the candidate footprint that are
        both at the correct height AND inside at least one LBCP.

        Each cell is a valid contact cell iff:
          - heightmap[gx, gy] is within CONTACT_TOL of z, AND
          - the cell centre (cx, cy) satisfies cx ∈ [lx_min, lx_max] AND
            cy ∈ [ly_min, ly_max] for at least one LBCP rectangle.

        Using rectangular LBCP approximation avoids the cost of a full
        convex-polygon inclusion test per cell, keeping the per-candidate
        cost O(footprint_cells).

        Args:
            x, y:     Bottom-left of candidate footprint.
            z:        Resting height.
            ol, ow:   Footprint dimensions.
            heightmap: Current heightmap (read-only).
            bin_cfg:  Bin configuration (resolution, grid_l, grid_w).
            lbcps:    List of (lx_min, ly_min, lx_max, ly_max) rectangles.

        Returns:
            List of (cx, cy) world-coordinate centres of valid contact cells.
        """
        resolution = bin_cfg.resolution
        gx = int(round(x / resolution))
        gy = int(round(y / resolution))
        gx_end = min(gx + int(round(ol / resolution)), bin_cfg.grid_l)
        gy_end = min(gy + int(round(ow / resolution)), bin_cfg.grid_w)

        tol = CONTACT_TOL
        contact_pts: List[Tuple[float, float]] = []

        for ix in range(gx, gx_end):
            for iy in range(gy, gy_end):
                # Height check.
                if abs(heightmap[ix, iy] - z) > tol:
                    continue
                # Cell centre in world coords.
                cx = (ix + 0.5) * resolution
                cy = (iy + 0.5) * resolution
                # LBCP membership check (at least one LBCP must contain (cx, cy)).
                for lx_min, ly_min, lx_max, ly_max in lbcps:
                    if lx_min <= cx <= lx_max and ly_min <= cy <= ly_max:
                        contact_pts.append((cx, cy))
                        break  # No need to check other LBCPs for this cell.

        return contact_pts

    # -------------------------------------------------------------------------
    # Contact ratio (bottom face only — fast)
    # -------------------------------------------------------------------------

    def _compute_contact_ratio(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        heightmap: np.ndarray,
        bin_cfg,
    ) -> float:
        """
        Fraction of the candidate's bottom face that rests on existing surfaces.

        For floor placements (z < 0.5): returns 1.0 (perfect floor contact).
        Otherwise: counts heightmap cells within the footprint that match z
        within CONTACT_TOL, divided by total footprint cells.

        Args:
            x, y:      Bottom-left corner of candidate.
            z:         Resting height.
            ol, ow:    Footprint dimensions.
            heightmap: Current heightmap (read-only).
            bin_cfg:   Bin configuration.

        Returns:
            Float in [0.0, 1.0].
        """
        if z < 0.5:
            return 1.0

        res = bin_cfg.resolution
        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), bin_cfg.grid_l)
        gy_end = min(gy + int(round(ow / res)), bin_cfg.grid_w)
        total_cells = (gx_end - gx) * (gy_end - gy)

        if total_cells == 0:
            return 0.0

        footprint = heightmap[gx:gx_end, gy:gy_end]
        matched = int(np.sum(np.abs(footprint - z) <= CONTACT_TOL))
        return matched / total_cells

    # -------------------------------------------------------------------------
    # Roughness delta (same approach as surface_contact)
    # -------------------------------------------------------------------------

    def _compute_roughness_delta(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        heightmap: np.ndarray,
        bin_cfg,
        resolution: float,
        current_roughness: float,
    ) -> float:
        """
        Estimate the change in local surface roughness from placing the box.

        Paints the box into a local copy of the heightmap and computes the
        change in mean absolute height-gradient in the affected region (+margin).
        Positive delta = rougher (penalised). Negative delta = smoother.

        Normalised by (current_roughness + 1) to keep the value bounded.

        Args:
            x, y, z:           Position of box bottom-left corner.
            ol, ow, oh:        Oriented dimensions.
            heightmap:         Current heightmap (read-only).
            bin_cfg:           Bin configuration.
            resolution:        Grid cell size.
            current_roughness: Current surface roughness scalar.

        Returns:
            Normalised roughness change, roughly in [-1, 1].
        """
        gx = int(round(x / resolution))
        gy = int(round(y / resolution))
        gx_end = min(gx + int(round(ol / resolution)), bin_cfg.grid_l)
        gy_end = min(gy + int(round(ow / resolution)), bin_cfg.grid_w)

        margin = 2
        rx_start = max(0, gx - margin)
        ry_start = max(0, gy - margin)
        rx_end = min(bin_cfg.grid_l, gx_end + margin)
        ry_end = min(bin_cfg.grid_w, gy_end + margin)

        region = heightmap[rx_start:rx_end, ry_start:ry_end].copy()
        if region.size < 2:
            return 0.0

        dx_before = np.abs(np.diff(region, axis=0))
        dy_before = np.abs(np.diff(region, axis=1))
        roughness_before = (
            (float(np.mean(dx_before)) if dx_before.size > 0 else 0.0)
            + (float(np.mean(dy_before)) if dy_before.size > 0 else 0.0)
        ) / 2.0

        # Paint box into local region copy.
        box_top = z + oh
        local_gx = gx - rx_start
        local_gy = gy - ry_start
        local_gx_end = gx_end - rx_start
        local_gy_end = gy_end - ry_start
        region[local_gx:local_gx_end, local_gy:local_gy_end] = np.maximum(
            region[local_gx:local_gx_end, local_gy:local_gy_end],
            box_top,
        )

        dx_after = np.abs(np.diff(region, axis=0))
        dy_after = np.abs(np.diff(region, axis=1))
        roughness_after = (
            (float(np.mean(dx_after)) if dx_after.size > 0 else 0.0)
            + (float(np.mean(dy_after)) if dy_after.size > 0 else 0.0)
        ) / 2.0

        delta = roughness_after - roughness_before
        normaliser = current_roughness + 1.0
        return delta / normaliser

    # -------------------------------------------------------------------------
    # Convex hull and point-in-polygon geometry
    # -------------------------------------------------------------------------

    def _convex_hull_2d(
        self,
        points: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Return the convex hull of a set of 2D points as an ordered polygon.

        Uses scipy.spatial.ConvexHull when available; otherwise falls back
        to an internal Graham scan implementation.

        Degenerate cases:
            - 0 points → empty list.
            - 1 point  → list with that one point.
            - 2 points → list with both points.
            - ≥ 3 collinear → list with the two extreme points only.

        Args:
            points: List of (x, y) tuples.

        Returns:
            List of (x, y) tuples forming the convex hull in counter-clockwise
            order.  May contain as few as 1 point.
        """
        if len(points) == 0:
            return []
        if len(points) == 1:
            return list(points)
        if len(points) == 2:
            return list(points)

        pts_arr = np.array(points, dtype=float)

        if _SCIPY_AVAILABLE:
            try:
                hull = _ScipyConvexHull(pts_arr)
                indices = hull.vertices
                return [points[i] for i in indices]
            except Exception:
                pass  # Fall through to Graham scan.

        return self._graham_scan(points)

    @staticmethod
    def _graham_scan(
        points: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Graham scan convex hull algorithm (O(n log n)).

        Returns the hull in counter-clockwise order.  Collinear points on
        the boundary are excluded.

        Args:
            points: List of (x, y) tuples (≥ 2 distinct points assumed).

        Returns:
            Convex hull vertices in CCW order.
        """
        # Find the pivot: lowest y, then leftmost x.
        pivot = min(points, key=lambda p: (p[1], p[0]))

        def polar_angle(p: Tuple[float, float]) -> float:
            return float(np.arctan2(p[1] - pivot[1], p[0] - pivot[0]))

        def cross(o, a, b) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        sorted_pts = sorted(
            set(points),
            key=lambda p: (polar_angle(p), (p[0] - pivot[0]) ** 2 + (p[1] - pivot[1]) ** 2),
        )

        if len(sorted_pts) < 2:
            return sorted_pts

        hull: List[Tuple[float, float]] = []
        for p in sorted_pts:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)

        return hull

    def _point_in_polygon(
        self,
        point: Tuple[float, float],
        polygon: List[Tuple[float, float]],
    ) -> bool:
        """
        Test whether *point* lies inside (or on the boundary of) *polygon*.

        Handles all degenerate cases gracefully:
            - Empty polygon  → False.
            - Single vertex  → True iff point equals vertex.
            - Two vertices   → True iff point lies on the segment.
            - General convex → cross-product winding test.

        For the general convex case the polygon must be in CCW order (as
        returned by _convex_hull_2d).  The test uses the sign of the
        cross-product for each edge: if the point is consistently on the
        left side (or on the edge) for every edge, it is inside.

        Args:
            point:   (px, py) to test.
            polygon: Ordered list of (x, y) hull vertices.

        Returns:
            True if the point is inside or on the boundary.
        """
        n = len(polygon)
        if n == 0:
            return False

        px, py = point

        if n == 1:
            return abs(px - polygon[0][0]) < 1e-9 and abs(py - polygon[0][1]) < 1e-9

        if n == 2:
            return self._point_on_segment(point, polygon[0], polygon[1])

        # General convex polygon: cross-product sign test.
        # The hull is in CCW order; for each edge (A → B), the point must be
        # on the left side (cross ≥ 0).
        for i in range(n):
            ax, ay = polygon[i]
            bx, by = polygon[(i + 1) % n]
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            if cross < -1e-9:
                return False

        return True

    @staticmethod
    def _point_on_segment(
        point: Tuple[float, float],
        seg_a: Tuple[float, float],
        seg_b: Tuple[float, float],
    ) -> bool:
        """
        Return True iff *point* lies on segment seg_a–seg_b.

        Checks both collinearity (via cross-product) and that the point
        is within the bounding box of the segment.

        Args:
            point: (px, py).
            seg_a: First endpoint.
            seg_b: Second endpoint.

        Returns:
            bool.
        """
        px, py = point
        ax, ay = seg_a
        bx, by = seg_b

        # Cross-product must be zero (collinear).
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if abs(cross) > 1e-9:
            return False

        # Dot product check: point is between A and B.
        if min(ax, bx) - 1e-9 <= px <= max(ax, bx) + 1e-9:
            if min(ay, by) - 1e-9 <= py <= max(ay, by) + 1e-9:
                return True

        return False

    # -------------------------------------------------------------------------
    # Candidate generation (identical approach to surface_contact)
    # -------------------------------------------------------------------------

    def _generate_candidates(
        self,
        bin_state: BinState,
        step: float,
    ) -> List[Tuple[float, float]]:
        """
        Generate candidate (x, y) positions.

        Two sources:
          1. Full grid scan at *step* size (covers uniform spacing).
          2. Corners of all placed boxes (natural high-contact positions).

        Duplicates are removed.  The list is sorted by estimated resting
        height (lowest first), so low-position candidates are evaluated
        first — favouring stable base-layer placements.

        Args:
            bin_state: Current bin state.
            step:      Grid scan step size (cm).

        Returns:
            List of unique (x, y) candidates sorted by (z_est, x, y).
        """
        bin_cfg = bin_state.config
        seen: Set[Tuple[float, float]] = set()
        candidates: List[Tuple[float, float]] = []

        # Source 1: Grid scan.
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

        # Source 2: Placed-box corners.
        for p in bin_state.placed_boxes:
            for pt in [
                (p.x, p.y),
                (p.x_max, p.y),
                (p.x, p.y_max),
                (p.x_max, p.y_max),
            ]:
                if (pt not in seen
                        and 0 <= pt[0] <= bin_cfg.length
                        and 0 <= pt[1] <= bin_cfg.width):
                    seen.add(pt)
                    candidates.append(pt)

        # Sort by position only (x then y) to keep a deterministic order.
        candidates.sort()
        return candidates
