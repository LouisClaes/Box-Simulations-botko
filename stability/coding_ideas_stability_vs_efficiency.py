"""
=============================================================================
CODING IDEAS: Static Stability vs Packing Efficiency for Online 3D Bin Packing
=============================================================================

Based on: Ali, Ramos, Oliveira (2025) - "Static stability versus packing
efficiency in online three-dimensional packing problems"
Computers & Operations Research, Vol. 178, Article 107005

PURPOSE: Implementation roadmap for stability-aware online 3D packing with
buffer support, targeting our semi-online 2-bounded-space system.

ESTIMATED TOTAL IMPLEMENTATION TIME: 3-5 weeks for core modules
COMPLEXITY: Medium-High (convex hull + point-in-polygon + EMS management)
PYTHON VERSION: 3.10+
KEY DEPENDENCIES: numpy, scipy.spatial (for ConvexHull validation),
                  matplotlib (for visualization)

=============================================================================
UPDATED: 2026-02-18 -- Deep research pass with full implementation code
         extracted from exhaustive 20-page paper analysis.
=============================================================================
"""

import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Numerical tolerance for floating-point comparisons
EPSILON = 1e-9


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class Position:
    """3D position (minimum vertex / deepest-bottom-left corner)."""
    x: float  # depth (from entrance to back; x=0 is back wall)
    y: float  # width (left-right; y=0 is left wall)
    z: float  # height (bottom-top; z=0 is floor)


@dataclass
class ItemDims:
    """Item dimensions in a specific orientation."""
    length: float  # x-axis extent
    width: float   # y-axis extent
    height: float  # z-axis extent

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def base_area(self) -> float:
        return self.length * self.width


@dataclass
class PlacedItem:
    """An item that has been placed in a bin."""
    item_id: int
    position: Position
    dims: ItemDims  # dims in the placed orientation
    weight: float = 1.0  # uniform density assumed; weight proportional to volume

    @property
    def top_z(self) -> float:
        return self.position.z + self.dims.height

    @property
    def x_max(self) -> float:
        return self.position.x + self.dims.length

    @property
    def y_max(self) -> float:
        return self.position.y + self.dims.width

    @property
    def cg_xy(self) -> Tuple[float, float]:
        """Center of gravity projected onto XY plane (uniform density)."""
        return (self.position.x + self.dims.length / 2.0,
                self.position.y + self.dims.width / 2.0)

    @property
    def cg_xyz(self) -> Tuple[float, float, float]:
        """Full 3D center of gravity (uniform density)."""
        return (self.position.x + self.dims.length / 2.0,
                self.position.y + self.dims.width / 2.0,
                self.position.z + self.dims.height / 2.0)


# ============================================================================
# MODULE 1: STABILITY CONSTRAINT CHECKERS -- FULL IMPLEMENTATIONS
# ============================================================================
# DESIGN PATTERN: Strategy pattern -- each constraint is a callable that
# takes (item dims, placement position, orientation, list of placed items)
# and returns a stability verdict with metadata.
#
# Paper reference: Section 3 (Static stability approaches), pages 3-6.
#
# KEY NUMBERS FROM PAPER (Table 8):
# +---------------------------------+---------+----------+-------------+
# | Constraint                      | Bins    | Stable   | Stability % |
# +---------------------------------+---------+----------+-------------+
# | Full-base support               | 173,662 | 2974240  | 100.00%     |
# | Partial-base support (80%)      | 139,630 | 2974050  |  99.99%     |
# | CoG polygon support             |  88,601 | 2618558  |  88.04%     |
# | Partial-base polygon (50%)      |  95,921 | 2741924  |  92.19%     |
# | No stability constraint         |  64,918 | 1489489  |  50.08%     |
# +---------------------------------+---------+----------+-------------+

@dataclass
class StabilityResult:
    """Result of a stability check with detailed metadata."""
    is_stable: bool
    constraint_name: str
    # Metadata for scoring and debugging
    supported_area: float = 0.0
    base_area: float = 0.0
    support_ratio: float = 0.0  # supported_area / base_area
    cg_status: str = 'unknown'  # 'inside', 'boundary', 'outside', 'floor', 'direct'
    sp_area: float = 0.0  # support polygon area
    sp_vertices: List[Tuple[float, float]] = field(default_factory=list)
    supporting_item_count: int = 0
    condition_used: int = 0  # 1=floor, 2=direct, 3=polygon


class StabilityChecker(ABC):
    """Abstract interface for stability checking."""

    @abstractmethod
    def check(self, item_dims: ItemDims, position: Position,
              placed_items: List[PlacedItem]) -> StabilityResult:
        """
        Check whether an item placed at the given position is stable.

        Args:
            item_dims: dimensions of the item in its chosen orientation
            position: (x, y, z) of the item's deepest-bottom-left corner
            placed_items: list of all items already placed in the bin

        Returns:
            StabilityResult with is_stable and metadata
        """
        raise NotImplementedError

    def is_stable(self, item_dims: ItemDims, position: Position,
                  orientation: int, placed_items: List[PlacedItem]) -> bool:
        """Convenience wrapper returning just the boolean."""
        return self.check(item_dims, position, placed_items).is_stable


# ---- Helper: compute supported area ----

def compute_supported_area(item_x: float, item_y: float,
                           item_l: float, item_w: float,
                           item_z: float,
                           placed_items: List[PlacedItem]) -> Tuple[float, List[PlacedItem]]:
    """
    Compute the total area of the item's base that is supported by
    underlying items whose top surface is at the item's bottom height.

    Returns:
        (supported_area, list_of_supporting_items)

    Note: Naive summation of intersection areas is used. This is correct
    when supporting items do not overlap at their top surfaces, which is
    guaranteed by the non-overlap constraint in 3D bin packing.
    """
    supported_area = 0.0
    supporting = []

    for p in placed_items:
        # Check if this item's top surface is at the new item's bottom
        if abs(p.top_z - item_z) > EPSILON:
            continue

        # Compute intersection of two axis-aligned rectangles
        ix_min = max(item_x, p.position.x)
        ix_max = min(item_x + item_l, p.x_max)
        iy_min = max(item_y, p.position.y)
        iy_max = min(item_y + item_w, p.y_max)

        if ix_max > ix_min + EPSILON and iy_max > iy_min + EPSILON:
            area = (ix_max - ix_min) * (iy_max - iy_min)
            supported_area += area
            supporting.append(p)

    return supported_area, supporting


class FullBaseSupport(StabilityChecker):
    """
    CONSTRAINT 1: Full-base support (most restrictive).

    The entire (100%) base area of an item must be supported by either
    the container floor or the top surface(s) of packed items.

    Paper results (Table 8):
        Bins used: 173,662 | Stability: 100.00% | Efficiency: 51%

    Paper results by size class (Tables J.1-J.3):
        Small: 1.01 bins, 100% stable
        Medium: 2.47 bins, 100% stable
        Large: 8.05 bins, 100% stable

    Processing time per item (Table 6):
        Small: 0.003s | Medium: 0.106s | Large: 1.743s

    Complexity: O(n) per check where n = placed items in bin.
    """

    def check(self, item_dims: ItemDims, position: Position,
              placed_items: List[PlacedItem]) -> StabilityResult:

        base_area = item_dims.base_area

        # Condition 1: Floor support
        if position.z < EPSILON:
            return StabilityResult(
                is_stable=True, constraint_name='full_base',
                supported_area=base_area, base_area=base_area,
                support_ratio=1.0, cg_status='floor',
                condition_used=1
            )

        # Compute supported area
        supported, supporting = compute_supported_area(
            position.x, position.y, item_dims.length, item_dims.width,
            position.z, placed_items
        )

        ratio = supported / base_area if base_area > EPSILON else 0.0
        is_stable = abs(supported - base_area) < EPSILON * max(1.0, base_area)

        return StabilityResult(
            is_stable=is_stable, constraint_name='full_base',
            supported_area=supported, base_area=base_area,
            support_ratio=ratio, cg_status='full_support' if is_stable else 'insufficient',
            supporting_item_count=len(supporting),
            condition_used=1 if is_stable else 0
        )


class PartialBaseSupport(StabilityChecker):
    """
    CONSTRAINT 2: Partial-base support (threshold = 80% by default).

    At least 80% of the item's base area must be supported.
    Threshold follows Christensen & Rousse (2009), Fuellerer et al. (2010),
    Koch et al. (2018).

    Paper results (Table 8):
        Bins used: 139,630 | Stability: 99.99% | Efficiency: 63%

    Paper results by size class:
        Small: 1.00 bins, 100% stable
        Medium: 2.10 bins, 100% stable
        Large: 6.38 bins, 100% stable

    Processing time per item:
        Small: 0.003s | Medium: 0.246s | Large: 3.048s

    WARNING: Does NOT check center of gravity position. An item with
    80% support on one side but CG over the unsupported side will pass
    this check but may be physically unstable.

    Complexity: O(n) per check.
    """

    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold

    def check(self, item_dims: ItemDims, position: Position,
              placed_items: List[PlacedItem]) -> StabilityResult:

        base_area = item_dims.base_area

        # Condition 1: Floor support
        if position.z < EPSILON:
            return StabilityResult(
                is_stable=True, constraint_name='partial_base',
                supported_area=base_area, base_area=base_area,
                support_ratio=1.0, cg_status='floor',
                condition_used=1
            )

        supported, supporting = compute_supported_area(
            position.x, position.y, item_dims.length, item_dims.width,
            position.z, placed_items
        )

        ratio = supported / base_area if base_area > EPSILON else 0.0
        is_stable = ratio >= self.threshold - EPSILON

        return StabilityResult(
            is_stable=is_stable, constraint_name='partial_base',
            supported_area=supported, base_area=base_area,
            support_ratio=ratio,
            cg_status='partial_support' if is_stable else 'insufficient',
            supporting_item_count=len(supporting),
            condition_used=2 if is_stable else 0
        )


class CoGPolygonSupport(StabilityChecker):
    """
    CONSTRAINT 3: Center-of-gravity polygon support.

    An item is stable if its CG lies within the interior or boundary of
    the support polygon (SP), which is the convex hull of intersection
    vertices between the item's base and supporting items' top surfaces.

    Three conditions (any sufficient):
    1. Floor support: z = 0
    2. Direct support: CG directly over a supporting item
    3. Polygon support: CG inside/on boundary of SP

    Paper results (Table 8):
        Bins used: 88,601 (FEWEST) | Stability: 88.04% | Efficiency: 100%

    Paper results by size class:
        Small: 1.00 bins, 99.96% stable
        Medium: 1.45 bins, 94.40% stable
        Large: 3.91 bins, 87.77% stable

    Processing time per item:
        Small: 0.004s | Medium: 0.683s | Large: 7.198s (SLOWEST)

    Complexity: O(n + k*log(k)) per check where n = placed items,
    k = intersection vertices (typically 4*n_supporting).
    """

    def check(self, item_dims: ItemDims, position: Position,
              placed_items: List[PlacedItem]) -> StabilityResult:

        base_area = item_dims.base_area

        # Condition 1: Floor support
        if position.z < EPSILON:
            return StabilityResult(
                is_stable=True, constraint_name='cog_polygon',
                supported_area=base_area, base_area=base_area,
                support_ratio=1.0, cg_status='floor',
                condition_used=1
            )

        # Compute CG position in XY plane
        x_cg = position.x + item_dims.length / 2.0
        y_cg = position.y + item_dims.width / 2.0

        # Find supporting items (top_z == item bottom_z)
        supporting = []
        for p in placed_items:
            if abs(p.top_z - position.z) > EPSILON:
                continue
            # Check XY overlap
            if (p.position.x < position.x + item_dims.length - EPSILON and
                p.x_max > position.x + EPSILON and
                p.position.y < position.y + item_dims.width - EPSILON and
                p.y_max > position.y + EPSILON):
                supporting.append(p)

        if not supporting:
            return StabilityResult(
                is_stable=False, constraint_name='cog_polygon',
                base_area=base_area, cg_status='no_support',
                condition_used=0
            )

        # Condition 2: Direct support -- CG directly over a single item
        for s in supporting:
            if (s.position.x <= x_cg + EPSILON and x_cg <= s.x_max + EPSILON and
                s.position.y <= y_cg + EPSILON and y_cg <= s.y_max + EPSILON):
                # Compute supported area for metadata
                sup_area, _ = compute_supported_area(
                    position.x, position.y, item_dims.length, item_dims.width,
                    position.z, placed_items
                )
                return StabilityResult(
                    is_stable=True, constraint_name='cog_polygon',
                    supported_area=sup_area, base_area=base_area,
                    support_ratio=sup_area / base_area if base_area > 0 else 0,
                    cg_status='direct',
                    supporting_item_count=len(supporting),
                    condition_used=2
                )

        # Condition 3: Polygon support -- CG inside convex hull of intersections
        # Step 1: Compute all intersection vertices
        all_vertices = []
        for s in supporting:
            vertices = compute_intersection_vertices(
                position.x, position.y, item_dims.length, item_dims.width,
                s.position.x, s.position.y, s.dims.length, s.dims.width
            )
            all_vertices.extend(vertices)

        if len(all_vertices) < 3:
            # Not enough points to form a polygon
            return StabilityResult(
                is_stable=False, constraint_name='cog_polygon',
                base_area=base_area, cg_status='degenerate_sp',
                supporting_item_count=len(supporting),
                condition_used=0
            )

        # Step 2: Deduplicate vertices
        unique_verts = deduplicate_points(all_vertices)

        if len(unique_verts) < 3:
            return StabilityResult(
                is_stable=False, constraint_name='cog_polygon',
                base_area=base_area, cg_status='degenerate_sp',
                supporting_item_count=len(supporting),
                condition_used=0
            )

        # Step 3: Convex hull
        hull = gift_wrapping_convex_hull(unique_verts)

        if len(hull) < 3:
            return StabilityResult(
                is_stable=False, constraint_name='cog_polygon',
                base_area=base_area, cg_status='degenerate_hull',
                sp_vertices=hull,
                supporting_item_count=len(supporting),
                condition_used=0
            )

        # Step 4: Point-in-polygon test
        cg_status = point_in_polygon_vp((x_cg, y_cg), hull)

        is_stable = cg_status in ('inside', 'boundary')

        # Compute SP area for metadata
        sp_area = polygon_area_shoelace(hull)
        sup_area, _ = compute_supported_area(
            position.x, position.y, item_dims.length, item_dims.width,
            position.z, placed_items
        )

        return StabilityResult(
            is_stable=is_stable, constraint_name='cog_polygon',
            supported_area=sup_area, base_area=base_area,
            support_ratio=sup_area / base_area if base_area > 0 else 0,
            cg_status=cg_status,
            sp_area=sp_area, sp_vertices=hull,
            supporting_item_count=len(supporting),
            condition_used=3
        )


class PartialBasePolygonSupport(StabilityChecker):
    """
    CONSTRAINT 4: Partial-base polygon support (THE PAPER'S NOVEL CONTRIBUTION).

    Combines CoG polygon support with a minimum support polygon area requirement.
    An item is stable if:
    - CG is within/on boundary of SP, AND
    - area(SP) >= area_threshold * base_area (default 50%)

    OR: on floor (Condition 1), OR CG directly supported (Condition 2).

    Paper results (Table 8):
        Bins used: 95,921 | Stability: 92.19% | Efficiency: 92%

    Paper results by size class:
        Small: 1.00 bins, 100% stable
        Medium: 1.57 bins, 96.24% stable
        Large: 4.25 bins, 92.16% stable

    Processing time per item:
        Small: 0.003s | Medium: 0.640s | Large: 6.754s

    THE RECOMMENDED DEFAULT for our thesis system:
    - Only 8% more bins than CoG polygon (4.25 vs 3.91 for large)
    - But 4 percentage points more stability (92.19% vs 88.04%)
    - 47% fewer bins than full-base support (4.25 vs 8.05 for large)

    The 50% threshold is tunable and was NOT sensitivity-tested in the paper.
    This is an explicit future work opportunity for our thesis.

    Complexity: O(n + k*log(k)) per check (same as CoG polygon, plus area check).
    """

    def __init__(self, area_threshold: float = 0.50):
        self.area_threshold = area_threshold

    def check(self, item_dims: ItemDims, position: Position,
              placed_items: List[PlacedItem]) -> StabilityResult:

        base_area = item_dims.base_area

        # Condition 1: Floor support
        if position.z < EPSILON:
            return StabilityResult(
                is_stable=True, constraint_name='partial_base_polygon',
                supported_area=base_area, base_area=base_area,
                support_ratio=1.0, cg_status='floor',
                sp_area=base_area, condition_used=1
            )

        # CG position
        x_cg = position.x + item_dims.length / 2.0
        y_cg = position.y + item_dims.width / 2.0

        # Find supporting items
        supporting = []
        for p in placed_items:
            if abs(p.top_z - position.z) > EPSILON:
                continue
            if (p.position.x < position.x + item_dims.length - EPSILON and
                p.x_max > position.x + EPSILON and
                p.position.y < position.y + item_dims.width - EPSILON and
                p.y_max > position.y + EPSILON):
                supporting.append(p)

        if not supporting:
            return StabilityResult(
                is_stable=False, constraint_name='partial_base_polygon',
                base_area=base_area, cg_status='no_support', condition_used=0
            )

        # Condition 2: Direct support
        for s in supporting:
            if (s.position.x <= x_cg + EPSILON and x_cg <= s.x_max + EPSILON and
                s.position.y <= y_cg + EPSILON and y_cg <= s.y_max + EPSILON):
                sup_area, _ = compute_supported_area(
                    position.x, position.y, item_dims.length, item_dims.width,
                    position.z, placed_items
                )
                return StabilityResult(
                    is_stable=True, constraint_name='partial_base_polygon',
                    supported_area=sup_area, base_area=base_area,
                    support_ratio=sup_area / base_area if base_area > 0 else 0,
                    cg_status='direct',
                    supporting_item_count=len(supporting),
                    condition_used=2
                )

        # Condition 3: Polygon support WITH area requirement
        all_vertices = []
        for s in supporting:
            vertices = compute_intersection_vertices(
                position.x, position.y, item_dims.length, item_dims.width,
                s.position.x, s.position.y, s.dims.length, s.dims.width
            )
            all_vertices.extend(vertices)

        unique_verts = deduplicate_points(all_vertices)
        if len(unique_verts) < 3:
            return StabilityResult(
                is_stable=False, constraint_name='partial_base_polygon',
                base_area=base_area, cg_status='degenerate_sp',
                supporting_item_count=len(supporting), condition_used=0
            )

        hull = gift_wrapping_convex_hull(unique_verts)
        if len(hull) < 3:
            return StabilityResult(
                is_stable=False, constraint_name='partial_base_polygon',
                base_area=base_area, cg_status='degenerate_hull',
                sp_vertices=hull, supporting_item_count=len(supporting),
                condition_used=0
            )

        # CG inside polygon?
        cg_status = point_in_polygon_vp((x_cg, y_cg), hull)
        cg_ok = cg_status in ('inside', 'boundary')

        # SP area meets threshold?
        sp_area = polygon_area_shoelace(hull)
        area_ok = sp_area >= self.area_threshold * base_area - EPSILON

        is_stable = cg_ok and area_ok

        sup_area, _ = compute_supported_area(
            position.x, position.y, item_dims.length, item_dims.width,
            position.z, placed_items
        )

        return StabilityResult(
            is_stable=is_stable, constraint_name='partial_base_polygon',
            supported_area=sup_area, base_area=base_area,
            support_ratio=sup_area / base_area if base_area > 0 else 0,
            cg_status=cg_status if cg_ok else f'cg_{cg_status}_area_{"ok" if area_ok else "fail"}',
            sp_area=sp_area, sp_vertices=hull,
            supporting_item_count=len(supporting),
            condition_used=3 if is_stable else 0
        )


# ============================================================================
# MODULE 2: SUPPORT POLYGON COMPUTATION -- FULL IMPLEMENTATIONS
# ============================================================================
# These are the geometric primitives required by Constraints 3 and 4.
# Paper reference: Section 3.3-3.4, pages 4-6, Figures 2-5.

def compute_intersection_vertices(
    bx: float, by: float, bl: float, bw: float,
    sx: float, sy: float, sl: float, sw: float
) -> List[Tuple[float, float]]:
    """
    Compute the 4 vertices of the intersection rectangle between
    item b's base and supporting item s's top face.

    From paper (Section 3.3, page 6):
        v_{j1} = (max(x_b, x_{bj}), max(y_b, y_{bj}))
        v_{j2} = (min(x_b + l_b, x_{bj} + l_{bj}), max(y_b, y_{bj}))
        v_{j3} = (max(x_b, x_{bj}), min(y_b + w_b, y_{bj} + w_{bj}))
        v_{j4} = (min(x_b + l_b, x_{bj} + l_{bj}), min(y_b + w_b, y_{bj} + w_{bj}))

    Returns empty list if rectangles don't intersect.
    """
    # Check intersection exists
    if bx + bl <= sx + EPSILON or sx + sl <= bx + EPSILON:
        return []
    if by + bw <= sy + EPSILON or sy + sw <= by + EPSILON:
        return []

    v1 = (max(bx, sx), max(by, sy))
    v2 = (min(bx + bl, sx + sl), max(by, sy))
    v3 = (max(bx, sx), min(by + bw, sy + sw))
    v4 = (min(bx + bl, sx + sl), min(by + bw, sy + sw))

    return [v1, v2, v3, v4]


def deduplicate_points(points: List[Tuple[float, float]],
                       eps: float = EPSILON) -> List[Tuple[float, float]]:
    """Remove duplicate points within epsilon tolerance."""
    if not points:
        return []
    unique = [points[0]]
    for p in points[1:]:
        is_dup = False
        for u in unique:
            if abs(p[0] - u[0]) < eps and abs(p[1] - u[1]) < eps:
                is_dup = True
                break
        if not is_dup:
            unique.append(p)
    return unique


def gift_wrapping_convex_hull(
    points: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Compute convex hull using the Gift Wrapping (Jarvis March) algorithm.
    Returns vertices in counterclockwise order.

    This is the specific algorithm cited in the paper (Jarvis, 1973).

    Algorithm (from paper Section 3.3, page 6):
    1. Find P_0 with minimum y-coordinate (min x to break ties)
    2. Compute next vertex by finding smallest CCW angle from current
    3. Repeat until returning to P_0
    4. Eliminate collinear points

    Complexity: O(nh) where n = input points, h = hull vertices.
    For our use case n <= ~40 (10 supporting items * 4 vertices), so O(n^2)
    worst case ~ O(1600), negligible.
    """
    if len(points) < 3:
        return list(points)

    # Remove exact duplicates first
    pts = deduplicate_points(points)
    if len(pts) < 3:
        return list(pts)

    # Find starting point: minimum y, break ties by minimum x
    start = min(pts, key=lambda p: (p[1], p[0]))
    hull = []
    current = start
    prev_angle = 0.0  # Start looking rightward

    for _ in range(len(pts) + 1):  # safety limit
        hull.append(current)
        best_next = None
        best_angle = float('inf')
        best_dist = -1.0

        for candidate in pts:
            if candidate == current:
                continue
            dx = candidate[0] - current[0]
            dy = candidate[1] - current[1]
            angle = math.atan2(dy, dx)

            # Normalize angle relative to previous direction
            # We want the smallest CCW turn
            relative = angle - prev_angle
            while relative < -EPSILON:
                relative += 2 * math.pi
            while relative >= 2 * math.pi - EPSILON:
                relative -= 2 * math.pi

            dist = math.sqrt(dx * dx + dy * dy)

            if (relative < best_angle - EPSILON or
                (abs(relative - best_angle) < EPSILON and dist > best_dist)):
                best_angle = relative
                best_next = candidate
                best_dist = dist

        if best_next is None or best_next == start:
            break

        prev_angle = math.atan2(best_next[1] - current[1],
                                best_next[0] - current[0])
        current = best_next

    # Remove collinear points
    if len(hull) >= 3:
        hull = _remove_collinear(hull)

    return hull


def _remove_collinear(hull: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Remove collinear points from a convex hull."""
    if len(hull) <= 2:
        return hull
    result = []
    n = len(hull)
    for i in range(n):
        p_prev = hull[(i - 1) % n]
        p_curr = hull[i]
        p_next = hull[(i + 1) % n]
        # Cross product
        cross = ((p_curr[0] - p_prev[0]) * (p_next[1] - p_prev[1]) -
                 (p_next[0] - p_prev[0]) * (p_curr[1] - p_prev[1]))
        if abs(cross) > EPSILON:
            result.append(p_curr)
    return result if len(result) >= 3 else hull


def point_in_polygon_vp(
    point: Tuple[float, float],
    polygon_ccw: List[Tuple[float, float]]
) -> str:
    """
    Test if a point lies inside/on a convex polygon using the Vector Product
    (VP) method from the paper.

    Paper formula (Section 3.3, page 6):
        VP = (x_CG - x_i) * (y_{i+1} - y_i) - (x_{i+1} - x_i) * (y_CG - y_i)

    Since SP was constructed counterclockwise:
    - VP < 0 for all edges: CG is inside
    - VP = 0 for any edge (and <= 0 for all): CG is on boundary
    - VP > 0 for any edge: CG is outside

    Returns: 'inside', 'boundary', or 'outside'
    """
    if len(polygon_ccw) < 3:
        return 'outside'

    x_cg, y_cg = point
    n = len(polygon_ccw)
    on_boundary = False

    for i in range(n):
        x_i, y_i = polygon_ccw[i]
        x_next, y_next = polygon_ccw[(i + 1) % n]

        vp = ((x_cg - x_i) * (y_next - y_i) -
              (x_next - x_i) * (y_cg - y_i))

        if vp > EPSILON:
            return 'outside'
        if abs(vp) <= EPSILON:
            on_boundary = True

    return 'boundary' if on_boundary else 'inside'


def polygon_area_shoelace(
    polygon: List[Tuple[float, float]]
) -> float:
    """
    Compute area of a simple polygon using the Shoelace formula.

    Paper formula (Section 3.4):
        area = 0.5 * |sum_{i=0}^{m-1} (x_i * y_{i+1} - x_{i+1} * y_i)|
    """
    if len(polygon) < 3:
        return 0.0

    n = len(polygon)
    area = 0.0
    for i in range(n):
        x_i, y_i = polygon[i]
        x_next, y_next = polygon[(i + 1) % n]
        area += x_i * y_next - x_next * y_i

    return abs(area) / 2.0


def compute_full_support_polygon(
    item_x: float, item_y: float, item_l: float, item_w: float,
    item_z: float,
    placed_items: List[PlacedItem]
) -> Dict[str, Any]:
    """
    Full support polygon computation pipeline.

    Returns a dict with:
        'vertices': convex hull vertices in CCW order
        'area': SP area
        'cg': (x_cg, y_cg)
        'cg_status': 'inside', 'boundary', or 'outside'
        'supporting_count': number of supporting items
        'support_ratio': SP area / base area
    """
    x_cg = item_x + item_l / 2.0
    y_cg = item_y + item_w / 2.0
    base_area = item_l * item_w

    # Find supporting items
    supporting = []
    for p in placed_items:
        if abs(p.top_z - item_z) > EPSILON:
            continue
        if (p.position.x < item_x + item_l - EPSILON and
            p.x_max > item_x + EPSILON and
            p.position.y < item_y + item_w - EPSILON and
            p.y_max > item_y + EPSILON):
            supporting.append(p)

    if not supporting:
        return {
            'vertices': [], 'area': 0.0,
            'cg': (x_cg, y_cg), 'cg_status': 'no_support',
            'supporting_count': 0, 'support_ratio': 0.0
        }

    # Compute intersection vertices
    all_verts = []
    for s in supporting:
        verts = compute_intersection_vertices(
            item_x, item_y, item_l, item_w,
            s.position.x, s.position.y, s.dims.length, s.dims.width
        )
        all_verts.extend(verts)

    unique = deduplicate_points(all_verts)
    if len(unique) < 3:
        return {
            'vertices': unique, 'area': 0.0,
            'cg': (x_cg, y_cg), 'cg_status': 'degenerate',
            'supporting_count': len(supporting), 'support_ratio': 0.0
        }

    hull = gift_wrapping_convex_hull(unique)
    sp_area = polygon_area_shoelace(hull) if len(hull) >= 3 else 0.0
    cg_status = point_in_polygon_vp((x_cg, y_cg), hull) if len(hull) >= 3 else 'outside'

    return {
        'vertices': hull,
        'area': sp_area,
        'cg': (x_cg, y_cg),
        'cg_status': cg_status,
        'supporting_count': len(supporting),
        'support_ratio': sp_area / base_area if base_area > 0 else 0.0
    }


# ============================================================================
# MODULE 3: STATIC MECHANICAL EQUILIBRIUM (GROUND TRUTH)
# ============================================================================
# Based on Ramos et al. (2016b): "A physical packing sequence algorithm for
# the container loading problem with static mechanical equilibrium conditions"
# Int. Trans. Oper. Res. 23(1-2), 215-238.
#
# This is the physics-based gold standard used in the paper to validate
# the four approximate stability constraints. Too slow for online use
# (O(n^2)), but essential for post-hoc validation.

def check_mechanical_equilibrium(item_idx: int,
                                  placed_items: List[PlacedItem]) -> bool:
    """
    Full static mechanical equilibrium check based on Newton's laws.

    An item is in mechanical equilibrium if one of:
    i)   Item lies on the container floor
    ii)  Application point of resultant downward force lies directly
         above another item
    iii) Application point of resultant force lies inside a support polygon

    This accounts for LOAD TRANSFER: when items are stacked multiple layers
    deep, forces cascade downward. An item stable at placement time may
    become unstable when more items are placed on top.

    This is the key weakness of greedy online approaches (see paper Fig. 4).

    Args:
        item_idx: index into placed_items of the item to check
        placed_items: all placed items in the bin (final configuration)

    Returns:
        True if item is in static mechanical equilibrium
    """
    item = placed_items[item_idx]

    # Condition i: On floor
    if item.position.z < EPSILON:
        return True

    # For conditions ii and iii, we need the resultant force application point.
    # For uniform density with only gravity, the application point of the
    # resultant downward force is the CG projected onto the base plane,
    # UNLESS items above are adding load unevenly.
    #
    # Simplified model (matching paper's approach):
    # Only gravity acts. All forces are vertical. The resultant force
    # application point equals the CG projection for items with no load
    # from above, or the weighted centroid for items carrying stacked load.

    # Compute the effective application point considering items stacked above
    total_force = item.weight
    moment_x = item.weight * (item.position.x + item.dims.length / 2.0)
    moment_y = item.weight * (item.position.y + item.dims.width / 2.0)

    # Find items resting directly on top of this item
    for other in placed_items:
        if other is item:
            continue
        if abs(other.position.z - item.top_z) < EPSILON:
            # Check XY overlap
            ix_min = max(item.position.x, other.position.x)
            ix_max = min(item.x_max, other.x_max)
            iy_min = max(item.position.y, other.position.y)
            iy_max = min(item.y_max, other.y_max)
            if ix_max > ix_min + EPSILON and iy_max > iy_min + EPSILON:
                # This item contributes load
                # Simplified: proportional to overlap area
                overlap_frac = ((ix_max - ix_min) * (iy_max - iy_min) /
                                (other.dims.base_area + EPSILON))
                added_force = other.weight * overlap_frac
                total_force += added_force
                ocg_x, ocg_y = other.cg_xy
                moment_x += added_force * ocg_x
                moment_y += added_force * ocg_y

    # Application point of resultant force
    app_x = moment_x / total_force if total_force > EPSILON else item.cg_xy[0]
    app_y = moment_y / total_force if total_force > EPSILON else item.cg_xy[1]

    # Now check if (app_x, app_y) is supported
    sp_info = compute_full_support_polygon(
        item.position.x, item.position.y,
        item.dims.length, item.dims.width,
        item.position.z, placed_items
    )

    # Condition ii: Direct support
    for s in [p for p in placed_items if abs(p.top_z - item.position.z) < EPSILON]:
        if (s.position.x <= app_x + EPSILON and app_x <= s.x_max + EPSILON and
            s.position.y <= app_y + EPSILON and app_y <= s.y_max + EPSILON):
            return True

    # Condition iii: Inside support polygon
    if len(sp_info['vertices']) >= 3:
        status = point_in_polygon_vp((app_x, app_y), sp_info['vertices'])
        if status in ('inside', 'boundary'):
            return True

    return False


def evaluate_all_stability(placed_items: List[PlacedItem]) -> Dict[str, Any]:
    """
    Evaluate mechanical equilibrium for ALL items in a bin.

    Returns:
        dict with 'total_items', 'stable_items', 'unstable_items',
        'stability_pct', 'unstable_indices'
    """
    total = len(placed_items)
    stable = 0
    unstable_indices = []

    for i in range(total):
        if check_mechanical_equilibrium(i, placed_items):
            stable += 1
        else:
            unstable_indices.append(i)

    return {
        'total_items': total,
        'stable_items': stable,
        'unstable_items': total - stable,
        'stability_pct': 100.0 * stable / total if total > 0 else 100.0,
        'unstable_indices': unstable_indices
    }


# ============================================================================
# MODULE 4: MULTI-OBJECTIVE EVALUATION (PARETO ANALYSIS)
# ============================================================================
# Replicates Figure 7 (non-dominated heuristics) and Figure 8
# (stability-efficiency tradeoff curve) from the paper.

def compute_pareto_front(
    results: List[Tuple[str, float, float]]
) -> List[Tuple[str, float, float]]:
    """
    Identify the Pareto front on (stability%, efficiency%) space.

    A result is non-dominated if no other result is better on BOTH objectives.

    Args:
        results: list of (name, stability_pct, efficiency_pct) tuples

    Returns:
        list of non-dominated tuples

    This replicates Figure 7 from the paper where:
    - Full-base: 2 non-dominated (F52, A52)
    - Partial-base: 1 non-dominated (F51)
    - CoG polygon: 17 non-dominated
    - Partial-base polygon: 16 non-dominated
    """
    non_dominated = []
    for i, (name_i, stab_i, eff_i) in enumerate(results):
        dominated = False
        for j, (name_j, stab_j, eff_j) in enumerate(results):
            if i == j:
                continue
            # j dominates i if j is >= on both and > on at least one
            if stab_j >= stab_i and eff_j >= eff_i:
                if stab_j > stab_i + EPSILON or eff_j > eff_i + EPSILON:
                    dominated = True
                    break
        if not dominated:
            non_dominated.append((name_i, stab_i, eff_i))
    return non_dominated


def compute_efficiency_relative(bins_used: int,
                                 reference_bins: int = 88601) -> float:
    """
    Compute relative efficiency as in the paper (Table 8).

    Efficiency% = (reference_bins / bins_used) * 100

    The reference is CoG polygon support with 88,601 bins (best constrained).
    """
    if bins_used <= 0:
        return 0.0
    return 100.0 * reference_bins / bins_used


# ============================================================================
# MODULE 5: ADAPTIVE STABILITY SELECTOR
# ============================================================================
# The paper uses one constraint for all items. With our buffer, we can
# switch constraints per-item based on the current bin state.

def select_stability_constraint(
    bin_fill_rate: float,
    placement_height: float,
    bin_height: float,
    stability_checkers: Dict[str, StabilityChecker]
) -> StabilityChecker:
    """
    Dynamically select which stability constraint to apply based on
    current placement context.

    ADAPTIVE STRATEGY (height-based gradient):
    - Bottom layer (height < 30% of bin): FullBaseSupport
      Bottom items bear the most load and must be maximally stable.
    - Middle layers (30-60%): PartialBasePolygonSupport(0.50)
      Balance efficiency and stability in the main body.
    - Top layers (>60%): CoGPolygonSupport
      Top items bear the least load; maximize remaining space usage.

    ALTERNATIVE STRATEGY (fill-based gradient):
    - Bin < 30% full: strict (building foundation)
    - Bin 30-70% full: balanced
    - Bin > 70% full: permissive (filling gaps)

    Args:
        bin_fill_rate: current volume utilization of the bin (0-1)
        placement_height: z-coordinate where item will be placed
        bin_height: total bin height
        stability_checkers: dict of available checkers

    Returns:
        The recommended StabilityChecker for this placement
    """
    height_ratio = placement_height / bin_height if bin_height > 0 else 0

    if height_ratio < 0.30:
        return stability_checkers.get('full_base',
               stability_checkers.get('partial_base_polygon'))
    elif height_ratio < 0.60:
        return stability_checkers.get('partial_base_polygon',
               stability_checkers.get('cog_polygon'))
    else:
        return stability_checkers.get('cog_polygon',
               stability_checkers.get('partial_base_polygon'))


# ============================================================================
# MODULE 6: VISUALIZATION (for thesis figures)
# ============================================================================

def visualize_support_polygon_data(
    item_x: float, item_y: float, item_l: float, item_w: float,
    supporting_items: List[PlacedItem],
    sp_vertices: List[Tuple[float, float]],
    cg: Tuple[float, float],
    cg_status: str
) -> Dict[str, Any]:
    """
    Prepare data for matplotlib visualization of a support polygon,
    replicating paper Figures 3, 4, 5.

    Returns a dict with all plot data ready for rendering:
    - 'item_rect': (x, y, w, h) for the item base
    - 'support_rects': list of (x, y, w, h) for supporting tops
    - 'intersection_rects': list of intersection rectangles
    - 'hull_polygon': list of (x, y) for convex hull
    - 'cg_point': (x, y)
    - 'cg_color': 'green' if inside, 'red' if outside, 'orange' if boundary
    """
    cg_color = {'inside': 'green', 'boundary': 'orange',
                'outside': 'red', 'direct': 'blue'}.get(cg_status, 'gray')

    support_rects = []
    intersection_rects = []
    for s in supporting_items:
        support_rects.append((s.position.x, s.position.y,
                              s.dims.length, s.dims.width))
        # Intersection rectangle
        ix1 = max(item_x, s.position.x)
        iy1 = max(item_y, s.position.y)
        ix2 = min(item_x + item_l, s.x_max)
        iy2 = min(item_y + item_w, s.y_max)
        if ix2 > ix1 and iy2 > iy1:
            intersection_rects.append((ix1, iy1, ix2 - ix1, iy2 - iy1))

    return {
        'item_rect': (item_x, item_y, item_l, item_w),
        'support_rects': support_rects,
        'intersection_rects': intersection_rects,
        'hull_polygon': sp_vertices,
        'cg_point': cg,
        'cg_color': cg_color,
    }


def visualize_tradeoff_data() -> Dict[str, Tuple[float, float]]:
    """
    Data points for the stability-efficiency tradeoff curve (paper Fig. 8).

    Returns dict mapping constraint name to (stability%, efficiency%).
    """
    return {
        'Full-base support':           (100.00, 51.04),
        'Partial-base support (80%)':  (99.99, 63.42),
        'CoG polygon support':         (88.04, 100.00),
        'Partial-base polygon (50%)':  (92.19, 92.37),
        'No stability constraint':     (50.08, 136.49),
    }


# ============================================================================
# MODULE 7: SENSITIVITY ANALYSIS FRAMEWORK
# ============================================================================
# The paper does NOT test different area thresholds for partial-base polygon.
# This is an explicit opportunity for our thesis contribution.

def run_threshold_sensitivity(
    instances: list,
    thresholds: List[float] = None,
    heuristic_name: str = 'A53'
) -> Dict[float, Dict[str, float]]:
    """
    Test the partial-base polygon support with different area thresholds.

    This fills a gap identified in the paper:
    "Note that this percentage is a parameter that can be adjusted within
    the decision support framework." (Section 3.4)

    Args:
        instances: list of test instances
        thresholds: list of area thresholds to test (default: 0.3-0.8)
        heuristic_name: which heuristic to use

    Returns:
        dict mapping threshold to {'avg_bins', 'avg_stability', 'avg_time'}

    Expected results (estimated, not from paper):
        0.30: ~89% stability, ~97% efficiency (near CoG polygon)
        0.40: ~90% stability, ~95% efficiency
        0.50: ~92% stability, ~92% efficiency (paper's setting)
        0.60: ~95% stability, ~82% efficiency
        0.70: ~97% stability, ~72% efficiency
        0.80: ~99% stability, ~65% efficiency (near partial-base)
    """
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    results = {}
    for t in thresholds:
        checker = PartialBasePolygonSupport(area_threshold=t)
        # Run benchmark with this checker
        # results[t] = run_single_benchmark(instances, checker, heuristic_name)
        results[t] = {'threshold': t, 'status': 'not_yet_run'}

    return results


# ============================================================================
# MODULE 8: CARGO HEIGHT ANALYSIS
# ============================================================================
# Paper Figures 9-10 show strong positive correlation between cargo height
# and number of unstable items. This module tracks height and adapts.

def analyze_height_instability_correlation(
    bins_results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Analyze the relationship between cargo height and instability,
    replicating paper Figures 9 and 10.

    The paper shows:
    - Under CoG polygon: at height ~1500mm, 20-80 unstable items
    - Under partial-base polygon: at height ~1500mm, 10-60 unstable items
    - Cargo height has MORE significant impact than number of items

    Returns:
        dict with correlation coefficients and regression parameters
    """
    heights = []
    unstable_counts = []

    for result in bins_results:
        max_height = max(
            (item.top_z for item in result.get('placed_items', [])),
            default=0
        )
        heights.append(max_height)
        unstable_counts.append(result.get('unstable_items', 0))

    if len(heights) < 2:
        return {'correlation': 0.0, 'slope': 0.0, 'intercept': 0.0}

    # Simple linear regression
    n = len(heights)
    mean_h = sum(heights) / n
    mean_u = sum(unstable_counts) / n

    cov = sum((h - mean_h) * (u - mean_u) for h, u in zip(heights, unstable_counts))
    var_h = sum((h - mean_h) ** 2 for h in heights)
    var_u = sum((u - mean_u) ** 2 for u in unstable_counts)

    slope = cov / var_h if var_h > 0 else 0.0
    intercept = mean_u - slope * mean_h
    correlation = cov / math.sqrt(var_h * var_u) if var_h > 0 and var_u > 0 else 0.0

    return {
        'correlation': correlation,
        'slope': slope,
        'intercept': intercept,
        'mean_height': mean_h,
        'mean_unstable': mean_u,
        'n_samples': n
    }


# ============================================================================
# IMPLEMENTATION PRIORITY & ROADMAP (UPDATED)
# ============================================================================
#
# PHASE 1 (Week 1-2): Core Infrastructure
#   1. support_polygon.py: convex hull, point-in-polygon, area
#      (compute_intersection_vertices, gift_wrapping_convex_hull,
#       point_in_polygon_vp, polygon_area_shoelace) -- 2 days
#   2. stability_checks.py: all 4 constraints as classes -- 2 days
#   3. Unit tests for geometry primitives -- 1 day
#   4. Unit tests for stability checkers -- 1 day
#
# PHASE 2 (Week 2-3): Integration with Heuristic Framework
#   5. Plug stability checkers into HeuristicEngine from
#      coding_ideas_160_heuristic_framework.py -- 1 day
#   6. Download 198 Mendeley instances, implement loader -- 1 day
#   7. Validate: reproduce paper's Table 5 and Table 8 -- 3 days
#
# PHASE 3 (Week 3-4): Evaluation & Analysis
#   8. Pareto front computation (compute_pareto_front) -- 1 day
#   9. Mechanical equilibrium checker for offline validation -- 2 days
#   10. Threshold sensitivity analysis (run_threshold_sensitivity) -- 1 day
#   11. Height-instability correlation analysis -- 1 day
#
# PHASE 4 (Week 4-5): Visualization & Thesis Figures
#   12. Support polygon visualization (replicating Figs. 3-5) -- 1 day
#   13. Tradeoff curve (replicating Fig. 8) -- 0.5 day
#   14. Pareto front plot (replicating Fig. 7) -- 0.5 day
#   15. 3D packing visualization (replicating Appendix I) -- 2 days
#
# ============================================================================
# KEY NUMBERS TO VALIDATE AGAINST
# ============================================================================
#
# From Table 5 (average bins per constraint per size class):
#   Full-base:   Small=1.006, Medium=2.472, Large=8.055
#   Partial-base: Small=1.001, Medium=2.098, Large=6.377
#   CoG polygon:  Small=1.001, Medium=1.450, Large=3.911
#   Partial-base polygon: Small=1.001, Medium=1.573, Large=4.247
#
# From Table 6 (avg processing time per item in seconds):
#   Full-base:   Small=0.003, Medium=0.106, Large=1.743
#   Partial-base: Small=0.003, Medium=0.246, Large=3.048
#   CoG polygon:  Small=0.004, Medium=0.683, Large=7.198
#   Partial-base polygon: Small=0.003, Medium=0.640, Large=6.754
#
# From Appendix F (per-item per-rule avg processing time in seconds):
#   Full-base:   Small=0.001, Medium=0.003, Large=0.008
#   Partial-base: Small=0.001, Medium=0.007, Large=0.017
#   CoG polygon:  Small=0.001, Medium=0.018, Large=0.047
#   Partial-base polygon: Small=0.001, Medium=0.017, Large=0.043
#
# From Table 7 (online vs offline):
#   Equal results: 140 (71%)
#   Offline wins: 58 (29%)
#   Online wins: 0 (0%)
#
# ============================================================================
"""
"""
