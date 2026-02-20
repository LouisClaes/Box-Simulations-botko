"""
=============================================================================
CODING IDEAS: 160-Heuristic Framework from Ali et al. (2025)
=============================================================================

Based on: Ali, Ramos, Oliveira (2025) - "Static stability versus packing
efficiency in online three-dimensional packing problems"
Computers & Operations Research, Vol. 178, Article 107005

Also: Ali, Ramos, Carravilla, Oliveira (2024) - "Heuristics for online
three-dimensional packing problems and algorithm selection framework for
semi-online with full look-ahead"
Applied Soft Computing, Vol. 151, Article 111168

This file covers the HEURISTIC FRAMEWORK: the 4x8x5 = 160 combinations of
bin selection, space selection, and orientation selection rules that form
the core online packing engine.

For stability-specific code, see: python/stability/coding_ideas_stability_vs_efficiency.py
For buffer integration, see: python/semi_online_buffer/coding_ideas_buffer_stability_integration.py

=============================================================================
UPDATED: 2026-02-17 -- Deep research pass with complete paper extraction
=============================================================================
"""

import math
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from itertools import product


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class Dimensions:
    """Item or bin dimensions in mm."""
    length: float  # x-axis (depth, from entrance to back)
    width: float   # y-axis (left-right)
    height: float  # z-axis (bottom-top)

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def base_area(self) -> float:
        return self.length * self.width


@dataclass
class Position:
    """3D position (minimum vertex / deepest-bottom-left corner)."""
    x: float  # depth position
    y: float  # width position
    z: float  # height position


@dataclass
class EMS:
    """
    Empty Maximal Space -- the fundamental free-space representation.

    Based on Parreno, Alvarez-Valdes, Tamarit & Oliveira (2008).
    Each EMS is the largest free parallelepiped, defined by two corners:
    - (x1, y1, z1): minimum vertex (deepest-bottom-left)
    - (x2, y2, z2): maximum vertex (front-top-right)
    """
    x1: float  # min x
    y1: float  # min y
    z1: float  # min z
    x2: float  # max x
    y2: float  # max y
    z2: float  # max z
    creation_order: int = 0  # for tie-breaking in space selection

    @property
    def volume(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1) * max(0, self.z2 - self.z1)

    @property
    def dims(self) -> Tuple[float, float, float]:
        return (self.x2 - self.x1, self.y2 - self.y1, self.z2 - self.z1)

    def contains(self, other: 'EMS') -> bool:
        """Check if this EMS fully contains another EMS."""
        return (self.x1 <= other.x1 and self.y1 <= other.y1 and self.z1 <= other.z1 and
                self.x2 >= other.x2 and self.y2 >= other.y2 and self.z2 >= other.z2)

    def intersects_item(self, pos: Position, dims: Dimensions) -> bool:
        """Check if an item at position with given dims intersects this EMS."""
        return not (pos.x + dims.length <= self.x1 or pos.x >= self.x2 or
                    pos.y + dims.width <= self.y1 or pos.y >= self.y2 or
                    pos.z + dims.height <= self.z1 or pos.z >= self.z2)

    def is_valid(self, min_dim: float = 1.0) -> bool:
        """Check if EMS has positive volume above a minimum dimension threshold."""
        return ((self.x2 - self.x1) >= min_dim and
                (self.y2 - self.y1) >= min_dim and
                (self.z2 - self.z1) >= min_dim)


@dataclass
class PlacedItem:
    """An item that has been placed in a bin."""
    item_id: int
    position: Position
    dims: Dimensions  # dimensions in the placed orientation
    original_dims: Dimensions  # original item dimensions
    orientation: int  # 0-5

    @property
    def top_z(self) -> float:
        return self.position.z + self.dims.height

    @property
    def volume(self) -> float:
        return self.dims.volume

    @property
    def base_area(self) -> float:
        return self.dims.length * self.dims.width

    @property
    def center_of_gravity(self) -> Tuple[float, float, float]:
        """CG at geometric center (uniform density assumption)."""
        return (self.position.x + self.dims.length / 2,
                self.position.y + self.dims.width / 2,
                self.position.z + self.dims.height / 2)


# ============================================================================
# ORIENTATION SYSTEM
# ============================================================================
# 6 possible orthogonal orientations for a box with dims (l, w, h):
# 0: (l, w, h) -- original
# 1: (l, h, w) -- rotate around x-axis
# 2: (w, l, h) -- rotate around z-axis
# 3: (w, h, l) -- rotate around z then x
# 4: (h, l, w) -- rotate around y-axis
# 5: (h, w, l) -- rotate around y then z

def get_oriented_dims(dims: Dimensions, orientation: int) -> Dimensions:
    """
    Return item dimensions in the specified orientation.

    The paper allows up to 6 orientations; allowed set is instance-specific.
    """
    l, w, h = dims.length, dims.width, dims.height
    orientations = [
        Dimensions(l, w, h),  # 0: original
        Dimensions(l, h, w),  # 1
        Dimensions(w, l, h),  # 2
        Dimensions(w, h, l),  # 3
        Dimensions(h, l, w),  # 4
        Dimensions(h, w, l),  # 5
    ]
    return orientations[orientation]


def fits_in_ems(item_dims: Dimensions, ems: EMS) -> bool:
    """Check if item with given dims fits inside the EMS."""
    ems_l = ems.x2 - ems.x1
    ems_w = ems.y2 - ems.y1
    ems_h = ems.z2 - ems.z1
    return (item_dims.length <= ems_l + 1e-9 and
            item_dims.width <= ems_w + 1e-9 and
            item_dims.height <= ems_h + 1e-9)


# ============================================================================
# BIN SELECTION RULES (4 rules)
# ============================================================================
# Paper findings (from Appendix G, Fig. G.1 radar charts):
#
# - Full-base & partial-base: First-fit (F) and All-bins (A) dominate
#   for large instances.
# - CoG polygon: Worst-fit (W) competitive for stability; F and B lead
#   for efficiency.
# - Partial-base polygon: F and B are best overall.
# - Small instances: rule choice barely matters.
#
# For k=2 system: "All bins" checks just 2 bins -- no computational
# overhead vs F or B. Use A as default.

class BinSelectionRule(Enum):
    FIRST_FIT = 'F'
    BEST_FIT = 'B'
    WORST_FIT = 'W'
    ALL_BINS = 'A'


def select_bins_first_fit(active_bins: list) -> list:
    """Select the first (oldest) open bin. Returns list with 1 bin."""
    if active_bins:
        return [active_bins[0]]
    return []


def select_bins_best_fit(active_bins: list) -> list:
    """Select the fullest (most utilized) bin. Returns list with 1 bin."""
    if not active_bins:
        return []
    return [max(active_bins, key=lambda b: b.used_volume)]


def select_bins_worst_fit(active_bins: list) -> list:
    """Select the emptiest (least utilized) bin. Returns list with 1 bin."""
    if not active_bins:
        return []
    return [min(active_bins, key=lambda b: b.used_volume)]


def select_bins_all(active_bins: list) -> list:
    """Return ALL bins for global best evaluation."""
    return list(active_bins)


BIN_SELECTOR_MAP = {
    BinSelectionRule.FIRST_FIT: select_bins_first_fit,
    BinSelectionRule.BEST_FIT: select_bins_best_fit,
    BinSelectionRule.WORST_FIT: select_bins_worst_fit,
    BinSelectionRule.ALL_BINS: select_bins_all,
}


# ============================================================================
# SPACE (EMS) SELECTION RULES (8 rules) -- FROM TABLE 1
# ============================================================================
# Paper findings (from Appendix G, Fig. G.2 radar charts):
#
# - Rules 1 and 5 (DBLF variants) are consistently best for minimizing
#   bins across ALL stability constraints and medium/large instances.
# - Rule 3 (bottom-first) produces the best stability results.
# - Rule 4 (smallest EMS) good for tight packing.
# - Rules 6 & 7 (corner-first) moderate results with good stability.
# - Rule 8 (DFTRC variant) compact but not consistently best.
#
# RECOMMENDED: Rule 5 (DBLF with corner preference).

def space_rule_1_key(ems: EMS, bin_dims: Dimensions = None) -> tuple:
    """
    Rule 1: min x_1 -> min z_1 -> min y_1 (Classic DBLF)

    Pushes items to the back, then floor, then left.
    Origin: Karabulut & Inceoglu (2004).
    """
    return (ems.x1, ems.z1, ems.y1)


def space_rule_2_key(ems: EMS, bin_dims: Dimensions = None) -> tuple:
    """
    Rule 2: min of (x_1, y_1, z_1) -> next min -> next min (Lexicographic Min)

    No axis preference; prefers EMSs generally close to the origin.
    """
    vals = sorted([ems.x1, ems.y1, ems.z1])
    return tuple(vals)


def space_rule_3_key(ems: EMS, bin_dims: Dimensions = None) -> tuple:
    """
    Rule 3: min z_1 -> min of (x_1, y_1) -> next min (Bottom-First)

    Prioritizes lowest height above all else. Builds layers from floor up.
    Best for stability; not most efficient for bin count.
    """
    return (ems.z1, min(ems.x1, ems.y1), max(ems.x1, ems.y1))


def space_rule_4_key(ems: EMS, bin_dims: Dimensions = None) -> tuple:
    """
    Rule 4: smallest volume EMS -> earliest created

    Fills tiny gaps first. Good waste avoidance but can fragment space.
    """
    return (ems.volume, ems.creation_order)


def space_rule_5_key(ems: EMS, bin_dims: Dimensions = None) -> tuple:
    """
    Rule 5: min x_1 -> min z_1 -> nearest to back-bottom corner (DBLF + Corner)

    The MOST CONSISTENTLY BEST RULE across all constraints and instance sizes.
    Appears in majority of non-dominated heuristics (F52, A52, F51, B52, etc.).

    Third criterion: distance to the nearest of the two back-bottom corners:
    (0, 0, 0) or (0, W, 0) where W = bin width.
    """
    if bin_dims is None:
        bin_width = 800  # default EUR pallet width
    else:
        bin_width = bin_dims.width

    # Distance to back-bottom-left corner (0, 0, 0)
    d_left = math.sqrt(ems.x1**2 + ems.y1**2 + ems.z1**2)
    # Distance to back-bottom-right corner (0, W, 0)
    d_right = math.sqrt(ems.x1**2 + (ems.y1 - bin_width)**2 + ems.z1**2)

    return (ems.x1, ems.z1, min(d_left, d_right))


def space_rule_6_key(ems: EMS, bin_dims: Dimensions = None) -> tuple:
    """
    Rule 6: nearest to back-bottom corner -> largest volume -> earliest

    Corner-first, then prefer large spaces. Good for diverse item sizes.
    Appears in non-dominated: B63, W63, A63, F63.
    """
    if bin_dims is None:
        bin_width = 800
    else:
        bin_width = bin_dims.width

    d_left = math.sqrt(ems.x1**2 + ems.y1**2 + ems.z1**2)
    d_right = math.sqrt(ems.x1**2 + (ems.y1 - bin_width)**2 + ems.z1**2)
    nearest_dist = min(d_left, d_right)

    return (nearest_dist, -ems.volume, ems.creation_order)


def space_rule_7_key(ems: EMS, bin_dims: Dimensions = None) -> tuple:
    """
    Rule 7: nearest to back-bottom corner -> smallest volume -> earliest

    Corner-first, then prefer small spaces. Fills corners with small items.
    """
    if bin_dims is None:
        bin_width = 800
    else:
        bin_width = bin_dims.width

    d_left = math.sqrt(ems.x1**2 + ems.y1**2 + ems.z1**2)
    d_right = math.sqrt(ems.x1**2 + (ems.y1 - bin_width)**2 + ems.z1**2)
    nearest_dist = min(d_left, d_right)

    return (nearest_dist, ems.volume, ems.creation_order)


def space_rule_8_key(ems: EMS, bin_dims: Dimensions = None) -> tuple:
    """
    Rule 8: max distance from DBL vertex to bin FTR corner -> earliest

    DFTRC variant: prioritizes EMSs far from the bin's front-top-right
    corner, effectively pushing items deep, low, and to the left.

    FTR corner = (L, W, H) where L, W, H are bin dimensions.
    """
    if bin_dims is None:
        L, W, H = 1200, 800, 1500
    else:
        L, W, H = bin_dims.length, bin_dims.width, bin_dims.height

    dist = math.sqrt((L - ems.x1)**2 + (W - ems.y1)**2 + (H - ems.z1)**2)
    return (-dist, ems.creation_order)  # negative for max-first sorting


SPACE_RULE_MAP = {
    1: space_rule_1_key,
    2: space_rule_2_key,
    3: space_rule_3_key,
    4: space_rule_4_key,
    5: space_rule_5_key,
    6: space_rule_6_key,
    7: space_rule_7_key,
    8: space_rule_8_key,
}

# For Rules 5, 6, 7: items may be placed at the back-bottom corner
# nearest to the EMS (left OR right), not always at the DBL corner.
# This affects the placement position computation.
CORNER_PLACEMENT_RULES = {5, 6, 7}


# ============================================================================
# ORIENTATION SELECTION RULES (5 rules) -- FROM TABLE 2
# ============================================================================
# Paper findings (from Appendix G, Fig. G.3 radar charts):
#
# - Rule 3 (largest base area, greatest x-direction occupancy) is
#   consistently the BEST for minimizing bins across all constraints.
# - Rule 1 (min margin) is a close second, especially under polygon
#   constraints (tighter fit = more stable).
# - Rule 4 (multi-EMS comparison) good but slower.
# - Rule 5 (max distance) competitive but inconsistent.
# - Rule 2 worst for large instances.
#
# RECOMMENDED: Rule 3 as default, Rule 1 as alternative.

def orientation_rule_1(item_dims_list: List[Dimensions], ems: EMS,
                       all_ems: List[EMS] = None) -> int:
    """
    Rule 1: Orientation with minimum margin from item surfaces to EMS surfaces.
    Only considers 1 EMS (the top-ranked one).

    Margin = sum of gaps between item faces and EMS faces in all 3 axes:
        margin = (ems_l - item_l) + (ems_w - item_w) + (ems_h - item_h)

    Selects the orientation with the SMALLEST total margin (tightest fit).
    Tie-break: random.
    """
    best_idx = 0
    best_margin = float('inf')
    ems_l = ems.x2 - ems.x1
    ems_w = ems.y2 - ems.y1
    ems_h = ems.z2 - ems.z1

    for i, dims in enumerate(item_dims_list):
        margin = (ems_l - dims.length) + (ems_w - dims.width) + (ems_h - dims.height)
        if 0 <= margin < best_margin:
            best_margin = margin
            best_idx = i

    return best_idx


def orientation_rule_2(item_dims_list: List[Dimensions], ems: EMS,
                       all_ems: List[EMS] = None) -> int:
    """
    Rule 2: Largest base area -> smallest x-direction occupancy.
    Only considers 1 EMS.

    Prefers wide footprint with shallow depth (item stays close to back wall).
    """
    best_idx = 0
    best_key = (-1, float('inf'))

    for i, dims in enumerate(item_dims_list):
        base = dims.length * dims.width
        x_occ = dims.length  # depth occupancy
        key = (-base, x_occ)  # max base (negative for min sort), min x-occ
        if key < best_key:
            best_key = key
            best_idx = i

    return best_idx


def orientation_rule_3(item_dims_list: List[Dimensions], ems: EMS,
                       all_ems: List[EMS] = None) -> int:
    """
    Rule 3: Largest base area -> greatest x-direction occupancy.
    Only considers 1 EMS.

    THE MOST EFFECTIVE RULE OVERALL per paper's analysis.

    Prefers wide footprint AND deep placement. Most aggressive space filling.
    """
    best_idx = 0
    best_key = (-1, -1)

    for i, dims in enumerate(item_dims_list):
        base = dims.length * dims.width
        x_occ = dims.length
        key = (-base, -x_occ)  # max base, max x-occ (both negative for sort)
        if key < best_key:
            best_key = key
            best_idx = i

    return best_idx


def orientation_rule_4(item_dims_list: List[Dimensions], ems: EMS,
                       all_ems: List[EMS] = None, n_ems: int = 3) -> int:
    """
    Rule 4: Among n EMSs, find the EMS-orientation pair with highest fill ratio.

    For each of the top n EMSs:
        For each orientation: compute fill ratio = item_volume / ems_volume
        Keep the pair with minimum margin (tightest fit)
    Among all n pairs: select the one with highest fill ratio.

    More expensive but potentially better matches.
    """
    if all_ems is None or len(all_ems) == 0:
        all_ems = [ems]

    best_idx = 0
    best_fill = -1.0

    for candidate_ems in all_ems[:n_ems]:
        for i, dims in enumerate(item_dims_list):
            if not fits_in_ems(dims, candidate_ems):
                continue
            fill = dims.volume / max(candidate_ems.volume, 1e-9)
            if fill > best_fill:
                best_fill = fill
                best_idx = i

    return best_idx


def orientation_rule_5(item_dims_list: List[Dimensions], ems: EMS,
                       all_ems: List[EMS] = None,
                       bin_dims: Dimensions = None) -> int:
    """
    Rule 5: Max distance from item's front-top-right corner to bin's FTR corner.
    Considers n EMSs.

    Pushes the item's far corner as deep, low, and left as possible.
    """
    if bin_dims is None:
        L, W, H = 1200, 800, 1500
    else:
        L, W, H = bin_dims.length, bin_dims.width, bin_dims.height

    if all_ems is None or len(all_ems) == 0:
        all_ems = [ems]

    best_idx = 0
    best_dist = -1.0

    for candidate_ems in all_ems[:3]:
        for i, dims in enumerate(item_dims_list):
            if not fits_in_ems(dims, candidate_ems):
                continue
            # Item FTR corner after placement at EMS minimum vertex
            item_ftr_x = candidate_ems.x1 + dims.length
            item_ftr_y = candidate_ems.y1 + dims.width
            item_ftr_z = candidate_ems.z1 + dims.height
            # Distance from item FTR to bin FTR
            dist = math.sqrt((L - item_ftr_x)**2 + (W - item_ftr_y)**2 + (H - item_ftr_z)**2)
            if dist > best_dist:
                best_dist = dist
                best_idx = i

    return best_idx


ORIENTATION_RULE_MAP = {
    1: orientation_rule_1,
    2: orientation_rule_2,
    3: orientation_rule_3,
    4: orientation_rule_4,
    5: orientation_rule_5,
}


# ============================================================================
# EMS MANAGER -- "Difference and Elimination" from Lai & Chan (1997)
# ============================================================================

class EMSManager:
    """
    Manages Empty Maximal Spaces for a single bin.

    The "difference and elimination" process:
    1. When an item is placed, find all EMSs that intersect with it
    2. For each intersecting EMS, generate up to 6 new EMSs by cutting
       along the 6 faces of the placed item
    3. Remove the original intersecting EMS
    4. Eliminate any new EMS that is fully contained within another
    5. Remove zero-volume or sub-threshold-dimension EMSs
    6. Apply blocking filter (EMSs not reachable from entrance)

    This is THE critical data structure for the entire packing system.
    """

    def __init__(self, bin_dims: Dimensions, min_dim: float = 1.0):
        self.bin_dims = bin_dims
        self.min_dim = min_dim
        self._creation_counter = 0
        # Initial EMS = entire bin
        initial_ems = EMS(0, 0, 0,
                          bin_dims.length, bin_dims.width, bin_dims.height,
                          creation_order=self._next_order())
        self.ems_list: List[EMS] = [initial_ems]

    def _next_order(self) -> int:
        self._creation_counter += 1
        return self._creation_counter

    def place_item(self, position: Position, dims: Dimensions):
        """
        Update EMSs after placing an item.

        For each EMS intersecting the item:
            Generate up to 6 sub-EMSs by cutting along item faces
            Remove original EMS
        Then eliminate contained EMSs.
        """
        new_ems_list = []
        item_x1, item_y1, item_z1 = position.x, position.y, position.z
        item_x2 = item_x1 + dims.length
        item_y2 = item_y1 + dims.width
        item_z2 = item_z1 + dims.height

        for ems in self.ems_list:
            if not self._intersects(ems, item_x1, item_y1, item_z1,
                                     item_x2, item_y2, item_z2):
                new_ems_list.append(ems)
                continue

            # Generate up to 6 new EMSs by cutting
            cuts = self._generate_cuts(ems, item_x1, item_y1, item_z1,
                                        item_x2, item_y2, item_z2)
            for cut in cuts:
                if cut.is_valid(self.min_dim):
                    cut.creation_order = self._next_order()
                    new_ems_list.append(cut)

        # Eliminate contained EMSs
        self.ems_list = self._eliminate_contained(new_ems_list)

    def _intersects(self, ems: EMS, x1, y1, z1, x2, y2, z2) -> bool:
        """Check if EMS intersects with axis-aligned box [x1,x2]x[y1,y2]x[z1,z2]."""
        return not (x1 >= ems.x2 or x2 <= ems.x1 or
                    y1 >= ems.y2 or y2 <= ems.y1 or
                    z1 >= ems.z2 or z2 <= ems.z1)

    def _generate_cuts(self, ems: EMS, ix1, iy1, iz1, ix2, iy2, iz2) -> List[EMS]:
        """
        Generate up to 6 sub-EMSs by cutting the EMS along the 6 faces
        of the placed item.
        """
        cuts = []

        # Cut 1: x < item_x1 (space behind the item)
        if ix1 > ems.x1:
            cuts.append(EMS(ems.x1, ems.y1, ems.z1, ix1, ems.y2, ems.z2))

        # Cut 2: x > item_x2 (space in front of the item)
        if ix2 < ems.x2:
            cuts.append(EMS(ix2, ems.y1, ems.z1, ems.x2, ems.y2, ems.z2))

        # Cut 3: y < item_y1 (space to the left of the item)
        if iy1 > ems.y1:
            cuts.append(EMS(ems.x1, ems.y1, ems.z1, ems.x2, iy1, ems.z2))

        # Cut 4: y > item_y2 (space to the right of the item)
        if iy2 < ems.y2:
            cuts.append(EMS(ems.x1, iy2, ems.z1, ems.x2, ems.y2, ems.z2))

        # Cut 5: z < item_z1 (space below the item)
        if iz1 > ems.z1:
            cuts.append(EMS(ems.x1, ems.y1, ems.z1, ems.x2, ems.y2, iz1))

        # Cut 6: z > item_z2 (space above the item)
        if iz2 < ems.z2:
            cuts.append(EMS(ems.x1, ems.y1, iz2, ems.x2, ems.y2, ems.z2))

        return cuts

    def _eliminate_contained(self, ems_list: List[EMS]) -> List[EMS]:
        """Remove EMSs that are fully contained within another EMS."""
        if len(ems_list) <= 1:
            return ems_list

        result = []
        for i, ems_a in enumerate(ems_list):
            contained = False
            for j, ems_b in enumerate(ems_list):
                if i != j and ems_b.contains(ems_a):
                    contained = True
                    break
            if not contained:
                result.append(ems_a)
        return result

    def get_sorted_ems(self, rule_id: int) -> List[EMS]:
        """Return EMSs sorted according to the specified space selection rule."""
        key_func = SPACE_RULE_MAP.get(rule_id, space_rule_1_key)
        return sorted(self.ems_list, key=lambda e: key_func(e, self.bin_dims))

    def filter_blocked(self, placed_items: List[PlacedItem]):
        """
        Remove EMSs blocked by the entrance constraint.

        An EMS is blocked if it is positioned behind a packed item
        such that loading an item there would require moving through
        an already-placed item. The blocking check verifies x-values.
        """
        # Implementation depends on entrance position (assumed at x=L, front)
        # An EMS at (x1, y1, z1) is blocked if there exists a placed item
        # between x1 and the entrance (x=L) that physically obstructs access.
        # Simplified: check if any placed item has x <= x_ems and
        # x + l > x_ems and overlaps in y and z ranges.
        # Full implementation requires careful geometric analysis.
        pass


# ============================================================================
# THE 9-STEP HEURISTIC ENGINE
# ============================================================================

class HeuristicEngine:
    """
    The master heuristic engine implementing the 9-step procedure
    from Appendix A of Ali et al. (2025).

    Configurable with:
    - bin_selection_rule: F, B, W, or A
    - space_selection_rule: 1-8
    - orientation_selection_rule: 1-5
    - stability_checker: any StabilityChecker instance
    """

    def __init__(self, bin_rule: str, space_rule: int, orient_rule: int,
                 stability_checker=None, bin_dims: Dimensions = None):
        self.bin_rule = BinSelectionRule(bin_rule)
        self.space_rule = space_rule
        self.orient_rule = orient_rule
        self.stability_checker = stability_checker
        self.bin_dims = bin_dims or Dimensions(1200, 800, 1500)
        self.name = f"{bin_rule}{space_rule}{orient_rule}"

    def pack(self, items: list, allowed_orientations: dict = None) -> list:
        """
        Pack all items using this heuristic.

        Args:
            items: list of item dicts with 'length', 'width', 'height', 'id'
            allowed_orientations: dict mapping item_id to list of allowed
                                  orientation indices (0-5). Default: all 6.

        Returns:
            list of bin states with placed items
        """
        bins = []  # list of (EMSManager, list[PlacedItem])
        bin_selector = BIN_SELECTOR_MAP[self.bin_rule]

        for item_data in items:
            item_dims = Dimensions(item_data['length'],
                                    item_data['width'],
                                    item_data['height'])
            item_id = item_data.get('id', 0)
            orient_set = (allowed_orientations or {}).get(item_id, range(6))

            placed = self._try_place_in_existing_bins(
                item_id, item_dims, orient_set, bins, bin_selector)

            if not placed:
                # Open new bin
                new_ems_mgr = EMSManager(self.bin_dims)
                new_bin = (new_ems_mgr, [])
                bins.append(new_bin)
                self._try_place_in_bin(item_id, item_dims, orient_set, new_bin)

        return bins

    def _try_place_in_existing_bins(self, item_id, item_dims, orient_set,
                                      bins, bin_selector) -> bool:
        """Try to place item in existing bins using bin selection rule."""
        if not bins:
            return False

        candidate_bins = bin_selector(bins)

        for bin_data in candidate_bins:
            if self._try_place_in_bin(item_id, item_dims, orient_set, bin_data):
                return True
        return False

    def _try_place_in_bin(self, item_id, item_dims, orient_set,
                           bin_data) -> bool:
        """
        Try to place item in a specific bin.
        Steps 2-8 of the 9-step procedure.
        """
        ems_mgr, placed_items = bin_data

        # Step 2: Get sorted EMS list
        sorted_ems = ems_mgr.get_sorted_ems(self.space_rule)

        for ems in sorted_ems:
            # Step 3: Find fitting orientations
            fitting_orientations = []
            for o in orient_set:
                oriented = get_oriented_dims(item_dims, o)
                if fits_in_ems(oriented, ems):
                    fitting_orientations.append((o, oriented))

            if not fitting_orientations:
                continue

            # Step 4: Filter by stability
            position = Position(ems.x1, ems.y1, ems.z1)
            stable_orientations = []
            for o, dims in fitting_orientations:
                if self.stability_checker is None:
                    stable_orientations.append((o, dims))
                else:
                    is_stable = self.stability_checker.is_stable(
                        item_dims, position, o, placed_items)
                    if is_stable:
                        stable_orientations.append((o, dims))

            if not stable_orientations:
                continue

            # Step 5: Select best orientation
            orient_func = ORIENTATION_RULE_MAP[self.orient_rule]
            dims_list = [d for _, d in stable_orientations]
            best_idx = orient_func(dims_list, ems)
            best_orient, best_dims = stable_orientations[best_idx]

            # Step 6: Place item
            placed = PlacedItem(item_id, position, best_dims, item_dims, best_orient)
            placed_items.append(placed)

            # Step 7: Update EMSs
            ems_mgr.place_item(position, best_dims)

            # Step 8: (EMS reordering happens implicitly in get_sorted_ems)

            return True

        return False


# ============================================================================
# HEURISTIC REGISTRY -- ALL 160 COMBINATIONS
# ============================================================================

def generate_all_heuristic_names() -> List[str]:
    """Generate all 160 heuristic names as [B][S][O] strings."""
    names = []
    for b, s, o in product(['F', 'B', 'W', 'A'], range(1, 9), range(1, 6)):
        names.append(f"{b}{s}{o}")
    assert len(names) == 160
    return names


def build_heuristic(name: str, stability_checker=None,
                     bin_dims: Dimensions = None) -> HeuristicEngine:
    """Build a heuristic from its 3-character name (e.g., 'F53')."""
    bin_rule = name[0]
    space_rule = int(name[1])
    orient_rule = int(name[2])
    return HeuristicEngine(bin_rule, space_rule, orient_rule,
                            stability_checker, bin_dims)


# ============================================================================
# TOP HEURISTIC COMBINATIONS (from paper's Pareto analysis)
# ============================================================================

# Non-dominated heuristics under partial-base polygon support (16 total):
PARETO_PARTIAL_BASE_POLYGON = [
    'A12', 'F12', 'B12', 'W53', 'B52', 'F51', 'A63', 'F63',
    'B63', 'A53', 'F53', 'B53', 'A52', 'F52', 'W63', 'W73'
]

# Non-dominated heuristics under CoG polygon support (17 total):
PARETO_COG_POLYGON = [
    'A12', 'F12', 'B12', 'W53', 'A63', 'F63', 'B63', 'W63',
    'W73', 'A53', 'F53', 'B53', 'A52', 'F52', 'A11', 'F11', 'B52'
]

# Non-dominated under full-base support (just 2):
PARETO_FULL_BASE = ['F52', 'A52']

# Non-dominated under partial-base support (just 1):
PARETO_PARTIAL_BASE = ['F51']

# Overall top performers for minimum bins (across all polygon constraints):
TOP_EFFICIENCY = ['A12', 'F12', 'B12', 'F53', 'A53']

# Overall top performers for maximum stability:
TOP_STABILITY = ['F52', 'A52', 'F51']

# Recommended defaults for our k=2 semi-online system:
# A52 or A53 -- "All bins" has zero overhead with k=2.
RECOMMENDED_DEFAULT = 'A53'


# ============================================================================
# BENCHMARKING HELPER
# ============================================================================

def run_full_benchmark(instances: list, stability_checkers: dict,
                        bin_dims: Dimensions = None) -> dict:
    """
    Run all 160 heuristics x all stability constraints on all instances.

    This replicates the paper's experimental setup:
    160 heuristics x 198 instances x 4 constraints = 126,720 runs.

    Args:
        instances: list of instance dicts (items, allowed_orientations)
        stability_checkers: dict mapping constraint_name to checker
        bin_dims: container dimensions

    Returns:
        dict mapping (constraint, heuristic, instance) to results
    """
    results = {}
    all_names = generate_all_heuristic_names()

    for constraint_name, checker in stability_checkers.items():
        for name in all_names:
            h = build_heuristic(name, checker, bin_dims)
            for inst in instances:
                bins = h.pack(inst['items'], inst.get('orientations'))
                num_bins = len(bins)
                total_items = sum(len(b[1]) for b in bins)
                results[(constraint_name, name, inst['name'])] = {
                    'num_bins': num_bins,
                    'total_items': total_items,
                    'bins': bins,
                }

    return results


# ============================================================================
# IMPLEMENTATION NOTES (UPDATED)
# ============================================================================
#
# 1. The 160-heuristic framework is primarily useful for BENCHMARKING
#    and ALGORITHM SELECTION, not for runtime use. In production, select
#    the top 3-5 heuristics and use those.
#
# 2. For the thesis: implement all 160 heuristics, download the 198
#    Mendeley instances, and REPRODUCE the paper's Table 5 and Table 8.
#    This validates implementation before adding buffer/bounded-space.
#
# 3. The paper's naming convention (e.g., "F13") is maintained in code.
#
# 4. All 160 heuristics are a Cartesian product:
#    from itertools import product
#    all_heuristics = list(product(['F','B','W','A'], range(1,9), range(1,6)))
#    assert len(all_heuristics) == 160
#
# 5. For k=2 bounded space: "All bins" (A) checks just 2 bins,
#    so computational overhead vs F or B is negligible. Use A always.
#
# 6. Key insight from Pareto analysis: under polygon-based constraints,
#    there are 16-17 non-dominated heuristics, while under base-support
#    constraints there are only 1-2. This means polygon constraints
#    unlock diversity in the heuristic space.
#
# 7. ESTIMATED IMPLEMENTATION TIME:
#    - Data structures (Bin, Item, EMS): 2 days
#    - EMS manager (difference & elimination): 2 days
#    - Selection rules (4+8+5): 2 days
#    - Main loop (9-step procedure): 1 day
#    - Testing & validation vs paper results: 2 days
#    Total: ~9 days
#
# ============================================================================
"""
"""
