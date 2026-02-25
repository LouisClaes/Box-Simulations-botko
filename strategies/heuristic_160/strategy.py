"""
160-Heuristic Framework for 3D Bin Packing (Ali et al. 2024/2025).

This strategy implements the combinatorial heuristic framework described in:

    Ali, I., Ramos, A.G., Carravilla, M.A., Oliveira, J.F. (2024).
    "A matheuristic for the online 3D bin packing problem in the
    slaughterhouse industry."
    Applied Soft Computing, Vol. 151, Article 111168.
    https://doi.org/10.1016/j.asoc.2023.111168

    Ali, I., Ramos, A.G., Oliveira, J.F. (2025).
    "A 2-phase matheuristic for the online 3D bin packing problem."
    Computers & Operations Research, Vol. 178, Article 107005.
    https://doi.org/10.1016/j.cor.2025.107005

Framework overview:
    The paper enumerates 160 heuristics as:
        4 bin-selection rules x 8 space-selection rules x 5 orientation-
        selection rules = 160 combinations.

    Since we operate in a single-bin context, bin selection is trivially
    "this bin", so the relevant combinatorial space is:
        8 space-selection rules x 5 orientation-selection rules = 40
    heuristics for each incoming box.

Core concept -- Empty Maximal Spaces (EMS):
    An EMS is an axis-aligned rectangular volume (x1,y1,z1)-(x2,y2,z2)
    that is completely free of placed boxes.  After each box placement, the
    EMS that contained that box is split into up to 6 sub-EMSs (one per
    face of the placed box that does not reach the EMS boundary). Dominated
    EMSs (those fully contained in another) are discarded.

    At each decide_placement() call, the EMS list is rebuilt from scratch
    using bin_state.placed_boxes.  This is O(N) per call but is reliable
    and avoids drift that accumulates with incremental tracking.

Default heuristic: "A53"
    Space rule  5 (DBLF + corner preference)
    Orient rule 3 (largest base, then greatest x-extent)

    This combination was empirically strong across the Ali et al. datasets
    and is a good default for EUR-pallet configurations.

Space-selection rules (8):
    Rule 1 (DBLF)        -- (min_x, min_z, min_y) -- leftmost-lowest-front
    Rule 2               -- (min_z, min_x, min_y) -- lowest first
    Rule 3               -- (min_z, min_y, min_x) -- lowest-front-left
    Rule 4               -- ascending EMS volume  -- smallest space first
    Rule 5 (DBLF+corner) -- Rule 1, but wall/box-adjacent EMSs rank higher
    Rule 6               -- (max_x, min_z, min_y) -- rightmost first
    Rule 7               -- min_z, then raw position priority
    Rule 8 (DFTRC)       -- (-x2, -y2, -z2)       -- deep/front/top-right corner

Orientation-selection rules (5):
    Rule 1 (min margin)     -- minimise wasted EMS volume
    Rule 2 (largest base)   -- largest horizontal footprint
    Rule 3 (base + max x)   -- largest footprint, then greatest x-extent
    Rule 4 (fill ratio)     -- best box/EMS volume ratio
    Rule 5 (height-base)    -- shortest height with largest base
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import Box, ExperimentConfig, Orientation, PlacementDecision
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Anti-float threshold: must match the simulator's MIN_ANTI_FLOAT_RATIO (0.30).
MIN_SUPPORT: float = 0.30

# When checking support, placements at or below this height are considered
# floor-level and always valid (avoids spurious rejections at z ~ 0).
FLOOR_Z_THRESHOLD: float = 0.5

# Numerical tolerance used in overlap / containment checks.
EPS: float = 1e-6

# Minimum dimension (cm) for an EMS to be kept.  Removes degenerate slivers.
MIN_EMS_DIM: float = 1.0


# ---------------------------------------------------------------------------
# EMS data structure
# ---------------------------------------------------------------------------

@dataclass
class EMS:
    """
    An Empty Maximal Space -- a 3-D axis-aligned free-space region.

    Defined by two corner points:
        lower-back-left:  (x1, y1, z1)
        upper-front-right: (x2, y2, z2)

    Attributes:
        x1, y1, z1: Lower corner (origin) of the free space.
        x2, y2, z2: Upper corner of the free space.
    """

    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def length(self) -> float:
        """Extent along the x-axis."""
        return self.x2 - self.x1

    @property
    def width(self) -> float:
        """Extent along the y-axis."""
        return self.y2 - self.y1

    @property
    def height(self) -> float:
        """Extent along the z-axis."""
        return self.z2 - self.z1

    @property
    def volume(self) -> float:
        """Volume of this EMS."""
        return self.length * self.width * self.height

    def fits_box(self, ol: float, ow: float, oh: float) -> bool:
        """Return True if a box of oriented dims (ol, ow, oh) fits inside."""
        return (
            ol <= self.length + EPS
            and ow <= self.width + EPS
            and oh <= self.height + EPS
        )

    def is_valid(self) -> bool:
        """Return True if all three dimensions are at least MIN_EMS_DIM."""
        return (
            self.length >= MIN_EMS_DIM
            and self.width >= MIN_EMS_DIM
            and self.height >= MIN_EMS_DIM
        )

    def contains(self, other: "EMS") -> bool:
        """
        Return True if *other* is fully contained within *self*.
        Used to detect dominated EMSs.
        """
        return (
            self.x1 <= other.x1 + EPS
            and self.y1 <= other.y1 + EPS
            and self.z1 <= other.z1 + EPS
            and self.x2 >= other.x2 - EPS
            and self.y2 >= other.y2 - EPS
            and self.z2 >= other.z2 - EPS
        )

    def __repr__(self) -> str:
        return (
            f"EMS(({self.x1:.0f},{self.y1:.0f},{self.z1:.0f})"
            f"-({self.x2:.0f},{self.y2:.0f},{self.z2:.0f}) "
            f"vol={self.volume:.0f})"
        )


# ---------------------------------------------------------------------------
# EMS geometry helpers
# ---------------------------------------------------------------------------

def _box_overlaps_ems(p_x1: float, p_y1: float, p_z1: float,
                       p_x2: float, p_y2: float, p_z2: float,
                       e: EMS) -> bool:
    """
    Return True if the placed box (p_x1..p_x2, p_y1..p_y2, p_z1..p_z2)
    overlaps with EMS *e* in all three axes.
    """
    return (
        p_x1 < e.x2 - EPS and p_x2 > e.x1 + EPS
        and p_y1 < e.y2 - EPS and p_y2 > e.y1 + EPS
        and p_z1 < e.z2 - EPS and p_z2 > e.z1 + EPS
    )


def _split_ems(
    p_x1: float, p_y1: float, p_z1: float,
    p_x2: float, p_y2: float, p_z2: float,
    e: EMS,
) -> List[EMS]:
    """
    Split EMS *e* around the placed box (p_x1..p_x2, p_y1..p_y2, p_z1..p_z2)
    using the Lai & Chan (1997) "difference and elimination" rule.

    Up to 6 sub-EMSs are generated, one for each face of the placed box
    that does not coincide with the corresponding boundary of *e*:

        Right  (along +x): from bx+bl to e.x2
        Left   (along -x): from e.x1  to bx
        Front  (along +y): from by+bw to e.y2
        Back   (along -y): from e.y1  to by
        Top    (along +z): from bz+bh to e.z2
        Bottom (along -z): impossible (boxes always rest on surfaces,
                           so p_z1 >= e.z1 is guaranteed)

    Each candidate is kept only if all three dimensions are >= MIN_EMS_DIM.

    Args:
        p_x1..p_z2: Bounding box of the placed item.
        e:          The EMS to split.

    Returns:
        List of valid sub-EMSs (may be empty).
    """
    result: List[EMS] = []

    # Right sub-EMS: everything to the right of the placed box
    if p_x2 < e.x2 - EPS:
        candidate = EMS(p_x2, e.y1, e.z1, e.x2, e.y2, e.z2)
        if candidate.is_valid():
            result.append(candidate)

    # Left sub-EMS: everything to the left of the placed box
    if e.x1 < p_x1 - EPS:
        candidate = EMS(e.x1, e.y1, e.z1, p_x1, e.y2, e.z2)
        if candidate.is_valid():
            result.append(candidate)

    # Front sub-EMS: everything in front of the placed box (+y direction)
    if p_y2 < e.y2 - EPS:
        candidate = EMS(e.x1, p_y2, e.z1, e.x2, e.y2, e.z2)
        if candidate.is_valid():
            result.append(candidate)

    # Back sub-EMS: everything behind the placed box (-y direction)
    if e.y1 < p_y1 - EPS:
        candidate = EMS(e.x1, e.y1, e.z1, e.x2, p_y1, e.z2)
        if candidate.is_valid():
            result.append(candidate)

    # Top sub-EMS: everything above the placed box
    if p_z2 < e.z2 - EPS:
        candidate = EMS(e.x1, e.y1, p_z2, e.x2, e.y2, e.z2)
        if candidate.is_valid():
            result.append(candidate)

    # Bottom sub-EMS: physically impossible for properly supported boxes.
    # A box always rests on the highest point of the footprint, so p_z1
    # should equal the surface height at that position (>= e.z1).
    # We skip this case entirely to avoid generating phantom spaces.

    return result


def _eliminate_dominated(ems_list: List[EMS]) -> List[EMS]:
    """
    Remove dominated EMSs: keep only those not fully contained in another.

    An EMS A is dominated if there exists another EMS B in the list such
    that B.contains(A).  O(N^2) but N is typically small (< 100).

    Args:
        ems_list: Candidate EMS list (may contain dominated entries).

    Returns:
        Filtered list with all dominated EMSs removed.
    """
    n = len(ems_list)
    if n <= 1:
        return ems_list

    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            # If j fully contains i, mark i as dominated
            if ems_list[j].contains(ems_list[i]):
                keep[i] = False
                break

    return [ems_list[i] for i in range(n) if keep[i]]


# ---------------------------------------------------------------------------
# EMS recomputation from placed boxes
# ---------------------------------------------------------------------------

def _recompute_ems(bin_state: BinState) -> List[EMS]:
    """
    Rebuild the EMS list from scratch using the placed boxes in *bin_state*.

    Algorithm (Lai & Chan 1997 / Ali et al. 2024):
        1. Start with one EMS covering the entire bin.
        2. For each placed box, iterate over all current EMSs:
             - If the box overlaps an EMS, split that EMS into sub-EMSs.
             - Otherwise, keep the EMS unchanged.
        3. After processing all boxes, remove dominated EMSs.

    This is O(N * M) where N = number of placed boxes, M = EMS list size.
    M typically stays small (< 3*N in practice).

    Args:
        bin_state: Current bin state (read-only).

    Returns:
        List of valid, non-dominated EMSs.
    """
    cfg = bin_state.config
    # Initial EMS: the entire bin volume
    ems_list: List[EMS] = [EMS(0.0, 0.0, 0.0, cfg.length, cfg.width, cfg.height)]

    for p in bin_state.placed_boxes:
        # Bounding box of the placed item
        px1, py1, pz1 = p.x, p.y, p.z
        px2 = px1 + p.oriented_l
        py2 = py1 + p.oriented_w
        pz2 = pz1 + p.oriented_h

        new_ems: List[EMS] = []
        for e in ems_list:
            if _box_overlaps_ems(px1, py1, pz1, px2, py2, pz2, e):
                new_ems.extend(_split_ems(px1, py1, pz1, px2, py2, pz2, e))
            else:
                new_ems.append(e)

        ems_list = _eliminate_dominated(new_ems)

    return ems_list


# ---------------------------------------------------------------------------
# Space-selection rules (sort key factories)
# ---------------------------------------------------------------------------

def _sort_key_rule1(e: EMS) -> Tuple:
    """Rule 1 (DBLF): leftmost, then lowest, then frontmost."""
    return (e.x1, e.z1, e.y1)


def _sort_key_rule2(e: EMS) -> Tuple:
    """Rule 2: lowest first, then leftmost, then frontmost."""
    return (e.z1, e.x1, e.y1)


def _sort_key_rule3(e: EMS) -> Tuple:
    """Rule 3: lowest, then frontmost, then leftmost."""
    return (e.z1, e.y1, e.x1)


def _sort_key_rule4(e: EMS) -> Tuple:
    """Rule 4: smallest EMS volume first (tightest fit)."""
    return (e.volume, e.x1, e.y1, e.z1)


def _sort_key_rule5(e: EMS, bin_length: float, bin_width: float) -> Tuple:
    """
    Rule 5 (DBLF + corner preference): Rule 1 ordering but wall-adjacent
    (corner) EMSs are ranked higher (lower sort key = earlier selection).

    An EMS is a "corner EMS" if at least one of its origin coordinates
    is 0 (touching a wall).  We encode this as a Boolean flag: 0 = corner,
    1 = interior.  This is prepended before the DBLF key so corner EMSs
    are always tried first within the same DBLF tier.
    """
    is_corner = not (
        abs(e.x1) < EPS or abs(e.y1) < EPS or abs(e.z1) < EPS
        or abs(e.x1 - bin_length) < EPS or abs(e.y1 - bin_width) < EPS
    )
    return (int(is_corner), e.x1, e.z1, e.y1)


def _sort_key_rule6(e: EMS) -> Tuple:
    """Rule 6: rightmost first (highest x2), then lowest, then frontmost."""
    return (-e.x2, e.z1, e.y1)


def _sort_key_rule7(e: EMS) -> Tuple:
    """Rule 7: lowest z1 only, then raw position."""
    return (e.z1, e.x1, e.y1, e.z1)


def _sort_key_rule8(e: EMS) -> Tuple:
    """Rule 8 (DFTRC): prefer deep-front-top-right corner -- descending x2,y2,z2."""
    return (-e.x2, -e.y2, -e.z2)


def _apply_space_rule(
    ems_list: List[EMS],
    rule: int,
    bin_length: float,
    bin_width: float,
) -> List[EMS]:
    """
    Sort *ems_list* in-place (ascending key = first to try) according to
    the specified space-selection rule (1-8).

    Args:
        ems_list:   List of EMS objects to sort.
        rule:       Space-selection rule index (1..8).
        bin_length: Bin length, required for rule 5.
        bin_width:  Bin width, required for rule 5.

    Returns:
        Sorted copy of *ems_list*.
    """
    if rule == 1:
        return sorted(ems_list, key=_sort_key_rule1)
    elif rule == 2:
        return sorted(ems_list, key=_sort_key_rule2)
    elif rule == 3:
        return sorted(ems_list, key=_sort_key_rule3)
    elif rule == 4:
        return sorted(ems_list, key=_sort_key_rule4)
    elif rule == 5:
        return sorted(
            ems_list,
            key=lambda e: _sort_key_rule5(e, bin_length, bin_width),
        )
    elif rule == 6:
        return sorted(ems_list, key=_sort_key_rule6)
    elif rule == 7:
        return sorted(ems_list, key=_sort_key_rule7)
    elif rule == 8:
        return sorted(ems_list, key=_sort_key_rule8)
    else:
        raise ValueError(f"Space rule must be 1-8, got {rule}")


# ---------------------------------------------------------------------------
# Orientation-selection rules (sort key factories)
# ---------------------------------------------------------------------------

def _orient_key_rule1(
    ol: float, ow: float, oh: float, e: EMS
) -> float:
    """
    Rule 1 (min margin): minimise wasted EMS volume after placing the box.
    waste = ems.volume - box.volume  (lower = better, so we negate for sort).

    We use -box_volume as a proxy for -waste when comparing orientations
    within the same EMS (ems.volume is constant per EMS).
    """
    # More negative = better (larger box volume uses more of the EMS)
    return -(ol * ow * oh)


def _orient_key_rule2(ol: float, ow: float, oh: float) -> Tuple:
    """Rule 2 (largest base): sort descending by base area (ol * ow)."""
    return (-(ol * ow),)


def _orient_key_rule3(ol: float, ow: float, oh: float) -> Tuple:
    """Rule 3 (base + max x): largest base, then greatest x-extent."""
    return (-(ol * ow), -ol)


def _orient_key_rule4(ol: float, ow: float, oh: float, e: EMS) -> float:
    """
    Rule 4 (fill ratio): best box/EMS volume ratio (descending).
    ratio = (ol*ow*oh) / (e.length * e.width * e.height)
    """
    ems_vol = e.volume
    if ems_vol < EPS:
        return 0.0
    return -(ol * ow * oh) / ems_vol


def _orient_key_rule5(ol: float, ow: float, oh: float) -> Tuple:
    """Rule 5 (height-base): ascending height, then descending base area."""
    return (oh, -(ol * ow))


def _rank_orientations(
    orientations: List[Tuple[float, float, float]],
    e: EMS,
    rule: int,
) -> List[Tuple[int, float, float, float]]:
    """
    Return orientations sorted by the given orientation-selection rule.

    Args:
        orientations: List of (ol, ow, oh) tuples from Orientation.get_*().
        e:            The EMS the box will be placed into (needed for
                      volume-dependent rules 1 and 4).
        rule:         Orientation rule index (1..5).

    Returns:
        List of (original_index, ol, ow, oh) sorted best-first.
    """
    if rule == 1:
        indexed = sorted(
            enumerate(orientations),
            key=lambda t: _orient_key_rule1(t[1][0], t[1][1], t[1][2], e),
        )
    elif rule == 2:
        indexed = sorted(
            enumerate(orientations),
            key=lambda t: _orient_key_rule2(t[1][0], t[1][1], t[1][2]),
        )
    elif rule == 3:
        indexed = sorted(
            enumerate(orientations),
            key=lambda t: _orient_key_rule3(t[1][0], t[1][1], t[1][2]),
        )
    elif rule == 4:
        indexed = sorted(
            enumerate(orientations),
            key=lambda t: _orient_key_rule4(t[1][0], t[1][1], t[1][2], e),
        )
    elif rule == 5:
        indexed = sorted(
            enumerate(orientations),
            key=lambda t: _orient_key_rule5(t[1][0], t[1][1], t[1][2]),
        )
    else:
        raise ValueError(f"Orientation rule must be 1-5, got {rule}")

    return [(oidx, ol, ow, oh) for oidx, (ol, ow, oh) in indexed]


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

@register_strategy
class Heuristic160Strategy(BaseStrategy):
    """
    160-Heuristic Framework for 3D Bin Packing (Ali et al. 2024/2025).

    Combines one of 8 space-selection rules with one of 5 orientation-
    selection rules to produce a single placement decision per call.  The
    default configuration "A53" (space rule 5 / orient rule 3) uses
    DBLF+corner ordering with largest-base-then-greatest-x orientation.

    EMS list management:
        The EMS list is recomputed from scratch on every call using the
        canonical Lai & Chan split-and-eliminate procedure.  This is safe,
        stateless (no drift), and compatible with the strategy interface
        which does not provide placement callbacks.

    Stability:
        MIN_SUPPORT = 0.30 is always enforced.  When ExperimentConfig
        enables stricter stability (enable_stability=True), the configurable
        min_support_ratio is applied instead.

    Attributes:
        name:        Strategy identifier for the registry ("heuristic_160").
        space_rule:  Space-selection rule index (1-8). Default 5.
        orient_rule: Orientation-selection rule index (1-5). Default 3.
    """

    name: str = "heuristic_160"

    def __init__(
        self,
        space_rule: int = 5,
        orient_rule: int = 3,
    ) -> None:
        """
        Initialise the strategy.

        Args:
            space_rule:  Which space-selection rule to apply (1-8).
                         Default 5 = DBLF + corner preference.
            orient_rule: Which orientation-selection rule to apply (1-5).
                         Default 3 = largest base + greatest x-extent.
        """
        super().__init__()
        if not 1 <= space_rule <= 8:
            raise ValueError(f"space_rule must be 1-8, got {space_rule}")
        if not 1 <= orient_rule <= 5:
            raise ValueError(f"orient_rule must be 1-5, got {orient_rule}")
        self.space_rule: int = space_rule
        self.orient_rule: int = orient_rule

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Propose a placement for *box* using the 160-Heuristic Framework.

        Algorithm:
            1. Rebuild the EMS list from bin_state.placed_boxes.
            2. Sort EMSs by the configured space-selection rule.
            3. For each EMS (in sorted order):
               a. Rank allowed orientations by the configured orient rule.
               b. For each orientation (best first):
                  - Check that the oriented box fits inside the EMS.
                  - Compute the actual resting z via get_height_at().
                  - Validate: bounds, height limit, support ratio.
                  - If valid, return immediately (first-valid-fit approach).
            4. If no EMS yields a valid placement, fall back to a coarse
               bottom-left-fill grid scan as a last resort.
            5. Return None if even the fallback fails.

        The first-valid-fit approach is consistent with the Ali et al.
        framework: the space/orientation rules determine priority, and the
        first feasible combination found is used.

        Args:
            box:       The box to place (original dimensions, pre-rotation).
            bin_state: Current 3D bin state (read-only).

        Returns:
            PlacementDecision(x, y, orientation_idx) or None.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        # ------------------------------------------------------------------
        # 1. Resolve allowed orientations
        # ------------------------------------------------------------------
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Quick pre-check: can any orientation fit in the bin at all?
        feasible_orientations = [
            (ol, ow, oh)
            for (ol, ow, oh) in orientations
            if ol <= bin_cfg.length and ow <= bin_cfg.width and oh <= bin_cfg.height
        ]
        if not feasible_orientations:
            return None

        # ------------------------------------------------------------------
        # 2. Rebuild EMS list from placed boxes
        # ------------------------------------------------------------------
        ems_list = _recompute_ems(bin_state)

        # ------------------------------------------------------------------
        # 3. Sort EMSs by space-selection rule
        # ------------------------------------------------------------------
        sorted_ems = _apply_space_rule(
            ems_list, self.space_rule, bin_cfg.length, bin_cfg.width
        )

        # ------------------------------------------------------------------
        # 4. Try each EMS with orientation ranking
        # ------------------------------------------------------------------
        for e in sorted_ems:
            # Rank orientations for this specific EMS
            ranked = _rank_orientations(feasible_orientations, e, self.orient_rule)

            for oidx_local, ol, ow, oh in ranked:
                # Skip if box doesn't fit in this EMS geometrically
                if not e.fits_box(ol, ow, oh):
                    continue

                # Placement position: lower-left-back corner of the EMS
                px, py = e.x1, e.y1

                # Strict bounds check against bin dimensions
                if px + ol > bin_cfg.length + EPS:
                    continue
                if py + ow > bin_cfg.width + EPS:
                    continue

                # Actual resting z from the heightmap (may exceed e.z1
                # if the EMS was generated from placed-box coordinates
                # that slightly overhang the EMS lower boundary)
                z = bin_state.get_height_at(px, py, ol, ow)

                # Height limit check
                if z + oh > bin_cfg.height + EPS:
                    continue

                # ----------------------------------------------------------
                # Support / anti-float check
                # ----------------------------------------------------------
                if z > FLOOR_Z_THRESHOLD:
                    support_ratio = bin_state.get_support_ratio(px, py, ol, ow, z)
                    # Hard minimum (always enforced)
                    if support_ratio < MIN_SUPPORT:
                        continue
                    # Stricter stability when enabled by config
                    if cfg.enable_stability and support_ratio < cfg.min_support_ratio:
                        continue

                # Margin check (box-to-box gap enforcement)
                if not bin_state.is_margin_clear(px, py, ol, ow, z, oh):
                    continue

                # Map local orientation index back to the global index that
                # matches the ordered list returned by Orientation.get_*().
                # Since feasible_orientations is a filtered view of the
                # original `orientations` list, we need the global index.
                global_oidx = orientations.index(
                    feasible_orientations[oidx_local]
                    if oidx_local < len(feasible_orientations)
                    else feasible_orientations[0]
                )

                return PlacementDecision(x=px, y=py, orientation_idx=global_oidx)

        # ------------------------------------------------------------------
        # 5. Fallback: coarse BLF grid scan
        # ------------------------------------------------------------------
        fallback = self._fallback_blf(bin_state, orientations)
        return fallback

    # ------------------------------------------------------------------
    # Fallback placement
    # ------------------------------------------------------------------

    def _fallback_blf(
        self,
        bin_state: BinState,
        orientations: List[Tuple[float, float, float]],
    ) -> Optional[PlacementDecision]:
        """
        Bottom-Left-Fill fallback scan when no EMS yields a valid placement.

        Sweeps a coarse grid (at bin resolution step) and picks the first
        position with valid support.  Prioritises low-z, then small-x,
        then small-y (DBLF ordering).

        Args:
            bin_state:    Current bin state (read-only).
            orientations: List of (ol, ow, oh) tuples to try.

        Returns:
            PlacementDecision or None.
        """
        cfg = self.config
        bin_cfg = cfg.bin
        step = max(1.0, bin_cfg.resolution)

        best: Optional[Tuple[float, float, float, int]] = None  # (z, x, y, oidx)

        for oidx, (ol, ow, oh) in enumerate(orientations):
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue

            x = 0.0
            while x + ol <= bin_cfg.length + EPS:
                y = 0.0
                while y + ow <= bin_cfg.width + EPS:
                    z = bin_state.get_height_at(x, y, ol, ow)

                    if z + oh <= bin_cfg.height + EPS:
                        # Anti-float check
                        valid = True
                        if z > FLOOR_Z_THRESHOLD:
                            sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                            if sr < MIN_SUPPORT:
                                valid = False
                            elif cfg.enable_stability and sr < cfg.min_support_ratio:
                                valid = False

                        if valid and bin_state.is_margin_clear(x, y, ol, ow, z, oh):
                            candidate = (z, x, y, oidx)
                            if best is None or candidate < best:
                                best = candidate

                    y += step
                x += step

        if best is None:
            return None
        return PlacementDecision(x=best[1], y=best[2], orientation_idx=best[3])

    # ------------------------------------------------------------------
    # Convenience factories for all 40 single-bin heuristic variants
    # ------------------------------------------------------------------

    @classmethod
    def make(cls, space_rule: int, orient_rule: int) -> "Heuristic160Strategy":
        """
        Create a Heuristic160Strategy instance for the given (space, orient)
        rule combination.  For convenience when sweeping heuristic variants.

        Args:
            space_rule:  Space-selection rule (1-8).
            orient_rule: Orientation-selection rule (1-5).

        Returns:
            A new Heuristic160Strategy instance.
        """
        return cls(space_rule=space_rule, orient_rule=orient_rule)
