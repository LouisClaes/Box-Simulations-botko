"""
PCT Expansion Scheme Strategy for 3D bin packing.

References:
    Zhao, H., She, Q., Zhu, C., Yang, Y., & Xu, K. (2022).
    "Online 3D Bin Packing with Constrained Deep Reinforcement Learning."
    Proceedings of the AAAI Conference on Artificial Intelligence, 35(6), 741-749.
    AAAI 2021. (DRL foundation)

    Zhao, H., Yu, Y., & Xu, K. (2022).
    "Attend2Pack: Bin Packing through Deep Reinforcement Learning with Attention."
    ICLR 2022 Workshop.

    Zhao, H., et al. (2022).
    "Deliberate Planning of 3D Bin Packing on Packing Configuration Trees."
    ICLR 2022. https://github.com/alexfrom0815/Online-3D-BPP-PCT

    Zhao, H., et al. (2025).
    "PCT: Packing Configuration Trees for Online 3D Bin Packing."
    International Journal of Robotics Research (IJRR), 2025.

Algorithm Overview:
    The PCT paper defines a family of four expansion schemes for generating
    placement candidate positions in a 3D bin.  This module implements the
    GREEDY (non-neural) version: the schemes enumerate candidates; a
    heuristic scoring function selects the best one.

    The four schemes -- from cheapest to most thorough -- are:

      CP  (Corner Points):
          Candidate positions are the XY-projected corners of all placed
          boxes, plus corner-to-corner projections between box pairs.
          Origin and bin-edge corners are always included.
          Complexity: O(n^2) candidates, typically 50-200.

      EP  (Event Points):
          Positions derived from height-transition events in the heightmap:
          wherever the height changes along a row or column, an event
          x- or y-coordinate is recorded; candidates are the cross-products
          of all such coordinates.  CP candidates are added as well.
          Complexity: O(m * |B_2D|), typically 100-500.

      EMS (Empty Maximal Spaces):
          A list of Empty Maximal Spaces (3D axis-aligned free-space boxes)
          is computed from scratch by splitting the full bin against each
          placed box in turn.  The lower-left-bottom corner of every EMS
          is a candidate.  Dominated EMSs are pruned.
          Complexity: O(|E|), typically 20-100 candidates.

      EV  (Event + Vertices):
          Union of EP candidates and EMS lower-left corners.  Most thorough
          but also the slowest.

    Scheme Selection:
        'cp'  -- when fewer than 5 boxes have been placed, or fill < 15 %
        'ems' -- when fill is in [15 %, 65 %)
        'ev'  -- when fill >= 65 % (dense bin, need thorough search)

    Scoring (volume-efficiency + height + stability):
        score = 3.0 * vol_efficiency
              - 2.0 * height_norm
              + 1.0 * support_ratio
              - 0.01 * position_penalty

        where
          vol_efficiency  = box.volume / max(remaining_bin_capacity, 1)
          height_norm     = z / bin_cfg.height
          support_ratio   = fraction of box base that is supported
          position_penalty= (x + y) distance from origin
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

from config import Box, ExperimentConfig, Orientation, PlacementDecision
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

# Anti-float threshold: matches the simulator's MIN_ANTI_FLOAT_RATIO.
MIN_SUPPORT: float = 0.30

# Scheme-selection thresholds
SCHEME_CP_FILL_THRESHOLD: float = 0.15   # Use CP below this fill rate
SCHEME_EMS_FILL_THRESHOLD: float = 0.65  # Use EMS up to this fill rate
SCHEME_CP_BOX_THRESHOLD: int = 5         # Use CP if fewer boxes placed

# Scoring weights
WEIGHT_VOL_EFFICIENCY: float = 3.0
WEIGHT_HEIGHT: float = 2.0
WEIGHT_SUPPORT: float = 1.0
WEIGHT_POSITION: float = 0.01

# Minimum EMS volume (cm^3) below which an EMS is discarded as trivially small.
MIN_EMS_VOLUME: float = 1.0

# Cap on the number of EP candidates to avoid O(n^2) blow-up for large bins.
MAX_EP_CANDIDATES: int = 600

# Cap on the EMS list size after splitting.  Keeps the N largest EMSs.
# Prevents exponential EMS growth when many boxes are placed in large bins.
MAX_EMS_LIST: int = 300

# Max number of recent placed boxes to use for CP cross-projections.
# Limits O(n^2) growth; recent boxes are most useful for new placements.
CP_RECENT_BOXES: int = 20

# Tolerance (cm) used when comparing coordinates for deduplication.
COORD_TOLERANCE: float = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# EMS data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EMSBox:
    """
    An Empty Maximal Space inside the bin.

    Represented as an axis-aligned box by two corner points:
      (x1, y1, z1) -- lower-left-bottom (origin) corner
      (x2, y2, z2) -- upper-right-top corner

    The placement candidate derived from this EMS is (x1, y1).

    Attributes:
        x1, y1, z1:  Origin corner.
        x2, y2, z2:  Opposite corner.
    """

    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float

    @property
    def volume(self) -> float:
        """Volume of this empty space."""
        return (self.x2 - self.x1) * (self.y2 - self.y1) * (self.z2 - self.z1)

    def fits_box(self, ol: float, ow: float, oh: float) -> bool:
        """Return True if a box with oriented dimensions (ol, ow, oh) fits."""
        return (
            ol <= self.x2 - self.x1 + COORD_TOLERANCE
            and ow <= self.y2 - self.y1 + COORD_TOLERANCE
            and oh <= self.z2 - self.z1 + COORD_TOLERANCE
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EMS(({self.x1:.1f},{self.y1:.1f},{self.z1:.1f})"
            f"->({self.x2:.1f},{self.y2:.1f},{self.z2:.1f}))"
        )


# ─────────────────────────────────────────────────────────────────────────────
# EMS helpers (module-level, independent of strategy instance)
# ─────────────────────────────────────────────────────────────────────────────

def _recompute_ems_from_placements(placed_boxes, bin_cfg) -> List[EMSBox]:
    """
    Recompute the EMS list from scratch given all placed boxes.

    The algorithm follows the standard EMS splitting approach (Lai & Chan 1997,
    as used by Zhao et al. 2022):

      1. Initialise with one EMS spanning the full bin.
      2. For each placed box, split every overlapping EMS into up to six
         non-overlapping sub-EMSs (one for each face of the placed box).
      3. Remove dominated EMSs (an EMS is dominated if it is fully contained
         within a strictly larger EMS).

    Args:
        placed_boxes:  Iterable of Placement objects (the bin's placed_boxes).
        bin_cfg:       BinConfig with length, width, height.

    Returns:
        List of EMSBox objects sorted by (z1, x1, y1) ascending.
    """
    # Start with the entire bin as a single EMS.
    ems_list: List[EMSBox] = [
        EMSBox(0.0, 0.0, 0.0, bin_cfg.length, bin_cfg.width, bin_cfg.height)
    ]

    for p in placed_boxes:
        ems_list = _split_ems_list(ems_list, p)
        # Cap INSIDE the loop to prevent exponential blow-up during splitting.
        # Keeping the largest EMSs preserves the most useful free spaces.
        if len(ems_list) > MAX_EMS_LIST:
            ems_list.sort(key=lambda e: e.volume, reverse=True)
            ems_list = ems_list[:MAX_EMS_LIST]
        if not ems_list:
            break

    ems_list = _remove_dominated(ems_list)

    # Keep only non-trivial EMSs.
    ems_list = [e for e in ems_list if e.volume >= MIN_EMS_VOLUME]

    # Cap the EMS list to prevent exponential growth in large bins.
    # Keep the largest EMSs (by volume) — small EMSs rarely yield placements.
    if len(ems_list) > MAX_EMS_LIST:
        ems_list.sort(key=lambda e: e.volume, reverse=True)
        ems_list = ems_list[:MAX_EMS_LIST]

    ems_list.sort(key=lambda e: (e.z1, e.x1, e.y1))
    return ems_list


def _split_ems_list(ems_list: List[EMSBox], placement) -> List[EMSBox]:
    """
    Split every EMS that overlaps *placement* into sub-EMSs.

    For an overlapping EMS *e* and a placed box occupying
    [px, px2) x [py, py2) x [pz, pz2), up to six residual EMSs can be
    carved out (one per face of the placed box):

      Right part:   x in [px2, e.x2),  y in [e.y1, e.y2), z in [e.z1, e.z2)
      Left part:    x in [e.x1, px),   y in [e.y1, e.y2), z in [e.z1, e.z2)
      Front part:   x in [e.x1, e.x2), y in [py2, e.y2),  z in [e.z1, e.z2)
      Back part:    x in [e.x1, e.x2), y in [e.y1, py),   z in [e.z1, e.z2)
      Top part:     x in [e.x1, e.x2), y in [e.y1, e.y2), z in [pz2, e.z2)
      Bottom part:  x in [e.x1, e.x2), y in [e.y1, e.y2), z in [e.z1, pz)
         (bottom part is typically zero for gravity-based packing and skipped
          when pz == e.z1, but included here for completeness)

    EMSs that do not overlap the placement are kept unchanged.

    Args:
        ems_list:   Current list of EMSBox objects.
        placement:  A Placement (has .x, .y, .z, .x_max, .y_max, .z_max).

    Returns:
        New list of non-overlapping EMSBox objects.
    """
    px1, py1, pz1 = placement.x, placement.y, placement.z
    px2, py2, pz2 = placement.x_max, placement.y_max, placement.z_max

    new_list: List[EMSBox] = []

    for e in ems_list:
        # Check 3D overlap: if no overlap, keep as-is.
        if (px2 <= e.x1 or px1 >= e.x2 or
                py2 <= e.y1 or py1 >= e.y2 or
                pz2 <= e.z1 or pz1 >= e.z2):
            new_list.append(e)
            continue

        # There is an overlap: generate up to six residual sub-EMSs.

        # Right sub-EMS: to the right of the placed box.
        if px2 < e.x2:
            _append_valid(new_list, EMSBox(px2, e.y1, e.z1, e.x2, e.y2, e.z2))

        # Left sub-EMS: to the left of the placed box.
        if px1 > e.x1:
            _append_valid(new_list, EMSBox(e.x1, e.y1, e.z1, px1, e.y2, e.z2))

        # Front sub-EMS: in front of the placed box (larger y).
        if py2 < e.y2:
            _append_valid(new_list, EMSBox(e.x1, py2, e.z1, e.x2, e.y2, e.z2))

        # Back sub-EMS: behind the placed box (smaller y).
        if py1 > e.y1:
            _append_valid(new_list, EMSBox(e.x1, e.y1, e.z1, e.x2, py1, e.z2))

        # Top sub-EMS: above the placed box.
        if pz2 < e.z2:
            _append_valid(new_list, EMSBox(e.x1, e.y1, pz2, e.x2, e.y2, e.z2))

        # Bottom sub-EMS: below the placed box (non-zero only if box floats).
        if pz1 > e.z1:
            _append_valid(new_list, EMSBox(e.x1, e.y1, e.z1, e.x2, e.y2, pz1))

    return new_list


def _append_valid(target: List[EMSBox], e: EMSBox) -> None:
    """Append *e* to *target* only if all three extents are positive."""
    if e.x2 > e.x1 + COORD_TOLERANCE and e.y2 > e.y1 + COORD_TOLERANCE and e.z2 > e.z1 + COORD_TOLERANCE:
        target.append(e)


def _remove_dominated(ems_list: List[EMSBox]) -> List[EMSBox]:
    """
    Remove EMSs that are strictly dominated (fully contained in another EMS).

    An EMS e1 is dominated by e2 if:
      e2.x1 <= e1.x1 and e2.y1 <= e1.y1 and e2.z1 <= e1.z1
      e2.x2 >= e1.x2 and e2.y2 >= e1.y2 and e2.z2 >= e1.z2
      and e2 != e1 (strict containment: at least one bound differs)

    For large lists this is O(n^2) but n rarely exceeds ~200 in practice.

    Args:
        ems_list:  List of EMSBox objects.

    Returns:
        Filtered list with dominated EMSs removed.
    """
    if len(ems_list) <= 1:
        return ems_list

    result: List[EMSBox] = []
    n = len(ems_list)

    for i in range(n):
        e1 = ems_list[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            e2 = ems_list[j]
            # e1 is dominated by e2 if e2 fully contains e1 (and is strictly larger).
            if (e2.x1 <= e1.x1 and e2.y1 <= e1.y1 and e2.z1 <= e1.z1 and
                    e2.x2 >= e1.x2 and e2.y2 >= e1.y2 and e2.z2 >= e1.z2 and
                    (e2.x1 < e1.x1 or e2.y1 < e1.y1 or e2.z1 < e1.z1 or
                     e2.x2 > e1.x2 or e2.y2 > e1.y2 or e2.z2 > e1.z2)):
                dominated = True
                break
        if not dominated:
            result.append(e1)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Candidate generators (CP, EP, EMS, EV)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_cp_candidates(bin_state: BinState) -> Set[Tuple[float, float]]:
    """
    CP (Corner Points) expansion scheme.

    Generates candidate (x, y) positions from:
      - The projected XY corners of each placed box: (x, y), (x_max, y),
        (x, y_max), (x_max, y_max).
      - Cross-projections between pairs of placed boxes: for boxes p and q,
        add (p.x_max, q.y), (p.x, q.y_max), (p.x_max, q.y_max),
        (q.x_max, p.y), (q.x, p.y_max).  These correspond to positions where
        a new box can be flush against both p and q simultaneously.
      - The four bin-edge corners: (0,0), (0, W), (L, 0), (L, W).

    Points with x > bin_length or y > bin_width are excluded.

    Args:
        bin_state:  Current bin state (read-only).

    Returns:
        Set of (x, y) candidate positions.
    """
    bin_cfg = bin_state.config
    L = bin_cfg.length
    W = bin_cfg.width
    placed = bin_state.placed_boxes

    candidates: Set[Tuple[float, float]] = set()

    # Always include the four bin corners.
    candidates.add((0.0, 0.0))
    candidates.add((0.0, W))
    candidates.add((L, 0.0))
    candidates.add((L, W))

    # Per-box corners.
    for p in placed:
        for cx, cy in (
            (p.x,     p.y),
            (p.x_max, p.y),
            (p.x,     p.y_max),
            (p.x_max, p.y_max),
        ):
            if cx <= L + COORD_TOLERANCE and cy <= W + COORD_TOLERANCE:
                candidates.add((cx, cy))

    # Cross-projections between box pairs (O(n^2), bounded by n^2 candidates).
    # Limit to the most recently placed boxes to prevent O(n^2) blow-up.
    recent = placed[-CP_RECENT_BOXES:] if len(placed) > CP_RECENT_BOXES else placed
    n = len(recent)
    for i in range(n):
        p = recent[i]
        for j in range(n):
            if j == i:
                continue
            q = recent[j]
            for cx, cy in (
                (p.x_max, q.y),
                (p.x,     q.y_max),
                (p.x_max, q.y_max),
                (q.x_max, p.y),
                (q.x,     p.y_max),
                (q.x_max, p.y_max),
            ):
                if cx <= L + COORD_TOLERANCE and cy <= W + COORD_TOLERANCE:
                    candidates.add((cx, cy))

    return candidates


def _generate_ep_candidates(bin_state: BinState) -> Set[Tuple[float, float]]:
    """
    EP (Event Points) expansion scheme.

    Collects x-coordinates wherever the heightmap transitions along a row
    and y-coordinates wherever it transitions along a column.  The cross-
    product of all event-x and event-y values forms the EP candidates.
    CP candidates are included as well (union).

    To prevent combinatorial explosion, the total number of EP candidates
    is capped at MAX_EP_CANDIDATES; excess candidates are dropped (the most
    extreme ones, to keep interior event points which are more useful).

    Args:
        bin_state:  Current bin state (read-only).

    Returns:
        Set of (x, y) candidate positions.
    """
    bin_cfg = bin_state.config
    heightmap = bin_state.heightmap
    res = bin_cfg.resolution
    grid_l = bin_cfg.grid_l
    grid_w = bin_cfg.grid_w

    event_x: Set[float] = {0.0, bin_cfg.length}
    event_y: Set[float] = {0.0, bin_cfg.width}

    # Scan rows: collect x positions where height changes along each row.
    for gy in range(grid_w):
        for gx in range(1, grid_l):
            if abs(heightmap[gx, gy] - heightmap[gx - 1, gy]) > COORD_TOLERANCE:
                event_x.add(gx * res)

    # Scan columns: collect y positions where height changes along each column.
    for gx in range(grid_l):
        for gy in range(1, grid_w):
            if abs(heightmap[gx, gy] - heightmap[gx, gy - 1]) > COORD_TOLERANCE:
                event_y.add(gy * res)

    # Cross-product, capped to avoid explosion.
    ex_list = sorted(event_x)
    ey_list = sorted(event_y)

    candidates: Set[Tuple[float, float]] = set()
    count = 0
    for ex in ex_list:
        for ey in ey_list:
            if count >= MAX_EP_CANDIDATES:
                break
            candidates.add((ex, ey))
            count += 1
        if count >= MAX_EP_CANDIDATES:
            break

    # Union with CP candidates.
    candidates |= _generate_cp_candidates(bin_state)
    return candidates


def _generate_ems_candidates(
    ems_list: List[EMSBox],
) -> Set[Tuple[float, float]]:
    """
    EMS expansion scheme.

    The candidate for each EMS is its lower-left-bottom projected corner
    (x1, y1) on the XY plane.

    Args:
        ems_list:  Pre-computed list of EMSBox objects.

    Returns:
        Set of (x, y) candidate positions.
    """
    return {(e.x1, e.y1) for e in ems_list}


# ─────────────────────────────────────────────────────────────────────────────
# Strategy class
# ─────────────────────────────────────────────────────────────────────────────

@register_strategy
class PCTExpansionStrategy(BaseStrategy):
    """
    PCT (Packing Configuration Tree) Expansion Scheme strategy.

    Implements the greedy (non-neural) version of the four expansion schemes
    described in Zhao et al. (ICLR 2022 / IJRR 2025).  Instead of a learned
    policy, placement candidates generated by the selected scheme are scored
    by a heuristic function based on volume efficiency, height, and support.

    Scheme selection is automatic:
      - CP  when fill < 15 % or fewer than 5 boxes placed (fast, sufficient).
      - EMS when 15 % <= fill < 65 % (best balance of quality and speed).
      - EV  when fill >= 65 % (most thorough search for dense bins).

    Attributes:
        name:  Strategy identifier used in the registry ("pct_expansion").
    """

    name: str = "pct_expansion"

    def __init__(self) -> None:
        super().__init__()

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Store config; no persistent state needed between calls."""
        super().on_episode_start(config)

    # ── Main entry point ─────────────────────────────────────────────────

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Find the best placement for *box* using PCT expansion schemes.

        Steps:
          1. Determine which expansion scheme to use (CP / EMS / EV) based
             on the current fill rate and number of placed boxes.
          2. Generate candidate (x, y) positions using the chosen scheme.
          3. For each candidate and each allowed orientation:
               a. Check bounds.
               b. Compute resting height z from the heightmap.
               c. Reject if height limit exceeded.
               d. Enforce MIN_SUPPORT = 0.30 (anti-float).
               e. Enforce cfg.min_support_ratio if enable_stability is set.
               f. Compute heuristic score.
          4. Return the PlacementDecision for the highest-scoring candidate,
             or None if no valid placement exists.

        Args:
            box:       Box to place (original, un-rotated dimensions).
            bin_state: Current bin state (read-only).

        Returns:
            PlacementDecision(x, y, orientation_idx) or None.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        # --- Resolve allowed orientations ---
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Quick early-out: can the box fit in the bin in any orientation?
        valid_orientations: List[Tuple[int, float, float, float]] = []
        for oidx, (ol, ow, oh) in enumerate(orientations):
            if (ol <= bin_cfg.length + COORD_TOLERANCE
                    and ow <= bin_cfg.width + COORD_TOLERANCE
                    and oh <= bin_cfg.height + COORD_TOLERANCE):
                valid_orientations.append((oidx, ol, ow, oh))
        if not valid_orientations:
            return None

        # --- Select expansion scheme and generate candidates ---
        fill_rate = bin_state.get_fill_rate()
        num_placed = len(bin_state.placed_boxes)
        scheme = self._select_scheme(fill_rate, num_placed)

        candidates: Set[Tuple[float, float]]

        if scheme == 'cp':
            candidates = _generate_cp_candidates(bin_state)

        elif scheme == 'ems':
            ems_list = _recompute_ems_from_placements(
                bin_state.placed_boxes, bin_cfg
            )
            candidates = _generate_ems_candidates(ems_list)
            # Fall back to CP if EMS yielded no candidates.
            if not candidates:
                candidates = _generate_cp_candidates(bin_state)

        else:  # 'ev' -- union of EP and EMS
            ems_list = _recompute_ems_from_placements(
                bin_state.placed_boxes, bin_cfg
            )
            candidates = _generate_ems_candidates(ems_list)
            candidates |= _generate_ep_candidates(bin_state)

        # Always guarantee at least the origin so the bin is never skipped.
        candidates.add((0.0, 0.0))

        # --- Evaluate all (candidate, orientation) pairs ---
        bin_vol = bin_cfg.volume
        placed_vol = sum(p.volume for p in bin_state.placed_boxes)
        remaining_capacity = max(bin_vol - placed_vol, 1.0)

        best_score: float = -float("inf")
        best_result: Optional[Tuple[float, float, int]] = None  # (x, y, oidx)

        for cx, cy in candidates:
            for oidx, ol, ow, oh in valid_orientations:
                # --- Bounds check ---
                if cx + ol > bin_cfg.length + COORD_TOLERANCE:
                    continue
                if cy + ow > bin_cfg.width + COORD_TOLERANCE:
                    continue

                # --- Resting height (gravity) ---
                z = bin_state.get_height_at(cx, cy, ol, ow)

                # --- Height limit ---
                if z + oh > bin_cfg.height + COORD_TOLERANCE:
                    continue

                # --- Anti-float: enforce MIN_SUPPORT = 0.30 always ---
                support_ratio = 1.0
                if z > 0.5:
                    support_ratio = bin_state.get_support_ratio(
                        cx, cy, ol, ow, z
                    )
                    if support_ratio < MIN_SUPPORT:
                        continue

                # --- Stricter stability check when enabled ---
                if cfg.enable_stability and z > 0.5:
                    if support_ratio < cfg.min_support_ratio:
                        continue

                # Margin check (box-to-box gap enforcement)
                if not bin_state.is_margin_clear(cx, cy, ol, ow, z, oh):
                    continue

                # --- Score ---
                score = self._compute_score(
                    cx, cy, z, ol, ow, oh,
                    support_ratio, box.volume,
                    remaining_capacity, bin_cfg,
                )

                if score > best_score:
                    best_score = score
                    best_result = (cx, cy, oidx)

        if best_result is None:
            return None

        return PlacementDecision(
            x=best_result[0],
            y=best_result[1],
            orientation_idx=best_result[2],
        )

    # ── Scheme selection ─────────────────────────────────────────────────

    @staticmethod
    def _select_scheme(fill_rate: float, num_placed: int) -> str:
        """
        Choose which PCT expansion scheme to use.

        Decision rule (ascending thoroughness):
          CP  -- early phase: few boxes or low fill (fast is sufficient)
          EMS -- mid phase: EMS gives best quality-to-speed ratio
          EV  -- late phase: dense bin requires thorough candidate coverage

        Args:
            fill_rate:   Current volumetric fill rate in [0, 1].
            num_placed:  Number of boxes already placed in this bin.

        Returns:
            One of 'cp', 'ems', or 'ev'.
        """
        if num_placed < SCHEME_CP_BOX_THRESHOLD or fill_rate < SCHEME_CP_FILL_THRESHOLD:
            return 'cp'
        if fill_rate < SCHEME_EMS_FILL_THRESHOLD:
            return 'ems'
        return 'ev'

    # ── Scoring ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_score(
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        support_ratio: float,
        box_volume: float,
        remaining_capacity: float,
        bin_cfg,
    ) -> float:
        """
        Compute the placement score for a candidate.

        Formula:
            score = WEIGHT_VOL_EFFICIENCY * vol_efficiency
                  - WEIGHT_HEIGHT         * height_norm
                  + WEIGHT_SUPPORT        * support_ratio
                  - WEIGHT_POSITION       * position_penalty

        where:
          vol_efficiency  = box_volume / remaining_capacity
                            (how much of the remaining space this box fills)
          height_norm     = z / bin_cfg.height
                            (normalised placement height; lower is better)
          support_ratio   = fraction of box base that is supported
                            (already computed upstream; 1.0 on the floor)
          position_penalty= (x + y)
                            (slight back-left-corner preference)

        Args:
            x, y, z:              Candidate position.
            ol, ow, oh:           Oriented box dimensions.
            support_ratio:        Pre-computed support fraction.
            box_volume:           Volume of the box being placed.
            remaining_capacity:   Remaining free bin volume (>= 1 to avoid /0).
            bin_cfg:              Bin configuration (for normalisation).

        Returns:
            Scalar score; higher is better.
        """
        vol_efficiency = box_volume / remaining_capacity
        height_norm = z / bin_cfg.height if bin_cfg.height > 0.0 else 0.0
        position_penalty = x + y

        return (
            WEIGHT_VOL_EFFICIENCY * vol_efficiency
            - WEIGHT_HEIGHT * height_norm
            + WEIGHT_SUPPORT * support_ratio
            - WEIGHT_POSITION * position_penalty
        )
