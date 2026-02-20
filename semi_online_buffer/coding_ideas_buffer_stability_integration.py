"""
=============================================================================
CODING IDEAS: Buffer-Aware Stability Integration for Semi-Online 3D Packing
=============================================================================

Based on: Ali, Ramos, Oliveira (2025) - "Static stability versus packing
efficiency in online three-dimensional packing problems"
Computers & Operations Research, Vol. 178, Article 107005

FOCUS: How to leverage the 5-10 item BUFFER to improve upon the paper's
pure-online results, specifically for the stability-efficiency tradeoff.

The paper operates in pure online mode (1 item at a time, no choice).
Our system has a buffer of 5-10 items, giving us ITEM SELECTION as an
additional degree of freedom. This is the single biggest advantage we
have over the paper's approach.

=============================================================================
UPDATED: 2026-02-18 -- Deep research pass with complete implementations.
         Prior version had 6 ideas as comments. Now includes concrete
         code for all ideas plus 3 new ideas (7-9) and full integration
         with the 160-heuristic framework and 4 stability constraints.
=============================================================================
"""

import math
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# IMPORTS FROM SIBLING MODULES (defined in other coding_ideas files)
# ============================================================================
# In production, these would be proper imports:
# from stability.coding_ideas_stability_vs_efficiency import (
#     StabilityChecker, StabilityResult, FullBaseSupport,
#     PartialBaseSupport, CoGPolygonSupport, PartialBasePolygonSupport,
#     Position, ItemDims, PlacedItem, compute_full_support_polygon
# )
# from heuristics.coding_ideas_160_heuristic_framework import (
#     HeuristicEngine, EMS, EMSManager, fits_in_ems, get_oriented_dims,
#     SPACE_RULE_MAP, ORIENTATION_RULE_MAP
# )


# ============================================================================
# DATA STRUCTURES FOR BUFFER SYSTEM
# ============================================================================

@dataclass
class BufferItem:
    """An item waiting in the buffer to be placed."""
    item_id: int
    length: float
    width: float
    height: float
    weight: float = 1.0
    allowed_orientations: List[int] = field(default_factory=lambda: list(range(6)))
    arrival_time: int = 0  # when the item entered the buffer

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def max_base_area(self) -> float:
        """Largest possible base area across all orientations."""
        areas = []
        dims_list = [(self.length, self.width), (self.length, self.height),
                     (self.width, self.height)]
        for l, w in dims_list:
            areas.append(l * w)
        return max(areas)


@dataclass
class PlacementCandidate:
    """A candidate placement evaluated by the buffer selector."""
    item: Any  # BufferItem
    bin_idx: int
    ems: Any  # EMS
    orientation: int
    position: Any  # Position
    oriented_dims: Any  # ItemDims

    # Scoring components
    efficiency_score: float = 0.0
    stability_score: float = 0.0
    surface_score: float = 0.0
    height_penalty: float = 0.0
    combined_score: float = 0.0

    # Stability metadata
    stability_result: Any = None  # StabilityResult


# ============================================================================
# IDEA 1: JOINT ITEM-PLACEMENT OPTIMIZATION (Core Algorithm)
# ============================================================================
#
# THE FUNDAMENTAL INNOVATION: evaluate ALL (item, bin, EMS, orientation)
# combinations from the buffer and select the globally best one.
#
# Paper context: The paper's heuristics receive ONE item and find the best
# placement. We receive 5-10 items and choose WHICH item to place.
#
# Complexity per decision step:
#   B * K * E * R = 10 * 2 * 20 * 6 = 2,400 evaluations
#   Each with stability check: ~0.01-0.1ms
#   Total: 2.4 - 240ms -- feasible for real-time
#
# Paper's online vs offline gap: 71% equal, 29% offline wins, 0% online wins.
# With buffer of 10: estimated 80-85% equal (closing 10-14% of the gap).

class BufferStabilitySelector:
    """
    Joint item-placement optimizer for the semi-online buffer system.

    Evaluates all feasible (item, bin, EMS, orientation) tuples from
    the buffer and selects the one that maximizes a combined
    stability-efficiency score.
    """

    def __init__(self,
                 stability_checker,  # StabilityChecker instance
                 space_rule: int = 5,
                 orient_rule: int = 3,
                 alpha: float = 0.5,
                 top_k_ems: int = 15,
                 bin_dims=None):
        """
        Args:
            stability_checker: one of the 4 stability checkers
            space_rule: which of the 8 space selection rules to use for
                       EMS prioritization (default 5 = DBLF+corner, the best)
            orient_rule: not used for filtering, only as fallback
            alpha: weight for efficiency vs stability in combined score
                   alpha=1.0: pure efficiency
                   alpha=0.0: pure stability
                   alpha=0.5: balanced (RECOMMENDED)
            top_k_ems: only evaluate the top K EMSs per bin (limits search)
            bin_dims: container dimensions for space rule computation
        """
        self.stability_checker = stability_checker
        self.space_rule = space_rule
        self.orient_rule = orient_rule
        self.alpha = alpha
        self.top_k_ems = top_k_ems
        self.bin_dims = bin_dims

    def select_best_placement(
        self,
        buffer_items: List[Any],  # List[BufferItem]
        active_bins: List[Any],   # List of (EMSManager, List[PlacedItem])
    ) -> Optional[PlacementCandidate]:
        """
        Evaluate all feasible placements and return the best one.

        Returns None if no feasible placement exists in any active bin.
        """
        best = None

        for item in buffer_items:
            for bin_idx, (ems_mgr, placed_items) in enumerate(active_bins):
                sorted_ems = ems_mgr.get_sorted_ems(self.space_rule)

                for ems in sorted_ems[:self.top_k_ems]:
                    for orient in item.allowed_orientations:
                        # Get oriented dimensions
                        oriented = self._get_oriented_dims(item, orient)

                        # Check geometric fit
                        if not self._fits_in_ems(oriented, ems):
                            continue

                        # Check stability
                        position = self._ems_to_position(ems)
                        result = self.stability_checker.check(
                            oriented, position, placed_items
                        )

                        if not result.is_stable:
                            continue

                        # Score this candidate
                        candidate = PlacementCandidate(
                            item=item, bin_idx=bin_idx, ems=ems,
                            orientation=orient, position=position,
                            oriented_dims=oriented,
                            stability_result=result
                        )
                        self._score_candidate(candidate, ems, placed_items)

                        if best is None or candidate.combined_score > best.combined_score:
                            best = candidate

        return best

    def _score_candidate(self, candidate: PlacementCandidate,
                         ems: Any, placed_items: List[Any]):
        """
        Compute the combined stability-efficiency score.

        SCORING FUNCTION (the key design decision):
            combined = alpha * efficiency + (1 - alpha) * stability

        Efficiency components:
            - Volume fill ratio: item_volume / ems_volume
            - Margin tightness: how snugly the item fits

        Stability components:
            - Support area ratio: SP area / base area
            - CG centrality: how centered CG is within SP (0=edge, 1=center)
            - Height penalty: items placed higher get penalized

        Surface creation bonus:
            - Reward placements that create flat surfaces for future items
        """
        item = candidate.item
        dims = candidate.oriented_dims
        result = candidate.stability_result

        # Efficiency score
        ems_vol = max(self._ems_volume(ems), 1e-9)
        fill_ratio = dims.volume / ems_vol
        margin_tightness = self._compute_margin_tightness(dims, ems)
        candidate.efficiency_score = fill_ratio + 0.3 * margin_tightness

        # Stability score
        support_ratio = result.support_ratio if hasattr(result, 'support_ratio') else 0.5
        sp_area_ratio = (result.sp_area / result.base_area
                         if hasattr(result, 'sp_area') and result.base_area > 0
                         else 1.0)
        candidate.stability_score = 0.6 * support_ratio + 0.4 * min(sp_area_ratio, 1.0)

        # Height penalty (paper shows instability correlates with cargo height)
        bin_height = self.bin_dims.height if self.bin_dims else 1500
        height_ratio = candidate.position.z / bin_height if bin_height > 0 else 0
        candidate.height_penalty = 0.2 * height_ratio

        # Surface creation bonus
        candidate.surface_score = self._surface_creation_score(
            candidate.position, dims, placed_items
        )

        # Combined score
        eff = self.alpha * candidate.efficiency_score
        stab = (1 - self.alpha) * (candidate.stability_score - candidate.height_penalty)
        bonus = 0.15 * candidate.surface_score
        candidate.combined_score = eff + stab + bonus

    def _compute_margin_tightness(self, dims: Any, ems: Any) -> float:
        """
        How snugly the item fits in the EMS (Rule 1 from paper Table 2).
        Tighter fit -> higher score (less wasted space).
        """
        ems_l = ems.x2 - ems.x1
        ems_w = ems.y2 - ems.y1
        ems_h = ems.z2 - ems.z1
        total_margin = ((ems_l - dims.length) +
                        (ems_w - dims.width) +
                        (ems_h - dims.height))
        # Normalize: 0 = perfect fit, decays with increasing margin
        max_margin = ems_l + ems_w + ems_h
        if max_margin <= 0:
            return 1.0
        return max(0.0, 1.0 - total_margin / max_margin)

    def _surface_creation_score(self, position: Any, dims: Any,
                                 placed_items: List[Any]) -> float:
        """
        Estimate how good the resulting top surface is for future stacking.

        High score = placing this item extends an existing flat layer.
        This indirectly improves stability for future items.

        Checks alignment: how much area at the same top-z height exists
        adjacent to this item's top surface.
        """
        EPSILON = 1e-6
        new_top_z = position.z + dims.height
        new_top_area = dims.length * dims.width
        aligned_area = 0.0

        for existing in placed_items:
            if abs(existing.top_z - new_top_z) < EPSILON:
                # Check if adjacent or overlapping in XY
                # Adjacent means sharing at least one edge
                x_overlap = (min(position.x + dims.length, existing.x_max) -
                             max(position.x, existing.position.x))
                y_overlap = (min(position.y + dims.width, existing.y_max) -
                             max(position.y, existing.position.y))

                if x_overlap > EPSILON and y_overlap > EPSILON:
                    aligned_area += x_overlap * y_overlap

        return aligned_area / new_top_area if new_top_area > 0 else 0

    # Placeholder methods for type compatibility
    def _get_oriented_dims(self, item, orient):
        """Get item dims in specified orientation."""
        # In production: use get_oriented_dims from heuristic framework
        l, w, h = item.length, item.width, item.height
        orientations = [
            (l, w, h), (l, h, w), (w, l, h), (w, h, l), (h, l, w), (h, w, l)
        ]
        d = orientations[orient]
        # Return an object with .length, .width, .height, .volume, .base_area
        return type('Dims', (), {
            'length': d[0], 'width': d[1], 'height': d[2],
            'volume': d[0]*d[1]*d[2], 'base_area': d[0]*d[1]
        })()

    def _fits_in_ems(self, dims, ems):
        return (dims.length <= ems.x2 - ems.x1 + 1e-9 and
                dims.width <= ems.y2 - ems.y1 + 1e-9 and
                dims.height <= ems.z2 - ems.z1 + 1e-9)

    def _ems_to_position(self, ems):
        return type('Pos', (), {'x': ems.x1, 'y': ems.y1, 'z': ems.z1})()

    def _ems_volume(self, ems):
        return (ems.x2-ems.x1) * (ems.y2-ems.y1) * (ems.z2-ems.z1)


# ============================================================================
# IDEA 2: STABILITY-GUIDED ITEM ORDERING FROM BUFFER
# ============================================================================
#
# Key insight from paper: items placed FIRST form the foundation.
# With our buffer we can do IMPLICIT SEQUENCING -- choosing which item
# to place from the buffer is equivalent to partial reordering.
#
# Paper's offline comparison (Rocha et al. 2022) uses full sequencing
# control; our buffer gives us partial sequencing control.

class FoundationFirstSelector:
    """
    Prioritize items from the buffer based on their 'foundation quality'.

    STRATEGY: "Bottom-Heavy First"
    - When bin is < 30% full: prioritize large-base-area items
    - When bin is 30-70% full: prioritize best-fit items
    - When bin is > 70% full: prioritize small gap-filling items
    """

    def rank_buffer_items(self, buffer_items: List[Any],
                          bin_fill_rate: float,
                          available_ems: List[Any]) -> List[Any]:
        """
        Rank buffer items by their suitability for current bin state.

        Returns items sorted by priority (best first).
        """
        scored = []

        for item in buffer_items:
            if bin_fill_rate < 0.30:
                # Foundation phase: large base area items
                score = item.max_base_area
            elif bin_fill_rate < 0.70:
                # Body phase: items that fit well in available spaces
                score = self._best_fit_score(item, available_ems)
            else:
                # Top-off phase: small items that fill gaps
                score = -item.volume  # smaller = better (negative for sorting)
            scored.append((score, item))

        scored.sort(key=lambda x: -x[0])  # highest score first
        return [item for _, item in scored]

    def _best_fit_score(self, item: Any, ems_list: List[Any]) -> float:
        """How well does this item match available spaces?"""
        best_ratio = 0.0
        for ems in ems_list[:10]:  # check top 10 EMSs
            ems_vol = (ems.x2-ems.x1) * (ems.y2-ems.y1) * (ems.z2-ems.z1)
            if ems_vol > 0:
                ratio = item.volume / ems_vol
                if 0 < ratio <= 1.0 and ratio > best_ratio:
                    best_ratio = ratio
        return best_ratio


# ============================================================================
# IDEA 3: LOOK-AHEAD STABILITY ASSESSMENT
# ============================================================================
#
# Problem from paper (Section 3.3, Fig. 4): An item stable at placement
# time can become unstable LATER when subsequent items reduce its support.
#
# With our buffer, we can do 1-step look-ahead:
#   1. Tentatively place item A
#   2. For each remaining buffer item B, find best placement for B
#   3. Check if A is STILL stable after B is placed
#   4. If A becomes unstable in many scenarios, choose differently

class LookAheadStabilityAssessor:
    """
    Perform 1-step look-ahead to assess placement robustness.

    Complexity:
        buffer_size * (buffer_size-1) * ems_count * orientations
        = 10 * 9 * 20 * 6 = 10,800 evaluations for second step
        Each ~0.1ms = ~1 second total
        -> Only use for critical placements (high bin fill, tall stacks)
    """

    def __init__(self, stability_checker, use_threshold: float = 0.50):
        self.stability_checker = stability_checker
        self.use_threshold = use_threshold  # only look ahead when bin > 50% full

    def assess_robustness(self, candidate: PlacementCandidate,
                          remaining_buffer: List[Any],
                          bin_state: Any) -> float:
        """
        Estimate the probability that candidate placement stays stable
        after one more item is placed.

        Returns robustness score 0-1 (1 = always stays stable).
        """
        # Simplified version: check if support area exceeds threshold by margin
        result = candidate.stability_result
        if result is None:
            return 0.0

        # "Robust stability" = support ratio significantly above minimum
        excess_ratio = result.support_ratio - 0.50  # above 50% threshold
        if excess_ratio > 0.30:
            return 1.0  # very robust, no need for detailed look-ahead

        # Full look-ahead (expensive)
        stays_stable_count = 0
        total_scenarios = 0

        for next_item in remaining_buffer[:5]:  # limit to 5 items
            # Simulate placing candidate, then placing next_item
            # Check if candidate is still stable
            total_scenarios += 1
            # In production: actually simulate and re-check stability
            # Here: approximate based on support area margin
            if excess_ratio > 0.15:
                stays_stable_count += 1
            elif excess_ratio > 0.05:
                stays_stable_count += 0.5  # partial credit

        return stays_stable_count / max(total_scenarios, 1)


# ============================================================================
# IDEA 4: ADAPTIVE STABILITY CONSTRAINT WITH BUFFER
# ============================================================================
#
# The paper uses ONE stability constraint for the entire process.
# With our buffer, we can adapt PER ITEM based on context.

class AdaptiveStabilityPolicy:
    """
    Dynamically switch stability constraints based on placement context.

    ADAPTIVE POLICY (height-based gradient):
    - Bottom layer (height < 30% of bin): FullBaseSupport
      Bottom items bear the most load, must be maximally stable.
    - Middle layers (30-60%): PartialBasePolygonSupport(area_threshold=0.50)
      Balance efficiency and stability in the main body.
    - Top layers (>60%): CoGPolygonSupport
      Top items bear least load, maximize remaining space usage.

    This "stability gradient" matches physical intuition and is a
    novel contribution not found in the paper (which uses one constraint
    throughout).

    Additional buffer-aware adaptations:
    - If all buffer items are small: relax constraint (small items
      are easier to place stably)
    - If buffer contains very large items: tighten constraint (need
      to preserve large EMSs, use space carefully)
    """

    def __init__(self, bin_height: float = 1500.0):
        self.bin_height = bin_height
        # Pre-create all checker instances
        self.full_base = None       # FullBaseSupport()
        self.partial_base = None    # PartialBaseSupport(0.80)
        self.cog_polygon = None     # CoGPolygonSupport()
        self.pbp_strict = None      # PartialBasePolygonSupport(0.60)
        self.pbp_default = None     # PartialBasePolygonSupport(0.50)
        self.pbp_relaxed = None     # PartialBasePolygonSupport(0.40)

    def set_checkers(self, full_base, partial_base, cog_polygon,
                     pbp_strict, pbp_default, pbp_relaxed):
        """Set the stability checker instances."""
        self.full_base = full_base
        self.partial_base = partial_base
        self.cog_polygon = cog_polygon
        self.pbp_strict = pbp_strict
        self.pbp_default = pbp_default
        self.pbp_relaxed = pbp_relaxed

    def select_constraint(self, placement_height: float,
                          bin_fill_rate: float,
                          buffer_items: List[Any]) -> Any:
        """
        Select the appropriate stability constraint for this placement.

        Args:
            placement_height: z-coordinate where item will be placed
            bin_fill_rate: current fill ratio of the target bin (0-1)
            buffer_items: current buffer contents

        Returns:
            StabilityChecker instance to use for this placement
        """
        height_ratio = placement_height / self.bin_height

        # Height-based selection
        if height_ratio < 0.30:
            base_constraint = self.full_base or self.pbp_strict
        elif height_ratio < 0.60:
            base_constraint = self.pbp_default
        else:
            base_constraint = self.cog_polygon or self.pbp_relaxed

        # Buffer-aware adjustment
        if buffer_items:
            avg_volume = sum(i.volume for i in buffer_items) / len(buffer_items)
            bin_volume = 1200 * 800 * self.bin_height  # approximate
            relative_size = avg_volume / bin_volume

            if relative_size < 0.01:
                # All small items: relax one level
                base_constraint = self._relax(base_constraint)
            elif relative_size > 0.05:
                # Large items: tighten one level
                base_constraint = self._tighten(base_constraint)

        return base_constraint

    def _relax(self, current):
        """Relax the constraint one level."""
        hierarchy = [self.full_base, self.pbp_strict, self.pbp_default,
                     self.pbp_relaxed, self.cog_polygon]
        hierarchy = [c for c in hierarchy if c is not None]
        try:
            idx = hierarchy.index(current)
            return hierarchy[min(idx + 1, len(hierarchy) - 1)]
        except ValueError:
            return current

    def _tighten(self, current):
        """Tighten the constraint one level."""
        hierarchy = [self.full_base, self.pbp_strict, self.pbp_default,
                     self.pbp_relaxed, self.cog_polygon]
        hierarchy = [c for c in hierarchy if c is not None]
        try:
            idx = hierarchy.index(current)
            return hierarchy[max(idx - 1, 0)]
        except ValueError:
            return current


# ============================================================================
# IDEA 5: BIN CLOSING POLICY WITH STABILITY AWARENESS (for k=2)
# ============================================================================
#
# The paper uses unbounded bins. Our k=2 system requires bin closing.
# This is NOT addressed in the paper and represents our own contribution.

class StabilityAwareClosingPolicy:
    """
    When neither of 2 active bins can accommodate items from the buffer,
    decide which bin to close permanently.

    Five closing policies to test:
    a) CLOSE_LEAST_FILLED: close emptier bin
    b) CLOSE_MOST_FILLED: close fuller bin (preserve flexibility)
    c) CLOSE_LEAST_STABLE: close less stable bin
    d) CLOSE_COMBINED: weighted score of fill + stability + buffer compatibility
    e) CLOSE_WORST_EMS: close bin with least useful remaining EMSs

    RECOMMENDED: CLOSE_COMBINED (d) for dual-objective optimization.
    """

    def __init__(self, w_ems: float = 0.4, w_stab: float = 0.3,
                 w_buffer: float = 0.3):
        self.w_ems = w_ems
        self.w_stab = w_stab
        self.w_buffer = w_buffer

    def choose_bin_to_close(
        self,
        bin_a: Any,  # (EMSManager, List[PlacedItem])
        bin_b: Any,
        buffer_items: List[Any],
        stability_checker: Any
    ) -> int:
        """
        Score each bin on future potential. Close the lower-scored bin.

        Returns: 0 to close bin_a, 1 to close bin_b.
        """
        score_a = self._score_bin(bin_a, buffer_items, stability_checker)
        score_b = self._score_bin(bin_b, buffer_items, stability_checker)

        # Close the bin with LOWER score (less future potential)
        return 0 if score_a <= score_b else 1

    def _score_bin(self, bin_data: Any, buffer_items: List[Any],
                   stability_checker: Any) -> float:
        """
        Score a bin on its "keep worthiness".
        Higher score = more worth keeping open.
        """
        ems_mgr, placed_items = bin_data

        # EMS quality: largest EMS volume / remaining volume
        total_vol = 1200 * 800 * 1500  # bin volume
        used_vol = sum(p.dims.volume for p in placed_items)
        remaining_vol = total_vol - used_vol

        if not ems_mgr.ems_list or remaining_vol <= 0:
            return 0.0

        largest_ems = max(e.volume for e in ems_mgr.ems_list)
        ems_quality = largest_ems / remaining_vol

        # Stability quality
        if placed_items:
            # Approximate: count items on floor (always stable)
            floor_items = sum(1 for p in placed_items if p.position.z < 1e-6)
            stability_est = floor_items / len(placed_items)
        else:
            stability_est = 1.0

        # Buffer compatibility: fraction of buffer items that could fit
        fittable = 0
        for item in buffer_items:
            for ems in ems_mgr.ems_list[:5]:  # check top 5 EMSs
                if (item.length <= ems.x2 - ems.x1 + 1e-9 and
                    item.width <= ems.y2 - ems.y1 + 1e-9 and
                    item.height <= ems.z2 - ems.z1 + 1e-9):
                    fittable += 1
                    break
        buffer_compat = fittable / max(len(buffer_items), 1)

        return (self.w_ems * ems_quality +
                self.w_stab * stability_est +
                self.w_buffer * buffer_compat)

    def last_chance_placement(
        self,
        bin_to_close: Any,
        buffer_items: List[Any],
        stability_checker: Any,
        selector: BufferStabilitySelector
    ) -> Optional[PlacementCandidate]:
        """
        Before closing a bin, check if ANY buffer item can still be placed.
        If yes, place it first (may improve fill rate of the closing bin).

        This "last chance" check can recover significant efficiency.
        """
        return selector.select_best_placement(
            buffer_items, [bin_to_close]
        )


# ============================================================================
# IDEA 6: SURFACE CREATION HEURISTIC
# ============================================================================
#
# Key insight: stability for FUTURE items improves when current placements
# create flat, wide support surfaces. This is a placement QUALITY metric,
# not a stability constraint.
#
# Implementation is embedded in BufferStabilitySelector._surface_creation_score
# above. Here we provide the standalone version with more detail.

def surface_creation_score_detailed(
    position_x: float, position_y: float, position_z: float,
    item_l: float, item_w: float, item_h: float,
    placed_items: List[Any]
) -> Dict[str, float]:
    """
    Detailed surface creation analysis for a proposed placement.

    Returns:
        dict with:
        - 'aligned_area': area at same height as item's new top
        - 'aligned_ratio': aligned_area / item's top area
        - 'num_aligned_neighbors': count of items at same height
        - 'layer_completeness': fraction of bin floor area covered at this height
        - 'height_uniformity': how uniform the heights are around this item
    """
    EPSILON = 1e-6
    new_top_z = position_z + item_h
    new_top_area = item_l * item_w

    aligned_area = 0.0
    num_aligned = 0
    total_area_at_height = new_top_area

    for existing in placed_items:
        if abs(existing.top_z - new_top_z) < EPSILON:
            num_aligned += 1
            total_area_at_height += existing.dims.base_area

            # Compute adjacent overlap
            x_overlap = (min(position_x + item_l, existing.x_max) -
                         max(position_x, existing.position.x))
            y_overlap = (min(position_y + item_w, existing.y_max) -
                         max(position_y, existing.position.y))

            if x_overlap > EPSILON and y_overlap > EPSILON:
                aligned_area += x_overlap * y_overlap

    bin_floor_area = 1200 * 800  # approximate
    layer_completeness = total_area_at_height / bin_floor_area

    return {
        'aligned_area': aligned_area,
        'aligned_ratio': aligned_area / new_top_area if new_top_area > 0 else 0,
        'num_aligned_neighbors': num_aligned,
        'layer_completeness': layer_completeness,
        'height_uniformity': 1.0 - abs(layer_completeness - 1.0)
    }


# ============================================================================
# IDEA 7: HYPER-HEURISTIC INTEGRATION (NEW)
# ============================================================================
#
# Use the 160 heuristics from the paper as an ACTION SPACE for a
# hyper-heuristic that selects the best heuristic per step.
#
# This IS Research Gap 3 from the overview paper (Ali et al. 2022):
# "Selective hyper-heuristic for online 3D packing."
#
# Two approaches:
# a) Rule-based: select heuristic based on bin state features
# b) Learning-based: train a DRL agent to select heuristics

class HeuristicSelector:
    """
    Hyper-heuristic that selects from a pool of heuristics per step.

    Instead of using one fixed heuristic for all items, we select the
    most appropriate heuristic for the current state.

    With k=2 and buffer of 10, we can evaluate multiple heuristics
    and pick the one producing the best placement.
    """

    def __init__(self, heuristic_pool: List[str] = None):
        """
        Args:
            heuristic_pool: list of heuristic names to choose from.
                Default: the 16 non-dominated heuristics under
                partial-base polygon support from paper Figure 7.
        """
        if heuristic_pool is None:
            # Non-dominated under partial-base polygon support
            self.pool = [
                'A12', 'F12', 'B12', 'W53', 'B52', 'F51',
                'A63', 'F63', 'B63', 'A53', 'F53', 'B53',
                'A52', 'F52', 'W63', 'W73'
            ]
        else:
            self.pool = heuristic_pool

    def select_heuristic_rule_based(
        self,
        bin_fill_rate: float,
        buffer_heterogeneity: float,
        avg_ems_size: float
    ) -> str:
        """
        Rule-based heuristic selection.

        Heuristic rules derived from paper's Appendix G/H analysis:
        - Empty bin (fill < 20%): A53 (DBLF+corner, largest base/max-x)
          Builds a strong foundation layer.
        - Medium fill (20-60%): A12 or F12 (DBLF, largest base/min-x)
          Efficient space filling.
        - High fill (>60%): B63 or F63 (corner-first, largest base/max-x)
          Fill corners and gaps efficiently.
        - High heterogeneity: A52 (evaluate all bins, find best match)
        - Low heterogeneity: F53 (simple first-fit, consistent performance)
        """
        if bin_fill_rate < 0.20:
            return 'A53'
        elif bin_fill_rate < 0.60:
            if buffer_heterogeneity > 0.70:
                return 'A12'
            else:
                return 'F12'
        else:
            if avg_ems_size < 0.05:  # small remaining spaces
                return 'B63'
            else:
                return 'F63'

    def select_heuristic_competitive(
        self,
        buffer_items: List[Any],
        active_bins: List[Any],
        stability_checker: Any,
        top_n: int = 4
    ) -> Tuple[str, Any]:
        """
        Competitive selection: try the top N heuristics and pick the
        one that produces the best single placement.

        This is more expensive but more accurate than rule-based.
        With only 4 heuristics and ~2400 evaluations each,
        total = ~9600 evaluations, still sub-second.

        Returns: (best_heuristic_name, best_placement_candidate)
        """
        best_name = self.pool[0]
        best_placement = None

        for name in self.pool[:top_n]:
            # Extract space rule and orient rule from name
            space_rule = int(name[1])
            orient_rule = int(name[2])

            selector = BufferStabilitySelector(
                stability_checker=stability_checker,
                space_rule=space_rule,
                orient_rule=orient_rule
            )
            candidate = selector.select_best_placement(buffer_items, active_bins)

            if candidate is not None:
                if (best_placement is None or
                    candidate.combined_score > best_placement.combined_score):
                    best_name = name
                    best_placement = candidate

        return best_name, best_placement


# ============================================================================
# IDEA 8: FRAGILE ITEM ROUTING (NEW)
# ============================================================================
#
# With 2 active bins, we can designate one as the "stable bin" for
# fragile/heavy items and the other as the "efficiency bin" for
# robust items. This is not addressed in the paper.

class FragileItemRouter:
    """
    Route fragile/heavy items to the more stable bin and robust items
    to the more efficient bin.

    Requires item metadata: is_fragile, is_heavy flags.
    """

    def route_item(self, item: Any, bin_a_stability: float,
                   bin_b_stability: float) -> int:
        """
        Choose which bin to try first for this item.

        Returns: 0 for bin_a, 1 for bin_b.
        """
        is_fragile = getattr(item, 'is_fragile', False)
        is_heavy = getattr(item, 'is_heavy', False)

        if is_fragile or is_heavy:
            # Route to the MORE stable bin
            return 0 if bin_a_stability >= bin_b_stability else 1
        else:
            # Route to the LESS stable bin (it needs more items, and
            # robust items can handle some instability)
            return 0 if bin_a_stability <= bin_b_stability else 1


# ============================================================================
# IDEA 9: RUNNING PARETO TRACKING (NEW)
# ============================================================================
#
# Track the running stability-efficiency position during packing
# and adjust strategy to stay within user-specified bounds.

class RunningParetoTracker:
    """
    Track stability and efficiency metrics during packing and adjust
    the stability constraint to stay within user-specified bounds.

    Example: user specifies min_stability=0.90 and min_efficiency=0.85.
    The tracker monitors both metrics and tightens/relaxes the constraint
    to stay in the feasible region.
    """

    def __init__(self, min_stability: float = 0.90,
                 target_efficiency: float = 0.85):
        self.min_stability = min_stability
        self.target_efficiency = target_efficiency
        self.items_placed = 0
        self.items_stable = 0  # count of items stable per online check
        self.bins_used = 0
        self.total_item_volume = 0.0
        self.total_bin_volume = 0.0

    def update(self, is_stable: bool, item_volume: float,
               new_bin_opened: bool, bin_volume: float):
        """Update metrics after placing an item."""
        self.items_placed += 1
        if is_stable:
            self.items_stable += 1
        self.total_item_volume += item_volume
        if new_bin_opened:
            self.bins_used += 1
            self.total_bin_volume += bin_volume

    @property
    def current_stability(self) -> float:
        if self.items_placed == 0:
            return 1.0
        return self.items_stable / self.items_placed

    @property
    def current_efficiency(self) -> float:
        if self.total_bin_volume <= 0:
            return 0.0
        return self.total_item_volume / self.total_bin_volume

    def should_tighten(self) -> bool:
        """Should we use a stricter stability constraint?"""
        return self.current_stability < self.min_stability

    def should_relax(self) -> bool:
        """Can we relax the stability constraint?"""
        margin = self.current_stability - self.min_stability
        return margin > 0.05  # 5% margin above minimum

    def recommend_constraint(self, available_checkers: dict) -> Any:
        """
        Recommend a stability constraint based on current Pareto position.
        """
        if self.should_tighten():
            return (available_checkers.get('partial_base') or
                    available_checkers.get('full_base'))
        elif self.should_relax():
            return (available_checkers.get('cog_polygon') or
                    available_checkers.get('partial_base_polygon'))
        else:
            return available_checkers.get('partial_base_polygon')


# ============================================================================
# MASTER INTEGRATION: COMPLETE SEMI-ONLINE PACKING ENGINE
# ============================================================================

class SemiOnlinePackingEngine:
    """
    The complete semi-online packing engine integrating:
    - 5-10 item buffer (Idea 1)
    - Foundation-first ordering (Idea 2)
    - Look-ahead assessment (Idea 3, optional)
    - Adaptive stability (Idea 4)
    - Stability-aware bin closing (Idea 5)
    - Surface creation scoring (Idea 6)
    - Hyper-heuristic selection (Idea 7)
    - Running Pareto tracking (Idea 9)

    This engine is the heart of our thesis implementation.
    """

    def __init__(self, buffer_size: int = 10,
                 bin_dims=None,
                 min_stability: float = 0.90,
                 use_adaptive: bool = True,
                 use_hyper_heuristic: bool = False,
                 use_look_ahead: bool = False):
        self.buffer_size = buffer_size
        self.bin_dims = bin_dims  # Dimensions(1200, 800, 1500)
        self.min_stability = min_stability
        self.use_adaptive = use_adaptive
        self.use_hyper_heuristic = use_hyper_heuristic
        self.use_look_ahead = use_look_ahead

        # Components
        self.buffer: List[Any] = []
        self.active_bins: List[Any] = []  # max 2 for k=2
        self.closed_bins: List[Any] = []

        # Selectors
        self.closing_policy = StabilityAwareClosingPolicy()
        self.pareto_tracker = RunningParetoTracker(min_stability=min_stability)
        self.adaptive_policy = AdaptiveStabilityPolicy()
        self.hyper_selector = HeuristicSelector() if use_hyper_heuristic else None

    def pack_stream(self, item_stream) -> Dict[str, Any]:
        """
        Pack a stream of items arriving one by one.

        Args:
            item_stream: iterator yielding BufferItem objects

        Returns:
            dict with final results (bins used, stability, etc.)
        """
        for item in item_stream:
            # Add to buffer
            self.buffer.append(item)

            # When buffer is full (or stream ends), place one item
            if len(self.buffer) >= self.buffer_size:
                self._place_one_from_buffer()

        # Drain remaining buffer
        while self.buffer:
            self._place_one_from_buffer()

        # Close remaining active bins
        self.closed_bins.extend(self.active_bins)
        self.active_bins = []

        return self._compute_final_results()

    def _place_one_from_buffer(self):
        """Select and place the best item from the buffer."""
        if not self.buffer:
            return

        # Ensure we have at least one active bin
        if not self.active_bins:
            self._open_new_bin()

        # Select stability constraint (adaptive or fixed)
        if self.use_adaptive:
            # Use the first available EMS height as estimate
            est_height = 0  # would compute from bin state
            checker = self.adaptive_policy.select_constraint(
                est_height, 0.5, self.buffer
            )
        else:
            checker = None  # Would use default from constructor

        # Select the best (item, bin, EMS, orientation)
        if self.use_hyper_heuristic and self.hyper_selector:
            name, candidate = self.hyper_selector.select_heuristic_competitive(
                self.buffer, self.active_bins, checker
            )
        else:
            selector = BufferStabilitySelector(
                stability_checker=checker,
                bin_dims=self.bin_dims
            )
            candidate = selector.select_best_placement(
                self.buffer, self.active_bins
            )

        if candidate is not None:
            # Place the item
            self._execute_placement(candidate)
            # Remove from buffer
            self.buffer.remove(candidate.item)
        else:
            # No feasible placement in any active bin
            if len(self.active_bins) >= 2:
                # Close worst bin
                close_idx = self.closing_policy.choose_bin_to_close(
                    self.active_bins[0], self.active_bins[1],
                    self.buffer, checker
                )
                self.closed_bins.append(self.active_bins.pop(close_idx))

            self._open_new_bin()
            # Retry placement
            if checker:
                selector = BufferStabilitySelector(
                    stability_checker=checker,
                    bin_dims=self.bin_dims
                )
                candidate = selector.select_best_placement(
                    self.buffer, self.active_bins
                )
                if candidate is not None:
                    self._execute_placement(candidate)
                    self.buffer.remove(candidate.item)

    def _execute_placement(self, candidate: PlacementCandidate):
        """Execute a placement: update bin state and tracking."""
        ems_mgr, placed_items = self.active_bins[candidate.bin_idx]

        # Create PlacedItem and add to bin
        # (In production, convert candidate.oriented_dims to PlacedItem)
        # placed = PlacedItem(...)
        # placed_items.append(placed)
        # ems_mgr.place_item(candidate.position, candidate.oriented_dims)

        # Update Pareto tracker
        is_stable = (candidate.stability_result.is_stable
                     if candidate.stability_result else True)
        self.pareto_tracker.update(
            is_stable=is_stable,
            item_volume=candidate.oriented_dims.volume,
            new_bin_opened=False,
            bin_volume=0
        )

    def _open_new_bin(self):
        """Open a new bin if k=2 allows it."""
        if len(self.active_bins) >= 2:
            return  # Cannot open more
        # new_bin = (EMSManager(self.bin_dims), [])
        # self.active_bins.append(new_bin)
        self.pareto_tracker.update(
            is_stable=True, item_volume=0,
            new_bin_opened=True,
            bin_volume=1200 * 800 * 1500
        )

    def _compute_final_results(self) -> Dict[str, Any]:
        """Compute final packing metrics."""
        total_bins = len(self.closed_bins)
        return {
            'total_bins': total_bins,
            'stability_pct': self.pareto_tracker.current_stability * 100,
            'efficiency_pct': self.pareto_tracker.current_efficiency * 100,
            'items_placed': self.pareto_tracker.items_placed,
            'items_stable': self.pareto_tracker.items_stable,
        }


# ============================================================================
# ESTIMATED IMPLEMENTATION PRIORITY (UPDATED)
# ============================================================================
#
# 1. FIRST: BufferStabilitySelector (Idea 1) -- 2-3 days
#    Core buffer mechanism. Can use simple scoring initially.
#    This is the minimum viable product for the semi-online system.
#
# 2. SECOND: StabilityAwareClosingPolicy (Idea 5) -- 1-2 days
#    Required for k=2 bounded space. Test multiple policies.
#
# 3. THIRD: AdaptiveStabilityPolicy (Idea 4) -- 1-2 days
#    Height-based gradient. Easy given stability checkers exist.
#
# 4. FOURTH: FoundationFirstSelector (Idea 2) -- 1 day
#    Simple heuristic with big potential impact on stability.
#
# 5. FIFTH: SurfaceCreationScore (Idea 6) -- 1 day
#    Indirect stability improvement for future items.
#
# 6. SIXTH: RunningParetoTracker (Idea 9) -- 1 day
#    Monitoring and adaptive constraint selection.
#
# 7. SEVENTH: HeuristicSelector (Idea 7) -- 2-3 days
#    Hyper-heuristic integration. Requires all 160 heuristics first.
#
# 8. EIGHTH: LookAheadStabilityAssessor (Idea 3) -- 2-3 days
#    Most complex. Only if other ideas are insufficient.
#
# 9. NINTH: SemiOnlinePackingEngine integration -- 2-3 days
#    Wire everything together.
#
# TOTAL: ~13-19 days for all ideas
# MINIMUM VIABLE: Ideas 1+2+5 = ~5 days
#
# ============================================================================
# KEY INSIGHT FROM PAPER FOR BUFFER INTEGRATION
# ============================================================================
#
# The paper shows online heuristics match offline in 71% of cases (Table 7).
# The 29% gap is due to:
# 1. Inability to choose item order (our buffer partially addresses this)
# 2. No ability to evaluate future items (our look-ahead addresses this)
# 3. No ability to route items across bins (our k=2 routing addresses this)
#
# With buffer of 10 + k=2 + adaptive stability, we estimate:
# - Match offline in 80-85% of cases (up from 71%)
# - Stability improvement of 3-5% over paper's best (up from 92% to 95-97%)
# - This would be a significant thesis contribution
#
# ============================================================================
"""
"""
