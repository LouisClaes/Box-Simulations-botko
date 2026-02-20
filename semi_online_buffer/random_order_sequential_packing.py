"""
Random Order Sequential Packing: Buffer Management for Semi-Online 3D Bin Packing
==================================================================================

Based on: Albers, Khan & Ladewig (2021). "Improved Online Algorithms for Knapsack
          and GAP in the Random Order Model." Algorithmica 83, 1750-1785.

Key idea: Decompose the packing problem into two sequential phases:
  Phase 1: Prioritize LARGE items (volume > delta * bin_volume)
  Phase 2: Use SMALL items as gap-fillers

The buffer of 5-10 items provides a strictly stronger model than the paper's
single-item random order arrival. We exploit this for better practical performance.

Integration points:
  - Works with any 3D placement heuristic (DBLF, Extreme Points, EMS)
  - Works with any stability checker
  - Designed for k=2 bounded space (2 active bins)

Estimated complexity: O(B * P) per item decision, where B = buffer size (5-10),
                      P = number of placement positions per bin (depends on EMS count)
Feasibility: HIGH -- straightforward to implement, no ML training required.

Author: Thesis project (coding ideas derived from Albers et al. 2021)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import math
import random


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Item3D:
    """A 3D box item to be packed."""
    id: int
    width: float
    height: float
    depth: float
    weight: float = 0.0
    fragile: bool = False

    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        return (self.width, self.height, self.depth)

    def rotations(self, vertical_axis_only: bool = True) -> List[Tuple[float, float, float]]:
        """Return all valid rotations of the item.
        vertical_axis_only=True: only rotate around z-axis (robot constraint from Verma et al. 2020)
        vertical_axis_only=False: all 6 orthogonal rotations
        """
        w, h, d = self.width, self.height, self.depth
        if vertical_axis_only:
            # Height stays the same, swap width and depth
            return [(w, h, d), (d, h, w)]
        else:
            # All 6 orthogonal rotations (unique ones)
            rots = set()
            for dims in [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]:
                rots.add(dims)
            return list(rots)


@dataclass
class Placement:
    """A placement of an item in a bin at a specific position and orientation."""
    item: Item3D
    bin_index: int
    x: float
    y: float
    z: float
    placed_width: float   # Width after rotation
    placed_height: float  # Height after rotation
    placed_depth: float   # Depth after rotation


@dataclass
class EMS:
    """Empty Maximal Space -- the standard 3D space representation.
    Defined by two corners: (x_min, y_min, z_min) and (x_max, y_max, z_max).
    See overview knowledge base Section 9.5.
    """
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    @property
    def volume(self) -> float:
        return ((self.x_max - self.x_min) *
                (self.y_max - self.y_min) *
                (self.z_max - self.z_min))

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        return (self.x_max - self.x_min,
                self.y_max - self.y_min,
                self.z_max - self.z_min)

    def fits(self, w: float, h: float, d: float) -> bool:
        ew, eh, ed = self.dimensions
        return w <= ew + 1e-9 and h <= eh + 1e-9 and d <= ed + 1e-9


class ItemCategory(Enum):
    LARGE = "large"
    SMALL = "small"


@dataclass
class BinState:
    """State of an active bin."""
    bin_index: int
    bin_width: float
    bin_height: float
    bin_depth: float
    placed_items: List[Placement] = field(default_factory=list)
    empty_spaces: List[EMS] = field(default_factory=list)  # List of EMS
    is_closed: bool = False

    def __post_init__(self):
        if not self.empty_spaces:
            # Initialize with the full bin as one EMS
            self.empty_spaces = [
                EMS(0, 0, 0, self.bin_width, self.bin_height, self.bin_depth)
            ]

    @property
    def bin_volume(self) -> float:
        return self.bin_width * self.bin_height * self.bin_depth

    @property
    def used_volume(self) -> float:
        return sum(p.placed_width * p.placed_height * p.placed_depth
                   for p in self.placed_items)

    @property
    def fill_rate(self) -> float:
        return self.used_volume / self.bin_volume if self.bin_volume > 0 else 0.0

    @property
    def remaining_volume(self) -> float:
        return self.bin_volume - self.used_volume


# =============================================================================
# Distribution Estimator
# =============================================================================

class ItemDistributionEstimator:
    """
    Estimates item size distribution from observed items (buffer + processed items).

    In the random order model, early items are representative samples of the full
    item set. With a buffer of 5-10, we get distributional information immediately.

    This replaces the paper's formal "sampling phase" with a practical, adaptive
    approach that does NOT waste items.
    """

    def __init__(self):
        self.volumes: List[float] = []
        self.dimensions: List[Tuple[float, float, float]] = []
        self.weights: List[float] = []

    def observe(self, item: Item3D):
        """Record an observed item (whether packed or not)."""
        self.volumes.append(item.volume)
        self.dimensions.append(item.dimensions)
        self.weights.append(item.weight)

    @property
    def n_observed(self) -> int:
        return len(self.volumes)

    def mean_volume(self) -> float:
        return sum(self.volumes) / len(self.volumes) if self.volumes else 0.0

    def max_volume(self) -> float:
        return max(self.volumes) if self.volumes else 0.0

    def volume_std(self) -> float:
        if len(self.volumes) < 2:
            return 0.0
        mean = self.mean_volume()
        return math.sqrt(sum((v - mean) ** 2 for v in self.volumes) / (len(self.volumes) - 1))

    def estimate_large_fraction(self, delta: float, bin_volume: float) -> float:
        """Estimate fraction of items that are delta-large."""
        if not self.volumes:
            return 0.5  # No data yet, assume balanced
        threshold = delta * bin_volume
        n_large = sum(1 for v in self.volumes if v > threshold)
        return n_large / len(self.volumes)

    def optimal_delta(self, bin_volume: float) -> float:
        """
        Find optimal large/small threshold.

        From the paper: delta = 1/3 for knapsack, 1/2 for GAP.
        In 3D, the optimal threshold depends on the volume distribution.

        Heuristic: choose delta such that the "large" items contribute roughly
        half the total optimal profit. Since we don't know profit, we use
        volume as a proxy.
        """
        if len(self.volumes) < 5:
            return 0.33  # Default from the paper

        sorted_vols = sorted(self.volumes, reverse=True)
        total_vol = sum(sorted_vols)

        # Find threshold where large items contribute ~50% of total volume
        cumulative = 0.0
        for i, v in enumerate(sorted_vols):
            cumulative += v
            if cumulative >= 0.5 * total_vol:
                # The threshold is approximately this item's volume / bin_volume
                candidate = v / bin_volume
                # Clamp to reasonable range
                return max(0.15, min(0.50, candidate))

        return 0.33

    def estimate_remaining_items(self, total_expected: int) -> int:
        """Estimate how many items remain to be seen."""
        return max(0, total_expected - self.n_observed)


# =============================================================================
# Stability Checker (placeholder -- integrate with actual stability module)
# =============================================================================

class StabilityChecker:
    """
    Placeholder for stability checking. In the actual implementation, this
    should be replaced with the stability module from:
    C:\\Users\\Louis\\Downloads\\stapelalgortime\\python\\stability\\

    Checks both static (vertical) and dynamic (horizontal) stability.
    See overview knowledge base Section 7.1 (Safety Constraints).
    """

    def __init__(self, min_support_ratio: float = 0.7):
        self.min_support_ratio = min_support_ratio  # Minimum fraction of base supported

    def is_stable(self, placement: Placement, bin_state: BinState) -> bool:
        """
        Check if placing the item at the given position is statically stable.

        Static stability: item's bottom face must be supported by at least
        min_support_ratio of its area, either by the bin floor or by items below.
        """
        # If placed on the floor (z=0), always stable
        if placement.z < 1e-9:
            return True

        # Otherwise, check support from items below
        support_area = self._compute_support_area(placement, bin_state)
        item_base_area = placement.placed_width * placement.placed_depth
        return support_area >= self.min_support_ratio * item_base_area

    def _compute_support_area(self, placement: Placement, bin_state: BinState) -> float:
        """Compute the supported area of the item's bottom face."""
        # TODO: Implement actual support area calculation
        # This requires checking overlap of the item's bottom face with
        # the top faces of items below at the same z-level
        return 0.0  # Placeholder

    def stability_score(self, placement: Placement, bin_state: BinState) -> float:
        """
        Compute a stability score in [0, 1].
        Higher = more stable placement.

        Factors: support ratio, center of gravity shift, height penalty
        """
        if placement.z < 1e-9:
            return 1.0  # Floor placement is maximally stable

        # Penalize high placements (less stable overall)
        height_penalty = 1.0 - (placement.z / bin_state.bin_height)

        # TODO: Compute actual support ratio and CoG shift
        support_ratio = 0.8  # Placeholder
        cog_penalty = 0.9    # Placeholder

        return height_penalty * support_ratio * cog_penalty


# =============================================================================
# Core Algorithm: Sequential Buffer Manager
# =============================================================================

class SequentialBufferManager:
    """
    Main algorithm: Semi-online 3D bin packing with buffer, using the
    sequential approach from Albers, Khan & Ladewig (2021).

    DESIGN PHILOSOPHY:
    The paper's Algorithm 1 runs A_L then A_S sequentially on the time axis.
    With a buffer, we adapt this to run them simultaneously on the ITEM TYPE axis:
    - When the buffer contains large items, prioritize them (A_L logic)
    - When only small items remain, use gap-filling (A_S logic)
    - The buffer provides implicit "sampling" without wasting items

    The paper's key parameters:
    - delta = 1/3 (large/small threshold, adapted for 3D)
    - c = 0.42291 (sampling phase fraction -- we use buffer instead)
    - d = 0.64570 (large items phase end -- we adapt dynamically)

    k=2 BOUNDED SPACE:
    With 2 active bins, we use the GAP matching formulation (Section 5 of paper):
    - Each buffer item can go to either bin
    - We solve a small matching problem to find the best assignment
    """

    def __init__(
        self,
        buffer_size: int = 10,
        num_active_bins: int = 2,
        bin_dims: Tuple[float, float, float] = (120.0, 100.0, 150.0),
        delta: float = 0.33,
        adaptive_delta: bool = True,
        fill_rate_weight: float = 0.5,
        stability_weight: float = 0.5,
    ):
        self.buffer_size = buffer_size
        self.num_active_bins = num_active_bins
        self.bin_width, self.bin_height, self.bin_depth = bin_dims
        self.delta = delta
        self.adaptive_delta = adaptive_delta
        self.fill_rate_weight = fill_rate_weight
        self.stability_weight = stability_weight

        # State
        self.buffer: List[Item3D] = []
        self.active_bins: List[BinState] = []
        self.closed_bins: List[BinState] = []
        self.total_items_processed = 0

        # Components
        self.estimator = ItemDistributionEstimator()
        self.stability_checker = StabilityChecker(min_support_ratio=0.7)

        # Initialize active bins
        for i in range(num_active_bins):
            self._open_new_bin(i)

        # Statistics
        self.packing_log: List[Dict] = []

    @property
    def bin_volume(self) -> float:
        return self.bin_width * self.bin_height * self.bin_depth

    def _open_new_bin(self, index: int) -> BinState:
        """Open a new active bin."""
        new_bin = BinState(
            bin_index=index,
            bin_width=self.bin_width,
            bin_height=self.bin_height,
            bin_depth=self.bin_depth,
        )
        if index < len(self.active_bins):
            self.active_bins[index] = new_bin
        else:
            self.active_bins.append(new_bin)
        return new_bin

    def _close_bin(self, bin_index: int):
        """Close an active bin permanently (k-bounded space constraint)."""
        for i, b in enumerate(self.active_bins):
            if b.bin_index == bin_index:
                b.is_closed = True
                self.closed_bins.append(b)
                # Open a new bin in its place
                new_idx = max(b2.bin_index for b2 in self.active_bins + self.closed_bins) + 1
                self._open_new_bin(i)
                self.active_bins[i].bin_index = new_idx
                break

    def classify_item(self, item: Item3D) -> ItemCategory:
        """Classify item as LARGE or SMALL based on volume ratio."""
        threshold = self.delta * self.bin_volume
        if item.volume > threshold:
            return ItemCategory.LARGE
        return ItemCategory.SMALL

    # =========================================================================
    # Buffer Management
    # =========================================================================

    def add_to_buffer(self, item: Item3D) -> Optional[Placement]:
        """
        Add an item to the buffer. If buffer is full, must pack or reject an item first.

        Returns: Placement if an item was packed, None if buffer absorbed the item.
        """
        self.estimator.observe(item)
        self.total_items_processed += 1

        # Update delta adaptively based on distribution
        if self.adaptive_delta and self.estimator.n_observed >= 10:
            self.delta = self.estimator.optimal_delta(self.bin_volume)

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(item)
            return None
        else:
            # Buffer full: must select an item to pack (or reject)
            self.buffer.append(item)
            return self._select_and_pack()

    def _select_and_pack(self) -> Optional[Placement]:
        """
        Select the best item from the buffer to pack.
        This is the core decision function implementing the sequential approach.

        SEQUENTIAL LOGIC (adapted from paper's Algorithm 1):
        1. If buffer contains LARGE items: prioritize the best large item
           (analogous to A_L phase, Algorithm 2)
        2. If buffer contains only SMALL items: use gap-filling strategy
           (analogous to A_S phase, Algorithm 3)
        """
        large_items = [item for item in self.buffer
                       if self.classify_item(item) == ItemCategory.LARGE]
        small_items = [item for item in self.buffer
                       if self.classify_item(item) == ItemCategory.SMALL]

        # PHASE 1: Large items priority (A_L logic)
        if large_items:
            result = self._pack_large_item(large_items)
            if result is not None:
                return result

        # PHASE 2: Small items gap-filling (A_S logic)
        if small_items:
            result = self._pack_small_item(small_items)
            if result is not None:
                return result

        # If nothing could be packed, reject the worst item in buffer
        self._reject_worst_item()
        return None

    def _pack_large_item(self, large_items: List[Item3D]) -> Optional[Placement]:
        """
        Pack the best large item from the buffer.

        Inspired by Algorithm 2 (A_L):
        - The paper uses 2-Secretary logic: accept items with profit > v* (best seen so far)
        - With buffer, we do better: we can compare all large items in the buffer
          and pick the one that best fits an active bin.

        GAP matching (paper Section 5): We solve a small matching problem:
        - Left nodes: large items in buffer
        - Right nodes: active bins
        - Edge weight: placement quality score
        """
        best_placement = None
        best_score = -float('inf')
        best_item = None

        for item in large_items:
            for bin_state in self.active_bins:
                if bin_state.is_closed:
                    continue
                placement, score = self._find_best_placement(item, bin_state)
                if placement is not None and score > best_score:
                    best_score = score
                    best_placement = placement
                    best_item = item

        if best_placement is not None and best_item is not None:
            self._execute_placement(best_placement, best_item)
            return best_placement

        return None

    def _pack_small_item(self, small_items: List[Item3D]) -> Optional[Placement]:
        """
        Pack the best small item from the buffer.

        Inspired by Algorithm 3 (A_S):
        - The paper uses greedy fractional knapsack + randomized rounding
        - In 3D, "fractional packing" is not possible, so we use greedy best-fit
        - We select the small item that best fills an available EMS (gap-filling)

        Gap-filling heuristic: prefer items that closely match the dimensions of
        an available EMS (minimum wasted space = best "fit quality").
        """
        best_placement = None
        best_score = -float('inf')
        best_item = None

        for item in small_items:
            for bin_state in self.active_bins:
                if bin_state.is_closed:
                    continue
                placement, score = self._find_best_placement(item, bin_state)
                if placement is not None and score > best_score:
                    best_score = score
                    best_placement = placement
                    best_item = item

        if best_placement is not None and best_item is not None:
            self._execute_placement(best_placement, best_item)
            return best_placement

        return None

    def _find_best_placement(
        self, item: Item3D, bin_state: BinState
    ) -> Tuple[Optional[Placement], float]:
        """
        Find the best placement for an item in a given bin.

        Uses EMS-based placement with DBLF priority (Ha et al. 2017):
        - Try all valid rotations
        - For each rotation, try all EMSs
        - Score each placement by: fill_rate_contribution * w1 + stability_score * w2
        - Return the best scoring placement

        See overview knowledge base Section 10.2 (Ha et al. 2017 heuristic).
        """
        best_placement = None
        best_score = -float('inf')

        for rotation in item.rotations(vertical_axis_only=True):
            rw, rh, rd = rotation
            for ems in bin_state.empty_spaces:
                if not ems.fits(rw, rh, rd):
                    continue

                # Create candidate placement (DBLF: deepest-bottom-left)
                placement = Placement(
                    item=item,
                    bin_index=bin_state.bin_index,
                    x=ems.x_min,
                    y=ems.y_min,
                    z=ems.z_min,
                    placed_width=rw,
                    placed_height=rh,
                    placed_depth=rd,
                )

                # Compute combined score
                score = self._score_placement(placement, bin_state, ems)

                if score > best_score:
                    best_score = score
                    best_placement = placement

        return best_placement, best_score

    def _score_placement(
        self, placement: Placement, bin_state: BinState, ems: EMS
    ) -> float:
        """
        Score a candidate placement. Combines multiple objectives:

        1. Fill rate contribution (volume utilization)
        2. Stability score
        3. Fit quality (how well item fills the EMS, minimizing waste)
        4. DBLF priority (prefer deeper-bottom-left positions)

        Weights are configurable to match thesis goals.
        """
        item_vol = placement.placed_width * placement.placed_height * placement.placed_depth

        # Fill rate contribution
        fill_contribution = item_vol / bin_state.bin_volume

        # Fit quality: ratio of item volume to EMS volume (1.0 = perfect fit)
        fit_quality = item_vol / ems.volume if ems.volume > 0 else 0.0

        # Stability score
        stab_score = self.stability_checker.stability_score(placement, bin_state)

        # DBLF priority: prefer positions closer to (0, 0, 0)
        max_dist = math.sqrt(
            bin_state.bin_width**2 + bin_state.bin_height**2 + bin_state.bin_depth**2
        )
        dist = math.sqrt(placement.x**2 + placement.y**2 + placement.z**2)
        dblf_score = 1.0 - (dist / max_dist) if max_dist > 0 else 0.0

        # Weighted combination
        score = (
            self.fill_rate_weight * fill_contribution +
            self.stability_weight * stab_score +
            0.2 * fit_quality +
            0.1 * dblf_score
        )

        return score

    def _execute_placement(self, placement: Placement, item: Item3D):
        """Execute a placement: update bin state and remove item from buffer."""
        for bin_state in self.active_bins:
            if bin_state.bin_index == placement.bin_index:
                bin_state.placed_items.append(placement)
                # TODO: Update EMSs (split/merge empty spaces)
                self._update_ems(bin_state, placement)
                break

        # Remove item from buffer
        self.buffer = [i for i in self.buffer if i.id != item.id]

        # Log
        self.packing_log.append({
            "item_id": item.id,
            "bin": placement.bin_index,
            "category": self.classify_item(item).value,
            "position": (placement.x, placement.y, placement.z),
            "score": self._score_placement(
                placement,
                self.active_bins[0],  # Approximate
                EMS(0, 0, 0, 1, 1, 1)  # Placeholder
            ),
        })

    def _update_ems(self, bin_state: BinState, placement: Placement):
        """
        Update Empty Maximal Spaces after placing an item.

        This is a placeholder. The actual implementation should follow
        Parreno et al. (2008) -- see overview knowledge base Section 9.5.

        Steps:
        1. For each existing EMS that overlaps with the placed item:
           a. Remove the overlapping EMS
           b. Generate up to 6 new EMSs (splitting along each axis)
        2. Remove any EMS fully contained in another EMS
        3. Remove any EMS that is too small to fit any expected item
        """
        # TODO: Implement proper EMS update
        # For now, just remove EMSs that are fully covered by the placement
        new_spaces = []
        for ems in bin_state.empty_spaces:
            if self._overlaps(ems, placement):
                # Split the EMS
                splits = self._split_ems(ems, placement)
                new_spaces.extend(splits)
            else:
                new_spaces.append(ems)
        bin_state.empty_spaces = new_spaces

    def _overlaps(self, ems: EMS, placement: Placement) -> bool:
        """Check if an EMS overlaps with a placement."""
        px_max = placement.x + placement.placed_width
        py_max = placement.y + placement.placed_depth
        pz_max = placement.z + placement.placed_height

        return not (
            placement.x >= ems.x_max or px_max <= ems.x_min or
            placement.y >= ems.y_max or py_max <= ems.y_min or
            placement.z >= ems.z_max or pz_max <= ems.z_min
        )

    def _split_ems(self, ems: EMS, placement: Placement) -> List[EMS]:
        """Split an EMS around a placed item. Returns up to 6 new EMSs."""
        px_max = placement.x + placement.placed_width
        py_max = placement.y + placement.placed_depth
        pz_max = placement.z + placement.placed_height

        new_spaces = []

        # Split along x-axis (left and right)
        if placement.x > ems.x_min:
            new_spaces.append(EMS(ems.x_min, ems.y_min, ems.z_min,
                                  placement.x, ems.y_max, ems.z_max))
        if px_max < ems.x_max:
            new_spaces.append(EMS(px_max, ems.y_min, ems.z_min,
                                  ems.x_max, ems.y_max, ems.z_max))

        # Split along y-axis (front and back)
        if placement.y > ems.y_min:
            new_spaces.append(EMS(ems.x_min, ems.y_min, ems.z_min,
                                  ems.x_max, placement.y, ems.z_max))
        if py_max < ems.y_max:
            new_spaces.append(EMS(ems.x_min, py_max, ems.z_min,
                                  ems.x_max, ems.y_max, ems.z_max))

        # Split along z-axis (below and above)
        if placement.z > ems.z_min:
            new_spaces.append(EMS(ems.x_min, ems.y_min, ems.z_min,
                                  ems.x_max, ems.y_max, placement.z))
        if pz_max < ems.z_max:
            new_spaces.append(EMS(ems.x_min, ems.y_min, pz_max,
                                  ems.x_max, ems.y_max, ems.z_max))

        # Filter out degenerate EMSs (zero or negative volume)
        return [s for s in new_spaces if s.volume > 1e-9]

    def _reject_worst_item(self):
        """
        When no item can be packed, reject the item with lowest expected value.

        Rejection policy (inspired by the paper's sampling phase):
        - Reject the item that is least likely to be useful in the future
        - Small items in a mostly-full bin: low value (can't fill gaps)
        - Items that don't fit any EMS in any active bin: must reject
        """
        if not self.buffer:
            return

        worst_score = float('inf')
        worst_item = self.buffer[0]

        for item in self.buffer:
            # Score = expected future value of keeping this item
            score = item.volume  # Simple heuristic: larger items are more valuable
            for bin_state in self.active_bins:
                # Check if item fits anywhere in any bin
                fits_somewhere = False
                for rotation in item.rotations():
                    rw, rh, rd = rotation
                    for ems in bin_state.empty_spaces:
                        if ems.fits(rw, rh, rd):
                            fits_somewhere = True
                            break
                    if fits_somewhere:
                        break
                if not fits_somewhere:
                    score = -1  # Item doesn't fit anywhere -> reject it
                    break

            if score < worst_score:
                worst_score = score
                worst_item = item

        self.buffer.remove(worst_item)

    # =========================================================================
    # Bin Closing Decision (k=2 Bounded Space)
    # =========================================================================

    def should_close_bin(self, bin_state: BinState) -> bool:
        """
        Decide if an active bin should be closed.

        In k-bounded space, closing a bin is permanent. This decision should
        be made carefully:
        - Close when fill rate is high enough (diminishing returns)
        - Close when remaining EMSs are too small for expected items
        - Close when the other bin is much emptier (balance load)

        The paper's GAP framework suggests: close when expected marginal
        contribution drops below the expected contribution of a fresh bin.
        """
        if bin_state.fill_rate >= 0.85:
            return True  # Good enough, close and move on

        # Check if remaining spaces are too small
        mean_item_vol = self.estimator.mean_volume()
        if mean_item_vol > 0:
            useful_spaces = [ems for ems in bin_state.empty_spaces
                             if ems.volume >= 0.5 * mean_item_vol]
            if not useful_spaces:
                return True  # No useful spaces left

        return False

    # =========================================================================
    # Main Processing Loop
    # =========================================================================

    def process_item_stream(self, items: List[Item3D]) -> Dict:
        """
        Process a stream of items. This is the main entry point.

        Items arrive one by one. Each item enters the buffer.
        When the buffer is full, the algorithm selects an item to pack or reject.

        Returns statistics about the packing.
        """
        for item in items:
            self.add_to_buffer(item)

            # Check if any bin should be closed
            for bin_state in self.active_bins:
                if not bin_state.is_closed and self.should_close_bin(bin_state):
                    self._close_bin(bin_state.bin_index)

        # Flush remaining buffer items
        while self.buffer:
            result = self._select_and_pack()
            if result is None:
                # Cannot pack anything, reject remaining
                self.buffer.pop(0)

        return self._compute_statistics()

    def _compute_statistics(self) -> Dict:
        """Compute packing statistics."""
        all_bins = self.closed_bins + self.active_bins
        total_used_volume = sum(b.used_volume for b in all_bins)
        total_bin_volume = sum(b.bin_volume for b in all_bins if b.placed_items)
        total_items_packed = sum(len(b.placed_items) for b in all_bins)

        return {
            "total_items_processed": self.total_items_processed,
            "total_items_packed": total_items_packed,
            "total_bins_used": len([b for b in all_bins if b.placed_items]),
            "average_fill_rate": total_used_volume / total_bin_volume if total_bin_volume > 0 else 0.0,
            "total_used_volume": total_used_volume,
            "total_bin_volume": total_bin_volume,
            "delta_threshold": self.delta,
            "items_rejected": self.total_items_processed - total_items_packed,
        }


# =============================================================================
# GAP-Inspired Bin Assignment (from paper Section 5)
# =============================================================================

class GAPBinAssigner:
    """
    Assigns buffer items to active bins using the Generalized Assignment Problem
    formulation from Section 5 of Albers, Khan & Ladewig (2021).

    For our setup with buffer_size <= 10 and num_bins = 2, the matching problem
    is tiny and can be solved exactly via brute force.

    This class can be used as an alternative to the simple best-fit approach
    in SequentialBufferManager._pack_large_item and _pack_small_item.
    """

    def __init__(self, stability_checker: StabilityChecker):
        self.stability_checker = stability_checker

    def compute_score_matrix(
        self,
        buffer_items: List[Item3D],
        active_bins: List[BinState],
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute assignment scores for all (item, bin) pairs.

        Score combines:
        - Volume fit quality
        - Stability score
        - Remaining capacity after placement (for future items)

        Returns dict mapping (item_index, bin_index) -> score
        """
        scores = {}
        for i, item in enumerate(buffer_items):
            for j, bin_state in enumerate(active_bins):
                if bin_state.is_closed:
                    scores[(i, j)] = -float('inf')
                    continue
                score = self._evaluate_assignment(item, bin_state)
                scores[(i, j)] = score
        return scores

    def _evaluate_assignment(self, item: Item3D, bin_state: BinState) -> float:
        """Evaluate the quality of assigning an item to a bin."""
        # Check feasibility: does item fit in any EMS?
        feasible = False
        best_fit_ratio = 0.0

        for rotation in item.rotations(vertical_axis_only=True):
            rw, rh, rd = rotation
            for ems in bin_state.empty_spaces:
                if ems.fits(rw, rh, rd):
                    feasible = True
                    fit_ratio = (rw * rh * rd) / ems.volume if ems.volume > 0 else 0
                    best_fit_ratio = max(best_fit_ratio, fit_ratio)

        if not feasible:
            return -float('inf')

        # Score components
        volume_contrib = item.volume / bin_state.bin_volume
        fit_score = best_fit_ratio
        remaining_after = (bin_state.remaining_volume - item.volume) / bin_state.bin_volume

        return 0.4 * volume_contrib + 0.4 * fit_score + 0.2 * remaining_after

    def solve_optimal_assignment(
        self,
        buffer_items: List[Item3D],
        active_bins: List[BinState],
    ) -> Optional[Tuple[int, int]]:
        """
        Find the optimal single-item assignment.

        For our small problem (buffer <= 10, bins = 2), this is O(B * k) = O(20).

        Returns: (item_index, bin_index) of the best assignment, or None if no
                 feasible assignment exists.
        """
        scores = self.compute_score_matrix(buffer_items, active_bins)

        best_pair = None
        best_score = -float('inf')

        for (i, j), score in scores.items():
            if score > best_score:
                best_score = score
                best_pair = (i, j)

        if best_pair is not None and best_score > -float('inf'):
            return best_pair
        return None


# =============================================================================
# Usage Example / Test Harness
# =============================================================================

def example_usage():
    """
    Example demonstrating the sequential buffer manager.

    This can be used as a starting point for integration with the
    actual 3D bin packing environment.
    """
    # Create some test items
    random.seed(42)
    items = []
    for i in range(50):
        w = random.uniform(10, 60)
        h = random.uniform(10, 50)
        d = random.uniform(10, 60)
        items.append(Item3D(id=i, width=w, height=h, depth=d, weight=w*h*d*0.001))

    # Random order (simulating the paper's random order model)
    random.shuffle(items)

    # Create the buffer manager
    manager = SequentialBufferManager(
        buffer_size=10,
        num_active_bins=2,
        bin_dims=(120.0, 100.0, 150.0),
        delta=0.33,
        adaptive_delta=True,
        fill_rate_weight=0.5,
        stability_weight=0.5,
    )

    # Process the item stream
    stats = manager.process_item_stream(items)

    print("=== Packing Results ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    example_usage()
