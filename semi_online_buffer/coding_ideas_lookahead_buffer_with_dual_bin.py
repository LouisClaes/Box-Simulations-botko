"""
==============================================================================
CODING IDEAS: Lookahead Buffer Management for Semi-Online Dual-Bin Packing
==============================================================================

Source Paper:
    Tsang, Y.P., Mo, D.Y., Chung, K.T., Lee, C.K.M. (2025).
    "A deep reinforcement learning approach for online and concurrent 3D bin
    packing optimisation with bin replacement strategies."
    Computers in Industry, 164, 104202.

Focus: Buffer / Lookahead management in the semi-online setting with
       2 active bins.

Key Insight from Paper:
    The paper treats the lookahead buffer as a set of k items where the
    DRL agent chooses WHICH item to pack next. This is critical: it's not
    FIFO (first-in-first-out) -- the agent can pick ANY item from the
    buffer. This transforms the problem from pure online to semi-online
    and is where much of the performance gain comes from.

    Performance by buffer size (single bin, DRL):
        k=5:  74.11% utilization
        k=10: 75.48% utilization  (best)
        k=15: 74.45% utilization  (DRL struggles, heuristics still improve)

    Performance by buffer size (dual bin, replaceMax, DRL):
        k=5:  76.35% utilization
        k=10: 77.11% utilization  (overall best)
        k=15: 73.87% utilization

Our Use Case:
    - Physical staging area with 5-10 boxes
    - Boxes arrive on conveyor, are scanned, and placed in staging buffer
    - Algorithm selects which box to pack next from buffer
    - Two pallets (bins) are active simultaneously
    - Buffer is FIFO-refillable: when a box is picked from buffer, the
      next box from conveyor fills the empty slot

This file addresses:
    1. Buffer data structure and management
    2. Item selection strategies (which item from buffer to pack next)
    3. Joint item-bin selection (which item + which bin)
    4. Integration of buffer with bin replacement triggers
    5. Strategies when buffer items don't fit in any bin

==============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from collections import deque
import random


# ==============================================================================
# SECTION 1: Buffer Data Structure
# ==============================================================================

@dataclass
class BufferItem:
    """An item currently in the lookahead buffer."""
    item_id: int
    w: float  # width
    l: float  # length
    h: float  # height
    weight: float = 1.0
    arrival_order: int = 0  # When it arrived in the buffer (for FIFO tiebreaking)

    @property
    def volume(self) -> float:
        return self.w * self.l * self.h

    @property
    def dimensions_sorted(self) -> Tuple[float, float, float]:
        """Dimensions sorted smallest to largest (rotation-invariant)."""
        return tuple(sorted([self.w, self.l, self.h]))

    @property
    def min_dim(self) -> float:
        return min(self.w, self.l, self.h)

    @property
    def max_dim(self) -> float:
        return max(self.w, self.l, self.h)

    @property
    def aspect_ratio(self) -> float:
        """Max dimension / min dimension. Closer to 1 = more cubic."""
        return self.max_dim / max(self.min_dim, 1e-9)


class LookaheadBuffer:
    """Manages the lookahead buffer of items awaiting packing.

    The buffer models a physical staging area where items from the conveyor
    are temporarily held. The packing algorithm can choose any item from
    the buffer to pack next (not necessarily FIFO).

    When an item is removed (packed), the next item from the conveyor
    automatically fills the empty slot (if available).

    Properties:
        capacity: Maximum number of items in the buffer (k in paper)
        items: Current items in the buffer
        conveyor: Remaining items on the conveyor (not yet in buffer)
    """

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.items: List[BufferItem] = []
        self.conveyor: deque = deque()  # Items waiting to enter buffer
        self._next_arrival_order = 0

    def load_conveyor(self, items: List[dict]):
        """Load items onto the conveyor belt.

        Args:
            items: List of dicts with keys 'w', 'l', 'h', optionally 'weight'
        """
        for i, item_data in enumerate(items):
            bi = BufferItem(
                item_id=i,
                w=item_data['w'],
                l=item_data['l'],
                h=item_data['h'],
                weight=item_data.get('weight', 1.0),
            )
            self.conveyor.append(bi)

    def fill_buffer(self):
        """Fill the buffer from the conveyor up to capacity."""
        while len(self.items) < self.capacity and self.conveyor:
            item = self.conveyor.popleft()
            item.arrival_order = self._next_arrival_order
            self._next_arrival_order += 1
            self.items.append(item)

    def remove_item(self, index: int) -> BufferItem:
        """Remove and return item at given index from the buffer.

        After removal, automatically refills from conveyor.

        Args:
            index: Index in self.items to remove

        Returns:
            The removed BufferItem
        """
        item = self.items.pop(index)
        self.fill_buffer()  # Auto-refill
        return item

    def peek_all(self) -> List[BufferItem]:
        """View all items currently in the buffer (non-destructive)."""
        return list(self.items)

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    @property
    def is_full(self) -> bool:
        return len(self.items) >= self.capacity

    @property
    def conveyor_remaining(self) -> int:
        return len(self.conveyor)

    @property
    def total_remaining(self) -> int:
        return len(self.items) + len(self.conveyor)


# ==============================================================================
# SECTION 2: Item Selection Strategies
# ==============================================================================

class ItemSelectionStrategy:
    """Strategies for selecting which item to pack from the buffer.

    In the paper, the DRL agent jointly selects item + rotation + position.
    Here we separate the ITEM SELECTION decision for use with any packing
    algorithm.

    These strategies answer: "Given the current bins and buffer, which
    item should we try to pack next?"
    """

    @staticmethod
    def largest_first(buffer: LookaheadBuffer, bin_states=None) -> int:
        """Select the largest item by volume.

        Classic greedy strategy. Large items are harder to fit later,
        so pack them first when more space is available.

        Used in many offline algorithms (decreasing volume order).
        """
        if buffer.is_empty:
            return -1
        return max(range(len(buffer.items)),
                   key=lambda i: buffer.items[i].volume)

    @staticmethod
    def smallest_first(buffer: LookaheadBuffer, bin_states=None) -> int:
        """Select the smallest item by volume.

        Can be useful for "filling gaps" when bins are mostly full.
        """
        if buffer.is_empty:
            return -1
        return min(range(len(buffer.items)),
                   key=lambda i: buffer.items[i].volume)

    @staticmethod
    def most_cubic_first(buffer: LookaheadBuffer, bin_states=None) -> int:
        """Select the most cubic item (aspect ratio closest to 1).

        Cubic items are easiest to place stably and fit in most spaces.
        Packing them first uses space efficiently.
        """
        if buffer.is_empty:
            return -1
        return min(range(len(buffer.items)),
                   key=lambda i: buffer.items[i].aspect_ratio)

    @staticmethod
    def best_fit_for_bins(buffer: LookaheadBuffer,
                          get_best_fit_score: Callable) -> int:
        """Select the item with the best fit score across all bins.

        This requires a scoring function that evaluates how well each
        item fits in the available spaces across all active bins.

        The scoring function should return higher scores for better fits.

        This is closest to the paper's approach, where the DRL agent
        evaluates Q-values for all (item, rotation, position, bin)
        combinations and picks the best.

        Args:
            buffer: The lookahead buffer
            get_best_fit_score: Function(BufferItem) -> float

        Returns:
            Index of best-fitting item in buffer
        """
        if buffer.is_empty:
            return -1
        scores = [get_best_fit_score(item) for item in buffer.items]
        if all(s <= 0 for s in scores):
            return -1  # No item fits anywhere
        return max(range(len(scores)), key=lambda i: scores[i])

    @staticmethod
    def fifo(buffer: LookaheadBuffer, bin_states=None) -> int:
        """Select the item that arrived first (FIFO).

        This is the degenerate case: ignoring the buffer entirely and
        packing in arrival order. Serves as a lower bound baseline.
        """
        if buffer.is_empty:
            return -1
        return min(range(len(buffer.items)),
                   key=lambda i: buffer.items[i].arrival_order)

    @staticmethod
    def random_choice(buffer: LookaheadBuffer, bin_states=None) -> int:
        """Select a random item from the buffer.

        Useful as a baseline and for epsilon-greedy exploration in DRL.
        """
        if buffer.is_empty:
            return -1
        return random.randint(0, len(buffer.items) - 1)


# ==============================================================================
# SECTION 3: Joint Item-Bin Selection
# ==============================================================================

class JointItemBinSelector:
    """Jointly selects which item AND which bin to pack into.

    This is the core decision-making component that combines:
    - Buffer management (which item?)
    - Multi-bin management (which bin?)
    - Placement heuristic (where in the bin?)

    The paper's DRL agent makes all three decisions simultaneously by
    evaluating Q-values for all (item, rotation, cuboid, bin) combinations.

    For a heuristic approach, we can decompose this into steps:
    1. For each item in buffer, for each bin, find the best placement
    2. Score each (item, bin, placement) triple
    3. Select the triple with the best score

    This is computationally cheaper than full DRL but captures the key
    insight that item selection and bin selection are coupled decisions.
    """

    def __init__(self, placement_heuristic: str = 'bssf',
                 scoring_method: str = 'volume_fit'):
        """
        Args:
            placement_heuristic: How to find placement within a bin.
                Options: 'bssf', 'bvf', 'bl', 'blsf'
            scoring_method: How to score (item, bin, placement) triples.
                Options: 'volume_fit', 'reward_based', 'stability_weighted'
        """
        self.placement_heuristic = placement_heuristic
        self.scoring_method = scoring_method

    def select(self, buffer: LookaheadBuffer,
               bins: list,
               mca_managers: list) -> Optional[dict]:
        """Find the best (item, bin, placement) action.

        Args:
            buffer: Current lookahead buffer
            bins: List of active Bin objects
            mca_managers: List of MaximalCuboidsAlgorithm instances (one per bin)

        Returns:
            Action dict or None if no feasible action exists
        """
        best_action = None
        best_score = float('-inf')

        for item_idx, item in enumerate(buffer.items):
            for bin_idx, (b, mca) in enumerate(zip(bins, mca_managers)):
                # Find all feasible placements for this item in this bin
                for rotation in self._get_rotations(item):
                    w, l, h = rotation
                    for cuboid_idx, mc in enumerate(b.maximal_cuboids):
                        if mc.can_fit(w, l, h):
                            # Compute placement score
                            score = self._score_placement(
                                item, rotation, mc, b, buffer)

                            if score > best_score:
                                best_score = score
                                best_action = {
                                    'item_idx': item_idx,
                                    'bin_idx': bin_idx,
                                    'rotation': rotation,
                                    'cuboid_idx': cuboid_idx,
                                    'w': w, 'l': l, 'h': h,
                                    'x': mc.x_min, 'y': mc.y_min, 'z': mc.z_min,
                                    'score': score,
                                }

        return best_action

    def _get_rotations(self, item: BufferItem) -> List[Tuple[float, float, float]]:
        """Get unique rotations of an item."""
        import itertools
        dims = (item.w, item.l, item.h)
        return list(set(itertools.permutations(dims)))

    def _score_placement(self, item, rotation, cuboid, bin_obj, buffer):
        """Score a potential placement.

        Different scoring methods emphasize different criteria.
        """
        w, l, h = rotation

        if self.scoring_method == 'volume_fit':
            # Best Short Side Fit style: minimize wasted space
            dx = cuboid.width - w
            dy = cuboid.height - h
            dz = cuboid.depth - l
            short_side = min(dx, dy, dz)
            long_side = max(dx, dy, dz)
            # Lower waste = better score (negate for maximization)
            return -(short_side + long_side)

        elif self.scoring_method == 'reward_based':
            # Simulate the paper's reward function
            # Would need to actually simulate the placement and compute reward
            # Placeholder: use volume ratio
            item_vol = w * l * h
            cuboid_vol = cuboid.volume
            return item_vol / max(cuboid_vol, 1e-9)

        elif self.scoring_method == 'stability_weighted':
            # Prefer placements with higher stability AND lower position
            position_score = -(cuboid.y_min)  # Lower = better
            fit_score = min(cuboid.width - w, cuboid.depth - l)  # Tighter = better
            return position_score * 0.5 + (-fit_score) * 0.5

        else:
            return 0.0


# ==============================================================================
# SECTION 4: Buffer-Aware Bin Replacement Integration
# ==============================================================================

class BufferAwareBinManager:
    """Integrates buffer management with bin replacement.

    This coordinator manages the interplay between:
    1. Lookahead buffer (which items are available)
    2. Active bins (where items can be placed)
    3. Replacement triggers (when no items fit in any bin)

    Decision Flow:
        1. Check if any buffer item fits in any active bin
        2. If YES: use JointItemBinSelector to pick best action
        3. If NO: trigger bin replacement strategy
        4. After replacement: re-check feasibility
        5. If still NO: skip item(s) or terminate

    Key question: What happens when NO buffer item fits in ANY bin?
    Options from the paper:
        a) Replace bin(s) [paper's approach]
    Additional options for our use case:
        b) Skip the largest item in buffer (push to overflow)
        c) Force-place the smallest item in the best available space
        d) Wait for different items (if conveyor has more)
    """

    def __init__(self,
                 buffer_capacity: int = 10,
                 n_bins: int = 2,
                 bin_config: Tuple[float, float, float] = (120, 80, 150),
                 replacement_strategy: str = 'replaceMax',
                 item_selector: str = 'joint'):
        self.buffer = LookaheadBuffer(capacity=buffer_capacity)
        self.n_bins = n_bins
        self.bin_config = bin_config
        self.replacement_strategy = replacement_strategy
        self.item_selector = item_selector

        # Statistics
        self.total_items_packed = 0
        self.total_items_skipped = 0
        self.replacements_triggered = 0

    def run_packing_episode(self, item_list: List[dict],
                            packing_function: Callable) -> dict:
        """Run a full packing episode.

        Args:
            item_list: All items to pack (conveyor order)
            packing_function: Function(buffer, bins) -> action or None

        Returns:
            Episode statistics
        """
        # Load conveyor
        self.buffer.load_conveyor(item_list)
        self.buffer.fill_buffer()

        # Initialize bins (placeholder)
        bins = self._create_bins()

        packed_items = []
        completed_bins = []
        step = 0

        while not self.buffer.is_empty:
            step += 1

            # Try to find a feasible action
            action = packing_function(self.buffer, bins)

            if action is not None:
                # Execute the packing action
                item = self.buffer.remove_item(action['item_idx'])
                # ... place item in bin ...
                packed_items.append(item)
                self.total_items_packed += 1

            else:
                # No feasible action -> trigger replacement
                self.replacements_triggered += 1

                # Close bin(s) according to strategy
                if self.replacement_strategy == 'replaceMax':
                    # Find fullest bin, close it, open new
                    fullest_idx = max(range(len(bins)),
                                      key=lambda i: bins[i].get('utilization', 0))
                    completed_bins.append(bins[fullest_idx])
                    bins[fullest_idx] = self._create_single_bin()

                elif self.replacement_strategy == 'replaceAll':
                    completed_bins.extend(bins)
                    bins = self._create_bins()

                # Retry after replacement
                action = packing_function(self.buffer, bins)
                if action is not None:
                    item = self.buffer.remove_item(action['item_idx'])
                    packed_items.append(item)
                    self.total_items_packed += 1
                else:
                    # Still can't fit -> skip item (overflow handling)
                    if not self.buffer.is_empty:
                        skipped = self.buffer.remove_item(0)  # Skip first item
                        self.total_items_skipped += 1

        # Close remaining active bins
        completed_bins.extend(bins)

        return {
            'total_packed': len(packed_items),
            'total_skipped': self.total_items_skipped,
            'total_bins_used': len(completed_bins),
            'replacements_triggered': self.replacements_triggered,
            'steps': step,
        }

    def _create_bins(self):
        """Create initial set of bins (placeholder)."""
        return [self._create_single_bin() for _ in range(self.n_bins)]

    def _create_single_bin(self):
        """Create a single empty bin (placeholder)."""
        W, L, H = self.bin_config
        return {'W': W, 'L': L, 'H': H, 'utilization': 0.0, 'placements': []}


# ==============================================================================
# SECTION 5: Buffer Size Analysis
# ==============================================================================

"""
ANALYSIS: Optimal Buffer Size for Our Setup

From the paper's results:

SINGLE BIN (S1):
    k=5:  DRL=74.11%, Best heuristic=70.10%, Gap=+4.01%
    k=10: DRL=75.48%, Best heuristic=71.37%, Gap=+4.11%
    k=15: DRL=74.45%, Best heuristic=72.06%, Gap=+2.39%

DUAL BIN, replaceMax (S2):
    k=5:  DRL=76.35%, Best heuristic=72.01%, Gap=+4.34%
    k=10: DRL=77.11%, Best heuristic=73.28%, Gap=+3.83%
    k=15: DRL=73.87%, Best heuristic=73.75%, Gap=+0.12%

OBSERVATIONS:
    1. k=10 is the sweet spot for DRL (highest absolute performance)
    2. k=5 gives the best DRL-vs-heuristic gap per buffer slot
    3. k=15 causes DRL degradation due to action space explosion
    4. Heuristics monotonically improve with k (no degradation)
    5. Dual bin with replaceMax consistently beats single bin

RECOMMENDATIONS FOR OUR THESIS:
    - Use buffer size k=10 as primary target
    - Test k=5 as well (simpler, still effective)
    - If using DRL: do NOT exceed k=10 without architectural changes
    - If using heuristics: larger buffer is always better
    - For hybrid (DRL + heuristic): DRL selects from top-5 items,
      heuristic handles placement -> effectively reduces k for DRL
      while giving heuristic full buffer benefit

PHYSICAL STAGING AREA CONSIDERATIONS:
    - 5-10 boxes in staging area means k=5-10 directly
    - Staging area size affects warehouse floor space usage
    - Larger buffer = more space needed = higher infrastructure cost
    - k=10 is a good balance of performance and physical feasibility
    - Consider: buffer could be a small conveyor loop (circular buffer)
      that holds 10 items and rotates to present them to the robotic arm
"""


# ==============================================================================
# SECTION 6: Integration with DRL and Heuristics
# ==============================================================================

"""
INTEGRATION POINTS:

1. WITH DRL (deep_rl/coding_ideas_tsang2025_ddqn_dual_bin.py):
   - The LookaheadBuffer replaces the paper's simple list of k items
   - The DRL agent's action space is defined by buffer contents
   - Buffer refill happens automatically after each item is packed
   - The DRL environment's _get_lookahead_items() should return
     buffer.peek_all() instead of item_queue[:k]

2. WITH HEURISTICS (heuristics/):
   - Use JointItemBinSelector with heuristic scoring
   - Item selection strategies (largest_first, best_fit) wrap heuristics
   - Buffer provides the candidate set; heuristic evaluates each

3. WITH MULTI-BIN (multi_bin/coding_ideas_dual_bin_replacement_strategies.py):
   - BufferAwareBinManager integrates buffer + bin replacement
   - When no buffer item fits: trigger ReplacementStrategy
   - After replacement: re-scan buffer for feasible placements

4. WITH STABILITY (stability/):
   - Stability check should be part of feasibility filtering
   - Items that would cause instability are excluded from feasible set
   - This may reduce effective buffer size (fewer feasible items)
   - Consider stability-aware item selection: pack stable items first
     to build a solid base, then fill with less stable placements

5. WITH HYBRID (hybrid_heuristic_ml/):
   - DRL selects (item_idx, bin_idx) from buffer
   - Heuristic selects (rotation, position) within chosen bin
   - This reduces DRL action space from O(6*k*|M|) to O(k*n_bins)
   - Addresses the k=15 degradation problem
   - Buffer management stays the same
"""

if __name__ == "__main__":
    print("Semi-Online Buffer Management Demo")
    print("=" * 50)

    # Create buffer with capacity 10
    buffer = LookaheadBuffer(capacity=10)

    # Generate random items for conveyor
    items = [{'w': random.randint(6, 12),
              'l': random.randint(6, 12),
              'h': random.randint(6, 12)}
             for _ in range(50)]

    buffer.load_conveyor(items)
    buffer.fill_buffer()

    print(f"Buffer capacity: {buffer.capacity}")
    print(f"Buffer items: {len(buffer.items)}")
    print(f"Conveyor remaining: {buffer.conveyor_remaining}")
    print()

    # Show buffer contents
    print("Buffer contents:")
    for i, item in enumerate(buffer.items):
        print(f"  [{i}] Item {item.item_id}: "
              f"{item.w}x{item.l}x{item.h} "
              f"vol={item.volume} "
              f"aspect={item.aspect_ratio:.2f}")

    # Test item selection strategies
    print()
    print("Item selection strategies:")
    print(f"  Largest first:     idx={ItemSelectionStrategy.largest_first(buffer)}")
    print(f"  Smallest first:    idx={ItemSelectionStrategy.smallest_first(buffer)}")
    print(f"  Most cubic first:  idx={ItemSelectionStrategy.most_cubic_first(buffer)}")
    print(f"  FIFO:              idx={ItemSelectionStrategy.fifo(buffer)}")
    print(f"  Random:            idx={ItemSelectionStrategy.random_choice(buffer)}")

    # Demonstrate removal and refill
    print()
    removed = buffer.remove_item(0)
    print(f"Removed item {removed.item_id} from buffer")
    print(f"Buffer items: {len(buffer.items)} (auto-refilled from conveyor)")
    print(f"Conveyor remaining: {buffer.conveyor_remaining}")


# ==============================================================================
# SECTION 7: Adaptive Buffer Size Strategy
# ==============================================================================

class AdaptiveBufferManager:
    """Dynamically adjust the effective buffer utilization based on bin state.

    Insight from the paper: DRL performance peaks at k=10 and degrades at k=15.
    The action space explosion at larger k is the culprit.

    This manager implements a DYNAMIC effective buffer:
        - When bins are mostly empty (early filling): use full buffer (k=10)
          to find the best items for a good foundation
        - When bins are mostly full (late filling): use smaller effective
          buffer (k=3-5) because fewer items fit anyway, and the DRL
          agent is more effective with fewer options
        - When replacement was just triggered: use full buffer for the
          new empty bin

    This addresses the k=15 degradation without reducing the physical
    buffer size. The physical buffer always holds k items; the EFFECTIVE
    buffer (presented to the DRL agent) is dynamically sized.
    """

    def __init__(self, physical_capacity: int = 10,
                 min_effective: int = 3,
                 utilization_threshold: float = 0.6):
        """
        Args:
            physical_capacity: Physical staging area size (always holds this many)
            min_effective: Minimum items presented to agent
            utilization_threshold: When avg bin utilization exceeds this,
                                    reduce effective buffer
        """
        self.physical_capacity = physical_capacity
        self.min_effective = min_effective
        self.utilization_threshold = utilization_threshold
        self.buffer = LookaheadBuffer(capacity=physical_capacity)

    def get_effective_buffer(self, bin_utilizations: List[float]) -> List[BufferItem]:
        """Get the items to present to the packing agent.

        Returns a subset of the physical buffer based on current bin states.
        """
        all_items = self.buffer.peek_all()
        if not all_items:
            return []

        avg_util = np.mean(bin_utilizations) if bin_utilizations else 0.0

        if avg_util < self.utilization_threshold:
            # Bins are not yet full -- use full buffer
            effective_k = len(all_items)
        else:
            # Bins are getting full -- reduce buffer to avoid DRL degradation
            # Linear interpolation from physical_capacity to min_effective
            # as utilization goes from threshold to 1.0
            progress = (avg_util - self.utilization_threshold) / \
                       (1.0 - self.utilization_threshold)
            effective_k = int(self.physical_capacity -
                              progress * (self.physical_capacity - self.min_effective))
            effective_k = max(self.min_effective, effective_k)

        # Select the top effective_k items (by a priority criterion)
        # Option 1: First k items (arrival order)
        # Option 2: Largest k items (pack big items first)
        # Option 3: Most cubic k items (easiest to place stably)
        # Using Option 2 by default (largest first priority)
        sorted_items = sorted(all_items, key=lambda x: x.volume, reverse=True)
        return sorted_items[:effective_k]


# ==============================================================================
# SECTION 8: Buffer Diversity Score
# ==============================================================================

class BufferDiversityAnalyzer:
    """Analyze the diversity of items in the buffer for decision support.

    A diverse buffer (items of many different sizes) is generally better
    than a homogeneous buffer (all items similar size) because diversity
    provides more options for fitting into various free spaces.

    This analyzer computes metrics that can be used for:
        1. Reward shaping: bonus for maintaining diverse buffer
        2. Replacement decisions: replace bin if buffer diversity is high
           (many options available) vs. low (fewer options, keep flexible bin)
        3. Item selection: prefer items that are UNUSUAL in the buffer
           (rare sizes should be used when good spaces exist, rather than
           waiting and hoping for a better opportunity)
    """

    @staticmethod
    def compute_diversity(buffer: LookaheadBuffer) -> dict:
        """Compute diversity metrics for current buffer contents.

        Returns dict with:
            - volume_diversity: coefficient of variation of item volumes
            - dim_diversity: average CoV across w, l, h dimensions
            - aspect_diversity: CoV of aspect ratios
            - size_classes: number of distinct size classes (small/medium/large)
        """
        if buffer.is_empty or len(buffer.items) < 2:
            return {'volume_diversity': 0.0, 'dim_diversity': 0.0,
                    'aspect_diversity': 0.0, 'size_classes': 0}

        volumes = [item.volume for item in buffer.items]
        widths = [item.w for item in buffer.items]
        lengths = [item.l for item in buffer.items]
        heights = [item.h for item in buffer.items]
        aspects = [item.aspect_ratio for item in buffer.items]

        def cov(values):
            """Coefficient of variation (std / mean)."""
            m = np.mean(values)
            return np.std(values) / m if m > 0 else 0.0

        vol_cov = cov(volumes)
        dim_cov = np.mean([cov(widths), cov(lengths), cov(heights)])
        aspect_cov = cov(aspects)

        # Classify items into size categories
        small = sum(1 for v in volumes if v < 1000)
        medium = sum(1 for v in volumes if 1000 <= v < 5000)
        large = sum(1 for v in volumes if v >= 5000)
        classes = sum(1 for c in [small, medium, large] if c > 0)

        return {
            'volume_diversity': vol_cov,
            'dim_diversity': dim_cov,
            'aspect_diversity': aspect_cov,
            'size_classes': classes,
        }

    @staticmethod
    def item_rarity_score(item: BufferItem, buffer: LookaheadBuffer) -> float:
        """How rare/unusual is this item compared to others in the buffer?

        High rarity = item is very different from the buffer average.
        Should be packed when a good opportunity exists (rare items
        are harder to place later).

        Returns value in [0, 1] where 1 = very rare.
        """
        if buffer.is_empty or len(buffer.items) < 2:
            return 0.5  # Default: moderately rare

        volumes = [it.volume for it in buffer.items]
        avg_vol = np.mean(volumes)
        std_vol = np.std(volumes)

        if std_vol < 1e-6:
            return 0.0  # All items are same size

        z_score = abs(item.volume - avg_vol) / std_vol
        # Normalize to [0, 1] using sigmoid-like function
        rarity = min(1.0, z_score / 3.0)
        return rarity


# ==============================================================================
# SECTION 9: Stability-Aware Item Selection
# ==============================================================================

class StabilityAwareItemSelector:
    """Select items from the buffer considering both fit AND stability impact.

    Key insight from cross-paper analysis:
        - Early in packing: prioritize LARGE, FLAT items to build a stable base
        - Middle of packing: prioritize items that FILL GAPS and maintain flatness
        - Late in packing: prioritize SMALL items that fit remaining spaces

    This implements a phase-aware item selection strategy that integrates
    with the LookaheadBuffer and bin state information.

    From One4Many-StablePacker (2025): flat top surfaces are crucial for
    subsequent item stability. So item selection should consider the
    IMPACT on surface flatness, not just volume fit.
    """

    def __init__(self, early_threshold: float = 0.3,
                 late_threshold: float = 0.7):
        """
        Args:
            early_threshold: Bin utilization below this = early phase
            late_threshold: Bin utilization above this = late phase
        """
        self.early_threshold = early_threshold
        self.late_threshold = late_threshold

    def select_item(self, buffer: LookaheadBuffer,
                     bin_utilizations: List[float],
                     feasible_actions: List[dict]) -> Optional[int]:
        """Select the best item index from the buffer.

        Args:
            buffer: Current lookahead buffer
            bin_utilizations: Utilization of each active bin
            feasible_actions: All feasible (item, bin, placement) combinations

        Returns:
            Index of selected item in buffer, or None if no feasible action
        """
        if not feasible_actions:
            return None

        avg_util = np.mean(bin_utilizations)

        # Score each item
        scores = {}
        for action in feasible_actions:
            idx = action['item_idx']
            if idx not in scores:
                scores[idx] = []

            item = buffer.items[idx]
            score = self._score_item(item, action, avg_util)
            scores[idx].append(score)

        if not scores:
            return None

        # For each item, take the BEST score across its feasible placements
        best_scores = {idx: max(s_list) for idx, s_list in scores.items()}

        # Return the item with the highest best score
        return max(best_scores, key=best_scores.get)

    def _score_item(self, item: BufferItem, action: dict,
                     avg_utilization: float) -> float:
        """Score an item based on the current packing phase."""

        base_area = item.w * item.l  # Largest possible base (may be rotated)
        volume = item.volume
        stability = action.get('stability', 0.5)

        if avg_utilization < self.early_threshold:
            # Early phase: prefer large, flat, heavy items for base
            flatness = min(item.w, item.l) / max(item.h, 1)  # Wider & shorter = better
            score = (0.4 * (volume / 50000.0) +   # Larger is better
                     0.3 * flatness +               # Flatter is better
                     0.3 * stability)                # More stable is better

        elif avg_utilization < self.late_threshold:
            # Middle phase: prefer gap-filling, surface-maintaining items
            # Lower placement height = better (fills gaps below)
            height_score = 1.0 - (action.get('y', 0) / 150.0)
            score = (0.3 * height_score +           # Lower placement = better
                     0.3 * stability +               # Stability matters
                     0.2 * (volume / 50000.0) +      # Still prefer larger
                     0.2 * (1.0 / max(item.aspect_ratio, 1.0)))  # More cubic = better

        else:
            # Late phase: prefer small items that fit tight
            tightness = volume / max(action.get('cuboid_volume', volume), 1)
            score = (0.5 * tightness +               # Tight fit is critical
                     0.3 * stability +                # Still need stability
                     0.2 * (1.0 - volume / 50000.0))  # Smaller preferred

        return score


# ==============================================================================
# SECTION 10: Complete Pipeline: Buffer + Dual Bin + Stability
# ==============================================================================

class CompleteSemiOnlinePipeline:
    """The full thesis pipeline combining all components.

    Architecture:
        Conveyor -> Buffer (k=10) -> Item Selector -> Bin Router -> Placer

    Components:
        1. LookaheadBuffer: Physical staging area management
        2. StabilityAwareItemSelector: Phase-aware item prioritization
        3. JointItemBinSelector: Heuristic placement scoring
        4. MultiBinManager (from multi_bin/): Bin lifecycle management
        5. ReplaceMax (from multi_bin/): Bin replacement strategy

    This pipeline works WITHOUT DRL and serves as:
        - A strong heuristic baseline
        - A testing framework for individual components
        - A fallback for when DRL inference is too slow

    Usage:
        pipeline = CompleteSemiOnlinePipeline(
            buffer_size=10,
            bin_config=(120, 80, 150),
            strategy='replaceMax',
            heuristic='bssf',
        )
        results = pipeline.run(item_list)
        print(f"Avg utilization: {results['avg_utilization']:.2%}")
    """

    def __init__(self, buffer_size: int = 10,
                 bin_config: Tuple[float, float, float] = (120, 80, 150),
                 n_bins: int = 2,
                 strategy: str = 'replaceMax',
                 heuristic: str = 'bssf',
                 stability_threshold: float = 0.7):
        self.buffer = LookaheadBuffer(capacity=buffer_size)
        self.item_selector = StabilityAwareItemSelector()
        self.joint_selector = JointItemBinSelector(
            placement_heuristic=heuristic,
            scoring_method='stability_weighted')
        self.bin_config = bin_config
        self.n_bins = n_bins
        self.strategy_name = strategy
        self.stability_threshold = stability_threshold

        # Runtime state
        self.bins = []
        self.completed_bins = []
        self.stats = {
            'items_packed': 0,
            'items_skipped': 0,
            'replacements': 0,
            'total_reward': 0.0,
        }

    def run(self, items: List[dict]) -> dict:
        """Run the complete packing pipeline on a list of items.

        Args:
            items: List of item dicts with keys 'w', 'l', 'h', optional 'weight'

        Returns:
            Results dictionary with utilization statistics
        """
        # Reset
        self.bins = [{'config': self.bin_config, 'placements': [], 'util': 0.0}
                     for _ in range(self.n_bins)]
        self.completed_bins = []
        self.stats = {'items_packed': 0, 'items_skipped': 0,
                      'replacements': 0, 'total_reward': 0.0}

        # Load conveyor
        self.buffer.load_conveyor(items)
        self.buffer.fill_buffer()

        step = 0
        max_steps = len(items) * 2  # Safety limit

        while not self.buffer.is_empty and step < max_steps:
            step += 1

            # Get current state
            bin_utils = [b['util'] for b in self.bins]
            buffer_items = self.buffer.peek_all()

            # Try to find a feasible action using heuristic
            action = self._find_best_action(buffer_items, bin_utils)

            if action is not None:
                # Execute placement
                self._execute_placement(action)
            else:
                # No feasible action -- trigger replacement
                self._trigger_replacement()

                # Retry after replacement
                buffer_items = self.buffer.peek_all()
                bin_utils = [b['util'] for b in self.bins]
                action = self._find_best_action(buffer_items, bin_utils)

                if action is not None:
                    self._execute_placement(action)
                else:
                    # Still stuck -- skip the largest item
                    if not self.buffer.is_empty:
                        largest_idx = max(range(len(self.buffer.items)),
                                          key=lambda i: self.buffer.items[i].volume)
                        self.buffer.remove_item(largest_idx)
                        self.stats['items_skipped'] += 1

        # Finalize
        self.completed_bins.extend(self.bins)
        active = [b for b in self.completed_bins if b['placements']]
        utils = [b['util'] for b in active]

        return {
            'avg_utilization': np.mean(utils) if utils else 0.0,
            'std_utilization': np.std(utils) if utils else 0.0,
            'max_utilization': np.max(utils) if utils else 0.0,
            'min_utilization': np.min(utils) if utils else 0.0,
            'num_bins_used': len(active),
            'items_packed': self.stats['items_packed'],
            'items_skipped': self.stats['items_skipped'],
            'replacements': self.stats['replacements'],
            'strategy': self.strategy_name,
            'buffer_size': self.buffer.capacity,
        }

    def _find_best_action(self, buffer_items, bin_utils) -> Optional[dict]:
        """Find the best (item, bin, placement) action using heuristics.

        Simplified version -- real implementation would use MCA + full
        feasibility checking from the DDQN coding ideas file.
        """
        W, L, H = self.bin_config
        best_action = None
        best_score = float('-inf')

        for item_idx, item in enumerate(buffer_items):
            for bin_idx, b in enumerate(self.bins):
                # Simplified fit check
                item_vol = item.w * item.l * item.h
                remaining = (1.0 - b['util']) * W * L * H
                if item_vol <= remaining * 1.2:  # Rough check with margin
                    # Score the placement
                    score = self._score_placement(item, b, bin_utils)
                    if score > best_score:
                        best_score = score
                        best_action = {
                            'item_idx': item_idx,
                            'bin_idx': bin_idx,
                            'volume': item_vol,
                            'score': score,
                        }

        return best_action

    def _score_placement(self, item, bin_state, bin_utils) -> float:
        """Score a placement using heuristic criteria."""
        W, L, H = self.bin_config
        item_vol = item.w * item.l * item.h
        remaining = (1.0 - bin_state['util']) * W * L * H
        fit_ratio = item_vol / max(remaining, 1)
        return fit_ratio

    def _execute_placement(self, action):
        """Execute a placement action."""
        item = self.buffer.remove_item(action['item_idx'])
        bin_idx = action['bin_idx']
        W, L, H = self.bin_config
        self.bins[bin_idx]['placements'].append(item)
        self.bins[bin_idx]['util'] += action['volume'] / (W * L * H)
        self.stats['items_packed'] += 1

    def _trigger_replacement(self):
        """Execute bin replacement."""
        self.stats['replacements'] += 1
        if self.strategy_name == 'replaceMax':
            max_idx = max(range(len(self.bins)),
                          key=lambda i: self.bins[i]['util'])
            self.completed_bins.append(self.bins[max_idx])
            self.bins[max_idx] = {'config': self.bin_config,
                                   'placements': [], 'util': 0.0}
        elif self.strategy_name == 'replaceAll':
            self.completed_bins.extend(self.bins)
            self.bins = [{'config': self.bin_config, 'placements': [],
                          'util': 0.0} for _ in range(self.n_bins)]


# ==============================================================================
# SECTION 11: Buffer Size Sensitivity Analysis Framework
# ==============================================================================

"""
BUFFER SIZE SENSITIVITY ANALYSIS

This framework systematically tests different buffer sizes to find the
optimal k for our specific setup.

From the paper (Tsang et al. 2025):
    Single bin DRL:     k=5: 74.11%, k=10: 75.48%, k=15: 74.45%
    Dual bin ReplaceMax: k=5: 76.35%, k=10: 77.11%, k=15: 73.87%

Expected behavior for our setup (EUR pallet, real items):
    - Larger items relative to bin may mean FEWER maximal cuboids
    - Heterogeneous item sizes may mean MORE benefit from larger buffer
    - Weight/stability constraints may reduce effective feasible actions
    - Overall: k=8-12 likely optimal, but MUST be empirically verified

EXPERIMENTAL PROTOCOL:
    1. Fix: replacement strategy = ReplaceMax, n_bins = 2
    2. Vary: k in {1, 2, 3, 5, 7, 10, 12, 15, 20}
    3. For each k:
        a. Run 100 instances of 200 items each
        b. Record: avg utilization, std, num bins, computation time
    4. Also test with DRL (after training) at k = {5, 7, 10}
    5. Also test with best heuristic at all k values

    Plot: utilization vs k (separate curves for DRL, BSSF, BVF, FIFO)
    Expected: DRL curve peaks around k=10, heuristic curves monotonically increase

KEY IMPLEMENTATION:
    buffer_sizes = [1, 2, 3, 5, 7, 10, 12, 15, 20]
    results = {}
    for k in buffer_sizes:
        pipeline = CompleteSemiOnlinePipeline(buffer_size=k)
        k_results = [pipeline.run(generate_items(200)) for _ in range(100)]
        results[k] = {
            'avg': np.mean([r['avg_utilization'] for r in k_results]),
            'std': np.std([r['avg_utilization'] for r in k_results]),
        }
"""
