"""
=============================================================================
CODING IDEAS: Semi-Online Buffer Management with Hierarchical Tree Search
Based on: Lee & Nam (2025) - "A Hierarchical Bin Packing Framework..."
=============================================================================

FOCUS: How to manage a buffer of 5-10 items in semi-online 3D BPP,
drawing on the paper's conveyor visibility model and tree search.

The paper's key insight: with more visible/accessible items, utilization
improves dramatically (78.88% with 1 item -> 98.59% with 5 items + rotation
+ dual-arm). This proves the buffer is extremely valuable.

=============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import numpy as np


# =============================================================================
# 1. BUFFER ITEM SELECTION STRATEGIES
# =============================================================================

"""
When a buffer holds 5-10 items, the key decision is: WHICH item to pack next?

The paper handles this via tree search (Algorithm 2): it evaluates all buffer
items x orientations x positions, sorts by reward, and selects the best path.

For our use case, we can implement multiple strategies and compare:
"""


class BufferSelectionStrategy(Enum):
    """Strategies for selecting which buffer item to pack next."""

    # Strategy 1: Use the tree search (paper's approach)
    TREE_SEARCH = "tree_search"

    # Strategy 2: Largest item first (classic heuristic)
    LARGEST_FIRST = "largest_first"

    # Strategy 3: Best fit (item that wastes least space)
    BEST_FIT = "best_fit"

    # Strategy 4: RL-based selection (learned policy over buffer)
    RL_SELECTION = "rl_selection"

    # Strategy 5: Combined score (fill + stability + future potential)
    COMBINED_SCORE = "combined_score"


@dataclass
class BufferItem:
    """An item in the buffer with associated metadata."""
    id: int
    width: float
    depth: float
    height: float
    weight: float = 1.0
    time_in_buffer: int = 0     # How many steps this item has waited
    volume: float = 0.0

    def __post_init__(self):
        self.volume = self.width * self.depth * self.height


class SmartBufferManager:
    """Buffer manager with intelligent item selection.

    Extends the paper's conveyor model (Section III, V-C2) with:
    - Multiple selection strategies
    - Time-in-buffer tracking (items waiting too long should be prioritized)
    - Look-ahead scoring (evaluate the CONSEQUENCE of packing an item)
    """

    def __init__(self, max_size: int = 10,
                 strategy: BufferSelectionStrategy = BufferSelectionStrategy.COMBINED_SCORE,
                 max_wait_time: int = 20):
        self.max_size = max_size
        self.strategy = strategy
        self.max_wait_time = max_wait_time  # Force pack after this many steps
        self.buffer: List[BufferItem] = []
        self.queue: List[BufferItem] = []

    def select_next_item(self, bin_states, tree_search=None) -> Optional[BufferItem]:
        """Select the next item to pack from the buffer.

        This is the core decision that the paper's tree search addresses.
        """
        if not self.buffer:
            return None

        # Check for items that have waited too long (forced selection)
        urgent = [i for i in self.buffer if i.time_in_buffer >= self.max_wait_time]
        if urgent:
            return max(urgent, key=lambda i: i.time_in_buffer)

        if self.strategy == BufferSelectionStrategy.TREE_SEARCH:
            return self._select_via_tree_search(bin_states, tree_search)
        elif self.strategy == BufferSelectionStrategy.LARGEST_FIRST:
            return self._select_largest_first()
        elif self.strategy == BufferSelectionStrategy.BEST_FIT:
            return self._select_best_fit(bin_states)
        elif self.strategy == BufferSelectionStrategy.COMBINED_SCORE:
            return self._select_combined_score(bin_states)
        else:
            return self.buffer[0]  # Default: FIFO

    def _select_via_tree_search(self, bin_states, tree_search) -> Optional[BufferItem]:
        """Use the paper's tree search to select the best item.

        The tree search evaluates all items x orientations x bins and
        returns the sequence with the highest cumulative reward.
        The first item in that sequence is our selection.
        """
        if tree_search is None:
            return self._select_largest_first()

        # Delegate to tree search (Algorithms 1-2 from paper)
        # The tree search returns a sequence of actions; we take the first
        action_sequence = tree_search.search(
            bin_states=bin_states,
            buffer_items=self.buffer,
            recognized_items=[],
            require_full_pack=False
        )

        if action_sequence:
            return action_sequence[0].item
        return self._select_largest_first()  # Fallback

    def _select_largest_first(self) -> Optional[BufferItem]:
        """Classic heuristic: pack the largest item first.

        Rationale: large items are harder to place later when the bin is
        more full. Placing them early gives more flexibility.

        The paper's REWARDSORTING uses descending item size as a tiebreaker,
        reflecting a similar intuition.
        """
        return max(self.buffer, key=lambda i: i.volume)

    def _select_best_fit(self, bin_states) -> Optional[BufferItem]:
        """Select the item that fits best in the current bin state.

        "Best fit" = the item whose placement wastes the least space.
        Inspired by the paper's adjacency reward: higher adjacency = tighter fit.
        """
        best_item = None
        best_score = -float('inf')

        for item in self.buffer:
            for bin_state in bin_states:
                # Simplified: check all positions and find the best adjacency
                for w, d, h in [(item.width, item.depth, item.height),
                                (item.depth, item.width, item.height)]:
                    w, d = int(w), int(d)
                    for x in range(bin_state.width - w + 1):
                        for z in range(bin_state.depth - d + 1):
                            if bin_state.can_place(w, d, h, x, z):
                                score = bin_state.compute_adjacency_reward(x, z, w, d)
                                if score > best_score:
                                    best_score = score
                                    best_item = item

        return best_item if best_item else self._select_largest_first()

    def _select_combined_score(self, bin_states) -> Optional[BufferItem]:
        """Score each buffer item on multiple criteria and select the best.

        Criteria:
        1. Volume (prefer larger items -- harder to place later)
        2. Best placement reward (adjacency + stability)
        3. Wait time (items waiting longer get priority)
        4. "Regret" estimation: how much WORSE would future placements be
           if we delay this item?

        This is our novel contribution, combining the paper's ideas with
        practical buffer management.
        """
        if not self.buffer:
            return None

        scores = {}
        max_volume = max(i.volume for i in self.buffer) or 1.0

        for item in self.buffer:
            # Component 1: Volume preference (normalize to [0, 1])
            volume_score = item.volume / max_volume

            # Component 2: Wait time urgency (normalize to [0, 1])
            wait_score = min(item.time_in_buffer / self.max_wait_time, 1.0)

            # Component 3: Best placement quality
            # (simplified; full version would query RL agent)
            placement_score = 0.0
            for bin_state in bin_states:
                for w, d, h in [(item.width, item.depth, item.height),
                                (item.depth, item.width, item.height)]:
                    w_int, d_int = int(w), int(d)
                    for x in range(bin_state.width - w_int + 1):
                        for z in range(bin_state.depth - d_int + 1):
                            if bin_state.can_place(w_int, d_int, h, x, z):
                                r = bin_state.compute_adjacency_reward(
                                    x, z, w_int, d_int)
                                placement_score = max(placement_score, r)

            # Normalize placement score
            max_adj = 2 * (max(i.width + i.depth for i in self.buffer))
            placement_score_norm = placement_score / max_adj if max_adj > 0 else 0

            # Combined score with tunable weights
            alpha = 0.3    # Volume weight
            beta = 0.2     # Wait time weight
            gamma = 0.5    # Placement quality weight

            total_score = (alpha * volume_score +
                           beta * wait_score +
                           gamma * placement_score_norm)

            scores[item.id] = total_score

        # Select item with highest combined score
        best_id = max(scores, key=scores.get)
        return next(i for i in self.buffer if i.id == best_id)

    def increment_wait_times(self):
        """Call at each step to track how long items have been in buffer."""
        for item in self.buffer:
            item.time_in_buffer += 1

    def add_item(self, item: BufferItem) -> bool:
        """Add an item to the buffer if space available."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(item)
            return True
        return False

    def remove_item(self, item_id: int):
        """Remove an item from the buffer (after packing)."""
        self.buffer = [i for i in self.buffer if i.id != item_id]

    def refill_from_queue(self):
        """Refill buffer from arrival queue."""
        while len(self.buffer) < self.max_size and self.queue:
            item = self.queue.pop(0)
            self.buffer.append(item)


# =============================================================================
# 2. LOOK-AHEAD EVALUATION (Adapted from Paper's Forward Simulation)
# =============================================================================

"""
The paper's forward simulation (Section IV-C3) evaluates candidate sequences
by simulating them forward and computing cumulative reward. We extend this
to evaluate the IMPACT of buffer decisions.

Key idea: For each candidate item from the buffer, simulate packing it
AND the likely next few items. The sequence that leads to the best
cumulative utilization wins.
"""


class LookAheadEvaluator:
    """Evaluate buffer decisions by simulating their consequences.

    Based on the paper's forward simulation + our buffer extension.
    """

    def __init__(self, rl_agent=None, look_ahead_depth: int = 3):
        self.rl_agent = rl_agent
        self.look_ahead_depth = look_ahead_depth

    def evaluate_item_choice(self, item: BufferItem, remaining_buffer: List[BufferItem],
                             bin_state, orientation: Tuple[float, float, float],
                             position: Tuple[int, int]) -> float:
        """Evaluate how good it is to pack this item next.

        Simulates packing this item, then greedily packing the next
        look_ahead_depth items from the remaining buffer.

        Returns: estimated cumulative reward (higher = better choice)
        """
        import copy
        sim_bin = copy.deepcopy(bin_state)
        sim_buffer = list(remaining_buffer)

        # Pack the candidate item
        w, d, h = int(orientation[0]), int(orientation[1]), orientation[2]
        total_reward = sim_bin.compute_adjacency_reward(
            position[0], position[1], w, d)
        sim_bin.place_item(item, position[0], position[1], orientation)

        # Simulate look-ahead: greedily pack next items
        for step in range(self.look_ahead_depth):
            if not sim_buffer:
                break

            best_next_reward = -float('inf')
            best_next_action = None
            best_next_item_idx = -1

            for idx, next_item in enumerate(sim_buffer):
                for next_orient in [(next_item.width, next_item.depth, next_item.height),
                                    (next_item.depth, next_item.width, next_item.height)]:
                    nw, nd, nh = int(next_orient[0]), int(next_orient[1]), next_orient[2]
                    for x in range(sim_bin.width - nw + 1):
                        for z in range(sim_bin.depth - nd + 1):
                            if sim_bin.can_place(nw, nd, nh, x, z):
                                r = sim_bin.compute_adjacency_reward(x, z, nw, nd)
                                if r > best_next_reward:
                                    best_next_reward = r
                                    best_next_action = (next_item, next_orient, (x, z))
                                    best_next_item_idx = idx

            if best_next_action is None:
                break

            # Apply best next action
            ni, no, np_ = best_next_action
            sim_bin.place_item(ni, np_[0], np_[1], no)
            total_reward += best_next_reward * (0.9 ** (step + 1))  # Discounted
            sim_buffer.pop(best_next_item_idx)

        return total_reward


# =============================================================================
# 3. BUFFER-AWARE BIN CLOSING DECISIONS
# =============================================================================

"""
In 2-bounded space, closing a bin is IRREVERSIBLE. The buffer contents
should influence this decision.

Paper context: The paper uses a single bin, so bin closing is not relevant.
We extend the framework with buffer-informed bin closing.
"""


class BufferAwareBinCloser:
    """Decide when to close a bin based on buffer contents.

    For 2-bounded space: closing a bin too early wastes space,
    closing too late blocks the other bin from receiving good items.
    """

    def __init__(self, min_utilization: float = 0.70,
                 target_utilization: float = 0.90):
        self.min_utilization = min_utilization
        self.target_utilization = target_utilization

    def should_close(self, bin_state, buffer: List[BufferItem],
                     other_bin_state) -> bool:
        """Decide whether to close this bin.

        Factors:
        1. Current utilization (don't close if too low)
        2. Can any buffer item fit? (close if nothing fits)
        3. Would the other bin benefit more from these items?
        4. Buffer composition: are remaining items too large for this bin?
        """
        util = bin_state.get_utilization()

        # Never close if below minimum utilization (waste)
        if util < self.min_utilization:
            return False

        # Close if target reached and buffer items fit better in other bin
        if util >= self.target_utilization:
            return True

        # Check if any buffer item can fit
        any_fits = False
        for item in buffer:
            for w, d, h in [(item.width, item.depth, item.height),
                            (item.depth, item.width, item.height)]:
                w_int, d_int = int(w), int(d)
                for x in range(bin_state.width - w_int + 1):
                    for z in range(bin_state.depth - d_int + 1):
                        if bin_state.can_place(w_int, d_int, h, x, z):
                            any_fits = True
                            break
                    if any_fits:
                        break
                if any_fits:
                    break
            if any_fits:
                break

        if not any_fits:
            return True  # Nothing fits -> close

        # Advanced: compare fit quality between the two bins
        # If most items fit better in the other bin, consider closing
        this_bin_avg_reward = self._avg_best_reward(bin_state, buffer)
        other_bin_avg_reward = self._avg_best_reward(other_bin_state, buffer)

        if (this_bin_avg_reward < other_bin_avg_reward * 0.5 and
                util >= self.min_utilization + 0.1):
            return True  # Items much better suited to other bin

        return False

    def _avg_best_reward(self, bin_state, buffer: List[BufferItem]) -> float:
        """Average best placement reward for buffer items in this bin."""
        rewards = []
        for item in buffer:
            best_r = 0.0
            for w, d, h in [(item.width, item.depth, item.height),
                            (item.depth, item.width, item.height)]:
                w_int, d_int = int(w), int(d)
                for x in range(bin_state.width - w_int + 1):
                    for z in range(bin_state.depth - d_int + 1):
                        if bin_state.can_place(w_int, d_int, h, x, z):
                            r = bin_state.compute_adjacency_reward(x, z, w_int, d_int)
                            best_r = max(best_r, r)
            rewards.append(best_r)
        return np.mean(rewards) if rewards else 0.0


# =============================================================================
# 4. PSEUDOCODE: COMPLETE SEMI-ONLINE LOOP WITH BUFFER
# =============================================================================

"""
ALGORITHM: Semi-Online 3D BPP with Buffer and 2-Bounded Space

This combines the paper's hierarchical framework with our buffer extensions.

Input:
    - item_stream: sequence of items arriving on conveyor
    - buffer_size: 5-10 (our use case)
    - k: 2 (bounded space parameter)

Output:
    - assignment of items to bins with positions and orientations
    - statistics: utilization per bin, total bins used, stability scores

Pseudocode:

1.  INITIALIZE:
      active_bins = [new Bin(), new Bin()]      # k=2
      buffer = []
      closed_bins = []
      rl_agent = load_pretrained_model()
      tree_search = HierarchicalTreeSearch(rl_agent)
      stability_checker = StabilityChecker()

2.  FILL buffer from item_stream (up to buffer_size items)

3.  WHILE buffer is not empty OR item_stream has items:

      3a. INCREMENT wait times for all buffer items

      3b. CHECK bin closing conditions:
          FOR each active_bin:
              IF should_close(bin, buffer, other_bin):
                  CLOSE bin, OPEN new bin

      3c. SELECT item from buffer:
          # Paper's tree search evaluates all items x orientations x bins
          action_sequence = tree_search.search(active_bins, buffer)

          IF action_sequence is empty:
              # No item fits in either bin
              CLOSE more-full bin, OPEN new bin
              CONTINUE

      3d. EXECUTE first action in sequence:
          item, orientation, position, bin_idx = action_sequence[0]
          PLACE item in active_bins[bin_idx]

      3e. OPTIONAL REPACKING (if time permits):
          IF utilization could improve:
              REPACK top-layer items in active_bins[bin_idx]

      3f. UPDATE buffer:
          REMOVE packed item from buffer
          REFILL buffer from item_stream

4.  CLOSE remaining active bins

5.  RETURN statistics
"""


# =============================================================================
# 5. EXPERIMENTAL DESIGN FOR THESIS
# =============================================================================

"""
SUGGESTED EXPERIMENTS (comparing buffer strategies):

Experiment 1: Buffer Size Impact
    - Fix strategy = TREE_SEARCH
    - Vary buffer_size in {1, 3, 5, 7, 10}
    - Measure: mean utilization, computation time, stability score
    - Expected: utilization increases with buffer size (paper shows this)

Experiment 2: Strategy Comparison
    - Fix buffer_size = 10
    - Compare: TREE_SEARCH, LARGEST_FIRST, BEST_FIT, COMBINED_SCORE
    - Measure: mean utilization, computation time per step, total bins used
    - Expected: TREE_SEARCH best utilization but slowest;
                COMBINED_SCORE best trade-off

Experiment 3: Look-Ahead Depth
    - Fix buffer_size = 10, strategy = COMBINED_SCORE
    - Vary look_ahead_depth in {0, 1, 2, 3, 5}
    - Measure: utilization improvement vs computation cost
    - Expected: diminishing returns after depth 2-3

Experiment 4: 2-Bounded vs Unbounded
    - Compare k=2 (our use case) vs k=infinity (unbounded)
    - Fix buffer_size = 10, strategy = TREE_SEARCH
    - Measure: total bins used, mean utilization, worst-case utilization
    - Expected: k=2 uses slightly more bins but is realistic

Experiment 5: Repacking Budget
    - Fix all else, vary repacking time_limit in {0, 0.5, 1.0, 2.0, 5.0} seconds
    - Measure: utilization improvement vs time spent
    - Expected: most gains in first 1 second (paper shows this in Figure 8)

Item distributions for all experiments:
    - Random: uniform U(1, W/2) per dimension (paper's distribution)
    - Realistic: based on common parcel sizes (e-commerce data)
    - Adversarial: items designed to be hard to pack (stress test)
"""
