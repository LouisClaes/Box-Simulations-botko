"""
Buffer-Aware Packing for Semi-Online 3D Bin Packing
====================================================

This file extends the Deep-Pack (Kundu et al. 2019) paradigm to handle
a semi-online setup where a buffer of 5-10 items is available. The agent
can choose WHICH item from the buffer to place next, adding an item
selection layer on top of the placement decision.

Context from overview KB (Section 4.1 - Buffering Model):
  "A buffer of fixed size holds items temporarily before packing. The
   algorithm can choose which buffered item to pack next (limited
   lookahead). Key study: Epstein & Kleiman (2009)."

Our specific setup:
  - Buffer size: 5-10 items (configurable)
  - 2-bounded space: 2 active bins
  - Items arrive on a conveyor; as one item is removed from buffer for
    packing, the next item from the conveyor enters the buffer
  - The agent sees all items in the buffer and both bin states
  - Decision: (which_item, which_bin, where_in_bin, what_orientation)

This file focuses on the BUFFER MANAGEMENT and ITEM SELECTION aspects.
The placement decision is handled by the DRL agent in deep_rl/.

Related files:
  - python/deep_rl/deep_pack_3d_coding_ideas.py  (DRL placement agent)
  - python/multi_bin/two_bounded_manager.py        (2-bounded logic)
  - python/stability/stability_checker.py          (stability checks)
"""

# =============================================================================
# SECTION 1: BUFFER DATA STRUCTURE AND MANAGEMENT
# =============================================================================

"""
1.1 Buffer Class

The buffer maintains a fixed-capacity list of items. When an item is
removed (placed in a bin or explicitly discarded), the next item from
the conveyor/queue enters the buffer automatically.

Key design decisions:
  - Buffer is FIFO for incoming items (conveyor order preserved)
  - Selection from buffer is FREE (any item can be chosen)
  - If buffer is full and conveyor has more items: conveyor blocks
  - If buffer is empty and conveyor is done: packing ends
"""

# from dataclasses import dataclass, field
# from typing import List, Optional, Tuple
# from collections import deque
# import numpy as np
#
# @dataclass
# class Box:
#     id: int
#     width: float
#     depth: float
#     height: float
#     weight: float = 1.0
#
# class ItemBuffer:
#     \"\"\"
#     Semi-online buffer for holding items before packing.
#
#     The buffer sits between the conveyor (item source) and the packing
#     station (bins). It allows the agent to choose which item to pack
#     next, providing a limited form of lookahead.
#     \"\"\"
#     def __init__(self, capacity: int = 10):
#         self.capacity = capacity
#         self.items: List[Box] = []
#         self.conveyor: deque = deque()  # remaining items on conveyor
#         self.total_arrived = 0
#         self.total_packed = 0
#         self.total_discarded = 0
#
#     def load_conveyor(self, items: List[Box]):
#         \"\"\"Load a sequence of items onto the conveyor.\"\"\"
#         self.conveyor = deque(items)
#         self.total_arrived = len(items)
#         self._fill_buffer()
#
#     def _fill_buffer(self):
#         \"\"\"Fill buffer from conveyor up to capacity.\"\"\"
#         while len(self.items) < self.capacity and self.conveyor:
#             self.items.append(self.conveyor.popleft())
#
#     def select_item(self, index: int) -> Box:
#         \"\"\"
#         Remove and return the item at the given index from the buffer.
#         Automatically refills from conveyor.
#         \"\"\"
#         if index < 0 or index >= len(self.items):
#             raise IndexError(f"Buffer index {index} out of range [0, {len(self.items)})")
#         item = self.items.pop(index)
#         self.total_packed += 1
#         self._fill_buffer()
#         return item
#
#     def discard_item(self, index: int) -> Box:
#         \"\"\"
#         Discard an item from the buffer (item cannot fit in any bin).
#         Still refills from conveyor.
#         \"\"\"
#         item = self.items.pop(index)
#         self.total_discarded += 1
#         self._fill_buffer()
#         return item
#
#     def peek(self) -> List[Box]:
#         \"\"\"Return current buffer contents without removing.\"\"\"
#         return list(self.items)
#
#     def is_empty(self) -> bool:
#         return len(self.items) == 0
#
#     @property
#     def current_size(self) -> int:
#         return len(self.items)
#
#     @property
#     def remaining_on_conveyor(self) -> int:
#         return len(self.conveyor)
#
#     def get_feature_vector(self) -> np.ndarray:
#         \"\"\"
#         Encode buffer contents as a fixed-size feature vector for the
#         neural network.
#
#         Returns:
#             np.ndarray of shape (capacity, 4): each row = (w, d, h, weight)
#             Padded with zeros for empty slots.
#         \"\"\"
#         features = np.zeros((self.capacity, 4), dtype=np.float32)
#         for i, item in enumerate(self.items):
#             features[i] = [item.width, item.depth, item.height, item.weight]
#         return features
#
#     def get_mask(self) -> np.ndarray:
#         \"\"\"
#         Binary mask indicating valid items in the buffer.
#
#         Returns:
#             np.ndarray of shape (capacity,): 1 for valid, 0 for empty slot
#         \"\"\"
#         mask = np.zeros(self.capacity, dtype=np.float32)
#         mask[:len(self.items)] = 1.0
#         return mask


# =============================================================================
# SECTION 2: ITEM SELECTION STRATEGIES
# =============================================================================

"""
2.1 Strategy Overview

The item selection problem: given the current buffer and bin states,
which item should be packed next?

This is a key difference from Deep-Pack, which has no choice (items
arrive one by one, take-it-or-leave-it).

Strategies range from simple heuristics to learned policies:

Strategy A: Largest-First
  - Select the item with the largest volume
  - Rationale: large items are harder to fit later; place them first
  - Classic offline wisdom applied to semi-online

Strategy B: Best-Fit-First
  - For each buffer item, compute the best placement score across both bins
  - Select the item with the highest best-placement-score
  - Rationale: place the item that fits best right now

Strategy C: Urgency-Based
  - Items that have been in the buffer longest get priority
  - Prevents indefinite deferral of awkward items
  - Can be combined with other criteria

Strategy D: DRL-Based (Deep-Pack extension)
  - Train a neural network to score buffer items
  - The network sees bin states + buffer contents
  - Outputs a probability distribution over buffer items
  - This is the approach detailed in deep_rl/deep_pack_3d_coding_ideas.py

Strategy E: Hybrid (Recommended for thesis)
  - Use heuristic pre-filtering to narrow buffer to top-k candidates
  - Then use DRL to select among the top-k
  - Reduces exploration space while maintaining learning flexibility
"""

# class ItemSelectionStrategy:
#     \"\"\"Base class for item selection strategies.\"\"\"
#     def select(self, buffer: 'ItemBuffer', bin_states: list) -> int:
#         raise NotImplementedError
#
#
# class LargestFirstStrategy(ItemSelectionStrategy):
#     \"\"\"Select the item with the largest volume.\"\"\"
#     def select(self, buffer: 'ItemBuffer', bin_states: list) -> int:
#         items = buffer.peek()
#         if not items:
#             return -1
#         volumes = [b.width * b.depth * b.height for b in items]
#         return int(np.argmax(volumes))
#
#
# class BestFitFirstStrategy(ItemSelectionStrategy):
#     \"\"\"
#     Select the item that has the best available placement across all bins.
#
#     For each item in the buffer:
#       1. Generate candidate placements in each active bin
#       2. Score each candidate using a heuristic (DBLF, support ratio, etc.)
#       3. The item's "best fit score" = max score across all candidates
#     Select the item with the highest best fit score.
#     \"\"\"
#     def __init__(self, candidate_generator, scorer):
#         self.candidate_generator = candidate_generator
#         self.scorer = scorer
#
#     def select(self, buffer: 'ItemBuffer', bin_states: list) -> int:
#         items = buffer.peek()
#         if not items:
#             return -1
#
#         best_scores = []
#         for item in items:
#             item_best = -float('inf')
#             for bin_state in bin_states:
#                 candidates = self.candidate_generator(bin_state, item)
#                 if candidates:
#                     scores = [self.scorer(bin_state, c) for c in candidates]
#                     item_best = max(item_best, max(scores))
#             best_scores.append(item_best)
#
#         return int(np.argmax(best_scores))
#
#
# class UrgencyWeightedStrategy(ItemSelectionStrategy):
#     \"\"\"
#     Combines fit score with urgency (time in buffer).
#
#     score = alpha * fit_score + (1 - alpha) * urgency_score
#
#     urgency_score = time_in_buffer / max_allowed_time
#     \"\"\"
#     def __init__(self, base_strategy: ItemSelectionStrategy, alpha=0.7):
#         self.base = base_strategy
#         self.alpha = alpha
#         self.arrival_times = {}  # item_id -> timestep when entered buffer
#         self.current_step = 0
#
#     def select(self, buffer: 'ItemBuffer', bin_states: list) -> int:
#         # Track arrival times for new items
#         items = buffer.peek()
#         for item in items:
#             if item.id not in self.arrival_times:
#                 self.arrival_times[item.id] = self.current_step
#
#         # Get base scores (from fit-based strategy)
#         base_idx = self.base.select(buffer, bin_states)
#
#         # Combine with urgency
#         max_wait = 20  # maximum acceptable wait (in steps)
#         urgency_scores = []
#         for item in items:
#             wait = self.current_step - self.arrival_times.get(item.id, self.current_step)
#             urgency_scores.append(min(wait / max_wait, 1.0))
#
#         # Simple weighted combination
#         # (In practice, you'd compute fit scores for each item individually)
#         combined = np.array(urgency_scores)  # placeholder
#         self.current_step += 1
#
#         return int(np.argmax(combined))


# =============================================================================
# SECTION 3: BUFFER-AWARE EPISODE MANAGEMENT
# =============================================================================

"""
3.1 Episode Loop with Buffer

An episode in our semi-online setup differs from Deep-Pack:

Deep-Pack episode:
  1. Generate W^2 items
  2. Send one by one
  3. Place or discard each immediately
  4. Episode ends when bin full or all items processed

Our episode:
  1. Generate N items (e.g., 50-200)
  2. Fill buffer with first BUFFER_SIZE items
  3. Repeat:
     a. Agent selects item from buffer (buffer selection strategy/policy)
     b. Agent selects bin and placement location (DRL placement policy)
     c. Item is removed from buffer; next conveyor item enters buffer
     d. If a bin is "closed" (too full), open a new one (2-bounded logic)
  4. Episode ends when buffer empty AND conveyor empty,
     OR when all bins are closed and no new bins allowed

Key metric: total volume packed across all bins / total volume of all bins used
Secondary metric: stability scores of all placed items (min, mean, median)
"""

# def run_episode(buffer: 'ItemBuffer',
#                 item_selector: 'ItemSelectionStrategy',
#                 placement_agent,  # DRL agent or heuristic
#                 bin_manager: 'TwoBoundedManager',
#                 items: list,
#                 max_steps: int = 1000) -> dict:
#     \"\"\"
#     Run one episode of semi-online bin packing with buffer.
#
#     Args:
#         buffer: ItemBuffer instance (initially empty)
#         item_selector: Strategy for selecting items from buffer
#         placement_agent: Agent that decides where to place selected item
#         bin_manager: Manages 2-bounded space (opening/closing bins)
#         items: Full list of items for this episode (conveyor sequence)
#         max_steps: Safety limit on number of steps
#
#     Returns:
#         dict with metrics: volume_utilization, stability_scores,
#         items_placed, items_discarded, bins_used, etc.
#     \"\"\"
#     buffer.load_conveyor(items)
#     step = 0
#     total_placed = 0
#     total_discarded = 0
#     rewards = []
#
#     while not buffer.is_empty() and step < max_steps:
#         # Get current bin states
#         active_bins = bin_manager.get_active_bins()
#
#         # Select item from buffer
#         item_idx = item_selector.select(buffer, active_bins)
#
#         if item_idx < 0:
#             break  # no item selectable (should not happen if buffer non-empty)
#
#         selected_item = buffer.peek()[item_idx]
#
#         # Try to place the item using the placement agent
#         placement_result = placement_agent.find_placement(
#             selected_item, active_bins
#         )
#
#         if placement_result is not None:
#             bin_id, x, y, orientation = placement_result
#             target_bin = active_bins[bin_id]
#
#             # Execute placement
#             placement = target_bin.place(selected_item, x, y, orientation)
#             buffer.select_item(item_idx)
#             total_placed += 1
#
#             # Compute reward
#             reward = compute_placement_reward(target_bin, placement)
#             rewards.append(reward)
#
#             # Check if bin should be closed
#             if bin_manager.should_close_bin(bin_id):
#                 bin_manager.close_bin(bin_id)
#
#         else:
#             # Item cannot be placed in any active bin
#             # Options:
#             #   a) Discard the item
#             #   b) Try another item from buffer first
#             #   c) Close a bin and open new one
#
#             # Strategy: try another item first (up to buffer_size attempts)
#             placed = False
#             for alt_idx in range(buffer.current_size):
#                 if alt_idx == item_idx:
#                     continue
#                 alt_item = buffer.peek()[alt_idx]
#                 alt_result = placement_agent.find_placement(alt_item, active_bins)
#                 if alt_result is not None:
#                     bin_id, x, y, orientation = alt_result
#                     target_bin = active_bins[bin_id]
#                     target_bin.place(alt_item, x, y, orientation)
#                     buffer.select_item(alt_idx)
#                     total_placed += 1
#                     placed = True
#                     break
#
#             if not placed:
#                 # No item from buffer fits -> close fullest bin, open new
#                 if bin_manager.can_close_and_open():
#                     bin_manager.close_fullest_and_open_new()
#                 else:
#                     # Must discard
#                     buffer.discard_item(item_idx)
#                     total_discarded += 1
#
#         step += 1
#
#     # Compute final metrics
#     metrics = {
#         'total_placed': total_placed,
#         'total_discarded': total_discarded,
#         'total_items': len(items),
#         'bins_used': bin_manager.total_bins_used,
#         'volume_utilization': bin_manager.average_volume_utilization(),
#         'mean_reward': np.mean(rewards) if rewards else 0,
#         'min_stability': bin_manager.min_stability_score(),
#         'mean_stability': bin_manager.mean_stability_score(),
#     }
#     return metrics


# =============================================================================
# SECTION 4: BUFFER SIZE ANALYSIS AND SENSITIVITY
# =============================================================================

"""
4.1 Impact of Buffer Size on Performance

Theory (from overview KB):
  - Buffer size = 0: Strictly online (Deep-Pack original)
  - Buffer size = 1: Trivially online (no choice, must pack or discard)
  - Buffer size = 5-10: Semi-online with limited lookahead
  - Buffer size = N (all items): Essentially offline (can see everything)

Expected behavior:
  - Increasing buffer from 0 to ~5 should show rapid improvement
    (the first few items of lookahead are most valuable)
  - Increasing from 5 to 10 should show diminishing returns
  - Beyond 10: marginal improvements, increasing computational cost

Experiment design for thesis:
  - Fix all other parameters (bin size, item distribution, algorithm)
  - Vary buffer size: 1, 2, 3, 5, 7, 10, 15, 20
  - Measure: volume utilization, stability, computation time per step
  - Plot: performance vs. buffer size curve (expect logarithmic shape)

This experiment directly contributes to thesis findings by quantifying
the value of lookahead in semi-online packing.
"""

# def buffer_size_experiment(item_sequences: list,
#                             buffer_sizes: list = [1, 2, 3, 5, 7, 10, 15, 20],
#                             num_runs: int = 100) -> dict:
#     \"\"\"
#     Experiment to measure the impact of buffer size on packing performance.
#
#     Args:
#         item_sequences: List of item sequences to test
#         buffer_sizes: Buffer sizes to evaluate
#         num_runs: Number of runs per configuration
#
#     Returns:
#         dict mapping buffer_size -> {mean_utilization, std_utilization,
#                                      mean_stability, computation_time}
#     \"\"\"
#     results = {}
#
#     for buf_size in buffer_sizes:
#         utils = []
#         stabs = []
#         times = []
#
#         for seq in item_sequences[:num_runs]:
#             buffer = ItemBuffer(capacity=buf_size)
#             bin_mgr = TwoBoundedManager(bin_w=40, bin_d=40, bin_h=40)
#             selector = BestFitFirstStrategy(...)
#             agent = ...  # trained DRL agent or heuristic
#
#             import time
#             t0 = time.time()
#             metrics = run_episode(buffer, selector, agent, bin_mgr, seq)
#             elapsed = time.time() - t0
#
#             utils.append(metrics['volume_utilization'])
#             stabs.append(metrics['mean_stability'])
#             times.append(elapsed)
#
#         results[buf_size] = {
#             'mean_utilization': np.mean(utils),
#             'std_utilization': np.std(utils),
#             'mean_stability': np.mean(stabs),
#             'std_stability': np.std(stabs),
#             'mean_time': np.mean(times),
#         }
#
#     return results


# =============================================================================
# SECTION 5: INTEGRATION WITH 2-BOUNDED SPACE
# =============================================================================

"""
5.1 Two-Bounded Manager Interface

The buffer interacts closely with the 2-bounded bin manager:
  - When selecting an item from buffer, we need to know which bins are active
  - When no item fits in any active bin, we may need to close a bin
  - Closing a bin is irreversible (overview KB Section 5)
  - The buffer provides flexibility: maybe another item in the buffer DOES fit

Decision flow:

  1. Buffer presents current items
  2. For each item, check if it fits in any active bin
  3. If yes: select best (item, bin, placement) combination
  4. If no item fits in active bins:
     a. Close the fullest active bin (maximize its utilization)
     b. Open a new empty bin
     c. Retry with new bin configuration
  5. If still no fit (item too large for any empty bin):
     a. Discard the item from buffer
     b. Try next item

The buffer gives us a CRUCIAL advantage over strictly online:
  We don't have to discard the first item that doesn't fit.
  We can try other items from the buffer before giving up.
  This is expected to significantly improve utilization.
"""

# class TwoBoundedManager:
#     \"\"\"
#     Manages the 2-bounded space constraint: at most 2 bins active at once.
#
#     Based on overview KB Section 5:
#       "Only a restricted, finite number k of bins is open (active).
#        If no active bin has enough space, one active bin is closed
#        permanently and a new one is opened. Once closed, a bin can
#        never be reopened."
#     \"\"\"
#     def __init__(self, bin_w, bin_d, bin_h, k=2):
#         self.bin_w = bin_w
#         self.bin_d = bin_d
#         self.bin_h = bin_h
#         self.k = k  # max active bins (2 for our case)
#
#         self.active_bins: List[BinState] = []
#         self.closed_bins: List[BinState] = []
#
#         # Start with k empty bins
#         for _ in range(k):
#             self._open_new_bin()
#
#     def _open_new_bin(self):
#         new_bin = BinState(
#             bin_width=self.bin_w,
#             bin_depth=self.bin_d,
#             bin_height=self.bin_h,
#             heightmap=np.zeros((self.bin_w, self.bin_d)),
#         )
#         self.active_bins.append(new_bin)
#
#     def get_active_bins(self) -> List['BinState']:
#         return self.active_bins
#
#     @property
#     def total_bins_used(self) -> int:
#         return len(self.active_bins) + len(self.closed_bins)
#
#     def should_close_bin(self, bin_idx: int, threshold: float = 0.95) -> bool:
#         \"\"\"
#         Heuristic: close a bin if its utilization exceeds threshold.
#         This frees a slot for a new empty bin.
#         \"\"\"
#         return self.active_bins[bin_idx].volume_utilization() >= threshold
#
#     def close_bin(self, bin_idx: int):
#         \"\"\"Close the specified active bin and open a new one.\"\"\"
#         closed = self.active_bins.pop(bin_idx)
#         self.closed_bins.append(closed)
#         self._open_new_bin()
#
#     def close_fullest_and_open_new(self):
#         \"\"\"Close the active bin with highest utilization; open new.\"\"\"
#         if not self.active_bins:
#             return
#         utils = [b.volume_utilization() for b in self.active_bins]
#         fullest_idx = int(np.argmax(utils))
#         self.close_bin(fullest_idx)
#
#     def can_close_and_open(self) -> bool:
#         \"\"\"Can we close a bin and open a new one? Always yes for online.\"\"\"
#         return len(self.active_bins) > 0
#
#     def average_volume_utilization(self) -> float:
#         all_bins = self.active_bins + self.closed_bins
#         if not all_bins:
#             return 0.0
#         return np.mean([b.volume_utilization() for b in all_bins])
#
#     def min_stability_score(self) -> float:
#         # Placeholder -- requires stability checker integration
#         return 0.0
#
#     def mean_stability_score(self) -> float:
#         return 0.0


# =============================================================================
# SECTION 6: COMPLEXITY AND FEASIBILITY
# =============================================================================

"""
6.1 Per-Step Complexity with Buffer

Without buffer (Deep-Pack):
  - 1 item, 1 bin, W*H actions -> O(W*H) action evaluation

With buffer (our extension):
  - B items in buffer, 2 bins, ~C candidates per (item, bin) pair
  - Total candidates: B * 2 * C
  - With B=10, C=50: 1000 total candidates
  - CNN forward pass per candidate (or batched): ~10-50ms on GPU
  - Total per step: ~50-100ms (acceptable for robotic operation)

6.2 Feasibility Notes

- Buffer management itself is O(B) per step -- negligible
- The main cost is in candidate generation and Q-network evaluation
- With heuristic pre-filtering (extreme points), candidates are ~50 per bin
- Batched GPU evaluation keeps inference fast
- Memory overhead of buffer: negligible (just B Box objects)

6.3 Training Considerations

- More decisions per episode: item selection + placement
- Larger state space: buffer contents add variability
- Recommend: pre-train placement policy on single-bin without buffer,
  then fine-tune with buffer (curriculum learning)
- Alternatively: train item selector and placement agent separately
  (hierarchical decomposition)
"""
