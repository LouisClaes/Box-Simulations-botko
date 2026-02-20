"""
=============================================================================
CODING IDEAS: 2-Bounded Space Bin Management for Semi-Online 3D BPP
Based on: Lee & Nam (2025) - "A Hierarchical Bin Packing Framework..."
         + Overview KB (Ali et al. 2022) - Section 5 on Bounded Space
=============================================================================

FOCUS: Extending the paper's single-bin hierarchical framework to 2-bounded
space (k=2), where only 2 bins/pallets are active at any time.

OVERVIEW KB CONTEXT (Section 5):
    "Only a restricted, finite number k of bins is open (active) at any time.
     If no active bin has enough space, one active bin is closed permanently
     and a new one is opened. Once closed, a bin can never be reopened."
    This is identified as Gap 4: "severe lack of studies on bounded 3D-PPs."

=============================================================================
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
import numpy as np


# =============================================================================
# 1. BIN SELECTION STRATEGIES
# =============================================================================

"""
The paper operates with a SINGLE bin. With k=2, we must decide for each item:
    "Which of the 2 active bins should receive this item?"

This is a NEW decision layer not present in the paper. We place it ABOVE
the paper's hierarchical search (between the buffer and the tree search).

Strategies range from simple heuristics to learned policies.
"""


class BinSelectionStrategy(Enum):
    """Strategies for selecting which active bin to use."""

    # Strategy 1: Best Fit -- pack into the bin where the item fits best
    BEST_FIT = "best_fit"

    # Strategy 2: Worst Fit -- pack into the bin with more remaining space
    # (keeps bins balanced, gives more options later)
    WORST_FIT = "worst_fit"

    # Strategy 3: Score-based -- run tree search on both bins, pick higher score
    TREE_SEARCH_SCORE = "tree_search_score"

    # Strategy 4: Fill one, then the other (sequential)
    SEQUENTIAL_FILL = "sequential_fill"

    # Strategy 5: Learned policy (RL agent decides bin)
    RL_BIN_SELECTOR = "rl_bin_selector"

    # Strategy 6: Specialize bins by item size (large -> bin A, small -> bin B)
    SIZE_SPECIALIZATION = "size_specialization"


class TwoBoundedBinSelector:
    """Select which of 2 active bins to pack an item into.

    This is our key contribution for extending the paper to 2-bounded space.
    """

    def __init__(self, strategy: BinSelectionStrategy = BinSelectionStrategy.TREE_SEARCH_SCORE):
        self.strategy = strategy
        self.primary_bin = 0  # For SEQUENTIAL_FILL

    def select_bin(self, item, active_bins: list, tree_search=None,
                   buffer: list = None) -> int:
        """Select which bin index (0 or 1) to use for this item.

        Returns: 0 or 1 (index into active_bins)
        """
        if self.strategy == BinSelectionStrategy.BEST_FIT:
            return self._best_fit(item, active_bins)
        elif self.strategy == BinSelectionStrategy.WORST_FIT:
            return self._worst_fit(active_bins)
        elif self.strategy == BinSelectionStrategy.TREE_SEARCH_SCORE:
            return self._tree_search_score(item, active_bins, tree_search)
        elif self.strategy == BinSelectionStrategy.SEQUENTIAL_FILL:
            return self._sequential_fill(item, active_bins)
        elif self.strategy == BinSelectionStrategy.SIZE_SPECIALIZATION:
            return self._size_specialization(item, active_bins)
        return 0  # Default

    def _best_fit(self, item, active_bins) -> int:
        """Pack into the bin where the item gets the highest adjacency reward.

        Adapts the paper's low-level reward concept to bin selection.
        """
        best_bin = 0
        best_reward = -float('inf')

        for bin_idx, bin_state in enumerate(active_bins):
            for w, d, h in [(item.width, item.depth, item.height),
                            (item.depth, item.width, item.height)]:
                w_int, d_int = int(w), int(d)
                for x in range(bin_state.width - w_int + 1):
                    for z in range(bin_state.depth - d_int + 1):
                        if bin_state.can_place(w_int, d_int, h, x, z):
                            r = bin_state.compute_adjacency_reward(x, z, w_int, d_int)
                            if r > best_reward:
                                best_reward = r
                                best_bin = bin_idx

        return best_bin

    def _worst_fit(self, active_bins) -> int:
        """Pack into the bin with MORE remaining space (lower utilization).

        Rationale: keeps both bins balanced, preserving future flexibility.
        Classic 1D online heuristic adapted to our setting.
        """
        utils = [b.get_utilization() for b in active_bins]
        return int(np.argmin(utils))

    def _tree_search_score(self, item, active_bins, tree_search) -> int:
        """Run the paper's tree search for each bin, select the higher-scoring bin.

        This is the most principled approach: evaluate the FULL consequences
        of placing this item in bin 0 vs bin 1, including future placements.

        Computationally expensive (2x the cost), but most accurate.
        """
        if tree_search is None:
            return self._best_fit(item, active_bins)

        scores = []
        for bin_idx in range(2):
            # Run search with only this bin active
            action_seq = tree_search.search(
                bin_states=[active_bins[bin_idx]],
                buffer_items=[item],
                recognized_items=[],
                require_full_pack=False
            )
            if action_seq:
                score = sum(a.reward for a in action_seq)
                scores.append((bin_idx, score))
            else:
                scores.append((bin_idx, -1.0))

        scores.sort(key=lambda x: -x[1])
        return scores[0][0]

    def _sequential_fill(self, item, active_bins) -> int:
        """Fill the primary bin until threshold, then switch to secondary.

        Simple but effective: achieves good utilization on the primary bin
        before moving to the secondary.
        """
        primary_util = active_bins[self.primary_bin].get_utilization()
        if primary_util >= 0.85:
            return 1 - self.primary_bin  # Switch to other bin
        return self.primary_bin

    def _size_specialization(self, item, active_bins) -> int:
        """Specialize bins by item size: large items in bin 0, small in bin 1.

        Rationale: mixing large and small items creates gaps.
        Keeping them separate allows tighter packing within each bin.

        Threshold: median volume of typical items.
        """
        volume = item.width * item.depth * item.height
        max_item_volume = (active_bins[0].width * active_bins[0].depth *
                           active_bins[0].max_height * 0.125)  # 1/8 of bin
        if volume >= max_item_volume:
            return 0  # Large items -> bin 0
        return 1      # Small items -> bin 1


# =============================================================================
# 2. BIN CLOSING LOGIC
# =============================================================================

"""
Bin closing is the most CRITICAL and IRREVERSIBLE decision in bounded-space
packing. The overview KB (Section 5) states:
    "Once closed, a bin can never be reopened."

The paper doesn't address this at all (single bin only).
We must design careful closing policies.
"""


class BinClosingTrigger(Enum):
    """What triggers a bin closing evaluation."""
    NO_FIT = "no_buffer_item_fits"
    HIGH_UTILIZATION = "utilization_above_threshold"
    TIMER = "time_based_after_n_steps"
    COMBINED = "combined_criteria"


@dataclass
class BinClosingConfig:
    """Configuration for bin closing decisions."""
    min_utilization_to_close: float = 0.70     # Never close below this
    target_utilization: float = 0.90           # Close if above this AND other criteria met
    max_consecutive_no_fit: int = 3            # Close after N items don't fit
    max_steps_without_placement: int = 5       # Close if no placement for N steps
    consider_buffer_composition: bool = True   # Look at buffer to decide


class BinClosingController:
    """Controller for bin closing decisions in 2-bounded space.

    Design principles:
    1. Never close a bin with very low utilization (waste)
    2. Close a bin when it's unlikely to receive more items efficiently
    3. Consider BOTH bins: don't close if the other bin is also nearly full
    4. The buffer contents should inform the decision
    """

    def __init__(self, config: BinClosingConfig = None):
        self.config = config or BinClosingConfig()
        self.no_fit_counters = [0, 0]  # Per-bin counter of consecutive no-fits
        self.steps_without_placement = [0, 0]

    def evaluate_closing(self, bin_idx: int, active_bins: list,
                         buffer: list) -> bool:
        """Decide whether to close bin at bin_idx.

        Returns True if the bin should be closed.
        """
        bin_state = active_bins[bin_idx]
        other_bin = active_bins[1 - bin_idx]
        util = bin_state.get_utilization()
        other_util = other_bin.get_utilization()

        # Rule 1: Never close below minimum utilization
        if util < self.config.min_utilization_to_close:
            return False

        # Rule 2: Never close BOTH bins at high utilization simultaneously
        # (would force opening 2 new bins with no items placed)
        if other_util >= self.config.target_utilization:
            # Other bin is also very full -- don't close this one yet
            # unless absolutely nothing fits
            if self._any_item_fits(bin_state, buffer):
                return False

        # Rule 3: Close if no buffer items fit
        if not self._any_item_fits(bin_state, buffer):
            self.no_fit_counters[bin_idx] += 1
            if self.no_fit_counters[bin_idx] >= self.config.max_consecutive_no_fit:
                return True
        else:
            self.no_fit_counters[bin_idx] = 0

        # Rule 4: Close if above target utilization
        if util >= self.config.target_utilization:
            if self.config.consider_buffer_composition:
                # Check if remaining items are better suited to a fresh bin
                fresh_bin_reward = self._estimate_fresh_bin_reward(buffer, bin_state)
                current_bin_reward = self._estimate_current_bin_reward(bin_state, buffer)
                if fresh_bin_reward > current_bin_reward * 1.5:
                    return True
            else:
                return True

        # Rule 5: Close if too many steps without a placement
        if self.steps_without_placement[bin_idx] >= self.config.max_steps_without_placement:
            return util >= self.config.min_utilization_to_close

        return False

    def record_placement(self, bin_idx: int, success: bool):
        """Record whether a placement was made in this bin."""
        if success:
            self.steps_without_placement[bin_idx] = 0
        else:
            self.steps_without_placement[bin_idx] += 1

    def _any_item_fits(self, bin_state, buffer: list) -> bool:
        """Check if any buffer item can fit in the bin."""
        for item in buffer:
            for w, d, h in [(item.width, item.depth, item.height),
                            (item.depth, item.width, item.height)]:
                w_int, d_int = int(w), int(d)
                for x in range(bin_state.width - w_int + 1):
                    for z in range(bin_state.depth - d_int + 1):
                        if bin_state.can_place(w_int, d_int, h, x, z):
                            return True
        return False

    def _estimate_fresh_bin_reward(self, buffer: list, current_bin) -> float:
        """Estimate how well buffer items would fit in a fresh (empty) bin."""
        # Simplified: sum of item volumes that could fit
        total_volume = sum(i.width * i.depth * i.height for i in buffer)
        bin_volume = current_bin.width * current_bin.depth * current_bin.max_height
        return min(total_volume / bin_volume, 1.0)

    def _estimate_current_bin_reward(self, bin_state, buffer: list) -> float:
        """Estimate remaining capacity utilization potential."""
        remaining_volume = (bin_state.width * bin_state.depth * bin_state.max_height *
                            (1.0 - bin_state.get_utilization()))
        fittable_volume = 0.0
        for item in buffer:
            vol = item.width * item.depth * item.height
            if vol <= remaining_volume:
                fittable_volume += vol
        return fittable_volume / (bin_state.width * bin_state.depth * bin_state.max_height)


# =============================================================================
# 3. INTEGRATED 2-BOUNDED SPACE PIPELINE
# =============================================================================

"""
Complete pipeline that integrates:
    - Paper's hierarchical search (adapted for 2 bins)
    - Buffer management
    - Bin selection
    - Bin closing

This is the main class that a thesis implementation would use.
"""


class TwoBoundedSpacePipeline:
    """Complete semi-online 3D BPP pipeline with 2-bounded space.

    Architecture:
        Buffer (5-10 items)
            |
        Bin Selector (which of 2 bins?)
            |
        Hierarchical Search (position + orientation + repacking)
            |
        Stability Check
            |
        Bin Closing Controller (close + open new?)
    """

    def __init__(self, bin_width: int, bin_depth: int, max_height: float,
                 buffer_size: int = 10,
                 bin_strategy: BinSelectionStrategy = BinSelectionStrategy.TREE_SEARCH_SCORE,
                 closing_config: BinClosingConfig = None):

        self.bin_width = bin_width
        self.bin_depth = bin_depth
        self.max_height = max_height

        # Core components
        self.active_bins = [self._new_bin(), self._new_bin()]
        self.closed_bins: list = []
        self.bin_selector = TwoBoundedBinSelector(strategy=bin_strategy)
        self.closing_controller = BinClosingController(closing_config)
        self.buffer_size = buffer_size

        # Statistics
        self.stats = {
            "total_items_packed": 0,
            "total_bins_used": 2,
            "items_per_bin": [],
            "utilization_per_bin": [],
            "closing_reasons": [],
        }

    def _new_bin(self):
        """Create a new empty bin.

        In a real implementation, this would use the Bin class from
        hybrid_heuristic_ml/coding_ideas_hierarchical_bpp.py
        """
        # Placeholder -- would import from the main module
        pass

    def process_step(self, buffer_items: list, tree_search=None) -> dict:
        """Process one step of the semi-online pipeline.

        This is called once per conveyor cycle (paper: ~1.7 seconds).

        Returns: dict with action taken and metadata.
        """
        result = {"action": None, "bin_idx": None, "bin_closed": False}

        # Step 1: Check bin closing
        for bin_idx in range(2):
            if self.closing_controller.evaluate_closing(
                    bin_idx, self.active_bins, buffer_items):
                self._close_bin(bin_idx)
                result["bin_closed"] = True
                self.stats["closing_reasons"].append(
                    f"Bin {bin_idx}: util={self.active_bins[bin_idx]}")

        # Step 2: Run tree search across both bins
        # (This is the paper's Algorithm 1, extended to consider both bins)
        if tree_search:
            action_sequence = tree_search.search(
                bin_states=self.active_bins,
                buffer_items=buffer_items,
                recognized_items=[],
                require_full_pack=False
            )

            if action_sequence:
                best_action = action_sequence[0]
                result["action"] = best_action
                result["bin_idx"] = best_action.bin_index
                self.closing_controller.record_placement(
                    best_action.bin_index, success=True)
                self.stats["total_items_packed"] += 1
            else:
                # No placements possible -> close more full bin
                utils = [0.0, 0.0]  # Would compute from active_bins
                close_idx = int(np.argmax(utils))
                self._close_bin(close_idx)
                result["bin_closed"] = True

        return result

    def _close_bin(self, bin_idx: int):
        """Close an active bin and open a new one."""
        closed_bin = self.active_bins[bin_idx]
        self.closed_bins.append(closed_bin)

        # Record statistics
        # util = closed_bin.get_utilization()
        # self.stats["utilization_per_bin"].append(util)
        # self.stats["items_per_bin"].append(len(closed_bin.packed_items))

        # Open new bin
        self.active_bins[bin_idx] = self._new_bin()
        self.stats["total_bins_used"] += 1

        # Reset closing controller for this slot
        self.closing_controller.no_fit_counters[bin_idx] = 0
        self.closing_controller.steps_without_placement[bin_idx] = 0

    def get_final_statistics(self) -> dict:
        """Get final performance statistics."""
        # Close remaining active bins
        for bin_idx in range(2):
            self._close_bin(bin_idx)

        return {
            **self.stats,
            "mean_utilization": (np.mean(self.stats["utilization_per_bin"])
                                 if self.stats["utilization_per_bin"] else 0.0),
            "min_utilization": (np.min(self.stats["utilization_per_bin"])
                                if self.stats["utilization_per_bin"] else 0.0),
            "bins_per_item": (self.stats["total_bins_used"] /
                              max(self.stats["total_items_packed"], 1)),
        }


# =============================================================================
# 4. COMPLEXITY ANALYSIS FOR 2-BOUNDED SPACE
# =============================================================================

"""
COMPUTATIONAL OVERHEAD vs. SINGLE BIN (paper's setting):

The 2-bounded space extension adds the following costs:

1. Bin Selection:
   - BEST_FIT: O(|buffer| * |orientations| * W * D * 2) -- 2x single bin
   - WORST_FIT: O(1) -- trivial
   - TREE_SEARCH_SCORE: O(2 * TreeSearchCost) -- 2x single bin
   - SEQUENTIAL_FILL: O(1) -- trivial

2. Bin Closing:
   - _any_item_fits check: O(|buffer| * |orientations| * W * D)
   - Called once per step per bin: O(2 * |buffer| * |orientations| * W * D)
   - With buffer=10, orientations=2, W=D=10: O(4000) -- negligible

3. Tree Search Extension:
   - Paper's branching factor: |buffer| * |orientations|
   - Our branching factor: |buffer| * |orientations| * 2 (bins)
   - With SELECTION beam width control: still manageable
   - Suggested: reduce beam width from 5 to 3-4 to compensate

4. Memory:
   - 2x bin state storage (heightmaps, packed items)
   - Tree nodes store 2 bin states instead of 1
   - Overhead: ~2x, but absolute values are small (10x10 grids)

OVERALL: The 2-bounded space extension roughly DOUBLES computation
time but remains well within the paper's reported planning budget
of 1-2 seconds per step.

FEASIBILITY:
    - Paper: < 1.3 seconds per step for 10x10 2D single bin
    - Ours (estimated): 2-3 seconds per step for 10x10 3D dual bins
    - Conveyor cycle: ~1.7 seconds (paper's physical setup)
    - May need: pipeline planning (plan next step during current execution)
      which the paper already does (Section V-E, Figure 10)
"""


# =============================================================================
# 5. EXPERIMENTAL DESIGN: 2-BOUNDED SPACE EVALUATION
# =============================================================================

"""
THESIS EXPERIMENTS for 2-bounded space:

Experiment 1: Bin Selection Strategy Comparison
    - Fix: buffer=10, 3D items, random distribution
    - Compare: all 6 BinSelectionStrategy variants
    - Measure: total bins used, mean util, computation time
    - Expected: TREE_SEARCH_SCORE best util but 2x slower;
                SIZE_SPECIALIZATION interesting if items are bimodal

Experiment 2: Bin Closing Policy Comparison
    - Fix: buffer=10, strategy=TREE_SEARCH_SCORE
    - Vary: min_utilization_to_close in {0.60, 0.70, 0.80, 0.90}
    - Vary: max_consecutive_no_fit in {1, 3, 5}
    - Measure: total bins used, mean util per bin, variance
    - Expected: aggressive closing (low threshold) -> more bins, lower variance
                conservative closing (high threshold) -> fewer bins, higher variance

Experiment 3: k=1 vs k=2 vs k=3
    - Compare bounded space with different k values
    - Fixed buffer, items, strategy
    - Measure: total bins used, mean util, worst-case util
    - Expected: k=2 significantly better than k=1 for heterogeneous items
                k=3 marginal improvement over k=2

Experiment 4: Integration with Repacking
    - 2-bounded space with and without repacking
    - Measure: utilization improvement from repacking
    - Expected: repacking helps more in bounded space (more constraint
                on placements -> more suboptimal decisions to fix)

Experiment 5: Stress Test -- Adversarial Item Sequences
    - Design item sequences that are hard for 2-bounded space
    - E.g., alternating very large and very small items
    - E.g., items that barely fit in bin (leave unusable gaps)
    - Compare strategies under stress
    - Expected: TREE_SEARCH_SCORE most robust; SEQUENTIAL_FILL worst

BENCHMARK INSTANCES:
    - Use the paper's Algorithm 4 (100% Set generation) adapted for 3D
    - Use standard BR instances from the 3D BPP literature
    - Generate semi-online scenarios by shuffling item order
"""
