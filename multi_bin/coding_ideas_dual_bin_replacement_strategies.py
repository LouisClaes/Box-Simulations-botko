"""
==============================================================================
CODING IDEAS: Multi-Bin Management & Replacement Strategies
==============================================================================

Source Paper:
    Tsang, Y.P., Mo, D.Y., Chung, K.T., Lee, C.K.M. (2025).
    "A deep reinforcement learning approach for online and concurrent 3D bin
    packing optimisation with bin replacement strategies."
    Computers in Industry, 164, 104202.

Focus: Bin replacement strategies for 2-bounded space online packing.

This file isolates the MULTI-BIN MANAGEMENT aspect of the paper, which is
relevant independently of whether DRL or heuristics are used for placement.

Key Concept:
    In a 2-bounded space setup, exactly 2 bins (pallets) are active at any
    time. When neither bin can accommodate any item from the lookahead buffer,
    a "bin replacement" is triggered: one or both bins are closed (permanently)
    and replaced with fresh empty bins.

    The choice of WHICH bin to close is a critical decision that affects
    overall space utilization.

Strategies from Paper:
    1. replaceAll: Close both bins, open 2 fresh ones
    2. replaceMax: Close the fuller bin, keep the emptier one

Extensions for Our Thesis:
    3. replaceThreshold: Close bins above a utilization threshold
    4. replaceSmartLookahead: Predict which bin is more useful for upcoming items
    5. replaceLearned: Use a small RL agent to learn the replacement policy

Integration Points:
    - Works with ANY packing algorithm (DRL, heuristic, hybrid)
    - The bin replacement decision is SEPARATE from the item placement decision
    - Can be combined with: MCA, DBLF, BSSF, BVF, or any other placement method

==============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from abc import ABC, abstractmethod


# ==============================================================================
# SECTION 1: Bin Data Structure (Minimal for replacement logic)
# ==============================================================================

@dataclass
class BinState:
    """Minimal bin state for replacement decisions.

    This is a lightweight view of a bin that captures the information
    needed for replacement strategy decisions without full geometry.
    """
    bin_id: int
    utilization: float  # Volume utilization [0, 1]
    max_height_ratio: float  # max_height / bin_height [0, 1]
    num_placements: int  # Number of items placed
    largest_free_cuboid_volume: float  # Volume of largest remaining free space
    total_free_volume: float  # Sum of all free cuboid volumes
    num_free_cuboids: int  # Number of maximal free cuboids

    @property
    def fragmentation(self) -> float:
        """Measure of how fragmented the remaining space is.

        High fragmentation = many small free spaces = harder to fit items.
        Low fragmentation = few large free spaces = easier to fit items.

        Returns a value in [0, 1] where 0 = no fragmentation (one large space)
        and 1 = highly fragmented.
        """
        if self.total_free_volume <= 0 or self.num_free_cuboids <= 1:
            return 0.0
        # Ratio of largest free space to total free space
        concentration = self.largest_free_cuboid_volume / self.total_free_volume
        return 1.0 - concentration  # Higher = more fragmented


# ==============================================================================
# SECTION 2: Abstract Replacement Strategy Interface
# ==============================================================================

class ReplacementStrategy(ABC):
    """Abstract base class for bin replacement strategies.

    A replacement strategy decides which bin(s) to close when no feasible
    action exists in any active bin for any item in the lookahead buffer.
    """

    @abstractmethod
    def select_bins_to_replace(self,
                               bin_states: List[BinState],
                               lookahead_items: List[dict] = None
                               ) -> List[int]:
        """Select which bin(s) to replace.

        Args:
            bin_states: Current state of each active bin
            lookahead_items: Optional info about upcoming items

        Returns:
            List of bin indices to close/replace
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ==============================================================================
# SECTION 3: Paper Strategies
# ==============================================================================

class ReplaceAll(ReplacementStrategy):
    """Close all active bins and open fresh ones.

    From Tsang et al. 2025, Section 3.2.2.
    Simple but may waste space in partially-filled bins.

    Paper results show this is INFERIOR to ReplaceMax in dual-bin scenarios.
    """

    def select_bins_to_replace(self, bin_states, lookahead_items=None):
        return list(range(len(bin_states)))

    @property
    def name(self):
        return "replaceAll"


class ReplaceMax(ReplacementStrategy):
    """Close the bin with HIGHEST utilization, keep the rest.

    From Tsang et al. 2025, Section 3.2.2.
    The less-utilized bin is kept because it has more available space.

    Paper results: Consistently outperforms ReplaceAll.
    - k=5: +1.53% utilization over ReplaceAll
    - k=10: +1.60%
    - k=15: +2.35%

    Rationale: The fuller bin has less remaining space, making it less
    likely to accommodate diverse future items. The emptier bin offers
    more flexibility for upcoming items.
    """

    def select_bins_to_replace(self, bin_states, lookahead_items=None):
        if not bin_states:
            return []
        max_util_idx = max(range(len(bin_states)),
                           key=lambda i: bin_states[i].utilization)
        return [max_util_idx]

    @property
    def name(self):
        return "replaceMax"


# ==============================================================================
# SECTION 4: Extended Strategies (Our Contributions)
# ==============================================================================

class ReplaceMin(ReplacementStrategy):
    """Close the bin with LOWEST utilization, keep the fuller one.

    INVERSE of paper's ReplaceMax. Worth testing as a baseline.

    Rationale: The emptier bin may indicate a "bad start" that won't
    improve. The fuller bin is closer to completion and might benefit
    from a few more items.

    Expected: Likely WORSE than ReplaceMax in most cases, but could
    be better when items are very heterogeneous and the fuller bin
    has strategically arranged free spaces.
    """

    def select_bins_to_replace(self, bin_states, lookahead_items=None):
        if not bin_states:
            return []
        min_util_idx = min(range(len(bin_states)),
                           key=lambda i: bin_states[i].utilization)
        return [min_util_idx]

    @property
    def name(self):
        return "replaceMin"


class ReplaceFragmented(ReplacementStrategy):
    """Close the bin with HIGHEST fragmentation.

    Novel strategy not in the paper.

    Rationale: Even if a bin has low utilization, if its remaining space
    is highly fragmented (many small disconnected free spaces), it's
    unlikely to fit standard-sized items. Better to close it and start
    fresh.

    This considers the QUALITY of remaining space, not just the QUANTITY.
    """

    def select_bins_to_replace(self, bin_states, lookahead_items=None):
        if not bin_states:
            return []
        most_frag_idx = max(range(len(bin_states)),
                            key=lambda i: bin_states[i].fragmentation)
        return [most_frag_idx]

    @property
    def name(self):
        return "replaceFragmented"


class ReplaceThreshold(ReplacementStrategy):
    """Close bins above a utilization threshold.

    Novel strategy. Configurable threshold (default 0.70).

    If ALL bins are above threshold: close the fullest (like ReplaceMax).
    If NO bins are above threshold: close the most fragmented.
    If SOME bins are above: close those above threshold.

    Rationale: Bins above 70% utilization are "good enough" -- close them
    to lock in the high utilization. Bins below 70% still have potential
    for improvement.
    """

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def select_bins_to_replace(self, bin_states, lookahead_items=None):
        above = [i for i, bs in enumerate(bin_states)
                 if bs.utilization >= self.threshold]

        if len(above) == len(bin_states):
            # All above threshold -> close the fullest
            return [max(range(len(bin_states)),
                        key=lambda i: bin_states[i].utilization)]
        elif len(above) == 0:
            # None above threshold -> close most fragmented
            return [max(range(len(bin_states)),
                        key=lambda i: bin_states[i].fragmentation)]
        else:
            return above

    @property
    def name(self):
        return f"replaceThreshold({self.threshold:.0%})"


class ReplaceLookaheadAware(ReplacementStrategy):
    """Close the bin that is least compatible with upcoming items.

    Novel strategy. Uses lookahead buffer information to predict which
    bin is more likely to accommodate future items.

    Algorithm:
    1. For each active bin, count how many lookahead items COULD POTENTIALLY
       fit in its largest free cuboid (ignoring rotation for speed).
    2. Close the bin with the FEWEST potential fits.

    Rationale: The bin that can accommodate more upcoming items should be
    kept. This uses the semi-online information (lookahead buffer) for
    the replacement decision, not just the placement decision.
    """

    def select_bins_to_replace(self, bin_states, lookahead_items=None):
        if not bin_states:
            return []
        if not lookahead_items:
            # Fallback to ReplaceMax if no lookahead info
            return ReplaceMax().select_bins_to_replace(bin_states)

        # Score each bin by number of items that could potentially fit
        scores = []
        for bs in bin_states:
            # Simplified: check if any dimension of each item is smaller
            # than the largest free cuboid dimensions
            # In practice, this would use the actual maximal cuboid list
            fit_count = 0
            lfcv = bs.largest_free_cuboid_volume
            for item in lookahead_items:
                item_vol = item.get('volume', item.get('w', 1) *
                                    item.get('l', 1) * item.get('h', 1))
                if item_vol <= lfcv:
                    fit_count += 1
            scores.append(fit_count)

        # Close the bin with fewest potential fits
        min_score_idx = min(range(len(bin_states)),
                            key=lambda i: scores[i])
        return [min_score_idx]

    @property
    def name(self):
        return "replaceLookaheadAware"


class ReplaceComposite(ReplacementStrategy):
    """Weighted composite of multiple replacement criteria.

    Novel strategy. Scores each bin on multiple dimensions and closes
    the bin with the lowest "keep score."

    Keep Score = w1 * (1 - utilization)     # Prefer keeping emptier bins
               + w2 * (1 - fragmentation)   # Prefer keeping less fragmented
               + w3 * fit_potential          # Prefer keeping bins that fit items
               + w4 * (1 - height_ratio)     # Prefer keeping lower-height bins

    Close the bin with the LOWEST keep score (least worth keeping).
    """

    def __init__(self,
                 w_util: float = 0.3,
                 w_frag: float = 0.2,
                 w_fit: float = 0.3,
                 w_height: float = 0.2):
        self.w_util = w_util
        self.w_frag = w_frag
        self.w_fit = w_fit
        self.w_height = w_height

    def select_bins_to_replace(self, bin_states, lookahead_items=None):
        if not bin_states:
            return []

        scores = []
        for bs in bin_states:
            keep_score = (
                self.w_util * (1 - bs.utilization) +
                self.w_frag * (1 - bs.fragmentation) +
                self.w_height * (1 - bs.max_height_ratio)
            )

            # Fit potential (if lookahead available)
            if lookahead_items and self.w_fit > 0:
                fit_count = 0
                for item in lookahead_items:
                    item_vol = item.get('volume', 1)
                    if item_vol <= bs.largest_free_cuboid_volume:
                        fit_count += 1
                fit_ratio = fit_count / max(1, len(lookahead_items))
                keep_score += self.w_fit * fit_ratio
            else:
                keep_score += self.w_fit * 0.5  # Neutral if no lookahead

            scores.append(keep_score)

        # Close the bin with the lowest keep score
        min_score_idx = min(range(len(bin_states)),
                            key=lambda i: scores[i])
        return [min_score_idx]

    @property
    def name(self):
        return "replaceComposite"


# ==============================================================================
# SECTION 5: Multi-Bin Manager
# ==============================================================================

class MultiBinManager:
    """Manages multiple active bins with a configurable replacement strategy.

    This is the coordinator that sits above individual bin packing algorithms.
    It handles:
    - Tracking active bins
    - Deciding when to trigger replacement
    - Executing the replacement strategy
    - Tracking completed (closed) bins and their utilizations

    Compatible with ANY packing algorithm (DRL, heuristic, hybrid) because
    the placement logic is external.

    Usage:
        manager = MultiBinManager(
            n_bins=2,
            bin_config=(120, 80, 150),
            strategy=ReplaceMax()
        )

        while items_remaining:
            feasible = manager.get_all_feasible_actions(lookahead_items)
            if not feasible:
                manager.trigger_replacement(lookahead_items)
                feasible = manager.get_all_feasible_actions(lookahead_items)
                if not feasible:
                    break  # Truly stuck
            action = select_action(feasible)  # Your packing algorithm here
            manager.execute_action(action)
    """

    def __init__(self,
                 n_bins: int,
                 bin_config: Tuple[float, float, float],
                 strategy: ReplacementStrategy):
        self.n_bins = n_bins
        self.bin_config = bin_config
        self.strategy = strategy
        self.active_bins = []  # List of active Bin objects (placeholder type)
        self.completed_bins = []  # Closed bins with their final utilizations
        self.replacement_count = 0

        # Initialize active bins
        self._init_bins()

    def _init_bins(self):
        """Create initial set of empty bins."""
        # This would create actual Bin objects with MCA
        # Placeholder: store as dicts
        self.active_bins = [
            {'id': i, 'utilization': 0.0, 'placements': []}
            for i in range(self.n_bins)
        ]

    def get_bin_states(self) -> List[BinState]:
        """Extract BinState from each active bin for the replacement strategy."""
        states = []
        for b in self.active_bins:
            states.append(BinState(
                bin_id=b['id'],
                utilization=b.get('utilization', 0.0),
                max_height_ratio=b.get('max_height_ratio', 0.0),
                num_placements=len(b.get('placements', [])),
                largest_free_cuboid_volume=b.get('largest_free_volume', 0.0),
                total_free_volume=b.get('total_free_volume', 0.0),
                num_free_cuboids=b.get('num_free_cuboids', 1),
            ))
        return states

    def trigger_replacement(self, lookahead_items=None):
        """Execute bin replacement using the configured strategy.

        Returns the indices of replaced bins for logging.
        """
        bin_states = self.get_bin_states()
        indices_to_replace = self.strategy.select_bins_to_replace(
            bin_states, lookahead_items)

        for idx in sorted(indices_to_replace, reverse=True):
            closed_bin = self.active_bins[idx]
            self.completed_bins.append({
                'bin': closed_bin,
                'final_utilization': closed_bin.get('utilization', 0.0)
            })
            # Replace with fresh bin
            self.active_bins[idx] = {
                'id': len(self.completed_bins) + self.n_bins,
                'utilization': 0.0,
                'placements': [],
            }

        self.replacement_count += 1
        return indices_to_replace

    def get_statistics(self) -> dict:
        """Get summary statistics of all completed bins."""
        if not self.completed_bins:
            return {
                'avg_utilization': 0.0,
                'min_utilization': 0.0,
                'max_utilization': 0.0,
                'num_completed': 0,
                'num_replacements': self.replacement_count,
            }

        utils = [b['final_utilization'] for b in self.completed_bins]
        return {
            'avg_utilization': np.mean(utils),
            'min_utilization': np.min(utils),
            'max_utilization': np.max(utils),
            'std_utilization': np.std(utils),
            'num_completed': len(self.completed_bins),
            'num_replacements': self.replacement_count,
            'strategy': self.strategy.name,
        }


# ==============================================================================
# SECTION 6: Benchmarking Framework
# ==============================================================================

def benchmark_replacement_strategies(
    n_experiments: int = 100,
    n_items: int = 200,
    bin_config: Tuple[float, float, float] = (32, 32, 32),
    n_bins: int = 2,
    lookahead_k: int = 10
) -> dict:
    """Run a benchmarking experiment comparing all replacement strategies.

    This creates a controlled experiment where the ONLY variable is the
    replacement strategy. The packing algorithm is held constant.

    In practice, pair this with a specific packing algorithm (e.g., BSSF
    heuristic) to isolate the effect of the replacement strategy.

    Args:
        n_experiments: Number of random instances to test
        n_items: Items per instance
        bin_config: (W, L, H) of bins
        n_bins: Number of active bins (2 for our use case)
        lookahead_k: Lookahead buffer size

    Returns:
        Dict mapping strategy name -> aggregate statistics
    """
    strategies = [
        ReplaceAll(),
        ReplaceMax(),
        ReplaceMin(),
        ReplaceFragmented(),
        ReplaceThreshold(0.65),
        ReplaceThreshold(0.70),
        ReplaceThreshold(0.75),
        ReplaceLookaheadAware(),
        ReplaceComposite(),
    ]

    results = {}

    for strategy in strategies:
        print(f"Testing strategy: {strategy.name}")
        experiment_utils = []

        for exp in range(n_experiments):
            manager = MultiBinManager(n_bins, bin_config, strategy)
            # ... run packing algorithm with this manager ...
            # stats = manager.get_statistics()
            # experiment_utils.append(stats['avg_utilization'])
            pass  # Placeholder -- integrate with actual packing algorithm

        results[strategy.name] = {
            # 'avg': np.mean(experiment_utils),
            # 'std': np.std(experiment_utils),
            # 'min': np.min(experiment_utils),
            # 'max': np.max(experiment_utils),
        }

    return results


# ==============================================================================
# SECTION 7: Implementation Priority and Integration Notes
# ==============================================================================

"""
IMPLEMENTATION PLAN FOR MULTI-BIN MANAGEMENT

Priority 1 (MUST HAVE):
    - Implement MultiBinManager with ReplaceMax and ReplaceAll
    - Integrate with MCA from deep_rl/coding_ideas_tsang2025_ddqn_dual_bin.py
    - Test with simple heuristic (BSSF) as baseline

Priority 2 (SHOULD HAVE):
    - Implement ReplaceThreshold and ReplaceFragmented
    - Benchmark all strategies against ReplaceMax baseline
    - Determine if any strategy consistently beats ReplaceMax

Priority 3 (NICE TO HAVE):
    - Implement ReplaceLookaheadAware and ReplaceComposite
    - Consider learning the replacement policy with a small RL agent
    - Sensitivity analysis on threshold values

INTEGRATION WITH OTHER CODE:
    - deep_rl/: The MultiBinManager wraps the DRL packing agent
    - heuristics/: The MultiBinManager wraps any heuristic packer
    - semi_online_buffer/: Buffer management is SEPARATE from bin replacement
        - Buffer decides WHICH ITEM to pack next
        - MultiBinManager decides WHICH BIN to close when stuck
    - stability/: Stability metrics can inform fragmentation scoring in
        replacement strategies (e.g., bins with unstable top surfaces
        are harder to pack on top of)

ESTIMATED EFFORT:
    - Core MultiBinManager + ReplaceMax/All: 2-3 hours
    - Extended strategies: 1 day
    - Benchmarking framework integration: 1-2 days
    - Full testing: 1 day
"""


if __name__ == "__main__":
    # Quick demo of replacement strategy selection
    print("Multi-Bin Replacement Strategy Demo")
    print("=" * 50)

    # Simulate two bin states
    bin_states = [
        BinState(bin_id=0, utilization=0.72, max_height_ratio=0.85,
                 num_placements=15, largest_free_cuboid_volume=3000,
                 total_free_volume=8000, num_free_cuboids=12),
        BinState(bin_id=1, utilization=0.58, max_height_ratio=0.65,
                 num_placements=10, largest_free_cuboid_volume=6000,
                 total_free_volume=14000, num_free_cuboids=6),
    ]

    strategies = [
        ReplaceAll(),
        ReplaceMax(),
        ReplaceMin(),
        ReplaceFragmented(),
        ReplaceThreshold(0.70),
        ReplaceComposite(),
    ]

    print(f"\nBin 0: util={bin_states[0].utilization:.0%}, "
          f"frag={bin_states[0].fragmentation:.2f}")
    print(f"Bin 1: util={bin_states[1].utilization:.0%}, "
          f"frag={bin_states[1].fragmentation:.2f}")
    print()

    for strategy in strategies:
        to_replace = strategy.select_bins_to_replace(bin_states)
        print(f"{strategy.name:30s} -> Replace bin(s): {to_replace}")


# ==============================================================================
# SECTION 8: Learned Replacement Strategy (DRL-based)
# ==============================================================================

class ReplaceByQLearning(ReplacementStrategy):
    """Learn the replacement policy using a small Q-network.

    Novel strategy for our thesis. Instead of hand-crafted rules,
    train a small neural network to predict which bin should be closed.

    State: [util_0, frag_0, height_0, util_1, frag_1, height_1, buffer_stats]
    Action: 0 = replace bin 0, 1 = replace bin 1, 2 = replace both
    Reward: utilization of the COMPLETED bin(s)

    This is a SEPARATE RL agent from the placement DQN. It only activates
    when a replacement is triggered and makes a one-shot decision.

    Architecture: Small FC network (no CNN needed -- input is a feature vector)
        Input: (14,) -- 7 features per bin
        FC(14, 64) -> ReLU -> FC(64, 32) -> ReLU -> FC(32, 3) -> Q-values

    Training: Can be trained jointly with the placement DQN or separately.
    Separate training recommended (simpler, less interference).
    """

    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Path to a trained Q-network weights file.
                        If None, falls back to ReplaceMax.
        """
        self.model_path = model_path
        self.model = None
        self._fallback = ReplaceMax()

    def select_bins_to_replace(self, bin_states, lookahead_items=None):
        if self.model is None:
            # No trained model available -- use fallback
            return self._fallback.select_bins_to_replace(bin_states,
                                                          lookahead_items)

        # Encode state as feature vector
        features = self._encode_state(bin_states, lookahead_items)

        # Forward pass through Q-network
        # q_values = self.model.predict(features)
        # action = argmax(q_values)

        # Decode action
        # if action == 0: return [0]
        # elif action == 1: return [1]
        # else: return [0, 1]

        # Placeholder until model is trained
        return self._fallback.select_bins_to_replace(bin_states, lookahead_items)

    def _encode_state(self, bin_states, lookahead_items=None) -> list:
        """Encode bin states and buffer info into a fixed-size feature vector.

        Features per bin (7):
            0: utilization [0,1]
            1: fragmentation [0,1]
            2: max_height_ratio [0,1]
            3: num_placements (normalized)
            4: largest_free_cuboid_volume (normalized)
            5: total_free_volume (normalized)
            6: num_free_cuboids (normalized)

        Total: 7 * n_bins = 14 features
        """
        features = []
        for bs in bin_states:
            features.extend([
                bs.utilization,
                bs.fragmentation,
                bs.max_height_ratio,
                bs.num_placements / 50.0,  # Normalize by expected max
                bs.largest_free_cuboid_volume / 100000.0,  # Normalize
                bs.total_free_volume / 200000.0,  # Normalize
                bs.num_free_cuboids / 100.0,  # Normalize
            ])
        return features

    @property
    def name(self):
        return "replaceByQLearning"


# ==============================================================================
# SECTION 9: Proactive Replacement (Close Before Getting Stuck)
# ==============================================================================

class ReplaceProactive(ReplacementStrategy):
    """Proactively close a bin when its remaining utility drops below threshold.

    Novel strategy for our thesis. Instead of waiting until NO items fit
    (reactive replacement), close a bin when its expected remaining
    contribution is low.

    Trigger condition (different from paper):
        Paper: triggered when NO feasible action exists across all bins
        This: triggered when a bin's "remaining potential" drops below threshold

    Remaining potential = (largest_free_cuboid_volume / avg_item_volume)

    If a bin's largest free space can fit fewer than `min_remaining_items`
    expected items, close it proactively.

    Benefit: Avoids the scenario where a bin slowly fills with tiny items
    at poor positions, wasting time and degrading overall utilization.

    Risk: May close bins too early, leaving unrealized utilization.
    """

    def __init__(self, min_remaining_items: float = 2.0,
                 avg_item_volume: float = 729.0):
        """
        Args:
            min_remaining_items: Close bin if it can fit fewer than this
                                 many average items in its largest free space
            avg_item_volume: Expected average item volume (default: 9^3 = 729
                             for the paper; adjust for real warehouse items)
        """
        self.min_remaining_items = min_remaining_items
        self.avg_item_volume = avg_item_volume

    def select_bins_to_replace(self, bin_states, lookahead_items=None):
        if not bin_states:
            return []

        to_replace = []
        for i, bs in enumerate(bin_states):
            remaining_capacity = bs.largest_free_cuboid_volume / self.avg_item_volume
            if remaining_capacity < self.min_remaining_items:
                to_replace.append(i)

        if not to_replace:
            # Nothing to proactively replace -- use ReplaceMax as fallback
            return ReplaceMax().select_bins_to_replace(bin_states, lookahead_items)

        return to_replace

    @property
    def name(self):
        return f"replaceProactive(min={self.min_remaining_items:.0f})"


# ==============================================================================
# SECTION 10: Replacement Strategy Comparison Analysis
# ==============================================================================

"""
COMPREHENSIVE COMPARISON OF ALL REPLACEMENT STRATEGIES

Strategy              | Decision Rule                          | Expected Performance | Complexity
------------------------------------------------------------------------------------------
ReplaceAll            | Close all bins                         | Baseline (lowest)    | O(1)
ReplaceMax            | Close fullest bin                      | Good (paper's best)  | O(n_bins)
ReplaceMin            | Close emptiest bin                     | Poor (expected)      | O(n_bins)
ReplaceFragmented     | Close most fragmented bin              | Good (novel)         | O(n_bins)
ReplaceThreshold      | Close bins above util threshold        | Good (tunable)       | O(n_bins)
ReplaceLookaheadAware | Close bin least compatible with buffer | Very good (novel)    | O(n_bins * k)
ReplaceComposite      | Weighted multi-criteria score          | Very good (novel)    | O(n_bins * k)
ReplaceByQLearning    | Learned policy                         | Unknown (needs data) | O(n_bins) + inference
ReplaceProactive      | Close before getting stuck             | Good (novel)         | O(n_bins)

PAPER RESULTS FOR REFERENCE (Dual bin, DRL, k=10):
    ReplaceAll: 75.89% average utilization
    ReplaceMax: 77.11% average utilization
    Difference: +1.22% (statistically significant)

EXPERIMENTAL PLAN FOR OUR THESIS:
    1. Implement all strategies
    2. Test with BSSF heuristic first (fast, consistent)
    3. Test with DDQN (after training)
    4. Report:
        - Average utilization (mean +/- std over 1000 instances)
        - Number of pallets used per 200 items
        - Replacement trigger frequency
        - Variance in utilization across pallets
        - Computation time per strategy

HYPOTHESIS:
    ReplaceLookaheadAware and ReplaceComposite should outperform ReplaceMax
    because they use more information (buffer contents + fragmentation) to
    make the replacement decision. ReplaceMax only uses utilization.

    ReplaceProactive may improve throughput by avoiding wasted placements
    in nearly-full bins, but may reduce average utilization per bin.
"""


# ==============================================================================
# SECTION 11: Integration with MCA for Full Pipeline
# ==============================================================================

class FullPipelineManager:
    """Complete multi-bin manager integrated with MCA and packing heuristics.

    This brings together:
        - MultiBinManager (this file) for bin lifecycle management
        - MaximalCuboidsAlgorithm (from deep_rl/ coding ideas) for space management
        - Replacement strategies (this file) for bin closure decisions
        - Heuristic scoring (configurable) for placement decisions

    This is a STANDALONE implementation that works WITHOUT DRL.
    It serves as:
        1. A heuristic baseline for comparison with DRL
        2. A testing framework for replacement strategies
        3. A deployment option when DRL inference is too slow
    """

    def __init__(self, bin_config: Tuple[float, float, float],
                 n_bins: int = 2,
                 strategy: ReplacementStrategy = None,
                 heuristic: str = 'bssf',
                 buffer_size: int = 10):
        self.bin_config = bin_config
        self.n_bins = n_bins
        self.strategy = strategy or ReplaceMax()
        self.heuristic = heuristic
        self.buffer_size = buffer_size

        self.bins = []
        self.completed_bins = []
        self.replacement_count = 0
        self.items_packed = 0
        self.items_skipped = 0

    def run_instance(self, items: List[dict]) -> dict:
        """Run a complete packing instance.

        Args:
            items: List of item dicts with 'w', 'l', 'h' keys

        Returns:
            Statistics dict
        """
        # Initialize
        W, L, H = self.bin_config
        self.bins = [{'W': W, 'L': L, 'H': H, 'util': 0.0, 'placements': [],
                       'free_vol': W * L * H, 'largest_free': W * L * H,
                       'num_free': 1}
                      for _ in range(self.n_bins)]
        self.completed_bins = []
        self.replacement_count = 0
        self.items_packed = 0
        self.items_skipped = 0

        queue = list(items)

        while queue:
            buffer = queue[:self.buffer_size]

            # Try to find a feasible placement
            placed = False
            for item in buffer:
                for bin_idx in range(len(self.bins)):
                    # Simplified: check if item volume fits in largest free space
                    item_vol = item['w'] * item['l'] * item['h']
                    if item_vol <= self.bins[bin_idx]['largest_free']:
                        # Place item (simplified -- real impl uses MCA)
                        self.bins[bin_idx]['util'] += item_vol / (W * L * H)
                        self.bins[bin_idx]['free_vol'] -= item_vol
                        self.bins[bin_idx]['largest_free'] *= 0.7  # Rough approx
                        self.bins[bin_idx]['placements'].append(item)
                        queue.remove(item)
                        self.items_packed += 1
                        placed = True
                        break
                if placed:
                    break

            if not placed:
                # Trigger replacement
                bin_states = [
                    BinState(
                        bin_id=i,
                        utilization=b['util'],
                        max_height_ratio=b['util'],  # Rough approximation
                        num_placements=len(b['placements']),
                        largest_free_cuboid_volume=b['largest_free'],
                        total_free_volume=b['free_vol'],
                        num_free_cuboids=b['num_free'],
                    )
                    for i, b in enumerate(self.bins)
                ]
                to_replace = self.strategy.select_bins_to_replace(
                    bin_states,
                    [{'volume': item['w'] * item['l'] * item['h']}
                     for item in buffer]
                )

                if to_replace:
                    for idx in sorted(to_replace, reverse=True):
                        self.completed_bins.append(self.bins[idx])
                        self.bins[idx] = {
                            'W': W, 'L': L, 'H': H, 'util': 0.0,
                            'placements': [], 'free_vol': W * L * H,
                            'largest_free': W * L * H, 'num_free': 1
                        }
                    self.replacement_count += 1
                else:
                    # Skip the first item in buffer
                    if queue:
                        queue.pop(0)
                        self.items_skipped += 1

        # Finalize
        self.completed_bins.extend(self.bins)
        active_bins = [b for b in self.completed_bins if b['placements']]
        utils = [b['util'] for b in active_bins]

        return {
            'strategy': self.strategy.name,
            'avg_utilization': np.mean(utils) if utils else 0.0,
            'std_utilization': np.std(utils) if utils else 0.0,
            'num_bins': len(active_bins),
            'items_packed': self.items_packed,
            'items_skipped': self.items_skipped,
            'replacements': self.replacement_count,
        }
