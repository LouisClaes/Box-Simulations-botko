"""
Tree of Packing (ToP) for Semi-Online Buffer + 2-Bounded Space
===============================================================

Based on: "Deliberate Planning of 3D Bin Packing on Packing Configuration Trees"
Authors: Zhao, Xu, Yu, Hu, Zhu, Du, Xu (2025)
Paper: ICLR 2022 (conference) + SAGE Int. J. Robotics Research 2025 (extended)
Code: https://github.com/alexfrom0815/Online-3D-BPP-PCT (~250 stars)

This file focuses specifically on adapting the ToP planning framework
for the thesis use case:
  - Semi-online: buffer of 5-10 items
  - 2-bounded space: exactly 2 pallets active at any time
  - Goals: maximize fill rate + ensure stability
  - Robotic/conveyor setup

KEY INSIGHT FROM THE PAPER:
The ToP framework (Section 3.4) already handles buffering packing
(s > 1, p = 0, u > 0) by modeling item selection as a search tree.
The paper does NOT handle multi-bin (k-bounded space).
Extending ToP to 2-bounded space is the primary thesis contribution.

APPROACH: Extend the ToP search tree so each decision node encodes:
  (which_item_from_buffer, which_bin_to_place_in)
instead of just (which_item_from_buffer).

This doubles the branching factor but remains tractable with MCTS.

THREE ARCHITECTURAL OPTIONS FOR 2-BOUNDED EXTENSION:
Option A: Dual-PCT with Separate Bin Selection Head
  - Two independent PCTs + learned bin selector MLP
  - Requires training the bin selector head
  - Sequential decisions (bin selection then placement)

Option B: Extended ToP Search (RECOMMENDED)
  - Same pre-trained pi_theta, NO modification
  - MCTS search tree extended: each node = (item_index, bin_index)
  - Branching factor = s * 2 (buffer_size * num_bins)
  - No retraining needed! Most aligned with paper framework.

Option C: Joint Multi-Bin PCT (Research-Level)
  - Single PCT encoding BOTH bins simultaneously
  - Items tagged with bin_id in the graph
  - Requires retraining from scratch; larger graph

RECOMMENDATION: Option B because:
  1. No retraining of pi_theta needed
  2. Most directly extends the paper's framework
  3. Implementation is well-defined
  4. Novel contribution: "ToP + k-bounded + stability-aware closing"

EXPECTED PERFORMANCE (from paper results + engineering estimates):
| Configuration                         | Fill Rate | Stability |
|---------------------------------------|-----------|-----------|
| Buffer=5, no stability                | 83-88%    | N/A       |
| Buffer=5, stability (c=0.1)          | 78-85%    | 70-85%    |
| Buffer=5, 2-bounded, stability        | 73-82%    | 70-85%    |
| Buffer=10, no stability               | 88-93%    | N/A       |
| Buffer=10, stability (c=0.1)         | 83-90%    | 70-85%    |
| Buffer=10, 2-bounded, stability       | 78-87%    | 70-85%    |
| Buffer=10, 2-bounded, physics verify  | 75-84%    | 95-100%   |

THESIS NOVEL CONTRIBUTION:
"Extending Tree of Packing (ToP) to k-bounded space (k=2) with
stability-aware bin closing for semi-online robotic packing."
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time


# =============================================================================
# SECTION 1: ToP SEARCH TREE FOR BUFFER + 2 BINS
# =============================================================================

@dataclass
class ToP_Action:
    """
    An action in the ToP search tree.

    In the original paper (single bin):
      action = item_index  (which item from buffer to pack next)

    In our extension (2-bounded space):
      action = (item_index, bin_index)

    The placement POSITION within the chosen bin is determined by
    the pre-trained PCT policy pi_theta -- it is NOT part of the search.
    This is a key design choice that keeps the search tractable.
    """
    item_index: int         # Index into the buffer
    bin_index: int          # 0 or 1 (which active bin)
    item_id: int = -1       # Unique item identifier
    estimated_value: float = 0.0  # pi_theta's evaluation of this placement

    def __hash__(self):
        return hash((self.item_index, self.bin_index))

    def __eq__(self, other):
        return (self.item_index == other.item_index and
                self.bin_index == other.bin_index)


@dataclass
class ToP_SearchState:
    """
    State at a node in the ToP search tree.

    Contains:
    - The current PCT state for each bin (or a compact representation)
    - Which items remain in the buffer
    - The value estimate V(.) for unknown future items
    """
    bin_states: List[Any]       # PCT states for each active bin
    remaining_buffer: List[int] # Indices of items still in buffer
    cumulative_volume: float    # Total volume placed so far in this path
    state_value: float = 0.0   # V(.) estimate for future items

    @property
    def total_value(self) -> float:
        """Path value = packed volumes + estimated future value."""
        return self.cumulative_volume + self.state_value


class ToP_SearchNode:
    """
    Node in the ToP search tree.

    Each node represents a state after a sequence of (item, bin) decisions.
    Children correspond to different next actions.

    The tree is built during MCTS search and shared across time steps
    via the global cache (from the paper).
    """

    def __init__(self, state: Optional[ToP_SearchState] = None,
                 parent: Optional['ToP_SearchNode'] = None,
                 action: Optional[ToP_Action] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[ToP_Action, 'ToP_SearchNode'] = {}

        # MCTS statistics
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior_probability: float = 1.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_terminal(self) -> bool:
        """No more items in buffer or no feasible placements."""
        if self.state is None:
            return False
        return len(self.state.remaining_buffer) == 0

    @property
    def q_value(self) -> float:
        """Average value of simulations through this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float = 2.0) -> float:
        """
        PUCT score for MCTS selection (Silver et al. 2016 style).

        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Where:
        - Q = average value
        - P = prior probability (from policy network)
        - N(s) = parent visit count
        - N(s,a) = this node's visit count
        """
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = (c_puct * self.prior_probability *
                       np.sqrt(parent_visits) / (1 + self.visit_count))
        return self.q_value + exploration

    def best_child(self, c_puct: float = 2.0) -> 'ToP_SearchNode':
        """Select child with highest UCB score."""
        return max(self.children.values(),
                   key=lambda n: n.ucb_score(c_puct))

    def most_visited_child(self) -> 'ToP_SearchNode':
        """Select child with most visits (for final decision)."""
        return max(self.children.values(),
                   key=lambda n: n.visit_count)


# =============================================================================
# SECTION 2: MCTS PLANNER FOR 2-BOUNDED BUFFER PACKING
# =============================================================================

class ToP_MCTS_Planner:
    """
    MCTS-based planner for Tree of Packing with buffer and 2-bounded space.

    The planner searches over:
    - Which item from the buffer to pack next
    - Which of the 2 active bins to place it in

    The WHERE to place within the bin is determined by the pre-trained
    PCT policy (pi_theta), not by the search.

    Key parameters:
    - num_simulations: MCTS iterations (default 200)
    - c_puct: exploration constant (default 2.0)
    - max_depth: maximum search depth (default = buffer_size)

    Based on paper Section 3.4 and MCTS from Silver et al. (2016).
    """

    def __init__(self, policy_network, value_network,
                 num_simulations: int = 200,
                 c_puct: float = 2.0,
                 temperature: float = 1.0):
        self.policy_network = policy_network
        self.value_network = value_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

        # Global cache for path reuse across time steps
        # Key: frozenset of remaining buffer item IDs
        # Value: best path found
        self.global_cache: Dict[frozenset, List[ToP_Action]] = {}

    def plan(self, bin_states: List[Any], buffer: List[Any],
             check_feasibility_fn, evaluate_placement_fn,
             get_state_value_fn) -> ToP_Action:
        """
        Run MCTS planning and return the best first action.

        Args:
            bin_states: current state of each active bin (PCT objects)
            buffer: list of items currently in the buffer
            check_feasibility_fn: (item, bin_idx, bin_state) -> bool
            evaluate_placement_fn: (item, bin_idx, bin_state) -> (placement, value)
            get_state_value_fn: (bin_states) -> float  [V(.) for future items]

        Returns:
            ToP_Action: the best (item_index, bin_index) to execute
        """
        # Check global cache first
        buffer_key = frozenset(id(item) for item in buffer)
        if buffer_key in self.global_cache:
            cached_path = self.global_cache[buffer_key]
            if cached_path:
                return cached_path[0]

        # Initialize root
        root_state = ToP_SearchState(
            bin_states=[s for s in bin_states],
            remaining_buffer=list(range(len(buffer))),
            cumulative_volume=0.0,
            state_value=get_state_value_fn(bin_states)
        )
        root = ToP_SearchNode(state=root_state)

        # Run MCTS simulations
        for sim in range(self.num_simulations):
            self._simulate(root, buffer, bin_states,
                           check_feasibility_fn, evaluate_placement_fn,
                           get_state_value_fn)

        # Select best action
        if not root.children:
            # No feasible action found
            return None

        best_child = root.most_visited_child()
        best_action = best_child.action

        # Cache the best path
        self._cache_best_path(root, buffer_key)

        return best_action

    def _simulate(self, root, buffer, bin_states,
                  check_feasibility_fn, evaluate_placement_fn,
                  get_state_value_fn):
        """
        Single MCTS simulation: SELECT -> EXPAND -> EVALUATE -> BACKPROPAGATE
        """
        node = root
        path = [node]

        # SELECTION: traverse existing tree using UCB
        while not node.is_leaf and not node.is_terminal:
            node = node.best_child(self.c_puct)
            path.append(node)

        # EXPANSION: add children for unexplored actions
        if not node.is_terminal:
            self._expand(node, buffer, check_feasibility_fn,
                         evaluate_placement_fn)

            # Select one new child for evaluation
            if node.children:
                node = node.best_child(self.c_puct)
                path.append(node)

        # EVALUATION: estimate value of this leaf node
        if node.state is not None:
            value = node.state.total_value
        else:
            value = 0.0

        # BACKPROPAGATION: update all ancestors
        for n in reversed(path):
            n.visit_count += 1
            n.total_value += value

    def _expand(self, node, buffer, check_feasibility_fn,
                evaluate_placement_fn):
        """
        Expand node by creating children for all feasible (item, bin) actions.

        For each remaining buffer item x each active bin (2 bins):
        - Check if item fits in bin
        - If yes, create child node with updated state
        - Use pi_theta to determine placement and estimate value
        """
        if node.state is None:
            return

        remaining = node.state.remaining_buffer
        if not remaining:
            return

        for item_idx in remaining:
            item = buffer[item_idx]
            for bin_idx in range(2):  # 2-bounded space
                # Check feasibility
                if not check_feasibility_fn(item, bin_idx, node.state.bin_states[bin_idx]):
                    continue

                # Use policy to evaluate placement
                placement, placement_value = evaluate_placement_fn(
                    item, bin_idx, node.state.bin_states[bin_idx])

                if placement is None:
                    continue

                action = ToP_Action(
                    item_index=item_idx,
                    bin_index=bin_idx,
                    estimated_value=placement_value
                )

                # Create child state
                new_remaining = [i for i in remaining if i != item_idx]
                new_cumulative = node.state.cumulative_volume + placement_value

                child_state = ToP_SearchState(
                    bin_states=node.state.bin_states,  # Note: need deep copy in practice
                    remaining_buffer=new_remaining,
                    cumulative_volume=new_cumulative,
                    state_value=0.0  # Will be estimated during evaluation
                )

                child = ToP_SearchNode(
                    state=child_state,
                    parent=node,
                    action=action
                )
                child.prior_probability = placement_value  # Use as prior

                node.children[action] = child

    def _cache_best_path(self, root, buffer_key):
        """Extract and cache the best path from root."""
        path = []
        node = root
        while node.children:
            best_child = node.most_visited_child()
            path.append(best_child.action)
            node = best_child
        self.global_cache[buffer_key] = path


# =============================================================================
# SECTION 3: SPATIAL ENSEMBLE FOR 2-BOUNDED SPACE
# =============================================================================

class SpatialEnsembleRanker:
    """
    Spatial Ensemble for cross-bin evaluation.

    From paper Section 3.3: When evaluating placements across multiple
    sub-bins (or in our case, multiple active bins), direct value comparison
    is unfair because score scales differ.

    Solution: Convert absolute scores to RANKS within each bin,
    then select the placement with the best worst-case rank.

    For 2-bounded space adaptation:
    - Each bin = a "sub-bin" in the spatial ensemble terminology
    - For each candidate leaf node, compute its rank within each bin
    - Select the placement that maximizes: max_l min_c rank(l, c)

    This ensures we don't systematically favor one bin over the other.
    """

    def rank_and_select(self,
                        candidates_bin0: List[Tuple[Any, float]],
                        candidates_bin1: List[Tuple[Any, float]],
                        ) -> Tuple[int, Any]:
        """
        Rank candidates across two bins and select the best.

        Args:
            candidates_bin0: list of (placement, score) for bin 0
            candidates_bin1: list of (placement, score) for bin 1

        Returns:
            (bin_index, best_placement)
        """
        # Rank within each bin (higher score = higher rank)
        ranks_bin0 = self._compute_ranks(candidates_bin0)
        ranks_bin1 = self._compute_ranks(candidates_bin1)

        # For items that appear in both bins (same item, different bin),
        # take the worst rank across bins
        best_rank = -1
        best_bin = 0
        best_placement = None

        for placement, rank in ranks_bin0.items():
            if rank > best_rank:
                best_rank = rank
                best_bin = 0
                best_placement = placement

        for placement, rank in ranks_bin1.items():
            if rank > best_rank:
                best_rank = rank
                best_bin = 1
                best_placement = placement

        return best_bin, best_placement

    def _compute_ranks(self, candidates: List[Tuple[Any, float]]) -> Dict[Any, float]:
        """Convert absolute scores to ranks (0 to 1 normalized)."""
        if not candidates:
            return {}

        sorted_candidates = sorted(candidates, key=lambda x: x[1])
        n = len(sorted_candidates)
        ranks = {}
        for rank, (placement, score) in enumerate(sorted_candidates):
            ranks[id(placement)] = (rank + 1) / n  # Normalized rank [1/n, 1]
        return ranks


# =============================================================================
# SECTION 4: BIN CLOSING STRATEGY
# =============================================================================

class BinClosingStrategy:
    """
    Strategy for deciding when to close a bin in 2-bounded space.

    This is crucial for good performance in k-bounded settings.
    Once a bin is closed, it can NEVER be reopened.

    Three strategies implemented:

    1. NO-FIT CLOSING: Close when no buffer item fits
       - Simple but may close bins prematurely

    2. VALUE-THRESHOLD CLOSING: Close when V(.) drops below threshold
       - Uses the learned state value to predict remaining utility
       - More sophisticated but requires calibrating the threshold

    3. UTILIZATION-THRESHOLD CLOSING: Close when utilization exceeds threshold
       - Pragmatic approach for high fill rates
       - Risk: may close bins that could still accommodate small items

    For the thesis: start with NO-FIT, then experiment with VALUE-THRESHOLD.
    """

    def __init__(self, strategy: str = 'no_fit',
                 value_threshold: float = 0.1,
                 utilization_threshold: float = 0.90):
        self.strategy = strategy
        self.value_threshold = value_threshold
        self.utilization_threshold = utilization_threshold

    def should_close(self, bin_pct, buffer: List[Any],
                     value_fn=None) -> bool:
        """
        Determine if a bin should be closed.

        Args:
            bin_pct: PackingConfigurationTree for the bin
            buffer: current buffer items
            value_fn: function to estimate V(.) for the bin state

        Returns:
            True if bin should be closed
        """
        if self.strategy == 'no_fit':
            return self._no_fit_closing(bin_pct, buffer)
        elif self.strategy == 'value_threshold':
            return self._value_threshold_closing(bin_pct, value_fn)
        elif self.strategy == 'utilization_threshold':
            return self._utilization_threshold_closing(bin_pct)
        elif self.strategy == 'hybrid':
            return self._hybrid_closing(bin_pct, buffer, value_fn)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _no_fit_closing(self, bin_pct, buffer) -> bool:
        """Close if no buffer item can fit."""
        for item in buffer:
            leaves = bin_pct.get_feasible_leaves(item)
            if len(leaves) > 0:
                return False
        return True

    def _value_threshold_closing(self, bin_pct, value_fn) -> bool:
        """Close if estimated future utility is below threshold."""
        if value_fn is None:
            return False
        v = value_fn(bin_pct)
        return v < self.value_threshold

    def _utilization_threshold_closing(self, bin_pct) -> bool:
        """Close if utilization exceeds threshold."""
        return bin_pct.utilization >= self.utilization_threshold

    def _hybrid_closing(self, bin_pct, buffer, value_fn) -> bool:
        """
        Hybrid: close if (no fit) OR (high utilization AND low future value).

        This is the recommended strategy for the thesis.
        """
        # Must close if nothing fits
        if self._no_fit_closing(bin_pct, buffer):
            return True

        # Consider closing if utilization is high and value is low
        if bin_pct.utilization > 0.80:
            if value_fn is not None:
                v = value_fn(bin_pct)
                if v < self.value_threshold * 2:  # Relaxed threshold at high utilization
                    return True

        return False


# =============================================================================
# SECTION 5: COMPLETE SEMI-ONLINE PIPELINE
# =============================================================================

class SemiOnline2BoundedPacker:
    """
    Complete semi-online packing system with buffer and 2-bounded space.

    This is the main class for the thesis implementation.

    Architecture:
    1. BufferManager: manages the 5-10 item buffer
    2. TwoBinPCTManager: manages 2 active bins with PCTs
    3. ToP_MCTS_Planner: searches over (item, bin) orderings
    4. SpatialEnsembleRanker: ensures fair cross-bin comparison
    5. BinClosingStrategy: decides when to close a bin
    6. StabilityChecker: verifies placement stability

    Flow per time step:
    1. Check if any bin should be closed
    2. Generate candidates for all buffer items in both bins
    3. Run MCTS to find best (item, bin) action
    4. Execute placement using pi_theta
    5. Update buffer (remove packed item, add new from stream)
    6. Repeat

    Expected performance (based on paper results):
    - Buffer=5: ~88-89% per-bin utilization
    - Buffer=10: ~89-91% per-bin utilization
    - With stability constraints: subtract 3-8%
    - With 2-bounded space overhead: subtract 2-5%
    - NET EXPECTED: 75-85% average utilization
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict with keys:
                - bin_size: np.ndarray (3,)
                - buffer_size: int (5-10)
                - num_orientations: int (2 or 6)
                - mcts_simulations: int (100-500)
                - closing_strategy: str ('no_fit', 'value_threshold', 'hybrid')
                - check_stability: bool
                - stability_support_threshold: float (0.6-1.0)
        """
        self.config = config
        self.bin_size = config['bin_size']
        self.buffer_size = config.get('buffer_size', 5)

        # Components (to be initialized with trained models)
        self.planner = None  # ToP_MCTS_Planner -- set after training
        self.ranker = SpatialEnsembleRanker()
        self.closing_strategy = BinClosingStrategy(
            strategy=config.get('closing_strategy', 'hybrid')
        )

        # State
        self.bins = None  # TwoBinPCTManager
        self.buffer = []
        self.statistics = defaultdict(list)

    def initialize(self, policy_network=None, value_network=None):
        """Initialize the packing system with trained models."""
        from pct_coding_ideas import TwoBinPCTManager
        self.bins = TwoBinPCTManager(
            self.bin_size,
            num_orientations=self.config.get('num_orientations', 2)
        )

        if policy_network and value_network:
            self.planner = ToP_MCTS_Planner(
                policy_network=policy_network,
                value_network=value_network,
                num_simulations=self.config.get('mcts_simulations', 200),
            )

    def pack_stream(self, item_stream) -> dict:
        """
        Main packing loop. Packs items from stream until exhausted.

        Returns dict with comprehensive statistics.
        """
        self.initialize()
        self._fill_buffer(item_stream)

        total_packed = 0
        total_volume_packed = 0.0
        bin_volume = float(np.prod(self.bin_size))

        start_time = time.time()

        while self.buffer:
            # Step 1: Check bin closing
            self._check_and_close_bins()

            # Step 2: Plan using MCTS (or fallback to greedy)
            action = self._plan_action()

            if action is None:
                # No feasible action for any item in any bin
                # Force close both bins and open new ones
                self._force_close_all_bins()
                continue

            # Step 3: Execute placement
            success = self._execute_action(action)

            if success:
                total_packed += 1
                total_volume_packed += action.estimated_value
                self.statistics['per_item_time'].append(
                    time.time() - start_time
                )

            # Step 4: Refill buffer
            self._fill_buffer(item_stream)

        elapsed = time.time() - start_time

        return {
            'total_items_packed': total_packed,
            'total_bins_used': self.bins.total_bins_used,
            'average_utilization': self.bins.get_total_utilization(),
            'total_volume_packed': total_volume_packed,
            'elapsed_time': elapsed,
            'avg_time_per_item': elapsed / max(1, total_packed),
            'items_per_bin': total_packed / max(1, self.bins.total_bins_used),
        }

    def _fill_buffer(self, item_stream):
        """Fill buffer from stream up to capacity."""
        while len(self.buffer) < self.buffer_size:
            try:
                item = next(item_stream)
                self.buffer.append(item)
            except StopIteration:
                break

    def _check_and_close_bins(self):
        """Check if any active bin should be closed."""
        for bin_idx in range(2):
            if self.bins.active_bins[bin_idx]:
                pct = self.bins.bins[bin_idx]
                if self.closing_strategy.should_close(pct, self.buffer):
                    self.bins.close_bin(bin_idx)

    def _plan_action(self) -> Optional[ToP_Action]:
        """
        Use MCTS planner (or greedy fallback) to select best action.
        """
        if self.planner is not None:
            return self.planner.plan(
                bin_states=[b for b in self.bins.bins],
                buffer=self.buffer,
                check_feasibility_fn=self._check_feasibility,
                evaluate_placement_fn=self._evaluate_placement,
                get_state_value_fn=self._get_state_value,
            )
        else:
            return self._greedy_action()

    def _greedy_action(self) -> Optional[ToP_Action]:
        """
        Greedy fallback: for each item x bin, find best fit.
        Select the (item, bin) with highest immediate value.

        This serves as a baseline before MCTS is implemented.
        """
        best_action = None
        best_value = -float('inf')

        for item_idx, item in enumerate(self.buffer):
            for bin_idx in range(2):
                if not self.bins.active_bins[bin_idx]:
                    continue
                candidates = self.bins.bins[bin_idx].get_feasible_leaves(item)
                if candidates:
                    # Simple heuristic: pick placement with best support
                    value = item.volume  # Basic: just volume
                    if value > best_value:
                        best_value = value
                        best_action = ToP_Action(
                            item_index=item_idx,
                            bin_index=bin_idx,
                            estimated_value=value
                        )

        return best_action

    def _execute_action(self, action: ToP_Action) -> bool:
        """Execute the chosen action: place item in bin."""
        item = self.buffer[action.item_index]
        bin_pct = self.bins.bins[action.bin_index]

        candidates = bin_pct.get_feasible_leaves(item, check_stability=True)
        if not candidates:
            return False

        # In practice: use pi_theta to select among candidates
        # For now: pick first feasible
        placement = candidates[0]
        bin_pct.place_item(item, placement)
        self.buffer.pop(action.item_index)
        return True

    def _force_close_all_bins(self):
        """Close all active bins and open new ones."""
        for bin_idx in range(2):
            if self.bins.active_bins[bin_idx]:
                self.bins.close_bin(bin_idx)

    def _check_feasibility(self, item, bin_idx, bin_state) -> bool:
        """Check if item has any feasible placement in the bin."""
        leaves = bin_state.get_feasible_leaves(item)
        return len(leaves) > 0

    def _evaluate_placement(self, item, bin_idx, bin_state):
        """Use policy to evaluate best placement for item in bin."""
        leaves = bin_state.get_feasible_leaves(item)
        if not leaves:
            return None, 0.0
        # In practice: use pi_theta to score each leaf
        return leaves[0], item.volume

    def _get_state_value(self, bin_states) -> float:
        """Estimate future value using critic network."""
        # In practice: use value_network
        return 0.0


# =============================================================================
# SECTION 6: TRAINING CONSIDERATIONS
# =============================================================================

"""
TRAINING STRATEGY FOR THESIS:

Phase 1: Train base PCT policy on single-bin online packing (no buffer, no multi-bin)
  - Follow paper's training setup (ACKTR, 64 parallel envs)
  - Use discrete solution space first (S^d = 10)
  - Train for ~500K steps
  - Validate: should achieve ~70-76% utilization (Setting 1)
  - Estimated time: 12-24 hours on single GPU

Phase 2: Validate ToP buffer planning with trained policy
  - Use trained pi_theta from Phase 1
  - Implement MCTS planner with buffer of 5 items
  - No retraining needed! (key advantage of ToP)
  - Compare against greedy buffer baseline
  - Expected improvement: +5-15% utilization over pure online

Phase 3: Extend to 2-bounded space
  - Add bin selection to MCTS search tree
  - Implement bin closing strategy
  - Implement spatial ensemble ranking
  - No retraining of pi_theta needed!
  - Validate on simulated conveyor streams

Phase 4: Add stability constraints
  - Integrate stability checking into leaf node filtering
  - Add stability reward term: w_t = max(0, v_t + c * f_stability)
  - Retrain pi_theta with stability-aware reward
  - Add physics-based test-time verification (PyBullet)

Phase 5 (optional): Fine-tune for distribution
  - If real warehouse item distribution is available
  - Fine-tune pi_theta on this distribution
  - The paper shows good generalization across distributions (Table 12)
    so this may not be strictly necessary

SIMPLIFICATIONS FOR THESIS:
1. Use PPO instead of ACKTR (easier to implement with stable-baselines3)
2. Use 8-16 parallel environments instead of 64
3. Start with integer coordinates before moving to continuous
4. Use simple support-area stability before physics-based
5. Reduce MCTS to 50-100 simulations for interactive debugging
6. Use beam search as MCTS alternative for simpler implementation

LIBRARIES:
- PyTorch >= 1.12 (neural networks)
- stable-baselines3 (PPO training alternative)
- PyBullet (physics simulation for stability)
- numpy (geometry computations)
- gymnasium (RL environment interface)
"""


# =============================================================================
# SECTION 7: EVALUATION METRICS
# =============================================================================

class ParkingMetrics:
    """
    Comprehensive metrics for evaluating the packing system.

    Following the paper's metrics:
    - Uti.: Average space utilization (volume packed / bin volume)
    - Var.: Variance of utilization (x10^-3)
    - Num.: Average number of packed items per bin
    - Gap: Difference relative to best method's utilization

    Additional metrics for thesis:
    - Stability score: fraction of placements that pass stability check
    - Bins used: total number of bins opened
    - Items per bin: average items packed per bin
    - Decision time: average time per placement decision
    """

    @staticmethod
    def compute_all(results: List[dict]) -> dict:
        """Compute aggregate metrics from multiple packing episodes."""
        utilizations = [r['average_utilization'] for r in results]
        items_per_bin = [r['items_per_bin'] for r in results]
        times = [r['avg_time_per_item'] for r in results]

        return {
            'mean_utilization': np.mean(utilizations),
            'std_utilization': np.std(utilizations),
            'variance_utilization_1e3': np.var(utilizations) * 1e3,
            'mean_items_per_bin': np.mean(items_per_bin),
            'mean_decision_time': np.mean(times),
            'total_bins_used': sum(r['total_bins_used'] for r in results),
            'total_items_packed': sum(r['total_items_packed'] for r in results),
        }


# =============================================================================
# SECTION 8: CONFIGURATION TEMPLATES
# =============================================================================

THESIS_CONFIG_DISCRETE = {
    'bin_size': np.array([10, 10, 10]),
    'buffer_size': 5,
    'num_orientations': 2,
    'mcts_simulations': 100,
    'closing_strategy': 'hybrid',
    'check_stability': True,
    'stability_support_threshold': 0.80,
    'description': 'Discrete space, small buffer, thesis baseline'
}

THESIS_CONFIG_CONTINUOUS = {
    'bin_size': np.array([1.0, 1.0, 1.0]),
    'buffer_size': 5,
    'num_orientations': 2,
    'mcts_simulations': 200,
    'closing_strategy': 'hybrid',
    'check_stability': True,
    'stability_support_threshold': 0.80,
    'description': 'Continuous space, matching paper setting'
}

THESIS_CONFIG_LARGE_BUFFER = {
    'bin_size': np.array([1.2, 1.0, 1.4]),  # 120x100x140cm real pallet
    'buffer_size': 10,
    'num_orientations': 2,
    'mcts_simulations': 300,
    'closing_strategy': 'hybrid',
    'check_stability': True,
    'stability_support_threshold': 0.80,
    'description': 'Real-world pallet dimensions, large buffer'
}


# =============================================================================
# SECTION 9: OPTION A -- DUAL-PCT WITH BIN SELECTION HEAD
#             (from deep summary Section 10.2)
# =============================================================================

"""
OPTION A: Dual-PCT with Separate Bin Selection Head

Architecture:
  PCT_1: PackingConfigurationTree for bin 1
  PCT_2: PackingConfigurationTree for bin 2

  For each item n_t in buffer:
    h_1 = GAT(PCT_1, n_t)  -> leaf features for bin 1
    h_2 = GAT(PCT_2, n_t)  -> leaf features for bin 2

    # Bin selection via additional attention head
    h_global = concat(mean(h_1), mean(h_2))
    bin_score = MLP_bin(h_global)  -> softmax over [bin_1, bin_2]

    # Within selected bin, use pointer mechanism as normal
    selected_bin = argmax(bin_score)
    pi = pointer(h_{selected_bin})

PROS:
  - Modular; can reuse pre-trained PCT encoder (shared weights)
  - End-to-end differentiable for bin selection
  - Fast at inference (one forward pass per bin)

CONS:
  - Requires training the bin selector head
  - Sequential decisions (bin then placement)
  - Bin selector may not capture long-horizon planning effects

When to use: if end-to-end learning is preferred over search-based planning.
"""


def dual_pct_bin_selector_pseudocode():
    """
    # PyTorch implementation for Option A:

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical

    class DualPCT_BinSelector(nn.Module):
        '''
        Option A: Two PCTs with a learned bin selection head.

        Architecture:
        1. Shared PCTEncoder processes each bin's state independently
        2. BinSelector MLP compares global contexts from both bins
        3. Pointer mechanism selects placement within the chosen bin

        Training:
        - Pre-train PCTEncoder on single-bin online packing (Phase 1-4)
        - Then train BinSelector head while fine-tuning encoder (Phase 6)
        - Alternatively: freeze encoder, train only BinSelector

        Parameters:
        - PCTEncoder: ~30K (shared between bins)
        - BinSelector: ~8K (new)
        - Pointer: ~8K (shared)
        - Total new: ~8K on top of existing PCT model
        '''
        def __init__(self, d_h=64, d_raw=8, num_bins=2):
            super().__init__()
            self.d_h = d_h
            self.num_bins = num_bins

            # Shared PCT encoder (freeze or fine-tune from pre-trained)
            self.pct_encoder = PCTEncoder(d_raw, d_h)  # From pct_coding_ideas.py

            # Bin selection head
            self.bin_selector = nn.Sequential(
                nn.Linear(num_bins * d_h, d_h),
                nn.ReLU(),
                nn.Linear(d_h, d_h // 2),
                nn.ReLU(),
                nn.Linear(d_h // 2, num_bins),
            )

            # Pointer mechanism (shared with PCT)
            self.pointer = PointerMechanism(d_h)

            # Critic for bin selection (separate from PCT critic)
            self.bin_critic = nn.Sequential(
                nn.Linear(num_bins * d_h, d_h),
                nn.ReLU(),
                nn.Linear(d_h, 1),
            )

        def forward(self, pct_features_1, pct_features_2, current_item):
            '''
            Args:
                pct_features_1: dict with internal_feat, leaf_feat, masks for bin 1
                pct_features_2: dict with internal_feat, leaf_feat, masks for bin 2
                current_item: current item features

            Returns:
                bin_probs: (batch, 2) probability over bins
                leaf_probs: (batch, max_leaves) probability over leaf nodes
                bin_value: (batch,) value estimate for bin selection
            '''
            # Encode each bin's state
            h_1 = self.pct_encoder(**pct_features_1, current=current_item)
            h_2 = self.pct_encoder(**pct_features_2, current=current_item)

            # Global context for each bin
            ctx_1 = h_1.mean(dim=1)  # (batch, d_h)
            ctx_2 = h_2.mean(dim=1)  # (batch, d_h)

            # Bin selection
            joint_ctx = torch.cat([ctx_1, ctx_2], dim=-1)  # (batch, 2*d_h)
            bin_logits = self.bin_selector(joint_ctx)       # (batch, 2)
            bin_probs = F.softmax(bin_logits, dim=-1)

            # Value estimate for this joint state
            bin_value = self.bin_critic(joint_ctx).squeeze(-1)

            # During training: sample bin; during testing: argmax
            bin_idx = Categorical(bin_probs).sample()

            # Select leaf from chosen bin
            h_selected = torch.where(
                bin_idx.unsqueeze(-1).unsqueeze(-1).expand_as(h_1) == 0,
                h_1, h_2
            )
            leaf_probs = self.pointer(h_selected)

            return bin_probs, leaf_probs, bin_value, bin_idx
    '''
    """
    pass


# =============================================================================
# SECTION 10: OPTION B -- EXTENDED ToP SEARCH (RECOMMENDED)
#             (from deep summary Section 10.2)
# =============================================================================

"""
OPTION B: Extended ToP Search -- RECOMMENDED FOR THESIS

Architecture:
  Same pre-trained PCT policy pi_theta -- NO modification needed!

  ToP search tree extended:
    Original: each node = (item_index)
    Extended: each node = (item_index, bin_index)
    Branching factor = s * 2 (buffer_size * num_bins)

  MCTS searches over:
    - Which item from buffer (s choices)
    - Which bin to place in (2 choices)

  Placement POSITION within the chosen bin determined by pi_theta (not searched).
  This is the key design choice keeping search tractable.

KEY INSIGHT (from paper Section 3.4):
  The same pi_theta trained on simple online packing can be deployed
  with ANY buffer size s, lookahead p, or even offline (s=|I|).
  No retraining, fine-tuning, or architectural changes needed!

  We exploit this by adding bin selection to the search without
  touching pi_theta at all.

BRANCHING FACTOR ANALYSIS:
  Original ToP (single bin): branching = s
  Extended ToP (2 bins):     branching = s * 2

  For s=5:   10 children per node (vs 5)
  For s=10:  20 children per node (vs 10)

  MCTS handles this easily -- the paper uses m=200 simulations
  for s=10, which suffices for branching factor 20.

COMPUTATIONAL COST:
  Per MCTS simulation:  ~1 pi_theta forward pass per expanded node
  Total per decision:   200 * ~20 = ~4000 forward passes
  Time per decision:    ~2-4 seconds on GPU (within 9.8s robot cycle)

GLOBAL CACHE FOR PATH REUSE:
  At adjacent time steps, MCTS paths share common sub-sequences.
  The paper's global cache stores previously visited paths to avoid
  redundant computations, nearly halving decision time.
  For 2-bounded space: cache key includes both bins' states.
"""


class ExtendedToP_MCTS:
    """
    Extended ToP MCTS planner for 2-bounded space.

    This is the RECOMMENDED implementation for the thesis.
    It adds bin selection to the standard ToP MCTS without modifying pi_theta.

    Key differences from standard ToP:
    1. Each action = (item_index, bin_index) instead of just item_index
    2. Rollout simulates both bins simultaneously
    3. Value function accounts for both bins' states
    4. Global cache key includes both bins' utilizations
    """

    def __init__(self, policy_network, value_network=None,
                 num_simulations: int = 200,
                 c_puct: float = 2.0,
                 max_depth: int = 10):
        self.policy_network = policy_network
        self.value_network = value_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.max_depth = max_depth

        # Global cache for path reuse across time steps
        self.global_cache: Dict[str, List[ToP_Action]] = {}

    def plan(self, bin_states: List[Any], buffer: List[Any],
             closing_strategy=None) -> Optional[ToP_Action]:
        """
        Run extended MCTS and return best first action.

        Args:
            bin_states: [PCT_bin_0, PCT_bin_1] current bin states
            buffer: list of items in buffer
            closing_strategy: optional BinClosingStrategy

        Returns:
            ToP_Action with (item_index, bin_index) or None if no feasible action

        Algorithm:
        1. Check global cache for previously computed paths
        2. Initialize root with current state
        3. Run num_simulations MCTS iterations:
           a. SELECT: traverse tree using PUCT
           b. EXPAND: create children for all (item, bin) combos
           c. SIMULATE: rollout using pi_theta for known items + V(.) for unknown
           d. BACKPROPAGATE: update all ancestors
        4. Select root child with highest visit count
        5. Cache the best path for reuse at next time step
        """
        if not buffer:
            return None

        # Check cache
        cache_key = self._make_cache_key(bin_states, buffer)
        if cache_key in self.global_cache:
            cached_path = self.global_cache[cache_key]
            if cached_path:
                return cached_path[0]

        # Initialize root
        root = ToP_SearchNode(state=ToP_SearchState(
            bin_states=[s for s in bin_states],
            remaining_buffer=list(range(len(buffer))),
            cumulative_volume=0.0,
        ))

        # Run MCTS
        for sim in range(self.num_simulations):
            self._simulate_one(root, buffer, bin_states, closing_strategy)

        # Select best action
        if not root.children:
            return None

        best_child = root.most_visited_child()
        best_action = best_child.action

        # Cache best path
        self._cache_path(root, cache_key)

        return best_action

    def _simulate_one(self, root, buffer, original_bin_states,
                       closing_strategy):
        """
        Single MCTS simulation: SELECT -> EXPAND -> EVALUATE -> BACKPROPAGATE.

        Extended for 2-bounded space: expansion creates children for all
        (item_index, bin_index) combinations where bin_index in {0, 1}.
        """
        import copy

        node = root
        path = [node]
        # We track simulated state changes along the path
        sim_buffer_remaining = list(root.state.remaining_buffer)
        sim_volumes = [0.0, 0.0]  # Volume added per bin in this path

        # SELECTION: traverse existing tree using PUCT
        while not node.is_leaf and not node.is_terminal:
            node = node.best_child(self.c_puct)
            path.append(node)

            # Apply action to simulated state
            if node.action is not None:
                item_idx = node.action.item_index
                bin_idx = node.action.bin_index
                if item_idx in sim_buffer_remaining:
                    sim_volumes[bin_idx] += node.action.estimated_value
                    sim_buffer_remaining.remove(item_idx)

        # EXPANSION: create children for all feasible (item, bin) actions
        if not node.is_terminal and sim_buffer_remaining:
            for item_idx in sim_buffer_remaining:
                item = buffer[item_idx]
                for bin_idx in range(2):
                    # Check feasibility using pi_theta
                    feasible, placement_value = self._evaluate_feasibility(
                        item, bin_idx, original_bin_states[bin_idx])

                    if not feasible:
                        continue

                    action = ToP_Action(
                        item_index=item_idx,
                        bin_index=bin_idx,
                        estimated_value=placement_value,
                    )

                    new_remaining = [i for i in sim_buffer_remaining
                                     if i != item_idx]
                    child_state = ToP_SearchState(
                        bin_states=original_bin_states,  # Deep copy in practice
                        remaining_buffer=new_remaining,
                        cumulative_volume=(root.state.cumulative_volume +
                                          sum(sim_volumes) + placement_value),
                    )

                    child = ToP_SearchNode(
                        state=child_state,
                        parent=node,
                        action=action,
                    )
                    # Prior probability: use pi_theta's action probability
                    child.prior_probability = max(0.01, placement_value)
                    node.children[action] = child

            # Select one new child for evaluation
            if node.children:
                node = node.best_child(self.c_puct)
                path.append(node)

        # EVALUATION: estimate value of this state
        # For known items in buffer: sum of placed volumes along this path
        # For unknown future items: V(.) from critic network
        value = sum(sim_volumes)
        if self.value_network is not None:
            # Add V(.) for future unknown items
            # value += self.value_network(bin_states_to_features(sim_bins))
            pass
        elif node.state is not None:
            value = node.state.cumulative_volume

        # BACKPROPAGATION
        for n in reversed(path):
            n.visit_count += 1
            n.total_value += value

    def _evaluate_feasibility(self, item, bin_idx, bin_state) -> Tuple[bool, float]:
        """
        Check if item has feasible placement in bin and estimate its value.

        Uses pi_theta to:
        1. Check if any leaf node exists for this item in this bin
        2. Estimate the value of placing the item here

        Returns: (feasible: bool, value: float)
        """
        # In practice: call bin_state.get_feasible_leaves(item)
        # and use policy_network to score the best placement
        # Placeholder:
        try:
            leaves = bin_state.get_feasible_leaves(item)
            if not leaves:
                return False, 0.0
            # Value = item volume (simplified; use pi_theta score in practice)
            return True, item.volume
        except Exception:
            return False, 0.0

    def _make_cache_key(self, bin_states, buffer) -> str:
        """Create cache key from bin states and buffer contents."""
        # Use utilizations + buffer item IDs as key
        utils = tuple(round(b.utilization, 3) for b in bin_states)
        items = tuple(sorted(id(item) for item in buffer))
        return str((utils, items))

    def _cache_path(self, root, cache_key):
        """Extract and cache the best path from root."""
        path = []
        node = root
        while node.children:
            best_child = node.most_visited_child()
            if best_child.action is not None:
                path.append(best_child.action)
            node = best_child
        self.global_cache[cache_key] = path


# =============================================================================
# SECTION 11: OPTION C -- JOINT MULTI-BIN PCT (RESEARCH-LEVEL)
#             (from deep summary Section 10.2)
# =============================================================================

"""
OPTION C: Joint Multi-Bin PCT

Architecture:
  Single PCT encoding BOTH bins simultaneously.

  Internal nodes from both bins, tagged with bin_id:
    b = (s_x, s_y, s_z, p_x, p_y, p_z, rho, cat, bin_id)

  Leaf nodes from both bins, tagged with bin_id:
    l = (s_o_x, s_o_y, s_o_z, p_x, p_y, p_z, bin_id)

  GAT operates over the COMBINED graph of both bins.
  The graph has N = |B_0| + |B_1| + |L_0| + |L_1| + 1 nodes.

  Pointer selects (leaf, bin) jointly -- the leaf node already encodes
  which bin it belongs to.

PROS:
  - End-to-end learned cross-bin dependencies
  - Can learn to balance bins without explicit heuristics
  - Captures interactions like "this item fits better in bin 1
    because bin 0 needs the space for a future large item"

CONS:
  - Requires retraining from scratch (cannot use pre-trained pi_theta)
  - Graph is ~2x larger -> O(N^2) GAT is ~4x slower
  - More complex to implement
  - Research-level novelty but higher risk

IMPLEMENTATION NOTES:
  - Add bin_id as an extra feature dimension in node descriptors
  - Internal node feature: d_raw += 1 (for bin_id 0/1)
  - Leaf node feature: d_raw += 1 (for bin_id 0/1)
  - Max nodes: 2 * max_internal + 2 * max_leaves + 1
  - Attention mask: can optionally restrict attention within bins
    (intra-bin attention) or allow full cross-bin attention

  For thesis: NOT recommended as primary approach. Consider as extension
  if Option B works well and there is time for exploration.
"""


class JointMultiBinPCT:
    """
    Joint PCT encoding for multiple bins (Option C).

    Merges internal nodes, leaf nodes from all active bins into a single
    graph with bin_id tags.

    This is provided for completeness / future exploration.
    Use Option B (ExtendedToP_MCTS) as the primary approach.
    """

    def __init__(self, bin_size, num_bins: int = 2, num_orientations: int = 2):
        self.bin_size = bin_size
        self.num_bins = num_bins
        self.num_orientations = num_orientations

    def merge_features(self, pct_list: List[Any], current_item: Any,
                        max_internal: int = 80,
                        max_leaves: int = 50) -> dict:
        """
        Merge features from multiple PCTs into a single feature tensor.

        Args:
            pct_list: list of PackingConfigurationTree objects (one per bin)
            current_item: the current item to place
            max_internal: max internal nodes per bin
            max_leaves: max leaf nodes per bin

        Returns:
            dict with merged feature tensors including bin_id dimension
        """
        d_raw = 9  # 8 standard features + 1 bin_id
        total_max_internal = max_internal * self.num_bins
        total_max_leaves = max_leaves * self.num_bins

        internal_feats = np.zeros((total_max_internal, d_raw))
        internal_mask = np.zeros(total_max_internal, dtype=bool)
        leaf_feats = np.zeros((total_max_leaves, d_raw))
        leaf_mask = np.zeros(total_max_leaves, dtype=bool)

        for bin_idx, pct in enumerate(pct_list):
            int_offset = bin_idx * max_internal
            leaf_offset = bin_idx * max_leaves

            # Internal nodes
            for i, p in enumerate(pct.packed_items[:max_internal]):
                internal_feats[int_offset + i] = [
                    p.size[0] / self.bin_size[0],
                    p.size[1] / self.bin_size[1],
                    p.size[2] / self.bin_size[2],
                    p.position[0] / self.bin_size[0],
                    p.position[1] / self.bin_size[1],
                    p.position[2] / self.bin_size[2],
                    1.0, 0.0,
                    bin_idx / max(1, self.num_bins - 1),  # bin_id normalized
                ]
                internal_mask[int_offset + i] = True

            # Leaf nodes
            for i, l in enumerate(pct.leaf_nodes[:max_leaves]):
                leaf_feats[leaf_offset + i] = [
                    l.size[0] / self.bin_size[0],
                    l.size[1] / self.bin_size[1],
                    l.size[2] / self.bin_size[2],
                    l.position[0] / self.bin_size[0],
                    l.position[1] / self.bin_size[1],
                    l.position[2] / self.bin_size[2],
                    l.orientation_idx / max(1, self.num_orientations - 1),
                    0.0,
                    bin_idx / max(1, self.num_bins - 1),  # bin_id
                ]
                leaf_mask[leaf_offset + i] = True

        # Current item (same for all bins)
        current_feats = np.array([[
            current_item.width / self.bin_size[0],
            current_item.depth / self.bin_size[1],
            current_item.height / self.bin_size[2],
            0.0, 0.0, 0.0,
            current_item.density if hasattr(current_item, 'density') else 1.0,
            0.0,
            -1.0,  # bin_id = -1 (not assigned yet)
        ]])

        return {
            'internal_features': internal_feats,
            'leaf_features': leaf_feats,
            'current_features': current_feats,
            'internal_mask': internal_mask,
            'leaf_mask': leaf_mask,
            'bin_ids': leaf_feats[:, -1],  # For decoding which bin each leaf belongs to
        }


# =============================================================================
# SECTION 12: CROSS-BIN SPATIAL ENSEMBLE (from paper Section 3.3, adapted)
# =============================================================================

"""
CROSS-BIN SPATIAL ENSEMBLE FOR 2-BOUNDED SPACE

The paper's spatial ensemble (Section 3.3) was designed for recursive packing
sub-bins. We adapt it for cross-bin evaluation in 2-bounded space.

PROBLEM: When comparing placements across bins with different fill levels,
direct score comparison is unfair:
  - A nearly empty bin will have high pi_theta scores for most items
  - A nearly full bin will have low scores even for good placements
  - Direct comparison systematically favors the emptier bin

SOLUTION: Convert absolute scores to RANKS within each bin, then compare.

ALGORITHM:
1. For each candidate leaf in bin 0: compute rank among bin 0 leaves
2. For each candidate leaf in bin 1: compute rank among bin 1 leaves
3. For items that could go in either bin, take the rank from each bin
4. Select the (item, bin) with the best rank overall

The original spatial ensemble uses max-min across sub-bins (Equation 9):
  l* = argmax_{l in L} min_{c_i in c} Phi_hat(l, c_i)

For 2-bounded space, this simplifies to:
  For each item in buffer:
    rank_in_bin0 = rank of best placement in bin 0
    rank_in_bin1 = rank of best placement in bin 1
    combined_score = min(rank_in_bin0, rank_in_bin1) if item goes to both
                   = rank_in_binX if only one bin is feasible
  Select the item with highest combined_score
"""


class CrossBinSpatialEnsemble:
    """
    Spatial ensemble for fair cross-bin evaluation in 2-bounded space.

    Converts absolute pi_theta scores to normalized ranks within each bin,
    enabling fair comparison of placements across bins with different
    fill levels.
    """

    def select_best_action(self,
                            buffer: List[Any],
                            bin_states: List[Any],
                            policy_network=None,
                            ) -> Optional[Tuple[int, int]]:
        """
        Select the best (item_index, bin_index) using spatial ensemble.

        Args:
            buffer: list of items in buffer
            bin_states: [PCT_bin_0, PCT_bin_1]
            policy_network: trained PCT policy for scoring

        Returns:
            (item_index, bin_index) or None if no feasible action
        """
        # Collect scores per bin
        # scores[bin_idx] = list of (item_idx, best_placement_score)
        scores_per_bin = [[], []]

        for item_idx, item in enumerate(buffer):
            for bin_idx in range(2):
                leaves = bin_states[bin_idx].get_feasible_leaves(item)
                if not leaves:
                    continue
                # Score using policy network (or volume as fallback)
                if policy_network is not None:
                    # pi_scores = policy_network.score_leaves(bin_states[bin_idx], item)
                    # best_score = max(pi_scores)
                    best_score = item.volume  # Placeholder
                else:
                    best_score = item.volume

                scores_per_bin[bin_idx].append((item_idx, best_score))

        # Convert to ranks within each bin
        ranks_per_bin = [
            self._to_normalized_ranks(scores)
            for scores in scores_per_bin
        ]

        # Find best (item, bin) by highest rank
        best_action = None
        best_rank = -1.0

        for bin_idx in range(2):
            for item_idx, rank in ranks_per_bin[bin_idx].items():
                if rank > best_rank:
                    best_rank = rank
                    best_action = (item_idx, bin_idx)

        return best_action

    def _to_normalized_ranks(self, scores: List[Tuple[int, float]]
                              ) -> Dict[int, float]:
        """
        Convert (item_idx, score) list to {item_idx: normalized_rank} dict.
        Rank is in [1/n, 1.0] where higher is better.
        """
        if not scores:
            return {}

        sorted_scores = sorted(scores, key=lambda x: x[1])
        n = len(sorted_scores)
        return {
            item_idx: (rank + 1) / n
            for rank, (item_idx, _) in enumerate(sorted_scores)
        }


# =============================================================================
# SECTION 13: ADVANCED BIN CLOSING STRATEGIES
#             (from deep summary Sections 10.4 and 10.6)
# =============================================================================

"""
BIN CLOSING is the CRITICAL decision in k-bounded space packing.
Once a bin is closed, it NEVER reopens. Poor closing decisions waste space.

The original PCT paper does NOT address bin closing (single-bin focus).
This is our thesis contribution area.

FOUR STRATEGIES (with increasing sophistication):

Strategy 1: NO-FIT
  Close when no buffer item fits. Simple but may close prematurely
  if items in the future stream could still fit.

Strategy 2: VALUE-THRESHOLD
  Close when V(.) from critic drops below threshold.
  More sophisticated but requires calibrating the threshold.
  V(.) predicts accumulated future reward from current state.

Strategy 3: UTILIZATION-THRESHOLD
  Close when utilization exceeds threshold (e.g., 90%).
  Pragmatic for high fill rates. Risk: may close bins that
  could still accommodate small items.

Strategy 4: HYBRID (RECOMMENDED)
  Combine no-fit with value and utilization thresholds.
  Close if: (no fit) OR (utilization > 85% AND V(.) < 0.05 * 2)

LEARNING-BASED CLOSING (future extension):
  Train a separate small network to predict whether closing is optimal.
  Input: bin state features + buffer summary features
  Output: close probability
  This could be trained via RL with a reward that penalizes premature closing.

CLOSING EVALUATION METRIC:
  "Wasted space" = bin_volume * (1 - utilization) for each closed bin.
  Good closing minimizes total wasted space across all bins.
"""


class AdvancedBinClosingStrategy:
    """
    Extended bin closing strategies for 2-bounded space.

    Adds fragmentation analysis and comparative closing to the basic strategies
    in BinClosingStrategy (Section 4 above).
    """

    def __init__(self, strategy: str = 'hybrid',
                 value_threshold: float = 0.05,
                 utilization_threshold: float = 0.85,
                 fragmentation_threshold: float = 0.3):
        self.strategy = strategy
        self.value_threshold = value_threshold
        self.utilization_threshold = utilization_threshold
        self.fragmentation_threshold = fragmentation_threshold

    def should_close(self, bin_pct, buffer: List[Any],
                     other_bin_pct=None,
                     value_fn=None) -> bool:
        """
        Determine if a bin should be closed.

        Extended with:
        - Fragmentation analysis (how scattered is remaining space?)
        - Comparative closing (close the worse bin when both are nearly full)
        - Buffer-aware closing (consider buffer item sizes)
        """
        if self.strategy == 'no_fit':
            return self._no_fit(bin_pct, buffer)
        elif self.strategy == 'value_threshold':
            return self._value_threshold(bin_pct, value_fn)
        elif self.strategy == 'utilization_threshold':
            return self._utilization_threshold(bin_pct)
        elif self.strategy == 'hybrid':
            return self._hybrid(bin_pct, buffer, value_fn)
        elif self.strategy == 'fragmentation_aware':
            return self._fragmentation_aware(bin_pct, buffer)
        elif self.strategy == 'comparative':
            return self._comparative(bin_pct, buffer, other_bin_pct, value_fn)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _no_fit(self, bin_pct, buffer) -> bool:
        """Close if no buffer item can fit."""
        for item in buffer:
            leaves = bin_pct.get_feasible_leaves(item)
            if len(leaves) > 0:
                return False
        return True

    def _value_threshold(self, bin_pct, value_fn) -> bool:
        """Close if V(.) is below threshold."""
        if value_fn is None:
            return False
        v = value_fn(bin_pct)
        return v < self.value_threshold

    def _utilization_threshold(self, bin_pct) -> bool:
        """Close if utilization exceeds threshold."""
        return bin_pct.utilization >= self.utilization_threshold

    def _hybrid(self, bin_pct, buffer, value_fn) -> bool:
        """
        Hybrid: close if (no fit) OR (high util AND low value).
        RECOMMENDED for thesis.
        """
        if self._no_fit(bin_pct, buffer):
            return True

        if bin_pct.utilization > self.utilization_threshold:
            if value_fn is not None:
                v = value_fn(bin_pct)
                if v < self.value_threshold * 2:
                    return True

        return False

    def _fragmentation_aware(self, bin_pct, buffer) -> bool:
        """
        Close if remaining space is too fragmented to be useful.

        Fragmentation = 1 - (max_ems_volume / total_remaining_volume)
        High fragmentation means space is scattered in small pockets.
        """
        if self._no_fit(bin_pct, buffer):
            return True

        total_remaining = float(np.prod(bin_pct.bin_size)) * (1 - bin_pct.utilization)
        if total_remaining < 1e-9:
            return True

        max_ems_vol = max(
            (e.volume for e in bin_pct.ems_manager.ems_list),
            default=0
        )
        fragmentation = 1.0 - (max_ems_vol / max(total_remaining, 1e-9))

        if fragmentation > self.fragmentation_threshold:
            # Space is fragmented -- check if any buffer item fits the largest EMS
            min_item_vol = min((item.volume for item in buffer), default=float('inf'))
            if max_ems_vol < min_item_vol * 0.5:
                return True

        return False

    def _comparative(self, bin_pct, buffer, other_bin_pct, value_fn) -> bool:
        """
        Comparative closing: when BOTH bins are nearly full, close the worse one.

        Logic:
        - If both bins > 80% utilization, close the one with lower V(.)
        - This frees up a slot for a fresh empty bin
        - Only one bin should be closed at a time
        """
        if other_bin_pct is None:
            return self._hybrid(bin_pct, buffer, value_fn)

        both_high = (bin_pct.utilization > 0.80 and
                     other_bin_pct.utilization > 0.80)

        if both_high and value_fn is not None:
            my_value = value_fn(bin_pct)
            other_value = value_fn(other_bin_pct)
            # Close the worse bin (this one) only if it has lower value
            if my_value < other_value and my_value < self.value_threshold * 3:
                return True

        return self._hybrid(bin_pct, buffer, value_fn)


# =============================================================================
# SECTION 14: PHYSICS-BASED STABILITY VERIFICATION PIPELINE
#             (from paper Section 4.7 and deep summary Section 10.5)
# =============================================================================

"""
STABILITY VERIFICATION PIPELINE FOR 2-BOUNDED SPACE:

Training time:
  - Use quasi-static equilibrium (from pct_coding_ideas.py Section 13)
  - Pre-filter leaf nodes: unstable placements never shown to DRL agent
  - Speed: >400 FPS
  - Integrated via check_stability=True in PCT.get_feasible_leaves()

Test time:
  - Use physics simulation (PyBullet) for placement verification
  - For each placement decision:
    1. pi_theta produces ranked list of candidate leaves
    2. Take top k_l = 5 candidates
    3. For each candidate, run k_d = 4-8 disturbance sets
    4. Accept the highest-ranked candidate that survives all disturbances
    5. If none survive, accept highest-ranked (with warning)

  DISTURBANCE MODEL (from paper):
    Each set = 10 random perturbations:
      Translation: [15, 20] cm along x, y axes
      Rotation: [-10, 10] degrees around z axis
      Linear velocity: 6 m/s
      Angular velocity: 30 degrees/s

  RESULTS (paper Table 11):
    k_d = 1 -> 70% transport stability
    k_d = 2 -> 85%
    k_d = 4 -> 95%
    k_d = 8 -> 100%

FOR 2-BOUNDED SPACE:
  Each bin needs independent stability verification.
  The verification is per-bin, not cross-bin (items in different bins
  don't interact physically).

  Cost: k_l * k_d * 10 simulations per placement decision per bin
  With k_l=5, k_d=4: 200 simulations per decision
  At ~0.01s per sim on GPU: ~2 seconds additional per decision
  Total with MCTS: ~4-6 seconds (still within 9.8s cycle)
"""


class StabilityVerificationPipeline:
    """
    Complete stability pipeline for 2-bounded space packing.

    Combines training-time quasi-static checks with test-time physics
    verification for robust stability guarantees.
    """

    def __init__(self, mode: str = 'training',
                 support_threshold: float = 0.80,
                 k_l: int = 5, k_d: int = 4,
                 use_pybullet: bool = False):
        """
        Args:
            mode: 'training' (fast quasi-static) or 'testing' (physics sim)
            support_threshold: min support area ratio for quasi-static
            k_l: number of top candidates for physics verification
            k_d: number of disturbance sets per candidate
            use_pybullet: whether PyBullet is available
        """
        self.mode = mode
        self.support_threshold = support_threshold
        self.k_l = k_l
        self.k_d = k_d
        self.use_pybullet = use_pybullet

    def filter_candidates(self, candidates: List[Any],
                           packed_items: List[Any],
                           bin_size=None) -> List[Any]:
        """
        Filter candidates for stability.

        During training: fast quasi-static check (support area ratio + COG)
        During testing: pass through (physics verification done later)
        """
        if self.mode == 'training':
            return [c for c in candidates
                    if self._quasi_static_check(c, packed_items)]
        else:
            return candidates  # Physics check done at selection time

    def select_stable_placement(self, candidates: List[Any],
                                  scores: List[float],
                                  packed_items: List[Any],
                                  bin_size=None) -> Optional[Any]:
        """
        Select the best stable placement at test time.

        Takes the top k_l candidates by score, then runs physics
        verification on each until one passes all disturbance sets.
        """
        if self.mode == 'training':
            # During training: just pick the top-scoring stable candidate
            stable = [(c, s) for c, s in zip(candidates, scores)
                      if self._quasi_static_check(c, packed_items)]
            if stable:
                return max(stable, key=lambda x: x[1])[0]
            return candidates[0] if candidates else None

        # Test-time: physics verification
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        top_k = ranked[:self.k_l]

        if self.use_pybullet:
            for candidate, score in top_k:
                if self._physics_verify(candidate, packed_items, bin_size):
                    return candidate

        # Fallback: return top candidate without physics check
        return top_k[0][0] if top_k else None

    def _quasi_static_check(self, placement, packed_items) -> bool:
        """Fast quasi-static equilibrium check (training time)."""
        if placement.position[2] < 1e-9:
            return True  # Floor placement

        support_area = 0.0
        placement_area = placement.size[0] * placement.size[1]

        for packed in packed_items:
            packed_top = packed.position[2] + packed.size[2]
            if abs(packed_top - placement.position[2]) > 1e-6:
                continue
            ox = max(0, min(placement.position[0] + placement.size[0],
                           packed.position[0] + packed.size[0]) -
                     max(placement.position[0], packed.position[0]))
            oy = max(0, min(placement.position[1] + placement.size[1],
                           packed.position[1] + packed.size[1]) -
                     max(placement.position[1], packed.position[1]))
            support_area += ox * oy

        ratio = support_area / max(placement_area, 1e-9)
        return ratio >= self.support_threshold

    def _physics_verify(self, placement, packed_items, bin_size) -> bool:
        """
        Physics verification using PyBullet (test time).

        Placeholder -- actual implementation requires:
          import pybullet as p
          p.connect(p.DIRECT)
          # Build scene from packed_items + placement
          # Apply k_d disturbance sets
          # Check for collapses
          p.disconnect()
        """
        # TODO: Implement PyBullet verification
        return True  # Placeholder: accept all


# =============================================================================
# SECTION 15: COMPLETE EVALUATION METRICS FRAMEWORK
#             (from paper Section 4 results tables)
# =============================================================================

"""
EVALUATION METRICS (matching paper's metrics exactly):

Primary metrics (from paper Tables 2-15):
  Uti.  = Average space utilization (volume packed / bin volume)
  Var.  = Variance of utilization (x10^-3)
  Num.  = Average number of packed items per bin
  Gap   = Difference relative to best method's utilization

Additional metrics for thesis (2-bounded space specific):
  Bins used     = Total bins opened across all episodes
  Items/bin     = Average items packed per bin
  Decision time = Average time per placement decision
  Stability     = Fraction of placements passing stability check
  Closing rate  = Average utilization at bin closing time
  Wasted space  = Total unused volume in closed bins

PAPER RESULT REFERENCE (for comparison baseline):
  Online, discrete, Setting 1 (stability):  75.8% (PCT & EMS)
  Online, discrete, Setting 2 (standard):   86.0% (PCT & EMS)
  Buffering s=5, Setting 2:                 88.3% (ToP)
  Buffering s=10, Setting 2:                93.5% (ToP)
  With stability (c=0.1):                   ~3-5% reduction
  2-bounded overhead estimate:              ~2-5% reduction

STATISTICAL SIGNIFICANCE:
  Use paired t-test or Wilcoxon signed-rank test on per-episode utilizations.
  Paper uses 2000 test instances for robust statistics.
  For thesis: minimum 200 test instances per configuration.
"""


class ComprehensiveMetrics:
    """
    Complete evaluation metrics for 2-bounded semi-online packing.

    Computes all metrics from the paper plus thesis-specific metrics.
    """

    @staticmethod
    def compute_episode_metrics(packing_result: dict) -> dict:
        """Compute metrics for a single packing episode."""
        return {
            'utilization': packing_result.get('average_utilization', 0.0),
            'items_packed': packing_result.get('total_items_packed', 0),
            'bins_used': packing_result.get('total_bins_used', 0),
            'items_per_bin': (packing_result.get('total_items_packed', 0) /
                              max(1, packing_result.get('total_bins_used', 1))),
            'elapsed_time': packing_result.get('elapsed_time', 0.0),
            'avg_decision_time': packing_result.get('avg_time_per_item', 0.0),
        }

    @staticmethod
    def compute_aggregate_metrics(episode_results: List[dict]) -> dict:
        """
        Compute aggregate metrics across multiple episodes.

        Matches paper's reporting format:
        - Uti. (mean utilization)
        - Var. (variance x 10^-3)
        - Num. (mean items per bin)
        """
        if not episode_results:
            return {}

        utils = [r['utilization'] for r in episode_results]
        items = [r['items_per_bin'] for r in episode_results]
        bins = [r['bins_used'] for r in episode_results]
        times = [r['avg_decision_time'] for r in episode_results]

        return {
            # Paper-standard metrics
            'mean_utilization': float(np.mean(utils)),
            'std_utilization': float(np.std(utils)),
            'variance_1e3': float(np.var(utils) * 1e3),
            'mean_items_per_bin': float(np.mean(items)),

            # Thesis-specific metrics
            'total_bins_used': int(np.sum(bins)),
            'mean_bins_per_episode': float(np.mean(bins)),
            'mean_decision_time_s': float(np.mean(times)),
            'p95_decision_time_s': float(np.percentile(times, 95)),

            # Utilization distribution
            'min_utilization': float(np.min(utils)),
            'max_utilization': float(np.max(utils)),
            'p25_utilization': float(np.percentile(utils, 25)),
            'p75_utilization': float(np.percentile(utils, 75)),

            # Episode count
            'num_episodes': len(episode_results),
        }

    @staticmethod
    def compute_gap(our_util: float, baseline_util: float) -> float:
        """
        Compute gap to a baseline method.
        Positive gap = we are better.
        """
        return our_util - baseline_util

    @staticmethod
    def format_results_table(results: dict, method_name: str = "Ours") -> str:
        """Format results as a text table matching paper format."""
        lines = [
            f"| Method       | Uti.   | Var.(1e-3) | Num.  | Time(s) |",
            f"|--------------|--------|------------|-------|---------|",
            f"| {method_name:12s} | "
            f"{results['mean_utilization']:.1%} | "
            f"{results['variance_1e3']:.2f}       | "
            f"{results['mean_items_per_bin']:.1f}  | "
            f"{results['mean_decision_time_s']:.3f}   |",
        ]
        return "\n".join(lines)

    @staticmethod
    def closing_quality_analysis(closed_bin_utilizations: List[float]) -> dict:
        """
        Analyze the quality of bin closing decisions.

        Good closing: bins closed at high utilization.
        Bad closing: bins closed with lots of empty space.
        """
        if not closed_bin_utilizations:
            return {'no_closed_bins': True}

        return {
            'mean_closing_util': float(np.mean(closed_bin_utilizations)),
            'std_closing_util': float(np.std(closed_bin_utilizations)),
            'min_closing_util': float(np.min(closed_bin_utilizations)),
            'max_closing_util': float(np.max(closed_bin_utilizations)),
            'wasted_volume_ratio': float(1.0 - np.mean(closed_bin_utilizations)),
            'bins_closed_above_80': int(sum(1 for u in closed_bin_utilizations
                                             if u >= 0.80)),
            'bins_closed_below_50': int(sum(1 for u in closed_bin_utilizations
                                             if u < 0.50)),
            'total_bins_closed': len(closed_bin_utilizations),
        }


# =============================================================================
# SECTION 16: ToP SEARCH TREE VARIANTS FROM PAPER (Section 3.4)
# =============================================================================

"""
The paper classifies BPP variants by item visibility:

| Setting                  | s    | p   | u   | Type        |
|--------------------------|------|-----|-----|-------------|
| s=1, p=0, u>0           | 1    | 0   | >0  | Online      |
| s=1, p>0, u>0           | 1    | >0  | >0  | Lookahead   |
| s>1, p=0, u>0           | >1   | 0   | >0  | Buffering   |
| s=|I|, p=0, u=0         | all  | 0   | 0   | Offline     |

Where:
  s = selectable items (within robot reach R_r)
  p = previewed items (visible via camera Fov_c but not yet reachable)
  u = unknown items (not yet visible)

  Formally: s = |R_r|, p = |Fov_c| - |R_r|, u = |complement(Fov_c)|

FOR OUR THESIS:
  s = buffer_size (5-10)
  p = 0 (no camera preview of conveyor)
  u > 0 (items keep arriving)
  Setting = BUFFERING PACKING

  If conveyor camera is available:
  s = buffer_size (5-10)
  p = camera_visible - buffer_size (e.g., 5)
  u > 0
  Setting = GENERAL PACKING

ToP HEATMAP RESULTS (from paper Figure 10, Setting 2):
| s   | p=0    | p=5    | p=9    |
|-----|--------|--------|--------|
| 1   | 86.0%  | 90.5%  | 91.8%  |
| 5   | 88.3%  | 91.2%  | 91.9%  |
| 10  | 93.5%  | 94.4%  | 95.0%  |

IMPLICATIONS:
- Going from s=1 to s=5: +2.3% (significant for free)
- Going from s=5 to s=10: +5.2% (worth the larger buffer)
- Adding p=5 preview at s=5: +2.9% (valuable if camera available)
- Diminishing returns beyond s=7 and p=5
"""


# =============================================================================
# SECTION 17: BEAM SEARCH ALTERNATIVE TO MCTS
# =============================================================================

class BeamSearchPlanner:
    """
    Beam search alternative to MCTS for simpler implementation.

    MCTS provides best results but beam search is:
    - Simpler to implement and debug
    - Deterministic (reproducible results)
    - Faster for small beam widths
    - Good baseline before implementing full MCTS

    Algorithm:
    1. Start with all possible first actions (item x bin)
    2. For each action, simulate the result using pi_theta
    3. Keep top beam_width results by cumulative value
    4. Expand each beam with all possible next actions
    5. Repeat for search_depth steps
    6. Return the first action of the highest-value beam

    Complexity: O(beam_width * branching * depth * pi_theta_cost)
    For beam=5, branching=20, depth=3: 300 pi_theta evaluations
    """

    def __init__(self, policy_network=None,
                 beam_width: int = 5,
                 search_depth: int = 3):
        self.policy_network = policy_network
        self.beam_width = beam_width
        self.search_depth = search_depth

    def plan(self, bin_states: List[Any], buffer: List[Any]) -> Optional[Tuple[int, int]]:
        """
        Run beam search and return best first action.

        Returns: (item_index, bin_index) or None
        """
        if not buffer:
            return None

        # Initialize beams: each beam = (first_action, cumulative_value, state)
        beams = []
        for item_idx, item in enumerate(buffer):
            for bin_idx in range(2):
                leaves = bin_states[bin_idx].get_feasible_leaves(item)
                if leaves:
                    value = item.volume  # Use pi_theta score in practice
                    beams.append({
                        'first_action': (item_idx, bin_idx),
                        'cumulative_value': value,
                        'remaining': [i for i in range(len(buffer)) if i != item_idx],
                        'depth': 1,
                    })

        if not beams:
            return None

        # Iteratively expand beams
        for depth in range(1, self.search_depth):
            # Sort by cumulative value, keep top beam_width
            beams.sort(key=lambda b: -b['cumulative_value'])
            beams = beams[:self.beam_width]

            new_beams = []
            for beam in beams:
                if not beam['remaining']:
                    new_beams.append(beam)
                    continue

                # Expand this beam with all possible next actions
                for item_idx in beam['remaining']:
                    item = buffer[item_idx]
                    for bin_idx in range(2):
                        leaves = bin_states[bin_idx].get_feasible_leaves(item)
                        if leaves:
                            value = item.volume
                            new_beams.append({
                                'first_action': beam['first_action'],
                                'cumulative_value': beam['cumulative_value'] + value,
                                'remaining': [i for i in beam['remaining']
                                              if i != item_idx],
                                'depth': depth + 1,
                            })

            beams = new_beams

        if not beams:
            return None

        # Return first action of best beam
        beams.sort(key=lambda b: -b['cumulative_value'])
        return beams[0]['first_action']


# =============================================================================
# SECTION 18: COMPLETE DATA FLOW AND INTEGRATION DIAGRAM
# =============================================================================

"""
COMPLETE DATA FLOW FOR 2-BOUNDED SEMI-ONLINE PACKING:

[Item Stream / Conveyor]
         |
         v
[Buffer Manager (s=5-10)]
         |
         v
[Bin Closing Check] -----> Close bin? --> [Replace with empty bin]
         |                                         |
         v                                         v
[Generate candidates for all buffer items in both bins]
         |
         v
[MCTS Planner (Option B)]  or  [Beam Search (baseline)]
  |                                    |
  v                                    v
[For each (item, bin) action:]
  |
  +-> [PCT Bin 0: GAT + Pointer]
  |         |
  |         v
  |   [Candidates scored by pi_theta]
  |
  +-> [PCT Bin 1: GAT + Pointer]
  |         |
  |         v
  |   [Candidates scored by pi_theta]
  |
  v
[Cross-Bin Spatial Ensemble Ranking]
         |
         v
[Select best (item, bin, leaf) action]
         |
         v
[Stability Verification]
  |  Training: quasi-static filter (fast)
  |  Testing: PyBullet physics sim (accurate)
         |
         v
[Execute Placement]
         |
         v
[Update PCT + EMS + Buffer]
         |
         v
[Refill buffer from stream]
         |
         v
[Record metrics: utilization, stability, timing]
         |
         v
[Repeat until stream exhausted]


MODULE DEPENDENCY MAP:

pct_coding_ideas.py:
  - Box, Placement, EMS, EMSManager
  - PackingConfigurationTree
  - PCTNetwork (GAT + Pointer)
  - PCTReward
  - QuasiStaticStabilityChecker
  - ConstraintRewards
  - BinPackingEnv

top_buffer_2bounded_coding_ideas.py (this file):
  - ToP_Action, ToP_SearchState, ToP_SearchNode
  - ToP_MCTS_Planner
  - ExtendedToP_MCTS (RECOMMENDED)
  - SpatialEnsembleRanker, CrossBinSpatialEnsemble
  - BinClosingStrategy, AdvancedBinClosingStrategy
  - SemiOnline2BoundedPacker
  - StabilityVerificationPipeline
  - ComprehensiveMetrics
  - BeamSearchPlanner
"""


# =============================================================================
# SECTION 19: EXTENDED CONFIGURATION TEMPLATES
# =============================================================================

# Experiment 1: Ablation on buffer size
ABLATION_BUFFER_CONFIGS = {
    f'buffer_{s}': {
        'bin_size': np.array([10, 10, 10]),
        'buffer_size': s,
        'num_orientations': 2,
        'mcts_simulations': min(s * 20, 300),
        'closing_strategy': 'hybrid',
        'check_stability': True,
        'stability_support_threshold': 0.80,
        'description': f'Buffer size ablation: s={s}'
    }
    for s in [1, 2, 3, 5, 7, 10]
}

# Experiment 2: Ablation on closing strategy
ABLATION_CLOSING_CONFIGS = {
    f'closing_{strat}': {
        'bin_size': np.array([10, 10, 10]),
        'buffer_size': 5,
        'num_orientations': 2,
        'mcts_simulations': 200,
        'closing_strategy': strat,
        'check_stability': True,
        'stability_support_threshold': 0.80,
        'description': f'Closing strategy ablation: {strat}'
    }
    for strat in ['no_fit', 'value_threshold', 'utilization_threshold',
                  'hybrid', 'fragmentation_aware', 'comparative']
}

# Experiment 3: Ablation on stability constraint weight c
ABLATION_STABILITY_CONFIGS = {
    f'stability_c{c}': {
        'bin_size': np.array([10, 10, 10]),
        'buffer_size': 5,
        'num_orientations': 2,
        'mcts_simulations': 200,
        'closing_strategy': 'hybrid',
        'check_stability': True,
        'stability_support_threshold': 0.80,
        'constraint_weight': c,
        'description': f'Stability weight ablation: c={c}'
    }
    for c in [0.0, 0.01, 0.1, 1.0, 10.0]
}

# Experiment 4: Compare 1-bounded vs 2-bounded
COMPARISON_BOUNDED_CONFIGS = {
    '1_bounded_s5': {
        'bin_size': np.array([10, 10, 10]),
        'buffer_size': 5,
        'num_bins': 1,
        'mcts_simulations': 200,
        'closing_strategy': 'no_fit',
        'check_stability': True,
        'description': '1-bounded baseline (s=5)'
    },
    '2_bounded_s5': {
        'bin_size': np.array([10, 10, 10]),
        'buffer_size': 5,
        'num_bins': 2,
        'mcts_simulations': 200,
        'closing_strategy': 'hybrid',
        'check_stability': True,
        'description': '2-bounded with hybrid closing (s=5)'
    },
}

# Experiment 5: Real-world pallet dimensions
REAL_WORLD_CONFIGS = {
    'pallet_small_buffer': {
        'bin_size': np.array([1.2, 1.0, 1.4]),  # 120x100x140cm
        'buffer_size': 5,
        'num_orientations': 2,
        'mcts_simulations': 200,
        'closing_strategy': 'hybrid',
        'check_stability': True,
        'stability_support_threshold': 0.80,
        'description': 'Real pallet, small buffer'
    },
    'pallet_large_buffer': {
        'bin_size': np.array([1.2, 1.0, 1.4]),
        'buffer_size': 10,
        'num_orientations': 2,
        'mcts_simulations': 300,
        'closing_strategy': 'hybrid',
        'check_stability': True,
        'stability_support_threshold': 0.80,
        'description': 'Real pallet, large buffer'
    },
}


if __name__ == "__main__":
    # Quick test of the planning infrastructure
    config = THESIS_CONFIG_DISCRETE
    packer = SemiOnline2BoundedPacker(config)
    print(f"Initialized packer with config: {config['description']}")
    print(f"  Bin size: {config['bin_size']}")
    print(f"  Buffer size: {config['buffer_size']}")
    print(f"  MCTS simulations: {config['mcts_simulations']}")
    print(f"  Closing strategy: {config['closing_strategy']}")
    print(f"  Stability check: {config['check_stability']}")

    # Test spatial ensemble
    ensemble = CrossBinSpatialEnsemble()
    print(f"\nCrossBinSpatialEnsemble initialized.")

    # Test advanced closing strategy
    closing = AdvancedBinClosingStrategy(strategy='hybrid')
    print(f"AdvancedBinClosingStrategy: {closing.strategy}")

    # Test beam search planner
    beam = BeamSearchPlanner(beam_width=5, search_depth=3)
    print(f"BeamSearchPlanner: width={beam.beam_width}, depth={beam.search_depth}")

    # Test metrics
    mock_results = [
        {'average_utilization': 0.82, 'total_items_packed': 18,
         'total_bins_used': 3, 'elapsed_time': 5.0, 'avg_time_per_item': 0.28},
        {'average_utilization': 0.78, 'total_items_packed': 15,
         'total_bins_used': 3, 'elapsed_time': 4.2, 'avg_time_per_item': 0.28},
    ]
    episode_metrics = [ComprehensiveMetrics.compute_episode_metrics(r) for r in mock_results]
    agg = ComprehensiveMetrics.compute_aggregate_metrics(episode_metrics)
    print(f"\nAggregate metrics:")
    print(f"  Mean utilization: {agg['mean_utilization']:.1%}")
    print(f"  Variance (1e-3):  {agg['variance_1e3']:.2f}")
    print(f"  Mean items/bin:   {agg['mean_items_per_bin']:.1f}")

    # Test closing quality analysis
    closing_utils = [0.85, 0.72, 0.91, 0.68, 0.88]
    closing_analysis = ComprehensiveMetrics.closing_quality_analysis(closing_utils)
    print(f"\nClosing quality analysis:")
    print(f"  Mean closing util: {closing_analysis['mean_closing_util']:.1%}")
    print(f"  Wasted volume ratio: {closing_analysis['wasted_volume_ratio']:.1%}")

    # Print experiment configs
    print(f"\nBuffer ablation configs: {len(ABLATION_BUFFER_CONFIGS)}")
    print(f"Closing strategy configs: {len(ABLATION_CLOSING_CONFIGS)}")
    print(f"Stability weight configs: {len(ABLATION_STABILITY_CONFIGS)}")

    print("\nReady for integration with trained PCT policy.")
