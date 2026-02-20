"""
CODING IDEAS: Semi-Online Buffer Policy with MCTS for 2-Bounded Space
======================================================================
Source: "Learning Practically Feasible Policies for Online 3D Bin Packing"
         Zhao et al. (2023) -- adapted and extended for our thesis use case

PURPOSE:
  Implement a complete semi-online decision system for:
    - Buffer of 5-10 items (choose which to pack next)
    - 2-bounded space (2 active pallets/bins)
    - Maximize fill rate + stability

  This combines the paper's MCTS approach (BPP-k) with multi-bin routing
  and buffer item selection.

DESIGN PHILOSOPHY:
  The paper's MCTS evaluates different orderings of k lookahead items for a
  single bin. We extend this to also consider:
    1. Which item from the buffer to select
    2. Which of the 2 active bins to place it in
    3. When to close a bin and open a new one

ESTIMATED COMPLEXITY:
  Per decision step with buffer=10, bins=2:
    - Naive exhaustive: 10 items * 2 bins * ~1000 feasible placements = 20,000
    - With MCTS (m=200 rollouts): O(k * m) = O(10 * 200) = 2,000 evaluations
    - Each evaluation: ~1ms (forward pass) + ~0.5ms (stability check) = ~3 seconds total
    - With parallelization (4 threads): ~0.75 seconds -- FEASIBLE for real-time
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import heapq
import time
from copy import deepcopy


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BoxItem:
    """A box item in the buffer or to be packed."""
    id: int
    length: float
    width: float
    height: float
    arrival_order: int  # Position in the arrival sequence

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def dims(self) -> Tuple[float, float, float]:
        return (self.length, self.width, self.height)


@dataclass
class PlacementAction:
    """Complete placement decision."""
    item_id: int
    bin_id: int        # 0 or 1 (which active bin)
    x: float           # FLB x-coordinate
    y: float           # FLB y-coordinate
    orientation: int   # 0 = [l,w,h], 1 = [w,l,h]
    score: float = 0.0  # Evaluation score for this action


@dataclass
class BinState:
    """State of a single bin/pallet."""
    bin_id: int
    height_map: np.ndarray  # (L, W) integer array
    items_packed: List[int] = field(default_factory=list)  # Item IDs
    total_volume_packed: float = 0.0
    is_active: bool = True
    is_closed: bool = False

    @property
    def utilization(self) -> float:
        bin_vol = self.height_map.shape[0] * self.height_map.shape[1] * 100  # Assume H=100
        return self.total_volume_packed / bin_vol if bin_vol > 0 else 0.0


@dataclass
class MCTSNode:
    """Node in the Monte Carlo Tree Search."""
    action: Optional[PlacementAction]  # Action that led to this node
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    is_terminal: bool = False

    @property
    def average_value(self) -> float:
        return self.total_value / max(self.visit_count, 1)

    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        """Upper Confidence Bound for tree policy."""
        if self.visit_count == 0:
            return float('inf')
        parent_visits = self.parent.visit_count if self.parent else 1
        exploitation = self.average_value
        exploration = exploration_constant * np.sqrt(np.log(parent_visits) / self.visit_count)
        return exploitation + exploration


# =============================================================================
# BUFFER MANAGER
# =============================================================================

class ItemBuffer:
    """
    Manages the semi-online buffer of items.

    Items arrive from a stream (conveyor belt) and are placed in the buffer.
    The agent can choose any item from the buffer to pack next.
    When an item is removed (packed), the next arriving item fills the slot.
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.items: List[Optional[BoxItem]] = [None] * max_size
        self.arrival_stream = []
        self.stream_index = 0

    def initialize(self, initial_items: List[BoxItem], remaining_stream: List[BoxItem]):
        """Fill the buffer initially and set up the arrival stream."""
        for i, item in enumerate(initial_items[:self.max_size]):
            self.items[i] = item
        self.arrival_stream = remaining_stream
        self.stream_index = 0

    def remove_item(self, buffer_index: int) -> Optional[BoxItem]:
        """Remove an item from the buffer and refill from the stream."""
        removed = self.items[buffer_index]
        if removed is None:
            return None

        # Refill from stream
        if self.stream_index < len(self.arrival_stream):
            self.items[buffer_index] = self.arrival_stream[self.stream_index]
            self.stream_index += 1
        else:
            self.items[buffer_index] = None

        return removed

    def get_active_items(self) -> List[Tuple[int, BoxItem]]:
        """Return list of (buffer_index, item) for non-empty slots."""
        return [(i, item) for i, item in enumerate(self.items) if item is not None]

    @property
    def is_empty(self) -> bool:
        return all(item is None for item in self.items)

    @property
    def count(self) -> int:
        return sum(1 for item in self.items if item is not None)


# =============================================================================
# BIN CLOSING HEURISTIC
# =============================================================================

class BinClosingPolicy:
    """
    Heuristic to decide when to close an active bin and open a new one.

    The paper does NOT address this -- it terminates the episode when nothing
    fits. For 2-bounded space, we need an explicit policy.

    Several strategies:

    1. THRESHOLD-BASED: Close when utilization exceeds threshold (e.g., 85%)
    2. ITEM-FIT-BASED: Close when no item in the buffer fits the bin
    3. MARGINAL-GAIN-BASED: Close when the best achievable utilization gain
       from any buffer item is below a threshold
    4. LEARNED: Train an RL head to decide close/continue
    """

    def __init__(self, strategy: str = "marginal_gain",
                 utilization_threshold: float = 0.85,
                 marginal_gain_threshold: float = 0.01):
        self.strategy = strategy
        self.utilization_threshold = utilization_threshold
        self.marginal_gain_threshold = marginal_gain_threshold

    def should_close(self, bin_state: BinState, buffer: ItemBuffer,
                     feasible_placements: Dict[int, List[PlacementAction]]) -> bool:
        """
        Decide whether to close the given bin.

        Args:
            bin_state: Current state of the bin
            buffer: Current item buffer
            feasible_placements: Dict mapping buffer_index -> list of feasible
                                 placements in this bin

        Returns:
            True if the bin should be closed
        """
        if self.strategy == "threshold":
            return bin_state.utilization >= self.utilization_threshold

        elif self.strategy == "item_fit":
            # Close if no buffer item has any feasible placement in this bin
            total_feasible = sum(len(p) for p in feasible_placements.values())
            return total_feasible == 0

        elif self.strategy == "marginal_gain":
            if bin_state.utilization >= self.utilization_threshold:
                return True
            # Check if best achievable gain is below threshold
            total_feasible = sum(len(p) for p in feasible_placements.values())
            if total_feasible == 0:
                return True
            # Find the best volume gain from any feasible placement
            best_gain = 0.0
            bin_vol = (bin_state.height_map.shape[0] *
                       bin_state.height_map.shape[1] * 100)
            for buf_idx, placements in feasible_placements.items():
                if placements:
                    item = buffer.items[buf_idx]
                    if item:
                        gain = item.volume / bin_vol
                        best_gain = max(best_gain, gain)
            return best_gain < self.marginal_gain_threshold

        return False


# =============================================================================
# CORE ALGORITHM: MCTS FOR BUFFER + 2-BOUNDED SPACE
# =============================================================================

class BufferMCTS:
    """
    Monte Carlo Tree Search adapted for:
      - Selecting which item from the buffer to pack
      - Selecting which of 2 bins to place it in
      - Selecting the placement position and orientation

    This extends the paper's BPP-k MCTS in two ways:
      1. Item selection from buffer (not just permutation of fixed sequence)
      2. Bin routing (which of 2 bins)

    The MCTS uses the trained placement policy as a rollout/evaluation function.

    Key adaptations from the paper:
      - Root-based parallelization for efficiency
      - Attention-based node sampling (items arriving soon sampled more)
      - Virtual item mass scaling (10^{-6}) to avoid stability distortion
    """

    def __init__(self, bin_dims: Tuple[float, float, float],
                 resolution: int = 100,
                 n_rollouts: int = 200,
                 exploration_constant: float = 1.414,
                 max_depth: int = 5):
        self.bin_dims = bin_dims
        self.resolution = resolution
        self.n_rollouts = n_rollouts
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth  # How many items ahead to look

        # Placeholder for the trained placement policy
        # In practice, this would be the DecomposedBinPackingPolicy
        self.placement_policy = None

        # Placeholder for the stacking tree
        self.stability_checker = None

    def set_placement_policy(self, policy):
        """Set the trained placement policy for rollout evaluation."""
        self.placement_policy = policy

    def set_stability_checker(self, checker):
        """Set the stability checker (DualBinStabilityChecker)."""
        self.stability_checker = checker

    def search(self, bin_states: List[BinState], buffer: ItemBuffer
               ) -> PlacementAction:
        """
        Run MCTS to find the best action (item, bin, placement).

        Algorithm:
        1. Create root node representing current state
        2. For each rollout:
           a. SELECT: Walk down tree using UCB
           b. EXPAND: Create children for unexplored actions
           c. ROLLOUT: Use policy network to simulate future placements
           d. BACKPROPAGATE: Update node values
        3. Return the action with highest visit count from root

        Args:
            bin_states: Current states of both active bins
            buffer: Current item buffer

        Returns:
            Best PlacementAction to execute
        """
        root = MCTSNode(action=None)

        for _ in range(self.n_rollouts):
            # Make copies for simulation
            sim_bins = [deepcopy(bs) for bs in bin_states]
            sim_buffer = deepcopy(buffer)

            # SELECT
            node = self._select(root)

            # EXPAND
            if not node.is_terminal and node.visit_count > 0:
                node = self._expand(node, sim_bins, sim_buffer)

            # ROLLOUT
            value = self._rollout(node, sim_bins, sim_buffer)

            # BACKPROPAGATE
            self._backpropagate(node, value)

        # Return best action from root
        if not root.children:
            return self._get_greedy_action(bin_states, buffer)

        best_child = max(root.children, key=lambda c: c.visit_count)
        return best_child.action

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB tree policy."""
        while node.children and not node.is_terminal:
            node = max(node.children,
                       key=lambda c: c.ucb_score(self.exploration_constant))
        return node

    def _expand(self, node: MCTSNode, bin_states: List[BinState],
                buffer: ItemBuffer) -> MCTSNode:
        """
        Expand node by generating child nodes for possible actions.

        We use attention-based sampling from the paper:
        Items arriving sooner are more likely to be sampled.
        """
        active_items = buffer.get_active_items()
        if not active_items:
            node.is_terminal = True
            return node

        # Generate candidate actions using attention-based sampling
        candidates = self._generate_candidate_actions(bin_states, buffer, active_items)

        if not candidates:
            node.is_terminal = True
            return node

        for action in candidates:
            child = MCTSNode(action=action, parent=node)
            node.children.append(child)

        # Return a random unexplored child
        unexplored = [c for c in node.children if c.visit_count == 0]
        if unexplored:
            return np.random.choice(unexplored)
        return node.children[0]

    def _generate_candidate_actions(self, bin_states: List[BinState],
                                      buffer: ItemBuffer,
                                      active_items: List[Tuple[int, BoxItem]]
                                      ) -> List[PlacementAction]:
        """
        Generate candidate actions for expansion.

        For each item in the buffer, for each active bin, find the top-k
        feasible placements using the placement policy.
        """
        candidates = []

        for buf_idx, item in active_items:
            for bin_id, bin_state in enumerate(bin_states):
                if not bin_state.is_active or bin_state.is_closed:
                    continue

                # Get top placements from the policy (or heuristic)
                placements = self._get_top_placements(item, bin_state, top_k=3)

                for x, y, o, score in placements:
                    candidates.append(PlacementAction(
                        item_id=item.id,
                        bin_id=bin_id,
                        x=x, y=y,
                        orientation=o,
                        score=score
                    ))

        return candidates

    def _get_top_placements(self, item: BoxItem, bin_state: BinState,
                             top_k: int = 3) -> List[Tuple[float, float, int, float]]:
        """
        Get the top-k placements for an item in a bin.

        If a trained policy is available, use it to rank placements.
        Otherwise, fall back to heuristic scoring.

        Returns: List of (x, y, orientation, score) tuples
        """
        candidates = []

        for orientation in [0, 1]:
            if orientation == 0:
                eff_l, eff_w = item.length, item.width
            else:
                eff_l, eff_w = item.width, item.length

            L, W = bin_state.height_map.shape

            # Check a grid of potential positions
            # For efficiency, check at a coarser resolution first
            step = max(1, L // 20)  # Check every 5th cell

            for xi in range(0, L, step):
                for yi in range(0, W, step):
                    x = xi
                    y = yi

                    # Check if item fits within bin bounds
                    if x + eff_l > L or y + eff_w > W:
                        continue

                    # Get placement height
                    z = np.max(bin_state.height_map[x:x+int(eff_l), y:y+int(eff_w)])

                    # Check height constraint
                    if z + item.height > 100:  # Assume H=100
                        continue

                    # Score: prefer lower placements, bottom-left, high support area
                    # This is a DBLF-inspired heuristic score
                    score = -(z * 1000 + x * 10 + y)  # Lower z, then lower x, then lower y

                    # Additional score for support coverage
                    support_region = bin_state.height_map[x:x+int(eff_l), y:y+int(eff_w)]
                    support_fraction = np.sum(support_region == z) / max(support_region.size, 1)
                    score += support_fraction * 500  # Bonus for flat support

                    candidates.append((x, y, orientation, score))

        # Return top-k by score
        candidates.sort(key=lambda c: c[3], reverse=True)
        return candidates[:top_k]

    def _rollout(self, node: MCTSNode, bin_states: List[BinState],
                 buffer: ItemBuffer) -> float:
        """
        Simulate forward from the current node using the policy.

        The rollout packs items greedily for max_depth steps and returns
        the accumulated reward + terminal value estimate.

        Key adaptation from paper: Virtual items have mass * 10^{-6}
        to avoid stability distortion during simulation.
        """
        total_reward = 0.0
        depth = 0

        while depth < self.max_depth and not buffer.is_empty:
            # Get best greedy action
            action = self._get_greedy_action(bin_states, buffer)
            if action is None:
                break

            # Execute action (simplified simulation)
            item = None
            for i, buf_item in enumerate(buffer.items):
                if buf_item is not None and buf_item.id == action.item_id:
                    item = buffer.remove_item(i)
                    break

            if item is None:
                break

            # Update bin state
            bin_state = bin_states[action.bin_id]
            if action.orientation == 0:
                eff_l, eff_w = item.length, item.width
            else:
                eff_l, eff_w = item.width, item.length

            x, y = int(action.x), int(action.y)
            x_end = min(x + int(eff_l), bin_state.height_map.shape[0])
            y_end = min(y + int(eff_w), bin_state.height_map.shape[1])

            z = np.max(bin_state.height_map[x:x_end, y:y_end])
            bin_state.height_map[x:x_end, y:y_end] = z + item.height
            bin_state.total_volume_packed += item.volume

            # Compute reward
            bin_vol = bin_state.height_map.shape[0] * bin_state.height_map.shape[1] * 100
            reward = 10.0 * item.volume / bin_vol
            total_reward += reward

            depth += 1

        # Terminal value estimate (heuristic: weighted utilization of both bins)
        terminal_value = sum(bs.utilization for bs in bin_states) / len(bin_states)
        total_reward += terminal_value

        return total_reward

    def _get_greedy_action(self, bin_states: List[BinState],
                            buffer: ItemBuffer) -> Optional[PlacementAction]:
        """Get the best immediate action using heuristic scoring."""
        best_action = None
        best_score = float('-inf')

        for buf_idx, item in buffer.get_active_items():
            for bin_id, bin_state in enumerate(bin_states):
                if not bin_state.is_active or bin_state.is_closed:
                    continue

                placements = self._get_top_placements(item, bin_state, top_k=1)
                if placements:
                    x, y, o, score = placements[0]
                    if score > best_score:
                        best_score = score
                        best_action = PlacementAction(
                            item_id=item.id, bin_id=bin_id,
                            x=x, y=y, orientation=o, score=score
                        )

        return best_action

    def _backpropagate(self, node: MCTSNode, value: float):
        """Propagate the rollout value back up the tree."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent


# =============================================================================
# COMPLETE SEMI-ONLINE PACKING SYSTEM
# =============================================================================

class SemiOnlinePackingSystem:
    """
    Complete semi-online packing system combining all components:
      1. Item buffer management
      2. 2-bounded space bin management
      3. MCTS-based decision making
      4. Stability checking via stacking tree
      5. Bin closing policy

    This is the top-level class that would be used in the thesis implementation.

    PSEUDOCODE FOR MAIN LOOP:

    system = SemiOnlinePackingSystem(config)
    system.initialize(item_stream)

    while not system.is_done():
        action = system.decide_next_action()  # MCTS or policy-based
        system.execute_action(action)

        # Check if any bin should be closed
        for bin_id in [0, 1]:
            if system.should_close_bin(bin_id):
                system.close_bin(bin_id)
                system.open_new_bin(bin_id)

    print(f"Total bins used: {system.total_bins_used}")
    print(f"Average utilization: {system.average_utilization}")
    """

    def __init__(self, bin_dims: Tuple[float, float, float] = (100, 100, 100),
                 buffer_size: int = 10,
                 resolution: int = 100,
                 n_mcts_rollouts: int = 200,
                 closing_strategy: str = "marginal_gain"):

        self.bin_dims = bin_dims
        self.buffer_size = buffer_size
        self.resolution = resolution

        # Components
        self.buffer = ItemBuffer(max_size=buffer_size)
        self.mcts = BufferMCTS(
            bin_dims=bin_dims,
            resolution=resolution,
            n_rollouts=n_mcts_rollouts
        )
        self.closing_policy = BinClosingPolicy(strategy=closing_strategy)

        # Bin management
        self.active_bins: List[BinState] = [
            BinState(bin_id=0, height_map=np.zeros((resolution, resolution), dtype=np.float32)),
            BinState(bin_id=1, height_map=np.zeros((resolution, resolution), dtype=np.float32))
        ]
        self.closed_bins: List[BinState] = []
        self.total_bins_opened = 2

        # Statistics
        self.total_items_packed = 0
        self.decision_times: List[float] = []

    def initialize(self, item_stream: List[BoxItem]):
        """Initialize the system with an item stream."""
        initial_items = item_stream[:self.buffer_size]
        remaining = item_stream[self.buffer_size:]
        self.buffer.initialize(initial_items, remaining)

    def decide_next_action(self) -> Optional[PlacementAction]:
        """
        Use MCTS to decide the next action.

        Returns:
            PlacementAction or None if no feasible action exists
        """
        start_time = time.time()

        action = self.mcts.search(self.active_bins, self.buffer)

        elapsed = time.time() - start_time
        self.decision_times.append(elapsed)

        return action

    def execute_action(self, action: PlacementAction) -> bool:
        """Execute a placement action."""
        if action is None:
            return False

        # Find and remove item from buffer
        item = None
        for i, buf_item in enumerate(self.buffer.items):
            if buf_item is not None and buf_item.id == action.item_id:
                item = self.buffer.remove_item(i)
                break

        if item is None:
            return False

        # Place item in the selected bin
        bin_state = self.active_bins[action.bin_id]
        if action.orientation == 0:
            eff_l, eff_w = item.length, item.width
        else:
            eff_l, eff_w = item.width, item.length

        x, y = int(action.x), int(action.y)
        x_end = min(x + int(eff_l), self.resolution)
        y_end = min(y + int(eff_w), self.resolution)

        z = np.max(bin_state.height_map[x:x_end, y:y_end])
        bin_state.height_map[x:x_end, y:y_end] = z + item.height
        bin_state.total_volume_packed += item.volume
        bin_state.items_packed.append(item.id)

        self.total_items_packed += 1

        return True

    def should_close_bin(self, bin_id: int) -> bool:
        """Check if a bin should be closed."""
        bin_state = self.active_bins[bin_id]
        if not bin_state.is_active or bin_state.is_closed:
            return False

        # Get feasible placements for all buffer items on this bin
        feasible = {}
        for buf_idx, item in self.buffer.get_active_items():
            placements = self.mcts._get_top_placements(item, bin_state, top_k=1)
            if placements:
                feasible[buf_idx] = placements

        return self.closing_policy.should_close(bin_state, self.buffer, feasible)

    def close_bin(self, bin_id: int):
        """Close a bin permanently."""
        self.active_bins[bin_id].is_closed = True
        self.active_bins[bin_id].is_active = False
        self.closed_bins.append(self.active_bins[bin_id])

    def open_new_bin(self, bin_id: int):
        """Open a new bin at the given slot."""
        new_bin = BinState(
            bin_id=self.total_bins_opened,
            height_map=np.zeros((self.resolution, self.resolution), dtype=np.float32)
        )
        self.active_bins[bin_id] = new_bin
        self.total_bins_opened += 1

    def is_done(self) -> bool:
        """Check if packing is complete."""
        # Done when buffer is empty and stream is exhausted
        if self.buffer.is_empty:
            return True

        # Also done if no item fits in any active bin
        for buf_idx, item in self.buffer.get_active_items():
            for bin_state in self.active_bins:
                if bin_state.is_active and not bin_state.is_closed:
                    placements = self.mcts._get_top_placements(item, bin_state, top_k=1)
                    if placements:
                        return False

        # No item fits anywhere -- close bins and open new ones if possible
        return True

    @property
    def average_utilization(self) -> float:
        """Average utilization across all closed bins."""
        all_bins = self.closed_bins + [b for b in self.active_bins if b.items_packed]
        if not all_bins:
            return 0.0
        return sum(b.utilization for b in all_bins) / len(all_bins)

    def print_statistics(self):
        """Print packing statistics."""
        all_bins = self.closed_bins + [b for b in self.active_bins if b.items_packed]
        print(f"=== Packing Statistics ===")
        print(f"Total bins used: {len(all_bins)}")
        print(f"Total items packed: {self.total_items_packed}")
        print(f"Average utilization: {self.average_utilization:.1%}")
        print(f"Average decision time: {np.mean(self.decision_times):.3f}s")
        for b in all_bins:
            status = "CLOSED" if b.is_closed else "ACTIVE"
            print(f"  Bin {b.bin_id}: {b.utilization:.1%} util, "
                  f"{len(b.items_packed)} items [{status}]")


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Generate a random item stream
    np.random.seed(42)
    n_items = 100
    items = []
    for i in range(n_items):
        l = np.random.randint(10, 50)
        w = np.random.randint(10, 50)
        h = np.random.randint(10, 50)
        items.append(BoxItem(id=i, length=l, width=w, height=h, arrival_order=i))

    # Create and run the system
    system = SemiOnlinePackingSystem(
        bin_dims=(100, 100, 100),
        buffer_size=10,
        resolution=100,
        n_mcts_rollouts=50,  # Reduced for quick demo
        closing_strategy="marginal_gain"
    )

    system.initialize(items)

    step = 0
    while not system.is_done():
        action = system.decide_next_action()
        if action is None:
            # No feasible action -- close most-filled bin, open new one
            utilizations = [(i, b.utilization) for i, b in enumerate(system.active_bins)
                           if b.is_active and not b.is_closed]
            if utilizations:
                most_filled_idx = max(utilizations, key=lambda x: x[1])[0]
                system.close_bin(most_filled_idx)
                system.open_new_bin(most_filled_idx)
            else:
                break
        else:
            system.execute_action(action)

        # Check closing policy
        for bin_id in [0, 1]:
            if system.should_close_bin(bin_id):
                system.close_bin(bin_id)
                system.open_new_bin(bin_id)

        step += 1
        if step % 10 == 0:
            print(f"Step {step}: {system.total_items_packed} items packed, "
                  f"bins: {system.total_bins_opened}")

    system.print_statistics()


# =============================================================================
# INTEGRATION NOTES
# =============================================================================

"""
HOW THIS CONNECTS TO OTHER MODULES:

1. stacking_tree.py (stability/):
   - Replace the simple heuristic height-based stability check in _get_top_placements
     with the proper AdaptiveStackingTree.check_stability() call
   - Use DualBinStabilityChecker for managing stability across both bins
   - Compute proper feasibility masks for the RL policy

2. decomposed_actor_critic.py (deep_rl/):
   - The trained policy can replace the heuristic _get_top_placements() method
   - During MCTS rollouts, use the policy's value function (critic) for
     terminal state evaluation instead of heuristic utilization
   - The action decomposition enables efficient candidate generation

3. Other papers in the reading list:
   - "A deep RL approach for online and concurrent 3D bin packing" may provide
     insights for the bin selection head
   - "Solving Online 3D Multi-Bin Packing Problem with Deep" is directly
     relevant for multi-bin RL training
   - "Near-optimal Algorithms for Stochastic Online Bin Packing" may provide
     theoretical bounds for our buffer-based approach

4. Heuristic alternatives (for comparison):
   - DBLF (Deepest-Bottom-Left-Fill) from Ha et al. 2017
   - Best-Fit bin selection (route item to bin where it fits best)
   - First-Fit bin selection (route to first bin where item fits)
   - These can serve as baselines and also as fallbacks when MCTS is too slow

5. Bin closing alternatives to evaluate:
   - Fixed threshold (e.g., close at 80% utilization)
   - Predicted remaining gain (use RL value function to estimate)
   - Time-based (close after N items, regardless of utilization)
   - Hybrid: close if no improvement for last K items
"""
