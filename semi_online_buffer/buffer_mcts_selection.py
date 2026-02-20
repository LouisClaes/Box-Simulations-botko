"""
=============================================================================
CODING IDEAS: Semi-Online Buffer Selection with MCTS
=============================================================================
Based on: Zhao et al. (2021) "Online 3D Bin Packing with Constrained DRL"
          Specifically the BPP-k extension with Monte Carlo Tree Search

Adapted for our thesis use case:
  - Buffer of 5-10 items (choose WHICH item to pack, not just WHERE)
  - 2-bounded space (2 active bins, close one and open new when full)
  - The paper's BPP-k uses MCTS to search permutations of lookahead items
  - We adapt this to search over (item selection, bin selection) decisions

KEY DIFFERENCE from the paper:
  Paper: Items must be packed in arrival order; lookahead is passive
  Ours:  Items can be selected from buffer in ANY order; buffer is active

This is actually EASIER than the paper's problem because we have more freedom.
The paper's MCTS permutation search already solves the harder variant.
=============================================================================
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


# =============================================================================
# 1. BUFFER DATA STRUCTURES
# =============================================================================

@dataclass
class BufferItem:
    """An item in the selection buffer."""
    dimensions: np.ndarray  # [l, w, h]
    item_id: int
    arrival_time: int

    @property
    def volume(self) -> float:
        return float(np.prod(self.dimensions))


@dataclass
class BinState:
    """Simplified bin state using height map."""
    height_map: np.ndarray
    L: int
    W: int
    H: int
    volume_used: float = 0.0
    items_packed: int = 0
    is_active: bool = True

    @property
    def utilization(self) -> float:
        return self.volume_used / (self.L * self.W * self.H)

    def copy(self) -> 'BinState':
        return BinState(
            height_map=self.height_map.copy(),
            L=self.L, W=self.W, H=self.H,
            volume_used=self.volume_used,
            items_packed=self.items_packed,
            is_active=self.is_active,
        )


# =============================================================================
# 2. MCTS FOR BUFFER ITEM AND BIN SELECTION
# =============================================================================

@dataclass
class MCTSNode:
    """Node in the MCTS search tree."""
    # State
    bin_states: List[BinState]
    remaining_buffer: List[BufferItem]

    # Action that led to this node
    item_idx: Optional[int] = None
    bin_idx: Optional[int] = None
    position: Optional[int] = None

    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0
    children: List['MCTSNode'] = field(default_factory=list)
    parent: Optional['MCTSNode'] = None

    @property
    def average_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, exploration_constant: float = 1.41) -> float:
        """Upper Confidence Bound score for node selection."""
        if self.visit_count == 0:
            return float('inf')
        if self.parent is None:
            return self.average_value

        exploitation = self.average_value
        exploration = exploration_constant * np.sqrt(
            np.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration


class BufferMCTS:
    """
    Monte Carlo Tree Search for buffer item and bin selection.

    At each decision point, the agent must decide:
    1. Which item from the buffer to pack next
    2. Which of the 2 active bins to place it in
    3. Where in that bin to place it (position on grid)

    The MCTS explores sequences of (item, bin, position) decisions
    to find the best FIRST action.

    Complexity: O(buffer_size * 2 * num_positions * num_simulations)
    For buffer=5, 2 bins, 100 positions, 50 simulations: ~50,000 evaluations
    Each evaluation is cheap (height map update + feasibility check).
    """

    def __init__(self,
                 network,  # ConstrainedPackingNetwork for position selection + value
                 num_simulations: int = 100,
                 exploration_constant: float = 1.41,
                 search_depth: int = 3,  # How many steps ahead to search
                 position_candidates: int = 5):  # Top-k positions per (item, bin)
        self.network = network
        self.num_simulations = num_simulations
        self.c = exploration_constant
        self.search_depth = search_depth
        self.position_candidates = position_candidates

    def search(self, bin_states: List[BinState],
               buffer: List[BufferItem],
               feasibility_fn) -> Tuple[int, int, int]:
        """
        Run MCTS to find the best (item_idx, bin_idx, position).

        Args:
            bin_states: Current state of the 2 active bins
            buffer: Current items in the buffer
            feasibility_fn: Function to compute feasibility masks

        Returns:
            (item_idx, bin_idx, position) -- the best first action
        """
        # Create root node
        root = MCTSNode(
            bin_states=[b.copy() for b in bin_states],
            remaining_buffer=list(buffer),
        )

        # Run simulations
        for _ in range(self.num_simulations):
            # 1. Selection: traverse tree using UCB
            node = self._select(root)

            # 2. Expansion: add child nodes
            if node.visit_count > 0 and len(node.remaining_buffer) > 0:
                node = self._expand(node, feasibility_fn)

            # 3. Simulation: rollout from node
            value = self._simulate(node, feasibility_fn)

            # 4. Backpropagation: update statistics
            self._backpropagate(node, value)

        # Return best child of root (most visited)
        if not root.children:
            # No valid actions -- return None
            return None, None, None

        best_child = max(root.children, key=lambda c: c.visit_count)
        return best_child.item_idx, best_child.bin_idx, best_child.position

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select leaf node using UCB."""
        while node.children:
            node = max(node.children, key=lambda c: c.ucb_score(self.c))
        return node

    def _expand(self, node: MCTSNode, feasibility_fn) -> MCTSNode:
        """
        Expand node by generating child nodes for all valid
        (item, bin, position) combinations.

        To keep the branching factor manageable:
        - Consider all items in buffer
        - Consider both bins
        - Consider only top-k positions (from network actor) per (item, bin)
        """
        for item_idx, item in enumerate(node.remaining_buffer):
            l, w, h = item.dimensions

            for bin_idx in range(2):
                if not node.bin_states[bin_idx].is_active:
                    continue

                bin_state = node.bin_states[bin_idx]
                mask = feasibility_fn(bin_state.height_map, l, w, h)

                if not np.any(mask):
                    continue

                # Get top-k positions from network (or from mask directly)
                positions = self._get_top_positions(
                    bin_state, item.dimensions, mask
                )

                for position in positions:
                    # Create child node with this action applied
                    new_bins = [b.copy() for b in node.bin_states]
                    x = position % bin_state.L
                    y = position // bin_state.L

                    # Place item
                    region = new_bins[bin_idx].height_map[x:x+l, y:y+w]
                    placement_z = np.max(region)
                    new_bins[bin_idx].height_map[x:x+l, y:y+w] = placement_z + h
                    new_bins[bin_idx].volume_used += l * w * h
                    new_bins[bin_idx].items_packed += 1

                    # Remove item from buffer
                    new_buffer = [
                        it for i, it in enumerate(node.remaining_buffer)
                        if i != item_idx
                    ]

                    child = MCTSNode(
                        bin_states=new_bins,
                        remaining_buffer=new_buffer,
                        item_idx=item_idx,
                        bin_idx=bin_idx,
                        position=position,
                        parent=node,
                    )
                    node.children.append(child)

        # Return a random child for simulation
        if node.children:
            return node.children[np.random.randint(len(node.children))]
        return node

    def _simulate(self, node: MCTSNode, feasibility_fn) -> float:
        """
        Simulate (rollout) from node using a simple policy.

        Uses greedy item-bin-position selection for fast rollout.
        Returns accumulated reward + critic value at terminal state.
        """
        bins = [b.copy() for b in node.bin_states]
        buffer = list(node.remaining_buffer)
        total_reward = 0.0

        for _ in range(min(self.search_depth, len(buffer))):
            if not buffer:
                break

            # Greedy: try each item in each bin, pick best
            best_reward = -1
            best_action = None

            for item_idx, item in enumerate(buffer):
                l, w, h = item.dimensions
                for bin_idx in range(2):
                    if not bins[bin_idx].is_active:
                        continue

                    mask = feasibility_fn(bins[bin_idx].height_map, l, w, h)
                    if not np.any(mask):
                        continue

                    # Best position by volume efficiency
                    volume_reward = 10.0 * (l * w * h) / (
                        bins[bin_idx].L * bins[bin_idx].W * bins[bin_idx].H)

                    if volume_reward > best_reward:
                        # Pick the first feasible position
                        pos = np.argmax(mask.flatten())
                        best_reward = volume_reward
                        best_action = (item_idx, bin_idx, pos)

            if best_action is None:
                break

            item_idx, bin_idx, pos = best_action
            item = buffer[item_idx]
            l, w, h = item.dimensions
            x = pos % bins[bin_idx].L
            y = pos // bins[bin_idx].L

            region = bins[bin_idx].height_map[x:x+l, y:y+w]
            placement_z = np.max(region)
            bins[bin_idx].height_map[x:x+l, y:y+w] = placement_z + h
            bins[bin_idx].volume_used += l * w * h

            total_reward += best_reward
            buffer.pop(item_idx)

        # Add critic value estimate for remaining potential
        # (Use average utilization as a simple heuristic)
        avg_util = np.mean([b.utilization for b in bins if b.is_active])
        estimated_future = avg_util * 5.0  # Rough heuristic

        return total_reward + estimated_future

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    def _get_top_positions(self, bin_state: BinState,
                           item_dims: np.ndarray,
                           mask: np.ndarray) -> List[int]:
        """
        Get top-k feasible positions for item in bin.

        Uses a scoring heuristic (DBLF-inspired) to rank positions.
        """
        L, W = bin_state.L, bin_state.W
        feasible_positions = []

        for pos in range(L * W):
            x, y = pos % L, pos // L
            if mask[x, y] > 0:
                # Score: prefer back-bottom-left (DBLF principle)
                # Lower x (deeper), lower z (height at position), lower y (left)
                l, w, h = item_dims
                region = bin_state.height_map[x:x+l, y:y+w]
                z = np.max(region)
                score = -(x * 1000 + z * 100 + y)  # negative for sorting
                feasible_positions.append((pos, score))

        # Sort by score (descending) and take top-k
        feasible_positions.sort(key=lambda p: p[1], reverse=True)
        return [pos for pos, _ in feasible_positions[:self.position_candidates]]


# =============================================================================
# 3. SIMPLIFIED BUFFER SELECTION (Without Full MCTS)
# =============================================================================

class GreedyBufferSelector:
    """
    Simpler buffer selection strategies for comparison with MCTS.

    These can serve as:
    1. Baselines for evaluating MCTS improvement
    2. Fast alternatives when MCTS is too slow
    3. Rollout policies within MCTS simulations
    """

    @staticmethod
    def largest_volume_first(buffer: List[BufferItem],
                             bin_states: List[BinState],
                             feasibility_fn) -> Optional[Tuple[int, int, int]]:
        """
        Pack the largest-volume item that fits in any bin.

        Rationale: Large items are hardest to place later, so pack them first.
        This is a variant of the "decreasing" offline heuristic applied to
        the buffer.
        """
        # Sort buffer by volume (descending)
        sorted_indices = sorted(
            range(len(buffer)),
            key=lambda i: buffer[i].volume,
            reverse=True
        )

        for item_idx in sorted_indices:
            item = buffer[item_idx]
            l, w, h = item.dimensions

            for bin_idx in range(2):
                if not bin_states[bin_idx].is_active:
                    continue

                mask = feasibility_fn(bin_states[bin_idx].height_map, l, w, h)
                if np.any(mask):
                    # DBLF position selection
                    pos = GreedyBufferSelector._dblf_position(
                        bin_states[bin_idx], mask, l, w, h
                    )
                    return item_idx, bin_idx, pos

        return None

    @staticmethod
    def best_fit_item(buffer: List[BufferItem],
                      bin_states: List[BinState],
                      feasibility_fn) -> Optional[Tuple[int, int, int]]:
        """
        Pack the item that creates the least wasted space.

        For each (item, bin, position) triple, compute the "waste":
        the gap between the item top and surrounding heights.
        Choose the triple with least waste.
        """
        best_waste = float('inf')
        best_action = None

        for item_idx, item in enumerate(buffer):
            l, w, h = item.dimensions

            for bin_idx in range(2):
                if not bin_states[bin_idx].is_active:
                    continue

                bin_state = bin_states[bin_idx]
                mask = feasibility_fn(bin_state.height_map, l, w, h)

                if not np.any(mask):
                    continue

                L, W = bin_state.L, bin_state.W
                for pos in range(L * W):
                    x, y = pos % L, pos // L
                    if mask[x, y] == 0:
                        continue

                    # Compute waste: height variance in footprint region
                    region = bin_state.height_map[x:x+l, y:y+w]
                    placement_z = np.max(region)
                    gap = placement_z * l * w - np.sum(region)
                    waste = gap  # Volume of air beneath the item

                    if waste < best_waste:
                        best_waste = waste
                        best_action = (item_idx, bin_idx, pos)

        return best_action

    @staticmethod
    def critic_guided(buffer: List[BufferItem],
                      bin_states: List[BinState],
                      feasibility_fn,
                      network) -> Optional[Tuple[int, int, int]]:
        """
        Use the critic network to evaluate each (item, bin) pair.

        This is the paper's multi-bin selection approach adapted
        for buffer selection:
        - For each item in buffer and each active bin
        - Compute critic value V(s') for the state after placing item
        - Choose the (item, bin) with highest V(s')
        - Then use actor for position selection

        This is O(buffer_size * 2) network evaluations.
        """
        import torch

        best_value = float('-inf')
        best_action = None

        for item_idx, item in enumerate(buffer):
            l, w, h = item.dimensions

            for bin_idx in range(2):
                if not bin_states[bin_idx].is_active:
                    continue

                bin_state = bin_states[bin_idx]
                mask = feasibility_fn(bin_state.height_map, l, w, h)

                if not np.any(mask):
                    continue

                # Create observation for this (item, bin)
                obs = np.zeros((bin_state.L, bin_state.W, 4), dtype=np.float32)
                obs[:, :, 0] = bin_state.height_map / bin_state.H
                obs[:, :, 1] = l / bin_state.L
                obs[:, :, 2] = w / bin_state.W
                obs[:, :, 3] = h / bin_state.H

                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                mask_tensor = torch.FloatTensor(mask).unsqueeze(0)

                with torch.no_grad():
                    output = network(obs_tensor, gt_mask=mask_tensor)
                    value = output['value'].item()
                    action_probs = output['action_probs'].squeeze(0)

                if value > best_value:
                    best_value = value
                    # Get best position from actor
                    pos = torch.argmax(action_probs).item()
                    best_action = (item_idx, bin_idx, pos)

        return best_action

    @staticmethod
    def _dblf_position(bin_state: BinState, mask: np.ndarray,
                       l: int, w: int, h: int) -> int:
        """Select position using DBLF (Deepest-Bottom-Left with Fill)."""
        L, W = bin_state.L, bin_state.W
        best_pos = None
        best_score = (float('inf'), float('inf'), float('inf'))

        for pos in range(L * W):
            x, y = pos % L, pos // L
            if mask[x, y] == 0:
                continue

            region = bin_state.height_map[x:x+l, y:y+w]
            z = np.max(region)

            # DBLF: minimize (x, z, y) lexicographically
            score = (x, z, y)
            if score < best_score:
                best_score = score
                best_pos = pos

        return best_pos


# =============================================================================
# 4. BIN CLOSING POLICY FOR 2-BOUNDED SPACE
# =============================================================================

class BinClosingPolicy:
    """
    Policy for deciding when to close a bin in 2-bounded space.

    In 2-bounded space:
    - Only 2 bins are active at any time
    - When a bin is closed, it is PERMANENTLY sealed
    - A new empty bin takes its slot

    Closing too early: wastes space (low utilization)
    Closing too late: blocks items that could fit in a fresh bin

    Several strategies, from simple to complex:
    """

    @staticmethod
    def no_fit_close(bin_state: BinState,
                     buffer: List[BufferItem],
                     feasibility_fn) -> bool:
        """
        Close bin when NO item in buffer fits.

        Simplest strategy. Used in the paper's multi-bin extension.
        Problem: by the time no item fits, utilization is already
        as high as it can get with current buffer.
        """
        for item in buffer:
            mask = feasibility_fn(bin_state.height_map, *item.dimensions)
            if np.any(mask):
                return False
        return True

    @staticmethod
    def threshold_close(bin_state: BinState,
                        threshold: float = 0.80) -> bool:
        """
        Close bin when utilization exceeds threshold.

        Simple and predictable. Threshold must be tuned:
        - Too low (0.5): wastes space
        - Too high (0.95): may never close, blocking new bins
        - Sweet spot: typically 0.75-0.85
        """
        return bin_state.utilization >= threshold

    @staticmethod
    def remaining_capacity_close(bin_state: BinState,
                                 buffer: List[BufferItem],
                                 feasibility_fn,
                                 min_items: int = 1) -> bool:
        """
        Close bin when fewer than min_items from buffer can still fit.

        More nuanced than no_fit_close: closes when the bin is becoming
        impractical rather than completely full.
        """
        fitting_count = 0
        for item in buffer:
            mask = feasibility_fn(bin_state.height_map, *item.dimensions)
            if np.any(mask):
                fitting_count += 1
                if fitting_count >= min_items:
                    return False
        return True

    @staticmethod
    def critic_based_close(bin_state: BinState,
                           network,
                           buffer: List[BufferItem],
                           value_threshold: float = 0.5) -> bool:
        """
        Close bin when critic predicts low future value.

        Use the trained critic to estimate the expected future reward
        from continuing to pack this bin. If it's below a threshold,
        close the bin and start fresh.

        This is the most sophisticated approach and requires a trained
        critic network.
        """
        import torch

        # Compute average critic value over buffer items
        total_value = 0.0
        evaluated = 0

        for item in buffer:
            l, w, h = item.dimensions
            obs = np.zeros((bin_state.L, bin_state.W, 4), dtype=np.float32)
            obs[:, :, 0] = bin_state.height_map / bin_state.H
            obs[:, :, 1] = l / bin_state.L
            obs[:, :, 2] = w / bin_state.W
            obs[:, :, 3] = h / bin_state.H

            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                output = network(obs_tensor)
                total_value += output['value'].item()
                evaluated += 1

        if evaluated == 0:
            return True

        avg_value = total_value / evaluated
        return avg_value < value_threshold


# =============================================================================
# 5. FULL PIPELINE: SEMI-ONLINE 2-BOUNDED WITH BUFFER
# =============================================================================

class SemiOnlineTwoBoundedPacker:
    """
    Complete packing pipeline combining all components.

    This is the main class that orchestrates:
    1. Buffer management (receiving items, maintaining buffer)
    2. Item selection (MCTS or greedy from buffer)
    3. Bin selection (critic-based or rule-based)
    4. Position selection (actor with feasibility mask)
    5. Bin closing (policy-based)

    Flow per step:
    1. New item arrives on conveyor -> add to buffer
    2. If buffer is full, must pack one item:
       a. MCTS/Greedy: select (item, bin, position)
       b. Place item, update bin height map
       c. Check bin closing conditions
       d. If close: seal bin, open new empty bin
    3. Repeat until all items processed
    """

    def __init__(self,
                 L: int = 10, W: int = 10, H: int = 10,
                 buffer_size: int = 5,
                 selection_strategy: str = 'critic',  # 'mcts', 'critic', 'greedy'
                 closing_strategy: str = 'no_fit',  # 'no_fit', 'threshold', 'critic'
                 network=None):
        self.L, self.W, self.H = L, W, H
        self.buffer_size = buffer_size
        self.selection_strategy = selection_strategy
        self.closing_strategy = closing_strategy
        self.network = network

        # State
        self.bins = [
            BinState(np.zeros((L, W), dtype=np.int32), L, W, H),
            BinState(np.zeros((L, W), dtype=np.int32), L, W, H),
        ]
        self.buffer: List[BufferItem] = []
        self.completed_bins: List[BinState] = []
        self.item_counter = 0

    def receive_item(self, dimensions: np.ndarray) -> Optional[Dict]:
        """
        Receive a new item from the conveyor.

        If buffer is not full, add to buffer and return None.
        If buffer is full, pack one item and return packing info.
        """
        item = BufferItem(
            dimensions=dimensions,
            item_id=self.item_counter,
            arrival_time=self.item_counter,
        )
        self.item_counter += 1
        self.buffer.append(item)

        if len(self.buffer) > self.buffer_size:
            return self._pack_one_item()

        return None

    def _pack_one_item(self) -> Dict:
        """Select and pack one item from the buffer."""

        # Select (item, bin, position) using chosen strategy
        if self.selection_strategy == 'mcts':
            mcts = BufferMCTS(self.network, num_simulations=100)
            item_idx, bin_idx, position = mcts.search(
                self.bins, self.buffer, self._feasibility_fn
            )
        elif self.selection_strategy == 'critic':
            result = GreedyBufferSelector.critic_guided(
                self.buffer, self.bins, self._feasibility_fn, self.network
            )
            if result:
                item_idx, bin_idx, position = result
            else:
                return self._handle_no_fit()
        else:  # greedy
            result = GreedyBufferSelector.largest_volume_first(
                self.buffer, self.bins, self._feasibility_fn
            )
            if result:
                item_idx, bin_idx, position = result
            else:
                return self._handle_no_fit()

        if item_idx is None:
            return self._handle_no_fit()

        # Place item
        item = self.buffer[item_idx]
        l, w, h = item.dimensions
        x = position % self.L
        y = position // self.L

        bin_state = self.bins[bin_idx]
        region = bin_state.height_map[x:x+l, y:y+w]
        placement_z = int(np.max(region))
        bin_state.height_map[x:x+l, y:y+w] = placement_z + h
        bin_state.volume_used += l * w * h
        bin_state.items_packed += 1

        # Remove from buffer
        self.buffer.pop(item_idx)

        # Check bin closing
        self._check_and_close_bins()

        return {
            'item': item,
            'bin_idx': bin_idx,
            'position': (x, y, placement_z),
            'utilizations': [b.utilization for b in self.bins],
            'completed_bins': len(self.completed_bins),
        }

    def _handle_no_fit(self) -> Dict:
        """Handle case where no item fits in any bin."""
        # Close the fuller bin and open a new one
        worse_idx = 0 if self.bins[0].utilization >= self.bins[1].utilization else 1
        self._close_bin(worse_idx)

        # Try again with the new empty bin
        result = GreedyBufferSelector.largest_volume_first(
            self.buffer, self.bins, self._feasibility_fn
        )
        if result:
            item_idx, bin_idx, position = result
            item = self.buffer[item_idx]
            l, w, h = item.dimensions
            x = position % self.L
            y = position // self.L
            bin_state = self.bins[bin_idx]
            region = bin_state.height_map[x:x+l, y:y+w]
            placement_z = int(np.max(region))
            bin_state.height_map[x:x+l, y:y+w] = placement_z + h
            bin_state.volume_used += l * w * h
            bin_state.items_packed += 1
            self.buffer.pop(item_idx)
            return {
                'item': item,
                'bin_idx': bin_idx,
                'position': (x, y, placement_z),
                'utilizations': [b.utilization for b in self.bins],
                'completed_bins': len(self.completed_bins),
            }

        # Truly cannot place anything -- drop item (should be rare)
        dropped = self.buffer.pop(0)
        return {'dropped': dropped}

    def _check_and_close_bins(self):
        """Check all bins against closing policy."""
        for idx in range(2):
            should_close = False

            if self.closing_strategy == 'no_fit':
                should_close = BinClosingPolicy.no_fit_close(
                    self.bins[idx], self.buffer, self._feasibility_fn
                )
            elif self.closing_strategy == 'threshold':
                should_close = BinClosingPolicy.threshold_close(
                    self.bins[idx], threshold=0.80
                )
            elif self.closing_strategy == 'critic' and self.network:
                should_close = BinClosingPolicy.critic_based_close(
                    self.bins[idx], self.network, self.buffer
                )

            if should_close:
                self._close_bin(idx)

    def _close_bin(self, idx: int):
        """Close bin at index and open a new empty bin."""
        old_bin = self.bins[idx]
        old_bin.is_active = False
        self.completed_bins.append(old_bin)

        # Open new empty bin
        self.bins[idx] = BinState(
            height_map=np.zeros((self.L, self.W), dtype=np.int32),
            L=self.L, W=self.W, H=self.H,
        )

    def _feasibility_fn(self, height_map: np.ndarray,
                        l: int, w: int, h: int) -> np.ndarray:
        """Compute feasibility mask (stability + containment + height)."""
        from stability.feasibility_mask_stability import FeasibilityMaskGenerator
        generator = FeasibilityMaskGenerator(self.L, self.W, self.H)
        return generator.compute_mask(height_map, l, w, h)

    def flush(self) -> List[Dict]:
        """Pack all remaining buffer items (end of stream)."""
        results = []
        while self.buffer:
            result = self._pack_one_item()
            results.append(result)
        return results

    def get_statistics(self) -> Dict:
        """Get packing statistics."""
        all_bins = self.completed_bins + [
            b for b in self.bins if b.items_packed > 0
        ]
        utilizations = [b.utilization for b in all_bins]
        return {
            'num_bins_used': len(all_bins),
            'avg_utilization': np.mean(utilizations) if utilizations else 0,
            'max_utilization': np.max(utilizations) if utilizations else 0,
            'min_utilization': np.min(utilizations) if utilizations else 0,
            'total_items_packed': sum(b.items_packed for b in all_bins),
        }


# =============================================================================
# 6. EXAMPLE USAGE
# =============================================================================

def demo_pipeline():
    """
    Demonstrate the full semi-online 2-bounded pipeline.

    This uses greedy selection (no neural network) for illustration.
    For full performance, replace with a trained ConstrainedPackingNetwork.
    """
    L, W, H = 10, 10, 10
    buffer_size = 5
    num_items = 100

    packer = SemiOnlineTwoBoundedPacker(
        L=L, W=W, H=H,
        buffer_size=buffer_size,
        selection_strategy='greedy',
        closing_strategy='no_fit',
    )

    # Simulate item stream
    np.random.seed(42)
    for i in range(num_items):
        dims = np.array([
            np.random.randint(1, L // 2 + 1),
            np.random.randint(1, W // 2 + 1),
            np.random.randint(1, H // 2 + 1),
        ])
        result = packer.receive_item(dims)
        if result and 'position' in result:
            print(f"Item {i}: packed in bin {result['bin_idx']} "
                  f"at {result['position']}, "
                  f"utilizations: {[f'{u:.1%}' for u in result['utilizations']]}")

    # Flush remaining buffer
    packer.flush()

    # Print statistics
    stats = packer.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Bins used: {stats['num_bins_used']}")
    print(f"  Avg utilization: {stats['avg_utilization']:.1%}")
    print(f"  Items packed: {stats['total_items_packed']}")


if __name__ == '__main__':
    demo_pipeline()


# =============================================================================
# 7. EXTENDED: PROGRESSIVE WIDENING MCTS FOR LARGE BUFFERS
# =============================================================================

class ProgressiveWideningMCTS(BufferMCTS):
    """
    MCTS with progressive widening for buffer sizes 7-10.

    Standard MCTS creates ALL children at expansion time. For buffer=10
    with 2 bins and top-5 positions, that is 10*2*5 = 100 children per node.
    This is too many for effective exploration.

    Progressive widening (Coulom 2007, Chaslot et al. 2008) limits the
    number of children based on the visit count:

        max_children(node) = C_pw * N(node)^alpha

    where:
    - C_pw = progressive widening constant (controls branching rate)
    - alpha = widening exponent (typically 0.5)
    - N(node) = visit count of the node

    This means early on, only a few children are explored. As a node
    gets more visits, more children are unlocked. This focuses search
    on promising branches early while ensuring broad exploration later.

    Key parameters for our setting:
    - C_pw = 2.0-4.0 (higher = wider search)
    - alpha = 0.5 (square root growth)
    - For buffer=10: starts with ~2-4 children, grows to ~20 after 100 visits

    This is the recommended MCTS variant for buffer sizes > 5.
    """

    def __init__(self, network, num_simulations: int = 200,
                 exploration_constant: float = 1.41,
                 search_depth: int = 3,
                 position_candidates: int = 3,
                 pw_constant: float = 3.0,
                 pw_alpha: float = 0.5):
        super().__init__(
            network=network,
            num_simulations=num_simulations,
            exploration_constant=exploration_constant,
            search_depth=search_depth,
            position_candidates=position_candidates,
        )
        self.pw_constant = pw_constant
        self.pw_alpha = pw_alpha

    def _expand(self, node: MCTSNode, feasibility_fn) -> MCTSNode:
        """
        Progressive widening expansion.

        Instead of generating ALL children, generate only up to
        max_children = C_pw * N(node)^alpha children.

        Priority for child generation:
        1. Largest items first (hardest to place later)
        2. Both bins (try bin with higher critic value first)
        3. Top positions from actor network
        """
        max_children = int(
            self.pw_constant * (node.visit_count + 1) ** self.pw_alpha
        )

        # If we already have enough children, just return a random one
        if len(node.children) >= max_children:
            if node.children:
                return node.children[np.random.randint(len(node.children))]
            return node

        # Generate candidate actions, sorted by priority
        candidates = []
        for item_idx, item in enumerate(node.remaining_buffer):
            l, w, h = item.dimensions
            volume = l * w * h

            for bin_idx in range(2):
                if not node.bin_states[bin_idx].is_active:
                    continue

                bin_state = node.bin_states[bin_idx]
                mask = feasibility_fn(bin_state.height_map, l, w, h)
                if not np.any(mask):
                    continue

                positions = self._get_top_positions(
                    bin_state, item.dimensions, mask
                )

                for position in positions:
                    # Priority: volume (larger items first)
                    candidates.append({
                        'item_idx': item_idx,
                        'bin_idx': bin_idx,
                        'position': position,
                        'priority': volume,
                    })

        # Sort by priority (largest volume first)
        candidates.sort(key=lambda c: c['priority'], reverse=True)

        # Add children up to the progressive widening limit
        existing_actions = {
            (c.item_idx, c.bin_idx, c.position) for c in node.children
        }

        for cand in candidates:
            if len(node.children) >= max_children:
                break

            action_key = (cand['item_idx'], cand['bin_idx'], cand['position'])
            if action_key in existing_actions:
                continue

            item_idx = cand['item_idx']
            bin_idx = cand['bin_idx']
            position = cand['position']

            item = node.remaining_buffer[item_idx]
            l, w, h = item.dimensions

            # Create child state
            new_bins = [b.copy() for b in node.bin_states]
            x = position % new_bins[bin_idx].L
            y = position // new_bins[bin_idx].L
            region = new_bins[bin_idx].height_map[x:x+l, y:y+w]
            placement_z = np.max(region)
            new_bins[bin_idx].height_map[x:x+l, y:y+w] = placement_z + h
            new_bins[bin_idx].volume_used += l * w * h
            new_bins[bin_idx].items_packed += 1

            new_buffer = [
                it for i, it in enumerate(node.remaining_buffer)
                if i != item_idx
            ]

            child = MCTSNode(
                bin_states=new_bins,
                remaining_buffer=new_buffer,
                item_idx=item_idx,
                bin_idx=bin_idx,
                position=position,
                parent=node,
            )
            node.children.append(child)
            existing_actions.add(action_key)

        if node.children:
            return node.children[np.random.randint(len(node.children))]
        return node


# =============================================================================
# 8. EXTENDED: LEARNED ROLLOUT POLICY FOR MCTS SIMULATION
# =============================================================================

class LearnedRolloutPolicy:
    """
    Use the trained DRL agent as the rollout policy within MCTS simulations.

    The paper uses the pre-trained actor to select actions during MCTS
    simulation (rollout phase). This is much better than random rollout
    because the learned policy already knows good placement positions.

    For our buffer setting, the rollout policy must handle:
    1. Item selection from remaining buffer items
    2. Bin selection between 2 active bins
    3. Position selection within the chosen bin

    Strategy:
    - Item selection: critic-guided (evaluate V(s) for each item-bin pair)
    - Bin selection: same critic evaluation
    - Position selection: actor network with feasibility mask projection

    This rollout is O(buffer_remaining * 2 * forward_pass) per step.
    For buffer=10 and depth=3: ~60 forward passes total. Very fast.
    """

    def __init__(self, network, L: int, W: int, H: int):
        self.network = network
        self.L, self.W, self.H = L, W, H

    def rollout(self, bin_states: List[BinState],
                buffer: List[BufferItem],
                feasibility_fn,
                max_steps: int = 3) -> float:
        """
        Perform a learned rollout from the given state.

        Uses the trained actor-critic to make decisions.
        Returns the accumulated reward + critic value at the end.
        """
        import torch

        bins = [b.copy() for b in bin_states]
        remaining = list(buffer)
        total_reward = 0.0

        for step in range(min(max_steps, len(remaining))):
            if not remaining:
                break

            # Find best (item, bin, position) using critic + actor
            best_value = float('-inf')
            best_action = None

            for item_idx, item in enumerate(remaining):
                l, w, h = item.dimensions
                for bin_idx in range(2):
                    if not bins[bin_idx].is_active:
                        continue

                    mask = feasibility_fn(bins[bin_idx].height_map, l, w, h)
                    if not np.any(mask):
                        continue

                    # Construct observation
                    obs = np.zeros((self.L, self.W, 4), dtype=np.float32)
                    obs[:, :, 0] = bins[bin_idx].height_map / self.H
                    obs[:, :, 1] = l / self.L
                    obs[:, :, 2] = w / self.W
                    obs[:, :, 3] = h / self.H

                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    mask_tensor = torch.FloatTensor(mask).unsqueeze(0)

                    with torch.no_grad():
                        output = self.network(obs_tensor, gt_mask=mask_tensor)
                        value = output['value'].item()
                        action_probs = output['action_probs'].squeeze(0)

                    if value > best_value:
                        best_value = value
                        best_pos = torch.argmax(action_probs).item()
                        best_action = (item_idx, bin_idx, best_pos)

            if best_action is None:
                break

            item_idx, bin_idx, pos = best_action
            item = remaining[item_idx]
            l, w, h = item.dimensions
            x = pos % bins[bin_idx].L
            y = pos // bins[bin_idx].L

            # Place item
            region = bins[bin_idx].height_map[x:x+l, y:y+w]
            pz = int(np.max(region))
            bins[bin_idx].height_map[x:x+l, y:y+w] = pz + h
            bins[bin_idx].volume_used += l * w * h
            bins[bin_idx].items_packed += 1

            reward = 10.0 * (l * w * h) / (self.L * self.W * self.H)
            total_reward += reward
            remaining.pop(item_idx)

        # Terminal: get critic value for remaining potential
        if remaining:
            item = remaining[0]
            l, w, h = item.dimensions
            # Use the fuller bin for the terminal value estimate
            bin_idx = 0 if bins[0].utilization >= bins[1].utilization else 1
            obs = np.zeros((self.L, self.W, 4), dtype=np.float32)
            obs[:, :, 0] = bins[bin_idx].height_map / self.H
            obs[:, :, 1] = l / self.L
            obs[:, :, 2] = w / self.W
            obs[:, :, 3] = h / self.H
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                output = self.network(obs_tensor)
                terminal_value = output['value'].item()
            total_reward += terminal_value

        return total_reward


# =============================================================================
# 9. EXTENDED: BIN REPLACEMENT STRATEGY FOR 2-BOUNDED SPACE
# =============================================================================

class BinReplacementStrategy:
    """
    Strategy for managing the 2-bounded bin space lifecycle.

    In 2-bounded space, we have exactly 2 active bins at all times.
    When we close a bin, we must immediately open a new empty bin.

    Key decisions:
    1. WHEN to close: no-fit, threshold, or critic-based
    2. WHICH bin to close (if both are candidates): the fuller one?
       The one with lower expected future value?
    3. HOW to adapt the buffer strategy when a fresh bin opens

    From Tsang et al. (2024, "A deep RL approach for online and concurrent
    3D bin packing with bin replacement strategies"): bin replacement
    strategies have significant impact on overall performance.

    Strategies from simple to complex:
    1. No-Fit: close when nothing in buffer fits
    2. Age-Based: close the oldest bin when space is tight
    3. Ratio-Based: close when utilization exceeds threshold
    4. Critic-Based: close when critic predicts low remaining value
    5. Learned: train a separate policy to decide closing
    """

    @staticmethod
    def smart_close_decision(bins: List[BinState],
                             buffer: List[BufferItem],
                             feasibility_fn,
                             network=None,
                             utilization_threshold: float = 0.80,
                             min_fit_items: int = 2) -> Optional[int]:
        """
        Smart bin closing decision combining multiple signals.

        Logic:
        1. If a bin has no items that fit from buffer -> close it
        2. If a bin exceeds utilization threshold AND the other bin
           is below 50% -> close the high-utilization bin
           (to maintain one "almost full" and one "fresh" bin)
        3. If both bins are above threshold AND fewer than min_fit_items
           fit in either -> close the fuller one

        Returns:
            Index of bin to close (0 or 1), or None if no close needed
        """
        fit_counts = [0, 0]
        for bin_idx in range(2):
            if not bins[bin_idx].is_active:
                continue
            for item in buffer:
                mask = feasibility_fn(bins[bin_idx].height_map, *item.dimensions)
                if np.any(mask):
                    fit_counts[bin_idx] += 1

        utils = [bins[0].utilization, bins[1].utilization]

        # Rule 1: No-fit close
        for idx in range(2):
            if bins[idx].is_active and fit_counts[idx] == 0:
                return idx

        # Rule 2: One bin very full, other fresh
        for idx in range(2):
            other = 1 - idx
            if (utils[idx] >= utilization_threshold and
                    utils[other] < 0.50 and
                    fit_counts[idx] < min_fit_items):
                return idx

        # Rule 3: Both full, close the fuller one
        if all(u >= utilization_threshold for u in utils):
            if fit_counts[0] < min_fit_items and fit_counts[1] < min_fit_items:
                return 0 if utils[0] >= utils[1] else 1

        return None  # No close needed

    @staticmethod
    def post_replacement_buffer_reeval(bins: List[BinState],
                                       buffer: List[BufferItem],
                                       feasibility_fn) -> dict:
        """
        After a bin replacement, re-evaluate buffer opportunities.

        When a fresh empty bin opens, items that did not fit anywhere
        now have a new opportunity. This method analyzes the new situation.

        Returns:
            dict with analysis of new placement opportunities
        """
        opportunities = {
            'items_that_now_fit': [],
            'items_that_still_dont_fit': [],
            'new_bin_feasible_positions': {},
        }

        for idx, item in enumerate(buffer):
            fits_anywhere = False
            for bin_idx in range(2):
                if not bins[bin_idx].is_active:
                    continue
                mask = feasibility_fn(bins[bin_idx].height_map, *item.dimensions)
                if np.any(mask):
                    fits_anywhere = True
                    # For the NEW bin (empty), count feasible positions
                    if bins[bin_idx].items_packed == 0:
                        opportunities['new_bin_feasible_positions'][idx] = int(
                            np.sum(mask > 0)
                        )
                    break

            if fits_anywhere:
                opportunities['items_that_now_fit'].append(idx)
            else:
                opportunities['items_that_still_dont_fit'].append(idx)

        return opportunities


# =============================================================================
# 10. EXTENDED: PERFORMANCE ESTIMATION MODEL
# =============================================================================

"""
PERFORMANCE ESTIMATION FOR THESIS

Based on the paper's results and our extensions, here is a performance
estimation model for our thesis setup:

Configuration: 2-bounded space, buffer size B, grid LxWxH

Baseline (BPP-1, no buffer, 1 bin, 10x10x10):
    RS:     50.5%  (from Table 3)
    CUT-1:  73.4%  (from Table 3)
    CUT-2:  66.9%  (from Table 3)

With buffer k (from Figure 8b, MCTS):
    k=1:  baseline
    k=2:  +3-5%
    k=3:  +7-10%
    k=5:  +12-15%
    k=8:  +15-18%
    k=10: +17-20% (extrapolated)

With 2 bins (from Table 2, interpolated):
    +2-3% over 1 bin

With item rotation (2 horizontal, from Section Scalability):
    +8-12% on RS
    +3-5% on CUT-1/CUT-2

COMBINED ESTIMATES (CUT-2 dataset, our primary benchmark):

| Setup | Est. Util. | Notes |
|---|---|---|
| BPP-1 baseline | 67% | Paper result |
| + buffer k=5, greedy | 75-78% | Critic-guided selection |
| + buffer k=5, MCTS | 78-82% | Full MCTS search |
| + 2 bins | 80-84% | Critic-based bin selection |
| + rotation | 83-87% | 2 orientations |
| + LBCP stability | 84-88% | Better stability = more positions |

For RS (hardest dataset):
| Setup | Est. Util. | Notes |
|---|---|---|
| BPP-1 baseline | 51% | Paper result |
| + buffer k=5, MCTS | 65-70% | |
| + 2 bins + rotation | 70-78% | |
| + LBCP | 72-80% | |

These estimates should be validated experimentally.
"""


# =============================================================================
# 11. EXTENDED: THESIS EXPERIMENT CONFIGURATION
# =============================================================================

class ExperimentConfig:
    """
    Configuration class for thesis experiments.

    Defines all the experiments to run and their parameters.
    """

    # Main experiments
    EXPERIMENTS = {
        'baseline_bpp1': {
            'buffer_size': 1,
            'num_bins': 1,
            'selection_strategy': 'greedy',
            'closing_strategy': 'no_fit',
            'rotation': False,
            'stability': 'paper_3tier',
            'description': 'Paper baseline (BPP-1, single bin)',
        },
        'buffer5_greedy': {
            'buffer_size': 5,
            'num_bins': 2,
            'selection_strategy': 'greedy',
            'closing_strategy': 'no_fit',
            'rotation': False,
            'stability': 'paper_3tier',
            'description': 'Buffer=5, greedy selection, 2 bins',
        },
        'buffer5_critic': {
            'buffer_size': 5,
            'num_bins': 2,
            'selection_strategy': 'critic',
            'closing_strategy': 'no_fit',
            'rotation': False,
            'stability': 'paper_3tier',
            'description': 'Buffer=5, critic-guided selection, 2 bins',
        },
        'buffer5_mcts': {
            'buffer_size': 5,
            'num_bins': 2,
            'selection_strategy': 'mcts',
            'closing_strategy': 'no_fit',
            'rotation': False,
            'stability': 'paper_3tier',
            'description': 'Buffer=5, MCTS selection, 2 bins',
        },
        'buffer10_mcts': {
            'buffer_size': 10,
            'num_bins': 2,
            'selection_strategy': 'mcts',
            'closing_strategy': 'smart',
            'rotation': False,
            'stability': 'paper_3tier',
            'description': 'Buffer=10, progressive MCTS, 2 bins, smart closing',
        },
        'full_system': {
            'buffer_size': 5,
            'num_bins': 2,
            'selection_strategy': 'mcts',
            'closing_strategy': 'smart',
            'rotation': True,
            'stability': 'lbcp',
            'description': 'Full system: buffer=5, MCTS, 2 bins, rotation, LBCP',
        },
    }

    # Ablation studies
    ABLATIONS = {
        'no_mask': {
            'description': 'Without feasibility mask predictor',
            'modify': {'use_mask_predictor': False},
        },
        'no_constraint': {
            'description': 'Reward shaping instead of CMDP constraint',
            'modify': {'use_constraint': False, 'use_reward_shaping': True},
        },
        'no_mcts': {
            'description': 'Buffer=5 with greedy instead of MCTS',
            'modify': {'selection_strategy': 'greedy'},
        },
        'no_entropy': {
            'description': 'Without feasibility-based entropy loss',
            'modify': {'psi': 0.0},
        },
        'single_bin': {
            'description': 'Buffer=5, MCTS, but only 1 bin',
            'modify': {'num_bins': 1},
        },
    }

    # Datasets to test on
    DATASETS = ['rs', 'cut1', 'cut2']

    # Number of test episodes per configuration
    NUM_TEST_EPISODES = 2000

    # Number of random seeds
    NUM_SEEDS = 5

    @staticmethod
    def total_experiments() -> int:
        """Calculate total number of experiment runs needed."""
        main = len(ExperimentConfig.EXPERIMENTS)
        ablations = len(ExperimentConfig.ABLATIONS)
        datasets = len(ExperimentConfig.DATASETS)
        seeds = ExperimentConfig.NUM_SEEDS
        return (main + ablations) * datasets * seeds
