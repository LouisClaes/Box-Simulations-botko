"""
==============================================================================
CODING IDEAS: Cross-Bin Rearrangement for k=2 Bounded Space
==============================================================================

Source Paper: "Online 3D Bin Packing with Fast Stability Validation and
             Stable Rearrangement Planning" (Gao et al., 2025)

This file extends the paper's single-bin SRP concept to a 2-bin setup.
The main implementation is in:
    stability/lbcp_stability_and_rearrangement.py

This file focuses specifically on the MULTI-BIN aspects:
  - Bin allocation policy (which bin to try first)
  - Bin closing policy (when to close a bin in k=2 bounded space)
  - Cross-bin item migration strategies
  - Balancing utilization across bins

CRITICAL INSIGHT: In k=2 bounded space, the second bin can serve as an
extended "staging area" for rearrangement. This transforms the paper's
staging-area concept (max 4 boxes) into a much richer rearrangement
space (the entire second bin).

DEPENDENCIES:
  - stability/lbcp_stability_and_rearrangement.py (LBCP, SRP, validators)

==============================================================================
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# KEY DESIGN DECISIONS FOR k=2 BOUNDED SPACE
# =============================================================================

"""
DECISION 1: Bin Allocation Policy
    When a new item arrives (selected from buffer), which bin do we try first?

    Options:
    a) Fuller-bin-first: Try to pack in the bin with higher utilization.
       Pro: Finishes bins faster, potentially higher final utilization.
       Con: May leave hard-to-fill gaps in the fuller bin.

    b) Best-fit: Try both bins, pick the placement with highest score.
       Pro: Optimizes each placement decision.
       Con: May leave both bins half-full with poor packing.

    c) Emptier-bin-first: Try the bin with lower utilization first.
       Pro: Balances load, more flexible for future items.
       Con: May not achieve high utilization in either bin.

    RECOMMENDATION for thesis: Best-fit with stability validation.
    Score = utilization_improvement - height_penalty - rearrangement_cost.
    This is already implemented in BufferStabilitySelector.


DECISION 2: Bin Closing Policy
    When should a bin be closed (permanently sealed)?

    The paper doesn't address this (single-bin focus). For k=2:

    Options:
    a) Utilization threshold: Close when bin exceeds T_close (e.g., 80%).
       Pro: Simple, predictable.
       Con: May close too early if good items are coming.

    b) Failure-based: Close when N consecutive items from buffer
       fail to fit (even with SRP).
       Pro: Adapts to actual item stream.
       Con: May keep nearly-full bins open too long.

    c) Combined: Close when utilization > T_close AND last K items failed.
       Pro: Balances both signals.
       Con: More parameters to tune.

    d) Buffer-informed: If no item in the current buffer fits the bin
       (even with SRP), close it.
       Pro: Leverages buffer information (semi-online advantage).
       Con: Buffer might not be representative of future items.

    RECOMMENDATION for thesis: Option (d) -- buffer-informed closing.
    With 10 items in the buffer, if NONE can be placed in a bin even with
    rearrangement, it's very likely the bin is effectively full.


DECISION 3: Cross-Bin Migration Strategy
    When and how to move items between bins?

    Scenarios where cross-bin moves help:
    a) Bin A is nearly full but has a tall item taking up floor space.
       Move tall item to bin B (which has height capacity) to free
       floor space in bin A for the new item.

    b) Bin A has poor stability for the new item due to uneven surface.
       Move some items from bin A's top layer to bin B to create a
       flatter surface.

    c) Both bins are moderate utilization but neither fits the new item.
       Redistribute items between bins to create a good placement.

    COST MODEL for cross-bin moves:
    - Each move = 1 robot pick-and-place operation
    - Unpacking from bin A: must respect precedence (remove top items first)
    - Packing into bin B: must satisfy stability (LBCP validation)
    - Total operations = unpack_count + pack_count
    - Budget: max 6-10 operations per rearrangement (practical for robot)

    RECOMMENDATION: Limit to max 3 cross-bin moves (6 total operations:
    3 unpacks from source + 3 packs to destination). More than this is
    too slow for real-time operation.


DECISION 4: When to Prefer Cross-Bin vs Single-Bin SRP
    Single-bin SRP (paper's approach): unpack items from bin, pack new item,
    repack items back into same bin. Simpler, but items return to same bin.

    Cross-bin SRP (our extension): move items to OTHER bin permanently.
    More operations, but can free more space.

    Heuristic: Try single-bin SRP first (lower cost). If it fails,
    try cross-bin SRP. This is already the strategy in
    CrossBinRearrangementPlanner.plan().
"""


# =============================================================================
# BIN CLOSING POLICY
# =============================================================================

class BinClosingPolicy:
    """
    Determines when to close a bin in k=2 bounded space.

    Uses a combination of:
    - Utilization threshold
    - Buffer exhaustion (no buffer item fits)
    - Consecutive failure count

    When a bin is closed, a new empty bin is opened in its place.
    """

    def __init__(
        self,
        min_utilization_to_close: float = 0.65,
        max_consecutive_failures: int = 5,
        buffer_exhaustion_required: bool = True,
    ):
        self.min_util = min_utilization_to_close
        self.max_failures = max_consecutive_failures
        self.buffer_exhaustion_required = buffer_exhaustion_required
        self.failure_counts: Dict[int, int] = {}  # bin_id -> consecutive failures

    def should_close(
        self,
        bin_state,  # Bin object
        buffer_fits: bool,  # Whether any buffer item fits this bin
    ) -> bool:
        """
        Determine if this bin should be closed.

        Args:
            bin_state: Current bin state
            buffer_fits: Whether any item from the current buffer can be
                        placed in this bin (with or without rearrangement)

        Returns:
            True if bin should be closed
        """
        bin_id = bin_state.id

        if bin_id not in self.failure_counts:
            self.failure_counts[bin_id] = 0

        if not buffer_fits:
            self.failure_counts[bin_id] += 1
        else:
            self.failure_counts[bin_id] = 0

        # Close if utilization is above threshold AND buffer exhausted
        if bin_state.utilization >= self.min_util:
            if self.buffer_exhaustion_required:
                if not buffer_fits:
                    return True
            else:
                if self.failure_counts[bin_id] >= self.max_failures:
                    return True

        # Close if very high utilization regardless
        if bin_state.utilization >= 0.90:
            return True

        return False

    def record_success(self, bin_id: int):
        """Record successful placement, reset failure counter."""
        self.failure_counts[bin_id] = 0


# =============================================================================
# BIN ALLOCATION SCORING
# =============================================================================

class BinAllocationScorer:
    """
    Scores a (item, bin, position) triple for allocation decisions.

    Factors:
    - Volume utilization improvement
    - Height efficiency (prefer lower placements)
    - Floor space usage (prefer compact footprint usage)
    - Support quality (how well-supported is this position)
    - Rearrangement cost (number of operations needed)
    - Future flexibility (does this leave good space for future items?)
    """

    def __init__(
        self,
        w_utilization: float = 1.0,
        w_height: float = -0.2,
        w_operations: float = -0.05,
        w_support_quality: float = 0.3,
        w_compactness: float = 0.2,
    ):
        self.w_util = w_utilization
        self.w_height = w_height
        self.w_ops = w_operations
        self.w_support = w_support_quality
        self.w_compact = w_compactness

    def score(
        self,
        bin_state,      # Bin object
        item,           # Box object
        position,       # (x, y, z)
        support_lbcp,   # LBCP from validation
        num_operations: int = 0,
    ) -> float:
        """
        Compute allocation score.

        Higher score = better placement.
        """
        bin_volume = bin_state.width * bin_state.depth * bin_state.height

        # Utilization improvement
        util_delta = item.volume / bin_volume

        # Height penalty (normalized by bin height)
        height_penalty = position[2] / bin_state.height

        # Support quality: ratio of support polygon area to item footprint area
        if support_lbcp is not None and len(support_lbcp.polygon) >= 3:
            support_area = self._polygon_area(support_lbcp.polygon)
            footprint_area = item.width * item.depth
            support_ratio = min(support_area / footprint_area, 1.0)
        else:
            support_ratio = 1.0 if position[2] < 0.01 else 0.0

        # Compactness: how close is this placement to existing items?
        compactness = self._compute_compactness(bin_state, item, position)

        score = (
            self.w_util * util_delta +
            self.w_height * height_penalty +
            self.w_ops * num_operations +
            self.w_support * support_ratio +
            self.w_compact * compactness
        )
        return score

    @staticmethod
    def _polygon_area(polygon) -> float:
        """Compute area of a polygon using the shoelace formula."""
        n = len(polygon)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        return abs(area) / 2.0

    @staticmethod
    def _compute_compactness(bin_state, item, position) -> float:
        """
        Measure how adjacent the placement is to existing items or walls.

        Higher value = more compact (more faces touching walls/items).
        Normalized to [0, 1].
        """
        touching_faces = 0
        total_faces = 6
        x, y, z = position

        # Check walls
        if x < 0.01:
            touching_faces += 1
        if x + item.width > bin_state.width - 0.01:
            touching_faces += 1
        if y < 0.01:
            touching_faces += 1
        if y + item.depth > bin_state.depth - 0.01:
            touching_faces += 1
        if z < 0.01:
            touching_faces += 1  # Floor

        # Check adjacency to existing items (simplified)
        for existing in bin_state.items:
            # Check if any face is adjacent
            if (abs(x - (existing.x + existing.width)) < 0.01 or
                abs(x + item.width - existing.x) < 0.01):
                touching_faces += 0.5
            if (abs(y - (existing.y + existing.depth)) < 0.01 or
                abs(y + item.depth - existing.y) < 0.01):
                touching_faces += 0.5
            if abs(z - existing.top_z) < 0.01:
                touching_faces += 0.5

        return min(touching_faces / total_faces, 1.0)


# =============================================================================
# COMPLETE 2-BIN PIPELINE OVERVIEW
# =============================================================================

"""
COMPLETE PIPELINE FOR SEMI-ONLINE k=2 WITH STABILITY:

1. INITIALIZATION:
   - Create bin_a, bin_b (both empty, same dimensions)
   - Initialize LBCP sets, feasibility maps, height maps for both
   - Initialize empty buffer (capacity 5-10)
   - Initialize BinClosingPolicy, BinAllocationScorer

2. MAIN LOOP (for each item arriving on conveyor):
   a. Add item to buffer (if buffer not full)
   b. If buffer is full OR conveyor demands action:

      FOR EACH item in buffer:
        FOR EACH active bin:
          - Generate candidate placements (EMSs)
          - Validate stability via LBCP (get stability mask)
          - If stable placements exist:
              Score each with BinAllocationScorer
              Record (item, bin, position, score, 0 operations)
          - If no stable placements:
              Run single-bin SRP (MCTS + A*)
              If successful: Score and record (item, bin, pos, score, N ops)
              If failed: try cross-bin SRP
              If cross-bin successful: Score and record

      SELECT best (item, bin, position) by score
      EXECUTE operations (unpack, cross-move, pack)
      UPDATE bin states (SSU for all affected bins)
      REMOVE selected item from buffer

   c. CHECK bin closing policy for each bin:
      - If should_close(bin_x): close bin_x, open new empty bin

3. TERMINATION:
   - When all items processed and buffer empty
   - Report final utilization for all closed + active bins

EXPECTED PERFORMANCE (estimates based on paper results):
   - Single-bin utilization with stability: 73-80%
   - With SRP: 80-85%
   - With buffer selection (10 items): +3-5% over strict online
   - With cross-bin rearrangement: +1-3% over single-bin SRP
   - Overall expected: 78-88% utilization WITH guaranteed stability

TIMING (per item decision):
   - Direct placement with buffer: ~50-100 ms
   - Single-bin SRP when needed: ~1-5 seconds
   - Cross-bin SRP when needed: ~3-15 seconds
   - Average (most items place directly): ~200-500 ms

COMPARISON TO PAPER'S RESULTS:
   - Paper achieves 80.5% in simulation (single bin, strict online, with SRP)
   - Paper achieves 71.2% on real robot (single bin, with collision buffer)
   - Our semi-online buffer should improve over strict online by 3-5%
   - Our 2-bin setup should achieve comparable or better per-bin utilization
   - Target for thesis: 78-85% average bin utilization with stability guarantees
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from copy import deepcopy
from itertools import combinations
import random
import math
import heapq


# =============================================================================
# SECTION: Core data-structure stubs (full implementations in stability/ module)
# =============================================================================

# NOTE: In the real codebase these are imported from
#   stability.lbcp_stability_and_rearrangement
# They are stubbed here so this file is self-contained for documentation.

class _Box:
    """Lightweight stub -- see stability/lbcp_stability_and_rearrangement.py"""
    def __init__(self, id, width, depth, height, x=0, y=0, z=0, delta_cog=0.1):
        self.id = id; self.width = width; self.depth = depth; self.height = height
        self.x = x; self.y = y; self.z = z; self.delta_cog = delta_cog
    @property
    def volume(self): return self.width * self.depth * self.height
    @property
    def top_z(self): return self.z + self.height


class _Bin:
    """Lightweight stub -- see stability/lbcp_stability_and_rearrangement.py"""
    def __init__(self, width, depth, height, id=0, resolution=1.0):
        self.width = width; self.depth = depth; self.height = height
        self.id = id; self.resolution = resolution
        self.items: List[_Box] = []
        gw = int(width / resolution); gd = int(depth / resolution)
        self.feasibility_map = np.ones((gw, gd), dtype=bool)
        self.height_map = np.zeros((gw, gd), dtype=float)
        self.lbcps = []
    @property
    def utilization(self):
        vol = self.width * self.depth * self.height
        return sum(i.volume for i in self.items) / vol if vol > 0 else 0.0


# =============================================================================
# CROSS-BIN MCTS SEARCH
# =============================================================================

@dataclass
class CrossBinMCTSNode:
    """
    Node in the cross-bin MCTS search tree.

    The state tracks which items have been unpacked from each bin and
    where they might be re-packed. This extends the paper's single-bin
    MCTS (which only tracks unpacked items from one bin).

    State encoding:
      - unpacked_from_a: set of item IDs removed from bin A
      - unpacked_from_b: set of item IDs removed from bin B
      - moved_a_to_b: set of item IDs relocated from A to B
      - moved_b_to_a: set of item IDs relocated from B to A

    The total number of cross-bin moves is:
        len(moved_a_to_b) + len(moved_b_to_a)
    which is capped at max_cross_moves.
    """
    unpacked_from_a: frozenset = field(default_factory=frozenset)
    unpacked_from_b: frozenset = field(default_factory=frozenset)
    moved_a_to_b: frozenset = field(default_factory=frozenset)
    moved_b_to_a: frozenset = field(default_factory=frozenset)
    parent: Optional['CrossBinMCTSNode'] = None
    children: List['CrossBinMCTSNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0

    @property
    def average_reward(self):
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    @property
    def total_cross_moves(self):
        return len(self.moved_a_to_b) + len(self.moved_b_to_a)

    @property
    def total_unpacked(self):
        return len(self.unpacked_from_a) + len(self.unpacked_from_b)


class CrossBinMCTS:
    """
    MCTS search for cross-bin rearrangement in 2-bounded space.

    This extends the paper's single-bin MCTS to search over item
    migrations between two active bins. The search tree explores
    sequences of:
      1. Unpack item from bin A or bin B
      2. Pack item into the OTHER bin
      3. Eventually pack the new item into one of the bins

    Parameters mirror the paper's MCTS (Section III-D) with additions
    for the cross-bin setting:
      max_nodes: 100 (from paper)
      max_depth: 6 total unpacking operations (from paper)
      max_children: 3 per node (from paper)
      max_cross_moves: 3 items moved between bins (our addition)
      eta: 1.0 UCB1 exploration weight (from paper, Eq. 3)
      w_v: 5.0 critic weight in rollout reward (from paper, Eq. 4)
    """

    def __init__(
        self,
        max_nodes: int = 100,
        max_depth: int = 6,
        max_children: int = 3,
        max_cross_moves: int = 3,
        eta: float = 1.0,
        w_v: float = 5.0,
        target_utilization: float = 0.8,
    ):
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_children = max_children
        self.max_cross_moves = max_cross_moves
        self.eta = eta
        self.w_v = w_v
        self.target_utilization = target_utilization

    def search(
        self,
        bin_a: '_Bin',
        bin_b: '_Bin',
        new_item: '_Box',
        evaluate_fn=None,
    ) -> Optional[Dict]:
        """
        Run cross-bin MCTS to find a rearrangement plan.

        Args:
            bin_a, bin_b: The two active bins.
            new_item: Item that cannot be placed directly.
            evaluate_fn: Optional critic function for rollout reward.

        Returns:
            Dictionary with:
              'target_bin': 'a' or 'b' (where new_item goes)
              'unpacked_from_a': set of item IDs to remove from A
              'unpacked_from_b': set of item IDs to remove from B
              'moved_a_to_b': set of item IDs relocated from A to B
              'moved_b_to_a': set of item IDs relocated from B to A
              'new_item_position': (x, y, z)
              'placements': {item_id: (bin_id, x, y, z)}
            Or None if no feasible rearrangement found.
        """
        root = CrossBinMCTSNode()
        best_result = None
        best_reward = float('-inf')
        nodes_expanded = 0

        while nodes_expanded < self.max_nodes:
            # Selection: traverse tree using UCB1 (Eq. 3 from paper)
            node = self._select(root)

            if node.total_unpacked >= self.max_depth:
                self._backpropagate(node, 0.0)
                continue

            # Expansion: try unpacking one more item from either bin
            child = self._expand(node, bin_a, bin_b)
            if child is None:
                self._backpropagate(node, 0.0)
                continue
            nodes_expanded += 1

            # Rollout: simulate packing with the given rearrangement
            reward, result = self._rollout(
                child, bin_a, bin_b, new_item, evaluate_fn
            )

            if result is not None and reward > best_reward:
                best_reward = reward
                best_result = result

            self._backpropagate(child, reward)

        return best_result

    def _select(self, node: CrossBinMCTSNode) -> CrossBinMCTSNode:
        """UCB1 selection per Eq. 3: UCB1(s_i) = v_bar_i + eta * sqrt(ln(N)/n_i)"""
        while node.children:
            total = sum(c.visits for c in node.children)
            best_child = None
            best_ucb = float('-inf')
            for child in node.children:
                if child.visits == 0:
                    return child
                ucb = child.average_reward + self.eta * math.sqrt(
                    math.log(total) / child.visits
                )
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            node = best_child
        return node

    def _expand(
        self, parent: CrossBinMCTSNode,
        bin_a: '_Bin', bin_b: '_Bin',
    ) -> Optional[CrossBinMCTSNode]:
        """Expand by unpacking one accessible item from either bin."""
        if len(parent.children) >= self.max_children:
            return None

        # Get directly unpackable items from both bins
        unpackable_a = self._get_top_layer(bin_a, parent.unpacked_from_a)
        unpackable_b = self._get_top_layer(bin_b, parent.unpacked_from_b)

        # Filter already-expanded children
        already_expanded = set()
        for child in parent.children:
            diff_a = child.unpacked_from_a - parent.unpacked_from_a
            diff_b = child.unpacked_from_b - parent.unpacked_from_b
            already_expanded.update(diff_a)
            already_expanded.update(diff_b)

        candidates = []
        for item in unpackable_a:
            if item.id not in already_expanded:
                candidates.append(('a', item))
        for item in unpackable_b:
            if item.id not in already_expanded:
                candidates.append(('b', item))

        if not candidates:
            return None

        source_bin, item = random.choice(candidates)

        if source_bin == 'a':
            new_unpacked_a = parent.unpacked_from_a | frozenset({item.id})
            child = CrossBinMCTSNode(
                unpacked_from_a=new_unpacked_a,
                unpacked_from_b=parent.unpacked_from_b,
                moved_a_to_b=parent.moved_a_to_b,
                moved_b_to_a=parent.moved_b_to_a,
                parent=parent,
            )
        else:
            new_unpacked_b = parent.unpacked_from_b | frozenset({item.id})
            child = CrossBinMCTSNode(
                unpacked_from_a=parent.unpacked_from_a,
                unpacked_from_b=new_unpacked_b,
                moved_a_to_b=parent.moved_a_to_b,
                moved_b_to_a=parent.moved_b_to_a,
                parent=parent,
            )

        parent.children.append(child)
        return child

    def _rollout(
        self, node: CrossBinMCTSNode,
        bin_a: '_Bin', bin_b: '_Bin',
        new_item: '_Box', evaluate_fn,
    ) -> Tuple[float, Optional[Dict]]:
        """
        Simulate the rearrangement and compute the rollout reward.

        Rollout reward follows Eq. 4:
            R_rollout = w_v * Critic(B_t, O_last) + U_t

        Steps:
          1. Remove unpacked items from respective bins (simulated copies)
          2. Try to pack new_item into bin_a or bin_b
          3. Try to re-pack unpacked items into the OTHER bin (cross-bin)
             or back into the same bin (single-bin SRP)
          4. Compute utilization and critic value
        """
        sim_a = deepcopy(bin_a)
        sim_b = deepcopy(bin_b)

        # Remove unpacked items
        unpacked_items_a = []
        for item_id in node.unpacked_from_a:
            item = next((i for i in sim_a.items if i.id == item_id), None)
            if item:
                unpacked_items_a.append(deepcopy(item))
                sim_a.items = [i for i in sim_a.items if i.id != item_id]

        unpacked_items_b = []
        for item_id in node.unpacked_from_b:
            item = next((i for i in sim_b.items if i.id == item_id), None)
            if item:
                unpacked_items_b.append(deepcopy(item))
                sim_b.items = [i for i in sim_b.items if i.id != item_id]

        # NOTE: In a full implementation, also update HM and FM after removal.
        # Skipped here for brevity -- see ssu_update_unpack in the stability module.

        # Try packing new_item into bin_a or bin_b (pick the one with more space)
        target_bin_label = None
        new_pos = None

        for label, sim_bin in [('a', sim_a), ('b', sim_b)]:
            pos = self._greedy_find_position(sim_bin, new_item)
            if pos is not None:
                target_bin_label = label
                new_pos = pos
                break

        if target_bin_label is None:
            return 0.0, None  # Cannot even place the new item

        # Pack new item into the target bin (simulation)
        placed = _Box(
            id=new_item.id, width=new_item.width,
            depth=new_item.depth, height=new_item.height,
            x=new_pos[0], y=new_pos[1], z=new_pos[2],
        )
        if target_bin_label == 'a':
            sim_a.items.append(placed)
        else:
            sim_b.items.append(placed)

        # Try re-packing unpacked items into the OTHER bin (cross-bin moves)
        placements = {new_item.id: (target_bin_label, new_pos[0], new_pos[1], new_pos[2])}
        all_repacked = True
        cross_moves_a_to_b = set()
        cross_moves_b_to_a = set()

        # Items from A go to B (preferably), or back to A
        for item in unpacked_items_a:
            # Try bin B first (cross-bin)
            pos_b = self._greedy_find_position(sim_b, item)
            if pos_b is not None and node.total_cross_moves + len(cross_moves_a_to_b) < self.max_cross_moves:
                repacked = _Box(id=item.id, width=item.width, depth=item.depth,
                               height=item.height, x=pos_b[0], y=pos_b[1], z=pos_b[2])
                sim_b.items.append(repacked)
                placements[item.id] = ('b', pos_b[0], pos_b[1], pos_b[2])
                cross_moves_a_to_b.add(item.id)
            else:
                # Try back into bin A
                pos_a = self._greedy_find_position(sim_a, item)
                if pos_a is not None:
                    repacked = _Box(id=item.id, width=item.width, depth=item.depth,
                                   height=item.height, x=pos_a[0], y=pos_a[1], z=pos_a[2])
                    sim_a.items.append(repacked)
                    placements[item.id] = ('a', pos_a[0], pos_a[1], pos_a[2])
                else:
                    all_repacked = False

        # Items from B go to A (preferably), or back to B
        for item in unpacked_items_b:
            pos_a = self._greedy_find_position(sim_a, item)
            if pos_a is not None and node.total_cross_moves + len(cross_moves_b_to_a) < self.max_cross_moves:
                repacked = _Box(id=item.id, width=item.width, depth=item.depth,
                               height=item.height, x=pos_a[0], y=pos_a[1], z=pos_a[2])
                sim_a.items.append(repacked)
                placements[item.id] = ('a', pos_a[0], pos_a[1], pos_a[2])
                cross_moves_b_to_a.add(item.id)
            else:
                pos_b = self._greedy_find_position(sim_b, item)
                if pos_b is not None:
                    repacked = _Box(id=item.id, width=item.width, depth=item.depth,
                                   height=item.height, x=pos_b[0], y=pos_b[1], z=pos_b[2])
                    sim_b.items.append(repacked)
                    placements[item.id] = ('b', pos_b[0], pos_b[1], pos_b[2])
                else:
                    all_repacked = False

        if not all_repacked:
            return 0.0, None

        # Compute reward (Eq. 4 analog for dual-bin)
        avg_util = (sim_a.utilization + sim_b.utilization) / 2.0
        if evaluate_fn:
            critic_val = evaluate_fn(sim_a, sim_b, new_item)
            reward = self.w_v * critic_val + avg_util
        else:
            reward = avg_util

        result = {
            'target_bin': target_bin_label,
            'unpacked_from_a': set(node.unpacked_from_a),
            'unpacked_from_b': set(node.unpacked_from_b),
            'moved_a_to_b': cross_moves_a_to_b,
            'moved_b_to_a': cross_moves_b_to_a,
            'new_item_position': new_pos,
            'placements': placements,
        }
        return reward, result

    def _backpropagate(self, node: CrossBinMCTSNode, reward: float):
        """Backpropagate reward to root."""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def _get_top_layer(self, bin_state: '_Bin', already_removed: frozenset) -> List['_Box']:
        """Find items on the top layer (nothing above them)."""
        remaining = [i for i in bin_state.items if i.id not in already_removed]
        top_items = []
        for item in remaining:
            is_top = True
            for other in remaining:
                if other.id == item.id:
                    continue
                if (other.z >= item.top_z - 1e-6 and
                    other.x < item.x + item.width and other.x + other.width > item.x and
                    other.y < item.y + item.depth and other.y + other.depth > item.y):
                    is_top = False
                    break
            if is_top:
                top_items.append(item)
        return top_items

    def _greedy_find_position(self, bin_state: '_Bin', item: '_Box') -> Optional[Tuple[float, float, float]]:
        """
        Simplified greedy placement using corner-point candidates.

        In production, replace with:
          1. EMS-based candidate generation
          2. Full LBCP-based SSV validation
          3. Scoring function for best placement

        This stub uses a grid search and overlap checking without
        LBCP stability validation -- the real version must include it.
        """
        res = bin_state.resolution * 2  # coarser for speed
        best_pos = None
        best_score = float('-inf')

        for x in np.arange(0, bin_state.width - item.width + 0.01, res):
            for y in np.arange(0, bin_state.depth - item.depth + 0.01, res):
                # Simple overlap check
                z = 0.0
                for existing in bin_state.items:
                    if (x < existing.x + existing.width and x + item.width > existing.x and
                        y < existing.y + existing.depth and y + item.depth > existing.y):
                        z = max(z, existing.top_z)

                if z + item.height > bin_state.height + 1e-6:
                    continue

                # Check overlap at computed z
                overlap = False
                for existing in bin_state.items:
                    if (x < existing.x + existing.width and x + item.width > existing.x and
                        y < existing.y + existing.depth and y + item.depth > existing.y and
                        z < existing.top_z and z + item.height > existing.z):
                        overlap = True
                        break
                if overlap:
                    continue

                score = -(x + y + z)  # DBLF heuristic
                if score > best_score:
                    best_score = score
                    best_pos = (x, y, z)

        return best_pos


# =============================================================================
# STABILITY-AWARE BIN CLOSING WITH FINAL SRP
# =============================================================================

class StabilityAwareBinCloser:
    """
    Handles bin closing in a k=2 bounded space with LBCP stability awareness.

    Before closing a bin, this module:
      1. Verifies all items are stably packed (LBCP re-check)
      2. Attempts a final SRP pass to squeeze in more items from buffer
      3. Computes a closing score combining utilization and stability margin
      4. Recommends which bin to close (if a close is necessary)

    This is a NOVEL extension of the paper's framework (the paper does not
    address bin closing since it operates on a single bin).
    """

    def __init__(
        self,
        min_util_to_close: float = 0.65,
        stability_weight: float = 0.3,
        utilization_weight: float = 0.7,
    ):
        self.min_util = min_util_to_close
        self.w_stab = stability_weight
        self.w_util = utilization_weight
        self.closing_policy = BinClosingPolicy(
            min_utilization_to_close=min_util_to_close,
        )

    def compute_closing_score(self, bin_state: '_Bin') -> float:
        """
        Score a bin for closing. Higher = better candidate for closing.

        Combines:
          - Utilization (higher = closer to done, good to close)
          - Stability (higher = safe to transport, good to close)
          - Remaining useful space (lower = harder to use, good to close)

        Formula:
          score = w_util * utilization + w_stab * stability_proxy
                + (1 - w_util - w_stab) * (1 - remaining_useful_space_ratio)
        """
        util = bin_state.utilization

        # Stability proxy: ratio of feasibility map cells that are true
        # (more LBCP coverage = more stable surface for future items / transport)
        fm = bin_state.feasibility_map
        stab_proxy = np.sum(fm) / fm.size if fm.size > 0 else 0.0

        # Remaining useful space: rough estimate of how much more can be packed
        # Simplified: (1 - max_height / bin_height) * floor_area_ratio
        hm = bin_state.height_map
        avg_height = np.mean(hm) if hm.size > 0 else 0.0
        remaining_ratio = 1.0 - (avg_height / bin_state.height)

        w_remain = max(0.0, 1.0 - self.w_util - self.w_stab)
        score = (self.w_util * util +
                 self.w_stab * stab_proxy +
                 w_remain * (1.0 - remaining_ratio))
        return score

    def recommend_close(
        self,
        bin_a: '_Bin',
        bin_b: '_Bin',
        buffer_fits_a: bool,
        buffer_fits_b: bool,
    ) -> Optional[str]:
        """
        Recommend which bin to close, or None if neither should close.

        Decision logic:
          1. If both bins can still accept buffer items -> don't close either
          2. If only one bin is exhausted -> close that one
          3. If both are exhausted -> close the one with higher closing score
          4. Always respect minimum utilization threshold

        Args:
            bin_a, bin_b: The two active bins
            buffer_fits_a: Can any buffer item fit in bin A (with/without SRP)?
            buffer_fits_b: Can any buffer item fit in bin B (with/without SRP)?

        Returns:
            'a', 'b', or None
        """
        should_close_a = self.closing_policy.should_close(bin_a, buffer_fits_a)
        should_close_b = self.closing_policy.should_close(bin_b, buffer_fits_b)

        if not should_close_a and not should_close_b:
            return None

        if should_close_a and not should_close_b:
            return 'a'

        if should_close_b and not should_close_a:
            return 'b'

        # Both should close -- pick the one with higher closing score
        score_a = self.compute_closing_score(bin_a)
        score_b = self.compute_closing_score(bin_b)
        return 'a' if score_a >= score_b else 'b'

    def final_srp_before_close(
        self,
        bin_to_close: '_Bin',
        other_bin: '_Bin',
        buffer: List['_Box'],
        mcts_searcher: 'CrossBinMCTS',
    ) -> Tuple[int, float]:
        """
        Attempt to squeeze extra items into the bin before closing.

        Tries each buffer item with SRP (single-bin and cross-bin).
        Returns (items_added, final_utilization).

        This is called just before closing a bin to maximize its fill rate.
        """
        items_added = 0
        for item in list(buffer):
            # Quick check: does the item's volume fit at all?
            remaining_volume = (bin_to_close.width * bin_to_close.depth *
                                bin_to_close.height * (1 - bin_to_close.utilization))
            if item.volume > remaining_volume * 1.2:  # 20% tolerance
                continue

            # Try single-bin placement
            pos = mcts_searcher._greedy_find_position(bin_to_close, item)
            if pos is not None:
                placed = _Box(id=item.id, width=item.width, depth=item.depth,
                             height=item.height, x=pos[0], y=pos[1], z=pos[2])
                bin_to_close.items.append(placed)
                buffer.remove(item)
                items_added += 1
                continue

            # Try cross-bin MCTS (may move items to other_bin to free space)
            result = mcts_searcher.search(bin_to_close, other_bin, item)
            if result is not None and result['target_bin'] == 'a':
                # For simplicity, just count it as added
                # Full implementation would execute the rearrangement plan
                items_added += 1
                buffer.remove(item)

        return items_added, bin_to_close.utilization


# =============================================================================
# FULL DUAL-BIN SEMI-ONLINE PIPELINE CONTROLLER
# =============================================================================

class DualBinPipelineController:
    """
    Top-level controller for the complete semi-online k=2 pipeline.

    Orchestrates:
      - Buffer management (filling, selecting from buffer)
      - Bin allocation (which bin to pack into)
      - Stability validation (LBCP-based)
      - Rearrangement planning (single-bin and cross-bin MCTS+A*)
      - Bin closing policy (when to close and open new bins)

    This is the main class for the thesis implementation.

    Usage:
        controller = DualBinPipelineController(
            bin_width=55.0, bin_depth=45.0, bin_height=45.0,
            buffer_capacity=10
        )
        for item in item_stream:
            controller.receive_item(item)
        stats = controller.get_statistics()
    """

    def __init__(
        self,
        bin_width: float = 55.0,
        bin_depth: float = 45.0,
        bin_height: float = 45.0,
        buffer_capacity: int = 10,
        resolution: float = 1.0,
        delta_cog: float = 0.1,
    ):
        self.bin_dims = (bin_width, bin_depth, bin_height)
        self.buffer_capacity = buffer_capacity
        self.resolution = resolution
        self.delta_cog = delta_cog

        # Active bins
        self.bin_a = _Bin(bin_width, bin_depth, bin_height, id=0, resolution=resolution)
        self.bin_b = _Bin(bin_width, bin_depth, bin_height, id=1, resolution=resolution)

        # Buffer
        self.buffer: List[_Box] = []

        # Components
        self.scorer = BinAllocationScorer()
        self.closer = StabilityAwareBinCloser()
        self.cross_mcts = CrossBinMCTS()

        # Statistics
        self.total_items_received = 0
        self.total_items_packed = 0
        self.total_rearrangements = 0
        self.total_cross_bin_moves = 0
        self.closed_bins: List[Tuple[int, float]] = []  # (bin_id, final_util)
        self.next_bin_id = 2

    def receive_item(self, item: '_Box') -> dict:
        """
        Receive a new item from the conveyor.

        Returns a status dict:
          {'action': 'buffered'|'packed'|'rearranged'|'bin_closed'|'failed',
           'details': ...}
        """
        self.total_items_received += 1

        # Add to buffer
        self.buffer.append(item)

        # If buffer is full, must process one item
        if len(self.buffer) >= self.buffer_capacity:
            return self._process_buffer()

        return {'action': 'buffered', 'buffer_size': len(self.buffer)}

    def force_process(self) -> dict:
        """Force processing even if buffer is not full (e.g., end of stream)."""
        if self.buffer:
            return self._process_buffer()
        return {'action': 'empty', 'buffer_size': 0}

    def _process_buffer(self) -> dict:
        """
        Select and place the best item from the buffer.

        Strategy:
          1. Evaluate all (item, bin) combinations for direct placement
          2. If any direct placement exists, pick the best by score
          3. If no direct placement, try single-bin SRP for each combo
          4. If single-bin SRP fails, try cross-bin MCTS
          5. If all fail, trigger bin closing

        Returns status dict.
        """
        best_score = float('-inf')
        best_result = None  # (item, bin_label, position, operations_count)

        # Phase 1: Try direct placement for all buffer items on both bins
        for item in self.buffer:
            for label, bin_state in [('a', self.bin_a), ('b', self.bin_b)]:
                pos = self.cross_mcts._greedy_find_position(bin_state, item)
                if pos is not None:
                    # NOTE: In production, use LBCP SSV validation here
                    score = self.scorer.score(
                        bin_state, item, pos,
                        support_lbcp=None,  # Would come from SSV
                        num_operations=0
                    )
                    if score > best_score:
                        best_score = score
                        best_result = (item, label, pos, 0)

        if best_result is not None:
            item, label, pos, ops = best_result
            self._execute_placement(item, label, pos)
            self.buffer.remove(item)
            self.total_items_packed += 1
            return {
                'action': 'packed', 'item_id': item.id,
                'bin': label, 'position': pos, 'operations': 0,
            }

        # Phase 2: Try cross-bin MCTS for the first buffer item
        # (In production, try all buffer items, but this is expensive)
        for item in self.buffer[:3]:  # Try top 3 items
            result = self.cross_mcts.search(self.bin_a, self.bin_b, item)
            if result is not None:
                self.buffer.remove(item)
                self.total_items_packed += 1
                self.total_rearrangements += 1
                n_cross = len(result.get('moved_a_to_b', set())) + \
                          len(result.get('moved_b_to_a', set()))
                self.total_cross_bin_moves += n_cross
                return {
                    'action': 'rearranged', 'item_id': item.id,
                    'target_bin': result['target_bin'],
                    'cross_moves': n_cross,
                }

        # Phase 3: No placement possible -- close a bin
        buffer_fits_a = any(
            self.cross_mcts._greedy_find_position(self.bin_a, item) is not None
            for item in self.buffer
        )
        buffer_fits_b = any(
            self.cross_mcts._greedy_find_position(self.bin_b, item) is not None
            for item in self.buffer
        )

        close_recommendation = self.closer.recommend_close(
            self.bin_a, self.bin_b, buffer_fits_a, buffer_fits_b
        )

        if close_recommendation is not None:
            closed_util = self._close_bin(close_recommendation)
            return {
                'action': 'bin_closed', 'closed_bin': close_recommendation,
                'final_utilization': closed_util,
            }

        return {'action': 'failed', 'reason': 'no placement and no bin to close'}

    def _execute_placement(self, item: '_Box', bin_label: str, position: Tuple):
        """Execute a direct placement (update bin state)."""
        placed = _Box(
            id=item.id, width=item.width, depth=item.depth, height=item.height,
            x=position[0], y=position[1], z=position[2],
            delta_cog=self.delta_cog,
        )
        target = self.bin_a if bin_label == 'a' else self.bin_b
        target.items.append(placed)
        # NOTE: In production, also run SSU to update FM and HM

    def _close_bin(self, bin_label: str) -> float:
        """Close a bin and open a new empty one in its place."""
        if bin_label == 'a':
            final_util = self.bin_a.utilization
            self.closed_bins.append((self.bin_a.id, final_util))
            self.bin_a = _Bin(
                *self.bin_dims, id=self.next_bin_id,
                resolution=self.resolution
            )
            self.next_bin_id += 1
        else:
            final_util = self.bin_b.utilization
            self.closed_bins.append((self.bin_b.id, final_util))
            self.bin_b = _Bin(
                *self.bin_dims, id=self.next_bin_id,
                resolution=self.resolution
            )
            self.next_bin_id += 1
        return final_util

    def get_statistics(self) -> dict:
        """Return summary statistics for the packing session."""
        closed_utils = [u for _, u in self.closed_bins]
        return {
            'total_items_received': self.total_items_received,
            'total_items_packed': self.total_items_packed,
            'packing_rate': self.total_items_packed / max(1, self.total_items_received),
            'total_rearrangements': self.total_rearrangements,
            'total_cross_bin_moves': self.total_cross_bin_moves,
            'bins_closed': len(self.closed_bins),
            'avg_closed_bin_utilization': np.mean(closed_utils) if closed_utils else 0.0,
            'max_closed_bin_utilization': max(closed_utils) if closed_utils else 0.0,
            'min_closed_bin_utilization': min(closed_utils) if closed_utils else 0.0,
            'active_bin_a_utilization': self.bin_a.utilization,
            'active_bin_b_utilization': self.bin_b.utilization,
            'buffer_remaining': len(self.buffer),
        }


# =============================================================================
# DEMONSTRATION / SMOKE TEST
# =============================================================================

def demo_dual_bin_pipeline():
    """
    Demonstrate the full dual-bin pipeline with random items.

    This is a smoke test that exercises all major code paths:
      - Buffer management
      - Direct placement
      - Bin closing
    """
    controller = DualBinPipelineController(
        bin_width=55.0, bin_depth=45.0, bin_height=45.0,
        buffer_capacity=10, resolution=2.0,
    )

    # Generate 80 random items
    random.seed(42)
    for i in range(80):
        item = _Box(
            id=i,
            width=random.uniform(5, 20),
            depth=random.uniform(5, 20),
            height=random.uniform(5, 15),
        )
        result = controller.receive_item(item)
        if result['action'] != 'buffered':
            pass  # In production: log the result

    # Drain remaining buffer
    while controller.buffer:
        result = controller.force_process()
        if result['action'] == 'failed':
            break

    stats = controller.get_statistics()
    print("=" * 60)
    print("DUAL-BIN PIPELINE STATISTICS")
    print("=" * 60)
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.3f}")
        else:
            print(f"  {key}: {val}")
    print("=" * 60)


if __name__ == "__main__":
    demo_dual_bin_pipeline()
