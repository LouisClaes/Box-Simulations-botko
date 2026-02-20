"""
=============================================================================
CODING IDEAS: Hierarchical Bin Packing Framework
Based on: Lee & Nam (2025) - "A Hierarchical Bin Packing Framework with
    Dual Manipulators via Heuristic Search and Deep Reinforcement Learning"
=============================================================================

TARGET USE CASE:
    - Semi-online 3D bin packing
    - Buffer of 5-10 boxes (items visible on conveyor/staging area)
    - 2-bounded space (k=2): only 2 pallets/bins active at any time
    - Dual objectives: maximize fill rate AND ensure stability
    - Python implementation for thesis

PAPER CONTEXT:
    The paper solves 2D BPP using a hierarchical architecture:
      Low-level: A3C RL agent selects placement positions on a grid
      High-level: DFS-BS tree search selects item order, orientation, repacking
    We adapt this to 3D with stability and 2-bounded space.

=============================================================================
"""

# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================

"""
1.1 Heightmap-Based Bin Representation (3D extension of paper's binary grid)

The paper uses a 2D binary grid B in {0,1}^(W x H). For 3D, we replace this
with a heightmap: a 2D array where each cell stores the maximum occupied height.

Advantages over full 3D voxel grid:
  - O(W*D) memory instead of O(W*D*H)
  - Compatible with 2D CNN architectures (the paper's A3C uses CNN on grid)
  - Captures "where can I place next" information compactly
  - Standard in the 3D online BPP literature (Zhao et al. 2021, Yang et al. 2023)

Limitation: Cannot represent overhangs or cavities. For most box-packing
scenarios, this is acceptable since we assume flat-top stacking.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
from enum import Enum
import copy


@dataclass
class Item:
    """A 3D box item to be packed."""
    id: int
    width: float        # x-dimension
    depth: float        # z-dimension (into the bin)
    height: float       # y-dimension (vertical, stacking direction)
    weight: float = 1.0
    fragility: int = 0  # 0 = not fragile, higher = more fragile
    is_accessible: bool = True   # Can a robot reach this item?
    is_recognized: bool = True   # Has the system seen this item's dimensions?

    def get_orientations(self) -> List[Tuple[float, float, float]]:
        """Return all valid 3D orientations (w, d, h) for this item.
        For vertical-axis rotation only (as in the paper): 2 orientations.
        For full rotation: up to 6 orientations (but we may restrict to 2-3
        for items that must remain upright)."""
        orientations = []
        dims = (self.width, self.depth, self.height)

        # Vertical-axis rotations only (paper approach, most practical)
        # Original: (w, d, h)
        orientations.append((dims[0], dims[1], dims[2]))
        # 90-degree rotation: (d, w, h)
        if dims[0] != dims[1]:  # Skip if square base
            orientations.append((dims[1], dims[0], dims[2]))

        # Optional: allow laying on side (if item is not fragile)
        # This gives up to 6 orientations but may violate stability
        # Uncomment for more flexibility:
        # if self.fragility == 0:
        #     orientations.append((dims[0], dims[2], dims[1]))
        #     orientations.append((dims[2], dims[0], dims[1]))
        #     orientations.append((dims[1], dims[2], dims[0]))
        #     orientations.append((dims[2], dims[1], dims[0]))

        # Remove duplicates
        return list(set(orientations))


@dataclass
class Placement:
    """A specific placement of an item in a bin."""
    item: Item
    x: int              # Grid x-coordinate (left edge)
    z: int              # Grid z-coordinate (front edge)
    y: float            # Height coordinate (bottom of item)
    orientation: Tuple[float, float, float]  # (w, d, h) after rotation


class Bin:
    """A 3D bin represented by a heightmap.

    This extends the paper's 2D binary grid to 3D.
    The paper uses B in {0,1}^(W+2 x H+2) with padding.
    We use heightmap in R^(W x D) where each cell stores max height.
    """

    def __init__(self, width: int, depth: int, max_height: float,
                 grid_resolution: float = 1.0):
        self.width = width      # Number of grid cells in x
        self.depth = depth      # Number of grid cells in z
        self.max_height = max_height
        self.grid_resolution = grid_resolution

        # Core representation: heightmap
        self.heightmap = np.zeros((self.width, self.depth), dtype=np.float32)

        # Packed items tracking (needed for repacking, stability checks)
        self.packed_items: List[Placement] = []

        # For stability analysis: weight map per cell
        self.weight_map = np.zeros((self.width, self.depth), dtype=np.float32)

    def can_place(self, item_w: int, item_d: int, item_h: float,
                  x: int, z: int) -> bool:
        """Check if an item can be placed at position (x, z).

        Corresponds to the paper's feasibility check (Section IV-A, point 2).
        Extended to 3D: checks heightmap instead of binary grid.
        """
        # Boundary check
        if x < 0 or z < 0 or x + item_w > self.width or z + item_d > self.depth:
            return False

        # Height check: item must fit below max height
        base_height = self.heightmap[x:x + item_w, z:z + item_d].max()
        if base_height + item_h > self.max_height:
            return False

        return True

    def get_placement_height(self, item_w: int, item_d: int,
                             x: int, z: int) -> float:
        """Get the height at which an item would rest if placed at (x, z).
        This is the max height in the footprint region."""
        return float(self.heightmap[x:x + item_w, z:z + item_d].max())

    def place_item(self, item: Item, x: int, z: int,
                   orientation: Tuple[float, float, float]) -> Placement:
        """Place an item and update the heightmap.

        Corresponds to the paper's pack(o, phi, x, y) primitive.
        """
        item_w, item_d, item_h = int(orientation[0]), int(orientation[1]), orientation[2]
        base_height = self.get_placement_height(item_w, item_d, x, z)

        # Update heightmap
        self.heightmap[x:x + item_w, z:z + item_d] = base_height + item_h

        # Update weight map
        self.weight_map[x:x + item_w, z:z + item_d] += item.weight

        placement = Placement(item=item, x=x, z=z, y=base_height,
                              orientation=orientation)
        self.packed_items.append(placement)
        return placement

    def unpack_item(self, placement: Placement):
        """Remove an item from the bin.

        Corresponds to the paper's unpack(o) primitive.
        WARNING: In 3D, this is only safe for top-layer items!
        Must check that nothing is stacked on top.
        """
        item_w = int(placement.orientation[0])
        item_d = int(placement.orientation[1])
        item_h = placement.orientation[2]

        # Remove from packed items
        self.packed_items.remove(placement)

        # Recompute heightmap in the affected region
        # (cannot simply subtract -- other items may overlap in footprint)
        self._recompute_heightmap_region(placement.x, placement.z,
                                         item_w, item_d)

        # Update weight map
        self.weight_map[placement.x:placement.x + item_w,
                        placement.z:placement.z + item_d] -= placement.item.weight

    def _recompute_heightmap_region(self, x: int, z: int, w: int, d: int):
        """Recompute heightmap for a region after unpacking."""
        self.heightmap[x:x + w, z:z + d] = 0.0
        for p in self.packed_items:
            pw, pd, ph = int(p.orientation[0]), int(p.orientation[1]), p.orientation[2]
            # Check overlap with region
            ox1 = max(x, p.x)
            oz1 = max(z, p.z)
            ox2 = min(x + w, p.x + pw)
            oz2 = min(z + d, p.z + pd)
            if ox1 < ox2 and oz1 < oz2:
                np.maximum(self.heightmap[ox1:ox2, oz1:oz2],
                           p.y + ph,
                           out=self.heightmap[ox1:ox2, oz1:oz2])

    def get_utilization(self) -> float:
        """Compute volume utilization ratio.

        Paper's metric: sum of item areas / bin area (2D).
        Our 3D version: sum of item volumes / bin volume.
        """
        total_item_volume = sum(
            p.orientation[0] * p.orientation[1] * p.orientation[2]
            for p in self.packed_items
        )
        bin_volume = self.width * self.depth * self.max_height
        return total_item_volume / bin_volume if bin_volume > 0 else 0.0

    def get_state_for_rl(self) -> np.ndarray:
        """Get state representation for the RL agent.

        Paper uses: B (binary grid) concatenated with l_o(phi) (item size).
        3D extension: heightmap (normalized to [0,1]) as a 2D image input to CNN.
        """
        return self.heightmap / self.max_height  # Normalize to [0, 1]

    def compute_adjacency_reward(self, x: int, z: int, item_w: int,
                                 item_d: int) -> float:
        """Compute the adjacency-based reward for a placement.

        This is the paper's core low-level reward (Equation 3):
        r_low = sum of occupied cells adjacent to the placed item's boundary.

        3D adaptation: count cells on the heightmap boundary that are occupied
        at or above the placement height.
        """
        base_height = self.get_placement_height(item_w, item_d, x, z)
        reward = 0.0

        # Left boundary (x - 1)
        if x > 0:
            for dz in range(item_d):
                if self.heightmap[x - 1, z + dz] >= base_height:
                    reward += 1.0

        # Right boundary (x + item_w)
        if x + item_w < self.width:
            for dz in range(item_d):
                if self.heightmap[x + item_w, z + dz] >= base_height:
                    reward += 1.0

        # Front boundary (z - 1)
        if z > 0:
            for dx in range(item_w):
                if self.heightmap[x + dx, z - 1] >= base_height:
                    reward += 1.0

        # Back boundary (z + item_d)
        if z + item_d < self.depth:
            for dx in range(item_w):
                if self.heightmap[x + dx, z + item_d] >= base_height:
                    reward += 1.0

        # Bottom: fraction of base area that is supported
        # (This is our stability-aware extension, not in the original paper)
        footprint = self.heightmap[x:x + item_w, z:z + item_d]
        support_ratio = np.sum(footprint >= base_height - 0.01) / (item_w * item_d)
        reward += support_ratio * item_w * item_d  # Weighted by footprint area

        return reward


# =============================================================================
# 2. STABILITY MODULE (NOT IN PAPER -- OUR EXTENSION)
# =============================================================================

"""
The paper operates in 2D and has no explicit stability modeling.
For our 3D use case, stability is critical. We add:
  1. Support ratio check (static vertical stability)
  2. Center-of-gravity tracking
  3. Stability-aware feasibility masking

These integrate with the paper's feasibility mask approach (Section IV-A, point 2).
"""


class StabilityChecker:
    """Check physical stability of placements.

    Integrates with the paper's feasibility mask:
    - Paper: b_j = 0 if action j violates placement constraints
    - Extension: b_j = 0 if action j also violates stability constraints
    """

    def __init__(self, min_support_ratio: float = 0.8,
                 max_cog_offset_ratio: float = 0.15):
        self.min_support_ratio = min_support_ratio
        self.max_cog_offset_ratio = max_cog_offset_ratio

    def check_support_ratio(self, bin_state: Bin, x: int, z: int,
                            item_w: int, item_d: int) -> float:
        """Compute what fraction of the item's base is supported.

        Full support (1.0): entire base rests on floor or other items.
        Paper's 2D assumption trivializes this (everything is on the floor).
        In 3D, items can overhang if the heightmap is uneven.
        """
        base_height = bin_state.get_placement_height(item_w, item_d, x, z)
        if base_height == 0:
            return 1.0  # On the floor -- fully supported

        footprint = bin_state.heightmap[x:x + item_w, z:z + item_d]
        supported_cells = np.sum(np.abs(footprint - base_height) < 0.01)
        total_cells = item_w * item_d
        return supported_cells / total_cells

    def check_placement_stability(self, bin_state: Bin, x: int, z: int,
                                  item_w: int, item_d: int) -> bool:
        """Check if a placement meets stability requirements.

        Returns True if stable enough (for use in feasibility mask).
        """
        support = self.check_support_ratio(bin_state, x, z, item_w, item_d)
        return support >= self.min_support_ratio

    def compute_stability_reward(self, bin_state: Bin, x: int, z: int,
                                 item_w: int, item_d: int) -> float:
        """Compute a stability-based reward component.

        This can be ADDED to the paper's adjacency reward to create
        a multi-objective reward:
            r_total = alpha * r_adjacency + beta * r_stability
        """
        support = self.check_support_ratio(bin_state, x, z, item_w, item_d)
        return support  # 0.0 to 1.0

    def get_feasibility_mask_3d(self, bin_state: Bin, item_w: int,
                                item_d: int, item_h: float) -> np.ndarray:
        """Generate feasibility mask for all positions on the heightmap.

        Extension of paper's feasibility mask (Section IV-A, point 2):
        - Paper: b_j = 0 if placement violates containment/overlap
        - Ours: b_j = 0 if also violates stability or height constraints

        Returns: 1D array of size W*D + 1 (last entry = no-position action)
        """
        W, D = bin_state.width, bin_state.depth
        mask = np.zeros(W * D + 1, dtype=np.float32)

        for x in range(W):
            for z in range(D):
                idx = x * D + z
                if bin_state.can_place(item_w, item_d, item_h, x, z):
                    if self.check_placement_stability(bin_state, x, z,
                                                      item_w, item_d):
                        mask[idx] = 1.0

        # No-position action: allowed only if no valid placements exist
        if mask[:W * D].sum() == 0:
            mask[W * D] = 1.0

        return mask


# =============================================================================
# 3. LOW-LEVEL RL AGENT (A3C, adapted for 3D)
# =============================================================================

"""
Paper's approach:
    - CNN encodes bin occupancy B
    - Concatenate with item size vector l_o(phi)
    - FC layers output actor (policy) and critic (value)
    - Feasibility mask applied to pre-softmax logits:
        z' = b * z + (1 - b) * (-10^8)
        pi = softmax(z')

3D adaptation:
    - Input: heightmap (W x D) instead of binary grid (W x H)
    - Item representation: 3D size vector (w, d, h) instead of 2D
    - Action space: W * D + 1 positions (heightmap cells + no-position)
    - Feasibility mask includes stability checks

Training (from paper):
    - A3C with 1 global net + 3 workers
    - 7.2M episodes, ~28 hours on RTX 4080 SUPER
    - Reward: adjacency count (equation 3)
    - Loss: L_total = L_actor + L_critic
    - Discount factor gamma_low for future rewards
"""

# PyTorch pseudocode for the A3C network:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionSelectionNetwork(nn.Module):
    '''
    A3C Actor-Critic for position selection.

    Paper reference: Figure 4, Section IV-A.

    Input: heightmap (W x D) + item_size (3,)
    Output: policy pi(a | s) over W*D+1 actions, value V(s)
    '''

    def __init__(self, bin_width, bin_depth, item_dim=3):
        super().__init__()

        # CNN for heightmap encoding (paper uses CNN on binary grid)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Feature dimension after CNN
        cnn_out_dim = 64 * bin_width * bin_depth

        # FC layers after concatenating CNN output + item size
        self.fc1 = nn.Linear(cnn_out_dim + item_dim, 512)
        self.fc2 = nn.Linear(512, 256)

        # Actor head (policy)
        self.actor = nn.Linear(256, bin_width * bin_depth + 1)

        # Critic head (value)
        self.critic = nn.Linear(256, 1)

        self.bin_width = bin_width
        self.bin_depth = bin_depth

    def forward(self, heightmap, item_size, feasibility_mask):
        '''
        Forward pass with feasibility masking.

        Args:
            heightmap: (batch, 1, W, D) - normalized heightmap
            item_size: (batch, 3) - [w, d, h] of item
            feasibility_mask: (batch, W*D+1) - binary mask

        Returns:
            policy: (batch, W*D+1) - action probabilities
            value: (batch, 1) - state value estimate
        '''
        # CNN encoding
        x = F.relu(self.conv1(heightmap))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten

        # Concatenate with item size
        x = torch.cat([x, item_size], dim=1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Actor: logits
        logits = self.actor(x)

        # Apply feasibility mask (paper equation: z' = b*z + (1-b)*(-10^8))
        masked_logits = (feasibility_mask * logits +
                        (1 - feasibility_mask) * (-1e8))

        # Policy
        policy = F.softmax(masked_logits, dim=1)

        # Critic: value estimate
        value = self.critic(x)

        return policy, value
"""


# =============================================================================
# 4. HIGH-LEVEL TREE SEARCH (DFS-BS, adapted for 3D + 2 bins)
# =============================================================================

"""
Paper's DFS-BS algorithm (Algorithm 2) adapted for:
    1. 3D items with multiple orientations
    2. 2-bounded space (2 active bins)
    3. Stability-aware candidate scoring

Key changes from paper:
    - Each tree node now includes bin_index (which of 2 bins)
    - Branching: for each item x orientation x bin = more candidates
    - SELECTION must be more aggressive to manage branching factor
    - Reward includes stability component
"""


@dataclass
class TreeNode:
    """A node in the search tree.

    Paper's vertex v = {B', N', I', C'}.
    Extended with: bin states for both active bins, stability info.
    """
    bin_states: List[Bin]           # State of each active bin (k=2)
    buffer_items: List[Item]        # Items in the buffer (accessible)
    recognized_items: List[Item]    # Items visible but not yet accessible
    packed_per_bin: List[List[Placement]]  # Items packed in each bin
    depth: int = 0
    parent: Optional['TreeNode'] = None


@dataclass
class CandidateAction:
    """A candidate action at a tree node.

    Paper's (o, phi, a_low, d(v')) tuple.
    Extended with bin_index for 2-bounded space.
    """
    item: Item
    orientation: Tuple[float, float, float]
    position: Tuple[int, int]       # (x, z) on heightmap
    bin_index: int                  # Which of the 2 active bins
    reward: float                   # Low-level reward (adjacency + stability)
    depth: int                      # Depth in tree


class HierarchicalTreeSearch:
    """High-level tree search for item sequencing and bin assignment.

    Implements Algorithms 1-3 from the paper, adapted for:
    - 3D items
    - 2-bounded space
    - Stability-aware scoring
    """

    def __init__(self, rl_agent, stability_checker: StabilityChecker,
                 max_beam_width: int = 5, max_depth: int = 10,
                 use_repack: bool = True, time_limit: float = 2.0):
        self.rl_agent = rl_agent
        self.stability_checker = stability_checker
        self.max_beam_width = max_beam_width
        self.max_depth = max_depth
        self.use_repack = use_repack
        self.time_limit = time_limit  # seconds

    def search(self, bin_states: List[Bin], buffer_items: List[Item],
               recognized_items: List[Item],
               require_full_pack: bool = False) -> List[CandidateAction]:
        """Main entry point: Algorithm 1 adapted for 2-bounded space.

        Returns a sequence of CandidateActions (item, orientation, position, bin).
        """
        root = TreeNode(
            bin_states=[copy.deepcopy(b) for b in bin_states],
            buffer_items=list(buffer_items),
            recognized_items=list(recognized_items),
            packed_per_bin=[list(b.packed_items) for b in bin_states]
        )

        # Phase 1: Tree expansion
        candidate_sequences = []
        self._tree_expansion(root, [], candidate_sequences, require_full_pack)

        if not candidate_sequences:
            return []  # No valid placements found

        # Phase 2: Forward simulation and selection
        best_sequence = self._select_best_sequence(candidate_sequences)

        # Phase 3: Repacking (if enabled and initial result is suboptimal)
        if self.use_repack and self._should_repack(best_sequence, bin_states):
            repack_result = self._repack_trial(root, best_sequence, require_full_pack)
            if repack_result is not None:
                best_sequence = repack_result

        return best_sequence

    def _tree_expansion(self, node: TreeNode, current_sequence: List[CandidateAction],
                        all_sequences: List[List[CandidateAction]],
                        require_full_pack: bool):
        """Recursive DFS-BS tree expansion (Algorithm 2).

        Adapted for 2-bounded space: each candidate includes bin_index.
        """
        candidates = []

        # For each accessible item in the buffer
        for item in node.buffer_items:
            if not item.is_accessible:
                continue

            # For each orientation
            for orientation in item.get_orientations():
                w, d, h = int(orientation[0]), int(orientation[1]), orientation[2]

                # For each active bin (2-bounded space extension)
                for bin_idx, bin_state in enumerate(node.bin_states):
                    # Query RL agent for best position
                    position, reward = self._query_rl_agent(
                        bin_state, w, d, h
                    )

                    if position is not None:
                        candidates.append(CandidateAction(
                            item=item,
                            orientation=orientation,
                            position=position,
                            bin_index=bin_idx,
                            reward=reward,
                            depth=node.depth
                        ))

        if not candidates:
            # Leaf node: no valid placements
            if current_sequence:
                all_sequences.append(list(current_sequence))
            return

        # Sort and select (paper: REWARDSORTING + SELECTION)
        candidates.sort(key=lambda c: (-c.reward, -c.item.width * c.item.depth * c.item.height))
        candidates = candidates[:self.max_beam_width]

        # Recurse on each selected candidate
        for candidate in candidates:
            if node.depth >= self.max_depth:
                all_sequences.append(current_sequence + [candidate])
                continue

            # Generate child node
            child = self._generate_child(node, candidate)
            self._tree_expansion(
                child,
                current_sequence + [candidate],
                all_sequences,
                require_full_pack
            )

    def _query_rl_agent(self, bin_state: Bin, item_w: int, item_d: int,
                        item_h: float) -> Tuple[Optional[Tuple[int, int]], float]:
        """Query the RL agent for the best position.

        Paper: a_low = pi_low(s_low) where s_low = {B, l_o(phi)}.
        3D extension: s_low = {heightmap, (w, d, h)}.
        """
        # Get state representation
        heightmap_state = bin_state.get_state_for_rl()

        # Get feasibility mask (geometric + stability)
        mask = self.stability_checker.get_feasibility_mask_3d(
            bin_state, item_w, item_d, item_h
        )

        # RL agent forward pass
        # action_idx = self.rl_agent.select_action(heightmap_state,
        #                                          [item_w, item_d, item_h],
        #                                          mask)
        # For pseudocode, we simulate:
        action_idx = 0  # Placeholder
        W, D = bin_state.width, bin_state.depth

        if action_idx == W * D:
            return None, 0.0  # No-position action

        x = action_idx // D
        z = action_idx % D
        reward = bin_state.compute_adjacency_reward(x, z, item_w, item_d)
        stability_reward = self.stability_checker.compute_stability_reward(
            bin_state, x, z, item_w, item_d
        )

        # Combined reward (our extension)
        alpha = 0.7  # Weight for adjacency (fill rate)
        beta = 0.3   # Weight for stability
        combined_reward = alpha * reward + beta * stability_reward

        return (x, z), combined_reward

    def _generate_child(self, parent: TreeNode,
                        action: CandidateAction) -> TreeNode:
        """Generate a child tree node by applying an action."""
        child = TreeNode(
            bin_states=[copy.deepcopy(b) for b in parent.bin_states],
            buffer_items=[i for i in parent.buffer_items if i.id != action.item.id],
            recognized_items=list(parent.recognized_items),
            packed_per_bin=[list(p) for p in parent.packed_per_bin],
            depth=parent.depth + 1,
            parent=parent
        )

        # Apply the pack action
        placement = child.bin_states[action.bin_index].place_item(
            action.item, action.position[0], action.position[1],
            action.orientation
        )
        child.packed_per_bin[action.bin_index].append(placement)

        return child

    def _select_best_sequence(self, sequences: List[List[CandidateAction]]) -> List[CandidateAction]:
        """Select the best candidate sequence via forward simulation.

        Paper: compute mu(chi_tilde) = cumulative reward, then util(chi_tilde).
        Select highest mu, then highest util, then smallest total depth.
        """
        best_seq = None
        best_score = -float('inf')

        for seq in sequences:
            score = sum(a.reward for a in seq)
            if score > best_score:
                best_score = score
                best_seq = seq

        return best_seq if best_seq else []

    def _should_repack(self, sequence: List[CandidateAction],
                       bin_states: List[Bin]) -> bool:
        """Decide whether repacking should be attempted."""
        if not sequence:
            return True
        # Check if any action has low reward (poor placement)
        avg_reward = sum(a.reward for a in sequence) / len(sequence)
        return avg_reward < 1.0  # Threshold

    def _repack_trial(self, root: TreeNode, current_best: List[CandidateAction],
                      require_full_pack: bool) -> Optional[List[CandidateAction]]:
        """Repacking search (Algorithm 3).

        Try unpacking recently placed items and re-searching.
        In 3D, only top-layer items can be safely unpacked.

        Returns improved sequence or None if no improvement found.
        """
        import time
        start_time = time.time()
        best_result = None
        best_util = self._compute_utilization(root, current_best)

        for bin_idx, bin_state in enumerate(root.bin_states):
            packed = list(bin_state.packed_items)

            # Try unpacking 1, then 2 items (last-packed-first, paper's strategy)
            for num_unpack in range(1, min(4, len(packed) + 1)):
                if time.time() - start_time > self.time_limit:
                    return best_result

                # Get candidates for unpacking (top-layer items only in 3D)
                unpack_candidates = self._get_top_layer_items(bin_state, packed)
                unpack_candidates = unpack_candidates[:num_unpack]

                if not unpack_candidates:
                    continue

                # Clone state and unpack
                trial_node = TreeNode(
                    bin_states=[copy.deepcopy(b) for b in root.bin_states],
                    buffer_items=list(root.buffer_items),
                    recognized_items=list(root.recognized_items),
                    packed_per_bin=[list(p) for p in root.packed_per_bin]
                )

                for placement in unpack_candidates:
                    trial_node.bin_states[bin_idx].unpack_item(placement)
                    # Add unpacked item back to buffer
                    trial_node.buffer_items.append(placement.item)

                # Re-search
                new_sequences = []
                self._tree_expansion(trial_node, [], new_sequences, require_full_pack)

                if new_sequences:
                    new_best = self._select_best_sequence(new_sequences)
                    new_util = self._compute_utilization(trial_node, new_best)
                    if new_util > best_util:
                        best_util = new_util
                        best_result = new_best

        return best_result

    def _get_top_layer_items(self, bin_state: Bin,
                             packed: List[Placement]) -> List[Placement]:
        """Get items that can be safely unpacked (nothing stacked on top).

        This is our 3D extension -- the paper doesn't need this in 2D.
        """
        top_items = []
        for p in reversed(packed):  # Last-packed-first (paper's strategy)
            w, d, h = int(p.orientation[0]), int(p.orientation[1]), p.orientation[2]
            top_height = p.y + h

            # Check if any other item is resting on top
            is_top = True
            for other in packed:
                if other is p:
                    continue
                ow, od = int(other.orientation[0]), int(other.orientation[1])
                # Check footprint overlap
                if (other.x < p.x + w and other.x + ow > p.x and
                        other.z < p.z + d and other.z + od > p.z):
                    if abs(other.y - top_height) < 0.01:
                        is_top = False
                        break
            if is_top:
                top_items.append(p)

        return top_items

    def _compute_utilization(self, node: TreeNode,
                             sequence: List[CandidateAction]) -> float:
        """Compute total utilization across all bins after applying a sequence."""
        total_util = 0.0
        for bin_state in node.bin_states:
            total_util += bin_state.get_utilization()
        return total_util / len(node.bin_states)


# =============================================================================
# 5. 2-BOUNDED SPACE MANAGER (OUR EXTENSION)
# =============================================================================

"""
The paper uses a single bin. We extend to 2-bounded space:
    - 2 bins are active at any time
    - When a bin is "closed" it cannot be reopened
    - The algorithm must decide: which bin to pack into, and when to close a bin

This is identified as Gap 4 in the overview knowledge base (Section 17):
"There is a severe lack of studies on bounded 3D-PPs."
"""


class BinClosingPolicy(Enum):
    """Policies for when to close a bin and open a new one."""
    UTILIZATION_THRESHOLD = "close_when_utilization_above_threshold"
    NO_FIT_ITEMS = "close_when_no_buffer_items_fit"
    COMBINED = "combined_threshold_and_fit"


class TwoBoundedSpaceManager:
    """Manages 2 active bins for semi-online 3D BPP.

    Implements k-bounded space (k=2) as described in Overview Section 5.
    Key property: once a bin is closed, it can NEVER be reopened.
    """

    def __init__(self, bin_width: int, bin_depth: int, max_height: float,
                 closing_policy: BinClosingPolicy = BinClosingPolicy.COMBINED,
                 close_threshold: float = 0.85):
        self.bin_width = bin_width
        self.bin_depth = bin_depth
        self.max_height = max_height
        self.closing_policy = closing_policy
        self.close_threshold = close_threshold

        # Two active bins
        self.active_bins: List[Bin] = [
            Bin(bin_width, bin_depth, max_height),
            Bin(bin_width, bin_depth, max_height)
        ]
        self.closed_bins: List[Bin] = []
        self.total_bins_used = 2

    def should_close_bin(self, bin_idx: int, buffer_items: List[Item]) -> bool:
        """Determine whether to close a bin.

        Policies:
        1. UTILIZATION_THRESHOLD: close if utilization exceeds threshold
        2. NO_FIT_ITEMS: close if no buffer item can fit
        3. COMBINED: close if threshold exceeded AND no items fit
        """
        bin_state = self.active_bins[bin_idx]
        util = bin_state.get_utilization()

        if self.closing_policy == BinClosingPolicy.UTILIZATION_THRESHOLD:
            return util >= self.close_threshold

        elif self.closing_policy == BinClosingPolicy.NO_FIT_ITEMS:
            return not self._any_item_fits(bin_state, buffer_items)

        elif self.closing_policy == BinClosingPolicy.COMBINED:
            return (util >= self.close_threshold and
                    not self._any_item_fits(bin_state, buffer_items))

        return False

    def close_bin(self, bin_idx: int):
        """Close a bin and open a new one."""
        self.closed_bins.append(self.active_bins[bin_idx])
        self.active_bins[bin_idx] = Bin(
            self.bin_width, self.bin_depth, self.max_height
        )
        self.total_bins_used += 1

    def select_bin(self, item: Item, orientation: Tuple[float, float, float],
                   tree_search: HierarchicalTreeSearch) -> int:
        """Select which active bin to place an item in.

        Strategy: evaluate placement quality in both bins, pick the better one.
        This extends the paper's single-bin approach to 2-bounded space.
        """
        scores = []
        w, d, h = int(orientation[0]), int(orientation[1]), orientation[2]

        for bin_idx, bin_state in enumerate(self.active_bins):
            position, reward = tree_search._query_rl_agent(
                bin_state, w, d, h
            )
            if position is not None:
                scores.append((bin_idx, reward))
            else:
                scores.append((bin_idx, -1.0))  # No valid position

        if not scores:
            return 0  # Default to first bin

        # Select bin with highest reward
        scores.sort(key=lambda x: -x[1])
        return scores[0][0]

    def _any_item_fits(self, bin_state: Bin, items: List[Item]) -> bool:
        """Check if any buffer item can fit in the bin."""
        for item in items:
            for orientation in item.get_orientations():
                w, d, h = int(orientation[0]), int(orientation[1]), orientation[2]
                for x in range(bin_state.width - w + 1):
                    for z in range(bin_state.depth - d + 1):
                        if bin_state.can_place(w, d, h, x, z):
                            return True
        return False

    def get_statistics(self) -> Dict:
        """Get performance statistics."""
        all_bins = self.closed_bins + self.active_bins
        utilizations = [b.get_utilization() for b in all_bins]
        return {
            "total_bins_used": self.total_bins_used,
            "mean_utilization": np.mean(utilizations) if utilizations else 0.0,
            "min_utilization": np.min(utilizations) if utilizations else 0.0,
            "max_utilization": np.max(utilizations) if utilizations else 0.0,
            "closed_bins": len(self.closed_bins),
            "active_bins": len(self.active_bins),
        }


# =============================================================================
# 6. BUFFER MANAGER (Semi-Online with 5-10 Items)
# =============================================================================

"""
The paper models a conveyor belt with varying visibility:
    - n_R recognized items (visible dimensions)
    - n_A accessible items per robot (can be picked)

For our use case:
    - Buffer of 5-10 items: these are our "accessible" items
    - All buffer items have known dimensions (recognized)
    - Items arrive as previous items are packed (conveyor advances)
"""


class BufferManager:
    """Manages the semi-online item buffer.

    Maps to the paper's conveyor model (Section III, Section V-C2).
    """

    def __init__(self, max_buffer_size: int = 10):
        self.max_buffer_size = max_buffer_size
        self.buffer: List[Item] = []
        self.arrival_queue: List[Item] = []  # Items not yet visible
        self.total_items_seen = 0

    def add_to_queue(self, items: List[Item]):
        """Add items to the arrival queue (not yet visible)."""
        self.arrival_queue.extend(items)

    def advance_conveyor(self, num_packed: int = 1):
        """Advance conveyor: fill buffer from arrival queue.

        Paper's conveyor logic (equations 7-9) adapted:
        When items are packed, new items become visible.
        """
        spaces = min(num_packed, self.max_buffer_size - len(self.buffer))
        for _ in range(spaces):
            if self.arrival_queue:
                item = self.arrival_queue.pop(0)
                item.is_accessible = True
                item.is_recognized = True
                self.buffer.append(item)
                self.total_items_seen += 1

    def remove_from_buffer(self, item: Item):
        """Remove a packed item from the buffer."""
        self.buffer = [i for i in self.buffer if i.id != item.id]

    def get_accessible_items(self) -> List[Item]:
        """Get all accessible items in the buffer."""
        return [i for i in self.buffer if i.is_accessible]

    def get_buffer_state(self) -> Dict:
        """Get buffer state for decision making."""
        return {
            "buffer_size": len(self.buffer),
            "accessible_count": len(self.get_accessible_items()),
            "remaining_in_queue": len(self.arrival_queue),
            "total_seen": self.total_items_seen,
        }


# =============================================================================
# 7. MAIN LOOP: Putting It All Together
# =============================================================================

"""
This is the main packing loop that combines all components.
It follows the paper's overall framework (Figure 3) adapted for:
    - 3D items
    - 2-bounded space
    - 5-10 item buffer
    - Stability-aware placement
"""


def main_packing_loop(
    items: List[Item],
    bin_width: int,
    bin_depth: int,
    max_height: float,
    buffer_size: int = 10,
    use_repack: bool = True,
    require_stability: bool = True,
    min_support_ratio: float = 0.8,
):
    """Main semi-online packing loop.

    This is the top-level algorithm that would be called in a thesis
    implementation. It integrates all components.
    """
    # Initialize components
    stability_checker = StabilityChecker(min_support_ratio=min_support_ratio)
    space_manager = TwoBoundedSpaceManager(bin_width, bin_depth, max_height)
    buffer_manager = BufferManager(max_buffer_size=buffer_size)
    rl_agent = None  # Would be loaded from trained model

    tree_search = HierarchicalTreeSearch(
        rl_agent=rl_agent,
        stability_checker=stability_checker,
        max_beam_width=5,
        max_depth=buffer_size,  # Search depth = buffer size
        use_repack=use_repack,
        time_limit=2.0
    )

    # Load items into arrival queue
    buffer_manager.add_to_queue(items)

    # Fill initial buffer
    for _ in range(buffer_size):
        buffer_manager.advance_conveyor()

    # Main loop: process items until all are packed or no more items
    step = 0
    while buffer_manager.buffer or buffer_manager.arrival_queue:
        step += 1

        # Get current state
        accessible_items = buffer_manager.get_accessible_items()
        if not accessible_items:
            break

        # Check if any active bin should be closed
        for bin_idx in range(len(space_manager.active_bins)):
            if space_manager.should_close_bin(bin_idx, accessible_items):
                space_manager.close_bin(bin_idx)

        # Run hierarchical search across both active bins
        action_sequence = tree_search.search(
            bin_states=space_manager.active_bins,
            buffer_items=accessible_items,
            recognized_items=[],  # All buffer items are recognized
            require_full_pack=False
        )

        if not action_sequence:
            # No valid placements in either bin -- close the more full bin
            utils = [b.get_utilization() for b in space_manager.active_bins]
            close_idx = int(np.argmax(utils))
            space_manager.close_bin(close_idx)
            continue

        # Execute the best action
        best_action = action_sequence[0]  # Take the first action in sequence
        bin_state = space_manager.active_bins[best_action.bin_index]
        bin_state.place_item(
            best_action.item,
            best_action.position[0],
            best_action.position[1],
            best_action.orientation
        )

        # Remove from buffer and advance conveyor
        buffer_manager.remove_from_buffer(best_action.item)
        buffer_manager.advance_conveyor(num_packed=1)

        # Log progress
        if step % 10 == 0:
            stats = space_manager.get_statistics()
            buf_state = buffer_manager.get_buffer_state()
            print(f"Step {step}: bins_used={stats['total_bins_used']}, "
                  f"mean_util={stats['mean_utilization']:.2%}, "
                  f"buffer={buf_state['buffer_size']}, "
                  f"remaining={buf_state['remaining_in_queue']}")

    # Final statistics
    return space_manager.get_statistics()


# =============================================================================
# 8. COMPLEXITY AND FEASIBILITY ANALYSIS
# =============================================================================

"""
COMPUTATIONAL COMPLEXITY:

Paper's analysis (implicit from algorithms):

1. Low-level RL agent: O(1) per query (single forward pass through CNN + FC)
   - In 3D with heightmap: same O(1) per query
   - CNN: O(W * D * num_filters * kernel_size^2)
   - For 10x10 bin: very fast (~1ms on GPU)

2. Tree expansion (Algorithm 2):
   - Branching factor: |buffer| * |orientations| * |bins|
   - With buffer=10, orientations=2, bins=2: branching = 40
   - After SELECTION with beam_width=5: effective branching = 5
   - Max depth: buffer_size = 10
   - Worst case: 5^10 = ~10M nodes (but pruning reduces this dramatically)
   - Paper reports < 1.3 seconds total planning time for 2D single bin

3. Repacking (Algorithm 3):
   - Tries unpacking 1, 2, ... items
   - For each subset: runs tree expansion
   - Time-bounded (configurable, paper uses 1 second)
   - With time limit: practical regardless of theoretical complexity

4. 2-bounded space extension (our addition):
   - Doubles the per-node computation (2 bins instead of 1)
   - Tree branching increases by factor 2
   - SELECTION beam width may need reduction to compensate

MEMORY:
    - Heightmap per bin: O(W * D) = O(100) for 10x10 grid
    - Tree nodes: each stores a copy of bin state(s)
    - With max_depth=10 and beam_width=5: ~50 nodes active
    - Total: ~50 * 2 * O(W * D) = manageable

TRAINING TIME (A3C agent):
    - Paper: 28 hours on RTX 4080 SUPER for 2D
    - 3D with heightmap: similar architecture, expect ~40-60 hours
    - One-time cost; agent is then deployed without retraining

FEASIBILITY ASSESSMENT:
    - Single-step planning: < 2 seconds (acceptable for conveyor @ 1.7s/item)
    - Total implementation: 16-20 weeks (see summary document)
    - Hardware: any modern GPU (RTX 3060+ sufficient)
    - Framework: PyTorch + custom environment (no Unity needed for planning)
"""


# =============================================================================
# 9. INTEGRATION POINTS WITH OTHER METHODS
# =============================================================================

"""
This framework can be combined with other papers in our reading list:

1. STABILITY INTEGRATION:
   - "Static stability versus packing efficiency" paper:
     Use their support-ratio and CoG formulations in our StabilityChecker
   - "Online 3D Bin Packing with Fast Stability Validation":
     Use their fast validation methods in the feasibility mask

2. MULTI-BIN EXTENSION:
   - "A Deep RL Approach for Online and Concurrent 3D Bin Packing" (Tsang et al.):
     Their multi-bin concurrent packing can inform our 2-bounded space bin selection
   - "Solving Online 3D Multi-Bin Packing" from the gelezen folder:
     Directly relevant for the multi-bin component

3. CONSTRAINED RL:
   - "Online 3D Bin Packing with Constrained DRL" (Zhao et al. 2021):
     Their CMDP formulation can replace/augment the feasibility mask for
     stability constraints
   - Their feasibility predictor could be used instead of the binary mask

4. HEURISTIC ALTERNATIVES FOR LOW-LEVEL:
   - Ha et al. (2017) DBLF + best-match-first:
     Can replace the RL agent entirely for a simpler baseline
   - Verma et al. (2020) stability scoring:
     Can be used as an alternative low-level placement strategy

5. BUFFER MANAGEMENT:
   - "Near-optimal Algorithms for Stochastic Online Bin Packing":
     Their stochastic analysis can inform buffer management strategy
   - "Online Bin Packing with Predictions":
     If item distribution is partially known, predictions can guide
     which items to prioritize from the buffer

6. HYPER-HEURISTIC WRAPPER (Overview Gap 3):
   The tree search's SELECTION function could be replaced by a selective
   hyper-heuristic that learns which placement rule to use based on
   current bin state. This would directly address Gap 3 from the overview.
"""
