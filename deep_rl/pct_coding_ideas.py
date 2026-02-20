"""
PCT-Based Deep RL for 3D Bin Packing -- Coding Ideas
=====================================================

Based on: "Deliberate Planning of 3D Bin Packing on Packing Configuration Trees"
Authors: Zhao, Xu, Yu, Hu, Zhu, Du, Xu (2025)
Paper: ICLR 2022 (conference) + SAGE Int. J. Robotics Research 2025 (extended journal)
Code: https://github.com/alexfrom0815/Online-3D-BPP-PCT (~250 stars)

This file contains concrete algorithm pseudocode, data structures, and implementation
guidance for building a PCT-based 3D bin packing system.

TARGET USE CASE:
- Semi-online with buffer of 5-10 boxes
- 2-bounded space (2 active pallets/bins)
- Maximize fill rate AND stability
- Python + PyTorch implementation
- Thesis project

ESTIMATED IMPLEMENTATION EFFORT (8-12 weeks total):
Phase 1: Core PCT data structures + EMS         -- 1 week
Phase 2: GAT + Pointer network (PyTorch)         -- 1 week
Phase 3: ACKTR/PPO training loop                 -- 1-2 weeks
Phase 4: Validation (reproduce paper ~70-76%)    -- 1 week
Phase 5: ToP MCTS planner with buffer s=5-10    -- 1-2 weeks
Phase 6: 2-bounded space extension               -- 1-2 weeks
Phase 7: Stability integration                   -- 1 week
Phase 8: Evaluation + experiments                -- 1-2 weeks

PAPER KEY RESULTS (for reference):
- Online, discrete, Setting 1 (stability): 75.8% utilization (PCT & EMS)
- Online, discrete, Setting 2 (no stability): 86.0% utilization (PCT & EMS)
- Buffering s=10: 93.5% (ToP, Setting 2)
- Buffering s=5: 88.3% (ToP, Setting 2)
- Offline (|I|=50): 95.2% (ToP, outperforms Gurobi)
- Real robot: 57.4% (large boxes), 0 collapses, 100% transport stability

EXPECTED THESIS PERFORMANCE:
- Buffer=10, 2-bounded, stability: 78-87% fill rate
- Buffer=5, 2-bounded, stability: 73-82% fill rate
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import heapq


# =============================================================================
# SECTION 1: CORE DATA STRUCTURES
# =============================================================================

class NodeType(Enum):
    INTERNAL = "internal"   # Packed item
    LEAF = "leaf"           # Candidate placement
    CURRENT = "current"     # Current item to be placed
    DUMMY = "dummy"         # Padding for batch computation


@dataclass
class Box:
    """Represents a physical box/item."""
    width: float    # x dimension
    depth: float    # y dimension
    height: float   # z dimension
    weight: float = 1.0
    density: float = 1.0
    category: int = 0
    box_id: int = -1

    @property
    def volume(self) -> float:
        return self.width * self.depth * self.height

    @property
    def size(self) -> np.ndarray:
        return np.array([self.width, self.depth, self.height])

    def get_orientations(self, num_orientations: int = 2) -> List[np.ndarray]:
        """
        Return possible orientations.
        num_orientations=2: only horizontal rotations (for top-down robot placement)
        num_orientations=6: all axis-aligned rotations
        """
        w, d, h = self.width, self.depth, self.height
        if num_orientations == 2:
            return [
                np.array([w, d, h]),
                np.array([d, w, h]),
            ]
        elif num_orientations == 6:
            return [
                np.array([w, d, h]),
                np.array([d, w, h]),
                np.array([w, h, d]),
                np.array([h, w, d]),
                np.array([d, h, w]),
                np.array([h, d, w]),
            ]
        else:
            raise ValueError(f"Unsupported num_orientations: {num_orientations}")


@dataclass
class Placement:
    """A candidate placement = position + oriented size."""
    position: np.ndarray    # FLB corner (p_x, p_y, p_z)
    size: np.ndarray        # (s_x, s_y, s_z) in this orientation
    orientation_idx: int
    node_id: int = -1
    parent_ems_id: int = -1  # Which EMS generated this candidate

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    @property
    def flb(self) -> np.ndarray:
        """Front-Left-Bottom corner."""
        return self.position

    @property
    def brt(self) -> np.ndarray:
        """Back-Right-Top corner."""
        return self.position + self.size

    def intersects(self, other: 'Placement') -> bool:
        """Check if two placements overlap."""
        for d in range(3):
            if (self.position[d] >= other.position[d] + other.size[d] or
                other.position[d] >= self.position[d] + self.size[d]):
                return False
        return True

    def is_inside_bin(self, bin_size: np.ndarray) -> bool:
        """Check containment constraint."""
        return np.all(self.position >= 0) and np.all(self.brt <= bin_size)


@dataclass
class EMS:
    """
    Empty Maximal Space -- the key space management primitive.

    An EMS is the largest empty rectangular parallelepiped within the bin.
    Represented by its FLB corner and size.

    Reference: Ha et al. (2017), Parreño et al. (2008)
    """
    position: np.ndarray    # FLB corner (min coordinates)
    size: np.ndarray        # (width, depth, height)
    ems_id: int = -1

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    @property
    def flb(self) -> np.ndarray:
        return self.position

    @property
    def brt(self) -> np.ndarray:
        return self.position + self.size

    def can_contain(self, item_size: np.ndarray) -> bool:
        """Check if an item of given size fits in this EMS."""
        return np.all(item_size <= self.size + 1e-9)

    def generate_candidates(self, item: Box, num_orientations: int = 2) -> List[Placement]:
        """
        Generate candidate placements for item within this EMS.

        For EMS expansion, candidates are placed at the FLB corner of the EMS
        and at the four FLB-adjacent corners of the EMS.

        In the PCT paper, candidates = FLB corners after EMS splitting.
        """
        candidates = []
        for o_idx, oriented_size in enumerate(item.get_orientations(num_orientations)):
            if self.can_contain(oriented_size):
                # Primary candidate: FLB corner of EMS
                candidates.append(Placement(
                    position=self.position.copy(),
                    size=oriented_size,
                    orientation_idx=o_idx,
                    parent_ems_id=self.ems_id
                ))
                # Additional candidates from EMS corner variants:
                # Left-up, right-up, left-bottom, right-bottom of the EMS
                # (See paper Appendix B, Figure 18(c))
                for corner_fn in [self._left_up, self._right_up,
                                  self._left_bottom, self._right_bottom]:
                    pos = corner_fn(oriented_size)
                    if pos is not None:
                        candidates.append(Placement(
                            position=pos,
                            size=oriented_size,
                            orientation_idx=o_idx,
                            parent_ems_id=self.ems_id
                        ))
        return candidates

    def _left_up(self, item_size: np.ndarray) -> Optional[np.ndarray]:
        pos = np.array([
            self.position[0],
            self.position[1] + self.size[1] - item_size[1],
            self.position[2]
        ])
        if np.all(pos >= 0):
            return pos
        return None

    def _right_up(self, item_size: np.ndarray) -> Optional[np.ndarray]:
        pos = np.array([
            self.position[0] + self.size[0] - item_size[0],
            self.position[1] + self.size[1] - item_size[1],
            self.position[2]
        ])
        if np.all(pos >= 0):
            return pos
        return None

    def _left_bottom(self, item_size: np.ndarray) -> Optional[np.ndarray]:
        # Same as FLB corner -- duplicate, skip in practice
        pos = np.array([
            self.position[0],
            self.position[1],
            self.position[2]
        ])
        return None  # Same as FLB, skip to avoid duplicate

    def _right_bottom(self, item_size: np.ndarray) -> Optional[np.ndarray]:
        pos = np.array([
            self.position[0] + self.size[0] - item_size[0],
            self.position[1],
            self.position[2]
        ])
        if pos[0] >= 0:
            return pos
        return None


class EMSManager:
    """
    Manages Empty Maximal Spaces for a single bin.

    When an item is placed, existing EMSs that overlap with the item are split
    into up to 3 new smaller EMSs (along each axis). EMSs fully covered by the
    item are removed.

    Complexity: O(|E|) per item placement, where |E| is the number of EMSs.

    Reference: Ha et al. (2017), Section 3.1 and Appendix B of the PCT paper.
    """

    def __init__(self, bin_size: np.ndarray):
        self.bin_size = bin_size
        self.ems_list: List[EMS] = []
        self.next_ems_id = 0
        # Initialize with the entire bin as one EMS
        self._add_ems(np.zeros(3), bin_size.copy())

    def _add_ems(self, position: np.ndarray, size: np.ndarray) -> int:
        """Add a new EMS if it has positive volume."""
        if np.all(size > 1e-9):
            ems = EMS(position=position, size=size, ems_id=self.next_ems_id)
            self.next_ems_id += 1
            self.ems_list.append(ems)
            return ems.ems_id
        return -1

    def place_item(self, placement: Placement) -> List[EMS]:
        """
        Update EMS list after placing an item.

        For each EMS that intersects the placed item:
        - Remove the original EMS
        - Create up to 3 new EMSs by splitting along each axis

        Returns: list of newly created EMSs
        """
        new_ems_list = []
        created_ems = []

        for ems in self.ems_list:
            if self._intersects(ems, placement):
                # Split along each axis
                splits = self._split_ems(ems, placement)
                for pos, size in splits:
                    ems_id = self._add_ems(pos, size)
                    if ems_id >= 0:
                        created_ems.append(self.ems_list[-1])
            else:
                new_ems_list.append(ems)

        # Add newly created EMSs
        new_ems_list.extend(created_ems)

        # Remove EMSs that are subsets of other EMSs (optional optimization)
        self.ems_list = self._remove_subsets(new_ems_list)

        return created_ems

    def _intersects(self, ems: EMS, placement: Placement) -> bool:
        """Check if EMS and placement overlap."""
        for d in range(3):
            if (ems.position[d] >= placement.position[d] + placement.size[d] or
                placement.position[d] >= ems.position[d] + ems.size[d]):
                return False
        return True

    def _split_ems(self, ems: EMS, placement: Placement) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Split an EMS along 3 axes around a placed item.
        Returns list of (position, size) tuples for new EMSs.

        This implements the EMS splitting algorithm from PCT paper Appendix B:
        When item n_t is placed within EMS e, up to 3 new EMSs are created:
          1. (p_e^x + s_n^x, p_e^y, p_e^z), size (s_e^x - s_n^x, s_e^y, s_e^z)
          2. (p_e^x, p_e^y + s_n^y, p_e^z), size (s_e^x, s_e^y - s_n^y, s_e^z)
          3. (p_e^x, p_e^y, p_e^z + s_n^z), size (s_e^x, s_e^y, s_e^z - s_n^z)

        If n_t only PARTIALLY overlaps an existing EMS, a similar volume
        subtraction is applied. The code below handles both cases:
        full overlap (item placed on FLB of EMS) and partial overlap.

        Reference: Ha et al. (2017), Section 3.1 and Appendix B of PCT paper.
        """
        splits = []

        # Split along x-axis (width)
        # Part before the item
        if placement.position[0] > ems.position[0]:
            new_size = ems.size.copy()
            new_size[0] = placement.position[0] - ems.position[0]
            splits.append((ems.position.copy(), new_size))

        # Part after the item
        item_end_x = placement.position[0] + placement.size[0]
        ems_end_x = ems.position[0] + ems.size[0]
        if item_end_x < ems_end_x:
            new_pos = ems.position.copy()
            new_pos[0] = item_end_x
            new_size = ems.size.copy()
            new_size[0] = ems_end_x - item_end_x
            splits.append((new_pos, new_size))

        # Split along y-axis (depth)
        if placement.position[1] > ems.position[1]:
            new_size = ems.size.copy()
            new_size[1] = placement.position[1] - ems.position[1]
            splits.append((ems.position.copy(), new_size))

        item_end_y = placement.position[1] + placement.size[1]
        ems_end_y = ems.position[1] + ems.size[1]
        if item_end_y < ems_end_y:
            new_pos = ems.position.copy()
            new_pos[1] = item_end_y
            new_size = ems.size.copy()
            new_size[1] = ems_end_y - item_end_y
            splits.append((new_pos, new_size))

        # Split along z-axis (height)
        if placement.position[2] > ems.position[2]:
            new_size = ems.size.copy()
            new_size[2] = placement.position[2] - ems.position[2]
            splits.append((ems.position.copy(), new_size))

        item_end_z = placement.position[2] + placement.size[2]
        ems_end_z = ems.position[2] + ems.size[2]
        if item_end_z < ems_end_z:
            new_pos = ems.position.copy()
            new_pos[2] = item_end_z
            new_size = ems.size.copy()
            new_size[2] = ems_end_z - item_end_z
            splits.append((new_pos, new_size))

        return splits

    def _remove_subsets(self, ems_list: List[EMS]) -> List[EMS]:
        """Remove EMSs that are completely contained within other EMSs."""
        # Simple O(n^2) approach -- sufficient for typical EMS counts
        result = []
        for i, ems_i in enumerate(ems_list):
            is_subset = False
            for j, ems_j in enumerate(ems_list):
                if i != j and self._is_subset(ems_i, ems_j):
                    is_subset = True
                    break
            if not is_subset:
                result.append(ems_i)
        return result

    def _is_subset(self, inner: EMS, outer: EMS) -> bool:
        """Check if inner EMS is completely contained in outer EMS."""
        return (np.all(inner.position >= outer.position - 1e-9) and
                np.all(inner.brt <= outer.brt + 1e-9) and
                inner.ems_id != outer.ems_id)

    def get_all_candidates(self, item: Box, num_orientations: int = 2) -> List[Placement]:
        """Generate all candidate placements across all EMSs."""
        candidates = []
        for ems in self.ems_list:
            candidates.extend(ems.generate_candidates(item, num_orientations))
        return candidates


class PackingConfigurationTree:
    """
    The Packing Configuration Tree (PCT) -- core data structure.

    A dynamically growing tree where:
    - Internal nodes (B_t) represent packed items with their spatial configurations
    - Leaf nodes (L_t) represent candidate placements for the current item
    - The current item node (n_t) is appended

    The PCT serves as both the STATE and ACTION SPACE for the RL agent.

    Key insight: The action space is proportional to |L_t| (number of leaf nodes),
    not the coordinate space resolution. This enables continuous-space packing.
    """

    def __init__(self, bin_size: np.ndarray, num_orientations: int = 2):
        self.bin_size = bin_size
        self.num_orientations = num_orientations
        self.ems_manager = EMSManager(bin_size)

        # Tree nodes
        self.packed_items: List[Placement] = []  # Internal nodes B_t
        self.leaf_nodes: List[Placement] = []    # Leaf nodes L_t
        self.next_node_id = 0

        # History for recursive packing
        self.placement_history: List[Tuple[Box, Placement]] = []

    @property
    def num_internal(self) -> int:
        return len(self.packed_items)

    @property
    def num_leaves(self) -> int:
        return len(self.leaf_nodes)

    @property
    def utilization(self) -> float:
        """Current space utilization."""
        packed_volume = sum(p.volume for p in self.packed_items)
        bin_volume = float(np.prod(self.bin_size))
        return packed_volume / bin_volume if bin_volume > 0 else 0.0

    def get_feasible_leaves(self, item: Box,
                            check_stability: bool = False,
                            bin_size_override: Optional[np.ndarray] = None) -> List[Placement]:
        """
        Get all feasible leaf nodes for the current item.

        Generates candidates from EMS manager, then filters by:
        1. Containment constraint (within bin)
        2. Non-overlap constraint (not intersecting packed items)
        3. Stability constraint (optional -- if check_stability=True)

        Returns list of feasible Placement objects.
        """
        bin_size = bin_size_override if bin_size_override is not None else self.bin_size
        candidates = self.ems_manager.get_all_candidates(item, self.num_orientations)

        feasible = []
        for c in candidates:
            # Check containment
            if not c.is_inside_bin(bin_size):
                continue

            # Check non-overlap with all packed items
            overlap = False
            for packed in self.packed_items:
                if c.intersects(packed):
                    overlap = True
                    break
            if overlap:
                continue

            # Optional: check stability
            if check_stability:
                if not self._check_stability(c):
                    continue

            c.node_id = self.next_node_id
            self.next_node_id += 1
            feasible.append(c)

        self.leaf_nodes = feasible
        return feasible

    def place_item(self, item: Box, placement: Placement):
        """
        Place an item at the given placement position.

        1. Add to packed items (internal nodes)
        2. Update EMS manager
        3. Record history
        """
        self.packed_items.append(placement)
        self.ems_manager.place_item(placement)
        self.placement_history.append((item, placement))

    def _check_stability(self, placement: Placement) -> bool:
        """
        Check static stability of a placement.

        A placement is stable if:
        - It rests on the floor (z = 0), OR
        - A sufficient fraction of its bottom face is supported by packed items below

        For full support: 100% of bottom face must be on floor or other items.
        For partial support: typically >= 60-80% suffices.

        The PCT paper uses quasi-static equilibrium estimation from
        Zhao et al. (2022b) for a more physically accurate check.
        """
        if placement.position[2] < 1e-9:
            return True  # On the floor

        # Check support from items below
        support_area = 0.0
        placement_area = placement.size[0] * placement.size[1]

        for packed in self.packed_items:
            # Check if packed item is directly below
            packed_top_z = packed.position[2] + packed.size[2]
            if abs(packed_top_z - placement.position[2]) > 1e-6:
                continue

            # Calculate overlap area in xy plane
            overlap_x = max(0, min(placement.position[0] + placement.size[0],
                                   packed.position[0] + packed.size[0]) -
                           max(placement.position[0], packed.position[0]))
            overlap_y = max(0, min(placement.position[1] + placement.size[1],
                                   packed.position[1] + packed.size[1]) -
                           max(placement.position[1], packed.position[1]))
            support_area += overlap_x * overlap_y

        # Require at least 80% support (configurable)
        support_ratio = support_area / placement_area if placement_area > 0 else 0
        return support_ratio >= 0.80

    def to_feature_tensors(self, current_item: Box,
                           max_internal: int = 80,
                           max_leaves: int = 50) -> dict:
        """
        Convert PCT state to feature tensors for the neural network.

        Returns dict with:
        - internal_features: (max_internal, d_h) -- packed item descriptors
        - leaf_features: (max_leaves, d_h) -- candidate placement descriptors
        - current_features: (1, d_h) -- current item descriptor
        - internal_mask: (max_internal,) -- True for real nodes, False for padding
        - leaf_mask: (max_leaves,) -- True for real nodes, False for padding
        - num_internal: int
        - num_leaves: int

        Feature format per node (before MLP projection):
        - Internal node: [s_x, s_y, s_z, p_x, p_y, p_z, density, category, ...]
        - Leaf node: [s_o_x, s_o_y, s_o_z, p_x, p_y, p_z, orientation_idx, ...]
        - Current item: [s_x, s_y, s_z, 0, 0, 0, density, category, ...]

        All positions/sizes are normalized by bin dimensions.
        """
        d_raw = 8  # Raw feature dimension before MLP

        # Internal nodes
        internal_feats = np.zeros((max_internal, d_raw))
        internal_mask = np.zeros(max_internal, dtype=bool)
        for i, p in enumerate(self.packed_items[:max_internal]):
            internal_feats[i] = [
                p.size[0] / self.bin_size[0],
                p.size[1] / self.bin_size[1],
                p.size[2] / self.bin_size[2],
                p.position[0] / self.bin_size[0],
                p.position[1] / self.bin_size[1],
                p.position[2] / self.bin_size[2],
                1.0,  # density placeholder
                0.0,  # category placeholder
            ]
            internal_mask[i] = True

        # Leaf nodes
        leaf_feats = np.zeros((max_leaves, d_raw))
        leaf_mask = np.zeros(max_leaves, dtype=bool)
        for i, l in enumerate(self.leaf_nodes[:max_leaves]):
            leaf_feats[i] = [
                l.size[0] / self.bin_size[0],
                l.size[1] / self.bin_size[1],
                l.size[2] / self.bin_size[2],
                l.position[0] / self.bin_size[0],
                l.position[1] / self.bin_size[1],
                l.position[2] / self.bin_size[2],
                l.orientation_idx / max(1, self.num_orientations - 1),
                0.0,
            ]
            leaf_mask[i] = True

        # Current item
        current_feats = np.array([[
            current_item.width / self.bin_size[0],
            current_item.depth / self.bin_size[1],
            current_item.height / self.bin_size[2],
            0.0, 0.0, 0.0,
            current_item.density,
            current_item.category / 10.0,
        ]])

        return {
            'internal_features': internal_feats,
            'leaf_features': leaf_feats,
            'current_features': current_feats,
            'internal_mask': internal_mask,
            'leaf_mask': leaf_mask,
            'num_internal': min(len(self.packed_items), max_internal),
            'num_leaves': min(len(self.leaf_nodes), max_leaves),
        }


# =============================================================================
# SECTION 2: NEURAL NETWORK ARCHITECTURE (PyTorch)
# =============================================================================

"""
The following is PyTorch pseudocode for the PCT neural network.
Actual implementation requires: pip install torch

Architecture (from paper Section 3.1 and Appendix A):
1. Three MLPs project heterogeneous node features to common dimension d_h
2. SINGLE GAT layer (NOT multi-head) captures spatial relations between all nodes
3. Skip connection + Feed-Forward MLP
4. Pointer mechanism selects a leaf node as the action
5. Critic head estimates state value V(s_t)

IMPORTANT: The paper explicitly states they do NOT use multi-head attention:
  "We don't extend GAT to employ the multi-head attention mechanism
   (Vaswani et al. 2017) since we find that additional attention heads
   cannot help the final performance."

Key hyperparameters (from paper Appendix A and D):
- d_h = d_k = d_v = 64 (feature/key/value dimensions)
- c_clip = 10 (compatibility logit clipping, Equation 7)
- Activation: LeakyReLU for MLPs, ReLU for Feed-Forward
- Max internal nodes: 80 (for batch padding)
- Max leaf nodes: 25 * |O| (50 for |O|=2, 150 for |O|=6)
- Parallel environments: 64 (ACKTR)
- Forward rollout steps: k_s = 5
- Total parameters: ~68,000 (very lightweight for real-time inference)

Parameter count breakdown:
  MLP_internal:  d_raw * d_h + d_h + d_h * d_h + d_h = ~8,400
  MLP_leaf:      similar ~8,400
  MLP_current:   similar ~8,400
  GAT (W^Q, W^K, W^V, W^O): 4 * d_k * d_h = 16,384
  Feed-Forward:  d_h * d_h + d_h + d_h * d_h + d_h = ~8,320
  Pointer (W^q, W^k): 2 * d_k * d_h = 8,192
  Critic: d_h * d_h + d_h + d_h * 1 + 1 = ~4,225
  TOTAL: ~68,000

Feature input format per setting:
  Setting 1 (|O|=2, stability): d_raw=6 (size+position for internal/leaf), 3 for current
  Setting 3 (density-aware):    d_raw=8 (size+position+density+category)
"""


def pct_network_pseudocode():
    """
    # PyTorch implementation sketch (complete):

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical

    class PCTNetwork(nn.Module):
        '''
        Complete PCT Actor-Critic Network.

        Forward pass flow:
        1. phi_B, phi_L, phi_n project heterogeneous features to d_h
        2. Concatenate all nodes: h_hat = [h_B, h_L, h_n] in R^{N x d_h}
        3. GAT layer: h' = h_hat + W^O * softmax(QK^T / sqrt(d_k)) * V
        4. Feed-forward: h = h' + FF(h')
        5. Global context: h_bar = mean(h)
        6. Pointer: pi = softmax(c_clip * tanh(q^T k / sqrt(d_k)))
        7. Critic: V(s) = MLP(h_bar)

        Equations reference:
        - GAT: Equation (4) in paper
        - Skip connection: Equation (5)
        - Compatibility: Equation (6)
        - Policy: Equation (7)
        - Loss: Equation (13a, 13b)
        '''
        def __init__(self, d_raw_internal=8, d_raw_leaf=8, d_raw_current=8,
                     d_h=64, d_k=64, c_clip=10.0,
                     max_internal=80, max_leaves=50):
            super().__init__()
            self.d_h = d_h
            self.d_k = d_k
            self.c_clip = c_clip
            self.max_internal = max_internal
            self.max_leaves = max_leaves

            # Node-wise MLPs for heterogeneous feature projection
            # Each is a 2-layer network: Linear -> LeakyReLU -> Linear -> LeakyReLU
            self.mlp_internal = nn.Sequential(
                nn.Linear(d_raw_internal, d_h),
                nn.LeakyReLU(),
                nn.Linear(d_h, d_h),
                nn.LeakyReLU(),
            )
            self.mlp_leaf = nn.Sequential(
                nn.Linear(d_raw_leaf, d_h),
                nn.LeakyReLU(),
                nn.Linear(d_h, d_h),
                nn.LeakyReLU(),
            )
            self.mlp_current = nn.Sequential(
                nn.Linear(d_raw_current, d_h),
                nn.LeakyReLU(),
                nn.Linear(d_h, d_h),
                nn.LeakyReLU(),
            )

            # GAT layer parameters (single head, scaled dot-product attention)
            # W^Q in R^{d_k x d_h}, W^K in R^{d_k x d_h}
            # W^V in R^{d_v x d_h}, W^O in R^{d_h x d_v}
            self.W_Q = nn.Linear(d_h, d_k, bias=False)
            self.W_K = nn.Linear(d_h, d_k, bias=False)
            self.W_V = nn.Linear(d_h, d_h, bias=False)  # d_v = d_h in paper
            self.W_O = nn.Linear(d_h, d_h, bias=False)

            # Feed-forward after GAT (node-wise, with ReLU not LeakyReLU)
            self.ff = nn.Sequential(
                nn.Linear(d_h, d_h),
                nn.ReLU(),    # NOTE: ReLU here, not LeakyReLU
                nn.Linear(d_h, d_h),
            )

            # Pointer mechanism (Vinyals et al. 2015)
            self.W_q_ptr = nn.Linear(d_h, d_k, bias=False)
            self.W_k_ptr = nn.Linear(d_h, d_k, bias=False)

            # Critic head: maps h_bar to scalar V(s)
            self.critic = nn.Sequential(
                nn.Linear(d_h, d_h),
                nn.ReLU(),
                nn.Linear(d_h, 1),
            )

        def forward(self, internal_feat, leaf_feat, current_feat,
                    internal_mask, leaf_mask):
            '''
            Args:
                internal_feat: (batch, max_internal, d_raw) packed item features
                leaf_feat:     (batch, max_leaves, d_raw) candidate placement features
                current_feat:  (batch, 1, d_raw) current item features
                internal_mask: (batch, max_internal) bool -- True for real nodes
                leaf_mask:     (batch, max_leaves) bool -- True for real nodes

            Returns:
                pi:      (batch, max_leaves) policy over leaf nodes
                V_state: (batch,) state value estimate
            '''
            batch = internal_feat.size(0)

            # 1. Project heterogeneous features to common d_h
            h_B = self.mlp_internal(internal_feat)   # (batch, max_int, d_h)
            h_L = self.mlp_leaf(leaf_feat)           # (batch, max_leaf, d_h)
            h_n = self.mlp_current(current_feat)     # (batch, 1, d_h)

            # Concatenate all nodes: N = max_int + max_leaf + 1
            h_hat = torch.cat([h_B, h_L, h_n], dim=1)  # (batch, N, d_h)

            # Build full attention mask
            current_mask = torch.ones(batch, 1, dtype=torch.bool,
                                       device=internal_feat.device)
            full_mask = torch.cat([internal_mask, leaf_mask, current_mask],
                                   dim=1)  # (batch, N)

            # 2. GAT layer -- Equation (4)
            Q = self.W_Q(h_hat)  # (batch, N, d_k)
            K = self.W_K(h_hat)
            V = self.W_V(h_hat)

            # Scaled dot-product attention
            attn_logits = torch.bmm(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)
            # (batch, N, N)

            # Mask dummy/padding nodes with -inf
            # Mask along key dimension (columns): padding keys should not be attended to
            attn_mask = full_mask.unsqueeze(1).expand_as(attn_logits)
            attn_logits = attn_logits.masked_fill(~attn_mask, float('-inf'))

            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = attn_weights.masked_fill(attn_weights != attn_weights, 0)
            # Handle NaN from all-masked rows

            attn_out = torch.bmm(attn_weights, V)  # (batch, N, d_h)
            gat_out = self.W_O(attn_out)

            # Skip connection -- Equation (5) first part
            h_prime = h_hat + gat_out

            # Feed-forward with skip -- Equation (5) second part
            h = h_prime + self.ff(h_prime)  # (batch, N, d_h)

            # 3. Pointer mechanism -- select leaf node
            # Global context: mean of non-padding nodes
            # h_bar = (1/N) * sum_{i=1}^{N} h_i  (only real nodes)
            mask_expanded = full_mask.unsqueeze(-1).float()  # (batch, N, 1)
            h_sum = (h * mask_expanded).sum(dim=1)  # (batch, d_h)
            n_real = full_mask.float().sum(dim=1, keepdim=True)  # (batch, 1)
            h_bar = h_sum / n_real.clamp(min=1.0)  # (batch, d_h)

            # Query from global context
            q = self.W_q_ptr(h_bar)  # (batch, d_k)

            # Keys from leaf nodes only
            leaf_start = self.max_internal
            leaf_end = self.max_internal + self.max_leaves
            h_leaves = h[:, leaf_start:leaf_end, :]  # (batch, max_leaf, d_h)
            k = self.W_k_ptr(h_leaves)  # (batch, max_leaf, d_k)

            # Compatibility scores -- Equation (6)
            u = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) / (self.d_k ** 0.5)
            # (batch, max_leaf)

            # Clip and mask -- Equation (7)
            u = self.c_clip * torch.tanh(u)
            u = u.masked_fill(~leaf_mask, float('-inf'))

            # Actor output: policy over leaf nodes
            pi = F.softmax(u, dim=-1)

            # Critic output: state value V(s_t)
            V_state = self.critic(h_bar).squeeze(-1)  # (batch,)

            return pi, V_state

        def select_action(self, internal_feat, leaf_feat, current_feat,
                          internal_mask, leaf_mask, deterministic=False):
            '''
            Select an action (leaf node index) from the policy.

            During training: sample from Categorical(pi) for exploration
            During testing: argmax(pi) for exploitation
            '''
            pi, V_state = self.forward(internal_feat, leaf_feat, current_feat,
                                        internal_mask, leaf_mask)
            if deterministic:
                action = pi.argmax(dim=-1)
                log_prob = torch.log(pi.gather(1, action.unsqueeze(1)) + 1e-8)
            else:
                dist = Categorical(pi)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            return action, log_prob, V_state, pi
    """
    pass


# =============================================================================
# SECTION 2B: LEAF NODE INTERCEPTION (from paper Section 3.1)
# =============================================================================

class LeafInterceptor:
    """
    When |L_t| exceeds a threshold, randomly subsample to fixed size.

    From paper:
      L_sub subset L_t, |L_sub| = c * |O|

    Where c ~ 25 (determined by grid search).
    During training: c * |O| = 25 * 2 = 50 for Setting 1 (|O|=2)
    During testing:  c * |O| = 25 * 2 to 150 * 2 (grid search, step 10)

    The paper shows interception barely hurts performance because:
    1. Sub-optimal solutions exist even in the intercepted set
    2. Random leaf selection forces exploration, helping escape local optima

    IMPORTANT: The interception constant c is FIXED during training but can
    be tuned via grid search at test time for up to +2-3% improvement.
    """

    def __init__(self, num_orientations: int = 2, c_train: int = 25,
                 c_test_range: Tuple = (25, 150), c_test_step: int = 10):
        self.num_orientations = num_orientations
        self.c_train = c_train
        self.max_leaves_train = c_train * num_orientations
        self.c_test_range = c_test_range
        self.c_test_step = c_test_step

    def intercept(self, leaves: List[Placement], training: bool = True,
                  c_override: Optional[int] = None) -> List[Placement]:
        """
        Subsample leaves if they exceed the threshold.

        Args:
            leaves: all feasible leaf nodes
            training: if True use fixed c_train; if False allow c_override
            c_override: optional override for test-time tuning

        Returns:
            Subsampled list of leaves (or original if below threshold)
        """
        if c_override is not None:
            max_leaves = c_override * self.num_orientations
        elif training:
            max_leaves = self.max_leaves_train
        else:
            max_leaves = self.max_leaves_train  # Default; override for grid search

        if len(leaves) <= max_leaves:
            return leaves

        # Random subsample (uniform, no replacement)
        indices = np.random.choice(len(leaves), size=max_leaves, replace=False)
        return [leaves[i] for i in sorted(indices)]

    def grid_search_c_test(self, pct_env, policy_network,
                            num_episodes: int = 100) -> int:
        """
        Grid search for optimal interception constant at test time.

        Evaluate across c_test_range with c_test_step increments.
        Returns c that maximizes average utilization.

        From paper: tested range 50-300 with step 10 for |O|=2.
        """
        best_c = self.c_train
        best_util = 0.0

        for c in range(self.c_test_range[0], self.c_test_range[1] + 1,
                       self.c_test_step):
            total_util = 0.0
            for _ in range(num_episodes):
                # Run episode with this c value
                util = self._evaluate_episode(pct_env, policy_network, c)
                total_util += util
            avg_util = total_util / num_episodes
            if avg_util > best_util:
                best_util = avg_util
                best_c = c

        return best_c

    def _evaluate_episode(self, pct_env, policy_network, c: int) -> float:
        """Run one episode with given interception constant c."""
        # Placeholder -- actual implementation hooks into the BPP environment
        return 0.0


# =============================================================================
# SECTION 2C: ALL FOUR EXPANSION SCHEMES (from paper Section 3.1)
# =============================================================================

"""
The PCT paper describes four leaf expansion schemes. The EMSManager above
implements EMS (the default). Below are pseudocode outlines for the other three
schemes, for completeness and potential hybrid use.

COMPARISON TABLE (from paper Table 3, Setting 2):
| Scheme | Complexity            | Paper Uti. | Default? |
|--------|-----------------------|------------|----------|
| CP     | O(c) constant         | 81.8%      | No       |
| EP     | O(m * |B_2D|)         | 78.1%      | No       |
| EMS    | O(|E|) linear         | 86.0%      | YES      |
| EV     | O(m * |B_2D|^2) quad  | 85.3%      | No       |
| FC     | O(coord^3)            | 76.9%      | No       |

KEY FINDING: EMS outperforms full coordinate space (FC) despite generating
fewer candidates. Heuristic-guided expansion provides locally optimal
candidates (proven in Appendix C), reducing the DRL agent's search burden.

For thesis: use EMS during training. Optionally use EV during testing for
maximum quality (superset of all other schemes).
"""


class CornerPointExpander:
    """
    Corner Point (CP) expansion -- Martello et al. (2000).

    When item n_t is placed at (p_x, p_y, p_z), new corner points are generated
    wherever the envelope of packed items changes from vertical to horizontal.

    New candidates:
      (p_x + s_x, p_y, p_z)  -- right of item along x
      (p_x, p_y + s_y, p_z)  -- behind item along y
      (p_x, p_y, p_z + s_z)  -- above item along z

    if the envelope of the corresponding 2D plane is changed by n_t.

    Pros: Very fast; O(c) constant time using heightmap.
    Cons: Misses valid positions in concavities below envelope; fewest candidates.
    """

    def __init__(self, bin_size: np.ndarray):
        self.bin_size = bin_size
        # Heightmap for fast envelope tracking (xoy plane)
        grid = int(bin_size[0]) + 1  # Assumes integer bin size for discrete
        self.heightmap = np.zeros((grid, grid))

    def generate_corners(self, placement: Placement) -> List[np.ndarray]:
        """Generate corner points from a new placement."""
        p = placement.position
        s = placement.size
        candidates = []

        # Right corner (along x)
        cx = p[0] + s[0]
        if cx < self.bin_size[0]:
            candidates.append(np.array([cx, p[1], p[2]]))

        # Back corner (along y)
        cy = p[1] + s[1]
        if cy < self.bin_size[1]:
            candidates.append(np.array([p[0], cy, p[2]]))

        # Top corner (along z)
        cz = p[2] + s[2]
        if cz < self.bin_size[2]:
            candidates.append(np.array([p[0], p[1], cz]))

        return candidates


class EventPointExpander:
    """
    Event Point (EV) expansion -- superset of CP, EP, and EMS.

    Combine ALL boundary points of packed items along {x, y} with the
    start/end points of n_t. For all distinct p^z values satisfying
    p_n^z <= p^z <= p_n^z + s_n^z, scan all boundary points in the z-plane.

    This produces the COMPLETE set of candidate positions.

    Complexity: O(m * |B_2D|^2) per update -- quadratic in packed items.

    EVF = Full event points (without interception / random subset sampling).

    For thesis: could use EV during testing for maximum quality
    while keeping EMS during training for efficiency.
    """

    def __init__(self, bin_size: np.ndarray):
        self.bin_size = bin_size

    def generate_event_points(self, packed_items: List[Placement],
                               current_item: Box,
                               num_orientations: int = 2) -> List[Placement]:
        """
        Generate all event point candidates.

        Algorithm:
        1. Collect all x-boundary values from packed items
        2. Collect all y-boundary values from packed items
        3. For each distinct z-level where placement could occur:
           a. Take cross product of x-boundaries x y-boundaries
           b. Each (x, y, z) combo is a candidate position
        4. Filter by containment and non-overlap
        5. Generate candidates for each valid orientation
        """
        # Collect all x-boundaries
        x_bounds = {0.0, float(self.bin_size[0])}
        y_bounds = {0.0, float(self.bin_size[1])}
        z_levels = {0.0}

        for p in packed_items:
            x_bounds.add(float(p.position[0]))
            x_bounds.add(float(p.position[0] + p.size[0]))
            y_bounds.add(float(p.position[1]))
            y_bounds.add(float(p.position[1] + p.size[1]))
            z_levels.add(float(p.position[2] + p.size[2]))

        candidates = []
        for o_idx, oriented_size in enumerate(
                current_item.get_orientations(num_orientations)):
            for z in sorted(z_levels):
                if z + oriented_size[2] > self.bin_size[2] + 1e-9:
                    continue
                for x in sorted(x_bounds):
                    if x + oriented_size[0] > self.bin_size[0] + 1e-9:
                        continue
                    for y in sorted(y_bounds):
                        if y + oriented_size[1] > self.bin_size[1] + 1e-9:
                            continue
                        pos = np.array([x, y, z])
                        cand = Placement(
                            position=pos,
                            size=oriented_size.copy(),
                            orientation_idx=o_idx,
                        )
                        # Check non-overlap with all packed items
                        overlap = False
                        for packed in packed_items:
                            if cand.intersects(packed):
                                overlap = True
                                break
                        if not overlap:
                            candidates.append(cand)

        return candidates


# =============================================================================
# SECTION 3: MDP AND REWARD
# =============================================================================

class PCTReward:
    """
    Reward computation for PCT-based 3D-BPP.

    Basic reward: r_t = c_r * v_t (volume of placed item)
    With constraints: w_t = max(0, v_t + c * f_hat(.))

    Where:
    - c_r = 10 / (S^x * S^y * S^z) -- normalizing constant
    - v_t = s_n^x * s_n^y * s_n^z -- volume of item n_t
    - c = 0.1 (default constraint weight)
    - f_hat(.) = f(.) / f_bar -- normalized constraint reward

    The max operator ensures non-negative weights, encouraging
    the policy to always pack items (never skip).
    """

    def __init__(self, bin_size: np.ndarray, constraint_weight: float = 0.1):
        self.bin_size = bin_size
        self.bin_volume = float(np.prod(bin_size))
        self.c_r = 10.0 / self.bin_volume
        self.c = constraint_weight

    def compute_basic_reward(self, item: Box) -> float:
        """Basic space utilization reward."""
        return self.c_r * item.volume

    def compute_reward_with_constraints(self, item: Box, placement: Placement,
                                        packed_items: List[Placement],
                                        constraints: dict) -> float:
        """
        Reward with practical constraints.

        constraints dict can include:
        - 'stability': stability score
        - 'load_balance': load balance score
        - 'isle_friendliness': clustering score
        - etc.

        Each constraint function f(.) is normalized by its average
        under random placement to get f_hat(.).
        """
        v_t = item.volume
        f_total = 0.0

        for name, (f_value, f_avg) in constraints.items():
            if f_avg != 0:
                f_hat = f_value / abs(f_avg)
            else:
                f_hat = f_value
            f_total += f_hat

        w_t = max(0, v_t + self.c * f_total)
        return self.c_r * w_t


# =============================================================================
# SECTION 4: ADAPTATION FOR 2-BOUNDED SPACE WITH BUFFER
# =============================================================================

class TwoBinPCTManager:
    """
    Extension of PCT for 2-bounded space (k=2) packing.

    Manages two active bins (pallets), each with its own PCT.
    Combined with a buffer of 5-10 items.

    This is the KEY EXTENSION needed for the thesis project.
    The PCT paper does not address multi-bin bounded-space packing.

    Design decisions:
    1. Each bin has its own PCT and EMS manager
    2. The ToP search tree is extended to include bin selection
    3. Spatial ensemble is used for cross-bin comparison
    4. A bin is closed when no feasible placement exists for ANY buffered item
    """

    def __init__(self, bin_size: np.ndarray, num_orientations: int = 2):
        self.bin_size = bin_size
        self.num_orientations = num_orientations
        self.bins: List[PackingConfigurationTree] = [
            PackingConfigurationTree(bin_size, num_orientations),
            PackingConfigurationTree(bin_size, num_orientations),
        ]
        self.active_bins = [True, True]
        self.closed_bins: List[PackingConfigurationTree] = []
        self.total_bins_used = 2

    def get_all_candidates(self, item: Box) -> List[Tuple[int, Placement]]:
        """
        Get feasible placements across both active bins.
        Returns list of (bin_index, placement) tuples.
        """
        candidates = []
        for bin_idx, (pct, active) in enumerate(zip(self.bins, self.active_bins)):
            if not active:
                continue
            leaves = pct.get_feasible_leaves(item, check_stability=True)
            for leaf in leaves:
                candidates.append((bin_idx, leaf))
        return candidates

    def place_item(self, item: Box, bin_idx: int, placement: Placement):
        """Place item in specified bin."""
        self.bins[bin_idx].place_item(item, placement)

    def should_close_bin(self, bin_idx: int, buffer: List[Box]) -> bool:
        """
        Determine if a bin should be closed.

        A bin should be closed when:
        1. No item in the buffer can fit in it, OR
        2. The state value V(.) is very low (minimal future utility), OR
        3. The utilization exceeds a threshold and remaining space is fragmented

        This is a heuristic decision -- could also be learned.
        """
        pct = self.bins[bin_idx]

        # Check if any buffered item fits
        any_fits = False
        for item in buffer:
            leaves = pct.get_feasible_leaves(item, check_stability=True)
            if len(leaves) > 0:
                any_fits = True
                break

        if not any_fits:
            return True

        # Optional: close if utilization is high and space is fragmented
        if pct.utilization > 0.85:
            max_ems_volume = max((e.volume for e in pct.ems_manager.ems_list), default=0)
            min_item_volume = min((item.volume for item in buffer), default=float('inf'))
            if max_ems_volume < min_item_volume:
                return True

        return False

    def close_bin(self, bin_idx: int):
        """Close a bin and open a new one."""
        self.closed_bins.append(self.bins[bin_idx])
        self.bins[bin_idx] = PackingConfigurationTree(self.bin_size, self.num_orientations)
        self.total_bins_used += 1

    def get_total_utilization(self) -> float:
        """Average utilization across all bins (active + closed)."""
        all_bins = self.closed_bins + [b for b, a in zip(self.bins, self.active_bins) if a]
        if not all_bins:
            return 0.0
        return sum(b.utilization for b in all_bins) / len(all_bins)


class BufferManager:
    """
    Manages a buffer of 5-10 items for semi-online packing.

    Items enter the buffer from a conveyor/stream.
    The planner selects which item to pack next and in which bin.
    """

    def __init__(self, buffer_size: int = 5):
        self.buffer_size = buffer_size
        self.buffer: List[Box] = []
        self.item_stream = None  # Set externally

    def fill_buffer(self, stream):
        """Fill buffer from item stream up to capacity."""
        while len(self.buffer) < self.buffer_size:
            try:
                item = next(stream)
                self.buffer.append(item)
            except StopIteration:
                break

    def remove_item(self, item: Box):
        """Remove a packed item from buffer."""
        self.buffer.remove(item)

    def is_empty(self) -> bool:
        return len(self.buffer) == 0

    def peek(self) -> List[Box]:
        """View all items in buffer without removing."""
        return list(self.buffer)


# =============================================================================
# SECTION 5: ToP SEARCH WITH MCTS FOR BUFFER + 2 BINS
# =============================================================================

class MCTSNode:
    """Node in the MCTS search tree for ToP planning."""

    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action  # (item_index, bin_index) tuple
        self.children: Dict[Tuple[int, int], 'MCTSNode'] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = 1.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_explore: float = 1.414) -> float:
        """Upper Confidence Bound for tree search."""
        if self.visit_count == 0:
            return float('inf')
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = c_explore * np.sqrt(np.log(parent_visits) / self.visit_count)
        return self.q_value + exploration


def top_mcts_planning(
    bins: TwoBinPCTManager,
    buffer: List[Box],
    policy_network,  # Pre-trained PCT actor-critic network
    num_simulations: int = 200,
    c_explore: float = 1.414,
) -> Tuple[int, int]:
    """
    MCTS-based planning for ToP with buffer and 2 bins.

    Searches over orderings of (item_choice, bin_choice) to find
    the best first action to execute.

    Returns: (item_index_in_buffer, bin_index) for the best first action.

    Algorithm:
    1. Each MCTS iteration: select path via UCB, expand, simulate, backpropagate
    2. Simulation uses pi_theta (policy network) for rollout
    3. Unknown future items valued by V(.) (critic network)
    4. After all simulations, select the root child with highest visit count

    Complexity: O(num_simulations * buffer_size * 2)
    For buffer=10, bins=2, sims=200: ~4000 policy evaluations
    """
    root = MCTSNode()

    for _ in range(num_simulations):
        node = root
        # Deep copy current state for simulation
        # (In practice, use incremental state updates + undo)
        sim_bins = _copy_bins(bins)
        sim_buffer = list(buffer)
        path = []

        # SELECTION: traverse tree using UCB
        while node.children and sim_buffer:
            best_child = None
            best_score = -float('inf')
            for action, child in node.children.items():
                score = child.ucb_score(c_explore)
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
            path.append(node)

            # Apply action to simulated state
            item_idx, bin_idx = node.action
            if item_idx < len(sim_buffer):
                item = sim_buffer[item_idx]
                candidates = sim_bins.get_all_candidates(item)
                bin_candidates = [(b, p) for b, p in candidates if b == bin_idx]
                if bin_candidates:
                    # Use policy to select placement within bin
                    _, placement = bin_candidates[0]  # Simplified; use pi_theta in practice
                    sim_bins.place_item(item, bin_idx, placement)
                    sim_buffer.pop(item_idx)

        # EXPANSION: create children for unvisited actions
        if sim_buffer:
            for item_idx in range(len(sim_buffer)):
                for bin_idx in range(2):
                    action = (item_idx, bin_idx)
                    if action not in node.children:
                        node.children[action] = MCTSNode(parent=node, action=action)

        # SIMULATION: rollout using policy network
        value = _simulate_rollout(sim_bins, sim_buffer, policy_network)

        # BACKPROPAGATION
        for n in reversed(path):
            n.visit_count += 1
            n.total_value += value

        root.visit_count += 1

    # Select best first action (most visited child)
    if not root.children:
        return (0, 0)  # Default: first item, first bin

    best_action = max(root.children.items(),
                      key=lambda x: x[1].visit_count)
    return best_action[0]


def _simulate_rollout(bins: TwoBinPCTManager, buffer: List[Box],
                       policy_network) -> float:
    """
    Simulate remaining packing using policy network.

    For items in buffer: use pi_theta for placement decisions.
    For unknown future items: use V(.) state value estimate.

    Returns: estimated total value (sum of volumes + V(.))
    """
    total_value = 0.0

    for item in buffer:
        candidates = bins.get_all_candidates(item)
        if candidates:
            # In practice: use policy_network to score candidates
            # Simplified: pick first feasible
            bin_idx, placement = candidates[0]
            bins.place_item(item, bin_idx, placement)
            total_value += item.volume

    # Add state value for unknown future items
    # V(.) estimated by critic network on current state
    # total_value += policy_network.critic(bins_to_features(bins))

    return total_value


def _copy_bins(bins: TwoBinPCTManager) -> TwoBinPCTManager:
    """Deep copy of bin manager for simulation."""
    import copy
    return copy.deepcopy(bins)


# =============================================================================
# SECTION 6: INTEGRATION POINTS WITH OTHER METHODS
# =============================================================================

"""
Integration with other methods in the thesis:

1. STABILITY (stability/ folder):
   - PCT's quasi-static equilibrium can be replaced with the fast stability
     validation from "Online 3D Bin Packing with Fast Stability Validation"
   - Physics-based test-time checking via PyBullet or Isaac Gym
   - Integration point: the _check_stability() method in PackingConfigurationTree

2. HEURISTICS (heuristics/ folder):
   - The EMS leaf expansion scheme is itself a heuristic from Ha et al. (2017)
   - DBLF, Corner Distances, DFTRC can serve as alternative/additional
     expansion schemes
   - Integration point: EMSManager.get_all_candidates()

3. MULTI-BIN (multi_bin/ folder):
   - The TwoBinPCTManager extends PCT to 2-bounded space
   - For more bins (k > 2), generalize the bin selection in MCTS
   - Connection to "Solving Online 3D Multi-Bin Packing with Deep RL"

4. SEMI-ONLINE BUFFER (semi_online_buffer/ folder):
   - BufferManager + MCTS planning is the core semi-online component
   - Connection to Puche & Lee (2022) TAP-NET++ for buffering
   - Connection to "Near-optimal Algorithms for Stochastic Online Bin Packing"
     for theoretical performance bounds

5. NOVEL IDEAS (novel_ideas/ folder):
   - PCT as a hyper-heuristic: let the DRL agent choose between different
     expansion schemes (CP, EP, EMS) depending on packing state
   - Curriculum learning: train PCT on easy (few items) then hard (many items)
   - Transfer learning: pre-train PCT on uniform distribution, fine-tune
     on real warehouse distribution
"""


# =============================================================================
# SECTION 7: COMPLETE SEMI-ONLINE PACKING LOOP
# =============================================================================

def semi_online_packing_loop(
    bin_size: np.ndarray,
    item_stream,
    buffer_size: int = 5,
    num_orientations: int = 2,
    mcts_simulations: int = 200,
    check_stability: bool = True,
):
    """
    Complete semi-online packing loop with buffer and 2-bounded space.

    This is the main algorithm for the thesis project.

    Args:
        bin_size: (3,) array of bin dimensions
        item_stream: iterator yielding Box objects
        buffer_size: number of items in buffer (5-10)
        num_orientations: 2 (top-down robot) or 6 (general)
        mcts_simulations: MCTS iterations per decision
        check_stability: whether to enforce stability constraints

    Returns:
        results dict with utilization, bins used, items packed, etc.
    """
    # Initialize
    bin_manager = TwoBinPCTManager(bin_size, num_orientations)
    buffer_mgr = BufferManager(buffer_size)
    buffer_mgr.fill_buffer(item_stream)

    total_items_packed = 0
    packing_log = []

    while not buffer_mgr.is_empty():
        buffer = buffer_mgr.peek()

        # Check if any bin needs closing
        for bin_idx in range(2):
            if bin_manager.active_bins[bin_idx]:
                if bin_manager.should_close_bin(bin_idx, buffer):
                    bin_manager.close_bin(bin_idx)

        # Get all feasible placements across both bins and all buffer items
        any_feasible = False
        for item in buffer:
            candidates = bin_manager.get_all_candidates(item)
            if candidates:
                any_feasible = True
                break

        if not any_feasible:
            # No placement possible -- close both bins and open new ones
            for bin_idx in range(2):
                if bin_manager.active_bins[bin_idx]:
                    bin_manager.close_bin(bin_idx)
            continue

        # MCTS planning: choose (item, bin) from buffer
        item_idx, bin_idx = top_mcts_planning(
            bin_manager, buffer, policy_network=None,  # Pass trained model
            num_simulations=mcts_simulations,
        )

        # Execute the chosen action
        chosen_item = buffer[item_idx]
        candidates = bin_manager.get_all_candidates(chosen_item)
        bin_candidates = [(b, p) for b, p in candidates if b == bin_idx]

        if bin_candidates:
            _, placement = bin_candidates[0]  # In practice, use pi_theta
            bin_manager.place_item(chosen_item, bin_idx, placement)
            buffer_mgr.remove_item(chosen_item)
            total_items_packed += 1

            packing_log.append({
                'item': chosen_item,
                'bin': bin_idx,
                'placement': placement,
                'utilization': bin_manager.get_total_utilization(),
            })

        # Refill buffer
        buffer_mgr.fill_buffer(item_stream)

    # Results
    return {
        'total_items_packed': total_items_packed,
        'total_bins_used': bin_manager.total_bins_used,
        'average_utilization': bin_manager.get_total_utilization(),
        'packing_log': packing_log,
    }


# =============================================================================
# SECTION 8: ESTIMATED COMPLEXITY AND FEASIBILITY
# =============================================================================

"""
COMPLEXITY ANALYSIS:

Per-item decision:
- EMS candidate generation: O(|E| * |O|) where |E| = num EMSs, |O| = orientations
  Typical: |E| < 100, |O| = 2 --> ~200 candidates
- Feasibility checking: O(candidates * |B|) where |B| = packed items
  Typical: 200 * 50 = 10,000 checks
- GAT forward pass: O(N^2 * d_h) where N = |B| + |L| + 1
  Typical: (50 + 50 + 1)^2 * 64 = ~650,000 operations
- Pointer mechanism: O(|L| * d_k)
  Typical: 50 * 64 = 3,200 operations
- MCTS (for buffer planning): O(sims * buffer_size * 2 * (GAT + pointer))
  Typical: 200 * 10 * 2 * 653,200 = ~2.6 billion operations

FEASIBILITY ASSESSMENT:
- Without MCTS (pure online, s=1): ~10ms per decision on GPU --> fully real-time
- With MCTS (buffer s=5, 100 sims): ~500ms per decision on GPU --> acceptable for 9.8s cycle
- With MCTS (buffer s=10, 200 sims): ~2s per decision on GPU --> tight but feasible
- The paper reports running costs of ~10^-2 to 10^-3 seconds per item

TRAINING REQUIREMENTS:
- 64 parallel environments (ACKTR)
- ~500K training steps
- GPU: NVIDIA TITAN V or equivalent
- Training time: ~12-24 hours (estimated)
- For thesis: can use fewer environments (8-16) with longer training

SIMPLIFICATIONS FOR THESIS PROTOTYPE:
1. Start with 1-bin PCT (no multi-bin) to validate the core approach
2. Use simpler stability check (support area ratio) before physics-based
3. Reduce MCTS simulations to 50-100 for faster iteration
4. Use PPO instead of ACKTR if implementation is simpler
5. Start with discrete solution space (integer coordinates) before continuous
"""


# =============================================================================
# SECTION 9: ACKTR TRAINING DETAILS (from paper Section 3.2 and Appendix D)
# =============================================================================

"""
ACKTR (Actor-Critic using Kronecker-Factored Trust Region) -- Wu et al. 2017

Why ACKTR over PPO/A2C:
1. Sample efficiency: 2-3x better than A2C and TRPO
2. Natural gradient: K-FAC approximates Fisher information matrix
3. Computational cost: Only 10-25% higher per-update than SGD
4. Trust region: Prevents catastrophically large policy updates
5. Precedent: Zhao et al. (2021) showed ACKTR > SAC for online 3D-BPP

K-FAC INTUITION:
  The Fisher information matrix F is N_params x N_params -- infeasible to invert.
  K-FAC approximates F as a Kronecker product:  F ~ A x B
  Where A, B are much smaller matrices from activation/gradient statistics.
  Inversion: (A x B)^{-1} = A^{-1} x B^{-1}

  Natural gradient: delta_theta = F^{-1} * grad_theta L
  This accounts for parameter space curvature, giving more efficient updates.

TRAINING CONFIGURATION (exact from paper):
  Parallel processes:     k_p = 64
  Forward rollout steps:  k_s = 5 (practical setting)
  Batch size:             k_p * k_s = 64 * 5 = 320 transitions per update
  Training steps:         ~500K (from learning curves, Figure 17)
  GAT throughput:         > 400 FPS even with stability checks
  Discount gamma:         1.0 (finite episodes, undiscounted)
  Loss weights:           alpha = beta = 1 (actor and critic equal weight)

  Loss functions:
    L_actor  = (r_t + gamma * V(s_{t+1}) - V(s_t)) * log pi(a_t | s_t)  (Eq 13a)
    L_critic = (r_t + gamma * V(s_{t+1}) - V(s_t))^2                     (Eq 13b)
    L_total  = alpha * L_actor + beta * L_critic

  Hardware: Gold 5117 CPU + GeForce TITAN V GPU
  Training time: ~12-24 hours estimated (from convergence curves)

THESIS SIMPLIFICATION:
  Use PPO instead of ACKTR if implementation is complex.
  PPO (Schulman et al. 2017) is widely available in stable-baselines3.
  Reduce to 8-16 parallel environments with longer training (~48 hours).
  Use stable-baselines3 PPO with custom network architecture.
"""


def acktr_training_pseudocode():
    """
    # ACKTR training loop pseudocode:

    import torch
    from kfac import KFACOptimizer  # K-FAC implementation

    def train_pct_acktr(env_factory, num_envs=64, k_s=5, total_steps=500000,
                         gamma=1.0, alpha=1.0, beta=1.0):
        '''
        Train PCT policy using ACKTR.

        env_factory: function returning a BPP environment instance
        num_envs:    number of parallel environments (k_p)
        k_s:         forward rollout steps
        total_steps: total training iterations
        '''
        # Create parallel environments
        envs = [env_factory() for _ in range(num_envs)]

        # Initialize network and optimizer
        network = PCTNetwork(d_raw=8, d_h=64, d_k=64, c_clip=10.0)
        optimizer = KFACOptimizer(network, lr=0.25)
        # Note: K-FAC learning rate is typically larger than SGD (~0.25 vs 0.001)

        # Initialize states
        states = [env.reset() for env in envs]

        for step in range(total_steps):
            # Collect k_s rollout steps from each environment
            rollout_buffer = []

            for t in range(k_s):
                # Convert states to tensor features
                features = batch_states_to_features(states, network)

                # Forward pass
                pi, V_state = network(**features)

                # Sample actions
                actions = Categorical(pi).sample()
                log_probs = Categorical(pi).log_prob(actions)

                # Step environments
                next_states, rewards, dones = [], [], []
                for i, (env, action) in enumerate(zip(envs, actions)):
                    s, r, d = env.step(action.item())
                    next_states.append(s)
                    rewards.append(r)
                    dones.append(d)
                    if d:
                        next_states[i] = env.reset()

                rollout_buffer.append({
                    'states': states,
                    'actions': actions,
                    'log_probs': log_probs,
                    'rewards': rewards,
                    'dones': dones,
                    'values': V_state,
                })

                states = next_states

            # Compute advantages using GAE or simple TD
            # A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            advantages = compute_advantages(rollout_buffer, gamma)

            # Compute actor and critic losses
            # L_actor = -A_t * log pi(a_t | s_t)    (policy gradient)
            # L_critic = A_t^2                        (value function MSE)
            actor_loss = -(advantages * log_probs_batch).mean()
            critic_loss = advantages.pow(2).mean()
            total_loss = alpha * actor_loss + beta * critic_loss

            # ACKTR update (natural gradient via K-FAC)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Evaluate periodically
            if step % 10000 == 0:
                avg_util = evaluate(network, env_factory, num_episodes=100)
                print(f"Step {step}: avg utilization = {avg_util:.2%}")

        return network
    '''
    """
    pass


def ppo_training_pseudocode():
    """
    # PPO alternative (simpler, recommended for thesis):

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def train_pct_ppo(env_factory, num_envs=16, total_timesteps=1000000):
        '''
        Train PCT policy using PPO via stable-baselines3.

        Requires: wrapping PCT environment in gymnasium API.
        '''
        # Create vectorized environments
        env = SubprocVecEnv([env_factory for _ in range(num_envs)])

        # PPO with custom network
        model = PPO(
            policy='MlpPolicy',  # Replace with custom PCTPolicy
            env=env,
            learning_rate=3e-4,
            n_steps=128,          # Rollout length per env
            batch_size=256,
            n_epochs=4,
            gamma=1.0,            # Undiscounted (finite episodes)
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,        # Encourage exploration
            vf_coef=0.5,
            verbose=1,
        )

        model.learn(total_timesteps=total_timesteps)
        return model
    '''
    """
    pass


# =============================================================================
# SECTION 10: ALL SIX CONSTRAINT REWARD FUNCTIONS (from paper Section 4.4)
# =============================================================================

class ConstraintRewards:
    """
    All six practical constraint reward functions from PCT paper Section 4.4.

    General formula:
      w_t = max(0, v_t + c * f_hat(.))    (Equation 16)
      where f_hat(.) = f(.) / f_bar (normalized by average under random policy)
      c = 0.1 (default, tested at 0.1, 1.0, 10.0)

    CONSTRAINT SENSITIVITY (from paper Table 18, Setting 1):
    | Constraint         | c=0.1 Uti. | c=1.0 Uti. | c=10 Uti. |
    |--------------------|------------|------------|-----------|
    | Isle Friendliness  | 71.8%      | 66.2%      | 59.5%     |
    | Load Balancing     | 70.9%      | 65.2%      | 59.0%     |
    | Height Uniformity  | 73.8%      | 73.2%      | 64.5%     |
    | Kinematic          | 72.7%      | 70.2%      | 68.5%     |
    | Load Bearing       | 69.6%      | 61.9%      | 57.2%     |
    | Bridging           | 68.8%      | 66.2%      | 62.0%     |

    Observations:
    - Kinematic constraints conflict least with utilization (-4.2% at c=10)
    - Load bearing conflicts most (-12.4% at c=10)
    - Default c=0.1 provides good balance
    - For thesis: use stability + height uniformity with c=0.1
    """

    @staticmethod
    def isle_friendliness(placement: Placement, packed_items: List[Placement],
                          categories: List[int], current_category: int) -> float:
        """
        f = -dist(n_t, B_t): negative average distance to same-category items.
        Encourages grouping items of same category together.

        For warehouse: group by SKU family, destination zone, etc.
        """
        if not packed_items:
            return 0.0
        same_cat_items = [p for p, cat in zip(packed_items, categories)
                          if cat == current_category]
        if not same_cat_items:
            return 0.0
        # Average Euclidean distance to same-category items
        center = placement.position + placement.size / 2
        distances = []
        for p in same_cat_items:
            other_center = p.position + p.size / 2
            distances.append(float(np.linalg.norm(center - other_center)))
        return -np.mean(distances)

    @staticmethod
    def load_balancing(placement: Placement, packed_items: List[Placement],
                       bin_size: np.ndarray) -> float:
        """
        f = -var(n_t, B_t): negative mass distribution variance on the floor.
        Encourages even weight distribution across bin footprint.

        Divide floor into quadrants, compute mass variance across quadrants.
        """
        if not packed_items:
            return 0.0
        # Divide floor into 2x2 quadrants
        half_x = bin_size[0] / 2
        half_y = bin_size[1] / 2
        quadrant_mass = [0.0, 0.0, 0.0, 0.0]  # TL, TR, BL, BR

        all_items = list(packed_items) + [placement]
        for p in all_items:
            cx = p.position[0] + p.size[0] / 2
            cy = p.position[1] + p.size[1] / 2
            mass = float(np.prod(p.size))  # Volume as proxy for mass
            q_idx = (0 if cx < half_x else 1) + (0 if cy < half_y else 2)
            quadrant_mass[q_idx] += mass

        return -float(np.var(quadrant_mass))

    @staticmethod
    def height_uniformity(placement: Placement, packed_items: List[Placement],
                          bin_size: np.ndarray, grid_res: int = 10) -> float:
        """
        f = -H_var: negative heightmap variance.
        Encourages flat top surfaces, which improves stability and stacking.

        Compute heightmap at grid resolution, then take variance.
        """
        dx = bin_size[0] / grid_res
        dy = bin_size[1] / grid_res
        heightmap = np.zeros((grid_res, grid_res))

        all_items = list(packed_items) + [placement]
        for p in all_items:
            top_z = p.position[2] + p.size[2]
            x_start = max(0, int(p.position[0] / dx))
            x_end = min(grid_res, int((p.position[0] + p.size[0]) / dx) + 1)
            y_start = max(0, int(p.position[1] / dy))
            y_end = min(grid_res, int((p.position[1] + p.size[1]) / dy) + 1)
            for ix in range(x_start, x_end):
                for iy in range(y_start, y_end):
                    heightmap[ix, iy] = max(heightmap[ix, iy], top_z)

        return -float(np.var(heightmap))

    @staticmethod
    def kinematic_safe_position(placement: Placement,
                                packed_items: List[Placement],
                                bin_size: np.ndarray,
                                clearance: float = 0.40) -> float:
        """
        f = V_safe: safe position reward from Zhao et al. (2022b).
        Requires clearance above surrounding boxes for robot gripper.

        Paper uses 40cm clearance above neighboring boxes.
        Returns 1.0 if safe, 0.0 if blocked.
        """
        # Check if gripper can reach the placement position from above
        p_top = placement.position[2] + placement.size[2]
        for packed in packed_items:
            # Check if any packed item is adjacent and too tall
            if placement.intersects(Placement(
                position=np.array([
                    packed.position[0],
                    packed.position[1],
                    placement.position[2]
                ]),
                size=np.array([
                    packed.size[0],
                    packed.size[1],
                    clearance
                ]),
                orientation_idx=0
            )):
                packed_top = packed.position[2] + packed.size[2]
                if packed_top > p_top + clearance:
                    return 0.0  # Blocked by taller neighbor
        return 1.0

    @staticmethod
    def load_bearing(placement: Placement, packed_items: List[Placement]) -> float:
        """
        f = -E_{b in B_t} bear(b, B_t): negative expected bearing force.
        Penalizes placing heavy items on top of fragile ones.

        Simplified: compute total weight above each item and penalize
        if any item supports too much relative weight.
        """
        if not packed_items:
            return 0.0
        max_bearing = 0.0
        for packed in packed_items:
            # How much weight does this item bear from items above?
            bearing = 0.0
            packed_top = packed.position[2] + packed.size[2]
            for other in packed_items:
                if other is packed:
                    continue
                if abs(other.position[2] - packed_top) < 1e-6:
                    # other rests on top of packed
                    bearing += float(np.prod(other.size))  # Volume as weight proxy
            # Include the new placement
            if abs(placement.position[2] - packed_top) < 1e-6:
                bearing += float(np.prod(placement.size))
            max_bearing = max(max_bearing, bearing)
        return -max_bearing

    @staticmethod
    def bridging(placement: Placement, packed_items: List[Placement]) -> float:
        """
        f = bridge(n_t, B_t): count of bridging-contributing items.
        Encourages staggered packing patterns that interlock items.

        An item contributes to bridging if it rests on top of 2+ items
        (spanning across a gap). More bridging = more structural stability.
        """
        support_count = 0
        for packed in packed_items:
            packed_top = packed.position[2] + packed.size[2]
            if abs(placement.position[2] - packed_top) < 1e-6:
                # Check overlap in xy plane
                ox = max(0, min(placement.position[0] + placement.size[0],
                               packed.position[0] + packed.size[0]) -
                         max(placement.position[0], packed.position[0]))
                oy = max(0, min(placement.position[1] + placement.size[1],
                               packed.position[1] + packed.size[1]) -
                         max(placement.position[1], packed.position[1]))
                if ox > 1e-6 and oy > 1e-6:
                    support_count += 1
        # Bridging = resting on 2+ items
        return float(max(0, support_count - 1))


# =============================================================================
# SECTION 11: RECURSIVE PACKING WITH SPATIAL ENSEMBLE
#             (from paper Section 3.3, for large-scale N=200-1000)
# =============================================================================

"""
For large-scale problems (N > ~30 items), direct PCT inference degrades:
- Long decision sequences cause training instability
- O(N^2) GAT becomes expensive
- Test distribution mismatch

SOLUTION: Decompose large PCT T into sub-trees T = {T^1, ..., T^n}

DECOMPOSITION ALGORITHM:
1. Given current item n and large-scale T:
2. Start at random leaf node l in T
3. Backtrack upward from l to find ancestor b^v such that |subtree(b^v)| > tau
4. The historical EMS at b^v defines sub-bin c^v
5. Detect all nodes overlapping with c^v
6. These overlapping nodes form T^v
7. Repeat until all leaf nodes assigned to at least one sub-tree

THRESHOLD tau = 30:
  Determined by generating 2000 random sequences from U(0.1, 0.5),
  evaluating pi_theta, and finding 95th percentile of packed item count = 30.4.
  Setting tau = 30 ensures sub-problems match the scale pi_theta was trained on.

SUB-TREE NORMALIZATION (Equation 8):
  b_hat^v = (b^v - FLB(c^v)) * S / s^v
  l_hat^v = (l^v - FLB(c^v)) * S / s^v
  n_hat^v = n * S / s^v

  Where S = original bin size, s^v = sub-bin size.
  This lets pre-trained pi_theta generalize to sub-problems without retraining.

SPATIAL ENSEMBLE INTEGRATION:
  Problem: Direct V(.) or pi(.) evaluation in individual sub-bins gives
  locally optimal but globally suboptimal solutions.

  Solution: Convert absolute scores to RANKS, then max-min across sub-bins.

  Score:   Phi(l, c_i) = pi(. | T_hat_i)
  Rank:    Phi_hat(l, c_i) = rank_L(Phi(l, c_i))
  Select:  l* = argmax_{l in L} min_{c_i in c} Phi_hat(l, c_i)    (Equation 9)

ALTERNATIVE INTEGRATION METHODS (from paper Table 7):
| Method              | N=200  | N=500  | N=1000 |
|---------------------|--------|--------|--------|
| Max State Value     | 60.5%  | 46.9%  | 41.4%  |
| Max Return          | 66.2%  | 50.0%  | 45.4%  |
| Max Volume          | 66.6%  | 61.5%  | 48.7%  |
| Min Surface Area    | 65.4%  | 55.5%  | 49.7%  |
| SPATIAL ENSEMBLE    | 76.9%  | 79.9%  | 81.2%  |  <-- dramatically best

KEY RESULT: Recursive packing performance IMPROVES with scale
(76.9% at N=200 -> 81.2% at N=1000), while direct DRL degrades
catastrophically (72.3% -> 56.4%).

For thesis: relevant if packing many small items. Not strictly needed
for buffer=10 per bin, but the spatial ensemble concept is reused for
cross-bin evaluation in 2-bounded space.
"""


class RecursivePackingDecomposer:
    """
    Decomposes a large PCT into sub-trees for recursive packing.

    Each sub-tree contains <= tau items and corresponds to a sub-bin
    region of the original bin. Sub-trees are normalized so pi_theta
    can process them as if they were standard-size bins.
    """

    def __init__(self, bin_size: np.ndarray, tau: int = 30):
        self.bin_size = bin_size
        self.tau = tau

    def decompose(self, pct: PackingConfigurationTree) -> List[dict]:
        """
        Decompose PCT into sub-trees.

        Returns list of dicts, each containing:
        - 'sub_bin_pos': FLB position of sub-bin
        - 'sub_bin_size': dimensions of sub-bin
        - 'items': list of Placement objects in this sub-tree
        - 'leaves': list of leaf Placement objects in this sub-tree
        - 'scale_factor': S / s^v for normalization
        """
        if pct.num_internal <= self.tau:
            # No decomposition needed
            return [{
                'sub_bin_pos': np.zeros(3),
                'sub_bin_size': self.bin_size.copy(),
                'items': list(pct.packed_items),
                'leaves': list(pct.leaf_nodes),
                'scale_factor': np.ones(3),
            }]

        sub_trees = []
        assigned_leaves = set()

        for leaf in pct.leaf_nodes:
            if id(leaf) in assigned_leaves:
                continue

            # Find a sub-bin region containing this leaf
            sub_bin_pos, sub_bin_size = self._find_sub_bin(leaf, pct)

            # Collect all items and leaves within this sub-bin
            items_in_sub = []
            leaves_in_sub = []

            for item in pct.packed_items:
                if self._overlaps_region(item, sub_bin_pos, sub_bin_size):
                    items_in_sub.append(item)

            for l in pct.leaf_nodes:
                if self._overlaps_region(l, sub_bin_pos, sub_bin_size):
                    leaves_in_sub.append(l)
                    assigned_leaves.add(id(l))

            # Compute normalization factor
            scale_factor = self.bin_size / np.maximum(sub_bin_size, 1e-9)

            sub_trees.append({
                'sub_bin_pos': sub_bin_pos,
                'sub_bin_size': sub_bin_size,
                'items': items_in_sub,
                'leaves': leaves_in_sub,
                'scale_factor': scale_factor,
            })

        return sub_trees

    def normalize_sub_tree(self, sub_tree: dict) -> dict:
        """
        Normalize a sub-tree so its coordinates map to the full bin.

        Equation 8:
          b_hat = (b - FLB(c^v)) * S / s^v
          l_hat = (l - FLB(c^v)) * S / s^v
          n_hat = n * S / s^v
        """
        offset = sub_tree['sub_bin_pos']
        scale = sub_tree['scale_factor']

        normalized_items = []
        for item in sub_tree['items']:
            norm_item = Placement(
                position=(item.position - offset) * scale,
                size=item.size * scale,
                orientation_idx=item.orientation_idx,
            )
            normalized_items.append(norm_item)

        normalized_leaves = []
        for leaf in sub_tree['leaves']:
            norm_leaf = Placement(
                position=(leaf.position - offset) * scale,
                size=leaf.size * scale,
                orientation_idx=leaf.orientation_idx,
            )
            normalized_leaves.append(norm_leaf)

        return {
            'items': normalized_items,
            'leaves': normalized_leaves,
            'scale_factor': scale,
            'offset': offset,
        }

    def _find_sub_bin(self, leaf: Placement,
                      pct: PackingConfigurationTree) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find a sub-bin region of appropriate size containing the leaf.
        Uses the EMS that contains this leaf as the sub-bin boundary.
        """
        # Find the EMS that contains this leaf position
        for ems in pct.ems_manager.ems_list:
            if (np.all(leaf.position >= ems.position - 1e-9) and
                np.all(leaf.position < ems.brt + 1e-9)):
                return ems.position.copy(), ems.size.copy()

        # Fallback: use a region around the leaf
        region_size = self.bin_size / 2
        region_pos = np.maximum(leaf.position - region_size / 4, 0)
        region_size = np.minimum(region_size, self.bin_size - region_pos)
        return region_pos, region_size

    def _overlaps_region(self, placement: Placement,
                         region_pos: np.ndarray,
                         region_size: np.ndarray) -> bool:
        """Check if a placement overlaps with a region."""
        for d in range(3):
            if (placement.position[d] >= region_pos[d] + region_size[d] or
                placement.position[d] + placement.size[d] <= region_pos[d]):
                return False
        return True


class SpatialEnsemble:
    """
    Spatial ensemble for integrating placement scores across sub-bins.

    From paper Equation (9):
      l* = argmax_{l in L} min_{c_i in c} Phi_hat(l, c_i)

    Where Phi_hat is the rank-normalized score.

    This is ALSO used in the 2-bounded space extension for cross-bin
    evaluation (see top_buffer_2bounded_coding_ideas.py).
    """

    @staticmethod
    def rank_based_selection(sub_bin_scores: List[List[Tuple[int, float]]]
                              ) -> int:
        """
        Select the best leaf index using rank-based spatial ensemble.

        Args:
            sub_bin_scores: for each sub-bin, list of (leaf_index, score) tuples

        Returns:
            leaf_index with best worst-case rank across sub-bins
        """
        # Convert absolute scores to ranks within each sub-bin
        all_leaf_indices = set()
        for scores in sub_bin_scores:
            for leaf_idx, _ in scores:
                all_leaf_indices.add(leaf_idx)

        if not all_leaf_indices:
            return 0

        # Compute ranks per sub-bin
        leaf_ranks = {}  # leaf_idx -> list of ranks across sub-bins
        for sub_idx, scores in enumerate(sub_bin_scores):
            sorted_scores = sorted(scores, key=lambda x: x[1])
            n = len(sorted_scores)
            for rank, (leaf_idx, _) in enumerate(sorted_scores):
                if leaf_idx not in leaf_ranks:
                    leaf_ranks[leaf_idx] = []
                leaf_ranks[leaf_idx].append((rank + 1) / max(n, 1))

        # Max-min: best worst-case rank
        best_leaf = 0
        best_min_rank = -1.0
        for leaf_idx, ranks in leaf_ranks.items():
            min_rank = min(ranks)
            if min_rank > best_min_rank:
                best_min_rank = min_rank
                best_leaf = leaf_idx

        return best_leaf


# =============================================================================
# SECTION 12: MULTI-SCALE TRAINING (from paper Section 3.3)
# =============================================================================

"""
For recursive packing, pi_theta must handle varying item size distributions
(because sub-bin normalization changes effective item sizes).

MULTI-SCALE TRAINING DISTRIBUTIONS:
The distribution changes randomly after each packing episode:
  - N(0.3, 0.1^2): large items
  - N(0.1, 0.2^2): medium items with high variance
  - N(0.5, 0.2^2): medium-large items

This forces the policy to be robust to different item scales.

For thesis: multi-scale training is important if using recursive packing.
For standard buffer=10 with 2-bounded space, can train with the target
item distribution directly. However, exposing to multiple distributions
during training can improve generalization (paper shows good cross-distribution
results in Table 12).
"""


class MultiScaleItemSampler:
    """
    Sample items from multiple size distributions for multi-scale training.

    Randomly switches between distributions after each episode.
    """

    def __init__(self, bin_size: np.ndarray, distributions: Optional[List] = None):
        self.bin_size = bin_size
        if distributions is None:
            # Default distributions from paper (continuous domain)
            self.distributions = [
                {'mean': 0.3, 'std': 0.1},   # Large items
                {'mean': 0.1, 'std': 0.2},   # Medium, high variance
                {'mean': 0.5, 'std': 0.2},   # Medium-large
            ]
        else:
            self.distributions = distributions
        self.current_dist_idx = 0

    def new_episode(self):
        """Switch to random distribution for next episode."""
        self.current_dist_idx = np.random.randint(len(self.distributions))

    def sample_item(self, box_id: int = -1) -> Box:
        """Sample a single item from current distribution."""
        dist = self.distributions[self.current_dist_idx]
        sizes = np.clip(
            np.random.normal(dist['mean'], dist['std'], size=3),
            0.1 * self.bin_size,
            0.5 * self.bin_size
        )
        return Box(
            width=float(sizes[0]),
            depth=float(sizes[1]),
            height=float(sizes[2]),
            box_id=box_id
        )

    def sample_episode(self, num_items: int = 100) -> List[Box]:
        """Sample a full episode worth of items."""
        self.new_episode()
        return [self.sample_item(box_id=i) for i in range(num_items)]


# =============================================================================
# SECTION 13: STABILITY INTEGRATION PIPELINE
#             (from paper Sections 4.5 and 4.7)
# =============================================================================

"""
TWO-PHASE STABILITY APPROACH:

Phase 1: TRAINING-TIME -- Quasi-static equilibrium (fast, approximate)
  - Pre-filter leaf nodes: unstable placements never shown to DRL agent
  - Speed: >400 FPS (compatible with ACKTR training throughput)
  - Accuracy: 55% transportation stability (Table 11)
  - Cost: negligible (simple geometric computation)

Phase 2: TEST-TIME -- Physics simulation via Isaac Gym (slow, accurate)
  - For each candidate from top k_l leaves:
    - Run k_d disturbance sets (each = 10 random translations + rotations)
    - If stack survives ALL disturbances: accept
    - Select accepted placement with highest pi_theta probability
  - k_d = 8 -> 100% transportation stability

Disturbance model:
  Translations: [15, 20] cm along x, y axes
  Rotations: [-10, 10] degrees around z axis
  Linear velocity: 6 m/s
  Angular velocity: 30 degrees/s

WHY NOT PHYSICS FOR TRAINING?
  1. Isaac Gym requires CPU-GPU sync for dynamic object creation -> slow
  2. Training-side physics causes occasional test instability (overfitting)

THESIS RECOMMENDATION:
  - Training: use quasi-static equilibrium (simple support ratio for v1,
    full center-of-gravity check for v2)
  - Testing: use PyBullet (lighter than Isaac Gym, easier to set up)
    with k_d = 4-8 disturbance sets

STABILITY RESULTS (from paper Table 11):
| Method                          | Transport stability |
|---------------------------------|---------------------|
| Quasi-static equilibrium only   | 55%                 |
| Physics-based (k_d = 1)        | 70%                 |
| Physics-based (k_d = 2)        | 85%                 |
| Physics-based (k_d = 4)        | 95%                 |
| Physics-based (k_d = 8)        | 100%                |
"""


class QuasiStaticStabilityChecker:
    """
    Fast quasi-static equilibrium estimation for training-time stability.

    A placement is stable if:
    1. It rests on the floor (z = 0), OR
    2. Its center of gravity projection falls within the support polygon

    The support polygon is the convex hull of the contact regions
    with items below.

    Speed: O(|B|) per check -- fast enough for >400 FPS training.
    """

    def __init__(self, support_threshold: float = 0.80,
                 use_cog_check: bool = True):
        self.support_threshold = support_threshold
        self.use_cog_check = use_cog_check

    def is_stable(self, placement: Placement,
                  packed_items: List[Placement]) -> bool:
        """
        Check if a placement is statically stable.

        Two checks:
        1. Support area ratio: bottom face coverage >= threshold
        2. COG projection: center of gravity falls within support polygon
        """
        if placement.position[2] < 1e-9:
            return True  # On the floor -- always stable

        # Check 1: Support area ratio
        support_area = 0.0
        placement_area = placement.size[0] * placement.size[1]

        support_contacts = []  # (x_min, y_min, x_max, y_max) of contact patches

        for packed in packed_items:
            packed_top_z = packed.position[2] + packed.size[2]
            if abs(packed_top_z - placement.position[2]) > 1e-6:
                continue  # Not directly below

            # Overlap area in xy plane
            ox_min = max(placement.position[0], packed.position[0])
            ox_max = min(placement.position[0] + placement.size[0],
                         packed.position[0] + packed.size[0])
            oy_min = max(placement.position[1], packed.position[1])
            oy_max = min(placement.position[1] + placement.size[1],
                         packed.position[1] + packed.size[1])

            if ox_max > ox_min and oy_max > oy_min:
                area = (ox_max - ox_min) * (oy_max - oy_min)
                support_area += area
                support_contacts.append((ox_min, oy_min, ox_max, oy_max))

        support_ratio = support_area / placement_area if placement_area > 0 else 0
        if support_ratio < self.support_threshold:
            return False

        # Check 2: Center of gravity projection (optional, more accurate)
        if self.use_cog_check and support_contacts:
            cog_x = placement.position[0] + placement.size[0] / 2
            cog_y = placement.position[1] + placement.size[1] / 2

            # Check if COG projection falls within union of support contacts
            cog_supported = False
            for (sx_min, sy_min, sx_max, sy_max) in support_contacts:
                if sx_min <= cog_x <= sx_max and sy_min <= cog_y <= sy_max:
                    cog_supported = True
                    break

            if not cog_supported:
                return False

        return True

    def filter_stable_leaves(self, leaves: List[Placement],
                              packed_items: List[Placement]) -> List[Placement]:
        """Filter out unstable leaf placements. Used during candidate generation."""
        return [l for l in leaves if self.is_stable(l, packed_items)]


class PhysicsStabilityVerifier:
    """
    Physics-based stability verification for test-time.

    Uses PyBullet (or Isaac Gym) to simulate disturbances and check
    if the packing configuration survives.

    DISTURBANCE MODEL (from paper):
      10 random perturbations per disturbance set:
        - Translation: [15, 20] cm along x, y
        - Rotation: [-10, 10] degrees around z
        - Linear velocity: 6 m/s
        - Angular velocity: 30 deg/s
    """

    def __init__(self, k_d: int = 4, k_l: int = 5,
                 use_pybullet: bool = True):
        """
        Args:
            k_d: number of disturbance sets to test (4-8 recommended)
            k_l: number of top candidates to verify (from pi_theta ranking)
            use_pybullet: if True use PyBullet; if False placeholder
        """
        self.k_d = k_d
        self.k_l = k_l
        self.use_pybullet = use_pybullet

    def verify_placement(self, placement: Placement,
                          packed_items: List[Placement],
                          bin_size: np.ndarray) -> bool:
        """
        Verify a single placement survives all disturbance sets.

        Returns True only if the packing survives ALL k_d disturbance sets.

        Pseudocode:
          for d in range(k_d):
              disturbance = generate_random_disturbance()
              survived = simulate(packed_items + [placement], disturbance)
              if not survived:
                  return False
          return True
        """
        if not self.use_pybullet:
            return True  # Placeholder: skip physics

        for d in range(self.k_d):
            disturbance = self._generate_disturbance()
            if not self._simulate_disturbance(placement, packed_items,
                                               bin_size, disturbance):
                return False
        return True

    def select_best_stable_placement(self, candidates: List[Placement],
                                       scores: List[float],
                                       packed_items: List[Placement],
                                       bin_size: np.ndarray) -> Optional[Placement]:
        """
        From the top k_l candidates (sorted by pi_theta score),
        select the highest-scoring one that passes physics verification.

        This is the test-time placement selection algorithm.
        """
        # Sort by score descending, take top k_l
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        top_k = ranked[:self.k_l]

        for placement, score in top_k:
            if self.verify_placement(placement, packed_items, bin_size):
                return placement

        # If none survive: fall back to highest-scoring (accept risk)
        return ranked[0][0] if ranked else None

    def _generate_disturbance(self) -> dict:
        """Generate a random disturbance set (10 perturbations)."""
        return {
            'translations': np.random.uniform(
                low=[-0.20, -0.20, 0],
                high=[0.20, 0.20, 0],
                size=(10, 3)
            ),
            'rotations': np.random.uniform(
                low=[-10, -10, -10],
                high=[10, 10, 10],
                size=(10, 3)
            ),
            'linear_velocity': 6.0,   # m/s
            'angular_velocity': 30.0,  # deg/s
        }

    def _simulate_disturbance(self, placement: Placement,
                               packed_items: List[Placement],
                               bin_size: np.ndarray,
                               disturbance: dict) -> bool:
        """
        Simulate disturbance in physics engine.

        Pseudocode for PyBullet integration:
          import pybullet as p
          p.connect(p.DIRECT)
          # Create bin as static plane/box
          # Create each packed item as rigid body
          # Add new placement
          # Apply disturbance (forces/velocities)
          # Step simulation for N steps
          # Check if all items remain within bin (no collapse)
          p.disconnect()
        """
        # TODO: Implement PyBullet simulation
        # For now: accept all placements (placeholder)
        return True


# =============================================================================
# SECTION 14: THEORETICAL ANALYSIS (from paper Appendix C)
# =============================================================================

"""
THEOREM (Local Optimality of Heuristic Candidates):

For a boundary point p in partial_E (boundary of the No-Fit Polygon E),
if p is a convex vertex, then:
  p = argmax_{q in N(p)} d^T q    (Equation 15)
for some open neighborhood N(p) and direction d in R^2.

TIGHTNESS MEASURE (Lemma 2):
  psi(p) = pi - theta
where theta is the interior angle of E at boundary point p.

INTERPRETATION:
Candidates generated by EMS and EV lie on convex vertices of the NFP (No-Fit
Polygon), which are provably locally optimal under the tightness measure.
The DRL agent selects the globally best among these locally optimal candidates.

This explains why heuristic-guided candidates outperform random or full-
coordinate-space candidates: they pre-filter to locally optimal positions,
reducing the search burden on the DRL agent.

PRACTICAL IMPLICATION FOR THESIS:
- EMS candidates are theoretically well-founded, not just heuristic
- The DRL agent needs only to learn global selection, not local placement
- This reduces the effective complexity of the learning problem
- Even with limited training (PPO, fewer environments), decent performance
  is expected because the candidate set is already high-quality
"""


# =============================================================================
# SECTION 15: GENERALIZATION TESTING SETUP (from paper Section 4.3)
# =============================================================================

"""
CROSS-DISTRIBUTION GENERALIZATION (paper Table 12):

Train on distribution D_train, test on distribution D_test:
| Train -> Test           | Utilization |
|-------------------------|-------------|
| U(1,5) -> U(1,5)       | 86.0%       |
| U(1,5) -> U(2,5)       | 81.1%       |
| U(1,5) -> U(3,5)       | 78.0%       |
| U(2,5) -> U(1,5)       | 79.0%       |
| U(2,5) -> U(2,5)       | 81.1%       |
| U(3,5) -> U(1,5)       | 72.0%       |

KEY FINDINGS:
- Training on broader distribution (U(1,5)) gives best generalization
- Performance degrades gracefully across distributions
- For thesis: train on broadest expected item range

ITEM SIZE DISTRIBUTIONS USED IN PAPER:
  Discrete: s^d in Z+, s^d <= S^d/2, drawn uniformly
  Continuous: s^d ~ U(a, S^d/2) where a = 0.1 (Setting 1,2) or variable

  For ICRA Stacking Challenge (Sim4Dexterity 2023):
  - Heterogeneous box sizes matching real warehouse
  - 8000 SKUs in real deployment
  - Box sizes up to 80 x 80 x 60 cm in 120 x 100 cm pallet
"""


class GeneralizationBenchmark:
    """
    Benchmark runner for testing generalization across item distributions.
    """

    STANDARD_DISTRIBUTIONS = {
        'discrete_uniform_1_5': {'type': 'discrete', 'low': 1, 'high': 5, 'bin': 10},
        'discrete_uniform_2_5': {'type': 'discrete', 'low': 2, 'high': 5, 'bin': 10},
        'discrete_uniform_3_5': {'type': 'discrete', 'low': 3, 'high': 5, 'bin': 10},
        'continuous_01_05': {'type': 'continuous', 'low': 0.1, 'high': 0.5, 'bin': 1.0},
        'warehouse_realistic': {
            'type': 'continuous',
            'sizes': [  # Common warehouse box sizes (meters)
                (0.3, 0.2, 0.15), (0.4, 0.3, 0.2), (0.5, 0.4, 0.3),
                (0.6, 0.4, 0.3), (0.8, 0.6, 0.4), (0.3, 0.3, 0.3),
            ],
            'bin': np.array([1.2, 1.0, 1.4]),  # Standard pallet
        },
    }

    @staticmethod
    def generate_items(dist_name: str, num_items: int = 100) -> List[Box]:
        """Generate items according to a named distribution."""
        config = GeneralizationBenchmark.STANDARD_DISTRIBUTIONS[dist_name]

        items = []
        for i in range(num_items):
            if config['type'] == 'discrete':
                sizes = np.random.randint(config['low'], config['high'] + 1, size=3)
            elif 'sizes' in config:
                # Sample from predefined sizes
                idx = np.random.randint(len(config['sizes']))
                sizes = np.array(config['sizes'][idx])
                # Add noise
                sizes = sizes * np.random.uniform(0.8, 1.2, size=3)
            else:
                sizes = np.random.uniform(config['low'], config['high'], size=3)

            items.append(Box(
                width=float(sizes[0]),
                depth=float(sizes[1]),
                height=float(sizes[2]),
                box_id=i
            ))
        return items

    @staticmethod
    def run_cross_distribution_test(policy_network,
                                     train_dist: str,
                                     test_dists: List[str],
                                     num_episodes: int = 100) -> dict:
        """
        Test a policy trained on train_dist across multiple test distributions.

        Returns dict mapping test_dist -> average utilization.
        """
        results = {}
        for test_dist in test_dists:
            total_util = 0.0
            for _ in range(num_episodes):
                items = GeneralizationBenchmark.generate_items(test_dist)
                # Run episode with policy_network (placeholder)
                # util = run_episode(policy_network, items, ...)
                util = 0.0  # placeholder
                total_util += util
            results[test_dist] = total_util / num_episodes
        return results


# =============================================================================
# SECTION 16: GYMNASIUM ENVIRONMENT WRAPPER
# =============================================================================

class BinPackingEnv:
    """
    Gymnasium-compatible environment for PCT-based 3D bin packing.

    This wraps the PCT data structures into a standard RL environment
    interface for training with ACKTR, PPO, or other algorithms.

    State: PCT feature tensors (internal + leaf + current node features)
    Action: leaf node index (integer)
    Reward: c_r * w_t (volume-based, optionally with constraints)
    Done: when no feasible placement exists for current item

    Compatible with:
    - stable-baselines3 (PPO, A2C)
    - Custom ACKTR implementation
    - Any gymnasium-compatible RL library
    """

    def __init__(self, bin_size: np.ndarray,
                 num_orientations: int = 2,
                 item_distribution: str = 'discrete_uniform_1_5',
                 max_internal: int = 80,
                 max_leaves: int = 50,
                 check_stability: bool = False,
                 constraint_weight: float = 0.1):
        self.bin_size = bin_size
        self.num_orientations = num_orientations
        self.item_distribution = item_distribution
        self.max_internal = max_internal
        self.max_leaves = max_leaves
        self.check_stability = check_stability
        self.reward_fn = PCTReward(bin_size, constraint_weight)

        # Will be initialized in reset()
        self.pct = None
        self.current_item = None
        self.items_packed = 0

    def reset(self) -> dict:
        """Reset environment and return initial observation."""
        self.pct = PackingConfigurationTree(self.bin_size, self.num_orientations)
        self.current_item = self._sample_item()
        self.items_packed = 0

        # Generate initial leaf nodes
        self.pct.get_feasible_leaves(self.current_item,
                                      check_stability=self.check_stability)

        return self._get_observation()

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        """
        Take an action (select leaf node index).

        Returns: (observation, reward, done, info)
        """
        # Validate action
        if action >= len(self.pct.leaf_nodes) or action < 0:
            return self._get_observation(), 0.0, True, {'error': 'invalid_action'}

        placement = self.pct.leaf_nodes[action]

        # Place item
        self.pct.place_item(self.current_item, placement)
        self.items_packed += 1

        # Compute reward
        reward = self.reward_fn.compute_basic_reward(self.current_item)

        # Sample next item
        self.current_item = self._sample_item()

        # Generate new leaf nodes
        leaves = self.pct.get_feasible_leaves(self.current_item,
                                               check_stability=self.check_stability)

        # Check termination
        done = len(leaves) == 0

        info = {
            'utilization': self.pct.utilization,
            'items_packed': self.items_packed,
            'num_leaves': len(leaves),
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> dict:
        """Convert PCT state to observation tensors."""
        return self.pct.to_feature_tensors(
            self.current_item,
            max_internal=self.max_internal,
            max_leaves=self.max_leaves
        )

    def _sample_item(self) -> Box:
        """Sample a random item from the configured distribution."""
        if 'discrete' in self.item_distribution:
            sizes = np.random.randint(1, int(self.bin_size[0] / 2) + 1, size=3)
        else:
            sizes = np.random.uniform(0.1, float(self.bin_size[0] / 2), size=3)
        return Box(
            width=float(sizes[0]),
            depth=float(sizes[1]),
            height=float(sizes[2]),
            box_id=self.items_packed
        )


# =============================================================================
# SECTION 17: RUNNING COST AND PERFORMANCE REFERENCE TABLE
# =============================================================================

"""
RUNNING COSTS (from paper Table 16):
| Domain     | Setting | Method         | Time/item (s)  |
|------------|---------|----------------|----------------|
| Discrete   | 1       | CDRL           | 5.51e-2        |
| Discrete   | 1       | PCT & EMS      | 2.68e-2        |
| Discrete   | 2       | PCT & EMS      | 1.25e-2        |
| Continuous | 1       | PCT & EV       | 4.46e-2        |
| Continuous | 2       | PCT & EMS      | 2.50e-2        |

ToP PLANNING TIMES (from paper Table 9):
| Setting                | Time/item (s) |
|------------------------|---------------|
| ToP (s=1, p=9)        | 1.1-1.8       |
| ToP (s=10, p=0)       | 1.2-2.1       |
| ToP (s=5, p=5)        | 1.3-2.0       |

All within 9.8-second robot cycle time.

INFERENCE SCALING (from paper Figure 25(c)):
  At test time (k_p=1, k_s=1): both GAT and pointer scale linearly
  with node count N. Linear at inference even though training is quadratic.
"""


if __name__ == "__main__":
    # Quick sanity test of data structures
    bin_size = np.array([10.0, 10.0, 10.0])

    # Create PCT
    pct = PackingConfigurationTree(bin_size, num_orientations=2)

    # Create a test item
    item = Box(width=3.0, depth=4.0, height=2.0, box_id=0)

    # Get feasible placements
    leaves = pct.get_feasible_leaves(item)
    print(f"Number of feasible placements for first item: {len(leaves)}")

    if leaves:
        # Place item at first feasible position
        pct.place_item(item, leaves[0])
        print(f"Placed item at {leaves[0].position}, utilization: {pct.utilization:.2%}")

        # Get placements for second item
        item2 = Box(width=2.0, depth=2.0, height=3.0, box_id=1)
        leaves2 = pct.get_feasible_leaves(item2)
        print(f"Number of feasible placements for second item: {len(leaves2)}")

    # Test 2-bin manager
    manager = TwoBinPCTManager(bin_size, num_orientations=2)
    candidates = manager.get_all_candidates(item)
    print(f"\nTotal candidates across 2 bins: {len(candidates)}")

    # Test reward computation
    reward_fn = PCTReward(bin_size, constraint_weight=0.1)
    basic_r = reward_fn.compute_basic_reward(item)
    print(f"\nBasic reward for item (3x4x2): {basic_r:.4f}")

    # Test stability checker
    checker = QuasiStaticStabilityChecker(support_threshold=0.80)
    if leaves:
        is_stable = checker.is_stable(leaves[0], [])
        print(f"Floor placement stable: {is_stable}")

    # Test constraint rewards
    print(f"\nHeight uniformity reward: "
          f"{ConstraintRewards.height_uniformity(leaves[0] if leaves else Placement(np.zeros(3), np.ones(3), 0), [], bin_size):.4f}")

    # Test multi-scale sampler
    sampler = MultiScaleItemSampler(bin_size)
    episode_items = sampler.sample_episode(num_items=10)
    print(f"\nMulti-scale sampler: generated {len(episode_items)} items")
    print(f"  Sizes: {[f'{i.width:.1f}x{i.depth:.1f}x{i.height:.1f}' for i in episode_items[:3]]}")

    print("\nAll data structures and algorithms initialized successfully.")
