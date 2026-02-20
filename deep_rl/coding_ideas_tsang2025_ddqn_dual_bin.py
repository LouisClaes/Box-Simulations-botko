"""
==============================================================================
CODING IDEAS: Double DQN for Online 3D Dual-Bin Packing (Tsang et al. 2025)
==============================================================================

Source Paper:
    Tsang, Y.P., Mo, D.Y., Chung, K.T., Lee, C.K.M. (2025).
    "A deep reinforcement learning approach for online and concurrent 3D bin
    packing optimisation with bin replacement strategies."
    Computers in Industry, 164, 104202.

Target Use Case:
    - Semi-online with 5-10 box buffer (lookahead)
    - 2-bounded space (2 active pallets/bins)
    - Maximize fill rate + stability
    - Real-world robotic/conveyor system
    - Python/PyTorch implementation

This file provides:
    1. Core data structures
    2. Maximal Cuboids Algorithm (MCA) -- 3D MAXRECTS extension
    3. Height map and action map computation
    4. Stability check
    5. Reward function (pyramid + compactness)
    6. Double DQN architecture for variable action spaces
    7. Bin replacement strategies (replaceAll, replaceMax)
    8. Full training loop skeleton
    9. Adaptation notes for our specific use case

Estimated Implementation Effort:
    - MCA alone: ~2-3 days
    - Full DQN system: ~2-3 weeks
    - Integration + testing: ~1-2 weeks

Complexity Analysis:
    - MCA per placement: O(|M|^2) where |M| = number of maximal cuboids
    - DQN inference per step: O(6 * k * |M|) forward passes
    - Training: O(iterations * episodes * steps_per_episode * inference_cost)
    - Memory: O(replay_buffer_size * state_size)

==============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
from collections import deque
import random

# ==============================================================================
# SECTION 1: Core Data Structures
# ==============================================================================

@dataclass
class Item:
    """Represents a 3D rectangular item (box) to be packed."""
    id: int
    w: float  # width (x-axis)
    l: float  # length (z-axis)
    h: float  # height (y-axis)
    weight: float = 1.0  # Extension: weight for stability calculations

    @property
    def volume(self) -> float:
        return self.w * self.l * self.h

    def get_rotations(self) -> List[Tuple[float, float, float]]:
        """Return all unique orthogonal rotations (w, l, h) of this item.

        There are 6 possible rotations for a cuboid. If dimensions are equal,
        some rotations are equivalent and are filtered out.
        """
        dims = (self.w, self.l, self.h)
        rotations = set()
        # All 6 permutations of (w, l, h)
        import itertools
        for perm in itertools.permutations(dims):
            rotations.add(perm)
        return list(rotations)


@dataclass
class Cuboid:
    """Represents a maximal free cuboid space in a bin.

    Defined by two corners: (x_min, y_min, z_min) and (x_max, y_max, z_max).
    Following the paper's convention:
        x = width axis, y = height axis, z = depth/length axis
    """
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def depth(self) -> float:
        return self.z_max - self.z_min

    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth

    def contains(self, other: 'Cuboid') -> bool:
        """Check if this cuboid fully contains another cuboid."""
        return (self.x_min <= other.x_min and self.x_max >= other.x_max and
                self.y_min <= other.y_min and self.y_max >= other.y_max and
                self.z_min <= other.z_min and self.z_max >= other.z_max)

    def intersects(self, other: 'Cuboid') -> bool:
        """Check if two cuboids overlap."""
        return not (self.x_max <= other.x_min or other.x_max <= self.x_min or
                    self.y_max <= other.y_min or other.y_max <= self.y_min or
                    self.z_max <= other.z_min or other.z_max <= self.z_min)

    def can_fit(self, w: float, l: float, h: float) -> bool:
        """Check if an item with given dimensions fits in this cuboid."""
        return (w <= self.width and l <= self.depth and h <= self.height)


@dataclass
class Placement:
    """Records where an item was placed in a bin."""
    item: Item
    x: float
    y: float
    z: float
    w: float  # rotated width
    l: float  # rotated length
    h: float  # rotated height

    @property
    def bounding_box(self) -> Cuboid:
        return Cuboid(self.x, self.y, self.z,
                      self.x + self.w, self.y + self.h, self.z + self.l)


@dataclass
class Bin:
    """Represents a 3D packing bin (pallet)."""
    W: float  # width
    L: float  # length (depth)
    H: float  # height
    placements: List[Placement] = field(default_factory=list)
    maximal_cuboids: List[Cuboid] = field(default_factory=list)
    closed: bool = False

    def __post_init__(self):
        if not self.maximal_cuboids:
            # Initialize with the entire bin as one maximal cuboid
            self.maximal_cuboids = [Cuboid(0, 0, 0, self.W, self.H, self.L)]

    @property
    def packed_volume(self) -> float:
        return sum(p.w * p.l * p.h for p in self.placements)

    @property
    def utilization(self) -> float:
        total = self.W * self.L * self.H
        if total == 0:
            return 0.0
        return self.packed_volume / total

    @property
    def max_height_reached(self) -> float:
        """Maximum y-coordinate reached by any placed item."""
        if not self.placements:
            return 0.0
        return max(p.y + p.h for p in self.placements)


# ==============================================================================
# SECTION 2: Maximal Cuboids Algorithm (MCA)
# ==============================================================================

class MaximalCuboidsAlgorithm:
    """3D extension of MAXRECTS for managing free space in a bin.

    This is the core space management data structure from the paper.
    After placing an item, the affected maximal cuboids are split along
    x, y, z axes, and redundant (fully contained) cuboids are removed.

    Reference: Algorithm 1 in paper, extending Jylanki (2010) MAXRECTS to 3D.
    """

    def __init__(self, bin_obj: Bin):
        self.bin = bin_obj

    def place_item(self, cuboid_idx: int, w: float, l: float, h: float) -> Placement:
        """Place an item at the bottom-left corner of the specified maximal cuboid.

        Args:
            cuboid_idx: Index of the maximal cuboid in bin.maximal_cuboids
            w, l, h: Rotated dimensions of the item

        Returns:
            Placement object recording the placement
        """
        m = self.bin.maximal_cuboids[cuboid_idx]
        x_min, y_min, z_min = m.x_min, m.y_min, m.z_min

        # Create bounding box of the placed item
        item_box = Cuboid(x_min, y_min, z_min,
                          x_min + w, y_min + h, z_min + l)

        # Split all affected maximal cuboids
        new_cuboids = []
        for mc in self.bin.maximal_cuboids:
            if mc.intersects(item_box):
                # This cuboid is affected -- split it
                splits = self._split_cuboid(mc, item_box)
                new_cuboids.extend(splits)
            else:
                # Not affected -- keep as is
                new_cuboids.append(mc)

        # Remove cuboids that are fully contained in another
        self.bin.maximal_cuboids = self._remove_contained(new_cuboids)

        return Placement(
            item=None,  # Will be set by caller
            x=x_min, y=y_min, z=z_min,
            w=w, l=l, h=h
        )

    def _split_cuboid(self, mc: Cuboid, item_box: Cuboid) -> List[Cuboid]:
        """Split a maximal cuboid around a placed item bounding box.

        Generates up to 6 new cuboids (2 per axis: left/right, below/above,
        front/back). Only creates cuboids with positive volume.

        This implements the core of Algorithm 1 lines 10-15.
        """
        results = []

        # Left of item (x-axis, negative direction)
        if item_box.x_min > mc.x_min:
            results.append(Cuboid(
                mc.x_min, mc.y_min, mc.z_min,
                item_box.x_min, mc.y_max, mc.z_max
            ))

        # Right of item (x-axis, positive direction)
        if item_box.x_max < mc.x_max:
            results.append(Cuboid(
                item_box.x_max, mc.y_min, mc.z_min,
                mc.x_max, mc.y_max, mc.z_max
            ))

        # Below item (y-axis, negative direction)
        if item_box.y_min > mc.y_min:
            results.append(Cuboid(
                mc.x_min, mc.y_min, mc.z_min,
                mc.x_max, item_box.y_min, mc.z_max
            ))

        # Above item (y-axis, positive direction)
        if item_box.y_max < mc.y_max:
            results.append(Cuboid(
                mc.x_min, item_box.y_max, mc.z_min,
                mc.x_max, mc.y_max, mc.z_max
            ))

        # In front of item (z-axis, negative direction)
        if item_box.z_min > mc.z_min:
            results.append(Cuboid(
                mc.x_min, mc.y_min, mc.z_min,
                mc.x_max, mc.y_max, item_box.z_min
            ))

        # Behind item (z-axis, positive direction)
        if item_box.z_max < mc.z_max:
            results.append(Cuboid(
                mc.x_min, mc.y_min, item_box.z_max,
                mc.x_max, mc.y_max, mc.z_max
            ))

        # Filter out zero or negative volume cuboids
        results = [c for c in results if c.volume > 1e-9]
        return results

    def _remove_contained(self, cuboids: List[Cuboid]) -> List[Cuboid]:
        """Remove cuboids that are fully contained within another.

        This is the pruning step in Algorithm 1, lines 17-21.
        O(n^2) complexity where n = number of cuboids.
        """
        if not cuboids:
            return []

        # Sort by volume descending for efficiency
        cuboids.sort(key=lambda c: c.volume, reverse=True)
        keep = []

        for i, c_i in enumerate(cuboids):
            contained = False
            for j, c_j in enumerate(cuboids):
                if i != j and c_j.contains(c_i):
                    contained = True
                    break
            if not contained:
                keep.append(c_i)

        return keep

    def get_feasible_actions(self, items: List[Item],
                             height_map: np.ndarray) -> List[dict]:
        """Generate all feasible (item, rotation, cuboid) combinations.

        For each item in the lookahead buffer, for each rotation, for each
        maximal cuboid: check if the item fits and meets stability requirements.

        Args:
            items: List of lookahead items
            height_map: Current height map of the bin (for stability check)

        Returns:
            List of action dicts with keys: item_idx, rotation, cuboid_idx,
            w, l, h (rotated dims), x, y, z (placement position)
        """
        actions = []
        for item_idx, item in enumerate(items):
            for rotation in item.get_rotations():
                w, l, h = rotation
                for cuboid_idx, mc in enumerate(self.bin.maximal_cuboids):
                    if mc.can_fit(w, l, h):
                        x, y, z = mc.x_min, mc.y_min, mc.z_min
                        # Check stability (50% base support)
                        tau = self._stability_check(x, y, z, w, l, height_map)
                        if tau >= 0.5:
                            actions.append({
                                'item_idx': item_idx,
                                'rotation': rotation,
                                'cuboid_idx': cuboid_idx,
                                'w': w, 'l': l, 'h': h,
                                'x': x, 'y': y, 'z': z,
                                'stability': tau
                            })
        return actions

    def _stability_check(self, x: float, y: float, z: float,
                          w: float, l: float,
                          height_map: np.ndarray) -> float:
        """Check what fraction of the item's base is supported.

        Implements Eq. 14 from the paper.
        tau = (supported cells) / (total base cells) >= 0.5

        The height map is a 2D grid where H[i][j] = height of the highest
        point at position (i, j). An item placed at y=y_min is supported at
        (i,j) if height_map[i][j] == y_min (or y_min == 0 for floor).

        Args:
            x, y, z: Bottom-left-back corner of item
            w, l: Width and length of item (base dimensions)
            height_map: 2D numpy array (W x L grid)

        Returns:
            tau: Support ratio [0, 1]
        """
        if y == 0:
            return 1.0  # Floor always provides full support

        # Discretize to grid coordinates
        x_start = int(x)
        z_start = int(z)
        x_end = int(x + w)
        z_end = int(z + l)

        total_cells = 0
        supported_cells = 0

        for xi in range(x_start, x_end):
            for zi in range(z_start, z_end):
                total_cells += 1
                if xi < height_map.shape[0] and zi < height_map.shape[1]:
                    # Cell is supported if height at this position matches
                    # the bottom of the item (within tolerance)
                    if abs(height_map[xi][zi] - y) < 1e-6:
                        supported_cells += 1

        if total_cells == 0:
            return 0.0
        return supported_cells / total_cells


# ==============================================================================
# SECTION 3: Height Map and Action Map Computation
# ==============================================================================

def compute_height_map(bin_obj: Bin, resolution: Tuple[int, int] = None) -> np.ndarray:
    """Compute a top-down height map of the bin.

    Each cell (i, j) contains the normalized height of the highest point
    at that (x, z) position, as a value in [0, 1].

    This is the primary visual state representation for the DQN (Fig. 3a).

    Args:
        bin_obj: The bin to compute height map for
        resolution: (width_cells, depth_cells). If None, uses bin dimensions.

    Returns:
        2D numpy array of shape (resolution) with values in [0, 1]
    """
    if resolution is None:
        resolution = (int(bin_obj.W), int(bin_obj.L))

    hmap = np.zeros(resolution, dtype=np.float32)

    for placement in bin_obj.placements:
        x_start = int(placement.x)
        x_end = int(placement.x + placement.w)
        z_start = int(placement.z)
        z_end = int(placement.z + placement.l)
        top = placement.y + placement.h

        for xi in range(max(0, x_start), min(resolution[0], x_end)):
            for zi in range(max(0, z_start), min(resolution[1], z_end)):
                hmap[xi][zi] = max(hmap[xi][zi], top)

    # Normalize to [0, 1] by bin height
    hmap /= bin_obj.H
    return hmap


def compute_action_map(bin_obj: Bin, action: dict,
                       resolution: Tuple[int, int] = None) -> np.ndarray:
    """Compute the action map for a specific placement action.

    The action map is a binary 2D grid showing where the item would be
    placed (Fig. 3b). This is concatenated with the height map as input
    to the CNN branch of the DQN.

    Args:
        bin_obj: The bin
        action: Action dict from get_feasible_actions
        resolution: Grid resolution

    Returns:
        2D numpy array of shape (resolution) with values 0 or 1
    """
    if resolution is None:
        resolution = (int(bin_obj.W), int(bin_obj.L))

    amap = np.zeros(resolution, dtype=np.float32)

    x_start = int(action['x'])
    x_end = int(action['x'] + action['w'])
    z_start = int(action['z'])
    z_end = int(action['z'] + action['l'])

    for xi in range(max(0, x_start), min(resolution[0], x_end)):
        for zi in range(max(0, z_start), min(resolution[1], z_end)):
            amap[xi][zi] = 1.0

    return amap


# ==============================================================================
# SECTION 4: Reward Function
# ==============================================================================

def compute_reward(bin_obj: Bin) -> float:
    """Compute the composite reward R = (R_pyramid + R_compactness) / 2.

    R_pyramid (Eq. 15): Ratio of packed item volume to the total volume
    under the height map surface. Encourages filling gaps.

    R_compactness (Eq. 16): Ratio of packed item volume to the bounding
    box defined by W * D * H_max. Encourages keeping height low.

    Args:
        bin_obj: The bin after placing an item

    Returns:
        Reward value in [0, 1]
    """
    if not bin_obj.placements:
        return 0.0

    packed_volume = bin_obj.packed_volume
    hmap_raw = compute_height_map(bin_obj) * bin_obj.H  # De-normalize

    # R_pyramid: packed volume / volume under height map surface
    # The volume under the height map is the sum of all height values
    # (since each cell has area 1x1 in the discretized grid)
    occupied_volume = float(np.sum(hmap_raw))
    if occupied_volume > 0:
        r_pyramid = packed_volume / occupied_volume
    else:
        r_pyramid = 0.0

    # R_compactness: packed volume / (W * D * H_max)
    h_max = bin_obj.max_height_reached
    if h_max > 0:
        bounding_volume = bin_obj.W * bin_obj.L * h_max
        r_compactness = packed_volume / bounding_volume
    else:
        r_compactness = 0.0

    return (r_pyramid + r_compactness) / 2.0


# ==============================================================================
# SECTION 5: Enhanced Reward (Extension for Our Use Case)
# ==============================================================================

def compute_enhanced_reward(bin_obj: Bin,
                            stability_weight: float = 0.2,
                            pyramid_weight: float = 0.4,
                            compactness_weight: float = 0.4) -> float:
    """Extended reward function incorporating stability more explicitly.

    EXTENSION BEYOND PAPER: The paper only checks stability as a hard
    constraint (tau >= 0.5). For our use case, we want to reward HIGHER
    stability within the feasible set.

    Components:
        1. R_pyramid: Gap-filling (from paper)
        2. R_compactness: Height minimization (from paper)
        3. R_stability: Average support ratio of all placed items (NEW)

    Args:
        bin_obj: The bin after placing an item
        stability_weight: Weight for stability component
        pyramid_weight: Weight for pyramid component
        compactness_weight: Weight for compactness component

    Returns:
        Weighted reward value
    """
    if not bin_obj.placements:
        return 0.0

    # Original components
    packed_volume = bin_obj.packed_volume
    hmap_raw = compute_height_map(bin_obj) * bin_obj.H

    occupied_volume = float(np.sum(hmap_raw))
    r_pyramid = packed_volume / occupied_volume if occupied_volume > 0 else 0.0

    h_max = bin_obj.max_height_reached
    bounding_volume = bin_obj.W * bin_obj.L * h_max if h_max > 0 else 1.0
    r_compactness = packed_volume / bounding_volume

    # NEW: Stability component
    # Average support ratio across all items (higher = more stable overall)
    if len(bin_obj.placements) > 0:
        hmap_for_stability = compute_height_map(bin_obj) * bin_obj.H
        # This is a simplified version; in practice, you'd compute tau for
        # each item when it was placed and track it
        r_stability = 1.0  # Placeholder: compute from stored tau values
    else:
        r_stability = 0.0

    return (pyramid_weight * r_pyramid +
            compactness_weight * r_compactness +
            stability_weight * r_stability)


# ==============================================================================
# SECTION 6: Double DQN Architecture
# ==============================================================================

"""
PyTorch implementation of the Double DQN for 3D bin packing.

Key architectural decision from the paper (Fig. 5b):
    - Standard DQN: one forward pass -> Q-values for ALL actions
    - This DQN: one forward pass per (state, action) pair -> single Q-value
    - Reason: action space size varies at each timestep

Network architecture (Fig. 6):
    CNN branch: height_map(32,32,1) + action_map(32,32,1) -> concat(32,32,2)
        -> 5x [Conv2D -> ReLU -> BatchNorm]
        -> GlobalAveragePooling -> (256,)
    FC branch: item_list(k-1, 3) -> FC(256)
    Merge: concat(512) -> FC(1000) -> FC(100) -> FC(1) -> Q-value

IMPORTANT: This requires PyTorch. Install with: pip install torch torchvision
"""

# --- PyTorch model definition (uncomment when ready to implement) ---

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
#
#
# class BinPackingDQN(nn.Module):
#     """Double DQN for 3D bin packing with variable action spaces.
#
#     Takes (state, action) pair as input, outputs single Q-value.
#     Must be called once per feasible action to find the best action.
#     """
#
#     def __init__(self, map_size: int = 32, max_items: int = 15):
#         super().__init__()
#         self.map_size = map_size
#
#         # CNN branch for height map + action map
#         # Input: (batch, 2, map_size, map_size)
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(2, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#         )
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # -> (batch, 256, 1, 1)
#
#         # FC branch for item dimensions
#         # Input: (batch, (max_items - 1) * 3) -- remaining items after selection
#         self.item_fc = nn.Sequential(
#             nn.Linear((max_items - 1) * 3, 256),
#             nn.ReLU(),
#         )
#
#         # Merge and output
#         self.merge_fc = nn.Sequential(
#             nn.Linear(512, 1000),
#             nn.ReLU(),
#             nn.Linear(1000, 100),
#             nn.ReLU(),
#             nn.Linear(100, 1),  # Single Q-value output
#         )
#
#     def forward(self, height_map, action_map, item_list):
#         """Forward pass for a single (state, action) pair.
#
#         Args:
#             height_map: (batch, 1, H, W) normalized height map
#             action_map: (batch, 1, H, W) binary action placement map
#             item_list: (batch, (k-1)*3) remaining item dimensions
#
#         Returns:
#             Q-value: (batch, 1)
#         """
#         # Concatenate maps along channel dimension
#         maps = torch.cat([height_map, action_map], dim=1)  # (batch, 2, H, W)
#
#         # CNN branch
#         conv_out = self.conv_layers(maps)
#         conv_out = self.global_avg_pool(conv_out)  # (batch, 256, 1, 1)
#         conv_out = conv_out.view(conv_out.size(0), -1)  # (batch, 256)
#
#         # FC branch
#         item_out = self.item_fc(item_list)  # (batch, 256)
#
#         # Merge
#         merged = torch.cat([conv_out, item_out], dim=1)  # (batch, 512)
#         q_value = self.merge_fc(merged)  # (batch, 1)
#
#         return q_value
#
#
# class ExperienceReplay:
#     """Experience replay buffer for DQN training."""
#
#     def __init__(self, capacity: int = 1_000_000):
#         self.buffer = deque(maxlen=capacity)
#
#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#
#     def sample(self, batch_size: int):
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
#         return states, actions, rewards, next_states, dones
#
#     def __len__(self):
#         return len(self.buffer)


# ==============================================================================
# SECTION 7: Bin Replacement Strategies
# ==============================================================================

class BinReplacementStrategy:
    """Bin replacement strategies for when no feasible action exists.

    In the 2-bounded space scenario, when neither bin can accommodate any
    of the lookahead items, a replacement is triggered.
    """

    @staticmethod
    def replace_all(bins: List[Bin], bin_config: Tuple[float, float, float]
                    ) -> List[Bin]:
        """ReplaceAll: Close all bins and open fresh ones.

        Simple but can waste space in partially-filled bins.
        """
        closed_bins = []
        for b in bins:
            b.closed = True
            closed_bins.append(b)

        W, L, H = bin_config
        new_bins = [Bin(W=W, L=L, H=H) for _ in range(len(bins))]
        return new_bins, closed_bins

    @staticmethod
    def replace_max(bins: List[Bin], bin_config: Tuple[float, float, float]
                    ) -> List[Bin]:
        """ReplaceMax: Close only the bin with highest utilization.

        The less-utilized bin is kept for continued packing.
        This is the PREFERRED strategy per the paper's findings.

        Rationale: The bin with higher utilization has less remaining space,
        so it's less likely to accommodate future items. Keep the one with
        more available space.
        """
        # Find bin with highest utilization
        max_util_idx = max(range(len(bins)), key=lambda i: bins[i].utilization)
        closed_bin = bins[max_util_idx]
        closed_bin.closed = True

        W, L, H = bin_config
        new_bins = list(bins)
        new_bins[max_util_idx] = Bin(W=W, L=L, H=H)
        return new_bins, [closed_bin]

    @staticmethod
    def replace_threshold(bins: List[Bin], bin_config: Tuple[float, float, float],
                          threshold: float = 0.7) -> List[Bin]:
        """EXTENSION: Replace bins above a utilization threshold.

        Not in the original paper. This is a hybrid strategy:
        - Close any bin above the threshold
        - Keep bins below the threshold

        This could be useful when items are heterogeneous and a
        partially-filled bin might still accommodate small items.
        """
        W, L, H = bin_config
        new_bins = []
        closed_bins = []
        for b in bins:
            if b.utilization >= threshold:
                b.closed = True
                closed_bins.append(b)
                new_bins.append(Bin(W=W, L=L, H=H))
            else:
                new_bins.append(b)

        # If no bins were replaced (all below threshold), force replace the fullest
        if not closed_bins:
            return BinReplacementStrategy.replace_max(bins, bin_config)

        return new_bins, closed_bins


# ==============================================================================
# SECTION 8: Full Training Loop Skeleton
# ==============================================================================

class OnlineDualBinPackingEnv:
    """Environment for the Online 3D Dual-Bin Packing Problem.

    This wraps the bin packing logic into an RL environment interface
    compatible with the Double DQN training loop.

    State: (height_maps, action_maps, item_lists) for each bin
    Action: (bin_idx, item_idx, rotation_idx, cuboid_idx)
    Reward: composite (pyramid + compactness)
    """

    def __init__(self, bin_config: Tuple[float, float, float],
                 n_bins: int = 2,
                 lookahead_k: int = 10,
                 item_generator=None,
                 replacement_strategy: str = 'replaceMax',
                 map_resolution: int = 32):
        self.bin_config = bin_config
        self.n_bins = n_bins
        self.k = lookahead_k
        self.item_generator = item_generator or self._default_item_generator
        self.replacement_strategy = replacement_strategy
        self.map_resolution = map_resolution

        self.bins = []
        self.mca_managers = []
        self.item_queue = []
        self.completed_bins = []
        self.step_count = 0

    def _default_item_generator(self, n: int = 200) -> List[Item]:
        """Generate random items as in the paper: dims in [6, 12]."""
        items = []
        for i in range(n):
            w = random.randint(6, 12)
            l = random.randint(6, 12)
            h = random.randint(6, 12)
            items.append(Item(id=i, w=w, l=l, h=h))
        return items

    def reset(self) -> dict:
        """Reset environment for a new episode."""
        W, L, H = self.bin_config
        self.bins = [Bin(W=W, L=L, H=H) for _ in range(self.n_bins)]
        self.mca_managers = [MaximalCuboidsAlgorithm(b) for b in self.bins]
        self.item_queue = self.item_generator()
        random.shuffle(self.item_queue)  # Randomize arrival order
        self.completed_bins = []
        self.step_count = 0
        return self._get_state()

    def _get_lookahead_items(self) -> List[Item]:
        """Get the next k items from the queue."""
        return self.item_queue[:self.k]

    def _get_state(self) -> dict:
        """Compute the full state representation."""
        resolution = (self.map_resolution, self.map_resolution)
        state = {
            'height_maps': [],
            'lookahead_items': self._get_lookahead_items(),
            'feasible_actions': [],
        }

        for bin_idx, (b, mca) in enumerate(zip(self.bins, self.mca_managers)):
            hmap = compute_height_map(b, resolution)
            state['height_maps'].append(hmap)

            # Get feasible actions for this bin
            actions = mca.get_feasible_actions(
                state['lookahead_items'], hmap * b.H  # De-normalize for stability
            )
            for a in actions:
                a['bin_idx'] = bin_idx
            state['feasible_actions'].extend(actions)

        return state

    def step(self, action: dict) -> Tuple[dict, float, bool, dict]:
        """Execute one packing action.

        Args:
            action: Dict with bin_idx, item_idx, rotation, cuboid_idx, etc.

        Returns:
            next_state, reward, done, info
        """
        bin_idx = action['bin_idx']
        item_idx = action['item_idx']
        w, l, h = action['w'], action['l'], action['h']
        cuboid_idx = action['cuboid_idx']

        # Execute placement
        b = self.bins[bin_idx]
        mca = self.mca_managers[bin_idx]
        placement = mca.place_item(cuboid_idx, w, l, h)
        placement.item = self.item_queue[item_idx]
        b.placements.append(placement)

        # Remove the packed item from the queue
        self.item_queue.pop(item_idx)

        # Compute reward
        reward = compute_reward(b)

        # Check if done
        self.step_count += 1
        done = len(self.item_queue) == 0

        # Get next state
        next_state = self._get_state()

        # If no feasible actions in next state, trigger bin replacement
        if not done and len(next_state['feasible_actions']) == 0:
            self._replace_bins()
            next_state = self._get_state()

            # If still no feasible actions after replacement, episode is done
            if len(next_state['feasible_actions']) == 0:
                done = True

        info = {
            'step': self.step_count,
            'bin_utilizations': [b.utilization for b in self.bins],
            'completed_bins': len(self.completed_bins),
            'items_remaining': len(self.item_queue),
        }

        return next_state, reward, done, info

    def _replace_bins(self):
        """Trigger bin replacement based on configured strategy."""
        if self.replacement_strategy == 'replaceAll':
            new_bins, closed = BinReplacementStrategy.replace_all(
                self.bins, self.bin_config)
        elif self.replacement_strategy == 'replaceMax':
            new_bins, closed = BinReplacementStrategy.replace_max(
                self.bins, self.bin_config)
        elif self.replacement_strategy == 'replaceThreshold':
            new_bins, closed = BinReplacementStrategy.replace_threshold(
                self.bins, self.bin_config, threshold=0.7)
        else:
            raise ValueError(f"Unknown strategy: {self.replacement_strategy}")

        self.completed_bins.extend(closed)
        self.bins = new_bins
        self.mca_managers = [MaximalCuboidsAlgorithm(b) for b in self.bins]


# ==============================================================================
# SECTION 9: Training Loop Pseudocode
# ==============================================================================

def train_ddqn_dual_bin_packing():
    """Pseudocode for the full DDQN training loop.

    This follows the paper's training configuration:
        - 100 iterations x 1000 episodes
        - Discount factor: 0.95
        - Replay memory: 1,000,000
        - Epsilon: 1.0 -> 0.05 (decay 0.99 per epoch)
        - Learning rate: 1e-3 with decay

    Implementation requires PyTorch. Uncomment the model code above first.
    """

    # --- Configuration ---
    BIN_CONFIG = (32, 32, 32)  # (W, L, H) -- paper uses 32^3
    # For real pallets: (120, 80, 150) or similar
    N_BINS = 2
    LOOKAHEAD_K = 10  # Sweet spot per paper
    REPLACEMENT_STRATEGY = 'replaceMax'
    MAP_RESOLUTION = 32  # Match bin width for 1:1 mapping

    # --- Hyperparameters from paper ---
    GAMMA = 0.95
    REPLAY_CAPACITY = 1_000_000
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.99  # Per epoch
    WARMUP_EPOCHS = 20
    WARMUP_LR = 1e-5
    MAIN_LR = 1e-3
    LR_DECAY_STEP = 10_000
    LR_DECAY_AMOUNT = 1e-5
    TARGET_UPDATE_FREQ = 10  # Update target network every 10 epochs
    N_ITERATIONS = 100
    N_EPISODES_PER_ITER = 1000
    BATCH_SIZE = 32

    print("=" * 60)
    print("DDQN Dual-Bin Packing Training")
    print(f"Bins: {N_BINS}, Lookahead: {LOOKAHEAD_K}")
    print(f"Strategy: {REPLACEMENT_STRATEGY}")
    print(f"Iterations: {N_ITERATIONS} x {N_EPISODES_PER_ITER} episodes")
    print("=" * 60)

    # --- Initialize ---
    env = OnlineDualBinPackingEnv(
        bin_config=BIN_CONFIG,
        n_bins=N_BINS,
        lookahead_k=LOOKAHEAD_K,
        replacement_strategy=REPLACEMENT_STRATEGY,
        map_resolution=MAP_RESOLUTION,
    )

    # Uncomment when PyTorch model is ready:
    # policy_net = BinPackingDQN(map_size=MAP_RESOLUTION, max_items=LOOKAHEAD_K)
    # target_net = BinPackingDQN(map_size=MAP_RESOLUTION, max_items=LOOKAHEAD_K)
    # target_net.load_state_dict(policy_net.state_dict())
    # target_net.eval()
    # optimizer = optim.Adam(policy_net.parameters(), lr=WARMUP_LR)
    # replay_buffer = ExperienceReplay(capacity=REPLAY_CAPACITY)

    epsilon = EPSILON_START
    total_epochs = 0

    for iteration in range(N_ITERATIONS):
        for episode in range(N_EPISODES_PER_ITER):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    # Random action from feasible set
                    if state['feasible_actions']:
                        action = random.choice(state['feasible_actions'])
                    else:
                        break  # No feasible actions
                else:
                    # Greedy action: evaluate Q-value for each feasible action
                    # For each action in state['feasible_actions']:
                    #     hmap = state['height_maps'][action['bin_idx']]
                    #     amap = compute_action_map(bins[action['bin_idx']], action)
                    #     items = encode_remaining_items(state, action)
                    #     q = policy_net(hmap, amap, items)
                    #     track best q
                    # action = action_with_max_q
                    if state['feasible_actions']:
                        action = random.choice(state['feasible_actions'])  # Placeholder
                    else:
                        break

                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                # Store experience
                # replay_buffer.push(state, action, reward, next_state, done)

                # Train on mini-batch
                # if len(replay_buffer) >= BATCH_SIZE:
                #     batch = replay_buffer.sample(BATCH_SIZE)
                #     loss = compute_ddqn_loss(policy_net, target_net, batch, GAMMA)
                #     optimizer.zero_grad()
                #     loss.backward()
                #     optimizer.step()

                state = next_state

            total_epochs += 1

            # Epsilon decay
            if total_epochs > WARMUP_EPOCHS:
                epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

            # Target network update
            # if total_epochs % TARGET_UPDATE_FREQ == 0:
            #     target_net.load_state_dict(policy_net.state_dict())

            # Learning rate scheduling
            # if total_epochs == WARMUP_EPOCHS:
            #     for pg in optimizer.param_groups:
            #         pg['lr'] = MAIN_LR
            # elif total_epochs > WARMUP_EPOCHS and total_epochs % LR_DECAY_STEP == 0:
            #     for pg in optimizer.param_groups:
            #         pg['lr'] = max(EPSILON_MIN, pg['lr'] - LR_DECAY_AMOUNT)

        # Log iteration results
        print(f"Iteration {iteration+1}/{N_ITERATIONS}, "
              f"Epsilon: {epsilon:.4f}")

    print("Training complete.")
    # torch.save(policy_net.state_dict(), 'ddqn_dual_bin_packing.pth')


# ==============================================================================
# SECTION 10: Adaptation Notes for Our Specific Use Case
# ==============================================================================

"""
ADAPTATION GUIDE: Tsang et al. 2025 -> Our Thesis Setup

1. BIN DIMENSIONS
   Paper: 32 x 32 x 32 (cubic, small)
   Ours: Likely EUR pallet 120 x 80 x ~150 cm
   Change: Update BIN_CONFIG and MAP_RESOLUTION
   Issue: 120x80 height map is much larger than 32x32 -> CNN needs more
   capacity or use downsampled resolution (e.g., 60x40 at 2cm precision)

2. ITEM SIZE DISTRIBUTION
   Paper: Uniform [6, 12] in each dimension
   Ours: Real distribution from warehouse data (likely wider range)
   Change: Replace _default_item_generator with real item distributions
   Issue: Very small items may create many maximal cuboids -> slow

3. STABILITY ENHANCEMENT
   Paper: 50% base support only
   Ours: Need weight-aware stability, center of gravity tracking
   Change:
   - Add weight field to Item dataclass (already included above)
   - Implement center of gravity calculation per bin
   - Add weight-on-weight stacking constraint
   - Consider dynamic stability (lateral forces during transport)
   - Augment reward function with stability component (see compute_enhanced_reward)

4. BUFFER / LOOKAHEAD
   Paper: k = 5, 10, 15 items visible
   Ours: Buffer of 5-10 boxes (physical staging area)
   Alignment: PERFECT for k=5-10 (paper's sweet spot)
   Possible enhancement: Buffer items can be chosen in any order (not just FIFO)
   -> This is already how the paper models it (choose any of k items)

5. 2-BOUNDED SPACE
   Paper: Exactly 2 bins active
   Ours: Exactly 2 pallets active
   Alignment: PERFECT
   Enhancement: Use replaceMax strategy (paper's recommendation)

6. COMPUTATIONAL SPEED
   Paper: Each action needs a DQN forward pass. With k=10, |M|~50:
          ~3000 forward passes per step on GPU
   Ours: May need real-time decisions (< 1 second per item)
   Mitigation:
   - Use GPU inference (NVIDIA on robot control computer)
   - Batch all (state, action) pairs into one GPU call
   - Pre-filter actions more aggressively (e.g., only top-5 cuboids by volume)
   - Consider using a simpler/smaller network for deployment

7. HYBRID APPROACH (RECOMMENDED)
   Instead of pure DRL, consider:
   - Use DRL to select WHICH ITEM and WHICH BIN
   - Use heuristic (DBLF, BSSF) to select WHERE in the chosen bin
   - This reduces action space from O(6*k*|M|) to O(k*n_bins)
   - Much faster inference, potentially more robust

8. INTEGRATION WITH OTHER METHODS
   - MCA (Section 2) is standalone and useful regardless of DRL/heuristic choice
   - Height map computation (Section 3) is useful for any visualization/state repr.
   - Bin replacement strategies (Section 7) are applicable with any packing method
   - The reward design (Section 4-5) can guide heuristic scoring functions too

9. SUGGESTED IMPLEMENTATION PRIORITY
   Phase 1: Implement MCA + heuristic baselines (BL, BVF, BSSF, BLSF) [1 week]
   Phase 2: Implement environment wrapper with buffer + dual bin [1 week]
   Phase 3: Implement DDQN + training loop [2 weeks]
   Phase 4: Add enhanced stability [1 week]
   Phase 5: Compare DRL vs heuristics vs hybrid [1 week]
   Phase 6: Tune for real pallet dimensions and item distributions [1 week]
"""

# ==============================================================================
# Quick test / demo
# ==============================================================================

if __name__ == "__main__":
    print("Tsang et al. 2025 - Online 3D Dual-Bin Packing")
    print("=" * 50)

    # Create a bin
    b = Bin(W=32, L=32, H=32)
    mca = MaximalCuboidsAlgorithm(b)

    print(f"Initial maximal cuboids: {len(b.maximal_cuboids)}")
    print(f"Cuboid 0: ({b.maximal_cuboids[0].x_min}, {b.maximal_cuboids[0].y_min}, "
          f"{b.maximal_cuboids[0].z_min}) -> ({b.maximal_cuboids[0].x_max}, "
          f"{b.maximal_cuboids[0].y_max}, {b.maximal_cuboids[0].z_max})")

    # Place a few items
    items = [
        Item(id=0, w=10, l=10, h=8),
        Item(id=1, w=8, l=12, h=6),
        Item(id=2, w=6, l=6, h=10),
    ]

    hmap = compute_height_map(b)
    for item in items:
        actions = mca.get_feasible_actions([item], hmap * b.H)
        if actions:
            action = actions[0]  # Take first feasible action
            placement = mca.place_item(action['cuboid_idx'],
                                       action['w'], action['l'], action['h'])
            placement.item = item
            b.placements.append(placement)
            hmap = compute_height_map(b)
            reward = compute_reward(b)
            print(f"Placed item {item.id} ({item.w}x{item.l}x{item.h}) at "
                  f"({placement.x}, {placement.y}, {placement.z}) "
                  f"rotated ({placement.w}x{placement.l}x{placement.h}) | "
                  f"Util: {b.utilization:.2%} | Reward: {reward:.4f} | "
                  f"Maximal cuboids: {len(b.maximal_cuboids)}")
        else:
            print(f"No feasible placement for item {item.id}")

    print(f"\nFinal utilization: {b.utilization:.2%}")
    print(f"Completed placing {len(b.placements)}/{len(items)} items")


# ==============================================================================
# SECTION 11: EUR Pallet Adaptation (120x80x150cm)
# ==============================================================================

class EURPalletConfig:
    """Configuration for European standard pallet (EUR/EPAL).

    The paper uses 32x32x32 bins. Our thesis uses real pallets.
    This class encapsulates the adaptation from the paper's
    normalized bins to physical pallet dimensions.

    Key differences from paper:
        - Rectangular base (120x80) vs square (32x32)
        - Much taller (150cm) vs cubic (32)
        - Items range from ~5x5x5 to 60x40x40cm
        - Need 2cm discretization for height map (60x40 grid)
    """

    # Pallet dimensions in centimeters
    WIDTH = 120       # x-axis (cm)
    LENGTH = 80       # z-axis (cm)
    HEIGHT = 150      # y-axis (cm)

    # Height map resolution (2cm discretization)
    MAP_WIDTH = 60    # WIDTH / 2
    MAP_LENGTH = 40   # LENGTH / 2
    CELL_SIZE = 2.0   # cm per grid cell

    # Maximum item dimensions (typical warehouse items)
    MAX_ITEM_DIM = 60   # cm
    MIN_ITEM_DIM = 5    # cm

    # Typical weight range
    MAX_ITEM_WEIGHT = 30.0   # kg
    MIN_ITEM_WEIGHT = 0.1    # kg

    # Maximum pallet load
    MAX_TOTAL_WEIGHT = 1000.0  # kg (EUR pallet rated for 1500kg static)

    @classmethod
    def create_bin(cls) -> Bin:
        """Create a Bin object with EUR pallet dimensions."""
        return Bin(W=cls.WIDTH, L=cls.LENGTH, H=cls.HEIGHT)

    @classmethod
    def compute_height_map(cls, bin_obj: Bin) -> np.ndarray:
        """Compute height map at 2cm resolution for EUR pallet.

        Returns a (60, 40) normalized array suitable for CNN input.
        """
        return compute_height_map(bin_obj,
                                  resolution=(cls.MAP_WIDTH, cls.MAP_LENGTH))

    @classmethod
    def compute_action_map(cls, bin_obj: Bin, action: dict) -> np.ndarray:
        """Compute action map at 2cm resolution for EUR pallet.

        Must account for the cell size when mapping physical
        coordinates to grid coordinates.
        """
        resolution = (cls.MAP_WIDTH, cls.MAP_LENGTH)
        amap = np.zeros(resolution, dtype=np.float32)

        # Convert physical coords to grid coords
        x_start = int(action['x'] / cls.CELL_SIZE)
        x_end = int((action['x'] + action['w']) / cls.CELL_SIZE)
        z_start = int(action['z'] / cls.CELL_SIZE)
        z_end = int((action['z'] + action['l']) / cls.CELL_SIZE)

        for xi in range(max(0, x_start), min(resolution[0], x_end)):
            for zi in range(max(0, z_start), min(resolution[1], z_end)):
                amap[xi][zi] = 1.0
        return amap


# ==============================================================================
# SECTION 12: Enhanced Stability via LBCP (Largest Base Contact Polygon)
# ==============================================================================

class LBCPStabilityChecker:
    """Enhanced stability checking beyond the paper's 50% base support.

    The paper checks: tau = (supported_area / total_area) >= 0.5
    This class adds:
        1. Center of gravity (CoG) must project within support polygon
        2. Weight capacity: items below must support the weight above
        3. Surface flatness: reward flat top surfaces

    Based on concepts from:
        - Ramos et al. (2016): LBCP stability model
        - One4Many-StablePacker (2025): height difference metric
        - Zhao et al. (2022): feasibility mask for stability

    For our thesis: integrate this as BOTH a hard constraint (feasibility
    filter) and a soft reward component.
    """

    def __init__(self, min_support_ratio: float = 0.7,
                 require_cg_in_polygon: bool = True,
                 max_weight_per_unit_area: float = 50.0):
        """
        Args:
            min_support_ratio: Minimum base area supported (paper: 0.5, we use 0.7)
            require_cg_in_polygon: Whether CoG must project within support polygon
            max_weight_per_unit_area: Maximum weight (kg) per cm^2 for stacking
        """
        self.min_support_ratio = min_support_ratio
        self.require_cg_in_polygon = require_cg_in_polygon
        self.max_weight_per_unit_area = max_weight_per_unit_area

    def check_feasibility(self, item: Item, position: Tuple[float, float, float],
                           bin_obj: Bin,
                           height_map_raw: np.ndarray) -> Tuple[bool, float]:
        """Full stability feasibility check.

        Returns:
            (is_feasible, stability_score) where:
                - is_feasible: boolean, hard constraint
                - stability_score: float in [0,1], for reward shaping
        """
        x, y, z = position
        w, l, h = item.w, item.l, item.h

        # 1. Base support ratio (from paper, Eq. 14)
        tau = self._base_support_ratio(x, y, z, w, l, height_map_raw)
        if tau < self.min_support_ratio:
            return False, tau

        # 2. Center of gravity check
        if self.require_cg_in_polygon and y > 0:
            cg_ok = self._check_cg_in_support(x, y, z, w, l, h,
                                               item.weight, height_map_raw)
            if not cg_ok:
                return False, tau * 0.5  # Penalize but keep score info

        # 3. Weight capacity check
        weight_ok = self._check_weight_capacity(x, y, z, w, l,
                                                 item.weight, bin_obj)
        if not weight_ok:
            return False, tau * 0.3

        # Compute composite stability score
        stability_score = tau  # Could be extended with CoG margin, etc.
        return True, stability_score

    def _base_support_ratio(self, x: float, y: float, z: float,
                             w: float, l: float,
                             height_map: np.ndarray) -> float:
        """Same as paper's Eq. 14 but with configurable threshold."""
        if y == 0:
            return 1.0
        x_start, z_start = int(x), int(z)
        x_end, z_end = int(x + w), int(z + l)
        total = 0
        supported = 0
        for xi in range(x_start, x_end):
            for zi in range(z_start, z_end):
                total += 1
                if (xi < height_map.shape[0] and zi < height_map.shape[1]):
                    if abs(height_map[xi][zi] - y) < 1e-6:
                        supported += 1
        return supported / max(total, 1)

    def _check_cg_in_support(self, x, y, z, w, l, h, weight,
                              height_map) -> bool:
        """Check if center of gravity projects within the support polygon.

        For a uniform-density item, CoG is at the geometric center.
        The support polygon is the convex hull of supported cells.
        Simplified: CoG must be within the bounding box of supported cells.
        """
        cg_x = x + w / 2.0
        cg_z = z + l / 2.0

        # Find support region bounds
        sup_x_min, sup_x_max = x + w, x  # Initialize inverted
        sup_z_min, sup_z_max = z + l, z
        has_support = False

        for xi in range(int(x), int(x + w)):
            for zi in range(int(z), int(z + l)):
                if (xi < height_map.shape[0] and zi < height_map.shape[1]):
                    if abs(height_map[xi][zi] - y) < 1e-6:
                        sup_x_min = min(sup_x_min, xi)
                        sup_x_max = max(sup_x_max, xi + 1)
                        sup_z_min = min(sup_z_min, zi)
                        sup_z_max = max(sup_z_max, zi + 1)
                        has_support = True

        if not has_support:
            return y == 0  # Only floor placement is okay without support

        # CoG must be within support bounding box
        return (sup_x_min <= cg_x <= sup_x_max and
                sup_z_min <= cg_z <= sup_z_max)

    def _check_weight_capacity(self, x, y, z, w, l, weight, bin_obj) -> bool:
        """Check if items below can support this item's weight.

        Simplified: compute total weight above each placed item's footprint
        and ensure it doesn't exceed the weight capacity.
        """
        # For now, always return True (placeholder for weight-aware logic)
        # Full implementation would track cumulative weight per surface cell
        return True

    def compute_surface_flatness(self, bin_obj: Bin) -> float:
        """Compute how flat the top surface of the bin is.

        From One4Many-StablePacker: reward flat surfaces.
        Returns value in [0, 1] where 1 = perfectly flat.
        """
        hmap = compute_height_map(bin_obj)
        if np.max(hmap) == 0:
            return 1.0  # Empty bin is trivially flat
        # Only consider non-zero cells
        nonzero = hmap[hmap > 0]
        if len(nonzero) == 0:
            return 1.0
        variance = np.var(nonzero)
        # Normalize: max possible variance is 0.25 (half empty, half full)
        flatness = max(0.0, 1.0 - variance / 0.25)
        return flatness


# ==============================================================================
# SECTION 13: DeepPack3D Integration Guide
# ==============================================================================

"""
DEEPPACK3D INTEGRATION
======================

The authors released DeepPack3D as an open-source Python package:
    GitHub: https://github.com/SoftwareImpacts/SIMPAC-2024-311
    Paper:  Software Impacts, Volume 23, 2025, Article 100732
    Python: 3.10
    DRL:    TensorFlow 2.10.0

WHAT DEEPPACK3D PROVIDES:
    1. Complete MCA implementation (maximal cuboids algorithm)
    2. Double DQN training and inference pipeline
    3. Four heuristic baselines (BL, BVF, BSSF, BLSF)
    4. Visualization tools for packing results
    5. Data loading utilities

HOW TO USE FOR OUR THESIS:
    Option A: Fork and modify
        - Clone the repo
        - Modify bin dimensions (32^3 -> 120x80x150)
        - Add enhanced stability checks
        - Port from TensorFlow to PyTorch (if desired)
        - Add dual-bin management with replacement strategies

    Option B: Re-implement core components
        - Use DeepPack3D as reference for MCA algorithm
        - Implement our own DQN in PyTorch
        - Take their reward function as starting point
        - Implement our own enhanced features

    Option C: Hybrid
        - Use DeepPack3D's MCA and heuristics directly
        - Build our own DRL layer on top in PyTorch
        - Use their visualization for comparison

RECOMMENDED: Option C. The MCA is the most tedious to implement correctly,
and DeepPack3D's implementation is tested. Build the DRL and enhanced
stability on top.

KEY FILES IN DEEPPACK3D REPO:
    - MCA implementation: check their space management module
    - DQN model: check their RL agent module
    - Heuristics: check their heuristic module
    - Environment: check their environment wrapper

PORTING NOTES (TensorFlow -> PyTorch):
    - tf.keras.layers.Conv2D -> torch.nn.Conv2d
    - tf.keras.layers.Dense -> torch.nn.Linear
    - tf.keras.layers.BatchNormalization -> torch.nn.BatchNorm2d
    - tf.keras.layers.GlobalAveragePooling2D -> torch.nn.AdaptiveAvgPool2d(1)
    - Optimizer: tf.keras.optimizers.Adam -> torch.optim.Adam
    - Experience replay is framework-agnostic (just a deque)
"""


# ==============================================================================
# SECTION 14: Complete Adapted System for EUR Pallet
# ==============================================================================

class AdaptedDualPalletSystem:
    """Complete system adapted for EUR pallet thesis scenario.

    Combines:
        - MCA for free space management (from paper)
        - Enhanced stability checking (LBCP, from our enhancement)
        - ReplaceMax bin replacement (from paper)
        - Lookahead buffer management (from paper)
        - Height map state representation (from paper, scaled to 60x40)

    This class orchestrates the full packing pipeline for evaluation
    and can serve as the RL environment for training.

    Usage:
        system = AdaptedDualPalletSystem(buffer_size=10)
        system.load_items(item_list)
        while system.has_items():
            state = system.get_state()
            action = agent.select_action(state)  # DRL or heuristic
            reward, done, info = system.step(action)
    """

    def __init__(self, buffer_size: int = 10,
                 n_pallets: int = 2,
                 replacement_strategy: str = 'replaceMax',
                 stability_threshold: float = 0.7):
        self.buffer_size = buffer_size
        self.n_pallets = n_pallets
        self.replacement_strategy = replacement_strategy
        self.stability_checker = LBCPStabilityChecker(
            min_support_ratio=stability_threshold)

        self.pallets = []
        self.mca_managers = []
        self.item_queue = []
        self.completed_pallets = []
        self.step_count = 0

    def reset(self, items: List[Item] = None):
        """Reset the system for a new episode.

        Args:
            items: Optional list of items. If None, generates random items.
        """
        self.pallets = [EURPalletConfig.create_bin() for _ in range(self.n_pallets)]
        self.mca_managers = [MaximalCuboidsAlgorithm(p) for p in self.pallets]
        self.completed_pallets = []
        self.step_count = 0

        if items is not None:
            self.item_queue = list(items)
        else:
            self.item_queue = self._generate_warehouse_items(200)
            random.shuffle(self.item_queue)

        return self._get_state()

    def _generate_warehouse_items(self, n: int) -> List[Item]:
        """Generate items from a realistic warehouse distribution.

        Instead of uniform [6,12] as in the paper, uses a distribution
        that mimics real e-commerce warehouses:
            - Small items (5-15cm): 40% of items
            - Medium items (15-35cm): 40% of items
            - Large items (35-60cm): 20% of items
        """
        items = []
        for i in range(n):
            r = random.random()
            if r < 0.4:  # Small
                dims = [random.randint(5, 15) for _ in range(3)]
                weight = random.uniform(0.1, 3.0)
            elif r < 0.8:  # Medium
                dims = [random.randint(15, 35) for _ in range(3)]
                weight = random.uniform(1.0, 10.0)
            else:  # Large
                dims = [random.randint(25, 60) for _ in range(3)]
                weight = random.uniform(5.0, 30.0)
            items.append(Item(id=i, w=dims[0], l=dims[1], h=dims[2],
                              weight=weight))
        return items

    def _get_lookahead(self) -> List[Item]:
        """Get the current lookahead buffer (first k items)."""
        return self.item_queue[:self.buffer_size]

    def _get_state(self) -> dict:
        """Build the full state representation for DRL or heuristic."""
        state = {
            'height_maps': [],
            'lookahead_items': self._get_lookahead(),
            'feasible_actions': [],
            'pallet_utilizations': [],
        }

        for pidx, (pallet, mca) in enumerate(zip(self.pallets, self.mca_managers)):
            hmap = EURPalletConfig.compute_height_map(pallet)
            state['height_maps'].append(hmap)
            state['pallet_utilizations'].append(pallet.utilization)

            # Generate feasible actions with enhanced stability
            hmap_raw = hmap * EURPalletConfig.HEIGHT
            for item_idx, item in enumerate(state['lookahead_items']):
                for rotation in item.get_rotations():
                    w, l, h = rotation
                    for cidx, mc in enumerate(pallet.maximal_cuboids):
                        if mc.can_fit(w, l, h):
                            pos = (mc.x_min, mc.y_min, mc.z_min)
                            feasible, stab_score = \
                                self.stability_checker.check_feasibility(
                                    Item(id=item.id, w=w, l=l, h=h,
                                         weight=item.weight),
                                    pos, pallet, hmap_raw)
                            if feasible:
                                state['feasible_actions'].append({
                                    'pallet_idx': pidx,
                                    'item_idx': item_idx,
                                    'rotation': rotation,
                                    'cuboid_idx': cidx,
                                    'w': w, 'l': l, 'h': h,
                                    'x': mc.x_min, 'y': mc.y_min,
                                    'z': mc.z_min,
                                    'stability': stab_score,
                                })
        return state

    def step(self, action: dict) -> Tuple[float, bool, dict]:
        """Execute a packing action.

        Returns: (reward, done, info)
        """
        pidx = action['pallet_idx']
        item_idx = action['item_idx']
        w, l, h = action['w'], action['l'], action['h']
        cidx = action['cuboid_idx']

        pallet = self.pallets[pidx]
        mca = self.mca_managers[pidx]

        # Place item
        placement = mca.place_item(cidx, w, l, h)
        placement.item = self.item_queue[item_idx]
        pallet.placements.append(placement)

        # Remove from queue
        self.item_queue.pop(item_idx)
        self.step_count += 1

        # Compute enhanced reward
        r_pyramid = self._compute_pyramid(pallet)
        r_compactness = self._compute_compactness(pallet)
        r_stability = action.get('stability', 0.5)
        r_flatness = self.stability_checker.compute_surface_flatness(pallet)

        reward = (0.30 * r_pyramid + 0.25 * r_compactness +
                  0.25 * r_stability + 0.20 * r_flatness)

        done = len(self.item_queue) == 0

        # Check for replacement trigger
        if not done:
            next_state = self._get_state()
            if len(next_state['feasible_actions']) == 0:
                self._trigger_replacement()
                next_state = self._get_state()
                if len(next_state['feasible_actions']) == 0:
                    done = True

        info = {
            'step': self.step_count,
            'pallet_utils': [p.utilization for p in self.pallets],
            'completed_count': len(self.completed_pallets),
            'remaining_items': len(self.item_queue),
            'reward_components': {
                'pyramid': r_pyramid,
                'compactness': r_compactness,
                'stability': r_stability,
                'flatness': r_flatness,
            }
        }
        return reward, done, info

    def _compute_pyramid(self, pallet: Bin) -> float:
        """R_pyramid from Eq. 15."""
        if not pallet.placements:
            return 0.0
        packed_vol = pallet.packed_volume
        hmap_raw = compute_height_map(pallet) * pallet.H
        occ_vol = float(np.sum(hmap_raw))
        return packed_vol / occ_vol if occ_vol > 0 else 0.0

    def _compute_compactness(self, pallet: Bin) -> float:
        """R_compactness from Eq. 16."""
        if not pallet.placements:
            return 0.0
        packed_vol = pallet.packed_volume
        h_max = pallet.max_height_reached
        bb_vol = pallet.W * pallet.L * h_max if h_max > 0 else 1.0
        return packed_vol / bb_vol

    def _trigger_replacement(self):
        """Execute bin replacement using configured strategy."""
        if self.replacement_strategy == 'replaceMax':
            max_idx = max(range(len(self.pallets)),
                          key=lambda i: self.pallets[i].utilization)
            self.completed_pallets.append(self.pallets[max_idx])
            self.pallets[max_idx] = EURPalletConfig.create_bin()
            self.mca_managers[max_idx] = MaximalCuboidsAlgorithm(
                self.pallets[max_idx])
        elif self.replacement_strategy == 'replaceAll':
            self.completed_pallets.extend(self.pallets)
            self.pallets = [EURPalletConfig.create_bin()
                            for _ in range(self.n_pallets)]
            self.mca_managers = [MaximalCuboidsAlgorithm(p)
                                 for p in self.pallets]

    def get_final_statistics(self) -> dict:
        """Get statistics after episode completes."""
        all_pallets = self.completed_pallets + self.pallets
        utils = [p.utilization for p in all_pallets if p.placements]
        return {
            'avg_utilization': np.mean(utils) if utils else 0.0,
            'std_utilization': np.std(utils) if utils else 0.0,
            'min_utilization': np.min(utils) if utils else 0.0,
            'max_utilization': np.max(utils) if utils else 0.0,
            'num_pallets_used': len(utils),
            'total_items_packed': sum(len(p.placements) for p in all_pallets),
            'total_items_queued': self.step_count + len(self.item_queue),
        }


# ==============================================================================
# SECTION 15: Batched DQN Inference for Real-Time Performance
# ==============================================================================

"""
PERFORMANCE OPTIMIZATION FOR REAL-TIME DEPLOYMENT

Problem: The paper's DQN evaluates each (state, action) pair individually.
With k=10 and |M|=50, that is ~3000 forward passes per step.
For EUR pallets with |M|~200: up to 12,000 forward passes.

Solution: BATCH all forward passes into a single GPU call.

Instead of:
    for action in feasible_actions:
        q = dqn.forward(hmap, action_map, items)

Do:
    batch_hmaps = stack([hmap] * len(feasible_actions))
    batch_amaps = stack([compute_action_map(a) for a in feasible_actions])
    batch_items = stack([encode_items(a) for a in feasible_actions])
    q_values = dqn.forward(batch_hmaps, batch_amaps, batch_items)

This converts O(|A|) sequential GPU calls into ONE batched call.
On an RTX 3060: ~12,000 samples/batch at ~50ms.

ADDITIONAL OPTIMIZATIONS:
    1. Pre-filter: only evaluate top-50 cuboids by volume
    2. Rotation pruning: only evaluate 3 rotations (skip if 2 dims equal)
    3. Action pruning: skip actions where item volume < 10% of cuboid volume
    4. Cache height maps: only recompute after placement, not per action

ESTIMATED THROUGHPUT:
    Paper setup (32^3, k=10): ~3000 actions, ~30ms per step (GPU)
    Our setup (120x80, k=10): ~6000 actions, ~100ms per step (GPU)
    With pre-filtering: ~1500 actions, ~25ms per step (GPU)

    TARGET: < 500ms per step (suitable for robotic arm cycle time)
"""
