"""
Deep-Pack 3D Extension: Coding Ideas for Vision-Based DRL Online Bin Packing
=============================================================================

Based on: Kundu, Dutta & Kumar (2019) "Deep-Pack: A Vision-Based 2D Online
Bin Packing Algorithm with Deep Reinforcement Learning"

Extended for:
  - 3D packing (heightmap representation)
  - Semi-online with 5-10 item buffer
  - 2-bounded space (2 active pallets/bins)
  - Stability constraints
  - Heuristic-filtered action space (inspired by Verma et al. 2020)

This file contains concrete pseudocode and architecture definitions that can
be turned into a working implementation. It is NOT a runnable script -- it is
a design document with inline Python-style pseudocode.

Target: Thesis project on semi-online 3D bin packing with DRL.

Related files:
  - python/semi_online_buffer/buffer_aware_packing.py  (buffer logic)
  - python/hybrid_heuristic_ml/ems_filtered_drl.py     (heuristic filtering)
  - python/stability/stability_checker.py              (stability module)
  - python/multi_bin/two_bounded_manager.py            (2-bounded logic)
"""

# =============================================================================
# SECTION 1: CORE DATA STRUCTURES
# =============================================================================

"""
1.1 Box (Item) Representation
"""
# from dataclasses import dataclass, field
# from typing import List, Tuple, Optional
# import numpy as np

# @dataclass
# class Box:
#     \"\"\"A 3D rectangular box to be packed.\"\"\"
#     id: int
#     width: float    # x-dimension
#     depth: float    # y-dimension
#     height: float   # z-dimension
#     weight: float = 1.0  # for CoG and load-bearing calculations
#
#     def volume(self) -> float:
#         return self.width * self.depth * self.height
#
#     def footprint_area(self) -> float:
#         return self.width * self.depth
#
#     def get_orientations(self, allow_rotations: str = "vertical_only") -> List[Tuple[float, float, float]]:
#         \"\"\"
#         Return possible (w, d, h) orientations.
#
#         allow_rotations:
#           'none': only original orientation
#           'vertical_only': rotate around z-axis (2 orientations for non-square base)
#           'all': all 6 axis-aligned orientations (3 axes * 2 per axis, deduplicated)
#         \"\"\"
#         orientations = set()
#         w, d, h = self.width, self.depth, self.height
#
#         if allow_rotations == "none":
#             return [(w, d, h)]
#         elif allow_rotations == "vertical_only":
#             orientations.add((w, d, h))
#             orientations.add((d, w, h))  # 90-degree rotation around z
#         elif allow_rotations == "all":
#             # All 6 permutations of (w, d, h) -- but height must go up
#             # Actually, 6 orientations from choosing which dimension is height
#             for dims in [(w,d,h), (w,h,d), (d,w,h), (d,h,w), (h,w,d), (h,d,w)]:
#                 orientations.add(dims)
#
#         return list(orientations)


"""
1.2 Placement Record
"""
# @dataclass
# class Placement:
#     \"\"\"Records where a box was placed in a bin.\"\"\"
#     box: Box
#     x: int          # discretized x position (top-left corner on floor grid)
#     y: int          # discretized y position
#     z: float        # computed z (height at which bottom of box rests)
#     orientation: Tuple[float, float, float]  # (placed_w, placed_d, placed_h)
#
#     @property
#     def placed_width(self): return self.orientation[0]
#     @property
#     def placed_depth(self): return self.orientation[1]
#     @property
#     def placed_height(self): return self.orientation[2]


"""
1.3 Bin State (Heightmap-Based)
"""
# @dataclass
# class BinState:
#     \"\"\"
#     State of a single 3D bin, represented primarily via a heightmap.
#
#     The heightmap is a 2D grid (W x D) where each cell stores the maximum
#     height of items at that (x, y) position. This is the 3D extension of
#     Deep-Pack's binary grid.
#
#     Key insight from Deep-Pack: The bin state IS the image. For 3D, the
#     heightmap IS the image (grayscale, normalized to [0,1]).
#     \"\"\"
#     bin_width: int    # discretized W
#     bin_depth: int    # discretized D
#     bin_height: float # maximum allowed height H
#
#     heightmap: np.ndarray          # shape (W, D), float, values in [0, H]
#     placements: List[Placement] = field(default_factory=list)
#
#     def normalized_heightmap(self) -> np.ndarray:
#         \"\"\"Heightmap normalized to [0, 1] for CNN input.\"\"\"
#         return self.heightmap / self.bin_height
#
#     def volume_utilization(self) -> float:
#         \"\"\"Total volume of placed items / bin volume.\"\"\"
#         total_item_vol = sum(p.box.volume() for p in self.placements)
#         bin_vol = self.bin_width * self.bin_depth * self.bin_height
#         return total_item_vol / bin_vol if bin_vol > 0 else 0.0
#
#     def can_place(self, box_w, box_d, box_h, x, y) -> bool:
#         \"\"\"Check if a box with footprint (box_w, box_d) fits at position (x, y).\"\"\"
#         if x + box_w > self.bin_width or y + box_d > self.bin_depth:
#             return False
#         # The box would rest at z = max height in the footprint region
#         z = np.max(self.heightmap[x:x+box_w, y:y+box_d])
#         if z + box_h > self.bin_height:
#             return False
#         return True
#
#     def place(self, box: 'Box', x: int, y: int, orientation: tuple) -> 'Placement':
#         \"\"\"Place a box and update the heightmap.\"\"\"
#         bw, bd, bh = orientation
#         z = np.max(self.heightmap[x:x+int(bw), y:y+int(bd)])
#         self.heightmap[x:x+int(bw), y:y+int(bd)] = z + bh
#         p = Placement(box=box, x=x, y=y, z=z, orientation=orientation)
#         self.placements.append(p)
#         return p


# =============================================================================
# SECTION 2: STATE REPRESENTATION FOR DRL
# =============================================================================

"""
2.1 State Construction

Deep-Pack (2D): State = concatenate(bin_binary_image, item_binary_image)
  -> Matrix of size W x 2H

Our 3D Extension: State = multi-channel image stack

Channel layout for 2-bounded space with buffer:

  Channel 0: Bin 1 normalized heightmap      (W x D)
  Channel 1: Bin 2 normalized heightmap      (W x D)
  Channel 2: Current item footprint mask     (W x D) -- 1 where item would be placed
  Channel 3: Support map for Bin 1           (W x D) -- fraction of each cell supported
  Channel 4: Support map for Bin 2           (W x D)

Total state shape: (W, D, 5)   -- or (5, W, D) for PyTorch conv2d

Buffer information is handled separately (see Section 4).
"""

# def construct_state(bin1: BinState, bin2: BinState,
#                     current_item: Box, orientation: tuple) -> np.ndarray:
#     \"\"\"
#     Construct the multi-channel state tensor for the DRL agent.
#
#     Returns:
#         np.ndarray of shape (5, W, D) -- channels-first for PyTorch
#     \"\"\"
#     W, D = bin1.bin_width, bin1.bin_depth
#     state = np.zeros((5, W, D), dtype=np.float32)
#
#     # Channel 0: Bin 1 heightmap (normalized)
#     state[0] = bin1.normalized_heightmap()
#
#     # Channel 1: Bin 2 heightmap (normalized)
#     state[1] = bin2.normalized_heightmap()
#
#     # Channel 2: Item footprint mask (normalized)
#     bw, bd, bh = orientation
#     item_mask = np.zeros((W, D), dtype=np.float32)
#     # Place the item footprint in the center for reference
#     # (actual placement location is determined by action)
#     item_mask[:int(bw), :int(bd)] = bh / bin1.bin_height  # height as intensity
#     state[2] = item_mask
#
#     # Channel 3-4: Support maps (fraction of each cell that is at a
#     # "supported" height level -- requires voxel analysis or approximation)
#     state[3] = compute_support_map(bin1)
#     state[4] = compute_support_map(bin2)
#
#     return state


"""
2.2 Support Map Computation

For each cell (x, y) in the heightmap, the support fraction is:
  - Look at the cell's current height z = heightmap[x, y]
  - Check how much of the area at height z is supported by items
    directly below (i.e., items whose top surface is at height z)
  - For floor-level cells (z = 0): support = 1.0 (floor always supports)

A simplified version: binary support -- is the cell touching items below
or the floor? More accurate: compute overlap between item bottom and
items/floor beneath.
"""

# def compute_support_map(bin_state: BinState) -> np.ndarray:
#     \"\"\"
#     Compute a support map for the bin.
#
#     For each cell (x, y): what fraction of the top surface at that
#     position is supported from below?
#
#     Simplified approach: A cell is "supported" if its height equals
#     the height of at least one of its 4-connected neighbors or the floor.
#     \"\"\"
#     hm = bin_state.heightmap
#     W, D = hm.shape
#     support = np.zeros((W, D), dtype=np.float32)
#
#     for x in range(W):
#         for y in range(D):
#             if hm[x, y] == 0:
#                 support[x, y] = 1.0  # floor level = fully supported
#             else:
#                 # Check if any neighbor has same or higher height
#                 neighbors = []
#                 if x > 0: neighbors.append(hm[x-1, y])
#                 if x < W-1: neighbors.append(hm[x+1, y])
#                 if y > 0: neighbors.append(hm[x, y-1])
#                 if y < D-1: neighbors.append(hm[x, y+1])
#                 # Supported if current height matches a neighboring column
#                 support[x, y] = 1.0 if any(n >= hm[x, y] for n in neighbors) else 0.5
#
#     return support


# =============================================================================
# SECTION 3: REWARD FUNCTION (3D EXTENSION OF DEEP-PACK)
# =============================================================================

"""
3.1 Reward Design Philosophy

Deep-Pack reward: cluster_size * compactness
  - Promotes placing items adjacent to existing items
  - Promotes rectangular (non-zigzag) clusters

3D extension: We decompose the reward into multiple components that each
target a specific packing objective.

R(s, a) = { -PENALTY                          if action is infeasible
           { w1*R_adj + w2*R_stab + w3*R_smooth  if action is feasible

Plus end-of-episode bonus: R_end = K * volume_utilization

Suggested weights (to be tuned via hyperparameter search):
  w1 = 1.0   (adjacency/compactness -- core Deep-Pack idea)
  w2 = 2.0   (stability -- critical for our use case, higher weight)
  w3 = 0.5   (smoothness -- encourages layer-like building)
  K  = 5.0   (end-of-episode bonus multiplier)
  PENALTY = 10.0  (infeasible action penalty)
"""

# def compute_reward(bin_state: BinState, placement: Placement,
#                    bin_state_before: BinState) -> float:
#     \"\"\"
#     Compute the reward for placing an item in a bin.
#
#     Components:
#       1. Adjacency reward (adapted from Deep-Pack cluster*compactness)
#       2. Stability reward (support ratio + CoG balance)
#       3. Smoothness reward (heightmap variance reduction)
#     \"\"\"
#     w1, w2, w3 = 1.0, 2.0, 0.5
#
#     r_adj = compute_adjacency_reward(bin_state, placement)
#     r_stab = compute_stability_reward(bin_state, placement)
#     r_smooth = compute_smoothness_reward(bin_state, bin_state_before)
#
#     return w1 * r_adj + w2 * r_stab + w3 * r_smooth


"""
3.2 Adjacency Reward (3D Extension of Deep-Pack Cluster Reward)

In 2D, Deep-Pack uses Connected Component Labelling on the binary grid.
In 3D, we compute the contact surface area between the placed item and
its surroundings (other items + bin walls).

contact_surface_area: total area of item faces that touch other items or walls
total_surface_area: total surface area of the placed item

R_adj = contact_surface_area / total_surface_area

Range: [0, 1] where 1 means the item is completely surrounded (maximum contact)
"""

# def compute_adjacency_reward(bin_state: BinState, placement: Placement) -> float:
#     \"\"\"
#     Compute adjacency reward: fraction of item surface in contact with
#     other items or bin walls.
#     \"\"\"
#     p = placement
#     bw, bd, bh = p.placed_width, p.placed_depth, p.placed_height
#     x, y, z = p.x, p.y, p.z
#
#     total_surface = 2 * (bw*bd + bw*bh + bd*bh)
#     contact = 0.0
#
#     # Bottom face contact (floor or items below)
#     if z == 0:
#         contact += bw * bd  # resting on floor
#     else:
#         # Count cells where heightmap_before == z (items supporting from below)
#         # This requires the heightmap BEFORE placement
#         pass  # detailed implementation in stability module
#
#     # Side face contacts: check neighboring cells in heightmap
#     # Left wall (x == 0) or items at x-1
#     if x == 0:
#         contact += bd * bh  # bin wall contact
#     # Right wall (x + bw == W)
#     if x + bw == bin_state.bin_width:
#         contact += bd * bh
#     # Front wall (y == 0)
#     if y == 0:
#         contact += bw * bh
#     # Back wall (y + bd == D)
#     if y + bd == bin_state.bin_depth:
#         contact += bw * bh
#
#     # Item-item side contacts require checking heightmap neighbors
#     # (simplified: check if adjacent columns have height >= z and <= z+bh)
#
#     return min(contact / total_surface, 1.0) if total_surface > 0 else 0.0


"""
3.3 Stability Reward

Two sub-components:
  a) Support ratio: fraction of item's bottom face that rests on solid support
  b) CoG balance: how centered the overall center of gravity remains

R_stab = 0.7 * support_ratio + 0.3 * cog_balance
"""

# def compute_stability_reward(bin_state: BinState, placement: Placement) -> float:
#     \"\"\"Compute stability reward combining support ratio and CoG balance.\"\"\"
#     support = compute_support_ratio(bin_state, placement)
#     cog_bal = compute_cog_balance(bin_state)
#     return 0.7 * support + 0.3 * cog_bal
#
# def compute_support_ratio(bin_state: BinState, placement: Placement) -> float:
#     \"\"\"
#     What fraction of the item's bottom face is supported?
#
#     The item rests at z = max(heightmap[footprint_region]) BEFORE placement.
#     Supported cells = cells in the footprint where heightmap == z.
#     Unsupported cells = cells where heightmap < z (there's a gap below).
#
#     support_ratio = supported_cells / total_footprint_cells
#     \"\"\"
#     p = placement
#     bw, bd = int(p.placed_width), int(p.placed_depth)
#     x, y = p.x, p.y
#
#     # We need the heightmap BEFORE this item was placed
#     # This should be passed in or stored
#     # heightmap_before = bin_state_before.heightmap
#     # footprint = heightmap_before[x:x+bw, y:y+bd]
#     # rest_height = np.max(footprint)  # item bottom rests here
#     # supported = np.sum(footprint == rest_height)
#     # total = bw * bd
#     # return supported / total
#     pass
#
# def compute_cog_balance(bin_state: BinState) -> float:
#     \"\"\"
#     How well-centered is the overall center of gravity?
#
#     Returns 1.0 if CoG is at bin center, decreasing toward 0 as it
#     moves to the edge.
#
#     CoG_x = sum(item_x_center * item_weight) / sum(item_weight)
#     CoG_y = sum(item_y_center * item_weight) / sum(item_weight)
#
#     balance = 1 - max(|CoG_x - W/2| / (W/2), |CoG_y - D/2| / (D/2))
#     \"\"\"
#     if not bin_state.placements:
#         return 1.0
#
#     W, D = bin_state.bin_width, bin_state.bin_depth
#     total_weight = 0
#     weighted_x = 0
#     weighted_y = 0
#
#     for p in bin_state.placements:
#         w = p.box.weight
#         cx = p.x + p.placed_width / 2
#         cy = p.y + p.placed_depth / 2
#         weighted_x += cx * w
#         weighted_y += cy * w
#         total_weight += w
#
#     if total_weight == 0:
#         return 1.0
#
#     cog_x = weighted_x / total_weight
#     cog_y = weighted_y / total_weight
#
#     dx = abs(cog_x - W/2) / (W/2)
#     dy = abs(cog_y - D/2) / (D/2)
#
#     return max(0.0, 1.0 - max(dx, dy))


"""
3.4 Smoothness Reward (Encourages Layer Building)

After placing an item, the heightmap should ideally become MORE uniform
(closer to a flat top surface), which enables easier future placements
and is characteristic of good layer-building strategies.

R_smooth = -delta_variance / max_possible_variance
         = (var_before - var_after) / var_before  (positive if variance decreased)

Or simplified: R_smooth = 1 - normalized_variance(heightmap_after)
"""

# def compute_smoothness_reward(bin_state_after: BinState,
#                                bin_state_before: BinState) -> float:
#     \"\"\"
#     Reward for making the heightmap more uniform (layer-like).
#     Positive if variance decreased, negative if increased.
#     \"\"\"
#     var_before = np.var(bin_state_before.heightmap)
#     var_after = np.var(bin_state_after.heightmap)
#
#     if var_before == 0:
#         return -var_after  # penalize creating variance from flat
#
#     # Normalized improvement: positive = variance decreased = good
#     return (var_before - var_after) / var_before


# =============================================================================
# SECTION 4: NETWORK ARCHITECTURE
# =============================================================================

"""
4.1 Q-Network for 3D Bin Packing

Deep-Pack uses a simple CNN with 5 conv layers and 2 FC layers.
Our 3D extension needs a deeper/wider network because:
  1. Multi-channel input (5 channels instead of 1)
  2. Larger spatial dimensions (e.g., 40x40 instead of 5x5)
  3. More complex action space

Architecture options:

Option A: ResNet-style (recommended for bins > 20x20)
  - Input: (5, W, D)
  - 3 residual blocks with skip connections
  - Global average pooling
  - FC layers for Q-values

Option B: Deep-Pack style scaled up (for bins <= 20x20)
  - Input: (5, W, D)
  - 5 conv layers with increasing filters
  - 3 max-pooling layers
  - 2 FC layers

Option C: Dueling DQN architecture (recommended for better value estimation)
  - Shared convolutional backbone
  - Two parallel heads:
    - Value stream: estimates V(s)
    - Advantage stream: estimates A(s, a)
  - Q(s, a) = V(s) + A(s, a) - mean(A(s, .))
"""

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class DeepPack3D_QNetwork(nn.Module):
#     \"\"\"
#     Dueling Double DQN for 3D bin packing.
#
#     Input: Multi-channel heightmap state (batch, channels, W, D)
#     Output: Q-values for each candidate action
#
#     The number of actions is NOT fixed at W*D (too large).
#     Instead, actions are pre-filtered by heuristics (see Section 5),
#     and we output Q-values for a fixed maximum number of candidates
#     (e.g., MAX_CANDIDATES = 100).
#     \"\"\"
#     def __init__(self, in_channels=5, grid_w=40, grid_d=40,
#                  max_actions=100):
#         super().__init__()
#
#         # Convolutional backbone
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#
#         # Compute flattened size after convolutions
#         # After 3 pools: spatial dims / 8
#         conv_out_w = grid_w // 8
#         conv_out_d = grid_d // 8
#         flat_size = 64 * max(1, conv_out_w) * max(1, conv_out_d)
#
#         # Dueling streams
#         # Value stream
#         self.value_fc1 = nn.Linear(flat_size, 256)
#         self.value_fc2 = nn.Linear(256, 1)
#
#         # Advantage stream
#         self.adv_fc1 = nn.Linear(flat_size, 256)
#         self.adv_fc2 = nn.Linear(256, max_actions)
#
#     def forward(self, x):
#         # Backbone
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
#         x = x.flatten(1)
#
#         # Dueling
#         val = F.relu(self.value_fc1(x))
#         val = self.value_fc2(val)
#
#         adv = F.relu(self.adv_fc1(x))
#         adv = self.adv_fc2(adv)
#
#         # Q = V + (A - mean(A))
#         q = val + adv - adv.mean(dim=1, keepdim=True)
#         return q


"""
4.2 Buffer-Aware Item Selection Network

For the semi-online buffer, we need a second decision: which item from
the buffer to pick next. Two approaches:

Approach A: Separate network for item selection
  - Input: buffer items (encoded as feature vectors) + bin states
  - Output: probability/score for each buffer item
  - Then the placement network decides WHERE to place the selected item

Approach B: Joint network
  - For each (item, bin, placement) triple, compute a joint Q-value
  - Select the triple with highest Q-value
  - More expensive but considers interactions

Approach C: Hierarchical RL (recommended)
  - High-level policy: selects item from buffer + target bin
  - Low-level policy: selects placement location + orientation
  - Train separately or jointly with hierarchical RL
"""

# class BufferSelector(nn.Module):
#     \"\"\"
#     Selects which item from the buffer to pack next.
#
#     Input:
#       - bin_features: flattened CNN features from both bins (from shared backbone)
#       - buffer_items: (max_buffer_size, 4) -- (w, d, h, weight) per item, padded
#       - buffer_mask: (max_buffer_size,) -- 1 for valid items, 0 for padding
#
#     Output:
#       - scores: (max_buffer_size,) -- score per buffer item
#     \"\"\"
#     def __init__(self, bin_feature_dim=256, item_feature_dim=4,
#                  hidden_dim=128, max_buffer_size=10):
#         super().__init__()
#
#         # Encode each item
#         self.item_encoder = nn.Sequential(
#             nn.Linear(item_feature_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64)
#         )
#
#         # Combine bin features with item features
#         self.score_net = nn.Sequential(
#             nn.Linear(bin_feature_dim + 64, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
#
#     def forward(self, bin_features, buffer_items, buffer_mask):
#         # bin_features: (batch, bin_feature_dim)
#         # buffer_items: (batch, max_buffer, 4)
#         # buffer_mask: (batch, max_buffer)
#
#         batch_size, max_buffer, _ = buffer_items.shape
#
#         # Encode items: (batch, max_buffer, 64)
#         item_feats = self.item_encoder(buffer_items)
#
#         # Expand bin features: (batch, max_buffer, bin_feature_dim)
#         bin_expanded = bin_features.unsqueeze(1).expand(-1, max_buffer, -1)
#
#         # Concatenate: (batch, max_buffer, bin_feature_dim + 64)
#         combined = torch.cat([bin_expanded, item_feats], dim=-1)
#
#         # Score: (batch, max_buffer, 1) -> (batch, max_buffer)
#         scores = self.score_net(combined).squeeze(-1)
#
#         # Mask invalid items with large negative value
#         scores = scores.masked_fill(buffer_mask == 0, -1e9)
#
#         return scores


# =============================================================================
# SECTION 5: ACTION SPACE REDUCTION (CRITICAL FOR SCALABILITY)
# =============================================================================

"""
5.1 Why Action Space Reduction is Essential

Deep-Pack's pixel-level action space: W*H + 1 actions
  - For 5x5 bin: 26 actions (manageable)
  - For 40x40 bin: 1601 actions (still feasible but sparse)
  - For 100x100 bin: 10001 actions (too many -- most infeasible)
  - For 40x40x40 3D with pixel-level: 64001 actions (intractable)

Solution: Use heuristic rules (from overview KB Section 9 and 10) to
generate a small set of CANDIDATE placements, then let DRL choose among them.

This follows Verma et al. (2020) who reduced the action space before
applying DQN.

Candidate generation methods:
  a) Extreme Points (Crainic et al. 2008) -- widely used in online algorithms
  b) Empty Maximal Spaces (EMS) (Parreno et al. 2008) -- the standard
  c) Corner Points (Martello et al. 2000) -- simpler but fewer candidates

For heightmap representation, extreme points are the most natural fit:
  - After placing a box, new extreme points form at the corners of the
    box projected down to the heightmap surface
"""

# def generate_candidate_placements(bin_state: BinState, box: Box,
#                                    max_candidates: int = 100) -> list:
#     \"\"\"
#     Generate candidate (x, y, orientation) placements using extreme points.
#
#     This replaces the raw pixel-level action space with a much smaller
#     set of heuristically-selected feasible placements.
#
#     Algorithm:
#       1. Compute extreme points from the current heightmap
#       2. For each extreme point (x, y):
#          a. For each valid orientation of the box:
#             - Check if placement is feasible (fits, stable enough)
#             - If feasible, add to candidate list
#       3. If too many candidates: rank by heuristic score and take top-k
#       4. If zero candidates: return [NO_ACTION]
#
#     Returns: List of (x, y, orientation, heuristic_score) tuples
#     \"\"\"
#     candidates = []
#     extreme_pts = compute_extreme_points(bin_state)
#
#     for (ex, ey) in extreme_pts:
#         for orient in box.get_orientations("vertical_only"):
#             bw, bd, bh = orient
#             bw_i, bd_i = int(bw), int(bd)
#
#             # Check basic feasibility
#             if not bin_state.can_place(bw_i, bd_i, bh, ex, ey):
#                 continue
#
#             # Compute heuristic score for ranking
#             score = heuristic_score(bin_state, ex, ey, bw_i, bd_i, bh)
#             candidates.append((ex, ey, orient, score))
#
#     # Sort by heuristic score (descending) and take top candidates
#     candidates.sort(key=lambda c: c[3], reverse=True)
#     return candidates[:max_candidates]
#
#
# def compute_extreme_points(bin_state: BinState) -> list:
#     \"\"\"
#     Compute extreme points from the heightmap.
#
#     In 3D with heightmap, extreme points are positions where a new item
#     could rest. Key locations:
#       - (0, 0): always an extreme point (bottom-left corner)
#       - After each placed item: right edge (x+w, y), back edge (x, y+d)
#       - Projected down to heightmap surface
#
#     For heightmap representation, we simplify: scan the heightmap for
#     positions where height transitions occur (edges of placed items).
#     \"\"\"
#     hm = bin_state.heightmap
#     W, D = hm.shape
#     points = set()
#     points.add((0, 0))  # origin always valid
#
#     for p in bin_state.placements:
#         bw, bd = int(p.placed_width), int(p.placed_depth)
#         # Right edge
#         if p.x + bw < W:
#             points.add((p.x + bw, p.y))
#         # Back edge
#         if p.y + bd < D:
#             points.add((p.x, p.y + bd))
#         # Diagonal corner
#         if p.x + bw < W and p.y + bd < D:
#             points.add((p.x + bw, p.y + bd))
#
#     return list(points)
#
#
# def heuristic_score(bin_state, x, y, bw, bd, bh) -> float:
#     \"\"\"
#     Score a candidate placement using DBLF-inspired heuristic.
#
#     Higher score = better placement according to heuristic.
#
#     Scoring criteria (from overview KB Section 10):
#       - Prefer deeper positions (lower x) -> DBLF
#       - Prefer lower positions (lower z rest height) -> bottom preference
#       - Prefer positions closer to walls -> corner preference
#       - Penalize positions that create large height variance
#     \"\"\"
#     W, D = bin_state.bin_width, bin_state.bin_depth
#     hm = bin_state.heightmap
#
#     z_rest = np.max(hm[x:x+bw, y:y+bd])
#
#     # DBLF: prefer min x, then min z, then min y
#     score_dblf = (W - x) + (bin_state.bin_height - z_rest) + (D - y)
#
#     # Wall adjacency bonus
#     wall_bonus = 0
#     if x == 0: wall_bonus += 1
#     if y == 0: wall_bonus += 1
#     if x + bw == W: wall_bonus += 1
#     if y + bd == D: wall_bonus += 1
#
#     # Support ratio bonus
#     footprint = hm[x:x+bw, y:y+bd]
#     support = np.sum(footprint == z_rest) / (bw * bd) if bw*bd > 0 else 0
#
#     return score_dblf + 2 * wall_bonus + 3 * support


# =============================================================================
# SECTION 6: TRAINING LOOP (ADAPTED FROM DEEP-PACK)
# =============================================================================

"""
6.1 Training Loop Pseudocode

Differences from Deep-Pack:
  1. Multi-channel heightmap state instead of binary image
  2. Buffer: agent selects which item from buffer to place
  3. 2-bounded: agent selects which bin to place in
  4. Candidate actions instead of pixel-level actions
  5. Stability-aware reward
  6. Dueling Double DQN instead of simple Double DQN
"""

# def train_deep_pack_3d():
#     \"\"\"
#     Main training loop for 3D Deep-Pack with buffer and 2-bounded space.
#     \"\"\"
#     # Hyperparameters
#     NUM_EPISODES = 500_000
#     BUFFER_SIZE = 10      # semi-online buffer
#     MAX_ITEMS = 200       # items per episode
#     BIN_W, BIN_D, BIN_H = 40, 40, 40  # discretized bin dimensions
#     GAMMA = 0.99
#     LR = 1e-4
#     REPLAY_CAPACITY = 100_000
#     BATCH_SIZE = 64
#     TARGET_UPDATE_FREQ = 1000
#     EPSILON_START = 1.0
#     EPSILON_END = 0.05
#     EPSILON_DECAY_STEPS = 200_000
#     MAX_CANDIDATES = 100
#
#     # Initialize networks
#     q_net = DeepPack3D_QNetwork(in_channels=5, grid_w=BIN_W, grid_d=BIN_D,
#                                  max_actions=MAX_CANDIDATES)
#     target_net = DeepPack3D_QNetwork(in_channels=5, grid_w=BIN_W, grid_d=BIN_D,
#                                       max_actions=MAX_CANDIDATES)
#     target_net.load_state_dict(q_net.state_dict())
#
#     buffer_selector = BufferSelector(bin_feature_dim=256, max_buffer_size=BUFFER_SIZE)
#
#     optimizer = torch.optim.Adam(
#         list(q_net.parameters()) + list(buffer_selector.parameters()),
#         lr=LR
#     )
#     replay = ReplayMemory(REPLAY_CAPACITY)
#
#     step = 0
#
#     for episode in range(NUM_EPISODES):
#         # Initialize 2 empty bins
#         bin1 = BinState(BIN_W, BIN_D, BIN_H, np.zeros((BIN_W, BIN_D)))
#         bin2 = BinState(BIN_W, BIN_D, BIN_H, np.zeros((BIN_W, BIN_D)))
#         bins_closed = 0
#
#         # Generate random item sequence
#         items = generate_random_items(MAX_ITEMS, BIN_W, BIN_D, BIN_H)
#         item_queue = list(items)
#
#         # Initialize buffer
#         buffer = []
#         for _ in range(min(BUFFER_SIZE, len(item_queue))):
#             buffer.append(item_queue.pop(0))
#
#         episode_reward = 0
#
#         while buffer:  # continue until buffer is empty and no more items
#             epsilon = max(EPSILON_END,
#                          EPSILON_START - step * (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS)
#
#             # STEP 1: Select item from buffer
#             # (either random or via buffer_selector network)
#             if np.random.random() < epsilon:
#                 item_idx = np.random.randint(len(buffer))
#             else:
#                 # Use buffer selector network
#                 # ... (forward pass through buffer_selector)
#                 item_idx = 0  # placeholder
#
#             selected_item = buffer[item_idx]
#
#             # STEP 2: Generate candidate placements for BOTH bins
#             candidates_bin1 = generate_candidate_placements(bin1, selected_item, MAX_CANDIDATES // 2)
#             candidates_bin2 = generate_candidate_placements(bin2, selected_item, MAX_CANDIDATES // 2)
#
#             # Combine: each candidate tagged with bin_id
#             all_candidates = (
#                 [(1, c) for c in candidates_bin1] +
#                 [(2, c) for c in candidates_bin2] +
#                 [(0, None)]  # "skip/discard" action
#             )
#
#             # STEP 3: Construct state and get Q-values
#             state = construct_state(bin1, bin2, selected_item,
#                                     selected_item.get_orientations()[0])
#
#             if np.random.random() < epsilon:
#                 action_idx = np.random.randint(len(all_candidates))
#             else:
#                 # Forward pass through Q-network
#                 # q_values = q_net(state_tensor)
#                 # action_idx = q_values[:len(all_candidates)].argmax()
#                 pass
#
#             # STEP 4: Execute action
#             bin_id, candidate = all_candidates[action_idx]
#
#             if bin_id == 0:
#                 # Discard item
#                 reward = -1.0
#             else:
#                 target_bin = bin1 if bin_id == 1 else bin2
#                 x, y, orient, _ = candidate
#                 bin_before = copy_bin_state(target_bin)
#                 placement = target_bin.place(selected_item, x, y, orient)
#                 reward = compute_reward(target_bin, placement, bin_before)
#
#             # STEP 5: Remove item from buffer, refill
#             buffer.pop(item_idx)
#             if item_queue:
#                 buffer.append(item_queue.pop(0))
#
#             # STEP 6: Store transition and train
#             next_state = construct_state(bin1, bin2,
#                                           buffer[0] if buffer else None, None)
#             replay.push(state, action_idx, reward, next_state, len(buffer) == 0)
#
#             if len(replay) >= BATCH_SIZE:
#                 train_batch(q_net, target_net, optimizer, replay, BATCH_SIZE, GAMMA)
#
#             step += 1
#             episode_reward += reward
#
#             # Update target network periodically
#             if step % TARGET_UPDATE_FREQ == 0:
#                 target_net.load_state_dict(q_net.state_dict())
#
#             # Check if bins should be closed (heuristic: close if >95% full)
#             # This is the 2-bounded logic
#
#         # End-of-episode bonus
#         pe1 = bin1.volume_utilization()
#         pe2 = bin2.volume_utilization()
#         end_reward = 5.0 * (pe1 + pe2) / 2
#         episode_reward += end_reward
#
#         if episode % 100 == 0:
#             print(f"Episode {episode}: reward={episode_reward:.2f}, "
#                   f"PE1={pe1:.3f}, PE2={pe2:.3f}")


# =============================================================================
# SECTION 7: COMPLEXITY ANALYSIS AND FEASIBILITY
# =============================================================================

"""
7.1 Computational Complexity

Per-step inference:
  - Candidate generation: O(P * R * W * D) where P = # extreme points,
    R = # orientations, W*D for feasibility check
    Typically: P ~ 50, R ~ 2-6, feasibility = O(footprint_area) ~ O(W)
    Total: O(50 * 6 * W) = O(300W) -- very fast
  - CNN forward pass: O(CNN_params * spatial_size)
    With moderate architecture: ~5-10ms on GPU, ~50-100ms on CPU
  - Total per-step: ~10-100ms -- feasible for real-time robotic operation

Training:
  - Experience replay sampling: O(batch_size)
  - Gradient update: O(CNN_params)
  - Total training time estimate: 500K episodes * 200 steps * 0.01s = ~1000 hours on CPU
  - With GPU: ~50-100 hours (feasible)
  - With smaller bins / fewer episodes: proportionally less

Memory:
  - Replay buffer: 100K transitions * (state_size + action + reward + next_state)
  - State size: 5 * 40 * 40 * 4 bytes = 32KB per state
  - Total replay: ~6.4 GB (manageable, can reduce to 50K if needed)


7.2 Feasibility Assessment for Thesis

| Aspect               | Assessment      | Notes                                      |
|----------------------|----------------|--------------------------------------------|
| Implementation       | Feasible       | Standard PyTorch + Gymnasium               |
| Training time        | Moderate       | 2-7 days on single GPU                     |
| Scalability to 40x40 | Feasible       | With action space reduction                |
| Scalability to 100x100| Challenging   | May need larger network, more training     |
| Real-world testing   | Feasible       | Depth camera -> heightmap is straightforward|
| Beating heuristics   | Uncertain      | DRL may or may not outperform tuned EMS+DBLF|
| Stability integration| Feasible       | Reward shaping + feasibility masking       |
| 2-bounded space      | Feasible       | Natural extension of multi-bin state       |
| Buffer integration   | Feasible       | Hierarchical RL or separate selector       |


7.3 Recommended Development Plan

Phase 1 (Week 1-2): Environment + Baselines
  - Implement heightmap bin environment (Gymnasium interface)
  - Implement EMS/extreme point candidate generation
  - Implement DBLF and Best-Fit heuristic baselines
  - Implement stability checker

Phase 2 (Week 3-4): Single-Bin DRL
  - Implement Dueling DDQN with candidate actions
  - Train on single bin, single item at a time (no buffer, no multi-bin)
  - Verify learning and compare with heuristic baselines
  - This validates the core 3D extension of Deep-Pack

Phase 3 (Week 5-6): Buffer + Multi-Bin
  - Add buffer management
  - Add 2-bounded bin management
  - Implement hierarchical decision: item selection -> placement
  - Retrain and evaluate

Phase 4 (Week 7-8): Stability + Tuning
  - Integrate stability reward components
  - Add feasibility masking for minimum support ratio
  - Hyperparameter tuning
  - Ablation studies (reward components, buffer size, etc.)

Phase 5 (Week 9-10): Evaluation + Writing
  - Compare with all baselines (Shelf, BF, DBLF, Skyline, random)
  - Analyze fill rate vs. stability tradeoff
  - Generate figures and tables
  - Write thesis sections
"""


# =============================================================================
# SECTION 8: KEY DIFFERENCES FROM ORIGINAL DEEP-PACK
# =============================================================================

"""
Summary of all modifications from original Deep-Pack (Kundu et al. 2019):

| Aspect              | Deep-Pack (Original)           | Our 3D Extension              |
|---------------------|-------------------------------|-------------------------------|
| Dimensionality      | 2D                             | 3D                             |
| State representation| Binary grid W x 2H            | Multi-channel heightmap 5xWxD |
| Action space        | W*H + 1 (pixel level)         | ~100 heuristic-filtered candidates|
| Reward              | cluster_size * compactness     | adjacency + stability + smoothness|
| Algorithm           | Double DQN                     | Dueling Double DQN             |
| Bins                | Single                         | 2-bounded (2 active)           |
| Online model        | Strictly online                | Semi-online (5-10 buffer)      |
| Rotation            | None                           | Vertical-axis (2 orientations) |
| Stability           | Implicit (2D)                  | Explicit (support + CoG)       |
| Item discard        | Permanent                      | Goes back to buffer or next bin|
| Scale tested        | 3x3 to 5x5                    | Target: 40x40x40               |
| Real-world sensor   | 2D camera + thresholding       | Depth camera -> heightmap      |
"""
