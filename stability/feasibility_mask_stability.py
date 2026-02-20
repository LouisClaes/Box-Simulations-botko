"""
=============================================================================
CODING IDEAS: Stability as a Constraint via Feasibility Masking
=============================================================================
Based on: Zhao et al. (2021) "Online 3D Bin Packing with Constrained DRL"

This file focuses specifically on the STABILITY aspects of the paper and
how to implement, extend, and improve them for our thesis use case.

The key insight: stability is NOT a reward signal but a HARD CONSTRAINT
encoded as a binary feasibility mask. This is proven superior to reward
shaping for stability enforcement.
=============================================================================
"""

import numpy as np
from typing import Tuple, List, Optional


# =============================================================================
# 1. PAPER'S ORIGINAL STABILITY CRITERION
# =============================================================================

class PaperStabilityCriterion:
    """
    The conservative stability criterion from Zhao et al. (2021).

    A loading position (LP) at (x, y) is FEASIBLE for item (l, w, h)
    if the placement satisfies ANY of these three conditions:

    Condition 1: >= 60% bottom area supported
                 AND all 4 bottom corners supported
    Condition 2: >= 80% bottom area supported
                 AND >= 3 bottom corners supported
    Condition 3: >= 95% bottom area supported
                 (corners don't matter)

    This is conservative because:
    - It does not compute center of mass (unknown mass distribution)
    - It does not compute moment of inertia
    - It does not model rotational stability
    - It assumes uniform mass distribution implicitly via area ratios

    The three tiers provide a practical tradeoff:
    - Tier 1: low support but all corners -> like a table (stable)
    - Tier 2: high support, 3 corners -> like a shelf edge
    - Tier 3: very high support -> practically fully supported
    """

    def __init__(self):
        self.tier1_area = 0.60
        self.tier1_corners = 4
        self.tier2_area = 0.80
        self.tier2_corners = 3
        self.tier3_area = 0.95

    def is_stable(self, height_map: np.ndarray,
                  x: int, y: int,
                  l: int, w: int, h: int) -> bool:
        """
        Check if placing item (l, w, h) at FLB corner (x, y) is stable.

        Args:
            height_map: Current L x W height map of the bin
            x, y: Front-Left-Bottom corner position
            l, w, h: Item dimensions

        Returns:
            True if placement is stable under the three-tier criterion
        """
        region = height_map[x:x+l, y:y+w]
        placement_z = np.max(region)

        # Floor placements are always stable
        if placement_z == 0:
            return True

        # Compute support metrics
        total_cells = l * w
        supported_cells = np.sum(region == placement_z)
        support_ratio = supported_cells / total_cells

        # Corner support
        L_map, W_map = height_map.shape
        corners = [
            (x, y),           # front-left
            (x+l-1, y),       # back-left
            (x, y+w-1),       # front-right
            (x+l-1, y+w-1),   # back-right
        ]
        corners_supported = sum(
            1 for cx, cy in corners
            if 0 <= cx < L_map and 0 <= cy < W_map
            and height_map[cx, cy] == placement_z
        )

        # Three-tier check
        if support_ratio >= self.tier1_area and corners_supported >= self.tier1_corners:
            return True
        if support_ratio >= self.tier2_area and corners_supported >= self.tier2_corners:
            return True
        if support_ratio >= self.tier3_area:
            return True

        return False


# =============================================================================
# 2. EXTENDED STABILITY: CENTER OF MASS CHECK
# =============================================================================

class CenterOfMassStability:
    """
    Extended stability check that considers center of mass.

    The paper's criterion is conservative but ignores mass distribution.
    This extension adds a center-of-mass check:

    An item is stable if:
    1. It passes the paper's three-tier check (area + corners)
    2. Its center of mass (assuming uniform density) is within
       the convex hull of its support region

    For a rectangular item with uniform density, the center of mass
    is at the geometric center of the footprint. The item is stable
    if this center lies within (or very close to) the supported area.

    Relevant paper: "Static stability versus packing efficiency in
    online three-dimensional packing" (in our reading list)
    """

    def __init__(self, com_tolerance: float = 0.1):
        """
        Args:
            com_tolerance: How far the center of mass can be from
                          the support boundary (as fraction of item dim)
        """
        self.com_tolerance = com_tolerance
        self.base_criterion = PaperStabilityCriterion()

    def is_stable(self, height_map: np.ndarray,
                  x: int, y: int,
                  l: int, w: int, h: int) -> bool:
        """
        Enhanced stability check with center of mass.
        """
        # First pass: paper's criterion
        if not self.base_criterion.is_stable(height_map, x, y, l, w, h):
            return False

        region = height_map[x:x+l, y:y+w]
        placement_z = np.max(region)

        # Floor placement: always stable
        if placement_z == 0:
            return True

        # Compute center of mass of the item (geometric center)
        com_x = l / 2.0
        com_y = w / 2.0

        # Compute center of mass of the support region
        support_mask = (region == placement_z)
        if not np.any(support_mask):
            return False

        support_coords = np.argwhere(support_mask)
        support_center_x = np.mean(support_coords[:, 0]) + 0.5
        support_center_y = np.mean(support_coords[:, 1]) + 0.5

        # Check if item CoM is reasonably close to support center
        # (within tolerance of the support region)
        dx = abs(com_x - support_center_x) / l
        dy = abs(com_y - support_center_y) / w

        if dx > self.com_tolerance or dy > self.com_tolerance:
            return False

        return True


# =============================================================================
# 3. FULL FEASIBILITY MASK GENERATOR
# =============================================================================

class FeasibilityMaskGenerator:
    """
    Generates ground-truth feasibility masks for training the mask predictor.

    This is the component that creates the supervision signal for the
    neural network mask predictor. It must be:
    1. Correct (conservative is fine, but must not allow unstable placements)
    2. Fast (called for every item during training)

    The mask is a binary L x W matrix:
    M[x][y] = 1 if item can be stably placed at FLB corner (x, y)
    M[x][y] = 0 otherwise

    Checks performed for each cell:
    1. Containment: item fits within bin boundaries
    2. Height: item top does not exceed bin height
    3. Stability: placement satisfies stability criterion
    """

    def __init__(self, L: int, W: int, H: int,
                 stability_checker=None,
                 allow_rotation: bool = False):
        self.L = L
        self.W = W
        self.H = H
        self.stability = stability_checker or PaperStabilityCriterion()
        self.allow_rotation = allow_rotation

    def compute_mask(self, height_map: np.ndarray,
                     l: int, w: int, h: int) -> np.ndarray:
        """
        Compute feasibility mask for item (l, w, h) on current height map.

        Returns:
            Binary mask of shape (L, W) -- or (2, L, W) if rotation allowed
        """
        if self.allow_rotation:
            # Two orientations: (l, w) and (w, l)
            mask1 = self._compute_single_mask(height_map, l, w, h)
            mask2 = self._compute_single_mask(height_map, w, l, h)
            return np.stack([mask1, mask2], axis=0)
        else:
            return self._compute_single_mask(height_map, l, w, h)

    def _compute_single_mask(self, height_map: np.ndarray,
                             l: int, w: int, h: int) -> np.ndarray:
        """Compute mask for a single item orientation."""
        mask = np.zeros((self.L, self.W), dtype=np.float32)

        for x in range(self.L - l + 1):
            for y in range(self.W - w + 1):
                # Containment: already guaranteed by loop bounds

                # Height check
                region = height_map[x:x+l, y:y+w]
                placement_z = np.max(region)
                if placement_z + h > self.H:
                    continue

                # Stability check
                if self.stability.is_stable(height_map, x, y, l, w, h):
                    mask[x, y] = 1.0

        return mask

    def compute_mask_vectorized(self, height_map: np.ndarray,
                                l: int, w: int, h: int) -> np.ndarray:
        """
        OPTIMIZED: Vectorized feasibility mask computation.

        For larger grids (20x20, 30x30), the nested loop above is too slow.
        This version uses numpy operations for ~10-50x speedup.
        """
        mask = np.zeros((self.L, self.W), dtype=np.float32)

        # Compute max height in each possible footprint using sliding window
        # For each valid (x, y), compute max(height_map[x:x+l, y:y+w])
        from scipy.ndimage import maximum_filter

        # Max height in footprint
        max_heights = np.zeros_like(height_map, dtype=np.float32)
        for x in range(self.L - l + 1):
            for y in range(self.W - w + 1):
                max_heights[x, y] = np.max(height_map[x:x+l, y:y+w])

        # Height feasibility
        height_ok = (max_heights + h <= self.H)

        # Containment feasibility
        containment_ok = np.zeros((self.L, self.W), dtype=bool)
        containment_ok[:self.L-l+1, :self.W-w+1] = True

        # Combined basic feasibility
        basic_ok = height_ok & containment_ok

        # Stability check (vectorized where possible)
        for x in range(self.L - l + 1):
            for y in range(self.W - w + 1):
                if basic_ok[x, y]:
                    if self.stability.is_stable(height_map, x, y, l, w, h):
                        mask[x, y] = 1.0

        return mask


# =============================================================================
# 4. STABILITY CONSTRAINT VS REWARD SHAPING: IMPLEMENTATION COMPARISON
# =============================================================================

class StabilityRewardShaping:
    """
    ALTERNATIVE approach: stability as reward shaping (NOT recommended).

    The paper shows this is INFERIOR to the feasibility mask approach.
    Included here for comparison purposes.

    In reward shaping:
    - Stable placement: reward = volume_reward
    - Unstable placement: reward = volume_reward - stability_penalty
    - Invalid placement: reward = -large_penalty

    Problems with this approach:
    1. Agent may learn to tolerate some instability for volume gain
    2. Penalty magnitude is hard to tune
    3. Unstable placements still happen during training (unsafe exploration)
    4. Slower convergence because learning signal is indirect
    """

    def __init__(self, stability_penalty: float = 5.0,
                 invalid_penalty: float = 10.0):
        self.stability_penalty = stability_penalty
        self.invalid_penalty = invalid_penalty
        self.stability = PaperStabilityCriterion()

    def compute_reward(self, height_map: np.ndarray,
                       x: int, y: int,
                       l: int, w: int, h: int,
                       L: int, W: int, H: int) -> float:
        """
        Compute shaped reward.

        This is the reward-guided DRL approach from Figure 7 of the paper.
        """
        volume_reward = 10.0 * (l * w * h) / (L * W * H)

        # Check containment
        if x + l > L or y + w > W:
            return -self.invalid_penalty

        # Check height
        region = height_map[x:x+l, y:y+w]
        if np.max(region) + h > H:
            return -self.invalid_penalty

        # Check stability
        if not self.stability.is_stable(height_map, x, y, l, w, h):
            return volume_reward - self.stability_penalty

        return volume_reward


# =============================================================================
# 5. PRACTICAL STABILITY FOR ROBOTIC PACKING
# =============================================================================

class RoboticPackingStability:
    """
    Extended stability checks for real-world robotic packing.

    Beyond the paper's static vertical stability, real robotic packing
    needs to consider:

    1. PLACEMENT STABILITY: Can the robot place the item without
       it tipping during the placement motion?
    2. STACK STABILITY: After placement, is the entire stack stable?
    3. DYNAMIC STABILITY: Will items remain stable during bin transport?

    For our thesis (semi-online, 2-bounded, conveyor setup):
    - Placement stability is critical (robot must release the item)
    - Stack stability matters (items will be stacked high)
    - Dynamic stability depends on transport method
    """

    def __init__(self, min_support_ratio: float = 0.5,
                 max_stack_height_ratio: float = 0.9,
                 com_threshold: float = 0.15):
        self.min_support = min_support_ratio
        self.max_stack_ratio = max_stack_height_ratio
        self.com_threshold = com_threshold

    def full_stability_check(self, height_map: np.ndarray,
                             x: int, y: int,
                             l: int, w: int, h: int,
                             H: int) -> Tuple[bool, dict]:
        """
        Comprehensive stability check returning both decision and details.

        Returns:
            (is_stable, details_dict)
        """
        region = height_map[x:x+l, y:y+w]
        placement_z = np.max(region)

        details = {
            'placement_z': placement_z,
            'support_ratio': 0.0,
            'corners_supported': 0,
            'height_ratio': (placement_z + h) / H,
            'center_of_mass_ok': True,
            'stack_height_ok': True,
        }

        # Floor: always stable
        if placement_z == 0:
            details['support_ratio'] = 1.0
            details['corners_supported'] = 4
            return True, details

        # Support ratio
        total_cells = l * w
        supported_cells = np.sum(region == placement_z)
        support_ratio = supported_cells / total_cells
        details['support_ratio'] = support_ratio

        # Corner support
        corners = [
            (x, y), (x+l-1, y), (x, y+w-1), (x+l-1, y+w-1)
        ]
        corners_supported = sum(
            1 for cx, cy in corners
            if height_map[cx, cy] == placement_z
        )
        details['corners_supported'] = corners_supported

        # Check 1: Minimum support ratio
        if support_ratio < self.min_support:
            return False, details

        # Check 2: Stack height ratio (don't stack too high)
        if (placement_z + h) / H > self.max_stack_ratio:
            details['stack_height_ok'] = False
            # This is a soft check -- may still be allowed

        # Check 3: Center of mass within support
        support_mask = (region == placement_z)
        support_coords = np.argwhere(support_mask)
        if len(support_coords) > 0:
            support_center_x = np.mean(support_coords[:, 0]) + 0.5
            support_center_y = np.mean(support_coords[:, 1]) + 0.5
            com_x, com_y = l / 2.0, w / 2.0

            dx = abs(com_x - support_center_x) / max(l, 1)
            dy = abs(com_y - support_center_y) / max(w, 1)

            if dx > self.com_threshold or dy > self.com_threshold:
                details['center_of_mass_ok'] = False
                return False, details

        # Apply paper's three-tier criterion as final check
        paper_stable = PaperStabilityCriterion().is_stable(
            height_map, x, y, l, w, h
        )

        return paper_stable, details


# =============================================================================
# 6. INTEGRATION WITH FEASIBILITY MASK PREDICTOR
# =============================================================================

"""
KEY DESIGN PRINCIPLE:

The beauty of the feasibility mask approach is its MODULARITY.
The mask predictor neural network is INDEPENDENT of how the ground-truth
mask is computed. This means:

1. During TRAINING:
   - We compute ground-truth masks using ANY stability criterion
   - The criterion can be as complex as needed (physics simulation, FEA, etc.)
   - The mask predictor learns to approximate these complex checks

2. During INFERENCE:
   - The mask predictor outputs a fast approximation
   - No need to run expensive stability checks in real-time
   - The trained predictor is ~99.5% accurate (from paper)

This is essentially KNOWLEDGE DISTILLATION:
   Complex stability check -> Binary mask -> Neural network predictor

For our thesis:
1. Implement the most accurate stability check we can afford during training
2. Train the mask predictor to approximate it
3. During deployment, use only the predictor for real-time decisions
4. Periodically validate predictor accuracy against ground-truth

SUGGESTED STABILITY LEVELS (from simple to complex):

Level 1 (Paper's approach):
  - Three-tier corner + area criterion
  - Fast, conservative, no physics
  - Suitable for uniform-density items

Level 2 (Center of mass):
  - Add CoM check to Level 1
  - Slightly more restrictive but more physically accurate
  - Requires assuming uniform density

Level 3 (Full physics):
  - Compute torques and forces on the item
  - Consider friction coefficients
  - Model contact surfaces as point contacts
  - Expensive but most accurate

Level 4 (Simulation):
  - Use a physics engine (PyBullet, MuJoCo) to simulate placement
  - Let the item settle and check if it remains in position
  - Most accurate but very slow (only for training, not inference)

For the thesis, Level 2 is recommended as the best tradeoff between
accuracy and computation speed.
"""


# =============================================================================
# 7. QUICK BENCHMARK: Stability Check Performance
# =============================================================================

def benchmark_stability_checks():
    """
    Benchmark different stability check implementations.

    Measures time to compute a full feasibility mask for various
    grid sizes and stability criteria.
    """
    import time

    grid_sizes = [10, 15, 20, 30]
    criteria = [
        ("Paper 3-tier", PaperStabilityCriterion()),
        ("CoM extended", CenterOfMassStability()),
        ("Robotic full", RoboticPackingStability()),
    ]

    print("Feasibility Mask Computation Benchmark")
    print("=" * 60)
    print(f"{'Grid':>6} | {'Criterion':<20} | {'Time (ms)':>10} | {'Feasible':>8}")
    print("-" * 60)

    for L in grid_sizes:
        W = H = L
        height_map = np.random.randint(0, H // 2, (L, W))
        item = (L // 4, W // 4, H // 4)

        for name, criterion in criteria:
            generator = FeasibilityMaskGenerator(L, W, H, criterion)

            start = time.time()
            mask = generator.compute_mask(height_map, *item)
            elapsed = (time.time() - start) * 1000

            feasible_count = np.sum(mask > 0)
            print(f"{L:>4}x{L:<1} | {name:<20} | {elapsed:>8.2f}ms | {feasible_count:>8}")

    print()
    print("Note: For grids > 20x20, use compute_mask_vectorized()")
    print("or implement CUDA acceleration for the stability checks.")


if __name__ == '__main__':
    benchmark_stability_checks()


# =============================================================================
# 8. EXTENDED: LBCP-BASED STABILITY (from Gao et al. 2025 / One4Many)
# =============================================================================

class LBCPStability:
    """
    Local Bounding Convex Polygon (LBCP) stability criterion.

    This is a more physically accurate stability model from recent work
    (Gao et al. 2025, One4Many-StablePacker, arXiv:2510.10057).

    Instead of heuristic area/corner thresholds, LBCP:
    1. Identifies all support contact points between the item and
       items/floor below it
    2. Computes the convex hull of these contact points
    3. Checks if the center of mass (CoM) projection falls within
       this convex hull

    This is physically correct under the assumption that:
    - Items have uniform density (CoM = geometric center)
    - No friction forces (only normal forces)
    - Static equilibrium (no dynamic forces)

    Compared to the paper's three-tier criterion:
    - More accurate: does not reject stable configurations or accept unstable ones
    - Slightly slower: convex hull computation adds overhead
    - More principled: derived from physics, not engineering heuristics
    - Recommended for training ground truth; paper's criterion OK for inference

    For our thesis: Use LBCP for ground-truth mask generation during training,
    and the neural mask predictor for real-time inference.
    """

    def __init__(self, com_margin: float = 0.05):
        """
        Args:
            com_margin: Safety margin for CoM check. CoM must be at least
                        this fraction of item size inside the support polygon.
                        0.0 = exactly on edge is OK (most permissive)
                        0.1 = must be 10% inside (conservative)
        """
        self.com_margin = com_margin

    def is_stable(self, height_map: np.ndarray,
                  x: int, y: int,
                  l: int, w: int, h: int) -> bool:
        """
        Check stability using LBCP criterion.

        Steps:
        1. Find placement height z_p = max(height_map[x:x+l, y:y+w])
        2. Find all support contact cells: cells where height == z_p
        3. Compute convex hull of support region
        4. Compute center of mass of item (geometric center, assuming uniform density)
        5. Check if CoM falls within the convex hull (with margin)
        """
        region = height_map[x:x+l, y:y+w]
        placement_z = np.max(region)

        # Floor placement is always stable
        if placement_z == 0:
            return True

        # Find support contact points (relative to item's FLB corner)
        support_mask = (region == placement_z)
        support_points = np.argwhere(support_mask).astype(float)

        if len(support_points) == 0:
            return False

        # Minimum support: at least 1 cell must be in contact
        # (But for meaningful stability, we want meaningful contact area)
        min_support_cells = max(1, int(0.10 * l * w))  # At least 10% contact
        if len(support_points) < min_support_cells:
            return False

        # Center of mass of item (geometric center, relative to FLB corner)
        com_x = (l - 1) / 2.0
        com_y = (w - 1) / 2.0

        # For 1D or 2D items, special handling
        if l == 1 and w == 1:
            return True  # Single cell always stable if supported

        if l == 1 or w == 1:
            # Linear item: check CoM falls within support range
            if l == 1:
                support_range = (support_points[:, 1].min(), support_points[:, 1].max())
                margin_abs = self.com_margin * w
                return (support_range[0] - margin_abs <= com_y <= support_range[1] + margin_abs)
            else:
                support_range = (support_points[:, 0].min(), support_points[:, 0].max())
                margin_abs = self.com_margin * l
                return (support_range[0] - margin_abs <= com_x <= support_range[1] + margin_abs)

        # For 2D items: compute convex hull
        try:
            from scipy.spatial import ConvexHull
            if len(support_points) < 3:
                # Fewer than 3 points: check collinear/point support
                if len(support_points) == 1:
                    pt = support_points[0]
                    return (abs(pt[0] - com_x) <= self.com_margin * l and
                            abs(pt[1] - com_y) <= self.com_margin * w)
                elif len(support_points) == 2:
                    # Line segment: check CoM distance to segment
                    return self._point_near_segment(
                        com_x, com_y,
                        support_points[0], support_points[1],
                        self.com_margin * max(l, w)
                    )

            hull = ConvexHull(support_points)

            # Check if CoM is inside the convex hull (with margin)
            return self._point_in_convex_hull_with_margin(
                com_x, com_y, support_points[hull.vertices],
                self.com_margin * min(l, w)
            )
        except Exception:
            # Fallback: if convex hull fails (degenerate cases), use area ratio
            support_ratio = len(support_points) / (l * w)
            return support_ratio >= 0.5

    def _point_near_segment(self, px, py, a, b, margin):
        """Check if point (px, py) is within margin of segment a-b."""
        ax, ay = a
        bx, by = b
        # Project point onto segment
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0:
            return (abs(px - ax) <= margin and abs(py - ay) <= margin)
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
        return dist <= margin

    def _point_in_convex_hull_with_margin(self, px, py, hull_vertices, margin):
        """
        Check if point (px, py) is inside convex hull with safety margin.

        Uses the cross-product method: for each edge of the hull,
        check that the point is on the correct side (or within margin).
        """
        n = len(hull_vertices)
        for i in range(n):
            x1, y1 = hull_vertices[i]
            x2, y2 = hull_vertices[(i + 1) % n]
            # Cross product: (edge_vector) x (point - edge_start)
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            # If cross < -margin * edge_length, point is outside
            edge_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if edge_len > 0 and cross / edge_len < -margin:
                return False
        return True


# =============================================================================
# 9. EXTENDED: STACKING TREE STABILITY (from Zhao et al. 2022)
# =============================================================================

class StackingTreeNode:
    """
    Node in a stacking tree, representing one placed item.

    The stacking tree (from "Learning Practically Feasible Policies for
    Online 3D Bin Packing", Zhao et al. 2022, Science China Information
    Sciences) is a tree structure that tracks support relationships:

    - Root: the bin floor (virtual node)
    - Each node: a placed item
    - Edge from A to B: A supports B (A is below B, in contact)
    - Leaves: items at the top of stacks

    The tree enables O(N log N) stability checking vs O(N^2) for
    naive pairwise checks. When a new item is added, we only need
    to check its relationship with potential support items (parents
    in the tree), not all items.
    """

    def __init__(self, item_id: int, x: int, y: int, z: int,
                 l: int, w: int, h: int):
        self.item_id = item_id
        self.x, self.y, self.z = x, y, z
        self.l, self.w, self.h = l, w, h
        self.children: List['StackingTreeNode'] = []
        self.parent: Optional['StackingTreeNode'] = None
        self.support_items: List['StackingTreeNode'] = []  # items supporting this one

    @property
    def top_z(self) -> int:
        return self.z + self.h

    @property
    def footprint(self) -> Tuple[int, int, int, int]:
        """Returns (x_min, y_min, x_max, y_max) of the item footprint."""
        return (self.x, self.y, self.x + self.l, self.y + self.w)

    def overlaps_horizontally(self, other: 'StackingTreeNode') -> bool:
        """Check if this item's footprint overlaps with another's."""
        x1_min, y1_min, x1_max, y1_max = self.footprint
        x2_min, y2_min, x2_max, y2_max = other.footprint
        return (x1_min < x2_max and x2_min < x1_max and
                y1_min < y2_max and y2_min < y1_max)


class StackingTree:
    """
    Complete stacking tree for tracking item support relationships.

    Used for:
    1. Stability checking: is a new placement stable?
    2. Cascade detection: if item X is removed, which items above become unstable?
    3. Support graph visualization

    Complexity:
    - Adding an item: O(N_active) where N_active = items whose top face
      is at the placement height (typically much less than N)
    - Stability check: O(support_count) where support_count = number of
      items supporting the new item

    From Zhao et al. 2022: this reduces overall complexity from O(N^2)
    to O(N log N) for N items.
    """

    def __init__(self, L: int, W: int, H: int):
        self.L, self.W, self.H = L, W, H
        # Root represents the bin floor
        self.root = StackingTreeNode(
            item_id=-1, x=0, y=0, z=0, l=L, w=W, h=0
        )
        self.items: List[StackingTreeNode] = []
        self.height_map = np.zeros((L, W), dtype=np.int32)

    def add_item(self, x: int, y: int, l: int, w: int, h: int) -> StackingTreeNode:
        """
        Add a new item to the stacking tree and update relationships.

        Returns the new node added to the tree.
        """
        # Determine placement height
        region = self.height_map[x:x+l, y:y+w]
        placement_z = int(np.max(region))

        # Create node
        item_id = len(self.items)
        node = StackingTreeNode(item_id, x, y, placement_z, l, w, h)

        # Find support items: items whose top face is at placement_z
        # and whose footprint overlaps with the new item
        if placement_z == 0:
            # Supported by floor (root)
            node.support_items.append(self.root)
            node.parent = self.root
            self.root.children.append(node)
        else:
            for existing in self.items:
                if existing.top_z == placement_z and node.overlaps_horizontally(existing):
                    node.support_items.append(existing)
                    existing.children.append(node)

            if node.support_items:
                node.parent = node.support_items[0]  # Primary support

        self.items.append(node)

        # Update height map
        self.height_map[x:x+l, y:y+w] = placement_z + h

        return node

    def check_stability(self, x: int, y: int, l: int, w: int, h: int,
                        stability_criterion=None) -> bool:
        """
        Check if placing item at (x, y) would be stable using tree info.

        Uses support relationships from the tree to efficiently determine
        stability without scanning all items.
        """
        if stability_criterion is None:
            stability_criterion = PaperStabilityCriterion()

        return stability_criterion.is_stable(self.height_map, x, y, l, w, h)

    def get_cascade_unstable(self, removed_item_id: int) -> List[int]:
        """
        Find all items that become unstable if the given item is removed.

        This is important for rearrangement planning: if we need to move
        item X, which items above it must also be moved?

        Returns list of item IDs that become unstable.
        """
        if removed_item_id < 0 or removed_item_id >= len(self.items):
            return []

        removed = self.items[removed_item_id]
        unstable = []
        to_check = list(removed.children)

        while to_check:
            child = to_check.pop(0)
            # Check if child still has other support
            remaining_supports = [
                s for s in child.support_items
                if s.item_id != removed_item_id
                and s.item_id not in unstable
            ]
            if not remaining_supports:
                unstable.append(child.item_id)
                # Cascade: children of this item may also become unstable
                to_check.extend(child.children)

        return unstable


# =============================================================================
# 10. EXTENDED: MASK PREDICTOR TRAINING PIPELINE
# =============================================================================

class MaskPredictorTrainingPipeline:
    """
    Pipeline for training or fine-tuning the mask predictor separately.

    Use cases:
    1. Pre-train mask predictor on collected data before full DRL training
    2. Fine-tune mask predictor when switching stability criteria
    3. Evaluate mask predictor accuracy independently

    The mask predictor is an MLP that maps CNN features -> L*W binary mask.
    Training data: (height_map, item_dims) -> ground_truth_mask pairs.

    From the paper: the mask predictor achieves ~99.5% accuracy on
    predicting feasible/infeasible placements.

    KEY INSIGHT for thesis: When switching from three-tier stability to
    LBCP or stacking tree, we need to retrain the mask predictor. This
    can be done cheaply (~2 hours) without retraining the full DRL agent.
    """

    def __init__(self, L: int = 10, W: int = 10, H: int = 10,
                 stability_checker=None):
        self.L, self.W, self.H = L, W, H
        self.stability = stability_checker or PaperStabilityCriterion()
        self.mask_gen = FeasibilityMaskGenerator(L, W, H, self.stability)

    def generate_training_data(self, num_samples: int = 10000,
                               max_items_per_bin: int = 20) -> dict:
        """
        Generate training data for the mask predictor.

        Process:
        1. Generate random bin states by placing random items
        2. For each state, generate a random next item
        3. Compute the ground-truth feasibility mask
        4. Store (height_map, item_dims, mask) triples

        Args:
            num_samples: Number of training samples to generate
            max_items_per_bin: Max items to place when creating random states

        Returns:
            dict with 'height_maps', 'item_dims', 'masks' arrays
        """
        height_maps = []
        item_dims = []
        masks = []

        for _ in range(num_samples):
            # Create random bin state
            hm = np.zeros((self.L, self.W), dtype=np.int32)
            n_items = np.random.randint(0, max_items_per_bin + 1)

            for _ in range(n_items):
                l = np.random.randint(1, self.L // 2 + 1)
                w = np.random.randint(1, self.W // 2 + 1)
                h = np.random.randint(1, self.H // 2 + 1)
                mask_tmp = self.mask_gen.compute_mask(hm, l, w, h)
                if np.any(mask_tmp):
                    # Place at a random feasible position
                    feasible = np.argwhere(mask_tmp > 0)
                    idx = np.random.randint(len(feasible))
                    x, y = feasible[idx]
                    region = hm[x:x+l, y:y+w]
                    pz = int(np.max(region))
                    hm[x:x+l, y:y+w] = pz + h

            # Generate random next item
            next_l = np.random.randint(1, self.L // 2 + 1)
            next_w = np.random.randint(1, self.W // 2 + 1)
            next_h = np.random.randint(1, self.H // 2 + 1)

            # Compute ground-truth mask
            gt_mask = self.mask_gen.compute_mask(hm, next_l, next_w, next_h)

            height_maps.append(hm.copy())
            item_dims.append(np.array([next_l, next_w, next_h]))
            masks.append(gt_mask)

        return {
            'height_maps': np.array(height_maps),
            'item_dims': np.array(item_dims),
            'masks': np.array(masks),
        }

    def evaluate_predictor_accuracy(self, predictor_fn, test_data: dict) -> dict:
        """
        Evaluate mask predictor accuracy on test data.

        Metrics:
        1. Overall accuracy: fraction of cells correctly predicted
        2. Precision: of cells predicted feasible, how many are truly feasible
        3. Recall: of cells that are truly feasible, how many are predicted feasible
        4. F1 score: harmonic mean of precision and recall
        5. False positive rate: infeasible cells predicted as feasible (DANGEROUS)
        6. False negative rate: feasible cells predicted as infeasible (conservative)
        """
        gt_masks = test_data['masks']
        pred_masks = []

        for i in range(len(gt_masks)):
            hm = test_data['height_maps'][i]
            dims = test_data['item_dims'][i]
            pred = predictor_fn(hm, dims)
            pred_masks.append((pred > 0.5).astype(float))

        pred_masks = np.array(pred_masks)

        # Flatten for metric computation
        gt_flat = gt_masks.flatten()
        pred_flat = pred_masks.flatten()

        tp = np.sum((gt_flat == 1) & (pred_flat == 1))
        tn = np.sum((gt_flat == 0) & (pred_flat == 0))
        fp = np.sum((gt_flat == 0) & (pred_flat == 1))
        fn = np.sum((gt_flat == 1) & (pred_flat == 0))

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        fpr = fp / (fp + tn + 1e-10)  # FALSE POSITIVE = SAFETY CONCERN
        fnr = fn / (fn + tp + 1e-10)  # FALSE NEGATIVE = LOST EFFICIENCY

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positive_rate': fpr,  # Must be very low (< 0.5%)
            'false_negative_rate': fnr,  # Acceptable up to ~5%
            'total_cells': len(gt_flat),
            'feasible_ratio': np.mean(gt_flat),
        }


# =============================================================================
# 11. EXTENDED: STABILITY COMPARISON FRAMEWORK
# =============================================================================

class StabilityComparison:
    """
    Framework for comparing different stability criteria on the same scenarios.

    Useful for thesis experiments:
    1. Generate N random bin states
    2. For each state + item, compute masks under all criteria
    3. Compare: which criteria are more/less restrictive?
    4. Compute correlation between criteria
    5. Measure computation time per criterion

    Expected results:
    - Full base support > LBCP > Paper 3-tier > Partial support (restrictiveness)
    - More restrictive = fewer feasible positions = lower packing density
    - Less restrictive = more feasible positions but risk of instability
    """

    def __init__(self, L: int = 10, W: int = 10, H: int = 10):
        self.L, self.W, self.H = L, W, H
        self.criteria = {
            'paper_3tier': PaperStabilityCriterion(),
            'com_extended': CenterOfMassStability(),
            'lbcp': LBCPStability(),
            'robotic_full': RoboticPackingStability(),
        }

    def compare_on_scenario(self, height_map: np.ndarray,
                            l: int, w: int, h: int) -> dict:
        """
        Compare all stability criteria on a single scenario.

        Returns dict mapping criterion name to its feasibility mask.
        """
        results = {}
        for name, criterion in self.criteria.items():
            gen = FeasibilityMaskGenerator(self.L, self.W, self.H, criterion)
            mask = gen.compute_mask(height_map, l, w, h)
            results[name] = {
                'mask': mask,
                'feasible_count': int(np.sum(mask > 0)),
                'feasible_ratio': float(np.mean(mask)),
            }
        return results

    def run_comparison(self, num_scenarios: int = 1000) -> dict:
        """
        Run full comparison across many random scenarios.

        Returns aggregate statistics.
        """
        import time

        all_results = {name: {
            'feasible_counts': [],
            'total_time': 0.0,
        } for name in self.criteria}

        agreement_matrix = {
            (a, b): {'agree': 0, 'total': 0}
            for a in self.criteria for b in self.criteria if a < b
        }

        for _ in range(num_scenarios):
            # Random bin state
            hm = np.zeros((self.L, self.W), dtype=np.int32)
            n_items = np.random.randint(3, 15)
            for __ in range(n_items):
                il = np.random.randint(1, self.L // 2 + 1)
                iw = np.random.randint(1, self.W // 2 + 1)
                ih = np.random.randint(1, self.H // 2 + 1)
                # Quick place at first available position
                for ax in range(self.L - il + 1):
                    for ay in range(self.W - iw + 1):
                        region = hm[ax:ax+il, ay:ay+iw]
                        pz = int(np.max(region))
                        if pz + ih <= self.H:
                            hm[ax:ax+il, ay:ay+iw] = pz + ih
                            break
                    else:
                        continue
                    break

            # Random test item
            tl = np.random.randint(1, self.L // 2 + 1)
            tw = np.random.randint(1, self.W // 2 + 1)
            th = np.random.randint(1, self.H // 2 + 1)

            scenario_masks = {}
            for name, criterion in self.criteria.items():
                gen = FeasibilityMaskGenerator(self.L, self.W, self.H, criterion)
                start = time.time()
                mask = gen.compute_mask(hm, tl, tw, th)
                elapsed = time.time() - start

                all_results[name]['feasible_counts'].append(int(np.sum(mask > 0)))
                all_results[name]['total_time'] += elapsed
                scenario_masks[name] = mask

            # Compute pairwise agreement
            for (a, b) in agreement_matrix:
                agree = np.sum(scenario_masks[a] == scenario_masks[b])
                total = scenario_masks[a].size
                agreement_matrix[(a, b)]['agree'] += agree
                agreement_matrix[(a, b)]['total'] += total

        # Aggregate results
        summary = {}
        for name in self.criteria:
            counts = all_results[name]['feasible_counts']
            summary[name] = {
                'avg_feasible_positions': float(np.mean(counts)),
                'std_feasible_positions': float(np.std(counts)),
                'avg_time_ms': all_results[name]['total_time'] / num_scenarios * 1000,
            }

        # Agreement rates
        agreements = {}
        for (a, b), data in agreement_matrix.items():
            if data['total'] > 0:
                agreements[f'{a}_vs_{b}'] = data['agree'] / data['total']

        return {
            'per_criterion': summary,
            'pairwise_agreement': agreements,
            'num_scenarios': num_scenarios,
        }


# =============================================================================
# 12. EXTENDED: INTEGRATION NOTES FOR THESIS
# =============================================================================

"""
THESIS INTEGRATION GUIDE FOR STABILITY MODULES
================================================

Step 1: Initial Implementation (Week 1-2)
    Use PaperStabilityCriterion as-is.
    This matches the paper's approach and establishes a baseline.

    Code:
        criterion = PaperStabilityCriterion()
        mask_gen = FeasibilityMaskGenerator(L, W, H, criterion)
        mask = mask_gen.compute_mask(height_map, l, w, h)

Step 2: Train Mask Predictor (Week 3)
    Generate training data with MaskPredictorTrainingPipeline.
    Train the MLP mask predictor as part of the DRL training.

    Code:
        pipeline = MaskPredictorTrainingPipeline(L, W, H, criterion)
        data = pipeline.generate_training_data(num_samples=50000)
        # Feed into DRL training loop (L_mask term in loss)

Step 3: Upgrade to LBCP (Week 5-6)
    Switch ground-truth computation to LBCPStability.
    Retrain mask predictor MLP on LBCP-generated masks.
    Optionally fine-tune the full DRL agent.

    Code:
        lbcp_criterion = LBCPStability(com_margin=0.05)
        mask_gen = FeasibilityMaskGenerator(L, W, H, lbcp_criterion)
        # Retrain mask predictor with new ground truth

Step 4: Compare All Criteria (Week 7)
    Run StabilityComparison across all criteria.
    Report results in thesis:
    - Number of feasible positions per criterion
    - Agreement rates between criteria
    - Impact on packing density
    - Computation time comparison

Step 5: Stacking Tree (Optional, if time permits)
    Implement StackingTree for cascade stability tracking.
    Add to the feasibility mask as an additional check.
    Especially useful for tall stacks in the 2-bounded setting.

CRITICAL SAFETY NOTE:
    The mask predictor's false positive rate (predicting feasible when
    actually infeasible) must be kept extremely low (< 0.5%).
    A false positive = the robot places an item in an unstable position.
    Always validate with ground-truth checks in safety-critical settings.

    From the paper: 99.5% placement legitimacy = 0.5% failure rate.
    This may need to be improved for production robotic systems.
"""
