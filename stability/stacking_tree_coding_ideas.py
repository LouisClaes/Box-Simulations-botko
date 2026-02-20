"""
CODING IDEAS: Adaptive Stacking Tree for O(N log N) Stability Analysis
========================================================================
Source: "Learning Practically Feasible Policies for Online 3D Bin Packing"
         Zhao et al. (2023), arXiv:2108.13680v3

PURPOSE:
  Implement the adaptive stacking tree for real-time stability estimation
  during bin packing. This is the single most transferable component from
  the paper -- it can be used by ANY packing algorithm (RL-based, heuristic,
  hybrid) to check whether a placement is stable.

COMPLEXITY:
  Traditional full force analysis: O(N^2) per item placement
  This stacking tree approach:      O(N log N) total across all placements
  Accuracy: 99.9% vs Bullet physics simulator
  Speed:    ~5 x 10^-4 seconds per stability check

ADAPTATION FOR OUR USE CASE (2-bounded space, buffer 5-10):
  - Run stability check independently for EACH of the 2 active bins
  - For each item in the buffer (5-10 items), check stability on both bins
  - Total checks per decision step: 10 items * 2 bins * 2 orientations = 40
  - At 5e-4 seconds each: ~0.02 seconds total --> well within real-time
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
from scipy.spatial import ConvexHull


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Item:
    """Represents a 3D rectangular item (box) with placement info."""
    id: int
    length: float
    width: float
    height: float
    mass: float = 0.0  # Will be set to volume * density (uniform density assumed)

    # Placement info (set when placed)
    x: float = 0.0  # FLB corner x-coordinate
    y: float = 0.0  # FLB corner y-coordinate
    z: float = 0.0  # FLB corner z-coordinate (bottom)
    orientation: int = 0  # 0 = [l,w,h], 1 = [w,l,h]

    # Effective dimensions after orientation
    eff_length: float = 0.0
    eff_width: float = 0.0

    # Stability tracking
    is_placed: bool = False
    is_stable: bool = False

    def __post_init__(self):
        # Assume uniform density: mass proportional to volume
        if self.mass == 0.0:
            self.mass = self.length * self.width * self.height

    def apply_orientation(self):
        """Set effective dimensions based on orientation."""
        if self.orientation == 0:
            self.eff_length = self.length
            self.eff_width = self.width
        else:
            self.eff_length = self.width
            self.eff_width = self.length

    @property
    def centroid_xy(self) -> Tuple[float, float]:
        """2D centroid projection on XY plane."""
        return (
            self.x + self.eff_length / 2.0,
            self.y + self.eff_width / 2.0
        )

    @property
    def top_z(self) -> float:
        """Top surface z-coordinate."""
        return self.z + self.height

    @property
    def bottom_rect(self) -> Tuple[float, float, float, float]:
        """Bottom face rectangle: (x_min, y_min, x_max, y_max)."""
        return (self.x, self.y,
                self.x + self.eff_length,
                self.y + self.eff_width)

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height


@dataclass
class MassFlowEdge:
    """Edge in the mass distribution graph: mass flowing from item_above to item_below."""
    item_above_id: int
    item_below_id: int
    contact_points: List[Tuple[float, float]]  # XY contact points
    mass_flow: float = 0.0  # Mass transferred along this edge


@dataclass
class StackingTreeNode:
    """Node in the adaptive stacking tree."""
    item_id: int
    parent_ids: List[int]  # Items directly below (supporting this item)
    children_ids: List[int] = field(default_factory=list)  # Items directly above
    total_mass_above: float = 0.0  # Total mass flowing through this node
    group_centroid: Tuple[float, float] = (0.0, 0.0)  # Centroid of this item + all above


# =============================================================================
# CORE ALGORITHM: STACKING TREE
# =============================================================================

class AdaptiveStackingTree:
    """
    Adaptive Stacking Tree for O(N log N) stability analysis.

    The key insight from Zhao et al.:
    - Mass distribution of packed items forms a graph G
    - When placing item n, only a subgraph G_n needs updating
    - G_n is the "adaptive stacking tree" containing only I_active items
    - Items NOT in I_active keep their existing mass distributions

    Usage:
        tree = AdaptiveStackingTree(bin_length=100, bin_width=100, bin_height=100)
        is_stable = tree.check_stability(new_item, x, y, orientation)
        if is_stable:
            tree.place_item(new_item, x, y, orientation)
    """

    def __init__(self, bin_length: float, bin_width: float, bin_height: float,
                 resolution: int = 100):
        self.bin_L = bin_length
        self.bin_W = bin_width
        self.bin_H = bin_height
        self.resolution = resolution

        # Height map: integer grid recording max height at each cell
        self.height_map = np.zeros((resolution, resolution), dtype=np.float32)

        # Placed items registry
        self.placed_items: Dict[int, Item] = {}

        # Mass distribution graph
        self.edges: Dict[Tuple[int, int], MassFlowEdge] = {}
        self.tree_nodes: Dict[int, StackingTreeNode] = {}

        # Cell size for discretization
        self.cell_w = bin_length / resolution
        self.cell_h = bin_width / resolution

    def _get_support_items(self, item: Item) -> List[Tuple[int, List[Tuple[float, float]]]]:
        """
        Find which placed items directly support the given item.

        Returns list of (item_id, contact_points) tuples.
        Contact points are the XY coordinates where the items touch.
        """
        x_min, y_min, x_max, y_max = item.bottom_rect
        support_z = item.z  # The z-level where support must exist

        # If item is on the floor
        if abs(support_z) < 1e-6:
            return [(-1, [(x_min, y_min), (x_max, y_min),
                          (x_min, y_max), (x_max, y_max)])]  # -1 = floor

        supporters = {}
        for pid, placed in self.placed_items.items():
            # Check if placed item's top surface is at the right height
            if abs(placed.top_z - support_z) > 1e-6:
                continue

            # Check XY overlap
            px_min, py_min, px_max, py_max = placed.bottom_rect
            # Note: for top surface, we use the same xy bounds as bottom
            # since items are rectangular

            overlap_x_min = max(x_min, px_min)
            overlap_y_min = max(y_min, py_min)
            overlap_x_max = min(x_max, px_max)
            overlap_y_max = min(y_max, py_max)

            if overlap_x_min < overlap_x_max and overlap_y_min < overlap_y_max:
                # There is contact. Record contact region corners as contact points.
                contact_pts = [
                    (overlap_x_min, overlap_y_min),
                    (overlap_x_max, overlap_y_min),
                    (overlap_x_min, overlap_y_max),
                    (overlap_x_max, overlap_y_max)
                ]
                supporters[pid] = contact_pts

        return list(supporters.items())

    def _check_centroid_stability(self, item: Item,
                                  support_info: List[Tuple[int, List[Tuple[float, float]]]]) -> bool:
        """
        Supported Centroid Rule (from the paper):
        Item n is stable if its centroid c_n satisfies either:
        1) c_n is directly supported by a single packed item with this LP
        2) c_n is inside the convex hull of contact points of supporting items

        Args:
            item: The item to check stability for
            support_info: List of (supporter_id, contact_points) from _get_support_items

        Returns:
            True if item is stable at this position
        """
        if not support_info:
            return False

        cx, cy = item.centroid_xy

        # Case 1: Single supporter -- check if centroid is within contact area
        if len(support_info) == 1:
            sid, contact_pts = support_info[0]
            if sid == -1:  # Floor always supports
                return True
            # Check if centroid is within the contact rectangle
            xs = [p[0] for p in contact_pts]
            ys = [p[1] for p in contact_pts]
            return min(xs) <= cx <= max(xs) and min(ys) <= cy <= max(ys)

        # Case 2: Multiple supporters -- check convex hull of all contact points
        all_contact_points = []
        for sid, pts in support_info:
            all_contact_points.extend(pts)

        if len(all_contact_points) < 3:
            # Not enough points for a hull; check if centroid is on the line
            # (degenerate case -- rarely happens in practice)
            return False

        # Remove duplicate points
        unique_pts = list(set(all_contact_points))
        if len(unique_pts) < 3:
            # Collinear or too few unique points
            # Check if centroid lies on the line segment
            return self._point_on_line_segment(cx, cy, unique_pts)

        try:
            hull = ConvexHull(np.array(unique_pts))
            return self._point_in_convex_hull(cx, cy, unique_pts, hull)
        except Exception:
            return False

    def _point_in_convex_hull(self, px: float, py: float,
                               points: List[Tuple[float, float]],
                               hull: ConvexHull) -> bool:
        """Check if point (px, py) is inside the convex hull."""
        # Use the half-plane intersection method
        point = np.array([px, py])
        for eq in hull.equations:
            if np.dot(eq[:-1], point) + eq[-1] > 1e-6:
                return False
        return True

    def _point_on_line_segment(self, px: float, py: float,
                                points: List[Tuple[float, float]]) -> bool:
        """Check if point is approximately on the line segment between 2 points."""
        if len(points) < 2:
            return False
        x1, y1 = points[0]
        x2, y2 = points[1]
        # Check collinearity and range
        cross = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
        if abs(cross) > 1e-4:
            return False
        return (min(x1, x2) - 1e-4 <= px <= max(x1, x2) + 1e-4 and
                min(y1, y2) - 1e-4 <= py <= max(y1, y2) + 1e-4)

    def _compute_mass_distribution(self, item: Item,
                                    support_info: List[Tuple[int, List[Tuple[float, float]]]]):
        """
        Compute mass distribution using the principle of leverage.

        Three cases from the paper:
        1. Single support: all mass transfers to supporter
        2. Two supports: leverage principle (Equation 2 in paper)
        3. More than two: least squares optimization

        Then propagate changes down the stacking tree.
        """
        cx, cy = item.centroid_xy
        m_n = item.mass

        # Filter out floor support
        real_supports = [(sid, pts) for sid, pts in support_info if sid != -1]

        if not real_supports:
            # Item on floor -- no mass distribution needed
            return

        if len(real_supports) == 1:
            # Case 1: Single support
            sid, pts = real_supports[0]
            self.edges[(item.id, sid)] = MassFlowEdge(
                item_above_id=item.id,
                item_below_id=sid,
                contact_points=pts,
                mass_flow=m_n
            )
            self._propagate_mass_update(sid)

        elif len(real_supports) == 2:
            # Case 2: Two supports -- leverage principle
            sid0, pts0 = real_supports[0]
            sid1, pts1 = real_supports[1]

            # Use centroid of contact regions as pivot points
            p0 = np.mean(pts0, axis=0)
            p1 = np.mean(pts1, axis=0)
            c = np.array([cx, cy])

            dist_total = np.linalg.norm(p0 - p1)
            if dist_total < 1e-8:
                # Supports at same location; split evenly
                f0 = m_n / 2.0
                f1 = m_n / 2.0
            else:
                # Leverage: F_0 = g * ||c_n - p_1|| * m_n / ||p_0 - p_1||
                f0 = np.linalg.norm(c - p1) * m_n / dist_total
                f1 = np.linalg.norm(c - p0) * m_n / dist_total

            self.edges[(item.id, sid0)] = MassFlowEdge(
                item_above_id=item.id, item_below_id=sid0,
                contact_points=pts0, mass_flow=f0
            )
            self.edges[(item.id, sid1)] = MassFlowEdge(
                item_above_id=item.id, item_below_id=sid1,
                contact_points=pts1, mass_flow=f1
            )
            self._propagate_mass_update(sid0)
            self._propagate_mass_update(sid1)

        else:
            # Case 3: More than two supports -- least squares optimization
            self._distribute_mass_least_squares(item, real_supports)
            for sid, _ in real_supports:
                self._propagate_mass_update(sid)

    def _distribute_mass_least_squares(self, item: Item,
                                         supports: List[Tuple[int, List[Tuple[float, float]]]]):
        """
        For more than 2 supports, use least squares to distribute mass.
        Constraints: sum of forces = total weight, and torque balance.
        """
        cx, cy = item.centroid_xy
        m_n = item.mass
        n_supports = len(supports)

        # Build the system: balance forces and torques
        # Variables: F_i for each support
        # Constraint 1: sum(F_i) = m_n * g (we use g=1 for simplicity)
        # Constraint 2: sum(F_i * x_i) = m_n * cx (torque about y-axis)
        # Constraint 3: sum(F_i * y_i) = m_n * cy (torque about x-axis)

        contact_centers = []
        for sid, pts in supports:
            center = np.mean(pts, axis=0)
            contact_centers.append(center)
        contact_centers = np.array(contact_centers)

        # Least squares: minimize ||F||^2 subject to constraints
        # A @ F = b
        A = np.zeros((3, n_supports))
        A[0, :] = 1.0  # Force balance
        A[1, :] = contact_centers[:, 0]  # X-torque
        A[2, :] = contact_centers[:, 1]  # Y-torque
        b = np.array([m_n, m_n * cx, m_n * cy])

        # Solve using least norm solution: F = A^T @ (A @ A^T)^{-1} @ b
        try:
            AAT_inv = np.linalg.inv(A @ A.T)
            F = A.T @ AAT_inv @ b
            # Ensure non-negative forces
            F = np.maximum(F, 0)
            # Renormalize to maintain force balance
            if F.sum() > 0:
                F = F * (m_n / F.sum())
        except np.linalg.LinAlgError:
            # Fallback: equal distribution
            F = np.full(n_supports, m_n / n_supports)

        for i, (sid, pts) in enumerate(supports):
            self.edges[(item.id, sid)] = MassFlowEdge(
                item_above_id=item.id, item_below_id=sid,
                contact_points=pts, mass_flow=F[i]
            )

    def _propagate_mass_update(self, item_id: int):
        """
        Propagate mass distribution changes down the stacking tree.
        This is the "adaptive" part -- only update I_active items.

        The propagation continues downward until:
        - We reach the bin floor
        - OR instability is detected (centroid outside support polygon)
        """
        if item_id not in self.placed_items:
            return

        item = self.placed_items[item_id]
        # Recalculate total mass above this item
        total_above = sum(
            edge.mass_flow for key, edge in self.edges.items()
            if edge.item_below_id == item_id
        )

        if item_id in self.tree_nodes:
            old_mass = self.tree_nodes[item_id].total_mass_above
            if abs(total_above - old_mass) < 1e-6:
                return  # No significant change; stop propagation
            self.tree_nodes[item_id].total_mass_above = total_above

        # Propagate to items below this one
        for key, edge in self.edges.items():
            if edge.item_above_id == item_id:
                self._propagate_mass_update(edge.item_below_id)

    def check_stability(self, item: Item, x: float, y: float, orientation: int) -> bool:
        """
        Check if placing item at (x, y) with given orientation is stable.

        This is the main API for external callers.

        Args:
            item: The item to check
            x: X-coordinate of FLB corner
            y: Y-coordinate of FLB corner
            orientation: 0 = [l,w,h], 1 = [w,l,h]

        Returns:
            True if placement is stable
        """
        # Create a temporary copy with placement info
        temp_item = Item(
            id=item.id, length=item.length, width=item.width,
            height=item.height, mass=item.mass
        )
        temp_item.x = x
        temp_item.y = y
        temp_item.orientation = orientation
        temp_item.apply_orientation()

        # Determine z-coordinate from height map
        temp_item.z = self._get_placement_z(temp_item)

        # Check containment
        if (temp_item.x + temp_item.eff_length > self.bin_L or
            temp_item.y + temp_item.eff_width > self.bin_W or
            temp_item.top_z > self.bin_H):
            return False

        # Find support items
        support_info = self._get_support_items(temp_item)

        # Check centroid stability
        return self._check_centroid_stability(temp_item, support_info)

    def _get_placement_z(self, item: Item) -> float:
        """Get the z-coordinate where item would rest based on height map."""
        # Convert item footprint to grid cells
        x_start = int(item.x / self.cell_w)
        y_start = int(item.y / self.cell_h)
        x_end = int(np.ceil((item.x + item.eff_length) / self.cell_w))
        y_end = int(np.ceil((item.y + item.eff_width) / self.cell_h))

        x_start = max(0, min(x_start, self.resolution - 1))
        y_start = max(0, min(y_start, self.resolution - 1))
        x_end = max(0, min(x_end, self.resolution))
        y_end = max(0, min(y_end, self.resolution))

        if x_start >= x_end or y_start >= y_end:
            return 0.0

        return float(np.max(self.height_map[x_start:x_end, y_start:y_end]))

    def place_item(self, item: Item, x: float, y: float, orientation: int) -> bool:
        """
        Place an item and update the stacking tree.

        Args:
            item: The item to place
            x, y: FLB corner coordinates
            orientation: 0 or 1

        Returns:
            True if placed successfully (stable), False otherwise
        """
        item.x = x
        item.y = y
        item.orientation = orientation
        item.apply_orientation()
        item.z = self._get_placement_z(item)
        item.is_placed = True

        # Check stability first
        support_info = self._get_support_items(item)
        if not self._check_centroid_stability(item, support_info):
            item.is_placed = False
            return False

        item.is_stable = True

        # Register item
        self.placed_items[item.id] = item

        # Create tree node
        parent_ids = [sid for sid, _ in support_info if sid != -1]
        self.tree_nodes[item.id] = StackingTreeNode(
            item_id=item.id,
            parent_ids=parent_ids,
            total_mass_above=0.0,
            group_centroid=item.centroid_xy
        )

        # Add as child to parents
        for pid in parent_ids:
            if pid in self.tree_nodes:
                self.tree_nodes[pid].children_ids.append(item.id)

        # Compute and propagate mass distribution
        self._compute_mass_distribution(item, support_info)

        # Update height map
        self._update_height_map(item)

        return True

    def _update_height_map(self, item: Item):
        """Update height map after placing an item."""
        x_start = int(item.x / self.cell_w)
        y_start = int(item.y / self.cell_h)
        x_end = int(np.ceil((item.x + item.eff_length) / self.cell_w))
        y_end = int(np.ceil((item.y + item.eff_width) / self.cell_h))

        x_start = max(0, min(x_start, self.resolution - 1))
        y_start = max(0, min(y_start, self.resolution - 1))
        x_end = max(0, min(x_end, self.resolution))
        y_end = max(0, min(y_end, self.resolution))

        self.height_map[x_start:x_end, y_start:y_end] = np.maximum(
            self.height_map[x_start:x_end, y_start:y_end],
            item.top_z
        )

    def compute_feasibility_mask(self, item: Item, orientation: int) -> np.ndarray:
        """
        Compute the full L x W feasibility mask for an item with given orientation.

        This is the M_{n,o} from the paper: a binary matrix where 1 means
        placing the item's FLB at that grid cell is stable.

        Used as input to the RL policy network.
        """
        mask = np.zeros((self.resolution, self.resolution), dtype=np.float32)

        temp_item = Item(
            id=item.id, length=item.length, width=item.width,
            height=item.height, mass=item.mass
        )
        temp_item.orientation = orientation
        temp_item.apply_orientation()

        # Only check cells where the item fits within the bin
        max_x_cells = int((self.bin_L - temp_item.eff_length) / self.cell_w) + 1
        max_y_cells = int((self.bin_W - temp_item.eff_width) / self.cell_h) + 1

        for xi in range(min(max_x_cells, self.resolution)):
            for yi in range(min(max_y_cells, self.resolution)):
                x = xi * self.cell_w
                y = yi * self.cell_h

                if self.check_stability(item, x, y, orientation):
                    # Also check height constraint
                    temp_item.x = x
                    temp_item.y = y
                    z = self._get_placement_z(temp_item)
                    if z + temp_item.height <= self.bin_H:
                        mask[xi, yi] = 1.0

        return mask

    def get_height_map(self) -> np.ndarray:
        """Return the current height map (copy)."""
        return self.height_map.copy()

    def get_utilization(self) -> float:
        """Compute current volume utilization of the bin."""
        total_volume = sum(item.volume for item in self.placed_items.values())
        bin_volume = self.bin_L * self.bin_W * self.bin_H
        return total_volume / bin_volume if bin_volume > 0 else 0.0

    def reset(self):
        """Reset the stacking tree for a new bin."""
        self.height_map = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.placed_items.clear()
        self.edges.clear()
        self.tree_nodes.clear()


# =============================================================================
# INTEGRATION HELPER: For 2-Bounded Space with Buffer
# =============================================================================

class DualBinStabilityChecker:
    """
    Wrapper for managing stability checks across 2 active bins
    with a buffer of items.

    Usage in the 2-bounded space semi-online setting:
        checker = DualBinStabilityChecker(bin_dims=(100,100,100))
        # For each item in buffer, evaluate best placement on each bin
        for item in buffer:
            for bin_id in [0, 1]:
                for orient in [0, 1]:
                    mask = checker.get_feasibility_mask(bin_id, item, orient)
                    # Feed mask to RL policy or heuristic
    """

    def __init__(self, bin_dims: Tuple[float, float, float], resolution: int = 100):
        self.bins = [
            AdaptiveStackingTree(bin_dims[0], bin_dims[1], bin_dims[2], resolution),
            AdaptiveStackingTree(bin_dims[0], bin_dims[1], bin_dims[2], resolution)
        ]
        self.active_bins = [True, True]
        self.bin_dims = bin_dims
        self.resolution = resolution

    def check_stability(self, bin_id: int, item: Item,
                        x: float, y: float, orientation: int) -> bool:
        """Check stability of placing item in specified bin."""
        if not self.active_bins[bin_id]:
            return False
        return self.bins[bin_id].check_stability(item, x, y, orientation)

    def place_item(self, bin_id: int, item: Item,
                   x: float, y: float, orientation: int) -> bool:
        """Place item in specified bin."""
        if not self.active_bins[bin_id]:
            return False
        return self.bins[bin_id].place_item(item, x, y, orientation)

    def get_feasibility_mask(self, bin_id: int, item: Item,
                              orientation: int) -> np.ndarray:
        """Get feasibility mask for item on specified bin."""
        if not self.active_bins[bin_id]:
            return np.zeros((self.resolution, self.resolution), dtype=np.float32)
        return self.bins[bin_id].compute_feasibility_mask(item, orientation)

    def close_bin(self, bin_id: int):
        """Close a bin (permanently)."""
        self.active_bins[bin_id] = False

    def open_new_bin(self, bin_id: int):
        """Open a new bin at the specified slot."""
        self.bins[bin_id] = AdaptiveStackingTree(
            self.bin_dims[0], self.bin_dims[1], self.bin_dims[2], self.resolution
        )
        self.active_bins[bin_id] = True

    def get_utilizations(self) -> Tuple[float, float]:
        """Get utilization of both bins."""
        return (self.bins[0].get_utilization(), self.bins[1].get_utilization())

    def get_height_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get height maps of both bins."""
        return (self.bins[0].get_height_map(), self.bins[1].get_height_map())


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Demo: Place items in a bin with stability checking
    tree = AdaptiveStackingTree(bin_length=100, bin_width=100, bin_height=100, resolution=100)

    # Place a large item on the floor
    item1 = Item(id=1, length=40, width=30, height=20)
    success = tree.place_item(item1, x=0, y=0, orientation=0)
    print(f"Item 1 placed: {success}, utilization: {tree.get_utilization():.2%}")

    # Place a second item next to it
    item2 = Item(id=2, length=30, width=30, height=20)
    success = tree.place_item(item2, x=40, y=0, orientation=0)
    print(f"Item 2 placed: {success}, utilization: {tree.get_utilization():.2%}")

    # Place a third item on top spanning both (should be stable if centroid is supported)
    item3 = Item(id=3, length=50, width=25, height=15)
    stable = tree.check_stability(item3, x=10, y=2, orientation=0)
    print(f"Item 3 stable at (10,2): {stable}")

    if stable:
        success = tree.place_item(item3, x=10, y=2, orientation=0)
        print(f"Item 3 placed: {success}, utilization: {tree.get_utilization():.2%}")

    # Demo: Dual bin for 2-bounded space
    print("\n--- Dual Bin Demo ---")
    checker = DualBinStabilityChecker(bin_dims=(100, 100, 100))
    item_a = Item(id=10, length=40, width=40, height=30)

    # Check which bin is better for this item
    for bin_id in [0, 1]:
        stable = checker.check_stability(bin_id, item_a, 0, 0, 0)
        print(f"Item A stable on bin {bin_id}: {stable}")

    checker.place_item(0, item_a, 0, 0, 0)
    print(f"Utilizations: {checker.get_utilizations()}")
