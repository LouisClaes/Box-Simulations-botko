"""
Extreme Points Strategy for 3D bin packing.

Algorithm overview:
    Instead of scanning every grid cell (O(L*W) per orientation), this strategy
    only evaluates "extreme points" (EPs) generated from the corners of already-
    placed boxes. This drastically reduces the number of candidate positions from
    ~9600 (120x80 grid) to typically 10-100, while still finding high-quality
    placements.

    Extreme points are the natural positions where a new box can be placed
    adjacent to existing boxes. For each placed box, three EPs are generated:
      - Right:  (x_max, y)     -- immediately to the right of the box
      - Front:  (x, y_max)     -- immediately in front of the box
      - Top:    (x, y)         -- on top of the box (projected to surface)

    The origin (0, 0) is always included as a fallback so the first box and
    any box that doesn't fit near existing ones can still be placed.

Scoring:
    Candidates are ranked by a weighted score combining:
      - Low z (prefer bottom placements)
      - High contact ratio (walls + adjacent boxes)
      - Low wasted space below
      - Slight back-left corner preference

References:
    Crainic, T.G., Perboli, G., & Tadei, R. (2008).
    "Extreme Point-Based Heuristics for Three-Dimensional Bin Packing."
    INFORMS Journal on Computing, 20(3), 368-384.
"""

from typing import Optional, List, Tuple, Set
import numpy as np

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants (tuning hyperparameters)
# ─────────────────────────────────────────────────────────────────────────────

# Anti-float threshold: must match the simulator's MIN_ANTI_FLOAT_RATIO (0.30)
MIN_SUPPORT: float = 0.30

# Scoring weights
WEIGHT_HEIGHT: float = -3.0       # Strongly prefer lower z placements
WEIGHT_CONTACT: float = 2.0       # Reward touching walls and adjacent boxes
WEIGHT_WASTED_SPACE: float = -1.0  # Penalize empty gaps below the box
WEIGHT_CORNER: float = -0.5       # Slight preference for back-left corner

# Contact detection tolerance: how close (cm) surfaces must be to count as touching
CONTACT_TOLERANCE: float = 1.5

# A score so good we can stop searching (theoretical perfect placement at z=0)
PERFECT_SCORE_THRESHOLD: float = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Strategy implementation
# ─────────────────────────────────────────────────────────────────────────────

@register_strategy
class ExtremePointsStrategy(BaseStrategy):
    """
    Extreme Points heuristic for 3D bin packing.

    Generates candidate positions only at the corners/edges of already-placed
    boxes, dramatically reducing the search space compared to full grid scans.
    Each candidate is evaluated with a multi-criteria scoring function that
    balances low placement height, surface contact, gap minimization, and
    corner packing.

    Attributes:
        name: Strategy identifier for the registry ("extreme_points").
    """

    name: str = "extreme_points"

    def __init__(self) -> None:
        super().__init__()

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Find the best extreme-point placement for the given box.

        Steps:
            1. Generate extreme points from all currently placed boxes.
            2. Sort EPs by (z-height-at-point, x, y) for early termination.
            3. For each EP and each allowed orientation, check feasibility
               (bounds, height limit, support ratio) and compute a score.
            4. Return the best-scoring feasible candidate, or None.

        Args:
            box:       The box to place (original dimensions before rotation).
            bin_state: Current bin state (read-only).

        Returns:
            PlacementDecision with (x, y, orientation_idx) or None if no
            valid placement exists.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        # Resolve allowed orientations based on experiment config
        orientations = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Quick check: can this box fit in any orientation at all?
        can_fit_any = False
        for ol, ow, oh in orientations:
            if ol <= bin_cfg.length and ow <= bin_cfg.width and oh <= bin_cfg.height:
                can_fit_any = True
                break
        if not can_fit_any:
            return None

        # Generate extreme points from the current set of placed boxes
        extreme_points = self._generate_extreme_points(bin_state)

        # Sort EPs by estimated quality: low height first, then back-left corner
        # This enables early stopping when a perfect-score candidate is found
        extreme_points = self._sort_extreme_points(extreme_points, bin_state)

        best_score: float = -float("inf")
        best_candidate: Optional[Tuple[float, float, int]] = None  # (x, y, oidx)

        for ex, ey in extreme_points:
            for oidx, (ol, ow, oh) in enumerate(orientations):
                # --- Feasibility checks ---

                # Bounds check: box must fit within bin dimensions
                if ex + ol > bin_cfg.length + 1e-6:
                    continue
                if ey + ow > bin_cfg.width + 1e-6:
                    continue

                # Compute resting height (gravity drop)
                z = bin_state.get_height_at(ex, ey, ol, ow)

                # Height limit check
                if z + oh > bin_cfg.height + 1e-6:
                    continue

                # Support ratio check (anti-float): only for stacked placements
                if z > 0.5:
                    support_ratio = bin_state.get_support_ratio(ex, ey, ol, ow, z)
                    if support_ratio < MIN_SUPPORT:
                        continue

                # Stability check (stricter threshold when enabled)
                if cfg.enable_stability and z > 0.5:
                    support_ratio = bin_state.get_support_ratio(ex, ey, ol, ow, z)
                    if support_ratio < cfg.min_support_ratio:
                        continue

                # --- Scoring ---
                score = self._compute_score(
                    ex, ey, z, ol, ow, oh, bin_state, bin_cfg
                )

                if score > best_score:
                    best_score = score
                    best_candidate = (ex, ey, oidx)

                    # Early termination: if we found a near-perfect placement
                    # (on the floor, in the corner, with full contact), stop
                    if best_score > PERFECT_SCORE_THRESHOLD:
                        return PlacementDecision(
                            x=best_candidate[0],
                            y=best_candidate[1],
                            orientation_idx=best_candidate[2],
                        )

        if best_candidate is None:
            return None

        return PlacementDecision(
            x=best_candidate[0],
            y=best_candidate[1],
            orientation_idx=best_candidate[2],
        )

    # ── Extreme point generation ──────────────────────────────────────────

    def _generate_extreme_points(
        self, bin_state: BinState
    ) -> List[Tuple[float, float]]:
        """
        Generate extreme points from all placed boxes in the bin.

        For each placed box, up to 3 extreme points are created:
          - Right point:  (x_max, y)   -- right edge of the box
          - Front point:  (x, y_max)   -- front edge of the box
          - Top point:    (x, y)       -- top-left corner (projects to surface)

        The origin (0, 0) is always included to handle the empty-bin case
        and as a fallback position.

        Points outside bin bounds are filtered out. Duplicates are removed.

        Args:
            bin_state: Current bin state (read-only).

        Returns:
            List of unique (x, y) candidate positions.
        """
        bin_cfg = bin_state.config
        seen: Set[Tuple[float, float]] = set()
        points: List[Tuple[float, float]] = []

        # Always include the origin as a fallback
        origin = (0.0, 0.0)
        seen.add(origin)
        points.append(origin)

        for p in bin_state.placed_boxes:
            # Right point: immediately to the right of this box
            rp = (p.x_max, p.y)
            if rp not in seen and rp[0] < bin_cfg.length:
                seen.add(rp)
                points.append(rp)

            # Front point: immediately in front of this box
            fp = (p.x, p.y_max)
            if fp not in seen and fp[1] < bin_cfg.width:
                seen.add(fp)
                points.append(fp)

            # Top point: on top of this box (same x, y corner)
            # The actual z will be determined by get_height_at, but the
            # (x, y) position is the key for candidate generation
            tp = (p.x, p.y)
            if tp not in seen:
                seen.add(tp)
                points.append(tp)

        return points

    def _sort_extreme_points(
        self,
        points: List[Tuple[float, float]],
        bin_state: BinState,
    ) -> List[Tuple[float, float]]:
        """
        Sort extreme points by estimated quality for early-stop optimization.

        Sorts by (height_at_point, x, y) ascending so that the most promising
        candidates (low, back-left) are evaluated first.

        Args:
            points:    List of (x, y) extreme point candidates.
            bin_state: Current bin state for height queries.

        Returns:
            Sorted list of (x, y) points.
        """
        def sort_key(point: Tuple[float, float]) -> Tuple[float, float, float]:
            x, y = point
            # Use a minimal 1x1 probe to get the surface height at this point
            z = bin_state.get_height_at(x, y, 1.0, 1.0)
            return (z, x, y)

        return sorted(points, key=sort_key)

    # ── Scoring ───────────────────────────────────────────────────────────

    def _compute_score(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        bin_state: BinState,
        bin_cfg,
    ) -> float:
        """
        Score a candidate placement using a weighted multi-criteria function.

        Components:
          1. Height penalty:    prefer low z placements (DBLF principle)
          2. Contact reward:    prefer positions touching walls/existing boxes
          3. Wasted space:      penalize large empty gaps below the box
          4. Corner preference: slight bias toward back-left corner (0, 0)

        The formula:
            score = WEIGHT_HEIGHT * z
                  + WEIGHT_CONTACT * contact_ratio
                  + WEIGHT_WASTED_SPACE * wasted_space_fraction
                  + WEIGHT_CORNER * (x + y) / (length + width)

        Args:
            x, y, z:          Position of the box's back-left-bottom corner.
            ol, ow, oh:       Oriented box dimensions.
            bin_state:         Current bin state (read-only).
            bin_cfg:           Bin configuration.

        Returns:
            Scalar score (higher is better).
        """
        # 1. Contact ratio: fraction of the box's 6 faces touching walls or boxes
        contact_ratio = self._compute_contact_ratio(
            x, y, z, ol, ow, oh, bin_state, bin_cfg
        )

        # 2. Wasted space below the box: the gap between actual box volume
        #    below and the space that *could* have been filled
        wasted_space = self._compute_wasted_space_below(
            x, y, z, ol, ow, bin_state, bin_cfg
        )

        # 3. Corner distance: normalized distance from origin
        corner_dist = (x + y) / (bin_cfg.length + bin_cfg.width)

        score = (
            WEIGHT_HEIGHT * z
            + WEIGHT_CONTACT * contact_ratio
            + WEIGHT_WASTED_SPACE * wasted_space
            + WEIGHT_CORNER * corner_dist
        )

        return score

    def _compute_contact_ratio(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        bin_state: BinState,
        bin_cfg,
    ) -> float:
        """
        Compute the fraction of the box's 6 faces that are in contact with
        walls or other boxes.

        A face is considered "in contact" if it is within CONTACT_TOLERANCE
        of a wall or if the heightmap/placed-box data indicates adjacency.

        Returns:
            Float in [0, 1] representing the contact ratio.
        """
        contacts = 0
        total_faces = 6

        # Bottom face: in contact if z > 0 (resting on something) or z == 0 (floor)
        # On the floor, the bottom always has contact
        if z < CONTACT_TOLERANCE:
            contacts += 1  # Floor contact
        else:
            # Check if there's something below (there should be, since z is
            # computed from heightmap — but the support ratio tells us how much)
            support = bin_state.get_support_ratio(x, y, ol, ow, z)
            if support > 0.0:
                contacts += 1

        # Left wall: x == 0
        if x < CONTACT_TOLERANCE:
            contacts += 1

        # Back wall: y == 0
        if y < CONTACT_TOLERANCE:
            contacts += 1

        # Right wall: x + ol == bin_length
        if abs(x + ol - bin_cfg.length) < CONTACT_TOLERANCE:
            contacts += 1

        # Front wall: y + ow == bin_width
        if abs(y + ow - bin_cfg.width) < CONTACT_TOLERANCE:
            contacts += 1

        # Top face: check if anything is directly above (rare, but possible
        # if we're filling a gap). We skip this as it's less relevant for
        # placement quality — top contact is not a goal during packing.

        # Check adjacency to other placed boxes (left, right, front, back faces)
        for p in bin_state.placed_boxes:
            # Vertical overlap required for lateral contact
            vert_overlap = (
                z < p.z_max and z + oh > p.z
            )
            if not vert_overlap:
                continue

            # Depth overlap (y-axis) required for left/right contact
            depth_overlap = (y < p.y_max and y + ow > p.y)

            # Width overlap (x-axis) required for front/back contact
            width_overlap = (x < p.x_max and x + ol > p.x)

            # Right face of candidate touches left face of placed box
            if depth_overlap and abs(x + ol - p.x) < CONTACT_TOLERANCE:
                contacts += 1

            # Left face of candidate touches right face of placed box
            if depth_overlap and abs(x - p.x_max) < CONTACT_TOLERANCE:
                contacts += 1

            # Front face of candidate touches back face of placed box
            if width_overlap and abs(y + ow - p.y) < CONTACT_TOLERANCE:
                contacts += 1

            # Back face of candidate touches front face of placed box
            if width_overlap and abs(y - p.y_max) < CONTACT_TOLERANCE:
                contacts += 1

        # Cap at total_faces (a face can only count once, but our simple
        # counting may double-count if a wall and a box share the same face)
        return min(contacts, total_faces) / total_faces

    def _compute_wasted_space_below(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        bin_state: BinState,
        bin_cfg,
    ) -> float:
        """
        Compute the fraction of wasted space below this box relative to
        total bin volume.

        Wasted space = the column of space from the floor up to z within the
        box's footprint, minus the volume of already-placed boxes occupying
        that column.

        For floor placements (z == 0), wasted space is 0.

        Returns:
            Float in [0, 1], fraction of bin volume that would be wasted.
        """
        if z < 1e-6:
            return 0.0

        # Total column volume below this box
        column_volume = z * ol * ow

        # Subtract volume of placed boxes that overlap in the footprint AND
        # are below z. This is an approximation using bounding-box overlap.
        filled_volume = 0.0
        for p in bin_state.placed_boxes:
            # Compute overlap in x-axis
            overlap_x = max(0.0, min(x + ol, p.x_max) - max(x, p.x))
            # Compute overlap in y-axis
            overlap_y = max(0.0, min(y + ow, p.y_max) - max(y, p.y))
            # Compute overlap in z-axis (only below z)
            overlap_z = max(0.0, min(z, p.z_max) - max(0.0, p.z))

            filled_volume += overlap_x * overlap_y * overlap_z

        wasted = max(0.0, column_volume - filled_volume)

        # Normalize by bin volume
        if bin_cfg.volume > 0:
            return wasted / bin_cfg.volume
        return 0.0
