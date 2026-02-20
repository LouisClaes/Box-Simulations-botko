"""
Wall Building Strategy -- back-to-front strip-based packing for 3D bins.

Algorithm overview:
    Inspired by how human palletizers work: build dense wall-like structures
    from the back of the bin forward. The bin is divided into vertical strips
    along the x-axis, and each strip is filled before moving to the next.

    The strategy tracks a "build front" that advances from y=0 (back wall)
    towards y=bin_width (front) as strips fill up. Within each strip, boxes
    are placed as close to the back wall and side walls as possible, with
    bonuses for wall contact and adjacency to existing boxes.

Key concepts:
    - Strip: A vertical column of the bin along the x-axis. Width is adaptive
      based on observed box dimensions (median width of boxes seen so far).
    - Build front: The y-coordinate beyond which the heightmap is essentially
      empty. Detected from the heightmap at each placement decision.
    - Wall bonus: Extra score for positions touching bin walls (back, left,
      right) which create stable, dense structures.
    - Adjacency: Reward for positions touching existing boxes on their sides.

Advantages:
    - Creates very dense structures against walls
    - Maximizes contact area for stability
    - Works especially well with rectangular boxes that align to strip widths
    - Natural layering behaviour emerges from back-to-front filling

References:
    Inspired by practical palletizing heuristics and the "wall building"
    approach described in Bischoff & Ratcliff (1995), "Issues in the
    development of approaches to container loading."
"""

from typing import Optional, List, Tuple
import numpy as np

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Default strip width (cm) used until enough boxes have been observed to
# compute a median width. Chosen to be a reasonable middle-ground for
# typical box sizes on a 120x80 pallet.
DEFAULT_STRIP_WIDTH: float = 25.0

# Minimum / maximum strip width bounds (cm). Prevents degenerate strips
# that are too narrow (wasted space between strips) or too wide (no benefit
# from strip-based ordering).
MIN_STRIP_WIDTH: float = 10.0
MAX_STRIP_WIDTH: float = 60.0

# Grid scan step size within a strip (cm).
SCAN_STEP: float = 1.0

# When a strip's average height exceeds this fraction of bin height in the
# back region, the strip is considered "full" and we prefer other strips.
STRIP_FULL_THRESHOLD: float = 0.85

# Number of boxes to observe before adapting strip width to the median.
ADAPT_AFTER_N_BOXES: int = 3

# Anti-float threshold -- matches simulator's MIN_ANTI_FLOAT_RATIO.
MIN_SUPPORT: float = 0.30

# Scoring weights
WEIGHT_WALL_BACK: float = 2.0      # Bonus for back wall contact (y ~ 0)
WEIGHT_WALL_LEFT: float = 1.5      # Bonus for left wall contact (x ~ 0)
WEIGHT_WALL_RIGHT: float = 1.5     # Bonus for right wall contact (x+ol ~ length)
WEIGHT_ADJACENCY: float = 2.0      # Bonus per adjacent face
WEIGHT_HEIGHT_PENALTY: float = 3.0  # Penalty for high z placement
WEIGHT_DEPTH_PENALTY: float = 0.5   # Penalty for being far from back wall (high y)

# Tolerance for detecting wall contact (cm).
WALL_TOLERANCE: float = 1.0

# Tolerance for detecting adjacency between box faces (cm).
ADJACENCY_TOLERANCE: float = 1.5


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@register_strategy
class WallBuildingStrategy(BaseStrategy):
    """
    Wall building strategy: fill the bin from back wall to front using strips.

    The bin is divided into vertical strips along the x-axis. Each strip is
    filled back-to-front, preferring positions that contact bin walls and
    existing boxes. When a strip becomes full, the strategy moves to the
    next one.

    Internal state (reset each episode):
        _strip_width:     Current adaptive strip width.
        _seen_widths:     Box widths observed so far for adaptive sizing.
        _box_count:       Number of boxes seen this episode.

    Attributes:
        name: Strategy identifier for the registry ("wall_building").
    """

    name: str = "wall_building"

    def __init__(self) -> None:
        super().__init__()
        self._strip_width: float = DEFAULT_STRIP_WIDTH
        self._seen_widths: List[float] = []
        self._box_count: int = 0
        self._scan_step: float = SCAN_STEP

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Reset internal state for a new episode."""
        super().on_episode_start(config)
        self._strip_width = DEFAULT_STRIP_WIDTH
        self._seen_widths = []
        self._box_count = 0
        self._scan_step = max(SCAN_STEP, config.bin.resolution)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Find the best wall-building position for the given box.

        Algorithm:
            1. Update internal state (observe box dimensions, adapt strips).
            2. Detect the current build front from the heightmap.
            3. Identify strips and sort by fill level (least full first).
            4. For each strip, scan positions and score them.
            5. If no position found in any strip, fall back to full-bin BLF.

        Args:
            box:       The box to place (original dimensions).
            bin_state: Current 3D bin state. NOT modified by this method.

        Returns:
            PlacementDecision(x, y, orientation_idx) or None if the box
            cannot be placed anywhere.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        # Track box dimensions for adaptive strip width
        self._observe_box(box)

        # Resolve allowed orientations
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Quick check: can the box fit in any orientation?
        fitting_orientations = [
            (oidx, ol, ow, oh) for oidx, (ol, ow, oh) in enumerate(orientations)
            if ol <= bin_cfg.length and ow <= bin_cfg.width and oh <= bin_cfg.height
        ]
        if not fitting_orientations:
            return None

        # Detect how far forward the packing has extended
        build_front = self._get_build_front(bin_state)

        # Generate strips along the x-axis
        strips = self._generate_strips(bin_cfg)

        # Sort strips: prefer the least-filled strip in the back region
        strips = self._sort_strips_by_fill(strips, bin_state, bin_cfg, build_front)

        # Try to place in each strip, starting with the most promising
        best_score: float = -np.inf
        best_candidate: Optional[Tuple[float, float, int]] = None

        for strip_start, strip_end in strips:
            result = self._scan_strip(
                box, bin_state, bin_cfg, cfg,
                strip_start, strip_end, build_front,
                fitting_orientations,
            )
            if result is not None:
                score, x, y, oidx = result
                if score > best_score:
                    best_score = score
                    best_candidate = (x, y, oidx)

        # If we found a good placement in any strip, return it
        if best_candidate is not None:
            return PlacementDecision(
                x=best_candidate[0],
                y=best_candidate[1],
                orientation_idx=best_candidate[2],
            )

        # Fallback: full bin BLF scan (ignoring strips)
        return self._fallback_blf(box, bin_state, bin_cfg, cfg, fitting_orientations)

    # ------------------------------------------------------------------
    # Adaptive strip width
    # ------------------------------------------------------------------

    def _observe_box(self, box: Box) -> None:
        """
        Record a box's dimensions to adapt strip width over time.

        After ADAPT_AFTER_N_BOXES boxes, the strip width is set to the
        median of the smaller horizontal dimension (min of length, width)
        of all observed boxes, clamped to [MIN_STRIP_WIDTH, MAX_STRIP_WIDTH].
        """
        self._box_count += 1
        # Use the smaller horizontal dimension as the "width" for strip sizing
        self._seen_widths.append(min(box.length, box.width))

        if self._box_count >= ADAPT_AFTER_N_BOXES and self._seen_widths:
            median_w = float(np.median(self._seen_widths))
            self._strip_width = max(MIN_STRIP_WIDTH, min(median_w, MAX_STRIP_WIDTH))

    # ------------------------------------------------------------------
    # Build front detection
    # ------------------------------------------------------------------

    def _get_build_front(self, bin_state: BinState) -> float:
        """
        Detect how far forward (along the y-axis) the packing extends.

        Scans columns from the front of the bin backwards. The build front
        is the y-position of the furthest forward column that has any
        non-zero height.

        Args:
            bin_state: Current bin state (read-only).

        Returns:
            The y-coordinate (in real-world cm) of the build front.
            Returns 0.0 if the bin is empty.
        """
        heightmap = bin_state.heightmap
        resolution = bin_state.config.resolution
        grid_w = heightmap.shape[1]

        for gy in range(grid_w - 1, -1, -1):
            col = heightmap[:, gy]
            if np.any(col > 0):
                return (gy + 1) * resolution

        return 0.0

    # ------------------------------------------------------------------
    # Strip management
    # ------------------------------------------------------------------

    def _generate_strips(self, bin_cfg) -> List[Tuple[float, float]]:
        """
        Divide the bin along the x-axis into strips of width _strip_width.

        The last strip may be narrower if the bin length is not evenly
        divisible by the strip width.

        Args:
            bin_cfg: Bin configuration.

        Returns:
            List of (strip_start_x, strip_end_x) tuples.
        """
        strips: List[Tuple[float, float]] = []
        x = 0.0
        while x < bin_cfg.length:
            strip_end = min(x + self._strip_width, bin_cfg.length)
            strips.append((x, strip_end))
            x = strip_end
        return strips

    def _sort_strips_by_fill(
        self,
        strips: List[Tuple[float, float]],
        bin_state: BinState,
        bin_cfg,
        build_front: float,
    ) -> List[Tuple[float, float]]:
        """
        Sort strips by their average fill level (least filled first).

        The fill level of a strip is the average height in its x-range
        across the entire y-axis. Strips that are already very full
        (above STRIP_FULL_THRESHOLD) are pushed to the back of the list.

        Args:
            strips:      List of (strip_start, strip_end) tuples.
            bin_state:   Current bin state.
            bin_cfg:     Bin configuration.
            build_front: Current build front y-position.

        Returns:
            Sorted list of strips.
        """
        heightmap = bin_state.heightmap
        resolution = bin_cfg.resolution
        bin_height = bin_cfg.height

        def strip_fill_key(strip: Tuple[float, float]) -> Tuple[int, float]:
            s_start, s_end = strip
            gx_start = int(round(s_start / resolution))
            gx_end = min(int(round(s_end / resolution)), bin_cfg.grid_l)
            if gx_start >= gx_end:
                return (1, 0.0)

            region = heightmap[gx_start:gx_end, :]
            avg_height = float(np.mean(region))
            is_full = 1 if avg_height > bin_height * STRIP_FULL_THRESHOLD else 0

            return (is_full, avg_height)

        return sorted(strips, key=strip_fill_key)

    # ------------------------------------------------------------------
    # Strip scanning
    # ------------------------------------------------------------------

    def _scan_strip(
        self,
        box: Box,
        bin_state: BinState,
        bin_cfg,
        cfg: ExperimentConfig,
        strip_start: float,
        strip_end: float,
        build_front: float,
        orientations: List[Tuple[int, float, float, float]],
    ) -> Optional[Tuple[float, float, float, int]]:
        """
        Scan positions within a single strip and return the best-scoring
        candidate as (score, x, y, orientation_idx).

        The scan covers:
        - x: from strip_start to strip_end (constrained by box dimensions)
        - y: from 0 to build_front + strip_width (allow some expansion)

        Args:
            box:          The box to place.
            bin_state:    Current bin state (read-only).
            bin_cfg:      Bin configuration.
            cfg:          Experiment configuration.
            strip_start:  Left edge of the strip (x-axis).
            strip_end:    Right edge of the strip (x-axis).
            build_front:  Current build front (y-axis).
            orientations: List of (oidx, ol, ow, oh) fitting orientations.

        Returns:
            (score, x, y, oidx) of the best candidate, or None if no valid
            position exists in this strip.
        """
        step = self._scan_step
        heightmap = bin_state.heightmap

        # Allow scanning slightly beyond the build front to start new layers
        y_limit = min(build_front + self._strip_width, bin_cfg.width)
        # Always scan at least the full bin width if the bin is nearly empty
        if build_front < self._strip_width:
            y_limit = bin_cfg.width

        best_score: float = -np.inf
        best_result: Optional[Tuple[float, float, float, int]] = None

        for oidx, ol, ow, oh in orientations:
            # x range: within this strip, accounting for box length
            x_min = strip_start
            x_max = min(strip_end, bin_cfg.length - ol + 1e-6)
            if x_min > x_max:
                continue

            x = x_min
            while x <= x_max + 1e-6:
                y = 0.0
                while y + ow <= y_limit + 1e-6:
                    # Compute resting height
                    z = bin_state.get_height_at(x, y, ol, ow)

                    # Height bounds check
                    if z + oh > bin_cfg.height + 1e-6:
                        y += step
                        continue

                    # Anti-float support check
                    if z > 0.5:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < MIN_SUPPORT:
                            y += step
                            continue

                    # Stability check (optional, stricter)
                    if cfg.enable_stability and z > 0.5:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < cfg.min_support_ratio:
                            y += step
                            continue

                    # Compute wall-building score
                    score = self._compute_score(
                        x, y, z, ol, ow, oh, bin_state, bin_cfg, heightmap,
                    )

                    if score > best_score:
                        best_score = score
                        best_result = (score, x, y, oidx)

                    y += step
                x += step

        return best_result

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

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
        heightmap: np.ndarray,
    ) -> float:
        """
        Score a candidate placement using wall-building criteria.

        Components:
            1. Wall bonuses: reward touching back, left, and right walls.
            2. Adjacency: reward touching existing boxes on lateral faces.
            3. Height penalty: prefer low placements.
            4. Depth penalty: prefer positions close to the back wall.

        Formula:
            score = wall_bonus
                    + WEIGHT_ADJACENCY * adjacency_count
                    - WEIGHT_HEIGHT_PENALTY * z / bin_height
                    - WEIGHT_DEPTH_PENALTY * y / bin_width

        Args:
            x, y, z:     Position of the box's back-left-bottom corner.
            ol, ow, oh:  Oriented box dimensions.
            bin_state:    Current bin state (read-only).
            bin_cfg:      Bin configuration.
            heightmap:    Current heightmap array (read-only).

        Returns:
            Scalar score (higher is better).
        """
        bin_height = bin_cfg.height
        bin_width = bin_cfg.width
        bin_length = bin_cfg.length

        # 1. Wall contact bonuses
        wall_bonus = 0.0
        if y < WALL_TOLERANCE:
            wall_bonus += WEIGHT_WALL_BACK       # Back wall
        if x < WALL_TOLERANCE:
            wall_bonus += WEIGHT_WALL_LEFT       # Left wall
        if x + ol > bin_length - WALL_TOLERANCE:
            wall_bonus += WEIGHT_WALL_RIGHT      # Right wall

        # 2. Adjacency to existing boxes
        adjacency = self._count_adjacent_faces(
            x, y, z, ol, ow, oh, bin_state, heightmap, bin_cfg,
        )

        # 3. Height penalty (normalized)
        height_pen = z / bin_height if bin_height > 0 else 0.0

        # 4. Depth penalty (prefer low y = close to back wall)
        depth_pen = y / bin_width if bin_width > 0 else 0.0

        score = (
            wall_bonus
            + WEIGHT_ADJACENCY * adjacency
            - WEIGHT_HEIGHT_PENALTY * height_pen
            - WEIGHT_DEPTH_PENALTY * depth_pen
        )

        return score

    def _count_adjacent_faces(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        bin_state: BinState,
        heightmap: np.ndarray,
        bin_cfg,
    ) -> float:
        """
        Count the number of faces of the candidate box that are adjacent
        to already-placed boxes.

        Uses a heightmap-based approximation for lateral adjacency: checks
        the columns immediately outside each of the four lateral faces. If
        the max height in those adjacent columns overlaps vertically with
        the candidate box, it counts as an adjacent face.

        Also checks the bottom face: if z > 0, there is support contact below.

        Args:
            x, y, z:     Position of the candidate box.
            ol, ow, oh:  Oriented dimensions.
            bin_state:    Current bin state.
            heightmap:    Current heightmap (read-only).
            bin_cfg:      Bin configuration.

        Returns:
            Float count of adjacent faces (0.0 to 5.0).
        """
        resolution = bin_cfg.resolution
        grid_l = bin_cfg.grid_l
        grid_w = bin_cfg.grid_w
        adjacency = 0.0

        gx = int(round(x / resolution))
        gy = int(round(y / resolution))
        gx_end = min(gx + int(round(ol / resolution)), grid_l)
        gy_end = min(gy + int(round(ow / resolution)), grid_w)

        box_z_min = z
        box_z_max = z + oh

        # Bottom face: if z > 0, we are resting on something
        if z > 0.5:
            adjacency += 1.0

        # Left face (x = gx - 1): check column just to the left
        if gx > 0:
            left_col = heightmap[gx - 1, gy:gy_end]
            if left_col.size > 0:
                max_left_h = float(np.max(left_col))
                # Vertical overlap: the adjacent column reaches into our z-range
                if max_left_h > box_z_min + ADJACENCY_TOLERANCE:
                    adjacency += 1.0

        # Right face (x = gx_end): check column just to the right
        if gx_end < grid_l:
            right_col = heightmap[gx_end, gy:gy_end]
            if right_col.size > 0:
                max_right_h = float(np.max(right_col))
                if max_right_h > box_z_min + ADJACENCY_TOLERANCE:
                    adjacency += 1.0

        # Back face (y = gy - 1): check column just behind
        if gy > 0:
            back_col = heightmap[gx:gx_end, gy - 1]
            if back_col.size > 0:
                max_back_h = float(np.max(back_col))
                if max_back_h > box_z_min + ADJACENCY_TOLERANCE:
                    adjacency += 1.0

        # Front face (y = gy_end): check column just in front
        if gy_end < grid_w:
            front_col = heightmap[gx:gx_end, gy_end]
            if front_col.size > 0:
                max_front_h = float(np.max(front_col))
                if max_front_h > box_z_min + ADJACENCY_TOLERANCE:
                    adjacency += 1.0

        return adjacency

    # ------------------------------------------------------------------
    # Fallback: full-bin BLF scan
    # ------------------------------------------------------------------

    def _fallback_blf(
        self,
        box: Box,
        bin_state: BinState,
        bin_cfg,
        cfg: ExperimentConfig,
        orientations: List[Tuple[int, float, float, float]],
    ) -> Optional[PlacementDecision]:
        """
        Full-bin Bottom-Left-Fill scan as a last resort when no strip
        placement is found.

        This is identical to the baseline BLF algorithm: scan every
        position on the grid (at the configured step) for every orientation
        and pick the lowest feasible placement (z, x, y priority).

        Args:
            box:          The box to place.
            bin_state:    Current bin state (read-only).
            bin_cfg:      Bin configuration.
            cfg:          Experiment configuration.
            orientations: List of (oidx, ol, ow, oh) fitting orientations.

        Returns:
            PlacementDecision or None.
        """
        step = self._scan_step
        best: Optional[Tuple[float, float, float, int]] = None  # (z, x, y, oidx)

        for oidx, ol, ow, oh in orientations:
            x = 0.0
            while x + ol <= bin_cfg.length + 1e-6:
                y = 0.0
                while y + ow <= bin_cfg.width + 1e-6:
                    z = bin_state.get_height_at(x, y, ol, ow)

                    # Height check
                    if z + oh > bin_cfg.height + 1e-6:
                        y += step
                        continue

                    # Anti-float check
                    if z > 0.5:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < MIN_SUPPORT:
                            y += step
                            continue

                    # Stability check
                    if cfg.enable_stability and z > 0.5:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < cfg.min_support_ratio:
                            y += step
                            continue

                    candidate = (z, x, y, oidx)
                    if best is None or candidate < best:
                        best = candidate

                    y += step
                x += step

        if best is None:
            return None

        _, bx, by, b_oidx = best
        return PlacementDecision(x=bx, y=by, orientation_idx=b_oidx)
