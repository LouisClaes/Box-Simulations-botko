"""
Column Fill Strategy -- virtual column-based 3D bin packing.

**NOVEL STRATEGY** — not derived from any published paper.

Algorithm overview
~~~~~~~~~~~~~~~~~~
Divides the pallet floor (120x80 cm) into a grid of virtual columns and fills
each column systematically from bottom to top.  Instead of scanning every grid
cell on every placement, the strategy selects the best *column* for each box,
producing dense vertical stacks with minimal wasted space between columns.

Workflow for each box:
  1. Resolve allowed orientations.
  2. For each column, for each orientation:
       - Check if the box physically fits within the column boundaries.
       - Score the column by area efficiency, remaining height room, and
         adjacency to already-filled columns.
  3. Pick the highest-scoring (column, orientation) pair.
  4. If no column can accommodate the box exactly:
       - Try spanning two adjacent columns (horizontal or vertical neighbour).
  5. Final fallback: full BLF grid scan (same as baseline).

Adaptive column sizing:
  The first N_CALIBRATION_BOXES boxes (default 5) are observed before the
  column grid is finalized.  Their median L/W dimensions determine column
  width and depth.  If no calibration data is available, a default 20x20 cm
  column grid is used.

Column selection priority (score_column):
    score = 2.0 * area_efficiency
          + 2.0 * height_room
          + 0.2 * adjacency_bonus

Hyperparameters (module-level constants):
    DEFAULT_COL_SIZE      — default column size when no calibration data (20 cm)
    N_CALIBRATION_BOXES   — how many boxes to observe before fixing column grid
    MIN_SUPPORT           — anti-float threshold (0.30)

This strategy does NOT modify the original bin_state.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ─────────────────────────────────────────────────────────────────────────────
# Constants / Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

# Anti-float threshold — matches the simulator's threshold.
MIN_SUPPORT: float = 0.30

# Default column size (cm) when no calibration data is available.
DEFAULT_COL_SIZE: float = 20.0

# Number of boxes to observe before finalizing the column grid.
# During calibration, a BLF fallback is used for placement.
N_CALIBRATION_BOXES: int = 5

# Minimum column dimension (cm).  Columns smaller than this are merged.
MIN_COL_DIM: float = 10.0

# Maximum column dimension (cm).  Columns larger than this are split.
MAX_COL_DIM: float = 50.0

# ── Scoring weights ──────────────────────────────────────────────────────────

# How much to reward area efficiency (box footprint / column footprint).
W_AREA_EFFICIENCY: float = 2.0

# How much to reward remaining height room in the column.
W_HEIGHT_ROOM: float = 2.0

# Bonus per filled neighbouring column (compactness).
W_ADJACENCY: float = 0.2

# BLF scan step size for fallback (cm).
BLF_STEP: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Column data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Column:
    """
    A virtual column on the pallet floor.

    Attributes:
        x:              X coordinate of the column origin.
        y:              Y coordinate of the column origin.
        width:          Extent along the X axis (length).
        depth:          Extent along the Y axis (width).
        current_height: Tracked internal height (for scoring priority).
        n_boxes:        Number of boxes placed in this column.
        col_idx:        Index in the column grid (row-major).
        row:            Row index in the grid.
        col:            Column index in the grid.
    """
    x: float
    y: float
    width: float
    depth: float
    current_height: float = 0.0
    n_boxes: int = 0
    col_idx: int = 0
    row: int = 0
    col: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

@register_strategy
class ColumnFillStrategy(BaseStrategy):
    """
    Column Fill placement strategy.

    Divides the pallet into virtual columns and fills each column from bottom
    to top.  Columns are sized adaptively based on the first few boxes.  When
    no column can accept a box, a spanning placement across adjacent columns
    is attempted, followed by a full BLF fallback.

    The strategy never modifies the original ``bin_state`` — it reads heights
    and support ratios and maintains its own lightweight column tracking for
    selection priority.
    """

    name = "column_fill"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = BLF_STEP
        # Column grid state (reset per episode).
        self._columns: List[Column] = []
        self._n_cols_x: int = 0
        self._n_cols_y: int = 0
        self._grid_ready: bool = False
        # Calibration buffer.
        self._calibration_boxes: List[Box] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Reset all column state at the start of a new packing episode."""
        super().on_episode_start(config)
        self._scan_step = max(1.0, config.bin.resolution)
        self._columns = []
        self._n_cols_x = 0
        self._n_cols_y = 0
        self._grid_ready = False
        self._calibration_boxes = []

    # ── Main entry point ──────────────────────────────────────────────────

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Decide where to place *box* using the column-fill heuristic.

        1. If the column grid is not yet ready, record calibration data
           and use BLF fallback.
        2. Try to fit the box into the best-scoring column.
        3. If no single column works, try spanning adjacent columns.
        4. Final fallback: full BLF grid scan.

        Args:
            box:       The box to place (original dimensions).
            bin_state: Current 3D bin state (read-only).

        Returns:
            PlacementDecision or None.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        orientations = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Quick reject: does the box fit at all?
        any_fits = any(
            ol <= bin_cfg.length and ow <= bin_cfg.width and oh <= bin_cfg.height
            for ol, ow, oh in orientations
        )
        if not any_fits:
            return None

        # ── Calibration phase ───────────────────────────────────────────
        if not self._grid_ready:
            self._calibration_boxes.append(box)
            if len(self._calibration_boxes) >= N_CALIBRATION_BOXES:
                self._build_column_grid(bin_cfg)
                self._grid_ready = True
            # During calibration, use BLF fallback.
            return self._blf_fallback(box, orientations, bin_state, bin_cfg)

        # ── Phase 1: try best column ────────────────────────────────────
        result = self._try_column_placement(box, orientations, bin_state, bin_cfg)
        if result is not None:
            return result

        # ── Phase 2: try spanning two adjacent columns ──────────────────
        result = self._try_spanning_placement(box, orientations, bin_state, bin_cfg)
        if result is not None:
            return result

        # ── Phase 3: full BLF fallback ──────────────────────────────────
        return self._blf_fallback(box, orientations, bin_state, bin_cfg)

    # ── Column grid construction ──────────────────────────────────────────

    def _build_column_grid(self, bin_cfg) -> None:
        """
        Build the virtual column grid based on calibration data.

        Uses the median length and width of the observed boxes to determine
        column dimensions.  Columns are snapped to the bin boundaries so no
        space is wasted along the edges.

        Args:
            bin_cfg: Bin configuration.
        """
        if not self._calibration_boxes:
            col_w = DEFAULT_COL_SIZE
            col_d = DEFAULT_COL_SIZE
        else:
            lengths = sorted(b.length for b in self._calibration_boxes)
            widths = sorted(b.width for b in self._calibration_boxes)
            median_l = lengths[len(lengths) // 2]
            median_w = widths[len(widths) // 2]

            # Column size = median box dimension, clamped to reasonable range.
            col_w = max(MIN_COL_DIM, min(MAX_COL_DIM, median_l))
            col_d = max(MIN_COL_DIM, min(MAX_COL_DIM, median_w))

        # Compute how many columns fit along each axis.
        n_cols_x = max(1, int(round(bin_cfg.length / col_w)))
        n_cols_y = max(1, int(round(bin_cfg.width / col_d)))

        # Adjust column sizes to exactly tile the bin (no gaps at edges).
        actual_col_w = bin_cfg.length / n_cols_x
        actual_col_d = bin_cfg.width / n_cols_y

        self._n_cols_x = n_cols_x
        self._n_cols_y = n_cols_y
        self._columns = []

        idx = 0
        for row in range(n_cols_y):
            for col in range(n_cols_x):
                self._columns.append(Column(
                    x=col * actual_col_w,
                    y=row * actual_col_d,
                    width=actual_col_w,
                    depth=actual_col_d,
                    col_idx=idx,
                    row=row,
                    col=col,
                ))
                idx += 1

    # ── Column-based placement ────────────────────────────────────────────

    def _try_column_placement(
        self,
        box: Box,
        orientations: List[Tuple[float, float, float]],
        bin_state: BinState,
        bin_cfg,
    ) -> Optional[PlacementDecision]:
        """
        Try to place the box inside a single column.

        For each (column, orientation) pair where the box fits within the
        column boundaries, compute a score and return the best.

        Args:
            box:          The box to place.
            orientations: List of (ol, ow, oh) tuples.
            bin_state:    Current bin state (read-only).
            bin_cfg:      Bin configuration.

        Returns:
            PlacementDecision or None.
        """
        cfg = self.config
        best_score: float = -1e18
        best_result: Optional[Tuple[float, float, int]] = None

        for column in self._columns:
            for oidx, (ol, ow, oh) in enumerate(orientations):
                # Skip orientations too large for the bin.
                if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                    continue

                # Check: does the box fit within this column?
                if ol > column.width + 1e-6 or ow > column.depth + 1e-6:
                    continue

                # Place at the column origin — the simulator computes z.
                x = column.x
                y = column.y
                z = bin_state.get_height_at(x, y, ol, ow)

                # Height bounds check.
                if z + oh > bin_cfg.height + 1e-6:
                    continue

                # Anti-float support check.
                if z > 0.5:
                    sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                    if sr < MIN_SUPPORT:
                        continue

                # Stricter stability when enabled.
                if cfg.enable_stability and z > 0.5:
                    sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                    if sr < cfg.min_support_ratio:
                        continue

                # ── Score this (column, orientation) pair ────────────
                score = self._score_column(column, ol, ow, oh, z, bin_cfg)

                if score > best_score:
                    best_score = score
                    best_result = (x, y, oidx)

        if best_result is None:
            return None

        bx, by, b_oidx = best_result
        # Update internal column tracking.
        self._update_column_after_placement(bx, by, bin_state, orientations[b_oidx])
        return PlacementDecision(x=bx, y=by, orientation_idx=b_oidx)

    def _score_column(
        self,
        column: Column,
        ol: float,
        ow: float,
        oh: float,
        z: float,
        bin_cfg,
    ) -> float:
        """
        Score a (column, orientation) pair.

        Components:
          - area_efficiency: (box footprint) / (column footprint).
            Penalizes placing a small box in a large column.
          - height_room: fraction of bin height still available above z+oh.
            Prefers shorter columns (more room to stack).
          - adjacency_bonus: number of filled neighbour columns.
            Encourages compact packing.

        Returns:
            Scalar score (higher is better).
        """
        col_area = column.width * column.depth
        if col_area <= 0:
            return -1e18

        area_efficiency = (ol * ow) / col_area

        # Prefer columns with more remaining room.
        top_after = z + oh
        height_room = 1.0 - top_after / bin_cfg.height if bin_cfg.height > 0 else 0.0
        height_room = max(height_room, 0.0)

        # Adjacency bonus: count how many of the 4 neighbours have boxes.
        adjacency = self._count_filled_neighbours(column)

        score = (
            W_AREA_EFFICIENCY * area_efficiency
            + W_HEIGHT_ROOM * height_room
            + W_ADJACENCY * adjacency
        )
        return score

    def _count_filled_neighbours(self, column: Column) -> int:
        """
        Count how many of the 4-connected neighbours of *column* have at
        least one box placed in them.

        Returns:
            Integer count in [0, 4].
        """
        count = 0
        row, col = column.row, column.col
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self._n_cols_y and 0 <= nc < self._n_cols_x:
                neighbour_idx = nr * self._n_cols_x + nc
                if self._columns[neighbour_idx].n_boxes > 0:
                    count += 1
        return count

    def _update_column_after_placement(
        self,
        px: float,
        py: float,
        bin_state: BinState,
        orientation: Tuple[float, float, float],
    ) -> None:
        """
        Update the internal column tracking after a box is placed.

        Finds the column whose origin is closest to (px, py) and increments
        its box count and tracked height.  The tracked height is advisory
        (actual z always comes from bin_state.get_height_at).

        Args:
            px, py:      Position of the placed box.
            bin_state:   Current bin state (for height query).
            orientation: (ol, ow, oh) of the placed box.
        """
        ol, ow, oh = orientation
        best_col = None
        best_dist = float("inf")
        for column in self._columns:
            dist = abs(column.x - px) + abs(column.y - py)
            if dist < best_dist:
                best_dist = dist
                best_col = column
        if best_col is not None:
            best_col.n_boxes += 1
            # Use the actual height from the bin state for accuracy.
            z = bin_state.get_height_at(px, py, ol, ow)
            best_col.current_height = max(best_col.current_height, z + oh)

    # ── Spanning placement (across 2 adjacent columns) ────────────────────

    def _try_spanning_placement(
        self,
        box: Box,
        orientations: List[Tuple[float, float, float]],
        bin_state: BinState,
        bin_cfg,
    ) -> Optional[PlacementDecision]:
        """
        Try to place the box spanning two horizontally or vertically
        adjacent columns.

        For each pair of adjacent columns, we test if the combined area
        can fit the box.  The placement origin is the top-left corner of
        the earlier column.

        Args:
            box:          The box to place.
            orientations: List of (ol, ow, oh) tuples.
            bin_state:    Current bin state (read-only).
            bin_cfg:      Bin configuration.

        Returns:
            PlacementDecision or None.
        """
        cfg = self.config
        best_score: float = -1e18
        best_result: Optional[Tuple[float, float, int]] = None

        for column in self._columns:
            row, col_i = column.row, column.col

            # Try right neighbour (span along x).
            if col_i + 1 < self._n_cols_x:
                right_idx = row * self._n_cols_x + (col_i + 1)
                right_col = self._columns[right_idx]
                span_w = column.width + right_col.width
                span_d = min(column.depth, right_col.depth)

                result = self._eval_span(
                    column.x, column.y, span_w, span_d,
                    orientations, bin_state, bin_cfg, cfg,
                    best_score,
                )
                if result is not None:
                    score, bx, by, b_oidx = result
                    if score > best_score:
                        best_score = score
                        best_result = (bx, by, b_oidx)

            # Try front neighbour (span along y).
            if row + 1 < self._n_cols_y:
                front_idx = (row + 1) * self._n_cols_x + col_i
                front_col = self._columns[front_idx]
                span_w = min(column.width, front_col.width)
                span_d = column.depth + front_col.depth

                result = self._eval_span(
                    column.x, column.y, span_w, span_d,
                    orientations, bin_state, bin_cfg, cfg,
                    best_score,
                )
                if result is not None:
                    score, bx, by, b_oidx = result
                    if score > best_score:
                        best_score = score
                        best_result = (bx, by, b_oidx)

        if best_result is None:
            return None

        bx, by, b_oidx = best_result
        # Update column tracking for the spanning placement.
        self._update_column_after_placement(bx, by, bin_state, orientations[b_oidx])
        return PlacementDecision(x=bx, y=by, orientation_idx=b_oidx)

    def _eval_span(
        self,
        x: float,
        y: float,
        span_w: float,
        span_d: float,
        orientations: List[Tuple[float, float, float]],
        bin_state: BinState,
        bin_cfg,
        cfg: ExperimentConfig,
        current_best: float,
    ) -> Optional[Tuple[float, float, float, int]]:
        """
        Evaluate a spanning placement region [x, x+span_w) x [y, y+span_d).

        Returns (score, x, y, oidx) if a valid placement is found that
        beats *current_best*, else None.
        """
        for oidx, (ol, ow, oh) in enumerate(orientations):
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue
            if ol > span_w + 1e-6 or ow > span_d + 1e-6:
                continue
            if x + ol > bin_cfg.length + 1e-6 or y + ow > bin_cfg.width + 1e-6:
                continue

            z = bin_state.get_height_at(x, y, ol, ow)
            if z + oh > bin_cfg.height + 1e-6:
                continue

            if z > 0.5:
                sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                if sr < MIN_SUPPORT:
                    continue

            if cfg.enable_stability and z > 0.5:
                sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                if sr < cfg.min_support_ratio:
                    continue

            # Score spanning placements using a DBLF-style metric.
            area_efficiency = (ol * ow) / (span_w * span_d) if (span_w * span_d) > 0 else 0
            height_room = max(0.0, 1.0 - (z + oh) / bin_cfg.height) if bin_cfg.height > 0 else 0
            score = W_AREA_EFFICIENCY * area_efficiency + W_HEIGHT_ROOM * height_room

            if score > current_best:
                return (score, x, y, oidx)

        return None

    # ── BLF fallback ──────────────────────────────────────────────────────

    def _blf_fallback(
        self,
        box: Box,
        orientations: List[Tuple[float, float, float]],
        bin_state: BinState,
        bin_cfg,
    ) -> Optional[PlacementDecision]:
        """
        Full Bottom-Left-Fill grid scan — identical to the baseline strategy.

        Used as a final fallback when no column or spanning placement works.
        Scans every grid position, every orientation, and returns the lowest
        feasible (z, x, y) candidate.

        Args:
            box:          The box to place.
            orientations: List of (ol, ow, oh) tuples.
            bin_state:    Current bin state (read-only).
            bin_cfg:      Bin configuration.

        Returns:
            PlacementDecision or None.
        """
        cfg = self.config
        step = self._scan_step
        best: Optional[Tuple[float, float, float, int]] = None  # (z, x, y, oidx)

        for oidx, (ol, ow, oh) in enumerate(orientations):
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue

            x = 0.0
            while x + ol <= bin_cfg.length + 1e-6:
                y = 0.0
                while y + ow <= bin_cfg.width + 1e-6:
                    z = bin_state.get_height_at(x, y, ol, ow)

                    if z + oh > bin_cfg.height + 1e-6:
                        y += step
                        continue

                    if z > 0.5:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < MIN_SUPPORT:
                            y += step
                            continue

                    if cfg.enable_stability:
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

        _, x_best, y_best, oidx_best = best
        # Update column tracking for the fallback placement too.
        if self._grid_ready:
            self._update_column_after_placement(
                x_best, y_best, bin_state,
                orientations[oidx_best],
            )
        return PlacementDecision(x=x_best, y=y_best, orientation_idx=oidx_best)
