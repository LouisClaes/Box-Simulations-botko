# Column Fill Strategy

**File:** `strategies/column_fill.py`
**Registry name:** `column_fill`
**Type:** NOVEL heuristic strategy -- not based on any published paper.

---

## Overview

The Column Fill Strategy divides the pallet floor into a grid of virtual columns and fills each column vertically (bottom-to-top) before moving to the next. This creates dense vertical stacks with minimal wasted horizontal space between columns.

The key insight is that by constraining boxes to predefined column regions, we avoid the fragmentation that often occurs with greedy position-scanning approaches (like BLF). Each column acts as a local sub-problem: "fill this rectangular area as high as possible."

---

## Algorithm

### Phase 1: Calibration (first 5 boxes)

During the first `N_CALIBRATION_BOXES` (default: 5) boxes, the strategy:
- Records each box's dimensions.
- Uses a standard BLF fallback for placement.
- After collecting enough data, computes the **median box length and width** and uses these to determine column dimensions.

### Phase 2: Column Grid Construction

After calibration:
1. `col_width = median(box.length)`, clamped to `[MIN_COL_DIM, MAX_COL_DIM]` (10--50 cm).
2. `col_depth = median(box.width)`, clamped similarly.
3. Compute `n_cols_x = round(bin_length / col_width)` and `n_cols_y = round(bin_width / col_depth)`.
4. Adjust column sizes to exactly tile the bin with no leftover gaps.
5. Create `n_cols_x * n_cols_y` Column objects, each tracking position, size, current height, and box count.

For the default 120x80 cm bin with ~20 cm boxes, this yields a 6x4 = 24 column grid.

### Phase 3: Column-Based Placement

For each incoming box:

**Step A -- Single-column fit:**
For each column and each orientation of the box:
- Check if `ol <= column.width` AND `ow <= column.depth`.
- Compute `z = bin_state.get_height_at(column.x, column.y, ol, ow)`.
- Verify height bounds and support constraints.
- Score the (column, orientation) pair using:

```
score = 2.0 * area_efficiency + 2.0 * height_room + 0.2 * adjacency_bonus
```

where:
- `area_efficiency = (ol * ow) / (column.width * column.depth)` -- tighter fit is better.
- `height_room = 1.0 - (z + oh) / bin_height` -- shorter columns are preferred.
- `adjacency_bonus = count of filled 4-connected neighbour columns` -- compact packing.

Return the highest-scoring candidate.

**Step B -- Spanning placement (2 adjacent columns):**
If no single column can fit the box, try combining two horizontally or vertically adjacent columns. The combined area (span_w x span_d) is tested for each orientation.

**Step C -- BLF fallback:**
If spanning also fails, perform a full Bottom-Left-Fill grid scan identical to the baseline strategy. This guarantees no valid position is missed.

### Internal State Tracking

Each Column dataclass maintains:
- `current_height` -- advisory tracked height (actual z comes from heightmap).
- `n_boxes` -- number of boxes placed in this column.

These are used for scoring (adjacency, height room) but never for computing z. The real `bin_state.get_height_at()` is always used for the actual resting height.

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_COL_SIZE` | 20.0 cm | Column size when no calibration data available |
| `N_CALIBRATION_BOXES` | 5 | Boxes to observe before building the column grid |
| `MIN_COL_DIM` | 10.0 cm | Minimum column dimension (prevents tiny columns) |
| `MAX_COL_DIM` | 50.0 cm | Maximum column dimension (prevents too-large columns) |
| `W_AREA_EFFICIENCY` | 2.0 | Weight for box/column area ratio |
| `W_HEIGHT_ROOM` | 2.0 | Weight for remaining vertical room |
| `W_ADJACENCY` | 0.2 | Per-neighbour bonus weight |
| `MIN_SUPPORT` | 0.30 | Anti-float support threshold |
| `BLF_STEP` | 1.0 cm | Grid scan step for BLF fallback |

### Tuning Guidelines

- **Larger `N_CALIBRATION_BOXES`** gives better column sizing but delays the adaptive phase.
- **Higher `W_AREA_EFFICIENCY`** discourages placing small boxes in large columns (reduces internal fragmentation but may leave tall columns unfilled).
- **Higher `W_HEIGHT_ROOM`** produces more uniform stack heights across columns.
- **Higher `W_ADJACENCY`** concentrates boxes near already-filled areas (good for stability, may leave corners empty longer).

---

## Expected Performance

| Metric | Expected Range | Rationale |
|--------|---------------|-----------|
| Fill rate | 55-70% | Dense vertical stacking within columns |
| Max height | Moderate | Balanced by height-room scoring |
| Speed | Fast | Only evaluates ~24 columns instead of ~9600 grid cells |
| Stability | Good | Column structure naturally creates wide support bases |

**Best for:** Box distributions where sizes cluster around 1-3 dominant sizes (e.g., e-commerce packaging with standard box types).

**Worst for:** Highly heterogeneous box sizes where few boxes fit neatly into the column grid, causing frequent BLF fallbacks.

---

## Edge Cases Handled

1. **Empty bin** -- Calibration phase uses BLF fallback; first box goes to (0,0).
2. **Box too large for any column** -- Spanning placement tries adjacent pairs; BLF fallback catches everything else.
3. **Box too large for the bin** -- Quick reject at the top of `decide_placement()`.
4. **All columns full (at bin height)** -- BLF fallback scans for any remaining valid position.
5. **No valid position exists** -- Returns `None`.

---

## Dependencies

- `numpy` (for heightmap operations via bin_state)
- `config.py` (Box, PlacementDecision, ExperimentConfig, Orientation)
- `robotsimulator.bin_state` (BinState)
- `strategies.base_strategy` (BaseStrategy, register_strategy)

No dependency on other strategy files. Fully standalone.

---

## Class Structure

```
ColumnFillStrategy(BaseStrategy)
    name = "column_fill"

    on_episode_start(config)     -- reset columns, calibration buffer
    decide_placement(box, state) -- main entry point

    _build_column_grid(bin_cfg)  -- construct column grid from calibration data
    _try_column_placement(...)   -- attempt single-column placement
    _score_column(...)           -- compute column selection score
    _count_filled_neighbours(..) -- adjacency check for scoring
    _update_column_after_placement(...) -- update internal tracking
    _try_spanning_placement(...) -- attempt 2-column spanning
    _eval_span(...)              -- evaluate a specific spanning region
    _blf_fallback(...)           -- full grid scan fallback
```

---

## Notes for Continuation

- The column grid is built **once** per episode (after calibration). A potential extension is to **rebuild** the grid mid-episode if the box distribution shifts.
- The spanning logic currently only considers pairs of adjacent columns. An extension could try 2x2 blocks (4 columns) for very large boxes.
- Column sizing uses the median; an alternative is to use the most common dimension (mode) or to create non-uniform columns with varying sizes.
- The strategy would benefit from a "column compaction" pass that shifts boxes within a column to minimize gaps (offline post-processing).
