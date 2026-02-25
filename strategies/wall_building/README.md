# Wall Building Strategy

## Overview

The Wall Building Strategy is inspired by how human palletizers work in practice: they build dense, wall-like structures from the back of the pallet forward. The bin is divided into vertical "strips" along the x-axis, and each strip is filled back-to-front before moving to the next.

This strategy creates stable structures with maximum wall contact, making it especially effective for rectangular boxes and scenarios where physical stability is important.

## Algorithm

### Step 1: Observe and Adapt

Each time a new box arrives, its smaller horizontal dimension is recorded. After `ADAPT_AFTER_N_BOXES` boxes (default 3), the strip width is set to the median of observed box widths, clamped to `[10, 60]` cm.

### Step 2: Detect Build Front

The "build front" is the furthest y-coordinate in the bin that has any non-zero height. It is detected by scanning the heightmap from front to back:

```python
for gy in range(grid_w - 1, -1, -1):
    if any height > 0 in column gy:
        build_front = (gy + 1) * resolution
        break
```

### Step 3: Generate and Sort Strips

The bin is divided into strips of `_strip_width` cm along the x-axis:

```
Strip 0: x in [0, strip_width)
Strip 1: x in [strip_width, 2*strip_width)
...
```

Strips are sorted by fill level (average height). The least-filled strip is tried first. Strips with average height above `STRIP_FULL_THRESHOLD` (85%) of bin height are deprioritized.

### Step 4: Scan Each Strip

For each strip, scan all positions within the strip bounds:

```
x: [strip_start, min(strip_end, bin_length - ol)]
y: [0, min(build_front + strip_width, bin_width)]
```

Each valid position is scored:

```
wall_bonus = 0
if y < 1.0: wall_bonus += 2.0  (back wall)
if x < 1.0: wall_bonus += 1.5  (left wall)
if x + ol > bin_length - 1.0: wall_bonus += 1.5  (right wall)

adjacency = count_adjacent_faces(x, y, z, ol, ow, oh)

score = wall_bonus
      + 2.0 * adjacency
      - 3.0 * z / bin_height
      - 0.5 * y / bin_width
```

### Step 5: Fallback BLF

If no valid placement is found in any strip, a full-bin Bottom-Left-Fill scan is performed as a last resort (same as the baseline strategy).

### Adjacency Detection

The adjacency count uses a heightmap-based approximation. For each of the four lateral faces of the candidate box, the strategy checks the columns immediately outside that face. If the max height in those adjacent columns overlaps vertically with the candidate box's z-range, it counts as an adjacent face. Bottom contact (z > 0) adds one more.

## Hyperparameters

All hyperparameters are module-level constants in `wall_building.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_STRIP_WIDTH` | 25.0 | Initial strip width before adaptation (cm). |
| `MIN_STRIP_WIDTH` | 10.0 | Minimum adaptive strip width (cm). |
| `MAX_STRIP_WIDTH` | 60.0 | Maximum adaptive strip width (cm). |
| `SCAN_STEP` | 1.0 | Grid scan step within strips (cm). |
| `STRIP_FULL_THRESHOLD` | 0.85 | Fraction of bin height above which a strip is "full". |
| `ADAPT_AFTER_N_BOXES` | 3 | Number of boxes to observe before adapting strip width. |
| `MIN_SUPPORT` | 0.30 | Anti-float threshold (must match simulator). |
| `WEIGHT_WALL_BACK` | 2.0 | Scoring bonus for back wall contact. |
| `WEIGHT_WALL_LEFT` | 1.5 | Scoring bonus for left wall contact. |
| `WEIGHT_WALL_RIGHT` | 1.5 | Scoring bonus for right wall contact. |
| `WEIGHT_ADJACENCY` | 2.0 | Scoring bonus per adjacent face. |
| `WEIGHT_HEIGHT_PENALTY` | 3.0 | Scoring penalty for high placement. |
| `WEIGHT_DEPTH_PENALTY` | 0.5 | Scoring penalty for distance from back wall. |
| `WALL_TOLERANCE` | 1.0 | Distance threshold for wall contact (cm). |
| `ADJACENCY_TOLERANCE` | 1.5 | Distance threshold for box adjacency (cm). |

## Internal State

The strategy maintains internal state that is reset at the start of each episode via `on_episode_start()`:

| Field | Type | Description |
|-------|------|-------------|
| `_strip_width` | float | Current adaptive strip width (starts at 25.0). |
| `_seen_widths` | List[float] | Min(length, width) of all observed boxes. |
| `_box_count` | int | Number of boxes seen this episode. |
| `_scan_step` | float | Resolved scan step (max of SCAN_STEP and bin resolution). |

## Performance Characteristics

- **Time complexity**: O(orientations * strip_area / step^2) per box. With the default 1.0 cm step and a 25x80 strip, this is ~2000 positions per orientation. Faster than full-bin scans but slower than extreme-point strategies.
- **Space complexity**: O(1) beyond the heightmap (no copies needed; reads heightmap in-place).
- **Quality**: Excels at creating dense, stable structures with high wall contact. Produces visually clean, layer-like results. Works best when boxes have similar sizes.
- **Weakness**: The strip decomposition can be suboptimal if box sizes vary wildly. The adaptive strip width mitigates this but does not eliminate it.

## Tuning Guide

- **Faster execution**: Increase `SCAN_STEP` to 2.0 or 3.0.
- **Denser packing**: Decrease `SCAN_STEP` to 0.5, increase `WEIGHT_ADJACENCY` to 3.0.
- **More wall contact**: Increase `WEIGHT_WALL_BACK` and `WEIGHT_WALL_LEFT` to 3.0.
- **Taller stacks**: Reduce `WEIGHT_HEIGHT_PENALTY` to 1.0 (less aversion to height).
- **Wider strips**: Increase `DEFAULT_STRIP_WIDTH` to 40.0 (fewer, wider columns).
- **Narrower strips**: Reduce `DEFAULT_STRIP_WIDTH` to 15.0 (more columns, tighter fit).

## Usage

```python
# Via the experiment runner CLI
python run_experiment.py --strategy wall_building

# Programmatic usage
from strategies.wall_building import WallBuildingStrategy

strategy = WallBuildingStrategy()
strategy.on_episode_start(config)
decision = strategy.decide_placement(box, bin_state)
```

## Dependencies

- `numpy` -- used for heightmap analysis, median computation, and column queries.
- `config` -- `Box`, `PlacementDecision`, `ExperimentConfig`, `Orientation`.
- `robotsimulator.bin_state` -- `BinState` (read-only access).
- `strategies.base_strategy` -- `BaseStrategy`, `register_strategy`.

## Extension Ideas

- **Two-pass strip assignment**: First assign each box to the best strip by volume, then optimize placement within the strip.
- **Dynamic strip boundaries**: Allow strip boundaries to shift based on the actual shapes of placed boxes rather than fixed-width divisions.
- **Front wall bonus**: Add a scoring bonus for front wall contact (`y + ow ~ bin_width`) once the bin is mostly full.
- **Layer detection**: Detect when a complete horizontal layer has been formed and start a new layer cleanly.
- **Combine with lookahead**: Use the wall-building score as a pre-filter, then apply lookahead evaluation on the top candidates.
- **Weight scheduling**: Gradually shift from high wall-contact preference (early) to high fill-rate preference (late) as the bin fills up.
