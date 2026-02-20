# Layer Building Strategy

**File:** `strategies/layer_building.py`
**Strategy name:** `layer_building`
**Class:** `LayerBuildingStrategy`

## Overview

The Layer Building strategy organizes box placements into horizontal layers,
filling each layer as completely as possible before moving to the next one. This
approach produces dense, stable packings with flat surfaces that are ideal for
further stacking.

Rather than placing boxes individually with no global plan, the strategy
maintains a concept of the "current active layer" characterized by a base height
and a target thickness. Boxes are scored highly when they fit within the current
layer and match its target height, and only after the layer is sufficiently full
does the strategy advance to a new layer.

## Algorithm

### High-Level Flow

```
for each incoming box:
    1. Detect or update the current layer (base height + target thickness)
    2. Try to place the box WITHIN the current layer (Phase 1)
    3. If no in-layer position found:
       a. Check if the current layer is >= 80% full
       b. If yes: advance to the next layer and try again (Phase 2)
    4. If still no position: fall back to unconstrained BLF scan (Phase 3)
    5. Return best position or None
```

### Layer Detection

The strategy detects the current layer state from the heightmap rather than
relying on fragile bookkeeping:

1. If more than half the bin footprint is at height 0, we are on the first layer
   (base = 0).
2. Otherwise, round all heightmap values to the nearest integer and find the
   **mode** (most frequent value). This is the top of the last completed layer
   and hence the base of the current one.
3. The target layer height is set to the tallest orientation height of the first
   box that arrives in a new layer.

### Layer Coverage

A layer is considered "full" when >= 80% of the bin footprint has a height at or
above `layer_base + layer_target_h` (within tolerance). Only then does the
strategy advance.

### Candidate Scanning

Candidates are scanned in grid order (x left-to-right, y back-to-front) at the
configured resolution step, identical to the baseline BLF scan. Each candidate
is checked for:

- Bin bounds
- Height overflow
- Anti-float support (>= 30%)
- Optional stricter stability check

### Scoring Formula

For each valid candidate at position `(x, y, z)` with oriented dimensions
`(ol, ow, oh)`:

```
in_layer_bonus = 3.0   if z in [layer_base, layer_base + target_h + tolerance]
                 0.0    otherwise

height_fit     = 1.0 - |oh - target_h| / target_h     (clamped to [0, 1])

area_fill      = (ol * ow) / (bin_length * bin_width)

height_penalty = z / bin_height

position_tie   = 0.001 * (1 - x/L) + 0.0005 * (1 - y/W)

score = in_layer_bonus
      + 2.0 * height_fit
      + 1.0 * area_fill
      - 1.0 * height_penalty
      + position_tie
```

## Hyperparameters

| Constant               | Default | Description                                                     |
|------------------------|---------|-----------------------------------------------------------------|
| `MIN_SUPPORT`          | 0.30    | Minimum support ratio to avoid floating (matches simulator)     |
| `LAYER_FULL_THRESHOLD` | 0.80    | Fraction of footprint that must be covered before layer advance |
| `HEIGHT_TOLERANCE_FRAC`| 0.35    | Fractional tolerance for box-height-to-layer matching           |
| `HEIGHT_TOLERANCE_ABS` | 2.0 cm  | Absolute tolerance added to height matching                     |
| `IN_LAYER_BONUS`       | 3.0     | Score bonus for in-layer placements                             |
| `HEIGHT_FIT_WEIGHT`    | 2.0     | Weight for height fit component                                 |
| `AREA_FILL_WEIGHT`     | 1.0     | Weight for footprint area component                             |
| `HEIGHT_PENALTY_WEIGHT`| 1.0     | Weight for z-height penalty                                     |

### Tuning Guide

- **`LAYER_FULL_THRESHOLD`**: Lower values (e.g. 0.6) advance layers sooner,
  which may leave gaps but allows more flexible placement. Higher values (e.g.
  0.95) force very dense layers but may reject more boxes.

- **`HEIGHT_TOLERANCE_FRAC`**: Controls how strictly box heights must match the
  layer target. Lower values create more uniform layers; higher values allow
  more variation (useful for heterogeneous box sets).

- **`IN_LAYER_BONUS`**: The dominant scoring term. Increasing it makes the
  strategy more rigid about staying within the current layer. Decreasing it
  gives more weight to height-fit and area-fill.

- **`HEIGHT_FIT_WEIGHT`**: Increase to strongly prefer orientations whose
  height matches the layer target. Decrease if box heights vary widely and
  exact matching is impractical.

## Usage

### Command Line

```bash
python run_experiment.py --strategy layer_building
python run_experiment.py --strategy layer_building --all-orientations
```

### Programmatic

```python
from strategies.layer_building import LayerBuildingStrategy
from config import ExperimentConfig, Box

strategy = LayerBuildingStrategy()
strategy.on_episode_start(ExperimentConfig())

# The simulator provides bin_state; strategy returns PlacementDecision or None
decision = strategy.decide_placement(box, bin_state)
```

## Performance Characteristics

### Strengths

- **High volumetric utilization** for box sets with similar heights. The layer
  approach naturally produces dense packings when boxes tile well.
- **Stable surfaces**: completed layers are flat, making subsequent stacking
  reliable and reducing floating rejections.
- **Predictable stacking pattern**: the layer structure is easy to reason about
  and debug visually.

### Weaknesses

- **Sensitive to box height diversity**: if incoming boxes have wildly different
  heights, the layer target may be a poor match for many boxes, increasing
  rejections.
- **Layer advancement can waste space**: if the current layer is 79% full and
  the threshold is 80%, the strategy falls back to BLF instead of starting a
  new layer, potentially missing efficient placements.
- **No lookahead**: the strategy does not consider future boxes when choosing
  the layer target height.

### Expected Fill Rates

- Uniform box sets (all same size): 70-85%
- Mixed but regular boxes (2-3 sizes): 55-70%
- Highly heterogeneous boxes: 40-55%

(These are rough estimates; actual performance depends on box dimensions, bin
size, and orientation settings.)

## Limitations and Potential Improvements

1. **Adaptive layer target**: instead of using the first box's height as the
   target, analyze the upcoming box queue (if available) and choose a target
   that matches the most boxes.

2. **Partial layer filling**: allow starting a new layer even if the current one
   is below threshold, if the new box clearly does not fit the current layer
   and would fit above it.

3. **Multi-layer lookahead**: use `bin_state.copy()` to simulate placing a box
   in different positions and evaluate which creates the best surface for future
   boxes.

4. **Dynamic threshold**: lower `LAYER_FULL_THRESHOLD` as the bin gets fuller,
   since there is less room for optimization at the top.

5. **Corner-point candidates**: add corner positions of placed boxes (like
   Best Fit Decreasing does) to find tighter placements within layers.

6. **Orientation preference**: when multiple orientations fit the layer, prefer
   the one that maximizes contact with neighbors (hybrid with Best Fit).
