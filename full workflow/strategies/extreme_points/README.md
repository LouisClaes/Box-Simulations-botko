# Extreme Points Strategy

## Overview

The Extreme Points (EP) strategy is a heuristic for online 3D bin packing that dramatically reduces the search space by only evaluating candidate positions generated from the corners and edges of already-placed boxes. Instead of scanning every grid cell in a 120x80 = 9,600 cell grid, this strategy typically evaluates only 10-100 candidate positions, achieving near-equivalent placement quality in a fraction of the time.

The key insight is that optimal placements almost always occur adjacent to existing boxes or walls -- positions in the middle of empty space are wasteful. Extreme points capture exactly these "interesting" positions.

## Algorithm

### Step 1: Extreme Point Generation

For each placed box `p` in `bin_state.placed_boxes`, three extreme points are generated:

| Point   | Coordinates      | Meaning                              |
|---------|------------------|--------------------------------------|
| Right   | (p.x_max, p.y)   | Immediately to the right of the box  |
| Front   | (p.x, p.y_max)   | Immediately in front of the box      |
| Top     | (p.x, p.y)       | On top of the box (same XY corner)   |

The origin `(0, 0)` is always included as a fallback. Duplicate points are removed, and points outside the bin bounds are filtered out.

For an empty bin, only `(0, 0)` exists -- the first box is placed in the back-left corner.

### Step 2: Sorting for Early Termination

Extreme points are sorted by `(z_at_point, x, y)` ascending, so the most promising candidates (low, back-left) are evaluated first. This enables early stopping when a near-perfect placement is found.

### Step 3: Candidate Evaluation

For each extreme point `(ex, ey)` and each allowed orientation `(ol, ow, oh)`:

1. **Bounds check**: `ex + ol <= bin_length`, `ey + ow <= bin_width`
2. **Height computation**: `z = bin_state.get_height_at(ex, ey, ol, ow)`
3. **Height limit**: `z + oh <= bin_height`
4. **Support check**: If `z > 0`, verify `support_ratio >= 0.30`
5. **Stability check** (optional): If enabled, verify `support_ratio >= min_support_ratio`
6. **Scoring**: Compute a multi-criteria score

### Step 4: Return Best

Return the candidate with the highest score, or `None` if no feasible placement exists.

## Scoring Function

```
score = WEIGHT_HEIGHT * z
      + WEIGHT_CONTACT * contact_ratio
      + WEIGHT_WASTED_SPACE * wasted_space_below
      + WEIGHT_CORNER * (x + y) / (bin_length + bin_width)
```

### Components

#### 1. Height penalty (`WEIGHT_HEIGHT = -3.0`)

Strongly prefers lower placements, implementing the "Deepest" part of DBLF. A box placed at z=0 scores 0 on this component; a box at z=50 scores -150.

#### 2. Contact ratio (`WEIGHT_CONTACT = 2.0`)

Measures how many of the box's 6 faces are touching walls or other boxes. Contact detection considers:

- **Floor contact**: Bottom face at z < tolerance
- **Wall contact**: Left (x=0), back (y=0), right (x+ol=length), front (y+ow=width)
- **Box adjacency**: Lateral faces touching placed boxes (checked by comparing face coordinates within `CONTACT_TOLERANCE = 1.5 cm`)

`contact_ratio` ranges from 0 (floating in empty space) to 1 (all 6 faces touching).

#### 3. Wasted space below (`WEIGHT_WASTED_SPACE = -1.0`)

Penalizes empty gaps below the box:

```
column_volume = z * ol * ow
filled_volume = sum of overlap volumes with placed boxes below z
wasted_space = (column_volume - filled_volume) / bin_volume
```

A box on the floor has 0 wasted space. A box placed high with nothing below gets a strong penalty.

#### 4. Corner preference (`WEIGHT_CORNER = -0.5`)

Slight bias toward the back-left corner `(0, 0)`:

```
corner_distance = (x + y) / (bin_length + bin_width)
```

Ranges from 0.0 (at origin) to 1.0 (at far corner). The negative weight means positions closer to the origin score higher.

## Hyperparameters

| Parameter                 | Default | Effect                                         |
|---------------------------|---------|-------------------------------------------------|
| `MIN_SUPPORT`             | 0.30    | Anti-float threshold; must match simulator      |
| `WEIGHT_HEIGHT`           | -3.0    | Increase magnitude to pack more bottom-first    |
| `WEIGHT_CONTACT`          | 2.0     | Increase to prefer tighter wall/box contact     |
| `WEIGHT_WASTED_SPACE`     | -1.0    | Increase magnitude to avoid gaps more aggressively |
| `WEIGHT_CORNER`           | -0.5    | Increase to pack more toward back-left corner   |
| `CONTACT_TOLERANCE`       | 1.5 cm  | How close faces must be to count as "touching"  |
| `PERFECT_SCORE_THRESHOLD` | 10.0    | Score above which we stop searching early       |

### Tuning advice

- **Tall, narrow bins**: Increase `WEIGHT_HEIGHT` magnitude to -5.0 or more
- **Wide, flat bins**: Increase `WEIGHT_CORNER` magnitude to pack tighter
- **Many small boxes**: `CONTACT_TOLERANCE` can be reduced to 1.0 for more precise fits
- **Performance**: `PERFECT_SCORE_THRESHOLD` can be lowered (e.g. 5.0) to stop searching sooner at the cost of potentially missing slightly better placements

## Performance Characteristics

### Time Complexity

- **EP generation**: O(n) where n = number of placed boxes
- **Sorting**: O(k log k) where k = number of unique EPs (typically 3n+1)
- **Evaluation**: O(k * m) where m = number of orientations (2 or 6)
- **Total per box**: O(n * m), which is dramatically faster than the baseline's O(L * W * m)

### Space Complexity

- O(n) for the EP list
- No heightmap copies or large auxiliary structures

### Expected Quality

- **Fill rate**: Typically 5-15% better than random placement, comparable to or slightly below full grid scan (baseline BLF)
- **Speed**: 10-100x faster than full grid scan for bins with many placed boxes
- **Best for**: Online packing where boxes arrive one at a time and decisions must be fast

## Usage

### In experiment config

```python
from config import ExperimentConfig
config = ExperimentConfig(strategy_name="extreme_points")
```

### Register in `strategies/__init__.py`

```python
import strategies.extreme_points  # register extreme_points strategy
```

### Direct instantiation (for testing)

```python
from strategies.extreme_points import ExtremePointsStrategy
from config import ExperimentConfig, Box

strategy = ExtremePointsStrategy()
strategy.on_episode_start(ExperimentConfig())

# strategy.decide_placement(box, bin_state) -> PlacementDecision or None
```

## Limitations

1. **Suboptimal in sparse bins**: When few boxes are placed, the EP list is very small and may miss good positions that are not adjacent to existing boxes. The origin fallback mitigates this somewhat.

2. **No lookahead**: The strategy is greedy -- it picks the best position for the current box without considering future boxes. For known box sequences, a lookahead wrapper using `bin_state.copy()` could improve results.

3. **2D projection**: EPs are generated as (x, y) pairs, with z computed from the heightmap. This means the strategy does not explicitly track 3D extreme points (e.g., "above box A but beside box B"). True 3D EP generation would add more candidates but at higher cost.

4. **Contact ratio approximation**: The contact detection iterates over all placed boxes, giving O(n) per candidate. For very large n (>500 boxes), this could become a bottleneck. A spatial index (e.g., grid-based lookup) would help.

5. **No rotation optimization**: The strategy tries orientations in a fixed order and picks the best-scoring one. It does not attempt to find orientations that maximize contact or minimize wasted space specifically.

## Potential Improvements

- **3D extreme points**: Generate (x, y, z) triples instead of (x, y) pairs, enabling more precise candidate positions for stacked placements.
- **Dominated EP pruning**: Remove EPs that are strictly dominated by others (same or worse on all scoring dimensions) to reduce evaluation cost.
- **Adaptive weights**: Adjust scoring weights based on bin fill rate -- e.g., increase contact weight as the bin fills up.
- **Lookahead integration**: Use `bin_state.copy()` to simulate placing the next 2-3 boxes and evaluate total score, turning this into a beam search.
- **Spatial index for contacts**: Replace the O(n) placed-box scan with a grid-based spatial index for O(1) adjacency queries.

## References

- Crainic, T.G., Perboli, G., & Tadei, R. (2008). "Extreme Point-Based Heuristics for Three-Dimensional Bin Packing." INFORMS Journal on Computing, 20(3), 368-384.
- Dyckhoff, H. (1990). "A typology of cutting and packing problems." European Journal of Operational Research, 44(2), 145-159.
