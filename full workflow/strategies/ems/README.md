# Empty Maximal Spaces (EMS) Strategy

## Overview

The Empty Maximal Spaces (EMS) strategy is one of the most widely cited approaches in the 3D bin packing literature. It maintains a list of the largest axis-aligned rectangular empty volumes in the bin, and places each new box into the space where it achieves the tightest fit. This naturally minimizes wasted volume and produces compact, stable packings.

This implementation uses a heightmap-based rebuild approach: instead of tracking EMSs incrementally (which is complex and prone to drift), we regenerate the EMS list from the current heightmap at each decision step. This is simpler, more robust, and still fast enough for online packing.

## Algorithm

### Step 1: Candidate Position Generation

Candidate (x, y) positions are collected from two sources:

1. **Coarse grid**: Every `GRID_STEP` cm (default 5 cm) along both axes. This ensures coverage of the entire bin.
2. **Placed-box corners**: All four projected corners of every placed box. This adds precision near existing placements where tight fits are most likely.

The origin `(0, 0)` is always included.

### Step 2: EMS Construction

For each candidate position `(cx, cy)`:

1. **Probe height**: `z = bin_state.get_height_at(cx, cy, 1, 1)` -- the surface height at this point.
2. **Available height**: `avail_h = bin_height - z`. If less than `MIN_EMS_DIMENSION`, skip.
3. **Expand rightward (x-axis)**: Starting from `cx`, move right cell by cell. Include each cell if its heightmap value is within `HEIGHT_TOLERANCE` of `z`. Stop when a cell exceeds the tolerance or the bin edge is reached. This gives `max_l`.
4. **Expand forward (y-axis)**: Starting from `cy`, move forward row by row. For each row, check the ENTIRE x-extent `[cx, cx+max_l)`. If all cells in that row are within tolerance of `z`, include the row. Stop otherwise. This gives `max_w`.
5. **Create EMS**: `EMS(cx, cy, z, max_l, max_w, avail_h)`.

The result is an axis-aligned rectangular space where the surface is approximately flat at height `z`, with `avail_h` of clearance above.

### Step 3: Candidate Evaluation

EMSs are sorted by `(z, x, y)` ascending (DBLF ordering). For each EMS and each allowed orientation:

1. **Fit check**: `ol <= ems.length`, `ow <= ems.width`, `oh <= ems.height`
2. **Position**: Place at the EMS origin `(ems.x, ems.y)`
3. **Actual height**: `z = bin_state.get_height_at(ems.x, ems.y, ol, ow)` (recalculated for the actual footprint)
4. **Height limit**: `z + oh <= bin_height`
5. **Support check**: If `z > 0`, verify `support_ratio >= 0.30`
6. **Stability check** (optional): If enabled, verify `support_ratio >= min_support_ratio`
7. **Scoring**: Compute DBLF + fit tightness score

### Step 4: Return Best

Return the candidate with the highest score, or `None` if no feasible placement exists.

## Scoring Function

```
fit_score = (ol * ow * oh) / (ems_length * ems_width * ems_height)

score = WEIGHT_Z * z
      + WEIGHT_X * x
      + WEIGHT_Y * y
      + WEIGHT_FIT * fit_score
```

### Components

#### 1. Height penalty (`WEIGHT_Z = -5.0`)

Very strong preference for low placements. This is the dominant term: a box at z=0 has no penalty, while a box at z=30 gets -150 points.

#### 2. Left preference (`WEIGHT_X = -1.0`)

Moderate preference for positions closer to the left wall (x=0). Helps build a compact wall from left to right.

#### 3. Back preference (`WEIGHT_Y = -0.5`)

Slight preference for positions closer to the back wall (y=0). Combined with the x-preference, this creates a back-left-first packing pattern.

#### 4. Fit tightness (`WEIGHT_FIT = 3.0`)

Rewards placements where the box fills a large fraction of the EMS volume:

```
fit_score = box_volume / ems_volume
```

- `fit_score = 1.0`: The box perfectly fills the EMS (ideal).
- `fit_score = 0.1`: The box uses only 10% of the available space (wasteful).

This component ensures that small boxes go into small spaces and large boxes go into large spaces, minimizing fragmentation.

## Hyperparameters

| Parameter           | Default | Effect                                           |
|---------------------|---------|--------------------------------------------------|
| `MIN_SUPPORT`       | 0.30    | Anti-float threshold; must match simulator       |
| `GRID_STEP`         | 5.0 cm  | Coarse grid resolution; smaller = more precise but slower |
| `HEIGHT_TOLERANCE`   | 1.0 cm  | How flat a surface must be to form one EMS       |
| `WEIGHT_Z`          | -5.0    | Increase magnitude to pack more bottom-first     |
| `WEIGHT_X`          | -1.0    | Increase magnitude to pack more left-first       |
| `WEIGHT_Y`          | -0.5    | Increase magnitude to pack more back-first       |
| `WEIGHT_FIT`        | 3.0     | Increase to prefer tighter fits over position    |
| `MIN_EMS_DIMENSION` | 1.0 cm  | Minimum EMS size on any axis; filters tiny spaces |

### Tuning Advice

- **Higher fill rates**: Decrease `GRID_STEP` to 2.0 or 1.0 for more candidate positions. This increases computation time but finds tighter fits.
- **Faster execution**: Increase `GRID_STEP` to 10.0. Fewer candidates, but placed-box corners still provide precision where it matters most.
- **Rough heightmaps**: Increase `HEIGHT_TOLERANCE` to 2.0 or 3.0 to allow EMSs over slightly uneven surfaces. Useful when boxes have varied heights.
- **Flat surfaces only**: Decrease `HEIGHT_TOLERANCE` to 0.5 to only create EMSs on perfectly flat surfaces.
- **Position vs fit tradeoff**: If `WEIGHT_Z` dominates too much (always picking the lowest point even with terrible fit), increase `WEIGHT_FIT` to 5.0 or reduce `WEIGHT_Z` to -3.0.

## Performance Characteristics

### Time Complexity

- **Candidate generation**: O(G + n) where G = (bin_length/GRID_STEP) * (bin_width/GRID_STEP) grid points and n = number of placed boxes (for corners)
- **EMS expansion**: O(G * grid_l) per candidate for x-expansion, O(G * grid_l * grid_w) worst case for y-expansion, but typically much less
- **Evaluation**: O(E * m) where E = number of EMSs and m = number of orientations
- **Total per box**: O(G * grid_area + E * m), typically a few hundred to a few thousand operations

With default `GRID_STEP = 5.0` on a 120x80 bin: G = 24 * 16 = 384 grid candidates + box corners. Typical E = 100-400 EMSs.

### Space Complexity

- O(E) for the EMS list
- O(1) additional -- only reads the existing heightmap, no copies

### Expected Quality

- **Fill rate**: Typically 5-20% better than naive bottom-left-fill due to fit-tightness optimization
- **Speed**: Faster than full grid scan, slower than pure extreme points (more candidates to evaluate)
- **Best for**: Scenarios with varied box sizes where fit tightness matters -- the EMS approach naturally pairs small boxes with small gaps

## Usage

### In experiment config

```python
from config import ExperimentConfig
config = ExperimentConfig(strategy_name="ems")
```

### Register in `strategies/__init__.py`

```python
import strategies.ems  # register ems strategy
```

### Direct instantiation (for testing)

```python
from strategies.ems import EMSStrategy
from config import ExperimentConfig, Box

strategy = EMSStrategy()
strategy.on_episode_start(ExperimentConfig())

# strategy.decide_placement(box, bin_state) -> PlacementDecision or None
```

## The EMS Data Structure

Each EMS is represented as an object with six attributes:

```
EMS(x, y, z, length, width, height)
```

- `(x, y, z)`: The origin (back-left-bottom corner) of the empty space
- `(length, width, height)`: The extent of the empty space along each axis

The `can_fit(ol, ow, oh)` method checks if a box with given oriented dimensions can fit inside the EMS. The `volume` property returns `length * width * height`.

### Example

In a 120x80x150 bin with one 40x30x20 box placed at (0, 0, 0):

```
Placed box: (0, 0, 0) -> (40, 30, 20)

Approximate EMSs generated:
  EMS(40, 0, 0, 80, 80, 150)    -- space to the right of the box
  EMS(0, 30, 0, 120, 50, 150)   -- space in front of the box
  EMS(0, 0, 20, 40, 30, 130)    -- space above the box
  EMS(40, 30, 0, 80, 50, 150)   -- diagonal space (right-front)
  ... and more from grid sampling
```

## Limitations

1. **Approximate EMSs**: The heightmap-based expansion produces rectangular spaces that may not be truly "maximal" in the formal sense. A cell-by-cell expansion can miss L-shaped or irregular empty regions.

2. **Height tolerance sensitivity**: If `HEIGHT_TOLERANCE` is too large, EMSs may span uneven surfaces, leading to poor support ratios. If too small, useful spaces may be fragmented into tiny unusable EMSs.

3. **Coarse grid gaps**: With `GRID_STEP = 5.0`, positions between grid points and box corners are not evaluated. This can miss optimal placements by up to 5 cm on each axis.

4. **No cross-EMS placement**: A box can only be placed at the origin of a single EMS. It cannot span multiple EMSs even if the combined space would fit it. This is a fundamental limitation of the per-EMS evaluation approach.

5. **Rebuild overhead**: Regenerating the full EMS list at each step is O(G * grid_area). For very large bins or very small grid steps, this could become expensive. An incremental update scheme would be more efficient but significantly more complex.

6. **No lookahead**: Like most greedy heuristics, this strategy does not consider future boxes when making placement decisions.

## Potential Improvements

- **Incremental EMS updates**: When a box is placed, split overlapping EMSs into sub-spaces (up to 6 per overlap) rather than rebuilding from scratch. This is the classical approach from the literature but requires careful bookkeeping.
- **Maximal rectangle detection**: Use the histogram-based O(n) algorithm to find truly maximal rectangles at each height level, rather than greedy expansion from seed points.
- **Multi-EMS spanning**: Allow boxes to be placed at positions that span multiple EMSs, using the heightmap to verify feasibility rather than requiring containment within a single EMS.
- **Adaptive grid step**: Start with a coarse grid and refine around the best-scoring candidates, similar to a branch-and-bound approach.
- **Lookahead**: Use `bin_state.copy()` to simulate future placements and choose the current position that maximizes long-term fill rate.
- **Weight annealing**: Gradually shift scoring weights as the bin fills -- e.g., increase `WEIGHT_FIT` and decrease `WEIGHT_Z` as fewer spaces remain.
- **Dominated EMS pruning**: Remove EMSs that are fully contained within larger ones at the same height level, reducing the evaluation set.

## References

- Gonçalves, J.F. & Resende, M.G.C. (2013). "A biased random key genetic algorithm for 2D and 3D bin packing problems." International Journal of Production Economics, 145(2), 500-510.
- Crainic, T.G., Perboli, G., & Tadei, R. (2008). "Extreme Point-Based Heuristics for Three-Dimensional Bin Packing." INFORMS Journal on Computing, 20(3), 368-384.
- Lai, K.K. & Chan, J.W.M. (1997). "Developing a simulated annealing algorithm for the cutting stock problem." Computers & Industrial Engineering, 32(1), 115-127.
