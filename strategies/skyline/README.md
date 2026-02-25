# Skyline Strategy

## Overview

The Skyline strategy is a height-profile-based heuristic for online 3D bin packing. Its core idea is straightforward: **always fill the lowest valley first**. By projecting the 2D heightmap onto a 1D "skyline profile" (minimum height per x-column), the strategy identifies the deepest gaps in the bin and prioritises filling them, naturally producing uniform horizontal layers.

This approach is inspired by 2D strip packing skyline algorithms, extended here to work with a full 3D heightmap.

## Algorithm in Plain English

1. **Build the skyline profile.** For each x-column in the heightmap, compute the minimum height across all y positions. This produces a 1D array where each entry represents the "deepest reachable point" at that x coordinate.

2. **Find the valleys.** Sort all x positions by their skyline height in ascending order. The position with the lowest skyline value is the deepest valley -- the most urgent gap to fill.

3. **Try to fill each valley.** Starting from the deepest valley, for each allowed orientation of the box:
   - Check that the box fits horizontally starting at this x position.
   - Scan across all valid y positions within this x-column.
   - For each `(x, y)`: compute the resting height `z`, check bounds and support, then score the candidate.

4. **Score candidates** based on three factors:
   - How low the placement is (lower = better).
   - How well the box fills the valley width (wider coverage = better).
   - How uniform the footprint region will be after placement (smoother = better).

5. **Return the highest-scoring valid candidate**, or `None` if nothing fits.

## Mathematical Formulation

### Skyline profile

```
skyline[gx] = min(heightmap[gx, :])   for gx in 0..grid_l-1
```

### Valley width

Starting from the deepest point `gx_center`, expand left and right while neighbouring columns have approximately the same height (within `1.5 * resolution`):

```
valley_width = (right_bound - left_bound + 1) * resolution
```

### Scoring function

For a candidate at position `(x, y)` with orientation `(ol, ow, oh)` and resting height `z`:

```
score = -W_z * z  +  W_fill * valley_fill  +  W_unif * uniformity
```

where:

| Component | Formula | Meaning |
|-----------|---------|---------|
| `z` (raw) | height from `get_height_at(x, y, ol, ow)` | Lower is better |
| `valley_fill` | `min(ol / valley_width, 1.0)` | Fraction of valley covered by box length |
| `uniformity` | `-Var(footprint_region_after_placement) / bin_height^2` | Non-positive; 0 = perfectly flat |

### Why this works

By always targeting the lowest valley:
- Boxes are pushed into gaps rather than stacked on peaks.
- The bin surface tends toward uniform layers, which is ideal for stability.
- Vertical space is used efficiently because voids are filled before new layers start.
- The valley-fill bonus rewards boxes that span the full valley width, which closes gaps and creates flat surfaces for future placements.

## Hyperparameters

| Constant | Default | Effect |
|----------|---------|--------|
| `WEIGHT_Z` | 3.0 | How strongly to prefer lower placements. The dominant factor -- ensures boxes go to the bottom first. |
| `WEIGHT_VALLEY_FILL` | 1.5 | Bonus for covering more of the valley width. Higher values prefer boxes that span the entire gap. |
| `WEIGHT_UNIFORMITY` | 0.5 | Bonus for post-placement surface uniformity. Higher values produce smoother surfaces. |
| `MAX_VALLEY_CANDIDATES` | 40 | Maximum number of valley x-positions to evaluate. Controls the speed/quality trade-off. |
| `Y_SCAN_STEP_MULT` | 1.0 | Multiplier for the y-axis scan step. Higher values scan fewer y positions (faster but coarser). |
| `MIN_SUPPORT` | 0.30 | Minimum base support ratio (matches simulator anti-float). |

### Tuning guidance

- **For speed:** reduce `MAX_VALLEY_CANDIDATES` (e.g. to 20) or increase `Y_SCAN_STEP_MULT` (e.g. to 2.0). This reduces the number of candidates evaluated but may miss better positions.
- **For quality:** increase `MAX_VALLEY_CANDIDATES` (e.g. to 80+) to evaluate more valleys, or decrease `Y_SCAN_STEP_MULT` to scan y positions more finely.
- **For flatter surfaces:** increase `WEIGHT_UNIFORMITY` (e.g. to 1.0 or higher).
- **For tighter valley fills:** increase `WEIGHT_VALLEY_FILL` (e.g. to 2.5).
- **To relax the "always go low" bias:** decrease `WEIGHT_Z` (e.g. to 1.5). This gives more influence to the valley-fill and uniformity bonuses.

## Performance Characteristics

### Strengths
- Fast: evaluates only the most promising x-positions (valleys), not the entire grid. Typically evaluates 40 x-positions instead of 120.
- Produces naturally layered packings with good surface uniformity.
- Simple and interpretable -- easy to reason about why a placement was chosen.
- Works well with uniform and mixed box sizes.
- Deterministic output for the same input.

### Time complexity
- Building the skyline: `O(grid_l * grid_w)` -- a single `np.min` over axis 1.
- Sorting valleys: `O(grid_l * log(grid_l))`.
- Evaluating candidates: `O(MAX_VALLEY_CANDIDATES * orientations * grid_w / step)`.
- For the default 120x80 bin: roughly `40 * 2 * 80 = 6,400` candidates per box (flat orientations), which is about 3x fewer than the Wall-E strategy's exhaustive scan.

### Expected fill rates
- Uniform random boxes (5-25 cm): **50-65%** fill rate.
- Performs comparably to BLF for well-distributed boxes, and better when there are many height-varying placements where valley-filling helps.
- Generally slightly below Wall-E Scoring due to the reduced search space, but significantly faster.

### Weaknesses
- The 1D skyline projection loses information: it only sees the *minimum* height per x-column, not the full 2D height landscape. This means it may miss good positions where the minimum-height y-cell is in an inconvenient location.
- The `MAX_VALLEY_CANDIDATES` limit can cause the strategy to miss valid positions in large bins with complex terrain.
- No lookahead: purely greedy, one box at a time.
- The valley width measurement is 1D (along x only), so it does not capture L-shaped or irregular valley geometries.

## Usage

### Command line
```bash
python run_experiment.py --strategy skyline --generate 40 --verbose --render
python run_experiment.py --strategy skyline --generate 50 --all-orientations --render -v
python run_experiment.py --strategy skyline --dataset dataset/test.json --stability --render
```

### Python API
```python
from config import ExperimentConfig, BinConfig
from run_experiment import run_experiment
from dataset.generator import generate_uniform

boxes = generate_uniform(40, min_dim=5.0, max_dim=25.0, seed=42)

config = ExperimentConfig(
    bin=BinConfig(length=120, width=80, height=150),
    strategy_name="skyline",
    allow_all_orientations=False,
    verbose=True,
)

result = run_experiment(config, boxes)
print(f"Fill rate: {result['metrics']['fill_rate']:.1%}")
```

### Modifying weights at runtime
```python
import strategies.skyline as sk
sk.WEIGHT_Z = 2.0              # less aggressive height preference
sk.WEIGHT_VALLEY_FILL = 2.5    # stronger valley coverage bonus
sk.MAX_VALLEY_CANDIDATES = 80  # evaluate more valleys
```

## File Structure

```
strategies/
    skyline.py              -- strategy implementation
    skyline_README.md       -- this file
    base_strategy.py        -- abstract base class
    __init__.py             -- auto-registration
```

## Comparison with Other Strategies

| Aspect | Baseline (BLF) | Wall-E Scoring | Skyline |
|--------|----------------|----------------|---------|
| Search space | Full grid | Full grid | Valley positions only |
| Scoring | Lexicographic (z, x, y) | 5-component weighted sum | 3-component weighted sum |
| Speed | Medium | Slow | Fast |
| Surface quality | Variable | Very good | Good |
| Wall contact | Not considered | Explicitly scored | Not considered |
| Valley handling | Incidental | Via G_high sub-score | Primary mechanism |

## Limitations and Potential Improvements

1. **2D valley detection:** The current skyline is 1D (min over y-axis). A more sophisticated approach would identify 2D rectangular valleys in the heightmap, e.g. using connected-component analysis on low regions.

2. **Adaptive candidate count:** Instead of a fixed `MAX_VALLEY_CANDIDATES`, dynamically increase the search budget when the bin is nearly full (positions are scarcer and each decision matters more).

3. **Hybrid approach:** Combine skyline's fast valley identification with Wall-E's detailed scoring. Use skyline to identify the top-k candidate regions, then apply Wall-E scoring within those regions only.

4. **Y-axis skyline:** Compute a second skyline along the y-axis and merge the two to identify truly low regions in 2D.

5. **Lookahead:** Use `bin_state.copy()` to simulate placing the current box and evaluate the resulting skyline profile. Choose the placement that creates the most "fillable" profile for future boxes.

6. **Box-size awareness:** Weight the valley-fill bonus by the ratio of box volume to remaining bin volume, so the strategy becomes more conservative as the bin fills up.

7. **Orientation priority:** When a box has a dimension close to the valley width, prefer the orientation that matches, even if it is slightly worse on other metrics.

## References

- The skyline concept in bin packing originates from 2D strip packing literature, where the "skyline" is the upper contour of placed rectangles. This strategy extends the idea to 3D by using the heightmap's minimum-per-column projection.
- Burke, E.K. et al. "A new placement heuristic for the orthogonal stock-cutting problem" -- early use of skyline-based placement in 2D cutting/packing.
