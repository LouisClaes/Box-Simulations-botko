# Skyline Strategy

## Overview

The Skyline strategy is a height-profile-based heuristic for online 3D bin packing. Its core idea is straightforward: **always fill the lowest horizontal band first**. By projecting the 2D heightmap onto a 1D "skyline profile" (mean height per y-band across all x columns), the strategy identifies the lowest bands in the bin and prioritises filling them, naturally producing uniform horizontal layers.

This approach is inspired by 2D strip packing skyline algorithms, extended here to work with a full 3D heightmap.

## Algorithm in Plain English

1. **Build the skyline profile.** For each y-band in the heightmap, compute the mean height across all x positions. This produces a 1D array where each entry represents the average surface level at that y coordinate.

2. **Find the valleys.** Sort all y positions by their skyline height in ascending order. The position with the lowest skyline value is the lowest band -- the most urgent area to fill.

3. **Try to fill each valley.** Starting from the lowest band, for each allowed orientation of the box:
   - Check that the box fits along the y-axis starting at this y position.
   - Scan across all valid x positions within this y-band.
   - For each `(x, y)`: compute the resting height `z`, check bounds and support, then score the candidate.

4. **Score candidates** based on three factors:
   - How low the placement is (lower = better).
   - How well the box fills the valley depth along y (wider coverage = better).
   - How uniform the footprint region will be after placement (smoother = better).

5. **Return the highest-scoring valid candidate**, or `None` if nothing fits.

## Mathematical Formulation

### Skyline profile

```
skyline[gy] = mean(heightmap[:, gy])   for gy in 0..grid_w-1
```

### Valley depth

Starting from the lowest point `gy_center`, expand backward and forward while neighbouring y-bands have approximately the same height (within `1.5 * resolution`):

```
valley_depth = (right_bound - left_bound + 1) * resolution
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
| `valley_fill` | `min(ow / valley_depth, 1.0)` | Fraction of valley covered by box width |
| `uniformity` | `-Var(footprint_region_after_placement) / bin_height^2` | Non-positive; 0 = perfectly flat |

### Why this works

By always targeting the lowest y-band:
- Boxes spread across the bin floor before stacking vertically.
- The bin surface tends toward uniform horizontal layers, which is ideal for stability.
- Vertical space is used efficiently because low bands are filled before starting higher layers.
- The valley-fill bonus rewards boxes that span the full valley depth, which closes gaps and creates flat surfaces for future placements.

## Hyperparameters

| Constant | Default | Effect |
|----------|---------|--------|
| `WEIGHT_Z` | 3.0 | How strongly to prefer lower placements. The dominant factor -- ensures boxes go to the bottom first. |
| `WEIGHT_VALLEY_FILL` | 1.5 | Bonus for covering more of the valley depth. Higher values prefer boxes that span the entire gap. |
| `WEIGHT_UNIFORMITY` | 0.5 | Bonus for post-placement surface uniformity. Higher values produce smoother surfaces. |
| `MAX_VALLEY_CANDIDATES` | 40 | Maximum number of valley y-positions to evaluate. Controls the speed/quality trade-off. |
| `X_SCAN_STEP_MULT` | 1.0 | Multiplier for the x-axis scan step. Higher values scan fewer x positions (faster but coarser). |
| `MIN_SUPPORT` | 0.30 | Minimum base support ratio (matches simulator anti-float). |

### Tuning guidance

- **For speed:** reduce `MAX_VALLEY_CANDIDATES` (e.g. to 20) or increase `X_SCAN_STEP_MULT` (e.g. to 2.0). This reduces the number of candidates evaluated but may miss better positions.
- **For quality:** increase `MAX_VALLEY_CANDIDATES` (e.g. to 80+) to evaluate more valleys, or decrease `X_SCAN_STEP_MULT` to scan x positions more finely.
- **For flatter surfaces:** increase `WEIGHT_UNIFORMITY` (e.g. to 1.0 or higher).
- **For tighter valley fills:** increase `WEIGHT_VALLEY_FILL` (e.g. to 2.5).
- **To relax the "always go low" bias:** decrease `WEIGHT_Z` (e.g. to 1.5). This gives more influence to the valley-fill and uniformity bonuses.

## Performance Characteristics

### Strengths
- Fast: evaluates only the most promising y-positions (valleys), not the entire grid. Typically evaluates 40 y-positions instead of 80.
- Produces naturally layered packings with good surface uniformity.
- Simple and interpretable -- easy to reason about why a placement was chosen.
- Works well with uniform and mixed box sizes.
- Deterministic output for the same input.

### Time complexity
- Building the skyline: `O(grid_l * grid_w)` -- a single `np.mean` over axis 0.
- Sorting valleys: `O(grid_w * log(grid_w))`.
- Evaluating candidates: `O(MAX_VALLEY_CANDIDATES * orientations * grid_l / step)`.
- For the default 120x80 bin: roughly `40 * 2 * 120 = 9,600` candidates per box (flat orientations).

### Expected fill rates
- Uniform random boxes (5-25 cm): **50-65%** fill rate.
- Performs comparably to BLF for well-distributed boxes, and better when there are many height-varying placements where valley-filling helps.
- Generally slightly below Wall-E Scoring due to the reduced search space, but significantly faster.

### Weaknesses
- The 1D skyline projection loses information: it only sees the *mean* height per y-band, not the full 2D height landscape. This means it may miss good positions where isolated pockets exist at specific x positions.
- The `MAX_VALLEY_CANDIDATES` limit can cause the strategy to miss valid positions in large bins with complex terrain.
- No lookahead: purely greedy, one box at a time.
- The valley depth measurement is 1D (along y only), so it does not capture L-shaped or irregular valley geometries.

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
import strategies.skyline.strategy as sk
sk.WEIGHT_Z = 2.0              # less aggressive height preference
sk.WEIGHT_VALLEY_FILL = 2.5    # stronger valley coverage bonus
sk.MAX_VALLEY_CANDIDATES = 80  # evaluate more valleys
```

## File Structure

```
strategies/
    skyline/
        strategy.py         -- strategy implementation
        README.md           -- this file
    base_strategy.py        -- abstract base class
    __init__.py             -- auto-registration
```

## Comparison with Other Strategies

| Aspect | Baseline (BLF) | Wall-E Scoring | Skyline |
|--------|----------------|----------------|---------|
| Search space | Full grid | Full grid | Valley y-positions only |
| Scoring | Lexicographic (z, x, y) | 5-component weighted sum | 3-component weighted sum |
| Speed | Medium | Slow | Fast |
| Surface quality | Variable | Very good | Good |
| Wall contact | Not considered | Explicitly scored | Not considered |
| Valley handling | Incidental | Via G_high sub-score | Primary mechanism |

## Limitations and Potential Improvements

1. **2D valley detection:** The current skyline is 1D (mean over x-axis). A more sophisticated approach would identify 2D rectangular valleys in the heightmap, e.g. using connected-component analysis on low regions.

2. **Adaptive candidate count:** Instead of a fixed `MAX_VALLEY_CANDIDATES`, dynamically increase the search budget when the bin is nearly full (positions are scarcer and each decision matters more).

3. **Hybrid approach:** Combine skyline's fast valley identification with Wall-E's detailed scoring. Use skyline to identify the top-k candidate regions, then apply Wall-E scoring within those regions only.

4. **X-axis skyline:** Compute a second skyline along the x-axis and merge the two to identify truly low regions in 2D.

5. **Lookahead:** Use `bin_state.copy()` to simulate placing the current box and evaluate the resulting skyline profile. Choose the placement that creates the most "fillable" profile for future boxes.

6. **Box-size awareness:** Weight the valley-fill bonus by the ratio of box volume to remaining bin volume, so the strategy becomes more conservative as the bin fills up.

7. **Orientation priority:** When a box has a dimension close to the valley depth, prefer the orientation that matches, even if it is slightly worse on other metrics.

## References

- The skyline concept in bin packing originates from 2D strip packing literature, where the "skyline" is the upper contour of placed rectangles. This strategy extends the idea to 3D by using the heightmap's mean-per-y-band projection.
- Burke, E.K. et al. "A new placement heuristic for the orthogonal stock-cutting problem" -- early use of skyline-based placement in 2D cutting/packing.
