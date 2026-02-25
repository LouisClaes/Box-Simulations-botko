# Best Fit Decreasing Strategy

**File:** `strategies/best_fit_decreasing.py`
**Strategy name:** `best_fit_decreasing`
**Class:** `BestFitDecreasingStrategy`

## Overview

The Best Fit Decreasing (BFD) strategy is a 3D adaptation of the classic 1D Best
Fit bin-packing heuristic. For every incoming box, it evaluates **all** feasible
positions and orientations and selects the one where the box fits most
**snugly** — maximizing surface contact with bin walls and neighboring boxes
while keeping the placement as low as possible.

The "Decreasing" in the name refers to the common practice of sorting boxes by
volume (largest first) before feeding them to the strategy. The strategy itself
is order-agnostic and simply picks the tightest fit for whatever box it receives.

## Algorithm

### High-Level Flow

```
for each incoming box:
    1. Generate candidate (x, y) positions:
       a. Regular grid scan at resolution step
       b. Corner points (x, y, x_max, y_max) of all placed boxes
    2. For each candidate and each allowed orientation:
       a. Compute z = get_height_at(x, y, ol, ow)
       b. Check bounds, height overflow, support ratio
       c. Compute tightness score (surface contact fraction)
       d. Compute wasted volume below the box
       e. Compute combined score
    3. Return the candidate with the highest score, or None
```

### Candidate Position Generation

Two sources of candidate positions ensure both thorough coverage and efficient
tight fits:

1. **Grid scan**: every position at resolution step intervals across the bin
   floor (same as the baseline strategy).
2. **Corner points**: for every already-placed box, the four corners
   `(x, y)`, `(x_max, y)`, `(x, y_max)`, `(x_max, y_max)` are added as
   candidates. These are the positions most likely to produce snug fits against
   existing boxes.

All candidate positions are deduplicated using a set.

### Tightness Computation

Tightness measures what fraction of the box's total surface area (all 6 faces)
is in contact with walls or other boxes.

```
tightness = contact_area / total_surface_area
```

Contact sources:

| Face          | Contact condition                                        |
|---------------|----------------------------------------------------------|
| Bottom        | Floor (z ~ 0): full face. Elevated: support_ratio * face |
| Left wall     | x < wall_tolerance: full left face (ow * oh)             |
| Right wall    | x + ol > length - wall_tolerance: full right face        |
| Back wall     | y < wall_tolerance: full back face (ol * oh)             |
| Front wall    | y + ow > width - wall_tolerance: full front face         |
| Left neighbor | Probe heightmap at x - 1 cell; overlap fraction * face   |
| Right neighbor| Probe heightmap at x + ol; overlap fraction * face       |
| Back neighbor | Probe heightmap at y - 1 cell; overlap fraction * face   |
| Front neighbor| Probe heightmap at y + ow; overlap fraction * face       |

**Neighbor contact probing:** for each of the four sides, a thin strip (one
resolution unit thick) adjacent to the box face is examined in the heightmap. A
cell contributes contact if its height extends into the vertical range
`[z, z + oh]` of the box being placed. The fraction of such cells is multiplied
by the face area to estimate contact.

### Wasted Volume Estimation

The "waste" is the air gap between the bottom of the box and the surface below
it:

```
for each cell (i, j) in the box footprint:
    gap(i, j) = max(z - heightmap[i, j], 0)

air_volume = sum(gap) * resolution^2
wasted_fraction = air_volume / box_volume
```

This penalizes placements that create air pockets, preferring positions where
the box sits flush on a flat surface.

### Combined Score

```
score = 3.0 * tightness
      - 2.0 * (z / bin_height)
      - 0.5 * wasted_fraction
      + 0.001 * (1 - x / bin_length)      # BLF tiebreaker
      + 0.0005 * (1 - y / bin_width)       # BLF tiebreaker
```

The three main terms are:

1. **Tightness** (weight 3.0): primary driver. Maximizes surface contact.
2. **Height penalty** (weight 2.0): secondary. Prefers low placements.
3. **Waste penalty** (weight 0.5): tertiary. Minimizes air gaps below the box.

The tiny BLF tiebreaker ensures deterministic ordering among identically-scored
candidates by preferring the back-left corner.

## Hyperparameters

| Constant                | Default | Description                                              |
|-------------------------|---------|----------------------------------------------------------|
| `MIN_SUPPORT`           | 0.30    | Minimum support ratio (matches simulator anti-float)     |
| `TIGHTNESS_WEIGHT`      | 3.0     | Score weight for surface contact tightness               |
| `HEIGHT_PENALTY_WEIGHT` | 2.0     | Score penalty weight for z-height                        |
| `WASTE_PENALTY_WEIGHT`  | 0.5     | Score penalty weight for air gaps below the box          |
| `WALL_TOLERANCE`        | 0.5 cm  | Distance threshold for considering a box "at the wall"   |
| `NEIGHBOR_SAMPLE_POINTS`| 5       | Reserved for future fine-grained neighbor probing        |

### Tuning Guide

- **`TIGHTNESS_WEIGHT`**: The dominant term. Increasing it makes the strategy
  strongly prefer corner and wall placements. Decreasing it gives more influence
  to height minimization.

- **`HEIGHT_PENALTY_WEIGHT`**: Controls how strongly the strategy avoids placing
  boxes high in the bin. For tall bins where height is not a constraint, this
  can be reduced. For shallow bins, increase it.

- **`WASTE_PENALTY_WEIGHT`**: Controls aversion to air gaps. Increase for box
  sets where gaps are common (heterogeneous heights). Decrease for uniform box
  sets where gaps rarely occur.

- **`WALL_TOLERANCE`**: Should be at least `resolution / 2`. Increasing it makes
  the strategy more generous about counting wall contact, which may help when
  boxes do not perfectly align to the grid.

## Usage

### Command Line

```bash
python run_experiment.py --strategy best_fit_decreasing
python run_experiment.py --strategy best_fit_decreasing --all-orientations
```

### Programmatic

```python
from strategies.best_fit_decreasing import BestFitDecreasingStrategy
from config import ExperimentConfig, Box

strategy = BestFitDecreasingStrategy()
strategy.on_episode_start(ExperimentConfig())

# The simulator provides bin_state; strategy returns PlacementDecision or None
decision = strategy.decide_placement(box, bin_state)
```

## Performance Characteristics

### Time Complexity

The candidate set size is approximately:

```
N_grid = (bin_length / step) * (bin_width / step)
N_corners = 4 * num_placed_boxes
N_candidates = N_grid + N_corners

Total evaluations per box = N_candidates * num_orientations
```

For the default 120x80 bin at resolution 1.0 with flat orientations (2), this
is roughly `(120 * 80 + 4 * k) * 2` evaluations per box, where `k` is the
number of already-placed boxes. This is slightly more than the baseline due to
corner point candidates, but the tightness computation for each candidate is
also more expensive (heightmap probing).

For large bins or many placed boxes, this strategy is noticeably slower than
the baseline. Consider increasing the scan step or reducing
`NEIGHBOR_SAMPLE_POINTS` for faster execution.

### Strengths

- **Tight packing**: maximizing surface contact naturally produces compact
  arrangements with minimal wasted space.
- **Wall utilization**: the strategy strongly prefers wall and corner positions,
  which is optimal for real-world palletizing where walls provide structural
  support.
- **Robust to box diversity**: unlike layer-based strategies, BFD does not
  assume boxes have similar heights. It adapts to heterogeneous box sets
  gracefully.
- **Low center of gravity**: the height penalty keeps heavy boxes near the
  bottom, improving overall stability.

### Weaknesses

- **Slower than baseline**: evaluating tightness at every candidate is more
  expensive than simple BLF comparison.
- **Greedy**: like all single-box heuristics, it cannot plan ahead. A locally
  tight fit may block a globally better arrangement.
- **No layer structure**: the packing may develop an irregular surface that
  makes later placements difficult, unlike the flat surfaces produced by
  layer-building strategies.

### Expected Fill Rates

- Uniform box sets: 65-80%
- Mixed but regular boxes: 55-70%
- Highly heterogeneous boxes: 45-60%

(Rough estimates. BFD tends to outperform baseline by 3-8 percentage points
on mixed box sets and performs comparably to layer building on heterogeneous
sets.)

## Limitations and Potential Improvements

1. **Coarse neighbor probing**: the current heightmap-based side contact
   estimation is approximate. A more precise approach would iterate over the
   actual placed boxes list and compute geometric face-to-face overlap.

2. **Adaptive weights**: the scoring weights could be adjusted dynamically based
   on the bin fill rate. Early in the episode, prefer height minimization; as
   the bin fills, prioritize tightness to avoid wasted space.

3. **Lookahead**: use `bin_state.copy()` to simulate placing the current box at
   top-N candidates and evaluate which leaves the best bin state for the next
   box.

4. **Top face contact**: the current tightness computation does not account for
   boxes placed on top of the current box. A future improvement could estimate
   how well the top face of this box would support common box sizes.

5. **Scan step optimization**: for large bins, use a coarse initial scan to find
   promising regions, then refine with a fine-grained scan in those regions.

6. **Corner point clustering**: when many boxes are placed, corner points
   cluster densely. Deduplication with spatial hashing or a coarser grid could
   reduce redundant evaluations.

7. **Pre-sorting integration**: pair this strategy with a box sorter that orders
   boxes by decreasing volume for the full "Best Fit Decreasing" experience.
   The current strategy works with any box order but benefits from largest-first
   ordering.
