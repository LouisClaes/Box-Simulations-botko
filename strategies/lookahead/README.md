# Lookahead Strategy

## Overview

The Lookahead Strategy is a one-step what-if simulation approach to 3D bin packing. Rather than choosing the locally "best" position for the current box (as greedy strategies do), it simulates placing the box at every candidate position, then evaluates which resulting bin state is best for accommodating future boxes.

This strategy uses `BinState.copy()` to create deep copies of the bin state for safe what-if simulation, and never modifies the original bin state.

## Algorithm

### Phase 1: Candidate Generation

The strategy performs a coarse grid scan over all allowed orientations:

```
For each orientation (ol, ow, oh):
    For x in range(0, bin_length - ol, SCAN_STEP):
        For y in range(0, bin_width - ow, SCAN_STEP):
            z = get_height_at(x, y, ol, ow)
            if z + oh > bin_height: skip
            if z > 80% of bin_height: skip (HEIGHT_CUTOFF_RATIO)
            if z > 0 and support_ratio < 0.30: skip
            Add (x, y, z, oidx, ol, ow, oh) to candidates
```

### Phase 2: Candidate Pruning

If the number of candidates exceeds `MAX_CANDIDATES` (default 50), they are sorted by a quick heuristic score and truncated:

```
quick_score = 3.0 * z/bin_height + x/bin_length + y/bin_width
```

Lower quick score means the candidate is more promising (low, near the origin). Only the top 50 survive.

### Phase 3: Lookahead Evaluation

For each surviving candidate:

1. **Clone** the bin state: `sim_state = bin_state.copy()`
2. **Apply** a virtual placement on the clone: `sim_state.apply_placement(Placement(...))`
3. **Evaluate** the resulting state using five sub-scores:

| Sub-Score | Description | Weight |
|-----------|-------------|--------|
| Uniformity | `1 - var(occupied_heights) / bin_height^2` -- low variance = level surface | 2.0 |
| Remaining | `1 - max_height / bin_height` -- lower peak = more room | 2.0 |
| Flatness | `1 - roughness / 20.0` -- smooth surface for future stacking | 1.5 |
| Fill | `get_fill_rate()` -- volumetric efficiency so far | 1.0 |
| Accessible | Fraction of cells at the most common height -- large flat areas | 1.5 |

**Final score** = `2.0 * uniformity + 2.0 * remaining + 1.5 * flatness + 1.0 * fill + 1.5 * accessible`

The candidate with the highest post-placement state score is returned.

## Hyperparameters

All hyperparameters are module-level constants in `lookahead.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SCAN_STEP` | 3.0 | Grid scan step size (cm). Increase for speed, decrease for quality. |
| `MAX_CANDIDATES` | 50 | Maximum candidates evaluated with full lookahead. |
| `QUICK_SCORE_WEIGHT_Z` | 3.0 | How aggressively the quick pre-filter favours low placements. |
| `HEIGHT_CUTOFF_RATIO` | 0.80 | Discard candidates above this fraction of bin height. |
| `MIN_SUPPORT` | 0.30 | Anti-float threshold (must match simulator). |
| `WEIGHT_UNIFORMITY` | 2.0 | State evaluation weight for height uniformity. |
| `WEIGHT_REMAINING` | 2.0 | State evaluation weight for remaining vertical capacity. |
| `WEIGHT_FLATNESS` | 1.5 | State evaluation weight for surface flatness. |
| `WEIGHT_FILL` | 1.0 | State evaluation weight for fill efficiency. |
| `WEIGHT_ACCESSIBLE` | 1.5 | State evaluation weight for accessible flat area. |

## Performance Characteristics

- **Time complexity**: O(candidates * grid_size) per box, where candidates is capped at 50. The `copy()` + `apply_placement()` are the expensive operations (~1ms each). Typical decision time: 50-200ms per box.
- **Space complexity**: O(grid_l * grid_w) per candidate evaluation (one heightmap copy at a time).
- **Quality**: Generally produces higher fill rates than greedy strategies (baseline, BLF) because it optimizes for future usability. Particularly effective when box sizes vary significantly.
- **Weakness**: Only looks one step ahead. Does not consider the actual future box sequence (which is unknown). May be slower than single-pass strategies on very large bins.

## Tuning Guide

- **Faster execution**: Increase `SCAN_STEP` to 5.0, reduce `MAX_CANDIDATES` to 30.
- **Better quality**: Decrease `SCAN_STEP` to 2.0, increase `MAX_CANDIDATES` to 100.
- **More uniform surfaces**: Increase `WEIGHT_UNIFORMITY` to 3.0.
- **Denser packing**: Increase `WEIGHT_FILL` to 2.0.
- **Lower average height**: Increase `WEIGHT_REMAINING` to 3.0.

## Usage

```python
# Via the experiment runner CLI
python run_experiment.py --strategy lookahead

# Programmatic usage
from strategies.lookahead import LookaheadStrategy

strategy = LookaheadStrategy()
strategy.on_episode_start(config)
decision = strategy.decide_placement(box, bin_state)
```

## Dependencies

- `numpy` -- used for heightmap operations, variance calculation, and height rounding.
- `config` -- `Box`, `PlacementDecision`, `ExperimentConfig`, `Orientation`, `Placement`.
- `robotsimulator.bin_state` -- `BinState` (uses `.copy()` extensively).
- `strategies.base_strategy` -- `BaseStrategy`, `register_strategy`.

## Extension Ideas

- **Multi-step lookahead**: Simulate 2-3 boxes ahead using a box size distribution model.
- **Monte Carlo rollouts**: Sample random future boxes and average the resulting state quality.
- **Adaptive weights**: Adjust evaluation weights based on bin fill level (e.g., prioritize flatness early, fill rate late).
- **Parallel evaluation**: Evaluate candidates in parallel using multiprocessing.
- **Extreme point integration**: Replace the coarse grid scan with extreme-point candidate generation for faster, sparser searches.
