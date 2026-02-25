# PCT-MACS Heuristic Strategy

## Overview
Implements the MACS (Maximal Available Container Spaces) heuristic from the PCT repository (Zhao et al. ICLR 2022), originally from TAP-Net (Liu et al. ACM TOG 2020). Evaluates each candidate placement by simulating its effect on the bin's remaining packing capacity.

## Papers
- Zhao, H., et al. (2022). "Deliberate Planning of 3D Bin Packing on Packing Configuration Trees." *ICLR 2022*. https://github.com/alexfrom0815/Online-3D-BPP-PCT
- Zhao, H., et al. (2025). "PCT: Packing Configuration Trees for Online 3D Bin Packing." *IJRR 2025*.
- Liu, B., Wang, H., Niu, B., Hao, J., & Zheng, C. (2020). "TAP-Net: Transport-and-Pack using Reinforcement Learning." *ACM Transactions on Graphics*, 39(6). (MACS heuristic origin)

## Algorithm

### MACS Score
For each candidate position (x, y, orientation):
1. Simulate the placement: compute what the heightmap would look like after placing the box.
2. Compute **remaining_space** = sum of all remaining vertical capacity over the grid: `∑(bin_height - heightmap_after[i,j])`.
3. Compute **roughness_penalty** = variance of the new heightmap (jagged surfaces hurt future placements).
4. Compute **height_penalty** = (z + oh) / bin_height (penalise high placements).

```
MACS_score = remaining_space
           - 0.1 × roughness_penalty
           - 100.0 × height_penalty
```

The score directly answers: *"after placing this box here, how much useful packing space is left?"* Higher = better.

### Original MACS (TAP-Net)
The original algorithm scans every height slice and computes the largest empty rectangle using the Histogram algorithm. Our implementation uses the heightmap directly (`∑ remaining_height`) which preserves the core intuition while being compatible with the heightmap-based simulator.

## Performance (50 boxes, 1200×800×2700mm EUR pallet, seed=42)
| Metric | Value |
|--------|-------|
| Fill rate | **58.6%** |
| Boxes placed | 40/50 (80%) |
| Computation time | 65.2s total (1.30s/box) |

**Computation note:** Each decision requires simulating the heightmap update for all candidates (~9,600 grid cells × N candidates). Early boxes (empty bin) are slower (~5s/box); late boxes (bin constrained, fewer valid candidates) are faster (~0.3s/box). Average: 1.30s/box well within the 5-7s/box budget.

## Class: `PCTMACSHeuristicStrategy(BaseStrategy)`
- **Registry name:** `pct_macs_heuristic`
- **Type:** Single-bin (`BaseStrategy`)

## Stability
- `MIN_SUPPORT = 0.30` always enforced
- Optional `enable_stability` with strict `min_support_ratio`

## Integration
```python
python run_experiment.py --strategy pct_macs_heuristic --generate 50
```
Note: expect ~200-400s for 50 boxes due to O(grid × candidates) per-decision cost.

## Why it's slow
`_macs_score` simulates the heightmap after placement for every candidate:
- Grid: 120 × 80 = 9,600 cells
- Candidates: ~500-1,000 per box (grid scan + corners)
- Per-box cost: 9,600 × 800 = 7.68M operations
- At 3s/box this is ~2.5M simple numpy ops/second = expected for pure Python

Potential speedup: vectorise candidate evaluation using numpy broadcasting (batch all candidates at once rather than one-at-a-time).
