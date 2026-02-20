# PCT Expansion Strategy

## Overview
Implements the four candidate expansion schemes from the Packing Configuration Tree (PCT) paper by Zhao et al. (ICLR 2022). The strategy dynamically selects the best expansion scheme based on current bin fill, then scores candidates using volume efficiency and stability.

## Papers
- Zhao, H., She, Q., Zhu, C., Yang, Y., & Xu, K. (2021). "Online 3D Bin Packing with Constrained Deep Reinforcement Learning." *AAAI Conference on Artificial Intelligence*. arXiv:2012.04412.
- Zhao, H., et al. (2022). "Deliberate Planning of 3D Bin Packing on Packing Configuration Trees." *ICLR 2022*. https://github.com/alexfrom0815/Online-3D-BPP-PCT
- Zhao, H., et al. (2025). "PCT: Packing Configuration Trees for Online 3D Bin Packing." *IJRR 2025*.

## Four Expansion Schemes

| Scheme | Description | Complexity | Used when |
|--------|-------------|------------|-----------|
| **CP** | Corner Points: box corners + cross-projections | O(n²) | fill < 15% OR n_placed < 5 |
| **EP** | Event Points: heightmap transitions × cross-product | O(grid) | — |
| **EMS** | Empty Maximal Spaces: lower-left corners of free spaces | O(N·M) | 15% ≤ fill < 65% |
| **EV** | Event + Vertices: union of EP and EMS | O(grid + N·M) | fill ≥ 65% |

**Adaptive selection:** uses the cheapest scheme that still gives good coverage for the current bin density.

## Scoring
```
score = WEIGHT_VOL_EFFICIENCY × vol_efficiency
       - WEIGHT_HEIGHT × height_norm
       + WEIGHT_SUPPORT × support_ratio
       - WEIGHT_POSITION × (x + y)

vol_efficiency = box.volume / remaining_bin_capacity
```

## Performance (50 boxes, 1200×800×2700mm EUR pallet, seed=42)
| Metric | Value |
|--------|-------|
| Fill rate | **52.8%** |
| Boxes placed | 36/50 (72%) |
| Computation time | 5.0s total (0.10s/box) |

## Performance Fix Applied
**Original issue:** exponential decision time growth (36s/box at box 25). Root cause: `_recompute_ems_from_placements` rebuilt full EMS list from scratch every call, with EMS list growing unboundedly through repeated splits.

**Fix:** Added `MAX_EMS_LIST = 300` cap **inside** the split loop (keeping largest EMSs by volume), and `CP_RECENT_BOXES = 20` limit on cross-projections. Result: constant-time decisions at ~0.10s/box regardless of bin fill.

## Class: `PCTExpansionStrategy(BaseStrategy)`
- **Registry name:** `pct_expansion`
- **Type:** Single-bin (`BaseStrategy`)

## Stability
- `MIN_SUPPORT = 0.30` always enforced

## Integration
```python
python run_experiment.py --strategy pct_expansion --generate 50
```

## Comparison with PCT-DRL
This is the **heuristic baseline** of the PCT system. The full PCT uses a tree-structured DRL policy (MCTS-guided action selection). The DRL version achieves 75–80% fill. See `python/external_repos/Online-3D-BPP-PCT/` for the RL implementation.
