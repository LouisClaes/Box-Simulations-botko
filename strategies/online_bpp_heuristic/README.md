# Online BPP Heuristic Strategy

## Overview
Heuristic adaptation of the deep reinforcement learning approach from Zhao et al. (AAAI 2021) for online 3D bin packing. Implements the key placement heuristics used as the baseline in the DRL paper: heightmap-based placement with surface contact scoring and gravity-balanced stacking.

## Paper
Zhao, H., She, Q., Zhu, C., Yang, Y., & Xu, K. (2021). "Online 3D Bin Packing with Constrained Deep Reinforcement Learning." *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(6), 741–749. arXiv:2012.04412. https://github.com/alexfrom0815/Online-3D-BPP-DRL

## Algorithm
1. Generate candidate positions: full grid scan + corners of placed boxes.
2. For each candidate × orientation:
   - Compute resting height `z = get_height_at(x, y, ol, ow)`
   - Validate: bounds, height limit, support ratio ≥ 0.30
   - Score: `CONTACT_WEIGHT × contact_ratio + SUPPORT_WEIGHT × support_ratio - HEIGHT_WEIGHT × height_norm`
3. Optionally compute gravity balance: check that placement doesn't create a toppling moment (simplified via CoG check).
4. Select highest-scoring valid placement.

### Key constants
| Constant | Value | Role |
|----------|-------|------|
| `CONTACT_WEIGHT` | 3.0 | Surface contact (compactness) |
| `SUPPORT_WEIGHT` | 2.0 | Stable support |
| `HEIGHT_WEIGHT` | 1.0 | Prefer low placements |
| `MIN_SUPPORT` | 0.30 | Anti-float |

## Performance (50 boxes, 1200×800×2700mm EUR pallet, seed=42)
| Metric | Value |
|--------|-------|
| Fill rate | **64.1%** |
| Boxes placed | 42/50 (84%) |
| Computation time | 23.7s total (0.47s/box) |

**Note:** Good placement rate (84%) and competitive fill rate (64.1%), close to `walle_scoring` and `surface_contact`. The full grid scan ensures no valid position is missed.

## Class: `OnlineBPPHeuristicStrategy(BaseStrategy)`
- **Registry name:** `online_bpp_heuristic`
- **Type:** Single-bin (`BaseStrategy`)

## Stability
- `MIN_SUPPORT = 0.30` always enforced
- Optional `enable_stability` with stricter `min_support_ratio`

## Integration
```python
python run_experiment.py --strategy online_bpp_heuristic --generate 50
```

## Comparison with DRL version
This is the **heuristic baseline**. The full Online-3D-BPP-DRL uses a deep Q-network (DQN) policy trained with constrained RL. The DRL version achieves higher fill rates especially for diverse box distributions. See `python/external_repos/Online-3D-BPP-DRL/` for the RL implementation.
