# GOPT Heuristic Strategy

## Overview
Heuristic implementation of the Geometry-aware Object Placement Transformer (GOPT) packing approach from Xiong et al. (RA-L 2024). Uses heightmap gradient analysis to detect stable placement corners, then scores with DBLF (Deepest Bottom Left Fill) logic.

## Paper
Xiong, R., Huang, Y., Feng, X., Gong, H. (2024). "GOPT: Generalizable Online 3D Bin Packing via Transformer-based Deep Reinforcement Learning." *IEEE Robotics and Automation Letters*, Vol. 9, No. 10, pp. 8282–8289. https://doi.org/10.1109/LRA.2024.3426028

GitHub: https://github.com/zxpeter/GOPT

## Algorithm
1. Compute the gradient of the current heightmap (Sobel-like finite differences).
2. Identify **corner candidates**: positions where both `∂h/∂x > 0` and `∂h/∂y > 0` — boxes piled against existing stacks or walls.
3. For each corner candidate × orientation:
   - Compute resting height `z = get_height_at(x, y, ol, ow)`
   - Validate bounds, height limit, and support ratio
   - Score: `DBLF_WEIGHT × (-z) + CONTACT_WEIGHT × contact_ratio + CORNER_BONUS` for true corners
4. Select the placement with the highest score.

### Key constants
| Constant | Value | Role |
|----------|-------|------|
| `DBLF_WEIGHT` | 3.0 | Prefers low placements |
| `CONTACT_WEIGHT` | 2.0 | Rewards flat surface contact |
| `CORNER_BONUS` | 1.0 | Bonus for gradient-detected corners |
| `MIN_SUPPORT` | 0.30 | Anti-float threshold |

## Performance (50 boxes, 1200×800×2700mm EUR pallet, seed=42)
| Metric | Value |
|--------|-------|
| Fill rate | **61.7%** |
| Boxes placed | 41/50 (82%) |
| Computation time | 4.4s total (0.09s/box) |

**Note:** Best placement rate of all new EMS-family strategies. The gradient-based corner detection finds compact positions that reduce fragmentation.

## Class: `GOPTHeuristicStrategy(BaseStrategy)`
- **Registry name:** `gopt_heuristic`
- **Type:** Single-bin (`BaseStrategy`)

## Stability
- `MIN_SUPPORT = 0.30` always enforced
- Respects `enable_stability` for stricter checks

## Integration
```python
python run_experiment.py --strategy gopt_heuristic --generate 50
```

## Comparison with DRL version
This is the **heuristic baseline**. The full GOPT uses a Transformer-based DRL policy (PPO) trained on 3D bin packing episodes. The DRL version achieves ~75-80% fill vs. ~62% for this heuristic. See `python/external_repos/GOPT/` for the RL implementation.
