# Stacking Tree Stability Strategy

## Overview
Implements stability-aware placement using a tree structure that tracks support relationships between placed boxes. Based on the stability analysis framework from Zhao et al. (2023), which models the bin as a directed support graph (stacking tree) and validates new placements by propagating stability checks through the tree.

## Paper
Zhao, H., She, Q., Zhu, C., Yang, Y., & Xu, K. (2021/2023). "Online 3D Bin Packing with Constrained Deep Reinforcement Learning." *AAAI Conference on Artificial Intelligence*. arXiv:2012.04412. https://github.com/alexfrom0815/Online-3D-BPP-DRL

## Algorithm
1. Maintain a **stacking tree**: each placed box is a node; edges point from supported to supporting items.
2. For each candidate position (grid + box corners):
   - Find support items: placed boxes with `z_max ≈ z` and overlapping footprint
   - Compute support polygon (convex hull of contact region)
   - Check if centre-of-gravity lies inside the support polygon
3. Score: `support_ratio × 5.0 - height_norm × 2.0 + contact_ratio × 3.0`
4. Select the placement with the highest stability score.

## Performance (50 boxes, 1200×800×2700mm EUR pallet, seed=42)
| Metric | Value |
|--------|-------|
| Fill rate | **49.2%** |
| Boxes placed | 33/50 (66%) |
| Computation time | 30.2s total (0.60s/box) |

**Note:** Lower fill rate because strict stability validation rejects more placements. The tree traversal per candidate (O(N) where N = placed boxes) grows as the bin fills, making later decisions slower.

## Class: `StackingTreeStabilityStrategy(BaseStrategy)`
- **Registry name:** `stacking_tree_stability`
- **Type:** Single-bin (`BaseStrategy`)

## Stability
- `MIN_SUPPORT = 0.30` always enforced
- Full stacking tree validation per candidate (stricter than simple support check)
- Respects `enable_stability` for additional checks

## Integration
```python
python run_experiment.py --strategy stacking_tree_stability --generate 50
```

## Comparison with LBCP
Both strategies do stability validation beyond the simple support ratio:
- `stacking_tree_stability`: uses support polygon + CoG check (Zhao et al. 2021)
- `lbcp_stability`: uses Load-Bearable Convex Polygon + CoG check (Gao et al. 2025)
LBCP is more rigorous (transmits forces through the full stack) but slower.
