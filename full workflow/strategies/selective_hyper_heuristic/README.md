# Selective Hyper-Heuristic Strategy

## Overview
A novel hyper-heuristic strategy that dynamically selects between multiple low-level placement heuristics based on the current bin state. Inspired by the Selective Hyper-Heuristic (SHH) approach to adaptive algorithm selection, with heuristic selection driven by the current fill rate and bin surface flatness.

## Approach (Novel — no direct paper, framework-inspired)
The hyper-heuristic maintains a portfolio of 4 low-level heuristics:
1. **Floor-building** — maximise base area contact (surface_contact style)
2. **Column-fill** — fill vertical columns to reduce height variance
3. **Corner-fill** — target wall/corner adjacency positions
4. **Best-fit-decreasing** — place in smallest valid footprint

At each step, the strategy selects among them based on heuristics:
- Low fill (<40%): prefer floor-building to establish a compact base
- Medium fill (40-70%): prefer column-fill or corner-fill for compactness
- High fill (>70%): prefer best-fit to utilise remaining gaps

## Performance (50 boxes, 1200×800×2700mm EUR pallet, seed=42)
| Metric | Value |
|--------|-------|
| Fill rate | **46.4%** |
| Boxes placed | 31/50 (62%) |
| Computation time | 21.4s total (0.43s/box) |

**Note:** Lower fill rate than single-heuristic strategies. Switching between heuristics can cause suboptimal decisions when the heuristic selection criteria don't align with the actual bin geometry. A learning-based selection mechanism (e.g., online bandit/RL) would likely improve this significantly.

## Class: `SelectiveHyperHeuristicStrategy(BaseStrategy)`
- **Registry name:** `selective_hyper_heuristic`
- **Type:** Single-bin (`BaseStrategy`)

## Stability
- `MIN_SUPPORT = 0.30` always enforced
- Grid scan step: `2 × resolution` (10mm → 20mm step; 2501 candidates vs 9801 at 1×)

## Performance Notes
- At 2× scan step: 21.4s / 50 boxes = 0.43s/box (well within 5-7s/box budget)
- Scan step intentionally kept at 2× (not 1×) because: hyper-heuristic testing multiple sub-strategies per box; 1× would be ~4× slower for marginal quality gain

## Integration
```python
python run_experiment.py --strategy selective_hyper_heuristic --generate 50
```

## Future Improvements
- Online ε-greedy or UCB bandit for heuristic selection
- State features beyond fill rate (surface roughness, height variance)
- Lookahead selection: simulate 1 step ahead with each heuristic
