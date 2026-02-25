# Strategy Testing Recommendations

## Available Strategies (24 total)

### Single-Bin (22):
1. **baseline** - Simple bottom-left placement ✅ FAST
2. **best_fit_decreasing** - Sort by size, best fit ✅ FAST
3. **blueprint_packing** - Layer-based approach ✅ FAST
4. **column_fill** - Column stacking ✅ FAST
5. **ems** - Empty Maximal Spaces ✅ FAST
6. **extreme_points** - Corner-based placement ✅ FAST
7. **gopt_heuristic** - Genetic optimization-inspired ⚠️ MODERATE
8. **gravity_balanced** - Stability-focused ✅ FAST
9. **heuristic_160** - Research paper heuristic ✅ FAST
10. **hybrid_adaptive** - Adaptive strategy ⚠️ SLOW (adaptive)
11. **layer_building** - Layer construction ✅ FAST
12. **lbcp_stability** - Stability constraints ✅ FAST
13. **lookahead** - Tree search ❌ VERY SLOW (search)
14. **online_bpp_heuristic** - Online algorithm ⚠️ MODERATE
15. **pct_expansion** - Pattern-based ✅ FAST
16. **pct_macs_heuristic** - MACS-based ✅ FAST
17. **selective_hyper_heuristic** - Hyper-heuristic ❌ SLOW (learning)
18. **skyline** - Skyline tracking ✅ FAST
19. **stacking_tree_stability** - Tree-based stability ⚠️ MODERATE
20. **surface_contact** - Contact maximization ✅ FAST
21. **wall_building** - Wall construction ✅ FAST
22. **walle_scoring** - WALL-E inspired ✅ FAST

### Multi-Bin (2):
23. **tsang_multibin** - Tsang's algorithm ✅ FAST
24. **two_bounded_best_fit** - Bounded fit ✅ FAST

## Recommendations for 2.5 Days

### EXCLUDE (3 strategies):
```python
EXCLUDED_STRATEGIES = [
    "lookahead",                    # Tree search - very slow
    "selective_hyper_heuristic",    # Learning-based - needs training
    "hybrid_adaptive",              # Adaptive - slow convergence
]
```

### TEST (21 strategies):
- All fast heuristics (baseline, ems, walle, etc.)
- Moderate ones (gopt, online_bpp, stacking_tree)
- Both multi-bin strategies

## Estimated Runtime with Exclusions

### Quick Run (`--quick` flag):
```
21 strategies × 5 datasets × 3 shuffles = 315 experiments (Phase 1)
Top 5 × 3 × 3 × 4 datasets = 180 experiments (Phase 2)
Total: 495 experiments × ~6 min = 49.5 hours (~2.1 days)
```
✅ **Fits comfortably in 2.5 days**

### Full Run (default):
```
21 strategies × 10 datasets × 3 shuffles = 630 experiments (Phase 1)
Top 5 × 3 × 3 × 8 datasets = 360 experiments (Phase 2)
Total: 990 experiments × ~6 min = 99 hours (~4.1 days)
```

## Usage

### Smoke Test (2 min):
```bash
python run_overnight_botko_telegram.py --smoke-test
```

### Quick Run - 2.5 days (RECOMMENDED):
```bash
python run_overnight_botko_telegram.py --quick
```

### Full Run - 4-5 days:
```bash
python run_overnight_botko_telegram.py
```

## Notes

- Excluded strategies can be tested separately after training/optimization
- Average 6 min/experiment is conservative (some finish in 2-3 min)
- With pallet closing logic working, experiments may be faster
- Resume capability means interruptions won't lose progress
