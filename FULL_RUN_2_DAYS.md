# Full Demo Run - 2 Days Allowed

## ✅ ALL STRATEGIES INCLUDED

Since 2 days (48 hours) is allowed, **ALL tested strategies are viable!**

### Final Configuration

**Excluded:** Only 1 strategy
```python
EXCLUDED_STRATEGIES = [
    "selective_hyper_heuristic",  # Needs training (no pre-trained model)
]
```

**Included:** 23 strategies total
- **Fast strategies (19):** 7 min/experiment
- **Slow strategies (2):** 11-12 min/experiment (run last)
- **Multi-bin (2):** 7 min/experiment

**Strategy execution order:**
1. Fast strategies first (experiments 1-114)
2. Slow strategies last (experiments 115-126) ← These run at the end
3. Multi-bin strategies (experiments 127-138)

---

## Runtime Estimate - FULL TEST

### Phase 1: All Strategies Baseline

**Fast strategies:**
```
19 single-bin × 3 datasets × 2 shuffles = 114 experiments
Time: 114 experiments × 7 min = 798 min
Parallel (2 CPUs): 798 / 2 = 399 min = 6.6 hours
```

**Slow strategies:**
```
2 single-bin × 3 datasets × 2 shuffles = 12 experiments
Time: 12 experiments × 11.5 min = 138 min
Parallel (2 CPUs): 138 / 2 = 69 min = 1.2 hours
```

**Multi-bin:**
```
2 multi-bin × 3 datasets × 2 shuffles = 12 experiments
Time: 12 experiments × 7 min = 84 min
Parallel (2 CPUs): 84 / 2 = 42 min = 0.7 hours
```

**Phase 1 Total:** 6.6 + 1.2 + 0.7 = **8.5 hours**

### Phase 2: Top-5 Parameter Sweep

```
5 strategies × 3 box selectors × 3 bin selectors × 2 datasets = 90 experiments
Time: 90 experiments × 7 min = 630 min
Parallel (2 CPUs): 630 / 2 = 315 min = 5.3 hours
```

**Phase 2 Total:** **5.3 hours**

### Total Runtime

```
Phase 1: 8.5 hours
Phase 2: 5.3 hours
Total: 13.8 hours
```

**With early termination optimization:** ~13.0 - 13.5 hours

**Total estimate:** **~14 hours** ✅ Well within 2-day (48-hour) limit!

---

## Experiment Breakdown

### Total Experiments: 228

| Category | Count | Time/Exp | Total Time |
|----------|-------|----------|------------|
| Fast single-bin | 114 | 7 min | 798 min |
| Slow single-bin | 12 | 11.5 min | 138 min |
| Multi-bin | 12 | 7 min | 84 min |
| **Phase 1 Total** | **138** | - | **1020 min** |
| Phase 2 sweep | 90 | 7 min | 630 min |
| **Grand Total** | **228** | - | **1650 min** |

**Parallel execution (2 CPUs):** 1650 / 2 = 825 min = **13.75 hours**

---

## Strategy List - COMPLETE

### Fast Single-Bin (19 strategies, ~7 min each)
1. baseline
2. best_fit_decreasing
3. blueprint_packing
4. column_fill
5. ems
6. extreme_points
7. gopt_heuristic
8. gravity_balanced
9. heuristic_160
10. layer_building
11. lbcp_stability
12. online_bpp_heuristic
13. pct_expansion
14. pct_macs_heuristic
15. skyline
16. stacking_tree_stability
17. surface_contact
18. wall_building
19. walle_scoring

### Slow Single-Bin (2 strategies, ~11-12 min each)
20. **lookahead** - Tree search (11.6 min)
21. **hybrid_adaptive** - Adaptive learning (10.2 min)

### Multi-Bin (2 strategies, ~7 min each)
22. tsang_multibin
23. two_bounded_best_fit

**Total: 23 strategies** (excluding only selective_hyper_heuristic)

---

## Why This Works for 2 Days

### Time Breakdown

**Actual runtime:** ~14 hours
**Available time:** 48 hours (2 days)
**Buffer:** 34 hours (70% buffer for safety)

**Benefits of 2-day timeline:**
- Can handle unexpected slowdowns
- Room for Raspberry Pi thermal throttling
- Can pause/resume if needed
- No rush, all strategies thoroughly tested

---

## RUN FULL DEMO NOW 🚀

### 1. Verify Configuration
```bash
cd /home/louis/Box-Simulations-botko
source venv/bin/activate

python << 'EOF'
from run_overnight_botko_telegram import EXCLUDED_STRATEGIES, SLOW_STRATEGIES
from strategies.base_strategy import STRATEGY_REGISTRY, MULTIBIN_STRATEGY_REGISTRY

included = [s for s in STRATEGY_REGISTRY.keys() if s not in EXCLUDED_STRATEGIES]
print("=" * 70)
print("FULL 2-DAY RUN CONFIGURATION")
print("=" * 70)
print(f"Fast strategies: {len([s for s in included if s not in SLOW_STRATEGIES])}")
print(f"Slow strategies: {len([s for s in included if s in SLOW_STRATEGIES])}")
print(f"Multi-bin: {len(MULTIBIN_STRATEGY_REGISTRY)}")
print(f"Total included: {len(included) + len(MULTIBIN_STRATEGY_REGISTRY)}")
print(f"Excluded: {len(EXCLUDED_STRATEGIES)}")
print(f"\nSlow strategies (run last):")
for s in SLOW_STRATEGIES:
    print(f"  - {s}")
print(f"\nPhase 1: {(len(included) + len(MULTIBIN_STRATEGY_REGISTRY)) * 3 * 2} experiments")
print(f"Phase 2: 90 experiments")
print(f"Total: {(len(included) + len(MULTIBIN_STRATEGY_REGISTRY)) * 3 * 2 + 90} experiments")
print(f"\nEstimated time: ~14 hours")
print(f"Available time: 48 hours (2 days)")
print(f"Buffer: {48 - 14} hours")
print("=" * 70)
print("✅ FULL TEST READY")
print("=" * 70)
EOF
```

Expected output:
```
======================================================================
FULL 2-DAY RUN CONFIGURATION
======================================================================
Fast strategies: 19
Slow strategies: 2
Multi-bin: 2
Total included: 23
Excluded: 1

Slow strategies (run last):
  - lookahead
  - hybrid_adaptive

Phase 1: 138 experiments
Phase 2: 90 experiments
Total: 228 experiments

Estimated time: ~14 hours
Available time: 48 hours (2 days)
Buffer: 34 hours
======================================================================
✅ FULL TEST READY
======================================================================
```

### 2. Run Full Demo
```bash
nohup python run_overnight_botko_telegram.py --demo --no-telegram > demo_run.log 2>&1 &
echo $! > demo_run.pid
echo "Full demo started with PID: $(cat demo_run.pid)"
echo "Start time: $(date)"
echo "Expected completion: $(date -d '+14 hours')"
```

### 3. Monitor Progress
```bash
# Live log
tail -f demo_run.log | grep -E "PHASE|%|CLOSE|lookahead|hybrid"

# Progress summary
watch -n 120 'cat output/botko_*/results.json 2>/dev/null | python -c "
import json, sys
try:
    d = json.load(sys.stdin)
    p1 = len(d.get(\"phase1_baseline\", []))
    p2 = len(d.get(\"phase2_sweep\", []))
    print(f\"Phase 1: {p1}/138 ({p1/138*100:.1f}%)\")
    print(f\"Phase 2: {p2}/90 ({p2/90*100:.1f}%)\")

    # Check if we hit slow strategies yet
    slow_count = sum(1 for r in d.get(\"phase1_baseline\", []) if r.get(\"strategy\") in [\"lookahead\", \"hybrid_adaptive\"])
    if slow_count > 0:
        print(f\"Slow strategies: {slow_count}/12 completed\")
except:
    print(\"Waiting for results...\")
"'
```

---

## What Makes This Different from Quick Run

### This Full Demo:
- **All 23 strategies** (only 1 excluded)
- Includes slow strategies: lookahead, hybrid_adaptive
- **228 total experiments**
- **~14 hours** runtime
- Most comprehensive test

### If You Had Used Quick Mode (`--quick`):
- 23 strategies × 5 datasets × 3 shuffles = 345 experiments
- ~24 hours runtime
- More datasets but same strategies

### If You Used Full Mode (no flags):
- 23 strategies × 10 datasets × 3 shuffles = 690 experiments
- ~48 hours runtime
- Maximum data but takes full 2 days

**Demo mode is perfect:** Good balance of coverage and time!

---

## Advantages of Running Slow Strategies Last

### If You Need to Stop Early:

Since slow strategies run **last** (experiments 115-126), you can:

1. **Monitor Phase 1 progress:**
   - If fast strategies finish in ~7 hours, you're doing great
   - If slow strategies start and take too long, you can stop

2. **Decision point at experiment 114:**
   - Fast strategies done
   - Check time remaining
   - If running late, can skip slow strategies

3. **Stop and resume:**
   ```bash
   # If you want to skip slow strategies partway through
   kill $(cat demo_run.pid)

   # Results still valid for fast strategies
   # Phase 2 will use top-5 from fast strategies only
   ```

**But with 2 days:** You likely won't need to stop! 14 hours << 48 hours.

---

## Success Criteria

After ~14 hours:

✅ **Phase 1:** 138/138 experiments (including 12 slow ones)
✅ **Phase 2:** 90/90 experiments
✅ **All strategies tested** (except selective_hyper_heuristic)
✅ **Comprehensive results** with slow strategy data
✅ **Top-5 identified** (may include lookahead or hybrid_adaptive!)

---

## 🎯 RECOMMENDATION: RUN FULL DEMO

With 2 days available, **run ALL strategies** for maximum insight!

```bash
cd /home/louis/Box-Simulations-botko
source venv/bin/activate
nohup python run_overnight_botko_telegram.py --demo --no-telegram > demo_run.log 2>&1 &
echo $! > demo_run.pid
```

**Expected completion: ~14 hours**
**Total experiments: 228**
**All strategies tested: 23 (only selective_hyper_heuristic excluded)**

**Good luck with the full test! 🚀**
