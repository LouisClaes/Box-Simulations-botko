# Final Demo Setup - Complete Reference

## Current Configuration

### Excluded Strategies (Need Training)
```python
EXCLUDED_STRATEGIES = [
    "selective_hyper_heuristic",  # Needs training - no pre-trained model
]
```

### Slow Strategies (Run Last)
```python
SLOW_STRATEGIES = [
    "lookahead",         # Tree search
    "hybrid_adaptive",   # Adaptive learning
]
```

**Strategy execution order:**
1. Fast strategies first (19 strategies)
2. Slow strategies last (2 strategies) - can be skipped if too slow

**Total strategies: 21 single-bin + 2 multi-bin = 23 strategies**

---

## Validation Test Running

**Test:** `test_slow_strategies.py`
**Dataset:** 50 boxes (reduced from 400 for quick test)
**Timeout:** 10 minutes
**Goal:** Determine if lookahead & hybrid_adaptive are viable

**Viability threshold:** < 10 minutes for 400 boxes

**Possible outcomes:**
- ✅ Both viable → Run full demo with all 23 strategies
- ⚠️ One viable → Run demo with 22 strategies (exclude the slow one)
- ✗ None viable → Run demo with 21 strategies (exclude both slow ones)

---

## Demo Run Parameters

### Dataset Configuration
```python
n_datasets = 3      # 3 different datasets
n_shuffles = 2      # 2 shuffles per dataset
n_boxes = 400       # 400 Rajapack boxes
```

### Phase 1: Baseline (All Strategies)
**If all 23 strategies viable:**
```
21 single-bin × 3 datasets × 2 shuffles = 126 experiments
2 multi-bin × 3 datasets × 2 shuffles = 12 experiments
Total: 138 experiments
```

**Execution order:**
- Experiments 1-114: Fast strategies (19 × 3 × 2)
- Experiments 115-126: Slow strategies (2 × 3 × 2) **← Can skip if needed**
- Experiments 127-138: Multi-bin strategies

### Phase 2: Parameter Sweep (Top-5)
```
5 strategies (top from Phase 1)
3 box selectors × 3 bin selectors = 9 combinations
2 datasets (reduced for demo)
Total: 90 experiments
```

---

## Runtime Estimates

### Scenario A: All Strategies Viable (23 total)
```
Phase 1: 138 experiments × 7 min = 966 min = 16.1 hours
Phase 2: 90 experiments × 7 min = 630 min = 10.5 hours
Total: 26.6 hours (with 2 CPUs in parallel: ~13.3 hours)
```

### Scenario B: Fast Strategies Only (21 total)
```
Phase 1: 126 experiments × 7 min = 882 min = 14.7 hours
Phase 2: 90 experiments × 7 min = 630 min = 10.5 hours
Total: 25.2 hours (with 2 CPUs in parallel: ~12.6 hours)
```

**Recommended:** Start with all strategies, skip slow ones if they exceed 15 min per experiment.

---

## How to Run

### 1. Check Slow Strategy Test Results
```bash
cd /home/louis/Box-Simulations-botko
cat /tmp/claude-1000/-home-louis-bot/tasks/b5e516a.output
```

**Decision tree:**
- If both viable → Proceed with full demo
- If one/both not viable → Update EXCLUDED_STRATEGIES list before running

### 2. Update Exclusions (If Needed)
```bash
# Edit run_overnight_botko_telegram.py line 324
# Add non-viable strategies to EXCLUDED_STRATEGIES list
```

### 3. Run Demo
```bash
cd /home/louis/Box-Simulations-botko
source venv/bin/activate

# Background with logging
nohup python run_overnight_botko_telegram.py --demo --no-telegram > demo_run.log 2>&1 &
echo $! > demo_run.pid
echo "Demo run started with PID: $(cat demo_run.pid)"
```

### 4. Monitor Progress
```bash
# Real-time log
tail -f demo_run.log

# Check phase progress
watch -n 60 'cat output/botko_*/results.json 2>/dev/null | python -c "import json, sys; d=json.load(sys.stdin); print(f\"Phase 1: {len(d.get(\\\"phase1_baseline\\\", []))}/138 | Phase 2: {len(d.get(\\\"phase2_sweep\\\", []))}/90\")" 2>/dev/null || echo "Waiting for results..."'

# Check if running
ps -p $(cat demo_run.pid 2>/dev/null) > /dev/null 2>&1 && echo "✓ Running" || echo "✗ Not running"
```

### 5. Stop If Needed
```bash
# Graceful stop (saves progress)
kill $(cat demo_run.pid)

# Force stop
kill -9 $(cat demo_run.pid)
```

### 6. Resume After Interruption
```bash
python run_overnight_botko_telegram.py --demo --resume output/botko_*/results.json
```

---

## What to Watch For

### Success Indicators ✅
- Phase 1 progressing steadily
- Experiments completing in 5-10 minutes each
- Pallets closing (should see `[PALLET CLOSED]` messages)
- Results.json updating every ~5%
- No crashes or errors

### Warning Signs ⚠️
- Experiments taking > 15 minutes
- Many experiments with 0 closed pallets
- High memory usage (> 6GB on 8GB Pi)
- Process stops unexpectedly

### If Slow Strategies Are Too Slow
**Signs:**
- lookahead/hybrid_adaptive experiments taking > 20 minutes
- Phase 1 completion estimate > 20 hours

**Action:**
1. Kill the process: `kill $(cat demo_run.pid)`
2. Update exclusions:
   ```python
   EXCLUDED_STRATEGIES = [
       "selective_hyper_heuristic",
       "lookahead",           # Too slow
       "hybrid_adaptive",     # Too slow
   ]
   ```
3. Resume: `python run_overnight_botko_telegram.py --demo --resume output/botko_*/results.json`

---

## After Completion

### Check Results
```bash
cd /home/louis/Box-Simulations-botko

# View summary
python << 'EOF'
import json
with open("output/botko_*/results.json") as f:
    data = json.load(f)

print("=== DEMO RUN RESULTS ===")
print(f"Phase 1: {len(data['phase1_baseline'])} experiments")
print(f"Phase 2: {len(data['phase2_sweep'])} experiments")
print(f"\nTop 5 strategies:")
for i, s in enumerate(data['metadata']['top_5'], 1):
    print(f"  {i}. {s}")

# Find best overall
best = max(data['phase1_baseline'], key=lambda x: x.get('avg_closed_fill', 0))
print(f"\nBest strategy: {best['strategy']}")
print(f"  Fill rate: {best['avg_closed_fill']:.1%}")
print(f"  Pallets closed: {best['pallets_closed']}")
EOF
```

### Validate Data
```bash
# Check all experiments succeeded
cat output/botko_*/results.json | python -c "
import json, sys
data = json.load(sys.stdin)
p1 = len(data.get('phase1_baseline', []))
p2 = len(data.get('phase2_sweep', []))
print(f'Phase 1: {p1} (expected 138)')
print(f'Phase 2: {p2} (expected 90)')
"
```

---

## Complete File Reference

### Files Modified
1. `run_overnight_botko_telegram.py` - Main runner (exclusions, slow strategy ordering)
2. `simulator/session.py` - Close logic, early termination, rejected boxes tracking

### Documentation Created
1. `CLOSE_LOGIC_EXPLAINED.md` - Pallet close system
2. `PALLET_DUO_METRICS.md` - Rejected boxes tracking
3. `EARLY_TERMINATION_OPTIMIZATION.md` - Time savings optimization
4. `DEMO_MODE_VALIDATION.md` - System validation report
5. `DEMO_RUN_READY.md` - Comprehensive demo guide
6. `FINAL_DEMO_SETUP.md` - This file

### Test Scripts
1. `test_lookahead_close.py` - Validate close logic
2. `test_slow_strategies.py` - Validate slow strategy viability

---

## Quick Command Reference

```bash
# Activate environment
cd /home/louis/Box-Simulations-botko && source venv/bin/activate

# Check slow strategy test results
cat /tmp/claude-1000/-home-louis-bot/tasks/b5e516a.output

# Run demo
nohup python run_overnight_botko_telegram.py --demo --no-telegram > demo_run.log 2>&1 & echo $! > demo_run.pid

# Monitor
tail -f demo_run.log | grep -E "Phase|%|CLOSE"

# Check progress
cat output/botko_*/results.json | python -c "import json, sys; d=json.load(sys.stdin); print(f'P1: {len(d.get(\"phase1_baseline\",[]))} | P2: {len(d.get(\"phase2_sweep\",[]))}')"

# Stop
kill $(cat demo_run.pid)

# Resume
python run_overnight_botko_telegram.py --demo --resume output/botko_*/results.json
```

---

**NEXT STEP:** Check slow strategy test results, then run demo! 🚀
