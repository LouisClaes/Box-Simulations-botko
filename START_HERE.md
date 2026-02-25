# 🚀 Botko Overnight Runner - START HERE

## ✅ System Status: PRODUCTION READY

All validations passed! Your 3D bin packing overnight runner is configured and tested.

---

## 🎯 Quick Start - Run Full Production Test

```bash
cd /home/louis/Box-Simulations-botko

# Start the overnight run (2-4 hours on Raspberry Pi 4)
nice -n 10 python run_overnight_botko_telegram.py
```

---

## 📊 What This Does

**Comprehensive benchmark of 25+ packing strategies:**
- ✅ **300 boxes** per dataset (Rajapack warehouse distribution)
- ✅ **10 datasets** (different box combinations)
- ✅ **3 sequences** per dataset (random, size-sorted, weight-sorted)
- ✅ **Total: 750 experiments** (25 strategies × 10 datasets × 3 sequences)
- ✅ **Fair comparison**: All strategies get EXACTLY the same boxes (fixed seeds)
- ✅ **Metrics**: Only closed pallets count (1800mm height reached)

---

## 📡 You'll Get Telegram Notifications For:

1. Experiment start
2. Progress every 25% (4 updates total)
3. Top 5 rankings when complete
4. Final summary with best strategy

---

## 🔧 Configuration

| Setting | Value |
|---------|-------|
| CPU Usage | 50% (2 cores) |
| Nice Level | 10 (background priority) |
| Runtime | 2-4 hours |
| Telegram | ✅ Configured & tested |
| Resume | ✅ Enabled (auto-saves every ~5%) |

---

## 📖 Full Documentation

- **PRODUCTION_READY.md** - Complete feature guide
- **QUICK_START.md** - Simple usage instructions
- **README_botko.md** - Original Botko system docs

---

## 🧪 Already Validated

✅ Telegram notifications sent successfully
✅ Smoke test completed (3 strategies tested)
✅ Resume capability verified (skips completed work)
✅ Dataset fairness confirmed (same boxes for all strategies)
✅ CPU limiting configured (50% = 2 workers on Pi 4)

---

## 🚦 Ready to Go!

```bash
# Start now:
cd /home/louis/Box-Simulations-botko
nice -n 10 python run_overnight_botko_telegram.py

# Or use wrapper script:
./scripts/run_overnight_full.sh
```

**Expected completion:** 2-4 hours from now
**Results location:** `output/botko_TIMESTAMP/results.json`

---

Good luck! 🤖📦
