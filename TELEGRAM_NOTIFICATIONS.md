# Telegram Notifications - Complete Reference

## Overview

You will receive **VERY FREQUENT** updates throughout the demo run to stay fully in the loop.

**Update frequency:** Every 5% completion (20 updates per phase)

---

## Notification Timeline

### 1. Demo Start
```
🚀 DEMO RUN STARTING
━━━━━━━━━━━━━━━━━━━━
📊 Phase 1: Baseline Testing
  Fast strategies: 19
  Slow strategies: 2 (run last)
  Multi-bin: 2
  Datasets: 3
  Shuffles per dataset: 2
  Total experiments: 138

⏱ Estimated time: ~8-9 hours
📈 Updates every 5%
```

### 2. Phase 1 Progress Updates (Every 5%)

**Fast strategies zone (0-82%):**
```
📈 Phase 1: 5%
Progress: 7/138 experiments
Zone: ⚡ Fast strategies
Elapsed: 24.5m | ETA: 465.2m
Pallets closed: 67
Avg fill: 68.3%
Rate: 17.1 exp/hour
```

**Frequency:**
- 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%,
  55%, 60%, 65%, 70%, 75%, 80%

### 3. Slow Strategies Alert (~83%)
```
🐌 ENTERING SLOW STRATEGIES ZONE
━━━━━━━━━━━━━━━━━━━━
Now testing: lookahead & hybrid_adaptive
These take 10-12 min per experiment
(vs 7 min for fast strategies)
Progress: 114/138
ETA: 138.0m
```

### 4. Phase 1 Slow Zone (83-91%)

```
📈 Phase 1: 85%
Progress: 117/138 experiments
Zone: 🐌 SLOW STRATEGIES
Elapsed: 512.3m | ETA: 92.1m
Pallets closed: 1245
Avg fill: 67.8%
Rate: 13.7 exp/hour
```

**Frequency:** 85%, 90%

### 5. Multi-bin Zone (91-100%)
```
📈 Phase 1: 95%
Progress: 131/138 experiments
Zone: ⚡ Fast strategies
Elapsed: 548.7m | ETA: 27.4m
Pallets closed: 1389
Avg fill: 68.1%
Rate: 14.3 exp/hour
```

**Frequency:** 95%, 100%

### 6. Phase 1 Complete
```
✅ Phase 1 COMPLETE!
Time: 510.2 min (8.5 hours)
Experiments: 138
Total pallets closed: 1518
Overall avg fill: 68.2%
Now starting Phase 2...
```

### 7. Phase 2 Start
```
📊 Phase 2 Starting: Bin/Box Selector Sweep
Top 5 strategies
Box selectors: 3
Bin selectors: 3
Variations: 2 datasets
```

### 8. Phase 2 Progress Updates (Every 5%)

```
📊 Phase 2: 10%
Progress: 9/90 configs
Testing: walle_scoring
  Box sel: biggest_volume_first
  Bin sel: focus_fill
Elapsed: 28.3m | ETA: 254.7m
Pallets: 96 | Avg fill: 69.1%
Rate: 19.1 exp/hour
```

**Frequency:**
- 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%,
  55%, 60%, 65%, 70%, 75%, 80%, 85%, 90%, 95%, 100%

### 9. Phase 2 Complete
```
✅ Phase 2 COMPLETE!
Time: 283.4 min (4.7 hours)
Configurations tested: 90
Pallets closed: 990
Avg fill: 69.5%
Preparing final summary...
```

### 10. Final Summary
```
🎉 BOTKO DEMO COMPLETE!
━━━━━━━━━━━━━━━━━━━━
⏱ Total Runtime: 793.6 min (13.2 hours)
  Phase 1: 510.2 min
  Phase 2: 283.4 min

🏆 Top 5 Strategies:
  1. walle_scoring (71.2%)
  2. surface_contact (70.8%)
  3. ems (69.4%)
  4. lookahead (68.9%)
  5. baseline (68.1%)

📊 Statistics:
  Experiments: 228
  Strategies tested: 23
  Overall avg fill: 68.2%

📦 Pallets Closed:
  Phase 1: 1518
  Phase 2: 990
  Total: 2508

💾 Results saved to:
  output/botko_20260223_*/results.json
```

---

## Total Notifications

### Phase 1
- **Start:** 1 notification
- **Progress updates (every 5%):** 20 notifications
- **Slow zone alert:** 1 notification
- **Completion:** 1 notification
- **Subtotal:** 23 notifications

### Phase 2
- **Start:** 1 notification
- **Progress updates (every 5%):** 20 notifications
- **Completion:** 1 notification
- **Subtotal:** 22 notifications

### Final
- **Summary:** 1 notification

**Grand Total:** ~46 notifications over ~13 hours

**Average frequency:** 1 notification every ~17 minutes

---

## What Each Field Means

### Progress Updates

**Zone:**
- `⚡ Fast strategies` - Testing fast strategies (~7 min/experiment)
- `🐌 SLOW STRATEGIES` - Testing slow strategies (~11 min/experiment)

**Elapsed:** Time since phase started

**ETA:** Estimated time remaining for current phase

**Pallets closed:** Total pallets that reached ≥50% fill and were closed

**Avg fill:** Average volumetric fill rate of closed pallets (primary metric)

**Rate:** Experiments completed per hour

### Phase 2 Specific

**Testing:** Current strategy being tested

**Box sel:** Box selection strategy (which box from pick window to try first)
- `default` - FIFO order
- `biggest_volume_first` - Largest volume first
- `biggest_footprint_first` - Largest base area first

**Bin sel:** Bin selection strategy (which pallet to prefer)
- `emptiest_first` - Pallet with most free space
- `focus_fill` - Concentrate on one pallet
- `flattest_first` - Pallet with flattest top surface

---

## Notification Setup

### With Telegram (Recommended)
```bash
# Ensure .env file has your Telegram credentials
cat /home/louis/Box-Simulations-botko/.env
# Should contain:
# TELEGRAM_BOT_TOKEN=your_token
# TELEGRAM_CHAT_ID=your_chat_id

# Run with Telegram enabled
python run_overnight_botko_telegram.py --demo
```

### Without Telegram
```bash
# Run without notifications (still logs to file)
python run_overnight_botko_telegram.py --demo --no-telegram
```

---

## Sample Timeline

Assuming start at 00:00:

| Time | Event | Notification |
|------|-------|--------------|
| 00:00 | Start | 🚀 Demo starting |
| 00:21 | 5% P1 | 📈 Phase 1: 5% |
| 00:42 | 10% P1 | 📈 Phase 1: 10% |
| 01:03 | 15% P1 | 📈 Phase 1: 15% |
| 01:24 | 20% P1 | 📈 Phase 1: 20% |
| ... | ... | ... |
| 07:00 | 83% P1 | 🐌 Entering slow zone |
| 07:15 | 85% P1 | 📈 Phase 1: 85% (slow) |
| 07:45 | 90% P1 | 📈 Phase 1: 90% (slow) |
| 08:00 | 95% P1 | 📈 Phase 1: 95% |
| 08:30 | 100% P1 | ✅ Phase 1 complete |
| 08:30 | P2 start | 📊 Phase 2 starting |
| 08:44 | 5% P2 | 📊 Phase 2: 5% |
| 08:58 | 10% P2 | 📊 Phase 2: 10% |
| ... | ... | ... |
| 13:00 | 100% P2 | ✅ Phase 2 complete |
| 13:00 | Done | 🎉 Demo complete! |

**Total notifications:** ~46 over 13 hours

---

## Troubleshooting

### Not Receiving Notifications

1. **Check .env file:**
   ```bash
   cat /home/louis/Box-Simulations-botko/.env
   ```

2. **Verify bot token and chat ID:**
   ```bash
   cd /home/louis/Box-Simulations-botko
   source venv/bin/activate
   python << 'EOF'
   import os
   from dotenv import load_dotenv
   load_dotenv()
   print(f"Bot token: {os.getenv('TELEGRAM_BOT_TOKEN', 'NOT SET')[:20]}...")
   print(f"Chat ID: {os.getenv('TELEGRAM_CHAT_ID', 'NOT SET')}")
   EOF
   ```

3. **Test manually:**
   ```bash
   python << 'EOF'
   from monitoring.telegram_notifier import send_telegram
   import asyncio
   asyncio.run(send_telegram("Test message from Botko!"))
   EOF
   ```

### Too Many Notifications

If 46 notifications over 13 hours is too much, you can:

**Option 1:** Reduce frequency to 10% (edit line 476):
```python
if use_telegram and (current_pct - last_telegram_pct >= 10 ...
```

**Option 2:** Disable Telegram, check log file instead:
```bash
python run_overnight_botko_telegram.py --demo --no-telegram
tail -f demo_run.log
```

---

## You Will Be VERY Well Informed! 📱

With updates every 5%, you'll always know:
- ✅ Exact progress percentage
- ✅ How many experiments completed
- ✅ Current zone (fast vs slow strategies)
- ✅ Time elapsed and ETA
- ✅ Performance metrics (pallets, fill rate)
- ✅ Completion rate (experiments per hour)

**You'll never wonder what's happening!** 🎯
