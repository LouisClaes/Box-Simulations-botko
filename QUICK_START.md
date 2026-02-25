# 🤖 Botko Overnight Runner - Quick Start Guide

## What This Does

Runs **10 datasets × 3 orderings × 300 boxes** overnight on your Raspberry Pi:
- ✅ CPU limited to ~50% (won't overheat your Pi)
- ✅ Telegram notifications at milestones
- ✅ Only closed pallets counted in metrics
- ✅ Results saved to CSV/JSON after each dataset
- ✅ Can run for days without issues

## Setup (One-Time)

### 1. Set up Telegram credentials

```bash
cd /home/louis/Box-Simulations-botko
nano .env
```

Add your Telegram token (copy from your trading bot):
```bash
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=-1003509475971
```

### 2. Test it works

```bash
# Quick 2-dataset test (30 seconds)
source venv/bin/activate
python test_overnight.py
```

You should see:
```
✅ Test completed successfully!
   Closed pallets: ~20-30
   Total boxes: 60
```

## Running Overnight Experiments

### Option 1: Simple Script (Recommended for testing)

```bash
cd /home/louis/Box-Simulations-botko
./scripts/start_overnight.sh
```

This runs in your terminal with CPU limiting. You'll see:
- Real-time progress updates
- Results saved every 2 datasets
- Telegram notifications
- Final summary at the end

**How long?** About 30-60 minutes for 10 datasets × 300 boxes on Raspberry Pi 4.

### Option 2: Systemd Service (Recommended for production)

For truly overnight/multi-day runs:

```bash
# Install the service
cd /home/louis/Box-Simulations-botko
sudo ./scripts/install_service.sh

# Start it
./scripts/service_manager.sh start

# Check status
./scripts/service_manager.sh status

# View live logs
./scripts/service_manager.sh logs

# Stop it
./scripts/service_manager.sh stop
```

The service will:
- Auto-restart on failure
- Run in background
- Limit CPU to 50%
- Log everything to systemd journal

### Option 3: Direct Python

```bash
cd /home/louis/Box-Simulations-botko
source venv/bin/activate
nice -n 10 python run_botko_overnight.py
```

## Monitoring Progress

### Via Telegram
You'll get notifications for:
- 🚀 Experiment start
- 📊 Every 2 datasets completed (with avg utilization)
- ✅ Final completion summary

### Via Logs (if using systemd)
```bash
cd /home/louis/Box-Simulations-botko
./scripts/service_manager.sh logs
# or
journalctl -u botko-packing -f
```

### Via Results Files
```bash
cd /home/louis/Box-Simulations-botko/results
ls -lh

# View latest results
cat exp_*_final.json | python -m json.tool
```

## Results Format

After each run, you get:

1. **JSON summary** (`exp_YYYYMMDD_HHMMSS_final.json`):
   ```json
   {
     "experiment_id": "exp_20260221_094315",
     "total_pallets": 1234,
     "total_boxes": 9000,
     "avg_utilization_pct": 42.5,
     "runtime_seconds": 1835.2
   }
   ```

2. **CSV per-pallet data** (`exp_YYYYMMDD_HHMMSS_final_pallets.csv`):
   ```
   pallet_id,boxes_placed,utilization_pct,dataset_id
   pallet_001,15,45.3,dataset_000_random
   pallet_002,18,52.1,dataset_000_random
   ...
   ```

3. **Interim files** (saved after each dataset for crash recovery)

## CPU Usage

The runner uses `nice -n 10` to keep CPU around 50%:
- Leaves resources for other processes
- Prevents Pi overheating
- Still completes in reasonable time

To check CPU while running:
```bash
htop
# or
top -u louis
```

## Troubleshooting

### "Virtual environment not found"
```bash
cd /home/louis/Box-Simulations-botko
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### No Telegram notifications
1. Check `.env` file exists with valid token
2. Test notification:
   ```bash
   source venv/bin/activate
   python -c "from src.monitoring.telegram_notifier import send_telegram; import asyncio; asyncio.run(send_telegram('Test from Botko!'))"
   ```

### Service won't start
```bash
# Check service status
./scripts/service_manager.sh status

# View error logs
journalctl -u botko-packing -n 50
```

## Stopping a Running Experiment

### If using script:
Press `Ctrl+C` - partial results will be saved

### If using systemd:
```bash
./scripts/service_manager.sh stop
```

## What Gets Counted

**IMPORTANT:** Only **CLOSED** pallets count in metrics!

A pallet is "closed" when:
- It's full and sealed
- No more boxes can fit
- Algorithm has moved to next pallet

Open/partial pallets are excluded from final metrics.

## Next Steps

1. Run a quick test: `python test_overnight.py`
2. Set up Telegram in `.env`
3. Run overnight: `./scripts/start_overnight.sh`
4. Check results in `results/` directory

## Questions?

The agents created comprehensive documentation:
- `README.md` - Project overview
- `scripts/SERVICE_README.md` - Systemd service details
- `config/experiment.yaml` - Configuration options

Happy packing! 🤖📦
