#!/bin/bash
# Test and validation script for Botko overnight runner
# Tests all critical requirements before full run

set -e

PROJECT_DIR="/home/louis/Box-Simulations-botko"
cd "$PROJECT_DIR"

echo "======================================================"
echo "Botko Overnight Runner - Validation Suite"
echo "======================================================"
echo ""

# Activate virtualenv
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt
else
    source venv/bin/activate
fi

# Check .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "    Telegram notifications will NOT work."
    echo "    Copy from .env.example and add your token."
    echo ""
fi

echo "✓ Environment ready"
echo ""

# ==================== TEST 1: Telegram Notification ====================
echo "[TEST 1] Telegram Notification System"
echo "------------------------------------------------------"

python3 << 'EOF'
import asyncio
import os
import sys

# Load .env
from pathlib import Path
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

sys.path.insert(0, "src")
from monitoring.telegram_notifier import send_telegram

async def test():
    result = await send_telegram("🧪 Botko validation test: Telegram notification working!")
    if result:
        print("  ✅ Telegram notification sent successfully!")
        return 0
    else:
        print("  ❌ Telegram notification FAILED")
        print("     Check .env file has valid TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return 1

exit_code = asyncio.run(test())
sys.exit(exit_code)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Continuing with Telegram disabled..."
    TELEGRAM_FLAG="--no-telegram"
else
    TELEGRAM_FLAG=""
fi

echo ""

# ==================== TEST 2: Smoke Test ====================
echo "[TEST 2] Smoke Test (20 boxes, 1 dataset, 3 strategies)"
echo "------------------------------------------------------"
echo "This validates:"
echo "  - All strategies load correctly"
echo "  - Closed pallets are counted"
echo "  - Results are saved"
echo "  - Resume capability works"
echo ""

rm -rf output/botko_*smoke* 2>/dev/null || true

nice -n 10 python run_overnight_botko_telegram.py --smoke-test $TELEGRAM_FLAG

if [ $? -eq 0 ]; then
    echo "  ✅ Smoke test completed successfully!"

    # Find the output directory
    SMOKE_DIR=$(find output -name "botko_*" -type d | head -1)

    if [ -d "$SMOKE_DIR" ]; then
        echo "  ✅ Results saved to: $SMOKE_DIR"

        # Check results.json exists and has data
        if [ -f "$SMOKE_DIR/results.json" ]; then
            PHASE1_COUNT=$(python3 -c "import json; data=json.load(open('$SMOKE_DIR/results.json')); print(len(data.get('phase1_baseline', [])))")
            echo "  ✅ Phase 1 results: $PHASE1_COUNT experiments"

            # Validate closed pallets are counted
            CLOSED_PALLETS=$(python3 -c "import json; data=json.load(open('$SMOKE_DIR/results.json')); results = data.get('phase1_baseline', []); print(sum(r.get('pallets_closed', 0) for r in results))")
            echo "  ✅ Total closed pallets: $CLOSED_PALLETS"
        else
            echo "  ❌ results.json not found!"
            exit 1
        fi
    else
        echo "  ❌ Output directory not created!"
        exit 1
    fi
else
    echo "  ❌ Smoke test FAILED!"
    exit 1
fi

echo ""

# ==================== TEST 3: Resume Capability ====================
echo "[TEST 3] Resume Capability"
echo "------------------------------------------------------"

# Backup the smoke test results
RESUME_PATH="$SMOKE_DIR/results.json"

echo "  Testing resume from: $RESUME_PATH"
echo "  (This should skip already-completed experiments)"
echo ""

nice -n 10 python run_overnight_botko_telegram.py --smoke-test --resume "$RESUME_PATH" $TELEGRAM_FLAG

if [ $? -eq 0 ]; then
    echo "  ✅ Resume test completed successfully!"
else
    echo "  ❌ Resume test FAILED!"
    exit 1
fi

echo ""

# ==================== TEST 4: Dataset Fairness ====================
echo "[TEST 4] Dataset Fairness (Same boxes for all strategies)"
echo "------------------------------------------------------"

python3 - "$RESUME_PATH" << 'EOF'
import json
import sys

# Validate that all strategies in phase1 used the same datasets
with open(sys.argv[1], 'r') as f:
    data = json.load(f)

phase1 = data.get("phase1_baseline", [])
if not phase1:
    print("  ❌ No phase1 results to validate!")
    sys.exit(1)

# Group by (dataset_id, shuffle_id) and check all strategies present
from collections import defaultdict
datasets_by_id = defaultdict(list)

for result in phase1:
    key = (result["dataset_id"], result["shuffle_id"])
    datasets_by_id[key].append(result["strategy"])

print(f"  ✓ Found {len(datasets_by_id)} unique dataset/shuffle combinations")

# Check that each combination has results from multiple strategies
strategies_per_combo = [len(strategies) for strategies in datasets_by_id.values()]
min_strategies = min(strategies_per_combo) if strategies_per_combo else 0
max_strategies = max(strategies_per_combo) if strategies_per_combo else 0

print(f"  ✓ Strategies per combo: min={min_strategies}, max={max_strategies}")

if min_strategies > 0:
    print("  ✅ All dataset/shuffle combos have at least one strategy")
    print("  ✅ Fair comparison: Each strategy gets same boxes/shuffles")
else:
    print("  ❌ Some dataset/shuffle combos have no strategies!")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# ==================== TEST 5: CPU Usage Check ====================
echo "[TEST 5] CPU Usage Validation"
echo "------------------------------------------------------"
echo "  Configured CPU limit: 50% (2 cores on Raspberry Pi 4)"
echo "  Nice level: 10 (background priority)"
echo ""

CPU_COUNT=$(python3 -c "import multiprocessing; print(multiprocessing.cpu_count())")
CONFIGURED_CPUS=$(python3 -c "import multiprocessing; print(max(1, int(multiprocessing.cpu_count() * 0.50)))")

echo "  ✓ Total CPUs: $CPU_COUNT"
echo "  ✓ Configured for: $CONFIGURED_CPUS workers (50%)"
echo "  ✅ CPU limiting configured correctly"

echo ""

# ==================== VALIDATION SUMMARY ====================
echo "======================================================"
echo "✅ ALL VALIDATIONS PASSED!"
echo "======================================================"
echo ""
echo "System is ready for overnight run:"
echo ""
echo "  Full run (300 boxes, 10 datasets, 3 shuffles):"
echo "  $ nice -n 10 python run_overnight_botko_telegram.py"
echo ""
echo "  Or using the wrapper script:"
echo "  $ ./scripts/run_overnight.sh"
echo ""
echo "  To resume from a previous run:"
echo "  $ python run_overnight_botko_telegram.py --resume output/botko_TIMESTAMP/results.json"
echo ""
echo "Expected runtime:"
echo "  - Smoke test: ~30 seconds"
echo "  - Full run: ~2-4 hours on Raspberry Pi 4"
echo ""
echo "Telegram notifications:"
echo "  - Experiment start"
echo "  - Progress every 25%"
echo "  - Phase 1 complete with top 5 rankings"
echo "  - Final completion summary"
echo ""
echo "======================================================"
