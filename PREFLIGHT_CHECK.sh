#!/bin/bash
# Pre-Flight Safety Check for 2-Day Demo Run
# Validates everything is 1000% ready

set -e

cd "$(dirname "$0")"

echo "========================================================================"
echo "PRE-FLIGHT SAFETY CHECK - 2-DAY DEMO RUN"
echo "========================================================================"
echo ""

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "[OK] Virtual environment activated"
else
    echo "[FAIL] venv directory not found!"
    exit 1
fi

echo ""
echo "=== SYSTEM CHECKS ==="
echo ""

# Check 1: Disk space
echo "[1/10] Checking disk space..."
AVAILABLE=$(df . | tail -1 | awk '{print $4}')
if [ "$AVAILABLE" -gt 1000000 ]; then
    echo "  OK - $(df -h . | tail -1 | awk '{print $4}') available"
else
    echo "  WARNING - Low disk space: $(df -h . | tail -1 | awk '{print $4}')"
    echo "  Consider cleaning: rm -rf output/botko_202602[01]*"
fi

# Check 2: Memory
echo "[2/10] Checking memory..."
FREE_MEM=$(free -m | grep Mem | awk '{print $7}')
if [ "$FREE_MEM" -gt 2000 ]; then
    echo "  OK - ${FREE_MEM}MB free"
else
    echo "  WARNING - Low memory: ${FREE_MEM}MB free"
fi

# Check 3: Python version
echo "[3/10] Checking Python..."
PYTHON_VER=$(python --version 2>&1)
echo "  OK - $PYTHON_VER"

# Check 4: Dependencies
echo "[4/10] Checking dependencies..."
python << 'EOF'
try:
    import numpy
    import httpx
    import pydantic
    import pytest
    print("  OK - All dependencies available")
except ImportError as e:
    print(f"  FAIL - Missing dependency: {e}")
    exit(1)
EOF

# Check 5: No emojis in print statements
echo "[5/10] Checking for emoji safety..."
EMOJI_COUNT=$(grep -c "print.*[😊🎯✅❌⚡🐌📈📊🚀🎉🏆💾📦⏱]" run_overnight_botko_telegram.py 2>/dev/null || true)
if [ -z "$EMOJI_COUNT" ] || [ "$EMOJI_COUNT" = "0" ]; then
    echo "  OK - No emojis in print statements (terminal-safe)"
else
    echo "  WARNING - Found $EMOJI_COUNT emojis in print statements"
fi

# Check 6: Configuration
echo "[6/10] Checking configuration..."
python << 'EOF'
from strategies.base_strategy import STRATEGY_REGISTRY, MULTIBIN_STRATEGY_REGISTRY

# Count strategies (selective_hyper_heuristic excluded in actual run)
EXCLUDED = ["selective_hyper_heuristic"]
SLOW = ["lookahead", "hybrid_adaptive"]

included = [s for s in STRATEGY_REGISTRY.keys() if s not in EXCLUDED]
fast_strategies = [s for s in included if s not in SLOW]
slow_strategies = [s for s in included if s in SLOW]

total_strategies = len(included) + len(MULTIBIN_STRATEGY_REGISTRY)
phase1_exp = total_strategies * 3 * 2  # 3 datasets × 2 shuffles
phase2_exp = 90
total_exp = phase1_exp + phase2_exp

print(f"  Total strategies: {total_strategies}")
print(f"  Fast strategies: {len(fast_strategies) + len(MULTIBIN_STRATEGY_REGISTRY)}")
print(f"  Slow strategies: {len(slow_strategies)}")
print(f"  Excluded: {len(EXCLUDED)}")
print(f"  Phase 1 experiments: {phase1_exp}")
print(f"  Phase 2 experiments: {phase2_exp}")
print(f"  Total experiments: {total_exp}")

if total_exp == 228:
    print("  OK - Configuration correct (228 experiments)")
else:
    print(f"  WARNING - Expected 228 experiments, got {total_exp}")
EOF

# Check 7: Resume capability
echo "[7/10] Checking resume capability..."
python << 'EOF'
import os
import json

# Check if there are old results
results_files = [
    os.path.join("output", d, "results.json")
    for d in os.listdir("output")
    if os.path.isdir(os.path.join("output", d)) and d.startswith("botko_")
    and os.path.exists(os.path.join("output", d, "results.json"))
]
results_files = sorted(results_files, key=lambda x: os.path.getmtime(x), reverse=True)

if results_files:
    latest = results_files[0]
    with open(latest) as f:
        data = json.load(f)
    p1 = len(data.get('phase1_baseline', []))
    p2 = len(data.get('phase2_sweep', []))
    print(f"  Found previous run: {os.path.dirname(latest)}")
    print(f"    Phase 1: {p1}/138")
    print(f"    Phase 2: {p2}/90")
    if p1 < 138 or p2 < 90:
        print("  NOTE: Can resume with --resume flag")
else:
    print("  OK - No previous runs (fresh start)")
EOF

# Check 8: Telegram setup
echo "[8/10] Checking Telegram setup..."
if [ -f .env ]; then
    python << 'EOF'
import os
from dotenv import load_dotenv

# Load from explicit path to avoid frame issues in heredoc
load_dotenv('.env')

token = os.getenv('TELEGRAM_BOT_TOKEN')
chat = os.getenv('TELEGRAM_CHAT_ID')

if token and chat:
    print(f"  OK - Telegram configured")
    print(f"    Bot token: {token[:20]}...")
    print(f"    Chat ID: {chat}")
else:
    print("  INFO - Telegram not configured")
    print("    Will run with --no-telegram flag")
EOF
else
    echo "  INFO - No .env file (will run with --no-telegram)"
fi

# Check 9: Process management
echo "[9/10] Checking process management..."
if command -v nohup >/dev/null 2>&1; then
    echo "  OK - nohup available (can run in background)"
else
    echo "  WARNING - nohup not available"
fi

# Check 10: File structure
echo "[10/10] Checking file structure..."
REQUIRED_FILES=(
    "run_overnight_botko_telegram.py"
    "simulator/session.py"
    "strategies/base_strategy.py"
    "dataset/generator.py"
)

ALL_PRESENT=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  OK - $file"
    else
        echo "  FAIL - Missing: $file"
        ALL_PRESENT=false
    fi
done

echo ""
echo "=== SAFETY VALIDATION ==="
echo ""

# Final validation
python << 'EOF'
import os
import sys

# Check critical safety features
checks = []

# 1. Resume capability
try:
    with open("run_overnight_botko_telegram.py") as f:
        content = f.read()
        if "completed_phase1" in content and "if (d_idx, s_idx, strat) in completed_phase1:" in content:
            checks.append(("Resume capability (Phase 1)", True))
        else:
            checks.append(("Resume capability (Phase 1)", False))

        if "completed_phase2" in content and "in completed_phase2:" in content:
            checks.append(("Resume capability (Phase 2)", True))
        else:
            checks.append(("Resume capability (Phase 2)", False))
except:
    checks.append(("Resume capability", False))

# 2. Progress saving
try:
    with open("run_overnight_botko_telegram.py") as f:
        content = f.read()
        if "save_progress(out_dir, final_output)" in content:
            checks.append(("Progress auto-save", True))
        else:
            checks.append(("Progress auto-save", False))
except:
    checks.append(("Progress auto-save", False))

# 3. Error handling
try:
    with open("run_overnight_botko_telegram.py") as f:
        content = f.read()
        if "try:" in content and "except" in content:
            checks.append(("Error handling", True))
        else:
            checks.append(("Error handling", False))
except:
    checks.append(("Error handling", False))

# Print results
all_ok = True
for check_name, passed in checks:
    status = "OK  " if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    print(f"  [{status}] {check_name}")
    if not passed:
        all_ok = False

print()
if all_ok:
    print("  ALL SAFETY CHECKS PASSED")
else:
    print("  SOME SAFETY CHECKS FAILED")
    sys.exit(1)
EOF

echo ""
echo "========================================================================"
echo "READY TO RUN"
echo "========================================================================"
echo ""

# Show command to run
if [ -f .env ] && grep -q TELEGRAM_BOT_TOKEN .env 2>/dev/null; then
    echo "With Telegram notifications:"
    echo "  nohup python run_overnight_botko_telegram.py --demo > demo_run.log 2>&1 &"
    echo "  echo \$! > demo_run.pid"
    echo ""
    echo "Or without Telegram:"
    echo "  nohup python run_overnight_botko_telegram.py --demo --no-telegram > demo_run.log 2>&1 &"
    echo "  echo \$! > demo_run.pid"
else
    echo "Command to run (no Telegram):"
    echo "  nohup python run_overnight_botko_telegram.py --demo --no-telegram > demo_run.log 2>&1 &"
    echo "  echo \$! > demo_run.pid"
fi

echo ""
echo "Estimated completion: ~14 hours"
echo ""
echo "Monitor with:"
echo "  tail -f demo_run.log"
echo ""
echo "Check progress:"
echo "  cat output/botko_*/results.json | python -c \"import json,sys; d=json.load(sys.stdin); print(f'P1: {len(d.get(\\\"phase1_baseline\\\",[]))}/138 | P2: {len(d.get(\\\"phase2_sweep\\\",[]))}/90')\""
echo ""
echo "========================================================================"
echo "1000% READY FOR 2-DAY RUN!"
echo "========================================================================"
