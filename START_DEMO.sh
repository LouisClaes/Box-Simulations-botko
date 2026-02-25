#!/bin/bash
# Demo Run Launcher - Box-Simulations-botko
# Full test with all strategies (2-day run, ~14 hours expected)

set -e

echo "========================================================================"
echo "BOX-SIMULATIONS-BOTKO DEMO RUN"
echo "========================================================================"
echo ""

# Change to project directory
cd "$(dirname "$0")"
echo "✓ Working directory: $(pwd)"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "✗ Error: venv directory not found!"
    exit 1
fi

# Verify configuration
echo ""
echo "Verifying configuration..."
python << 'EOF'
from run_overnight_botko_telegram import EXCLUDED_STRATEGIES, SLOW_STRATEGIES
from strategies.base_strategy import STRATEGY_REGISTRY, MULTIBIN_STRATEGY_REGISTRY

included = [s for s in STRATEGY_REGISTRY.keys() if s not in EXCLUDED_STRATEGIES]
total = len(included) + len(MULTIBIN_STRATEGY_REGISTRY)

print(f"  Strategies: {total} (excluding {len(EXCLUDED_STRATEGIES)})")
print(f"  - Fast: {len([s for s in included if s not in SLOW_STRATEGIES])}")
print(f"  - Slow (run last): {len(SLOW_STRATEGIES)}")
print(f"  - Multi-bin: {len(MULTIBIN_STRATEGY_REGISTRY)}")
print(f"  Phase 1: {total * 3 * 2} experiments")
print(f"  Phase 2: 90 experiments")
print(f"  Total: {total * 3 * 2 + 90} experiments")
print(f"  Estimated time: ~14 hours")
EOF

echo ""
echo "========================================================================"
echo "READY TO RUN"
echo "========================================================================"
echo ""
echo "Starting demo run in 5 seconds..."
echo "Press Ctrl+C to cancel"
echo ""

sleep 5

# Run demo in background
nohup python run_overnight_botko_telegram.py --demo --no-telegram > demo_run.log 2>&1 &
PID=$!
echo $PID > demo_run.pid

echo "✓ Demo run started!"
echo "  PID: $PID"
echo "  Start time: $(date)"
echo "  Expected completion: $(date -d '+14 hours' 2>/dev/null || date -v+14H 2>/dev/null || echo 'in ~14 hours')"
echo ""
echo "Monitor with:"
echo "  tail -f demo_run.log"
echo ""
echo "Check progress:"
echo "  cat output/botko_*/results.json | python -c \"import json,sys; d=json.load(sys.stdin); print(f'P1: {len(d.get(\\\"phase1_baseline\\\", []))}/138 | P2: {len(d.get(\\\"phase2_sweep\\\", []))}/90')\""
echo ""
echo "Stop if needed:"
echo "  kill $PID"
echo ""
echo "========================================================================"
