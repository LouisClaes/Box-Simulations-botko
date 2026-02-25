#!/bin/bash
# Check status of Box-Simulations-botko experiments

set -e

cd /home/louis/Box-Simulations-botko

echo "=== Box-Simulations-botko Status ==="
echo ""

# Check if process is running
if pgrep -f "src.runner.experiment" > /dev/null; then
    echo "✓ Experiment runner is ACTIVE"
    echo ""
    echo "Process info:"
    ps aux | grep "src.runner.experiment" | grep -v grep
else
    echo "✗ Experiment runner is NOT running"
fi

echo ""
echo "=== Recent Results ==="
if [ -d "results" ]; then
    ls -lht results/ | head -n 6
else
    echo "No results directory found"
fi

echo ""
echo "=== Disk Usage ==="
du -sh results/ 2>/dev/null || echo "No results yet"

echo ""
echo "=== System Resources ==="
echo "CPU Temperature: $(vcgencmd measure_temp 2>/dev/null || echo 'N/A')"
echo "Memory Usage:"
free -h | grep Mem

echo ""
echo "=== Latest Log Entries ==="
if [ -f "results/latest.log" ]; then
    tail -n 10 results/latest.log
else
    echo "No log file found"
fi
