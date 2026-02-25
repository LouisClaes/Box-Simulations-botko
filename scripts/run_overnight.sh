#!/bin/bash
# Run overnight experiments for Box-Simulations-botko
# Designed for Raspberry Pi 4 with CPU limiting
# Usage: ./scripts/run_overnight.sh [datasets] [boxes]
# Example: ./scripts/run_overnight.sh 10 300

set -e

PROJECT_DIR="/home/louis/Box-Simulations-botko"
cd "$PROJECT_DIR"

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found at $PROJECT_DIR/venv"
    exit 1
fi

source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Telegram notifications will be disabled."
    echo "Copy .env.example to .env and configure your Telegram credentials."
fi

# Default values
DATASETS="${1:-10}"
BOXES="${2:-300}"

# Run experiment with CPU limiting
# - nice level 10: ~50% CPU priority
# - taskset -c 1-3: Use cores 1-3, leave core 0 for system
echo "======================================================"
echo "Box-Simulations-botko - Overnight Experiment Runner"
echo "======================================================"
echo "Datasets: $DATASETS"
echo "Boxes per dataset: $BOXES"
echo "Total experiments: $((DATASETS * 3)) (3 orderings per dataset)"
echo "CPU nice level: 10 (~50% priority)"
echo "CPU affinity: cores 1-3"
echo "Start time: $(date)"
echo "======================================================"
echo ""

taskset -c 1-3 nice -n 10 python -m src.runner.experiment \
    --datasets "$DATASETS" \
    --boxes "$BOXES"

echo ""
echo "======================================================"
echo "Overnight experiments completed at $(date)"
echo "Results saved to: $PROJECT_DIR/results/"
echo "======================================================"
