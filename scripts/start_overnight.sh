#!/bin/bash
# Start overnight Botko experiments with CPU limiting for Raspberry Pi
# Designed to use 50% CPU and run for hours/days

set -e

echo "🤖 Starting Botko Overnight Experiments"
echo "=========================================="
echo ""

# Navigate to project directory
cd /home/louis/Box-Simulations-botko

# Check virtualenv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if Telegram token is set
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "⚠️  TELEGRAM_BOT_TOKEN not set - loading from .env if available"
    if [ -f ".env" ]; then
        export $(cat .env | grep -v '^#' | xargs)
    else
        echo "⚠️  Warning: No .env file found. Telegram notifications will be disabled."
    fi
fi

# Create results directory if it doesn't exist
mkdir -p results

echo "Configuration:"
echo "  📦 Datasets: 10"
echo "  📦 Boxes per dataset: 300"
echo "  📦 Orderings per dataset: 3"
echo "  💻 CPU limit: ~50% (nice level 10)"
echo "  💬 Telegram: ${TELEGRAM_BOT_TOKEN:+Enabled}${TELEGRAM_BOT_TOKEN:-Disabled}"
echo ""
echo "Starting experiment..."
echo ""

# Run with CPU limiting
# nice -n 10 = lower priority, allows other processes to run
# This keeps CPU usage around 50% on Raspberry Pi
nice -n 10 python run_botko_overnight.py 2>&1 | tee "results/overnight_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ Experiment completed!"
echo "   Check results/ directory for output files"
