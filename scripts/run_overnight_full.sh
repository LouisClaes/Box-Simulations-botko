#!/bin/bash
# Full overnight run: 300 boxes × 10 datasets × 3 shuffles
# Uses 50% CPU, sends Telegram notifications

set -e

PROJECT_DIR="/home/louis/Box-Simulations-botko"
cd "$PROJECT_DIR"

echo "======================================================"
echo "Botko Overnight Runner - FULL PRODUCTION RUN"
echo "======================================================"
echo "Configuration:"
echo "  - 300 boxes per dataset"
echo "  - 10 datasets"
echo "  - 3 shuffle sequences per dataset"
echo "  - All strategies (~25 total)"
echo "  - 50% CPU usage (2 cores)"
echo "  - Telegram notifications enabled"
echo "======================================================"
echo ""

# Check virtualenv
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Run ./test_and_validate.sh first"
    exit 1
fi

source venv/bin/activate

# Check .env
if [ ! -f ".env" ]; then
    echo "Warning: .env not found. Telegram disabled."
    TELEGRAM_FLAG="--no-telegram"
else
    TELEGRAM_FLAG=""
fi

# Confirm with user
echo "This will run for approximately 2-4 hours."
echo "Results will be saved to: output/botko_TIMESTAMP/"
echo ""
read -p "Start overnight run? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting overnight run..."
echo "Monitor progress via:"
echo "  1. Telegram notifications"
echo "  2. tail -f output/botko_*/results.json"
echo "  3. htop (to check CPU usage)"
echo ""

# Run with nice level 10 for 50% CPU priority
nice -n 10 python run_overnight_botko_telegram.py $TELEGRAM_FLAG

echo ""
echo "======================================================"
echo "OVERNIGHT RUN COMPLETE!"
echo "======================================================"
echo ""
echo "Results saved to latest output/botko_* directory"
echo ""
