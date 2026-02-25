#!/bin/bash
cd /home/louis/Box-Simulations-botko
source venv/bin/activate
nohup nice -n 10 python run_overnight_botko_telegram.py > overnight_run.log 2>&1 &
echo "Started"
