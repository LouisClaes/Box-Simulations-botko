import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_overnight_botko import run_multibin_experiment
from dataset.generator import generate_rajapack
import traceback

boxes = generate_rajapack(300, seed=42)
args = {
    "strategy_type": "multi_bin",
    "dataset_id": 0,
    "shuffle_id": 0,
    "strategy_name": "two_bounded_best_fit",
    "boxes": boxes,
    "generate_gifs": False,
}

res = run_multibin_experiment(args)
if not res["success"]:
    print("FAILED!")
    print(res["error"])
    print(res["traceback"])
else:
    print("SUCCESS!")
    print(json.dumps(res["summary"], indent=2))
