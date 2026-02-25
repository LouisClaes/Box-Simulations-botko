#!/usr/bin/env python3
"""Run single experiments for multiple strategies and render a 1xN stacking grid.

Usage:
  python python/scripts/run_and_render_strategies.py --out python/output/botko_20260223_133414/stacking_grid_5.png

This runs one experiment per strategy (generating N boxes) and renders final
packing snapshots (no conveyor) into a 1xN PNG grid using the visualization
module. Runs are executed sequentially to keep CPU usage reasonable.
"""

import argparse
import os
import sys
import random
from typing import List, Dict

# Ensure package imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + os.sep + '..')

from config import BinConfig, ExperimentConfig, Placement
from dataset.generator import generate_uniform
from run_experiment import run_experiment
from visualization.grid_creator import create_png_grid


DEFAULT_STRATEGIES = [
    "walle_scoring",
    "surface_contact",
    "best_fit_decreasing",
    "hybrid_adaptive",
    "extreme_points",
]


def run_one(strategy: str, boxes, bin_config: BinConfig) -> Dict:
    cfg = ExperimentConfig(bin=bin_config, strategy_name=strategy, render_3d=False, verbose=False)
    result = run_experiment(cfg, boxes)
    placements = [Placement.from_dict(p) for p in result["placements"]]
    m = result["metrics"]
    return {
        "placements": placements,
        "run_index": 0,
        "fill_rate": m.get("fill_rate", 0.0),
        "computation_time_ms": m.get("computation_time_ms", 0.0),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="*", default=DEFAULT_STRATEGIES)
    parser.add_argument("--generate", type=int, default=400, help="Number of boxes to generate per run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-strategy-runs", type=int, default=5,
                        help="Number of runs (and grid columns) per strategy")
    parser.add_argument("--out-dir", default=None, help="Output directory to save per-strategy grids")
    args = parser.parse_args(argv)

    bin_config = BinConfig(length=1200.0, width=800.0, height=2700.0, resolution=10.0)

    # Generate a base set of boxes once; shuffle per run to create variation
    boxes_base = generate_uniform(args.generate, 200.0, 500.0, save_path=None, seed=args.seed)

    out_dir = args.out_dir or os.path.join("python", "output", "botko_20260223_133414")
    os.makedirs(out_dir, exist_ok=True)

    for s in args.strategies:
        print(f"Running strategy: {s} ...")
        runs_data = []
        for run_idx in range(args.per_strategy_runs):
            # Shuffle deterministically per run
            rng = random.Random(args.seed + run_idx)
            shuffled = list(boxes_base)
            rng.shuffle(shuffled)

            rd = run_one(s, shuffled, bin_config)
            rd["run_index"] = run_idx
            runs_data.append(rd)

        save_path = os.path.join(out_dir, f"stacking_{s}.png")
        print(f"Rendering grid for {s} → {save_path}")
        create_png_grid(runs_data, bin_config, save_path, strategy_name=s, cols=args.per_strategy_runs)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
