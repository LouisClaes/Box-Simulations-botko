"""
Strategy tester — run each strategy 100× on the same dataset with random
box orderings, produce photo-grid PNGs + GIFs for 10 of those runs, and
save detailed JSON results under each strategy's own output folder.

Output per strategy  →  strategies/<name>/output/
  ├── results_<timestamp>.json         all 100 runs + aggregated stats
  ├── packing_grid_<timestamp>.png     2×5 grid of 10 final packings
  └── stacking_grid_<timestamp>.gif    2×5 animated grid of 10 stacking sequences

Usage:
    python strategy_tester.py --strategies baseline walle_scoring --boxes 30
    python strategy_tester.py --all-strategies --boxes 50 --seed 42
    python strategy_tester.py --strategies baseline --runs 100 --viz-count 10 -v
"""

import argparse
import json
import os
import sys
import time
import random
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# Root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, must be before other mpl imports

from config import Box, Placement, BinConfig, ExperimentConfig
from dataset.generator import generate_uniform, generate_rajapack
from dataset.loader import load_dataset
from run_experiment import run_experiment
from strategies.base_strategy import STRATEGY_REGISTRY
from visualization.grid_creator import create_png_grid, create_gif_grid
from result_manager import ResultManager

# Register all strategies
import strategies  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def shuffle_boxes(boxes: List[Box], seed: int) -> List[Box]:
    """Create a deterministic shuffled copy of the box list."""
    rng = random.Random(seed)
    shuffled = list(boxes)
    rng.shuffle(shuffled)
    return shuffled


# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy_test(
    strategy_name: str,
    boxes: List[Box],
    bin_config: BinConfig,
    n_runs: int = 100,
    n_viz: int = 10,
    base_seed: int = 42,
    verbose: bool = False,
    gif_fps: int = 2,
) -> Dict:
    """
    Run `n_runs` experiments for one strategy with shuffled box orderings.
    Render visualisations for the first `n_viz` runs only.

    Returns a dict with all run results, aggregated stats, and output paths.
    """
    # Initialize result manager
    manager = ResultManager(strategy_name)
    
    # We still need config for run_experiment, but we don't need to manually make dir
    # result_manager does that on init.

    config = ExperimentConfig(
        bin=bin_config,
        strategy_name=strategy_name,
        render_3d=False,
        verbose=False,
    )

    all_results: List[Dict] = []
    viz_data: List[Dict] = []  # first n_viz runs for grid rendering

    print(f"\n{'─'*65}")
    print(f"  Strategy: {strategy_name}  |  {n_runs} runs  |  {len(boxes)} boxes")
    print(f"{'─'*65}")

    t_strat_start = time.perf_counter()

    for run_idx in range(n_runs):
        shuffle_seed = base_seed * 10000 + run_idx
        shuffled = shuffle_boxes(boxes, shuffle_seed)

        try:
            result = run_experiment(config, shuffled)
        except Exception as e:
            print(f"  ERROR run #{run_idx}: {e}")
            continue

        m = result["metrics"]
        n_boxes = m["boxes_total"]
        sec_per_box = (m["computation_time_ms"] / 1000.0) / max(n_boxes, 1)

        run_record = {
            "run_index": run_idx,
            "shuffle_seed": shuffle_seed,
            "fill_rate": round(m["fill_rate"], 6),
            "boxes_placed": m["boxes_placed"],
            "boxes_rejected": m["boxes_rejected"],
            "boxes_total": n_boxes,
            "max_height": round(m["max_height"], 2),
            "computation_time_ms": round(m["computation_time_ms"], 2),
            "sec_per_box": round(sec_per_box, 6),
            "stability_rate": round(m["stability_rate"], 4),
            "completed": result.get("completed", True),
        }
        all_results.append(run_record)

        # Collect placement data for visualisation (first n_viz runs)
        if run_idx < n_viz:
            placements = [Placement.from_dict(p) for p in result["placements"]]
            viz_data.append({
                "placements": placements,
                "run_index": run_idx,
                "fill_rate": m["fill_rate"],
                "computation_time_ms": m["computation_time_ms"],
            })

        # Progress
        if verbose:
            print(
                f"  [{run_idx + 1:4d}/{n_runs}]  "
                f"fill={m['fill_rate']:.1%}  "
                f"placed={m['boxes_placed']}/{n_boxes}  "
                f"time={m['computation_time_ms']:.0f}ms  "
                f"sec/box={sec_per_box:.4f}"
            )
        elif (run_idx + 1) % 10 == 0 or run_idx + 1 == n_runs:
            elapsed = time.perf_counter() - t_strat_start
            eta = (elapsed / (run_idx + 1)) * (n_runs - run_idx - 1)
            print(
                f"  Progress: {run_idx + 1}/{n_runs} "
                f"({(run_idx + 1) / n_runs:.0%}) | "
                f"ETA: {eta:.0f}s"
            )

    strat_elapsed = time.perf_counter() - t_strat_start

    # ── Aggregated statistics ────────────────────────────────────────────
    if all_results:
        fill_rates = [r["fill_rate"] for r in all_results]
        times_ms = [r["computation_time_ms"] for r in all_results]
        sec_per_box_list = [r["sec_per_box"] for r in all_results]
        placed_counts = [r["boxes_placed"] for r in all_results]

        n = len(fill_rates)
        mean_fr = sum(fill_rates) / n
        std_fr = (
            (sum((x - mean_fr) ** 2 for x in fill_rates) / max(n - 1, 1)) ** 0.5
            if n > 1 else 0.0
        )

        aggregate = {
            "n_runs": n,
            "mean_fill_rate": round(mean_fr, 6),
            "std_fill_rate": round(std_fr, 6),
            "min_fill_rate": round(min(fill_rates), 6),
            "max_fill_rate": round(max(fill_rates), 6),
            "avg_sec_per_box": round(sum(sec_per_box_list) / n, 6),
            "avg_computation_time_ms": round(sum(times_ms) / n, 2),
            "avg_boxes_placed": round(sum(placed_counts) / n, 2),
            "total_elapsed_s": round(strat_elapsed, 2),
        }
    else:
        aggregate = {"n_runs": 0, "error": "No successful runs"}

    # ── Save JSON ────────────────────────────────────────────────────────
    # Construct standard aggregate result
    dataset_info = {
        "n_boxes": len(boxes),
        "seed": base_seed,
    }
    
    json_data = ResultManager.build_aggregate_result(
        strategy_name=strategy_name,
        dataset_info=dataset_info,
        bin_config=bin_config,
        aggregate_stats=aggregate,
        runs=all_results
    )
    
    json_path = manager.save_json(json_data)
    print(f"  ✅ JSON saved:  {json_path}")

    # ── Render PNG grid ──────────────────────────────────────────────────
    img_type = f"packing_grid_mean{aggregate['mean_fill_rate']*100:.1f}pct_max{aggregate['max_fill_rate']*100:.1f}pct"
    png_path = manager.get_render_path(type=img_type, ext="png")
    if viz_data:
        print(f"  🖼  Rendering PNG grid ({len(viz_data)} cells)...")
        create_png_grid(
            viz_data, bin_config, png_path,
            strategy_name=strategy_name,
        )
        print(f"  ✅ PNG grid:    {png_path}")
    else:
        png_path = ""
        print("  ⚠  No vis data — PNG grid skipped.")

    # ── Render GIF grid ──────────────────────────────────────────────────
    gif_type = f"stacking_grid_mean{aggregate['mean_fill_rate']*100:.1f}pct_max{aggregate['max_fill_rate']*100:.1f}pct"
    gif_path = manager.get_render_path(type=gif_type, ext="gif")
    if viz_data:
        print(f"  🎞  Rendering GIF grid ({len(viz_data)} cells)...")
        create_gif_grid(
            viz_data, bin_config, gif_path,
            strategy_name=strategy_name,
            fps=gif_fps,
        )
        print(f"  ✅ GIF grid:    {gif_path}")
    else:
        gif_path = ""
        print("  ⚠  No vis data — GIF grid skipped.")

    # ── Console summary ──────────────────────────────────────────────────
    if aggregate.get("n_runs", 0) > 0:
        print(f"\n  {'─'*50}")
        print(f"  {strategy_name} SUMMARY ({aggregate['n_runs']} runs)")
        print(f"  {'─'*50}")
        print(f"  Mean fill rate:    {aggregate['mean_fill_rate']:.2%} ± {aggregate['std_fill_rate']:.2%}")
        print(f"  Min / Max fill:    {aggregate['min_fill_rate']:.2%} / {aggregate['max_fill_rate']:.2%}")
        print(f"  Avg sec/box:       {aggregate['avg_sec_per_box']:.6f}")
        print(f"  Avg time (ms):     {aggregate['avg_computation_time_ms']:.1f}")
        print(f"  Avg boxes placed:  {aggregate['avg_boxes_placed']:.1f}")
        print(f"  Total elapsed:     {aggregate['total_elapsed_s']:.1f}s")
        print(f"  {'─'*50}")

    return {
        "strategy": strategy_name,
        "aggregate": aggregate,
        "json_path": json_path,
        "png_path": png_path,
        "gif_path": gif_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Final comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(all_strategy_results: List[Dict]) -> None:
    """Print a ranked comparison table of all tested strategies."""
    # Sort by mean fill rate descending
    valid = [r for r in all_strategy_results if r["aggregate"].get("n_runs", 0) > 0]
    sorted_results = sorted(
        valid, key=lambda r: r["aggregate"]["mean_fill_rate"], reverse=True,
    )

    print(f"\n{'='*95}")
    print(f"  STRATEGY COMPARISON — {len(sorted_results)} strategies tested")
    print(f"{'='*95}")
    print(
        f"  {'#':<4s} {'Strategy':<25s} {'Runs':>5s} "
        f"{'Mean Fill':>10s} {'±Std':>7s} "
        f"{'Min':>7s} {'Max':>7s} "
        f"{'Sec/Box':>10s} {'Placed':>7s}"
    )
    print(f"  {'-'*89}")

    for rank, r in enumerate(sorted_results, 1):
        a = r["aggregate"]
        print(
            f"  {rank:<4d} {r['strategy']:<25s} {a['n_runs']:>5d} "
            f"{a['mean_fill_rate']:>9.2%} {a['std_fill_rate']:>6.2%} "
            f"{a['min_fill_rate']:>6.2%} {a['max_fill_rate']:>6.2%} "
            f"{a['avg_sec_per_box']:>9.6f} {a['avg_boxes_placed']:>6.1f}"
        )

    print(f"  {'-'*89}")
    if sorted_results:
        best = sorted_results[0]
        print(
            f"\n  🏆 BEST: {best['strategy']} — "
            f"{best['aggregate']['mean_fill_rate']:.2%} mean fill rate  |  "
            f"{best['aggregate']['avg_sec_per_box']:.6f} sec/box"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Strategy Tester — 100 shuffled runs per strategy with photo grids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python strategy_tester.py --strategies baseline --boxes 30
  python strategy_tester.py --strategies baseline walle_scoring --boxes 50 --seed 42
  python strategy_tester.py --all-strategies --boxes 30 -v
  python strategy_tester.py --strategies baseline --runs 20 --viz-count 5
        """,
    )

    # Strategy selection
    sg = parser.add_mutually_exclusive_group()
    sg.add_argument("--strategies", nargs="+",
                    help="List of strategy names to test")
    sg.add_argument("--all-strategies", action="store_true",
                    help="Test all registered strategies")

    # Dataset
    parser.add_argument("--dataset", type=str,
                        help="Path to an existing dataset JSON (overrides --generate)")
    parser.add_argument("--boxes", type=int, default=150,
                        help="Number of boxes to generate (default: 150)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Dataset generation seed (default: 42)")
    parser.add_argument("--gen-min", type=float, default=200.0,
                        help="Min box dimension (default: 200.0)")
    parser.add_argument("--gen-max", type=float, default=500.0,
                        help="Max box dimension (default: 500.0)")
    parser.add_argument("--generator", type=str, default="uniform",
                        choices=["uniform", "rajapack"],
                        help="Box generator method (uniform or rajapack)")

    # Run parameters
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of shuffled runs per strategy (default: 100)")
    parser.add_argument("--viz-count", type=int, default=10,
                        help="How many runs to visualize in the grid (default: 10)")

    # Bin
    parser.add_argument("--bin-length", type=float, default=1200.0)
    parser.add_argument("--bin-width", type=float, default=800.0)
    parser.add_argument("--bin-height", type=float, default=2700.0)
    parser.add_argument("--resolution", type=float, default=10.0)

    # GIF
    parser.add_argument("--gif-fps", type=int, default=2,
                        help="GIF frames per second (default: 2)")

    # Output
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # ── Resolve strategies ───────────────────────────────────────────────
    if args.all_strategies:
        strategy_list = list(STRATEGY_REGISTRY.keys())
    elif args.strategies:
        strategy_list = args.strategies
    else:
        strategy_list = list(STRATEGY_REGISTRY.keys())

    # Validate strategy names
    for s in strategy_list:
        if s not in STRATEGY_REGISTRY:
            print(f"  ERROR: Strategy '{s}' not registered. "
                  f"Available: {sorted(STRATEGY_REGISTRY.keys())}")
            sys.exit(1)

    # ── Bin config ───────────────────────────────────────────────────────
    bin_config = BinConfig(
        length=args.bin_length, width=args.bin_width,
        height=args.bin_height, resolution=args.resolution,
    )

    # ── Dataset ──────────────────────────────────────────────────────────
    if args.dataset:
        print(f"\n  Loading dataset: {args.dataset}")
        boxes = load_dataset(args.dataset)
    else:
        print(f"\n  Generating {args.boxes} boxes using {args.generator} "
              f"(dims {args.gen_min}-{args.gen_max}, seed={args.seed})")
        
        if args.generator == "rajapack":
            boxes = generate_rajapack(
                args.boxes, min_dim=args.gen_min, max_dim=args.gen_max,
                seed=args.seed,
            )
        else:
            boxes = generate_uniform(
                args.boxes, min_dim=args.gen_min, max_dim=args.gen_max,
                seed=args.seed,
            )

    # ── Banner ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  STRATEGY TESTER")
    print(f"  Strategies:  {len(strategy_list)} — {', '.join(strategy_list)}")
    print(f"  Boxes:       {len(boxes)}")
    print(f"  Runs/strat:  {args.runs}")
    print(f"  Visualize:   {args.viz_count} of {args.runs} runs")
    print(f"  Bin:         {bin_config.length}×{bin_config.width}×{bin_config.height}")
    print(f"  Seed:        {args.seed}")
    print(f"{'='*65}")

    # ── Run all strategies ───────────────────────────────────────────────
    all_strategy_results: List[Dict] = []
    t_total = time.perf_counter()

    for strat_name in strategy_list:
        result = run_strategy_test(
            strategy_name=strat_name,
            boxes=boxes,
            bin_config=bin_config,
            n_runs=args.runs,
            n_viz=args.viz_count,
            base_seed=args.seed,
            verbose=args.verbose,
            gif_fps=args.gif_fps,
        )
        all_strategy_results.append(result)

    total_elapsed = time.perf_counter() - t_total

    # ── Final comparison ─────────────────────────────────────────────────
    print_comparison_table(all_strategy_results)
    print(f"  Total time: {total_elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
