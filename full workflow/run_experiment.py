"""
Experiment runner — main entry point for the box stacking framework.

Orchestrates the full pipeline:
  1. Load or generate a dataset of boxes
  2. Initialise the robot simulator and selected strategy
  3. For each box:  strategy.decide_placement() → simulator.attempt_placement()
  4. Save structured JSON results to output/
  5. Optionally render a 3D image of the packed bin

Usage (CLI):
    python run_experiment.py --strategy baseline --generate 40 --verbose --render
    python run_experiment.py --strategy baseline --dataset dataset/test.json --render

Usage (Python):
    from run_experiment import run_experiment
    results = run_experiment(config, boxes)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

# Put the full-workflow root on the path so all packages resolve cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Box, Placement, BinConfig, ExperimentConfig
from dataset.generator import generate_uniform
from dataset.loader import load_dataset
from simulator.pipeline_simulator import PipelineSimulator
from strategies.base_strategy import get_strategy, STRATEGY_REGISTRY
from visualization.render_3d import render_packing, render_step_sequence
from visualization.step_logger import StepLogger
from visualization.gif_creator import create_stacking_gif

# Import strategies package to auto-register all strategies
import strategies  # noqa: F401
from result_manager import ResultManager


# ─────────────────────────────────────────────────────────────────────────────
# Core API
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    config: ExperimentConfig,
    boxes: Optional[List[Box]] = None,
) -> dict:
    """
    Run a full experiment: strategy × dataset → result dict.

    Data flow (clean OOP):
        strategy.decide_placement(box, bin_state)  → PlacementDecision
        simulator.attempt_placement(box, x, y, o)  → Placement | None
        simulator.record_rejection(box, reason)     (when strategy returns None)

    Args:
        config: Full experiment configuration.
        boxes:  Pre-loaded box list.  If None, loads from config.dataset_path.

    Returns:
        dict with keys: experiment, metrics, placements, step_log.
    """
    if boxes is None:
        boxes = load_dataset(config.dataset_path)

    # ── Initialise ───────────────────────────────────────────────────────
    if config.verbose:
        print("DEBUG: Initialising Simulator...")
    simulator = PipelineSimulator(config)
    if config.verbose:
        print("DEBUG: Initialising Strategy...")
    strategy = get_strategy(config.strategy_name)
    strategy.on_episode_start(config)
    logger = StepLogger(verbose=config.verbose)

    if config.verbose:
        print(f"\n  Strategy:     {config.strategy_name}")
        print(f"  Boxes:        {len(boxes)}")
        print(f"  Bin:          {config.bin.length}×{config.bin.width}×{config.bin.height}")
        print(f"  Stability:    {'ON' if config.enable_stability else 'OFF'}")
        print(f"  Orientations: {'all 6' if config.allow_all_orientations else 'flat (2)'}")
        print("-" * 65)


    # ── Run ──────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    stop_reason = None  # None = completed normally

    if config.verbose:
        print("DEBUG: Starting placement loop...")

    for box in boxes:
        if config.verbose:
            print(f"DEBUG: Processing Box {box.id}...")
        # 1. Strategy receives full 3D bin state
        bin_state = simulator.get_bin_state()

        # 2. Strategy proposes a single placement
        decision = strategy.decide_placement(box, bin_state)

        # 3a. Strategy says "this box doesn't fit" — expected, skip it
        if decision is None:
            simulator.record_rejection(box, "Strategy found no valid placement")
            latest = simulator.get_latest_step()
            if latest is not None:
                logger.log_step(latest)
            continue

        # 3b. Strategy proposed coordinates → simulator validates
        result = simulator.attempt_placement(
            box, decision.x, decision.y, decision.orientation_idx,
        )

        latest = simulator.get_latest_step()
        if latest is not None:
            logger.log_step(latest)

        # 3c. Simulator rejected the proposal → STOP (strategy bug)
        if result is None:
            reason = latest.rejection_reason if latest else "Unknown"
            stop_reason = (
                f"Simulator rejected placement for box #{box.id} "
                f"at ({decision.x:.1f}, {decision.y:.1f}, orient={decision.orientation_idx}): "
                f"{reason}"
            )
            if config.verbose:
                print(f"\n  ⛔ SIMULATION STOPPED: {stop_reason}")
            break

    computation_ms = (time.perf_counter() - t_start) * 1000
    strategy.on_episode_end(simulator.get_summary())

    summary = simulator.get_summary()
    if config.verbose:
        logger.print_summary(summary)

    result = ResultManager.build_single_run_result(
        config=config,
        summary=summary,
        placements=[p.to_dict() for p in simulator.get_bin_state().placed_boxes],
        logs=logger.get_records(),
        completed=stop_reason is None,
        stop_reason=stop_reason
    )
    result["metrics"]["computation_time_ms"] = round(computation_ms, 2)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Box Stacking Strategy Tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --strategy baseline --generate 40 --verbose --render
  python run_experiment.py --strategy baseline --dataset dataset/test.json --render
  python run_experiment.py --strategy baseline --generate 30 --stability --render -v
        """,
    )

    # Strategy
    parser.add_argument("--strategy", default="baseline",
                        help=f"Strategy name ({list(STRATEGY_REGISTRY.keys())})")

    # Dataset
    ds = parser.add_mutually_exclusive_group()
    ds.add_argument("--dataset", type=str, help="Path to dataset JSON")
    ds.add_argument("--generate", type=int, metavar="N",
                    help="Generate N uniform-random boxes")

    parser.add_argument("--gen-min", type=float, default=200.0)
    parser.add_argument("--gen-max", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=42)

    # Bin
    parser.add_argument("--bin-length", type=float, default=1200.0)
    parser.add_argument("--bin-width", type=float, default=800.0)
    parser.add_argument("--bin-height", type=float, default=2700.0)
    parser.add_argument("--resolution", type=float, default=10.0)

    # Constraints
    parser.add_argument("--stability", action="store_true")
    parser.add_argument("--min-support", type=float, default=0.8)
    parser.add_argument("--all-orientations", action="store_true")

    # Output
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-steps", action="store_true")
    parser.add_argument("--gif", action="store_true",
                        help="Create animated GIF of the stacking process")
    parser.add_argument("--gif-fps", type=int, default=2,
                        help="GIF frames per second (default: 2)")
    parser.add_argument("--verbose", "-v", action="store_true")
    # Output dir is now optional; if None, ResultManager uses standard strategies/<name>/output
    parser.add_argument("--output-dir", default=None,
                        help="Override default output directory (strategies/<name>/output)")

    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────
    bin_config = BinConfig(
        length=args.bin_length, width=args.bin_width,
        height=args.bin_height, resolution=args.resolution,
    )
    config = ExperimentConfig(
        bin=bin_config, strategy_name=args.strategy,
        enable_stability=args.stability, min_support_ratio=args.min_support,
        allow_all_orientations=args.all_orientations,
        render_3d=args.render, verbose=args.verbose,
    )

    # ── Dataset ──────────────────────────────────────────────────────────
    if args.dataset:
        print(f"\n  Loading dataset: {args.dataset}")
        boxes = load_dataset(args.dataset)
        config.dataset_path = args.dataset
    elif args.generate:
        print(f"\n  Generating {args.generate} boxes "
              f"(dims {args.gen_min}-{args.gen_max}, seed={args.seed})")
        ds_path = os.path.join("dataset", f"generated_{args.generate}_{args.seed}.json")
        boxes = generate_uniform(args.generate, args.gen_min, args.gen_max,
                                 save_path=ds_path, seed=args.seed)
        config.dataset_path = ds_path
    else:
        print("\n  No dataset specified — generating 150 boxes (seed=42)")
        ds_path = os.path.join("dataset", "generated_150_42.json")
        boxes = generate_uniform(150, 200.0, 500.0, save_path=ds_path, seed=42)
        config.dataset_path = ds_path

    # ── Run ──────────────────────────────────────────────────────────────
    result = run_experiment(config, boxes)

    # ── Save ─────────────────────────────────────────────────────────────
    # Initialize Manager with explicit output dir if provided, else standard
    manager = ResultManager(args.strategy, mode="single_bin", base_output_dir=args.output_dir)
    
    # Save using standard logic
    json_path = manager.save_json(result)
    print(f"  Results saved: {json_path}")

    # ── Render ───────────────────────────────────────────────────────────
    placements = [Placement.from_dict(p) for p in result["placements"]]

    if args.render:
        m = result["metrics"]
        img_type = f"packing_{m['boxes_placed']}of{m['boxes_total']}_{m['fill_rate']*100:.1f}pct"
        img_path = manager.get_render_path(run_id=None, type=img_type, ext="png")
        render_packing(placements, bin_config, img_path,
                       title="")
        print(f"  3D render:  {img_path}")

    if args.render_steps:
        # Steps are a directory, let's keep adjacent to the JSON
        steps_dir = json_path.replace(".json", "_steps")
        render_step_sequence(placements, bin_config, steps_dir)
        print(f"  Step renders: {steps_dir}/")

    if args.gif:
        m = result["metrics"]
        gif_type = f"stacking_{m['boxes_placed']}of{m['boxes_total']}_{m['fill_rate']*100:.1f}pct"
        gif_path = manager.get_render_path(run_id=None, type=gif_type, ext="gif")
        create_stacking_gif(
            placements, bin_config, gif_path,
            title="",
            fps=args.gif_fps,
        )
        print(f"  GIF animation: {gif_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    m = result["metrics"]
    print(f"\n  Fill rate: {m['fill_rate']:.1%}  |  "
          f"Placed: {m['boxes_placed']}/{m['boxes_total']}  |  "
          f"Time: {m['computation_time_ms']:.0f}ms\n")


if __name__ == "__main__":
    main()
