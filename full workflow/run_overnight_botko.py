"""
Botko BV Overnight Sweep — thesis-quality benchmark for dual-pallet robotic palletizer.

Models the Botko BV robot setup using the simulator.session.PackingSession:
  - 2 EUR pallets (1200x800mm), close at 1800mm height
  - FIFO conveyor buffer of 8 boxes with pick window of 4
  - Pluggable close policy, box selectors, bin selectors
  - Only closed pallets count in primary metrics

Phases:
  1. Baseline: all strategies × all datasets × all shuffles (default selectors)
  2. Parameter sweep: top-5 strategies × box selectors × bin selectors
  3. GIF generation: top-3 per dataset for visualization
"""

import argparse
import json
import os
import sys
import time
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Box, BinConfig
from dataset.generator import generate_rajapack
from strategies.base_strategy import (
    STRATEGY_REGISTRY, MULTIBIN_STRATEGY_REGISTRY,
    get_strategy, get_multibin_strategy,
)
from simulator.session import (
    PackingSession, SessionConfig,
    get_box_selector, get_bin_selector,
)
from simulator.close_policy import HeightClosePolicy

# ─────────────────────────────────────────────────────────────────────────────
# Botko BV constants
# ─────────────────────────────────────────────────────────────────────────────

BOTKO_BINS = 2
BOTKO_BUFFER_SIZE = 8
BOTKO_PICK_WINDOW = 4
BOTKO_PALLET = BinConfig(
    length=1200.0, width=800.0, height=2700.0, resolution=10.0,
)
MAX_CONSECUTIVE_REJECTS = 10
PALLET_CLOSE_HEIGHT = 1800.0

BOTKO_SESSION_CONFIG = SessionConfig(
    bin_config=BOTKO_PALLET,
    num_bins=BOTKO_BINS,
    buffer_size=BOTKO_BUFFER_SIZE,
    pick_window=BOTKO_PICK_WINDOW,
    close_policy=HeightClosePolicy(max_height=PALLET_CLOSE_HEIGHT),
    max_consecutive_rejects=MAX_CONSECUTIVE_REJECTS,
    enable_stability=False,
    allow_all_orientations=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# Single-bin strategy experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_singlebin_experiment(args: dict) -> dict:
    try:
        boxes = args["boxes"]
        strategy_name = args["strategy_name"]
        box_selector_name = args.get("box_selector", "default")
        bin_selector_name = args.get("bin_selector", "emptiest_first")
        generate_gifs = args.get("generate_gifs", False)

        strategy = get_strategy(strategy_name)
        box_selector = get_box_selector(box_selector_name)
        bin_selector = get_bin_selector(bin_selector_name)

        session = PackingSession(BOTKO_SESSION_CONFIG)

        # GIF recording callback
        conveyor_steps = []
        if generate_gifs:
            def on_step(step_num, step_result, obs):
                try:
                    from visualization.conveyor_gif_creator import ConveyorStep
                    p0 = list(session.stations[0].bin_state.placed_boxes)
                    p1 = list(session.stations[1].bin_state.placed_boxes)
                    box = step_result.box
                    if box is not None:
                        cs = ConveyorStep(
                            step=step_num, box=box,
                            bin_index=step_result.bin_index if step_result.placed else 0,
                            placed=step_result.placed,
                            placement=step_result.placement if step_result.placed else None,
                            buffer_snapshot=list(obs.buffer_view),
                            stream_remaining=obs.stream_remaining,
                            pallet0_placements=p0, pallet1_placements=p1,
                        )
                        conveyor_steps.append(cs)
                except Exception:
                    pass
        else:
            on_step = None

        result = session.run(boxes, strategy, box_selector, bin_selector, on_step=on_step)

        summary = result.to_dict()
        summary["strategy"] = strategy_name
        summary["strategy_type"] = "single_bin"
        summary["box_selector"] = box_selector_name
        summary["bin_selector"] = bin_selector_name

        return {
            "success": True,
            "summary": summary,
            "args_echo": {k: v for k, v in args.items() if k != "boxes"},
            "conveyor_steps": conveyor_steps if generate_gifs else None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "args_echo": {k: v for k, v in args.items() if k != "boxes"},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-bin strategy experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_multibin_experiment(args: dict) -> dict:
    try:
        boxes = args["boxes"]
        strategy_name = args["strategy_name"]
        generate_gifs = args.get("generate_gifs", False)

        strategy = get_multibin_strategy(strategy_name)

        session = PackingSession(BOTKO_SESSION_CONFIG)

        conveyor_steps = []
        if generate_gifs:
            def on_step(step_num, step_result, obs):
                try:
                    from visualization.conveyor_gif_creator import ConveyorStep
                    p0 = list(session.stations[0].bin_state.placed_boxes)
                    p1 = list(session.stations[1].bin_state.placed_boxes)
                    box = step_result.box
                    if box is not None:
                        cs = ConveyorStep(
                            step=step_num, box=box,
                            bin_index=step_result.bin_index if step_result.placed else 0,
                            placed=step_result.placed,
                            placement=step_result.placement if step_result.placed else None,
                            buffer_snapshot=list(obs.buffer_view),
                            stream_remaining=obs.stream_remaining,
                            pallet0_placements=p0, pallet1_placements=p1,
                        )
                        conveyor_steps.append(cs)
                except Exception:
                    pass
        else:
            on_step = None

        result = session.run(boxes, strategy, on_step=on_step)

        summary = result.to_dict()
        summary["strategy"] = strategy_name
        summary["strategy_type"] = "multi_bin"
        summary["box_selector"] = "native"
        summary["bin_selector"] = "native"

        return {
            "success": True,
            "summary": summary,
            "args_echo": {k: v for k, v in args.items() if k != "boxes"},
            "conveyor_steps": conveyor_steps if generate_gifs else None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "args_echo": {k: v for k, v in args.items() if k != "boxes"},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Process entry points
# ─────────────────────────────────────────────────────────────────────────────

def process_chunk(kwargs):
    if kwargs.get("strategy_type") == "multi_bin":
        return run_multibin_experiment(kwargs)
    return run_singlebin_experiment(kwargs)


def save_progress(out_dir, final_output):
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(final_output, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Botko BV Overnight Sweep")
    parser.add_argument("--smoke-test", action="store_true", help="Run a small quick test")
    parser.add_argument("--resume", type=str, help="Path to a previous results.json to resume from")
    args = parser.parse_args()

    smoketest = args.smoke_test
    import warnings
    warnings.filterwarnings("ignore")

    # ── Param setup ────────────────────────────────────────────────────────
    n_datasets = 1 if smoketest else 10
    n_shuffles = 1 if smoketest else 3
    n_boxes = 20 if smoketest else 300

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", f"botko_{timestamp}")

    final_output = {
        "metadata": {
            "timestamp": timestamp,
            "smoke_test": smoketest,
            "n_datasets": n_datasets,
            "n_shuffles": n_shuffles,
            "n_boxes": n_boxes,
            "botko_config": BOTKO_SESSION_CONFIG.to_dict(),
            "top_5": [],
        },
        "phase1_baseline": [],
        "phase2_sweep": [],
    }

    if args.resume:
        with open(args.resume, 'r') as f:
            final_output = json.load(f)
        out_dir = os.path.dirname(os.path.abspath(args.resume))
        print(f"\n[RESUME] Resuming from {args.resume}")
        smoketest = final_output["metadata"].get("smoke_test", smoketest)
        n_datasets = final_output["metadata"].get("n_datasets", n_datasets)
        n_shuffles = final_output["metadata"].get("n_shuffles", n_shuffles)
        n_boxes = final_output["metadata"].get("n_boxes", n_boxes)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "gifs"), exist_ok=True)

    num_cpus = max(1, int(multiprocessing.cpu_count() * 0.70))

    # ── Strategy lists ─────────────────────────────────────────────────────
    if smoketest:
        singlebin_strategies = ["surface_contact", "walle_scoring", "baseline"]
        multibin_strategies = list(MULTIBIN_STRATEGY_REGISTRY.keys())[:1]
    else:
        singlebin_strategies = list(STRATEGY_REGISTRY.keys())
        multibin_strategies = list(MULTIBIN_STRATEGY_REGISTRY.keys())

    all_strategy_names = singlebin_strategies + multibin_strategies

    print(f"\n{'='*75}")
    print(f"  BOTKO BV OVERNIGHT SWEEP -- {'SMOKE TEST' if smoketest else 'FULL RUN'}")
    print(f"{'='*75}")
    print(f"  CPUs:        {num_cpus} ({70}%)")
    print(f"  Output:      {out_dir}")
    print(f"  Datasets:    {n_datasets} x {n_boxes} Rajapack boxes, {n_shuffles} shuffles")
    print(f"  Single-bin:  {len(singlebin_strategies)} strategies")
    print(f"  Multi-bin:   {len(multibin_strategies)} strategies ({', '.join(multibin_strategies)})")
    print(f"  Pallet:      {BOTKO_PALLET.length:.0f}x{BOTKO_PALLET.width:.0f}x{BOTKO_PALLET.height:.0f}mm")
    print(f"  Buffer:      {BOTKO_BUFFER_SIZE} boxes, pick window {BOTKO_PICK_WINDOW}")
    print(f"  Close:       {BOTKO_SESSION_CONFIG.close_policy.describe()}")
    print(f"  Reject:      FIFO advance, front box exits (max {MAX_CONSECUTIVE_REJECTS} consecutive)")
    print(f"  Stats:       only closed pallets count in avg fill rate")

    # ── Generate datasets ──────────────────────────────────────────────────
    print(f"\n  Generating {n_datasets} datasets...")
    datasets = []
    for d in range(n_datasets):
        ds_boxes = generate_rajapack(n_boxes, seed=42 + d)
        shuffles = []
        for s in range(n_shuffles):
            shuffled = list(ds_boxes)
            random.Random(1000 * d + s + 100).shuffle(shuffled)
            shuffles.append(shuffled)
        datasets.append({
            "dataset_id": d,
            "shuffles": shuffles,
            "original_boxes": ds_boxes,
        })

    # ── Phase 1: Baseline ──────────────────────────────────────────────────
    completed_phase1 = set()
    for r in final_output.get("phase1_baseline", []):
        completed_phase1.add((r["dataset_id"], r["shuffle_id"], r["strategy"]))

    print(f"\n[PHASE 1] Baseline: all strategies with default selectors")
    tasks = []
    for d_idx, ds in enumerate(datasets):
        for s_idx, boxes_shuffled in enumerate(ds["shuffles"]):
            for strat in singlebin_strategies:
                if (d_idx, s_idx, strat) in completed_phase1:
                    continue
                tasks.append({
                    "strategy_type": "single_bin",
                    "dataset_id": d_idx,
                    "shuffle_id": s_idx,
                    "strategy_name": strat,
                    "box_selector": "default",
                    "bin_selector": "emptiest_first",
                    "boxes": boxes_shuffled,
                    "generate_gifs": False,
                })
            for strat in multibin_strategies:
                if (d_idx, s_idx, strat) in completed_phase1:
                    continue
                tasks.append({
                    "strategy_type": "multi_bin",
                    "dataset_id": d_idx,
                    "shuffle_id": s_idx,
                    "strategy_name": strat,
                    "boxes": boxes_shuffled,
                    "generate_gifs": False,
                })

    print(f"  Enqueuing {len(tasks)} tasks...")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = {executor.submit(process_chunk, t): t for t in tasks}
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            if res["success"]:
                entry = res["summary"].copy()
                entry["dataset_id"] = res["args_echo"]["dataset_id"]
                entry["shuffle_id"] = res["args_echo"]["shuffle_id"]
                final_output["phase1_baseline"].append(entry)
            else:
                strat_name = res["args_echo"].get("strategy_name", "?")
                print(f"    FAIL [{strat_name}]: {res.get('error', '?')[:80]}")

            completed += 1
            if len(tasks) > 0 and (completed % max(1, len(tasks) // 20) == 0 or completed == len(tasks)):
                print(f"    {completed}/{len(tasks)} ({completed / len(tasks):.0%})")
                save_progress(out_dir, final_output)

    elapsed1 = time.perf_counter() - t0
    print(f"  Phase 1 completed in {elapsed1:.1f}s.")
    save_progress(out_dir, final_output)

    # ── Aggregate Phase 1 to find top-5 ────────────────────────────────────
    strat_scores: Dict[str, List[float]] = {s: [] for s in all_strategy_names}
    strat_pallets: Dict[str, List[int]] = {s: [] for s in all_strategy_names}
    for r in final_output["phase1_baseline"]:
        sname = r["strategy"]
        if sname in strat_scores:
            strat_scores[sname].append(r.get("avg_closed_fill", 0.0))
            strat_pallets[sname].append(r.get("pallets_closed", 0))

    avg_scores = []
    for s, scores in strat_scores.items():
        if scores:
            avg_scores.append((
                s,
                sum(scores) / len(scores),
                len(scores),
                sum(strat_pallets.get(s, [0])),
            ))

    avg_scores.sort(key=lambda x: x[1], reverse=True)
    top_5_strategies = [x[0] for x in avg_scores[:5]]

    if smoketest:
        top_5_strategies = top_5_strategies[:2]

    print(f"\n  Phase 1 Rankings (Avg Closed-Pallet Fill):")
    print(f"  {'RANK':<5} {'STRATEGY':<32} {'FILL':>7} {'RUNS':>5} {'PALS':>5} {'TYPE':>10}")
    print(f"  {'-'*5} {'-'*32} {'-'*7} {'-'*5} {'-'*5} {'-'*10}")
    for i, (s, score, n, total_pals) in enumerate(avg_scores, 1):
        stype = "multi-bin" if s in multibin_strategies else "single-bin"
        marker = " <--" if s in top_5_strategies else ""
        print(f"  {i:<5} {s:<32} {score:>6.2%} {n:>5} {total_pals:>5} {stype:>10}{marker}")

    # ── Phase 1.5: GIF generation ──────────────────────────────────────────
    print(f"\n[PHASE 1.5] Generating GIFs for top-3 strategies per dataset...")
    gif_tasks = []
    for d_idx in range(n_datasets):
        ds_scores: Dict[str, List[float]] = {s: [] for s in all_strategy_names}
        for r_p1 in final_output["phase1_baseline"]:
            if r_p1["dataset_id"] == d_idx and r_p1["strategy"] in ds_scores:
                ds_scores[r_p1["strategy"]].append(r_p1.get("avg_closed_fill", 0.0))

        ds_avg = [(s, sum(sc) / len(sc) if sc else 0) for s, sc in ds_scores.items()]
        ds_avg.sort(key=lambda x: x[1], reverse=True)
        top_3_here = [x[0] for x in ds_avg[:3]]

        for strat in top_3_here:
            is_multibin = strat in multibin_strategies
            task = {
                "strategy_type": "multi_bin" if is_multibin else "single_bin",
                "dataset_id": d_idx,
                "shuffle_id": 0,
                "strategy_name": strat,
                "boxes": datasets[d_idx]["shuffles"][0],
                "generate_gifs": True,
            }
            if not is_multibin:
                task["box_selector"] = "default"
                task["bin_selector"] = "emptiest_first"
            gif_tasks.append(task)

    dataset_conveyor_steps: Dict[int, list] = {d: [] for d in range(n_datasets)}
    with ProcessPoolExecutor(max_workers=max(1, num_cpus // 2)) as executor:
        futures = {executor.submit(process_chunk, t): t for t in gif_tasks}
        for future in as_completed(futures):
            res = future.result()
            if res["success"]:
                d_id = res["args_echo"]["dataset_id"]
                s_name = res["args_echo"]["strategy_name"]
                dataset_conveyor_steps[d_id].append({
                    "label": s_name,
                    "steps": res["conveyor_steps"],
                })

    try:
        from visualization.conveyor_gif_creator import create_conveyor_grid_gif
        for d_id, exps in dataset_conveyor_steps.items():
            if exps:
                grid_path = os.path.join(out_dir, "gifs", f"dataset_{d_id:02d}_grid.gif")
                create_conveyor_grid_gif(
                    exps, BOTKO_PALLET, grid_path,
                    grid_cols=1, title=f"Dataset {d_id} Top 3", fps=4,
                )
                print(f"    Saved GIF: {grid_path}")
    except Exception as e:
        print(f"  Warning: Could not create grid GIFs: {e}")

    # ── Phase 2: Parameter sweep on top-5 ──────────────────────────────────
    box_selectors = ["default", "biggest_volume_first", "biggest_footprint_first"]
    bin_selectors = ["emptiest_first", "focus_fill", "flattest_first"]

    if smoketest:
        box_selectors = ["default", "biggest_volume_first"]
        bin_selectors = ["emptiest_first"]

    runs_per_sweep = 2 if smoketest else 8

    sweep_strategies = [s for s in top_5_strategies if s in singlebin_strategies]

    print(f"\n[PHASE 2] Parameter sweep on top-{len(sweep_strategies)} single-bin strategies")
    print(f"  Box selectors: {box_selectors}")
    print(f"  Bin selectors: {bin_selectors}")
    if any(s in multibin_strategies for s in top_5_strategies):
        print(f"  Note: multi-bin strategies use native routing (no sweep)")

    completed_phase2 = set()
    for r in final_output.get("phase2_sweep", []):
        completed_phase2.add((
            r["dataset_id"], r["shuffle_id"], r["strategy"],
            r["box_selector"], r["bin_selector"],
        ))

    phase2_tasks = []
    rng = random.Random(42)
    all_variations = [(d, s) for d in range(n_datasets) for s in range(n_shuffles)]
    selected_variations = rng.sample(all_variations, min(runs_per_sweep, len(all_variations)))

    for strat in sweep_strategies:
        for box_sel in box_selectors:
            for bin_sel in bin_selectors:
                for d_idx, s_idx in selected_variations:
                    if (d_idx, s_idx, strat, box_sel, bin_sel) in completed_phase2:
                        continue
                    phase2_tasks.append({
                        "strategy_type": "single_bin",
                        "dataset_id": d_idx,
                        "shuffle_id": s_idx,
                        "strategy_name": strat,
                        "box_selector": box_sel,
                        "bin_selector": bin_sel,
                        "boxes": datasets[d_idx]["shuffles"][s_idx],
                        "generate_gifs": False,
                    })

    print(f"  Enqueuing {len(phase2_tasks)} tasks...")
    final_output["metadata"]["top_5"] = top_5_strategies
    save_progress(out_dir, final_output)

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = {executor.submit(process_chunk, t): t for t in phase2_tasks}
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            if res["success"]:
                entry = res["summary"].copy()
                entry["dataset_id"] = res["args_echo"]["dataset_id"]
                entry["shuffle_id"] = res["args_echo"]["shuffle_id"]
                final_output["phase2_sweep"].append(entry)
            else:
                print(f"    FAIL: {res.get('error', '?')[:80]}")

            completed += 1
            if len(phase2_tasks) > 0 and (completed % max(1, len(phase2_tasks) // 10) == 0 or completed == len(phase2_tasks)):
                print(f"    {completed}/{len(phase2_tasks)} ({completed / len(phase2_tasks):.0%})")
                save_progress(out_dir, final_output)

    save_progress(out_dir, final_output)
    print(f"\nDone! Results saved to {os.path.join(out_dir, 'results.json')}")
    print(f"Run: python analyze_botko_results.py --input {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
