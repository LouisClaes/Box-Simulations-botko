"""
Botko BV Overnight Sweep with Telegram notifications - Raspberry Pi optimized.

Enhanced version of run_overnight_botko.py with:
- Telegram progress notifications
- 50% CPU usage limit for Raspberry Pi
- Resume capability
- Fair comparison (same datasets for all strategies via fixed seeds)
"""

import argparse
import json
import os
import sys
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
import asyncio
try:
    import resource  # Unix-only (available on Raspberry Pi/Linux)
except ImportError:
    resource = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_env_file_if_present() -> None:
    """
    Load KEY=VALUE pairs from local .env into os.environ (non-destructive).

    This is especially useful on Raspberry Pi where runs are often started
    as services/cron jobs without exported shell env vars.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file_if_present()

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
from simulator.close_policy_custom import FullestOnConsecutiveRejectsPolicy

# Import Telegram notifier
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from monitoring.telegram_notifier import send_telegram

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
PALLET_CLOSE_AFTER_REJECTS = 4  # Close pallet after 4 consecutive boxes can't fit

BOTKO_SESSION_CONFIG = SessionConfig(
    bin_config=BOTKO_PALLET,
    num_bins=BOTKO_BINS,
    buffer_size=BOTKO_BUFFER_SIZE,
    pick_window=BOTKO_PICK_WINDOW,
    close_policy=FullestOnConsecutiveRejectsPolicy(
        max_consecutive=PALLET_CLOSE_AFTER_REJECTS,
        min_fill_to_close=0.5  # Only close pallets that are at least 50% full
    ),
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


async def send_telegram_async(message: str):
    """Send Telegram message (async wrapper)."""
    try:
        await send_telegram(message)
    except Exception:
        pass  # Non-critical, silently fail


def send_telegram_sync(message: str):
    """Send Telegram message (sync wrapper for use in main)."""
    try:
        asyncio.run(send_telegram_async(message))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Botko BV Overnight Sweep with Telegram")
    parser.add_argument("--smoke-test", action="store_true", help="Run a small quick test (2 min)")
    parser.add_argument("--demo", action="store_true", help="Demo run for proof-of-concept (3 datasets, ~1 day)")
    parser.add_argument("--quick", action="store_true", help="Quick run (5 datasets, ~2.5 days)")
    parser.add_argument("--datasets", type=int, help="Override number of Rajapack datasets")
    parser.add_argument("--shuffles", type=int, help="Override number of shuffled sequences per dataset")
    parser.add_argument("--boxes", type=int, help="Override number of boxes per dataset")
    parser.add_argument("--phase1-only", action="store_true", help="Run only Phase 1 baseline (skip Phase 2)")
    parser.add_argument("--resume", type=str, help="Path to a previous results.json to resume from")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram notifications")
    args = parser.parse_args()

    smoketest = args.smoke_test
    demorun = args.demo
    quickrun = args.quick
    phase1_only = args.phase1_only
    use_telegram = not args.no_telegram

    import warnings
    warnings.filterwarnings("ignore")

    # ── Param setup ────────────────────────────────────────────────────────
    if smoketest:
        n_datasets = 1
        n_shuffles = 1
        n_boxes = 20
    elif demorun:
        n_datasets = 3  # Demo/proof-of-concept: ~22 hours
        n_shuffles = 2
        n_boxes = 400
    elif quickrun:
        n_datasets = 5  # 2.5 days: ~49 hours
        n_shuffles = 3
        n_boxes = 400
    else:
        n_datasets = 10  # Full run: ~99 hours (4+ days)
        n_shuffles = 3
        n_boxes = 400

    # Explicit overrides (highest priority).
    if args.datasets is not None:
        n_datasets = args.datasets
    if args.shuffles is not None:
        n_shuffles = args.shuffles
    if args.boxes is not None:
        n_boxes = args.boxes

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", f"botko_{timestamp}")

    final_output = {
        "metadata": {
            "timestamp": timestamp,
            "smoke_test": smoketest,
            "n_datasets": n_datasets,
            "n_shuffles": n_shuffles,
            "n_boxes": n_boxes,
            "phase1_only": phase1_only,
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
        phase1_only = final_output["metadata"].get("phase1_only", phase1_only)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "gifs"), exist_ok=True)

    # RASPBERRY PI: Use 50% of CPUs (2 cores on Pi 4)
    num_cpus = max(1, int(multiprocessing.cpu_count() * 0.50))

    # Set process priority to be nice to other processes
    # nice value: 10 = lower priority, yields CPU when others need it
    try:
        os.nice(10)
        print(f"  Process nice level set to 10 (background priority)")
    except (OSError, AttributeError) as e:
        print(f"  Warning: Could not set nice level: {e}")

    # Limit CPU time per worker (soft throttling)
    # This makes workers yield CPU more readily to other processes
    if resource is not None:
        try:
            # Get current limits
            soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
            # Don't set hard limit, just informational
            print(f"  CPU limits: soft={soft}, hard={hard}")
        except (ValueError, AttributeError):
            pass  # Not all systems support this

    # ── Strategy lists ─────────────────────────────────────────────────────
    # Exclude only strategies that need training
    EXCLUDED_STRATEGIES = [
        "selective_hyper_heuristic",    # Learning-based - needs training
    ]

    # Slow strategies - run LAST (11-12 min per experiment, acceptable for 2-day run)
    SLOW_STRATEGIES = [
        "lookahead",                    # Tree search - 11.6 min/experiment
        "hybrid_adaptive",              # Adaptive - 10.2 min/experiment
    ]

    if smoketest:
        singlebin_strategies = ["surface_contact", "walle_scoring", "baseline"]
        multibin_strategies = list(MULTIBIN_STRATEGY_REGISTRY.keys())[:1]
    else:
        # Get all strategies except excluded
        all_single = [s for s in STRATEGY_REGISTRY.keys() if s not in EXCLUDED_STRATEGIES]

        # Separate fast and slow strategies
        fast_strategies = [s for s in all_single if s not in SLOW_STRATEGIES]
        slow_strategies = [s for s in all_single if s in SLOW_STRATEGIES]

        # Fast strategies first, slow strategies last
        singlebin_strategies = fast_strategies + slow_strategies
        multibin_strategies = list(MULTIBIN_STRATEGY_REGISTRY.keys())

    all_strategy_names = singlebin_strategies + multibin_strategies

    print(f"\n{'='*75}")
    print(f"  BOTKO BV OVERNIGHT SWEEP -- {'SMOKE TEST' if smoketest else 'FULL RUN'}")
    print(f"{'='*75}")
    print(f"  CPUs:        {num_cpus} ({50}% for Raspberry Pi)")
    print(f"  Output:      {out_dir}")
    print(f"  Datasets:    {n_datasets} x {n_boxes} Rajapack boxes, {n_shuffles} shuffles")
    print(f"  Single-bin:  {len(singlebin_strategies)} strategies")
    print(f"  Multi-bin:   {len(multibin_strategies)} strategies ({', '.join(multibin_strategies)})")
    print(f"  Pallet:      {BOTKO_PALLET.length:.0f}x{BOTKO_PALLET.width:.0f}x{BOTKO_PALLET.height:.0f}mm")
    print(f"  Buffer:      {BOTKO_BUFFER_SIZE} boxes, pick window {BOTKO_PICK_WINDOW}")
    print(f"  Close:       {BOTKO_SESSION_CONFIG.close_policy.describe()}")
    print(f"  Reject:      FIFO advance, front box exits (max {MAX_CONSECUTIVE_REJECTS} consecutive)")
    print(f"  Stats:       only closed pallets count in avg fill rate")
    print(f"  Phase mode:  {'PHASE 1 ONLY' if phase1_only else 'PHASE 1 + PHASE 2'}")
    print(f"  Telegram:    {'ENABLED' if use_telegram else 'DISABLED'}")

    # Telegram: Experiment start
    if use_telegram and not args.resume:
        msg = (
            f"🚀 Botko Overnight Sweep Started\n"
            f"Mode: {'Smoke Test' if smoketest else 'Full Run'}\n"
            f"Phase mode: {'Phase 1 only' if phase1_only else 'Phase 1 + Phase 2'}\n"
            f"Datasets: {n_datasets} × {n_boxes} boxes × {n_shuffles} shuffles\n"
            f"Strategies: {len(all_strategy_names)} total\n"
            f"CPUs: {num_cpus}/4 (50%)\n"
            f"Output: {os.path.basename(out_dir)}"
        )
        send_telegram_sync(msg)

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
    if use_telegram:
        fast_count = len([s for s in singlebin_strategies if s not in SLOW_STRATEGIES])
        slow_count = len([s for s in singlebin_strategies if s in SLOW_STRATEGIES])
        total_experiments = (len(singlebin_strategies) + len(multibin_strategies)) * n_datasets * n_shuffles
        send_telegram_sync(
            f"🚀 DEMO RUN STARTING\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Phase 1: Baseline Testing\n"
            f"  Fast strategies: {fast_count}\n"
            f"  Slow strategies: {slow_count} (run last)\n"
            f"  Multi-bin: {len(multibin_strategies)}\n"
            f"  Datasets: {n_datasets}\n"
            f"  Shuffles per dataset: {n_shuffles}\n"
            f"  Total experiments: {total_experiments}\n"
            f"\n⏱ Estimated time: ~8-9 hours\n"
            f"📈 Updates every 5%"
        )

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
        last_telegram_pct = 0
        slow_zone_notified = False  # Track if we've notified about slow strategies

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
            current_pct = int((completed / len(tasks)) * 100)
            elapsed = time.perf_counter() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            remaining_tasks = len(tasks) - completed
            eta_seconds = remaining_tasks / rate if rate > 0 else 0

            # Save progress every ~5%
            if len(tasks) > 0 and (completed % max(1, len(tasks) // 20) == 0 or completed == len(tasks)):
                print(f"    {completed}/{len(tasks)} ({current_pct}%) - ETA: {eta_seconds/60:.1f}m")
                save_progress(out_dir, final_output)

                # Check if we just entered slow strategies zone
                recent_strategies = [r.get("strategy", "") for r in final_output["phase1_baseline"][-5:]]
                in_slow_zone = any(s in ["lookahead", "hybrid_adaptive"] for s in recent_strategies)

                # Send notification when entering slow zone
                if use_telegram and in_slow_zone and not slow_zone_notified:
                    slow_zone_notified = True
                    send_telegram_sync(
                        f"🐌 ENTERING SLOW STRATEGIES ZONE\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"Now testing: lookahead & hybrid_adaptive\n"
                        f"These take 10-12 min per experiment\n"
                        f"(vs 7 min for fast strategies)\n"
                        f"Progress: {completed}/{len(tasks)}\n"
                        f"ETA: {eta_seconds/60:.1f}m"
                    )

                # Telegram updates every 5% for frequent updates
                if use_telegram and (current_pct - last_telegram_pct >= 5 or completed == len(tasks)):
                    avg_fill = np.mean([r.get("avg_closed_fill", 0) for r in final_output["phase1_baseline"][-50:]]) if final_output["phase1_baseline"] else 0
                    total_closed = sum(r.get("pallets_closed", 0) for r in final_output["phase1_baseline"])

                    zone_indicator = "🐌 SLOW STRATEGIES" if in_slow_zone else "⚡ Fast strategies"

                    send_telegram_sync(
                        f"📈 Phase 1: {current_pct}%\n"
                        f"Progress: {completed}/{len(tasks)} experiments\n"
                        f"Zone: {zone_indicator}\n"
                        f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_seconds/60:.1f}m\n"
                        f"Pallets closed: {total_closed}\n"
                        f"Avg fill: {avg_fill:.1%}\n"
                        f"Rate: {rate*60:.1f} exp/hour"
                    )
                    last_telegram_pct = current_pct

    elapsed1 = time.perf_counter() - t0
    print(f"  Phase 1 completed in {elapsed1:.1f}s.")
    save_progress(out_dir, final_output)

    # Send Phase 1 completion notification
    if use_telegram:
        total_closed_p1 = sum(r.get("pallets_closed", 0) for r in final_output["phase1_baseline"])
        avg_fill_p1 = np.mean([r.get("avg_closed_fill", 0) for r in final_output["phase1_baseline"]]) if final_output["phase1_baseline"] else 0
        send_telegram_sync(
            f"✅ Phase 1 COMPLETE!\n"
            f"Time: {elapsed1/60:.1f} min ({elapsed1/3600:.1f} hours)\n"
            f"Experiments: {len(final_output['phase1_baseline'])}\n"
            f"Total pallets closed: {total_closed_p1}\n"
            f"Overall avg fill: {avg_fill_p1:.1%}\n"
            + ("Now starting Phase 2..." if not phase1_only else "Run finished (Phase 1 only).")
        )

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

    rankings_msg = "🏆 Phase 1 Complete - Top 5 Strategies:\n"
    for i, (s, score, n, total_pals) in enumerate(avg_scores, 1):
        stype = "multi-bin" if s in multibin_strategies else "single-bin"
        marker = " <--" if s in top_5_strategies else ""
        print(f"  {i:<5} {s:<32} {score:>6.2%} {n:>5} {total_pals:>5} {stype:>10}{marker}")

        if i <= 5:
            rankings_msg += f"{i}. {s}: {score:.1%} fill ({total_pals} pallets)\n"

    if use_telegram:
        send_telegram_sync(rankings_msg)

    final_output["metadata"]["top_5"] = top_5_strategies
    final_output["metadata"]["phase1_elapsed_s"] = elapsed1
    save_progress(out_dir, final_output)

    if phase1_only:
        total_elapsed = elapsed1
        final_output["metadata"]["phase2_elapsed_s"] = 0.0
        final_output["metadata"]["total_elapsed_s"] = total_elapsed
        save_progress(out_dir, final_output)

        print(f"\n{'='*75}")
        print(f"  OVERNIGHT SWEEP COMPLETE (PHASE 1 ONLY)")
        print(f"{'='*75}")
        print(f"  Results saved to: {out_dir}/results.json")
        print(f"  Phase 1: {elapsed1:.1f}s ({elapsed1/60:.1f} min)")
        print(f"  Total:   {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

        if use_telegram:
            total_closed_all = sum(r.get("pallets_closed", 0) for r in final_output["phase1_baseline"])
            avg_fill_all = np.mean([r.get("avg_closed_fill", 0) for r in final_output["phase1_baseline"]]) if final_output["phase1_baseline"] else 0
            send_telegram_sync(
                f"🎉 BOTKO RUN COMPLETE (PHASE 1 ONLY)\n"
                f"⏱ Runtime: {total_elapsed/60:.1f} min\n"
                f"Datasets: {n_datasets}, Shuffles: {n_shuffles}, Boxes: {n_boxes}\n"
                f"Pallets closed: {total_closed_all}\n"
                f"Avg closed fill: {avg_fill_all:.1%}\n"
                f"Results: {out_dir}/results.json"
            )
        return

    # ── Phase 2: Parameter sweep on top-5 ──────────────────────────────────
    box_selectors = ["default", "biggest_volume_first", "biggest_footprint_first"]
    bin_selectors = ["emptiest_first", "focus_fill", "flattest_first"]

    if smoketest:
        box_selectors = ["default", "biggest_volume_first"]
        bin_selectors = ["emptiest_first"]
        runs_per_sweep = 2
    elif demorun:
        runs_per_sweep = 2  # Use 2 datasets for demo Phase 2
    else:
        runs_per_sweep = 4 if quickrun else 8

    sweep_strategies = [s for s in top_5_strategies if s in singlebin_strategies]

    print(f"\n[PHASE 2] Parameter sweep on top-{len(sweep_strategies)} single-bin strategies")
    print(f"  Box selectors: {box_selectors}")
    print(f"  Bin selectors: {bin_selectors}")
    if any(s in multibin_strategies for s in top_5_strategies):
        print(f"  Note: multi-bin strategies use native routing (no sweep)")

    if use_telegram:
        send_telegram_sync(
            f"📊 Phase 2 Starting: Bin/Box Selector Sweep\n"
            f"Top {len(sweep_strategies)} strategies\n"
            f"Box selectors: {len(box_selectors)}\n"
            f"Bin selectors: {len(bin_selectors)}\n"
            f"Variations: {runs_per_sweep} datasets"
        )

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
    save_progress(out_dir, final_output)

    t2 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = {executor.submit(run_singlebin_experiment, t): t for t in phase2_tasks}
        completed = 0
        last_telegram_pct = 0

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
            current_pct = int((completed / len(phase2_tasks)) * 100) if phase2_tasks else 100
            elapsed2 = time.perf_counter() - t2
            rate2 = completed / elapsed2 if elapsed2 > 0 else 0
            remaining2 = len(phase2_tasks) - completed
            eta2_seconds = remaining2 / rate2 if rate2 > 0 else 0

            if len(phase2_tasks) > 0 and (completed % max(1, len(phase2_tasks) // 10) == 0 or completed == len(phase2_tasks)):
                print(f"    {completed}/{len(phase2_tasks)} ({current_pct}%) - ETA: {eta2_seconds/60:.1f}m")
                save_progress(out_dir, final_output)

                # Telegram updates every 5% for frequent updates
                if use_telegram and (current_pct - last_telegram_pct >= 5 or completed == len(phase2_tasks)):
                    total_closed_p2 = sum(r.get("pallets_closed", 0) for r in final_output["phase2_sweep"])
                    avg_fill_p2 = np.mean([r.get("avg_closed_fill", 0) for r in final_output["phase2_sweep"][-20:]]) if final_output["phase2_sweep"] else 0

                    # Show current configuration being tested
                    recent = final_output["phase2_sweep"][-1] if final_output["phase2_sweep"] else {}
                    current_strat = recent.get("strategy", "?")
                    current_box_sel = recent.get("box_selector", "?")
                    current_bin_sel = recent.get("bin_selector", "?")

                    send_telegram_sync(
                        f"📊 Phase 2: {current_pct}%\n"
                        f"Progress: {completed}/{len(phase2_tasks)} configs\n"
                        f"Testing: {current_strat}\n"
                        f"  Box sel: {current_box_sel}\n"
                        f"  Bin sel: {current_bin_sel}\n"
                        f"Elapsed: {elapsed2/60:.1f}m | ETA: {eta2_seconds/60:.1f}m\n"
                        f"Pallets: {total_closed_p2} | Avg fill: {avg_fill_p2:.1%}\n"
                        f"Rate: {rate2*60:.1f} exp/hour"
                    )
                    last_telegram_pct = current_pct

    elapsed2 = time.perf_counter() - t2
    print(f"  Phase 2 completed in {elapsed2:.1f}s.")
    save_progress(out_dir, final_output)

    # Send Phase 2 completion notification
    if use_telegram:
        total_closed_p2 = sum(r.get("pallets_closed", 0) for r in final_output["phase2_sweep"])
        avg_fill_p2 = np.mean([r.get("avg_closed_fill", 0) for r in final_output["phase2_sweep"]]) if final_output["phase2_sweep"] else 0
        send_telegram_sync(
            f"✅ Phase 2 COMPLETE!\n"
            f"Time: {elapsed2/60:.1f} min ({elapsed2/3600:.1f} hours)\n"
            f"Configurations tested: {len(final_output['phase2_sweep'])}\n"
            f"Pallets closed: {total_closed_p2}\n"
            f"Avg fill: {avg_fill_p2:.1%}\n"
            f"Preparing final summary..."
        )

    total_elapsed = elapsed1 + elapsed2
    final_output["metadata"]["phase2_elapsed_s"] = elapsed2
    final_output["metadata"]["total_elapsed_s"] = total_elapsed
    save_progress(out_dir, final_output)

    print(f"\n{'='*75}")
    print(f"  OVERNIGHT SWEEP COMPLETE")
    print(f"{'='*75}")
    print(f"  Results saved to: {out_dir}/results.json")
    print(f"  Phase 1: {elapsed1:.1f}s ({elapsed1/60:.1f} min)")
    print(f"  Phase 2: {elapsed2:.1f}s ({elapsed2/60:.1f} min)")
    print(f"  Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    if use_telegram:
        total_closed_all = sum(r.get("pallets_closed", 0) for r in final_output["phase1_baseline"])
        total_closed_p2 = sum(r.get("pallets_closed", 0) for r in final_output.get("phase2_sweep", []))
        avg_fill_all = np.mean([r.get("avg_closed_fill", 0) for r in final_output["phase1_baseline"]]) if final_output["phase1_baseline"] else 0

        # Build top 5 summary
        top_5_summary = "\n".join([
            f"  {i+1}. {s} ({score:.1%})"
            for i, (s, score, _, _) in enumerate(avg_scores[:5])
        ])

        send_telegram_sync(
            f"🎉 BOTKO DEMO COMPLETE!\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⏱ Total Runtime: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hours)\n"
            f"  Phase 1: {elapsed1/60:.1f} min\n"
            f"  Phase 2: {elapsed2/60:.1f} min\n"
            f"\n🏆 Top 5 Strategies:\n"
            f"{top_5_summary}\n"
            f"\n📊 Statistics:\n"
            f"  Experiments: {len(final_output['phase1_baseline']) + len(final_output.get('phase2_sweep', []))}\n"
            f"  Strategies tested: {len([s for s in all_strategy_names if any(r['strategy'] == s for r in final_output['phase1_baseline'])])}\n"
            f"  Overall avg fill: {avg_fill_all:.1%}\n"
            f"\n📦 Pallets Closed:\n"
            f"  Phase 1: {total_closed_all}\n"
            f"  Phase 2: {total_closed_p2}\n"
            f"  Total: {total_closed_all + total_closed_p2}\n"
            f"\n💾 Results saved to:\n"
            f"  {out_dir}/results.json"
        )


if __name__ == "__main__":
    main()
