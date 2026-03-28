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
from collections import defaultdict
from datetime import datetime, timedelta
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
    margin=20.0,  # 2cm gap between boxes and walls (per Botko BV spec)
)
MAX_CONSECUTIVE_REJECTS = 20  # Safety valve: session terminates if close logic is blocked (normal close at 8 rejects)
PALLET_CLOSE_AFTER_REJECTS = 4  # Condition 1: lookahead trigger after 4 consecutive fails

BOTKO_SESSION_CONFIG = SessionConfig(
    bin_config=BOTKO_PALLET,
    num_bins=BOTKO_BINS,
    buffer_size=BOTKO_BUFFER_SIZE,
    pick_window=BOTKO_PICK_WINDOW,
    close_policy=FullestOnConsecutiveRejectsPolicy(
        max_consecutive=PALLET_CLOSE_AFTER_REJECTS,
        min_fill_to_close=0.40  # Close pallets at ≥40% fill (strategies cap at 41-47%)
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


def _eta_str(seconds: float) -> str:
    """Format seconds into a human-readable ETA string like '2h 14m' or '45m'."""
    if seconds <= 0:
        return "done"
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    h += td.days * 24
    m = rem // 60
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def _pbar(done: int, total: int, width: int = 12) -> str:
    """Return a compact text progress bar like [████████░░░░] 67%."""
    pct = done / total if total > 0 else 1.0
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:.0%}"


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
    parser.add_argument("--skip-phase1", action="store_true", help="Skip Phase 1, load baseline from --phase1-from and go straight to Phase 2")
    parser.add_argument("--phase1-from", type=str, help="Path to existing results.json with Phase 1 data (required with --skip-phase1)")
    parser.add_argument("--resume", type=str, help="Path to a previous results.json to resume from")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram notifications")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated list of strategies to run (e.g. gravity_balanced,extreme_points). "
                             "If set, only these strategies are included in Phase 1.")
    parser.add_argument("--cpu-fraction", type=float, default=None,
                        help="Fraction of available CPUs to use (0-1). Overrides default 0.75")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="Base seed for dataset generation (changes datasets when different)")
    args = parser.parse_args()

    smoketest = args.smoke_test
    demorun = args.demo
    quickrun = args.quick
    phase1_only = args.phase1_only
    skip_phase1 = args.skip_phase1
    use_telegram = not args.no_telegram

    if skip_phase1 and not args.phase1_from:
        print("ERROR: --skip-phase1 requires --phase1-from <path/to/results.json>")
        sys.exit(1)

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

    # CPU fraction: default 75% (Raspberry Pi optimization) but overridable via --cpu-fraction
    cpu_frac = args.cpu_fraction if args.cpu_fraction is not None else 0.75
    cpu_frac = max(0.01, min(1.0, float(cpu_frac)))
    num_cpus = max(1, int(multiprocessing.cpu_count() * cpu_frac))

    # Set process priority so other processes preempt us if needed
    # nice value: 15 = below-normal priority, OS scheduler will prefer other processes
    try:
        os.nice(15)
        print(f"  Process nice level set to 15 (yields to other processes automatically)")
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
    # Exclude strategies that need training (RL) or are learning-based
    EXCLUDED_STRATEGIES = [
        "selective_hyper_heuristic",    # Learning-based - needs training
        "rl_dqn",                       # RL - needs trained neural network weights
        "rl_hybrid_hh",                 # RL - needs trained weights
        "rl_mcts_hybrid",               # RL - MCTS hybrid needs training
        "rl_a2c_masked",                # RL - needs trained weights
        "rl_pct_transformer",           # RL - transformer needs training
        "rl_ppo",                       # RL - PPO needs trained weights
    ]
    EXCLUDED_MULTIBIN_STRATEGIES = [
        "rl_mcts_hybrid_multibin",      # RL multibin - needs trained weights
    ]

    # Slow strategies - run LAST (11-12 min per experiment, acceptable for 2-day run)
    SLOW_STRATEGIES = [
        "lookahead",                    # Tree search - 11.6 min/experiment
        "hybrid_adaptive",              # Adaptive - 10.2 min/experiment
    ]

    if smoketest:
        singlebin_strategies = ["surface_contact", "walle_scoring", "baseline"]
        multibin_strategies = [s for s in MULTIBIN_STRATEGY_REGISTRY.keys() if s not in EXCLUDED_MULTIBIN_STRATEGIES][:1]
    else:
        # Get all non-RL, non-excluded strategies
        all_single = [s for s in STRATEGY_REGISTRY.keys() if s not in EXCLUDED_STRATEGIES]

        # Separate fast and slow strategies
        fast_strategies = [s for s in all_single if s not in SLOW_STRATEGIES]
        slow_strategies = [s for s in all_single if s in SLOW_STRATEGIES]

        # Fast strategies first, slow strategies last
        singlebin_strategies = fast_strategies + slow_strategies
        multibin_strategies = [s for s in MULTIBIN_STRATEGY_REGISTRY.keys() if s not in EXCLUDED_MULTIBIN_STRATEGIES]

    # Optional: restrict to a specific subset of strategies (--strategies flag)
    if args.strategies:
        target = [s.strip() for s in args.strategies.split(",") if s.strip()]
        singlebin_strategies = [s for s in singlebin_strategies if s in target]
        multibin_strategies  = [s for s in multibin_strategies  if s in target]

    all_strategy_names = singlebin_strategies + multibin_strategies

    print(f"\n{'='*75}")
    print(f"  BOTKO BV OVERNIGHT SWEEP -- {'SMOKE TEST' if smoketest else 'FULL RUN'}")
    print(f"{'='*75}")
    print(f"  CPUs:        {num_cpus} ({cpu_frac:.0%} of detected CPUs, nice 15 — yields to other processes)")
    print(f"  Output:      {out_dir}")
    print(f"  Datasets:    {n_datasets} x {n_boxes} Rajapack boxes, {n_shuffles} shuffles")
    print(f"  Single-bin:  {len(singlebin_strategies)} strategies")
    print(f"  Multi-bin:   {len(multibin_strategies)} strategies ({', '.join(multibin_strategies)})")
    print(f"  Pallet:      {BOTKO_PALLET.length:.0f}x{BOTKO_PALLET.width:.0f}x{BOTKO_PALLET.height:.0f}mm")
    print(f"  Buffer:      {BOTKO_BUFFER_SIZE} boxes, pick window {BOTKO_PICK_WINDOW}")
    print(f"  Close:       {BOTKO_SESSION_CONFIG.close_policy.describe()}")
    print(f"  Reject:      FIFO advance, front box exits (max {MAX_CONSECUTIVE_REJECTS} consecutive)")
    print(f"  Stats:       only closed pallets count in avg fill rate")
    phase_mode_str = "PHASE 1 ONLY" if phase1_only else ("SKIP PHASE 1 → PHASE 2 ONLY" if skip_phase1 else "PHASE 1 + PHASE 2")
    print(f"  Phase mode:  {phase_mode_str}")
    print(f"  Telegram:    {'ENABLED' if use_telegram else 'DISABLED'}")

    # Telegram: Experiment start
    if use_telegram and not args.resume:
        msg = (
            f"🚀 Botko Overnight Sweep Started\n"
            f"Mode: {'Smoke Test' if smoketest else 'Full Run'}\n"
            f"Phase mode: {'Phase 1 only' if phase1_only else 'Phase 1 + Phase 2'}\n"
            f"Datasets: {n_datasets} × {n_boxes} boxes × {n_shuffles} shuffles\n"
            f"Strategies: {len(all_strategy_names)} total\n"
            f"CPUs: {num_cpus}/4 (75%, nice 15)\n"
            f"Output: {os.path.basename(out_dir)}"
        )
        send_telegram_sync(msg)

    # ── Generate datasets ──────────────────────────────────────────────────
    seed_base = int(args.seed_base)
    print(f"\n  Generating {n_datasets} datasets (seed base={seed_base})...")
    datasets = []
    for d in range(n_datasets):
        ds_boxes = generate_rajapack(n_boxes, seed=seed_base + d)
        shuffles = []
        for s in range(n_shuffles):
            shuffled = list(ds_boxes)
            random.Random(1000 * (seed_base + d) + s + 100).shuffle(shuffled)
            shuffles.append(shuffled)
        datasets.append({
            "dataset_id": d,
            "shuffles": shuffles,
            "original_boxes": ds_boxes,
        })

    # ── Phase 1: Baseline (or skip) ────────────────────────────────────────
    elapsed1 = 0.0

    if skip_phase1:
        # Load Phase 1 results from an existing run, jump straight to Phase 2
        print(f"\n[PHASE 1] SKIPPED — loading baseline from {args.phase1_from}")
        with open(args.phase1_from, 'r') as f:
            p1_data = json.load(f)
        final_output["phase1_baseline"] = p1_data.get("phase1_baseline", [])
        final_output["metadata"]["top_5"] = p1_data["metadata"].get("top_5", [])
        final_output["metadata"]["phase1_elapsed_s"] = p1_data["metadata"].get("phase1_elapsed_s", 0.0)
        print(f"  Loaded {len(final_output['phase1_baseline'])} Phase 1 results")
        print(f"  Top 5 from loaded data: {final_output['metadata']['top_5']}")
        if use_telegram:
            send_telegram_sync(
                f"⏭️ Phase 1 SKIPPED\n"
                f"Loaded {len(final_output['phase1_baseline'])} results from existing run\n"
                f"Top 5: {', '.join(final_output['metadata']['top_5'])}\n"
                f"Going straight to Phase 2 sweep..."
            )
        save_progress(out_dir, final_output)

    else:
        completed_phase1 = set()
        for r in final_output.get("phase1_baseline", []):
            completed_phase1.add((r["dataset_id"], r["shuffle_id"], r["strategy"]))

        print(f"\n[PHASE 1] Baseline: all strategies with default selectors")
        if use_telegram:
            fast_count = len([s for s in singlebin_strategies if s not in SLOW_STRATEGIES])
            slow_count = len([s for s in singlebin_strategies if s in SLOW_STRATEGIES])
            total_experiments = (len(singlebin_strategies) + len(multibin_strategies)) * n_datasets * n_shuffles
            send_telegram_sync(
                f"🚀 BOTKO SWEEP STARTING\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Phase 1: Baseline Testing\n"
                f"  Fast strategies: {fast_count}\n"
                f"  Slow strategies: {slow_count} (run last)\n"
                f"  Multi-bin: {len(multibin_strategies)}\n"
                f"  Datasets: {n_datasets} × {n_shuffles} shuffles\n"
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
                        "bin_selector": "focus_fill",
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

        # How many runs we expect per strategy (for per-strategy completion detection)
        runs_per_strategy = n_datasets * n_shuffles
        strat_results_p1: Dict[str, list] = defaultdict(list)
        strat_notified_p1: set = set()
        HEARTBEAT_SECS = 1800  # 30-minute heartbeat if nothing else fires

        print(f"  Enqueuing {len(tasks)} tasks...")
        t0 = time.perf_counter()
        last_notification_time = t0

        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = {executor.submit(process_chunk, t): t for t in tasks}
            completed = 0
            last_telegram_pct = 0
            slow_zone_notified = False

            for future in as_completed(futures):
                res = future.result()
                if res["success"]:
                    entry = res["summary"].copy()
                    entry["dataset_id"] = res["args_echo"]["dataset_id"]
                    entry["shuffle_id"] = res["args_echo"]["shuffle_id"]
                    final_output["phase1_baseline"].append(entry)
                    strat_results_p1[entry["strategy"]].append(entry)
                else:
                    strat_name = res["args_echo"].get("strategy_name", "?")
                    print(f"    FAIL [{strat_name}]: {res.get('error', '?')[:80]}")
                    if use_telegram:
                        send_telegram_sync(
                            f"⚠️ Experiment failed\n"
                            f"Strategy: {strat_name}\n"
                            f"Error: {res.get('error', '?')[:120]}"
                        )
                        last_notification_time = time.perf_counter()

                completed += 1
                current_pct = int((completed / len(tasks)) * 100)
                now = time.perf_counter()
                elapsed = now - t0
                rate = completed / elapsed if elapsed > 0 else 0
                remaining_tasks = len(tasks) - completed
                eta_seconds = remaining_tasks / rate if rate > 0 else 0

                # ── Per-strategy completion notification ──────────────────
                if use_telegram and res["success"]:
                    sname = entry["strategy"]
                    done_runs = len(strat_results_p1[sname])
                    if sname not in strat_notified_p1 and done_runs >= runs_per_strategy:
                        strat_notified_p1.add(sname)
                        s_fills = [r.get("avg_closed_fill", 0) for r in strat_results_p1[sname]]
                        s_placed = [r.get("total_placed", 0) for r in strat_results_p1[sname]]
                        s_pals = [r.get("pallets_closed", 0) for r in strat_results_p1[sname]]
                        is_slow = sname in SLOW_STRATEGIES
                        tag = "🐌" if is_slow else "✓"
                        send_telegram_sync(
                            f"{tag} {sname}\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"Avg fill:   {np.mean(s_fills):.1%}  (best {max(s_fills):.1%})\n"
                            f"Placed:     {np.mean(s_placed):.0f} boxes avg\n"
                            f"Pallets:    {np.mean(s_pals):.1f} avg closed\n"
                            f"Runs:       {done_runs}/{runs_per_strategy}\n"
                            f"\n"
                            f"Overall {_pbar(completed, len(tasks))}\n"
                            f"ETA: {_eta_str(eta_seconds)}"
                        )
                        last_notification_time = now

                # ── Save + console print every ~5% ────────────────────────
                if len(tasks) > 0 and (completed % max(1, len(tasks) // 20) == 0 or completed == len(tasks)):
                    print(f"    {completed}/{len(tasks)} ({current_pct}%) - ETA: {_eta_str(eta_seconds)}")
                    save_progress(out_dir, final_output)

                    recent_strategies = [r.get("strategy", "") for r in final_output["phase1_baseline"][-5:]]
                    in_slow_zone = any(s in SLOW_STRATEGIES for s in recent_strategies)

                    if use_telegram and in_slow_zone and not slow_zone_notified:
                        slow_zone_notified = True
                        send_telegram_sync(
                            f"🐌 Entering slow strategies\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"Now testing: lookahead & hybrid_adaptive\n"
                            f"~10-12 min per experiment\n"
                            f"{_pbar(completed, len(tasks))} ETA {_eta_str(eta_seconds)}"
                        )
                        last_notification_time = now

                    if use_telegram and (current_pct - last_telegram_pct >= 10 or completed == len(tasks)):
                        avg_fill = np.mean([r.get("avg_closed_fill", 0) for r in final_output["phase1_baseline"][-50:]]) if final_output["phase1_baseline"] else 0
                        total_closed = sum(r.get("pallets_closed", 0) for r in final_output["phase1_baseline"])
                        strats_done = len(strat_notified_p1)
                        total_strats = len(all_strategy_names)
                        send_telegram_sync(
                            f"📈 Phase 1 — {current_pct}%\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"{_pbar(completed, len(tasks))}\n"
                            f"Experiments: {completed}/{len(tasks)}\n"
                            f"Strategies done: {strats_done}/{total_strats}\n"
                            f"Pallets closed: {total_closed}\n"
                            f"Recent avg fill: {avg_fill:.1%}\n"
                            f"Rate: {rate * 3600:.0f} exp/hr\n"
                            f"ETA: {_eta_str(eta_seconds)}"
                        )
                        last_telegram_pct = current_pct
                        last_notification_time = now

                # ── Heartbeat: send if 30 min of silence ──────────────────
                if use_telegram and (now - last_notification_time) >= HEARTBEAT_SECS:
                    send_telegram_sync(
                        f"💓 Still running — Phase 1\n"
                        f"{_pbar(completed, len(tasks))}\n"
                        f"Experiments: {completed}/{len(tasks)}\n"
                        f"ETA: {_eta_str(eta_seconds)}"
                    )
                    last_notification_time = now

        elapsed1 = time.perf_counter() - t0
        print(f"  Phase 1 completed in {elapsed1:.1f}s.")
        save_progress(out_dir, final_output)

        if use_telegram:
            total_closed_p1 = sum(r.get("pallets_closed", 0) for r in final_output["phase1_baseline"])
            avg_fill_p1 = np.mean([r.get("avg_closed_fill", 0) for r in final_output["phase1_baseline"]]) if final_output["phase1_baseline"] else 0
            send_telegram_sync(
                f"✅ Phase 1 complete!\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Time:        {_eta_str(elapsed1)} ({elapsed1/3600:.1f}h)\n"
                f"Experiments: {len(final_output['phase1_baseline'])}\n"
                f"Pallets:     {total_closed_p1} closed\n"
                f"Avg fill:    {avg_fill_p1:.1%}\n"
                f"\n{'⏩ Starting Phase 2 sweep...' if not phase1_only else '🏁 Run finished (Phase 1 only).'}"
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
        sel_combos = len(box_selectors) * len(bin_selectors)
        total_p2_exp = len(sweep_strategies) * sel_combos * runs_per_sweep
        send_telegram_sync(
            f"🔬 Phase 2 — Selector Sweep\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Top strategies: {', '.join(sweep_strategies)}\n"
            f"Box selectors:  {', '.join(box_selectors)}\n"
            f"Bin selectors:  {', '.join(bin_selectors)}\n"
            f"Combos/strategy: {sel_combos}\n"
            f"Total runs:     ~{total_p2_exp}\n"
            f"Datasets/combo: {runs_per_sweep}"
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

    # Track best combo seen so far in Phase 2
    combo_scores: Dict[str, list] = defaultdict(list)  # "strat|box|bin" -> list of placement rates
    last_notification_time_p2 = time.perf_counter()

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
                combo_key = f"{entry['strategy']}|{entry.get('box_selector','?')}|{entry.get('bin_selector','?')}"
                combo_scores[combo_key].append(entry.get("placement_rate", entry.get("avg_closed_fill", 0)))
            else:
                print(f"    FAIL: {res.get('error', '?')[:80]}")
                if use_telegram:
                    ae = res.get("args_echo", {})
                    send_telegram_sync(
                        f"⚠️ Phase 2 experiment failed\n"
                        f"Strategy: {ae.get('strategy_name','?')}\n"
                        f"Box sel: {ae.get('box_selector','?')} | Bin sel: {ae.get('bin_selector','?')}\n"
                        f"Error: {res.get('error','?')[:100]}"
                    )
                    last_notification_time_p2 = time.perf_counter()

            completed += 1
            current_pct = int((completed / len(phase2_tasks)) * 100) if phase2_tasks else 100
            now2 = time.perf_counter()
            elapsed2 = now2 - t2
            rate2 = completed / elapsed2 if elapsed2 > 0 else 0
            remaining2 = len(phase2_tasks) - completed
            eta2_seconds = remaining2 / rate2 if rate2 > 0 else 0

            if len(phase2_tasks) > 0 and (completed % max(1, len(phase2_tasks) // 10) == 0 or completed == len(phase2_tasks)):
                print(f"    {completed}/{len(phase2_tasks)} ({current_pct}%) - ETA: {_eta_str(eta2_seconds)}")
                save_progress(out_dir, final_output)

                if use_telegram and (current_pct - last_telegram_pct >= 10 or completed == len(phase2_tasks)):
                    # Compute best combo so far
                    best_combo_key = max(combo_scores, key=lambda k: np.mean(combo_scores[k])) if combo_scores else "—"
                    best_combo_score = np.mean(combo_scores[best_combo_key]) if combo_scores else 0.0
                    best_strat, best_box, best_bin = (best_combo_key.split("|") + ["?", "?", "?"])[:3]

                    recent = final_output["phase2_sweep"][-1] if final_output["phase2_sweep"] else {}
                    send_telegram_sync(
                        f"📊 Phase 2 — {current_pct}%\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"{_pbar(completed, len(phase2_tasks))}\n"
                        f"Configs: {completed}/{len(phase2_tasks)}\n"
                        f"Testing now:\n"
                        f"  {recent.get('strategy','?')} | {recent.get('box_selector','?')} | {recent.get('bin_selector','?')}\n"
                        f"\n🏆 Best so far:\n"
                        f"  {best_strat}\n"
                        f"  box={best_box} | bin={best_bin}\n"
                        f"  Score: {best_combo_score:.1%}\n"
                        f"\nETA: {_eta_str(eta2_seconds)}"
                    )
                    last_telegram_pct = current_pct
                    last_notification_time_p2 = now2

            # Heartbeat for Phase 2
            if use_telegram and (now2 - last_notification_time_p2) >= HEARTBEAT_SECS:
                send_telegram_sync(
                    f"💓 Still running — Phase 2\n"
                    f"{_pbar(completed, len(phase2_tasks))}\n"
                    f"Configs: {completed}/{len(phase2_tasks)}\n"
                    f"ETA: {_eta_str(eta2_seconds)}"
                )
                last_notification_time_p2 = now2

    elapsed2 = time.perf_counter() - t2
    print(f"  Phase 2 completed in {elapsed2:.1f}s.")
    save_progress(out_dir, final_output)

    # Phase 2 completion — show best combo found
    if use_telegram:
        total_closed_p2 = sum(r.get("pallets_closed", 0) for r in final_output["phase2_sweep"])
        avg_fill_p2 = np.mean([r.get("avg_closed_fill", 0) for r in final_output["phase2_sweep"]]) if final_output["phase2_sweep"] else 0
        # Best combo
        if combo_scores:
            best_k = max(combo_scores, key=lambda k: np.mean(combo_scores[k]))
            b_strat, b_box, b_bin = (best_k.split("|") + ["?", "?", "?"])[:3]
            b_score = np.mean(combo_scores[best_k])
            best_line = f"🥇 {b_strat}\n   box={b_box} | bin={b_bin}\n   Score: {b_score:.1%}"
        else:
            best_line = "—"
        send_telegram_sync(
            f"✅ Phase 2 complete!\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Time:    {_eta_str(elapsed2)} ({elapsed2/3600:.1f}h)\n"
            f"Configs: {len(final_output['phase2_sweep'])}\n"
            f"Pallets: {total_closed_p2} | Avg fill: {avg_fill_p2:.1%}\n"
            f"\nBest configuration:\n{best_line}\n"
            f"\nBuilding final summary..."
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

        # Top 5 phase 1 rankings
        top_5_lines = "\n".join([
            f"  {'🥇🥈🥉  5️⃣ 6️⃣'.split()[i] if i < 3 else str(i+1)+'.'} {s} — {score:.1%}"
            for i, (s, score, _, _) in enumerate(avg_scores[:5])
        ])

        # Best phase 2 combo
        if combo_scores:
            best_k = max(combo_scores, key=lambda k: np.mean(combo_scores[k]))
            b_strat, b_box, b_bin = (best_k.split("|") + ["?", "?", "?"])[:3]
            b_score = np.mean(combo_scores[best_k])
            best_p2_line = f"  {b_strat}\n  box={b_box} | bin={b_bin} — {b_score:.1%}"
        else:
            best_p2_line = "  —"

        total_experiments = len(final_output['phase1_baseline']) + len(final_output.get('phase2_sweep', []))
        strats_tested = len({r['strategy'] for r in final_output['phase1_baseline']})

        send_telegram_sync(
            f"🎉 BOTKO SWEEP COMPLETE\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⏱  Total:   {_eta_str(total_elapsed)} ({total_elapsed/3600:.1f}h)\n"
            f"   Phase 1: {_eta_str(elapsed1)}\n"
            f"   Phase 2: {_eta_str(elapsed2)}\n"
            f"\n📊 Phase 1 — Top 5 Strategies:\n"
            f"{top_5_lines}\n"
            f"\n🔬 Phase 2 — Best Config:\n"
            f"{best_p2_line}\n"
            f"\n📦 Stats:\n"
            f"  Experiments:    {total_experiments}\n"
            f"  Strategies:     {strats_tested}\n"
            f"  Pallets closed: {total_closed_all + total_closed_p2}\n"
            f"  Avg fill (P1):  {avg_fill_all:.1%}\n"
            f"\n💾 {os.path.basename(out_dir)}/results.json"
        )


if __name__ == "__main__":
    main()
