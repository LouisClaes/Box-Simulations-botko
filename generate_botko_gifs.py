"""
Generate one conveyor GIF per strategy from the Botko BV run.

Reproduces the same box sequence as botko_20260223_133414 (dataset 0, shuffle 0,
seed=42, 400 boxes).  Each strategy is run exactly once; the full placement
sequence is recorded and rendered as an animated conveyor GIF.

GIFs saved to:  output/botko_20260223_133414/gifs/<strategy_name>.gif

Usage:
    python generate_botko_gifs.py                   # all 23 strategies
    python generate_botko_gifs.py --strategies baseline walle_scoring
    python generate_botko_gifs.py --max-frames 60   # fewer frames → faster render
    python generate_botko_gifs.py --fps 3
"""

import argparse
import multiprocessing
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Constants matching botko_20260223_133414 ──────────────────────────────────
N_BOXES = 400
DATASET_SEED = 42       # generate_rajapack seed for dataset 0
SHUFFLE_SEED = 100      # random.Random seed for dataset 0, shuffle 0
MAX_GIF_FRAMES = 80     # subsample total steps to at most this many frames
GIF_FPS = 4

OUTPUT_GIF_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output", "botko_20260223_133414", "gifs",
)

# Strategies that appeared in the original Botko run
ORIGINAL_SINGLEBIN = [
    "baseline", "best_fit_decreasing", "blueprint_packing", "column_fill",
    "ems", "extreme_points", "gopt_heuristic", "gravity_balanced",
    "heuristic_160", "hybrid_adaptive", "layer_building", "lbcp_stability",
    "lookahead", "online_bpp_heuristic", "pct_expansion", "pct_macs_heuristic",
    "skyline", "stacking_tree_stability", "surface_contact",
    "wall_building", "walle_scoring",
]
ORIGINAL_MULTIBIN = ["tsang_multibin"]


# ─────────────────────────────────────────────────────────────────────────────
# Worker — runs in a subprocess
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args: dict):
    """
    Run one strategy, capture ConveyorSteps, render GIF.
    Returns (strategy_name, success: bool, message: str).
    """
    # Each subprocess needs its own sys.path setup
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    strategy_name = args["strategy_name"]
    is_multibin   = args["is_multibin"]
    boxes         = args["boxes"]          # List[Box] — pre-generated in main
    out_path      = args["out_path"]
    max_frames    = args["max_frames"]
    fps           = args["fps"]

    try:
        from config import BinConfig
        from simulator.session import (
            PackingSession, SessionConfig, get_box_selector, get_bin_selector,
        )
        from simulator.close_policy import HeightClosePolicy
        from visualization.conveyor_gif_creator import ConveyorStep, create_conveyor_gif

        bin_config = BinConfig(
            length=1200.0, width=800.0, height=2700.0, resolution=10.0,
        )
        session_cfg = SessionConfig(
            bin_config=bin_config,
            num_bins=2,
            buffer_size=8,
            pick_window=4,
            close_policy=HeightClosePolicy(max_height=1800.0),
            max_consecutive_rejects=10,
            enable_stability=False,
            allow_all_orientations=False,
        )

        session = PackingSession(session_cfg)
        conveyor_steps = []

        def on_step(step_num, step_result, obs):
            try:
                p0 = list(session.stations[0].bin_state.placed_boxes)
                p1 = list(session.stations[1].bin_state.placed_boxes)
                box = step_result.box
                if box is not None:
                    cs = ConveyorStep(
                        step=step_num,
                        box=box,
                        bin_index=step_result.bin_index if step_result.placed else 0,
                        placed=step_result.placed,
                        placement=step_result.placement if step_result.placed else None,
                        buffer_snapshot=list(obs.buffer_view),
                        stream_remaining=obs.stream_remaining,
                        pallet0_placements=p0,
                        pallet1_placements=p1,
                    )
                    conveyor_steps.append(cs)
            except Exception:
                pass

        # Run strategy exactly once
        if is_multibin:
            from strategies.base_strategy import get_multibin_strategy
            strategy = get_multibin_strategy(strategy_name)
            result = session.run(boxes, strategy, on_step=on_step)
        else:
            from strategies.base_strategy import get_strategy
            strategy = get_strategy(strategy_name)
            box_sel = get_box_selector("default")
            bin_sel = get_bin_selector("emptiest_first")
            result = session.run(boxes, strategy, box_sel, bin_sel, on_step=on_step)

        # Subsample steps to keep GIF size manageable
        steps = conveyor_steps
        if len(steps) > max_frames:
            interval = len(steps) / max_frames
            indices = sorted({
                int(round(i * interval)) for i in range(max_frames)
            } | {0, len(steps) - 1})
            steps = [steps[i] for i in indices if i < len(steps)]

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        create_conveyor_gif(steps, bin_config, out_path, title=strategy_name, fps=fps)

        fill   = result.avg_closed_fill
        pals   = result.closed_pallets
        placed = result.total_placed
        return (strategy_name, True,
                f"fill={fill:.1%}  pallets={pals}  placed={placed}  frames={len(steps)}")

    except Exception as exc:
        tb = traceback.format_exc()
        return (strategy_name, False, f"{exc}\n{tb[:400]}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Botko conveyor GIFs")
    parser.add_argument(
        "--strategies", nargs="*",
        help="Specific strategy names to process (default: all original strategies)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=MAX_GIF_FRAMES,
        help=f"Max frames per GIF (default: {MAX_GIF_FRAMES})",
    )
    parser.add_argument(
        "--fps", type=int, default=GIF_FPS,
        help=f"GIF frames per second (default: {GIF_FPS})",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate GIFs even if they already exist",
    )
    cli = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    # ── Generate box sequence matching the original run ───────────────────
    print(f"Generating {N_BOXES} boxes (seed={DATASET_SEED}, shuffle seed={SHUFFLE_SEED})...")
    from dataset.generator import generate_rajapack
    boxes = list(generate_rajapack(N_BOXES, seed=DATASET_SEED))
    random.Random(SHUFFLE_SEED).shuffle(boxes)
    print(f"  {len(boxes)} boxes ready")

    # ── Collect strategies ────────────────────────────────────────────────
    if cli.strategies:
        from strategies.base_strategy import STRATEGY_REGISTRY, MULTIBIN_STRATEGY_REGISTRY
        multibin_set = set(MULTIBIN_STRATEGY_REGISTRY.keys())
        selected = [(s, s in multibin_set) for s in cli.strategies]
    else:
        selected = (
            [(s, False) for s in ORIGINAL_SINGLEBIN] +
            [(s, True)  for s in ORIGINAL_MULTIBIN]
        )

    # ── Build task list (skip already-done unless --overwrite) ────────────
    os.makedirs(OUTPUT_GIF_DIR, exist_ok=True)
    tasks = []
    skipped = []
    for strat_name, is_mb in selected:
        out_path = os.path.join(OUTPUT_GIF_DIR, f"{strat_name}.gif")
        if not cli.overwrite and os.path.exists(out_path):
            skipped.append(strat_name)
            continue
        tasks.append({
            "strategy_name": strat_name,
            "is_multibin":   is_mb,
            "boxes":         boxes,
            "out_path":      out_path,
            "max_frames":    cli.max_frames,
            "fps":           cli.fps,
        })

    if skipped:
        print(f"  Skipping {len(skipped)} already-done: {', '.join(skipped)}")
        print(f"  (use --overwrite to regenerate)")

    if not tasks:
        print("All GIFs already exist. Done.")
        return

    # ── Parallel execution at ~50% CPU ───────────────────────────────────
    num_workers = max(1, multiprocessing.cpu_count() // 2)
    print(f"\n  Strategies to render : {len(tasks)}")
    print(f"  Workers (50% CPU)    : {num_workers}")
    print(f"  Max frames per GIF   : {cli.max_frames}")
    print(f"  FPS                  : {cli.fps}")
    print(f"  Output dir           : {OUTPUT_GIF_DIR}")
    print()

    t0 = time.perf_counter()
    done = 0
    ok = []
    failed = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker, t): t["strategy_name"] for t in tasks}
        for future in as_completed(futures):
            strat_name = futures[future]
            try:
                name, success, msg = future.result()
            except Exception as exc:
                name, success, msg = strat_name, False, str(exc)

            done += 1
            elapsed = time.perf_counter() - t0
            eta_s = elapsed / done * (len(tasks) - done) if done < len(tasks) else 0
            status = "OK  " if success else "FAIL"
            print(f"  [{status}] {name:<30} {msg}")
            print(f"         ({done}/{len(tasks)}, elapsed {elapsed:.0f}s, ETA ~{eta_s:.0f}s)")

            if success:
                ok.append(name)
            else:
                failed.append(name)

    total = time.perf_counter() - t0
    print(f"\n{'─'*60}")
    print(f"  Done in {total:.0f}s")
    print(f"  Success : {len(ok)}  ({', '.join(ok)})")
    if failed:
        print(f"  Failed  : {len(failed)}  ({', '.join(failed)})")
    print(f"  GIFs    : {OUTPUT_GIF_DIR}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
