"""
Generate one Botko-visualizer GIF per strategy from the Botko BV experiment.

Reproduces the same box sequence as botko_20260223_133414 (dataset 0, shuffle 0,
seed=42, 400 boxes).  Each strategy is run with the exact same session config
including FullestOnConsecutiveRejectsPolicy(4, 0.5).

GIFs saved to:  output/botko_20260223_133414/gifs/<strategy_name>.gif

Usage:
    python generate_botko_strategy_gifs.py                     # all 23 strategies
    python generate_botko_strategy_gifs.py --strategies baseline walle_scoring
    python generate_botko_strategy_gifs.py --max-frames 10     # fewer frames
    python generate_botko_strategy_gifs.py --fps 3
    python generate_botko_strategy_gifs.py --overwrite
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
DATASET_SEED = 42
SHUFFLE_SEED = 100
MAX_GIF_FRAMES = 80
GIF_FPS = 3

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
    Run one strategy, capture BotkoSteps, render GIF.
    Returns (strategy_name, success: bool, message: str).
    """
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    strategy_name = args["strategy_name"]
    is_multibin   = args["is_multibin"]
    boxes         = args["boxes"]
    out_path      = args["out_path"]
    max_frames    = args["max_frames"]
    fps           = args["fps"]

    try:
        from config import BinConfig
        from simulator.session import (
            PackingSession, SessionConfig, get_box_selector, get_bin_selector,
        )
        from simulator.close_policy_custom import FullestOnConsecutiveRejectsPolicy
        from visualization.botko_visualizer import BotkoStep, create_botko_gif

        bin_config = BinConfig(
            length=1200.0, width=800.0, height=2700.0, resolution=10.0,
        )

        close_policy = FullestOnConsecutiveRejectsPolicy(
            max_consecutive=4,
            min_fill_to_close=0.5,
        )

        session_cfg = SessionConfig(
            bin_config=bin_config,
            num_bins=2,
            buffer_size=8,
            pick_window=4,
            close_policy=close_policy,
            max_consecutive_rejects=10,
            enable_stability=False,
            allow_all_orientations=False,
        )

        session = PackingSession(session_cfg)
        botko_steps: list = []
        total_placed_counter = [0]
        total_rejected_counter = [0]
        pallets_closed_counter = [0]

        def on_step(step_num, step_result, obs):
            try:
                p0 = list(session.stations[0].bin_state.placed_boxes)
                p1 = list(session.stations[1].bin_state.placed_boxes)
                p0_fill = session.stations[0].bin_state.get_fill_rate()
                p1_fill = session.stations[1].bin_state.get_fill_rate()

                box = step_result.box
                if box is None:
                    return

                if step_result.placed:
                    total_placed_counter[0] += 1
                    action = f"Placed on Pallet {step_result.bin_index}"
                else:
                    total_rejected_counter[0] += 1
                    action = "Rejected — no fit"

                if step_result.pallet_closed:
                    pallets_closed_counter[0] += 1

                bs = BotkoStep(
                    step=step_num,
                    total_steps=len(boxes),
                    box=box,
                    bin_index=step_result.bin_index if step_result.placed else -1,
                    placed=step_result.placed,
                    placement=step_result.placement if step_result.placed else None,
                    buffer_snapshot=list(obs.buffer_view),
                    stream_remaining=obs.stream_remaining,
                    pallet0_placements=p0,
                    pallet1_placements=p1,
                    strategy_name=strategy_name,
                    action=action,
                    consecutive_rejects=session._consecutive_rejects,
                    pallets_closed_so_far=pallets_closed_counter[0],
                    pallet0_fill=p0_fill,
                    pallet1_fill=p1_fill,
                    close_policy_desc=close_policy.describe(),
                    total_placed=total_placed_counter[0],
                    total_rejected=total_rejected_counter[0],
                )
                botko_steps.append(bs)
            except Exception:
                pass

        # Run strategy
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
        steps = botko_steps
        if len(steps) > max_frames:
            interval = len(steps) / max_frames
            indices = sorted({
                int(round(i * interval)) for i in range(max_frames)
            } | {0, len(steps) - 1})
            steps = [steps[i] for i in indices if i < len(steps)]

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        create_botko_gif(steps, bin_config, out_path, fps=fps)

        fill   = result.avg_closed_fill
        pals   = result.closed_pallets
        placed = result.total_placed
        return (strategy_name, True,
                f"fill={fill:.1%}  pallets={len(pals)}  placed={placed}  frames={len(steps)}")

    except Exception as exc:
        tb = traceback.format_exc()
        return (strategy_name, False, f"{exc}\n{tb[:400]}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Botko strategy GIFs")
    parser.add_argument(
        "--strategies", nargs="*",
        help="Specific strategy names (default: all 23)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=MAX_GIF_FRAMES,
        help=f"Max frames per GIF (default: {MAX_GIF_FRAMES})",
    )
    parser.add_argument(
        "--fps", type=int, default=GIF_FPS,
        help=f"GIF FPS (default: {GIF_FPS})",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate even if GIF exists",
    )
    cli = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    # ── Generate box sequence matching the original run ───────────────────
    print(f"Generating {N_BOXES} boxes (seed={DATASET_SEED}, shuffle={SHUFFLE_SEED})...")
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

    # ── Build task list ───────────────────────────────────────────────────
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
