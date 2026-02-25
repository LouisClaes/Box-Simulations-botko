#!/usr/bin/env python3
"""
Generate GIFs for each strategy for presentation slides.
Runs with minimal CPU usage (nice 19) to not interfere with main demo run.
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Box, BinConfig
from dataset.generator import generate_rajapack
from strategies.base_strategy import STRATEGY_REGISTRY, MULTIBIN_STRATEGY_REGISTRY
from simulator.session import PackingSession, SessionConfig, get_box_selector, get_bin_selector
from simulator.close_policy_custom import FullestOnConsecutiveRejectsPolicy
from visualization.conveyor_gif_creator import ConveyorStep, create_conveyor_gif

# Botko BV configuration (same as demo)
BOTKO_PALLET = BinConfig(
    length=1200.0, width=800.0, height=2700.0, resolution=10.0,
)

BOTKO_SESSION_CONFIG = SessionConfig(
    bin_config=BOTKO_PALLET,
    num_bins=2,
    buffer_size=8,
    pick_window=4,
    close_policy=FullestOnConsecutiveRejectsPolicy(
        max_consecutive=4,
        min_fill_to_close=0.5
    ),
    max_consecutive_rejects=10,
    enable_stability=False,
    allow_all_orientations=False,
)

def run_experiment_with_gif(strategy_name: str, strategy_type: str, output_dir: str, n_boxes: int = 30):
    """Run a single experiment and generate GIF."""
    print(f"  Generating GIF for {strategy_name}...")

    # Generate boxes
    boxes = generate_rajapack(n_boxes, seed=42)

    # Get strategy
    if strategy_type == "single_bin":
        from strategies.base_strategy import get_strategy
        strategy = get_strategy(strategy_name)
        box_selector = get_box_selector("default")
        bin_selector = get_bin_selector("emptiest_first")
    else:  # multi_bin
        from strategies.base_strategy import get_multibin_strategy
        strategy = get_multibin_strategy(strategy_name)
        box_selector = None
        bin_selector = None

    # Create session
    session = PackingSession(BOTKO_SESSION_CONFIG)

    # Capture steps
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
        except Exception as e:
            print(f"    Warning: Could not capture step {step_num}: {e}")

    # Run packing
    try:
        start = time.time()
        if strategy_type == "single_bin":
            result = session.run(boxes, strategy, box_selector, bin_selector, on_step=on_step)
        else:
            result = session.run(boxes, strategy, on_step=on_step)
        elapsed = time.time() - start

        # Get results
        pallets_closed = len(session.closed_pallets)
        if pallets_closed > 0:
            avg_fill = sum(p.fill_rate for p in session.closed_pallets) / pallets_closed
        else:
            avg_fill = 0.0

        print(f"    ✓ Packed {n_boxes} boxes in {elapsed:.1f}s: {pallets_closed} pallets, {avg_fill:.1%} fill")

        # Generate GIF
        if conveyor_steps:
            gif_path = os.path.join(output_dir, f"{strategy_name}.gif")
            create_conveyor_gif(
                conveyor_steps,
                BOTKO_PALLET,
                save_path=gif_path,
                title=strategy_name,
                fps=2,
            )
            print(f"    ✓ GIF saved: {gif_path}")
            return True
        else:
            print(f"    ✗ No steps captured")
            return False

    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def main():
    print("="*75)
    print("  STRATEGY GIF GENERATOR FOR SLIDES")
    print("="*75)
    print(f"  Running with LOW PRIORITY (nice 19) - max ~10% CPU")
    print(f"  Using {30} boxes per strategy for clear visualization")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Set low priority
    try:
        os.nice(19)
        print("  ✓ Process priority set to nice 19 (background)")
    except (OSError, AttributeError) as e:
        print(f"  Warning: Could not set nice level: {e}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"strategy_gifs_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output: {output_dir}\n")

    # Get all strategies
    single_bin_strategies = list(STRATEGY_REGISTRY.keys())
    multi_bin_strategies = list(MULTIBIN_STRATEGY_REGISTRY.keys())

    total_strategies = len(single_bin_strategies) + len(multi_bin_strategies)
    print(f"  Generating GIFs for {total_strategies} strategies:")
    print(f"    Single-bin: {len(single_bin_strategies)}")
    print(f"    Multi-bin: {len(multi_bin_strategies)}")
    print()

    # Process single-bin strategies
    success_count = 0
    print("[1/2] Single-bin strategies:")
    for i, strat in enumerate(single_bin_strategies, 1):
        print(f"  [{i}/{len(single_bin_strategies)}] {strat}")
        if run_experiment_with_gif(strat, "single_bin", output_dir):
            success_count += 1
        time.sleep(1)  # Brief pause between experiments

    # Process multi-bin strategies
    print(f"\n[2/2] Multi-bin strategies:")
    for i, strat in enumerate(multi_bin_strategies, 1):
        print(f"  [{i}/{len(multi_bin_strategies)}] {strat}")
        if run_experiment_with_gif(strat, "multi_bin", output_dir):
            success_count += 1
        time.sleep(1)

    print()
    print("="*75)
    print(f"  COMPLETE: {success_count}/{total_strategies} GIFs generated")
    print(f"  Output: {output_dir}/")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*75)


if __name__ == "__main__":
    main()
