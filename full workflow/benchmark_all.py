"""
Benchmark all registered strategies with 50 boxes.

Single-bin: all BaseStrategy implementations via PipelineSimulator.
Multi-bin:  all MultiBinStrategy implementations via MultiBinPipeline.

Usage:
    python benchmark_all.py
    python benchmark_all.py --n_boxes 50 --seed 42
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Box, BinConfig, ExperimentConfig
from dataset.generator import generate_uniform
from simulator.pipeline_simulator import PipelineSimulator
from simulator.multi_bin_pipeline import MultiBinPipeline, PipelineConfig
from simulator.buffer import BufferPolicy
from strategies.base_strategy import get_strategy, STRATEGY_REGISTRY, MULTIBIN_STRATEGY_REGISTRY
import strategies  # registers everything


def run_single_bin(strategy_name: str, boxes, config: ExperimentConfig) -> dict:
    """Run a single-bin strategy and return result dict."""
    try:
        sim = PipelineSimulator(config)
        strategy = get_strategy(strategy_name)
        strategy.on_episode_start(config)
        t0 = time.perf_counter()
        for box in boxes:
            decision = strategy.decide_placement(box, sim.get_bin_state())
            if decision is not None:
                sim.attempt_placement(box, decision.x, decision.y, decision.orientation_idx)
        elapsed = (time.perf_counter() - t0) * 1000
        strategy.on_episode_end(sim.get_summary())
        summary = sim.get_summary()
        return {
            "fill_rate": sim.get_bin_state().get_fill_rate(),
            "placed": summary["boxes_placed"],
            "rejected": summary["boxes_rejected"],
            "time_ms": elapsed,
            "ms_per_box": elapsed / max(len(boxes), 1),
            "error": None,
        }
    except Exception as e:
        return {"fill_rate": 0.0, "placed": 0, "rejected": 0, "time_ms": 0,
                "ms_per_box": 0, "error": str(e)}


def run_multi_bin(strategy_name: str, boxes, pipeline_config: PipelineConfig) -> dict:
    """Run a multi-bin strategy via MultiBinPipeline."""
    from strategies.base_strategy import MULTIBIN_STRATEGY_REGISTRY
    try:
        strategy_cls = MULTIBIN_STRATEGY_REGISTRY[strategy_name]
        strategy = strategy_cls()
        pipeline = MultiBinPipeline(strategy=strategy, config=pipeline_config)
        t0 = time.perf_counter()
        result = pipeline.run(boxes)
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "fill_rate": result.aggregate_fill_rate,
            "mean_fill": result.mean_fill_rate,
            "placed": result.total_placed,
            "rejected": result.total_rejected,
            "bins_used": result.bins_used,
            "time_ms": elapsed,
            "ms_per_box": elapsed / max(len(boxes), 1),
            "error": None,
        }
    except Exception as e:
        return {"fill_rate": 0.0, "mean_fill": 0.0, "placed": 0, "rejected": 0,
                "bins_used": 0, "time_ms": 0, "ms_per_box": 0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boxes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Strategy names to skip (e.g. slow ones)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  FULL STRATEGY BENCHMARK")
    print(f"  {args.n_boxes} boxes  |  seed={args.seed}")
    print(f"  Bin: 120x80x270 cm (EUR pallet), resolution=10cm")
    print(f"{'='*70}\n")

    # Generate shared box dataset
    boxes = generate_uniform(args.n_boxes, seed=args.seed,
                              min_dim=200.0, max_dim=500.0)

    bin_config = BinConfig(
        length=1200.0, width=800.0, height=2700.0, resolution=10.0
    )
    exp_config = ExperimentConfig(
        bin=bin_config,
        strategy_name="benchmark",
        enable_stability=False,
        min_support_ratio=0.8,
        allow_all_orientations=False,
        render_3d=False,
        verbose=False,
    )

    # ── Single-bin strategies ─────────────────────────────────────────────
    print(f"  {'STRATEGY':<30} {'FILL':>6}  {'PLACED':>7}  {'REJECTED':>8}  {'ms/box':>7}  STATUS")
    print(f"  {'-'*30} {'-'*6}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*20}")

    single_results = {}
    skipped = set(args.skip)
    # Sort: fast strategies first, known-slow ones at the end
    slow = {"lbcp_stability", "selective_hyper_heuristic", "stacking_tree_stability",
            "blueprint_packing"}
    strategy_order = (
        sorted(k for k in STRATEGY_REGISTRY if k not in slow and k not in skipped) +
        sorted(k for k in STRATEGY_REGISTRY if k in slow and k not in skipped)
    )

    for name in strategy_order:
        exp_config.strategy_name = name
        r = run_single_bin(name, boxes, exp_config)
        single_results[name] = r
        status = f"ERROR: {r['error'][:40]}" if r["error"] else "OK"
        print(
            f"  {name:<30} {r['fill_rate']:>5.1%}  "
            f"{r['placed']:>7d}  {r['rejected']:>8d}  "
            f"{r['ms_per_box']:>7.0f}  {status}"
        )

    # ── Multi-bin strategies ──────────────────────────────────────────────
    print(f"\n  {'MULTI-BIN STRATEGY':<30} {'AGG':>6}  {'MEAN':>6}  "
          f"{'PLACED':>7}  {'BINS':>5}  {'ms/box':>7}  STATUS")
    print(f"  {'-'*30} {'-'*6}  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*20}")

    pipeline_config = PipelineConfig(
        n_bins=2,
        buffer_size=5,
        buffer_policy=BufferPolicy.LARGEST_FIRST,
        bin_config=bin_config,
        enable_stability=False,
        min_support_ratio=0.8,
        allow_all_orientations=False,
    )

    for name in sorted(MULTIBIN_STRATEGY_REGISTRY.keys()):
        if name in skipped:
            continue
        r = run_multi_bin(name, list(boxes), pipeline_config)
        status = f"ERROR: {r['error'][:40]}" if r["error"] else "OK"
        print(
            f"  {name:<30} {r['fill_rate']:>5.1%}  {r.get('mean_fill',0):>5.1%}  "
            f"{r['placed']:>7d}  {r.get('bins_used',0):>5d}  "
            f"{r['ms_per_box']:>7.0f}  {status}"
        )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TOP 5 SINGLE-BIN STRATEGIES (by fill rate):")
    ranked = sorted(single_results.items(),
                    key=lambda kv: kv[1]["fill_rate"], reverse=True)
    for i, (name, r) in enumerate(ranked[:5], 1):
        if r["error"]:
            continue
        print(f"    {i}. {name:<30} {r['fill_rate']:.1%}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
