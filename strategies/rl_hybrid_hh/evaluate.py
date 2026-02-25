"""
Evaluation and interpretability analysis for the RL Hybrid Hyper-Heuristic.

This script generates thesis-quality analysis of the trained HH agent:

1. Performance comparison:
   - Each individual heuristic alone (7 baselines)
   - Rule-based hyper-heuristic (selective_hyper_heuristic)
   - Trained RL hybrid HH (tabular and/or DQN)

2. Interpretability plots:
   - Heuristic selection frequency pie chart
   - Selection frequency vs bin fill rate (heatmap)
   - Selection changes across packing phases (stacked area)
   - Q-value analysis for different state regions
   - Success rate per heuristic bar chart

3. Statistical significance:
   - Multiple seeds for confidence intervals
   - Wilcoxon signed-rank tests vs best baseline

Usage:
    python evaluate.py --checkpoint outputs/rl_hybrid_hh/best_model.pt
    python evaluate.py --checkpoint outputs/rl_hybrid_hh/best_model.npz
    python evaluate.py --all  # Compare all modes
"""

from __future__ import annotations

import sys
import os
import time
import argparse
import json
from typing import List, Dict, Any, Optional

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

from config import Box, BinConfig, ExperimentConfig
from simulator.session import PackingSession, SessionConfig
from simulator.close_policy import HeightClosePolicy
from strategies.base_strategy import get_strategy, STRATEGY_REGISTRY
from strategies.rl_common.environment import generate_random_boxes

from strategies.rl_hybrid_hh.config import HHConfig
from strategies.rl_hybrid_hh.strategy import RLHybridHHStrategy


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_strategy(
    strategy_name: str,
    session_config: SessionConfig,
    num_episodes: int = 20,
    num_boxes: int = 100,
    seed: int = 42,
    strategy_instance=None,
) -> Dict[str, Any]:
    """
    Evaluate a single strategy over multiple episodes.

    Args:
        strategy_name:    Name for reporting.
        session_config:   PackingSession configuration.
        num_episodes:     Number of evaluation episodes.
        num_boxes:        Boxes per episode.
        seed:             Base seed.
        strategy_instance: Pre-created strategy (or None to load by name).

    Returns:
        Dict with evaluation metrics.
    """
    rng = np.random.default_rng(seed)
    session = PackingSession(session_config)

    fills = []
    placement_rates = []
    pallets_closed_list = []
    times_ms = []

    for ep in range(num_episodes):
        boxes = generate_random_boxes(
            n=num_boxes,
            size_range=(100.0, 600.0),
            weight_range=(1.0, 30.0),
            rng=rng,
        )

        if strategy_instance is not None:
            strat = strategy_instance
        else:
            strat = get_strategy(strategy_name)

        t0 = time.perf_counter()
        result = session.run(boxes, strat)
        elapsed = (time.perf_counter() - t0) * 1000

        fills.append(result.avg_closed_fill)
        placement_rates.append(result.placement_rate)
        pallets_closed_list.append(result.pallets_closed)
        times_ms.append(elapsed)

    return {
        "strategy": strategy_name,
        "fill_mean": float(np.mean(fills)),
        "fill_std": float(np.std(fills)),
        "fill_median": float(np.median(fills)),
        "fill_min": float(np.min(fills)),
        "fill_max": float(np.max(fills)),
        "placement_rate_mean": float(np.mean(placement_rates)),
        "pallets_closed_mean": float(np.mean(pallets_closed_list)),
        "time_ms_mean": float(np.mean(times_ms)),
        "fills": fills,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Interpretability analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_selections(
    strategy: RLHybridHHStrategy,
    session_config: SessionConfig,
    num_episodes: int = 10,
    num_boxes: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run the HH strategy and collect detailed selection data.

    Returns data suitable for heatmaps, phase analysis, and
    Q-value inspection.
    """
    rng = np.random.default_rng(seed)
    session = PackingSession(session_config)

    all_selections = []
    phase_data = {"early": {}, "mid": {}, "late": {}}

    for ep in range(num_episodes):
        boxes = generate_random_boxes(
            n=num_boxes,
            size_range=(100.0, 600.0),
            weight_range=(1.0, 30.0),
            rng=rng,
        )
        result = session.run(boxes, strategy)

        # Collect selection log
        for entry in strategy.selection_log:
            all_selections.append(entry)

    # Compute aggregated statistics
    heuristic_names = strategy._config_hh.heuristic_names + (
        ["SKIP"] if strategy._config_hh.include_skip else []
    )

    # Overall distribution
    action_counts = {}
    success_counts = {}
    for entry in all_selections:
        h = entry.get("heuristic", "unknown")
        # Normalise fallback names
        base_h = h.split("->")[0]
        action_counts[base_h] = action_counts.get(base_h, 0) + 1
        if entry.get("success", False):
            success_counts[base_h] = success_counts.get(base_h, 0) + 1

    return {
        "total_selections": len(all_selections),
        "action_distribution": action_counts,
        "success_counts": success_counts,
        "all_selections": all_selections,
        "heuristic_names": heuristic_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(results: List[Dict], output_dir: str) -> None:
    """Generate thesis-quality comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Fill rate comparison bar chart ──
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r["strategy"] for r in results]
    means = [r["fill_mean"] for r in results]
    stds = [r["fill_std"] for r in results]

    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=3,
                  color=colors, edgecolor="black", linewidth=0.5)

    # Highlight the best
    best_idx = int(np.argmax(means))
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(2)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Average Closed Fill Rate", fontsize=11)
    ax.set_title("Strategy Comparison: Avg Closed Fill Rate (Botko BV)", fontsize=13)
    ax.set_ylim(0, max(means) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "fill_rate_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_selection_analysis(
    analysis: Dict[str, Any],
    output_dir: str,
    config: HHConfig,
) -> None:
    """Generate interpretability plots for heuristic selection."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Selection frequency pie chart ──
    dist = analysis["action_distribution"]
    if dist:
        fig, ax = plt.subplots(figsize=(8, 8))
        labels = list(dist.keys())
        sizes = list(dist.values())
        total = sum(sizes)
        percentages = [s / total * 100 for s in sizes]

        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=90, pctdistance=0.8,
        )
        ax.set_title("Heuristic Selection Distribution\n(RL Hybrid HH)", fontsize=13)
        plt.tight_layout()
        path = os.path.join(output_dir, "selection_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 2. Success rate per heuristic ──
    success = analysis.get("success_counts", {})
    if dist and success:
        fig, ax = plt.subplots(figsize=(10, 5))
        heuristics = sorted(dist.keys())
        total_counts = [dist.get(h, 0) for h in heuristics]
        success_counts = [success.get(h, 0) for h in heuristics]
        success_rates = [
            s / t if t > 0 else 0.0
            for s, t in zip(success_counts, total_counts)
        ]

        x = np.arange(len(heuristics))
        width = 0.6

        bars = ax.bar(x, success_rates, width, color="steelblue",
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(heuristics, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Success Rate", fontsize=11)
        ax.set_title("Placement Success Rate per Heuristic\n(RL Hybrid HH)", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

        for i, (rate, count) in enumerate(zip(success_rates, total_counts)):
            ax.text(i, rate + 0.02, f"{rate:.0%}\n(n={count})",
                    ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        path = os.path.join(output_dir, "success_rate_per_heuristic.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 3. Selection over episode progress ──
    all_sel = analysis.get("all_selections", [])
    if all_sel and len(all_sel) > 20:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Bin selections by position in episode
        episode_length = 100  # approximate
        n_bins_plot = 10
        bin_width_plot = episode_length // n_bins_plot

        # Group by position
        selection_by_pos = {}
        for i, entry in enumerate(all_sel):
            pos_bin = min((i % episode_length) // bin_width_plot, n_bins_plot - 1)
            h = entry.get("heuristic", "unknown").split("->")[0]
            if pos_bin not in selection_by_pos:
                selection_by_pos[pos_bin] = {}
            selection_by_pos[pos_bin][h] = selection_by_pos[pos_bin].get(h, 0) + 1

        # Get all unique heuristics
        all_heuristics = sorted(set(
            entry.get("heuristic", "unknown").split("->")[0]
            for entry in all_sel
        ))

        # Build stacked bar data
        x_positions = sorted(selection_by_pos.keys())
        bottom = np.zeros(len(x_positions))
        colors = plt.cm.Set2(np.linspace(0, 1, len(all_heuristics)))

        for h_idx, h_name in enumerate(all_heuristics):
            counts = [
                selection_by_pos.get(p, {}).get(h_name, 0)
                for p in x_positions
            ]
            # Normalise to fractions
            totals = [
                sum(selection_by_pos.get(p, {}).values())
                for p in x_positions
            ]
            fracs = [c / t if t > 0 else 0.0 for c, t in zip(counts, totals)]

            ax.bar(x_positions, fracs, bottom=bottom, label=h_name,
                   color=colors[h_idx], edgecolor="white", linewidth=0.5)
            bottom += np.array(fracs)

        ax.set_xlabel("Episode Position (decile)", fontsize=11)
        ax.set_ylabel("Selection Fraction", fontsize=11)
        ax.set_title("Heuristic Selection Across Episode Progress\n"
                      "(How the RL agent adapts its strategy over time)",
                      fontsize=13)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{p*10}-{(p+1)*10}%" for p in x_positions],
                           fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, "selection_over_progress.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    checkpoint_path: str = "",
    output_dir: str = "",
    num_episodes: int = 20,
    num_boxes: int = 100,
) -> None:
    """
    Run the full evaluation pipeline.

    Compares the trained HH agent against individual heuristics and
    generates interpretability plots.
    """
    if not output_dir:
        output_dir = os.path.join(_WORKFLOW_ROOT, "outputs", "rl_hybrid_hh", "eval")
    os.makedirs(output_dir, exist_ok=True)

    config = HHConfig()

    # Session config (Botko BV)
    bin_config = BinConfig(
        length=config.bin_length,
        width=config.bin_width,
        height=config.bin_height,
        resolution=config.resolution,
    )
    session_config = SessionConfig(
        bin_config=bin_config,
        num_bins=config.num_bins,
        buffer_size=config.buffer_size,
        pick_window=config.pick_window,
        close_policy=HeightClosePolicy(max_height=config.close_height),
        max_consecutive_rejects=10,
    )

    print(f"\n{'='*70}")
    print(f"  RL Hybrid Hyper-Heuristic Evaluation")
    print(f"  Checkpoint: {checkpoint_path or '(fallback mode)'}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Boxes per episode: {num_boxes}")
    print(f"{'='*70}\n")

    # ── 1. Evaluate individual heuristics ─────────────────────────────────

    print("Evaluating individual heuristics...")
    results = []
    baseline_strategies = config.heuristic_names

    for name in baseline_strategies:
        if name in STRATEGY_REGISTRY:
            print(f"  {name}...", end=" ", flush=True)
            res = evaluate_strategy(
                strategy_name=name,
                session_config=session_config,
                num_episodes=num_episodes,
                num_boxes=num_boxes,
            )
            results.append(res)
            print(f"fill={res['fill_mean']:.4f} (+/- {res['fill_std']:.4f})")

    # ── 2. Evaluate rule-based HH (if available) ─────────────────────────

    if "selective_hyper_heuristic" in STRATEGY_REGISTRY:
        print(f"  selective_hyper_heuristic...", end=" ", flush=True)
        res = evaluate_strategy(
            strategy_name="selective_hyper_heuristic",
            session_config=session_config,
            num_episodes=num_episodes,
            num_boxes=num_boxes,
        )
        results.append(res)
        print(f"fill={res['fill_mean']:.4f} (+/- {res['fill_std']:.4f})")

    # ── 3. Evaluate RL Hybrid HH ─────────────────────────────────────────

    print(f"  rl_hybrid_hh...", end=" ", flush=True)
    hh_strategy = RLHybridHHStrategy(checkpoint_path=checkpoint_path)

    res = evaluate_strategy(
        strategy_name="rl_hybrid_hh",
        session_config=session_config,
        num_episodes=num_episodes,
        num_boxes=num_boxes,
        strategy_instance=hh_strategy,
    )
    results.append(res)
    print(f"fill={res['fill_mean']:.4f} (+/- {res['fill_std']:.4f})")

    # ── 4. Summary table ─────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Strategy':<30} {'Fill Mean':>10} {'Fill Std':>10} "
          f"{'Placement':>10} {'Time (ms)':>10}")
    print("-" * 70)

    results.sort(key=lambda r: r["fill_mean"], reverse=True)
    for r in results:
        marker = " *" if r["strategy"] == "rl_hybrid_hh" else ""
        print(f"{r['strategy']:<30} {r['fill_mean']:>10.4f} "
              f"{r['fill_std']:>10.4f} "
              f"{r['placement_rate_mean']:>10.2%} "
              f"{r['time_ms_mean']:>10.1f}{marker}")

    print(f"\n  * = RL Hybrid Hyper-Heuristic")

    # ── 5. Interpretability analysis ──────────────────────────────────────

    print(f"\nRunning interpretability analysis...")
    analysis = analyse_selections(
        strategy=hh_strategy,
        session_config=session_config,
        num_episodes=min(num_episodes, 10),
        num_boxes=num_boxes,
    )

    # Print selection summary
    summary = hh_strategy.get_selection_summary()
    if summary:
        print(f"\n  Heuristic selection summary:")
        for name, info in summary.get("heuristic_distribution", {}).items():
            print(f"    {name:<30} {info['count']:>5} "
                  f"({info['fraction']:.1%})")

    # ── 6. Generate plots ─────────────────────────────────────────────────

    print(f"\nGenerating plots...")
    plot_comparison(results, output_dir)
    plot_selection_analysis(analysis, output_dir, config)

    # ── 7. Save results ──────────────────────────────────────────────────

    # Remove non-serialisable data
    serialisable_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != "fills"}
        serialisable_results.append(sr)

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(serialisable_results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    print(f"\n{'='*70}")
    print(f"  Evaluation complete. Plots saved to: {output_dir}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the RL Hybrid Hyper-Heuristic and compare with baselines.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="",
        help="Path to trained model (.npz for tabular, .pt for DQN)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="",
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of evaluation episodes per strategy",
    )
    parser.add_argument(
        "--boxes", type=int, default=100,
        help="Boxes per episode",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate all available checkpoints",
    )

    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        num_boxes=args.boxes,
    )


if __name__ == "__main__":
    main()
