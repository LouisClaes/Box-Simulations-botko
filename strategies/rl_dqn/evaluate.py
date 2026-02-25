"""
Evaluation script for the trained DDQN bin packing agent.

Loads a trained model checkpoint and runs N evaluation episodes with
no exploration (greedy policy).  Reports comprehensive statistics and
generates comparison plots against heuristic baselines.

Usage:
    # Evaluate best model
    python evaluate.py --checkpoint outputs/rl_dqn/checkpoints/best_network.pt

    # Custom evaluation
    python evaluate.py --checkpoint outputs/rl_dqn/checkpoints/best_network.pt \
                       --episodes 100 --seed 12345

    # Compare with heuristic baselines
    python evaluate.py --checkpoint outputs/rl_dqn/checkpoints/best_network.pt \
                       --compare baseline walle_scoring surface_contact

Output:
    - Console summary with mean/std/min/max for all metrics
    - Results JSON saved to output directory
    - Comparison bar chart (if --compare specified)
    - Per-episode fill rate distribution histogram
"""

from __future__ import annotations

import sys
import os
import argparse
import json
import time
from typing import List, Dict, Any, Optional

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────
_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

import torch

from config import Box, BinConfig, ExperimentConfig
from simulator.session import PackingSession, SessionConfig
from simulator.close_policy import HeightClosePolicy
from strategies.rl_common.environment import generate_random_boxes
from strategies.rl_common.rewards import RewardShaper, RewardConfig

from strategies.rl_dqn.config import DQNConfig
from strategies.rl_dqn.network import DQNNetwork
from strategies.rl_dqn.candidate_generator import CandidateGenerator
from strategies.rl_dqn.train import DQNAgent, run_episode, build_state_tensors


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic baseline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_heuristic_baseline(
    strategy_name: str,
    boxes_list: List[List[Box]],
    bin_config: BinConfig,
    session_config: SessionConfig,
) -> Dict[str, float]:
    """
    Run a heuristic strategy on the same box sequences.

    Args:
        strategy_name: Name of the registered strategy.
        boxes_list:    List of box lists (one per episode).
        bin_config:    Bin configuration.
        session_config: Session configuration.

    Returns:
        Dict with mean fill, std, etc.
    """
    from strategies.base_strategy import get_strategy
    from simulator.session import PackingSession, FIFOBoxSelector

    strategy = get_strategy(strategy_name)
    session = PackingSession(session_config)

    fills = []
    placements = []

    for boxes in boxes_list:
        result = session.run(
            boxes,
            strategy=strategy,
            box_selector=FIFOBoxSelector(),
        )
        fills.append(result.avg_closed_fill)
        placements.append(result.placement_rate)

    return {
        "strategy": strategy_name,
        "fill_mean": float(np.mean(fills)),
        "fill_std": float(np.std(fills)),
        "fill_min": float(np.min(fills)),
        "fill_max": float(np.max(fills)),
        "placement_rate_mean": float(np.mean(placements)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    checkpoint_path: str,
    num_episodes: int = 50,
    seed: int = 12345,
    compare_strategies: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained DDQN model.

    Args:
        checkpoint_path: Path to network checkpoint (.pt file).
        num_episodes:    Number of evaluation episodes.
        seed:            Random seed for reproducible box generation.
        compare_strategies: List of heuristic strategy names to compare with.
        output_dir:      Directory for saving results and plots.

    Returns:
        Results dictionary with all metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model: {checkpoint_path}")

    # ── Load model ────────────────────────────────────────────────────────
    model = DQNNetwork.load(checkpoint_path, device=device)
    config = model.config
    model.eval()
    print(f"Model parameters: {model.count_parameters():,}")

    # ── Setup ─────────────────────────────────────────────────────────────
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
        enable_stability=False,
        allow_all_orientations=(config.num_orientations >= 6),
    )

    # Create agent with loaded weights
    agent = DQNAgent(config, device)
    agent.online_net.load_state_dict(model.state_dict())
    agent.online_net.eval()

    session = PackingSession(session_config)
    reward_shaper = RewardShaper(RewardConfig())

    # ── Generate evaluation box sequences ─────────────────────────────────
    rng = np.random.default_rng(seed)
    boxes_list: List[List[Box]] = []
    for _ in range(num_episodes):
        boxes = generate_random_boxes(
            n=config.num_boxes_per_episode,
            size_range=config.box_size_range,
            weight_range=config.box_weight_range,
            rng=rng,
        )
        boxes_list.append(boxes)

    # ── Run evaluation ────────────────────────────────────────────────────
    print(f"\nRunning {num_episodes} evaluation episodes (no exploration)...")
    t_start = time.time()

    fills = []
    rewards = []
    placement_rates = []
    pallets_closed_list = []
    times_per_box = []

    for ep_idx, boxes in enumerate(boxes_list):
        ep_start = time.time()

        metrics = run_episode(
            agent=agent,
            session=session,
            boxes=boxes,
            bin_config=bin_config,
            reward_shaper=reward_shaper,
            explore=False,
            train=False,
        )

        ep_time = time.time() - ep_start
        ms_per_box = (ep_time / max(metrics["placements"], 1)) * 1000

        fills.append(metrics["fill"])
        rewards.append(metrics["reward"])
        placement_rates.append(metrics["placement_rate"])
        pallets_closed_list.append(metrics["pallets_closed"])
        times_per_box.append(ms_per_box)

        if (ep_idx + 1) % 10 == 0:
            print(
                f"  Episode {ep_idx+1}/{num_episodes} | "
                f"Fill: {metrics['fill']:.4f} | "
                f"Reward: {metrics['reward']:.2f} | "
                f"ms/box: {ms_per_box:.1f}"
            )

    total_time = time.time() - t_start

    # ── Results ───────────────────────────────────────────────────────────
    results = {
        "model": checkpoint_path,
        "num_episodes": num_episodes,
        "seed": seed,
        "device": str(device),
        "fill_mean": float(np.mean(fills)),
        "fill_std": float(np.std(fills)),
        "fill_min": float(np.min(fills)),
        "fill_max": float(np.max(fills)),
        "fill_median": float(np.median(fills)),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "placement_rate_mean": float(np.mean(placement_rates)),
        "pallets_closed_mean": float(np.mean(pallets_closed_list)),
        "ms_per_box_mean": float(np.mean(times_per_box)),
        "ms_per_box_p95": float(np.percentile(times_per_box, 95)),
        "total_time_s": total_time,
        "fills": fills,
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes:           {num_episodes}")
    print(f"  Avg Fill Rate:      {results['fill_mean']:.4f} +/- {results['fill_std']:.4f}")
    print(f"  Fill Range:         [{results['fill_min']:.4f}, {results['fill_max']:.4f}]")
    print(f"  Median Fill:        {results['fill_median']:.4f}")
    print(f"  Avg Reward:         {results['reward_mean']:.2f} +/- {results['reward_std']:.2f}")
    print(f"  Placement Rate:     {results['placement_rate_mean']:.4f}")
    print(f"  Pallets Closed:     {results['pallets_closed_mean']:.1f}")
    print(f"  Speed (ms/box):     {results['ms_per_box_mean']:.1f} (p95: {results['ms_per_box_p95']:.1f})")
    print(f"  Total Time:         {total_time:.1f}s")
    print("=" * 60)

    # ── Compare with baselines ────────────────────────────────────────────
    baseline_results = []
    if compare_strategies:
        print(f"\nRunning baseline comparisons: {compare_strategies}")
        for strat_name in compare_strategies:
            try:
                baseline = run_heuristic_baseline(
                    strat_name, boxes_list, bin_config, session_config,
                )
                baseline_results.append(baseline)
                print(
                    f"  {strat_name:25s} | "
                    f"Fill: {baseline['fill_mean']:.4f} +/- {baseline['fill_std']:.4f}"
                )
            except Exception as e:
                print(f"  {strat_name:25s} | ERROR: {e}")

        results["baselines"] = baseline_results

    # ── Save results ──────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(checkpoint_path), "..", "eval")
    os.makedirs(output_dir, exist_ok=True)

    # JSON (without per-episode fills for compactness)
    results_compact = {k: v for k, v in results.items() if k != "fills"}
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results_compact, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Fill rate histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(fills, bins=30, alpha=0.7, color="steelblue", edgecolor="navy")
        ax.axvline(results["fill_mean"], color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {results['fill_mean']:.4f}")
        ax.set_title("DDQN: Fill Rate Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Average Closed Fill Rate")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "fill_distribution.png"), dpi=150)
        plt.close(fig)

        # Comparison bar chart
        if baseline_results:
            all_strategies = [{"strategy": "rl_dqn", **{
                "fill_mean": results["fill_mean"],
                "fill_std": results["fill_std"],
            }}] + baseline_results

            fig, ax = plt.subplots(figsize=(10, 6))
            names = [s["strategy"] for s in all_strategies]
            means = [s["fill_mean"] for s in all_strategies]
            stds = [s.get("fill_std", 0.0) for s in all_strategies]

            colors = ["steelblue"] + ["lightcoral"] * len(baseline_results)
            bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors,
                          edgecolor="navy", alpha=0.8)

            ax.set_title("Strategy Comparison: Avg Closed Fill Rate",
                         fontsize=14, fontweight="bold")
            ax.set_ylabel("Fill Rate")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis="y")

            # Value labels
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{mean:.3f}", ha="center", fontsize=10, fontweight="bold")

            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, "strategy_comparison.png"), dpi=150)
            plt.close(fig)

        print(f"Plots saved: {plots_dir}")

    except ImportError:
        print("matplotlib not available — skipping plots")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained DDQN bin packing agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--compare", nargs="*", default=None,
        help="Heuristic strategies to compare with (e.g., baseline walle_scoring)",
    )

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        seed=args.seed,
        compare_strategies=args.compare,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
