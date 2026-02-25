"""
Evaluation and comparison script for the PPO bin packing agent.

Provides:
    1. Load a trained checkpoint and run evaluation episodes
    2. Compare PPO performance against heuristic baselines
    3. Generate thesis-quality plots (fill distribution, learning curves,
       strategy comparison bar charts)

Usage:
    python evaluate.py --checkpoint outputs/rl_ppo/logs/checkpoints/best_model.pt
    python evaluate.py --checkpoint best_model.pt --num_episodes 100
    python evaluate.py --checkpoint best_model.pt --compare baseline walle_scoring surface_contact
    python evaluate.py --checkpoint best_model.pt --compare_all

The comparison mode runs the same box sequences through both the PPO agent
and the specified heuristic strategies, producing paired-sample statistics.
"""

from __future__ import annotations

import sys
import os
import argparse
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

import torch

from config import Box, BinConfig
from strategies.rl_ppo.config import PPOConfig
from strategies.rl_ppo.network import ActorCritic
from strategies.rl_ppo.train import (
    obs_to_tensor, compute_action_masks, decomposed_to_flat_action,
    load_checkpoint, _make_env_config,
)
from strategies.rl_common.environment import (
    BinPackingEnv, EnvConfig, generate_random_boxes,
)
from simulator.session import PackingSession, SessionConfig
from simulator.close_policy import HeightClosePolicy


# ─────────────────────────────────────────────────────────────────────────────
# PPO evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ppo(
    model: ActorCritic,
    config: PPOConfig,
    device: torch.device,
    num_episodes: int = 100,
    seed: int = 12345,
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """
    Evaluate the PPO agent over multiple episodes.

    Each episode uses a fresh random box sequence.  The agent uses
    deterministic (greedy) action selection.

    Args:
        model:        Trained actor-critic network.
        config:       PPO configuration.
        device:       Compute device.
        num_episodes: Number of evaluation episodes.
        seed:         Base random seed.
        verbose:      Print per-episode results.

    Returns:
        Dict with lists: 'fill_rates', 'returns', 'placement_rates',
        'pallets_closed', 'episode_lengths'.
    """
    model.eval()
    env_config = _make_env_config(config)
    env_config.seed = seed
    env = BinPackingEnv(config=env_config)

    results: Dict[str, List[float]] = {
        'fill_rates': [],
        'returns': [],
        'placement_rates': [],
        'pallets_closed': [],
        'episode_lengths': [],
    }

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_return = 0.0
        steps = 0
        done = False

        while not done:
            with torch.no_grad():
                obs_t = obs_to_tensor(obs, device, config)
                masks = compute_action_masks(obs, config, device)
                actions, _, _, _ = model(obs_t, action_masks=masks, deterministic=True)

            flat = decomposed_to_flat_action(actions, config)
            action = int(flat[0])
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            steps += 1
            done = terminated or truncated

        fill = info.get('final_avg_fill', 0.0)
        pr = info.get('placement_rate', 0.0)
        pc = info.get('pallets_closed', 0)

        results['fill_rates'].append(fill)
        results['returns'].append(ep_return)
        results['placement_rates'].append(pr)
        results['pallets_closed'].append(float(pc))
        results['episode_lengths'].append(float(steps))

        if verbose:
            print(f"  Episode {ep+1:4d}/{num_episodes}: "
                  f"fill={fill:.3f}, return={ep_return:.2f}, "
                  f"placed={pr:.1%}, closed={pc}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic baseline evaluation (for comparison)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_heuristic(
    strategy_name: str,
    config: PPOConfig,
    num_episodes: int = 100,
    seed: int = 12345,
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """
    Evaluate a heuristic strategy using PackingSession.

    Uses the same box sequences (same seed) as PPO evaluation
    for paired comparison.

    Args:
        strategy_name: Name of a registered BaseStrategy.
        config:        PPO config (for bin dimensions).
        num_episodes:  Number of episodes.
        seed:          Base random seed.
        verbose:       Print per-episode results.

    Returns:
        Same format as evaluate_ppo().
    """
    from strategies.base_strategy import get_strategy

    results: Dict[str, List[float]] = {
        'fill_rates': [],
        'returns': [],
        'placement_rates': [],
        'pallets_closed': [],
        'episode_lengths': [],
    }

    bin_config = BinConfig(
        length=config.bin_length,
        width=config.bin_width,
        height=config.bin_height,
        resolution=config.bin_resolution,
    )
    session_config = SessionConfig(
        bin_config=bin_config,
        num_bins=config.num_bins,
        buffer_size=config.buffer_size,
        pick_window=config.pick_window,
        close_policy=HeightClosePolicy(config.close_height),
        max_consecutive_rejects=config.max_consecutive_rejects,
    )

    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        # Generate same boxes as PPO (same seed progression)
        ep_rng = np.random.default_rng(seed + ep)
        boxes = generate_random_boxes(
            n=config.num_boxes_per_episode,
            size_range=config.box_size_range,
            weight_range=config.box_weight_range,
            rng=ep_rng,
        )

        strategy = get_strategy(strategy_name)
        session = PackingSession(session_config)
        result = session.run(boxes, strategy)

        results['fill_rates'].append(result.avg_closed_fill)
        results['returns'].append(0.0)  # No RL return for heuristics
        results['placement_rates'].append(result.placement_rate)
        results['pallets_closed'].append(float(result.pallets_closed))
        results['episode_lengths'].append(float(result.total_placed + result.total_rejected))

        if verbose:
            print(f"  [{strategy_name}] Episode {ep+1:4d}/{num_episodes}: "
                  f"fill={result.avg_closed_fill:.3f}, "
                  f"placed={result.placement_rate:.1%}, "
                  f"closed={result.pallets_closed}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(
    all_results: Dict[str, Dict[str, List[float]]],
    output_dir: str,
) -> None:
    """
    Generate thesis-quality comparison plots.

    Creates:
        1. Bar chart: mean fill rate per strategy with error bars
        2. Box plot: fill rate distributions
        3. Bar chart: placement rate comparison
        4. Fill rate histogram overlay

    Args:
        all_results: {strategy_name: evaluation_results_dict}
        output_dir:  Directory to save plots.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except ImportError:
        print("[evaluate] matplotlib not available -- skipping plots")
        return

    # Thesis-quality style
    rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })

    os.makedirs(output_dir, exist_ok=True)

    names = list(all_results.keys())
    fills = {n: all_results[n]['fill_rates'] for n in names}
    placements = {n: all_results[n]['placement_rates'] for n in names}

    # Colour palette (colourblind-safe)
    colours = ['#4C72B0', '#55A868', '#C44E52', '#8172B2',
               '#CCB974', '#64B5CD', '#8C8C8C', '#E88C30']

    # ── 1. Bar chart: mean fill rate ──
    fig, ax = plt.subplots(figsize=(10, 5))
    means = [np.mean(fills[n]) for n in names]
    stds = [np.std(fills[n]) for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=colours[:len(names)], edgecolor='black', linewidth=0.5,
                  alpha=0.85, error_kw={'linewidth': 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Average Closed Fill Rate')
    ax.set_title('Strategy Comparison: Fill Rate')
    ax.set_ylim(0, min(1.0, max(means) * 1.3 + 0.05))
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fill_rate_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 2. Box plot: fill rate distributions ──
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [fills[n] for n in names]
    bp = ax.boxplot(data, labels=names, patch_artist=True, notch=True,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch, colour in zip(bp['boxes'], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)
    ax.set_ylabel('Average Closed Fill Rate')
    ax.set_title('Fill Rate Distribution by Strategy')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fill_rate_boxplot.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 3. Placement rate comparison ──
    fig, ax = plt.subplots(figsize=(10, 5))
    pr_means = [np.mean(placements[n]) * 100 for n in names]
    pr_stds = [np.std(placements[n]) * 100 for n in names]
    bars = ax.bar(x, pr_means, yerr=pr_stds, capsize=4,
                  color=colours[:len(names)], edgecolor='black', linewidth=0.5,
                  alpha=0.85, error_kw={'linewidth': 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Placement Rate (%)')
    ax.set_title('Strategy Comparison: Placement Rate')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    for bar, mean in zip(bars, pr_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'placement_rate_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 4. Fill rate histograms (overlay) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(names):
        ax.hist(fills[name], bins=30, alpha=0.4, color=colours[i % len(colours)],
                label=f'{name} (mean={np.mean(fills[name]):.3f})',
                edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Average Closed Fill Rate')
    ax.set_ylabel('Count')
    ax.set_title('Fill Rate Distribution Overlay')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fill_rate_histogram.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[evaluate] Plots saved to: {output_dir}")


def print_summary(
    all_results: Dict[str, Dict[str, List[float]]],
) -> None:
    """Print a formatted summary table of all strategy results."""
    print("\n" + "=" * 80)
    print(f"{'Strategy':<25} {'Fill Rate':>12} {'Std':>8} "
          f"{'Place%':>8} {'Closed':>8}")
    print("-" * 80)

    for name, res in sorted(all_results.items(),
                            key=lambda x: np.mean(x[1]['fill_rates']),
                            reverse=True):
        fill_mean = np.mean(res['fill_rates'])
        fill_std = np.std(res['fill_rates'])
        pr_mean = np.mean(res['placement_rates']) * 100
        closed_mean = np.mean(res['pallets_closed'])
        print(f"{name:<25} {fill_mean:>12.4f} {fill_std:>8.4f} "
              f"{pr_mean:>7.1f}% {closed_mean:>8.1f}")

    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare PPO bin packing agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed for evaluation")
    parser.add_argument("--compare", nargs="*", default=[],
                        help="Heuristic strategies to compare against "
                             "(e.g., baseline walle_scoring surface_contact)")
    parser.add_argument("--compare_all", action="store_true",
                        help="Compare against all registered strategies")
    parser.add_argument("--output_dir", type=str, default="outputs/rl_ppo/evaluation",
                        help="Output directory for plots and results")
    parser.add_argument("--device", type=str, default="auto",
                        help="PyTorch device")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-episode results")

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Load model
    print(f"[evaluate] Loading checkpoint: {args.checkpoint}")
    config = PPOConfig()
    model, ckpt_info = load_checkpoint(args.checkpoint, config, device)
    print(f"[evaluate] Checkpoint info: update={ckpt_info.get('update', '?')}, "
          f"global_step={ckpt_info.get('global_step', '?')}, "
          f"best_fill={ckpt_info.get('best_fill', '?')}")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_results: Dict[str, Dict[str, List[float]]] = {}

    # Evaluate PPO
    print(f"\n[evaluate] Running PPO evaluation ({args.num_episodes} episodes)...")
    ppo_results = evaluate_ppo(
        model, config, device,
        num_episodes=args.num_episodes,
        seed=args.seed,
        verbose=args.verbose,
    )
    all_results['rl_ppo'] = ppo_results
    print(f"  PPO fill: {np.mean(ppo_results['fill_rates']):.4f} "
          f"+/- {np.std(ppo_results['fill_rates']):.4f}")

    # Determine which heuristics to compare
    compare_strategies = list(args.compare)
    if args.compare_all:
        from strategies.base_strategy import STRATEGY_REGISTRY
        compare_strategies = sorted(STRATEGY_REGISTRY.keys())
        # Exclude very slow strategies if doing many episodes
        if args.num_episodes > 20:
            slow = {'lbcp_stability', 'pct_expansion'}
            compare_strategies = [s for s in compare_strategies if s not in slow]

    # Evaluate heuristics
    for strategy_name in compare_strategies:
        print(f"\n[evaluate] Running {strategy_name} ({args.num_episodes} episodes)...")
        try:
            h_results = evaluate_heuristic(
                strategy_name, config,
                num_episodes=args.num_episodes,
                seed=args.seed,
                verbose=args.verbose,
            )
            all_results[strategy_name] = h_results
            print(f"  {strategy_name} fill: {np.mean(h_results['fill_rates']):.4f} "
                  f"+/- {np.std(h_results['fill_rates']):.4f}")
        except Exception as e:
            print(f"  [WARNING] Failed to evaluate {strategy_name}: {e}")

    # Summary
    print_summary(all_results)

    # Save raw results as JSON
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    json_results = {
        name: {k: [float(v) for v in vals] for k, vals in res.items()}
        for name, res in all_results.items()
    }
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n[evaluate] Results saved to: {results_path}")

    # Plots
    if len(all_results) > 1:
        plot_comparison(all_results, output_dir)
    else:
        # Single strategy plots
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(ppo_results['fill_rates'], bins=30, color='steelblue',
                    edgecolor='black', alpha=0.8)
            ax.axvline(np.mean(ppo_results['fill_rates']), color='red',
                       linestyle='--', linewidth=2,
                       label=f"Mean: {np.mean(ppo_results['fill_rates']):.3f}")
            ax.set_xlabel('Average Closed Fill Rate')
            ax.set_ylabel('Count')
            ax.set_title('PPO Agent: Fill Rate Distribution')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, 'ppo_fill_distribution.png'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        except ImportError:
            pass


if __name__ == "__main__":
    main()
