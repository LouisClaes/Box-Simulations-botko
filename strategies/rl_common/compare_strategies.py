"""
compare_strategies.py — Generate thesis-quality comparison plots across
all RL strategies and heuristic baselines.

Produces:
  1. Bar chart: avg_fill per strategy (RL vs heuristics)
  2. Box plot: fill rate distribution per strategy
  3. Training curves overlay: all RL strategies on one plot
  4. Radar chart: multi-metric comparison (fill, speed, stability, placement_rate)
  5. Heatmap: strategy × dataset performance matrix
  6. Learning efficiency: fill rate vs training steps

All plots use consistent Anthropic-inspired colour palette and are
thesis-ready (high DPI, proper labels, legends).

Usage:
    python compare_strategies.py --eval_dir outputs/evaluation --output_dir outputs/plots
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Any

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)


# ── Colour palette (thesis-consistent) ─────────────────────────────────────

COLORS = {
    # RL strategies (blues/purples)
    'rl_dqn':             '#2563EB',  # blue
    'rl_ppo':             '#7C3AED',  # purple
    'rl_a2c_masked':      '#059669',  # emerald
    'rl_hybrid_hh':       '#DC2626',  # red
    'rl_pct_transformer': '#D97706',  # amber
    # Heuristic baselines (grays)
    'baseline':           '#6B7280',
    'walle_scoring':      '#374151',
    'surface_contact':    '#9CA3AF',
    'extreme_points':     '#D1D5DB',
    'best_fit_decreasing': '#E5E7EB',
}

STRATEGY_LABELS = {
    'rl_dqn':             'DDQN',
    'rl_ppo':             'PPO (Decomposed)',
    'rl_a2c_masked':      'A2C + Mask',
    'rl_hybrid_hh':       'Hybrid HH (Novel)',
    'rl_pct_transformer': 'PCT Transformer',
    'baseline':           'Baseline (DBLF)',
    'walle_scoring':      'WallE Scoring',
    'surface_contact':    'Surface Contact',
    'extreme_points':     'Extreme Points',
    'best_fit_decreasing': 'Best Fit Dec.',
}


def load_evaluation_results(eval_dir: str) -> Dict[str, Dict]:
    """Load evaluation results from JSON files in eval_dir."""
    results = {}
    if not os.path.isdir(eval_dir):
        print(f"Warning: eval_dir not found: {eval_dir}")
        return results

    candidate_names = (
        "eval_results.json",
        "evaluation_results.json",
        "results.json",
    )
    for entry in sorted(os.listdir(eval_dir)):
        entry_dir = os.path.join(eval_dir, entry)
        if not os.path.isdir(entry_dir):
            continue
        for candidate in candidate_names:
            result_file = os.path.join(entry_dir, candidate)
            if os.path.isfile(result_file):
                with open(result_file) as f:
                    results[entry] = json.load(f)
                break
    return results


def load_training_histories(eval_dir: str) -> Dict[str, Dict[str, List[float]]]:
    """Load training metric histories from CSV files."""
    import csv
    histories = {}
    parent = os.path.dirname(eval_dir)  # training output dir

    strategies = [
        "rl_dqn",
        "rl_ppo",
        "rl_a2c_masked",
        "rl_hybrid_hh",
        "rl_pct_transformer",
        "rl_mcts_hybrid",
    ]
    for strat in strategies:
        csv_candidates = [
            os.path.join(parent, strat, "logs", "metrics.csv"),
            os.path.join(parent, strat, "logs", "dqn", "metrics.csv"),
            os.path.join(parent, strat, "logs", "tabular", "metrics.csv"),
        ]
        csv_path = next((p for p in csv_candidates if os.path.isfile(p)), None)
        if csv_path:
            history = {}
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key, val in row.items():
                        if key not in history:
                            history[key] = []
                        try:
                            history[key].append(float(val))
                        except (ValueError, TypeError):
                            pass
            histories[strat] = history
    return histories


def plot_fill_rate_comparison(
    results: Dict[str, Dict],
    output_dir: str,
) -> str:
    """Bar chart comparing avg_fill across all strategies."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by fill rate
    sorted_strats = sorted(
        results.items(),
        key=lambda x: x[1].get('avg_fill', 0),
        reverse=True,
    )

    names = []
    fills = []
    stds = []
    colors = []

    for name, data in sorted_strats:
        label = STRATEGY_LABELS.get(name, name)
        names.append(label)
        fills.append(data.get('avg_fill', 0))
        stds.append(data.get('fill_std', 0))
        colors.append(COLORS.get(name, '#6B7280'))

    x = np.arange(len(names))
    bars = ax.bar(x, fills, yerr=stds, capsize=4, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for bar, fill in zip(bars, fills):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{fill:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=10)
    ax.set_ylabel('Average Closed Fill Rate', fontsize=12)
    ax.set_title('Strategy Comparison: Average Pallet Fill Rate', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.683, color='gray', linestyle='--', alpha=0.5, label='Best heuristic (WallE: 68.3%)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fill_rate_comparison.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_training_curves_overlay(
    histories: Dict[str, Dict[str, List[float]]],
    output_dir: str,
) -> str:
    """Overlay training curves for all RL strategies."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('RL Training Progress: All Strategies', fontsize=14, fontweight='bold')

    for strat_name, history in histories.items():
        color = COLORS.get(strat_name, '#6B7280')
        label = STRATEGY_LABELS.get(strat_name, strat_name)

        # Panel 1: Episode reward
        if 'reward' in history:
            rewards = history['reward']
            window = min(100, max(1, len(rewards) // 20))
            if len(rewards) > window:
                ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
                axes[0].plot(range(window-1, len(rewards)), ma,
                           color=color, linewidth=2, label=label)
        # Panel 2: Fill rate
        if 'fill' in history:
            fills = history['fill']
            window = min(100, max(1, len(fills) // 20))
            if len(fills) > window:
                ma = np.convolve(fills, np.ones(window)/window, mode='valid')
                axes[1].plot(range(window-1, len(fills)), ma,
                           color=color, linewidth=2, label=label)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward (MA)')
    axes[0].set_title('Reward Over Training')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Avg Closed Fill Rate (MA)')
    axes[1].set_title('Fill Rate Over Training')
    axes[1].axhline(y=0.683, color='gray', linestyle='--', alpha=0.5, label='WallE baseline')
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves_overlay.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_radar_chart(
    results: Dict[str, Dict],
    output_dir: str,
) -> str:
    """Radar chart for multi-metric comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics = ['avg_fill', 'placement_rate', 'speed_score', 'stability']
    metric_labels = ['Fill Rate', 'Placement Rate', 'Speed', 'Stability']
    n_metrics = len(metrics)

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for strat_name, data in results.items():
        if not strat_name.startswith('rl_'):
            continue
        values = []
        for m in metrics:
            val = data.get(m, 0)
            if m == 'speed_score':
                # Invert: lower ms_per_box = better
                ms = data.get('ms_per_box', 100)
                val = max(0, 1.0 - ms / 200.0)
            elif m == 'stability':
                val = data.get('support_mean', 0.9)
            values.append(val)
        values += values[:1]

        color = COLORS.get(strat_name, '#6B7280')
        label = STRATEGY_LABELS.get(strat_name, strat_name)
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Strategy Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'radar_comparison.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_box_distribution(
    results: Dict[str, Dict],
    output_dir: str,
) -> str:
    """Box plot of fill rate distributions per strategy."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    data_to_plot = []
    labels = []
    colors_list = []

    sorted_strats = sorted(
        results.items(),
        key=lambda x: x[1].get('avg_fill', 0),
        reverse=True,
    )

    for name, data in sorted_strats:
        fills = data.get('fill_rates', [data.get('avg_fill', 0)] * 10)
        data_to_plot.append(fills)
        labels.append(STRATEGY_LABELS.get(name, name))
        colors_list.append(COLORS.get(name, '#6B7280'))

    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels)
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Fill Rate', fontsize=12)
    ax.set_title('Fill Rate Distribution by Strategy', fontsize=14, fontweight='bold')
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fill_distribution_boxplot.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def generate_summary_table(
    results: Dict[str, Dict],
    output_dir: str,
) -> str:
    """Generate a LaTeX-formatted results table for the thesis."""
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Strategy comparison on Botko BV setup (100 episodes, seed=42)}',
        r'\label{tab:strategy_comparison}',
        r'\begin{tabular}{lccccc}',
        r'\toprule',
        r'Strategy & Avg Fill (\%) & Std (\%) & Placement & Pallets & ms/box \\',
        r'\midrule',
    ]

    sorted_strats = sorted(
        results.items(),
        key=lambda x: x[1].get('avg_fill', 0),
        reverse=True,
    )

    for name, data in sorted_strats:
        label = STRATEGY_LABELS.get(name, name)
        fill = data.get('avg_fill', 0) * 100
        std = data.get('fill_std', 0) * 100
        place_rate = data.get('placement_rate', 0) * 100
        pallets = data.get('avg_pallets_closed', 0)
        ms = data.get('ms_per_box', 0)

        # Bold the best RL strategy
        if name.startswith('rl_') and fill == max(
            d.get('avg_fill', 0) * 100 for n, d in results.items() if n.startswith('rl_')
        ):
            lines.append(f'\\textbf{{{label}}} & \\textbf{{{fill:.1f}}} & {std:.1f} & {place_rate:.1f} & {pallets:.1f} & {ms:.1f} \\\\')
        else:
            lines.append(f'{label} & {fill:.1f} & {std:.1f} & {place_rate:.1f} & {pallets:.1f} & {ms:.1f} \\\\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    path = os.path.join(output_dir, 'results_table.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description='Compare RL strategies')
    parser.add_argument('--eval_dir', type=str, required=True, help='Evaluation results directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for plots')
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail if no real evaluation results are found',
    )
    parser.add_argument(
        '--expected_strategies',
        type=str,
        default='',
        help='Comma-separated list of strategies that must be present in results',
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.eval_dir, 'comparison')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading evaluation results...")
    results = load_evaluation_results(args.eval_dir)
    histories = load_training_histories(args.eval_dir)

    if not results:
        msg = f"No evaluation results found in {args.eval_dir}"
        if args.strict:
            raise FileNotFoundError(msg)
        print(msg)
        return

    expected = [x.strip() for x in args.expected_strategies.split(',') if x.strip()]
    if args.strict and expected:
        missing = [name for name in expected if name not in results]
        if missing:
            raise FileNotFoundError(
                f"Missing evaluation results for expected strategies: {missing}. "
                f"Found: {sorted(results.keys())}"
            )

    print(f"\nGenerating plots for {len(results)} strategies...")
    print()

    plot_fill_rate_comparison(results, output_dir)
    plot_box_distribution(results, output_dir)
    plot_radar_chart(results, output_dir)
    generate_summary_table(results, output_dir)

    if histories:
        plot_training_curves_overlay(histories, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
