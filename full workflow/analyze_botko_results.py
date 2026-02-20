"""
Botko BV Results Analyzer — thesis-quality charts and tables.

Reads results.json from run_overnight_botko.py and produces:
  1. Strategy ranking bar chart (aggregate + effective fill)
  2. Box selector x bin selector heatmap for top strategies
  3. Per-pallet height profile comparison
  4. Computation time analysis
  5. Stability & roughness comparison
  6. Summary LaTeX table

Usage:
    python analyze_botko_results.py --input output/botko_XXXXXXXX/results.json
    python analyze_botko_results.py --input output/botko_XXXXXXXX/results.json --format pdf
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS_SINGLE = "#2E86AB"
COLORS_MULTI = "#A23B72"
COLORS_EFFECTIVE = "#F18F01"


def load_results(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Strategy Ranking
# ─────────────────────────────────────────────────────────────────────────────

def plot_strategy_ranking(data: dict, out_dir: str, fmt: str):
    """Bar chart: all strategies ranked by avg closed-pallet fill."""
    phase1 = data["phase1_baseline"]
    if not phase1:
        print("  No Phase 1 data to plot.")
        return

    # Aggregate by strategy
    strat_agg: Dict[str, List[float]] = defaultdict(list)
    strat_eff: Dict[str, List[float]] = defaultdict(list)
    strat_placed: Dict[str, List[float]] = defaultdict(list)
    strat_pals: Dict[str, List[int]] = defaultdict(list)
    strat_type: Dict[str, str] = {}

    for r in phase1:
        s = r["strategy"]
        strat_agg[s].append(r.get("avg_closed_fill", r.get("aggregate_fill", 0)))
        strat_eff[s].append(r.get("avg_closed_effective_fill", r.get("aggregate_effective_fill", 0)))
        strat_placed[s].append(r["total_placed"])
        strat_pals[s].append(r.get("pallets_closed", 0))
        strat_type[s] = r.get("strategy_type", "single_bin")

    # Sort by mean closed-pallet fill descending
    ranked = sorted(strat_agg.keys(), key=lambda s: np.mean(strat_agg[s]), reverse=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    x = np.arange(len(ranked))
    width = 0.35

    agg_means = [np.mean(strat_agg[s]) * 100 for s in ranked]
    agg_stds = [np.std(strat_agg[s]) * 100 for s in ranked]
    eff_means = [np.mean(strat_eff[s]) * 100 for s in ranked]
    eff_stds = [np.std(strat_eff[s]) * 100 for s in ranked]

    colors = [COLORS_MULTI if strat_type.get(s) == "multi_bin" else COLORS_SINGLE for s in ranked]

    bars1 = ax1.bar(x - width / 2, agg_means, width, yerr=agg_stds, capsize=3,
                     color=colors, alpha=0.8, label="Volumetric Fill (L*W*H)")
    bars2 = ax1.bar(x + width / 2, eff_means, width, yerr=eff_stds, capsize=3,
                     color=COLORS_EFFECTIVE, alpha=0.7, label="Effective Fill (L*W*max_h)")

    ax1.set_ylabel("Fill Rate (%)")
    ax1.set_title("Strategy Ranking: Avg Closed-Pallet Fill (Botko BV)")
    ax1.legend(loc="upper right")

    # Add value labels
    for bar, val in zip(bars1, agg_means):
        if val > 1:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=6)

    # Bottom: placed boxes
    placed_means = [np.mean(strat_placed[s]) for s in ranked]
    ax2.bar(x, placed_means, width * 2, color=colors, alpha=0.6)
    ax2.set_ylabel("Boxes Placed")
    ax2.set_xlabel("Strategy")

    ax2.set_xticks(x)
    ax2.set_xticklabels(ranked, rotation=45, ha="right")

    legend_elements = [
        Patch(facecolor=COLORS_SINGLE, alpha=0.8, label="Single-bin strategy"),
        Patch(facecolor=COLORS_MULTI, alpha=0.8, label="Multi-bin strategy"),
    ]
    ax1.legend(handles=legend_elements + ax1.get_legend_handles_labels()[0][-1:],
               loc="upper right")

    plt.tight_layout()
    path = os.path.join(out_dir, f"01_strategy_ranking.{fmt}")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parameter Sweep Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_sweep_heatmap(data: dict, out_dir: str, fmt: str):
    """Heatmap: box_selector x bin_selector for each top strategy."""
    phase2 = data.get("phase2_sweep", [])
    if not phase2:
        print("  No Phase 2 data to plot.")
        return

    # Group by strategy
    strat_data: Dict[str, Dict[Tuple[str, str], List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in phase2:
        key = (r["box_selector"], r["bin_selector"])
        strat_data[r["strategy"]][key].append(r.get("avg_closed_fill", r.get("aggregate_fill", 0)))

    strategies = sorted(strat_data.keys())
    if not strategies:
        return

    n_strats = len(strategies)
    fig, axes = plt.subplots(1, n_strats, figsize=(5 * n_strats, 4), squeeze=False)

    for idx, strat in enumerate(strategies):
        ax = axes[0, idx]
        cells = strat_data[strat]

        box_sels = sorted(set(k[0] for k in cells.keys()))
        bin_sels = sorted(set(k[1] for k in cells.keys()))

        matrix = np.zeros((len(box_sels), len(bin_sels)))
        for i, bs in enumerate(box_sels):
            for j, bns in enumerate(bin_sels):
                vals = cells.get((bs, bns), [0])
                matrix[i, j] = np.mean(vals) * 100

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(bin_sels)))
        ax.set_xticklabels([s.replace("_", "\n") for s in bin_sels], fontsize=7)
        ax.set_yticks(range(len(box_sels)))
        ax.set_yticklabels([s.replace("_", "\n") for s in box_sels], fontsize=7)
        ax.set_xlabel("Bin Selector")
        ax.set_ylabel("Box Selector")
        ax.set_title(strat.replace("_", " ").title(), fontsize=9)

        for i in range(len(box_sels)):
            for j in range(len(bin_sels)):
                ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center",
                        fontsize=8, color="white" if matrix[i, j] > matrix.mean() else "black")

        fig.colorbar(im, ax=ax, shrink=0.8, label="Fill %")

    fig.suptitle("Phase 2: Parameter Sweep (box selector x bin selector)", fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f"02_sweep_heatmap.{fmt}")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Height Profile
# ─────────────────────────────────────────────────────────────────────────────

def plot_height_profile(data: dict, out_dir: str, fmt: str):
    """Closed-pallet height distribution per strategy."""
    phase1 = data["phase1_baseline"]
    if not phase1:
        return

    strat_heights: Dict[str, List[float]] = defaultdict(list)
    for r in phase1:
        s = r["strategy"]
        for cp in r.get("closed_pallets", []):
            strat_heights[s].append(cp.get("max_height", 0))

    if not any(strat_heights.values()):
        print("  No closed pallets — skipping height profile.")
        return

    ranked = sorted(strat_heights.keys(),
                    key=lambda s: np.mean(strat_heights[s]) if strat_heights[s] else 0,
                    reverse=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ranked))

    means = [np.mean(strat_heights[s]) if strat_heights[s] else 0 for s in ranked]
    stds = [np.std(strat_heights[s]) if len(strat_heights[s]) > 1 else 0 for s in ranked]

    ax.bar(x, means, yerr=stds, capsize=3, color="#2E86AB", alpha=0.8)

    ax.axhline(y=1800, color="red", linestyle="--", alpha=0.5, label="Close threshold (1800mm)")
    ax.axhline(y=2200, color="orange", linestyle="--", alpha=0.5, label="Robot reach (2200mm)")

    ax.set_ylabel("Avg Closed-Pallet Height (mm)")
    ax.set_title("Closed-Pallet Height Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(ranked, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, f"03_height_profile.{fmt}")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Computation Time
# ─────────────────────────────────────────────────────────────────────────────

def plot_computation_time(data: dict, out_dir: str, fmt: str):
    """Per-box computation time across strategies."""
    phase1 = data["phase1_baseline"]
    if not phase1:
        return

    strat_ms: Dict[str, List[float]] = defaultdict(list)
    for r in phase1:
        strat_ms[r["strategy"]].append(r.get("ms_per_box", 0))

    ranked = sorted(strat_ms.keys(), key=lambda s: np.mean(strat_ms[s]))

    fig, ax = plt.subplots(figsize=(10, 5))
    means = [np.mean(strat_ms[s]) for s in ranked]
    stds = [np.std(strat_ms[s]) for s in ranked]

    colors = ["#A23B72" if np.mean(strat_ms[s]) > 1000 else "#2E86AB" for s in ranked]
    ax.barh(range(len(ranked)), means, xerr=stds, capsize=3, color=colors, alpha=0.8)
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(ranked)
    ax.set_xlabel("ms per box (lower is better)")
    ax.set_title("Computation Time per Box")

    # Add value labels
    for i, (m, s) in enumerate(zip(means, ranked)):
        ax.text(m + 5, i, f"{m:.0f}ms", va="center", fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, f"04_computation_time.{fmt}")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Stability & Roughness
# ─────────────────────────────────────────────────────────────────────────────

def plot_stability_roughness(data: dict, out_dir: str, fmt: str):
    """Scatter: mean support ratio vs surface roughness, sized by fill rate."""
    phase1 = data["phase1_baseline"]
    if not phase1:
        return

    strat_support: Dict[str, List[float]] = defaultdict(list)
    strat_rough: Dict[str, List[float]] = defaultdict(list)
    strat_fill: Dict[str, List[float]] = defaultdict(list)

    for r in phase1:
        s = r["strategy"]
        for cp in r.get("closed_pallets", []):
            strat_support[s].append(cp.get("support_mean", 1.0))
            strat_rough[s].append(cp.get("surface_roughness", 0.0))
            strat_fill[s].append(cp.get("fill_rate", 0.0))

    if not any(strat_support.values()):
        print("  No closed pallets — skipping stability plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for s in strat_support:
        sup_mean = np.mean(strat_support[s])
        rough_mean = np.mean(strat_rough[s])
        fill_mean = np.mean(strat_fill[s])
        size = max(50, fill_mean * 5000)
        ax.scatter(rough_mean, sup_mean, s=size, alpha=0.7, edgecolors="black", linewidth=0.5)
        ax.annotate(s.replace("_", " "), (rough_mean, sup_mean),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    ax.set_xlabel("Surface Roughness (lower = smoother)")
    ax.set_ylabel("Mean Support Ratio (higher = more stable)")
    ax.set_title("Stability vs Surface Quality (bubble size = fill rate)")

    plt.tight_layout()
    path = os.path.join(out_dir, f"05_stability_roughness.{fmt}")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary_table(data: dict, out_dir: str):
    """Generate a plain-text and LaTeX summary table."""
    phase1 = data["phase1_baseline"]
    if not phase1:
        return

    strat_data: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for r in phase1:
        s = r["strategy"]
        strat_data[s]["agg_fill"].append(r.get("avg_closed_fill", r.get("aggregate_fill", 0)))
        strat_data[s]["eff_fill"].append(r.get("avg_closed_effective_fill", r.get("aggregate_effective_fill", 0)))
        strat_data[s]["placed"].append(r["total_placed"])
        strat_data[s]["pallets"].append(r.get("pallets_closed", 0))
        strat_data[s]["ms"].append(r.get("ms_per_box", 0))
        strat_data[s]["type"].append(r.get("strategy_type", "single_bin"))

    ranked = sorted(strat_data.keys(),
                    key=lambda s: np.mean(strat_data[s]["agg_fill"]),
                    reverse=True)

    # Plain text
    header = f"{'#':>3} {'Strategy':<32} {'Type':>10} {'Fill%':>7} {'Eff%':>7} {'Placed':>7} {'Pals':>5} {'ms/box':>8}"
    sep = "-" * len(header)
    lines = [header, sep]

    for i, s in enumerate(ranked, 1):
        d = strat_data[s]
        stype = d["type"][0] if d["type"] else "?"
        lines.append(
            f"{i:>3} {s:<32} {stype:>10} "
            f"{np.mean(d['agg_fill'])*100:>6.2f}% "
            f"{np.mean(d['eff_fill'])*100:>6.1f}% "
            f"{np.mean(d['placed']):>7.0f} "
            f"{sum(d['pallets']):>5} "
            f"{np.mean(d['ms']):>8.0f}"
        )

    table_txt = "\n".join(lines)

    txt_path = os.path.join(out_dir, "06_summary_table.txt")
    with open(txt_path, "w") as f:
        f.write(table_txt)
    print(f"  Saved: {txt_path}")

    # LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Strategy ranking on Botko BV dual-pallet setup (closed pallets only)}",
        r"\label{tab:botko-ranking}",
        r"\begin{tabular}{rlrrrrrr}",
        r"\toprule",
        r"\# & Strategy & Type & Fill (\%) & Eff.\ Fill (\%) & Placed & Pallets & ms/box \\",
        r"\midrule",
    ]
    for i, s in enumerate(ranked, 1):
        d = strat_data[s]
        stype = "multi" if d["type"][0] == "multi_bin" else "single"
        name_escaped = s.replace("_", r"\_")
        latex_lines.append(
            f"  {i} & {name_escaped} & {stype} & "
            f"{np.mean(d['agg_fill'])*100:.2f} & "
            f"{np.mean(d['eff_fill'])*100:.1f} & "
            f"{np.mean(d['placed']):.0f} & "
            f"{sum(d['pallets'])} & "
            f"{np.mean(d['ms']):.0f} \\\\"
        )
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex_path = os.path.join(out_dir, "06_summary_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"  Saved: {tex_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze Botko overnight results")
    parser.add_argument("--input", required=True, help="Path to results.json")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                        help="Output image format")
    args = parser.parse_args()

    data = load_results(args.input)
    out_dir = os.path.dirname(os.path.abspath(args.input))

    meta = data.get("metadata", {})
    print(f"\nAnalyzing: {args.input}")
    print(f"  Smoke test: {meta.get('smoke_test', '?')}")
    print(f"  Datasets: {meta.get('n_datasets', '?')} x {meta.get('n_boxes', '?')} boxes")
    print(f"  Phase 1 entries: {len(data.get('phase1_baseline', []))}")
    print(f"  Phase 2 entries: {len(data.get('phase2_sweep', []))}")
    print()

    plot_strategy_ranking(data, out_dir, args.format)
    plot_sweep_heatmap(data, out_dir, args.format)
    plot_height_profile(data, out_dir, args.format)
    plot_computation_time(data, out_dir, args.format)
    plot_stability_roughness(data, out_dir, args.format)
    generate_summary_table(data, out_dir)

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
