"""
Combine the three major Botko overnight Phase 1 runs into CSV summaries
and a LaTeX table.

The selected runs are:
1. 2026-02-28 baseline run with known zero-fill failures
2. 2026-03-01 targeted rerun after fixing the broken strategies
3. 2026-03-01 full rerun across all strategies

Outputs:
- output/botko_last_three_major_runs_report/phase1_overcoupled_rows.csv
- output/botko_last_three_major_runs_report/phase1_overcoupled_strategy_summary.csv
- output/botko_last_three_major_runs_report/phase1_overcoupled_strategy_table.tex
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "botko_last_three_major_runs_report"

RUNS = [
    {
        "label": "broken_baseline",
        "short_label": "Feb28",
        "description": "Pre-fix full run with zero-fill failures",
        "path": ROOT / "output" / "botko_20260228_102631" / "results.json",
    },
    {
        "label": "targeted_fixed",
        "short_label": "Mar01Fix",
        "description": "Targeted rerun after fixing broken strategies",
        "path": ROOT / "output" / "botko_20260301_165923" / "results.json",
    },
    {
        "label": "full_new_seed",
        "short_label": "Mar01Full",
        "description": "Final full rerun across all strategies",
        "path": ROOT / "output" / "botko_20260301_223315" / "results.json",
    },
]

STRATEGY_DISPLAY = {
    "baseline": "Baseline",
    "best_fit_decreasing": "Best-Fit Decreasing",
    "blueprint_packing": "Blueprint Packing",
    "column_fill": "Column Fill",
    "ems": "EMS",
    "extreme_points": "Extreme Points",
    "gopt_heuristic": "GOpt Heuristic",
    "gravity_balanced": "Gravity Balanced",
    "heuristic_160": "Heuristic-160",
    "hybrid_adaptive": "Hybrid Adaptive",
    "layer_building": "Layer Building",
    "lbcp_stability": "LBCP Stability",
    "lookahead": "Lookahead",
    "online_bpp_heuristic": "Online BPP",
    "pct_expansion": "PCT Expansion",
    "pct_macs_heuristic": "PCT-MACS",
    "skyline": "Skyline",
    "stacking_tree_stability": "Stacking-Tree",
    "surface_contact": "Surface Contact",
    "tsang_multibin": "Tsang Multi-Bin",
    "two_bounded_best_fit": "2-Bounded BF",
    "wall_building": "Wall Building",
    "walle_scoring": "WALL-E Scoring",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def pct(value: float) -> float:
    return value * 100.0


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def fmt_secs(value_ms: float) -> str:
    return f"{value_ms / 1000.0:.1f}"


def run_phase1_rows(run_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = load_json(run_cfg["path"])
    meta = data.get("metadata", {})
    rows = []
    for row in data.get("phase1_baseline", []):
        combined = dict(row)
        combined["run_label"] = run_cfg["label"]
        combined["run_short_label"] = run_cfg["short_label"]
        combined["run_description"] = run_cfg["description"]
        combined["run_timestamp"] = meta.get("timestamp", "")
        combined["run_datasets"] = meta.get("n_datasets", "")
        combined["run_shuffles"] = meta.get("n_shuffles", "")
        combined["run_boxes"] = meta.get("n_boxes", "")
        rows.append(combined)
    return rows


def build_corrected_rows(all_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    replacement_strategies = {
        row["strategy"]
        for row in all_rows
        if row["run_label"] == "targeted_fixed"
    }

    corrected_rows: List[Dict[str, Any]] = []
    for row in all_rows:
        if row["run_label"] == "broken_baseline" and row["strategy"] in replacement_strategies:
            continue
        corrected_rows.append(row)
    return corrected_rows


def write_combined_rows_csv(all_rows: List[Dict[str, Any]]) -> Path:
    out_path = OUTPUT_DIR / "phase1_overcoupled_rows.csv"
    fieldnames = [
        "experiment_id",
        "strategy",
        "strategy_type",
        "dataset_id",
        "shuffle_id",
        "box_selector",
        "bin_selector",
        "avg_closed_fill",
        "avg_closed_effective_fill",
        "placement_rate",
        "pallets_closed",
        "total_placed",
        "total_rejected",
        "ms_per_box",
        "elapsed_ms",
        "consecutive_rejects_triggered",
        "remaining_boxes",
        "closed_pallets",
        "active_pallets",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(all_rows, start=1):
            serial = {name: row.get(name, "") for name in fieldnames}
            serial["experiment_id"] = index
            serial["closed_pallets"] = json.dumps(row.get("closed_pallets", []), separators=(",", ":"))
            serial["active_pallets"] = json.dumps(row.get("active_pallets", []), separators=(",", ":"))
            writer.writerow(serial)
    return out_path


def safe_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    avg = safe_mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def compute_rej_per_pallet(rows: List[Dict[str, Any]]) -> float:
    vals = []
    for row in rows:
        pallets_closed = float(row.get("pallets_closed", 0) or 0)
        total_rejected = float(row.get("total_rejected", 0) or 0)
        if pallets_closed > 0:
            vals.append(total_rejected / pallets_closed)
            continue
        active = row.get("active_pallets", []) or []
        if active:
            vals.append(total_rejected / len(active))
    return safe_mean(vals)


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fills = [float(row.get("avg_closed_fill", 0) or 0) for row in rows]
    effective_fills = [float(row.get("avg_closed_effective_fill", 0) or 0) for row in rows]
    placement_rates = [float(row.get("placement_rate", 0) or 0) for row in rows]
    pallets = [float(row.get("pallets_closed", 0) or 0) for row in rows]
    placed = [float(row.get("total_placed", 0) or 0) for row in rows]
    rejected = [float(row.get("total_rejected", 0) or 0) for row in rows]
    ms_per_box = [float(row.get("ms_per_box", 0) or 0) for row in rows]
    nonzero_fills = [value for value in fills if value > 0]
    nonzero_effective_fills = [value for value in effective_fills if value > 0]
    return {
        "tests": len(rows),
        "zero_fill_tests": sum(1 for value in fills if value <= 0),
        "closed_fill_mean_pct": pct(safe_mean(fills)),
        "closed_fill_std_pct": pct(safe_std(fills)),
        "closed_fill_nonzero_mean_pct": pct(safe_mean(nonzero_fills)),
        "effective_fill_mean_pct": pct(safe_mean(effective_fills)),
        "effective_fill_std_pct": pct(safe_std(effective_fills)),
        "effective_fill_nonzero_mean_pct": pct(safe_mean(nonzero_effective_fills)),
        "placement_mean_pct": pct(safe_mean(placement_rates)),
        "placement_std_pct": pct(safe_std(placement_rates)),
        "pallets_closed_mean": safe_mean(pallets),
        "placed_boxes_mean": safe_mean(placed),
        "rejected_boxes_mean": safe_mean(rejected),
        "ms_per_box_mean": safe_mean(ms_per_box),
        "ms_per_box_std": safe_std(ms_per_box),
        "secs_per_box_mean": safe_mean(ms_per_box) / 1000.0,
        "secs_per_box_std": safe_std(ms_per_box) / 1000.0,
        "rej_per_pallet_mean": compute_rej_per_pallet(rows),
    }


def build_summary(all_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_strategy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    strategy_type: Dict[str, str] = {}

    for row in all_rows:
        strategy = row["strategy"]
        by_strategy[strategy].append(row)
        strategy_type[strategy] = row.get("strategy_type", "")

    summaries: List[Dict[str, Any]] = []
    for strategy in sorted(by_strategy.keys()):
        rows = by_strategy[strategy]
        summary: Dict[str, Any] = {
            "strategy": strategy,
            "strategy_display": STRATEGY_DISPLAY.get(strategy, strategy),
            "strategy_type": strategy_type.get(strategy, ""),
        }
        summary.update(summarize_rows(rows))
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            row["closed_fill_mean_pct"],
            row["effective_fill_mean_pct"],
            row["tests"],
        ),
        reverse=True,
    )
    return summaries


def write_summary_csv(summary_rows: List[Dict[str, Any]]) -> Path:
    out_path = OUTPUT_DIR / "phase1_overcoupled_strategy_summary.csv"
    fieldnames = [
        "strategy",
        "strategy_display",
        "strategy_type",
        "tests",
        "zero_fill_tests",
        "closed_fill_mean_pct",
        "closed_fill_std_pct",
        "closed_fill_nonzero_mean_pct",
        "effective_fill_mean_pct",
        "effective_fill_std_pct",
        "effective_fill_nonzero_mean_pct",
        "placement_mean_pct",
        "placement_std_pct",
        "ms_per_box_mean",
        "ms_per_box_std",
        "secs_per_box_mean",
        "secs_per_box_std",
        "pallets_closed_mean",
        "placed_boxes_mean",
        "rejected_boxes_mean",
        "rej_per_pallet_mean",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return out_path


def build_latex(summary_rows: List[Dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Combined Phase 1 strategy comparison after replacing flawed strategy rows with their rerun data.}",
        r"\label{tab:botko-overcoupled-phase1}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\definecolor{headerblue}{HTML}{1B365D}",
        r"\definecolor{rowalt}{HTML}{EDF2F7}",
        r"\definecolor{bestgreen}{HTML}{2F855A}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"\rowcolor{headerblue}",
        r"\textcolor{white}{\textbf{Strategy}} & \multicolumn{2}{c}{\textcolor{white}{\textbf{Closed Fill [\%]}}} & \multicolumn{2}{c}{\textcolor{white}{\textbf{Placement [\%]}}} & \textcolor{white}{\textbf{secs/box}} & \textcolor{white}{\textbf{$\overline{P_c}$}} & \textcolor{white}{\textbf{Placed}} & \textcolor{white}{\textbf{Rej./P}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"\rowcolor{headerblue}",
        r" & \textcolor{white}{$\mu$} & \textcolor{white}{$\sigma$} & \textcolor{white}{$\mu$} & \textcolor{white}{$\sigma$} & & & & \\",
        r"\midrule",
    ]

    best_fill = max(row["closed_fill_mean_pct"] for row in summary_rows)
    best_placement = max(row["placement_mean_pct"] for row in summary_rows)
    best_rej = min(row["rej_per_pallet_mean"] for row in summary_rows)

    for index, row in enumerate(summary_rows):
        strategy_name = latex_escape(str(row["strategy_display"]))
        fill_mu = f"{row['closed_fill_mean_pct']:.1f}"
        placement_mu = f"{row['placement_mean_pct']:.1f}"
        rej = f"{row['rej_per_pallet_mean']:.1f}"

        if abs(row["closed_fill_mean_pct"] - best_fill) < 1e-9:
            fill_mu = rf"\textcolor{{bestgreen}}{{\textbf{{{fill_mu}}}}}"
        if abs(row["placement_mean_pct"] - best_placement) < 1e-9:
            placement_mu = rf"\textcolor{{bestgreen}}{{\textbf{{{placement_mu}}}}}"
        if abs(row["rej_per_pallet_mean"] - best_rej) < 1e-9:
            rej = rf"\textcolor{{bestgreen}}{{\textbf{{{rej}}}}}"

        prefix = r"\rowcolor{rowalt} " if index % 2 == 1 else ""
        lines.append(
            f"{prefix}{strategy_name} & "
            f"{fill_mu} & {row['closed_fill_std_pct']:.1f} & "
            f"{placement_mu} & {row['placement_std_pct']:.1f} & "
            f"{fmt_secs(row['ms_per_box_mean'])} & {row['pallets_closed_mean']:.1f} & "
            f"{row['placed_boxes_mean']:.0f} & {rej} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def build_standalone_latex(table_fragment: str) -> str:
    return "\n".join(
        [
            r"\documentclass[border=8pt]{standalone}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{booktabs}",
            r"\usepackage{array}",
            r"\usepackage[table]{xcolor}",
            r"\begin{document}",
            table_fragment,
            r"\end{document}",
        ]
    )


def write_latex(summary_rows: List[Dict[str, Any]]) -> Path:
    out_path = OUTPUT_DIR / "phase1_overcoupled_strategy_table.tex"
    out_path.write_text(build_latex(summary_rows), encoding="utf-8")
    return out_path


def write_standalone_latex(summary_rows: List[Dict[str, Any]]) -> Path:
    out_path = OUTPUT_DIR / "phase1_overcoupled_strategy_table_standalone.tex"
    out_path.write_text(build_standalone_latex(build_latex(summary_rows)), encoding="utf-8")
    return out_path


def latex_to_png(tex_content: str, output_png: Path, dpi: int = 300) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="botko_latex_"))
    tex_path = tmpdir / "table.tex"
    pdf_path = tmpdir / "table.pdf"
    tex_path.write_text(tex_content, encoding="utf-8")

    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tmpdir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdflatex failed: {result.stderr or result.stdout[-1000:]}")
    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected PDF was not generated: {pdf_path}")

    output_png = output_png.resolve()

    if shutil.which("magick"):
        convert = subprocess.run(
            [
                "magick",
                "-density",
                str(dpi),
                str(pdf_path),
                "-quality",
                "100",
                "-alpha",
                "remove",
                "-alpha",
                "off",
                str(output_png),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if convert.returncode == 0 and output_png.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)
            return output_png

    if shutil.which("pdftoppm"):
        out_base = str(output_png).removesuffix(".png")
        convert = subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf_path), out_base],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if convert.returncode == 0 and output_png.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)
            return output_png

    raise RuntimeError("No PDF-to-PNG converter available (need magick or pdftoppm)")


def render_summary_png(summary_rows: List[Dict[str, Any]], output_png: Path) -> Path:
    fig_height = max(7, 0.34 * (len(summary_rows) + 3))
    fig, ax = plt.subplots(figsize=(15.5, fig_height))
    ax.axis("off")

    header_top = [
        "Strategy",
        "Closed Fill [%]",
        "",
        "Placement [%]",
        "",
        "secs/box",
        r"$\overline{P_c}$",
        "Placed",
        "Rej/P",
    ]
    header_sub = ["", r"$\mu$", r"$\sigma$", r"$\mu$", r"$\sigma$", "", "", "", ""]
    rows = []
    for row in summary_rows:
        rows.append(
            [
                row["strategy_display"],
                f"{row['closed_fill_mean_pct']:.1f}",
                f"{row['closed_fill_std_pct']:.1f}",
                f"{row['placement_mean_pct']:.1f}",
                f"{row['placement_std_pct']:.1f}",
                fmt_secs(row["ms_per_box_mean"]),
                f"{row['pallets_closed_mean']:.1f}",
                f"{row['placed_boxes_mean']:.0f}",
                f"{row['rej_per_pallet_mean']:.1f}",
            ]
        )

    table = ax.table(
        cellText=[header_top, header_sub] + rows,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1, 1.22)

    best_fill = max(row["closed_fill_mean_pct"] for row in summary_rows)
    best_placement = max(row["placement_mean_pct"] for row in summary_rows)
    best_rej = min(row["rej_per_pallet_mean"] for row in summary_rows)

    for (r, c), cell in table.get_celld().items():
        if r in (0, 1):
            cell.set_facecolor("#1B365D")
            cell.set_text_props(color="white", weight="bold")
            cell.set_edgecolor("#1B365D")
        else:
            cell.set_edgecolor("#D9E2EC")
            if (r - 2) % 2 == 1:
                cell.set_facecolor("#EDF2F7")
            else:
                cell.set_facecolor("#FFFFFF")

    for idx, row in enumerate(summary_rows, start=2):
        if abs(row["closed_fill_mean_pct"] - best_fill) < 1e-9:
            table[idx, 1].get_text().set_color("#2F855A")
            table[idx, 1].get_text().set_weight("bold")
        if abs(row["placement_mean_pct"] - best_placement) < 1e-9:
            table[idx, 3].get_text().set_color("#2F855A")
            table[idx, 3].get_text().set_weight("bold")
        if abs(row["rej_per_pallet_mean"] - best_rej) < 1e-9:
            table[idx, 8].get_text().set_color("#2F855A")
            table[idx, 8].get_text().set_weight("bold")

    col_widths = {
        0: 0.34,
        1: 0.09,
        2: 0.07,
        3: 0.09,
        4: 0.07,
        5: 0.10,
        6: 0.07,
        7: 0.07,
        8: 0.07,
    }
    total_rows = len(rows) + 2
    for c, width in col_widths.items():
        for r in range(total_rows):
            table[r, c].set_width(width)

    plt.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_png.resolve()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for run_cfg in RUNS:
        if not run_cfg["path"].exists():
            raise FileNotFoundError(f"Missing input file: {run_cfg['path']}")
        all_rows.extend(run_phase1_rows(run_cfg))

    corrected_rows = build_corrected_rows(all_rows)
    combined_csv = write_combined_rows_csv(corrected_rows)
    summary_rows = build_summary(corrected_rows)
    summary_csv = write_summary_csv(summary_rows)
    summary_tex = write_latex(summary_rows)
    summary_tex_standalone = write_standalone_latex(summary_rows)
    summary_png = None
    if shutil.which("pdflatex") and (shutil.which("magick") or shutil.which("pdftoppm")):
        summary_png = latex_to_png(
            summary_tex_standalone.read_text(encoding="utf-8"),
            OUTPUT_DIR / "phase1_overcoupled_strategy_table.png",
            dpi=300,
        )
    else:
        summary_png = render_summary_png(
            summary_rows,
            OUTPUT_DIR / "phase1_overcoupled_strategy_table.png",
        )

    print("Wrote combined report artifacts:")
    print(f"  {combined_csv}")
    print(f"  {summary_csv}")
    print(f"  {summary_tex}")
    print(f"  {summary_tex_standalone}")
    if summary_png is not None:
        print(f"  {summary_png}")
    print("")
    replacement_strategies = sorted(
        {
            row["strategy"]
            for row in all_rows
            if row["run_label"] == "targeted_fixed"
        }
    )
    print("Replacement logic:")
    print(
        "  Replaced Feb28 rows with Mar01Fix rows for: "
        + ", ".join(replacement_strategies)
    )
    print("")
    print("Selected runs:")
    for run_cfg in RUNS:
        data = load_json(run_cfg["path"])
        meta = data.get("metadata", {})
        print(
            f"  {run_cfg['short_label']}: {run_cfg['path'].parent.name} "
            f"({len(data.get('phase1_baseline', []))} phase1 rows, "
            f"{meta.get('n_datasets', '?')} datasets x {meta.get('n_shuffles', '?')} shuffles)"
        )


if __name__ == "__main__":
    main()
