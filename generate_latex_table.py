"""
Generate ultra-clean LaTeX comparison tables from botko results.json.
Produces two tables:
  1) Phase 1 -- All 23 strategies baseline comparison
  2) Phase 2 -- Selector sweep for top strategies
Renders both to high-DPI PNG for PowerPoint embedding.

Metrics include: Closed Fill %, Placement %, ms/box, Pallets Closed, Rej./Pallet
"""
import json
import statistics
import subprocess
import os
import shutil
import tempfile

RESULTS_PATH = "output/botko_20260223_133414/results.json"
OUTPUT_DIR = "output/botko_20260223_133414"

with open(RESULTS_PATH) as f:
    data = json.load(f)

p1 = data["phase1_baseline"]
p2 = data["phase2_sweep"]
meta = data["metadata"]

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
    "walle_scoring": r"WALL\textperiodcentered E Scoring",
}

BOX_SEL_DISPLAY = {
    "default": "Default",
    "biggest_volume_first": r"Vol.\,$\downarrow$",
    "biggest_footprint_first": r"Foot.\,$\downarrow$",
    "native": "Native",
}

BIN_SEL_DISPLAY = {
    "emptiest_first": "Emptiest",
    "flattest_first": "Flattest",
    "focus_fill": "Focus-Fill",
    "native": "Native",
}


def safe_mean(vals):
    return statistics.mean(vals) if vals else 0.0

def safe_std(vals):
    return statistics.stdev(vals) if len(vals) > 1 else 0.0

def fmt_pct(v):
    return f"{v*100:.1f}"

def fmt_ms(v):
    if v < 1:
        return f"{v*1000:.0f}\\,us"
    if v < 100:
        return f"{v:.1f}"
    if v < 1000:
        return f"{v:.0f}"
    return f"{v/1000:.1f}k"

def compute_rej_per_pallet(runs):
    """Compute mean rejected boxes per closed pallet across runs."""
    vals = []
    for r in runs:
        pc = r["pallets_closed"]
        if pc > 0:
            vals.append(r["total_rejected"] / pc)
        else:
            # No closed pallets -> use total pallets (active)
            total_p = len(r.get("active_pallets", []))
            if total_p > 0:
                vals.append(r["total_rejected"] / total_p)
            else:
                vals.append(float("inf"))
    finite = [v for v in vals if v != float("inf")]
    return safe_mean(finite) if finite else float("inf")


# --- Phase 1 ---
def compute_phase1_stats():
    strats = sorted(set(r["strategy"] for r in p1))
    rows = []
    for s in strats:
        runs = [r for r in p1 if r["strategy"] == s]
        fills = [r["avg_closed_fill"] for r in runs if r.get("avg_closed_fill")]
        pr = [r["placement_rate"] for r in runs]
        ms = [r["ms_per_box"] for r in runs]
        pc = [r["pallets_closed"] for r in runs]
        tp = [r["total_placed"] for r in runs]
        tr = [r["total_rejected"] for r in runs]
        rej_pp = compute_rej_per_pallet(runs)
        rows.append({
            "strategy": s, "n": len(runs),
            "fill_mean": safe_mean(fills), "fill_std": safe_std(fills),
            "pr_mean": safe_mean(pr), "pr_std": safe_std(pr),
            "ms_mean": safe_mean(ms),
            "pc_mean": safe_mean(pc),
            "placed_mean": safe_mean(tp),
            "rejected_mean": safe_mean(tr),
            "rej_per_pallet": rej_pp,
        })
    rows.sort(key=lambda r: r["fill_mean"], reverse=True)
    return rows


def build_phase1_latex(rows):
    best_fill = max(r["fill_mean"] for r in rows)
    best_pr = max(r["pr_mean"] for r in rows)
    best_ms = min(r["ms_mean"] for r in rows if r["ms_mean"] > 0)
    finite_rej = [r["rej_per_pallet"] for r in rows if r["rej_per_pallet"] != float("inf")]
    best_rpp = min(finite_rej) if finite_rej else float("inf")

    lines = []
    lines.append(r"\documentclass[border=10pt]{standalone}")
    lines.append(r"\usepackage[T1]{fontenc}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{array}")
    lines.append(r"\usepackage{xcolor}")
    lines.append(r"\usepackage{colortbl}")
    lines.append(r"\usepackage{textcomp}")
    lines.append(r"\usepackage{lmodern}")
    lines.append(r"\usepackage{helvet}")
    lines.append(r"\renewcommand{\familydefault}{\sfdefault}")
    lines.append("")
    lines.append(r"\definecolor{headerblue}{HTML}{1B365D}")
    lines.append(r"\definecolor{headertxt}{HTML}{FFFFFF}")
    lines.append(r"\definecolor{rowalt}{HTML}{F5F7FA}")
    lines.append(r"\definecolor{bestgreen}{HTML}{1A8C42}")
    lines.append(r"\definecolor{topfive}{HTML}{E3F0FD}")
    lines.append("")
    lines.append(r"\newcommand{\best}[1]{\textcolor{bestgreen}{\textbf{#1}}}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.28}")
    lines.append(r"\footnotesize")
    lines.append("")
    # Columns: Strategy | Fill mu | Fill sigma | Place mu | Place sigma | ms/box | P_closed | Placed | Rej/Pallet
    lines.append(r"\begin{tabular}{" +
                 r">{\raggedright\arraybackslash}p{3.0cm}" +
                 r"  r r  r r  r  r  r  r" +
                 r"}")
    lines.append("")
    lines.append(r"\toprule")
    lines.append(r"\rowcolor{headerblue}")
    hdr = (r"\textcolor{headertxt}{\textbf{Strategy}}"
           r" & \multicolumn{2}{c}{\textcolor{headertxt}{\textbf{Closed Fill [\%]}}}"
           r" & \multicolumn{2}{c}{\textcolor{headertxt}{\textbf{Placement [\%]}}}"
           r" & \textcolor{headertxt}{\textbf{ms/box}}"
           r" & \textcolor{headertxt}{\textbf{$\overline{\mathit{P}}_c$}}"
           r" & \textcolor{headertxt}{\textbf{Placed}}"
           r" & \textcolor{headertxt}{\textbf{Rej./P}} \\")
    lines.append(hdr)

    lines.append(r"\rowcolor{headerblue}")
    sub = (r" & \textcolor{headertxt}{$\mu$} & \textcolor{headertxt}{$\sigma$}"
           r" & \textcolor{headertxt}{$\mu$} & \textcolor{headertxt}{$\sigma$}"
           r" & & & & \\")
    lines.append(sub)
    lines.append(r"\midrule")

    top5 = set(meta.get("top_5", []))

    for i, row in enumerate(rows):
        s = row["strategy"]
        name = STRATEGY_DISPLAY.get(s, s)

        is_top5 = s in top5
        fill_v = fmt_pct(row["fill_mean"])
        fill_s = fmt_pct(row["fill_std"])
        pr_v = fmt_pct(row["pr_mean"])
        pr_s = fmt_pct(row["pr_std"])
        ms_v = fmt_ms(row["ms_mean"])
        pc_v = f"{row['pc_mean']:.1f}"
        placed_v = f"{row['placed_mean']:.0f}"
        rpp_v = f"{row['rej_per_pallet']:.1f}" if row["rej_per_pallet"] != float("inf") else "--"

        if abs(row["fill_mean"] - best_fill) < 1e-6:
            fill_v = r"\best{" + fill_v + "}"
        if abs(row["pr_mean"] - best_pr) < 1e-4:
            pr_v = r"\best{" + pr_v + "}"
        if abs(row["ms_mean"] - best_ms) < 0.1:
            ms_v = r"\best{" + ms_v + "}"
        if row["rej_per_pallet"] != float("inf") and abs(row["rej_per_pallet"] - best_rpp) < 0.05:
            rpp_v = r"\best{" + rpp_v + "}"

        if is_top5:
            prefix = r"\rowcolor{topfive}"
        elif i % 2 == 1:
            prefix = r"\rowcolor{rowalt}"
        else:
            prefix = ""

        line = f"{prefix}{name} & {fill_v} & {fill_s} & {pr_v} & {pr_s} & {ms_v} & {pc_v} & {placed_v} & {rpp_v} \\\\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{document}")
    return "\n".join(lines)


# --- Phase 2 ---
def compute_phase2_stats():
    combos = sorted(set(
        (r["strategy"], r.get("box_selector","default"), r.get("bin_selector",""))
        for r in p2
    ))
    rows = []
    for s, bs, bis in combos:
        runs = [r for r in p2
                if r["strategy"] == s
                and r.get("box_selector","default") == bs
                and r.get("bin_selector","") == bis]
        if not runs:
            continue
        fills = [r["avg_closed_fill"] for r in runs if r.get("avg_closed_fill")]
        pr = [r["placement_rate"] for r in runs]
        ms = [r["ms_per_box"] for r in runs]
        pc = [r["pallets_closed"] for r in runs]
        rej_pp = compute_rej_per_pallet(runs)
        rows.append({
            "strategy": s, "box_selector": bs, "bin_selector": bis,
            "n": len(runs),
            "fill_mean": safe_mean(fills), "fill_std": safe_std(fills),
            "pr_mean": safe_mean(pr), "ms_mean": safe_mean(ms),
            "pc_mean": safe_mean(pc),
            "rej_per_pallet": rej_pp,
        })
    # Group: walle first, then surface, then rest -- sorted by fill descending
    walle = sorted([r for r in rows if r["strategy"] == "walle_scoring"],
                   key=lambda r: r["fill_mean"], reverse=True)
    surface = sorted([r for r in rows if r["strategy"] == "surface_contact"],
                     key=lambda r: r["fill_mean"], reverse=True)
    rest = sorted([r for r in rows if r["strategy"] not in ("walle_scoring","surface_contact")],
                  key=lambda r: r["fill_mean"], reverse=True)
    return walle + surface + rest


def build_phase2_latex(rows):
    best_fill = max(r["fill_mean"] for r in rows)
    best_pr = max(r["pr_mean"] for r in rows)
    best_ms = min(r["ms_mean"] for r in rows if r["ms_mean"] > 0)
    finite_rej = [r["rej_per_pallet"] for r in rows if r["rej_per_pallet"] != float("inf")]
    best_rpp = min(finite_rej) if finite_rej else float("inf")

    lines = []
    lines.append(r"\documentclass[border=10pt]{standalone}")
    lines.append(r"\usepackage[T1]{fontenc}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{array}")
    lines.append(r"\usepackage{xcolor}")
    lines.append(r"\usepackage{colortbl}")
    lines.append(r"\usepackage{textcomp}")
    lines.append(r"\usepackage{lmodern}")
    lines.append(r"\usepackage{helvet}")
    lines.append(r"\renewcommand{\familydefault}{\sfdefault}")
    lines.append(r"\usepackage{multirow}")
    lines.append("")
    lines.append(r"\definecolor{headerblue}{HTML}{1B365D}")
    lines.append(r"\definecolor{headertxt}{HTML}{FFFFFF}")
    lines.append(r"\definecolor{rowalt}{HTML}{F5F7FA}")
    lines.append(r"\definecolor{bestgreen}{HTML}{1A8C42}")
    lines.append(r"\definecolor{wallecolor}{HTML}{FFF8E1}")
    lines.append(r"\definecolor{surfacecolor}{HTML}{E8F5E9}")
    lines.append("")
    lines.append(r"\newcommand{\best}[1]{\textcolor{bestgreen}{\textbf{#1}}}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.28}")
    lines.append(r"\footnotesize")
    lines.append("")
    # Columns: Strategy | Box Sel | Bin Sel | Fill | Place | ms/box | P_c | Rej/P
    lines.append(r"\begin{tabular}{" +
                 r">{\raggedright\arraybackslash}p{2.8cm}" +
                 r"  >{\centering\arraybackslash}p{1.8cm}" +
                 r"  >{\centering\arraybackslash}p{1.6cm}" +
                 r"  r  r  r  r  r" +
                 r"}")
    lines.append("")
    lines.append(r"\toprule")
    lines.append(r"\rowcolor{headerblue}")
    hdr = (r"\textcolor{headertxt}{\textbf{Strategy}}"
           r" & \textcolor{headertxt}{\textbf{Box Sel.}}"
           r" & \textcolor{headertxt}{\textbf{Bin Sel.}}"
           r" & \textcolor{headertxt}{\textbf{Fill [\%]}}"
           r" & \textcolor{headertxt}{\textbf{Place [\%]}}"
           r" & \textcolor{headertxt}{\textbf{ms/box}}"
           r" & \textcolor{headertxt}{\textbf{$\overline{\mathit{P}}_c$}}"
           r" & \textcolor{headertxt}{\textbf{Rej./P}} \\")
    lines.append(hdr)
    lines.append(r"\midrule")

    prev_strat = None
    for i, row in enumerate(rows):
        s = row["strategy"]
        name = STRATEGY_DISPLAY.get(s, s)
        bs = BOX_SEL_DISPLAY.get(row["box_selector"], row["box_selector"])
        bis = BIN_SEL_DISPLAY.get(row["bin_selector"], row["bin_selector"])

        fill_v = fmt_pct(row["fill_mean"])
        pr_v = fmt_pct(row["pr_mean"])
        ms_v = fmt_ms(row["ms_mean"])
        pc_v = f"{row['pc_mean']:.1f}"
        rpp_v = f"{row['rej_per_pallet']:.1f}" if row["rej_per_pallet"] != float("inf") else "--"

        if abs(row["fill_mean"] - best_fill) < 1e-6:
            fill_v = r"\best{" + fill_v + "}"
        if abs(row["pr_mean"] - best_pr) < 1e-4:
            pr_v = r"\best{" + pr_v + "}"
        if abs(row["ms_mean"] - best_ms) < 0.1:
            ms_v = r"\best{" + ms_v + "}"
        if row["rej_per_pallet"] != float("inf") and abs(row["rej_per_pallet"] - best_rpp) < 0.05:
            rpp_v = r"\best{" + rpp_v + "}"

        if "walle" in s:
            color = r"\rowcolor{wallecolor}"
        elif "surface" in s:
            color = r"\rowcolor{surfacecolor}"
        else:
            color = r"\rowcolor{rowalt}" if i % 2 == 1 else ""

        if prev_strat is not None and s != prev_strat:
            lines.append(r"\midrule")
        prev_strat = s

        line = f"{color}{name} & {bs} & {bis} & {fill_v} & {pr_v} & {ms_v} & {pc_v} & {rpp_v} \\\\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{document}")
    return "\n".join(lines)


# --- Compile LaTeX -> PNG ---
def latex_to_png(tex_content, output_png, dpi=300):
    tmpdir = tempfile.mkdtemp(prefix="latex_tbl_")
    tex_path = os.path.join(tmpdir, "table.tex")
    pdf_path = os.path.join(tmpdir, "table.pdf")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "table.tex"],
        cwd=tmpdir, capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print("=== LaTeX STDOUT (last 2000 chars) ===")
        print(result.stdout[-2000:])
        print("=== LaTeX STDERR ===")
        print(result.stderr[-1000:])
        raise RuntimeError(f"pdflatex failed (exit {result.returncode})")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not generated at {pdf_path}")

    output_png_abs = os.path.abspath(output_png)

    if shutil.which("magick"):
        r2 = subprocess.run(
            ["magick", "-density", str(dpi), pdf_path,
             "-quality", "100", "-alpha", "remove", "-alpha", "off",
             output_png_abs],
            capture_output=True, text=True, timeout=60
        )
        if r2.returncode == 0 and os.path.exists(output_png_abs):
            print(f"  [magick] OK -> {output_png_abs}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return output_png_abs
        else:
            print(f"  [magick] failed: {r2.stderr[:300]}")

    if shutil.which("pdftoppm"):
        out_base = output_png_abs.replace(".png", "")
        r2 = subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", pdf_path, out_base],
            capture_output=True, text=True, timeout=30
        )
        if r2.returncode == 0 and os.path.exists(output_png_abs):
            print(f"  [pdftoppm] OK -> {output_png_abs}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return output_png_abs

    pdf_out = output_png_abs.replace(".png", ".pdf")
    shutil.copy2(pdf_path, pdf_out)
    print(f"  [WARNING] Could not convert to PNG. PDF -> {pdf_out}")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return pdf_out


if __name__ == "__main__":
    print("=" * 60)
    print("  Botko Strategy Comparison -- LaTeX Table Generator v2")
    print("  Now includes: Rej./Pallet metric")
    print("=" * 60)

    p1_stats = compute_phase1_stats()
    p1_tex = build_phase1_latex(p1_stats)
    p1_tex_path = os.path.join(OUTPUT_DIR, "table_phase1_strategies.tex")
    with open(p1_tex_path, "w", encoding="utf-8") as f:
        f.write(p1_tex)
    print(f"\n[1/4] .tex saved: {p1_tex_path}")

    p2_stats = compute_phase2_stats()
    p2_tex = build_phase2_latex(p2_stats)
    p2_tex_path = os.path.join(OUTPUT_DIR, "table_phase2_selectors.tex")
    with open(p2_tex_path, "w", encoding="utf-8") as f:
        f.write(p2_tex)
    print(f"[2/4] .tex saved: {p2_tex_path}")

    try:
        p1_png = os.path.join(OUTPUT_DIR, "table_phase1_strategies.png")
        result = latex_to_png(p1_tex, p1_png, dpi=300)
        print(f"[3/4] Phase 1 -> {result}")
    except Exception as e:
        print(f"[3/4] Phase 1 FAILED: {e}")

    try:
        p2_png = os.path.join(OUTPUT_DIR, "table_phase2_selectors.png")
        result = latex_to_png(p2_tex, p2_png, dpi=300)
        print(f"[4/4] Phase 2 -> {result}")
    except Exception as e:
        print(f"[4/4] Phase 2 FAILED: {e}")

    print("\nDone!")
