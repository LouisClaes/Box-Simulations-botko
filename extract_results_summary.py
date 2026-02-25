"""Extract summary stats from results.json and write to a text file."""
import json
import statistics
import sys

d = json.load(open('output/botko_20260223_133414/results.json'))
p1 = d.get('phase1_baseline', [])
p2 = d.get('phase2_sweep', [])
meta = d.get('metadata', {})

lines = []
lines.append(f"Metadata: {json.dumps(meta, indent=2)}")
lines.append(f"\nP1 entries: {len(p1)}")
lines.append(f"P2 entries: {len(p2)}")

strats = sorted(set(r['strategy'] for r in p1 + p2))
bsel = sorted(set(r.get('box_selector', '') for r in p1 + p2))
bisel = sorted(set(r.get('bin_selector', '') for r in p1 + p2))
lines.append(f"\nAll strategies: {strats}")
lines.append(f"All box_selectors: {bsel}")
lines.append(f"All bin_selectors: {bisel}")

# Phase 1 combos
lines.append("\n=== PHASE 1 COMBOS ===")
combos = sorted(set((r['strategy'], r.get('box_selector','default'), r.get('bin_selector',''))
                     for r in p1))
for c in combos:
    cnt = sum(1 for r in p1 if (r['strategy'], r.get('box_selector','default'),
              r.get('bin_selector','')) == c)
    lines.append(f"  {c[0]:25s} | {c[1]:18s} | {c[2]:18s} | n={cnt}")

# Phase 2 combos
lines.append("\n=== PHASE 2 COMBOS ===")
combos2 = sorted(set((r['strategy'], r.get('box_selector','default'), r.get('bin_selector',''))
                      for r in p2))
for c in combos2:
    cnt = sum(1 for r in p2 if (r['strategy'], r.get('box_selector','default'),
              r.get('bin_selector','')) == c)
    lines.append(f"  {c[0]:25s} | {c[1]:18s} | {c[2]:18s} | n={cnt}")

# Phase 1 aggregated stats per strategy (default only)
lines.append("\n=== PHASE 1 AGGREGATED (default selector, emptiest_first) ===")
for s in sorted(set(r['strategy'] for r in p1)):
    runs = [r for r in p1 if r['strategy'] == s
            and r.get('box_selector', 'default') == 'default'
            and r.get('bin_selector', '') == 'emptiest_first']
    if not runs:
        continue
    fills = [r.get('avg_closed_fill', 0) for r in runs if r.get('avg_closed_fill')]
    eff_fills = [r.get('avg_closed_effective_fill', 0) for r in runs if r.get('avg_closed_effective_fill')]
    p_rates = [r['placement_rate'] for r in runs]
    ms_per = [r['ms_per_box'] for r in runs]
    pclose = [r['pallets_closed'] for r in runs]
    placed = [r['total_placed'] for r in runs]
    rejected = [r['total_rejected'] for r in runs]

    lines.append(f"\n{s} (n={len(runs)}):")
    if fills:
        lines.append(f"  avg_closed_fill:      mean={statistics.mean(fills):.4f}  std={statistics.stdev(fills) if len(fills) > 1 else 0:.4f}")
    if eff_fills:
        lines.append(f"  avg_closed_eff_fill:  mean={statistics.mean(eff_fills):.4f}  std={statistics.stdev(eff_fills) if len(eff_fills) > 1 else 0:.4f}")
    lines.append(f"  placement_rate:       mean={statistics.mean(p_rates):.4f}  std={statistics.stdev(p_rates) if len(p_rates) > 1 else 0:.4f}")
    lines.append(f"  ms_per_box:           mean={statistics.mean(ms_per):.2f}  std={statistics.stdev(ms_per) if len(ms_per) > 1 else 0:.2f}")
    lines.append(f"  pallets_closed:       mean={statistics.mean(pclose):.1f}")
    lines.append(f"  total_placed:         mean={statistics.mean(placed):.1f}")
    lines.append(f"  total_rejected:       mean={statistics.mean(rejected):.1f}")

# Phase 2 aggregated stats per combo
lines.append("\n=== PHASE 2 AGGREGATED (per strategy + selectors) ===")
for c in combos2:
    runs = [r for r in p2 if (r['strategy'], r.get('box_selector','default'),
            r.get('bin_selector','')) == c]
    fills = [r.get('avg_closed_fill', 0) for r in runs if r.get('avg_closed_fill')]
    eff_fills = [r.get('avg_closed_effective_fill', 0) for r in runs if r.get('avg_closed_effective_fill')]
    p_rates = [r['placement_rate'] for r in runs]
    ms_per = [r['ms_per_box'] for r in runs]
    pclose = [r['pallets_closed'] for r in runs]

    lines.append(f"\n{c[0]} | box={c[1]} | bin={c[2]} (n={len(runs)}):")
    if fills:
        lines.append(f"  avg_closed_fill:      mean={statistics.mean(fills):.4f}")
    if eff_fills:
        lines.append(f"  avg_closed_eff_fill:  mean={statistics.mean(eff_fills):.4f}")
    lines.append(f"  placement_rate:       mean={statistics.mean(p_rates):.4f}")
    lines.append(f"  ms_per_box:           mean={statistics.mean(ms_per):.2f}")
    lines.append(f"  pallets_closed:       mean={statistics.mean(pclose):.1f}")

text = "\n".join(lines)
with open("_results_summary.txt", "w") as f:
    f.write(text)
print("Written to _results_summary.txt")
