import json, statistics
d = json.load(open('output/botko_20260223_133414/results.json'))
p1 = d['phase1_baseline']
for s in ['walle_scoring','surface_contact','best_fit_decreasing','extreme_points','baseline','heuristic_160']:
    runs = [r for r in p1 if r['strategy']==s and r.get('box_selector','default')=='default']
    if not runs: continue
    rej_per_closed = []
    for r in runs:
        pc = r['pallets_closed']
        if pc > 0:
            rej_per_closed.append(r['total_rejected'] / pc)
        else:
            rej_per_closed.append(float('inf'))
    total_pallets = [r['pallets_closed'] + len(r.get('active_pallets',[])) for r in runs]
    rej_per_total = [r['total_rejected']/tp for tp, r in zip(total_pallets, runs)]
    pc_mean = statistics.mean([r['pallets_closed'] for r in runs])
    rej_mean = statistics.mean([r['total_rejected'] for r in runs])
    print(f"{s:25s}: closed={pc_mean:.1f}  rej={rej_mean:.1f}  rej/closed={statistics.mean(rej_per_closed):.1f}  rej/total={statistics.mean(rej_per_total):.1f}")
