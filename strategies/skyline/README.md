# Skyline Strategy

## What Changed
This version applies three PRD-style fixes:

1. Real anti-tower penalty: tall outliers are explicitly penalized.
2. Non-degenerate uniformity metric: local smoothness delta is measured in an expanded neighborhood, not only inside the footprint.
3. Safe fallback search: if valley-first search finds no candidate, a full-grid fallback scan is executed.

## Scoring Formula
For candidate `(x, y, z, ol, ow, oh)`:

```
score =
    - WEIGHT_Z * z
    + WEIGHT_VALLEY_FILL * valley_fill
    + WEIGHT_UNIFORMITY * uniformity_delta
    - WEIGHT_TOWER * tower_penalty
```

where:

```
valley_fill = min(ow / valley_depth, 1.0)
```

```
uniformity_delta = (roughness_before - roughness_after) / bin_height
```

`roughness_*` is local mean absolute gradient in an expanded window around the footprint:

```
roughness(region) = mean( mean(|diff_x(region)|), mean(|diff_y(region)|) )
```

Tower penalty:

```
local_excess  = max(0, box_top - (local_ring_p75 + allowance))
global_excess = max(0, box_top - (global_p90     + allowance))

tower_penalty = 0.7 * (local_excess / bin_height)^2
              + 0.3 * (global_excess / bin_height)^2
```

with:

```
box_top   = z + oh
allowance = TOWER_ALLOWED_EXTRA_RATIO * bin_height
```

## Search Logic
1. Build skyline profile: `skyline[gy] = mean(heightmap[:, gy])`.
2. Sort y-bands by ascending skyline.
3. Evaluate up to `MAX_VALLEY_CANDIDATES` valleys, scanning x positions per orientation.
4. If no valid candidate is found: run `_fallback_grid_search()` over the full grid at resolution step.
5. Return best scored candidate or `None`.

## Physical Validity Gates
Each candidate must satisfy:

1. In-bin bounds.
2. Height bound (`z + oh <= bin_height`).
3. Anti-float (`support_ratio >= MIN_SUPPORT` when `z > 0`).
4. Optional stricter stability threshold (`cfg.min_support_ratio`).
5. Margin clearance (`is_margin_clear`).

## Constants

- `MIN_SUPPORT = 0.30`
- `WEIGHT_Z = 3.0`
- `WEIGHT_VALLEY_FILL = 1.5`
- `WEIGHT_UNIFORMITY = 1.1`
- `WEIGHT_TOWER = 4.5`
- `MAX_VALLEY_CANDIDATES = 40`
- `UNIFORMITY_WINDOW_MARGIN_CELLS = 2`
- `TOWER_WINDOW_MARGIN_CELLS = 3`
- `TOWER_ALLOWED_EXTRA_RATIO = 0.03`

## Validation Commands
Run from `python/`:

```bash
python -m py_compile strategies/skyline/strategy.py
```

```bash
python - <<'PY'
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath('run_experiment.py')))
from config import BinConfig, ExperimentConfig
from dataset.generator import generate_uniform
from run_experiment import run_experiment
boxes = generate_uniform(n=10, min_dim=200.0, max_dim=350.0, seed=11)
cfg = ExperimentConfig(
    bin=BinConfig(length=1200.0, width=800.0, height=2700.0, resolution=10.0),
    strategy_name='skyline',
    enable_stability=True,
    min_support_ratio=0.8,
    allow_all_orientations=False,
    render_3d=False,
    verbose=False,
)
r = run_experiment(cfg, boxes)
m = r['metrics']
print(f"skyline: completed={r.get('completed', True)} placed={m['boxes_placed']}/{m['boxes_total']} fill={m['fill_rate']:.4f} stability={m['stability_rate']:.4f} time_ms={m['computation_time_ms']:.1f}")
PY
```

Observed output:

```
skyline: completed=True placed=10/10 fill=0.0748 stability=1.0000 time_ms=7762.9
```
