# LBCP Stability Strategy

## What Changed
This version applies three PRD-style fixes:

1. Bracing-aware stability relaxation (bounded): if CoG is slightly outside the support polygon, limited relaxation is allowed only when support and lateral bracing are strong enough.
2. Improved support-contact detection: support is computed from overlap area between footprint cells and support patches, not only cell-center inclusion.
3. Diagnostics-friendly behavior: each decision records rejection/accept counters and brace-relaxation usage via `get_last_diagnostics()`.

## Core Validation Logic

### 1) Support patch construction
For each support item `p`:

```
patch_p = candidate_footprint intersect top_face(p) intersect lbcp(p)
```

where `lbcp(p)` is:

- full top face for floor items (`z < 0.5`)
- inward-shrunk rectangle (`LBCP_SHRINK_STACKED`) for stacked items

### 2) Improved support ratio (area-based)
For each candidate footprint cell:

1. Height must match resting `z` within `CONTACT_TOL`.
2. Cell rectangle must overlap a support patch by at least:

```
MIN_CELL_CONTACT_FRACTION * cell_area
```

Support ratio:

```
support_ratio = total_contact_area / total_footprint_area
```

### 3) Strict LBCP test
Build support polygon as convex hull of contact cell centers.
Strictly stable if CoG is inside polygon.

## Bracing-Aware Relaxation (Bounded)
If CoG is outside support polygon, compute:

```
brace_factor in [0, 1]
```

from four side channels (`left/right/front/back`) using:

- wall brace (near wall)
- neighbor side-contact with sufficient vertical overlap

Allowance is bounded:

```
geometric_cap = min(min(ol, ow) * BRACE_RELAX_MAX_FRACTION,
                    resolution * BRACE_RELAX_MAX_CELLS)

support_scale = (support_ratio - BRACE_RELAX_MIN_SUPPORT) / (1 - BRACE_RELAX_MIN_SUPPORT)
support_scale clipped to [0, 1]

allowance = geometric_cap * brace_factor * support_scale
```

Relaxed stability is accepted only if all are true:

1. `support_ratio >= BRACE_RELAX_MIN_SUPPORT`
2. `brace_factor >= BRACE_RELAX_MIN_BRACE`
3. `outside_distance(CoG, support_polygon) <= allowance`

This makes relaxation explicitly bounded and impossible when support/bracing is weak.

## Scoring Formula

```
score =
    WEIGHT_STABILITY * support_ratio
    + WEIGHT_CONTACT * contact_ratio
    - WEIGHT_HEIGHT * height_norm
    - WEIGHT_ROUGHNESS_DELTA * roughness_delta
```

with:

```
height_norm = z / bin_height
```

## Diagnostics
`get_last_diagnostics()` returns counters from the latest `decide_placement()` call, including:

- `candidates_generated`
- `orientation_checks`
- `rejected_bounds`
- `rejected_height`
- `rejected_min_support`
- `rejected_cfg_support`
- `rejected_stability`
- `rejected_margin`
- `accepted_candidates`
- `brace_relaxed_accepts`
- `best_found`
- `best_score`

When `config.verbose=True`, a one-line summary is printed per decision call.

## Constants

- `MIN_SUPPORT = 0.30`
- `CONTACT_TOL = 0.5`
- `MIN_CELL_CONTACT_FRACTION = 0.05`
- `LBCP_SHRINK_STACKED = 0.15`
- `BRACE_RELAX_MIN_SUPPORT = 0.45`
- `BRACE_RELAX_MIN_BRACE = 0.25`
- `BRACE_RELAX_MAX_FRACTION = 0.12`
- `BRACE_RELAX_MAX_CELLS = 2.0`

## Validation Commands
Run from `python/`:

```bash
python -m py_compile strategies/lbcp_stability/strategy.py
```

```bash
python - <<'PY'
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath('run_experiment.py')))
from config import BinConfig, ExperimentConfig, Box
from dataset.generator import generate_uniform
from simulator.bin_state import BinState
from strategies.base_strategy import get_strategy
from run_experiment import run_experiment

boxes = generate_uniform(n=10, min_dim=200.0, max_dim=350.0, seed=11)
base_bin = BinConfig(length=1200.0, width=800.0, height=2700.0, resolution=10.0)

cfg = ExperimentConfig(
    bin=base_bin,
    strategy_name='lbcp_stability',
    enable_stability=True,
    min_support_ratio=0.8,
    allow_all_orientations=False,
    render_3d=False,
    verbose=False,
)
r = run_experiment(cfg, boxes)
m = r['metrics']
print(f"lbcp_stability: completed={r.get('completed', True)} placed={m['boxes_placed']}/{m['boxes_total']} fill={m['fill_rate']:.4f} stability={m['stability_rate']:.4f} time_ms={m['computation_time_ms']:.1f}")

strat = get_strategy('lbcp_stability')
strat.on_episode_start(ExperimentConfig(bin=base_bin, strategy_name='lbcp_stability'))
_ = strat.decide_placement(Box(id=999, length=300.0, width=200.0, height=150.0, weight=1.0), BinState(base_bin))
diag = strat.get_last_diagnostics()
print('diag_has_rejected_min_support:', 'rejected_min_support' in diag)
print('diag_has_brace_counter:', 'brace_relaxed_accepts' in diag)
PY
```

Observed output:

```
lbcp_stability: completed=True placed=10/10 fill=0.0748 stability=1.0000 time_ms=44249.0
diag_has_rejected_min_support: True
diag_has_brace_counter: True
```
