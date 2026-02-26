# PCT Expansion Strategy

## Core Logic
`pct_expansion` is a single-bin heuristic using PCT expansion schemes:
- `cp` when early/low fill,
- `ems` in mid-fill,
- `ev` in dense late-fill.

The selected base scheme is enriched with margin-aware, orientation-aware anchors. If no feasible placement is found, a fallback pass expands to EV-like candidates plus a dense margin-compliant grid.

## Candidate Generation
Base candidates come from CP/EMS/EV, then are augmented with:
- legal wall anchors (`margin`-compliant corners),
- box-edge anchors,
- orientation offsets (`p.x - ol - margin`, `p.y - ow - margin`, etc.),
- optional local jitter in fallback mode.

Fallback also adds a dense interior grid (capped by `MAX_FALLBACK_GRID_CANDIDATES`).

## Scoring Model
For feasible candidate `(x, y, z, ol, ow, oh)`:

```text
contact_ratio   = base cells at resting z
support_ratio   = supported base fraction
footprint_norm  = (ol * ow) / (L * W)
height_norm     = z / H
height_growth   = max(0, (z + oh - current_max_h) / H)
top_norm        = (z + oh) / H
tower_penalty   = top_norm^2 + height_growth * (1 - contact_ratio)
position_penalty= (x + y) / (L + W)

score = + 5.0 * contact_ratio
        + 1.5 * support_ratio
        + 0.8 * footprint_norm
        - 2.5 * height_norm
        - 6.0 * height_growth
        - 4.0 * tower_penalty
        - 0.3 * position_penalty
```

This explicitly rewards flat, supported dense placements and penalizes height spikes/tower growth.

## Feasibility Rules
Both primary and fallback passes keep strict validity checks:
- anti-float support `>= 0.30`,
- optional strict stability threshold,
- `is_margin_clear(...)`,
- bounds and height limit.

## Quick Validation
From `python/`:

```powershell
python -m compileall strategies/pct_expansion/strategy.py
```

```powershell
python run_experiment.py --strategy pct_expansion --generate 12 --seed 42 --verbose
```
