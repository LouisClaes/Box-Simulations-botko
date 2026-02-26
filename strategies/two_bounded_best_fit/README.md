# Two-Bounded Best-Fit Strategy

## Overview
`two_bounded_best_fit` is a native `MultiBinStrategy` for dual-pallet Botko runs.
At each decision it evaluates both active bins, finds each bin's best placement,
then routes to the globally best `(bin, x, y, orientation)` combination.

The strategy now uses:
- 3-phase fallback search (strict -> relaxed margin -> relaxed support)
- Lexicographic low-`z` preference inside each phase (DBLF-like layer bias)
- Height-aware bin bonus taper to reduce low-efficiency tower routing

## Current Scoring
Per-bin candidate score:

```text
score =
  + WEIGHT_SUPPORT * support_ratio
  - WEIGHT_HEIGHT * height_norm
  + WEIGHT_CONTACT * contact_base_ratio
  - WEIGHT_HEIGHT_GROWTH * height_growth
  - WEIGHT_POSITION * position_penalty
  - phase_penalty
```

where:
- `height_norm = z / bin_height`
- `height_growth = max(0, (z + oh - current_max_height) / bin_height)`
- `position_penalty = (x + y) / (bin_length + bin_width)`
- `phase_penalty` is `0` (phase 1), `4` (phase 2), `10` (phase 3)

Global routing score:

```text
total_score = in_bin_score + fill_bonus
fill_bonus = fill_rate * FILL_BONUS_WEIGHT * height_taper
height_taper = max(0, 1 - max_height / (bin_height * FILL_BONUS_HEIGHT_RATIO))
```

## Constants
| Constant | Value |
|---|---:|
| `MIN_SUPPORT` | `0.30` |
| `WEIGHT_SUPPORT` | `14.0` |
| `WEIGHT_HEIGHT` | `120.0` |
| `WEIGHT_CONTACT` | `8.0` |
| `WEIGHT_HEIGHT_GROWTH` | `40.0` |
| `WEIGHT_POSITION` | `0.5` |
| `WEIGHT_ROUGHNESS_DELTA` | `0.0` (disabled for speed) |
| `FILL_BONUS_WEIGHT` | `1.5` |
| `FILL_BONUS_HEIGHT_RATIO` | `0.67` |
| `PHASE2_PENALTY` | `4.0` |
| `PHASE3_PENALTY` | `10.0` |

## Validation Snapshot (Botko Config)
Config used:
- 2 bins
- 1200x800x2700 mm pallet
- close at 1800 mm
- buffer 8, pick window 4
- max consecutive rejects 10
- `allow_all_orientations = False`

Measured run (`generate_rajapack(400, seed=42)`):
- Placement rate: `97.75%`
- Avg closed fill: `33.25%`
- Avg closed effective fill: `47.51%`
- Closed pallets: `11`
- Rejected boxes: `0`
- Runtime: `~170s`

Status against PRD target for this strategy:
- Placement target (`>85%`): met
- Closed fill target (`>65%`): not yet met

## Notes
- The original reject-lockup failure mode is resolved (`0` rejects in Botko-scale runs).
- Remaining gap is geometric efficiency: pallets still close with low volumetric density.
