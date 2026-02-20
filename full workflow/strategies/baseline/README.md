# Baseline Strategy — Bottom-Left-Fill (DBLF)

## Overview

The baseline strategy implements the classic **Deepest-Bottom-Left-Fill (DBLF)** algorithm,
the most widely used heuristic for 3D bin packing. It serves as the reference benchmark
against which all other strategies are compared.

## Algorithm

1. For each allowed orientation of the box:
2. Scan grid positions: x left→right, y back→front
3. For each (x, y, orient): compute resting z, check bounds & support
4. Record the candidate with lowest (z, x, y)
5. Return the best candidate, or None if nothing fits.

**Priority:** Lowest z first (stack low), then lowest x (pack left), then lowest y (pack back).

## Scoring

```
candidate = (z, x, y, orientation_idx)
best = min(candidates)  # lexicographic comparison
```

No weighted scoring — pure lexicographic DBLF ordering.

## Hyperparameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `MIN_SUPPORT` | 0.30 | Anti-float threshold (matches simulator) |
| `_scan_step` | max(1.0, resolution) | Grid scan step size |

## Performance

- **Fill rate:** ~62.8% (mean across multiple seeds/shuffles)
- **Speed:** ~2.5s for 40 boxes in 60×40×60 bin
- **Stability:** ~57% (no explicit stability optimization)

## Usage

```bash
python run_experiment.py --strategy baseline --generate 40 --verbose --render
```

## References

```
Johnson, D.S. (1974).
Fast Algorithms for Bin Packing.
Journal of Computer and System Sciences, 8(3), 272-314.
```

## Limitations

- No lookahead or state quality evaluation
- Greedy single-step decisions
- Does not optimize for surface smoothness or future packability
- Fixed priority order (z→x→y) may not be optimal for all box distributions

## Potential Improvements

1. Adaptive scan step based on box dimensions
2. Secondary scoring for tie-breaking
3. Wall/corner preference for stability
4. Height-dependent priority adjustment
