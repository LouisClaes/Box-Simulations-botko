# Surface Contact Maximizer Strategy

**NOVEL STRATEGY -- This algorithm is NOT from any published paper.** It is an original placement heuristic designed for the 3D bin packing simulator in this repository.

---

## Overview

The Surface Contact Maximizer selects placement positions that maximize the total surface area of a new box that is in physical contact with walls, the floor, and the faces of already-placed boxes.

The core insight is that **contact area is a powerful proxy for packing quality**:

- **High bottom contact** = strong base support = physically stable placement
- **High lateral contact** = tight packing against walls and neighbors = fewer gaps
- **High total contact** = compact structure = high volumetric fill rate

By directly optimizing for contact, the strategy simultaneously achieves good fill rates, physical stability, and compact structures -- without needing separate sub-objectives for each.

---

## Algorithm

### 1. Candidate Generation

Candidates are generated from two sources:

1. **Full grid scan** at resolution step size (default 1 cm). Every grid position is a candidate.
2. **Placed-box corners**: the four corners of each already-placed box (origin, right edge, front edge, diagonal). These are natural high-contact positions.

Candidates are sorted by `(estimated_z, x, y)` so low-height positions are evaluated first.

### 2. Feasibility Checks

For each candidate `(x, y)` and each allowed orientation `(ol, ow, oh)`:

| Check | Condition | Action on failure |
|-------|-----------|-------------------|
| Bounds | `x + ol <= bin_length` and `y + ow <= bin_width` | Skip |
| Height limit | `z + oh <= bin_height` | Skip |
| Anti-float | `support_ratio >= 0.30` (for z > 0.5) | Skip |
| Stability | `support_ratio >= min_support_ratio` (if enabled) | Skip |

### 3. Six-Face Contact Computation

For each feasible candidate, the strategy computes contact area on all six faces of the box:

#### Bottom face (ol x ow)
- **On floor** (z ~ 0): full face area counts as contact.
- **Stacked**: count heightmap cells in the footprint `[gx, gx+ol) x [gy, gy+ow)` where `|heightmap[i,j] - z| <= tolerance`. Each matching cell contributes `resolution^2` of contact area.

#### Top face (ol x ow)
- Check all placed boxes: if any box has its base at `z + oh` and overlaps in x-y, the overlap area is contact.
- Usually 0 for new placements, but important when filling gaps.

#### Left face (ow x oh, at x=0 or x boundary)
- **Against left wall** (x ~ 0): full face area `ow * oh`.
- **Against neighbor**: examine heightmap column at `x - 1`. For each cell, compute how much of the vertical range `[z, z+oh]` is covered by the neighbor's height. Sum across all cells in `[gy, gy+ow)`.

#### Right face (ow x oh, at x+ol boundary)
- **Against right wall** (`x + ol ~ bin_length`): full face area.
- **Against neighbor**: examine heightmap column at `x + ol`. Same vertical overlap calculation.

#### Back face (ol x oh, at y=0 boundary)
- **Against back wall** (y ~ 0): full face area `ol * oh`.
- **Against neighbor**: examine heightmap row at `y - 1`.

#### Front face (ol x oh, at y+ow boundary)
- **Against front wall** (`y + ow ~ bin_width`): full face area.
- **Against neighbor**: examine heightmap row at `y + ow`.

The **contact ratio** is:
```
contact_ratio = total_contact_area / (2 * (ol*ow + ol*oh + ow*oh))
```

### 4. Surface Roughness Delta

The strategy simulates the placement on a local copy of the heightmap (with a 2-cell margin) and computes the change in surface roughness:

```
roughness_delta = (roughness_after - roughness_before) / (current_roughness + 1)
```

Positive delta = surface got rougher (penalized). Negative delta = surface got smoother (rewarded, e.g., filling a valley).

### 5. Scoring

```python
score = 5.0 * contact_ratio           # Primary: maximize contact
      - 2.0 * (z / bin_height)        # Secondary: prefer low placements
      + 1.0 * support_ratio           # Tertiary: prefer stable base
      - 0.3 * roughness_delta         # Quaternary: keep surface smooth
```

The candidate with the highest score is selected.

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `WEIGHT_CONTACT` | 5.0 | Weight for contact ratio (primary driver) |
| `WEIGHT_HEIGHT` | 2.0 | Weight for height penalty (prefer low z) |
| `WEIGHT_SUPPORT` | 1.0 | Weight for support ratio bonus |
| `WEIGHT_ROUGHNESS_DELTA` | 0.3 | Weight for roughness change penalty |
| `CONTACT_TOLERANCE` | 0.5 cm | Height matching tolerance for contact detection |
| `MIN_SUPPORT` | 0.30 | Anti-float threshold (matches simulator) |

### Tuning guidance

- **Increase `WEIGHT_CONTACT`** to pack more tightly against walls and neighbors. May increase computation time as the strategy more aggressively favors tight positions.
- **Increase `WEIGHT_HEIGHT`** to prioritize bottom-up filling. Reduces contact quality at higher layers but creates more level surfaces.
- **Increase `WEIGHT_ROUGHNESS_DELTA`** to create smoother surfaces. May reduce contact when filling jagged valleys is penalized.
- **Decrease `CONTACT_TOLERANCE`** for stricter contact matching (only exact height matches count). Increase for more lenient matching on imprecise datasets.

---

## Expected Performance

### Fill Rate
- Expected 55-75% fill rate on random box datasets (comparable to Wall-E scoring).
- Excels with box sets that have compatible dimensions (many matching faces).
- Slightly lower than extreme-points on datasets with many small uniform boxes.

### Computation Time
- Full grid scan: O(L * W * orientations) per box = ~9600-57600 evaluations per box.
- Plus box-corner candidates: +4 per placed box.
- Contact computation per candidate: O(ol + ow) for lateral faces, O(footprint_cells) for bottom.
- Total: typically 20-100ms per box at 1cm resolution on a 120x80cm bin.

### Stability
- Naturally produces stable stacking due to high contact requirement.
- Bottom contact maximization ensures good support ratios.
- Wall contact creates lateral constraint.

---

## Usage

### Command line
```bash
python run_experiment.py --strategy surface_contact
```

### Programmatic
```python
from strategies.surface_contact import SurfaceContactStrategy

strategy = SurfaceContactStrategy()
strategy.on_episode_start(config)
decision = strategy.decide_placement(box, bin_state)
```

### With all orientations enabled
```bash
python run_experiment.py --strategy surface_contact --allow-all-orientations
```

---

## File Structure

```
strategies/
    surface_contact.py         -- Strategy implementation
    README_surface_contact.md  -- This file
```

---

## Design Decisions

1. **Grid scan + box corners** (not just extreme points): The full grid scan ensures we never miss a high-contact position that falls between box corners. The box corners add O(4n) candidates that are likely to be high-quality.

2. **Per-face contact computation** (not binary flush detection): Unlike the Wall-E scoring strategy which counts binary "flush or not" per face, this strategy computes actual contact AREA. A half-face contact contributes half the score, which is more nuanced and leads to better decisions.

3. **Heightmap-based lateral contact** (not placed-box iteration): Lateral contact is estimated from heightmap columns/rows adjacent to the box. This is O(ol + ow) per candidate rather than O(n_placed_boxes), which scales better as the bin fills up.

4. **Local roughness delta** (not global): Computing global roughness after each candidate would require copying the full heightmap. Instead, we compute roughness change on a local region (footprint + 2-cell margin), which is much faster.

---

## Continuation Notes for AI Developers

If you are continuing development on this strategy, here are the key areas for improvement:

1. **Candidate pruning**: Skip candidates where the box clearly cannot achieve high contact (e.g., middle of a flat open area far from walls). This could cut evaluation count by 30-50%.

2. **Adaptive weights**: Adjust scoring weights based on bin fill level. Early in packing, favor wall contact (build from corners). Late in packing, favor bottom contact (fill gaps).

3. **Multi-box lookahead**: Use `bin_state.copy()` to simulate placing 2-3 boxes ahead and pick the position that maximizes total contact over the sequence.

4. **Vectorized contact computation**: The lateral face contact computation currently uses a Python loop over placed boxes for the top face. This could be vectorized with numpy for a 2-5x speedup.

5. **Contact caching**: If the same (x, y) is evaluated with multiple orientations, the wall-contact component stays the same. Cache it.
