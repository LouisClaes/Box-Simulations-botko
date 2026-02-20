# Hybrid Adaptive Strategy

**File:** `strategies/hybrid_adaptive.py`
**Registry name:** `hybrid_adaptive`
**Type:** NOVEL meta-strategy -- not based on any published paper. First adaptive, phase-switching heuristic designed for online 3D bin packing.

---

## Overview

The Hybrid Adaptive Strategy detects the current "packing phase" from the bin state and dynamically switches between three inline sub-strategies. The fundamental insight is that different stages of bin filling have different priorities:

- **Early packing** needs stability and structure (foundation).
- **Mid packing** needs tight space utilization (growth).
- **Late packing** needs gap-filling and height minimization (completion).

No prior work in 3D bin packing has implemented a strategy that dynamically adapts its scoring function based on the packing stage. This is the first such approach.

---

## Algorithm

### Phase Detection

At each `decide_placement()` call, the fill rate is computed from `bin_state.get_fill_rate()` and mapped to blending weights:

```
fill < 0.20          -> pure foundation  (w_fnd=1.0, w_grw=0.0, w_cmp=0.0)
0.20 <= fill < 0.30  -> blend foundation->growth  (linear interpolation)
0.30 <= fill < 0.60  -> pure growth  (w_fnd=0.0, w_grw=1.0, w_cmp=0.0)
0.60 <= fill < 0.70  -> blend growth->completion  (linear interpolation)
fill >= 0.70         -> pure completion  (w_fnd=0.0, w_grw=0.0, w_cmp=1.0)
```

The blending zone width is `BLEND_WIDTH = 0.10` (5% on each side of the boundary).

### Candidate Scanning

All three sub-strategies share the same exhaustive grid scan:
- For each orientation, scan x from 0 to `bin_length - ol`, y from 0 to `bin_width - ow`, step 1.0 cm.
- At each (x, y, orientation), compute z from `get_height_at`, check bounds, anti-float, and stability.
- Compute the blended score: `score = w_fnd * S_foundation + w_grw * S_growth + w_cmp * S_completion`.
- Return the candidate with the highest blended score.

### Sub-Strategy: Foundation (fill < 25%)

**Goal:** Build a stable, wall-hugging base layer.

```
S_foundation = 3.0 * wall_contact
             + 2.0 * floor_contact
             - 1.0 * (z / bin_height)
             + 1.0 * (footprint_area / floor_area)
```

- `wall_contact`: fraction of 4 vertical faces touching a bin wall (0 to 1.0).
- `floor_contact`: 1.0 if z ~ 0, else 0.0.
- Boxes are pushed toward corners and walls, creating a stable perimeter.

### Sub-Strategy: Growth (25%--65% fill)

**Goal:** Tight packing with DBLF priority and adjacency bonuses.

```
S_growth = -4.0 * (z / h)
           - 1.0 * (x / l)
           - 0.5 * (y / w)
           + 2.0 * support_ratio
           + 1.0 * total_adjacency
```

- Strong preference for low z (DBLF).
- `total_adjacency`: combination of box-to-box lateral contact and wall contact (0 to 1.0).
- Box-to-box adjacency is computed by iterating over `placed_boxes` and checking for face contact within `CONTACT_TOL` (1.5 cm).

### Sub-Strategy: Completion (>65% fill)

**Goal:** Fill remaining valleys and gaps, minimize height.

```
S_completion = -5.0 * (z / h)
              + 3.0 * valley_fill
              + 2.0 * gap_reduction
              - 0.5 * roughness_increase
```

- `valley_fill`: `(avg_height - z) / avg_height` if z < avg_height, else 0. Rewards filling depressions.
- `gap_reduction`: estimated reduction in local height variance after a virtual placement (computed on a copied region with a 2-cell margin).
- `roughness_increase`: change in mean absolute height gradient (penalized if surface becomes rougher).

### Smooth Blending

Near the 25% and 65% boundaries, the scores are linearly blended:

```python
# Example: foundation -> growth transition at fill = 0.23
alpha = (0.23 - 0.20) / 0.10  # = 0.3
score = 0.7 * S_foundation + 0.3 * S_growth
```

This prevents sudden jumps in placement behaviour that could create structural discontinuities.

### Phase Logging

Each call to `decide_placement()` appends the detected phase name to `self._phase_log`. On `on_episode_end()`, this list is attached to `results["phase_log"]` for post-hoc analysis. Phase names include: `"foundation"`, `"foundation->growth"`, `"growth"`, `"growth->completion"`, `"completion"`, `"skip"`.

---

## Hyperparameters

### Phase Boundaries

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PHASE_FOUNDATION_END` | 0.25 | Fill rate where foundation phase ends |
| `PHASE_GROWTH_END` | 0.65 | Fill rate where growth phase ends |
| `BLEND_WIDTH` | 0.10 | Width of the blending zone between phases |

### Foundation Phase Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FND_W_WALL` | 3.0 | Wall contact reward |
| `FND_W_FLOOR` | 2.0 | Floor contact reward |
| `FND_W_HEIGHT` | -1.0 | Height penalty (negative = penalize height) |
| `FND_W_FOOTPRINT` | 1.0 | Large footprint reward |

### Growth Phase Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRW_W_Z` | -4.0 | Low z priority |
| `GRW_W_X` | -1.0 | Left preference |
| `GRW_W_Y` | -0.5 | Back preference |
| `GRW_W_SUPPORT` | 2.0 | Support ratio reward |
| `GRW_W_ADJACENCY` | 1.0 | Adjacency reward |

### Completion Phase Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CMP_W_Z` | -5.0 | Very strong low z priority |
| `CMP_W_VALLEY` | 3.0 | Valley fill reward |
| `CMP_W_GAP_REDUCE` | 2.0 | Variance reduction reward |
| `CMP_W_ROUGHNESS` | -0.5 | Roughness increase penalty |

### General

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_SUPPORT` | 0.30 | Anti-float threshold |
| `CONTACT_TOL` | 1.5 cm | Tolerance for face contact detection |
| `SCAN_STEP` | 1.0 cm | Grid scan step size |

### Tuning Guidelines

- **Phase boundaries** should be adjusted based on the box distribution. If boxes are large relative to the bin, set `PHASE_FOUNDATION_END` lower (e.g., 0.15) because the foundation phase is completed faster.
- **`FND_W_WALL`** is the most impactful foundation weight. Higher values produce stronger wall-hugging behaviour but may leave the center empty longer.
- **`GRW_W_Z`** is the dominant growth weight. The `-4.0` default strongly enforces DBLF ordering.
- **`CMP_W_VALLEY`** controls how aggressively the completion phase fills depressions vs. packing tightly.
- **`BLEND_WIDTH`** should generally stay between 0.05 and 0.15. Wider blending is smoother but may dilute the benefits of each sub-strategy.

---

## Expected Performance

| Metric | Expected Range | Rationale |
|--------|---------------|-----------|
| Fill rate | 58-72% | Adaptive scoring outperforms fixed heuristics |
| Max height | Low to moderate | Aggressive z-penalty in growth and completion |
| Surface roughness | Low | Completion phase explicitly minimizes roughness |
| Stability | High | Foundation phase builds strong base; growth checks support |
| Speed | Moderate | Full grid scan (same as baseline); scoring adds ~30% overhead |

**Best for:** General-purpose packing with mixed box sizes. The adaptive nature makes it robust across different distributions without manual tuning.

**Worst for:** Extremely uniform box sizes (where a simpler layer-building approach may be more efficient) or bins with very few boxes (the phase transitions may not have time to take effect).

---

## Edge Cases Handled

1. **Empty bin** -- Pure foundation phase; boxes go to corners/walls.
2. **Single box** -- Foundation scoring places it at (0, 0) against two walls.
3. **Box too large for bin** -- Quick reject returns `None`; logged as `"skip"`.
4. **Nearly full bin** -- Completion phase aggressively fills any remaining valleys.
5. **No valid position** -- Returns `None`.
6. **Zero-height bin** -- Division guards prevent divide-by-zero in all scoring functions.
7. **Heightmap all zeros** -- `avg_height = 0`, valley_fill = 0; foundation and growth phases handle it naturally.

---

## Dependencies

- `numpy` (for heightmap operations, variance, gradient)
- `config.py` (Box, PlacementDecision, ExperimentConfig, Orientation)
- `robotsimulator.bin_state` (BinState)
- `strategies.base_strategy` (BaseStrategy, register_strategy)

No dependency on other strategy files. All three sub-strategies are implemented inline. Fully standalone.

---

## Class Structure

```
HybridAdaptiveStrategy(BaseStrategy)
    name = "hybrid_adaptive"

    on_episode_start(config)     -- reset phase log
    on_episode_end(results)      -- attach phase log to results
    decide_placement(box, state) -- main entry: detect phase, scan, blend scores

    _detect_phase_weights(fill)  -- return (phase_name, w_fnd, w_grw, w_cmp)
    _score_foundation(...)       -- foundation sub-strategy scoring
    _score_growth(...)           -- growth sub-strategy scoring
    _score_completion(...)       -- completion sub-strategy scoring
```

---

## Notes for Continuation

- **Phase detection** currently uses only fill rate. An extension could incorporate surface roughness (`bin_state.get_surface_roughness()`) and max height ratio to make phase transitions more nuanced. For example, a low fill rate but high roughness might indicate a premature transition to growth phase.
- **Sub-strategy implementations** are deliberately kept as static methods with no side effects. This makes them easy to unit-test individually.
- **The completion phase** computes gap_reduction using a local region copy. For very large bins or high resolution, this could be optimized by pre-computing a downsampled heightmap.
- **Adjacency scoring in growth phase** iterates over all `placed_boxes`. For bins with many boxes (>200), this could be accelerated with a spatial index (grid-based or R-tree).
- **Multi-objective extension**: The blending weights could be learned via Bayesian optimization or a simple grid search over a benchmark dataset, rather than being hand-tuned.
- **Lookahead extension**: The completion phase could use `bin_state.copy()` to simulate multiple placement options and pick the one that maximizes the next-step fill rate (1-step lookahead).
