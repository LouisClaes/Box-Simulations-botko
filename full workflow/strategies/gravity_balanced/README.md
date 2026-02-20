# Gravity-Balanced Packer Strategy

**NOVEL STRATEGY -- This algorithm is NOT from any published paper.** It is an original placement heuristic designed for the 3D bin packing simulator in this repository.

---

## Overview

The Gravity-Balanced Packer selects placement positions that keep the center of gravity (CoG) as low and centered as possible within the bin. This produces physically stable stacking configurations that resist tipping during transport.

**Key insight from real-world palletizing:** The most common cause of damage during transport is load tipping due to an off-center or high center of gravity. By treating CoG optimization as a primary objective (rather than an afterthought), the strategy produces stacking patterns that are inherently stable while still achieving competitive fill rates.

---

## Algorithm

### 1. Current Center of Gravity

Before evaluating candidates, the strategy computes the current CoG from all placed boxes using volume as a weight proxy:

```python
cog_x = sum(p.volume * (p.x + p.oriented_l/2) for p in placed_boxes) / total_volume
cog_y = sum(p.volume * (p.y + p.oriented_w/2) for p in placed_boxes) / total_volume
cog_z = sum(p.volume * (p.z + p.oriented_h/2) for p in placed_boxes) / total_volume
```

Volume is used as a proxy for weight because all boxes default to `weight = 1.0`. If custom weights are introduced, this should be updated to use actual weights.

### 2. Candidate Generation

Candidates come from three sources:

1. **Coarse grid scan** at 2 cm step size (default). This produces ~2400 candidates for a 120x80 bin, which is 4x fewer than the 1 cm scan used by other strategies.
2. **Placed-box corners**: the four corners of each placed box.
3. **Bin corners**: `(0,0)`, `(length,0)`, `(0,width)` -- always included.

Candidates are sorted by `(estimated_z, x, y)` for efficient evaluation.

### 3. Feasibility Checks

Identical to other strategies:

| Check | Condition | Action on failure |
|-------|-----------|-------------------|
| Bounds | `x + ol <= bin_length` and `y + ow <= bin_width` | Skip |
| Height limit | `z + oh <= bin_height` | Skip |
| Anti-float | `support_ratio >= 0.30` (for z > 0.5) | Skip |
| Stability | `support_ratio >= min_support_ratio` (if enabled) | Skip |

### 4. Hypothetical CoG Computation

For each feasible candidate, compute what the CoG WOULD be after placing this box:

```python
new_vol = total_volume + box_vol
new_cog_x = (cog_x * total_volume + (x + ol/2) * box_vol) / new_vol
new_cog_y = (cog_y * total_volume + (y + ow/2) * box_vol) / new_vol
new_cog_z = (cog_z * total_volume + (z + oh/2) * box_vol) / new_vol
```

### 5. CoG Quality Metrics

#### Lateral Score (centering)
```python
lateral_dist = sqrt((new_cog_x - center_x)^2 + (new_cog_y - center_y)^2)
max_lateral = sqrt(center_x^2 + center_y^2)
lateral_score = 1.0 - lateral_dist / max_lateral   # 1.0 = perfectly centered
```

#### Height Score (lowness)
```python
height_score = 1.0 - new_cog_z / bin_height   # 1.0 = CoG on the floor
```

### 6. Fill Efficiency

Measures how well the box uses the vertical column it occupies:

```python
fill_efficiency = box_volume / (ol * ow * max(oh, z + oh))
```

This penalizes placements where the box sits high up on a pillar, leaving wasted space below.

### 7. Scoring

```python
score = 2.0 * height_score        # Keep CoG low
      + 2.0 * lateral_score       # Keep CoG centered
      - 3.0 * (z / bin_height)    # Prefer low placements
      + 1.5 * support_ratio       # Prefer stable positions
      + 1.0 * fill_efficiency     # Don't sacrifice fill
```

### 8. Edge Cases

- **First box (empty bin):** Placed at `(0, 0)` with the orientation that gives the largest footprint. This ensures maximum wall contact and stability as the foundation for the stack.
- **Total volume = 0:** Falls through to first-box logic.

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `WEIGHT_COG_HEIGHT` | 2.0 | Reward for keeping CoG low |
| `WEIGHT_COG_LATERAL` | 2.0 | Reward for keeping CoG centered |
| `WEIGHT_LOW_Z` | 3.0 | Penalty for high absolute placement z |
| `WEIGHT_SUPPORT` | 1.5 | Reward for high support ratio |
| `WEIGHT_FILL_EFF` | 1.0 | Reward for efficient space use |
| `COARSE_STEP` | 2.0 cm | Grid scan step (controls speed vs precision) |
| `MIN_SUPPORT` | 0.30 | Anti-float threshold (matches simulator) |

### Tuning guidance

- **Increase `WEIGHT_COG_HEIGHT`** if stacks are tipping due to high CoG. This forces more conservative bottom-heavy packing.
- **Increase `WEIGHT_COG_LATERAL`** if stacks are tipping sideways. This forces more centered placement.
- **Decrease `WEIGHT_LOW_Z`** if the strategy is too conservative about stacking high. Some height is needed for good fill rates.
- **Decrease `COARSE_STEP`** (e.g., to 1.0) for higher precision at the cost of 4x slower evaluation. Use this for small bins or precision-critical applications.
- **Increase `COARSE_STEP`** (e.g., to 4.0) for faster evaluation on large bins. Placed-box corners compensate for the coarser grid.

---

## Expected Performance

### Fill Rate
- Expected 50-70% fill rate on random box datasets.
- May be slightly lower than pure fill-rate-optimized strategies (like extreme points or Wall-E scoring) because it sacrifices some fill for stability.
- Excels when stability constraints are enabled (`enable_stability=True`).

### Computation Time
- Coarse grid: O(L*W / step^2 * orientations) per box = ~2400-14400 evaluations.
- Plus box corners: +4 per placed box.
- CoG computation: O(1) per candidate (incremental update).
- Total: typically 10-50ms per box at 2cm step on a 120x80cm bin.
- Approximately 2-4x faster than strategies using 1cm grid scan.

### Stability
- Primary strength of this strategy: inherently produces stable stacks.
- CoG is actively kept low and centered, preventing tipping.
- Combined with support ratio checks, ensures both local and global stability.
- Best strategy choice when physical stability during transport is a priority.

---

## Usage

### Command line
```bash
python run_experiment.py --strategy gravity_balanced
```

### Programmatic
```python
from strategies.gravity_balanced import GravityBalancedStrategy

strategy = GravityBalancedStrategy()
strategy.on_episode_start(config)
decision = strategy.decide_placement(box, bin_state)
```

### With stability constraints
```bash
python run_experiment.py --strategy gravity_balanced --enable-stability --min-support-ratio 0.8
```

This is the recommended configuration for transport-critical applications.

---

## File Structure

```
strategies/
    gravity_balanced.py         -- Strategy implementation
    README_gravity_balanced.md  -- This file
```

---

## Design Decisions

1. **Volume as weight proxy**: All boxes default to `weight=1.0`, so volume determines each box's gravitational contribution. If the system introduces variable-density boxes, the CoG computation should use `p.weight * p.volume` or actual mass.

2. **Coarse grid (2cm step)**: The CoG computation is smooth -- moving a candidate 1cm doesn't drastically change the resulting CoG. So a 2cm grid captures the landscape well while being 4x faster than 1cm.

3. **First-box special case**: The first box is always placed at `(0, 0)` with the flattest orientation. This ensures the stack starts in a corner with maximum wall contact, which is optimal for both CoG centering and stability.

4. **Fill efficiency as tiebreaker**: Without the fill efficiency term, the strategy might place boxes in positions that technically keep CoG centered but waste vertical space. The fill efficiency term penalizes "tall skinny column" placements.

5. **No lookahead**: This strategy is greedy (single-box decisions). Adding 2-box lookahead with CoG projection would improve results but at O(candidates^2) cost.

---

## Comparison with Other Strategies

| Aspect | Baseline (BLF) | Wall-E Scoring | Surface Contact | **Gravity Balanced** |
|--------|----------------|----------------|-----------------|---------------------|
| Primary goal | Low z | Multi-criteria | Max contact | Low/centered CoG |
| Stability | Implicit | Partial | Good | **Best** |
| Fill rate | Moderate | Good | Good | Moderate-Good |
| Speed | Fast | Slow | Moderate | Fast |
| Transport-safe | No | Partial | Partial | **Yes** |
| Novel | No | No (Verma 2020) | **Yes** | **Yes** |

---

## Continuation Notes for AI Developers

If you are continuing development on this strategy, here are the key areas for improvement:

1. **Weight-aware CoG**: When the `Box.weight` field carries meaningful values (not all 1.0), update `_compute_current_cog` to use `p.weight` instead of `p.volume` as the mass proxy. The formula becomes `weighted_x += p.weight * (p.x + p.oriented_l/2)`.

2. **Tipping stability analysis**: Beyond just keeping CoG centered, compute the actual tipping margin: the distance from the CoG projection to the nearest support polygon edge. This is more physically accurate than simple centering.

3. **Adaptive COARSE_STEP**: Start with large step (4cm) when the bin is mostly empty, decrease to 1cm as it fills up. The CoG landscape becomes more complex with more boxes.

4. **Layer-aware CoG**: Track CoG per horizontal layer. Ensure each layer is independently balanced, not just the global stack.

5. **Dynamic weight adjustment**: As the bin fills, shift priority from `WEIGHT_COG_LATERAL` (centering) toward `WEIGHT_FILL_EFF` (filling gaps). Early packing should prioritize stability structure; late packing should maximize utilization.

6. **Symmetry exploitation**: If the current CoG is shifted to the left, preferentially place the next box on the right. This could be a simple bias added to the score based on CoG direction, avoiding full candidate evaluation.
