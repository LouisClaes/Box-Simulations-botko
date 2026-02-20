# Wall-E Scoring Strategy

## Overview

The Wall-E Scoring strategy is a deterministic, multi-criteria placement heuristic for online 3D bin packing. It is inspired by the approach described by **Verma et al. (2020)**, which combines several geometric quality metrics into a single composite score to rank all candidate placements.

The strategy exhaustively evaluates every feasible `(x, y, orientation)` position on the heightmap grid, computes a weighted score for each, and selects the single highest-scoring candidate. It does not use lookahead or randomness -- each box is placed greedily based on the current bin state.

## Algorithm in Plain English

1. **For each allowed orientation** of the incoming box (2 flat or up to 6 with rotations):
2. **For each grid position** `(x, y)` with a configurable step size:
   - Compute the resting height `z` using the heightmap.
   - Reject the position if the box would exceed the bin height or lack sufficient support (anti-float check).
   - Compute five sub-scores that capture different quality aspects of this placement.
   - Combine them into a single composite score `S`.
3. **Return the position with the highest S**, or `None` if no valid position exists.

## Mathematical Formulation

### Sub-scores

All sub-scores are normalised to roughly the `[0, 1]` range.

| Symbol | Name | Direction | Formula |
|--------|------|-----------|---------|
| `G_var` | Surface variance | **minimise** | `Var(neighbourhood_after_placement) / bin_height^2` |
| `G_high` | Valley nesting | **maximise** | `(max_neighbour_height - z) / bin_height` |
| `G_flush` | Flush faces | **maximise** | `count_of_flush_faces / 6` |
| `pos_pen` | Position penalty | **minimise** | `(x + y) / (bin_length + bin_width)` |
| `h_pen` | Height penalty | **minimise** | `z / bin_height` |

### Composite score

```
S = -alpha_var   * G_var
    + alpha_high  * G_high
    + alpha_flush * G_flush
    - alpha_pos   * pos_pen
    - alpha_height * h_pen
```

The candidate with the **maximum S** is selected.

### Sub-score details

**G_var (surface variance):**
Take a rectangular region around the box footprint (padded by `VARIANCE_MARGIN = 2` cells on each side). Copy this region, paint the box's top surface height into the footprint area, and compute the variance of the resulting height values. Normalise by `bin_height^2`. A lower value means the placement produces a more uniform local surface.

**G_high (valley nesting):**
Compute the maximum height in the neighbourhood around the box footprint. The score is `(max_h - z) / bin_height`. If the surrounding terrain is much higher than the resting height `z`, the box is "nestling" into a valley, which is desirable. Clamped to `[0, 1]`.

**G_flush (flush faces):**
Count how many of the box's six faces are flush (in contact) with bin walls or adjacent boxes:
- **Left face** (`x = 0`): flush with left wall
- **Right face** (`x + ol = bin_length`): flush with right wall
- **Back face** (`y = 0`): flush with back wall
- **Front face** (`y + ow = bin_width`): flush with front wall
- **Bottom face**: flush with floor (z = 0) or with the supporting surface below (partial credit based on heightmap matching)
- **Top face**: partial credit if the box top height matches adjacent column heights

The tolerance for flush detection is `FLUSH_TOLERANCE = 1.0 cm`.

**Position penalty:**
`(x + y) / (bin_length + bin_width)` -- mild bias toward placing boxes near the origin corner `(0, 0)`. This produces a more structured fill pattern.

**Height penalty:**
`z / bin_height` -- penalises high placements, encouraging the strategy to fill lower regions first and build compact layers.

## Hyperparameters

| Constant | Default | Effect |
|----------|---------|--------|
| `ALPHA_VAR` | 0.75 | Weight for surface variance penalty. Higher values produce smoother surfaces but may sacrifice wall contact. |
| `ALPHA_HIGH` | 1.0 | Weight for valley-nesting bonus. Higher values push boxes deeper into gaps. |
| `ALPHA_FLUSH` | 1.0 | Weight for flush-face bonus. Higher values maximise wall and box contact. |
| `ALPHA_POS` | 0.01 | Weight for position penalty. Kept small -- mainly a tie-breaker. Increasing it forces a strict corner-first pattern. |
| `ALPHA_HEIGHT` | 1.0 | Weight for height penalty. Higher values strongly prefer low placements. |
| `VARIANCE_MARGIN` | 2 | Grid cells of padding around the footprint for variance computation. Larger values consider a wider neighbourhood. |
| `FLUSH_TOLERANCE` | 1.0 | Maximum gap (cm) to still consider two faces as flush. |
| `MIN_SUPPORT` | 0.30 | Minimum base support ratio (matches simulator anti-float). |

### Tuning guidance

- **To prioritise compactness:** increase `ALPHA_HEIGHT` and `ALPHA_VAR`.
- **To prioritise wall contact:** increase `ALPHA_FLUSH`.
- **To prioritise filling valleys:** increase `ALPHA_HIGH`.
- **To get a strict corner-first order:** increase `ALPHA_POS` to 0.5 or higher.
- **For noisy/varied box sizes:** increase `VARIANCE_MARGIN` to smooth out local effects.

## Performance Characteristics

### Strengths
- Produces very compact, visually tidy packings with good wall contact.
- The flush-face metric naturally encourages tight clusters, reducing gaps.
- Deterministic -- same input always produces the same output.
- No lookahead or simulation copies needed (fast per-candidate evaluation).

### Time complexity
- `O(orientations * grid_l * grid_w)` per box, where grid dimensions are `bin_length / resolution` and `bin_width / resolution`.
- For the default 120x80 bin at 1 cm resolution: up to `2 * 120 * 80 = 19,200` candidates per box (flat orientations), or `6 * 120 * 80 = 57,600` with all orientations.
- Each candidate evaluation involves a small numpy region copy and variance computation, which is fast but not free.

### Expected fill rates
- Uniform random boxes (5-25 cm): **55-70%** fill rate depending on box distribution.
- Typically 5-15% higher than the baseline BLF strategy due to the multi-criteria scoring.

### Weaknesses
- Exhaustive grid scan can be slow for large bins or fine resolutions. For a 120x80 bin at 0.5 cm resolution, the grid becomes 240x160 = 38,400 cells per orientation.
- The weights are hand-tuned and may not generalise perfectly to all box distributions.
- No lookahead: the strategy is greedy and cannot reason about future boxes.
- The variance computation creates a numpy copy per candidate, which adds memory pressure for very fine grids.

## Usage

### Command line
```bash
python run_experiment.py --strategy walle_scoring --generate 40 --verbose --render
python run_experiment.py --strategy walle_scoring --generate 50 --all-orientations --render -v
python run_experiment.py --strategy walle_scoring --dataset dataset/test.json --stability --render
```

### Python API
```python
from config import ExperimentConfig, BinConfig
from run_experiment import run_experiment
from dataset.generator import generate_uniform

boxes = generate_uniform(40, min_dim=5.0, max_dim=25.0, seed=42)

config = ExperimentConfig(
    bin=BinConfig(length=120, width=80, height=150),
    strategy_name="walle_scoring",
    allow_all_orientations=False,
    verbose=True,
)

result = run_experiment(config, boxes)
print(f"Fill rate: {result['metrics']['fill_rate']:.1%}")
```

### Modifying weights
All weights are module-level constants in `walle_scoring.py`. To experiment:

```python
import strategies.walle_scoring as ws
ws.ALPHA_VAR = 1.5    # stronger smoothness preference
ws.ALPHA_FLUSH = 0.5  # weaker wall-contact preference
```

## File Structure

```
strategies/
    walle_scoring.py          -- strategy implementation
    walle_scoring_README.md   -- this file
    base_strategy.py          -- abstract base class
    __init__.py               -- auto-registration
```

## Limitations and Potential Improvements

1. **Speed:** The exhaustive grid scan is the bottleneck. Potential speedups:
   - Use a coarser initial scan (e.g. 5 cm step) to find promising regions, then refine with 1 cm step in the top-k regions.
   - Pre-compute an "extreme point" set and only evaluate those positions.
   - Vectorise the scoring with numpy broadcasting instead of the Python while-loop.

2. **Weight optimisation:** The five weights could be auto-tuned via:
   - Bayesian optimisation (e.g. Optuna) on a representative box dataset.
   - Evolutionary strategies with fill rate as the fitness function.
   - Learning from expert demonstrations (inverse RL).

3. **Lookahead:** Adding 1-step or k-step lookahead (using `bin_state.copy()`) could significantly improve quality at the cost of speed.

4. **Online adaptation:** The weights could be adjusted during an episode based on the current fill rate and remaining bin capacity (e.g. switch from flush-focused to height-focused when the bin is nearly full).

5. **G_var improvement:** Instead of local neighbourhood variance, use the global surface roughness metric (`bin_state.get_surface_roughness()`) on a state copy for more holistic smoothness evaluation.

6. **Weight for box volume:** Larger boxes could receive a bonus for being placed early/low, reserving flexibility for smaller boxes later.

## References

- Verma, S. et al. (2020). "A Generalized Reinforcement Learning Algorithm for Online 3D Bin-Packing." -- the scoring function design is adapted from the heuristic baseline described in this paper.
- The Wall-E name is a project-internal reference to the structured, layer-building approach reminiscent of the robotic compactor in the animated film.
