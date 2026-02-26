# Packing Strategies — Complete Guide

## Overview

This directory contains **13 standalone placement strategies** for the 3D bin packing simulator.
Every strategy implements the same `BaseStrategy` interface and can be used interchangeably via
the `--strategy <name>` CLI flag or the `ExperimentConfig.strategy_name` parameter.

**Focus:** Maximizing fill rate of a 2K-bound pallet setup (2 pallets available, 5-10 box buffer, semi-online).

## RL Pipeline Note (2026-02-25)

RL training/evaluation now uses a unified orchestrator:

```bash
bash strategies/rl_common/hpc/train_all.sh
```

This runs the full RL pipeline (train + evaluate + thesis visualizations) and prioritizes `rl_mcts_hybrid` first for production hardening.

---

## Strategy Catalogue

### Tier 1 — Best Performers (65-75% fill)

| Strategy | Name | Type | Origin | Mean Fill | Speed |
|----------|------|------|--------|-----------|-------|
| **Surface Contact** | `surface_contact` | Novel | Original | **70.2%** | ~12s |
| **WallE Scoring** | `walle_scoring` | Heuristic | Verma et al. 2020 | **68.4%** | ~9s |
| **Best Fit Decreasing** | `best_fit_decreasing` | Heuristic | Classic BPP | **67.6%** | ~6s |
| **Hybrid Adaptive** | `hybrid_adaptive` | Novel Meta | Original | **65.6%** | ~2.5s |

### Tier 2 — Competitive (58-63% fill)

| Strategy | Name | Type | Origin | Mean Fill | Speed |
|----------|------|------|--------|-----------|-------|
| **Baseline (DBLF)** | `baseline` | Heuristic | Classic | 62.8% | ~2.5s |
| **Layer Building** | `layer_building` | Systematic | Original | 61.5% | ~3.7s |
| **EMS** | `ems` | Space-tracking | Gonçalves 2013 | 60.7% | ~1.5s |
| **Gravity Balanced** | `gravity_balanced` | Novel | Original | 60.5% | ~1.0s |
| **Extreme Points** | `extreme_points` | Space-tracking | Literature | 59.4% | **~0.1s** |
| **Lookahead** | `lookahead` | Meta/Simulation | Original | 59.1% | ~1.6s |

### Tier 3 — Baseline-level (55-59% fill)

| Strategy | Name | Type | Origin | Mean Fill | Speed |
|----------|------|------|--------|-----------|-------|
| **Skyline** | `skyline` | Profile-based | Literature | 58.8% | ~3.7s |
| **Wall Building** | `wall_building` | Systematic | Original | 56.7% | ~2.9s |
| **Column Fill** | `column_fill` | Novel | Original | 56.0% | ~0.9s |

*Results from 156 runs: 3 datasets × 2 seeds × 2 shuffles × 13 strategies, 40 boxes in 60×40×60 bin.*

---

## Strategy Interface

Every strategy implements:

```python
from strategies.base_strategy import BaseStrategy, register_strategy

@register_strategy
class MyStrategy(BaseStrategy):
    name = "my_strategy"  # Used in CLI: --strategy my_strategy

    def decide_placement(self, box: Box, bin_state: BinState) -> Optional[PlacementDecision]:
        # Read from bin_state (heightmap, placed_boxes, query methods)
        # Return PlacementDecision(x, y, orientation_idx) or None
        ...
```

### What strategies receive:
- `box`: The box to place (id, length, width, height, weight, volume)
- `bin_state`: Full 3D state with:
  - `.heightmap` — 2D numpy grid of current heights
  - `.placed_boxes` — List[Placement] with full 3D info
  - `.get_height_at(x, y, w, d)` — resting z for footprint
  - `.get_support_ratio(x, y, w, d, z)` — base support fraction
  - `.get_fill_rate()` — volumetric utilization
  - `.get_max_height()` — peak height
  - `.get_surface_roughness()` — surface smoothness
  - `.copy()` — deep copy for lookahead simulation

### What strategies return:
- `PlacementDecision(x, y, orientation_idx)` — where to place
- `None` — if box cannot be placed

### Rules:
- Do NOT modify `bin_state` (read-only)
- Do NOT compute z (simulator does that)
- Use `bin_state.copy()` for what-if simulations

---

## Strategy Descriptions

### 1. Surface Contact Maximizer (`surface_contact`) — NOVEL
Maximizes total surface area in contact with walls and existing boxes.
Computes contact for all 6 faces (left, right, front, back, top, bottom).
High contact = tight packing + natural stability.

### 2. WallE Scoring (`walle_scoring`)
Based on Verma et al. 2020. Composite scoring: height variance penalty,
valley-nestling bonus, wall-flush bonus, corner distance penalty, height penalty.
8 weighted sub-scores combined into a single deterministic decision.

### 3. Best Fit Decreasing (`best_fit_decreasing`)
Classic BPP heuristic adapted to 3D. Evaluates every valid position and picks
the one with the tightest fit — minimizing wasted space around the box.
Computes surface contact ratio for each candidate.

### 4. Hybrid Adaptive (`hybrid_adaptive`) — NOVEL
Detects packing phase (foundation/growth/completion) from bin state.
Foundation: wall-hugging, stable base building.
Growth: efficient DBLF with adjacency scoring.
Completion: gap-filling with valley priority.
Smooth transitions between phases.

### 5. Baseline DBLF (`baseline`)
Bottom-Left-Fill: scans all grid positions, picks lowest (z, x, y).
Simple, fast, reliable. The benchmark all others are compared against.

### 6. Layer Building (`layer_building`)
Builds uniform horizontal layers. Detects current layer from heightmap,
prefers orientations matching layer height, fills current layer before starting next.

### 7. Empty Maximal Spaces (`ems`)
Tracks maximal empty rectangular volumes via heightmap expansion.
Places boxes at EMS origins scored by DBLF + fit tightness.
Falls back to grid scan when no EMS works.

### 8. Gravity Balanced (`gravity_balanced`) — NOVEL
Optimizes center of gravity: keeps CoG low and centered.
Computes what CoG would be after each candidate placement.
Combines CoG quality with fill efficiency.

### 9. Extreme Points (`extreme_points`)
Generates candidate positions from corners of placed boxes.
Only evaluates these ~50-200 positions (vs ~10000 for grid scan).
Fastest strategy by 10-100x. Scores by contact ratio and DBLF.

### 10. Lookahead (`lookahead`)
Simulates each candidate placement using BinState.copy().
Evaluates resulting state with multi-factor quality function
(uniformity, remaining capacity, flatness, fill, accessible area).
Picks placement that leaves the best state for future boxes.

### 11. Skyline (`skyline`)
Tracks height profile and fills valleys first. Creates uniform
layers naturally by prioritizing low points.

### 12. Wall Building (`wall_building`)
Builds from back wall forward. Prioritizes wall contact and
adjacency. Creates dense wall-like structures.

### 13. Column Fill (`column_fill`) — NOVEL
Divides pallet into virtual columns. Fills each column vertically
before moving to next. Adaptive column sizing based on box dimensions.

---

## Running Experiments

### Single strategy:
```bash
python run_experiment.py --strategy surface_contact --generate 50 --render --verbose
```

### All strategies (batch):
```bash
python batch_runner.py --all-strategies --datasets 5 --seeds 3 --boxes 40
```

### Specific strategies:
```bash
python batch_runner.py --strategies surface_contact walle_scoring baseline --datasets 10 --seeds 5
```

### With stability constraints:
```bash
python batch_runner.py --all-strategies --stability --min-support 0.8 --datasets 5 --seeds 3
```

### EUR pallet dimensions:
```bash
python batch_runner.py --all-strategies --bin-length 120 --bin-width 80 --bin-height 150 --boxes 80
```

---

## Adding a New Strategy

1. Create `strategies/my_strategy.py`:
```python
from typing import Optional
from config import Box, PlacementDecision, ExperimentConfig, Orientation
from robotsimulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy

MIN_SUPPORT = 0.30

@register_strategy
class MyStrategy(BaseStrategy):
    name = "my_strategy"

    def on_episode_start(self, config: ExperimentConfig) -> None:
        super().on_episode_start(config)
        # Initialize any internal state here

    def decide_placement(self, box: Box, bin_state: BinState) -> Optional[PlacementDecision]:
        cfg = self.config
        bin_cfg = cfg.bin

        orientations = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        best = None
        for oidx, (ol, ow, oh) in enumerate(orientations):
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue
            # ... scan positions, score candidates ...
            # Use bin_state.get_height_at(x, y, ol, ow) for z
            # Check z > 0.5 → support_ratio >= MIN_SUPPORT
            # Return PlacementDecision(x, y, oidx)

        return best
```

2. Register in `strategies/__init__.py`:
```python
import strategies.my_strategy
```

3. Test:
```bash
python run_experiment.py --strategy my_strategy --generate 30 --verbose
```

---

## Multi-Bin Results (2K-Bound Setup)

All strategies now work in multi-bin mode via the orchestrator. Results with
120 boxes, 2 bins (60x40x60), NEVER replacement, K=7 buffer, LARGEST_FIRST:

| Strategy | Agg Fill | Placed | Rejected |
|----------|----------|--------|----------|
| **surface_contact** | **77.9%** | 78 | 42 |
| walle_scoring | 76.8% | 78 | 42 |
| best_fit_decreasing | 75.9% | 77 | 43 |
| hybrid_adaptive | 74.1% | 73 | 47 |
| baseline | 70.8% | 70 | 50 |
| extreme_points | 67.2% | 68 | 52 |
| skyline | 61.2% | 62 | 58 |
| wall_building | 60.9% | 63 | 57 |

### Buffer Size Effect (surface_contact)
| K | Agg Fill | Placed |
|---|----------|--------|
| 1 | 74.1% | 75 |
| 3 | **79.7%** | 81 |
| 5 | 75.5% | 76 |
| 7 | 77.9% | 78 |
| 10 | 75.0% | 72 |

**Finding:** K=3 is optimal. Larger buffers don't always help — overly greedy
selection can miss better box orderings.

### Running Multi-Bin Experiments

```bash
# Single strategy
python run_multibin_experiment.py --strategy surface_contact --generate 80 --verbose

# Compare all strategies
python run_multibin_experiment.py --all-strategies --generate 100 --compare

# Batch sweep
python batch_multibin_runner.py --all-strategies --sweep-buffer 1 3 5 7 10
```

See `orchestrator/README.md` for full documentation.

---

## Performance Notes

- **Extreme Points** is the fastest (0.1s) — ideal for real-time applications
- **Surface Contact** is the most accurate (70.2% single / 77.9% multi-bin) — ideal for optimization
- **Hybrid Adaptive** offers the best speed/quality tradeoff (65.6% in 2.5s)
- All 13 strategies work in both single-bin and multi-bin mode

---

## Botko BV — Real-World Production Mode

### What is Botko BV?

Botko BV is a real-world packing scenario that matches the physical robot+conveyor setup:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| **Pallets** | 2 | EUR pallets (1200×800×2700 mm) side by side |
| **Buffer** | 8 | 8 boxes are visible on the conveyor (pre-known) |
| **Pick window** | 4 | Only the first 4 boxes are physically reachable by the robot |
| **Box source** | Rajapack | Real-world box dimensions from catalog (mm) |

The robot sees 8 upcoming boxes but can only grab the nearest 4. It must decide:
1. **Which box** to pick (from the 4 grippable)
2. **Which pallet** to place it on (left or right)

### Running Botko BV

```bash
# Single strategy
python run_botko.py --strategy surface_contact --verbose

# With conveyor animation GIF
python run_botko.py --strategy walle_scoring --boxes 40 --gif

# Compare ALL strategies in Botko BV mode
python run_botko.py --all-strategies --compare

# Grid comparison GIF (top strategies stacked vertically)
python run_botko.py --all-strategies --compare --grid
```

### Output Structure

Results are saved to: `strategies/<strategy>/output/botko_bv/<timestamp>/`

```
strategies/surface_contact/output/
├── single_bin/            # Standard single-bin experiments
│   └── 20260219_135114/
│       └── results.json
├── multibin/              # Multi-bin orchestrator experiments
├── buffer/                # Buffer sweep experiments
└── botko_bv/              # ← Botko BV real-world experiments
    └── 20260219_180000/
        ├── results.json       # Full metrics + step log
        └── botko_conveyor.gif # Animated conveyor visualization
```

### Botko BV results.json format

```json
{
  "experiment": {
    "mode": "botko_bv",
    "strategy_name": "surface_contact",
    "botko_params": {
      "bins": 2,
      "buffer_size": 8,
      "pick_window": 4,
      "pallet_mm": "1200×800×2700"
    }
  },
  "metrics": {
    "total_boxes": 30,
    "total_placed": 28,
    "total_rejected": 2,
    "placement_rate": 0.933,
    "pallet_0": { "fill_rate": 0.05, "boxes": 15, "max_height": 300 },
    "pallet_1": { "fill_rate": 0.04, "boxes": 13, "max_height": 250 },
    "aggregate_fill": 0.045
  },
  "step_log": [
    {
      "step": 0,
      "box_id": 2,
      "box_dims": [220, 310, 200],
      "pallet": 0,
      "position": [0.0, 0.0, 0.0],
      "orientation": 0,
      "placed": true
    }
  ]
}
```

### Adding a Strategy Optimized for Botko BV

Any existing `BaseStrategy` works with Botko BV automatically — `run_botko.py` wraps
every single-bin strategy with the 2-pallet + 4-grippable selection logic.

To build a strategy **specifically optimized** for the Botko BV constraints:

1. **Understand the constraints**: Your strategy sees one `BinState` at a time, but `run_botko.py` calls it for each (box, pallet) combination and picks the best. To influence which box is selected, make your scoring discriminative — return high-quality `PlacementDecision`s for good fits and `None` for poor ones.

2. **Create your strategy normally**:
```python
from strategies.base_strategy import BaseStrategy, register_strategy

@register_strategy
class MyBotkoStrategy(BaseStrategy):
    name = "my_botko_strategy"

    def decide_placement(self, box, bin_state):
        # Key insight: this will be called multiple times per step
        # (once per grippable box × per pallet).
        # Return None aggressively for poor fits — this helps
        # the Botko runner pick the best (box, pallet) combo.
        ...
```

3. **Register in `strategies/__init__.py`**:
```python
import strategies.my_botko_strategy
```

4. **Test in Botko BV mode**:
```bash
python run_botko.py --strategy my_botko_strategy --boxes 30 --verbose --gif
```

5. **Compare against all strategies**:
```bash
python run_botko.py --all-strategies --compare
```

### Key Design Notes for Botko BV Strategies

- **Box selection matters more than placement**: Since only 4 of 8 boxes are grippable, choosing the RIGHT box is critical. Strategies that return `None` for poor fits naturally guide the runner to pick better boxes.
- **Balance both pallets**: The runner scores pallet emptiness. Strategies can complement this by penalizing placements that create uneven surfaces.
- **Rajapack box sizes**: Boxes are 100-430mm in each dimension. Strategies should handle this size range gracefully.
- **EUR pallet ratio**: 1200×800mm footprint — strategies that exploit the 3:2 aspect ratio perform better.

---

## For Continuing Development

Each strategy has its own folder with `strategy.py`, `__init__.py`, and `README.md`.
The `orchestrator/` module wraps them for multi-bin + buffer management.

Key improvement areas:
1. **RL training**: Train neural networks for strategies that support it (PCT, DDQN)
2. **MCTS buffer search**: Tree search over box selection order (Fang et al. 2026)
3. **Stability integration**: LBCP / stacking tree for physics-accurate stability (Gao et al. 2025)
4. **Ensemble methods**: Combine top strategies with voting/scoring
5. **TF→PyTorch port**: Port SIMPAC-2024-311 dual-bin DDQN to PyTorch
6. **Benchmark testing**: Run against PCT pre-generated data and Mendeley instances
