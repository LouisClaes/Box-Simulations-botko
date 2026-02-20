# 3D Bin Packing — Strategy Comparison Framework

A modular Python framework for testing and comparing 3D bin packing strategies.
The simulator enforces physical constraints (bounds, overlap, stability) while
giving strategies full 3D state access for intelligent placement decisions.

---

## Quick Start

```bash
cd "python/full workflow"

# Run baseline strategy on 30 random boxes (renders 3D image)
python run_experiment.py --strategy baseline --generate 30 --render --verbose

# Custom bin + stability constraints
python run_experiment.py --strategy baseline --generate 50 \
    --bin-length 120 --bin-width 80 --bin-height 150 \
    --stability --min-support 0.8 --all-orientations --render -v

# Load existing dataset
python run_experiment.py --strategy baseline --dataset dataset/my_boxes.json --render

# Step-by-step images (one PNG per placement)
python run_experiment.py --strategy baseline --generate 20 --render-steps -v

# Animated GIF of the stacking process
python run_experiment.py --strategy baseline --generate 20 --gif --gif-fps 3 -v
```

**Output:** JSON results + 3D PNG / GIF in `output/`.

---

## Architecture

```
full workflow/
├── config.py                     # Core data models (all frozen/immutable)
├── run_experiment.py             # CLI entry point + experiment orchestration
│
├── dataset/
│   ├── generator.py              # uniform / warehouse / identical generators
│   └── loader.py                 # JSON load / save
│
├── robotsimulator/
│   ├── bin_state.py              # BinState — heightmap + placed_boxes (3D)
│   ├── validator.py              # Physical constraint validation
│   └── simulator.py              # RobotSimulator — placement orchestration
│
├── strategies/
│   ├── base_strategy.py          # Abstract interface + registry
│   └── baseline.py               # Bottom-Left-Fill (DBLF) reference
│
└── visualization/
    ├── render_3d.py              # 3D matplotlib rendering
    ├── gif_creator.py            # Animated GIF of stacking process
    └── step_logger.py            # Console step logger
```

---

## Data Flow

```
┌─────────────┐     BinState        ┌──────────────────┐
│   Strategy   │◄────────────────────│  RobotSimulator   │
│              │                     │                    │
│ decide_      │ PlacementDecision   │ attempt_placement()│
│ placement()  │────────────────────►│ record_rejection() │
└─────────────┘     (x, y, orient)  └──────────────────┘
                                          │
                                          ▼
                                    ┌──────────┐
                                    │ BinState  │
                                    │───────────│
                                    │ heightmap │
                                    │ placed_   │
                                    │   boxes   │
                                    └──────────┘
```

**Key principle:** The strategy *reads* the bin state, *proposes* a placement, and
the simulator *validates and commits* it. Strategies never modify state directly.

---

## How Strategies Access 3D State

The `BinState` object passed to `decide_placement()` gives full 3D access:

| Property / Method            | Returns                        | Use case                          |
|------------------------------|--------------------------------|-----------------------------------|
| `.heightmap`                 | `np.ndarray` (2D grid)         | Fast spatial queries              |
| `.placed_boxes`              | `List[Placement]` (frozen)     | Full 3D info per box (x,y,z,dims)|
| `.get_height_at(x,y,w,d)`   | `float` — resting z            | Where would a box land?           |
| `.get_support_ratio(...)`    | `float` — 0.0–1.0              | Is this placement stable?         |
| `.get_fill_rate()`           | `float` — volumetric %         | How full is the bin?              |
| `.get_max_height()`          | `float`                        | Peak height                       |
| `.get_surface_roughness()`   | `float`                        | Surface smoothness metric         |
| `.copy()`                    | `BinState` — deep copy         | **Lookahead / what-if simulation**|

### Example: using lookahead

```python
def decide_placement(self, box, bin_state):
    # Try a placement on a copy (doesn't affect real state)
    test_state = bin_state.copy()
    # ... simulate and evaluate ...
```

---

## Creating a New Strategy

1. Create `strategies/my_strategy.py`:

```python
from strategies.base_strategy import BaseStrategy, register_strategy
from config import Box, PlacementDecision, ExperimentConfig
from robotsimulator.bin_state import BinState

@register_strategy
class MyStrategy(BaseStrategy):
    """Custom strategy description."""
    name = "my_strategy"

    def on_episode_start(self, config: ExperimentConfig) -> None:
        super().on_episode_start(config)
        # Initialise any strategy-specific state here

    def decide_placement(self, box: Box, bin_state: BinState):
        # bin_state.placed_boxes  — list of all placed boxes (full 3D)
        # bin_state.heightmap     — 2D height grid
        # bin_state.copy()        — safe copy for what-if simulation
        #
        # Return PlacementDecision(x, y, orientation_idx) or None
        return PlacementDecision(x=0, y=0, orientation_idx=0)
```

2. Import it in `strategies/__init__.py`:

```python
import strategies.my_strategy  # registers via @register_strategy
```

3. Run it:

```bash
python run_experiment.py --strategy my_strategy --generate 30 --render -v
```

---

## CLI Reference

| Flag                 | Default | Description                                |
|----------------------|---------|--------------------------------------------|
| `--strategy NAME`    | baseline| Strategy name (from registry)              |
| `--dataset PATH`     | —       | Load boxes from a JSON file                |
| `--generate N`       | —       | Generate N random boxes instead            |
| `--gen-min`          | 5.0     | Min dimension for generated boxes          |
| `--gen-max`          | 25.0    | Max dimension for generated boxes          |
| `--seed`             | 42      | Random seed for reproducibility            |
| `--bin-length`       | 60.0    | Bin X dimension (cm)                       |
| `--bin-width`        | 40.0    | Bin Y dimension (cm)                       |
| `--bin-height`       | 60.0    | Bin Z dimension (cm)                       |
| `--resolution`       | 1.0     | Heightmap grid cell size (cm)              |
| `--stability`        | off     | Enable base support checking               |
| `--min-support`      | 0.8     | Min supported fraction (0.0–1.0)           |
| `--all-orientations` | off     | Allow all 6 rotations (default: 2 flat)    |
| `--render`           | off     | Save 3D PNG of the final result            |
| `--render-steps`     | off     | Save one PNG per placement step            |
| `--gif`              | off     | Create animated GIF of stacking process    |
| `--gif-fps`          | 2       | GIF frames per second                      |
| `--verbose` / `-v`   | off     | Step-by-step console output                |
| `--output-dir`       | output  | Directory for JSON and PNG output          |

---

## Output Format

### JSON result structure

```json
{
  "experiment": {
    "strategy_name": "baseline",
    "dataset_path": "dataset/generated_30_42.json",
    "timestamp": "2026-02-18T14:02:25",
    "config": { "bin": {...}, "enable_stability": false, ... }
  },
  "metrics": {
    "fill_rate": 0.594,
    "boxes_placed": 29,
    "boxes_total": 30,
    "boxes_rejected": 1,
    "max_height": 55.7,
    "computation_time_ms": 1287,
    "stability_rate": 1.0
  },
  "placements": [
    { "step": 0, "box_id": 0, "dims": [12, 8, 5], "position": [0, 0, 0], "orientation": 0 }
  ],
  "step_log": [ ... ]
}
```

### 3D Render

Clean 3D bar chart with colored semi-transparent boxes on a white background.
Generated with `--render` flag and saved alongside the JSON.

---

## Core Data Models

All exchange objects are **frozen** (immutable) for safe data flow:

| Class               | Mutable? | Purpose                                |
|---------------------|----------|----------------------------------------|
| `Box`               | No       | Input box with dimensions              |
| `Placement`         | No       | Validated position + oriented dims     |
| `PlacementDecision` | No       | Strategy's proposed (x, y, orient)     |
| `BinConfig`         | No       | Bin dimensions and grid resolution     |
| `StepRecord`        | No       | Logged placement attempt               |
| `ExperimentConfig`  | Yes      | Tuneable experiment parameters         |

---

## Dependencies

- Python ≥ 3.8
- numpy
- matplotlib
- Pillow (for GIF creation)

```bash
pip install numpy matplotlib Pillow
```
