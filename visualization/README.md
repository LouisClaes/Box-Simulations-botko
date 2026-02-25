# Visualization Module

All rendering tools for the box-stacking framework. Every function uses the same **Matplotlib 3D bar3d** style: white background, black text, HSV colour-cycled solid boxes.

## Quick Reference

| Function | Module | What it does |
|----------|--------|-------------|
| `render_packing()` | `render_3d` | Save a single final packing state as **PNG** |
| `render_step_sequence()` | `render_3d` | Save one **PNG per step** (box-by-box) |
| `get_figure()` | `render_3d` | Return a `plt.Figure` (for embedding / custom use) |
| `render_interactive_3d()` | `interactive_viewer` | Save an **HTML** interactive viewer |
| `create_stacking_gif()` | `gif_creator` | Animated **GIF** of a single-bin stacking |
| `create_png_grid()` | `grid_creator` | Static **PNG grid** of N final packing states |
| `create_gif_grid()` | `grid_creator` | Animated **GIF grid** of N single-bin stackings |
| `create_conveyor_gif()` | `conveyor_gif_creator` | Animated **GIF**: conveyor belt + 2 pallets |
| `create_conveyor_grid_gif()` | `conveyor_gif_creator` | Animated **GIF grid** of N conveyor experiments |
| `StepLogger` | `step_logger` | Console logging of each placement step |

---

## 1. Single-Bin Static Render

Render the final packing state as a PNG image.

```python
from visualization.render_3d import render_packing

render_packing(
    placements,        # List[Placement]
    bin_config,        # BinConfig
    save_path="output/packing.png",
    title="My Experiment",
)
```

### Step-by-Step Sequence

Save one PNG per placement step:

```python
from visualization.render_3d import render_step_sequence

render_step_sequence(
    placements,
    bin_config,
    output_dir="output/steps/",  # creates step_00.png, step_01.png, ...
)
```

### Raw Figure

Get the `plt.Figure` object for custom embedding:

```python
from visualization.render_3d import get_figure

fig = get_figure(placements, bin_config, title="Custom")
fig.savefig("custom.png", dpi=150)
plt.close(fig)
```

---

## 2. Single-Bin Animated GIF

Animated GIF showing boxes being placed one by one.

```python
from visualization.gif_creator import create_stacking_gif

create_stacking_gif(
    placements,
    bin_config,
    save_path="output/stacking.gif",
    title="Baseline Strategy",
    fps=2,
)
```

---

## 3. Multi-Run Grid

### Static PNG Grid

Compare N runs side-by-side (e.g. 2×5 grid of 10 random-order runs).

```python
from visualization.grid_creator import create_png_grid

runs_data = [
    {
        "placements": placements_list,  # List[Placement]
        "run_index": i,
        "fill_rate": 0.73,
        "computation_time_ms": 42.0,
    }
    for i in range(10)
]

create_png_grid(
    runs_data, bin_config,
    save_path="output/grid.png",
    strategy_name="surface_contact",
    cols=5,       # 5 columns → 2×5 grid for 10 runs
)
```

### Animated GIF Grid

Same layout but animated step-by-step:

```python
from visualization.grid_creator import create_gif_grid

create_gif_grid(
    runs_data, bin_config,
    save_path="output/grid.gif",
    strategy_name="surface_contact",
    fps=2,
    cols=5,
)
```

---

## 4. Conveyor Belt + Dual Pallet

Real-world production visualization: boxes arrive on a conveyor belt, a buffer provides lookahead, and a robot places them onto one of 2 pallets.

### Single Experiment GIF

```python
from visualization.conveyor_gif_creator import create_conveyor_gif, ConveyorStep

# steps: List[ConveyorStep] — collected during your packing loop
# Each ConveyorStep records:
#   - box, bin_index, placed, placement
#   - buffer_snapshot (boxes visible on conveyor)
#   - pallet0_placements, pallet1_placements (cumulative)

create_conveyor_gif(
    steps,
    bin_config,
    save_path="output/conveyor.gif",
    title="Rajapack Conveyor — surface_contact",
    fps=2,
)
```

### Grid Comparison GIF

Compare N strategies in a grid (e.g. 3 rows × 1 column):

```python
from visualization.conveyor_gif_creator import create_conveyor_grid_gif

experiments = [
    {"label": "surface_contact", "steps": steps_sc},
    {"label": "walle_scoring",   "steps": steps_ws},
    {"label": "skyline",         "steps": steps_sk},
]

create_conveyor_grid_gif(
    experiments,
    bin_config,
    save_path="output/conveyor_grid.gif",
    grid_cols=1,    # 1 column → stacked vertically
    title="Strategy Comparison",
    fps=2,
)
```

### CLI Shortcut

The `validate_conveyor_packing.py` script handles everything end-to-end:

```bash
# Single strategy
python validate_conveyor_packing.py --boxes 20 --strategy surface_contact --verbose

# Grid comparison (3 strategies stacked vertically)
python validate_conveyor_packing.py --grid \
    --grid-strategies surface_contact walle_scoring skyline \
    --grid-cols 1 --boxes 20

# Grid comparison (3 strategies side-by-side)
python validate_conveyor_packing.py --grid \
    --grid-strategies surface_contact walle_scoring skyline \
    --grid-cols 3 --boxes 20
```

Output goes to `results/conveyor_validation/`.

---

## 5. Step Logger

Console logging utility for verbose experiment output:

```python
from visualization.step_logger import StepLogger

logger = StepLogger(verbose=True)

# During the packing loop:
logger.log_step(step_record)   # StepRecord from PipelineSimulator

# After the loop:
logger.print_summary(summary_dict)
records = logger.get_records()  # List[dict] for JSON export
```

---

## File Overview

```
visualization/
├── __init__.py                 # Public exports
├── render_3d.py                # Single-bin 3D renderer (PNG / Figure)
├── interactive_viewer.py       # Interactive HTML Plotly viewer
├── gif_creator.py              # Single-bin animated GIF
├── grid_creator.py             # Multi-run PNG / GIF grids
├── conveyor_gif_creator.py     # Conveyor + dual-pallet GIF (single + grid)
├── step_logger.py              # Console step logger
└── README.md                   # This file
```
