# Botko BV Simulation — How It Works

This document explains the simulation model used for benchmarking 3D bin-packing
strategies on the Botko BV robotic palletizer.

The core packing logic lives in the `simulator/` package as reusable OOP classes.
The overnight script (`run_overnight_botko.py`) uses these classes for experiments.

---

## Physical Setup

```
                     ROBOT ARM
                        |
         +---------+    |    +---------+
         | Pallet  |    |    | Pallet  |
         |    0    |  (pick) |    1    |
         +---------+    |    +---------+
                        |
    =====[8][7][6][5]|[4][3][2][1]=====>  CONVEYOR BELT
                     |   pick window
                     |   (4 boxes)
              buffer visible
              (8 boxes total)
```

| Component | Value |
|-----------|-------|
| Pallets | 2 EUR pallets, 1200 x 800 mm |
| Pallet max height | 2700 mm (simulator limit) |
| Pallet close height | **1800 mm** (triggers pallet swap) |
| Conveyor buffer | 8 boxes visible on the belt |
| Pick window | First 4 boxes the robot can reach |
| Box catalog | Rajapack NL — 8 real box types (100–600 mm) |

---

## Conveyor Belt (FIFO Queue)

The conveyor is a strict **FIFO queue** — no recirculation.

```
Stream → [new boxes enter at BACK] ... [boxes exit at FRONT] → Robot / Reject
```

### When the robot picks a box

1. The robot selects a box from the pick window (front 4 positions).
2. That box is removed from the belt.
3. All boxes behind it shift forward by one position.
4. A new box enters from the stream at the back.

```
BEFORE:  [8] [7] [6] [5] | [4] [3] [2*] [1]    (* = picked)
AFTER:   [9] [8] [7] [6] | [5] [4] [3]  [1]    (9 enters from stream)
```

### When no box can be placed (rejection)

1. No box in the pick window fits on any pallet.
2. The belt **advances**: the front box exits the system permanently.
3. All remaining boxes shift forward.
4. A new box enters from the stream at the back.
5. The exited box goes to a reject/overflow bin (tracked in `buf.rejected`).

```
BEFORE:  [8] [7] [6] [5] | [4] [3] [2] [1]     (none fit)
AFTER:   [9] [8] [7] [6] | [5] [4] [3] [2]     ([1] exits, gone forever)
```

This models a real conveyor: the belt keeps moving. A box that passes
the robot without being picked is gone.

---

## Pallet Lifecycle

### Close condition

A pallet is **closed** (sealed, ready for shipping) when:

> **max_height >= 1800 mm**

This is the standard safe stacking height for EU road transport.

### What happens when a pallet closes

1. The pallet's stats are snapshotted (fill rate, effective fill, height,
   boxes placed, surface roughness, support ratio, timing).
2. A forklift removes the closed pallet.
3. A fresh empty pallet is placed on the same station.
4. The consecutive-reject counter resets to 0 (fresh pallet = new opportunity).
5. The robot continues placing boxes.

```
Step 47: Box placed on Pallet 0 → height = 1820mm → CLOSE
         Pallet 0 stats saved to closed_pallets[]
         Pallet 0 replaced with empty pallet
         Robot continues...

Step 48: Box placed on new Pallet 0 → height = 300mm → continue
```

### Active vs. closed pallets

| | Closed pallets | Active pallets |
|---|---|---|
| When | max_height >= 1800 mm | Still being filled at end of test |
| Count in avg fill? | **Yes** | **No** |
| Why | Complete — shipped | Incomplete — would wait for more boxes |

---

## Test Termination

The test ends when **either** condition is met:

1. **Stream exhausted**: all boxes have been processed (placed, rejected,
   or still in buffer) and the buffer is empty.

2. **Safety valve**: 10 consecutive rejections across ALL pallets
   (even fresh ones can't accept the remaining boxes).

At termination, any boxes still on the belt or in the stream are counted
as `remaining_boxes`. Any pallets with boxes that haven't reached the
close height are reported as `active_pallets` (not included in the
primary fill-rate metric).

---

## Strategy Evaluation

### Single-bin strategies (22 total)

These strategies decide placement for **one pallet at a time**.
The Botko simulation wraps them with external bin selection:

1. For each box in the pick window (optionally sorted by `box_selector`):
2. Try placing it on each pallet (scored by `bin_selector`).
3. Pick the (box, pallet, position) combo with the best score.
4. If nothing fits → conveyor advances.

**Box selectors** (which box to try first):
- `default` — conveyor order (FIFO)
- `biggest_volume_first` — largest box gets priority
- `biggest_footprint_first` — largest footprint (L × W)
- `heaviest_first` — heaviest box

**Bin selectors** (which pallet to prefer):
- `emptiest_first` — prefer the emptier pallet (spread load)
- `focus_fill` — prefer the fuller pallet (fill one up fast)
- `flattest_first` — prefer the pallet with lowest max height

### Multi-bin strategies (2 total)

These strategies (`tsang_multibin`, `two_bounded_best_fit`) natively
see **all pallet states** and decide both which pallet AND where to place.
No external bin selector is used.

---

## Metrics

### Primary metric: `avg_closed_fill`

Mean volumetric fill rate across all **closed** pallets:

```
avg_closed_fill = mean( placed_volume / (L × W × H) for each closed pallet )
```

where L = 1200, W = 800, H = 2700 (full pallet volume).

### Secondary metrics

| Metric | Description |
|--------|-------------|
| `avg_closed_effective_fill` | Mean of `placed_vol / (L × W × max_height)` per closed pallet |
| `pallets_closed` | Number of pallets that reached close height |
| `total_placed` | Total boxes placed across all pallets (closed + active) |
| `total_rejected` | Boxes that exited the belt without being placed |
| `remaining_boxes` | Boxes still on belt + in stream at termination |
| `ms_per_box` | Average computation time per placed box |

### Per-pallet stats (in `closed_pallets[]` and `active_pallets[]`)

| Field | Description |
|-------|-------------|
| `fill_rate` | Volumetric fill (placed_vol / full_bin_vol) |
| `effective_fill` | placed_vol / (L × W × max_height) |
| `max_height` | Tallest point on the pallet (mm) |
| `boxes_placed` | Number of boxes on this pallet |
| `surface_roughness` | Std dev of heightmap — flat = good |
| `support_mean` | Mean base-support ratio (1.0 = fully supported) |

---

## Simulator Architecture (`simulator/` package)

The packing lifecycle is baked into the simulator as reusable OOP classes:

```
simulator/
├── session.py          ← PackingSession: full orchestrator
├── conveyor.py         ← FIFOConveyor: belt model
├── close_policy.py     ← ClosePolicy ABC + implementations
├── pipeline_simulator.py ← PipelineSimulator: single-pallet physics
├── bin_state.py        ← BinState: heightmap + queries
├── validator.py        ← Placement validation
└── __init__.py         ← All public exports
```

### PackingSession — the orchestrator

```python
from simulator import PackingSession, SessionConfig, HeightClosePolicy
from config import BinConfig

config = SessionConfig(
    bin_config=BinConfig(length=1200, width=800, height=2700, resolution=10),
    num_bins=2,
    buffer_size=8,
    pick_window=4,
    close_policy=HeightClosePolicy(max_height=1800.0),
    max_consecutive_rejects=10,
)

session = PackingSession(config)
result = session.run(boxes, strategy, box_selector, bin_selector)

print(result.avg_closed_fill)        # Primary metric
print(result.pallets_closed)         # How many pallets were sealed
print(result.closed_pallets[0].to_dict())  # Per-pallet stats
```

### Close policies (pluggable)

```python
from simulator import HeightClosePolicy, RejectClosePolicy, CombinedClosePolicy

# Simple: close at 1800mm
policy = HeightClosePolicy(max_height=1800.0)

# Combined: close at height OR after 5 consecutive idle steps
policy = CombinedClosePolicy([
    HeightClosePolicy(max_height=1800.0),
    RejectClosePolicy(max_consecutive=5),
])

# Custom: subclass ClosePolicy and implement should_close()
```

### Box and bin selectors

```python
from simulator import get_box_selector, get_bin_selector

box_sel = get_box_selector("biggest_volume_first")
bin_sel = get_bin_selector("focus_fill")

result = session.run(boxes, strategy, box_sel, bin_sel)
```

### Step mode (for RL / custom control)

```python
session = PackingSession(config)
obs = session.reset(boxes)

while not obs.done:
    # obs.grippable     — boxes the robot can reach
    # obs.bin_states    — current state of each pallet
    # obs.buffer_view   — all visible boxes on belt
    # ... your decision logic ...
    result = session.step(box_id, bin_index, x, y, orientation_idx)
    if result.pallet_closed:
        print(f"Pallet closed! Fill: {result.closed_pallet_result.fill_rate:.1%}")
    obs = session.observe()

final = session.result()
```

---

## Constants

Default Botko BV constants (in `run_overnight_botko.py`):

```python
BOTKO_BINS = 2                        # Number of pallet stations
BOTKO_BUFFER_SIZE = 8                 # Visible boxes on conveyor
BOTKO_PICK_WINDOW = 4                 # Front N boxes the robot can reach
BOTKO_PALLET = BinConfig(
    length=1200, width=800,           # EUR pallet footprint (mm)
    height=2700,                      # Simulator height limit (mm)
    resolution=10,                    # Heightmap grid cell size (mm)
)
MAX_CONSECUTIVE_REJECTS = 10          # Safety valve — stop if stuck
PALLET_CLOSE_HEIGHT = 1800.0          # Close pallet when height >= this (mm)
```

---

## Running

```bash
# Full overnight run (10 datasets × 300 boxes × 3 shuffles × all strategies)
python run_overnight_botko.py

# Quick smoke test (1 dataset × 20 boxes × 1 shuffle × 4 strategies)
python run_overnight_botko.py --smoke-test

# Resume from a previous run
python run_overnight_botko.py --resume output/botko_XXXXXXXX/results.json

# Generate thesis figures from results
python analyze_botko_results.py --input output/botko_XXXXXXXX/results.json --format pdf
```

---

## Output Structure

```
output/botko_YYYYMMDD_HHMMSS/
├── results.json              ← all metrics, closed/active pallets, per-run data
├── gifs/
│   ├── dataset_00_grid.gif   ← animated GIF: top-3 strategies side-by-side
│   ├── dataset_01_grid.gif
│   └── ...
├── 01_strategy_ranking.png   ← bar chart (from analyze_botko_results.py)
├── 02_sweep_heatmap.png      ← box×bin selector heatmap
├── 03_height_profile.png     ← per-pallet height comparison
├── 04_computation_time.png   ← ms/box bar chart
├── 05_stability_roughness.png← roughness vs support scatter
└── 06_summary_table.tex      ← LaTeX table ready for thesis
```
