# Experiment Runner - Implementation Summary

## Overview

Successfully built the main experiment runner for Box-Simulations-botko that processes multiple datasets with different ordering strategies and packs boxes into pallets using the SimplePacker algorithm.

## Components Created

### 1. Core Data Models (`src/core/models.py`)
- **Box**: 3D box with dimensions (width, height, depth) and weight
  - Properties: `volume`, `__repr__`
  - Dimensions: 10-100cm each
  - Weight: 0.5-50kg

- **PlacedBox**: Box with position coordinates (x, y, z) on a pallet
  - Properties: `volume`, `weight`

- **Pallet**: Container for boxes with constraints
  - Dimensions: 120cm × 80cm × 200cm (standard EUR pallet)
  - Max weight: 1000kg
  - Methods:
    - `can_add_box()`: Check if box fits
    - `add_box()`: Add box to pallet (simple vertical stacking)
    - `close()`: Mark pallet as closed
  - Properties:
    - `current_height`: Height of stacked boxes
    - `current_weight`: Total weight on pallet
    - `utilization`: Volume utilization percentage
    - `is_closed`: Closed status

### 2. Dataset Generator (`src/runner/dataset.py`)
- **generate_boxes()**: Generate random boxes with optional seed for reproducibility
- **Ordering Strategies** (3 total):
  1. `random_order()`: Shuffle boxes randomly
  2. `size_sorted_order()`: Sort by volume (largest first)
  3. `weight_sorted_order()`: Sort by weight (heaviest first)
- **get_ordering_strategy()**: Strategy factory function

### 3. Simple Packing Algorithm (`src/algorithms/simple_packer.py`)
- **SimplePacker**: First-fit bin packing
  - Tries to add each box to current pallet
  - Creates new pallet when current is full
  - Closes pallets when starting a new one
  - Returns only CLOSED pallets (as specified)

### 4. Experiment Runner (`src/runner/experiment.py`)
- **ExperimentRunner**: Main orchestrator class
  - Parameters:
    - `algorithm`: Algorithm name (default: "SimplePacker")
    - `results_dir`: Output directory (default: "results")
    - `send_telegram_updates`: Enable/disable notifications

  - **run_experiment()** method:
    1. Generate experiment ID (`exp_YYYYMMDD_HHMMSS`)
    2. Send start notification (Telegram)
    3. For each dataset (default: 10):
       - Generate boxes (default: 300)
       - For each ordering strategy (3):
         - Apply ordering
         - Pack boxes into pallets
         - Collect metrics from CLOSED pallets only
         - Save interim results
       - Send progress update every 2 datasets
    4. Mark experiment complete
    5. Save final results (JSON + CSV)
    6. Send final summary (Telegram)

  - **Metrics tracking**:
    - Per-pallet: ID, boxes placed, utilization %, volume used/total
    - Aggregate: total pallets, total boxes, avg/median/min/max utilization
    - Runtime tracking

  - **Resume capability**: Saves interim results after each dataset

### 5. Overnight Run Script (`scripts/run_overnight.sh`)
- Usage: `./scripts/run_overnight.sh [datasets] [boxes]`
- Default: 10 datasets × 300 boxes = 30 experiments (3 orderings each)
- **CPU limiting**:
  - `nice -n 10`: ~50% CPU priority
  - `taskset -c 1-3`: Use cores 1-3, leave core 0 for system
- Validates venv and .env existence
- Comprehensive output with start/end timestamps

## Test Results

### Small Test (2 datasets × 30 boxes)
```
Datasets Processed: 6 (2 × 3 orderings)
Total Pallets: 67
Total Boxes: 180
Avg Utilization: 31.0%
Min Utilization: 2.9%
Max Utilization: 73.7%
Runtime: 0.037s
```

### Medium Test (1 dataset × 20 boxes via overnight script)
```
Datasets Processed: 3 (1 × 3 orderings)
Total Pallets: 24
Total Boxes: 60
Avg Utilization: 33.5%
Min Utilization: 7.0%
Max Utilization: 72.8%
Runtime: 0.0s
```

## Architecture Flow

```
┌─────────────────────────────────────────────────────────┐
│ ExperimentRunner.run_experiment()                       │
├─────────────────────────────────────────────────────────┤
│ 1. Generate experiment ID & create metrics              │
│ 2. Send Telegram start notification                     │
│ 3. FOR dataset in [0..num_datasets]:                    │
│    ├─ generate_boxes(count=300, seed=dataset_idx)       │
│    ├─ FOR strategy in [random, size_sorted, weight]:    │
│    │  ├─ ordered_boxes = strategy(boxes)                │
│    │  ├─ pallets = SimplePacker().pack(ordered_boxes)   │
│    │  ├─ FOR pallet in pallets:                         │
│    │  │  └─ IF pallet.is_closed: collect metrics        │
│    │  └─ save interim results                           │
│    └─ IF dataset % 2 == 0: send progress update         │
│ 4. Mark complete & save final results                   │
│ 5. Send Telegram final summary                          │
└─────────────────────────────────────────────────────────┘
```

## File Structure

```
src/
├── core/
│   ├── __init__.py (exports Box, Pallet, PlacedBox)
│   └── models.py (244 lines)
├── algorithms/
│   ├── __init__.py (exports SimplePacker)
│   └── simple_packer.py (63 lines)
├── runner/
│   ├── __init__.py
│   ├── dataset.py (106 lines)
│   └── experiment.py (263 lines)
└── monitoring/
    ├── telegram_notifier.py (228 lines, pre-existing)
    └── metrics.py (299 lines, pre-existing)

scripts/
└── run_overnight.sh (52 lines, updated)

results/
├── exp_YYYYMMDD_HHMMSS_interim_N.json (summary)
├── exp_YYYYMMDD_HHMMSS_interim_N_pallets.csv (per-pallet)
├── exp_YYYYMMDD_HHMMSS_final.json (full results)
└── exp_YYYYMMDD_HHMMSS_final_pallets.csv (per-pallet)
```

## Usage Examples

### Quick Test
```bash
cd /home/louis/Box-Simulations-botko
source venv/bin/activate
python test_runner.py
```

### Command Line
```bash
python -m src.runner.experiment --datasets 5 --boxes 100
```

### Overnight Run (Production)
```bash
./scripts/run_overnight.sh 10 300
```

## Integration with Existing Components

✅ **Telegram Notifier**: Uses `format_experiment_start()`, `format_dataset_milestone()`, `format_final_summary()`
✅ **Metrics System**: Uses `ExperimentMetrics`, `PalletMetrics`, `export_to_json()`, `export_to_csv()`
✅ **CPU Limiting**: Configured via `nice` and `taskset` in run_overnight.sh
✅ **Resume Capability**: Interim results saved after each dataset

## Known Limitations

1. **SimplePacker**:
   - Uses first-fit strategy (not optimal)
   - Simple vertical stacking (no 3D placement optimization)
   - Boxes too large/heavy for empty pallet are skipped

2. **Utilization**:
   - Average ~30-35% with current simple algorithm
   - Wide range (2% - 73%) indicates room for improvement

3. **Pallet Dimensions**:
   - Hardcoded to standard EUR pallet (120×80×200cm, 1000kg)

## Next Steps (for other agents)

1. **algorithms-agent**: Implement advanced packing algorithms (Best-Fit, 3D placement)
2. **testing-agent**: Write unit tests for dataset, packer, experiment runner
3. **systemd-agent**: Configure systemd service to use run_overnight.sh

## Performance Notes

- **Speed**: ~0.037s for 180 boxes (6 datasets × 30 boxes)
- **Scalability**: 10 datasets × 300 boxes = 9000 operations (~1-2 seconds estimated)
- **Memory**: Minimal (all in-memory, results saved incrementally)
- **CPU**: Limited to ~50% via nice level 10

## Files Modified

1. Created: `src/core/models.py`
2. Created: `src/runner/dataset.py`
3. Created: `src/algorithms/simple_packer.py`
4. Created: `src/runner/experiment.py`
5. Updated: `scripts/run_overnight.sh`
6. Updated: `src/core/__init__.py`
7. Updated: `src/algorithms/__init__.py`
8. Created: `test_runner.py` (test script)

**Total new code**: ~676 lines (excluding test script and comments)

## Verification

All components tested end-to-end:
- ✅ Imports work correctly
- ✅ Dataset generation produces valid boxes
- ✅ Packing algorithm creates closed pallets
- ✅ Experiment runner completes successfully
- ✅ Results saved to JSON and CSV
- ✅ Overnight script works with CPU limiting
- ✅ Metrics tracking accurate
