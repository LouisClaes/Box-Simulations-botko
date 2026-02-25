# Pallet Duo Metrics - Implementation Summary

## Overview

The system now tracks **rejected boxes per pallet duo** as a critical metric for optimization analysis.

## What Changed

### 1. PalletResult Class (session.py:286-318)

Added `rejected_boxes` field to track rejections for each closed pallet:

```python
class PalletResult:
    """Stats for one closed (or active) pallet."""
    bin_slot: int
    fill_rate: float
    effective_fill: float
    max_height: float
    boxes_placed: int
    placed_volume: float
    surface_roughness: float
    support_mean: float
    support_min: float
    ms_per_box_mean: float
    rejected_boxes: int = 0  # NEW: Duo metric
    placements: List[Placement] = field(default_factory=list)
```

### 2. to_dict() Method (session.py:306-318)

Updated to include `rejected_boxes` in serialization:

```python
def to_dict(self) -> dict:
    return {
        # ... existing fields ...
        "rejected_boxes": self.rejected_boxes,  # NEW
    }
```

### 3. PalletStation.snapshot() Method (session.py:488-518)

Added `rejected_boxes` parameter to capture duo reject count:

```python
def snapshot(self, bin_config: BinConfig, rejected_boxes: int = 0) -> PalletResult:
    """
    Take a frozen snapshot of this pallet's metrics.

    Args:
        bin_config: Bin configuration for metrics calculation
        rejected_boxes: Number of boxes rejected while this pallet was active
                       (duo metric - tracks rejections for the pallet pair)
    """
    # ... existing snapshot logic ...
    return PalletResult(
        # ... existing fields ...
        rejected_boxes=rejected_boxes,  # NEW
        placements=list(state.placed_boxes),
    )
```

### 4. All snapshot() Calls Updated

**Close Policy Trigger (line 730):**
```python
closed_result = station.snapshot(self._config.bin_config, rejected_boxes=self._consecutive_rejects)
```

**Max Rejects Fallback (line 771):**
```python
closed_result = fullest.snapshot(self._config.bin_config, rejected_boxes=self._consecutive_rejects)
```

**Active Pallets (line 820):**
```python
active.append(st.snapshot(self._config.bin_config, rejected_boxes=self._consecutive_rejects))
```

**Lookahead/Max Rejects Close (line 1005):**
```python
closed_result = fullest.snapshot(self._config.bin_config, rejected_boxes=self._consecutive_rejects)
```

## How It Works

### Reject Tracking Flow

1. **Box Rejection**: When no box in the pick window fits ANY pallet:
   - `self._consecutive_rejects` increments (session-level counter)
   - This tracks rejections for the current pallet DUO

2. **Pallet Close**: When a pallet closes (any trigger):
   - Snapshot captures `self._consecutive_rejects` as `rejected_boxes`
   - This represents ALL boxes rejected while this pallet was active
   - Counter resets to 0 for the new duo (closed pallet replaced with fresh one)

3. **Duo Lifecycle**:
   ```
   Pallet A + Pallet B (duo 1)
     → Box 1 rejected → counter = 1
     → Box 2 rejected → counter = 2
     → Box 3 placed on A → counter = 0 (reset on success)
     → Box 4 rejected → counter = 1
     → Box 5 rejected → counter = 2
     → [CLOSE B] rejected_boxes = 2

   Pallet A + Pallet C (duo 2, fresh start)
     → Counter = 0 (new duo)
     → Box 6 rejected → counter = 1
     → ...
   ```

## Key Concepts

### Duo Metric

- **Definition**: A rejection is a duo-level event (box couldn't fit on EITHER pallet)
- **Recording**: When a pallet closes, its `rejected_boxes` captures the duo's total rejects
- **Reset**: Counter resets when pallet closes (new pallet = new duo)

### Metrics Usage

**ONLY CLOSED PALLETS count toward primary metrics:**
- Fill rate optimization
- Efficiency scoring
- Benchmark comparisons

**Active pallets (still being filled):**
- Recorded for completeness
- NOT included in final optimization metrics
- Represent incomplete work (would continue in production)

## Validation

✅ Import test passed
✅ PalletResult creation with rejected_boxes passed
✅ to_dict() serialization includes rejected_boxes
✅ All snapshot() calls updated

## Next Steps

Run full validation test:
```bash
python test_lookahead_close.py
```

Expected output should now include rejected_boxes in closed pallet data.

## Documentation

See `CLOSE_LOGIC_EXPLAINED.md` for comprehensive explanation of:
- Lookahead close logic
- Max rejects limit
- Rejected boxes tracking
- Safety checks
- Example scenarios
