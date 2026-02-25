# Pallet Close Logic - Complete Documentation

## Overview

The pallet close logic prevents wasted boxes by intelligently closing pallets when they're stuck (can't accept more boxes).

## The Problem

**Without smart closing:**
- Boxes keep getting rejected indefinitely
- Wasted boxes pile up (rejected = lost)
- System gets stuck cycling through boxes that don't fit

## The Solution: Dual-Condition Close Logic

### Condition 1: LOOKAHEAD (Smart Prediction)

**Trigger:** Current 4 + next 4 boxes ALL can't fit on ANY pallet

**How it works:**
```
Buffer: 8 boxes total [B1, B2, B3, B4, B5, B6, B7, B8]
Pick window: First 4 [B1, B2, B3, B4]
Next 4: [B5, B6, B7, B8] (would be grippable after rejecting 4)

Check 1: Try placing any of [B1, B2, B3, B4] on either pallet
  → ALL fail

Check 2: Try placing any of [B5, B6, B7, B8] on either pallet
  → ALL fail

Conclusion: We'd need to reject 8 boxes to MAYBE find one that fits
Action: Close fullest pallet NOW (save those boxes)
```

**Why 8 boxes?**
- FIFO principle: Rejecting 1 box advances conveyor by 1
- Pick window = 4 boxes
- To check next 4, we'd reject current 4
- Total visibility: current 4 + next 4 = 8 boxes

### Condition 2: MAX REJECTS (Hard Limit)

**Trigger:** 8+ boxes rejected cumulatively

**Why needed:**
Prevents edge case where lookahead never triggers:
```
Cycle:
  Box 1-4: None fit
  Box 5: FITS! (places successfully) → resets counter
  Box 6-9: None fit
  Box 10: FITS! → resets counter
  ...infinite cycle...

With max rejects limit:
  After 8 total rejections → FORCE CLOSE regardless
```

**Prevents:**
- Infinite cycling (every Nth box fits)
- Wasting too many boxes
- Stuck simulations

## Implementation Details

### Location
`simulator/session.py` lines ~920-990 in `_run_singlebin_step()`

### Flow Chart
```
┌─────────────────────────────────────┐
│ Try to place box from pick window   │
└──────────┬──────────────────────────┘
           │
    ┌──────▼──────┐
    │ Box fits?   │
    └──┬───────┬──┘
       │       │
      YES     NO
       │       │
       │       ▼
       │  ┌────────────────────────────┐
       │  │ Increment reject counter   │
       │  │ _consecutive_rejects += 1  │
       │  └────────┬───────────────────┘
       │           │
       │      ┌────▼────────────────────────┐
       │      │ Check CONDITION 1:          │
       │      │ Current 4 + next 4 all fail?│
       │      └────┬────────────────────────┘
       │           │
       │      ┌────▼────────────────────────┐
       │      │ Check CONDITION 2:          │
       │      │ Rejects >= 8?               │
       │      └────┬────────────────────────┘
       │           │
       │      ┌────▼────────┐
       │      │ Either TRUE?│
       │      └──┬───────┬──┘
       │         │       │
       │        YES     NO
       │         │       │
       │    ┌────▼───────▼─────────┐
       │    │ >1 active pallet?    │
       │    │ Fill >= 50%?         │
       │    └──┬──────────┬────────┘
       │       │          │
       │      YES        NO
       │       │          │
       │  ┌────▼─────┐   │
       │  │ CLOSE    │   │
       │  │ FULLEST  │   │
       │  │ PALLET   │   │
       │  └────┬─────┘   │
       │       │         │
       │  ┌────▼─────────▼──┐
       │  │ Reset counter=0 │
       │  └─────────────────┘
       │
       ▼
  ┌──────────────────┐
  │ Place box        │
  │ Reset counter=0  │
  └──────────────────┘
```

### Code Location

**Main logic:** `simulator/session.py` lines 920-990
```python
def _run_singlebin_step(...):
    # ...try placement...

    if best_box is not None:
        # SUCCESS: Place box, reset counter
        return self.step(...)
    else:
        # FAILURE: Check close conditions
        self._consecutive_rejects += 1

        # Condition 1: Lookahead
        lookahead_stuck = check_next_4_boxes(...)

        # Condition 2: Max rejects
        max_rejects_reached = self._consecutive_rejects >= 8

        # Close if either condition true
        if (lookahead_stuck or max_rejects_reached) and can_close:
            close_fullest_pallet()
            self._consecutive_rejects = 0
            return StepResult(pallet_closed=True, ...)

        # Otherwise: reject box and continue
        return advance_conveyor()
```

## Safety Checks

### 1. Never Close Last Pallet
```python
active_pallets = [st for st in stations if st.boxes_placed > 0]
can_close = len(active_pallets) > 1
```
- Always keep at least 1 pallet open
- Prevents closing both pallets and getting stuck

### 2. Only Close Full Pallets
```python
stations_to_close = [
    st for st in stations
    if st.get_fill_rate() >= 0.5  # 50% threshold
]
```
- Won't close nearly-empty pallets
- Ensures closed pallets have meaningful utilization

### 3. Close Fullest First
```python
fullest = max(stations_to_close, key=lambda st: st.get_fill_rate())
```
- When multiple pallets qualify, close the fullest
- Maximizes efficiency of closed pallets

## Counter Management

### When Counter Resets to 0:
1. **Successful placement** (line 685)
   ```python
   if result is not None:
       self._consecutive_rejects = 0  # Box placed!
   ```

2. **Pallet closed** (line 965)
   ```python
   if stations_to_close:
       close_pallet()
       self._consecutive_rejects = 0  # Fresh start with remaining pallet
   ```

### When Counter Increments:
```python
else:  # No placement found
    self._consecutive_rejects += 1
```

## Example Scenarios

### Scenario 1: Lookahead Trigger
```
Buffer: [BoxA, BoxB, BoxC, BoxD, BoxE, BoxF, BoxG, BoxH]
Pallet 1: 72% full
Pallet 2: 68% full

Attempt: Try BoxA-D on both pallets → ALL fail
Lookahead: Try BoxE-H on both pallets → ALL fail

Decision: Close Pallet 1 (72% > 68%)
Result: Try BoxA-H again on Pallet 2 only
```

### Scenario 2: Max Rejects Trigger
```
Rejects: 0
Box 1: Reject → 1
Box 2: Reject → 2
Box 3: Reject → 3
Box 4: Reject → 4
Box 5: Place (counter resets) → 0
Box 6: Reject → 1
Box 7: Reject → 2
Box 8: Reject → 3
Box 9: Reject → 4
Box 10: Reject → 5
Box 11: Reject → 6
Box 12: Reject → 7
Box 13: Reject → 8 → FORCE CLOSE (max rejects)

Total rejected: 12 boxes (but counter hit 8 limit)
```

### Scenario 3: Safety Checks Prevent Close
```
Pallet 1: 35% full (below 50% threshold)
Pallet 2: 40% full (below 50% threshold)

Lookahead: Both conditions trigger (stuck)
Safety: No pallets >= 50% full
Decision: DON'T close, continue rejecting
```

## Benefits

1. **Prevents waste**: Stops rejecting boxes when stuck
2. **Smart timing**: Uses lookahead to predict when to close
3. **Hard limits**: Max rejects prevents infinite loops
4. **Efficiency**: Closes fullest pallet first
5. **Safety**: Never closes last pallet or nearly-empty pallets
6. **Metrics tracking**: Each closed pallet records how many boxes were rejected while it was active (duo metric)

## Configuration

Default values:
- **Max rejects:** 8 boxes
- **Min fill:** 50% (0.5)
- **Lookahead:** 4 + 4 = 8 boxes

Can be tuned in `simulator/session.py` line 958:
```python
max_rejects_reached = self._consecutive_rejects >= 8  # Tune this value
```

And line 969:
```python
if st.bin_state.get_fill_rate() >= 0.5  # Tune this threshold
```

## Metrics Tracking

### Rejected Boxes Per Pallet Duo

Each `PalletResult` includes a `rejected_boxes` field tracking how many boxes were rejected while that pallet was active as part of a duo.

**Key points:**
- When a box is rejected, it's rejected from the ENTIRE pallet duo (both pallets)
- When a pallet closes, its `rejected_boxes` count represents all rejections that occurred while it was active
- The counter resets when a pallet closes (fresh pallet + remaining pallet = new duo)
- Only CLOSED pallets count toward primary metrics (fill rate, optimization scores)
- Active pallets at the end of a run are NOT included in final metrics

**Example:**
```
Pallet 1 + Pallet 2 active
  → Box 1 rejected (both pallets)
  → Box 2 rejected
  → Box 3 placed on Pallet 1
  → Box 4 rejected
  → Box 5 rejected
  → [CLOSE] Pallet 2 (rejected_boxes: 4)

Pallet 1 + Pallet 3 (fresh) active
  → Counter reset to 0
  → Box 6 rejected (new duo)
  → Box 7 placed on Pallet 3
  → ...
```

## Testing

Run validation test:
```bash
python test_lookahead_close.py
```

Look for output:
```
[PALLET CLOSED] Station X: Y boxes, Z% fill
  Reason: LOOKAHEAD (current+next 4) | Total rejects: N
```

Or:
```
[PALLET CLOSED] Station X: Y boxes, Z% fill
  Reason: MAX REJECTS (8+) | Total rejects: 8
```
