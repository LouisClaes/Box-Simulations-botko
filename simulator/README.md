# Simulator Package — Definitive Reference

> **MANDATORY FOR ALL AIs**: Before ANY change to ANY file in this repo, you MUST:
> 1. Read **this file in full** (you are doing that now)
> 2. Read the specific source files listed in the "Read First" column of each task
> 3. Follow the step-by-step workflow for your task type
> 4. Never skip validation steps — they prevent silent breakage

---

## Table of Contents

1. [What This Package Does](#1-what-this-package-does)
2. [File Map — Every File, Every Purpose](#2-file-map)
3. [Data Models — The Language of the System](#3-data-models)
4. [Class Reference — Complete Interface](#4-class-reference)
5. [Data Flow — How a Box Moves Through the System](#5-data-flow)
6. [Invariants — Rules That Must Never Break](#6-invariants)
7. [Strategy Interface — How Algorithms Plug In](#7-strategy-interface)
8. [Task Workflows — Step-by-Step Guides](#8-task-workflows)
9. [Critical Anti-Patterns — What Will Break Everything](#9-critical-anti-patterns)
10. [Import Graph — What Depends On What](#10-import-graph)
11. [Architecture Diagram](#11-architecture-diagram)
12. [Change Checklist — Run This Before Every PR](#12-change-checklist)

---

## 1. What This Package Does

This package is a **3D bin-packing simulation framework**. It answers the question: *"Given a stream of boxes arriving on a conveyor belt, how do you stack them onto pallets as efficiently as possible?"*

The system has four distinct concerns, each handled by separate files:

| Concern | Handled By | What It Does |
|---------|-----------|--------------|
| **3D Physics** | `bin_state.py`, `validator.py` | Tracks heightmaps, validates placements |
| **Single Pallet** | `pipeline_simulator.py` | Places boxes on one pallet, logs results |
| **Physical Belt** | `conveyor.py` | Models FIFO conveyor with pick window |
| **Orchestration** | `session.py` | Coordinates belts, multiple pallets, policies |

A **strategy** (in `strategies/`) decides *where* to place a box. The simulator decides *if* the placement is valid. The session decides *when* to close a pallet. These are intentionally separate — a strategy should never know about the belt or pallet lifecycle.

---

## 2. File Map

### 2a. Simulator Package (`python/simulator/`)

Read these before touching anything:

| File | Lines | Read First When You Want To... |
|------|-------|-------------------------------|
| `__init__.py` | ~30 | See what is exported publicly |
| `bin_state.py` | ~200 | Understand 3D state, heightmaps, spatial queries |
| `validator.py` | ~120 | Understand placement validation constraints |
| `pipeline_simulator.py` | ~350 | Understand single-pallet physics engine |
| `conveyor.py` | ~180 | Understand belt model, FIFO, pick window |
| `close_policy.py` | ~150 | Understand when pallets are sealed |
| `session.py` | ~600 | Understand multi-pallet orchestration |
| `buffer.py` | ~100 | (Legacy) Understand old buffer model |
| `multi_bin_pipeline.py` | ~300 | Understand direct multi-bin strategy runner |

### 2b. Top-Level Project Files (`python/`)

| File | Read First When You Want To... |
|------|-------------------------------|
| `config.py` | Change Box, Placement, BinConfig data models |
| `run_overnight_botko.py` | Change the main Botko experiment |
| `benchmark_all.py` | Change the quick benchmark runner |
| `run_experiment.py` | Change single-bin experiment runner |
| `batch_runner.py` | Change batch single-bin runner |
| `strategy_tester.py` | Change strategy testing harness |
| `result_manager.py` | Change output path / JSON format |

### 2c. Strategies (`python/strategies/`)

| File/Dir | Purpose |
|----------|---------|
| `base_strategy.py` | `BaseStrategy` ABC, `MultiBinStrategy` ABC, registries |
| `baseline/strategy.py` | Simplest strategy — read this first to understand the pattern |
| `surface_contact/strategy.py` | Best performing heuristic |
| `walle_scoring/strategy.py` | Highest scoring heuristic |
| `rl_common/` | Shared RL infrastructure (env, rewards, obs utils) |
| `rl_mcts_hybrid/` | Most complex strategy — read last |

### 2d. Rule: Reading Order for Any Change

```
ALWAYS READ IN THIS ORDER:
1. This README (already reading)
2. python/config.py          ← data models used everywhere
3. The specific simulator file(s) you are changing
4. The strategies that CALL the code you are changing
5. The test/runner scripts that USE those strategies
```

---

## 3. Data Models

All data models live in `python/config.py`. They are the shared language. Every other file imports from here.

### Box — Input Data (Immutable)

```python
@dataclass(frozen=True)
class Box:
    id: int
    length: float
    width: float
    height: float
    weight: float = 1.0

    @property
    def volume(self) -> float: ...

    def to_dict(self) -> dict: ...

    @staticmethod
    def from_dict(d: dict) -> "Box": ...
```

**Rules:**
- Immutable (`frozen=True`) — never try to modify a Box object
- Dimensions are in **millimeters**
- `id` uniquely identifies a box in a session — never reuse IDs
- Strategies receive Box objects as read-only input

---

### Placement — Validated Output (Immutable)

```python
@dataclass(frozen=True)
class Placement:
    box_id: int
    x: float          # left edge, mm
    y: float          # front edge, mm
    z: float          # bottom edge, mm (computed by simulator, NOT strategy)
    oriented_l: float # length after rotation
    oriented_w: float # width after rotation
    oriented_h: float # height after rotation
    orientation_idx: int
    step: int         # step counter when placed

    @property
    def x_max(self) -> float: x + oriented_l
    @property
    def y_max(self) -> float: y + oriented_w
    @property
    def z_max(self) -> float: z + oriented_h
    @property
    def volume(self) -> float: ...
```

**Critical:** `z` is **never set by a strategy**. The simulator computes it from the heightmap. Strategies propose `(x, y, orientation_idx)` only.

---

### PlacementDecision — Strategy Output (Immutable)

```python
@dataclass(frozen=True)
class PlacementDecision:
    x: float
    y: float
    orientation_idx: int
```

**This is what strategies return.** No `z`. No `box_id`. The simulator fills in the rest after validation.

---

### MultiBinDecision — Multi-Bin Strategy Output (Immutable)

```python
@dataclass(frozen=True)
class MultiBinDecision:
    bin_index: int    # which pallet to place on
    x: float
    y: float
    orientation_idx: int
```

**Only used by `MultiBinStrategy` subclasses.** Single-bin strategies return `PlacementDecision`, never `MultiBinDecision`.

---

### BinConfig — Physical Pallet Dimensions (Immutable)

```python
@dataclass(frozen=True)
class BinConfig:
    length: float = 1200.0   # mm, X axis
    width: float = 800.0     # mm, Y axis
    height: float = 2700.0   # mm, Z axis
    resolution: float = 10.0 # mm per grid cell

    @property
    def grid_l(self) -> int: ceil(length / resolution)   # 120 for standard
    @property
    def grid_w(self) -> int: ceil(width / resolution)    # 80 for standard
    @property
    def volume(self) -> float: length * width * height
```

**Standard Botko setup:** `BinConfig(1200, 800, 2700, 10)` → 120×80 grid.

---

### Orientation — Rotation Utility

```python
class Orientation:
    @staticmethod
    def get_all(l, w, h) -> List[Tuple[float, float, float]]:
        """Up to 6 unique axis-aligned rotations."""

    @staticmethod
    def get_flat(l, w, h) -> List[Tuple[float, float, float]]:
        """2 flat-base rotations only (Z-axis rotation)."""
```

`orientation_idx` in PlacementDecision indexes into whichever list the strategy is using. If `allow_all_orientations=True`, use `get_all()`; otherwise `get_flat()`.

---

### ExperimentConfig — Single-Pallet Run Settings

```python
@dataclass
class ExperimentConfig:
    bin: BinConfig
    strategy_name: str
    dataset_path: str
    enable_stability: bool = False
    min_support_ratio: float = 0.8
    allow_all_orientations: bool = False
    render_3d: bool = True
    verbose: bool = False
```

Passed to `PipelineSimulator` and to `strategy.on_episode_start()`.

---

## 4. Class Reference

### BinState — 3D Pallet State

**File:** `simulator/bin_state.py`

Represents the current 3D state of one pallet as a heightmap + list of placements.

```python
class BinState:
    config: BinConfig              # pallet dimensions
    heightmap: np.ndarray          # shape (grid_l, grid_w), dtype float64
    placed_boxes: List[Placement]  # append-only during episode
    _step_counter: int             # increments on apply_placement()
```

**Methods strategies can safely call (read-only):**

```python
def get_height_at(self, x: float, y: float, w: float, d: float) -> float:
    """Max height in footprint [x, x+w) × [y, y+d). Used to find resting z."""

def get_support_ratio(self, x: float, y: float, w: float, d: float, z: float) -> float:
    """Fraction of base cells at height z (within tolerance). 1.0 for floor."""

def get_fill_rate(self) -> float:
    """sum(placed volumes) / bin_config.volume. Range [0, 1]."""

def get_max_height(self) -> float:
    """Peak z in heightmap."""

def get_surface_roughness(self) -> float:
    """Mean absolute height diff between neighboring cells."""

def copy(self) -> "BinState":
    """Deep copy. Safe for lookahead without affecting real state."""
```

**Method the simulator calls (NEVER call from strategy):**

```python
def apply_placement(self, placement: Placement) -> None:
    """Updates heightmap and placed_boxes. PRIVATE TO SIMULATOR."""
```

---

### PipelineSimulator — Single-Pallet Physics Engine

**File:** `simulator/pipeline_simulator.py`

```python
class PipelineSimulator:
    def __init__(self, config: ExperimentConfig): ...

    @property
    def config(self) -> ExperimentConfig: ...

    def get_bin_state(self) -> BinState:
        """Returns the live BinState (same object reference each call)."""

    def attempt_placement(
        self,
        box: Box,
        x: float,
        y: float,
        orientation_idx: int
    ) -> Optional[Placement]:
        """
        Physics pipeline:
        1. Resolve orientation_idx → (ol, ow, oh)
        2. z = bin_state.get_height_at(x, y, ol, ow)
        3. validate_placement(...)
        4. If valid: apply to bin_state, return Placement
        5. If invalid: log failure, return None
        """

    def record_rejection(self, box: Box, reason: str) -> None:
        """Call when strategy returns None (no placement found)."""

    def get_step_log(self) -> List[StepRecord]:
        """Immutable log of all placement attempts."""

    def get_summary(self) -> dict:
        """Aggregated metrics dict."""
```

---

### FIFOConveyor — Physical Belt

**File:** `simulator/conveyor.py`

```python
class FIFOConveyor:
    def __init__(self, boxes: List[Box], buffer_size: int = 8, pick_window: int = 4): ...

    @property
    def grippable(self) -> List[Box]:
        """visible[:pick_window] — boxes robot can reach."""

    @property
    def is_empty(self) -> bool:
        """True iff visible is empty AND stream exhausted."""

    @property
    def stream_remaining(self) -> int: ...

    @property
    def total_remaining(self) -> int: ...

    @property
    def total_loaded(self) -> int: ...

    @property
    def total_rejected(self) -> int: ...

    def pick(self, box_id: int) -> Optional[Box]:
        """Remove box from grippable by ID. Shifts belt, refills from stream."""

    def advance(self) -> Optional[Box]:
        """Front box exits rejected. Shifts belt, refills from stream."""

    def snapshot(self) -> List[Box]:
        """Shallow copy of visible for logging."""
```

---

### ClosePolicy — When to Seal a Pallet

**File:** `simulator/close_policy.py`

```python
class ClosePolicy(ABC):
    name: str = "base"

    @abstractmethod
    def should_close(self, bin_state: BinState, pallet_stats: dict) -> bool: ...

    def describe(self) -> str: ...
```

`pallet_stats` keys:
- `"consecutive_idle"` — steps since last placement on this pallet
- `"boxes_placed"` — total boxes placed on this pallet
- `"total_rejects"` — failed attempts on this pallet
- `"global_consecutive_rejects"` — cumulative rejects across all pallets

**Implementations:**

| Class | Close Condition |
|-------|----------------|
| `HeightClosePolicy(max_height)` | `bin_state.get_max_height() >= max_height` |
| `FillClosePolicy(min_fill)` | `bin_state.get_fill_rate() >= min_fill` |
| `RejectClosePolicy(max_consecutive)` | `pallet_stats["consecutive_idle"] >= max_consecutive` |
| `CombinedClosePolicy(policies)` | ANY child returns True |
| `NeverClosePolicy()` | Always False |

---

### PackingSession — Multi-Pallet Orchestrator

**File:** `simulator/session.py`

```python
class SessionConfig:
    bin_config: BinConfig = BinConfig()
    num_bins: int = 2
    buffer_size: int = 8
    pick_window: int = 4
    close_policy: ClosePolicy = HeightClosePolicy(1800.0)
    max_consecutive_rejects: int = 10
    enable_stability: bool = False
    min_support_ratio: float = 0.8
    allow_all_orientations: bool = False

class PackingSession:
    def __init__(self, config: SessionConfig): ...

    # Mode 1: Batch (recommended)
    def run(
        self,
        boxes: List[Box],
        strategy,                    # BaseStrategy or MultiBinStrategy
        box_selector: BoxSelector = None,
        bin_selector: BinSelector = None,
        on_step: Callable = None,    # callback for GIF/logging
    ) -> SessionResult: ...

    # Mode 2: Step (for RL / custom control)
    def reset(self, boxes: List[Box], strategy_name: str = "") -> StepObservation: ...
    def observe(self) -> StepObservation: ...
    def step(self, box_id: int, bin_index: int, x: float, y: float, orientation_idx: int) -> StepResult: ...
    def result(self) -> SessionResult: ...
```

**Key result dataclasses:**

```python
@dataclass
class SessionResult:
    total_boxes: int
    total_placed: int
    total_rejected: int
    remaining_boxes: int
    closed_pallets: List[PalletResult]   # PRIMARY METRIC SOURCE
    active_pallets: List[PalletResult]   # metadata only
    elapsed_ms: float
    consecutive_rejects_triggered: bool

    @property
    def avg_closed_fill(self) -> float:  # PRIMARY METRIC
        """Mean fill of CLOSED pallets only. Active pallets excluded."""

    @property
    def placement_rate(self) -> float:
        """total_placed / total_boxes"""

@dataclass
class PalletResult:
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
    rejected_boxes: int = 0
    placements: List[Placement] = field(default_factory=list)

@dataclass
class StepObservation:
    grippable: List[Box]       # front pick_window boxes robot can reach
    buffer_view: List[Box]     # all visible boxes
    bin_states: List[BinState] # current state of each pallet
    stream_remaining: int      # boxes waiting to enter belt
    done: bool
    step_num: int

@dataclass
class StepResult:
    placed: bool
    placement: Optional[Placement]
    box: Optional[Box]
    bin_index: int = -1
    pallet_closed: bool = False
    closed_pallet_result: Optional[PalletResult] = None
```

---

### BoxSelector and BinSelector

**File:** `simulator/session.py`

```python
class BoxSelector(ABC):
    def sort(self, grippable: List[Box]) -> List[Box]:
        """Reorder grippable boxes by preference. First = tried first."""

# Implementations: FIFOBoxSelector, LargestVolumeFirst, LargestFootprintFirst, HeaviestFirst
# Registry: BOX_SELECTORS = {"fifo": ..., "largest_volume": ..., ...}
# Lookup: get_box_selector(name: str) -> BoxSelector

class BinSelector(ABC):
    def score(self, bin_index: int, bin_state: BinState, box: Box) -> float:
        """Higher score = more preferred. Used by session to pick best bin."""

# Implementations: EmptiestFirst, FocusFill, FlatFirst
# Registry: BIN_SELECTORS = {"emptiest": ..., "focus_fill": ..., "flat_first": ...}
# Lookup: get_bin_selector(name: str) -> BinSelector
```

---

### Validator

**File:** `simulator/validator.py`

```python
def validate_placement(
    heightmap: np.ndarray,
    bin_config: BinConfig,
    x: float, y: float, z: float,
    oriented_l: float, oriented_w: float, oriented_h: float,
    enable_stability: bool = False,
    min_support_ratio: float = 0.8,
) -> bool:
    """
    Raises:
      OutOfBoundsError        — box extends outside pallet
      OverlapError            — box clips existing boxes (z too low)
      FloatingError           — <30% of base supported (ALWAYS enforced)
      UnstablePlacementError  — <min_support_ratio (only if enable_stability=True)
    Returns True on success.
    """
```

**Constant (never change without updating strategies):**
```python
MIN_ANTI_FLOAT_RATIO = 0.30  # always enforced, even if stability disabled
```

---

## 5. Data Flow

### Complete Box Lifecycle

```
boxes (List[Box])
    │
    ▼
FIFOConveyor
    │  buffer_size=8 boxes visible, pick_window=4 reachable
    │  FIFO: front exits first, stream loads from back
    │
    ▼
PackingSession._run_singlebin_step()  OR  _run_multibin_step()
    │
    │  SINGLE-BIN PATH:
    │  ┌─ For each box in grippable (sorted by BoxSelector):
    │  │    For each pallet station:
    │  │      bin_state = station.bin_state             ← READ ONLY
    │  │      decision = strategy.decide_placement(box, bin_state)
    │  │      if decision: score = bin_selector.score(bin_idx, bin_state, box)
    │  │    Track best (box, bin, decision) by highest score
    │  └─ Call session.step(best_box, best_bin, x, y, orient)
    │
    │  MULTI-BIN PATH:
    │  ┌─ For each box in grippable:
    │  │    bin_states = [all active pallets]
    │  │    decision = strategy.decide_placement(box, bin_states)  ← MultiBinDecision
    │  └─ Call session.step(box.id, decision.bin_index, x, y, orient)
    │
    ▼
PackingSession.step(box_id, bin_index, x, y, orient)
    │
    │  1. conveyor.pick(box_id)
    │  2. station[bin_index].sim.attempt_placement(box, x, y, orient)
    │     │
    │     │  a. Resolve orient → (ol, ow, oh)
    │     │  b. z = bin_state.get_height_at(x, y, ol, ow)
    │     │  c. validate_placement(heightmap, bin_config, x, y, z, ol, ow, oh)
    │     │     ├─ bounds check
    │     │     ├─ overlap check
    │     │     ├─ anti-float (≥30%)
    │     │     └─ stability (if enabled)
    │     │  d. If valid: bin_state.apply_placement(placement)
    │     │               ← heightmap updated
    │     │               ← placed_boxes updated
    │     │  e. Return Placement or None
    │     │
    │  3. If placed: check close_policy.should_close(bin_state, pallet_stats)
    │     If True: snapshot → closed_pallets, replace with fresh pallet
    │  4. If not placed: advance conveyor (box permanently lost)
    │     Check consecutive_rejects safety valve
    │
    ▼
SessionResult
    ├─ closed_pallets → avg_closed_fill (PRIMARY METRIC)
    └─ active_pallets → metadata
```

### How Strategy Returns Flow Through the System

```
STRATEGY RETURNS PlacementDecision(x=200, y=150, orientation_idx=0)
    │
    ▼
PipelineSimulator.attempt_placement(box, x=200, y=150, orientation_idx=0)
    │
    ├─ Resolves orientation_idx → (ol=400, ow=300, oh=200)
    ├─ z = heightmap at [200..600) × [150..450) = 250mm (gravity drop)
    ├─ Validates: bounds OK, no overlap, 85% supported
    │
    ▼
Returns Placement(box_id=5, x=200, y=150, z=250, ol=400, ow=300, oh=200, ...)
    │
    ▼
bin_state.apply_placement(placement)
    ├─ heightmap[20:60, 15:45] = max(current, 250+200=450)
    └─ placed_boxes.append(placement)
```

---

## 6. Invariants

These are rules that MUST hold at all times. Breaking any invariant causes silent corruption.

### BinState Invariants

1. `heightmap[i, j]` is always in range `[0.0, bin_config.height]`
2. Every Placement in `placed_boxes` satisfies:
   - `0 ≤ x ≤ x+oriented_l ≤ bin_config.length`
   - `0 ≤ y ≤ y+oriented_w ≤ bin_config.width`
   - `0 ≤ z ≤ z+oriented_h ≤ bin_config.height`
   - No overlap with any other placement
3. `placed_boxes` is **append-only** — never remove or reorder
4. All Placement objects are frozen (immutable)
5. Heightmap is always synchronized with placed_boxes — never edit heightmap directly

### Validator Invariants

1. `MIN_ANTI_FLOAT_RATIO = 0.30` is **always enforced** (even if `enable_stability=False`)
2. Stability check (`min_support_ratio`) is additional, only when `enable_stability=True`
3. `z` must equal `get_height_at(x, y, oriented_l, oriented_w)` — gravity always applies
4. Validation functions are **stateless** — they never modify anything

### PipelineSimulator Invariants

1. Every call to `attempt_placement()` adds exactly one entry to `_step_log`
2. `_step_counter` is monotonically increasing — never reset during a run
3. Box objects are never mutated
4. Each Placement's `step` field equals `_step_counter` at time of placement

### FIFOConveyor Invariants

1. **No recirculation** — boxes never re-enter the visible window after pick or advance
2. `visible[0]` is always the oldest box (exits first)
3. `len(visible) ≤ buffer_size` always
4. After `pick()` or `advance()`, `_refill()` is called automatically — never call manually
5. No box is ever silently lost: every box either becomes a Placement or a rejection

### PackingSession Invariants

1. `avg_closed_fill` counts **closed pallets only** — active pallets do not contribute
2. The last active pallet is **never closed** (requires at least one active pallet always)
3. A pallet is only closed when `close_policy.should_close()` returns True
4. `consecutive_rejects` resets to 0 on every successful placement or pallet close
5. `total_placed + total_rejected + remaining_boxes == total_boxes` always

### Strategy Invariants

1. **NEVER mutate `bin_state`** — it is a reference to live state
2. **NEVER mutate `box`** — it is frozen
3. **NEVER return a Placement** — return `PlacementDecision` or `MultiBinDecision` only
4. **NEVER return z** — the simulator computes z; strategies provide only `(x, y, orient_idx)`
5. Return `None` when no valid placement exists — never guess an invalid position
6. `bin_state.copy()` is safe for lookahead without corrupting real state

---

## 7. Strategy Interface

### How to Create a New Single-Bin Strategy

**File to model from:** `strategies/baseline/strategy.py`

```python
from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy

@register_strategy
class MyNewStrategy(BaseStrategy):
    name = "my_new_strategy"   # UNIQUE name, used in registry

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Called once before the first box. Store config, reset state."""
        super().on_episode_start(config)
        self._config = config
        # Initialize any per-episode state here

    def on_episode_end(self, results: dict) -> None:
        """Called after the last box. Clean up if needed."""
        pass

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Your algorithm here.

        Available reads:
          box.length, box.width, box.height, box.weight, box.volume
          bin_state.heightmap           (numpy array, read-only)
          bin_state.placed_boxes        (List[Placement], read-only)
          bin_state.get_height_at(x, y, w, d)
          bin_state.get_support_ratio(x, y, w, d, z)
          bin_state.get_fill_rate()
          bin_state.get_max_height()
          bin_state.copy()              (safe for lookahead)

        Config reads:
          self._config.bin.length       (pallet length, default 1200)
          self._config.bin.width        (pallet width, default 800)
          self._config.bin.height       (pallet height, default 2700)
          self._config.bin.resolution   (grid cell size, default 10)
          self._config.allow_all_orientations

        Returns:
          PlacementDecision(x, y, orientation_idx)   if a valid spot found
          None                                        if no valid spot

        NEVER return:
          Placement (that is the simulator's job, not the strategy's)
          Negative coordinates
          Coordinates where box would exceed pallet bounds
        """
        config = self._config.bin
        orientations = (
            Orientation.get_all(box.length, box.width, box.height)
            if self._config.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        for orient_idx, (ol, ow, oh) in enumerate(orientations):
            # Scan grid for valid placement positions
            step = config.resolution
            x = 0.0
            while x + ol <= config.length:
                y = 0.0
                while y + ow <= config.width:
                    z = bin_state.get_height_at(x, y, ol, ow)
                    if z + oh <= config.height:
                        # Valid position found
                        return PlacementDecision(x=x, y=y, orientation_idx=orient_idx)
                    y += step
                x += step

        return None  # No position found
```

**Registration:** The `@register_strategy` decorator automatically adds your strategy to `STRATEGY_REGISTRY` with key `"my_new_strategy"`. No other file needs editing.

---

### How to Create a New Multi-Bin Strategy

```python
from config import Box, BinConfig
from simulator.bin_state import BinState
from strategies.base_strategy import MultiBinStrategy, register_multibin_strategy

@register_multibin_strategy
class MyMultiBinStrategy(MultiBinStrategy):
    name = "my_multibin_strategy"

    def decide_placement(
        self,
        box: Box,
        bin_states: List[BinState],  # ALL active bins
    ) -> Optional[MultiBinDecision]:
        """
        Same rules as single-bin, but you also choose which bin.
        Returns MultiBinDecision(bin_index=..., x=..., y=..., orientation_idx=...)
        """
        for bin_idx, bin_state in enumerate(bin_states):
            # Try to place on this bin
            # ...
            if found:
                return MultiBinDecision(bin_index=bin_idx, x=x, y=y, orientation_idx=idx)
        return None
```

---

### Strategy Registry Lookup

```python
from strategies.base_strategy import STRATEGY_REGISTRY, MULTIBIN_STRATEGY_REGISTRY

# Get a strategy by name
StrategyClass = STRATEGY_REGISTRY["my_new_strategy"]
strategy = StrategyClass()

# All registered names
print(list(STRATEGY_REGISTRY.keys()))
```

---

## 8. Task Workflows

### Task A: Add a New Strategy

```
STEP 1: Read
  - This README (done)
  - python/config.py (Box, PlacementDecision, BinConfig)
  - simulator/bin_state.py (BinState methods)
  - strategies/base_strategy.py (ABCs, registries)
  - strategies/baseline/strategy.py (pattern to follow)

STEP 2: Create directory
  mkdir python/strategies/my_strategy/
  touch python/strategies/my_strategy/__init__.py
  create python/strategies/my_strategy/strategy.py

STEP 3: Implement (follow template in Section 7 above)
  - name = "my_strategy"  ← must be globally unique
  - Decorate with @register_strategy
  - Implement decide_placement() returning PlacementDecision or None

STEP 4: Register the module
  Open strategies/__init__.py or base_strategy.py
  Add import: from strategies.my_strategy.strategy import MyNewStrategy
  (The @register_strategy decorator handles the rest)

STEP 5: Test
  Run: python strategy_tester.py --strategy my_strategy
  Confirm: strategy appears in STRATEGY_REGISTRY output
  Confirm: no validation errors in placement attempts
```

---

### Task B: Modify How Placements Are Validated

```
STEP 1: Read
  - This README (done)
  - simulator/validator.py (full file — ~120 lines)
  - simulator/pipeline_simulator.py (how validator is called)
  - simulator/bin_state.py (get_height_at, get_support_ratio)

STEP 2: Understand what you're changing
  - validate_placement() raises exceptions on failure, returns True on success
  - PipelineSimulator catches the exceptions and returns None
  - MIN_ANTI_FLOAT_RATIO = 0.30 is always enforced — do NOT remove this check

STEP 3: Make the change
  - If adding a new check: raise a new subclass of PlacementError
  - If changing thresholds: update constants at top of validator.py
  - NEVER change the function signature (heightmap, bin_config, x, y, z, ol, ow, oh, ...)

STEP 4: Verify
  - Run benchmark_all.py — confirm placements still succeed
  - Confirm no strategy now breaks due to stricter constraints
```

---

### Task C: Add a New Close Policy

```
STEP 1: Read
  - This README (done)
  - simulator/close_policy.py (full file — ~150 lines)
  - simulator/session.py lines where close_policy is called

STEP 2: Implement
  class MyClosePolicy(ClosePolicy):
      name = "my_close"

      def should_close(self, bin_state: BinState, pallet_stats: dict) -> bool:
          # Read bin_state and pallet_stats, return True/False
          # MUST be stateless/idempotent (called repeatedly)
          return bin_state.get_fill_rate() > 0.9

STEP 3: Export
  Add to simulator/__init__.py exports

STEP 4: Use it
  SessionConfig(close_policy=MyClosePolicy())
```

---

### Task D: Change the Botko BV Experiment

```
STEP 1: Read
  - This README (done)
  - run_overnight_botko.py (full file)
  - simulator/session.py (SessionConfig, PackingSession)
  - simulator/close_policy.py (HeightClosePolicy)
  - result_manager.py (output format)

STEP 2: Identify what to change
  Botko defaults:
    bin_config = BinConfig(1200, 800, 2700, 10)    ← DO NOT CHANGE (EU standard)
    num_bins = 2                                     ← 2 pallets in parallel
    buffer_size = 8                                  ← conveyor visible window
    pick_window = 4                                  ← robot reach
    close_policy = HeightClosePolicy(1800)           ← EU road limit
    max_consecutive_rejects = 10                     ← safety valve

STEP 3: Make changes ONLY to run_overnight_botko.py configuration
  Do not change SessionConfig defaults in session.py (other scripts use those)
```

---

### Task E: Add a New BoxSelector or BinSelector

```
STEP 1: Read
  - This README (done)
  - simulator/session.py — BoxSelector and BinSelector classes

STEP 2: Implement
  class MyBoxSelector(BoxSelector):
      def sort(self, grippable: List[Box]) -> List[Box]:
          return sorted(grippable, key=lambda b: b.weight, reverse=True)

  # Register in session.py
  BOX_SELECTORS["my_selector"] = MyBoxSelector

STEP 3: Use it
  session.run(boxes, strategy, box_selector=MyBoxSelector())
  # OR by name:
  session.run(boxes, strategy, box_selector=get_box_selector("my_selector"))
```

---

### Task F: Modify BinState Spatial Queries

```
STEP 1: Read
  - This README (done)
  - simulator/bin_state.py (FULL file)
  - simulator/pipeline_simulator.py (how it calls bin_state)
  - All strategy files that call get_height_at or get_support_ratio

STEP 2: Extreme caution
  - get_height_at() is called by EVERY strategy and the simulator
  - Changing its behavior changes ALL strategy behavior simultaneously
  - If you change the grid conversion (_to_grid), verify ALL coordinate math

STEP 3: Only change if necessary
  - If you must change, add unit tests FIRST
  - Run ALL strategies in benchmark_all.py and compare fill rates before/after
```

---

## 9. Critical Anti-Patterns

These will silently corrupt results or cause hard-to-debug failures.

### Never Do This

```python
# WRONG: Strategy mutating bin_state
def decide_placement(self, box, bin_state):
    bin_state.placed_boxes.append(something)  # NEVER — corrupts live state
    bin_state.heightmap[0, 0] = 999           # NEVER — breaks heightmap

# WRONG: Strategy returning Placement (not PlacementDecision)
def decide_placement(self, box, bin_state):
    return Placement(box_id=box.id, x=0, y=0, z=150, ...)  # NEVER
    # Correct: return PlacementDecision(x=0, y=0, orientation_idx=0)

# WRONG: Strategy setting z
def decide_placement(self, box, bin_state):
    z = bin_state.get_height_at(0, 0, box.length, box.width)
    return PlacementDecision(x=0, y=0, z=z, orientation_idx=0)  # NEVER — z not in PlacementDecision

# WRONG: Using unsafe copy for lookahead
def decide_placement(self, box, bin_state):
    copy = bin_state  # NEVER — this is same reference, not a copy
    copy = bin_state.heightmap.copy()  # NEVER — only copies heightmap, not placed_boxes
    # Correct:
    copy = bin_state.copy()            # deep copy via bin_state.copy()

# WRONG: Ignoring orientation_idx bounds
def decide_placement(self, box, bin_state):
    orientations = Orientation.get_flat(box.length, box.width, box.height)
    return PlacementDecision(x=0, y=0, orientation_idx=99)  # NEVER — index out of range

# WRONG: Proposing out-of-bounds coordinates
def decide_placement(self, box, bin_state):
    return PlacementDecision(x=-10, y=0, orientation_idx=0)  # NEVER — negative coords
    return PlacementDecision(x=1200, y=0, orientation_idx=0) # NEVER — at edge, box would overflow

# WRONG: Calling apply_placement directly
def decide_placement(self, box, bin_state):
    bin_state.apply_placement(some_placement)  # NEVER — only simulator calls this
```

### Never Change These Without Reading Everything That Depends On Them

```
MIN_ANTI_FLOAT_RATIO in validator.py        — all strategies depend on ≥30% support
BinConfig.resolution default (10mm)         — all grid conversions depend on this
PlacementDecision fields (x, y, orientation_idx) — every strategy returns this
BinState.get_height_at() signature          — called by every strategy
PackingSession.avg_closed_fill property     — primary metric for all experiments
FIFOConveyor no-recirculation logic         — changing breaks physical model fidelity
```

---

## 10. Import Graph

```
config.py
  Box, Placement, PlacementDecision, MultiBinDecision, BinConfig,
  ExperimentConfig, Orientation
  └── imported by: EVERYTHING

simulator/bin_state.py
  imports: config.BinConfig, config.Placement, numpy
  exports: BinState
  imported by: pipeline_simulator, session, validator, all strategies

simulator/validator.py
  imports: config.BinConfig, numpy
  exports: validate_placement, PlacementError, OutOfBoundsError, ...
  imported by: pipeline_simulator

simulator/pipeline_simulator.py
  imports: config.*, bin_state.BinState, validator.validate_placement
  exports: PipelineSimulator, StepRecord
  imported by: session, multi_bin_pipeline, run_experiment

simulator/conveyor.py
  imports: config.Box
  exports: FIFOConveyor
  imported by: session

simulator/close_policy.py
  imports: bin_state.BinState
  exports: ClosePolicy, HeightClosePolicy, FillClosePolicy, ...
  imported by: session

simulator/buffer.py
  imports: config.Box
  exports: BoxBuffer, BufferPolicy
  imported by: multi_bin_pipeline (legacy)

simulator/multi_bin_pipeline.py
  imports: config.*, bin_state, pipeline_simulator, buffer
  exports: MultiBinPipeline, PipelineConfig
  imported by: benchmark_all

simulator/session.py
  imports: config.*, bin_state, pipeline_simulator, conveyor, close_policy
  imports: strategies.base_strategy.BaseStrategy, MultiBinStrategy
  exports: PackingSession, SessionConfig, SessionResult, PalletResult, ...
  imported by: run_overnight_botko, benchmark_all

strategies/base_strategy.py
  imports: config.*, bin_state.BinState
  exports: BaseStrategy, MultiBinStrategy, STRATEGY_REGISTRY,
           MULTIBIN_STRATEGY_REGISTRY, register_strategy, register_multibin_strategy
  imported by: ALL strategy files, session

strategies/*/strategy.py
  imports: config, bin_state, base_strategy
  NO file may import another strategy file (strategies are independent)
```

---

## 11. Architecture Diagram

```
INPUT: List[Box]
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│  PackingSession                                              │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  FIFOConveyor                                       │    │
│  │  buffer_size=8  |  pick_window=4                    │    │
│  │  [box1][box2][box3][box4] | [box5][box6][box7][box8]│    │
│  │   ^^^^^^^^^^^^^^^^^^^^         visible but not      │    │
│  │   grippable (robot reach)      reachable            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────┐  ┌─────────────────────────────┐   │
│  │  _PalletStation 0   │  │  _PalletStation 1            │  │
│  │  PipelineSimulator  │  │  PipelineSimulator           │  │
│  │  BinState           │  │  BinState                    │  │
│  │  [heightmap]        │  │  [heightmap]                 │  │
│  │  [placed_boxes]     │  │  [placed_boxes]              │  │
│  └─────────────────────┘  └─────────────────────────────┘   │
│                                                              │
│  ClosePolicy → HeightClosePolicy(1800)                       │
│  BoxSelector → FIFOBoxSelector (default)                     │
│  BinSelector → EmptiestFirst (default)                       │
└──────────────────────────────────────────────────────────────┘
                   │
   ┌───────────────┼───────────────────────┐
   │               │                       │
   ▼               ▼                       ▼
BaseStrategy   MultiBinStrategy        RL Strategies
decide_placement   decide_placement    (same interface,
(box, bin_state)   (box, bin_states)    trained weights)
   │               │
   ▼               ▼
PlacementDecision  MultiBinDecision
(x, y, orient)     (bin_idx, x, y, orient)
   │
   ▼
PipelineSimulator.attempt_placement()
   │
   ├─ Orientation.get_flat/all → (ol, ow, oh)
   ├─ z = BinState.get_height_at()
   ├─ validator.validate_placement()
   │   ├─ bounds
   │   ├─ overlap
   │   ├─ anti-float ≥30%
   │   └─ stability (if enabled)
   └─ BinState.apply_placement()
       ├─ heightmap updated
       └─ placed_boxes appended
   │
   ▼
ClosePolicy.should_close(bin_state, pallet_stats)
   │
   ├─ True  → snapshot PalletResult → closed_pallets
   └─ False → continue packing
   │
   ▼
OUTPUT: SessionResult
   ├─ avg_closed_fill (PRIMARY METRIC, closed pallets only)
   ├─ closed_pallets: List[PalletResult]
   └─ active_pallets: List[PalletResult]
```

---

## 12. Change Checklist

Before committing any change, verify:

### For Strategy Changes
- [ ] Strategy has a unique `name` string (check `STRATEGY_REGISTRY.keys()`)
- [ ] `decide_placement()` returns `Optional[PlacementDecision]` (not `Placement`, not tuple)
- [ ] No mutations to `box` or `bin_state` inside the strategy
- [ ] `orientation_idx` is within bounds of the orientations list used
- [ ] All coordinates are non-negative and within pallet bounds (minus box dimensions)
- [ ] Strategy is importable: `from strategies.my_strategy.strategy import MyStrategy`
- [ ] `@register_strategy` decorator is present
- [ ] Runs without error in `python strategy_tester.py --strategy my_strategy`

### For Simulator Changes
- [ ] Read ALL files that import the file you changed (see Section 10)
- [ ] Function signatures unchanged (or all callers updated)
- [ ] `MIN_ANTI_FLOAT_RATIO` still enforced in validator
- [ ] `placed_boxes` still append-only (no remove, no reorder)
- [ ] `heightmap` still synchronized with `placed_boxes`
- [ ] `BinState.copy()` still returns a true deep copy

### For Session/Conveyor Changes
- [ ] FIFO order not changed (no recirculation)
- [ ] `avg_closed_fill` still counts closed pallets only
- [ ] Last active pallet cannot be closed
- [ ] `consecutive_rejects` resets on success or pallet close
- [ ] `total_placed + total_rejected + remaining_boxes == total_boxes`

### For Data Model Changes (config.py)
- [ ] All frozen dataclasses remain frozen (immutable)
- [ ] `PlacementDecision` still has only `(x, y, orientation_idx)` — no `z`
- [ ] `Placement` still has `z` (set by simulator, not strategy)
- [ ] `Box.id` is still unique within a session
- [ ] All `.to_dict()` / `.from_dict()` methods updated if fields changed

### Final Smoke Test (always run this)
```bash
cd python
python benchmark_all.py
```
Expected: All strategies complete, fill rates roughly match known baselines:
- `walle_scoring` ~68.3%, `surface_contact` ~67.4%, `baseline` ~64.8%

---

*This README was generated 2026-02-25. If you change any class interface, update the relevant section here before merging.*
