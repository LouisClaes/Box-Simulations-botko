"""
simulator — core pipeline engine for box placement, validation, and session management.

This package provides the full packing lifecycle:

  **Core simulation** (single-pallet physics):
    PipelineSimulator   — validates and stacks boxes on one pallet
    BinState            — 3D state of one pallet (heightmap, placed_boxes)
    StepRecord          — immutable log entry per placement attempt

  **Conveyor belt** (FIFO box stream):
    FIFOConveyor        — models a physical conveyor with pick window

  **Close policies** (when to seal a pallet):
    ClosePolicy         — ABC for custom policies
    HeightClosePolicy   — close at max_height >= threshold (default 1800mm)
    FillClosePolicy     — close at fill_rate >= threshold
    RejectClosePolicy   — close after N consecutive idle steps
    CombinedClosePolicy — close when ANY child policy triggers
    NeverClosePolicy    — never close (single-bin mode)

  **Packing session** (multi-pallet orchestrator):
    PackingSession      — boxes stream in → strategy places → pallets close
    SessionConfig       — all session parameters in one object
    SessionResult       — complete metrics (closed + active pallets)
    PalletResult        — per-pallet stats snapshot
    StepObservation     — what the agent sees before acting
    StepResult          — outcome of one placement step

  **Selectors** (box/bin ordering for single-bin strategies):
    BoxSelector, BinSelector  — ABCs
    FIFOBoxSelector, LargestVolumeFirst, ...  — built-in box selectors
    EmptiestFirst, FocusFill, FlatFirst       — built-in bin selectors

  **Legacy** (backward compatibility):
    BoxBuffer, BufferPolicy   — generic buffer (used by MultiBinPipeline)

Public API:
    from simulator import PackingSession, SessionConfig, SessionResult
    from simulator import HeightClosePolicy, CombinedClosePolicy
    from simulator import FIFOConveyor
    from simulator import PipelineSimulator, BinState
"""

from simulator.pipeline_simulator import PipelineSimulator, StepRecord
from simulator.bin_state import BinState
from simulator.validator import (
    validate_placement,
    PlacementError,
    OutOfBoundsError,
    OverlapError,
    FloatingError,
    UnstablePlacementError,
)
from simulator.buffer import BoxBuffer, BufferPolicy
from simulator.conveyor import FIFOConveyor
from simulator.close_policy import (
    ClosePolicy,
    HeightClosePolicy,
    FillClosePolicy,
    RejectClosePolicy,
    CombinedClosePolicy,
    NeverClosePolicy,
)
from simulator.session import (
    PackingSession,
    SessionConfig,
    SessionResult,
    PalletResult,
    StepObservation,
    StepResult,
    BoxSelector,
    BinSelector,
    FIFOBoxSelector,
    LargestVolumeFirst,
    LargestFootprintFirst,
    HeaviestFirst,
    EmptiestFirst,
    FocusFill,
    FlatFirst,
    get_box_selector,
    get_bin_selector,
    BOX_SELECTORS,
    BIN_SELECTORS,
)

__all__ = [
    # Core simulation
    "PipelineSimulator", "StepRecord", "BinState",
    # Validation
    "validate_placement", "PlacementError",
    "OutOfBoundsError", "OverlapError", "FloatingError", "UnstablePlacementError",
    # Conveyor
    "FIFOConveyor",
    # Close policies
    "ClosePolicy", "HeightClosePolicy", "FillClosePolicy",
    "RejectClosePolicy", "CombinedClosePolicy", "NeverClosePolicy",
    # Session
    "PackingSession", "SessionConfig", "SessionResult",
    "PalletResult", "StepObservation", "StepResult",
    # Selectors
    "BoxSelector", "BinSelector",
    "FIFOBoxSelector", "LargestVolumeFirst", "LargestFootprintFirst", "HeaviestFirst",
    "EmptiestFirst", "FocusFill", "FlatFirst",
    "get_box_selector", "get_bin_selector",
    "BOX_SELECTORS", "BIN_SELECTORS",
    # Legacy buffer
    "BoxBuffer", "BufferPolicy",
]
