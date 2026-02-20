"""
PackingSession — unified orchestrator for multi-pallet box packing.

This module bakes the full packing lifecycle into the simulator package:
  boxes stream in → strategy places → pallets close → stats saved → repeat

The PackingSession handles:
  - FIFO conveyor belt with configurable buffer and pick window
  - N pallet stations with independent PipelineSimulator instances
  - Pluggable close policies (height, fill, rejects, combined)
  - Box selectors (which grippable box to try first)
  - Bin selectors (which pallet to prefer for single-bin strategies)
  - Pallet lifecycle: close → snapshot → replace with fresh pallet
  - Only closed pallets count in primary metrics
  - Safety valve: max consecutive rejects to prevent infinite loops

Supports BOTH strategy interfaces:
  - BaseStrategy  (single-bin): session wraps with external bin selection
  - MultiBinStrategy (multi-bin): strategy sees all bin states natively

Two usage modes:
  1. Batch mode: ``session.run(boxes, strategy)`` — runs everything, returns results
  2. Step mode:  ``session.reset(boxes)`` → ``observe()`` → ``step(...)`` loop

Example (batch mode):
    >>> from simulator.session import PackingSession, SessionConfig
    >>> from simulator.close_policy import HeightClosePolicy
    >>> config = SessionConfig(
    ...     bin_config=BinConfig(length=1200, width=800, height=2700, resolution=10),
    ...     num_bins=2,
    ...     close_policy=HeightClosePolicy(max_height=1800),
    ... )
    >>> session = PackingSession(config)
    >>> result = session.run(boxes, strategy)
    >>> print(result.avg_closed_fill)

Example (step mode):
    >>> session = PackingSession(config)
    >>> obs = session.reset(boxes)
    >>> while not obs.done:
    ...     # Your custom logic here
    ...     obs = session.observe()
    ...     # Decide placement...
    ...     step_result = session.step(box_id, bin_index, x, y, orient_idx)
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod

import numpy as np

from config import Box, BinConfig, ExperimentConfig, Placement
from simulator.bin_state import BinState
from simulator.pipeline_simulator import PipelineSimulator
from simulator.conveyor import FIFOConveyor
from simulator.close_policy import ClosePolicy, HeightClosePolicy, NeverClosePolicy


# ─────────────────────────────────────────────────────────────────────────────
# Box selectors — which grippable box to try first
# ─────────────────────────────────────────────────────────────────────────────

class BoxSelector(ABC):
    """
    Abstract base for box selection from the grippable window.

    Determines the order in which the session tries grippable boxes.
    The first box that fits on any pallet gets placed.
    """

    name: str = "base"

    @abstractmethod
    def sort(self, boxes: List[Box]) -> List[Box]:
        """
        Return the grippable boxes in preferred order.

        Args:
            boxes: Grippable boxes from the conveyor (front of belt).

        Returns:
            Same boxes, reordered by preference (first = try first).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class FIFOBoxSelector(BoxSelector):
    """Keep conveyor order — first-in, first-tried."""
    name = "default"

    def sort(self, boxes: List[Box]) -> List[Box]:
        return list(boxes)


class LargestVolumeFirst(BoxSelector):
    """Try the largest-volume box first."""
    name = "biggest_volume_first"

    def sort(self, boxes: List[Box]) -> List[Box]:
        return sorted(boxes, key=lambda b: b.volume, reverse=True)


class LargestFootprintFirst(BoxSelector):
    """Try the box with the largest footprint (L x W) first."""
    name = "biggest_footprint_first"

    def sort(self, boxes: List[Box]) -> List[Box]:
        return sorted(boxes, key=lambda b: b.length * b.width, reverse=True)


class HeaviestFirst(BoxSelector):
    """Try the heaviest box first."""
    name = "heaviest_first"

    def sort(self, boxes: List[Box]) -> List[Box]:
        return sorted(boxes, key=lambda b: b.weight, reverse=True)


BOX_SELECTORS: Dict[str, type] = {
    "default": FIFOBoxSelector,
    "biggest_volume_first": LargestVolumeFirst,
    "biggest_footprint_first": LargestFootprintFirst,
    "heaviest_first": HeaviestFirst,
}


def get_box_selector(name: str) -> BoxSelector:
    """Look up a BoxSelector by name and return a new instance."""
    if name not in BOX_SELECTORS:
        available = ", ".join(sorted(BOX_SELECTORS.keys()))
        raise ValueError(f"Unknown box selector '{name}'. Available: [{available}]")
    return BOX_SELECTORS[name]()


# ─────────────────────────────────────────────────────────────────────────────
# Bin selectors — which pallet to prefer (single-bin strategies only)
# ─────────────────────────────────────────────────────────────────────────────

class BinSelector(ABC):
    """
    Abstract base for pallet (bin) preference scoring.

    When a single-bin strategy finds valid placements on multiple pallets,
    the BinSelector decides which pallet to use.

    Not used for MultiBinStrategy (which chooses its own pallet).
    """

    name: str = "base"

    @abstractmethod
    def score(self, bin_index: int, bin_state: BinState, box: Box) -> float:
        """
        Score a pallet — higher = more preferred.

        Args:
            bin_index: Which pallet station (0-indexed).
            bin_state: Current state of that pallet.
            box:       The box being considered.

        Returns:
            A score.  The pallet with the highest score is chosen.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class EmptiestFirst(BinSelector):
    """Prefer the emptier pallet — spread load evenly."""
    name = "emptiest_first"

    def score(self, bin_index: int, bin_state: BinState, box: Box) -> float:
        fill = bin_state.get_fill_rate()
        max_h = bin_state.get_max_height()
        return (1.0 - fill) * 1000 + 1.0 / (1.0 + max_h)


class FocusFill(BinSelector):
    """Prefer the fuller pallet — fill one up fast before starting the next."""
    name = "focus_fill"

    def score(self, bin_index: int, bin_state: BinState, box: Box) -> float:
        fill = bin_state.get_fill_rate()
        max_h = bin_state.get_max_height()
        return fill * 1000 + 1.0 / (1.0 + max_h)


class FlatFirst(BinSelector):
    """Prefer the pallet with the lowest max height — keep surfaces flat."""
    name = "flattest_first"

    def score(self, bin_index: int, bin_state: BinState, box: Box) -> float:
        max_h = bin_state.get_max_height()
        fill = bin_state.get_fill_rate()
        return -max_h - fill * 100


BIN_SELECTORS: Dict[str, type] = {
    "emptiest_first": EmptiestFirst,
    "focus_fill": FocusFill,
    "flattest_first": FlatFirst,
}


def get_bin_selector(name: str) -> BinSelector:
    """Look up a BinSelector by name and return a new instance."""
    if name not in BIN_SELECTORS:
        available = ", ".join(sorted(BIN_SELECTORS.keys()))
        raise ValueError(f"Unknown bin selector '{name}'. Available: [{available}]")
    return BIN_SELECTORS[name]()


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionConfig:
    """
    All parameters for a PackingSession.

    This fully describes the physical setup and decision policies.
    Pass to ``PackingSession(config)`` to create a session.

    Attributes:
        bin_config:              Pallet dimensions and grid resolution.
        num_bins:                Number of pallet stations.
        buffer_size:             Conveyor belt capacity (visible boxes).
        pick_window:             Front N boxes the robot can reach.
        close_policy:            When to close a pallet (pluggable).
        max_consecutive_rejects: Safety valve — stop after this many
                                 consecutive rejects across ALL pallets.
        enable_stability:        Enable anti-float validation.
        min_support_ratio:       Minimum base support for stability.
        allow_all_orientations:  Allow all 6 box orientations (vs 2 flat).
    """

    bin_config: BinConfig = field(default_factory=lambda: BinConfig())
    num_bins: int = 2
    buffer_size: int = 8
    pick_window: int = 4
    close_policy: ClosePolicy = field(default_factory=lambda: HeightClosePolicy(1800.0))
    max_consecutive_rejects: int = 10
    enable_stability: bool = False
    min_support_ratio: float = 0.8
    allow_all_orientations: bool = False

    def to_experiment_config(self, strategy_name: str = "session") -> ExperimentConfig:
        """Convert to ExperimentConfig for PipelineSimulator creation."""
        return ExperimentConfig(
            bin=self.bin_config,
            strategy_name=strategy_name,
            enable_stability=self.enable_stability,
            min_support_ratio=self.min_support_ratio,
            allow_all_orientations=self.allow_all_orientations,
            render_3d=False,
            verbose=False,
        )

    def to_dict(self) -> dict:
        return {
            "bin_config": self.bin_config.to_dict(),
            "num_bins": self.num_bins,
            "buffer_size": self.buffer_size,
            "pick_window": self.pick_window,
            "close_policy": self.close_policy.describe(),
            "max_consecutive_rejects": self.max_consecutive_rejects,
            "enable_stability": self.enable_stability,
            "min_support_ratio": self.min_support_ratio,
            "allow_all_orientations": self.allow_all_orientations,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PalletResult:
    """
    Stats for one closed (or active) pallet.

    Frozen snapshot taken at the moment the pallet was closed.
    """

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
    placements: List[Placement] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bin_slot": self.bin_slot,
            "fill_rate": round(self.fill_rate, 6),
            "effective_fill": round(self.effective_fill, 6),
            "max_height": round(self.max_height, 2),
            "boxes_placed": self.boxes_placed,
            "placed_volume": round(self.placed_volume, 2),
            "surface_roughness": round(self.surface_roughness, 4),
            "support_mean": round(self.support_mean, 4),
            "support_min": round(self.support_min, 4),
            "ms_per_box_mean": round(self.ms_per_box_mean, 3),
        }


@dataclass
class SessionResult:
    """
    Complete result of a PackingSession run.

    Primary metric: ``avg_closed_fill`` — mean fill rate of closed pallets.
    Active pallets (still being filled at termination) are NOT included in
    the primary metric because in production they would wait for more boxes.
    """

    total_boxes: int = 0
    total_placed: int = 0
    total_rejected: int = 0
    remaining_boxes: int = 0
    closed_pallets: List[PalletResult] = field(default_factory=list)
    active_pallets: List[PalletResult] = field(default_factory=list)
    elapsed_ms: float = 0.0
    consecutive_rejects_triggered: bool = False

    @property
    def pallets_closed(self) -> int:
        return len(self.closed_pallets)

    @property
    def avg_closed_fill(self) -> float:
        """Mean volumetric fill across closed pallets (primary metric)."""
        if not self.closed_pallets:
            return 0.0
        return float(np.mean([p.fill_rate for p in self.closed_pallets]))

    @property
    def avg_closed_effective_fill(self) -> float:
        """Mean effective fill (placed_vol / L*W*max_height) across closed pallets."""
        if not self.closed_pallets:
            return 0.0
        return float(np.mean([p.effective_fill for p in self.closed_pallets]))

    @property
    def avg_closed_height(self) -> float:
        if not self.closed_pallets:
            return 0.0
        return float(np.mean([p.max_height for p in self.closed_pallets]))

    @property
    def ms_per_box(self) -> float:
        if self.total_placed == 0:
            return 0.0
        return self.elapsed_ms / self.total_placed

    @property
    def placement_rate(self) -> float:
        if self.total_boxes == 0:
            return 0.0
        return self.total_placed / self.total_boxes

    def to_dict(self) -> dict:
        return {
            "total_boxes": self.total_boxes,
            "total_placed": self.total_placed,
            "total_rejected": self.total_rejected,
            "remaining_boxes": self.remaining_boxes,
            "placement_rate": round(self.placement_rate, 4),
            "pallets_closed": self.pallets_closed,
            "avg_closed_fill": round(self.avg_closed_fill, 6),
            "avg_closed_effective_fill": round(self.avg_closed_effective_fill, 6),
            "avg_closed_height": round(self.avg_closed_height, 2),
            "closed_pallets": [p.to_dict() for p in self.closed_pallets],
            "active_pallets": [p.to_dict() for p in self.active_pallets],
            "elapsed_ms": round(self.elapsed_ms, 1),
            "ms_per_box": round(self.ms_per_box, 2),
            "consecutive_rejects_triggered": self.consecutive_rejects_triggered,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Step-mode observation / result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepObservation:
    """
    What the external agent sees before deciding an action.

    Returned by ``session.observe()`` and ``session.reset()``.
    """

    grippable: List[Box]
    """Boxes the robot can reach (front of conveyor)."""

    buffer_view: List[Box]
    """All visible boxes on the belt (for visualization)."""

    bin_states: List[BinState]
    """Current state of each pallet station."""

    stream_remaining: int
    """Boxes still in the stream (not yet on belt)."""

    done: bool
    """True if the session is finished (belt empty or safety valve)."""

    step_num: int = 0
    """Current step number."""


@dataclass
class StepResult:
    """
    Result of a single ``session.step()`` call.
    """

    placed: bool
    """Whether the box was successfully placed."""

    placement: Optional[Placement]
    """The validated Placement if placed, else None."""

    box: Optional[Box]
    """The box that was attempted (or rejected)."""

    bin_index: int = -1
    """Which pallet the box was placed on (-1 if not placed)."""

    pallet_closed: bool = False
    """Whether the placement triggered a pallet close."""

    closed_pallet_result: Optional[PalletResult] = None
    """Stats of the closed pallet (if pallet_closed is True)."""


# ─────────────────────────────────────────────────────────────────────────────
# Internal: pallet station
# ─────────────────────────────────────────────────────────────────────────────

class _PalletStation:
    """
    Internal class managing one pallet slot.

    Tracks the simulator, per-pallet counters, and snapshot logic.
    Not part of the public API — used only by PackingSession.
    """

    __slots__ = ("slot_index", "sim", "consecutive_idle", "boxes_placed", "total_rejects")

    def __init__(self, slot_index: int, sim: PipelineSimulator) -> None:
        self.slot_index = slot_index
        self.sim = sim
        self.consecutive_idle: int = 0
        self.boxes_placed: int = 0
        self.total_rejects: int = 0

    @property
    def bin_state(self) -> BinState:
        return self.sim.get_bin_state()

    @property
    def pallet_stats(self) -> dict:
        """Per-pallet counters passed to ClosePolicy.should_close()."""
        return {
            "consecutive_idle": self.consecutive_idle,
            "boxes_placed": self.boxes_placed,
            "total_rejects": self.total_rejects,
        }

    def snapshot(self, bin_config: BinConfig) -> PalletResult:
        """Take a frozen snapshot of this pallet's metrics."""
        state = self.sim.get_bin_state()
        log = self.sim.get_step_log()
        placed_records = [r for r in log if r.success]
        placed_vol = sum(p.volume for p in state.placed_boxes)
        max_h = state.get_max_height()
        eff_vol = bin_config.length * bin_config.width * max_h if max_h > 0 else 1.0
        support_ratios = [r.support_ratio for r in placed_records]
        timings = [r.elapsed_ms for r in placed_records]

        return PalletResult(
            bin_slot=self.slot_index,
            fill_rate=state.get_fill_rate(),
            effective_fill=placed_vol / eff_vol if eff_vol > 0 else 0.0,
            max_height=max_h,
            boxes_placed=len(placed_records),
            placed_volume=placed_vol,
            surface_roughness=state.get_surface_roughness(),
            support_mean=float(np.mean(support_ratios)) if support_ratios else 0.0,
            support_min=float(np.min(support_ratios)) if support_ratios else 0.0,
            ms_per_box_mean=float(np.mean(timings)) if timings else 0.0,
            placements=list(state.placed_boxes),
        )

    def reset(self, sim: PipelineSimulator) -> None:
        """Replace with a fresh pallet."""
        self.sim = sim
        self.consecutive_idle = 0
        self.boxes_placed = 0
        self.total_rejects = 0


# ─────────────────────────────────────────────────────────────────────────────
# PackingSession
# ─────────────────────────────────────────────────────────────────────────────

class PackingSession:
    """
    Unified orchestrator for multi-pallet box packing.

    Manages the full lifecycle:
      1. Boxes stream in via a FIFO conveyor belt.
      2. Strategy decides placement (single-bin or multi-bin).
      3. Pallet stations accept or reject placements.
      4. Close policy checks trigger pallet snapshots and replacement.
      5. Only closed pallets are counted in primary metrics.

    Works with both strategy interfaces:
      - ``BaseStrategy``:     Session wraps with BoxSelector + BinSelector
      - ``MultiBinStrategy``: Strategy sees all bin states directly

    Two usage modes:

    **Batch mode** (recommended for experiments):
        ``result = session.run(boxes, strategy, box_selector, bin_selector)``

    **Step mode** (for RL / custom control):
        ``obs = session.reset(boxes)``
        ``while not obs.done: result = session.step(box_id, bin_idx, x, y, orient)``
    """

    def __init__(self, config: SessionConfig) -> None:
        self._config = config
        self._conveyor: Optional[FIFOConveyor] = None
        self._stations: List[_PalletStation] = []
        self._closed: List[PalletResult] = []
        self._step_num: int = 0
        self._placed_count: int = 0
        self._rejected_count: int = 0
        self._consecutive_rejects: int = 0
        self._done: bool = True
        self._t0: float = 0.0
        self._exp_config: Optional[ExperimentConfig] = None
        self._on_step: Optional[Callable] = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def config(self) -> SessionConfig:
        return self._config

    @property
    def conveyor(self) -> Optional[FIFOConveyor]:
        """The active conveyor belt (None before reset)."""
        return self._conveyor

    @property
    def stations(self) -> List[_PalletStation]:
        """Active pallet stations (for advanced introspection)."""
        return self._stations

    @property
    def closed_pallets(self) -> List[PalletResult]:
        """All closed pallets so far."""
        return list(self._closed)

    @property
    def done(self) -> bool:
        return self._done

    # ── Step mode API ─────────────────────────────────────────────────────

    def reset(self, boxes: List[Box], strategy_name: str = "session") -> StepObservation:
        """
        Initialize a new session with a fresh box stream.

        Creates the conveyor, pallet stations, and returns the first
        observation for step-mode control.

        Args:
            boxes:         Full list of boxes to process.
            strategy_name: Label for ExperimentConfig (metadata only).

        Returns:
            Initial StepObservation.
        """
        self._exp_config = self._config.to_experiment_config(strategy_name)
        self._conveyor = FIFOConveyor(
            boxes,
            buffer_size=self._config.buffer_size,
            pick_window=self._config.pick_window,
        )
        self._stations = [
            _PalletStation(i, PipelineSimulator(self._exp_config))
            for i in range(self._config.num_bins)
        ]
        self._closed = []
        self._step_num = 0
        self._placed_count = 0
        self._rejected_count = 0
        self._consecutive_rejects = 0
        self._done = False
        self._t0 = time.perf_counter()

        return self.observe()

    def observe(self) -> StepObservation:
        """
        Current observation (grippable boxes, bin states, done flag).

        Call this before deciding your next action in step mode.
        """
        conv = self._conveyor
        return StepObservation(
            grippable=list(conv.grippable) if conv else [],
            buffer_view=list(conv.visible) if conv else [],
            bin_states=[st.bin_state for st in self._stations],
            stream_remaining=conv.stream_remaining if conv else 0,
            done=self._done,
            step_num=self._step_num,
        )

    def step(
        self,
        box_id: int,
        bin_index: int,
        x: float,
        y: float,
        orientation_idx: int,
    ) -> StepResult:
        """
        Execute one placement step.

        The caller specifies which box to pick, which pallet to use, and
        where to place it.  The session validates via PipelineSimulator,
        handles conveyor movement, and checks the close policy.

        If placement fails (simulator rejects), the conveyor advances
        (front box exits).

        Args:
            box_id:          ID of the box to pick from the grippable window.
            bin_index:       Which pallet station to place on.
            x, y:            Position within the pallet.
            orientation_idx: Box orientation index.

        Returns:
            StepResult with placement outcome and pallet close info.
        """
        if self._done:
            return StepResult(placed=False, placement=None, box=None)

        conv = self._conveyor
        station = self._stations[bin_index]

        # Pick the box from the conveyor
        picked = conv.pick(box_id)
        if picked is None:
            # Box not in grippable window — treat as reject
            self._handle_reject()
            return StepResult(placed=False, placement=None, box=None)

        # Try placement
        result = station.sim.attempt_placement(picked, x, y, orientation_idx)

        if result is not None:
            # Success
            station.boxes_placed += 1
            station.consecutive_idle = 0
            self._placed_count += 1
            self._consecutive_rejects = 0

            # Mark other stations as idle for this step
            for st in self._stations:
                if st.slot_index != bin_index:
                    st.consecutive_idle += 1

            # Check close policy
            pallet_closed = False
            closed_result = None
            if self._config.close_policy.should_close(station.bin_state, station.pallet_stats):
                closed_result = station.snapshot(self._config.bin_config)
                self._closed.append(closed_result)
                station.reset(PipelineSimulator(self._exp_config))
                self._consecutive_rejects = 0
                pallet_closed = True

            self._step_num += 1
            self._check_done()

            return StepResult(
                placed=True,
                placement=result,
                box=picked,
                bin_index=bin_index,
                pallet_closed=pallet_closed,
                closed_pallet_result=closed_result,
            )
        else:
            # Placement failed — return box concept: conveyor advances
            # (The box was already removed from conveyor by pick(); it's lost)
            self._rejected_count += 1
            self._consecutive_rejects += 1
            for st in self._stations:
                st.consecutive_idle += 1

            self._step_num += 1
            self._check_done()

            return StepResult(
                placed=False,
                placement=None,
                box=picked,
                bin_index=bin_index,
            )

    def advance_conveyor(self) -> Optional[Box]:
        """
        Manually advance the conveyor (front box exits).

        Call this in step mode when no grippable box can be placed.
        Returns the rejected box.
        """
        if self._done or self._conveyor is None:
            return None

        rejected = self._conveyor.advance()
        if rejected is not None:
            self._rejected_count += 1
            self._consecutive_rejects += 1
            for st in self._stations:
                st.consecutive_idle += 1

        self._step_num += 1
        self._check_done()
        return rejected

    def result(self) -> SessionResult:
        """
        Build the final SessionResult from the current session state.

        Can be called at any time — returns metrics for all closed pallets
        and any active (incomplete) pallets.
        """
        active = []
        for st in self._stations:
            if st.boxes_placed > 0:
                active.append(st.snapshot(self._config.bin_config))

        conv = self._conveyor
        remaining = conv.total_remaining if conv else 0

        return SessionResult(
            total_boxes=conv.total_loaded if conv else 0,
            total_placed=self._placed_count,
            total_rejected=self._rejected_count,
            remaining_boxes=remaining,
            closed_pallets=list(self._closed),
            active_pallets=active,
            elapsed_ms=(time.perf_counter() - self._t0) * 1000,
            consecutive_rejects_triggered=(
                self._consecutive_rejects >= self._config.max_consecutive_rejects
            ),
        )

    # ── Batch mode API ────────────────────────────────────────────────────

    def run(
        self,
        boxes: List[Box],
        strategy,
        box_selector: Optional[BoxSelector] = None,
        bin_selector: Optional[BinSelector] = None,
        on_step: Optional[Callable] = None,
    ) -> SessionResult:
        """
        Run a complete packing session in batch mode.

        Automatically handles the full loop: conveyor → strategy →
        placement → close → replace → repeat until done.

        Works with both strategy types:
          - ``BaseStrategy``:     Uses box_selector + bin_selector to wrap
          - ``MultiBinStrategy``: Calls strategy.decide_placement(box, all_states)

        Args:
            boxes:        Full list of boxes to process.
            strategy:     A BaseStrategy or MultiBinStrategy instance.
            box_selector: How to order grippable boxes (default: FIFO).
            bin_selector: How to score pallets (default: emptiest_first).
                          Ignored for MultiBinStrategy.
            on_step:      Optional callback ``fn(step_num, step_result, observation)``
                          called after every step (for GIF recording, logging, etc).

        Returns:
            SessionResult with all metrics.
        """
        from strategies.base_strategy import BaseStrategy, MultiBinStrategy

        if box_selector is None:
            box_selector = FIFOBoxSelector()
        if bin_selector is None:
            bin_selector = EmptiestFirst()

        strategy_name = getattr(strategy, "name", "unknown")
        is_multibin = isinstance(strategy, MultiBinStrategy)

        obs = self.reset(boxes, strategy_name=strategy_name)

        # Initialize strategy
        strategy.on_episode_start(self._exp_config)

        while not obs.done:
            grippable = obs.grippable
            if not grippable:
                break

            if is_multibin:
                step_result = self._run_multibin_step(strategy, grippable)
            else:
                step_result = self._run_singlebin_step(
                    strategy, grippable, box_selector, bin_selector,
                )

            obs = self.observe()

            if on_step is not None:
                on_step(self._step_num - 1, step_result, obs)

        strategy.on_episode_end({})
        return self.result()

    # ── Private helpers ───────────────────────────────────────────────────

    def _run_singlebin_step(
        self,
        strategy,
        grippable: List[Box],
        box_selector: BoxSelector,
        bin_selector: BinSelector,
    ) -> StepResult:
        """One step for a single-bin strategy with external bin selection."""
        sorted_boxes = box_selector.sort(grippable)

        best_score = -1e18
        best_box = None
        best_bin = None
        best_decision = None

        for box in sorted_boxes:
            for station in self._stations:
                state = station.bin_state
                decision = strategy.decide_placement(box, state)
                if decision is not None:
                    score = bin_selector.score(station.slot_index, state, box)
                    if score > best_score:
                        best_score = score
                        best_box = box
                        best_bin = station.slot_index
                        best_decision = decision

        if best_box is not None and best_decision is not None:
            return self.step(
                best_box.id, best_bin,
                best_decision.x, best_decision.y,
                best_decision.orientation_idx,
            )
        else:
            # No placement found — advance conveyor
            rejected = self.advance_conveyor()
            return StepResult(
                placed=False, placement=None,
                box=rejected, bin_index=-1,
            )

    def _run_multibin_step(
        self,
        strategy,
        grippable: List[Box],
    ) -> StepResult:
        """One step for a multi-bin strategy (native bin selection)."""
        bin_states = [st.bin_state for st in self._stations]

        best_box = None
        best_decision = None

        for box in grippable:
            decision = strategy.decide_placement(box, bin_states)
            if decision is not None:
                best_box = box
                best_decision = decision
                break  # Multi-bin strategy picks first viable box

        if best_box is not None and best_decision is not None:
            return self.step(
                best_box.id, best_decision.bin_index,
                best_decision.x, best_decision.y,
                best_decision.orientation_idx,
            )
        else:
            rejected = self.advance_conveyor()
            return StepResult(
                placed=False, placement=None,
                box=rejected, bin_index=-1,
            )

    def _handle_reject(self) -> None:
        """Handle a rejection (box not in window or sim rejected)."""
        self._rejected_count += 1
        self._consecutive_rejects += 1
        for st in self._stations:
            st.consecutive_idle += 1
        self._step_num += 1
        self._check_done()

    def _check_done(self) -> None:
        """Update the done flag based on termination conditions."""
        conv = self._conveyor
        if conv is None:
            self._done = True
            return

        if conv.is_empty:
            self._done = True
            return

        if self._consecutive_rejects >= self._config.max_consecutive_rejects:
            self._done = True
            return
