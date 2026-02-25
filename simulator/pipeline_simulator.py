"""
Pipeline simulator - the central authority for box placement.

Data flow:
  1. Strategy calls  simulator.get_bin_state()  -> receives BinState
     (heightmap + placed_boxes, full 3D info, copy() for lookahead).
  2. Strategy returns a PlacementDecision(x, y, orientation_idx).
  3. Runner calls    simulator.attempt_placement(box, x, y, orient)
     -> simulator computes z, validates, updates state, logs step.
  4. If strategy returns None -> runner calls
     simulator.record_rejection(box, reason) (public API).

Usage:
    sim = PipelineSimulator(config)
    state = sim.get_bin_state()          # strategy reads this
    result = sim.attempt_placement(...)  # simulator validates & stacks
    sim.record_rejection(box, reason)    # log when strategy cannot place
"""

import time
from typing import Optional, List
from dataclasses import dataclass

from config import Box, Placement, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from simulator.validator import validate_placement, PlacementError


# ---------------------------------------------------------------------------
# StepRecord -- immutable log entry for each placement attempt
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepRecord:
    """
    Log of a single placement attempt (success or rejection).

    Frozen so it can be safely examined without risk of mutation.
    """
    step: int
    box: Box
    success: bool
    placement: Optional[Placement] = None
    rejection_reason: str = ""
    fill_rate_after: float = 0.0
    support_ratio: float = 1.0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "step": self.step,
            "box_id": self.box.id,
            "box_dims": [self.box.length, self.box.width, self.box.height],
            "success": self.success,
            "fill_rate_after": round(self.fill_rate_after, 6),
            "support_ratio": round(self.support_ratio, 4),
            "elapsed_ms": round(self.elapsed_ms, 3),
        }
        if self.success and self.placement is not None:
            d["placement"] = self.placement.to_dict()
        else:
            d["rejection_reason"] = self.rejection_reason
        return d


# ---------------------------------------------------------------------------
# PipelineSimulator
# ---------------------------------------------------------------------------

class PipelineSimulator:
    """
    Physics engine for 3D box placement in a single bin.

    Receives box dimensions + coordinates from a strategy, physically
    stacks them, and enforces:

    * No box outside the stacking area
    * No overlapping boxes
    * Optional stability constraints (configurable support ratio)

    This is the core building block of the placement pipeline -- used
    directly for single-bin experiments and internally by the orchestrator
    and MultiBinPipeline for multi-bin runs.

    Public interface
    ~~~~~~~~~~~~~~~~
    get_bin_state()          -> BinState   (strategies read this)
    attempt_placement(...)   -> Placement | None
    record_rejection(...)    -> None       (log when strategy cannot place)
    get_step_log()           -> List[StepRecord]
    get_summary()            -> dict
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._bin_state = BinState(config.bin)
        self._step_log: List[StepRecord] = []
        self._step_counter: int = 0

    # -- Public: state access ------------------------------------------------

    @property
    def config(self) -> ExperimentConfig:
        """Read-only experiment configuration."""
        return self._config

    def get_bin_state(self) -> BinState:
        """
        Current bin state, exposing full 3D information:

        * .heightmap          -- 2D numpy grid of current heights
        * .placed_boxes       -- List[Placement] with x/y/z + dims
        * .get_height_at()    -- query resting z for a footprint
        * .get_support_ratio()-- base support query
        * .get_fill_rate()    -- volumetric utilisation
        * .copy()             -- deep copy for lookahead / what-if
        """
        return self._bin_state

    # -- Public: placement ---------------------------------------------------

    def attempt_placement(
        self, box: Box, x: float, y: float, orientation_idx: int,
    ) -> Optional[Placement]:
        """
        Try to place *box* at (x, y) with the given orientation.

        The simulator:
          1. Resolves oriented dimensions from the orientation index.
          2. Computes z from the heightmap (box drops to surface).
          3. Validates bounds, overlap, floating, and optional stability.
          4. On success: updates state, logs the step, returns Placement.
          5. On failure: logs rejection with reason, returns None.

        After this call, get_bin_state() returns the updated state
        that the strategy can use for the next decision.

        Returns:
            Placement on success, None on rejection.
        """
        t0 = time.perf_counter()
        step = self._step_counter

        # 1 -- Resolve orientation
        orientations = self._get_orientations(box)
        if orientation_idx < 0 or orientation_idx >= len(orientations):
            self._log_rejection(
                step, box, t0,
                f"Invalid orientation {orientation_idx} (max {len(orientations) - 1})",
            )
            return None

        ol, ow, oh = orientations[orientation_idx]

        # 2 -- Compute z (gravity drop)
        z = self._bin_state.get_height_at(x, y, ol, ow)

        # 3 -- Validate
        try:
            validate_placement(
                heightmap=self._bin_state.heightmap,
                bin_config=self._config.bin,
                x=x, y=y, z=z,
                oriented_l=ol, oriented_w=ow, oriented_h=oh,
                enable_stability=self._config.enable_stability,
                min_support_ratio=self._config.min_support_ratio,
                placed_boxes=self._bin_state.placed_boxes,
            )
        except PlacementError as e:
            self._log_rejection(step, box, t0, str(e))
            return None

        # 4 -- Commit placement
        placement = Placement(
            box_id=box.id, x=x, y=y, z=z,
            oriented_l=ol, oriented_w=ow, oriented_h=oh,
            orientation_idx=orientation_idx, step=step,
        )
        support = self._bin_state.get_support_ratio(x, y, ol, ow, z)
        self._bin_state.apply_placement(placement)

        elapsed = (time.perf_counter() - t0) * 1000
        self._step_log.append(StepRecord(
            step=step, box=box, success=True, placement=placement,
            fill_rate_after=self._bin_state.get_fill_rate(),
            support_ratio=support, elapsed_ms=elapsed,
        ))
        self._step_counter += 1
        return placement

    def record_rejection(self, box: Box, reason: str = "No valid placement") -> None:
        """
        Record a rejection when the strategy itself cannot find a
        valid placement.

        This is the **public API** for logging rejections -- the runner
        calls this instead of accessing private fields.
        """
        self._step_log.append(StepRecord(
            step=self._step_counter,
            box=box,
            success=False,
            rejection_reason=reason,
        ))
        self._step_counter += 1

    # -- Public: logs & summary ----------------------------------------------

    def get_step_log(self) -> List[StepRecord]:
        """Return a copy of the full step log."""
        return list(self._step_log)

    def get_latest_step(self) -> Optional[StepRecord]:
        """Return the most recent step record, or None if empty."""
        return self._step_log[-1] if self._step_log else None

    def get_summary(self) -> dict:
        """
        Compute a summary dict of the simulation results.

        Keys: fill_rate, boxes_total, boxes_placed, boxes_rejected,
              max_height, computation_time_ms, stability_rate.
        """
        placed = [r for r in self._step_log if r.success]
        total_time = sum(r.elapsed_ms for r in self._step_log)

        return {
            "fill_rate": self._bin_state.get_fill_rate(),
            "boxes_total": len(self._step_log),
            "boxes_placed": len(placed),
            "boxes_rejected": len(self._step_log) - len(placed),
            "max_height": self._bin_state.get_max_height(),
            "computation_time_ms": round(total_time, 2),
            "stability_rate": (
                sum(1 for r in placed
                    if r.support_ratio >= self._config.min_support_ratio)
                / max(len(placed), 1)
            ),
        }

    # -- Private helpers -----------------------------------------------------

    def _get_orientations(self, box: Box):
        if self._config.allow_all_orientations:
            return Orientation.get_all(box.length, box.width, box.height)
        return Orientation.get_flat(box.length, box.width, box.height)

    def _log_rejection(self, step: int, box: Box, t0: float, reason: str) -> None:
        elapsed = (time.perf_counter() - t0) * 1000
        self._step_log.append(StepRecord(
            step=step, box=box, success=False,
            rejection_reason=reason, elapsed_ms=elapsed,
        ))
        self._step_counter += 1


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

# RobotSimulator is the old name -- keep as alias so existing code still works.
RobotSimulator = PipelineSimulator
