"""
Multi-bin pipeline -- direct multi-bin simulation for advanced strategies.

Unlike the orchestrator (which *wraps* a single-bin strategy with external
multi-bin logic), the MultiBinPipeline exposes ALL active bin states to
the strategy simultaneously.  This enables advanced strategies to implement
their own cross-bin reasoning: choosing which bin to fill, reasoning about
joint bin state, etc.

Architecture
~~~~~~~~~~~~
                     +-----------------------------+
  Input stream       |      MultiBinPipeline        |
  (boxes)  -------->|                              |
                     |  BoxBuffer (K boxes)          |
                     |       |                       |
                     |       v                       |
                     |  MultiBinStrategy             |
                     |  .decide_placement(           |
                     |      box,                     |
                     |      bin_states  <-----------+ |
                     |  ) -> MultiBinDecision        | |
                     |       |                      | |
                     |       v                      | |
                     |  PipelineSimulator[bin_idx]  | |
                     |  .attempt_placement(...)      | |
                     |       |                      | |
                     |       +---- update ----------+ |
                     +-----------------------------+

Usage:
    from simulator.multi_bin_pipeline import MultiBinPipeline, PipelineConfig
    from simulator.buffer import BufferPolicy

    config = PipelineConfig(
        n_bins=2,
        buffer_size=5,
        buffer_policy=BufferPolicy.LARGEST_FIRST,
    )
    pipeline = MultiBinPipeline(strategy=my_strategy, config=config)
    result = pipeline.run(boxes)
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from config import Box, BinConfig, ExperimentConfig, Placement
from simulator.bin_state import BinState
from simulator.pipeline_simulator import PipelineSimulator
from simulator.buffer import BoxBuffer, BufferPolicy


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Configuration for the MultiBinPipeline.

    Attributes:
        n_bins:          Number of active bins (default 2).
        buffer_size:     Lookahead buffer size K (default 5).
        buffer_policy:   Which box to pick from buffer next.
        bin_config:      Physical bin dimensions.
        enable_stability: Pass through to individual simulators.
        min_support_ratio: Pass through to individual simulators.
        allow_all_orientations: Pass through to individual simulators.
        verbose:         Print step-by-step decisions.
    """
    n_bins: int = 2
    buffer_size: int = 5
    buffer_policy: BufferPolicy = BufferPolicy.LARGEST_FIRST
    bin_config: BinConfig = field(default_factory=BinConfig)
    enable_stability: bool = False
    min_support_ratio: float = 0.8
    allow_all_orientations: bool = False
    verbose: bool = False

    def to_dict(self) -> dict:
        return {
            "n_bins": self.n_bins,
            "buffer_size": self.buffer_size,
            "buffer_policy": self.buffer_policy.value,
            "bin_config": self.bin_config.to_dict(),
            "enable_stability": self.enable_stability,
            "min_support_ratio": self.min_support_ratio,
            "allow_all_orientations": self.allow_all_orientations,
        }

    @property
    def bin(self) -> BinConfig:
        """Compatibility shim: allows strategies to use config.bin like ExperimentConfig."""
        return self.bin_config

    def to_experiment_config(self, strategy_name: str = "multibin") -> ExperimentConfig:
        """Create an ExperimentConfig for a single bin in this pipeline."""
        return ExperimentConfig(
            bin=self.bin_config,
            strategy_name=strategy_name,
            enable_stability=self.enable_stability,
            min_support_ratio=self.min_support_ratio,
            allow_all_orientations=self.allow_all_orientations,
            render_3d=False,
            verbose=False,
        )


# ---------------------------------------------------------------------------
# Per-bin result
# ---------------------------------------------------------------------------

@dataclass
class BinResult:
    """Result for a single bin at end of pipeline run."""
    bin_id: int
    fill_rate: float = 0.0
    boxes_placed: int = 0
    max_height: float = 0.0
    placements: List[Placement] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bin_id": self.bin_id,
            "fill_rate": round(self.fill_rate, 6),
            "boxes_placed": self.boxes_placed,
            "max_height": round(self.max_height, 2),
            "placements": [p.to_dict() for p in self.placements],
        }


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Aggregate result from a MultiBinPipeline run.

    This is the multi-bin equivalent of the single-bin experiment result dict.
    """
    total_boxes: int = 0
    total_placed: int = 0
    total_rejected: int = 0
    total_unplaceable: int = 0
    bins_used: int = 0
    aggregate_fill_rate: float = 0.0
    mean_fill_rate: float = 0.0
    max_fill_rate: float = 0.0
    min_fill_rate: float = 0.0
    computation_time_ms: float = 0.0
    bin_results: List[BinResult] = field(default_factory=list)
    decision_log: List[dict] = field(default_factory=list)
    config: Optional[PipelineConfig] = None

    def to_dict(self) -> dict:
        return {
            "total_boxes": self.total_boxes,
            "total_placed": self.total_placed,
            "total_rejected": self.total_rejected,
            "total_unplaceable": self.total_unplaceable,
            "bins_used": self.bins_used,
            "aggregate_fill_rate": round(self.aggregate_fill_rate, 6),
            "mean_fill_rate": round(self.mean_fill_rate, 6),
            "max_fill_rate": round(self.max_fill_rate, 6),
            "min_fill_rate": round(self.min_fill_rate, 6),
            "computation_time_ms": round(self.computation_time_ms, 2),
            "bin_results": [br.to_dict() for br in self.bin_results],
            "config": self.config.to_dict() if self.config else {},
        }


# ---------------------------------------------------------------------------
# Multi-bin pipeline
# ---------------------------------------------------------------------------

class MultiBinPipeline:
    """
    Direct multi-bin pipeline for advanced (MultiBinStrategy) strategies.

    The pipeline exposes ALL active bin states to the strategy simultaneously,
    allowing it to reason across bins, decide placement order, and implement
    sophisticated cross-bin packing logic that the single-bin orchestrator
    cannot support.

    Key difference from MultiBinOrchestrator:
      - Orchestrator wraps a SINGLE-BIN strategy and adds multi-bin logic externally
      - MultiBinPipeline gives the strategy DIRECT ACCESS to all bin states

    The pipeline still manages the box buffer (BoxBuffer) for semi-online
    packing, handing the strategy only the K buffered boxes and the N bin states.

    Usage:
        config = PipelineConfig(n_bins=3, buffer_size=7)
        pipeline = MultiBinPipeline(strategy=my_strategy, config=config)
        result = pipeline.run(boxes)
    """

    def __init__(self, strategy, config: PipelineConfig) -> None:
        """
        Args:
            strategy: A MultiBinStrategy instance (see strategies/base_strategy.py).
            config:   Pipeline configuration.
        """
        self._strategy = strategy
        self._config = config
        self._simulators: List[PipelineSimulator] = []
        self._buffer = BoxBuffer(config.buffer_size, config.buffer_policy)

    def run(self, boxes: List[Box]) -> PipelineResult:
        """
        Run the full multi-bin packing episode.

        The strategy receives the full list of BinState objects at each step
        and returns a MultiBinDecision specifying which bin and where to place.

        Args:
            boxes: Complete list of input boxes (in arrival order).

        Returns:
            PipelineResult with per-bin and aggregate metrics.
        """
        t_start = time.perf_counter()

        # Initialize N bins
        exp_config = self._config.to_experiment_config()
        self._simulators = [PipelineSimulator(exp_config) for _ in range(self._config.n_bins)]

        # Initialize buffer
        self._buffer.load_stream(boxes)
        self._buffer.refill()

        # Notify strategy
        self._strategy.on_episode_start(self._config)

        decision_log: List[dict] = []
        total_placed = 0
        total_rejected = 0
        total_unplaceable = 0
        step = 0

        if self._config.verbose:
            print(f"  MultiBinPipeline")
            print(f"  Strategy:    {type(self._strategy).__name__}")
            print(f"  Bins:        {self._config.n_bins}")
            print(f"  Buffer size: {self._config.buffer_size}")
            print(f"  Buffer pol:  {self._config.buffer_policy.value}")
            print(f"  Total boxes: {len(boxes)}")
            print(f"  {chr(45)*50}")

        while not self._buffer.is_empty:
            self._buffer.refill()
            if not self._buffer.buffer:
                break

            # Pick next box from buffer
            if self._config.buffer_policy == BufferPolicy.BEST_FIT_SCORE:
                box = self._pick_best_fit_box()
            else:
                box = self._buffer.pick_next()

            if box is None:
                break

            # Collect all bin states for the strategy
            bin_states = [sim.get_bin_state() for sim in self._simulators]

            # Ask the strategy to decide (it sees ALL bin states)
            decision = self._strategy.decide_placement(box, bin_states)

            placed = False
            if decision is not None:
                bin_idx = decision.bin_index
                if 0 <= bin_idx < len(self._simulators):
                    sim = self._simulators[bin_idx]
                    result = sim.attempt_placement(
                        box, decision.x, decision.y, decision.orientation_idx,
                    )
                    if result is not None:
                        placed = True
                        total_placed += 1
                        decision_log.append({
                            "step": step,
                            "box_id": box.id,
                            "bin_id": bin_idx,
                            "action": "placed",
                            "position": [result.x, result.y, result.z],
                            "fill_after": sim.get_bin_state().get_fill_rate(),
                        })
                        if self._config.verbose:
                            fill = sim.get_bin_state().get_fill_rate()
                            print(
                                f"    Step {step:4d}: Box #{box.id:3d} -> Bin {bin_idx} "
                                f"at ({result.x:.0f},{result.y:.0f},{result.z:.0f}) "
                                f"fill={fill:.1%}"
                            )

            if not placed:
                total_rejected += 1
                decision_log.append({
                    "step": step,
                    "box_id": box.id,
                    "action": "rejected",
                    "reason": "Strategy returned None or placement failed",
                })
                if self._config.verbose:
                    print(f"    Step {step:4d}: Box #{box.id:3d} -> REJECTED")

            step += 1

        computation_ms = (time.perf_counter() - t_start) * 1000

        # Notify strategy
        self._strategy.on_episode_end({})

        # Build result
        bin_results = []
        for i, sim in enumerate(self._simulators):
            state = sim.get_bin_state()
            summary = sim.get_summary()
            if summary["boxes_placed"] > 0:
                bin_results.append(BinResult(
                    bin_id=i,
                    fill_rate=state.get_fill_rate(),
                    boxes_placed=summary["boxes_placed"],
                    max_height=state.get_max_height(),
                    placements=list(state.placed_boxes),
                ))

        # Aggregate metrics — use ALL n_bins (not just bins with boxes)
        # to avoid inflating fill rate when some bins are empty.
        all_fill_rates = [0.0] * self._config.n_bins
        for br in bin_results:
            all_fill_rates[br.bin_id] = br.fill_rate
        mean_fill = sum(all_fill_rates) / max(self._config.n_bins, 1)
        max_fill = max(all_fill_rates)
        min_fill = min(all_fill_rates)

        total_placed_vol = sum(
            sum(p.volume for p in br.placements) for br in bin_results
        )
        total_bin_vol = self._config.bin_config.volume * self._config.n_bins
        agg_fill = total_placed_vol / total_bin_vol if total_bin_vol > 0 else 0.0

        result = PipelineResult(
            total_boxes=len(boxes),
            total_placed=total_placed,
            total_rejected=total_rejected,
            total_unplaceable=total_unplaceable,
            bins_used=len(bin_results),
            aggregate_fill_rate=agg_fill,
            mean_fill_rate=mean_fill,
            max_fill_rate=max_fill,
            min_fill_rate=min_fill,
            computation_time_ms=computation_ms,
            bin_results=bin_results,
            decision_log=decision_log,
            config=self._config,
        )

        if self._config.verbose:
            self._print_summary(result)

        return result

    def _pick_best_fit_box(self) -> Optional[Box]:
        """Pick the box from buffer that is easiest to place (BEST_FIT_SCORE)."""
        if not self._buffer.buffer:
            return None
        best_idx = max(range(len(self._buffer.buffer)),
                       key=lambda i: self._buffer.buffer[i].volume)
        return self._buffer.buffer.pop(best_idx)

    def get_bin_states(self) -> List[BinState]:
        """Return current state of all active bins (read-only)."""
        return [sim.get_bin_state() for sim in self._simulators]

    def get_simulators(self) -> List[PipelineSimulator]:
        """Return all active simulators (for visualization/logging)."""
        return list(self._simulators)

    @staticmethod
    def _print_summary(result: PipelineResult) -> None:
        print(f"  {chr(61)*50}")
        print(f"  PIPELINE RESULTS")
        print(f"  {chr(61)*50}")
        print(f"  Total boxes:      {result.total_boxes}")
        print(f"  Placed:           {result.total_placed}")
        print(f"  Rejected:         {result.total_rejected}")
        print(f"  Bins used:        {result.bins_used}")
        print(f"  Aggregate fill:   {result.aggregate_fill_rate:.1%}")
        print(f"  Mean fill (bins): {result.mean_fill_rate:.1%}")
        print(f"  Time:             {result.computation_time_ms:.0f}ms")
        print(f"  {chr(45)*50}")
        for br in result.bin_results:
            print(
                f"  Bin {br.bin_id}: fill={br.fill_rate:.1%}  "
                f"boxes={br.boxes_placed}  max_h={br.max_height:.0f}"
            )
        print()
