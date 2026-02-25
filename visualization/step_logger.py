"""
Step logger — console output and structured recording of each step.

Usage:
    logger = StepLogger(verbose=True)
    logger.log_step(step_record)
    logger.print_summary(summary_dict)
"""

from typing import List
from simulator.pipeline_simulator import StepRecord


class StepLogger:
    """Logs placement steps to console and stores them for JSON output."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._records: List[dict] = []

    def log_step(self, record: StepRecord) -> None:
        """Log a single step (success or rejection)."""
        self._records.append(record.to_dict())

        if not self.verbose:
            return

        box = record.box
        dims_str = f"{box.length:.0f}x{box.width:.0f}x{box.height:.0f}"

        if record.success and record.placement is not None:
            p = record.placement
            print(
                f"  Step {record.step:3d}: "
                f"Box #{box.id:3d} ({dims_str}) "
                f"-> ({p.x:.0f}, {p.y:.0f}, {p.z:.0f}) "
                f"orient={p.orientation_idx}  "
                f"fill={record.fill_rate_after:.1%}  "
                f"support={record.support_ratio:.0%}  "
                f"[{record.elapsed_ms:.1f}ms]  OK"
            )
        else:
            print(
                f"  Step {record.step:3d}: "
                f"Box #{box.id:3d} ({dims_str}) "
                f"-> REJECTED: {record.rejection_reason}  "
                f"[{record.elapsed_ms:.1f}ms]"
            )

    def print_summary(self, summary: dict) -> None:
        """Print a formatted experiment summary block."""
        print("\n" + "=" * 65)
        print("  EXPERIMENT SUMMARY")
        print("=" * 65)
        print(f"  Fill rate:        {summary['fill_rate']:.1%}")
        print(f"  Boxes placed:     {summary['boxes_placed']} / {summary['boxes_total']}")
        print(f"  Boxes rejected:   {summary['boxes_rejected']}")
        print(f"  Max height:       {summary['max_height']:.1f}")
        print(f"  Stability rate:   {summary['stability_rate']:.1%}")
        print(f"  Computation time: {summary['computation_time_ms']:.1f} ms")
        print("=" * 65 + "\n")

    def get_records(self) -> List[dict]:
        """All logged step records as dicts (for JSON output)."""
        return list(self._records)
