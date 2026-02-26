import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BinConfig, Box, PlacementDecision
from simulator.close_policy import HeightClosePolicy
from simulator.session import PackingSession, SessionConfig
from strategies.base_strategy import BaseStrategy


class FlatStackStrategy(BaseStrategy):
    """Always stack at one fixed XY to trigger height-based close quickly."""

    name = "flat_stack_test"

    def decide_placement(self, box: Box, bin_state):
        return PlacementDecision(
            x=bin_state.config.margin,
            y=bin_state.config.margin,
            orientation_idx=0,
        )


def main():
    config = SessionConfig(
        bin_config=BinConfig(length=1200.0, width=800.0, height=2700.0, resolution=10.0),
        num_bins=2,
        buffer_size=8,
        pick_window=4,
        close_policy=HeightClosePolicy(max_height=250.0),  # force quick close
        max_consecutive_rejects=20,
        enable_stability=False,
        allow_all_orientations=False,
    )

    boxes = [
        Box(id=i, length=300.0, width=200.0, height=100.0, weight=1.0)
        for i in range(1, 31)
    ]

    session = PackingSession(config)
    result = session.run(boxes, FlatStackStrategy())

    print("VALIDATE_CLOSE_LOGIC")
    print(f"total_boxes={result.total_boxes}")
    print(f"total_placed={result.total_placed}")
    print(f"total_rejected={result.total_rejected}")
    print(f"closed_pallets={len(result.closed_pallets)}")
    print(f"avg_closed_fill={result.avg_closed_fill:.6f}")
    if result.closed_pallets:
        print("closed_heights=" + ",".join(f"{p.max_height:.1f}" for p in result.closed_pallets))

    # Hard assertion for this validation.
    assert len(result.closed_pallets) > 0, "No pallets were auto-closed."


if __name__ == "__main__":
    main()
