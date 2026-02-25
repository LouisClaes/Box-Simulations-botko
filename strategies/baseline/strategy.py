"""
Baseline strategy — Bottom-Left-Fill (BLF / DBLF).

Algorithm:
  1. For each allowed orientation of the box:
  2.   Scan grid positions:  x left→right,  y back→front
  3.   For each (x, y, orient): compute resting z, check bounds & support
  4.   Record the candidate with lowest (z, x, y)
  5. Return the best candidate, or None if nothing fits.

The strategy uses bin_state to make smart decisions (height queries,
support checks) so it only proposes positions the simulator will accept.
"""

from typing import Optional, Tuple

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy

# Minimum support ratio the strategy requires before proposing a position.
# Matches the simulator's anti-float threshold so proposals don't get rejected.
MIN_SUPPORT = 0.30


@register_strategy
class BaselineStrategy(BaseStrategy):
    """
    Bottom-Left-Fill (DBLF) baseline.

    Scans all grid positions trying each orientation and picks the
    lowest feasible placement (z → x → y priority).  Uses the full
    3D bin state for height queries and support ratio checks to ensure
    proposed positions are physically valid.
    """

    name = "baseline"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = 1.0

    def on_episode_start(self, config: ExperimentConfig) -> None:
        super().on_episode_start(config)
        self._scan_step = max(1.0, config.bin.resolution)

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Find the best BLF position for *box*.

        Queries ``bin_state`` for heights and support ratios at each
        candidate position, and returns the single best one.
        """
        cfg = self.config
        bin_cfg = cfg.bin
        step = self._scan_step

        # Resolve allowed orientations
        orientations = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        best: Optional[Tuple[float, float, float, int]] = None  # (z, x, y, oidx)

        for oidx, (ol, ow, oh) in enumerate(orientations):
            # Quick skip: orientation can never fit
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue

            x = 0.0
            while x + ol <= bin_cfg.length + 1e-6:
                y = 0.0
                while y + ow <= bin_cfg.width + 1e-6:
                    z = bin_state.get_height_at(x, y, ol, ow)

                    # Height check
                    if z + oh > bin_cfg.height + 1e-6:
                        y += step
                        continue

                    # Support check — use bin_state to avoid proposing
                    # positions the simulator would reject as floating
                    if z > 0.5:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < MIN_SUPPORT:
                            y += step
                            continue

                    # Stability check (stricter, when enabled)
                    if cfg.enable_stability:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < cfg.min_support_ratio:
                            y += step
                            continue

                    # Margin check (box-to-box gap enforcement)
                    if not bin_state.is_margin_clear(x, y, ol, ow, z, oh):
                        y += step
                        continue

                    candidate = (z, x, y, oidx)
                    if best is None or candidate < best:
                        best = candidate

                    y += step
                x += step

        if best is None:
            return None

        _, x_best, y_best, oidx_best = best
        return PlacementDecision(x=x_best, y=y_best, orientation_idx=oidx_best)
