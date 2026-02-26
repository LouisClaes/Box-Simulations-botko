"""
Skyline strategy with valley-first search, tower suppression, and safe fallback.

Candidate score:
    score = -W_z * z
            + W_fill * valley_fill
            + W_unif * uniformity_delta
            - W_tower * tower_penalty

where:
  - valley_fill      = min(box_width / valley_depth, 1)
  - uniformity_delta = local_roughness_before - local_roughness_after
  - tower_penalty    = squared excess of box top above local/global references

If valley-first search finds no valid candidate (for example because the
valley budget is exhausted), the strategy runs a full safe fallback scan
on the grid.
"""

from typing import List, Optional, Tuple

import numpy as np

from config import Box, ExperimentConfig, Orientation, PlacementDecision
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# Matches simulator anti-float threshold.
MIN_SUPPORT: float = 0.30

# Score weights.
WEIGHT_Z: float = 3.0
WEIGHT_VALLEY_FILL: float = 1.5
WEIGHT_UNIFORMITY: float = 1.1
WEIGHT_TOWER: float = 4.5

# Skyline search budget.
MAX_VALLEY_CANDIDATES: int = 40
X_SCAN_STEP_MULT: float = 1.0

# Fallback search uses full-grid scan at this multiplier of resolution.
FALLBACK_SCAN_STEP_MULT: float = 1.0

# Uniformity uses roughness delta in a local window around footprint.
UNIFORMITY_WINDOW_MARGIN_CELLS: int = 2

# Tower penalty compares new top to local ring and global percentile.
TOWER_WINDOW_MARGIN_CELLS: int = 3
TOWER_ALLOWED_EXTRA_RATIO: float = 0.03


@register_strategy
class SkylineStrategy(BaseStrategy):
    """Valley-first skyline strategy with anti-tower behavior."""

    name = "skyline"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = 1.0
        self._fallback_step: float = 1.0

    def on_episode_start(self, config: ExperimentConfig) -> None:
        super().on_episode_start(config)
        self._scan_step = max(1.0, config.bin.resolution) * X_SCAN_STEP_MULT
        self._fallback_step = max(1.0, config.bin.resolution) * FALLBACK_SCAN_STEP_MULT

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        cfg = self.config
        bin_cfg = cfg.bin
        heightmap = bin_state.heightmap

        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Skyline per y-band: average height across x-columns.
        skyline = np.mean(heightmap, axis=0)
        valley_order = np.argsort(skyline)
        valley_depths = self._precompute_valley_depths(skyline, bin_cfg.resolution)
        global_height_ref = self._global_height_reference(heightmap)

        best_score = -np.inf
        best_candidate: Optional[Tuple[float, float, int]] = None

        valleys_tried = 0
        for gy_start in valley_order:
            if valleys_tried >= MAX_VALLEY_CANDIDATES:
                break

            y_start = float(gy_start) * bin_cfg.resolution
            valley_depth = valley_depths[int(gy_start)]

            for oidx, (ol, ow, oh) in enumerate(orientations):
                if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                    continue
                if y_start + ow > bin_cfg.width + 1e-6:
                    continue

                x = 0.0
                while x + ol <= bin_cfg.length + 1e-6:
                    z = self._feasible_height(x, y_start, ol, ow, oh, bin_state)
                    if z is not None:
                        score = self._score_candidate(
                            heightmap=heightmap,
                            x=x,
                            y=y_start,
                            z=z,
                            ol=ol,
                            ow=ow,
                            oh=oh,
                            valley_depth=valley_depth,
                            global_height_ref=global_height_ref,
                            bin_cfg=bin_cfg,
                        )
                        if score > best_score:
                            best_score = score
                            best_candidate = (x, y_start, oidx)
                    x += self._scan_step

            valleys_tried += 1

        # Safe fallback: full-grid scan if skyline budget finds no candidate.
        if best_candidate is None:
            best_candidate = self._fallback_grid_search(
                bin_state=bin_state,
                orientations=orientations,
                skyline=skyline,
                valley_depths=valley_depths,
                global_height_ref=global_height_ref,
            )

        if best_candidate is None:
            return None

        return PlacementDecision(
            x=best_candidate[0],
            y=best_candidate[1],
            orientation_idx=best_candidate[2],
        )

    def _fallback_grid_search(
        self,
        bin_state: BinState,
        orientations: List[Tuple[float, float, float]],
        skyline: np.ndarray,
        valley_depths: np.ndarray,
        global_height_ref: float,
    ) -> Optional[Tuple[float, float, int]]:
        """Exhaustive fallback scan on the full grid (bounded by resolution step)."""
        bin_cfg = self.config.bin
        heightmap = bin_state.heightmap
        res = bin_cfg.resolution

        best_score = -np.inf
        best_candidate: Optional[Tuple[float, float, int]] = None

        y = 0.0
        while y <= bin_cfg.width + 1e-6:
            gy = int(round(y / res))
            if gy < 0 or gy >= len(skyline):
                y += self._fallback_step
                continue
            valley_depth = float(valley_depths[gy])

            for oidx, (ol, ow, oh) in enumerate(orientations):
                if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                    continue
                if y + ow > bin_cfg.width + 1e-6:
                    continue

                x = 0.0
                while x + ol <= bin_cfg.length + 1e-6:
                    z = self._feasible_height(x, y, ol, ow, oh, bin_state)
                    if z is not None:
                        score = self._score_candidate(
                            heightmap=heightmap,
                            x=x,
                            y=y,
                            z=z,
                            ol=ol,
                            ow=ow,
                            oh=oh,
                            valley_depth=valley_depth,
                            global_height_ref=global_height_ref,
                            bin_cfg=bin_cfg,
                        )
                        if score > best_score:
                            best_score = score
                            best_candidate = (x, y, oidx)
                    x += self._fallback_step

            y += self._fallback_step

        return best_candidate

    def _feasible_height(
        self,
        x: float,
        y: float,
        ol: float,
        ow: float,
        oh: float,
        bin_state: BinState,
    ) -> Optional[float]:
        """Return resting z if candidate is physically valid, else None."""
        cfg = self.config
        bin_cfg = cfg.bin

        if x + ol > bin_cfg.length + 1e-6:
            return None
        if y + ow > bin_cfg.width + 1e-6:
            return None

        z = bin_state.get_height_at(x, y, ol, ow)
        if z + oh > bin_cfg.height + 1e-6:
            return None

        if z > 0.5:
            support_ratio = bin_state.get_support_ratio(x, y, ol, ow, z)
            if support_ratio < MIN_SUPPORT:
                return None
            if cfg.enable_stability and support_ratio < cfg.min_support_ratio:
                return None

        if not bin_state.is_margin_clear(x, y, ol, ow, z, oh):
            return None

        return z

    def _score_candidate(
        self,
        *,
        heightmap: np.ndarray,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        valley_depth: float,
        global_height_ref: float,
        bin_cfg,
    ) -> float:
        valley_fill = self._valley_fill_bonus(ow, valley_depth)
        uniformity = self._uniformity_bonus(heightmap, x, y, z, ol, ow, oh, bin_cfg)
        tower_penalty = self._tower_penalty(
            heightmap=heightmap,
            x=x,
            y=y,
            z=z,
            ol=ol,
            ow=ow,
            oh=oh,
            global_height_ref=global_height_ref,
            bin_cfg=bin_cfg,
        )
        return (
            -WEIGHT_Z * z
            + WEIGHT_VALLEY_FILL * valley_fill
            + WEIGHT_UNIFORMITY * uniformity
            - WEIGHT_TOWER * tower_penalty
        )

    @staticmethod
    def _precompute_valley_depths(skyline: np.ndarray, resolution: float) -> np.ndarray:
        depths = np.zeros_like(skyline, dtype=float)
        for gy in range(len(skyline)):
            depths[gy] = SkylineStrategy._measure_valley_depth(
                skyline=skyline,
                gy_center=gy,
                valley_floor=float(skyline[gy]),
                resolution=resolution,
            )
        return depths

    @staticmethod
    def _measure_valley_depth(
        skyline: np.ndarray,
        gy_center: int,
        valley_floor: float,
        resolution: float,
    ) -> float:
        tolerance = resolution * 1.5
        n = len(skyline)

        left = gy_center
        while left > 0 and abs(float(skyline[left - 1]) - valley_floor) <= tolerance:
            left -= 1

        right = gy_center
        while right < n - 1 and abs(float(skyline[right + 1]) - valley_floor) <= tolerance:
            right += 1

        return float(right - left + 1) * resolution

    @staticmethod
    def _valley_fill_bonus(box_width: float, valley_depth: float) -> float:
        if valley_depth <= 0:
            return 0.0
        return min(box_width / valley_depth, 1.0)

    @staticmethod
    def _local_roughness(region: np.ndarray) -> float:
        if region.size < 2:
            return 0.0
        parts: List[float] = []
        dx = np.abs(np.diff(region, axis=0))
        dy = np.abs(np.diff(region, axis=1))
        if dx.size > 0:
            parts.append(float(np.mean(dx)))
        if dy.size > 0:
            parts.append(float(np.mean(dy)))
        if not parts:
            return 0.0
        return float(np.mean(parts))

    @staticmethod
    def _uniformity_bonus(
        heightmap: np.ndarray,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        bin_cfg,
    ) -> float:
        """
        Non-degenerate smoothness metric.

        This computes local roughness delta in an expanded window:
            uniformity = (roughness_before - roughness_after) / bin_height
        """
        res = bin_cfg.resolution
        grid_l = bin_cfg.grid_l
        grid_w = bin_cfg.grid_w

        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), grid_l)
        gy_end = min(gy + int(round(ow / res)), grid_w)
        if gx >= gx_end or gy >= gy_end:
            return 0.0

        m = UNIFORMITY_WINDOW_MARGIN_CELLS
        wx0 = max(0, gx - m)
        wy0 = max(0, gy - m)
        wx1 = min(grid_l, gx_end + m)
        wy1 = min(grid_w, gy_end + m)

        before = heightmap[wx0:wx1, wy0:wy1].copy()
        if before.size == 0:
            return 0.0

        after = before.copy()
        lx0 = gx - wx0
        ly0 = gy - wy0
        lx1 = gx_end - wx0
        ly1 = gy_end - wy0
        top = z + oh
        after[lx0:lx1, ly0:ly1] = np.maximum(after[lx0:lx1, ly0:ly1], top)

        rough_before = SkylineStrategy._local_roughness(before)
        rough_after = SkylineStrategy._local_roughness(after)
        normalizer = max(1.0, float(bin_cfg.height))
        return (rough_before - rough_after) / normalizer

    @staticmethod
    def _global_height_reference(heightmap: np.ndarray) -> float:
        if heightmap.size == 0:
            return 0.0
        return float(np.percentile(heightmap, 90.0))

    @staticmethod
    def _tower_penalty(
        *,
        heightmap: np.ndarray,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        global_height_ref: float,
        bin_cfg,
    ) -> float:
        """
        Penalize vertical outliers.

        local_excess  = max(0, box_top - (local_ring_p75 + allowance))
        global_excess = max(0, box_top - (global_p90      + allowance))

        tower_penalty = 0.7 * (local_excess / H)^2 + 0.3 * (global_excess / H)^2
        """
        res = bin_cfg.resolution
        grid_l = bin_cfg.grid_l
        grid_w = bin_cfg.grid_w
        h = max(1.0, float(bin_cfg.height))

        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), grid_l)
        gy_end = min(gy + int(round(ow / res)), grid_w)
        if gx >= gx_end or gy >= gy_end:
            return 0.0

        m = TOWER_WINDOW_MARGIN_CELLS
        wx0 = max(0, gx - m)
        wy0 = max(0, gy - m)
        wx1 = min(grid_l, gx_end + m)
        wy1 = min(grid_w, gy_end + m)

        window = heightmap[wx0:wx1, wy0:wy1]
        if window.size == 0:
            return 0.0

        # Ring excludes the footprint itself.
        ring_mask = np.ones(window.shape, dtype=bool)
        lx0 = gx - wx0
        ly0 = gy - wy0
        lx1 = gx_end - wx0
        ly1 = gy_end - wy0
        ring_mask[lx0:lx1, ly0:ly1] = False
        ring_vals = window[ring_mask]

        if ring_vals.size > 0:
            local_ref = float(np.percentile(ring_vals, 75.0))
        else:
            local_ref = global_height_ref

        allowance = TOWER_ALLOWED_EXTRA_RATIO * float(bin_cfg.height)
        box_top = z + oh
        local_excess = max(0.0, box_top - (local_ref + allowance))
        global_excess = max(0.0, box_top - (global_height_ref + allowance))

        local_term = (local_excess / h) ** 2
        global_term = (global_excess / h) ** 2
        return 0.7 * local_term + 0.3 * global_term
