"""
Tsang Multi-Bin heuristic strategy.

Implements a dual-bin best-fit routing policy with margin-aware candidate
sampling, dense fallback search, and anti-tower placement scoring.
"""

import numpy as np
from typing import List, Optional, Set, Tuple

from config import Box, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import (
    MultiBinDecision,
    MultiBinStrategy,
    register_multibin_strategy,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SUPPORT: float = 0.30

# Bin-routing bonus (best-fit) with a height taper to avoid late-stage towers.
FILL_BONUS_WEIGHT: float = 2.0
FILL_BONUS_HEIGHT_RATIO: float = 0.72

# In-bin placement scoring.
CONTACT_WEIGHT: float = 6.0
SUPPORT_WEIGHT: float = 1.5
HEIGHT_PENALTY_WEIGHT: float = 3.0
HEIGHT_GROWTH_WEIGHT: float = 7.0
TOWER_PENALTY_WEIGHT: float = 5.0
POSITION_WEIGHT: float = 0.5

CONTACT_TOLERANCE: float = 0.5


@register_multibin_strategy
class TsangMultiBinStrategy(MultiBinStrategy):
    """Dual-bin heuristic with best-fit routing and dense fallback search."""

    name: str = "tsang_multibin"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = 1.0

    def on_episode_start(self, config) -> None:
        super().on_episode_start(config)
        # Support both ExperimentConfig (.bin) and PipelineConfig (.bin_config).
        self._bin_cfg = getattr(config, "bin", None) or getattr(config, "bin_config", None)
        self._allow_all = getattr(config, "allow_all_orientations", False)
        self._scan_step = max(1.0, self._bin_cfg.resolution) if self._bin_cfg else 1.0

    def decide_placement(
        self,
        box: Box,
        bin_states: List[BinState],
    ) -> Optional[MultiBinDecision]:
        """Choose (bin, x, y, orientation) with highest global score."""
        bin_cfg = getattr(self, "_bin_cfg", None)
        if bin_cfg is None:
            return None

        allow_all = getattr(self, "_allow_all", False)
        orientations = (
            Orientation.get_all(box.length, box.width, box.height)
            if allow_all
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        valid_orients = [
            (oidx, ol, ow, oh)
            for oidx, (ol, ow, oh) in enumerate(orientations)
            if ol <= bin_cfg.length + 1e-6
            and ow <= bin_cfg.width + 1e-6
            and oh <= bin_cfg.height + 1e-6
        ]
        if not valid_orients:
            return None

        best_score: float = -np.inf
        best_decision: Optional[MultiBinDecision] = None

        for bin_idx, bin_state in enumerate(bin_states):
            result = self._best_in_bin(
                bin_state=bin_state,
                valid_orients=valid_orients,
                step=self._scan_step,
                cfg=self._config,
            )
            if result is None:
                continue

            x, y, oidx, placement_score = result

            # Height-tapered best-fit bonus: still prefer fuller bins, but reduce
            # bias when a bin is already too tall.
            soft_close_h = max(1.0, bin_state.config.height * FILL_BONUS_HEIGHT_RATIO)
            height_taper = max(0.0, 1.0 - (bin_state.get_max_height() / soft_close_h))
            fill_bonus = bin_state.get_fill_rate() * FILL_BONUS_WEIGHT * height_taper
            total_score = placement_score + fill_bonus

            if total_score > best_score:
                best_score = total_score
                best_decision = MultiBinDecision(
                    bin_index=bin_idx,
                    x=x,
                    y=y,
                    orientation_idx=oidx,
                )

        return best_decision

    def _best_in_bin(
        self,
        bin_state: BinState,
        valid_orients: List[Tuple[int, float, float, float]],
        step: float,
        cfg: ExperimentConfig,
    ) -> Optional[Tuple[float, float, int, float]]:
        """Find best placement in one bin with strict+fallback passes."""
        bin_cfg = bin_state.config
        heightmap = bin_state.heightmap
        res = bin_cfg.resolution

        current_max_h = bin_state.get_max_height()
        max_xy = max(1.0, bin_cfg.length + bin_cfg.width)

        # Pass 1: normal density candidates.
        primary_candidates = self._generate_candidates(
            bin_state=bin_state,
            valid_orients=valid_orients,
            step=step,
            dense=False,
        )
        best = self._score_candidates(
            candidates=primary_candidates,
            bin_state=bin_state,
            valid_orients=valid_orients,
            cfg=cfg,
            heightmap=heightmap,
            bin_cfg=bin_cfg,
            res=res,
            current_max_h=current_max_h,
            max_xy=max_xy,
        )
        if best is not None:
            return best

        # Pass 2 fallback: denser candidate set to reduce None returns.
        dense_step = max(1.0, step * 0.5)
        fallback_candidates = self._generate_candidates(
            bin_state=bin_state,
            valid_orients=valid_orients,
            step=dense_step,
            dense=True,
        )
        return self._score_candidates(
            candidates=fallback_candidates,
            bin_state=bin_state,
            valid_orients=valid_orients,
            cfg=cfg,
            heightmap=heightmap,
            bin_cfg=bin_cfg,
            res=res,
            current_max_h=current_max_h,
            max_xy=max_xy,
        )

    def _score_candidates(
        self,
        candidates: List[Tuple[float, float]],
        bin_state: BinState,
        valid_orients: List[Tuple[int, float, float, float]],
        cfg: ExperimentConfig,
        heightmap: np.ndarray,
        bin_cfg,
        res: float,
        current_max_h: float,
        max_xy: float,
    ) -> Optional[Tuple[float, float, int, float]]:
        """Evaluate all feasible (candidate, orientation) pairs."""
        best_score: float = -np.inf
        best: Optional[Tuple[float, float, int, float]] = None

        for cx, cy in candidates:
            for oidx, ol, ow, oh in valid_orients:
                if cx + ol > bin_cfg.length + 1e-6:
                    continue
                if cy + ow > bin_cfg.width + 1e-6:
                    continue

                z = bin_state.get_height_at(cx, cy, ol, ow)
                if z + oh > bin_cfg.height + 1e-6:
                    continue

                support = 1.0
                if z > res * 0.5:
                    support = bin_state.get_support_ratio(cx, cy, ol, ow, z)
                    if support < MIN_SUPPORT:
                        continue

                if cfg.enable_stability and z > res * 0.5:
                    if support < cfg.min_support_ratio:
                        continue

                if not bin_state.is_margin_clear(cx, cy, ol, ow, z, oh):
                    continue

                contact_ratio = self._contact_ratio(
                    x=cx,
                    y=cy,
                    z=z,
                    ol=ol,
                    ow=ow,
                    heightmap=heightmap,
                    bin_cfg=bin_cfg,
                    res=res,
                    tol=CONTACT_TOLERANCE,
                )

                height_norm = z / bin_cfg.height if bin_cfg.height > 0 else 0.0
                top_norm = (z + oh) / bin_cfg.height if bin_cfg.height > 0 else 0.0
                height_growth = (
                    max(0.0, (z + oh - current_max_h) / bin_cfg.height)
                    if bin_cfg.height > 0
                    else 0.0
                )
                position_penalty = (cx + cy) / max_xy

                # Penalize high, peak-growing placements more strongly than broad,
                # low-contact placements to reduce tower behavior.
                tower_penalty = (top_norm * top_norm) + height_growth * (1.0 - contact_ratio)

                score = (
                    CONTACT_WEIGHT * contact_ratio
                    + SUPPORT_WEIGHT * support
                    - HEIGHT_PENALTY_WEIGHT * height_norm
                    - HEIGHT_GROWTH_WEIGHT * height_growth
                    - TOWER_PENALTY_WEIGHT * tower_penalty
                    - POSITION_WEIGHT * position_penalty
                )

                if score > best_score:
                    best_score = score
                    best = (cx, cy, oidx, score)

        return best

    def _generate_candidates(
        self,
        bin_state: BinState,
        valid_orients: List[Tuple[int, float, float, float]],
        step: float,
        dense: bool,
    ) -> List[Tuple[float, float]]:
        """Generate margin-aware candidates with orientation-offset anchors."""
        bin_cfg = bin_state.config
        m = max(0.0, bin_cfg.margin)
        x_min = m
        y_min = m
        x_max = bin_cfg.length - m
        y_max = bin_cfg.width - m

        if x_min > x_max or y_min > y_max:
            return [(0.0, 0.0)]

        seen: Set[Tuple[float, float]] = set()
        candidates: List[Tuple[float, float]] = []

        unique_lengths = sorted({ol for _, ol, _, _ in valid_orients})
        unique_widths = sorted({ow for _, _, ow, _ in valid_orients})

        def add(cx: float, cy: float) -> None:
            if cx < x_min - 1e-6 or cy < y_min - 1e-6:
                return
            if cx > x_max + 1e-6 or cy > y_max + 1e-6:
                return
            pt = (round(cx, 3), round(cy, 3))
            if pt not in seen:
                seen.add(pt)
                candidates.append(pt)

        # Margin-aware wall anchors.
        add(x_min, y_min)
        add(x_min, y_max)
        add(x_max, y_min)
        add(x_max, y_max)

        # Anchors around placed boxes.
        for p in bin_state.placed_boxes:
            x_anchors = [p.x, p.x_max, p.x_max + m]
            y_anchors = [p.y, p.y_max, p.y_max + m]

            for ol in unique_lengths:
                x_anchors.extend([p.x - ol, p.x - ol - m, p.x_max - ol])
            for ow in unique_widths:
                y_anchors.extend([p.y - ow, p.y - ow - m, p.y_max - ow])

            for cx in x_anchors:
                for cy in y_anchors:
                    add(cx, cy)

            if dense:
                local_step = max(1.0, step)
                jitters = (0.0, -local_step, local_step)
                for bx in (p.x, p.x_max + m):
                    for by in (p.y, p.y_max + m):
                        for jx in jitters:
                            for jy in jitters:
                                add(bx + jx, by + jy)

        # Interior grid sweep.
        grid_step = max(1.0, step)
        gx = x_min
        while gx <= x_max + 1e-6:
            gy = y_min
            while gy <= y_max + 1e-6:
                add(gx, gy)
                gy += grid_step
            gx += grid_step

        if not candidates:
            add(x_min, y_min)

        # Low (x+y) points first to favor compact packing.
        candidates.sort(key=lambda p: (p[0] + p[1], p[1], p[0]))
        return candidates

    @staticmethod
    def _contact_ratio(
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        heightmap: np.ndarray,
        bin_cfg,
        res: float,
        tol: float,
    ) -> float:
        """Fraction of footprint cells at resting height z."""
        if z < res * 0.5:
            return 1.0

        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + max(1, int(round(ol / res))), bin_cfg.grid_l)
        gy_end = min(gy + max(1, int(round(ow / res))), bin_cfg.grid_w)

        region = heightmap[gx:gx_end, gy:gy_end]
        if region.size == 0:
            return 0.0

        matched = int(np.sum(np.abs(region - z) <= tol))
        return matched / region.size
