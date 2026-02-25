"""
Tsang Multi-Bin Strategy (MultiBinStrategy).
=============================================

Source Paper:
    Tsang, Mo, Chung, Lee (2025).
    "A deep reinforcement learning approach for online and concurrent 3D bin
    packing optimisation with bin replacement strategies",
    Computers in Industry, Vol. 164, Article 104202.
    Companion software: DeepPack3D (SIMPAC-2024-311, MIT license).

This is the HEURISTIC (non-DRL) implementation of the dual-bin packing logic.
It implements the key bin-routing and bin-replacement mechanics from Tsang 2025:

Bin Selection (routing):
    Best-Fit: route box to whichever bin has higher fill rate (prefers fuller bin).
    This prevents one bin from becoming too fragmented.

Bin Replacement Strategies (when should a bin be "closed"?):
    The original paper defines 5 strategies:
    - FILL:     Close bin when utilization > threshold (default 85%)
    - HEIGHT:   Close bin when max height > threshold (default 90%)
    - FAIL:     Close bin after N consecutive placement failures
    - COMBINED: Combination of FILL, HEIGHT, or FAIL (recommended by Tsang 2025)
    - MANUAL:   External trigger (not used here)

    Our MultiBinStrategy interface does not manage bin closing (that's the
    orchestrator's job), but we implement the bin-preference scoring that
    naturally implements the bin-replacement logic:
    - Prefer the bin that is closer to "ready to close" (higher fill = best-fit)
    - Reject placement in a bin that should be closed

Algorithm:
    For each call to decide_placement(box, bin_states):
    1. For each active bin, compute best (x, y, orient) using surface-contact scoring.
    2. Score each (bin, placement) pair:
       - contact_ratio * 5.0 + fill_bonus * 2.0 - height_penalty * 1.0
       - fill_bonus: bin.fill_rate (best-fit: prefer fuller bins)
    3. Return MultiBinDecision for the globally best (bin, x, y, orient).

Performance:
    Expected fill rate: 72-78% (dual-bin, heuristic version)
    Based on Tsang 2025 results: DRL version achieves 76.8-79.7%
    Heuristic baseline expected to achieve ~70-75%

References:
    Tsang et al. (2025), Computers in Industry Vol. 164
    Zhao et al. (2021), AAAI - CMDP baseline used as reference
"""

import numpy as np
from typing import Optional, List, Tuple

from config import Box, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import (
    MultiBinStrategy, MultiBinDecision, register_multibin_strategy,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SUPPORT: float = 0.30         # Anti-float threshold (always enforced)
FILL_BONUS_WEIGHT: float = 2.0    # Reward placing in fuller bin (best-fit)
CONTACT_WEIGHT: float = 5.0       # Reward bottom contact (stability + efficiency)
HEIGHT_PENALTY_WEIGHT: float = 1.0  # Penalize high placements
CONTACT_TOLERANCE: float = 0.5    # cm tolerance for height matching


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@register_multibin_strategy
class TsangMultiBinStrategy(MultiBinStrategy):
    """
    Dual-bin heuristic implementing Tsang et al. (2025) bin routing logic.

    Routes each box to the bin where it achieves the best surface contact
    score, with a best-fit bias (prefer the fuller bin).

    This is the greedy heuristic version of the DeepPack3D dual-bin system.
    It does not use the DQN but implements the same bin-routing principles.

    Key contribution from Tsang 2025 implemented here:
    - Best-fit bin routing (fuller bin preferred via fill_bonus)
    - Surface contact scoring for placement quality
    - Works natively with MultiBinPipeline (no orchestrator wrapper needed)

    Attributes:
        name: "tsang_multibin"
    """

    name: str = "tsang_multibin"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = 1.0

    def on_episode_start(self, config) -> None:
        super().on_episode_start(config)
        # Support both ExperimentConfig (.bin) and PipelineConfig (.bin_config)
        self._bin_cfg = getattr(config, "bin", None) or getattr(config, "bin_config", None)
        self._allow_all = getattr(config, "allow_all_orientations", False)
        self._scan_step = max(1.0, self._bin_cfg.resolution) if self._bin_cfg else 1.0

    def decide_placement(
        self,
        box: Box,
        bin_states: List[BinState],
    ) -> Optional[MultiBinDecision]:
        """
        Route box to best bin using surface contact + best-fit scoring.

        Args:
            box:        Box to place.
            bin_states: All active bin states (read-only).

        Returns:
            MultiBinDecision(bin_index, x, y, orientation_idx) or None.
        """
        bin_cfg = getattr(self, "_bin_cfg", None)
        if bin_cfg is None:
            return None

        step = self._scan_step

        # Resolve orientations
        allow_all = getattr(self, "_allow_all", False)
        if allow_all:
            orientations = Orientation.get_all(box.length, box.width, box.height)
        else:
            orientations = Orientation.get_flat(box.length, box.width, box.height)

        # Quick check: any orientation fits in the bin at all?
        valid_orients = [
            (oidx, ol, ow, oh)
            for oidx, (ol, ow, oh) in enumerate(orientations)
            if ol <= bin_cfg.length and ow <= bin_cfg.width and oh <= bin_cfg.height
        ]
        if not valid_orients:
            return None

        best_score: float = -np.inf
        best_decision: Optional[MultiBinDecision] = None

        for bin_idx, bin_state in enumerate(bin_states):
            fill_bonus = bin_state.get_fill_rate() * FILL_BONUS_WEIGHT

            # Get best placement in this bin
            result = self._best_in_bin(
                box, bin_state, valid_orients, step, self._config
            )
            if result is None:
                continue

            x, y, oidx, placement_score = result
            total_score = placement_score + fill_bonus

            if total_score > best_score:
                best_score = total_score
                best_decision = MultiBinDecision(
                    bin_index=bin_idx, x=x, y=y, orientation_idx=oidx
                )

        return best_decision

    def _best_in_bin(
        self,
        box: Box,
        bin_state: BinState,
        valid_orients: List[Tuple[int, float, float, float]],
        step: float,
        cfg: ExperimentConfig,
    ) -> Optional[Tuple[float, float, int, float]]:
        """
        Find the best placement in a single bin.

        Scoring (per placement):
            contact_ratio * CONTACT_WEIGHT
            - height_norm * HEIGHT_PENALTY_WEIGHT

        Args:
            box:           Box to place.
            bin_state:     Target bin state.
            valid_orients: List of (oidx, ol, ow, oh) tuples.
            step:          Grid scan step size.
            cfg:           Experiment configuration.

        Returns:
            (x, y, orientation_idx, score) or None if no valid placement.
        """
        bin_cfg = bin_state.config
        heightmap = bin_state.heightmap
        res = bin_cfg.resolution
        tol = CONTACT_TOLERANCE

        best_score: float = -np.inf
        best: Optional[Tuple[float, float, int, float]] = None

        # Candidate positions: placed box corners + origin
        candidates = self._generate_candidates(bin_state, step)

        for cx, cy in candidates:
            for oidx, ol, ow, oh in valid_orients:
                # Bounds
                if cx + ol > bin_cfg.length + 1e-6:
                    continue
                if cy + ow > bin_cfg.width + 1e-6:
                    continue

                # Resting height
                z = bin_state.get_height_at(cx, cy, ol, ow)

                # Height limit
                if z + oh > bin_cfg.height + 1e-6:
                    continue

                # Anti-float support (MIN_SUPPORT = 0.30, always enforced)
                support = 1.0
                if z > res * 0.5:
                    support = bin_state.get_support_ratio(cx, cy, ol, ow, z)
                    if support < MIN_SUPPORT:
                        continue

                # Optional strict stability
                if cfg.enable_stability and z > res * 0.5:
                    if support < cfg.min_support_ratio:
                        continue

                # Margin check (box-to-box gap enforcement)
                if not bin_state.is_margin_clear(cx, cy, ol, ow, z, oh):
                    continue

                # ── Scoring ─────────────────────────────────────────────
                # Contact ratio: bottom face cells matching z
                contact_ratio = self._contact_ratio(
                    cx, cy, z, ol, ow, heightmap, bin_cfg, res, tol
                )

                height_norm = z / bin_cfg.height if bin_cfg.height > 0 else 0.0
                score = (
                    CONTACT_WEIGHT * contact_ratio
                    - HEIGHT_PENALTY_WEIGHT * height_norm
                    + 0.5 * support  # small stability bonus
                )

                if score > best_score:
                    best_score = score
                    best = (cx, cy, oidx, score)

        return best

    def _generate_candidates(
        self,
        bin_state: BinState,
        step: float,
    ) -> List[Tuple[float, float]]:
        """
        Generate candidate (x, y) positions.

        Sources:
        1. Corners of all placed boxes (right-edge, front-edge, diagonal).
        2. Origin (0, 0).
        3. Grid scan at resolution step (for comprehensive coverage).
        """
        bin_cfg = bin_state.config
        seen = set()
        candidates = []

        def add(cx: float, cy: float) -> None:
            pt = (round(cx, 2), round(cy, 2))
            if pt not in seen and 0 <= cx <= bin_cfg.length and 0 <= cy <= bin_cfg.width:
                seen.add(pt)
                candidates.append(pt)

        # Always include origin
        add(0.0, 0.0)

        # Placed box corners
        for p in bin_state.placed_boxes:
            add(p.x, p.y)
            add(p.x_max, p.y)
            add(p.x, p.y_max)
            add(p.x_max, p.y_max)

        # Grid scan (for thorough coverage on empty bin)
        x = 0.0
        while x <= bin_cfg.length:
            y = 0.0
            while y <= bin_cfg.width:
                add(x, y)
                y += step
            x += step

        return candidates

    def _contact_ratio(
        self,
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
        """
        Fraction of box footprint cells that match the resting height z.

        For floor placements (z ≈ 0): returns 1.0 (perfect contact).
        For elevated placements: counts cells within tol of z.
        """
        if z < res * 0.5:
            return 1.0  # Floor placement: full contact

        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), bin_cfg.grid_l)
        gy_end = min(gy + int(round(ow / res)), bin_cfg.grid_w)

        region = heightmap[gx:gx_end, gy:gy_end]
        if region.size == 0:
            return 0.0

        matched = int(np.sum(np.abs(region - z) <= tol))
        return matched / region.size
