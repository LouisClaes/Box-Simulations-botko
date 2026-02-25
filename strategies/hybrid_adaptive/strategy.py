"""
Hybrid Adaptive Strategy -- phase-aware meta-strategy for 3D bin packing.

**NOVEL META-STRATEGY** — not derived from any published paper.
First adaptive, phase-switching strategy designed for online 3D bin packing.

Algorithm overview
~~~~~~~~~~~~~~~~~~
Detects the current packing "phase" from the bin state and dynamically switches
between three inline sub-strategies.  The intuition is that different stages of
packing benefit from fundamentally different heuristics:

  * **Foundation (fill < 25%)** — The bin is mostly empty.  Priority: create a
    stable, wall-hugging base.  Score rewards wall contact, floor placement,
    and large footprint coverage.

  * **Growth (25%-65% fill)** — The bin is partially filled.  Priority: pack
    tightly using DBLF-style scanning with adjacency bonuses.  Score rewards
    low z, good support, and side contact with existing boxes.

  * **Completion (>65% fill)** — The bin is crowded.  Priority: fill remaining
    gaps and valleys.  Score heavily penalizes height and rewards placements
    below the average surface level.

Smooth transitions:
    Near phase boundaries (20-30% and 55-65%), scores are linearly blended
    so there is no hard discontinuity in placement behaviour.

Phase tracking:
    The strategy logs which phase was used for each box in ``self._phase_log``.
    This list is attached to the results dict in ``on_episode_end`` for
    analysis and visualization.

Hyperparameters (module-level constants):
    MIN_SUPPORT           — anti-float threshold (0.30)
    PHASE_FOUNDATION_END  — fill rate where foundation ends (0.25)
    PHASE_GROWTH_END      — fill rate where growth ends (0.65)
    BLEND_WIDTH           — width of the blending zone (0.10)
    + per-phase scoring weight constants

This strategy does NOT modify the original bin_state.
"""

from typing import Optional, List, Tuple
import numpy as np

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ─────────────────────────────────────────────────────────────────────────────
# Constants / Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

# Anti-float threshold — matches the simulator.
MIN_SUPPORT: float = 0.30

# Phase boundaries (by volumetric fill rate).
PHASE_FOUNDATION_END: float = 0.25
PHASE_GROWTH_END: float = 0.65

# Width of the blending zone between phases.
BLEND_WIDTH: float = 0.10

# ── Foundation phase weights ──────────────────────────────────────────────────
FND_W_WALL: float = 3.0       # Wall contact reward
FND_W_FLOOR: float = 2.0      # Floor contact reward
FND_W_HEIGHT: float = -1.0    # Height penalty (normalized z/h)
FND_W_FOOTPRINT: float = 1.0  # Footprint area reward

# ── Growth phase weights ──────────────────────────────────────────────────────
GRW_W_Z: float = -4.0         # Low z priority
GRW_W_X: float = -1.0         # Left preference
GRW_W_Y: float = -0.5         # Back preference
GRW_W_SUPPORT: float = 2.0    # Support ratio reward
GRW_W_ADJACENCY: float = 1.0  # Adjacency to existing boxes

# ── Completion phase weights ──────────────────────────────────────────────────
CMP_W_Z: float = -5.0          # Very strong low z
CMP_W_VALLEY: float = 3.0      # Valley fill reward
CMP_W_GAP_REDUCE: float = 2.0  # Gap reduction reward
CMP_W_ROUGHNESS: float = -0.5  # Roughness penalty

# Contact detection tolerance (cm).
CONTACT_TOL: float = 1.5

# Scan step size (cm) — used for grid scanning in all phases.
SCAN_STEP: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

@register_strategy
class HybridAdaptiveStrategy(BaseStrategy):
    """
    Hybrid Adaptive meta-strategy for 3D bin packing.

    Detects the current packing phase (foundation / growth / completion) and
    applies the most appropriate inline sub-strategy.  Near phase boundaries,
    scores are linearly blended for smooth transitions.

    Attributes:
        name: Strategy identifier for the registry ("hybrid_adaptive").
    """

    name = "hybrid_adaptive"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = SCAN_STEP
        self._phase_log: List[str] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Reset phase log and scan step at the start of each episode."""
        super().on_episode_start(config)
        self._scan_step = max(1.0, config.bin.resolution)
        self._phase_log = []

    def on_episode_end(self, results: dict) -> None:
        """Attach the phase log to the results dict for analysis."""
        results["phase_log"] = list(self._phase_log)

    # ── Main entry point ──────────────────────────────────────────────────

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Decide where to place *box* using the adaptive phase-based approach.

        1. Detect the current phase from the bin state.
        2. Compute per-phase scores for all candidates.
        3. Blend scores near phase boundaries.
        4. Return the best candidate or None.

        Args:
            box:       The box to place (original dimensions).
            bin_state: Current 3D bin state (read-only).

        Returns:
            PlacementDecision or None.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        orientations = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Quick reject.
        any_fits = any(
            ol <= bin_cfg.length and ow <= bin_cfg.width and oh <= bin_cfg.height
            for ol, ow, oh in orientations
        )
        if not any_fits:
            self._phase_log.append("skip")
            return None

        # ── Detect phase and blending weights ───────────────────────────
        fill = bin_state.get_fill_rate()
        phase, w_fnd, w_grw, w_cmp = self._detect_phase_weights(fill)
        self._phase_log.append(phase)

        # Pre-compute state metrics needed for scoring.
        heightmap = bin_state.heightmap
        avg_height = float(np.mean(heightmap)) if heightmap.size > 0 else 0.0
        height_var = float(np.var(heightmap)) if heightmap.size > 0 else 0.0

        # ── Scan all candidates ─────────────────────────────────────────
        step = self._scan_step
        best_score: float = -1e18
        best_candidate: Optional[Tuple[float, float, int]] = None

        for oidx, (ol, ow, oh) in enumerate(orientations):
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue

            x = 0.0
            while x + ol <= bin_cfg.length + 1e-6:
                y = 0.0
                while y + ow <= bin_cfg.width + 1e-6:
                    z = bin_state.get_height_at(x, y, ol, ow)

                    # Height bounds.
                    if z + oh > bin_cfg.height + 1e-6:
                        y += step
                        continue

                    # Anti-float check.
                    if z > 0.5:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < MIN_SUPPORT:
                            y += step
                            continue
                    else:
                        sr = 1.0

                    # Stricter stability.
                    if cfg.enable_stability and z > 0.5:
                        sr2 = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr2 < cfg.min_support_ratio:
                            y += step
                            continue

                    # ── Compute blended score ───────────────────────
                    score = 0.0

                    if w_fnd > 0:
                        score += w_fnd * self._score_foundation(
                            x, y, z, ol, ow, oh, sr, bin_cfg,
                        )

                    if w_grw > 0:
                        score += w_grw * self._score_growth(
                            x, y, z, ol, ow, oh, sr, bin_state, bin_cfg,
                        )

                    if w_cmp > 0:
                        score += w_cmp * self._score_completion(
                            x, y, z, ol, ow, oh, avg_height, height_var,
                            heightmap, bin_cfg,
                        )

                    if score > best_score:
                        best_score = score
                        best_candidate = (x, y, oidx)

                    y += step
                x += step

        if best_candidate is None:
            return None

        bx, by, b_oidx = best_candidate
        return PlacementDecision(x=bx, y=by, orientation_idx=b_oidx)

    # ── Phase detection ───────────────────────────────────────────────────

    @staticmethod
    def _detect_phase_weights(
        fill: float,
    ) -> Tuple[str, float, float, float]:
        """
        Detect the current packing phase and compute blending weights.

        Returns:
            (phase_name, w_foundation, w_growth, w_completion) where weights
            sum to 1.0.  In pure phase regions one weight is 1.0 and the
            others are 0.0.  In blending zones weights interpolate linearly.
        """
        # Pure foundation.
        if fill < PHASE_FOUNDATION_END - BLEND_WIDTH / 2:
            return ("foundation", 1.0, 0.0, 0.0)

        # Blend: foundation -> growth.
        blend_start = PHASE_FOUNDATION_END - BLEND_WIDTH / 2
        blend_end = PHASE_FOUNDATION_END + BLEND_WIDTH / 2
        if fill < blend_end:
            alpha = (fill - blend_start) / BLEND_WIDTH
            alpha = max(0.0, min(1.0, alpha))
            return ("foundation->growth", 1.0 - alpha, alpha, 0.0)

        # Pure growth.
        if fill < PHASE_GROWTH_END - BLEND_WIDTH / 2:
            return ("growth", 0.0, 1.0, 0.0)

        # Blend: growth -> completion.
        blend_start_2 = PHASE_GROWTH_END - BLEND_WIDTH / 2
        blend_end_2 = PHASE_GROWTH_END + BLEND_WIDTH / 2
        if fill < blend_end_2:
            alpha = (fill - blend_start_2) / BLEND_WIDTH
            alpha = max(0.0, min(1.0, alpha))
            return ("growth->completion", 0.0, 1.0 - alpha, alpha)

        # Pure completion.
        return ("completion", 0.0, 0.0, 1.0)

    # ── Foundation sub-strategy scoring ───────────────────────────────────

    @staticmethod
    def _score_foundation(
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        support_ratio: float,
        bin_cfg,
    ) -> float:
        """
        Foundation phase: wall-hugging, stable base building.

        Score:
            FND_W_WALL * wall_contact
            + FND_W_FLOOR * floor_contact
            + FND_W_HEIGHT * (z / bin_height)
            + FND_W_FOOTPRINT * (footprint / floor_area)

        Wall contact counts how many of the 4 vertical faces touch a bin wall.
        Floor contact is 1.0 if z ~ 0, else 0.0.
        """
        # Wall contact: count touches with the 4 bin walls.
        wall_contact = 0.0
        if x < CONTACT_TOL:
            wall_contact += 1.0
        if abs(x + ol - bin_cfg.length) < CONTACT_TOL:
            wall_contact += 1.0
        if y < CONTACT_TOL:
            wall_contact += 1.0
        if abs(y + ow - bin_cfg.width) < CONTACT_TOL:
            wall_contact += 1.0
        # Normalize to [0, 1].
        wall_contact /= 4.0

        # Floor contact.
        floor_contact = 1.0 if z < CONTACT_TOL else 0.0

        # Height penalty (normalized).
        h_norm = z / bin_cfg.height if bin_cfg.height > 0 else 0.0

        # Footprint coverage.
        floor_area = bin_cfg.length * bin_cfg.width
        footprint = (ol * ow) / floor_area if floor_area > 0 else 0.0

        score = (
            FND_W_WALL * wall_contact
            + FND_W_FLOOR * floor_contact
            + FND_W_HEIGHT * h_norm
            + FND_W_FOOTPRINT * footprint
        )
        return score

    # ── Growth sub-strategy scoring ───────────────────────────────────────

    @staticmethod
    def _score_growth(
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        support_ratio: float,
        bin_state: BinState,
        bin_cfg,
    ) -> float:
        """
        Growth phase: efficient DBLF-style packing with adjacency scoring.

        Score:
            GRW_W_Z * (z / bin_height)
            + GRW_W_X * (x / bin_length)
            + GRW_W_Y * (y / bin_width)
            + GRW_W_SUPPORT * support_ratio
            + GRW_W_ADJACENCY * adjacency_ratio

        Adjacency is computed by checking how many placed boxes share a face
        with this placement (simplified: count touching box faces).
        """
        h = bin_cfg.height if bin_cfg.height > 0 else 1.0
        l = bin_cfg.length if bin_cfg.length > 0 else 1.0
        w = bin_cfg.width if bin_cfg.width > 0 else 1.0

        z_norm = z / h
        x_norm = x / l
        y_norm = y / w

        # Count lateral adjacency with placed boxes.
        adj_count = 0
        max_adj = 4  # left, right, front, back

        for p in bin_state.placed_boxes:
            # Need vertical overlap.
            if z >= p.z_max or z + oh <= p.z:
                continue

            # Left/right adjacency (y overlap required).
            y_overlap = (y < p.y_max and y + ow > p.y)
            if y_overlap:
                if abs(x + ol - p.x) < CONTACT_TOL:
                    adj_count += 1
                elif abs(x - p.x_max) < CONTACT_TOL:
                    adj_count += 1

            # Front/back adjacency (x overlap required).
            x_overlap = (x < p.x_max and x + ol > p.x)
            if x_overlap:
                if abs(y + ow - p.y) < CONTACT_TOL:
                    adj_count += 1
                elif abs(y - p.y_max) < CONTACT_TOL:
                    adj_count += 1

        adjacency = min(adj_count, max_adj) / max_adj

        # Also reward wall adjacency as a form of "contact".
        wall_adj = 0.0
        if x < CONTACT_TOL:
            wall_adj += 0.25
        if abs(x + ol - bin_cfg.length) < CONTACT_TOL:
            wall_adj += 0.25
        if y < CONTACT_TOL:
            wall_adj += 0.25
        if abs(y + ow - bin_cfg.width) < CONTACT_TOL:
            wall_adj += 0.25

        total_adjacency = min(1.0, adjacency + wall_adj)

        score = (
            GRW_W_Z * z_norm
            + GRW_W_X * x_norm
            + GRW_W_Y * y_norm
            + GRW_W_SUPPORT * support_ratio
            + GRW_W_ADJACENCY * total_adjacency
        )
        return score

    # ── Completion sub-strategy scoring ───────────────────────────────────

    @staticmethod
    def _score_completion(
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        avg_height: float,
        height_var: float,
        heightmap: np.ndarray,
        bin_cfg,
    ) -> float:
        """
        Completion phase: gap-filling with valley priority.

        Score:
            CMP_W_Z * (z / bin_height)
            + CMP_W_VALLEY * valley_fill
            + CMP_W_GAP_REDUCE * gap_reduction
            + CMP_W_ROUGHNESS * roughness_increase

        valley_fill:     (avg_height - z) / avg_height  if z < avg  else 0
        gap_reduction:   estimated reduction in height variance after placement
        roughness_increase: estimated change in local roughness (penalized)
        """
        h = bin_cfg.height if bin_cfg.height > 0 else 1.0
        z_norm = z / h

        # Valley fill: how much below average this position is.
        if avg_height > 1e-6 and z < avg_height:
            valley_fill = (avg_height - z) / avg_height
        else:
            valley_fill = 0.0

        # Gap reduction: estimate how much the placement reduces height variance.
        # We compute the variance of the footprint region before and after
        # a virtual placement.  Uses a copy of the relevant slice.
        res = bin_cfg.resolution
        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), bin_cfg.grid_l)
        gy_end = min(gy + int(round(ow / res)), bin_cfg.grid_w)

        gap_reduction = 0.0
        roughness_increase = 0.0

        if gx < gx_end and gy < gy_end:
            # Expand region slightly for context.
            margin = 2
            rx_s = max(0, gx - margin)
            ry_s = max(0, gy - margin)
            rx_e = min(bin_cfg.grid_l, gx_end + margin)
            ry_e = min(bin_cfg.grid_w, gy_end + margin)

            region_before = heightmap[rx_s:rx_e, ry_s:ry_e]
            if region_before.size > 0:
                var_before = float(np.var(region_before))

                # Virtual placement: copy and paint.
                region_after = region_before.copy()
                local_gx = gx - rx_s
                local_gy = gy - ry_s
                local_gx_end = gx_end - rx_s
                local_gy_end = gy_end - ry_s
                box_top = z + oh
                region_after[local_gx:local_gx_end, local_gy:local_gy_end] = np.maximum(
                    region_after[local_gx:local_gx_end, local_gy:local_gy_end],
                    box_top,
                )
                var_after = float(np.var(region_after))

                # Positive gap_reduction means we reduced variance (good).
                max_var = h * h
                if max_var > 0:
                    gap_reduction = (var_before - var_after) / max_var
                    gap_reduction = max(-1.0, min(1.0, gap_reduction))

                # Roughness increase: difference in mean absolute gradient.
                if region_before.shape[0] > 1 and region_before.shape[1] > 1:
                    rough_before = (
                        float(np.mean(np.abs(np.diff(region_before, axis=0))))
                        + float(np.mean(np.abs(np.diff(region_before, axis=1))))
                    ) / 2.0
                    rough_after = (
                        float(np.mean(np.abs(np.diff(region_after, axis=0))))
                        + float(np.mean(np.abs(np.diff(region_after, axis=1))))
                    ) / 2.0
                    roughness_increase = (rough_after - rough_before) / h if h > 0 else 0.0

        score = (
            CMP_W_Z * z_norm
            + CMP_W_VALLEY * valley_fill
            + CMP_W_GAP_REDUCE * gap_reduction
            + CMP_W_ROUGHNESS * roughness_increase
        )
        return score
