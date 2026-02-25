"""
Selective Hyper-Heuristic strategy for 3D bin packing.

NOVEL THESIS CONTRIBUTION -- not derived from any single published paper.
This is the primary thesis contribution: a selective hyper-heuristic that
dynamically chooses which placement heuristic to apply based on the current
packing state, expressed as a compact 6-dimensional feature vector.

Algorithm overview
~~~~~~~~~~~~~~~~~~
At each step the strategy:

  1. Extracts a 6-dimensional state-feature vector from the bin:
       f1  phase             -- normalised progress (step_count / est_total)
       f2  roughness         -- surface roughness / bin_height
       f3  fill_fraction     -- volumetric fill rate
       f4  item_size_ratio   -- box.volume / bin_cfg.volume
       f5  height_ratio      -- max_height / bin_cfg.height
       f6  surface_flatness  -- 1 - roughness_normalised

  2. Passes the features to a rule-based selector that picks one of four
     low-level heuristics:
       H1  WallE scoring      (Verma et al. AAAI 2020)
       H2  DBLF               (Karabulut & Inceoglu 2004; Zhu & Lim 2012)
       H3  Floor-building     (Classic heuristic)
       H4  Best-fit by volume (Classic bin-packing heuristic)

  3. Evaluates all feasible (x, y, orientation) candidates with the chosen
     heuristic and returns the best-scoring PlacementDecision.

Stability is enforced identically across all heuristics: every candidate must
satisfy MIN_SUPPORT >= 0.30 and, when enable_stability is True, the stricter
config.min_support_ratio.

Selector rules (no training needed):
    phase < 0.20                              -> floor_building (early foundation)
    roughness > 0.15                          -> floor_building (smooth the surface)
    fill > 0.65 AND item_ratio < 0.05         -> best_fit_volume (gap filling)
    fill > 0.50                               -> dblf (mid-late tight packing)
    default                                   -> walle (balanced stability+efficiency)

Paper references for component heuristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* WallE (H1):
    Verma et al., "A Generalized Reinforcement Learning Algorithm for Online
    3D Bin-Packing", AAAI 2020.

* DBLF (H2):
    Karabulut, K. & Inceoglu, M.M., "A hybrid genetic algorithm for packing in
    3D with deepest bottom left with fill method", 2004.
    Zhu, W. & Lim, A., "A new iterative-doubling Greedy-Lookahead algorithm
    for the single container loading problem", EJOR 2012.

* Floor-building (H3):
    Classic shelf/layer heuristic; see e.g. Martello et al. (2000),
    "Three-dimensional bin packing problems", Operations Research 48(2).

* Best-fit by volume (H4):
    Classic bin-packing heuristic; see Coffman et al. (1984),
    "Approximation algorithms for bin-packing -- an updated survey".
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from config import Box, ExperimentConfig, Orientation, PlacementDecision
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Anti-float threshold: always enforced, mirrors the simulator's own check.
MIN_SUPPORT: float = 0.30

# Roughness threshold above which floor-building is preferred.
ROUGHNESS_OVERRIDE_THRESHOLD: float = 0.15

# Fill thresholds for selector phase transitions.
FILL_LATE_THRESHOLD: float = 0.65
FILL_MID_THRESHOLD: float = 0.50

# Item size ratio below which gap-filling is preferred in the late phase.
ITEM_RATIO_SMALL_THRESHOLD: float = 0.05

# Estimated total steps for phase computation (overridden by on_episode_start
# if the config exposes it, but this reasonable default handles most datasets).
DEFAULT_ESTIMATED_TOTAL: int = 50

# WallE scoring weights (Verma et al. 2020, tuned for this simulator).
WALLE_W_VAR: float = 0.75    # Penalise placing on an uneven surface.
WALLE_W_HIGH: float = 1.0    # Reward having walls around the placement.
WALLE_W_FLUSH: float = 1.0   # Reward flush (level) placement.
WALLE_W_POS: float = 0.01    # Mild penalty for distance from the origin.
WALLE_W_Z: float = 1.0       # Penalty for absolute resting height.

# DBLF lexicographic ordering weights (left-most, lowest, front-most).
DBLF_W_X: float = 1e6
DBLF_W_Z: float = 1e3
DBLF_W_Y: float = 1.0

# Best-fit-by-volume scoring weights.
BFV_W_SUPPORT: float = 10.0     # Reward full support from below.
BFV_W_HEIGHT: float = 2.0       # Penalise high placements.
BFV_W_CONTACT: float = 3.0      # Reward cells at exactly z in the footprint.

# Roughness delta weight in floor-building (keep delta very small to avoid
# swamping the dominant minimise-z objective).
FLOOR_W_ROUGHNESS: float = 0.001

# Neighbor margin (in grid cells) for WallE neighborhood analysis.
WALLE_NEIGHBOR_MARGIN: int = 1

# Flush detection tolerance (fraction of resolution).
FLUSH_TOL_FACTOR: float = 0.5


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@register_strategy
class SelectiveHyperHeuristicStrategy(BaseStrategy):
    """
    Selective Hyper-Heuristic: dynamically picks the best placement heuristic.

    A rule-based hyper-heuristic that extracts a compact state-feature vector
    from the current bin state and uses it to select the most appropriate
    low-level placement heuristic for each incoming box.  No training or
    offline data is required.

    Attributes:
        name:                 Registry key ``"selective_hyper_heuristic"``.
        _step_count:          Number of boxes placed so far in this episode.
        _estimated_total:     Expected total boxes; used for phase feature.
        _heuristic_log:       Per-step record of which heuristic was selected.
        _scan_step:           Grid scan resolution (cm), set from bin config.
    """

    name: str = "selective_hyper_heuristic"

    def __init__(self) -> None:
        super().__init__()
        self._step_count: int = 0
        self._estimated_total: int = DEFAULT_ESTIMATED_TOTAL
        self._heuristic_log: List[str] = []
        self._scan_step: float = 1.0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Reset per-episode state and store config."""
        super().on_episode_start(config)
        self._step_count = 0
        self._heuristic_log = []
        # Use 2× resolution to limit candidate count without losing coverage.
        self._scan_step = max(1.0, config.bin.resolution * 2.0)

    def on_episode_end(self, results: dict) -> None:
        """Attach heuristic selection log to results for post-analysis."""
        results["heuristic_log"] = list(self._heuristic_log)

    # ── Main entry point ──────────────────────────────────────────────────

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Select the appropriate heuristic and return the best placement.

        Steps:
          1. Compute 6-dimensional state features.
          2. Select heuristic via rule-based selector.
          3. Evaluate candidates under the chosen heuristic.
          4. Increment step counter and return the best PlacementDecision.

        Args:
            box:       The box to place (original dimensions, read-only).
            bin_state: Current 3D state of the bin (read-only).

        Returns:
            PlacementDecision(x, y, orientation_idx), or None if the box
            cannot be placed anywhere in this bin.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        # ── Resolve orientations ─────────────────────────────────────────
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Quick reject: if no orientation fits the bin at all, bail early.
        any_fits = any(
            ol <= bin_cfg.length and ow <= bin_cfg.width and oh <= bin_cfg.height
            for ol, ow, oh in orientations
        )
        if not any_fits:
            self._heuristic_log.append("skip")
            self._step_count += 1
            return None

        # ── State features ───────────────────────────────────────────────
        features = self._compute_state_features(box, bin_state)

        # ── Heuristic selection ──────────────────────────────────────────
        selected = self._select_heuristic(features, self._step_count)
        self._heuristic_log.append(selected)

        # ── Evaluate candidates under the chosen heuristic ───────────────
        best_candidate: Optional[Tuple[float, float, int]] = None
        best_score: float = -1e18

        if selected == "walle":
            best_candidate, best_score = self._heuristic_walle(
                box, bin_state, orientations,
            )
        elif selected == "dblf":
            best_candidate, best_score = self._heuristic_dblf(
                box, bin_state, orientations,
            )
        elif selected == "floor_building":
            best_candidate, best_score = self._heuristic_floor_building(
                box, bin_state, orientations,
            )
        elif selected == "best_fit_volume":
            best_candidate, best_score = self._heuristic_best_fit_volume(
                box, bin_state, orientations,
            )

        self._step_count += 1

        if best_candidate is None:
            return None

        bx, by, b_oidx = best_candidate
        return PlacementDecision(x=bx, y=by, orientation_idx=b_oidx)

    # ── State feature computation ─────────────────────────────────────────

    def _compute_state_features(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Extract a 6-dimensional feature vector describing the current state.

        Returns:
            (phase, roughness_norm, fill_fraction, item_size_ratio,
             height_ratio, surface_flatness)

            f1  phase           = step_count / estimated_total       (0 -> 1)
            f2  roughness_norm  = surface_roughness / bin_height
            f3  fill_fraction   = volumetric fill rate               (0 -> 1)
            f4  item_size_ratio = box.volume / bin_cfg.volume
            f5  height_ratio    = max_height / bin_cfg.height
            f6  surface_flat    = 1 - roughness_norm
        """
        bin_cfg = self.config.bin

        # f1: phase
        est = max(1, self._estimated_total)
        phase = min(self._step_count / est, 1.0)

        # f2: roughness normalised by bin height
        raw_roughness = bin_state.get_surface_roughness()
        roughness_norm = (
            raw_roughness / bin_cfg.height if bin_cfg.height > 0 else 0.0
        )

        # f3: volumetric fill fraction
        fill_fraction = bin_state.get_fill_rate()

        # f4: item size relative to bin volume
        bin_vol = bin_cfg.volume
        item_size_ratio = (box.volume / bin_vol) if bin_vol > 0 else 0.0

        # f5: current max height fraction
        max_h = bin_state.get_max_height()
        height_ratio = (max_h / bin_cfg.height) if bin_cfg.height > 0 else 0.0

        # f6: surface flatness (complement of roughness)
        surface_flatness = 1.0 - roughness_norm

        return (
            phase,
            roughness_norm,
            fill_fraction,
            item_size_ratio,
            height_ratio,
            surface_flatness,
        )

    # ── Heuristic selector ────────────────────────────────────────────────

    @staticmethod
    def _select_heuristic(
        state_features: Tuple[float, float, float, float, float, float],
        step_count: int,
    ) -> str:
        """
        Rule-based heuristic selector.

        Reads the 6-dimensional state-feature vector and returns one of:
            'floor_building', 'best_fit_volume', 'dblf', 'walle'

        Rules (evaluated in priority order):
            1. Early phase (< 20%): build a flat, stable foundation.
            2. Rough surface (> 15% normalised): smooth it out.
            3. Late stage (fill > 65%) with small items (ratio < 5%):
               best-fit gap-filling.
            4. Mid-late stage (fill > 50%): DBLF for consistent packing.
            5. Default (mid phase): WallE for balanced stability+efficiency.

        Args:
            state_features: (phase, roughness, fill, item_ratio,
                              height_ratio, surface_flatness)
            step_count:     Current step index (unused in rules, available
                            for future extension).

        Returns:
            Heuristic key string.
        """
        phase = state_features[0]
        roughness = state_features[1]
        fill = state_features[2]
        item_ratio = state_features[3]

        # Rule 1: early phase -- lay a flat, stable foundation.
        if phase < 0.20:
            return "floor_building"

        # Rule 2: rough surface -- priority is to smooth it out.
        if roughness > ROUGHNESS_OVERRIDE_THRESHOLD:
            return "floor_building"

        # Rule 3: late stage with small items -- fill residual gaps.
        if fill > FILL_LATE_THRESHOLD and item_ratio < ITEM_RATIO_SMALL_THRESHOLD:
            return "best_fit_volume"

        # Rule 4: mid-to-late stage -- DBLF for consistent tight packing.
        if fill > FILL_MID_THRESHOLD:
            return "dblf"

        # Rule 5: default mid-phase -- WallE for balanced approach.
        return "walle"

    # ── Candidate generator ───────────────────────────────────────────────

    def _get_candidates(
        self,
        bin_state: BinState,
        step: float,
        include_grid: bool = True,
    ) -> List[Tuple[float, float]]:
        """
        Build a deduplicated list of (x, y) candidate positions.

        Sources:
          - Origin (0, 0) always included.
          - Corners of all placed boxes (high-value positions for snug fits).
          - Full grid scan at *step* resolution (only when include_grid=True).

        Args:
            bin_state:    Current bin state (for placed_boxes and config).
            step:         Grid scan resolution in cm.
            include_grid: Whether to add the regular grid scan candidates.

        Returns:
            List of unique (x, y) pairs within bin boundaries.
        """
        bin_cfg = bin_state.config
        candidates: set = set()

        # Origin always included.
        candidates.add((0.0, 0.0))

        # Corners of already-placed boxes.
        for p in bin_state.placed_boxes:
            for cx, cy in [
                (p.x,     p.y),
                (p.x_max, p.y),
                (p.x,     p.y_max),
                (p.x_max, p.y_max),
            ]:
                if 0 <= cx <= bin_cfg.length and 0 <= cy <= bin_cfg.width:
                    candidates.add((round(cx, 1), round(cy, 1)))

        # Regular grid scan.
        if include_grid:
            x = 0.0
            while x <= bin_cfg.length + 1e-6:
                y = 0.0
                while y <= bin_cfg.width + 1e-6:
                    candidates.add((round(x, 1), round(y, 1)))
                    y += step
                x += step

        return list(candidates)

    # ── Stability filter (shared across all heuristics) ───────────────────

    def _is_feasible(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        bin_state: BinState,
    ) -> Tuple[bool, float]:
        """
        Check physical feasibility and return (feasible, support_ratio).

        Enforces in order:
          1. Bin boundary checks (x+ol, y+ow, z+oh within limits).
          2. MIN_SUPPORT >= 0.30 for all non-floor placements.
          3. config.min_support_ratio when enable_stability is True.

        Returns:
            (True, support_ratio) if the candidate passes all checks,
            (False, 0.0) otherwise.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        # Bounds
        if x + ol > bin_cfg.length + 1e-6:
            return False, 0.0
        if y + ow > bin_cfg.width + 1e-6:
            return False, 0.0
        if z + oh > bin_cfg.height + 1e-6:
            return False, 0.0

        # Support
        if z > 0.5:
            sr = bin_state.get_support_ratio(x, y, ol, ow, z)
            if sr < MIN_SUPPORT:
                return False, 0.0
            if cfg.enable_stability and sr < cfg.min_support_ratio:
                return False, 0.0
        else:
            sr = 1.0

        # Margin check (box-to-box gap enforcement)
        if not bin_state.is_margin_clear(x, y, ol, ow, z, oh):
            return False, 0.0

        return True, sr

    # ── H1: WallE scoring ─────────────────────────────────────────────────

    def _heuristic_walle(
        self,
        box: Box,
        bin_state: BinState,
        orientations: List[Tuple[float, float, float]],
    ) -> Tuple[Optional[Tuple[float, float, int]], float]:
        """
        H1: WallE scoring (Verma et al. AAAI 2020).

        Evaluates placements using a composite score:
            S = -WALLE_W_VAR  * G_var
              + WALLE_W_HIGH  * G_high
              + WALLE_W_FLUSH * G_flush
              - WALLE_W_POS   * (x + y)
              - WALLE_W_Z     * z

        G_var:   variance of the heightmap in the footprint region (penalise
                 placing on an uneven surface).
        G_high:  number of neighboring cells higher than the new box top
                 (reward having surrounding "walls").
        G_flush: number of neighboring cells at exactly z (reward flush/level
                 contact before placement).

        Candidates: full grid scan + placed box corners.

        Returns:
            (best_candidate, best_score) where best_candidate is
            (x, y, orientation_idx) or None if no feasible position found.
        """
        bin_cfg = self.config.bin
        step = self._scan_step

        candidates = self._get_candidates(bin_state, step, include_grid=True)

        best_score: float = -1e18
        best_candidate: Optional[Tuple[float, float, int]] = None

        for oidx, (ol, ow, oh) in enumerate(orientations):
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue

            for (cx, cy) in candidates:
                z = bin_state.get_height_at(cx, cy, ol, ow)

                feasible, _ = self._is_feasible(cx, cy, z, ol, ow, oh, bin_state)
                if not feasible:
                    continue

                score = self._walle_score(cx, cy, z, ol, ow, oh, bin_state)

                if score > best_score:
                    best_score = score
                    best_candidate = (cx, cy, oidx)

        return best_candidate, best_score

    def _walle_score(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        bin_state: BinState,
    ) -> float:
        """
        Compute the WallE composite score for a single candidate.

        Uses the current heightmap (read-only) to calculate:
            G_var   -- variance in the footprint (penalise uneven surface)
            G_high  -- neighbor cells above new box top (reward having walls)
            G_flush -- neighbor cells flush with z (reward level contact)

        Score formula:
            -WALLE_W_VAR * G_var + WALLE_W_HIGH * G_high
            + WALLE_W_FLUSH * G_flush - WALLE_W_POS * (x + y)
            - WALLE_W_Z * z
        """
        heightmap = bin_state.heightmap
        cfg = bin_state.config
        res = cfg.resolution

        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), cfg.grid_l)
        gy_end = min(gy + int(round(ow / res)), cfg.grid_w)

        # G_var: variance of the footprint cells (on the current surface).
        footprint = heightmap[gx:gx_end, gy:gy_end]
        g_var = float(np.var(footprint)) if footprint.size > 0 else 0.0

        # Neighborhood: 1-cell border around footprint.
        m = WALLE_NEIGHBOR_MARGIN
        nx1 = max(0, gx - m)
        ny1 = max(0, gy - m)
        nx2 = min(cfg.grid_l, gx_end + m)
        ny2 = min(cfg.grid_w, gy_end + m)
        neighborhood = heightmap[nx1:nx2, ny1:ny2]

        new_top = z + oh
        tol = res * FLUSH_TOL_FACTOR

        # G_high: cells in neighborhood that are above the new top height.
        g_high = float(np.sum(neighborhood > new_top))

        # G_flush: cells in neighborhood at exactly the resting height z.
        g_flush = float(np.sum(np.abs(neighborhood - z) <= tol))

        score = (
            -WALLE_W_VAR   * g_var
            + WALLE_W_HIGH  * g_high
            + WALLE_W_FLUSH * g_flush
            - WALLE_W_POS   * (x + y)
            - WALLE_W_Z     * z
        )
        return score

    # ── H2: DBLF (Down-Back-Left-First) ──────────────────────────────────

    def _heuristic_dblf(
        self,
        box: Box,
        bin_state: BinState,
        orientations: List[Tuple[float, float, float]],
    ) -> Tuple[Optional[Tuple[float, float, int]], float]:
        """
        H2: Down-Back-Left-First (DBLF).

        Lexicographic ordering of candidate positions:
            Score = -(x * DBLF_W_X + z * DBLF_W_Z + y * DBLF_W_Y)

        This strongly prefers placements that are:
          1. Left-most  (smallest x)
          2. Lowest     (smallest z)
          3. Front-most (smallest y)

        Candidates: placed box corners + origin (no full grid scan needed
        since the lexicographic score favours corner-generated positions).

        Reference:
            Karabulut & Inceoglu 2004; Zhu & Lim 2012.

        Returns:
            (best_candidate, best_score) or (None, -inf).
        """
        bin_cfg = self.config.bin
        step = self._scan_step

        # DBLF benefits from corner candidates; grid adds breadth for safety.
        candidates = self._get_candidates(bin_state, step, include_grid=False)

        best_score: float = -1e18
        best_candidate: Optional[Tuple[float, float, int]] = None

        for oidx, (ol, ow, oh) in enumerate(orientations):
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue

            for (cx, cy) in candidates:
                z = bin_state.get_height_at(cx, cy, ol, ow)

                feasible, _ = self._is_feasible(cx, cy, z, ol, ow, oh, bin_state)
                if not feasible:
                    continue

                # Lexicographic DBLF score: minimise x, then z, then y.
                score = -(cx * DBLF_W_X + z * DBLF_W_Z + cy * DBLF_W_Y)

                if score > best_score:
                    best_score = score
                    best_candidate = (cx, cy, oidx)

        return best_candidate, best_score

    # ── H3: Floor-building ────────────────────────────────────────────────

    def _heuristic_floor_building(
        self,
        box: Box,
        bin_state: BinState,
        orientations: List[Tuple[float, float, float]],
    ) -> Tuple[Optional[Tuple[float, float, int]], float]:
        """
        H3: Floor-building heuristic.

        Minimises the resting height (z) to build a flat, uniform layer,
        with a very small secondary penalty on the roughness delta to
        prefer placements that smooth the surface:

            Score = -(z + FLOOR_W_ROUGHNESS * roughness_delta)

        roughness_delta: estimated change in surface roughness after placing
        the box, computed as the mean absolute height difference in the
        footprint region before vs after a virtual placement.

        Candidates: full grid scan (needed to find the globally lowest slot).

        Reference:
            Martello, Pisinger & Vigo (2000), Operations Research 48(2).

        Returns:
            (best_candidate, best_score) or (None, -inf).
        """
        bin_cfg = self.config.bin
        step = self._scan_step

        candidates = self._get_candidates(bin_state, step, include_grid=True)
        heightmap = bin_state.heightmap
        res = bin_cfg.resolution

        best_score: float = -1e18
        best_candidate: Optional[Tuple[float, float, int]] = None

        for oidx, (ol, ow, oh) in enumerate(orientations):
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue

            for (cx, cy) in candidates:
                z = bin_state.get_height_at(cx, cy, ol, ow)

                feasible, _ = self._is_feasible(cx, cy, z, ol, ow, oh, bin_state)
                if not feasible:
                    continue

                # Estimate roughness delta from virtual placement.
                roughness_delta = self._estimate_roughness_delta(
                    cx, cy, z, ol, ow, oh, heightmap, bin_cfg,
                )

                score = -(z + FLOOR_W_ROUGHNESS * roughness_delta)

                if score > best_score:
                    best_score = score
                    best_candidate = (cx, cy, oidx)

        return best_candidate, best_score

    @staticmethod
    def _estimate_roughness_delta(
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        heightmap: np.ndarray,
        bin_cfg,
    ) -> float:
        """
        Estimate the change in local surface roughness after a virtual placement.

        Computes the mean absolute height difference (a simple roughness proxy)
        in the footprint region before and after painting z+oh into a copy
        of the heightmap.  Returns (roughness_after - roughness_before).
        A negative delta means the placement smooths the surface (good).
        """
        res = bin_cfg.resolution
        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), bin_cfg.grid_l)
        gy_end = min(gy + int(round(ow / res)), bin_cfg.grid_w)

        if gx >= gx_end or gy >= gy_end:
            return 0.0

        # Use a 1-cell margin for context.
        margin = 1
        rx_s = max(0, gx - margin)
        ry_s = max(0, gy - margin)
        rx_e = min(bin_cfg.grid_l, gx_end + margin)
        ry_e = min(bin_cfg.grid_w, gy_end + margin)

        region_before = heightmap[rx_s:rx_e, ry_s:ry_e]
        if region_before.size < 2:
            return 0.0

        # Virtual placement: paint box top into a local copy.
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

        # Simple roughness: mean absolute difference between adjacent cells.
        def _rough(r: np.ndarray) -> float:
            if r.shape[0] < 2 or r.shape[1] < 2:
                return 0.0
            return float(
                (np.mean(np.abs(np.diff(r, axis=0)))
                 + np.mean(np.abs(np.diff(r, axis=1)))) / 2.0
            )

        return _rough(region_after) - _rough(region_before)

    # ── H4: Best-fit by volume ────────────────────────────────────────────

    def _heuristic_best_fit_volume(
        self,
        box: Box,
        bin_state: BinState,
        orientations: List[Tuple[float, float, float]],
    ) -> Tuple[Optional[Tuple[float, float, int]], float]:
        """
        H4: Best-fit by volume (gap-filling).

        Evaluates each candidate using:
            Score = BFV_W_SUPPORT * support_ratio
                  - BFV_W_HEIGHT  * height_norm
                  + BFV_W_CONTACT * contact_base_ratio

        support_ratio:      fraction of the footprint at exactly z (from
                            bin_state.get_support_ratio).
        height_norm:        z / bin_height (normalised resting height).
        contact_base_ratio: fraction of footprint cells whose height equals z
                            (counts cells that perfectly flush-support the box
                            base, using resolution/2 tolerance).

        Candidates: full grid scan + placed box corners.

        Reference:
            Coffman, Garey & Johnson (1984); Martello & Toth (1990).

        Returns:
            (best_candidate, best_score) or (None, -inf).
        """
        bin_cfg = self.config.bin
        step = self._scan_step
        res = bin_cfg.resolution

        candidates = self._get_candidates(bin_state, step, include_grid=True)
        heightmap = bin_state.heightmap

        best_score: float = -1e18
        best_candidate: Optional[Tuple[float, float, int]] = None

        for oidx, (ol, ow, oh) in enumerate(orientations):
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue

            for (cx, cy) in candidates:
                z = bin_state.get_height_at(cx, cy, ol, ow)

                feasible, sr = self._is_feasible(cx, cy, z, ol, ow, oh, bin_state)
                if not feasible:
                    continue

                # contact_base_ratio: fraction of footprint at exactly z.
                gx = int(round(cx / res))
                gy = int(round(cy / res))
                gx_end = min(gx + int(round(ol / res)), bin_cfg.grid_l)
                gy_end = min(gy + int(round(ow / res)), bin_cfg.grid_w)

                footprint = heightmap[gx:gx_end, gy:gy_end]
                contact_base_ratio = 0.0
                if footprint.size > 0:
                    tol = res * FLUSH_TOL_FACTOR
                    contact_cells = int(np.sum(np.abs(footprint - z) <= tol))
                    contact_base_ratio = contact_cells / footprint.size

                # height_norm: normalised resting height.
                height_norm = z / bin_cfg.height if bin_cfg.height > 0 else 0.0

                score = (
                    BFV_W_SUPPORT  * sr
                    - BFV_W_HEIGHT * height_norm
                    + BFV_W_CONTACT * contact_base_ratio
                )

                if score > best_score:
                    best_score = score
                    best_candidate = (cx, cy, oidx)

        return best_candidate, best_score
