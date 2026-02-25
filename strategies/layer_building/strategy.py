"""
Layer Building Strategy — builds uniform horizontal layers for dense packing.

Algorithm overview
~~~~~~~~~~~~~~~~~~
Instead of placing boxes one at a time with no global plan, this strategy
thinks in horizontal **layers**.  Each layer has a base height (``layer_base``)
and a target thickness (``layer_target_h``).  The strategy attempts to fill the
current layer as completely as possible before advancing to the next one.

Layer detection is derived from the heightmap rather than rigid bookkeeping,
making it resilient to imperfect earlier placements.

Steps for each box:
  1. Detect the current layer level from the heightmap (mode-based).
  2. Prefer orientations whose height matches the layer target.
  3. Scan positions within the current layer zone using DBLF order.
  4. Score candidates: in-layer bonus + height-fit + area-fill - height penalty.
  5. If the current layer is sufficiently full (>80%), allow placement above it.
  6. Fall back to unconstrained BLF if no in-layer position is found.

Hyperparameters (module-level constants):
  LAYER_FULL_THRESHOLD     — fraction of footprint area that must be filled
                             before advancing to the next layer.
  HEIGHT_TOLERANCE_FRAC    — how much the box height can deviate from the
                             target layer thickness and still count as "in-layer".
  IN_LAYER_BONUS           — score bonus for placements inside the active layer.
  HEIGHT_FIT_WEIGHT        — weight for how well the box height matches the target.
  AREA_FILL_WEIGHT         — weight for the footprint area contribution.
  HEIGHT_PENALTY_WEIGHT    — weight for penalizing high placements.
"""

from typing import Optional, List, Tuple
import numpy as np

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy

# ─────────────────────────────────────────────────────────────────────────────
# Constants / Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

# Minimum support ratio — mirrors the simulator's anti-float threshold.
MIN_SUPPORT: float = 0.30

# Fraction of the bin footprint that must be covered at the current layer
# height before the strategy advances to the next layer.
LAYER_FULL_THRESHOLD: float = 0.80

# Fractional tolerance for matching a box height to the layer target.
# A box with height within target_h * HEIGHT_TOLERANCE_FRAC of the target
# is considered a good fit for the current layer.
HEIGHT_TOLERANCE_FRAC: float = 0.35

# Absolute tolerance (cm) added to height matching to handle very small boxes.
HEIGHT_TOLERANCE_ABS: float = 2.0

# ── Scoring weights ──────────────────────────────────────────────────────────

# Large bonus applied when a placement falls within the active layer zone.
IN_LAYER_BONUS: float = 3.0

# Weight for how well the oriented box height matches the layer target.
HEIGHT_FIT_WEIGHT: float = 2.0

# Weight for the footprint area that the box covers (larger footprints
# fill the layer faster, which is desirable).
AREA_FILL_WEIGHT: float = 1.0

# Penalty weight for the absolute z-position — lower placements are preferred.
HEIGHT_PENALTY_WEIGHT: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

@register_strategy
class LayerBuildingStrategy(BaseStrategy):
    """
    Layer Building placement strategy.

    Builds dense horizontal layers by:
      - Detecting the current layer level from the heightmap.
      - Preferring orientations that match the layer thickness.
      - Scoring candidates to pack tightly within each layer.
      - Advancing to the next layer only when the current one is full.

    The strategy never modifies ``bin_state``; it only reads heights,
    support ratios, and placed-box information.
    """

    name = "layer_building"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = 1.0
        # Persistent layer tracking across boxes within one episode.
        self._layer_base: float = 0.0
        self._layer_target_h: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Reset layer state at the start of a new episode."""
        super().on_episode_start(config)
        self._scan_step = max(1.0, config.bin.resolution)
        self._layer_base = 0.0
        self._layer_target_h = 0.0

    # ── Main decision ─────────────────────────────────────────────────────

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Decide where to place *box* using the layer-building heuristic.

        Returns a ``PlacementDecision`` or ``None`` if no valid position exists.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        # Resolve orientations depending on configuration.
        orientations = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Quick check: does any orientation fit the bin at all?
        any_fits = any(
            ol <= bin_cfg.length and ow <= bin_cfg.width and oh <= bin_cfg.height
            for ol, ow, oh in orientations
        )
        if not any_fits:
            return None

        # ── Detect / update layer state ───────────────────────────────────
        self._update_layer_state(bin_state, orientations)

        layer_base = self._layer_base
        layer_target_h = self._layer_target_h

        # Height tolerance for considering a box "within" the current layer.
        tolerance = max(
            layer_target_h * HEIGHT_TOLERANCE_FRAC,
            HEIGHT_TOLERANCE_ABS,
        )

        # ── Phase 1: try to place within the current layer ────────────────
        best = self._scan_candidates(
            orientations, bin_state, bin_cfg, layer_base, layer_target_h,
            tolerance, constrain_to_layer=True,
        )

        if best is not None:
            return best

        # ── Phase 2: check if layer is full enough to advance ─────────────
        layer_coverage = self._compute_layer_coverage(
            bin_state, layer_base, layer_target_h,
        )

        if layer_coverage >= LAYER_FULL_THRESHOLD and layer_target_h > 0:
            # Advance to next layer.
            self._layer_base = layer_base + layer_target_h
            self._layer_target_h = 0.0  # Will be set by the next box.

            # Re-detect target height for the new layer.
            self._update_layer_state(bin_state, orientations)
            new_layer_base = self._layer_base
            new_target_h = self._layer_target_h
            new_tolerance = max(
                new_target_h * HEIGHT_TOLERANCE_FRAC,
                HEIGHT_TOLERANCE_ABS,
            )

            best = self._scan_candidates(
                orientations, bin_state, bin_cfg, new_layer_base,
                new_target_h, new_tolerance, constrain_to_layer=True,
            )
            if best is not None:
                return best

        # ── Phase 3: unconstrained BLF fallback ──────────────────────────
        best = self._scan_candidates(
            orientations, bin_state, bin_cfg, layer_base, layer_target_h,
            tolerance, constrain_to_layer=False,
        )

        return best

    # ── Layer state management ────────────────────────────────────────────

    def _update_layer_state(
        self,
        bin_state: BinState,
        orientations: List[Tuple[float, float, float]],
    ) -> None:
        """
        Detect / update the active layer parameters from the heightmap.

        Uses the heightmap to find the dominant height levels and derives
        layer_base and layer_target_h from them.  If the layer target
        height is still zero (first box in a new layer), sets it to the
        tallest flat orientation height of the current box.
        """
        hmap = bin_state.heightmap
        bin_cfg = bin_state.config

        if self._layer_target_h == 0.0:
            # New layer — detect from heightmap or set from box.
            layer_base = self._detect_layer_base(hmap, bin_cfg)
            self._layer_base = layer_base

            # Set target height to the tallest *flat* orientation of this box.
            # Flat orientations keep the box's natural height upward, which
            # tends to produce the most uniform layers.
            flat_heights = [oh for _, _, oh in orientations]
            if flat_heights:
                self._layer_target_h = max(flat_heights)

    def _detect_layer_base(
        self,
        hmap: np.ndarray,
        bin_cfg,
    ) -> float:
        """
        Detect the bottom of the current active layer from the heightmap.

        Strategy:
          - If most of the area is at 0.0, we are on the first layer.
          - Otherwise, find the mode (most common non-zero rounded height)
            of the heightmap.  This is approximately the top of the last
            completed layer, and hence the base of the current one.
        """
        total_cells = hmap.size
        if total_cells == 0:
            return 0.0

        # Count cells at floor level.
        floor_cells = int(np.sum(hmap < 0.5))
        if floor_cells > total_cells * 0.5:
            # More than half the bin is empty — first layer.
            return 0.0

        # Round heights to the nearest integer and find the mode.
        rounded = np.round(hmap).astype(int)
        non_zero = rounded[rounded > 0]
        if non_zero.size == 0:
            return 0.0

        # Find the mode (most frequent height value).
        values, counts = np.unique(non_zero, return_counts=True)
        mode_height = float(values[np.argmax(counts)])

        # The mode is the top of the most recent complete layer, so the
        # base of the next layer.  But we should also consider that the
        # current layer might still be in progress — if the current
        # tracked base is close to the mode, keep it.
        if abs(self._layer_base - mode_height) < HEIGHT_TOLERANCE_ABS:
            return self._layer_base

        # If mode_height is above our current base, it means the current
        # layer has been (mostly) completed — use mode as new base.
        if mode_height > self._layer_base:
            return mode_height

        return self._layer_base

    def _compute_layer_coverage(
        self,
        bin_state: BinState,
        layer_base: float,
        layer_target_h: float,
    ) -> float:
        """
        Compute what fraction of the bin footprint is covered at the
        current layer's top (layer_base + layer_target_h).

        Returns a float in [0.0, 1.0].
        """
        if layer_target_h <= 0:
            return 0.0

        hmap = bin_state.heightmap
        layer_top = layer_base + layer_target_h
        tolerance = max(layer_target_h * 0.3, HEIGHT_TOLERANCE_ABS)

        # A cell is "covered" if its height is at or above the layer top
        # (within tolerance).
        covered = np.sum(hmap >= layer_top - tolerance)
        return float(covered) / float(hmap.size)

    # ── Candidate scanning & scoring ──────────────────────────────────────

    def _scan_candidates(
        self,
        orientations: List[Tuple[float, float, float]],
        bin_state: BinState,
        bin_cfg,
        layer_base: float,
        layer_target_h: float,
        tolerance: float,
        constrain_to_layer: bool,
    ) -> Optional[PlacementDecision]:
        """
        Scan all (x, y, orientation) candidates and return the best one.

        When *constrain_to_layer* is True, only positions that fall within
        the active layer zone [layer_base, layer_base + layer_target_h]
        are considered.  When False, any valid position is accepted (BLF
        fallback).

        Returns a ``PlacementDecision`` or None.
        """
        cfg = self.config
        step = self._scan_step
        bin_height = bin_cfg.height
        bin_length = bin_cfg.length
        bin_width = bin_cfg.width

        best_score: float = -1e18
        best_result: Optional[Tuple[float, float, int]] = None  # (x, y, oidx)

        for oidx, (ol, ow, oh) in enumerate(orientations):
            # Skip orientations that cannot fit the bin dimensions.
            if ol > bin_length or ow > bin_width or oh > bin_height:
                continue

            x = 0.0
            while x + ol <= bin_length + 1e-6:
                y = 0.0
                while y + ow <= bin_width + 1e-6:
                    z = bin_state.get_height_at(x, y, ol, ow)

                    # Height bounds check.
                    if z + oh > bin_height + 1e-6:
                        y += step
                        continue

                    # Support check (anti-float).
                    if z > 0.5:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < MIN_SUPPORT:
                            y += step
                            continue

                    # Optional stricter stability check.
                    if cfg.enable_stability and z > 0.5:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < cfg.min_support_ratio:
                            y += step
                            continue

                    # ── Layer constraint ──────────────────────────────
                    in_layer = (
                        layer_target_h > 0
                        and z >= layer_base - 1e-6
                        and z + oh <= layer_base + layer_target_h + tolerance
                    )

                    if constrain_to_layer and not in_layer:
                        y += step
                        continue

                    # Margin check (box-to-box gap enforcement)
                    if not bin_state.is_margin_clear(x, y, ol, ow, z, oh):
                        y += step
                        continue

                    # ── Scoring ───────────────────────────────────────
                    score = self._score_candidate(
                        x, y, z, ol, ow, oh,
                        layer_base, layer_target_h, bin_length, bin_width,
                        bin_height, in_layer,
                    )

                    if score > best_score:
                        best_score = score
                        best_result = (x, y, oidx)

                    y += step
                x += step

        if best_result is None:
            return None

        bx, by, b_oidx = best_result
        return PlacementDecision(x=bx, y=by, orientation_idx=b_oidx)

    @staticmethod
    def _score_candidate(
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        layer_base: float,
        layer_target_h: float,
        bin_length: float,
        bin_width: float,
        bin_height: float,
        in_layer: bool,
    ) -> float:
        """
        Score a candidate placement position.

        Components:
          - **in_layer_bonus**: large bonus if the box sits within the
            active layer zone, encouraging dense layer filling.
          - **height_fit**: how closely the box height matches the layer
            target (1.0 = perfect match, 0.0 = completely off).
          - **area_fill**: normalized footprint area — bigger boxes fill
            layers faster.
          - **height_penalty**: absolute z position normalized by bin
            height — lower is better.
          - **position_tiebreak**: small BLF-style tiebreaker favouring
            the bottom-left-back corner.

        Returns a scalar score (higher is better).
        """
        # In-layer bonus.
        bonus = IN_LAYER_BONUS if in_layer else 0.0

        # Height fit: how well the box height matches the layer target.
        if layer_target_h > 0:
            height_fit = 1.0 - abs(oh - layer_target_h) / layer_target_h
            height_fit = max(height_fit, 0.0)  # Clamp to [0, 1].
        else:
            height_fit = 0.5  # Neutral when no target is set.

        # Area fill: footprint fraction of the total bin area.
        area_fill = (ol * ow) / (bin_length * bin_width)

        # Height penalty: prefer placements as low as possible.
        height_penalty = z / bin_height if bin_height > 0 else 0.0

        # Small BLF tiebreaker so that among equal-score candidates,
        # the one closest to the back-left corner wins.
        position_tiebreak = (
            0.001 * (1.0 - x / bin_length) + 0.0005 * (1.0 - y / bin_width)
        )

        score = (
            bonus
            + HEIGHT_FIT_WEIGHT * height_fit
            + AREA_FILL_WEIGHT * area_fill
            - HEIGHT_PENALTY_WEIGHT * height_penalty
            + position_tiebreak
        )
        return score
