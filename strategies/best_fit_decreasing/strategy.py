"""
Best Fit Decreasing Strategy — 3D adaptation of the classic Best Fit heuristic.

Algorithm overview
~~~~~~~~~~~~~~~~~~
For every incoming box the strategy evaluates **all** feasible positions and
orientations, then selects the one with the **tightest fit** — i.e. the
position where the box makes the most surface contact with walls and
neighbouring boxes, while sitting as low as possible.

This is the 3D analogue of the 1D Best Fit Decreasing (BFD) bin-packing
algorithm.  "Decreasing" refers to the common practice of sorting boxes
largest-first *before* feeding them to the strategy; the strategy itself
is order-agnostic and simply picks the tightest fit for whatever box it
receives.

Candidate generation:
  1. Regular grid scan at resolution step (same as baseline).
  2. **Corner points** of every already-placed box — these are the most
     likely positions for snug fits.

Tightness scoring:
  - Bottom contact (floor or support from below).
  - Left / right / back / front wall contact.
  - Side contact with adjacent boxes (detected via heightmap probing).
  - Normalized by total surface area to get a 0..1 tightness ratio.

Combined score:
  score = TIGHTNESS_WEIGHT * tightness
        - HEIGHT_PENALTY_WEIGHT * (z / bin_height)
        - WASTE_PENALTY_WEIGHT * wasted_fraction

Hyperparameters (module-level constants):
  TIGHTNESS_WEIGHT       — primary driver; surface contact maximization.
  HEIGHT_PENALTY_WEIGHT  — prefer low placements.
  WASTE_PENALTY_WEIGHT   — penalize air gaps below the box.
"""

from typing import Optional, List, Tuple, Set
import numpy as np

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy

# ─────────────────────────────────────────────────────────────────────────────
# Constants / Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

# Minimum support ratio — mirrors the simulator's anti-float threshold.
MIN_SUPPORT: float = 0.30

# ── Scoring weights ──────────────────────────────────────────────────────────

# Primary weight: how much of the box's surface is in contact with walls/boxes.
TIGHTNESS_WEIGHT: float = 3.0

# Penalty for placing high in the bin (normalized z).
HEIGHT_PENALTY_WEIGHT: float = 2.0

# Penalty for wasted (air) volume directly below the box footprint.
WASTE_PENALTY_WEIGHT: float = 0.5

# Small tolerance used for floating-point wall/edge detection (cm).
WALL_TOLERANCE: float = 0.5

# Number of sample points along each side used for neighbor contact probing.
# Higher values are more accurate but slower.
NEIGHBOR_SAMPLE_POINTS: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

@register_strategy
class BestFitDecreasingStrategy(BaseStrategy):
    """
    Best Fit Decreasing placement strategy.

    Evaluates all feasible (x, y, orientation) candidates and picks the
    one with the highest **tightness** score — maximizing surface contact
    with walls and adjacent boxes while keeping placements low.

    The strategy never modifies ``bin_state``; it only reads from it.
    """

    name = "best_fit_decreasing"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = 1.0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_episode_start(self, config: ExperimentConfig) -> None:
        super().on_episode_start(config)
        self._scan_step = max(1.0, config.bin.resolution)

    # ── Main decision ─────────────────────────────────────────────────────

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Find the tightest-fit position for *box*.

        Generates candidates from a grid scan plus corner points of placed
        boxes, evaluates each for tightness, and returns the best.
        """
        cfg = self.config
        bin_cfg = cfg.bin
        step = self._scan_step

        # Resolve orientations.
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

        # ── Build candidate x, y positions ────────────────────────────────
        candidate_positions = self._generate_candidate_positions(
            bin_state, bin_cfg, step,
        )

        # Cache the heightmap reference for neighbor probing.
        hmap = bin_state.heightmap

        best_score: float = -1e18
        best_result: Optional[Tuple[float, float, int]] = None

        for oidx, (ol, ow, oh) in enumerate(orientations):
            # Skip orientations that exceed bin dimensions.
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue

            for (cx, cy) in candidate_positions:
                # Bounds check for this orientation at this position.
                if cx + ol > bin_cfg.length + 1e-6:
                    continue
                if cy + ow > bin_cfg.width + 1e-6:
                    continue

                z = bin_state.get_height_at(cx, cy, ol, ow)

                # Height overflow.
                if z + oh > bin_cfg.height + 1e-6:
                    continue

                # Anti-float support check.
                if z > 0.5:
                    sr = bin_state.get_support_ratio(cx, cy, ol, ow, z)
                    if sr < MIN_SUPPORT:
                        continue

                # Optional stricter stability.
                if cfg.enable_stability and z > 0.5:
                    sr = bin_state.get_support_ratio(cx, cy, ol, ow, z)
                    if sr < cfg.min_support_ratio:
                        continue

                # Margin check (box-to-box gap enforcement)
                if not bin_state.is_margin_clear(cx, cy, ol, ow, z, oh):
                    continue

                # ── Compute tightness ─────────────────────────────────
                tightness = self._compute_tightness(
                    cx, cy, z, ol, ow, oh, bin_state, bin_cfg,
                )

                # ── Compute wasted volume fraction below the box ──────
                wasted = self._compute_waste(cx, cy, z, ol, ow, oh, bin_state)

                # ── Combined score ────────────────────────────────────
                normalized_z = z / bin_cfg.height if bin_cfg.height > 0 else 0.0
                score = (
                    TIGHTNESS_WEIGHT * tightness
                    - HEIGHT_PENALTY_WEIGHT * normalized_z
                    - WASTE_PENALTY_WEIGHT * wasted
                )

                # Tiny BLF tiebreaker (prefer back-left-bottom).
                score += 0.001 * (1.0 - cx / bin_cfg.length)
                score += 0.0005 * (1.0 - cy / bin_cfg.width)

                if score > best_score:
                    best_score = score
                    best_result = (cx, cy, oidx)

        if best_result is None:
            return None

        bx, by, b_oidx = best_result
        return PlacementDecision(x=bx, y=by, orientation_idx=b_oidx)

    # ── Candidate position generation ─────────────────────────────────────

    def _generate_candidate_positions(
        self,
        bin_state: BinState,
        bin_cfg,
        step: float,
    ) -> List[Tuple[float, float]]:
        """
        Build a list of unique (x, y) candidate positions.

        Sources:
          1. Regular grid scan at *step* resolution.
          2. Corner points of all already-placed boxes — these positions
             are likely to produce snug fits against existing boxes.
        """
        positions: Set[Tuple[float, float]] = set()

        # ── Grid scan ─────────────────────────────────────────────────
        x = 0.0
        while x <= bin_cfg.length + 1e-6:
            y = 0.0
            while y <= bin_cfg.width + 1e-6:
                positions.add((round(x, 6), round(y, 6)))
                y += step
            x += step

        # ── Corner points of placed boxes ─────────────────────────────
        for p in bin_state.placed_boxes:
            # Each placed box has corners at (x, y), (x_max, y),
            # (x, y_max), (x_max, y_max).  All four are valid candidate
            # origins for the next box.
            corners = [
                (p.x, p.y),
                (p.x_max, p.y),
                (p.x, p.y_max),
                (p.x_max, p.y_max),
            ]
            for (cx, cy) in corners:
                # Only include if within bin boundaries.
                if 0 <= cx <= bin_cfg.length and 0 <= cy <= bin_cfg.width:
                    positions.add((round(cx, 6), round(cy, 6)))

        return list(positions)

    # ── Tightness computation ─────────────────────────────────────────────

    def _compute_tightness(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        bin_state: BinState,
        bin_cfg,
    ) -> float:
        """
        Compute the tightness score for placing a box at (x, y, z) with
        oriented dimensions (ol, ow, oh).

        Tightness is the fraction of the box's total surface area that is
        in contact with the floor, bin walls, or adjacent boxes.

        Returns a float in [0.0, 1.0].
        """
        # Total surface area of the box (all 6 faces).
        total_surface = 2.0 * (ol * ow + ol * oh + ow * oh)
        if total_surface < 1e-9:
            return 0.0

        contact = 0.0

        # ── Bottom face (ol x ow) ────────────────────────────────────
        if z < 0.01:
            # Sitting on the floor — full bottom contact.
            contact += ol * ow
        else:
            # Partial contact from support below.
            sr = bin_state.get_support_ratio(x, y, ol, ow, z)
            contact += sr * ol * ow

        # ── Wall contact ──────────────────────────────────────────────
        # Left wall (x = 0).
        if x < WALL_TOLERANCE:
            contact += ow * oh

        # Right wall (x + ol = bin_length).
        if x + ol > bin_cfg.length - WALL_TOLERANCE:
            contact += ow * oh

        # Back wall (y = 0).
        if y < WALL_TOLERANCE:
            contact += ol * oh

        # Front wall (y + ow = bin_width).
        if y + ow > bin_cfg.width - WALL_TOLERANCE:
            contact += ol * oh

        # ── Side contact with adjacent boxes (heightmap probing) ──────
        hmap = bin_state.heightmap
        res = bin_cfg.resolution

        # Left side: probe the column just to the left of the box (x - 1 cell).
        contact += self._probe_side_contact(
            hmap, res, bin_cfg,
            probe_x=x - res, probe_y=y,
            probe_length=res, probe_width=ow,
            z=z, oh=oh,
            contact_face_area=ow * oh,
        )

        # Right side: probe the column just to the right.
        contact += self._probe_side_contact(
            hmap, res, bin_cfg,
            probe_x=x + ol, probe_y=y,
            probe_length=res, probe_width=ow,
            z=z, oh=oh,
            contact_face_area=ow * oh,
        )

        # Back side: probe the row just behind the box (y - 1 cell).
        contact += self._probe_side_contact(
            hmap, res, bin_cfg,
            probe_x=x, probe_y=y - res,
            probe_length=ol, probe_width=res,
            z=z, oh=oh,
            contact_face_area=ol * oh,
        )

        # Front side: probe the row just in front.
        contact += self._probe_side_contact(
            hmap, res, bin_cfg,
            probe_x=x, probe_y=y + ow,
            probe_length=ol, probe_width=res,
            z=z, oh=oh,
            contact_face_area=ol * oh,
        )

        # Clamp to [0, 1]: contact can slightly exceed total_surface if
        # wall contact and neighbor contact overlap for corner/edge cases.
        tightness = min(contact / total_surface, 1.0)
        return tightness

    @staticmethod
    def _probe_side_contact(
        hmap: np.ndarray,
        res: float,
        bin_cfg,
        probe_x: float,
        probe_y: float,
        probe_length: float,
        probe_width: float,
        z: float,
        oh: float,
        contact_face_area: float,
    ) -> float:
        """
        Probe a thin strip adjacent to one face of the box and estimate
        how much of that face is in contact with an existing box.

        The strip is one resolution unit thick and runs along the face.
        We check how many cells in that strip have a height that overlaps
        the vertical extent [z, z + oh] of the box being placed.

        Returns the estimated contact area (not normalized).
        """
        # Convert probe region to grid coordinates.
        gx = int(round(probe_x / res))
        gy = int(round(probe_y / res))
        gx_end = int(round((probe_x + probe_length) / res))
        gy_end = int(round((probe_y + probe_width) / res))

        # Clamp to valid heightmap range.
        gx = max(gx, 0)
        gy = max(gy, 0)
        gx_end = min(gx_end, bin_cfg.grid_l)
        gy_end = min(gy_end, bin_cfg.grid_w)

        if gx >= gx_end or gy >= gy_end:
            return 0.0

        region = hmap[gx:gx_end, gy:gy_end]
        if region.size == 0:
            return 0.0

        # A neighboring cell contributes contact if its height overlaps
        # with the vertical range [z, z + oh].  That is, the neighbor's
        # height must be > z (it extends into our vertical range).
        z_top = z + oh
        # Fraction of cells where the neighbor's column has material in
        # the range [z, z+oh].
        overlap_cells = np.sum(region > z + 0.01)
        overlap_fraction = float(overlap_cells) / float(region.size)

        return overlap_fraction * contact_face_area

    # ── Wasted volume estimation ──────────────────────────────────────────

    @staticmethod
    def _compute_waste(
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        bin_state: BinState,
    ) -> float:
        """
        Estimate the wasted (air gap) volume fraction below the box.

        The "waste" is the empty space between the bottom of the box (z)
        and the actual surface of whatever is below it in the footprint.
        This is normalized by the box's own volume so that larger boxes
        are not unfairly penalized.

        Returns a float >= 0.0 (0.0 means no waste, i.e. sitting on floor
        or perfectly supported).
        """
        box_volume = ol * ow * oh
        if box_volume < 1e-9 or z < 0.01:
            # On the floor — no waste.
            return 0.0

        hmap = bin_state.heightmap
        res = bin_state.config.resolution

        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), bin_state.config.grid_l)
        gy_end = min(gy + int(round(ow / res)), bin_state.config.grid_w)

        if gx >= gx_end or gy >= gy_end:
            return 0.0

        region = hmap[gx:gx_end, gy:gy_end]

        # Total air volume below the box: for each cell, the gap is
        # (z - cell_height) if positive, summed over the footprint,
        # scaled by resolution^2 to get real area per cell.
        gaps = np.maximum(z - region, 0.0)
        air_volume = float(np.sum(gaps)) * (res * res)

        # Normalize by box volume.
        return air_volume / box_volume
