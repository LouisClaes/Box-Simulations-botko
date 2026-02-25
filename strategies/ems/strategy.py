"""
Empty Maximal Spaces (EMS) Strategy for 3D bin packing.

Algorithm overview:
    This strategy maintains a list of Empty Maximal Spaces (EMSs) -- the largest
    axis-aligned rectangular volumes in the bin that contain no placed boxes.
    Instead of scanning every grid cell, we only try to place boxes at the origins
    of these empty spaces, ensuring tight fits and minimal wasted volume.

    EMS generation is done from the heightmap at each step (rebuild approach)
    rather than incremental tracking, which is simpler and avoids accumulation
    of numerical drift. The approach:

    1. Generate candidate (x, y) positions from a coarse grid (step=5 cm) plus
       all placed-box corners (for precision near edges).
    2. At each candidate, probe the heightmap to determine the surface height z.
    3. Expand rightward and forward to find the maximal rectangle of uniform
       height starting at that position.
    4. Each such rectangle defines an approximate EMS: (x, y, z, max_l, max_w, avail_h).
    5. For each EMS, try fitting the box in each allowed orientation.
    6. Score candidates by DBLF priority plus "fit tightness" (how well the box
       fills the EMS volume).

Scoring:
    score = -z * 5.0          (strong low-placement preference)
            - x * 1.0          (left preference)
            - y * 0.5          (back preference)
            + fit_score * 3.0   (tight-fit reward)

    Where fit_score = (box_volume) / (ems_volume), clamped to [0, 1].

References:
    Gonçalves, J.F. & Resende, M.G.C. (2013).
    "A biased random key genetic algorithm for 2D and 3D bin packing problems."
    International Journal of Production Economics, 145(2), 500-510.

    Lai, K.K. & Chan, J.W.M. (1997).
    "Developing a simulated annealing algorithm for the cutting stock problem."
    Computers & Industrial Engineering, 32(1), 115-127.
"""

from typing import Optional, List, Tuple, Set
import numpy as np

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants (tuning hyperparameters)
# ─────────────────────────────────────────────────────────────────────────────

# Anti-float threshold: must match the simulator's MIN_ANTI_FLOAT_RATIO (0.30)
MIN_SUPPORT: float = 0.30

# Coarse grid step size (cm) for candidate position generation.
# Smaller = more candidates = better packing but slower.
# Larger = fewer candidates = faster but may miss good positions.
GRID_STEP: float = 5.0

# Height tolerance (cm) for considering a surface "flat" when expanding EMSs.
# A cell is included in the flat region if its height differs by at most this
# amount from the seed height. Generous tolerance to avoid overly tiny EMSs.
HEIGHT_TOLERANCE: float = 5.0

# Scoring weights for the DBLF + fit-tightness function
WEIGHT_Z: float = -5.0       # Strong preference for low placements
WEIGHT_X: float = -1.0       # Prefer left positions
WEIGHT_Y: float = -0.5       # Prefer back positions
WEIGHT_FIT: float = 3.0      # Reward tight fits (box fills EMS well)

# Minimum EMS dimension (cm): skip EMSs smaller than this on any axis
MIN_EMS_DIMENSION: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Data structure for an Empty Maximal Space
# ─────────────────────────────────────────────────────────────────────────────

class EMS:
    """
    Represents an Empty Maximal Space in the bin.

    An EMS is an axis-aligned rectangular volume defined by its origin
    (x, y, z) and extent (length, width, height). It describes a region
    where boxes could potentially be placed.

    Attributes:
        x, y, z:  Origin coordinates (back-left-bottom corner).
        length:   Extent along the x-axis.
        width:    Extent along the y-axis.
        height:   Extent along the z-axis (up to bin ceiling).
    """

    __slots__ = ("x", "y", "z", "length", "width", "height")

    def __init__(
        self, x: float, y: float, z: float,
        length: float, width: float, height: float,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.length = length
        self.width = width
        self.height = height

    @property
    def volume(self) -> float:
        """Volume of this EMS."""
        return self.length * self.width * self.height

    def can_fit(self, ol: float, ow: float, oh: float) -> bool:
        """Check if a box with oriented dimensions fits inside this EMS."""
        return (
            ol <= self.length + 1e-6
            and ow <= self.width + 1e-6
            and oh <= self.height + 1e-6
        )

    def __repr__(self) -> str:
        return (
            f"EMS(origin=({self.x:.0f},{self.y:.0f},{self.z:.0f}), "
            f"size=({self.length:.0f},{self.width:.0f},{self.height:.0f}))"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy implementation
# ─────────────────────────────────────────────────────────────────────────────

@register_strategy
class EMSStrategy(BaseStrategy):
    """
    Empty Maximal Spaces (EMS) strategy for 3D bin packing.

    Tracks maximal empty rectangular volumes in the bin and places boxes
    to maximize space utilization. Candidates are scored by a combination
    of DBLF (Deepest Bottom-Left Fill) priority and fit tightness, which
    measures how well the box fills the available empty space.

    The EMS list is rebuilt from the heightmap at each step for robustness.

    Attributes:
        name: Strategy identifier for the registry ("ems").
    """

    name: str = "ems"

    def __init__(self) -> None:
        super().__init__()

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Find the best EMS-based placement for the given box.

        Steps:
            1. Generate the current list of EMSs from the heightmap and
               placed-box corners.
            2. For each EMS, try each allowed orientation of the box.
            3. If the box fits, compute a placement score.
            4. Return the highest-scoring feasible candidate, or None.

        Args:
            box:       The box to place (original dimensions before rotation).
            bin_state: Current bin state (read-only).

        Returns:
            PlacementDecision with (x, y, orientation_idx) or None if no
            valid placement exists.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        # Resolve allowed orientations
        orientations = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Quick check: can the box fit in the bin in any orientation?
        can_fit_any = False
        for ol, ow, oh in orientations:
            if ol <= bin_cfg.length and ow <= bin_cfg.width and oh <= bin_cfg.height:
                can_fit_any = True
                break
        if not can_fit_any:
            return None

        # Generate the list of empty maximal spaces from the current state
        ems_list = self._generate_ems_list(bin_state)

        best_score: float = -float("inf")
        best_candidate: Optional[Tuple[float, float, int]] = None  # (x, y, oidx)

        for ems in ems_list:
            for oidx, (ol, ow, oh) in enumerate(orientations):
                # Check if box fits within this EMS dimensions
                if not ems.can_fit(ol, ow, oh):
                    continue

                # Position: place at the EMS origin
                px, py = ems.x, ems.y

                # Bounds check (redundant with EMS generation but safe)
                if px + ol > bin_cfg.length + 1e-6:
                    continue
                if py + ow > bin_cfg.width + 1e-6:
                    continue

                # Compute actual resting height from the heightmap
                # (may differ from ems.z due to height variations in footprint)
                z = bin_state.get_height_at(px, py, ol, ow)

                # Height limit check
                if z + oh > bin_cfg.height + 1e-6:
                    continue

                # Anti-float support check
                if z > 0.5:
                    support_ratio = bin_state.get_support_ratio(px, py, ol, ow, z)
                    if support_ratio < MIN_SUPPORT:
                        continue

                # Stability check (stricter, when enabled)
                if cfg.enable_stability and z > 0.5:
                    support_ratio = bin_state.get_support_ratio(px, py, ol, ow, z)
                    if support_ratio < cfg.min_support_ratio:
                        continue

                # Margin check (box-to-box gap enforcement)
                if not bin_state.is_margin_clear(px, py, ol, ow, z, oh):
                    continue

                # Score this candidate
                score = self._compute_score(px, py, z, ol, ow, oh, ems, bin_cfg)

                if score > best_score:
                    best_score = score
                    best_candidate = (px, py, oidx)

        # Fallback: if no EMS candidate found, do a coarse grid scan (BLF)
        if best_candidate is None:
            best_candidate = self._fallback_grid_scan(box, bin_state, orientations)

        if best_candidate is None:
            return None

        return PlacementDecision(
            x=best_candidate[0],
            y=best_candidate[1],
            orientation_idx=best_candidate[2],
        )

    # ── EMS generation ────────────────────────────────────────────────────

    def _generate_ems_list(self, bin_state: BinState) -> List[EMS]:
        """
        Generate a list of Empty Maximal Spaces from the current heightmap.

        Uses a hybrid approach:
          1. Coarse grid sampling (every GRID_STEP cm) covers the whole bin.
          2. Placed-box corners add precision near existing placements.
          3. At each unique (x, y), we probe the height and expand to find
             the maximal flat rectangle, defining an EMS.

        The resulting EMSs are sorted by (z, x, y) to facilitate DBLF ordering.

        Args:
            bin_state: Current bin state (read-only).

        Returns:
            List of EMS objects, sorted by (z, x, y) ascending.
        """
        bin_cfg = bin_state.config
        heightmap = bin_state.heightmap

        # Collect candidate (x, y) positions
        candidates: Set[Tuple[float, float]] = set()

        # 1. Coarse grid positions
        x = 0.0
        while x < bin_cfg.length:
            y = 0.0
            while y < bin_cfg.width:
                candidates.add((x, y))
                y += GRID_STEP
            x += GRID_STEP

        # 2. Placed-box corner positions (for precision)
        for p in bin_state.placed_boxes:
            # All four projected corners of the box on the XY plane
            corners = [
                (p.x, p.y),
                (p.x_max, p.y),
                (p.x, p.y_max),
                (p.x_max, p.y_max),
            ]
            for cx, cy in corners:
                if 0 <= cx < bin_cfg.length and 0 <= cy < bin_cfg.width:
                    candidates.add((cx, cy))

        # 3. Always include origin
        candidates.add((0.0, 0.0))

        # Build EMSs by expanding each candidate into a maximal rectangle
        ems_list: List[EMS] = []
        resolution = bin_cfg.resolution

        for cx, cy in candidates:
            # Get the height at this point
            z = bin_state.get_height_at(cx, cy, 1.0, 1.0)

            # Available height above this point
            avail_h = bin_cfg.height - z
            if avail_h < MIN_EMS_DIMENSION:
                continue

            # Expand rightward (along x-axis): find how far we can go while
            # the surface height stays within tolerance of z
            max_l = self._expand_along_x(
                cx, cy, z, heightmap, bin_cfg, resolution
            )
            if max_l < MIN_EMS_DIMENSION:
                continue

            # Expand forward (along y-axis): find how far we can go while
            # keeping the full x-extent at uniform height
            max_w = self._expand_along_y(
                cx, cy, z, max_l, heightmap, bin_cfg, resolution
            )
            if max_w < MIN_EMS_DIMENSION:
                continue

            ems_list.append(EMS(
                x=cx, y=cy, z=z,
                length=max_l, width=max_w, height=avail_h,
            ))

        # Sort by DBLF: lowest z first, then leftmost x, then backmost y
        ems_list.sort(key=lambda e: (e.z, e.x, e.y))

        return ems_list

    def _expand_along_x(
        self,
        cx: float,
        cy: float,
        z: float,
        heightmap: np.ndarray,
        bin_cfg,
        resolution: float,
    ) -> float:
        """
        Starting from (cx, cy), expand along the x-axis as far as the
        surface height stays within HEIGHT_TOLERANCE of z.

        Args:
            cx, cy:     Starting position.
            z:          Reference height at the starting position.
            heightmap:  The current 2D heightmap (read-only access).
            bin_cfg:    Bin configuration.
            resolution: Grid resolution.

        Returns:
            Maximum length (cm) of the flat region along x.
        """
        gx_start = int(round(cx / resolution))
        gy = int(round(cy / resolution))

        if gy >= bin_cfg.grid_w:
            return 0.0

        max_gx = bin_cfg.grid_l
        gx = gx_start

        while gx < max_gx:
            cell_height = heightmap[gx, gy]
            if abs(cell_height - z) > HEIGHT_TOLERANCE:
                break
            gx += 1

        return (gx - gx_start) * resolution

    def _expand_along_y(
        self,
        cx: float,
        cy: float,
        z: float,
        max_l: float,
        heightmap: np.ndarray,
        bin_cfg,
        resolution: float,
    ) -> float:
        """
        Starting from (cx, cy), expand along the y-axis as far as the
        ENTIRE x-extent [cx, cx+max_l) stays within HEIGHT_TOLERANCE of z.

        This ensures the resulting rectangle is fully uniform in height.

        Args:
            cx, cy:     Starting position.
            z:          Reference height.
            max_l:      The x-extent already determined.
            heightmap:  The current 2D heightmap (read-only access).
            bin_cfg:    Bin configuration.
            resolution: Grid resolution.

        Returns:
            Maximum width (cm) of the flat region along y.
        """
        gx_start = int(round(cx / resolution))
        gx_end = min(gx_start + int(round(max_l / resolution)), bin_cfg.grid_l)
        gy_start = int(round(cy / resolution))

        if gx_start >= gx_end:
            return 0.0

        max_gy = bin_cfg.grid_w
        gy = gy_start

        while gy < max_gy:
            # Check the entire x-extent for this y row
            row_slice = heightmap[gx_start:gx_end, gy]
            if np.any(np.abs(row_slice - z) > HEIGHT_TOLERANCE):
                break
            gy += 1

        return (gy - gy_start) * resolution

    # ── Scoring ───────────────────────────────────────────────────────────

    def _compute_score(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        ems: EMS,
        bin_cfg,
    ) -> float:
        """
        Score a candidate placement using DBLF priority + fit tightness.

        The fit_score measures how well the box fills the EMS it is placed
        into. A box that exactly fills an EMS gets fit_score = 1.0, which
        means no wasted space. A small box in a large EMS gets a low
        fit_score.

        Formula:
            fit_score = (ol * ow * oh) / (ems.length * ems.width * ems.height)
            score = WEIGHT_Z * z
                  + WEIGHT_X * x
                  + WEIGHT_Y * y
                  + WEIGHT_FIT * fit_score

        Args:
            x, y, z:       Placement position.
            ol, ow, oh:    Oriented box dimensions.
            ems:           The EMS this box is being placed into.
            bin_cfg:       Bin configuration.

        Returns:
            Scalar score (higher is better).
        """
        # Fit tightness: how well the box fills the EMS
        ems_volume = ems.volume
        if ems_volume > 0:
            fit_score = (ol * ow * oh) / ems_volume
        else:
            fit_score = 0.0

        # Clamp fit_score to [0, 1] (should already be, but safety)
        fit_score = min(1.0, max(0.0, fit_score))

        score = (
            WEIGHT_Z * z
            + WEIGHT_X * x
            + WEIGHT_Y * y
            + WEIGHT_FIT * fit_score
        )

        return score

    # ── Fallback: grid scan ────────────────────────────────────────────

    def _fallback_grid_scan(
        self,
        box: Box,
        bin_state: BinState,
        orientations: list,
    ) -> Optional[Tuple[float, float, int]]:
        """
        Fallback BLF grid scan when no EMS candidate works.

        Scans at a coarser grid step than baseline for speed, still using
        the same DBLF priority (lowest z, then x, then y).
        """
        cfg = self.config
        bin_cfg = cfg.bin
        step = max(1.0, bin_cfg.resolution)

        best: Optional[Tuple[float, float, float, int]] = None  # (z, x, y, oidx)

        for oidx, (ol, ow, oh) in enumerate(orientations):
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue

            x = 0.0
            while x + ol <= bin_cfg.length + 1e-6:
                y = 0.0
                while y + ow <= bin_cfg.width + 1e-6:
                    z = bin_state.get_height_at(x, y, ol, ow)
                    if z + oh > bin_cfg.height + 1e-6:
                        y += step
                        continue
                    if z > 0.5:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < MIN_SUPPORT:
                            y += step
                            continue
                    if cfg.enable_stability and z > 0.5:
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
        return (best[1], best[2], best[3])
