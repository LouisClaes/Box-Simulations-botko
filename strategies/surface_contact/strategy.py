"""
Surface Contact Maximizer strategy for 3D bin packing.

NOVEL STRATEGY -- not based on any published paper. This is an original
algorithm designed specifically for this simulator.

Core idea:
    Maximize the total surface area of the new box that is in contact with
    walls, the floor, and faces of already-placed boxes.  High contact area
    simultaneously reduces wasted space (tight packing), increases physical
    stability (more support points), and creates compact structures naturally.

Algorithm:
    1. Generate candidate positions: full grid scan at resolution step size
       PLUS corners of already-placed boxes (known high-value positions).
    2. For every (x, y, orientation) candidate:
       a. Compute resting z from the heightmap.
       b. Reject if out-of-bounds, over height limit, or insufficient support.
       c. Compute contact area for all 6 faces of the box:
          - Bottom: cells in the heightmap footprint matching z (+/- tolerance)
          - Top:    placed boxes whose base sits at z+oh in the footprint
          - Left/Right/Back/Front: wall contact (full face) OR heightmap
            adjacency analysis for box-to-box lateral contact
       d. Aggregate into a contact ratio (total contact / total surface area).
    3. Score each candidate with a weighted combination of:
       - Contact ratio (primary, weight 5.0)
       - Height penalty (secondary, weight -2.0)
       - Support bonus (tertiary, weight 1.0)
       - Surface roughness delta (quaternary, weight -0.3)
    4. Return the highest-scoring candidate, or None.

Hyperparameters:
    WEIGHT_CONTACT          = 5.0   -- strongly reward high contact ratios
    WEIGHT_HEIGHT           = 2.0   -- prefer lower placements
    WEIGHT_SUPPORT          = 1.0   -- prefer stable bases
    WEIGHT_ROUGHNESS_DELTA  = 0.3   -- keep the surface smooth
    CONTACT_TOLERANCE       = 0.5   -- cm tolerance for matching heights
    MIN_SUPPORT             = 0.30  -- anti-float threshold (matches simulator)
"""

import numpy as np
from typing import Optional, List, Tuple, Set

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ---------------------------------------------------------------------------
# Module-level constants (hyperparameters)
# ---------------------------------------------------------------------------

# Anti-float threshold -- matches the simulator's rejection limit.
MIN_SUPPORT: float = 0.30

# Scoring weights
WEIGHT_CONTACT: float = 5.0          # Primary: maximize surface contact
WEIGHT_HEIGHT: float = 2.0           # Secondary: prefer low placements
WEIGHT_SUPPORT: float = 1.0          # Tertiary: prefer high support ratio
WEIGHT_ROUGHNESS_DELTA: float = 0.3  # Quaternary: keep surface smooth

# Tolerance for height matching when computing contact areas (cm).
CONTACT_TOLERANCE: float = 0.5


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

@register_strategy
class SurfaceContactStrategy(BaseStrategy):
    """
    Surface Contact Maximizer: placement strategy that maximizes the total
    surface area of a box that touches walls, floor, or other boxes.

    The insight is that contact area is a proxy for packing quality:
    - High bottom contact = good support (stable)
    - High lateral contact = tight packing (space-efficient)
    - High total contact = compact structure (fewer gaps)

    This strategy scans all grid positions plus placed-box corners,
    evaluates each with a detailed 6-face contact calculation, and returns
    the candidate with the best weighted score.

    Attributes:
        name: Strategy identifier for the registry ("surface_contact").
    """

    name: str = "surface_contact"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = 1.0

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Store config and derive the grid scan step from bin resolution."""
        super().on_episode_start(config)
        self._scan_step = max(1.0, config.bin.resolution)

    # -- Main entry point ---------------------------------------------------

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Find the best placement that maximizes surface contact.

        Steps:
            1. Build candidate (x, y) positions from grid scan + box corners.
            2. For each candidate and each orientation, check feasibility.
            3. Compute 6-face contact area and aggregate into a score.
            4. Return the highest-scoring feasible candidate.

        Args:
            box:       The box to place (original dimensions before rotation).
            bin_state: Current 3D bin state (read-only -- must NOT be modified).

        Returns:
            PlacementDecision(x, y, orientation_idx) or None if no valid
            placement exists.
        """
        cfg = self.config
        bin_cfg = cfg.bin
        step = self._scan_step

        # Resolve allowed orientations
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Quick check: can the box fit in any orientation at all?
        valid_orientations = [
            (oidx, ol, ow, oh)
            for oidx, (ol, ow, oh) in enumerate(orientations)
            if ol <= bin_cfg.length and ow <= bin_cfg.width and oh <= bin_cfg.height
        ]
        if not valid_orientations:
            return None

        # Read-only references
        heightmap = bin_state.heightmap
        resolution = bin_cfg.resolution

        # Build candidate positions: grid scan + box-corner positions
        candidates = self._generate_candidates(bin_state, step)

        # Pre-compute current surface roughness for delta calculation
        current_roughness = bin_state.get_surface_roughness()

        best_score: float = -np.inf
        best_candidate: Optional[Tuple[float, float, int]] = None

        for cx, cy in candidates:
            for oidx, ol, ow, oh in valid_orientations:
                # --- Bounds check ---
                if cx + ol > bin_cfg.length + 1e-6:
                    continue
                if cy + ow > bin_cfg.width + 1e-6:
                    continue

                # --- Resting height ---
                z = bin_state.get_height_at(cx, cy, ol, ow)

                # --- Height limit ---
                if z + oh > bin_cfg.height + 1e-6:
                    continue

                # --- Anti-float support check ---
                support_ratio = 1.0
                if z > 0.5:
                    support_ratio = bin_state.get_support_ratio(
                        cx, cy, ol, ow, z
                    )
                    if support_ratio < MIN_SUPPORT:
                        continue

                # --- Stricter stability when enabled ---
                if cfg.enable_stability and z > 0.5:
                    if support_ratio < cfg.min_support_ratio:
                        continue

                # --- Margin check (box-to-box gap enforcement) ---
                if not bin_state.is_margin_clear(cx, cy, ol, ow, z, oh):
                    continue

                # --- Compute 6-face contact area ---
                contact_ratio = self._compute_contact_ratio(
                    cx, cy, z, ol, ow, oh,
                    heightmap, bin_state, bin_cfg, resolution,
                )

                # --- Compute surface roughness delta ---
                roughness_delta = self._compute_roughness_delta(
                    cx, cy, z, ol, ow, oh,
                    heightmap, bin_cfg, resolution,
                    current_roughness,
                )

                # --- Composite score ---
                height_norm = z / bin_cfg.height if bin_cfg.height > 0 else 0.0
                score = (
                    WEIGHT_CONTACT * contact_ratio
                    - WEIGHT_HEIGHT * height_norm
                    + WEIGHT_SUPPORT * support_ratio
                    - WEIGHT_ROUGHNESS_DELTA * roughness_delta
                )

                if score > best_score:
                    best_score = score
                    best_candidate = (cx, cy, oidx)

        if best_candidate is None:
            return None

        return PlacementDecision(
            x=best_candidate[0],
            y=best_candidate[1],
            orientation_idx=best_candidate[2],
        )

    # -- Candidate generation -----------------------------------------------

    def _generate_candidates(
        self,
        bin_state: BinState,
        step: float,
    ) -> List[Tuple[float, float]]:
        """
        Generate candidate (x, y) positions from two sources:

        1. Full grid scan at the given step size.
        2. Corners of all placed boxes (right edge, front edge, origin).

        The placed-box corners are important because they represent natural
        contact points where new boxes can achieve high contact ratios.

        Duplicates are removed via a set. The list is sorted by (z, x, y)
        for efficiency -- low positions are evaluated first, and if a very
        good candidate is found early the higher ones have lower impact.

        Args:
            bin_state: Current bin state (read-only).
            step:      Grid scanning step size (cm).

        Returns:
            List of unique (x, y) candidate positions.
        """
        bin_cfg = bin_state.config
        seen: Set[Tuple[float, float]] = set()
        candidates: List[Tuple[float, float]] = []

        # Source 1: Grid scan positions
        x = 0.0
        while x <= bin_cfg.length:
            y = 0.0
            while y <= bin_cfg.width:
                pt = (x, y)
                if pt not in seen:
                    seen.add(pt)
                    candidates.append(pt)
                y += step
            x += step

        # Source 2: Corners of placed boxes
        for p in bin_state.placed_boxes:
            corner_points = [
                (p.x, p.y),           # origin corner of box
                (p.x_max, p.y),       # right edge
                (p.x, p.y_max),       # front edge
                (p.x_max, p.y_max),   # diagonal corner
            ]
            for pt in corner_points:
                if (pt not in seen
                        and 0 <= pt[0] <= bin_cfg.length
                        and 0 <= pt[1] <= bin_cfg.width):
                    seen.add(pt)
                    candidates.append(pt)

        # Sort by estimated height (lowest first) then by (x, y)
        # This helps find good low-contact positions early
        def sort_key(pt: Tuple[float, float]) -> Tuple[float, float, float]:
            z_est = bin_state.get_height_at(pt[0], pt[1], 1.0, 1.0)
            return (z_est, pt[0], pt[1])

        candidates.sort(key=sort_key)
        return candidates

    # -- Contact computation ------------------------------------------------

    def _compute_contact_ratio(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        heightmap: np.ndarray,
        bin_state: BinState,
        bin_cfg,
        resolution: float,
    ) -> float:
        """
        Compute the fraction of total surface area that is in contact with
        walls, floor, or other boxes.

        Checks all 6 faces:
            - Bottom (ol x ow): heightmap cells matching z
            - Top    (ol x ow): placed boxes whose base is at z+oh
            - Left   (ow x oh): wall at x=0, or heightmap column at x-1
            - Right  (ow x oh): wall at x+ol=length, or column at x+ol
            - Back   (ol x oh): wall at y=0, or heightmap row at y-1
            - Front  (ol x oh): wall at y+ow=width, or row at y+ow

        Args:
            x, y, z:       Position of the box's bottom-back-left corner.
            ol, ow, oh:    Oriented box dimensions.
            heightmap:     Current heightmap (read-only).
            bin_state:     Current bin state (for placed_boxes queries).
            bin_cfg:       Bin configuration.
            resolution:    Grid resolution.

        Returns:
            Float in [0.0, 1.0] -- fraction of total surface area in contact.
        """
        # Grid indices for the box footprint
        gx = int(round(x / resolution))
        gy = int(round(y / resolution))
        gx_end = min(gx + int(round(ol / resolution)), bin_cfg.grid_l)
        gy_end = min(gy + int(round(ow / resolution)), bin_cfg.grid_w)

        # Total surface area (max possible contact)
        total_surface = 2.0 * (ol * ow + ol * oh + ow * oh)
        if total_surface <= 0:
            return 0.0

        total_contact = 0.0
        tol = CONTACT_TOLERANCE

        # ---- Bottom face (ol x ow) ----
        if z < tol:
            # On the floor: full bottom contact
            total_contact += ol * ow
        else:
            # Count heightmap cells that match z within tolerance
            footprint = heightmap[gx:gx_end, gy:gy_end]
            if footprint.size > 0:
                matched = np.sum(np.abs(footprint - z) <= tol)
                cell_area = resolution * resolution
                total_contact += float(matched) * cell_area

        # ---- Top face (ol x ow) ----
        box_top = z + oh
        for p in bin_state.placed_boxes:
            # Check if any placed box has its base at our top
            if abs(p.z - box_top) <= tol:
                # Compute footprint overlap
                overlap_x = max(0.0, min(x + ol, p.x_max) - max(x, p.x))
                overlap_y = max(0.0, min(y + ow, p.y_max) - max(y, p.y))
                total_contact += overlap_x * overlap_y

        # ---- Left face (ow x oh, at x=x) ----
        if x <= tol:
            # Flush with left wall
            total_contact += ow * oh
        else:
            # Check heightmap column at x-1 for overlap with [z, z+oh]
            left_col_idx = max(0, gx - 1)
            if left_col_idx < bin_cfg.grid_l:
                left_col = heightmap[left_col_idx, gy:gy_end]
                if left_col.size > 0:
                    # For each cell in the column, compute vertical overlap
                    # with the box's left face [z, z+oh]
                    # The heightmap gives the top of whatever is at that cell.
                    # Contact exists where the neighbour height is in [z, z+oh].
                    contact_cells = np.sum(
                        (left_col > z + tol)  # neighbour is above z
                    )
                    # Approximate: each contacting cell contributes
                    # resolution * min(neighbour_height - z, oh) of contact
                    heights_above_z = np.clip(left_col - z, 0, oh)
                    lateral_contact = float(np.sum(heights_above_z)) * resolution
                    total_contact += min(lateral_contact, ow * oh)

        # ---- Right face (ow x oh, at x=x+ol) ----
        if abs(x + ol - bin_cfg.length) <= tol:
            # Flush with right wall
            total_contact += ow * oh
        else:
            right_col_idx = min(gx_end, bin_cfg.grid_l - 1)
            if right_col_idx >= 0:
                right_col = heightmap[right_col_idx, gy:gy_end]
                if right_col.size > 0:
                    heights_above_z = np.clip(right_col - z, 0, oh)
                    lateral_contact = float(np.sum(heights_above_z)) * resolution
                    total_contact += min(lateral_contact, ow * oh)

        # ---- Back face (ol x oh, at y=y) ----
        if y <= tol:
            # Flush with back wall
            total_contact += ol * oh
        else:
            back_row_idx = max(0, gy - 1)
            if back_row_idx < bin_cfg.grid_w:
                back_row = heightmap[gx:gx_end, back_row_idx]
                if back_row.size > 0:
                    heights_above_z = np.clip(back_row - z, 0, oh)
                    lateral_contact = float(np.sum(heights_above_z)) * resolution
                    total_contact += min(lateral_contact, ol * oh)

        # ---- Front face (ol x oh, at y=y+ow) ----
        if abs(y + ow - bin_cfg.width) <= tol:
            # Flush with front wall
            total_contact += ol * oh
        else:
            front_row_idx = min(gy_end, bin_cfg.grid_w - 1)
            if front_row_idx >= 0:
                front_row = heightmap[gx:gx_end, front_row_idx]
                if front_row.size > 0:
                    heights_above_z = np.clip(front_row - z, 0, oh)
                    lateral_contact = float(np.sum(heights_above_z)) * resolution
                    total_contact += min(lateral_contact, ol * oh)

        # Clamp to [0, 1]
        contact_ratio = total_contact / total_surface
        return min(max(contact_ratio, 0.0), 1.0)

    # -- Roughness delta computation ----------------------------------------

    def _compute_roughness_delta(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        heightmap: np.ndarray,
        bin_cfg,
        resolution: float,
        current_roughness: float,
    ) -> float:
        """
        Compute the change in surface roughness that would result from
        placing the box at (x, y, z) with dimensions (ol, ow, oh).

        Uses a local copy of the heightmap to simulate the placement
        without modifying the real state.

        A positive delta means roughness increased (bad). A negative delta
        means roughness decreased (good -- the box filled a valley).

        Normalized by current_roughness + 1 to prevent division by zero
        and keep the value bounded.

        Args:
            x, y, z:            Position of the box.
            ol, ow, oh:         Oriented dimensions.
            heightmap:          Current heightmap (read-only).
            bin_cfg:            Bin configuration.
            resolution:         Grid cell size.
            current_roughness:  Current surface roughness value.

        Returns:
            Normalized roughness change, roughly in [-1, 1].
        """
        gx = int(round(x / resolution))
        gy = int(round(y / resolution))
        gx_end = min(gx + int(round(ol / resolution)), bin_cfg.grid_l)
        gy_end = min(gy + int(round(ow / resolution)), bin_cfg.grid_w)

        # We only need a local copy of the affected region + some margin
        # for the roughness calculation to be accurate locally.
        margin = 2
        rx_start = max(0, gx - margin)
        ry_start = max(0, gy - margin)
        rx_end = min(bin_cfg.grid_l, gx_end + margin)
        ry_end = min(bin_cfg.grid_w, gy_end + margin)

        region = heightmap[rx_start:rx_end, ry_start:ry_end].copy()

        # Compute roughness of the region before placement
        if region.size < 2:
            return 0.0
        dx_before = np.abs(np.diff(region, axis=0))
        dy_before = np.abs(np.diff(region, axis=1))
        roughness_before = (
            (float(np.mean(dx_before)) if dx_before.size > 0 else 0.0)
            + (float(np.mean(dy_before)) if dy_before.size > 0 else 0.0)
        ) / 2.0

        # Paint the box into the local copy
        box_top = z + oh
        local_gx = gx - rx_start
        local_gy = gy - ry_start
        local_gx_end = gx_end - rx_start
        local_gy_end = gy_end - ry_start
        region[local_gx:local_gx_end, local_gy:local_gy_end] = np.maximum(
            region[local_gx:local_gx_end, local_gy:local_gy_end],
            box_top,
        )

        # Compute roughness after placement
        dx_after = np.abs(np.diff(region, axis=0))
        dy_after = np.abs(np.diff(region, axis=1))
        roughness_after = (
            (float(np.mean(dx_after)) if dx_after.size > 0 else 0.0)
            + (float(np.mean(dy_after)) if dy_after.size > 0 else 0.0)
        ) / 2.0

        # Normalized delta: positive means surface got rougher
        delta = roughness_after - roughness_before
        normalizer = current_roughness + 1.0  # avoid division by zero
        return delta / normalizer
