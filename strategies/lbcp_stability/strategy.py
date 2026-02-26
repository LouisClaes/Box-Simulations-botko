"""
LBCP Stability strategy with bounded bracing-aware relaxation.

This strategy keeps the original LBCP (load-bearable convex polygon) core:
  1) Build support contacts under the candidate footprint.
  2) Build support polygon as convex hull of those contacts.
  3) Require CoG to be inside support polygon.

Enhancements in this version:
  - Bracing-aware relaxation (bounded): if CoG is slightly outside support
    polygon but the box is laterally braced, permit a small offset.
  - Improved support-contact detection: contact is based on overlap area
    between grid cells and support patches, not only cell-center inclusion.
  - Diagnostics-friendly behavior: per-decision rejection counters and
    brace-usage metrics are captured in `get_last_diagnostics()`.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import Box, ExperimentConfig, Orientation, PlacementDecision
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

MIN_SUPPORT: float = 0.30

LBCP_SHRINK_FLOOR: float = 0.0
LBCP_SHRINK_STACKED: float = 0.15

CONTACT_TOL: float = 0.5
MIN_CELL_CONTACT_FRACTION: float = 0.05

# Bracing-aware relaxation (bounded).
BRACE_CONTACT_TOL_MULT: float = 1.0
BRACE_MIN_VERTICAL_OVERLAP_RATIO: float = 0.35
BRACE_RELAX_MIN_SUPPORT: float = 0.45
BRACE_RELAX_MIN_BRACE: float = 0.25
BRACE_RELAX_MAX_FRACTION: float = 0.12
BRACE_RELAX_MAX_CELLS: float = 2.0

# Score weights.
WEIGHT_STABILITY: float = 10.0
WEIGHT_CONTACT: float = 5.0
WEIGHT_HEIGHT: float = 2.0
WEIGHT_ROUGHNESS_DELTA: float = 1.0


# Avoid heavy optional dependency import during strategy registration.
# The internal Graham scan fallback is deterministic and sufficient here.
_ScipyConvexHull = None
_SCIPY_AVAILABLE = False


@register_strategy
class LBCPStabilityStrategy(BaseStrategy):
    """Single-bin LBCP strategy with bounded bracing relaxation."""

    name: str = "lbcp_stability"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = 1.0
        self._last_diagnostics: Dict[str, float] = {}

    def on_episode_start(self, config: ExperimentConfig) -> None:
        super().on_episode_start(config)
        self._scan_step = max(1.0, config.bin.resolution * 2.0)
        self._last_diagnostics = {}

    def get_last_diagnostics(self) -> Dict[str, float]:
        """Return counters from the most recent `decide_placement` call."""
        return dict(self._last_diagnostics)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        cfg = self.config
        bin_cfg = cfg.bin
        step = self._scan_step

        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        valid_orientations = [
            (oidx, ol, ow, oh)
            for oidx, (ol, ow, oh) in enumerate(orientations)
            if ol <= bin_cfg.length and ow <= bin_cfg.width and oh <= bin_cfg.height
        ]
        if not valid_orientations:
            self._last_diagnostics = {
                "candidates_generated": 0.0,
                "orientation_checks": 0.0,
                "accepted_candidates": 0.0,
                "best_found": 0.0,
            }
            return None

        heightmap = bin_state.heightmap
        resolution = bin_cfg.resolution
        candidates = self._generate_candidates(bin_state, step)
        current_roughness = bin_state.get_surface_roughness()

        diag: Dict[str, float] = {
            "candidates_generated": float(len(candidates)),
            "orientation_checks": 0.0,
            "rejected_bounds": 0.0,
            "rejected_height": 0.0,
            "rejected_min_support": 0.0,
            "rejected_cfg_support": 0.0,
            "rejected_stability": 0.0,
            "rejected_margin": 0.0,
            "accepted_candidates": 0.0,
            "brace_relaxed_accepts": 0.0,
            "best_found": 0.0,
            "best_score": -np.inf,
        }

        best_score = -np.inf
        best_candidate: Optional[Tuple[float, float, int]] = None

        for cx, cy in candidates:
            for oidx, ol, ow, oh in valid_orientations:
                diag["orientation_checks"] += 1.0

                if cx + ol > bin_cfg.length + 1e-6 or cy + ow > bin_cfg.width + 1e-6:
                    diag["rejected_bounds"] += 1.0
                    continue

                z = bin_state.get_height_at(cx, cy, ol, ow)
                if z + oh > bin_cfg.height + 1e-6:
                    diag["rejected_height"] += 1.0
                    continue

                if z < 0.5:
                    is_stable = True
                    support_ratio = 1.0
                    used_brace_relax = False
                else:
                    is_stable, support_ratio, used_brace_relax = self._validate_lbcp(
                        x=cx,
                        y=cy,
                        z=z,
                        ol=ol,
                        ow=ow,
                        oh=oh,
                        bin_state=bin_state,
                    )

                if support_ratio < MIN_SUPPORT:
                    diag["rejected_min_support"] += 1.0
                    continue

                if not is_stable:
                    diag["rejected_stability"] += 1.0
                    continue

                if cfg.enable_stability and z > 0.5 and support_ratio < cfg.min_support_ratio:
                    diag["rejected_cfg_support"] += 1.0
                    continue

                if not bin_state.is_margin_clear(cx, cy, ol, ow, z, oh):
                    diag["rejected_margin"] += 1.0
                    continue

                contact_ratio = self._compute_contact_ratio(
                    x=cx,
                    y=cy,
                    z=z,
                    ol=ol,
                    ow=ow,
                    heightmap=heightmap,
                    bin_cfg=bin_cfg,
                )

                if z < resolution * 0.5:
                    roughness_delta = 0.0
                else:
                    roughness_delta = self._compute_roughness_delta(
                        x=cx,
                        y=cy,
                        z=z,
                        ol=ol,
                        ow=ow,
                        oh=oh,
                        heightmap=heightmap,
                        bin_cfg=bin_cfg,
                        resolution=resolution,
                        current_roughness=current_roughness,
                    )

                height_norm = z / bin_cfg.height if bin_cfg.height > 0 else 0.0
                score = (
                    WEIGHT_STABILITY * support_ratio
                    + WEIGHT_CONTACT * contact_ratio
                    - WEIGHT_HEIGHT * height_norm
                    - WEIGHT_ROUGHNESS_DELTA * roughness_delta
                )

                diag["accepted_candidates"] += 1.0
                if used_brace_relax:
                    diag["brace_relaxed_accepts"] += 1.0

                if score > best_score:
                    best_score = score
                    best_candidate = (cx, cy, oidx)

        if best_candidate is not None:
            diag["best_found"] = 1.0
            diag["best_score"] = float(best_score)

        self._last_diagnostics = diag

        if cfg.verbose:
            print(
                "[lbcp_stability] "
                f"cand={int(diag['candidates_generated'])} "
                f"checks={int(diag['orientation_checks'])} "
                f"accepted={int(diag['accepted_candidates'])} "
                f"brace_relaxed={int(diag['brace_relaxed_accepts'])} "
                f"rej_stability={int(diag['rejected_stability'])} "
                f"rej_support={int(diag['rejected_min_support'])}"
            )

        if best_candidate is None:
            return None

        return PlacementDecision(
            x=best_candidate[0],
            y=best_candidate[1],
            orientation_idx=best_candidate[2],
        )

    # ------------------------------------------------------------------
    # LBCP validation with bounded bracing relaxation
    # ------------------------------------------------------------------

    def _validate_lbcp(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        oh: float,
        bin_state: BinState,
    ) -> Tuple[bool, float, bool]:
        """
        Return (is_stable, support_ratio, used_brace_relaxation).

        support_ratio is area-based:
            support_ratio = contact_area / footprint_area

        For strict LBCP:
            CoG must lie inside support polygon.

        Bounded relaxation:
            if CoG is outside, allow a small outside distance if:
              - support_ratio >= BRACE_RELAX_MIN_SUPPORT
              - brace_factor >= BRACE_RELAX_MIN_BRACE
              - outside_dist <= allowance
        """
        bin_cfg = bin_state.config
        heightmap = bin_state.heightmap

        support_items = self._get_support_items(x, y, z, ol, ow, bin_state)
        if not support_items:
            return False, 0.0, False

        support_patches = self._build_support_patches(
            x=x,
            y=y,
            ol=ol,
            ow=ow,
            support_items=support_items,
        )
        if not support_patches:
            return False, 0.0, False

        contact_points, support_ratio = self._compute_contact_points(
            x=x,
            y=y,
            z=z,
            ol=ol,
            ow=ow,
            heightmap=heightmap,
            bin_cfg=bin_cfg,
            support_patches=support_patches,
        )
        if support_ratio <= 0.0 or not contact_points:
            return False, support_ratio, False

        hull_pts = self._convex_hull_2d(contact_points)
        cog = (x + ol / 2.0, y + ow / 2.0)

        if self._point_in_polygon(cog, hull_pts):
            return True, support_ratio, False

        brace_factor = self._compute_bracing_factor(
            x=x,
            y=y,
            z=z,
            ol=ol,
            ow=ow,
            oh=oh,
            bin_state=bin_state,
        )
        allowance = self._compute_brace_allowance(
            ol=ol,
            ow=ow,
            support_ratio=support_ratio,
            brace_factor=brace_factor,
            resolution=bin_cfg.resolution,
        )
        if allowance <= 0.0:
            return False, support_ratio, False

        outside_dist = self._distance_to_polygon(cog, hull_pts)
        if outside_dist <= allowance + 1e-9:
            return True, support_ratio, True

        return False, support_ratio, False

    def _get_support_items(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        bin_state: BinState,
    ) -> List:
        """
        Return boxes that can support the candidate.

        Conditions:
          - top height close to candidate resting z
          - positive footprint overlap with candidate
        """
        result = []
        x_max = x + ol
        y_max = y + ow
        height_tol = max(CONTACT_TOL, bin_state.config.resolution * 0.5)

        for p in bin_state.placed_boxes:
            if abs(p.z_max - z) > height_tol:
                continue

            overlap_x = min(p.x_max, x_max) - max(p.x, x)
            overlap_y = min(p.y_max, y_max) - max(p.y, y)
            if overlap_x <= 1e-6 or overlap_y <= 1e-6:
                continue

            result.append(p)

        return result

    def _get_lbcp(self, placed_box) -> Tuple[float, float, float, float]:
        if placed_box.z < 0.5:
            shrink = LBCP_SHRINK_FLOOR
        else:
            shrink = LBCP_SHRINK_STACKED

        sx = placed_box.oriented_l * shrink / 2.0
        sy = placed_box.oriented_w * shrink / 2.0

        return (
            placed_box.x + sx,
            placed_box.y + sy,
            placed_box.x_max - sx,
            placed_box.y_max - sy,
        )

    def _build_support_patches(
        self,
        x: float,
        y: float,
        ol: float,
        ow: float,
        support_items: List,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Build support patches as intersections:
            candidate footprint intersect support top face intersect support LBCP.
        """
        x_max = x + ol
        y_max = y + ow
        patches: List[Tuple[float, float, float, float]] = []

        for p in support_items:
            lx_min, ly_min, lx_max, ly_max = self._get_lbcp(p)
            px0 = max(x, p.x, lx_min)
            py0 = max(y, p.y, ly_min)
            px1 = min(x_max, p.x_max, lx_max)
            py1 = min(y_max, p.y_max, ly_max)
            if px1 - px0 <= 1e-6 or py1 - py0 <= 1e-6:
                continue
            patches.append((px0, py0, px1, py1))

        return patches

    def _compute_contact_points(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        heightmap: np.ndarray,
        bin_cfg,
        support_patches: List[Tuple[float, float, float, float]],
    ) -> Tuple[List[Tuple[float, float]], float]:
        """
        Improved contact detection.

        A cell contributes support if:
          - cell height matches z within CONTACT_TOL
          - cell rectangle overlaps at least one support patch by
            MIN_CELL_CONTACT_FRACTION * cell_area
        """
        resolution = bin_cfg.resolution
        gx = int(round(x / resolution))
        gy = int(round(y / resolution))
        gx_end = min(gx + int(round(ol / resolution)), bin_cfg.grid_l)
        gy_end = min(gy + int(round(ow / resolution)), bin_cfg.grid_w)

        total_cells = (gx_end - gx) * (gy_end - gy)
        if total_cells <= 0:
            return [], 0.0

        cell_area = resolution * resolution
        min_overlap_area = MIN_CELL_CONTACT_FRACTION * cell_area
        tol = CONTACT_TOL

        contact_pts: List[Tuple[float, float]] = []
        contact_area = 0.0

        for ix in range(gx, gx_end):
            for iy in range(gy, gy_end):
                if abs(heightmap[ix, iy] - z) > tol:
                    continue

                cell_x0 = ix * resolution
                cell_y0 = iy * resolution
                cell_x1 = cell_x0 + resolution
                cell_y1 = cell_y0 + resolution

                best_overlap = 0.0
                for px0, py0, px1, py1 in support_patches:
                    ox = min(cell_x1, px1) - max(cell_x0, px0)
                    if ox <= 0.0:
                        continue
                    oy = min(cell_y1, py1) - max(cell_y0, py0)
                    if oy <= 0.0:
                        continue
                    area = ox * oy
                    if area > best_overlap:
                        best_overlap = area

                if best_overlap >= min_overlap_area:
                    contact_area += best_overlap
                    contact_pts.append(
                        (cell_x0 + 0.5 * resolution, cell_y0 + 0.5 * resolution)
                    )

        total_area = total_cells * cell_area
        support_ratio = contact_area / total_area if total_area > 0 else 0.0
        support_ratio = min(max(support_ratio, 0.0), 1.0)
        return contact_pts, support_ratio

    def _compute_bracing_factor(
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
        Estimate lateral bracing strength in [0, 1].

        Four brace channels are considered: left, right, front, back.
        Each side gets max score from wall brace or neighbor side-contact
        with sufficient vertical overlap.
        """
        bin_cfg = bin_state.config
        tol = max(CONTACT_TOL, bin_cfg.resolution * BRACE_CONTACT_TOL_MULT)
        min_vert_overlap = max(
            bin_cfg.resolution * 0.5,
            oh * BRACE_MIN_VERTICAL_OVERLAP_RATIO,
        )
        min_planar_overlap = bin_cfg.resolution * 0.25

        x_max = x + ol
        y_max = y + ow
        z_max = z + oh
        oh_safe = max(oh, 1e-6)

        side_score = {
            "left": 0.0,
            "right": 0.0,
            "front": 0.0,
            "back": 0.0,
        }

        # Wall brace.
        if x <= tol:
            side_score["left"] = 1.0
        if bin_cfg.length - x_max <= tol:
            side_score["right"] = 1.0
        if y <= tol:
            side_score["front"] = 1.0
        if bin_cfg.width - y_max <= tol:
            side_score["back"] = 1.0

        # Neighbor brace.
        for p in bin_state.placed_boxes:
            z_overlap = min(z_max, p.z_max) - max(z, p.z)
            if z_overlap <= min_vert_overlap:
                continue
            z_score = min(1.0, z_overlap / oh_safe)

            overlap_y = min(y_max, p.y_max) - max(y, p.y)
            overlap_x = min(x_max, p.x_max) - max(x, p.x)

            if overlap_y > min_planar_overlap:
                if abs(p.x_max - x) <= tol:
                    side_score["left"] = max(side_score["left"], z_score)
                if abs(p.x - x_max) <= tol:
                    side_score["right"] = max(side_score["right"], z_score)

            if overlap_x > min_planar_overlap:
                if abs(p.y_max - y) <= tol:
                    side_score["front"] = max(side_score["front"], z_score)
                if abs(p.y - y_max) <= tol:
                    side_score["back"] = max(side_score["back"], z_score)

        return (
            side_score["left"]
            + side_score["right"]
            + side_score["front"]
            + side_score["back"]
        ) / 4.0

    @staticmethod
    def _compute_brace_allowance(
        ol: float,
        ow: float,
        support_ratio: float,
        brace_factor: float,
        resolution: float,
    ) -> float:
        """
        Bounded CoG offset allowance used by bracing relaxation.

        allowance = geometric_cap * brace_factor * support_scale
        geometric_cap = min(min(ol, ow) * BRACE_RELAX_MAX_FRACTION,
                            resolution * BRACE_RELAX_MAX_CELLS)
        """
        if support_ratio < BRACE_RELAX_MIN_SUPPORT:
            return 0.0
        if brace_factor < BRACE_RELAX_MIN_BRACE:
            return 0.0

        support_scale = (
            (support_ratio - BRACE_RELAX_MIN_SUPPORT)
            / max(1e-9, 1.0 - BRACE_RELAX_MIN_SUPPORT)
        )
        support_scale = min(max(support_scale, 0.0), 1.0)

        geometric_cap = min(
            min(ol, ow) * BRACE_RELAX_MAX_FRACTION,
            resolution * BRACE_RELAX_MAX_CELLS,
        )
        return geometric_cap * brace_factor * support_scale

    def _distance_to_polygon(
        self,
        point: Tuple[float, float],
        polygon: List[Tuple[float, float]],
    ) -> float:
        """Shortest Euclidean distance from point to polygon boundary."""
        n = len(polygon)
        if n == 0:
            return float("inf")

        if self._point_in_polygon(point, polygon):
            return 0.0

        if n == 1:
            return float(
                np.hypot(point[0] - polygon[0][0], point[1] - polygon[0][1])
            )
        if n == 2:
            return self._distance_point_to_segment(point, polygon[0], polygon[1])

        best = float("inf")
        for i in range(n):
            a = polygon[i]
            b = polygon[(i + 1) % n]
            d = self._distance_point_to_segment(point, a, b)
            if d < best:
                best = d
        return best

    @staticmethod
    def _distance_point_to_segment(
        p: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> float:
        ax, ay = a
        bx, by = b
        px, py = p

        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        denom = abx * abx + aby * aby
        if denom <= 1e-12:
            return float(np.hypot(px - ax, py - ay))

        t = (apx * abx + apy * aby) / denom
        t = min(max(t, 0.0), 1.0)
        qx = ax + t * abx
        qy = ay + t * aby
        return float(np.hypot(px - qx, py - qy))

    # ------------------------------------------------------------------
    # Contact ratio and roughness score terms
    # ------------------------------------------------------------------

    def _compute_contact_ratio(
        self,
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        heightmap: np.ndarray,
        bin_cfg,
    ) -> float:
        if z < 0.5:
            return 1.0

        res = bin_cfg.resolution
        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), bin_cfg.grid_l)
        gy_end = min(gy + int(round(ow / res)), bin_cfg.grid_w)
        total_cells = (gx_end - gx) * (gy_end - gy)

        if total_cells <= 0:
            return 0.0

        footprint = heightmap[gx:gx_end, gy:gy_end]
        matched = int(np.sum(np.abs(footprint - z) <= CONTACT_TOL))
        return matched / total_cells

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
        gx = int(round(x / resolution))
        gy = int(round(y / resolution))
        gx_end = min(gx + int(round(ol / resolution)), bin_cfg.grid_l)
        gy_end = min(gy + int(round(ow / resolution)), bin_cfg.grid_w)

        margin = 2
        rx_start = max(0, gx - margin)
        ry_start = max(0, gy - margin)
        rx_end = min(bin_cfg.grid_l, gx_end + margin)
        ry_end = min(bin_cfg.grid_w, gy_end + margin)

        region = heightmap[rx_start:rx_end, ry_start:ry_end].copy()
        if region.size < 2:
            return 0.0

        dx_before = np.abs(np.diff(region, axis=0))
        dy_before = np.abs(np.diff(region, axis=1))
        roughness_before = (
            (float(np.mean(dx_before)) if dx_before.size > 0 else 0.0)
            + (float(np.mean(dy_before)) if dy_before.size > 0 else 0.0)
        ) / 2.0

        top = z + oh
        lx0 = gx - rx_start
        ly0 = gy - ry_start
        lx1 = gx_end - rx_start
        ly1 = gy_end - ry_start
        region[lx0:lx1, ly0:ly1] = np.maximum(region[lx0:lx1, ly0:ly1], top)

        dx_after = np.abs(np.diff(region, axis=0))
        dy_after = np.abs(np.diff(region, axis=1))
        roughness_after = (
            (float(np.mean(dx_after)) if dx_after.size > 0 else 0.0)
            + (float(np.mean(dy_after)) if dy_after.size > 0 else 0.0)
        ) / 2.0

        delta = roughness_after - roughness_before
        return delta / (current_roughness + 1.0)

    # ------------------------------------------------------------------
    # Convex hull and point-in-polygon
    # ------------------------------------------------------------------

    def _convex_hull_2d(
        self,
        points: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        if len(points) <= 2:
            return list(points)

        pts_arr = np.array(points, dtype=float)
        if _SCIPY_AVAILABLE:
            try:
                hull = _ScipyConvexHull(pts_arr)
                return [points[i] for i in hull.vertices]
            except Exception:
                pass

        return self._graham_scan(points)

    @staticmethod
    def _graham_scan(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        pts = sorted(set(points))
        if len(pts) <= 2:
            return pts

        def cross(o, a, b) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: List[Tuple[float, float]] = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper: List[Tuple[float, float]] = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Remove duplicated endpoints.
        return lower[:-1] + upper[:-1]

    def _point_in_polygon(
        self,
        point: Tuple[float, float],
        polygon: List[Tuple[float, float]],
    ) -> bool:
        n = len(polygon)
        if n == 0:
            return False

        px, py = point

        if n == 1:
            return abs(px - polygon[0][0]) < 1e-9 and abs(py - polygon[0][1]) < 1e-9
        if n == 2:
            return self._point_on_segment(point, polygon[0], polygon[1])

        for i in range(n):
            ax, ay = polygon[i]
            bx, by = polygon[(i + 1) % n]
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            if cross < -1e-9:
                return False
        return True

    @staticmethod
    def _point_on_segment(
        point: Tuple[float, float],
        seg_a: Tuple[float, float],
        seg_b: Tuple[float, float],
    ) -> bool:
        px, py = point
        ax, ay = seg_a
        bx, by = seg_b

        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if abs(cross) > 1e-9:
            return False

        return (
            min(ax, bx) - 1e-9 <= px <= max(ax, bx) + 1e-9
            and min(ay, by) - 1e-9 <= py <= max(ay, by) + 1e-9
        )

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        bin_state: BinState,
        step: float,
    ) -> List[Tuple[float, float]]:
        bin_cfg = bin_state.config
        seen: Set[Tuple[float, float]] = set()
        candidates: List[Tuple[float, float]] = []

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

        for p in bin_state.placed_boxes:
            for pt in [
                (p.x, p.y),
                (p.x_max, p.y),
                (p.x, p.y_max),
                (p.x_max, p.y_max),
            ]:
                if (
                    pt not in seen
                    and 0 <= pt[0] <= bin_cfg.length
                    and 0 <= pt[1] <= bin_cfg.width
                ):
                    seen.add(pt)
                    candidates.append(pt)

        candidates.sort()
        return candidates
