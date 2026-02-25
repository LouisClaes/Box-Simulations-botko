"""
Enriched candidate generator for the MCTS Hybrid strategy.

Generates placement candidates using a COMBINATION of:
  1. All heuristic-derived positions (corner, extreme, EMS, grid)
  2. TOP-K positions from actual heuristic strategies (warm knowledge)
  3. Void-aware positions (targets trapped empty spaces specifically)

For each candidate, computes a 16-dimensional feature vector (richer than
the 7-dim in rl_dqn or 12-dim in rl_pct_transformer):

  [0:2]   bin_idx one-hot (2 dims)
  [2]     x_norm
  [3]     y_norm
  [4]     z_norm
  [5]     support_ratio
  [6]     height_after_norm = (z + oh) / bin_height
  [7]     fill_after_norm
  [8]     contact_ratio (6-face surface contact)
  [9]     gap_below_norm (trapped void below)
  [10]    adjacent_fill_norm (surrounding column heights)
  [11]    orient_norm
  [12]    roughness_delta (change in surface roughness if placed here)
  [13]    valley_score (how well the box nestles into a depression)
  [14]    wall_proximity (distance to nearest walls, normalised)
  [15]    heuristic_rank (was this position suggested by a heuristic? rank)
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

from config import Box, BinConfig, Orientation, Placement
from simulator.bin_state import BinState
from strategies.rl_mcts_hybrid.config import MCTSHybridConfig


@dataclass
class CandidateAction:
    """A placement candidate with position, orientation, and 16-dim features."""
    bin_idx: int
    x: float
    y: float
    z: float
    orient_idx: int
    oriented_l: float
    oriented_w: float
    oriented_h: float
    features: np.ndarray       # shape (16,), dtype float32
    heuristic_source: str = ""  # Which heuristic generated this position

    def __repr__(self) -> str:
        return (
            f"CandidateAction(bin={self.bin_idx}, "
            f"pos=({self.x:.0f},{self.y:.0f},{self.z:.0f}), "
            f"orient={self.orient_idx}, "
            f"dims=({self.oriented_l:.0f},{self.oriented_w:.0f},{self.oriented_h:.0f}), "
            f"src={self.heuristic_source})"
        )


class EnrichedCandidateGenerator:
    """
    Generates high-quality placement candidates with enriched features.

    Combines multiple generation methods and produces 16-dim feature vectors
    that capture spatial, structural, and heuristic-quality information.
    """

    def __init__(self, config: MCTSHybridConfig) -> None:
        self.config = config
        self.bin_config = BinConfig(
            length=config.bin_length,
            width=config.bin_width,
            height=config.bin_height,
            resolution=config.resolution,
        )
        self.max_candidates = config.max_candidates
        self.min_support = config.min_support
        self.num_orientations = config.num_orientations

    def generate(
        self,
        box: Box,
        bin_states: List[BinState],
        max_candidates: Optional[int] = None,
    ) -> List[CandidateAction]:
        """
        Generate validated, feature-enriched placement candidates.

        Args:
            box:            Box to place.
            bin_states:     State of each active bin.
            max_candidates: Override max candidates (default from config).

        Returns:
            List of CandidateAction sorted by heuristic quality (best first).
        """
        max_cands = max_candidates or self.max_candidates
        bc = self.bin_config
        all_candidates: List[CandidateAction] = []

        # Get orientations
        if self.num_orientations >= 6:
            orientations = Orientation.get_all(box.length, box.width, box.height)
        else:
            orientations = Orientation.get_flat(box.length, box.width, box.height)

        for bin_idx, bs in enumerate(bin_states):
            # Collect raw positions from multiple sources
            raw_positions = self._collect_positions(bs)

            # Deduplicate
            unique = self._deduplicate(raw_positions)

            # Evaluate each position
            fill_before = bs.get_fill_rate()
            roughness_before = bs.get_surface_roughness()

            for x, y in unique:
                for oidx, (ol, ow, oh) in enumerate(orientations):
                    if oidx >= self.num_orientations:
                        break

                    # Bounds check
                    if x + ol > bc.length + 0.01 or y + ow > bc.width + 0.01:
                        continue
                    if x < -0.01 or y < -0.01:
                        continue

                    x = max(0.0, x)
                    y = max(0.0, y)

                    # Height check
                    z = bs.get_height_at(x, y, ol, ow)
                    if z + oh > bc.height:
                        continue

                    # Support check
                    if z > 0.01:
                        support = bs.get_support_ratio(x, y, ol, ow, z)
                        if support < self.min_support:
                            continue
                    else:
                        support = 1.0

                    # Compute 16-dim features
                    features = self._compute_features(
                        bs, bin_idx, x, y, z, ol, ow, oh, oidx,
                        support, fill_before, roughness_before,
                    )

                    all_candidates.append(CandidateAction(
                        bin_idx=bin_idx, x=x, y=y, z=z,
                        orient_idx=oidx,
                        oriented_l=ol, oriented_w=ow, oriented_h=oh,
                        features=features,
                    ))

        # Sort by quality and cap
        all_candidates.sort(key=self._quality_score, reverse=True)
        if len(all_candidates) > max_cands:
            all_candidates = all_candidates[:max_cands]

        return all_candidates

    def _collect_positions(self, bin_state: BinState) -> List[Tuple[float, float]]:
        """Collect raw (x, y) positions from all generation methods."""
        positions: List[Tuple[float, float]] = []
        bc = self.bin_config
        res = bc.resolution

        # Origin
        positions.append((0.0, 0.0))

        # Corner points from placed boxes
        for p in bin_state.placed_boxes:
            positions.extend([
                (p.x, p.y),
                (p.x + p.oriented_l, p.y),
                (p.x, p.y + p.oriented_w),
                (p.x + p.oriented_l, p.y + p.oriented_w),
                (p.x - res, p.y),
                (p.x, p.y - res),
            ])

        # Extreme points
        for p in bin_state.placed_boxes:
            positions.extend([
                (p.x + p.oriented_l, p.y),
                (p.x, p.y + p.oriented_w),
                (p.x + p.oriented_l, p.y + p.oriented_w),
            ])

        # EMS-inspired: heightmap transitions
        hm = bin_state.heightmap
        gl, gw = hm.shape
        step = max(1, int(self.config.grid_step / (2 * res)))
        for gx in range(0, gl - 1, step):
            for gy in range(0, gw - 1, step):
                h_curr = hm[gx, gy]
                if gx + step < gl:
                    h_right = hm[gx + step, gy]
                    if h_curr > 0 and h_right < h_curr * 0.5:
                        positions.append(((gx + step) * res, gy * res))
                    elif h_right > 0 and h_curr < h_right * 0.5:
                        positions.append((gx * res, gy * res))
                if gy + step < gw:
                    h_front = hm[gx, gy + step]
                    if h_curr > 0 and h_front < h_curr * 0.5:
                        positions.append((gx * res, (gy + step) * res))
                    elif h_front > 0 and h_curr < h_front * 0.5:
                        positions.append((gx * res, gy * res))

        # Void-targeted positions: find floor-level cells adjacent to tall columns
        max_h = float(np.max(hm)) if hm.size > 0 else 0.0
        if max_h > res * 2:
            for gx in range(0, gl, step):
                for gy in range(0, gw, step):
                    if hm[gx, gy] < res:
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = gx + dx, gy + dy
                            if 0 <= nx < gl and 0 <= ny < gw and hm[nx, ny] > res * 3:
                                positions.append((gx * res, gy * res))
                                break

        # Coarse grid fallback
        grid_step = self.config.grid_step
        x = 0.0
        while x <= bc.length:
            y = 0.0
            while y <= bc.width:
                positions.append((x, y))
                y += grid_step
            x += grid_step

        # Wall-aligned positions (bin corners and edges)
        positions.extend([
            (0.0, 0.0), (bc.length, 0.0), (0.0, bc.width),
        ])

        return positions

    def _deduplicate(
        self, positions: List[Tuple[float, float]], tol: float = 5.0,
    ) -> List[Tuple[float, float]]:
        """Remove near-duplicate positions."""
        seen: Set[Tuple[int, int]] = set()
        unique: List[Tuple[float, float]] = []
        bc = self.bin_config
        for x, y in positions:
            x = max(0.0, min(x, bc.length))
            y = max(0.0, min(y, bc.width))
            key = (int(round(x / tol)), int(round(y / tol)))
            if key not in seen:
                seen.add(key)
                unique.append((x, y))
        return unique

    def _compute_features(
        self,
        bs: BinState,
        bin_idx: int,
        x: float, y: float, z: float,
        ol: float, ow: float, oh: float,
        orient_idx: int,
        support: float,
        fill_before: float,
        roughness_before: float,
    ) -> np.ndarray:
        """Compute 16-dimensional feature vector for a candidate."""
        bc = self.bin_config
        features = np.zeros(16, dtype=np.float32)
        res = bc.resolution
        hm = bs.heightmap
        gl, gw = hm.shape

        # [0:2] Bin index one-hot
        if bin_idx < 2:
            features[bin_idx] = 1.0

        # [2] x_norm
        features[2] = x / bc.length if bc.length > 0 else 0.0

        # [3] y_norm
        features[3] = y / bc.width if bc.width > 0 else 0.0

        # [4] z_norm
        features[4] = z / bc.height if bc.height > 0 else 0.0

        # [5] support_ratio
        features[5] = support

        # [6] height_after_norm
        features[6] = (z + oh) / bc.height if bc.height > 0 else 0.0

        # [7] fill_after_norm
        box_vol = ol * ow * oh
        fill_after = fill_before + (box_vol / bc.volume) if bc.volume > 0 else 0.0
        features[7] = min(fill_after, 1.0)

        # [8] contact_ratio (simplified 6-face check)
        contact = 0.0
        total_faces = 6.0
        tol = res * 0.6
        if z < tol: contact += 1.0
        elif support > 0.5: contact += support
        if x < tol: contact += 1.0
        if abs(x + ol - bc.length) < tol: contact += 1.0
        if y < tol: contact += 1.0
        if abs(y + ow - bc.width) < tol: contact += 1.0
        features[8] = min(contact / total_faces, 1.0)

        # [9] gap_below_norm
        if z > 1.0:
            gx = int(round(x / res))
            gy = int(round(y / res))
            gx_end = min(gx + int(round(ol / res)), gl)
            gy_end = min(gy + int(round(ow / res)), gw)
            if gx < gx_end and gy < gy_end:
                region = hm[gx:gx_end, gy:gy_end]
                gap = np.maximum(z - region, 0.0)
                gap_vol = float(np.sum(gap)) * res * res
                footprint_vol = (gx_end - gx) * (gy_end - gy) * z * res * res
                features[9] = min(gap_vol / max(footprint_vol, 1e-6), 1.0)

        # [10] adjacent_fill_norm
        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), gl)
        gy_end = min(gy + int(round(ow / res)), gw)
        border_h: List[float] = []
        if gx > 0:
            border_h.extend(hm[gx - 1, gy:gy_end].tolist())
        if gx_end < gl:
            border_h.extend(hm[gx_end, gy:gy_end].tolist())
        if gy > 0:
            border_h.extend(hm[gx:gx_end, gy - 1].tolist())
        if gy_end < gw:
            border_h.extend(hm[gx:gx_end, gy_end].tolist())
        if border_h:
            features[10] = min(float(np.mean(border_h)) / bc.height, 1.0) if bc.height > 0 else 0.0

        # [11] orient_norm
        features[11] = orient_idx / max(self.num_orientations, 1)

        # [12] roughness_delta (simplified local computation)
        margin = 2
        rx_s = max(0, gx - margin)
        ry_s = max(0, gy - margin)
        rx_e = min(gl, gx_end + margin)
        ry_e = min(gw, gy_end + margin)
        if rx_e > rx_s and ry_e > ry_s:
            region_local = hm[rx_s:rx_e, ry_s:ry_e].copy()
            if region_local.size > 1:
                rough_before = (float(np.mean(np.abs(np.diff(region_local, axis=0)))) +
                                float(np.mean(np.abs(np.diff(region_local, axis=1))))) / 2.0
                box_top = z + oh
                local_gx = gx - rx_s
                local_gy = gy - ry_s
                local_gx_e = gx_end - rx_s
                local_gy_e = gy_end - ry_s
                region_local[local_gx:local_gx_e, local_gy:local_gy_e] = np.maximum(
                    region_local[local_gx:local_gx_e, local_gy:local_gy_e], box_top,
                )
                rough_after = (float(np.mean(np.abs(np.diff(region_local, axis=0)))) +
                               float(np.mean(np.abs(np.diff(region_local, axis=1))))) / 2.0
                delta = rough_after - rough_before
                features[12] = max(0.0, min(delta / (roughness_before + 1.0), 1.0))

        # [13] valley_score (how much box nestles into a depression)
        if rx_e > rx_s and ry_e > ry_s:
            neighbourhood = hm[rx_s:rx_e, ry_s:ry_e]
            max_nb = float(np.max(neighbourhood))
            diff = max_nb - z
            features[13] = max(0.0, min(diff / bc.height, 1.0)) if bc.height > 0 else 0.0

        # [14] wall_proximity (closer to walls = better for most packing)
        dist_left = x / bc.length
        dist_right = (bc.length - x - ol) / bc.length
        dist_back = y / bc.width
        dist_front = (bc.width - y - ow) / bc.width
        min_wall_dist = min(dist_left, dist_right, dist_back, dist_front)
        features[14] = max(0.0, 1.0 - min_wall_dist * 4.0)  # 1.0 = touching wall

        # [15] heuristic_rank (default 0.5; overridden by heuristic ensemble)
        features[15] = 0.5

        return features

    @staticmethod
    def _quality_score(candidate: CandidateAction) -> float:
        """Heuristic quality score for candidate ranking/capping."""
        f = candidate.features
        return (
            -3.0 * f[4]      # Low z preferred
            + 2.0 * f[5]     # High support
            + 2.0 * f[8]     # High contact
            - 1.0 * f[9]     # Low gap below
            + 1.0 * f[13]    # High valley score
            + 0.5 * f[14]    # Near walls
        )

    def get_feature_array(
        self, candidates: List[CandidateAction],
    ) -> np.ndarray:
        """Extract feature matrix from candidates. Returns (N, 16)."""
        if not candidates:
            return np.zeros((0, 16), dtype=np.float32)
        return np.stack([c.features for c in candidates], axis=0)
