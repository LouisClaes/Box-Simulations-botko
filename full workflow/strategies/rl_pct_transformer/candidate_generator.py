"""
Candidate placement generator for the PCT Transformer strategy.

This is the critical component that determines the action space at each step.
Instead of a fixed grid of (x, y, orient) actions, we generate a variable-size
set of PLACEMENT CANDIDATES, each fully characterised by position, orientation,
and contextual features.  The Transformer then attends over these candidates
and selects the best one via a pointer mechanism.

Candidate Generation Methods (all used, then deduplicated):
  1. Corner Points (CP): project corners outward from each placed box
  2. Extreme Points (EP): scan heightmap transitions for step changes
  3. Floor Scan: coarse grid on the empty floor (z=0)
  4. Residual Spaces: gaps between placed boxes and bin walls

For each candidate position (x, y, bin_idx):
  - Try all allowed orientations
  - Compute z from heightmap
  - Validate: bounds check, height limit, support >= min_support
  - Extract 12-dimensional context feature vector

References:
    Crainic et al. (2008): Extreme point heuristics for 3D BPP
    Zhao et al. (ICLR 2022): Packing Configuration Trees (leaf node candidates)
    Martello et al. (2000): Corner point generation
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


# ─────────────────────────────────────────────────────────────────────────────
# CandidateAction dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateAction:
    """
    A single placement candidate with position, orientation, and features.

    The features vector is the 12-dimensional input to the CandidateEncoder:
        [0:2]  bin_idx one-hot (2 dims for 2 bins)
        [2]    x_norm           = x / bin_length
        [3]    y_norm           = y / bin_width
        [4]    z_norm           = z / bin_height
        [5]    support_ratio    = fraction of base that is supported [0, 1]
        [6]    height_after_norm = (z + oriented_h) / bin_height
        [7]    fill_after_norm  = estimated fill rate after placement [0, 1]
        [8]    contact_ratio    = fraction of box surface touching walls/boxes [0, 1]
        [9]    gap_below_norm   = total empty volume below box / box volume [0, 1]
        [10]   adjacent_fill_norm = fill rate of immediately adjacent columns [0, 1]
        [11]   orient_norm      = orientation_idx / num_orientations
    """

    bin_idx: int
    x: float
    y: float
    z: float
    orient_idx: int
    oriented_l: float
    oriented_w: float
    oriented_h: float
    features: np.ndarray       # shape (12,), dtype float32

    def __repr__(self) -> str:
        return (
            f"CandidateAction(bin={self.bin_idx}, pos=({self.x:.0f},{self.y:.0f},{self.z:.0f}), "
            f"orient={self.orient_idx}, dims=({self.oriented_l:.0f},{self.oriented_w:.0f},{self.oriented_h:.0f}))"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CandidateGenerator
# ─────────────────────────────────────────────────────────────────────────────

class CandidateGenerator:
    """
    Generates placement candidates for a box across one or more bins.

    Combines four generation methods (corner points, extreme points, floor scan,
    residual spaces), deduplicates nearby positions, validates each candidate
    against physical constraints, and computes context features.

    Usage:
        gen = CandidateGenerator(bin_config, min_support=0.30, num_orientations=2)
        candidates = gen.generate(box, bin_states, max_candidates=200)

    Typical output: 30-150 candidates per step, depending on bin occupancy.
    Empty bins produce ~20-50 candidates (mostly floor scan).
    Occupied bins produce more candidates from corner/extreme points.
    """

    def __init__(
        self,
        bin_config: BinConfig,
        min_support: float = 0.30,
        num_orientations: int = 2,
        floor_scan_step: float = 50.0,
        dedup_tolerance: float = 5.0,
    ) -> None:
        """
        Args:
            bin_config:       Pallet/bin dimensions and grid resolution.
            min_support:      Minimum base support ratio for a valid candidate.
            num_orientations: 2 = flat-only, 6 = all axis-aligned rotations.
            floor_scan_step:  Grid step (mm) for floor-level candidate scan.
            dedup_tolerance:  Merge positions within this distance (mm).
        """
        self.bin_config = bin_config
        self.min_support = min_support
        self.num_orientations = num_orientations
        self.floor_scan_step = floor_scan_step
        self.dedup_tolerance = dedup_tolerance

    def generate(
        self,
        box: Box,
        bin_states: List[BinState],
        max_candidates: int = 200,
    ) -> List[CandidateAction]:
        """
        Generate all valid placement candidates for the given box.

        Combines candidates from all bins and all generation methods,
        deduplicates, validates, extracts features, and caps at max_candidates
        (sorted by a heuristic quality score to keep the best ones).

        Args:
            box:            The box to place.
            bin_states:     List of BinState for each active bin.
            max_candidates: Maximum number of candidates to return.

        Returns:
            List of CandidateAction, sorted by heuristic quality (best first).
            Empty list if no valid placement exists.
        """
        all_candidates: List[CandidateAction] = []

        for bin_idx, bs in enumerate(bin_states):
            # Step 1: collect raw positions (x, y) from all methods
            raw_positions = self._collect_raw_positions(box, bs)

            # Step 2: deduplicate nearby positions
            unique_positions = self._deduplicate(raw_positions)

            # Step 3: for each position, try each orientation
            candidates = self._evaluate_positions(
                box, bs, bin_idx, unique_positions,
            )
            all_candidates.extend(candidates)

        # Step 4: sort by heuristic quality and cap
        all_candidates.sort(key=self._quality_score, reverse=True)
        if len(all_candidates) > max_candidates:
            all_candidates = all_candidates[:max_candidates]

        return all_candidates

    # ── Position collection methods ──────────────────────────────────────

    def _collect_raw_positions(
        self, box: Box, bin_state: BinState,
    ) -> List[Tuple[float, float]]:
        """Collect raw (x, y) positions from all generation methods."""
        positions: List[Tuple[float, float]] = []

        # Always include origin
        positions.append((0.0, 0.0))

        # Method 1: Corner Points from placed boxes
        positions.extend(self._corner_points(bin_state))

        # Method 2: Extreme Points from heightmap transitions
        positions.extend(self._extreme_points(bin_state))

        # Method 3: Floor scan (coarse grid at z=0)
        positions.extend(self._floor_scan())

        # Method 4: Residual spaces (gaps between boxes and walls)
        positions.extend(self._residual_spaces(bin_state))

        return positions

    def _corner_points(self, bin_state: BinState) -> List[Tuple[float, float]]:
        """
        Generate corner points from each placed box.

        For each placed box, project outward from its corners:
          - Right edge:  (x + oriented_l, y)
          - Front edge:  (x, y + oriented_w)
          - Diagonal:    (x + oriented_l, y + oriented_w)
          - Top-origin:  (x, y) — for stacking on top

        These are the natural positions where a new box can sit flush
        against an existing box, maximising surface contact.
        """
        points: List[Tuple[float, float]] = []
        for p in bin_state.placed_boxes:
            # Right of box
            points.append((p.x + p.oriented_l, p.y))
            # Front of box
            points.append((p.x, p.y + p.oriented_w))
            # Diagonal corner
            points.append((p.x + p.oriented_l, p.y + p.oriented_w))
            # On top, same origin
            points.append((p.x, p.y))
            # Right-front on top
            points.append((p.x + p.oriented_l, p.y))
            # Front-left on top
            points.append((p.x, p.y + p.oriented_w))
        return points

    def _extreme_points(self, bin_state: BinState) -> List[Tuple[float, float]]:
        """
        Generate extreme points from heightmap transitions.

        Scan the heightmap for height discontinuities (step changes)
        and generate candidate points at these transitions.  These
        positions often represent ledges or shelves where boxes can rest.
        """
        points: List[Tuple[float, float]] = []
        hm = bin_state.heightmap
        res = self.bin_config.resolution
        gl, gw = hm.shape

        # Scan horizontal transitions (along x-axis)
        for gy in range(gw):
            for gx in range(1, gl):
                if abs(hm[gx, gy] - hm[gx - 1, gy]) > res:
                    # Height step detected — candidate at the lower side
                    if hm[gx, gy] < hm[gx - 1, gy]:
                        points.append((gx * res, gy * res))
                    else:
                        points.append(((gx - 1) * res, gy * res))

        # Scan vertical transitions (along y-axis)
        for gx in range(gl):
            for gy in range(1, gw):
                if abs(hm[gx, gy] - hm[gx, gy - 1]) > res:
                    if hm[gx, gy] < hm[gx, gy - 1]:
                        points.append((gx * res, gy * res))
                    else:
                        points.append((gx * res, (gy - 1) * res))

        return points

    def _floor_scan(self) -> List[Tuple[float, float]]:
        """
        Generate a coarse grid of positions on the floor.

        Ensures the agent can always consider floor-level placements
        even if no extreme or corner points exist at z=0.
        """
        points: List[Tuple[float, float]] = []
        step = self.floor_scan_step
        x = 0.0
        while x < self.bin_config.length:
            y = 0.0
            while y < self.bin_config.width:
                points.append((x, y))
                y += step
            x += step
        return points

    def _residual_spaces(self, bin_state: BinState) -> List[Tuple[float, float]]:
        """
        Generate candidates from residual spaces between boxes and bin walls.

        Identifies gaps where a box could fit between:
          - A placed box and the right wall (x-axis)
          - A placed box and the front wall (y-axis)
          - Two adjacent placed boxes
        """
        points: List[Tuple[float, float]] = []
        bc = self.bin_config

        for p in bin_state.placed_boxes:
            # Gap between box right edge and bin right wall
            gap_x = bc.length - (p.x + p.oriented_l)
            if gap_x > 0:
                points.append((p.x + p.oriented_l, p.y))

            # Gap between box front edge and bin front wall
            gap_y = bc.width - (p.y + p.oriented_w)
            if gap_y > 0:
                points.append((p.x, p.y + p.oriented_w))

            # From bin walls looking inward
            points.append((bc.length - p.oriented_l, p.y))
            points.append((p.x, bc.width - p.oriented_w))

        # Corners of the bin
        points.append((0.0, 0.0))
        points.append((bc.length, 0.0))
        points.append((0.0, bc.width))

        return points

    # ── Deduplication ────────────────────────────────────────────────────

    def _deduplicate(
        self, positions: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Remove near-duplicate positions within dedup_tolerance.

        Uses a simple grid-based bucketing approach for O(n) performance.
        """
        if not positions:
            return []

        tol = self.dedup_tolerance
        seen: Set[Tuple[int, int]] = set()
        unique: List[Tuple[float, float]] = []

        for x, y in positions:
            # Clamp to bin bounds
            x = max(0.0, min(x, self.bin_config.length))
            y = max(0.0, min(y, self.bin_config.width))

            # Bucket key
            key = (int(round(x / tol)), int(round(y / tol)))
            if key not in seen:
                seen.add(key)
                unique.append((x, y))

        return unique

    # ── Position evaluation ──────────────────────────────────────────────

    def _evaluate_positions(
        self,
        box: Box,
        bin_state: BinState,
        bin_idx: int,
        positions: List[Tuple[float, float]],
    ) -> List[CandidateAction]:
        """
        For each (x, y) position, try all orientations and return valid candidates.

        A candidate is valid if:
          - The oriented box fits within bin bounds
          - z + oriented_h <= bin_height (height limit)
          - Support ratio >= min_support (anti-float)
        """
        bc = self.bin_config
        candidates: List[CandidateAction] = []

        # Get orientations
        if self.num_orientations >= 6:
            orientations = Orientation.get_all(box.length, box.width, box.height)
        else:
            orientations = Orientation.get_flat(box.length, box.width, box.height)

        # Pre-compute bin fill before placement
        fill_before = bin_state.get_fill_rate()
        bin_vol = bc.volume

        for x, y in positions:
            for orient_idx, (ol, ow, oh) in enumerate(orientations):
                if orient_idx >= self.num_orientations:
                    break

                # Bounds check
                if x + ol > bc.length + 0.01:
                    continue
                if y + ow > bc.width + 0.01:
                    continue

                # Compute resting z
                z = bin_state.get_height_at(x, y, ol, ow)

                # Height limit check
                if z + oh > bc.height:
                    continue

                # Support check
                support = bin_state.get_support_ratio(x, y, ol, ow, z)
                if z > 0.01 and support < self.min_support:
                    continue

                # Compute features
                features = self._compute_features(
                    box, bin_state, bin_idx, x, y, z, ol, ow, oh,
                    orient_idx, support, fill_before, bin_vol,
                )

                candidates.append(CandidateAction(
                    bin_idx=bin_idx,
                    x=x, y=y, z=z,
                    orient_idx=orient_idx,
                    oriented_l=ol, oriented_w=ow, oriented_h=oh,
                    features=features,
                ))

        return candidates

    # ── Feature computation ──────────────────────────────────────────────

    def _compute_features(
        self,
        box: Box,
        bin_state: BinState,
        bin_idx: int,
        x: float, y: float, z: float,
        ol: float, ow: float, oh: float,
        orient_idx: int,
        support: float,
        fill_before: float,
        bin_vol: float,
    ) -> np.ndarray:
        """
        Compute the 12-dimensional feature vector for a candidate.

        Features are normalised to [0, 1] where possible for stable training.
        """
        bc = self.bin_config
        features = np.zeros(12, dtype=np.float32)

        # [0:2] Bin index one-hot (2 bins)
        if bin_idx < 2:
            features[bin_idx] = 1.0

        # [2] x_norm
        features[2] = x / bc.length if bc.length > 0 else 0.0

        # [3] y_norm
        features[3] = y / bc.width if bc.width > 0 else 0.0

        # [4] z_norm
        features[4] = z / bc.height if bc.height > 0 else 0.0

        # [5] support_ratio (already computed)
        features[5] = support

        # [6] height_after_norm = (z + oh) / bin_height
        features[6] = (z + oh) / bc.height if bc.height > 0 else 0.0

        # [7] fill_after_norm (estimated fill rate after placing this box)
        box_vol = ol * ow * oh
        fill_after = fill_before + (box_vol / bin_vol) if bin_vol > 0 else 0.0
        features[7] = min(fill_after, 1.0)

        # [8] contact_ratio: fraction of box surfaces touching walls or other boxes
        features[8] = self._compute_contact_ratio(
            bin_state, x, y, z, ol, ow, oh,
        )

        # [9] gap_below_norm: ratio of empty space below box to box volume
        features[9] = self._compute_gap_below(
            bin_state, x, y, z, ol, ow,
        )

        # [10] adjacent_fill_norm: average fill of surrounding columns
        features[10] = self._compute_adjacent_fill(
            bin_state, x, y, ol, ow,
        )

        # [11] orient_norm
        features[11] = orient_idx / max(self.num_orientations, 1)

        return features

    def _compute_contact_ratio(
        self,
        bin_state: BinState,
        x: float, y: float, z: float,
        ol: float, ow: float, oh: float,
    ) -> float:
        """
        Fraction of box surface area touching walls or other boxes.

        Checks each of the 6 faces:
          - Bottom: counted if z > 0 and support exists
          - Left/Right/Front/Back walls
          - Adjacent boxes on each face
        """
        bc = self.bin_config
        total_faces = 0
        touching = 0
        tol = bc.resolution * 0.6

        # Bottom face
        total_faces += 1
        if z < tol:
            touching += 1  # On floor
        elif bin_state.get_support_ratio(x, y, ol, ow, z) > 0.5:
            touching += 1

        # Left wall (x = 0)
        total_faces += 1
        if x < tol:
            touching += 1

        # Right wall (x + ol = bin_length)
        total_faces += 1
        if abs(x + ol - bc.length) < tol:
            touching += 1

        # Back wall (y = 0)
        total_faces += 1
        if y < tol:
            touching += 1

        # Front wall (y + ow = bin_width)
        total_faces += 1
        if abs(y + ow - bc.width) < tol:
            touching += 1

        # Top (always open when placing)
        total_faces += 1

        # Check adjacency with placed boxes for side faces
        for p in bin_state.placed_boxes:
            # Check if boxes overlap vertically
            z_overlap = (
                z < p.z + p.oriented_h and
                z + oh > p.z
            )
            if not z_overlap:
                continue

            # Right face of new box touches left face of placed box
            if abs(x + ol - p.x) < tol:
                y_overlap = (y < p.y + p.oriented_w and y + ow > p.y)
                if y_overlap:
                    touching += 0.5

            # Left face of new box touches right face of placed box
            if abs(x - (p.x + p.oriented_l)) < tol:
                y_overlap = (y < p.y + p.oriented_w and y + ow > p.y)
                if y_overlap:
                    touching += 0.5

            # Front face of new box touches back face of placed box
            if abs(y + ow - p.y) < tol:
                x_overlap = (x < p.x + p.oriented_l and x + ol > p.x)
                if x_overlap:
                    touching += 0.5

            # Back face of new box touches front face of placed box
            if abs(y - (p.y + p.oriented_w)) < tol:
                x_overlap = (x < p.x + p.oriented_l and x + ol > p.x)
                if x_overlap:
                    touching += 0.5

        return min(touching / total_faces, 1.0)

    def _compute_gap_below(
        self,
        bin_state: BinState,
        x: float, y: float, z: float,
        ol: float, ow: float,
    ) -> float:
        """
        Compute the ratio of empty space below the box footprint.

        gap_volume = integral of (z - heightmap[gx,gy]) over the footprint
        Returns gap_volume / (footprint_area * z), normalised to [0, 1].
        """
        if z < 1.0:
            return 0.0  # On floor, no gap

        res = self.bin_config.resolution
        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), self.bin_config.grid_l)
        gy_end = min(gy + int(round(ow / res)), self.bin_config.grid_w)

        if gx >= gx_end or gy >= gy_end:
            return 0.0

        region = bin_state.heightmap[gx:gx_end, gy:gy_end]
        gap = np.maximum(z - region, 0.0)
        gap_volume = float(np.sum(gap)) * res * res  # Convert grid area to real area
        footprint_volume = (gx_end - gx) * (gy_end - gy) * z * res * res

        if footprint_volume < 1e-6:
            return 0.0

        return min(float(gap_volume / footprint_volume), 1.0)

    def _compute_adjacent_fill(
        self,
        bin_state: BinState,
        x: float, y: float,
        ol: float, ow: float,
    ) -> float:
        """
        Average normalised height in columns adjacent to the box footprint.

        Looks at a 1-cell border around the box footprint and averages
        the heightmap values.  This indicates how well the box "fits in"
        with its surroundings.
        """
        bc = self.bin_config
        res = bc.resolution
        hm = bin_state.heightmap
        gl, gw = hm.shape

        gx = int(round(x / res))
        gy = int(round(y / res))
        gx_end = min(gx + int(round(ol / res)), gl)
        gy_end = min(gy + int(round(ow / res)), gw)

        # Collect border cell heights
        border_heights: List[float] = []

        # Left border (gx - 1)
        if gx > 0:
            for j in range(gy, gy_end):
                border_heights.append(float(hm[gx - 1, j]))

        # Right border (gx_end)
        if gx_end < gl:
            for j in range(gy, gy_end):
                border_heights.append(float(hm[gx_end, j]))

        # Back border (gy - 1)
        if gy > 0:
            for i in range(gx, gx_end):
                border_heights.append(float(hm[i, gy - 1]))

        # Front border (gy_end)
        if gy_end < gw:
            for i in range(gx, gx_end):
                border_heights.append(float(hm[i, gy_end]))

        if not border_heights:
            return 0.0

        avg_height = np.mean(border_heights)
        return min(float(avg_height / bc.height), 1.0) if bc.height > 0 else 0.0

    # ── Quality scoring ──────────────────────────────────────────────────

    @staticmethod
    def _quality_score(candidate: CandidateAction) -> float:
        """
        Heuristic quality score for sorting candidates.

        Used only for capping when we exceed max_candidates.
        Higher is better.  Prefers:
          - Low z (bottom filling)
          - High support
          - High contact ratio
          - Low gap below
        """
        f = candidate.features
        z_norm = f[4]
        support = f[5]
        contact = f[8]
        gap = f[9]
        return (
            -3.0 * z_norm +
            2.0 * support +
            2.0 * contact +
            -1.0 * gap
        )
