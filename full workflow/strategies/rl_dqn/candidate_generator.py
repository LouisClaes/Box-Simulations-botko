"""
Candidate placement generator for DDQN action space reduction.

Instead of evaluating ALL 120x80x2x2 = 38,400 grid positions per step,
this module generates a smart subset of 50-200 candidate placements by
combining four complementary heuristics:

  1. Corner-aligned positions (Verma/PackMan approach)
     - Positions aligned to corners of already-placed boxes
     - Bottom-left, bottom-right, top-left, top-right of each box footprint

  2. Extreme points (EP) from placed boxes
     - Points at (x_max, y, z), (x, y_max, z), (x, y, z_max) of each box
     - Classical approach from Crainic et al. (2008)

  3. EMS-inspired positions (Empty Maximal Spaces)
     - Corner points of large empty rectangular regions
     - Simplified version of full EMS — just origin corners

  4. Coarse grid fallback
     - Regular grid at configurable step size (default 100mm)
     - Ensures coverage even when other heuristics miss good spots

For each candidate position, the generator computes rich features:
  (bin_idx, x_norm, y_norm, orient_idx, z_norm, support_ratio, height_ratio)

These features become the input to the action MLP branch.

References:
    - Crainic et al. (2008): Extreme points heuristic
    - Verma et al. (AAAI 2020): PackMan corner-aligned candidates
    - Tsang et al. (2025): Action space reduction for DDQN
    - Xiong et al. (RA-L 2024): EMS candidate generation
"""

from __future__ import annotations

import sys
import os
from typing import List, Tuple, Optional, NamedTuple

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

from config import Box, BinConfig, Orientation
from simulator.bin_state import BinState


# ─────────────────────────────────────────────────────────────────────────────
# Candidate data structure
# ─────────────────────────────────────────────────────────────────────────────

class Candidate(NamedTuple):
    """A single placement candidate with computed features."""
    bin_idx: int
    x: float
    y: float
    orient_idx: int
    z: float
    support_ratio: float
    oriented_l: float
    oriented_w: float
    oriented_h: float


# ─────────────────────────────────────────────────────────────────────────────
# Candidate Generator
# ─────────────────────────────────────────────────────────────────────────────

class CandidateGenerator:
    """
    Generates and scores candidate placements for a given box across all bins.

    The generator combines multiple heuristics to produce a compact but
    high-quality set of candidates that cover the most promising positions.
    Each candidate is validated (bounds, height, support) before inclusion.

    This reduces the action space from ~38,400 to ~50-200, making DDQN
    training feasible without losing placement quality.

    Args:
        bin_config:           Physical bin dimensions.
        num_bins:             Number of simultaneous pallets.
        max_candidates:       Hard cap on total candidates returned.
        use_corner_positions: Include corner-aligned heuristic.
        use_extreme_points:   Include extreme point heuristic.
        use_ems_positions:    Include EMS-inspired heuristic.
        use_grid_fallback:    Include coarse grid fallback.
        grid_step:            Grid spacing for fallback (mm).
        num_orientations:     2 (flat only) or 6 (all rotations).
        min_support:          Minimum support ratio for a valid candidate.
    """

    def __init__(
        self,
        bin_config: BinConfig,
        num_bins: int = 2,
        max_candidates: int = 200,
        use_corner_positions: bool = True,
        use_extreme_points: bool = True,
        use_ems_positions: bool = True,
        use_grid_fallback: bool = True,
        grid_step: float = 100.0,
        num_orientations: int = 2,
        min_support: float = 0.30,
    ) -> None:
        self.bin_config = bin_config
        self.num_bins = num_bins
        self.max_candidates = max_candidates
        self.use_corner_positions = use_corner_positions
        self.use_extreme_points = use_extreme_points
        self.use_ems_positions = use_ems_positions
        self.use_grid_fallback = use_grid_fallback
        self.grid_step = grid_step
        self.num_orientations = num_orientations
        self.min_support = min_support

    def generate(
        self,
        box: Box,
        bin_states: List[BinState],
    ) -> Tuple[List[Candidate], np.ndarray]:
        """
        Generate validated candidate placements for a box across all bins.

        Args:
            box:        The box to place.
            bin_states: Current state of each bin.

        Returns:
            candidates: List of Candidate objects.
            features:   (N, 7) array of normalised features for the network.
                        Columns: (bin_idx_norm, x_norm, y_norm, orient_norm,
                                  z_norm, support_ratio, height_ratio)
        """
        bc = self.bin_config

        # Get orientations
        if self.num_orientations >= 6:
            orientations = Orientation.get_all(box.length, box.width, box.height)
        else:
            orientations = Orientation.get_flat(box.length, box.width, box.height)

        # Collect raw positions (x, y) from all heuristics
        raw_positions: set[Tuple[float, float]] = set()

        for bin_idx, bs in enumerate(bin_states):
            if self.use_corner_positions:
                raw_positions.update(self._corner_positions(bs, bc))

            if self.use_extreme_points:
                raw_positions.update(self._extreme_points(bs, bc))

            if self.use_ems_positions:
                raw_positions.update(self._ems_positions(bs, bc))

        if self.use_grid_fallback:
            raw_positions.update(self._grid_positions(bc))

        # Always include origin
        raw_positions.add((0.0, 0.0))

        # Validate each (position, orientation, bin) combination
        candidates: List[Candidate] = []
        seen: set[Tuple[int, float, float, int]] = set()

        for bin_idx, bs in enumerate(bin_states):
            for x, y in raw_positions:
                for oidx, (ol, ow, oh) in enumerate(orientations):
                    if oidx >= self.num_orientations:
                        break

                    # Deduplicate
                    key = (bin_idx, round(x, 1), round(y, 1), oidx)
                    if key in seen:
                        continue
                    seen.add(key)

                    # Bounds check
                    if x + ol > bc.length + 0.01 or x < -0.01:
                        continue
                    if y + ow > bc.width + 0.01 or y < -0.01:
                        continue

                    # Clamp to valid range
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

                    candidates.append(Candidate(
                        bin_idx=bin_idx,
                        x=x,
                        y=y,
                        orient_idx=oidx,
                        z=z,
                        support_ratio=support,
                        oriented_l=ol,
                        oriented_w=ow,
                        oriented_h=oh,
                    ))

        # Cap at max_candidates — prioritise by quality
        if len(candidates) > self.max_candidates:
            candidates = self._rank_and_trim(candidates)

        # Build feature array
        features = self._build_features(candidates)
        return candidates, features

    # ── Position heuristics ───────────────────────────────────────────────

    def _corner_positions(
        self,
        bin_state: BinState,
        bin_config: BinConfig,
    ) -> List[Tuple[float, float]]:
        """
        Corner-aligned positions from placed boxes (Verma/PackMan).

        For each placed box, generate positions at its four corners:
        (x, y), (x+l, y), (x, y+w), (x+l, y+w).
        """
        positions: List[Tuple[float, float]] = []
        res = bin_config.resolution

        for p in bin_state.placed_boxes:
            corners = [
                (p.x, p.y),
                (p.x + p.oriented_l, p.y),
                (p.x, p.y + p.oriented_w),
                (p.x + p.oriented_l, p.y + p.oriented_w),
                # Also try placing at the far edges snapped to resolution
                (p.x - res, p.y),
                (p.x, p.y - res),
            ]
            for cx, cy in corners:
                # Snap to grid resolution
                cx = round(cx / res) * res
                cy = round(cy / res) * res
                if 0.0 <= cx <= bin_config.length and 0.0 <= cy <= bin_config.width:
                    positions.append((cx, cy))

        return positions

    def _extreme_points(
        self,
        bin_state: BinState,
        bin_config: BinConfig,
    ) -> List[Tuple[float, float]]:
        """
        Extreme points from placed boxes (Crainic et al. 2008).

        For each placed box, generate three extreme points:
        - (x_max, y, z): right face
        - (x, y_max, z): front face
        - Project x and y extreme points onto the heightmap
        """
        positions: List[Tuple[float, float]] = []

        for p in bin_state.placed_boxes:
            # Right extreme point
            positions.append((p.x + p.oriented_l, p.y))
            # Front extreme point
            positions.append((p.x, p.y + p.oriented_w))
            # Diagonal extreme point
            positions.append((p.x + p.oriented_l, p.y + p.oriented_w))

        return positions

    def _ems_positions(
        self,
        bin_state: BinState,
        bin_config: BinConfig,
    ) -> List[Tuple[float, float]]:
        """
        EMS-inspired positions: origin corners of empty maximal spaces.

        Simplified approach: scan the heightmap for transitions from
        occupied to empty regions and place candidates at those boundaries.
        """
        positions: List[Tuple[float, float]] = []
        hm = bin_state.heightmap
        res = bin_config.resolution
        gl, gw = hm.shape

        # Find cells where height drops significantly compared to neighbours
        # These transition points indicate EMS origins
        step = max(1, int(self.grid_step / (2 * res)))

        for gx in range(0, gl - 1, step):
            for gy in range(0, gw - 1, step):
                h_curr = hm[gx, gy]
                # Check right neighbour
                if gx + step < gl:
                    h_right = hm[gx + step, gy]
                    if h_curr > 0 and h_right < h_curr * 0.5:
                        positions.append(((gx + step) * res, gy * res))
                    elif h_right > 0 and h_curr < h_right * 0.5:
                        positions.append((gx * res, gy * res))
                # Check front neighbour
                if gy + step < gw:
                    h_front = hm[gx, gy + step]
                    if h_curr > 0 and h_front < h_curr * 0.5:
                        positions.append((gx * res, (gy + step) * res))
                    elif h_front > 0 and h_curr < h_front * 0.5:
                        positions.append((gx * res, gy * res))

        # Also add positions at the base of "walls" in the heightmap
        max_h = float(np.max(hm)) if hm.size > 0 else 0.0
        if max_h > 0:
            # Find columns at zero height adjacent to non-zero columns
            for gx in range(0, gl, step):
                for gy in range(0, gw, step):
                    if hm[gx, gy] < res:
                        has_neighbour = False
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = gx + dx, gy + dy
                            if 0 <= nx < gl and 0 <= ny < gw and hm[nx, ny] > res:
                                has_neighbour = True
                                break
                        if has_neighbour:
                            positions.append((gx * res, gy * res))

        return positions

    def _grid_positions(self, bin_config: BinConfig) -> List[Tuple[float, float]]:
        """
        Coarse grid fallback positions.

        Generates a regular grid at self.grid_step spacing to ensure
        basic coverage even when heuristics fail.
        """
        positions: List[Tuple[float, float]] = []
        step = self.grid_step

        x = 0.0
        while x <= bin_config.length:
            y = 0.0
            while y <= bin_config.width:
                positions.append((x, y))
                y += step
            x += step

        return positions

    # ── Ranking and trimming ──────────────────────────────────────────────

    def _rank_and_trim(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Rank candidates by quality heuristic and trim to max_candidates.

        Quality score:
          + support_ratio   (prefer well-supported)
          - z / bin_height  (prefer low placements)
          + coverage bonus  (ensure bin diversity)
        """
        bc = self.bin_config

        def score(c: Candidate) -> float:
            z_norm = c.z / max(bc.height, 1.0)
            return c.support_ratio - 0.5 * z_norm

        # Sort by score descending
        candidates.sort(key=score, reverse=True)

        # Ensure all bins are represented
        selected: List[Candidate] = []
        per_bin: dict[int, int] = {}
        min_per_bin = max(10, self.max_candidates // (self.num_bins * 2))

        # First pass: ensure minimum per bin
        for c in candidates:
            count = per_bin.get(c.bin_idx, 0)
            if count < min_per_bin:
                selected.append(c)
                per_bin[c.bin_idx] = count + 1

        # Second pass: fill remaining slots from top candidates
        selected_set = set(id(c) for c in selected)
        for c in candidates:
            if len(selected) >= self.max_candidates:
                break
            if id(c) not in selected_set:
                selected.append(c)

        return selected[:self.max_candidates]

    # ── Feature building ──────────────────────────────────────────────────

    def _build_features(self, candidates: List[Candidate]) -> np.ndarray:
        """
        Build normalised feature array for the action MLP branch.

        Features per candidate (7 dimensions):
          0: bin_idx / (num_bins - 1)        — which bin [0, 1]
          1: x / bin_length                   — normalised x [0, 1]
          2: y / bin_width                    — normalised y [0, 1]
          3: orient_idx / max_orients         — orientation [0, 1]
          4: z / bin_height                   — normalised z [0, 1]
          5: support_ratio                    — support [0, 1]
          6: (z + oh) / bin_height            — height utilisation [0, 1]
        """
        bc = self.bin_config
        n = len(candidates)

        if n == 0:
            return np.zeros((0, 7), dtype=np.float32)

        features = np.zeros((n, 7), dtype=np.float32)

        bin_norm = max(self.num_bins - 1, 1)
        orient_norm = max(self.num_orientations - 1, 1)

        for i, c in enumerate(candidates):
            features[i, 0] = c.bin_idx / bin_norm
            features[i, 1] = c.x / max(bc.length, 1.0)
            features[i, 2] = c.y / max(bc.width, 1.0)
            features[i, 3] = c.orient_idx / orient_norm
            features[i, 4] = c.z / max(bc.height, 1.0)
            features[i, 5] = c.support_ratio
            features[i, 6] = (c.z + c.oriented_h) / max(bc.height, 1.0)

        return features

    # ── Utility ───────────────────────────────────────────────────────────

    def features_for_single(
        self,
        candidate: Candidate,
    ) -> np.ndarray:
        """Build feature vector for a single candidate. Returns (7,) array."""
        return self._build_features([candidate])[0]
