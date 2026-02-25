"""
Bin state — tracks the 3D state of a bin/pallet.

The BinState is the primary data object exchanged between the simulator
and strategies.  It provides:

  Spatial queries:
    .get_height_at(x, y, w, d)   — max height in a footprint region
    .get_support_ratio(...)       — fraction of base that is supported
    .get_fill_rate()              — volumetric utilisation
    .get_max_height()             — peak height anywhere
    .get_surface_roughness()      — surface smoothness metric

  Full 3D state:
    .placed_boxes                 — List[Placement], full 3D info per box
    .heightmap                    — np.ndarray of current heights
    .config                       — BinConfig dimensions

  Safe cloning:
    .copy()                       — deep copy for lookahead simulations

Usage:
    state = simulator.get_bin_state()
    z = state.get_height_at(x, y, box_l, box_w)
    all_boxes = state.placed_boxes  # read-only (Placement is frozen)
"""

import numpy as np
from typing import List, Tuple

from config import BinConfig, Placement


class BinState:
    """
    Manages the 3D state of a single bin/pallet.

    Internally uses a 2D heightmap (grid_l × grid_w) where each cell
    stores the maximum height at that grid position.  The resolution
    parameter controls the mapping from real-world to grid coordinates.

    The placed_boxes list contains immutable Placement objects that
    give strategies full 3D positional information about every box.
    """

    __slots__ = ("config", "heightmap", "placed_boxes", "_step_counter")

    def __init__(self, config: BinConfig) -> None:
        self.config: BinConfig = config
        self.heightmap: np.ndarray = np.zeros(
            (config.grid_l, config.grid_w), dtype=np.float64,
        )
        self.placed_boxes: List[Placement] = []
        self._step_counter: int = 0

    # ── Coordinate conversion ────────────────────────────────────────────

    def _to_grid(self, real_val: float) -> int:
        """Convert a real-world coordinate to a grid index."""
        return int(round(real_val / self.config.resolution))

    # ── Spatial queries (for strategies) ─────────────────────────────────

    def get_height_at(self, x: float, y: float, w: float, d: float) -> float:
        """
        Maximum height in the footprint [x, x+w) × [y, y+d).
        This is the z at which a box placed here would rest.
        """
        gx = self._to_grid(x)
        gy = self._to_grid(y)
        gx_end = min(gx + self._to_grid(w), self.config.grid_l)
        gy_end = min(gy + self._to_grid(d), self.config.grid_w)

        if gx >= gx_end or gy >= gy_end:
            return 0.0

        return float(np.max(self.heightmap[gx:gx_end, gy:gy_end]))

    def get_support_ratio(
        self, x: float, y: float, w: float, d: float, z: float,
    ) -> float:
        """
        Fraction of the box's base area that is supported at height *z*.

        Returns 1.0 for floor placements (z ≈ 0).
        """
        if z < 1e-9:
            return 1.0

        gx = self._to_grid(x)
        gy = self._to_grid(y)
        gx_end = min(gx + self._to_grid(w), self.config.grid_l)
        gy_end = min(gy + self._to_grid(d), self.config.grid_w)

        region = self.heightmap[gx:gx_end, gy:gy_end]
        if region.size == 0:
            return 0.0

        tolerance = self.config.resolution * 0.5
        supported = int(np.sum(np.abs(region - z) <= tolerance))
        return supported / region.size

    def get_fill_rate(self) -> float:
        """Volumetric fill rate = placed_volume / bin_volume."""
        bin_vol = self.config.volume
        if bin_vol == 0:
            return 0.0
        return sum(p.volume for p in self.placed_boxes) / bin_vol

    def get_max_height(self) -> float:
        """Current peak height anywhere in the bin."""
        return float(np.max(self.heightmap))

    def get_surface_roughness(self) -> float:
        """Mean absolute height difference between neighbouring cells."""
        if self.heightmap.size < 2:
            return 0.0
        dx = np.abs(np.diff(self.heightmap, axis=0))
        dy = np.abs(np.diff(self.heightmap, axis=1))
        return float(np.mean(dx) + np.mean(dy)) / 2.0

    def get_heightmap_copy(self) -> np.ndarray:
        """Safe copy of the heightmap for strategy use."""
        return self.heightmap.copy()

    @property
    def step_count(self) -> int:
        """Number of placements applied so far."""
        return self._step_counter

    # ── State mutation (simulator only — NOT for strategies) ─────────────

    def apply_placement(self, placement: Placement) -> None:
        """
        Update the heightmap and box list after a validated placement.

        **Called by the Simulator only** — strategies must not call this.
        """
        gx = self._to_grid(placement.x)
        gy = self._to_grid(placement.y)
        gx_end = min(gx + self._to_grid(placement.oriented_l), self.config.grid_l)
        gy_end = min(gy + self._to_grid(placement.oriented_w), self.config.grid_w)

        new_top = placement.z + placement.oriented_h
        self.heightmap[gx:gx_end, gy:gy_end] = np.maximum(
            self.heightmap[gx:gx_end, gy:gy_end], new_top,
        )
        self.placed_boxes.append(placement)
        self._step_counter += 1

    # ── Cloning (for strategy lookahead) ─────────────────────────────────

    def copy(self) -> "BinState":
        """
        Deep copy of this state.

        Strategies can safely use copies for what-if simulations
        without affecting the real bin state.
        """
        clone = BinState(self.config)
        clone.heightmap = self.heightmap.copy()
        clone.placed_boxes = list(self.placed_boxes)  # Placement is frozen
        clone._step_counter = self._step_counter
        return clone

    # ── Representation ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"BinState(boxes={len(self.placed_boxes)}, "
            f"fill={self.get_fill_rate():.1%}, "
            f"max_h={self.get_max_height():.1f}/{self.config.height})"
        )
