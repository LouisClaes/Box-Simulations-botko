"""
Trapped void detector for the MCTS Hybrid strategy.

Addresses Gap #5: NO spatial waste detection in any existing strategy.

A "trapped void" is an empty volume in the bin that can NEVER be filled because
it is completely enclosed by placed boxes, walls, and the heightmap surface.
Detecting these voids allows:
  1. Penalising placements that create trapped voids (reward shaping)
  2. Training the world model to predict void creation (auxiliary loss)
  3. Avoiding candidate positions that would seal off empty space

Algorithm (3D layer-by-layer flood-fill):
  We discretise the bin height into layers of resolution thickness.
  For each layer z, we build a 2D occupancy grid: a cell is "occupied" if
  any placed box covers that cell at height z, or if z >= heightmap[gx,gy]
  (above the surface — implicitly sealed).

  A cell at layer z is "accessible" if there is a path from the top of the
  bin down to it through unoccupied cells (4-connected in xy, connected
  vertically through layers).

  Any empty cell (not occupied by a box) that is NOT accessible is trapped void.

This runs in O(grid_l * grid_w * num_layers) which is acceptable for
step-wise evaluation (~10ms for typical setups).
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import List, Tuple

from config import BinConfig, Placement
from simulator.bin_state import BinState


def compute_void_fraction(bin_state: BinState) -> float:
    """
    Compute the fraction of bin volume that is trapped (unreachable) void.

    Uses a 3D layer-by-layer approach: discretises height into layers,
    builds a 3D occupancy grid, then flood-fills from the top to find
    all accessible empty cells. Anything empty but not accessible is void.

    Args:
        bin_state: Current bin state.

    Returns:
        Trapped void fraction in [0.0, 1.0].
    """
    bc = bin_state.config
    hm = bin_state.heightmap
    gl, gw = hm.shape
    res = bc.resolution

    if not bin_state.placed_boxes:
        return 0.0

    max_h = float(np.max(hm))
    if max_h < res:
        return 0.0  # Nearly empty bin

    # Number of vertical layers to check (up to max heightmap value)
    # Use coarser vertical resolution (2x) for speed
    layer_res = res * 2.0
    num_layers = max(1, int(np.ceil(max_h / layer_res)))

    # Step 1: Build 3D occupancy grid (True = occupied by a box)
    occupied = np.zeros((gl, gw, num_layers), dtype=bool)
    for p in bin_state.placed_boxes:
        px0 = int(round(p.x / res))
        py0 = int(round(p.y / res))
        px1 = min(px0 + max(1, int(round(p.oriented_l / res))), gl)
        py1 = min(py0 + max(1, int(round(p.oriented_w / res))), gw)
        pz0 = max(0, int(p.z / layer_res))
        pz1 = min(int(np.ceil((p.z + p.oriented_h) / layer_res)), num_layers)
        occupied[px0:px1, py0:py1, pz0:pz1] = True

    # Step 2: Determine which cells are "below heightmap" (inside the bin
    # volume that matters). A cell at layer z is "inside" if:
    #   z * layer_res < hm[gx, gy]  (below the surface)
    # Cells at or above the heightmap surface are outside (open air)
    layer_heights = np.arange(num_layers) * layer_res  # (num_layers,)
    # inside[gx, gy, z] = True if this cell is below the heightmap surface
    inside = layer_heights[np.newaxis, np.newaxis, :] < hm[:, :, np.newaxis]

    # Step 3: Empty interior cells = inside AND not occupied
    empty_interior = inside & (~occupied)

    total_empty = int(np.sum(empty_interior))
    if total_empty == 0:
        return 0.0

    # Step 4: 3D flood-fill from top to find accessible empty cells
    # Start from all cells that are empty and at the top layer of their column
    # (i.e., the layer just below the heightmap surface)
    accessible = np.zeros((gl, gw, num_layers), dtype=bool)
    queue = deque()

    # Seed: for each column, the topmost empty-interior cell is accessible
    # (it can be reached from above through the heightmap surface)
    for gx in range(gl):
        for gy in range(gw):
            col_h = hm[gx, gy]
            if col_h < layer_res:
                continue
            # Find the topmost interior layer for this column
            top_layer = min(int(col_h / layer_res), num_layers) - 1
            if top_layer >= 0 and empty_interior[gx, gy, top_layer]:
                accessible[gx, gy, top_layer] = True
                queue.append((gx, gy, top_layer))

    # Also seed from edge columns at all empty interior layers
    # (gaps on the side of the bin are accessible)
    for gx in range(gl):
        for gy in range(gw):
            if gx == 0 or gx == gl - 1 or gy == 0 or gy == gw - 1:
                for z in range(num_layers):
                    if empty_interior[gx, gy, z] and not accessible[gx, gy, z]:
                        accessible[gx, gy, z] = True
                        queue.append((gx, gy, z))

    # 3D flood-fill: 6-connected (4 horizontal + 2 vertical)
    while queue:
        cx, cy, cz = queue.popleft()
        for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if (0 <= nx < gl and 0 <= ny < gw and 0 <= nz < num_layers
                    and empty_interior[nx, ny, nz]
                    and not accessible[nx, ny, nz]):
                accessible[nx, ny, nz] = True
                queue.append((nx, ny, nz))

    # Step 5: Trapped void = empty interior cells that are NOT accessible
    trapped_cells = empty_interior & (~accessible)
    trapped_volume = float(np.sum(trapped_cells)) * res * res * layer_res
    total_volume = bc.volume

    return min(trapped_volume / total_volume, 1.0) if total_volume > 0 else 0.0


def compute_void_delta(
    bin_state: BinState,
    x: float, y: float, z: float,
    ol: float, ow: float, oh: float,
) -> float:
    """
    Estimate the change in trapped void fraction if a box were placed here.

    Uses a fast approximation: checks if placing the box would seal off
    any empty cells beneath it that were previously accessible.

    Args:
        bin_state: Current bin state.
        x, y, z:   Position.
        ol, ow, oh: Oriented dimensions.

    Returns:
        Estimated void delta (positive = creates more trapped void).
    """
    bc = bin_state.config
    hm = bin_state.heightmap
    res = bc.resolution
    gl, gw = hm.shape

    gx = int(round(x / res))
    gy = int(round(y / res))
    gx_end = min(gx + max(1, int(round(ol / res))), gl)
    gy_end = min(gy + max(1, int(round(ow / res))), gw)

    if gx >= gx_end or gy >= gy_end:
        return 0.0

    # Check: does placing this box create empty space below it?
    footprint = hm[gx:gx_end, gy:gy_end]

    # Gap below the box: height difference between box bottom (z) and heightmap
    gap = np.maximum(z - footprint, 0.0)
    gap_volume = float(np.sum(gap)) * res * res

    if gap_volume < 1e-6:
        return 0.0  # No gap created

    # After placement, the heightmap in the footprint will be z + oh.
    # Check if the gap below can still be accessed from any direction.
    # Heuristic: check all border cells of the footprint. If any border
    # cell has heightmap lower than z, the gap is still accessible laterally.
    # Also check if the gap connects vertically to space above.

    border_accessible = False

    for border_gx, border_gy in _border_cells(gx, gy, gx_end, gy_end, gl, gw):
        if hm[border_gx, border_gy] < z - res:
            border_accessible = True
            break

    if border_accessible:
        return 0.0  # Gap is accessible, not trapped

    # Approximate trapped void volume as fraction of bin volume
    return gap_volume / bc.volume if bc.volume > 0 else 0.0


def _border_cells(
    gx: int, gy: int, gx_end: int, gy_end: int,
    gl: int, gw: int,
) -> List[Tuple[int, int]]:
    """Get grid cells immediately adjacent to the footprint boundary."""
    cells: List[Tuple[int, int]] = []

    # Left border
    if gx > 0:
        for j in range(gy, gy_end):
            cells.append((gx - 1, j))
    # Right border
    if gx_end < gl:
        for j in range(gy, gy_end):
            cells.append((gx_end, j))
    # Back border
    if gy > 0:
        for i in range(gx, gx_end):
            cells.append((i, gy - 1))
    # Front border
    if gy_end < gw:
        for i in range(gx, gx_end):
            cells.append((i, gy_end))

    return cells
