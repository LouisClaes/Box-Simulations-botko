"""
WallE Heuristic for Online 3D Bin Packing - Coding Ideas
=========================================================

Source: "A Generalized Reinforcement Learning Algorithm for Online 3D Bin-Packing"
        Verma et al. (2020), AAAI 2020

WallE is a novel constructive heuristic that blends characteristics of:
- Floor Building (prefer low placements)
- First Fit (prefer corner/origin placements)
- Wall Building (promote smooth surfaces and snug fits)

It is fully deterministic, requires no training, and serves as an excellent
baseline for both standalone use and as a component within ML-based systems.

ESTIMATED COMPLEXITY AND FEASIBILITY
-------------------------------------
- Implementation time: 3-5 days
- Time per box decision: ~10ms (paper reports 0.0106 sec)
- Fill rate: ~81.8% average (paper result)
- No training required
- Immediately deployable
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================================
# DATA STRUCTURES (shared with packman_dqn_coding_ideas.py)
# ============================================================================

@dataclass
class Box:
    """Represents a box/parcel to be packed."""
    id: int
    length: int  # in grid cells (1 cell = 1 cm)
    width: int
    height: int
    weight: float = 1.0

    def rotated(self) -> 'Box':
        """Return box with length and width swapped (z-axis 90-degree rotation)."""
        return Box(self.id, self.width, self.length, self.height, self.weight)

    @property
    def volume(self) -> int:
        return self.length * self.width * self.height


@dataclass
class Container:
    """Represents a bin/container using a 2D heightmap."""
    id: int
    length: int
    width: int
    max_height: int
    heightmap: np.ndarray = field(default=None)
    is_open: bool = True
    packed_boxes: List = field(default_factory=list)

    def __post_init__(self):
        if self.heightmap is None:
            self.heightmap = np.zeros((self.length, self.width), dtype=np.int32)

    @property
    def fill_fraction(self) -> float:
        total = self.length * self.width * self.max_height
        used = sum(b.volume for b in self.packed_boxes)
        return used / total if total > 0 else 0.0

    def is_feasible(self, box: Box, i: int, j: int) -> bool:
        """Check if placing box at (i, j) is feasible."""
        l, w, h = box.length, box.width, box.height
        if i + l > self.length or j + w > self.width:
            return False
        region = self.heightmap[i:i+l, j:j+w]
        base_height = region.flat[0]
        if not np.all(region == base_height):
            return False
        if base_height + h > self.max_height:
            return False
        return True

    def place_box(self, box: Box, i: int, j: int) -> None:
        """Place box at (i, j). Assumes feasibility checked."""
        l, w, h = box.length, box.width, box.height
        base_height = self.heightmap[i, j]
        self.heightmap[i:i+l, j:j+w] = base_height + h
        self.packed_boxes.append(box)


# ============================================================================
# WALLE STABILITY SCORE
# ============================================================================

def compute_walle_score(
    container: Container,
    box: Box,
    i: int,
    j: int,
    alpha1: float = 0.75,
    alpha2: float = 1.0,
    alpha3: float = 1.0,
    alpha4: float = 0.01,
    alpha5: float = 1.0
) -> float:
    """
    Compute the WallE stability score S for placing a box at (i, j).

    S = -alpha1 * G_var + alpha2 * G_high + alpha3 * G_flush
        - alpha4 * (i + j) - alpha5 * h_{i,j}

    Components:
    -----------
    G_var:   Net variation = sum of |height differences| between the box's
             new top surface and all neighboring cells around its border.
             Lower means smoother placement. (Penalized, hence negative.)

    G_high:  Count of bordering cells HIGHER than the box's new top surface.
             Higher means the box is nestled into a valley. (Rewarded.)

    G_flush: Count of bordering cells at EXACTLY the same height as the
             box's new top surface. Higher means smooth resulting surface. (Rewarded.)

    (i+j):   Distance from origin. Penalty encourages packing toward corner. (Penalized.)

    h_{i,j}: Base height at placement location. Penalty encourages low placement
             (floor building). (Penalized.)

    Parameters
    ----------
    container : Container with current heightmap state
    box : Box to place (in its current orientation)
    i, j : Top-left grid position for placement
    alpha1..5 : Weight parameters (defaults from paper)

    Returns
    -------
    float : The stability score S. Higher is better.
    """
    l, w, h = box.length, box.width, box.height
    hmap = container.heightmap
    L, B = container.length, container.width

    base_height = int(hmap[i, j])
    new_top = base_height + h  # height of box top surface after placement

    # Collect all bordering cells
    # A bordering cell is any cell adjacent to the box footprint but not under it
    border_cells = set()

    for di in range(l):
        for dj in range(w):
            ci, cj = i + di, j + dj
            # Check 4-connected neighbors of each cell in the box footprint
            for ni, nj in [(ci-1, cj), (ci+1, cj), (ci, cj-1), (ci, cj+1)]:
                # Neighbor must be outside the box footprint
                if not (i <= ni < i + l and j <= nj < j + w):
                    border_cells.add((ni, nj))

    # Compute G_var, G_high, G_flush
    G_var = 0.0
    G_high = 0
    G_flush = 0

    for (bi, bj) in border_cells:
        if 0 <= bi < L and 0 <= bj < B:
            # Cell is within container
            neighbor_h = int(hmap[bi, bj])
            G_var += abs(new_top - neighbor_h)

            if neighbor_h > new_top:
                G_high += 1
            elif neighbor_h == new_top:
                G_flush += 1
        else:
            # Cell is a wall (outside container bounds)
            # Wall cells contribute 0 to G_var (box is flush with wall)
            # and can be considered "flush" or "high" depending on interpretation
            # Paper: "cells flush with the wall, those quantities are filled by zeroes"
            # This means walls contribute 0 to G_var (which is good)
            pass

    S = (-alpha1 * G_var
         + alpha2 * G_high
         + alpha3 * G_flush
         - alpha4 * (i + j)
         - alpha5 * base_height)

    return S


# ============================================================================
# EXTENDED STABILITY SCORE (for our use case)
# ============================================================================

def compute_extended_walle_score(
    container: Container,
    box: Box,
    i: int,
    j: int,
    alpha1: float = 0.75,
    alpha2: float = 1.0,
    alpha3: float = 1.0,
    alpha4: float = 0.01,
    alpha5: float = 1.0,
    alpha6: float = 2.0,   # NEW: support area weight
    alpha7: float = 0.5,   # NEW: center of gravity weight
    min_support_fraction: float = 0.8  # minimum support for feasibility
) -> Tuple[float, bool]:
    """
    Extended WallE score with formal stability metrics.

    EXTENSIONS beyond original paper:
    1. Support area fraction: what percentage of the box's base is supported
       by the surface below (floor or other boxes)?
    2. Center of gravity penalty: how far is the box's CoG from the center
       of its support polygon?

    Returns
    -------
    (score, is_stable) : float score and boolean stability check
    """
    l, w, h = box.length, box.width, box.height
    hmap = container.heightmap
    L, B = container.length, container.width

    base_height = int(hmap[i, j])

    # ----- Original WallE score -----
    S_original = compute_walle_score(container, box, i, j,
                                      alpha1, alpha2, alpha3, alpha4, alpha5)

    # ----- Support area fraction -----
    # Check what fraction of the box's base area is supported
    # A cell (ci, cj) under the box is "supported" if:
    #   - It is on the floor (base_height == 0), OR
    #   - The cell height equals the base_height (meaning there is a box below)
    #
    # Note: the basic feasibility check already ensures all cells are at the same
    # height. So if feasible, support is either 100% (all on floor or all on boxes)
    # or it is a more nuanced partial support check.
    #
    # For partial support (e.g., box overhanging edge of another box),
    # we would need a more sophisticated heightmap check.
    # With the flat-base constraint from the paper, support is always 100%.
    # We relax this for our extension:

    region = hmap[i:i+l, j:j+w]
    # Count cells at the base_height level (fully supported)
    supported_cells = np.sum(region == base_height)
    total_cells = l * w
    support_fraction = supported_cells / total_cells

    is_stable = support_fraction >= min_support_fraction

    # ----- Center of gravity offset -----
    # Compute the center of the support polygon
    # For simplicity, compute the centroid of supported cells
    supported_positions = np.argwhere(region == base_height)
    if len(supported_positions) > 0:
        support_centroid = np.mean(supported_positions, axis=0)
        box_centroid = np.array([l / 2.0, w / 2.0])
        cog_offset = np.linalg.norm(support_centroid - box_centroid)
        # Normalize by box diagonal
        box_diagonal = np.sqrt(l**2 + w**2)
        cog_penalty = cog_offset / box_diagonal if box_diagonal > 0 else 0
    else:
        cog_penalty = 1.0  # worst case: no support

    # ----- Combined score -----
    S_extended = (S_original
                  + alpha6 * support_fraction
                  - alpha7 * cog_penalty)

    return S_extended, is_stable


# ============================================================================
# WALLE PLACEMENT ALGORITHM
# ============================================================================

def walle_place_single_bin(
    container: Container,
    box: Box
) -> Optional[Tuple[int, int, int, float]]:
    """
    Find the best placement for a box in a single container using WallE score.

    Returns: (i, j, orientation, score) or None if no feasible placement.
    orientation: 0 = original, 1 = rotated
    """
    best_score = float('-inf')
    best_placement = None

    for orientation in [0, 1]:
        current_box = box if orientation == 0 else box.rotated()
        l, w = current_box.length, current_box.width

        for i in range(container.length - l + 1):
            for j in range(container.width - w + 1):
                if container.is_feasible(current_box, i, j):
                    score = compute_walle_score(container, current_box, i, j)
                    if score > best_score:
                        best_score = score
                        best_placement = (i, j, orientation, score)

    return best_placement


def walle_place_multi_bin(
    containers: List[Container],
    box: Box,
    open_new_if_needed: bool = True,
    next_container_id: int = None,
    container_dims: Tuple[int, int, int] = None
) -> Tuple[Optional[int], Optional[Tuple[int, int, int, float]]]:
    """
    Find the best placement for a box across multiple containers using WallE.

    For our k=2 bounded setup: searches both active bins.

    Returns: (container_id, (i, j, orientation, score)) or (None, None) if no fit.
    """
    best_score = float('-inf')
    best_container = None
    best_placement = None

    for container in containers:
        if not container.is_open:
            continue

        placement = walle_place_single_bin(container, box)
        if placement is not None:
            i, j, orientation, score = placement
            if score > best_score:
                best_score = score
                best_container = container.id
                best_placement = placement

    return best_container, best_placement


def walle_place_with_buffer(
    containers: List[Container],
    buffer_boxes: List[Box]
) -> Optional[Tuple[int, int, Tuple[int, int, int, float]]]:
    """
    EXTENSION FOR SEMI-ONLINE WITH BUFFER:
    Select which box from the buffer to pack and where.

    Strategy: evaluate WallE score for ALL (box, container, location, orientation)
    combinations and select the globally best one.

    This is more computationally expensive but leverages the buffer advantage.

    Returns: (box_id, container_id, (i, j, orientation, score)) or None
    """
    best_overall_score = float('-inf')
    best_box_id = None
    best_container_id = None
    best_placement = None

    for box in buffer_boxes:
        for container in containers:
            if not container.is_open:
                continue

            placement = walle_place_single_bin(container, box)
            if placement is not None:
                i, j, orientation, score = placement
                if score > best_overall_score:
                    best_overall_score = score
                    best_box_id = box.id
                    best_container_id = container.id
                    best_placement = placement

    if best_box_id is not None:
        return (best_box_id, best_container_id, best_placement)
    return None


# ============================================================================
# OPTIMIZED WALLE (skip full grid scan, use corner locations only)
# ============================================================================

def walle_place_corners_only(
    container: Container,
    box: Box
) -> Optional[Tuple[int, int, int, float]]:
    """
    Optimized WallE that only evaluates corner-aligned locations
    (same as PackMan's selective search).

    This significantly reduces computation while keeping most of the quality,
    since the best placements are almost always at corner locations.

    Speedup: from O(L * B * 2) to O(num_corners * 2) per box,
    where num_corners is typically 10-50 for a partially filled container.
    """
    best_score = float('-inf')
    best_placement = None

    for orientation in [0, 1]:
        current_box = box if orientation == 0 else box.rotated()
        l, w = current_box.length, current_box.width

        # Find corner locations (using the selective search from PackMan)
        corner_locations = _find_corner_locations(container, current_box)

        for (i, j) in corner_locations:
            if container.is_feasible(current_box, i, j):
                score = compute_walle_score(container, current_box, i, j)
                if score > best_score:
                    best_score = score
                    best_placement = (i, j, orientation, score)

    return best_placement


def _find_corner_locations(
    container: Container,
    box: Box
) -> List[Tuple[int, int]]:
    """
    Find corner-aligned candidate locations for the box.
    A candidate is where a corner of the box aligns with:
    - A corner of the container
    - An edge transition in the heightmap
    """
    hmap = container.heightmap
    L, B = container.length, container.width
    l, w = box.length, box.width

    candidates = set()

    # Container corners
    for ci in [0, max(0, L - l)]:
        for cj in [0, max(0, B - w)]:
            candidates.add((ci, cj))

    # Height transition points
    for i in range(L):
        for j in range(B):
            is_transition = False
            h = hmap[i, j]

            # Check if height changes at this cell boundary
            if j + 1 < B and hmap[i, j + 1] != h:
                is_transition = True
            if i + 1 < L and hmap[i + 1, j] != h:
                is_transition = True
            if h > 0 and (i == 0 or j == 0):
                is_transition = True

            if is_transition:
                # Box top-left at this corner
                candidates.add((i, j))
                # Box top-right: start at (i, j - w + 1)
                if j - w + 1 >= 0:
                    candidates.add((i, j - w + 1))
                # Box bottom-left: start at (i - l + 1, j)
                if i - l + 1 >= 0:
                    candidates.add((i - l + 1, j))
                # Box bottom-right: start at (i - l + 1, j - w + 1)
                if i - l + 1 >= 0 and j - w + 1 >= 0:
                    candidates.add((i - l + 1, j - w + 1))

    # Filter valid
    return [(ci, cj) for ci, cj in candidates
            if 0 <= ci <= L - l and 0 <= cj <= B - w]


# ============================================================================
# BIN CLOSING HEURISTIC FOR K=2 BOUNDED SPACE
# ============================================================================

def should_close_bin(
    container: Container,
    buffer_boxes: List[Box],
    fill_threshold: float = 0.85,
    no_fit_threshold: int = 3
) -> bool:
    """
    Heuristic to decide when to close a bin in k=2 bounded space.

    Decision criteria:
    1. Fill threshold: if bin fill fraction exceeds threshold, consider closing
    2. No-fit count: if N consecutive buffer items cannot fit, consider closing
    3. Volume comparison: if remaining volume is less than smallest buffer item

    For our 2-bounded setup, closing one bin means opening a fresh one,
    which provides new space but loses the partially-filled bin's potential.

    Returns: True if the bin should be closed.
    """
    # Criterion 1: High fill rate
    if container.fill_fraction >= fill_threshold:
        return True

    # Criterion 2: Remaining capacity too small for buffer items
    max_possible_height = container.max_height
    hmap = container.heightmap
    remaining_heights = max_possible_height - hmap
    max_remaining = np.max(remaining_heights)

    if max_remaining <= 0:
        return True  # Completely full

    # Check if any buffer item can fit
    fit_count = 0
    for box in buffer_boxes:
        placement = walle_place_single_bin(container, box)
        if placement is not None:
            fit_count += 1

    if fit_count == 0:
        return True  # No buffer items fit

    # Criterion 3: Very few items fit relative to buffer size
    if fit_count <= no_fit_threshold and container.fill_fraction > 0.6:
        return True

    return False


# ============================================================================
# FULL WALLE PIPELINE FOR K=2 BOUNDED + BUFFER
# ============================================================================

def walle_full_pipeline(
    box_stream: List[Box],
    buffer_capacity: int = 10,
    k: int = 2,
    container_length: int = 45,
    container_width: int = 80,
    container_height: int = 50,
    use_corner_optimization: bool = True,
    verbose: bool = False
) -> dict:
    """
    Complete WallE pipeline for semi-online 3D bin packing
    with buffer and 2-bounded space.

    Parameters
    ----------
    box_stream : list of Box objects arriving from conveyor
    buffer_capacity : size of buffer (5-10 for our use case)
    k : bounded space parameter (2 for our use case)
    container_* : container dimensions
    use_corner_optimization : if True, use corner-only search for speed
    verbose : print progress

    Returns
    -------
    dict with keys:
        'containers': list of all used containers (open and closed)
        'num_bins_used': total number of non-empty bins
        'avg_fill_fraction': average fill across first Opt bins
        'total_volume_packed': total box volume packed
        'placements_log': list of (box_id, container_id, i, j, orientation, score)
    """
    # Initialize
    active_containers = [
        Container(id=cid, length=container_length,
                  width=container_width, max_height=container_height)
        for cid in range(k)
    ]
    all_containers = list(active_containers)
    next_container_id = k

    # Fill buffer
    buffer = []
    stream_idx = 0
    while len(buffer) < buffer_capacity and stream_idx < len(box_stream):
        buffer.append(box_stream[stream_idx])
        stream_idx += 1

    placements_log = []
    total_volume_packed = 0
    step = 0

    while buffer:
        step += 1

        # Try to find best (box, placement) across buffer and active bins
        if use_corner_optimization:
            result = _find_best_corner_placement(active_containers, buffer)
        else:
            result = walle_place_with_buffer(active_containers, buffer)

        if result is not None:
            box_id, container_id, (i, j, orientation, score) = result

            # Execute placement
            target_box = next(b for b in buffer if b.id == box_id)
            placed_box = target_box if orientation == 0 else target_box.rotated()
            target_container = next(c for c in active_containers if c.id == container_id)

            target_container.place_box(placed_box, i, j)
            total_volume_packed += target_box.volume

            # Log
            placements_log.append((box_id, container_id, i, j, orientation, score))

            # Remove from buffer, add new from stream
            buffer = [b for b in buffer if b.id != box_id]
            if stream_idx < len(box_stream):
                buffer.append(box_stream[stream_idx])
                stream_idx += 1

            if verbose and step % 50 == 0:
                fills = [c.fill_fraction for c in active_containers]
                print(f"Step {step}: fills={[f'{f:.1%}' for f in fills]}, "
                      f"buffer={len(buffer)}")

        else:
            # No placement possible for any buffer item in any active bin
            # Close the fullest bin and open a new one
            bin_to_close = max(active_containers, key=lambda c: c.fill_fraction)
            bin_to_close.is_open = False

            if verbose:
                print(f"Step {step}: Closing bin {bin_to_close.id} "
                      f"(fill={bin_to_close.fill_fraction:.1%})")

            new_container = Container(
                id=next_container_id,
                length=container_length,
                width=container_width,
                max_height=container_height
            )
            next_container_id += 1
            all_containers.append(new_container)

            # Replace closed bin in active list
            active_containers = [
                c if c.is_open else new_container
                for c in active_containers
            ]
            # Edge case: if we replaced, mark new one properly
            for idx, c in enumerate(active_containers):
                if c.id == bin_to_close.id:
                    active_containers[idx] = new_container
                    break

        # Proactive bin closing check
        for container in active_containers:
            if container.is_open and len(container.packed_boxes) > 0:
                if should_close_bin(container, buffer):
                    # Only close if we have another active bin or can open one
                    open_count = sum(1 for c in active_containers if c.is_open)
                    if open_count > 1:
                        container.is_open = False
                        new_container = Container(
                            id=next_container_id,
                            length=container_length,
                            width=container_width,
                            max_height=container_height
                        )
                        next_container_id += 1
                        all_containers.append(new_container)
                        for idx, c in enumerate(active_containers):
                            if c.id == container.id:
                                active_containers[idx] = new_container
                                break
                        if verbose:
                            print(f"Step {step}: Proactively closing bin {container.id}")

    # Final results
    used_containers = [c for c in all_containers if len(c.packed_boxes) > 0]
    num_bins = len(used_containers)
    avg_fill = (np.mean([c.fill_fraction for c in used_containers])
                if used_containers else 0.0)

    return {
        'containers': all_containers,
        'num_bins_used': num_bins,
        'avg_fill_fraction': avg_fill,
        'total_volume_packed': total_volume_packed,
        'placements_log': placements_log
    }


def _find_best_corner_placement(
    containers: List[Container],
    buffer: List[Box]
) -> Optional[Tuple[int, int, Tuple[int, int, int, float]]]:
    """Helper: find best placement using corner-optimized search."""
    best_score = float('-inf')
    best_result = None

    for box in buffer:
        for container in containers:
            if not container.is_open:
                continue
            placement = walle_place_corners_only(container, box)
            if placement is not None:
                i, j, orientation, score = placement
                if score > best_score:
                    best_score = score
                    best_result = (box.id, container.id, placement)

    return best_result


# ============================================================================
# MAIN: DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    import time

    # Generate test boxes
    np.random.seed(42)
    num_boxes = 200
    boxes = [
        Box(id=i,
            length=np.random.randint(5, 25),
            width=np.random.randint(5, 25),
            height=np.random.randint(5, 20))
        for i in range(num_boxes)
    ]

    total_box_volume = sum(b.volume for b in boxes)
    container_volume = 45 * 80 * 50
    theoretical_min_bins = int(np.ceil(total_box_volume / container_volume))

    print(f"Test: {num_boxes} boxes, total volume = {total_box_volume}")
    print(f"Container volume = {container_volume}")
    print(f"Theoretical minimum bins = {theoretical_min_bins}")
    print()

    # Run WallE pipeline
    start = time.time()
    results = walle_full_pipeline(
        box_stream=boxes,
        buffer_capacity=10,
        k=2,
        use_corner_optimization=True,
        verbose=True
    )
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Bins used: {results['num_bins_used']}")
    print(f"  Avg fill fraction: {results['avg_fill_fraction']:.1%}")
    print(f"  Competitive ratio: {results['num_bins_used'] / theoretical_min_bins:.2f}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Time per box: {elapsed / num_boxes * 1000:.1f}ms")
