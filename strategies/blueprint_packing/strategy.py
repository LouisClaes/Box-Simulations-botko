"""
Blueprint Packing Strategy for 3D bin packing.

Paper:
    Ayyadevara, V., Dabas, N., Khan, A., Sreenivas, R. S. (2022).
    "Near-optimal Algorithms for Stochastic Online Bin Packing."
    Proc. 49th International Colloquium on Automata, Languages, and Programming
    (ICALP 2022). arXiv:2205.03622.

Core concept:
    When boxes arrive from a stationary (or near-stationary) distribution,
    it is possible to exploit distribution knowledge to pack near-optimally.
    The strategy operates in two phases:

    Phase 1 — Learning (first min(30, N/4) boxes):
        Observe arriving boxes without special placement intelligence.
        Accumulate them in a buffer.  Use BFD fallback for actual placement
        so that the bin is still populated during learning.

    Phase 2 — Blueprint construction (triggered once, after learning):
        Sort the observed sample by volume descending (Best-Fit Decreasing
        order) and greedily pack them into a virtual (empty) heightmap to
        create a sequence of "proxy placements" — the blueprint.  These
        proxy placements encode good positions for the distribution.

    Phase 3 — Online packing (after blueprint is built):
        For each new box, attempt "upright matching": find a blueprint proxy
        slot whose dimensions are at least as large as the box's oriented
        dimensions and whose position is still geometrically valid in the
        current (live) bin state.  If a match is found, place the box there.
        If no match is found, fall back to the BFD heuristic.

    This mirrors the paper's algorithm structure: use an offline blueprint
    derived from samples, then match online arrivals to blueprint slots.
    The 3-D adaptation uses volume-decreasing sort and greedy heightmap
    packing for blueprint construction, with upright dimensional matching
    for the online phase.

Algorithm parameters (module-level constants):
    LEARNING_RATIO          -- fraction of total boxes used for learning
    MIN_LEARNING_BOXES      -- minimum number of boxes to observe
    MAX_LEARNING_BOXES      -- maximum cap on the learning buffer
    BLUEPRINT_GRID_STEP     -- grid step for virtual blueprint construction
    BFD_SCORE_SUPPORT_W     -- support weight in BFD fallback scoring
    BFD_SCORE_HEIGHT_W      -- height penalty weight in BFD fallback scoring
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ---------------------------------------------------------------------------
# Module-level constants (hyperparameters)
# ---------------------------------------------------------------------------

# Anti-float threshold -- must match the simulator's rejection limit.
MIN_SUPPORT: float = 0.30

# Fraction of episode boxes to observe during the learning phase.
LEARNING_RATIO: float = 0.25

# Hard limits on the learning buffer size.
MIN_LEARNING_BOXES: int = 10
MAX_LEARNING_BOXES: int = 30

# Grid step used when constructing the virtual blueprint heightmap.
# Smaller = finer blueprint, slower; larger = coarser, faster.
BLUEPRINT_GRID_STEP: float = 1.0

# BFD fallback scoring weights
BFD_SCORE_SUPPORT_W: float = 2.0   # reward for good support ratio
BFD_SCORE_HEIGHT_W: float = 1.0    # penalty for high placement

# Upright match tolerance: allow proxy slots slightly smaller than the box
# (1 cm each side) so minor size variation does not prevent matching.
PROXY_FIT_TOLERANCE: float = 1.0


# ---------------------------------------------------------------------------
# Internal data class for blueprint proxy placements
# ---------------------------------------------------------------------------

@dataclass
class _ProxyPlacement:
    """A position in the virtual blueprint packing."""
    x: float
    y: float
    z: float
    l: float   # oriented length of the proxy box
    w: float   # oriented width  of the proxy box
    h: float   # oriented height of the proxy box


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

@register_strategy
class BlueprintPackingStrategy(BaseStrategy):
    """
    Blueprint Packing: observe-then-pack strategy based on stochastic
    online bin packing theory.

    During the learning phase the strategy accumulates observed boxes while
    using a BFD fallback for placement.  After enough samples are collected
    it constructs an offline "blueprint" by greedily packing the observed
    boxes in volume-decreasing order into a virtual bin.  Subsequent boxes
    are matched to blueprint proxy slots by dimension compatibility; if no
    match is found the BFD fallback is used.

    Reference:
        Ayyadevara et al., "Near-optimal Algorithms for Stochastic Online
        Bin Packing", ICALP 2022. arXiv:2205.03622.

    Attributes:
        name: Strategy identifier for the registry ("blueprint_packing").
    """

    name: str = "blueprint_packing"

    def __init__(self) -> None:
        super().__init__()
        self._buffer: List[Box] = []
        self._blueprint: Optional[List[_ProxyPlacement]] = None
        self._is_learning: bool = True
        self._learning_target: int = MIN_LEARNING_BOXES
        self._scan_step: float = BLUEPRINT_GRID_STEP

    # -- Lifecycle -----------------------------------------------------------

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Reset all learning and blueprint state for a fresh episode."""
        super().on_episode_start(config)
        self._buffer = []
        self._blueprint = None
        self._is_learning = True
        self._scan_step = max(BLUEPRINT_GRID_STEP, config.bin.resolution)
        # Derive the learning target from episode config where possible.
        # ExperimentConfig does not expose total_boxes directly, so we
        # rely on MIN/MAX_LEARNING_BOXES and LEARNING_RATIO as a post-hoc
        # adjustment when the buffer is finalized.
        self._learning_target = MIN_LEARNING_BOXES

    # -- Main entry point ----------------------------------------------------

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Propose a placement for *box* using the blueprint strategy.

        Phases:
            1. If still learning: add box to buffer; if buffer reaches
               target, build the blueprint.  Always place using BFD.
            2. If blueprint ready: attempt upright matching; fall back
               to BFD if no match found.

        Args:
            box:       The box to place (original dimensions).
            bin_state: Current 3D bin state (read-only).

        Returns:
            PlacementDecision(x, y, orientation_idx) or None.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        # Resolve allowed orientations for this box.
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # ---- Phase 1: Learning ----
        if self._is_learning:
            self._buffer.append(box)

            # Dynamically update the learning target based on observed count.
            # The first time we see a large episode, bump the target up a bit.
            n = len(self._buffer)
            desired = max(
                MIN_LEARNING_BOXES,
                min(MAX_LEARNING_BOXES, int(n / max(LEARNING_RATIO, 1e-9))),
            )
            # We stop learning once we have enough samples for a stable estimate
            # of the distribution, but never exceed MAX_LEARNING_BOXES.
            if n >= desired or n >= MAX_LEARNING_BOXES:
                self._blueprint = self._build_blueprint(self._buffer, bin_cfg)
                self._is_learning = False

            # Still fall back to BFD during learning (bin needs to be filled).
            return self._bfd_placement(box, orientations, bin_state, bin_cfg)

        # ---- Phase 2 / 3: Blueprint matching then BFD ----
        if self._blueprint:
            match = self._find_proxy_match(box, orientations, bin_state, bin_cfg)
            if match is not None:
                return PlacementDecision(
                    x=match[0],
                    y=match[1],
                    orientation_idx=match[2],
                )

        # BFD fallback (no blueprint yet, or no match found).
        return self._bfd_placement(box, orientations, bin_state, bin_cfg)

    # -- Blueprint construction ----------------------------------------------

    def _build_blueprint(
        self,
        sample_boxes: List[Box],
        bin_cfg,
    ) -> List[_ProxyPlacement]:
        """
        Pack the observed sample boxes in BFD order into a virtual bin to
        produce a list of proxy placements (the "blueprint").

        Algorithm:
            1. Sort sample_boxes by volume descending.
            2. Maintain a virtual heightmap (all zeros).
            3. For each box (and each orientation), scan the virtual heightmap
               for the lowest-leftmost valid position.
            4. Record each successful placement as a _ProxyPlacement.

        The resulting blueprint defines a set of (position, dimensions) slots
        that are geometrically compatible with the observed distribution.

        Args:
            sample_boxes: Boxes observed during the learning phase.
            bin_cfg:      Physical bin dimensions.

        Returns:
            List of _ProxyPlacement entries (may be empty if bin is tiny).
        """
        # Sort by volume descending (BFD ordering).
        sorted_boxes = sorted(sample_boxes, key=lambda b: b.volume, reverse=True)

        # Virtual heightmap for the blueprint construction.
        virtual_hmap = np.zeros((bin_cfg.grid_l, bin_cfg.grid_w), dtype=np.float64)
        res = bin_cfg.resolution
        step = self._scan_step
        proxy_placements: List[_ProxyPlacement] = []

        for box in sorted_boxes:
            orientations = Orientation.get_flat(box.length, box.width, box.height)
            best = self._find_blueprint_slot(
                box, orientations, virtual_hmap, bin_cfg, res, step
            )
            if best is None:
                continue
            vx, vy, vz, vol, vow, voh = best

            # Record proxy placement.
            proxy_placements.append(
                _ProxyPlacement(x=vx, y=vy, z=vz, l=vol, w=vow, h=voh)
            )

            # Update virtual heightmap.
            gx = int(round(vx / res))
            gy = int(round(vy / res))
            gx_end = min(gx + int(round(vol / res)), bin_cfg.grid_l)
            gy_end = min(gy + int(round(vow / res)), bin_cfg.grid_w)
            if gx < gx_end and gy < gy_end:
                new_top = vz + voh
                virtual_hmap[gx:gx_end, gy:gy_end] = np.maximum(
                    virtual_hmap[gx:gx_end, gy:gy_end], new_top
                )

        return proxy_placements

    def _find_blueprint_slot(
        self,
        box: Box,
        orientations: List[Tuple[float, float, float]],
        virtual_hmap: np.ndarray,
        bin_cfg,
        res: float,
        step: float,
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Find the lowest-leftmost valid position for *box* in the virtual
        heightmap used for blueprint construction.

        Returns:
            (x, y, z, ol, ow, oh) tuple or None if no position is valid.
        """
        best_z = np.inf
        best_pos = None

        for ol, ow, oh in orientations:
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue

            x = 0.0
            while x + ol <= bin_cfg.length + 1e-6:
                y = 0.0
                while y + ow <= bin_cfg.width + 1e-6:
                    # Query resting height from virtual map.
                    gx = int(round(x / res))
                    gy = int(round(y / res))
                    gx_end = min(gx + max(1, int(round(ol / res))), bin_cfg.grid_l)
                    gy_end = min(gy + max(1, int(round(ow / res))), bin_cfg.grid_w)

                    if gx >= gx_end or gy >= gy_end:
                        y += step
                        continue

                    z = float(np.max(virtual_hmap[gx:gx_end, gy:gy_end]))

                    if z + oh > bin_cfg.height + 1e-6:
                        y += step
                        continue

                    # Prefer lowest z, then leftmost x, then y.
                    if z < best_z - 1e-6 or (
                        abs(z - best_z) <= 1e-6 and best_pos is not None
                        and (x < best_pos[0] - 1e-6
                             or (abs(x - best_pos[0]) <= 1e-6
                                 and y < best_pos[1] - 1e-6))
                    ):
                        best_z = z
                        best_pos = (x, y, z, ol, ow, oh)

                    y += step
                x += step

        return best_pos

    # -- Online proxy matching -----------------------------------------------

    def _find_proxy_match(
        self,
        box: Box,
        orientations: List[Tuple[float, float, float]],
        bin_state: BinState,
        bin_cfg,
    ) -> Optional[Tuple[float, float, int]]:
        """
        Find a blueprint proxy slot that the box can fit into.

        Matching rule (upright matching):
            For each proxy slot and each box orientation, the box fits if:
                oriented_l <= proxy.l + tolerance
                oriented_w <= proxy.w + tolerance
                oriented_h <= proxy.h + tolerance

            AND the proxy position is still geometrically valid in the
            current (live) bin state:
                - In-bounds
                - Height limit not exceeded
                - Support ratio >= MIN_SUPPORT

        Returns the first valid match found (proxy list is ordered by
        blueprint construction sequence, which is BFD-ordered, so the
        first match tends to be the tightest fit).

        Args:
            box:          The box to place.
            orientations: Allowed orientations for the box.
            bin_state:    Current live bin state (read-only).
            bin_cfg:      Bin configuration.

        Returns:
            (x, y, orientation_idx) or None.
        """
        tol = PROXY_FIT_TOLERANCE

        for proxy in self._blueprint:
            for oidx, (ol, ow, oh) in enumerate(orientations):
                # Dimensional compatibility check (upright matching).
                if ol > proxy.l + tol:
                    continue
                if ow > proxy.w + tol:
                    continue
                if oh > proxy.h + tol:
                    continue

                x, y = proxy.x, proxy.y

                # Bounds check.
                if x + ol > bin_cfg.length + 1e-6:
                    continue
                if y + ow > bin_cfg.width + 1e-6:
                    continue

                # Resting height from the live bin.
                z = bin_state.get_height_at(x, y, ol, ow)

                # Height limit.
                if z + oh > bin_cfg.height + 1e-6:
                    continue

                # Support check.
                support = 1.0
                if z > 0.5:
                    support = bin_state.get_support_ratio(x, y, ol, ow, z)
                    if support < MIN_SUPPORT:
                        continue

                # Optional stricter stability.
                cfg = self.config
                if cfg.enable_stability and z > 0.5:
                    if support < cfg.min_support_ratio:
                        continue

                return (x, y, oidx)

        return None

    # -- BFD fallback --------------------------------------------------------

    def _bfd_placement(
        self,
        box: Box,
        orientations: List[Tuple[float, float, float]],
        bin_state: BinState,
        bin_cfg,
    ) -> Optional[PlacementDecision]:
        """
        Best-Fit Decreasing fallback heuristic.

        Scans a set of candidate positions (origin + corners of placed
        boxes) and scores each valid placement by:

            score = support_ratio * BFD_SCORE_SUPPORT_W
                    - (z / bin_height) * BFD_SCORE_HEIGHT_W

        Returns the highest-scoring candidate.

        Args:
            box:          The box to place.
            orientations: Allowed orientations for this box.
            bin_state:    Current live bin state (read-only).
            bin_cfg:      Bin configuration.

        Returns:
            PlacementDecision or None.
        """
        cfg = self.config

        # Gather candidate positions: origin + corners of placed boxes.
        candidates: List[Tuple[float, float]] = [(0.0, 0.0)]
        for p in bin_state.placed_boxes:
            candidates.extend([
                (p.x, p.y),
                (p.x_max, p.y),
                (p.x, p.y_max),
                (p.x_max, p.y_max),
            ])

        # Deduplicate while preserving order.
        seen = set()
        unique_candidates: List[Tuple[float, float]] = []
        for cx, cy in candidates:
            key = (round(cx, 4), round(cy, 4))
            if key not in seen:
                seen.add(key)
                unique_candidates.append((cx, cy))

        best_score: float = -np.inf
        best: Optional[Tuple[float, float, int]] = None

        for cx, cy in unique_candidates:
            for oidx, (ol, ow, oh) in enumerate(orientations):
                # Bounds check.
                if cx + ol > bin_cfg.length + 1e-6:
                    continue
                if cy + ow > bin_cfg.width + 1e-6:
                    continue

                # Resting height.
                z = bin_state.get_height_at(cx, cy, ol, ow)

                # Height limit.
                if z + oh > bin_cfg.height + 1e-6:
                    continue

                # Support check.
                support = 1.0
                if z > 0.5:
                    support = bin_state.get_support_ratio(cx, cy, ol, ow, z)
                    if support < MIN_SUPPORT:
                        continue

                # Optional stricter stability.
                if cfg.enable_stability and z > 0.5:
                    if support < cfg.min_support_ratio:
                        continue

                # Score: high support + low height.
                height_norm = z / bin_cfg.height if bin_cfg.height > 0 else 0.0
                score = (
                    BFD_SCORE_SUPPORT_W * support
                    - BFD_SCORE_HEIGHT_W * height_norm
                )

                if score > best_score:
                    best_score = score
                    best = (cx, cy, oidx)

        if best is None:
            return None

        return PlacementDecision(x=best[0], y=best[1], orientation_idx=best[2])
