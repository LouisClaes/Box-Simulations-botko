"""
Coding Ideas: Stochastic Blueprint Packing for Semi-Online 3D Bin Packing
===========================================================================

Based on: "Near-optimal Algorithms for Stochastic Online Bin Packing"
          by Ayyadevara, Dabas, Khan & Sreenivas (ICALP 2022, arXiv 2025)

Adapted for:
  - Semi-online setting with buffer of 5-10 boxes
  - 2-bounded space (k=2 active bins/pallets)
  - 3D bin packing with stability constraints
  - Fill rate maximization + stability goals
  - Python implementation for robotic/conveyor setup

This file contains concrete algorithm pseudocode, data structures, and
implementation guidance for adapting stochastic blueprint packing to our
thesis use case.

==========================================================================
"""

# =============================================================================
# PART 1: DATA STRUCTURES
# =============================================================================

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Deque
from collections import deque
import numpy as np
# For the actual implementation, also: from scipy import stats


@dataclass
class Box3D:
    """Represents a 3D box (item to be packed)."""
    box_id: int
    length: float      # x-dimension
    width: float        # y-dimension
    height: float       # z-dimension
    weight: float
    fragile: bool = False

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def base_area(self) -> float:
        return self.length * self.width

    @property
    def volume_normalized(self) -> float:
        """Volume as fraction of bin capacity (0,1]. Analogous to 1D 'size'."""
        # Normalize against bin volume; set bin_volume externally
        pass

    def orientations(self) -> list:
        """Return all valid orientations (up to 6 for full rotation,
        or 2-3 for vertical-axis-only rotation)."""
        l, w, h = self.length, self.width, self.height
        # Vertical-axis rotation only (robot constraint):
        return [
            (l, w, h),
            (w, l, h),
        ]


@dataclass
class Placement:
    """A box placed at a specific position and orientation in a bin."""
    box: Box3D
    x: float
    y: float
    z: float
    orientation: Tuple[float, float, float]  # (placed_l, placed_w, placed_h)


@dataclass
class Bin3D:
    """Represents a 3D bin (pallet/container)."""
    bin_id: int
    length: float       # bin x-dimension
    width: float        # bin y-dimension
    height: float       # bin z-dimension
    placements: List[Placement] = field(default_factory=list)
    is_active: bool = True
    is_closed: bool = False

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def used_volume(self) -> float:
        return sum(
            p.orientation[0] * p.orientation[1] * p.orientation[2]
            for p in self.placements
        )

    @property
    def fill_rate(self) -> float:
        return self.used_volume / self.volume if self.volume > 0 else 0.0

    @property
    def remaining_volume(self) -> float:
        return self.volume - self.used_volume


@dataclass
class EMS:
    """Empty Maximal Space -- standard 3D space management structure.
    Defined by two corner points: (x_min, y_min, z_min) and (x_max, y_max, z_max).
    """
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    @property
    def volume(self) -> float:
        return max(0, (self.x_max - self.x_min) *
                      (self.y_max - self.y_min) *
                      (self.z_max - self.z_min))

    def can_fit(self, l: float, w: float, h: float) -> bool:
        return (l <= self.x_max - self.x_min and
                w <= self.y_max - self.y_min and
                h <= self.z_max - self.z_min)


# =============================================================================
# PART 2: DISTRIBUTION LEARNING MODULE
# =============================================================================

class BoxDistributionLearner:
    """
    Learns and tracks the distribution of incoming box sizes from historical
    and live data. This is the 3D adaptation of the paper's assumption that
    items are drawn i.i.d. from an unknown distribution F.

    In practice, warehouse box sizes are NOT perfectly i.i.d., but they
    typically come from a relatively stable discrete distribution (finite
    set of SKU packaging sizes). This module tracks that distribution.

    Key insight from the paper:
    - The blueprint packing approach works as long as consecutive batches
      have "similar" distributional properties.
    - We do NOT need to know F exactly; we just need successive batches
      to be statistically similar.
    - With real data, we can validate this using the tests below.
    """

    def __init__(self, window_size: int = 500, drift_threshold: float = 0.05):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.history: Deque[Tuple[float, float, float]] = deque(maxlen=window_size)
        self.volume_history: Deque[float] = deque(maxlen=window_size)
        # For drift detection: keep two half-windows
        self._recent_half: Deque[float] = deque(maxlen=window_size // 2)
        self._older_half: Deque[float] = deque(maxlen=window_size // 2)

    def observe(self, box: Box3D):
        """Record an observed box in the distribution history."""
        dims = (box.length, box.width, box.height)
        self.history.append(dims)
        vol = box.volume
        self.volume_history.append(vol)

        # Drift detection: maintain two half-windows
        if len(self._recent_half) >= self.window_size // 2:
            oldest = self._recent_half.popleft()
            self._older_half.append(oldest)
        self._recent_half.append(vol)

    def get_volume_distribution(self) -> np.ndarray:
        """Return empirical volume distribution as sorted array."""
        return np.sort(np.array(self.volume_history))

    def estimate_large_item_fraction(self, delta: float) -> float:
        """
        Estimate the fraction of 'large' items in the distribution.

        From the paper: an item is 'large' if its size >= delta.
        In 3D, we define 'large' as volume >= delta * bin_volume.

        This fraction determines whether to use blueprint packing
        (many large items) or simple Next-Fit style (few large items).
        """
        if len(self.volume_history) == 0:
            return 0.0
        large_count = sum(1 for v in self.volume_history if v >= delta)
        return large_count / len(self.volume_history)

    def get_proxy_items(self) -> List[Tuple[float, float, float]]:
        """
        Return the 'large' items from the historical window, sorted by
        volume. These serve as proxy items for blueprint packing.

        In the paper: proxy items = large items in J1 (the blueprint batch).
        Here: proxy items = large items from recent history.
        """
        # Using the normalized volume as the sorting criterion
        items = sorted(self.history, key=lambda dims: dims[0]*dims[1]*dims[2])
        return items

    def detect_drift(self) -> bool:
        """
        Detect if the distribution has shifted significantly.

        If drift is detected, the blueprint from the old distribution
        may no longer be valid and should be recomputed.

        Uses a simple Kolmogorov-Smirnov-like test on the two half-windows.
        """
        if len(self._recent_half) < 20 or len(self._older_half) < 20:
            return False  # Not enough data

        recent = np.array(self._recent_half)
        older = np.array(self._older_half)

        # Quick KS-like statistic: max difference in empirical CDFs
        # For production, use scipy.stats.ks_2samp
        all_vals = np.sort(np.concatenate([recent, older]))
        n_recent = len(recent)
        n_older = len(older)

        max_diff = 0.0
        for v in all_vals:
            cdf_recent = np.sum(recent <= v) / n_recent
            cdf_older = np.sum(older <= v) / n_older
            max_diff = max(max_diff, abs(cdf_recent - cdf_older))

        return max_diff > self.drift_threshold

    def is_i_i_d_plausible(self) -> bool:
        """
        Test whether the i.i.d. assumption is plausible for the observed data.
        Checks:
          1. No significant autocorrelation in consecutive item volumes
          2. No significant trend
          3. Distribution is approximately stationary
        """
        if len(self.volume_history) < 50:
            return True  # Assume yes until we have enough data

        vols = np.array(self.volume_history)

        # Test 1: Autocorrelation at lag 1
        mean_v = np.mean(vols)
        var_v = np.var(vols)
        if var_v == 0:
            return True
        autocorr = np.corrcoef(vols[:-1], vols[1:])[0, 1]

        # Test 2: Simple trend check (linear regression slope)
        x = np.arange(len(vols))
        slope = np.polyfit(x, vols, 1)[0]
        relative_slope = abs(slope * len(vols) / mean_v)

        # Accept i.i.d. if autocorrelation is low and no strong trend
        return abs(autocorr) < 0.15 and relative_slope < 0.1


# =============================================================================
# PART 3: BLUEPRINT PACKING ADAPTED FOR 3D + BUFFER + k=2
# =============================================================================

class BlueprintPacker3D:
    """
    3D adaptation of the paper's blueprint packing procedure (Algorithm 2).

    Adaptations for our use case:
    1. 3D geometric packing instead of 1D weight packing
    2. Buffer of 5-10 items provides lookahead for better proxy matching
    3. k=2 bounded space: only 2 active bins at a time
    4. Stability constraints checked for each placement

    Architecture:
    - DistributionLearner feeds into this packer
    - A separate stability checker validates placements
    - The buffer manager provides the set of candidate items
    """

    def __init__(self,
                 bin_dims: Tuple[float, float, float],
                 buffer_size: int = 7,
                 k_bounded: int = 2,
                 delta: float = 0.15,
                 offline_solver=None,
                 stability_checker=None):
        """
        Args:
            bin_dims: (length, width, height) of bins
            buffer_size: number of items visible in the buffer (5-10)
            k_bounded: max number of active bins (2 for our use case)
            delta: threshold for large/small item classification
                   (as fraction of bin volume)
            offline_solver: callable that takes a list of Box3D and returns
                           a packing solution (for blueprint computation)
            stability_checker: callable that checks if a placement is stable
        """
        self.bin_dims = bin_dims
        self.bin_volume = bin_dims[0] * bin_dims[1] * bin_dims[2]
        self.buffer_size = buffer_size
        self.k_bounded = k_bounded
        self.delta = delta * self.bin_volume  # Absolute volume threshold
        self.offline_solver = offline_solver
        self.stability_checker = stability_checker

        # State
        self.distribution_learner = BoxDistributionLearner()
        self.active_bins: List[Bin3D] = []
        self.closed_bins: List[Bin3D] = []
        self.buffer: List[Box3D] = []
        self.blueprint: Optional[Dict] = None
        self.proxy_items: List[Box3D] = []  # Unmatched proxy items
        self.items_seen: int = 0
        self.stage_size: int = 50  # delta^2 * n equivalent
        self.current_stage_items: List[Box3D] = []
        self.previous_stage_items: List[Box3D] = []

    def is_large(self, box: Box3D) -> bool:
        """Classify item as large or small (paper's delta threshold)."""
        return box.volume >= self.delta

    def add_to_buffer(self, box: Box3D):
        """Add a new box to the buffer (conveyor arrival)."""
        self.buffer.append(box)
        self.distribution_learner.observe(box)
        self.items_seen += 1

    def select_and_pack(self) -> Optional[Placement]:
        """
        Select the best box from the buffer and pack it.

        This is where the buffer advantage over the paper's model is
        exploited: instead of being forced to pack the next arriving item,
        we can choose the BEST item from the buffer.

        Strategy (adapted from blueprint packing):
        1. If we have a blueprint, score each buffer item by how well it
           matches available proxy slots or S-slots.
        2. Pick the buffer item with the best match score.
        3. Place it according to the blueprint mapping.

        Without a blueprint (early stage / sampling stage):
        - Use Best-Fit or DBLF heuristic directly.
        """
        if not self.buffer:
            return None

        if not self.active_bins:
            self._open_new_bin()

        if self.blueprint is not None and self.proxy_items:
            return self._pack_with_blueprint()
        else:
            return self._pack_greedy_with_buffer()

    def _pack_with_blueprint(self) -> Optional[Placement]:
        """
        Blueprint-guided packing with buffer lookahead.

        For each item in the buffer, compute a 'blueprint match score':
        - For large items: how well does it match any available proxy item?
          Score = 1 / (proxy_volume - item_volume + epsilon) if proxy >= item
          Higher score = tighter fit (less wasted space)
        - For small items: how well does it fit in available S-slots?
          Score based on remaining EMS capacity after placement.

        Select the buffer item with the highest score.

        CRITICAL ADAPTATION FOR k=2:
        Since we only have 2 active bins, we must be strategic about
        which bin to use. The blueprint maps items to specific bins,
        but with k=2 we may need to close a bin before its blueprint
        is fully realized. The strategy is:
        - Prioritize filling the more-complete active bin first
        - Only switch focus when the current bin's blueprint targets
          are likely exhausted from the buffer
        """
        best_score = -1.0
        best_item_idx = -1
        best_placement_info = None

        for i, box in enumerate(self.buffer):
            if self.is_large(box):
                # Find best proxy match
                score, proxy_info = self._score_large_item_blueprint(box)
            else:
                # Find best S-slot match
                score, proxy_info = self._score_small_item_blueprint(box)

            if score > best_score:
                best_score = score
                best_item_idx = i
                best_placement_info = proxy_info

        if best_item_idx >= 0 and best_placement_info is not None:
            box = self.buffer.pop(best_item_idx)
            return self._execute_placement(box, best_placement_info)

        # Fallback to greedy
        return self._pack_greedy_with_buffer()

    def _score_large_item_blueprint(self, box: Box3D) -> Tuple[float, Optional[dict]]:
        """
        Score a large item against available proxy items.

        Paper's approach (1D): find smallest proxy d >= x.
        3D adaptation: find proxy whose occupied volume is closest to
        (but >= ) the box's volume, AND whose 3D footprint is compatible.
        """
        best_score = -1.0
        best_proxy_info = None

        for j, proxy in enumerate(self.proxy_items):
            # Check 3D compatibility: can box fit in proxy's allocated space?
            # This requires checking the EMS where the proxy was placed
            # in the blueprint packing.
            proxy_volume = proxy.volume
            box_volume = box.volume

            if proxy_volume >= box_volume * 0.8:
                # Volume compatibility (relaxed vs paper's strict >= )
                # Tighter match = higher score
                waste = proxy_volume - box_volume
                score = 1.0 / (waste + 0.01 * self.bin_volume)

                # Bonus for 3D shape compatibility
                # (proxy and box have similar aspect ratios)
                shape_similarity = self._shape_similarity(box, proxy)
                score *= (1.0 + shape_similarity)

                if score > best_score:
                    best_score = score
                    best_proxy_info = {'proxy_idx': j, 'type': 'large_proxy'}

        return best_score, best_proxy_info

    def _score_small_item_blueprint(self, box: Box3D) -> Tuple[float, Optional[dict]]:
        """
        Score a small item against available S-slots (EMSs in active bins).

        Paper's approach: pack into S-slots via Next-Fit.
        3D adaptation: find the EMS with the best fit for this box
        across all active bins.
        """
        best_score = -1.0
        best_info = None

        for bin_obj in self.active_bins:
            # Get EMSs for this bin
            ems_list = self._compute_ems(bin_obj)
            for ems_idx, ems in enumerate(ems_list):
                for orientation in box.orientations():
                    l, w, h = orientation
                    if ems.can_fit(l, w, h):
                        # Score: prefer tight fits (less wasted EMS volume)
                        remaining = ems.volume - (l * w * h)
                        score = 1.0 / (remaining + 0.01 * self.bin_volume)

                        # Stability bonus
                        if self.stability_checker:
                            stability_score = self._check_stability_score(
                                bin_obj, box, ems, orientation
                            )
                            score *= (1.0 + stability_score)

                        # DBLF preference: prefer deeper, bottom, left
                        dblf_score = self._dblf_priority(ems)
                        score *= (1.0 + 0.5 * dblf_score)

                        if score > best_score:
                            best_score = score
                            best_info = {
                                'bin_id': bin_obj.bin_id,
                                'ems_idx': ems_idx,
                                'orientation': orientation,
                                'type': 'small_sslot'
                            }

        return best_score, best_info

    def _pack_greedy_with_buffer(self) -> Optional[Placement]:
        """
        Greedy packing when no blueprint is available.

        Strategy: For each item in the buffer, evaluate all possible
        placements in all active bins. Select the (item, bin, position,
        orientation) tuple with the best score.

        Score combines:
        - Fill rate improvement
        - Stability score
        - DBLF priority (deepest-bottom-left-fill)
        - Future flexibility (how much usable EMS remains)

        This is essentially a multi-item Best-Match-First approach,
        adapted from Ha et al. (2017) but with buffer lookahead.
        """
        best_score = -1.0
        best_item_idx = -1
        best_placement = None

        for i, box in enumerate(self.buffer):
            for bin_obj in self.active_bins:
                ems_list = self._compute_ems(bin_obj)
                for ems_idx, ems in enumerate(ems_list):
                    for orientation in box.orientations():
                        l, w, h = orientation
                        if ems.can_fit(l, w, h):
                            score = self._compute_placement_score(
                                bin_obj, box, ems, orientation
                            )
                            if score > best_score:
                                best_score = score
                                best_item_idx = i
                                best_placement = {
                                    'bin_id': bin_obj.bin_id,
                                    'position': (ems.x_min, ems.y_min, ems.z_min),
                                    'orientation': orientation
                                }

        if best_item_idx >= 0 and best_placement is not None:
            box = self.buffer.pop(best_item_idx)
            return self._execute_placement(box, best_placement)

        # No valid placement found in active bins; manage k=2 constraint
        return self._handle_no_fit()

    def _handle_no_fit(self) -> Optional[Placement]:
        """
        Handle the case where no buffer item fits in any active bin.

        With k=2 bounded space:
        1. Close the bin with the highest fill rate (it's "done").
        2. Open a new bin.
        3. Try placing again.

        This is the critical k=2 decision: WHEN to close a bin.

        Informed by distribution: if we know the upcoming items are
        likely to be large and neither bin can fit them, close the
        less promising bin.
        """
        if len(self.active_bins) >= self.k_bounded:
            # Close the bin with highest fill rate (most complete)
            best_fill = -1
            close_idx = 0
            for idx, b in enumerate(self.active_bins):
                if b.fill_rate > best_fill:
                    best_fill = b.fill_rate
                    close_idx = idx

            closed_bin = self.active_bins.pop(close_idx)
            closed_bin.is_active = False
            closed_bin.is_closed = True
            self.closed_bins.append(closed_bin)

        # Open new bin
        self._open_new_bin()

        # Try greedy packing again (recursive but bounded depth 1)
        return self._pack_greedy_with_buffer()

    def _open_new_bin(self):
        """Open a new active bin."""
        new_id = len(self.active_bins) + len(self.closed_bins)
        new_bin = Bin3D(
            bin_id=new_id,
            length=self.bin_dims[0],
            width=self.bin_dims[1],
            height=self.bin_dims[2],
        )
        self.active_bins.append(new_bin)

    def update_blueprint(self):
        """
        Recompute the blueprint using the latest stage data.

        This is called at stage boundaries (every stage_size items).

        Paper's approach: at end of stage T_{j-1}, compute A_alpha(T_{j-1})
        and use as blueprint for T_j.

        3D adaptation: run offline 3D solver on previous stage's items
        to get a high-quality packing, then extract proxy mappings.
        """
        if not self.previous_stage_items or self.offline_solver is None:
            return

        # Run offline solver on previous stage
        offline_packing = self.offline_solver(self.previous_stage_items)

        # Extract blueprint:
        # - For each bin in offline solution, identify large items (proxies)
        #   and the EMS (S-slots) left after removing small items
        self.proxy_items = []
        self.blueprint = {'bins': [], 's_slots': []}

        for bin_packing in offline_packing:
            large_items = [item for item in bin_packing if self.is_large(item)]
            small_weight = sum(item.volume for item in bin_packing
                              if not self.is_large(item))
            self.proxy_items.extend(large_items)
            self.blueprint['bins'].append({
                'large_items': large_items,
                's_slot_volume': small_weight,
            })

    def process_stage_boundary(self):
        """
        Called when stage_size items have been processed.
        Shift current stage to previous, prepare new stage, update blueprint.
        """
        self.previous_stage_items = list(self.current_stage_items)
        self.current_stage_items = []
        self.update_blueprint()

    # ---- Helper methods (to be implemented) ----

    def _compute_ems(self, bin_obj: Bin3D) -> List[EMS]:
        """Compute Empty Maximal Spaces for a bin. Standard 3D-PP operation."""
        # Implementation: track EMSs incrementally as items are placed
        # See: Parreno et al. (2008) for the EMS algorithm
        raise NotImplementedError("Implement EMS computation")

    def _shape_similarity(self, box1: Box3D, box2: Box3D) -> float:
        """Compute shape similarity between two boxes (0 to 1)."""
        dims1 = sorted([box1.length, box1.width, box1.height])
        dims2 = sorted([box2.length, box2.width, box2.height])
        ratios = [min(a, b) / max(a, b) if max(a, b) > 0 else 1.0
                  for a, b in zip(dims1, dims2)]
        return sum(ratios) / 3.0

    def _check_stability_score(self, bin_obj, box, ems, orientation) -> float:
        """Compute stability score for a placement (0 to 1)."""
        # Delegate to stability_checker module
        raise NotImplementedError("Implement stability scoring")

    def _dblf_priority(self, ems: EMS) -> float:
        """DBLF priority score: prefer deeper (min x), bottom (min z), left (min y)."""
        # Normalize to [0, 1] range based on bin dimensions
        x_score = 1.0 - ems.x_min / self.bin_dims[0]
        z_score = 1.0 - ems.z_min / self.bin_dims[2]
        y_score = 1.0 - ems.y_min / self.bin_dims[1]
        return 0.5 * x_score + 0.3 * z_score + 0.2 * y_score

    def _compute_placement_score(self, bin_obj, box, ems, orientation) -> float:
        """Combined placement score for greedy mode."""
        l, w, h = orientation
        volume_used = l * w * h
        volume_ratio = volume_used / ems.volume  # Tighter fit is better
        dblf = self._dblf_priority(ems)
        # TODO: add stability score
        return 0.6 * volume_ratio + 0.4 * dblf

    def _execute_placement(self, box: Box3D, info: dict) -> Optional[Placement]:
        """Execute a placement decision."""
        # Implementation depends on the placement info type
        raise NotImplementedError("Implement placement execution")


# =============================================================================
# PART 4: UPRIGHT MATCHING FOR 3D PROXY ASSIGNMENT
# =============================================================================

class UprightMatcher3D:
    """
    Adaptation of the paper's upright matching (Algorithm 1) for 3D items.

    In 1D: match items by size (proxy size >= real item size).
    In 3D: match items by volume AND shape compatibility.

    The upright matching problem:
    - Plus points: real items arriving online, represented as (arrival_order, volume)
    - Minus points: proxy items from the blueprint, represented as (blueprint_order, volume)
    - Match: real item to proxy item where proxy volume >= real item volume
    - Objective: maximize the number of matched pairs

    For 3D, we extend this with a secondary shape compatibility check.
    """

    @staticmethod
    def maximum_upright_matching(
        proxy_items: List[Box3D],
        real_items: List[Box3D],
    ) -> List[Tuple[int, int]]:
        """
        Compute maximum upright matching between proxy and real items.

        This is Algorithm 1 from the paper, adapted for 3D.

        Args:
            proxy_items: list of proxy items from blueprint (minus points)
            real_items: list of real items arriving online (plus points)

        Returns:
            List of (real_item_idx, proxy_item_idx) pairs

        Paper guarantees: at most O(sqrt(m) * (log m)^{3/4}) unmatched
        items when items are i.i.d.
        """
        # Create points: minus = (order, volume) for proxies
        #                plus  = (order, volume) for real items
        points = []  # (x_coord, y_coord, is_plus, original_index)

        for i, proxy in enumerate(proxy_items):
            points.append((i, proxy.volume, False, i))

        for i, real in enumerate(real_items):
            points.append((len(proxy_items) + i, real.volume, True, i))

        # Sort by x-coordinate (order)
        points.sort(key=lambda p: p[0])

        # Unmatched minus points, sorted by y-coordinate (volume)
        # Use a sorted structure for efficient lookup
        import sortedcontainers
        unmatched_minus = sortedcontainers.SortedList(key=lambda p: p[1])
        matches = []

        for x, y, is_plus, orig_idx in points:
            if not is_plus:
                # Minus point (proxy): add to unmatched set
                unmatched_minus.add((x, y, orig_idx))
            else:
                # Plus point (real item): find best match
                # Need minus point with y_minus <= y (proxy volume >= real volume)
                # Among those, pick the one with maximum y_minus (tightest fit)
                candidates = [p for p in unmatched_minus if p[1] >= y]
                if candidates:
                    # Find the one with minimum y (smallest proxy that fits)
                    # This is the paper's "smallest proxy d >= x" rule
                    best = min(candidates, key=lambda p: p[1])
                    matches.append((orig_idx, best[2]))
                    unmatched_minus.remove(best)

        return matches

    @staticmethod
    def online_upright_matching_with_buffer(
        proxy_items: List[Box3D],
        buffer: List[Box3D],
    ) -> Optional[Tuple[int, int]]:
        """
        Online version: given the current buffer and available proxies,
        find the best (buffer_item, proxy) pair to match next.

        This is where the BUFFER ADVANTAGE is exploited:
        - The paper processes items sequentially (no choice).
        - With a buffer, we choose which item to match first.
        - We can pick the buffer item with the tightest proxy match,
          leaving more flexible proxies for future (unknown) items.

        Strategy: Match the buffer item whose best proxy match leaves
        the MOST flexibility for remaining items. This is a greedy
        heuristic inspired by the paper's matching but enhanced by choice.
        """
        if not proxy_items or not buffer:
            return None

        best_score = -1.0
        best_pair = None

        for buf_idx, real_box in enumerate(buffer):
            real_vol = real_box.volume
            # Find smallest proxy that fits
            fitting_proxies = [(p_idx, p) for p_idx, p in enumerate(proxy_items)
                               if p.volume >= real_vol]

            if fitting_proxies:
                # Smallest fitting proxy = tightest match
                best_proxy_idx, best_proxy = min(fitting_proxies,
                                                  key=lambda x: x[1].volume)
                waste = best_proxy.volume - real_vol
                # Score: prefer tight matches (low waste)
                score = 1.0 / (waste + 0.001)

                if score > best_score:
                    best_score = score
                    best_pair = (buf_idx, best_proxy_idx)

        return best_pair


# =============================================================================
# PART 5: DOUBLING TRICK FOR UNKNOWN STREAM LENGTH
# =============================================================================

class DoublingTrickManager:
    """
    Implementation of the paper's "ImpAlg" doubling trick (Algorithm 4)
    for handling unknown input length.

    When we don't know how many items will arrive (realistic in a warehouse),
    we use geometrically increasing "super-stages":
    - Gamma_0: n0 items (initial guess)
    - Gamma_1: mu * n0 items
    - Gamma_2: mu * (1+mu) * n0 items
    - etc.

    At each super-stage boundary, re-estimate the distribution and
    recompute the blueprint.
    """

    def __init__(self, n0: int = 50, mu: float = 0.25):
        """
        Args:
            n0: Initial guess for stream length (1/delta^3 in paper)
            mu: Growth factor for super-stages (delta^2 in paper)
        """
        self.n0 = n0
        self.mu = mu
        self.current_stage = 0
        self.items_in_current_stage = 0
        self.current_stage_size = n0
        self.total_items = 0
        self.n_values = [n0]  # n_0, n_1, n_2, ...

    def item_arrived(self) -> bool:
        """
        Called when a new item arrives.
        Returns True if a stage boundary has been reached.
        """
        self.total_items += 1
        self.items_in_current_stage += 1

        if self.items_in_current_stage >= self.current_stage_size:
            return True
        return False

    def advance_stage(self):
        """Move to the next super-stage."""
        self.current_stage += 1
        next_n = int(self.n_values[-1] * (1 + self.mu))
        self.n_values.append(next_n)
        self.current_stage_size = next_n - self.n_values[-2]
        self.items_in_current_stage = 0

    def is_sampling_stage(self) -> bool:
        """Are we in the initial sampling stage?"""
        return self.current_stage == 0

    def get_current_stage_fraction(self) -> float:
        """What fraction of the estimated total is the current stage?"""
        estimated_total = self.n_values[-1]
        return self.current_stage_size / estimated_total


# =============================================================================
# PART 6: MAIN ORCHESTRATOR -- COMBINING ALL COMPONENTS
# =============================================================================

class StochasticBlueprintOrchestrator:
    """
    Main class that orchestrates the complete semi-online packing system.

    Combines:
    1. Distribution learning (from paper's i.i.d. assumption)
    2. Blueprint packing (from paper's Algorithms 2-4)
    3. Buffer management (our extension beyond the paper)
    4. k=2 bounded space management (our constraint)
    5. Stability checking (our additional constraint)

    Lifecycle:
    1. SAMPLING PHASE: first n0 items packed greedily while learning distribution
    2. BLUEPRINT PHASE: subsequent stages use blueprint packing
    3. DRIFT DETECTION: if distribution changes, reset and re-learn

    For the thesis:
    - Compare this approach against:
      a. Pure greedy (no distribution learning)
      b. Pure RL (no theoretical guarantees)
      c. Hybrid approaches (RL + heuristic)
    - Measure: fill rate, stability score, computational time
    """

    def __init__(self,
                 bin_dims: Tuple[float, float, float],
                 buffer_size: int = 7,
                 offline_solver=None,
                 stability_checker=None):

        self.packer = BlueprintPacker3D(
            bin_dims=bin_dims,
            buffer_size=buffer_size,
            k_bounded=2,
            offline_solver=offline_solver,
            stability_checker=stability_checker,
        )
        self.doubling = DoublingTrickManager(n0=50, mu=0.25)
        self.mode = 'sampling'  # 'sampling' or 'blueprint'
        self.results: List[Placement] = []

    def receive_item(self, box: Box3D) -> Optional[Placement]:
        """
        Main entry point: receive a new box from the conveyor.

        Returns the placement if one was made, or None if the item
        was added to the buffer and nothing was placed yet.
        """
        # Add to buffer and distribution learner
        self.packer.add_to_buffer(box)

        # Check if buffer is full enough to make a placement decision
        if len(self.packer.buffer) < min(3, self.packer.buffer_size):
            return None  # Wait for more items in buffer

        # Check stage boundaries
        if self.doubling.item_arrived():
            self._handle_stage_boundary()

        # Check for distribution drift
        if self.packer.distribution_learner.detect_drift():
            self._handle_drift()

        # Select and pack
        placement = self.packer.select_and_pack()
        if placement:
            self.results.append(placement)
        return placement

    def _handle_stage_boundary(self):
        """Handle transition between stages."""
        if self.doubling.is_sampling_stage():
            # Transition from sampling to blueprint mode
            large_frac = self.packer.distribution_learner.estimate_large_item_fraction(
                self.packer.delta / self.packer.bin_volume
            )
            if large_frac > 0.1:  # Enough large items to benefit from blueprint
                self.mode = 'blueprint'
                self.packer.process_stage_boundary()
            # else: stay in greedy/NF mode (paper's "few large items" case)

        self.doubling.advance_stage()

        if self.mode == 'blueprint':
            self.packer.process_stage_boundary()

    def _handle_drift(self):
        """Handle detected distribution drift."""
        # Reset the blueprint and rebuild from recent data
        self.packer.blueprint = None
        self.packer.proxy_items = []
        # Will be rebuilt at next stage boundary

    def get_statistics(self) -> dict:
        """Return packing statistics for evaluation."""
        all_bins = self.packer.active_bins + self.packer.closed_bins
        return {
            'total_bins': len(all_bins),
            'total_items': self.packer.items_seen,
            'avg_fill_rate': (
                np.mean([b.fill_rate for b in all_bins]) if all_bins else 0
            ),
            'min_fill_rate': (
                min(b.fill_rate for b in all_bins) if all_bins else 0
            ),
            'max_fill_rate': (
                max(b.fill_rate for b in all_bins) if all_bins else 0
            ),
            'mode': self.mode,
            'distribution_iid': (
                self.packer.distribution_learner.is_i_i_d_plausible()
            ),
            'current_stage': self.doubling.current_stage,
        }


# =============================================================================
# PART 7: INTEGRATION POINTS WITH OTHER METHODS
# =============================================================================
"""
Integration with other modules in the thesis project:

1. STABILITY MODULE (stability/):
   - The stability_checker parameter accepts any function that takes
     (bin_state, box, position, orientation) and returns a stability score.
   - Integrate with physics-based or rule-based stability checkers.
   - See: stability/ folder for implementations.

2. DEEP RL MODULE (deep_rl/):
   - The offline_solver can be replaced with a trained RL agent that
     produces high-quality offline packings (for blueprint computation).
   - The buffer selection policy (which item to pick from buffer) could
     be learned via RL instead of the greedy heuristic above.
   - Key insight: RL learns WHICH item to pack next; blueprint packing
     tells WHERE to pack it.

3. HEURISTICS MODULE (heuristics/):
   - DBLF, Corner Distances, DFTRC placement rules can be used as the
     fallback when no blueprint is available.
   - Best-Fit (from the paper's random-order analysis) is the natural
     baseline heuristic.

4. HYPER-HEURISTICS (hyper_heuristics/):
   - The distribution learner can inform a hyper-heuristic selector:
     switch between placement rules based on the observed distribution.
   - E.g., use tight-fit heuristics when items are mostly large,
     use fill-maximizing heuristics when items are mostly small.

5. MULTI-BIN (multi_bin/):
   - The k=2 bounded space constraint is managed here.
   - The bin-closing decision (which bin to close when both are full)
     could be informed by the distribution: if large items are expected,
     close the bin that's harder to fit large items into.

6. THEORETICAL BOUNDS (theoretical_bounds/):
   - The (1+epsilon) competitive ratio from this paper provides a
     LOWER BOUND on achievable fill rate for i.i.d. distributions.
   - Use the stochastic_bounds_estimator to compute expected performance
     for your specific distribution.
"""


# =============================================================================
# PART 8: COMPLEXITY AND FEASIBILITY ANALYSIS
# =============================================================================
"""
COMPUTATIONAL COMPLEXITY:

1. Distribution Learning:
   - observe(): O(1) per item
   - detect_drift(): O(window_size) per check (can be amortized)
   - estimate_large_fraction(): O(window_size) per check

2. Blueprint Computation (offline solver):
   - Depends on the chosen offline solver
   - AFPTAS: O(C_eps + n log 1/eps) -- very fast for practical n
   - MFFD: O(n log n) -- even faster
   - For 3D offline: typically NP-hard, but good heuristics exist
   - Called once per stage (every ~50-200 items), NOT per item

3. Buffer Selection:
   - O(buffer_size * |active_bins| * |EMS_per_bin| * |orientations|)
   - With buffer=7, k=2, ~10 EMSs per bin, 2 orientations:
     O(7 * 2 * 10 * 2) = O(280) per item -- very fast

4. Upright Matching:
   - O(m log m) for m items using sorted structures
   - Only needed for large items (fraction of total)

5. Overall: O(n * (buffer_ops + stage_ops/stage_size))
   - Practical runtime: well under 100ms per item for typical parameters
   - Suitable for real-time conveyor/robotic operations

FEASIBILITY ASSESSMENT:

- HIGH feasibility for the distribution learning component
  (simple statistics, minimal overhead)
- HIGH feasibility for the buffer selection with greedy scoring
- MEDIUM feasibility for the full blueprint approach in 3D
  (3D proxy matching is more complex than 1D)
- LOW-MEDIUM feasibility for the theoretical guarantees in 3D
  (the paper's proofs rely on 1D structure; 3D adaptation is heuristic)

RECOMMENDED IMPLEMENTATION ORDER:
1. Distribution learner + drift detection (1-2 days)
2. Buffer-enhanced greedy packing with DBLF (2-3 days)
3. 1D blueprint packing prototype (2-3 days)
4. 3D blueprint packing with EMS management (1-2 weeks)
5. Integration with stability checker (1 week)
6. k=2 bounded space management (2-3 days)
7. Doubling trick for unknown stream length (1-2 days)
8. Evaluation against baselines (1 week)
"""
