"""
CODING IDEAS: Hybrid Heuristic-ML Feasibility-Guided Packing
==============================================================
Source: "Learning Practically Feasible Policies for Online 3D Bin Packing"
         Zhao et al. (2023) -- Hybrid approach combining learned policy + heuristics

PURPOSE:
  The paper's key insight is that physics-based feasibility (stability checking)
  can be combined with learned placement policies. This hybrid approach is more
  robust than pure RL or pure heuristics.

  For our thesis, this is the RECOMMENDED starting approach:
    - Use heuristic rules (DBLF, Best-Fit) for item selection and bin routing
    - Use the stacking tree for stability (physics-based feasibility)
    - Use a simpler RL policy OR scoring function for placement within a bin
    - Use MCTS to tie everything together for the buffer

WHY HYBRID:
  1. Pure RL is hard to train for multi-bin + buffer (huge state/action space)
  2. Pure heuristics miss complex packing patterns
  3. Hybrid: heuristics handle the combinatorial structure, RL handles spatial placement
  4. The feasibility mask bridges both worlds

COMPLEXITY: Lower than full RL -- no GPU training needed for heuristic components
FEASIBILITY: High -- can be implemented incrementally
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# PLACEMENT SCORING HEURISTICS (from Overview KB Section 10.5)
# =============================================================================

class PlacementRule(Enum):
    """Available placement rules from the literature."""
    DBLF = "dblf"              # Deepest-Bottom-Left-Fill (Karabulut & Inceoglu 2004)
    CORNER_DISTANCES = "cd"     # Corner Distances (Zhu & Lim 2012)
    DFTRC = "dftrc"            # Distance to Front-Top-Right Corner (Goncalves & Resende 2013)
    MIN_VERTEX = "mv"          # Minimum Vertex Coordinates (Li & Zhang 2015)
    BACK_BOTTOM = "bb"         # Back Bottom (Ramos et al. 2016)
    BEST_MATCH = "bm"          # Best Match First (Li & Zhang 2015)
    STABILITY_SCORE = "ss"     # Stability Score (Verma et al. 2020)


def score_dblf(x: float, y: float, z: float,
               bin_L: float, bin_W: float, bin_H: float) -> float:
    """
    DBLF: Deepest-Bottom-Left-Fill scoring.
    Priority: min x (deepest) -> min z (bottom) -> min y (left)
    Higher score = better placement.
    """
    return -(x * 10000 + z * 100 + y)


def score_corner_distances(x: float, y: float, z: float,
                           item_l: float, item_w: float, item_h: float,
                           bin_L: float, bin_W: float, bin_H: float) -> float:
    """
    Corner Distances: Manhattan distance to nearest container corner.
    Prefer placements near container corners.
    """
    # 8 container corners
    corners = [
        (0, 0, 0), (bin_L, 0, 0), (0, bin_W, 0), (0, 0, bin_H),
        (bin_L, bin_W, 0), (bin_L, 0, bin_H), (0, bin_W, bin_H), (bin_L, bin_W, bin_H)
    ]

    # Item corners (all 8)
    item_corners = [
        (x, y, z), (x + item_l, y, z), (x, y + item_w, z), (x, y, z + item_h),
        (x + item_l, y + item_w, z), (x + item_l, y, z + item_h),
        (x, y + item_w, z + item_h), (x + item_l, y + item_w, z + item_h)
    ]

    # Find minimum Manhattan distance across all corner pairs
    min_dist = float('inf')
    for ic in item_corners:
        for cc in corners:
            dist = abs(ic[0] - cc[0]) + abs(ic[1] - cc[1]) + abs(ic[2] - cc[2])
            min_dist = min(min_dist, dist)

    return -min_dist  # Negative because we want to MINIMIZE distance


def score_dftrc(x: float, y: float, z: float,
                item_l: float, item_w: float, item_h: float,
                bin_L: float, bin_W: float, bin_H: float) -> float:
    """
    DFTRC: Distance to Front-Top-Right Corner.
    Prefer placements that MAXIMIZE distance to front-top-right.
    This pushes items to the back-bottom-left, leaving space at front-top-right.
    """
    # Front-top-right corner of the bin
    ftr = (bin_L, bin_W, bin_H)

    # Front-top-right corner of the placed item
    item_ftr = (x + item_l, y + item_w, z + item_h)

    # Euclidean distance
    dist = np.sqrt(
        (ftr[0] - item_ftr[0]) ** 2 +
        (ftr[1] - item_ftr[1]) ** 2 +
        (ftr[2] - item_ftr[2]) ** 2
    )

    return dist  # Maximize this


def score_back_bottom(x: float, y: float, z: float,
                      bin_L: float, bin_W: float) -> float:
    """
    Back Bottom: minimal x (back), then minimal z (bottom),
    then closest to one of two back-bottom corners.
    """
    # Distance to back-bottom-left and back-bottom-right corners
    dist_left = abs(y)
    dist_right = abs(y - bin_W)
    corner_dist = min(dist_left, dist_right)

    return -(x * 10000 + z * 100 + corner_dist)


def score_stability_aware(x: float, y: float, z: float,
                          height_map: np.ndarray,
                          item_l: int, item_w: int) -> float:
    """
    Stability-aware scoring inspired by Verma et al. (2020).
    Prefer placements that create smooth, flat surfaces.
    """
    xi, yi = int(x), int(y)
    x_end = min(xi + item_l, height_map.shape[0])
    y_end = min(yi + item_w, height_map.shape[1])

    if xi >= x_end or yi >= y_end:
        return float('-inf')

    support_region = height_map[xi:x_end, yi:y_end]

    # Flatness score: how uniform is the support surface
    flatness = 1.0 / (1.0 + np.std(support_region))

    # Support coverage: what fraction of the bottom face is supported at z
    coverage = np.sum(support_region == z) / max(support_region.size, 1)

    # Height preference: lower is better
    height_penalty = z / 100.0  # Normalize by bin height

    return flatness * 0.4 + coverage * 0.4 - height_penalty * 0.2


# =============================================================================
# ITEM-BIN ROUTING HEURISTICS (for 2-bounded space)
# =============================================================================

class BinRoutingStrategy(Enum):
    """Strategies for routing items to bins in 2-bounded space."""
    FIRST_FIT = "ff"       # Place in first bin where item fits
    BEST_FIT = "bf"        # Place in bin where item fits best (smallest waste)
    WORST_FIT = "wf"       # Place in bin with most remaining space
    BALANCED = "balanced"  # Balance utilization across bins
    VOLUME_RATIO = "vr"    # Match item volume to bin remaining capacity


def route_first_fit(item_volume: float,
                    bin_utilizations: List[float],
                    bin_has_space: List[bool]) -> int:
    """First Fit: Return first bin that has space."""
    for i, has_space in enumerate(bin_has_space):
        if has_space:
            return i
    return -1  # No bin fits


def route_best_fit(item_volume: float,
                   bin_remaining_volumes: List[float],
                   bin_has_space: List[bool]) -> int:
    """Best Fit: Return bin where item leaves the least remaining space."""
    best_bin = -1
    min_remaining = float('inf')
    for i, (remaining, has_space) in enumerate(zip(bin_remaining_volumes, bin_has_space)):
        if has_space and remaining - item_volume < min_remaining and remaining >= item_volume:
            min_remaining = remaining - item_volume
            best_bin = i
    return best_bin


def route_worst_fit(item_volume: float,
                    bin_remaining_volumes: List[float],
                    bin_has_space: List[bool]) -> int:
    """Worst Fit: Return bin with the most remaining space."""
    best_bin = -1
    max_remaining = -1
    for i, (remaining, has_space) in enumerate(zip(bin_remaining_volumes, bin_has_space)):
        if has_space and remaining > max_remaining:
            max_remaining = remaining
            best_bin = i
    return best_bin


def route_balanced(bin_utilizations: List[float],
                   bin_has_space: List[bool]) -> int:
    """Balanced: Route to the bin with lower utilization."""
    best_bin = -1
    min_util = float('inf')
    for i, (util, has_space) in enumerate(zip(bin_utilizations, bin_has_space)):
        if has_space and util < min_util:
            min_util = util
            best_bin = i
    return best_bin


# =============================================================================
# ITEM SELECTION HEURISTICS (for buffer)
# =============================================================================

class ItemSelectionStrategy(Enum):
    """Strategies for selecting which item to pack from the buffer."""
    LARGEST_FIRST = "lf"   # Pack largest volume item first
    BEST_SCORING = "bs"    # Pack item with best placement score
    ARRIVAL_ORDER = "ao"   # Pack in arrival order (FIFO)
    RANDOM = "random"      # Random selection
    URGENCY = "urgency"    # Pack items that have been in buffer longest


def select_largest_first(buffer_items: List[Tuple[int, 'BoxItem']]) -> int:
    """Select the item with largest volume from the buffer."""
    if not buffer_items:
        return -1
    return max(buffer_items, key=lambda x: x[1].volume)[0]


def select_arrival_order(buffer_items: List[Tuple[int, 'BoxItem']]) -> int:
    """Select the item that arrived earliest (FIFO)."""
    if not buffer_items:
        return -1
    return min(buffer_items, key=lambda x: x[1].arrival_order)[0]


def select_best_scoring(buffer_items: List[Tuple[int, 'BoxItem']],
                        scoring_fn: Callable,
                        bin_states: List) -> int:
    """
    Select the item that achieves the best placement score
    across all bins. This evaluates every item-bin combination.
    """
    best_idx = -1
    best_score = float('-inf')

    for buf_idx, item in buffer_items:
        for bin_state in bin_states:
            score = scoring_fn(item, bin_state)
            if score > best_score:
                best_score = score
                best_idx = buf_idx

    return best_idx


# =============================================================================
# SELECTIVE HYPER-HEURISTIC (Gap 3 from Overview KB)
# =============================================================================

class SelectiveHyperHeuristic:
    """
    A selective hyper-heuristic that dynamically chooses the best
    placement rule based on the current bin state.

    From Overview KB Section 14:
    "No selective HH has been applied to 3D-PPs. This is a major opportunity,
    especially for online problems where a set of placement rules could be
    dynamically selected based on the current packing state."

    This is a NOVEL contribution opportunity for the thesis!

    The idea:
    1. Maintain a set of placement rules (DBLF, DFTRC, Corner Distances, etc.)
    2. For each packing step, evaluate the current bin state
    3. Select the placement rule most likely to perform well
    4. Selection can be learned (bandit algorithm, RL) or rule-based

    State features for selection:
    - Current utilization
    - Height map variance (flat vs uneven surface)
    - Remaining volume fraction
    - Item size relative to remaining space
    - Number of items packed so far
    """

    def __init__(self, rules: List[PlacementRule] = None):
        if rules is None:
            self.rules = list(PlacementRule)
        else:
            self.rules = rules

        # Performance tracking for each rule (for online learning)
        self.rule_successes: Dict[PlacementRule, int] = {r: 0 for r in self.rules}
        self.rule_attempts: Dict[PlacementRule, int] = {r: 0 for r in self.rules}

        # Exploration parameter for UCB-based selection
        self.exploration_weight = 0.5

    def select_rule(self, bin_state_features: Dict[str, float]) -> PlacementRule:
        """
        Select the best placement rule for the current state.

        Uses UCB (Upper Confidence Bound) to balance exploitation
        (pick the historically best rule) and exploration (try others).
        """
        total_attempts = sum(self.rule_attempts.values())

        best_rule = None
        best_ucb = float('-inf')

        for rule in self.rules:
            attempts = self.rule_attempts[rule]
            if attempts == 0:
                return rule  # Try unexplored rules first

            success_rate = self.rule_successes[rule] / attempts
            exploration = self.exploration_weight * np.sqrt(
                np.log(total_attempts) / attempts
            )
            ucb = success_rate + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_rule = rule

        return best_rule

    def update(self, rule: PlacementRule, reward: float):
        """Update the performance tracking after using a rule."""
        self.rule_attempts[rule] += 1
        self.rule_successes[rule] += reward  # reward in [0, 1]

    def select_rule_feature_based(self, bin_state_features: Dict[str, float]) -> PlacementRule:
        """
        Alternative: Rule-based selection using bin state features.

        These are heuristic rules derived from understanding each
        placement strategy's strengths:

        - DBLF: Good when bin is mostly empty (back-bottom packing)
        - Corner Distances: Good when bin is partially filled (corner consolidation)
        - DFTRC: Good for leaving space for future items
        - Stability Score: Good when surface is uneven
        - Back Bottom: Good for robot accessibility (far-to-near)
        """
        utilization = bin_state_features.get('utilization', 0.0)
        surface_variance = bin_state_features.get('height_variance', 0.0)
        item_size_ratio = bin_state_features.get('item_size_ratio', 0.5)

        if utilization < 0.3:
            # Early packing: use DBLF for systematic filling
            return PlacementRule.DBLF
        elif utilization < 0.6:
            if surface_variance > 0.3:
                # Uneven surface: prioritize stability
                return PlacementRule.STABILITY_SCORE
            else:
                # Even surface: maximize distance from front-top-right
                return PlacementRule.DFTRC
        elif utilization < 0.8:
            # Getting full: consolidate near corners
            return PlacementRule.CORNER_DISTANCES
        else:
            # Nearly full: best match to remaining spaces
            return PlacementRule.BEST_MATCH


# =============================================================================
# COMPLETE HYBRID SYSTEM
# =============================================================================

class HybridPackingSystem:
    """
    Complete hybrid system combining:
    1. Heuristic item selection from buffer
    2. Heuristic bin routing (2-bounded space)
    3. Physics-based stability (stacking tree feasibility mask)
    4. Selective hyper-heuristic for placement rule selection
    5. Optional: RL policy for placement scoring (if trained model available)

    This is a simpler alternative to the full MCTS approach
    (buffer_mcts_policy_coding_ideas.py) that may be good enough
    for the thesis while being much faster to implement.

    Expected performance:
    - Without RL: ~65-70% utilization (based on OnlineBPH numbers + stability)
    - With RL for placement: ~70-75% utilization
    - With MCTS on top: ~75-82% utilization
    """

    def __init__(self, bin_dims: Tuple[float, float, float] = (100, 100, 100),
                 buffer_size: int = 10,
                 resolution: int = 100,
                 item_strategy: str = "largest_first",
                 bin_strategy: str = "best_fit",
                 use_hyper_heuristic: bool = True):

        self.bin_dims = bin_dims
        self.buffer_size = buffer_size
        self.resolution = resolution

        # Strategy selection
        self.item_strategy = item_strategy
        self.bin_strategy = bin_strategy

        # Hyper-heuristic for placement rule selection
        self.hyper_heuristic = SelectiveHyperHeuristic() if use_hyper_heuristic else None
        self.default_rule = PlacementRule.DBLF

        # Initialize scoring functions
        self.scoring_functions = {
            PlacementRule.DBLF: score_dblf,
            PlacementRule.DFTRC: score_dftrc,
            PlacementRule.CORNER_DISTANCES: score_corner_distances,
            PlacementRule.BACK_BOTTOM: score_back_bottom,
        }

    def decide(self, buffer_items, bin_states, height_maps) -> Optional[Dict]:
        """
        Make a complete placement decision.

        Returns dict with: item_idx, bin_id, x, y, orientation, score
        Or None if no feasible placement exists.
        """
        # Step 1: Select item from buffer
        if self.item_strategy == "largest_first":
            item_idx = select_largest_first(buffer_items)
        elif self.item_strategy == "arrival_order":
            item_idx = select_arrival_order(buffer_items)
        else:
            item_idx = select_largest_first(buffer_items)  # Default

        if item_idx < 0:
            return None

        selected_item = dict(buffer_items)[item_idx]

        # Step 2: For each bin, find the best placement
        best_per_bin = []
        for bin_id, (bin_state, hmap) in enumerate(zip(bin_states, height_maps)):
            if not bin_state.get('is_active', True):
                best_per_bin.append(None)
                continue

            # Select placement rule via hyper-heuristic
            if self.hyper_heuristic:
                features = {
                    'utilization': bin_state.get('utilization', 0.0),
                    'height_variance': float(np.std(hmap)),
                    'item_size_ratio': selected_item.volume / np.prod(self.bin_dims)
                }
                rule = self.hyper_heuristic.select_rule(features)
            else:
                rule = self.default_rule

            # Find best placement using selected rule
            best_placement = self._find_best_placement(
                selected_item, hmap, rule
            )
            best_per_bin.append(best_placement)

        # Step 3: Route to best bin
        if self.bin_strategy == "best_fit":
            # Select bin where placement score is highest
            valid_bins = [(i, p) for i, p in enumerate(best_per_bin) if p is not None]
            if not valid_bins:
                return None
            bin_id, placement = max(valid_bins, key=lambda x: x[1]['score'])
        else:
            # First fit
            for i, p in enumerate(best_per_bin):
                if p is not None:
                    bin_id, placement = i, p
                    break
            else:
                return None

        return {
            'item_idx': item_idx,
            'bin_id': bin_id,
            **placement
        }

    def _find_best_placement(self, item, height_map: np.ndarray,
                               rule: PlacementRule) -> Optional[Dict]:
        """
        Find the best placement for an item on a bin using the specified rule.

        Scans feasible positions, scores each, returns the best.
        """
        L, W = height_map.shape
        best = None
        best_score = float('-inf')

        for orientation in [0, 1]:
            if orientation == 0:
                eff_l, eff_w = int(item.length), int(item.width)
            else:
                eff_l, eff_w = int(item.width), int(item.length)

            for xi in range(L - eff_l + 1):
                for yi in range(W - eff_w + 1):
                    # Get z from height map
                    z = float(np.max(height_map[xi:xi+eff_l, yi:yi+eff_w]))

                    # Check height constraint
                    if z + item.height > self.bin_dims[2]:
                        continue

                    # Score using selected rule
                    if rule == PlacementRule.DBLF:
                        score = score_dblf(xi, yi, z, *self.bin_dims)
                    elif rule == PlacementRule.DFTRC:
                        score = score_dftrc(xi, yi, z, eff_l, eff_w, item.height, *self.bin_dims)
                    elif rule == PlacementRule.CORNER_DISTANCES:
                        score = score_corner_distances(xi, yi, z, eff_l, eff_w, item.height,
                                                       *self.bin_dims)
                    elif rule == PlacementRule.BACK_BOTTOM:
                        score = score_back_bottom(xi, yi, z, self.bin_dims[0], self.bin_dims[1])
                    elif rule == PlacementRule.STABILITY_SCORE:
                        score = score_stability_aware(xi, yi, z, height_map, eff_l, eff_w)
                    else:
                        score = score_dblf(xi, yi, z, *self.bin_dims)

                    if score > best_score:
                        best_score = score
                        best = {
                            'x': xi, 'y': yi, 'z': z,
                            'orientation': orientation,
                            'score': score
                        }

        return best


# =============================================================================
# IMPLEMENTATION ROADMAP FOR THESIS
# =============================================================================

"""
RECOMMENDED IMPLEMENTATION ORDER:

Phase 1: Baseline System (Week 1-2)
  1. Implement height map representation (numpy array)
  2. Implement DBLF placement rule
  3. Implement First-Fit bin routing for 2 bins
  4. Implement FIFO item selection from buffer
  5. Test on random item sequences
  Expected: ~55-60% utilization

Phase 2: Stability Integration (Week 2-3)
  1. Implement stacking tree (stacking_tree_coding_ideas.py)
  2. Add feasibility mask computation
  3. Filter placements by stability before scoring
  4. Test stability accuracy against simple physics simulation
  Expected: ~60-65% utilization with 95%+ stability

Phase 3: Better Heuristics (Week 3-4)
  1. Implement all 6 placement rules
  2. Implement selective hyper-heuristic (UCB-based rule selection)
  3. Implement Best-Fit bin routing
  4. Implement Largest-First item selection
  5. Compare all combinations
  Expected: ~65-70% utilization

Phase 4: RL Integration (Week 4-6)
  1. Implement decomposed actor-critic (decomposed_actor_critic_coding_ideas.py)
  2. Train on single-bin BPP-1 setting
  3. Replace heuristic placement scoring with RL policy
  4. Evaluate on single-bin first, then extend to 2-bin
  Expected: ~70-75% utilization

Phase 5: MCTS Buffer Search (Week 6-8)
  1. Implement MCTS for buffer item selection
  2. Integrate RL value function as MCTS evaluation
  3. Extend MCTS to search over item + bin combinations
  4. Add parallelization
  5. Tune MCTS parameters (rollout budget, exploration constant)
  Expected: ~75-82% utilization

Phase 6: Evaluation & Writing (Week 8-10)
  1. Generate benchmark instances (RS, CUT-1, CUT-2 from paper)
  2. Run ablation study (each component on/off)
  3. Compare against baselines (OnlineBPH, boundary rule)
  4. Measure stability rates
  5. Measure computation time per decision
  6. Write thesis chapters

TOTAL ESTIMATED TIME: 10 weeks
"""
