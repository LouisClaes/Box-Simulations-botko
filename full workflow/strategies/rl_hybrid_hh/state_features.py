"""
State feature extraction for the RL Hybrid Hyper-Heuristic.

Converts the raw packing state (BinState, Box, conveyor info) into a compact
handcrafted feature vector of ~39 dimensions.  This is the observation that
the Q-learning agent uses to decide which heuristic to apply.

Design rationale for each feature group:

1. Per-bin features (2 bins x 8 = 16 features):
   These capture the spatial state of each pallet.  The agent needs to know
   how full each bin is, how smooth/rough the surface is, and whether there
   are large gaps available -- different heuristics excel in different states.

2. Current box features (5 features):
   The box's shape and size strongly influence which heuristic works best.
   Small boxes benefit from gap-filling heuristics; large boxes benefit
   from surface-contact maximisation.

3. Buffer features (4 features):
   The upcoming boxes on the conveyor affect the value of the current
   placement.  If many similar boxes are coming, layer-building is good.
   If diverse boxes are coming, flexible heuristics are better.

4. Episode progress (3 features):
   Early, mid, and late game require different strategies.  Early: build
   foundation.  Mid: maximise density.  Late: fill gaps.

5. History features (8 features):
   Recent heuristic choices and their success rate help the agent learn
   temporal patterns.  If a heuristic has been failing, try another.

6. Packing phase indicators (3 features):
   Binary indicators for early/mid/late phase provide explicit phase
   information that makes learning easier (helps the network specialise).

All features are normalised to approximately [0, 1] for stable training.
"""

from __future__ import annotations

import sys
import os
from typing import List, Optional, Tuple
from collections import deque

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

from config import Box, BinConfig
from simulator.bin_state import BinState
from strategies.rl_hybrid_hh.config import HHConfig


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Maximum expected number of boxes per bin (for normalisation)
MAX_BOXES_PER_BIN: int = 80

# Maximum expected surface roughness (for normalisation)
MAX_ROUGHNESS: float = 200.0

# Phase thresholds (fill rate)
EARLY_PHASE_THRESHOLD: float = 0.3
LATE_PHASE_THRESHOLD: float = 0.6


# ─────────────────────────────────────────────────────────────────────────────
# Feature Tracker (maintains history across steps)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureTracker:
    """
    Maintains running state for features that depend on history.

    This tracker persists across steps within an episode and provides
    features like recent heuristic choices, success rates, and counters.

    Attributes:
        heuristic_history: Last N heuristic choices (as action indices).
        success_history:   Last M placement outcomes (True/False).
        total_placed:      Total boxes placed this episode.
        total_rejected:    Total boxes rejected/skipped this episode.
        pallets_closed:    Number of pallets closed this episode.
    """

    def __init__(self, config: HHConfig) -> None:
        self._config = config
        self.heuristic_history: deque = deque(maxlen=10)
        self.success_history: deque = deque(maxlen=10)
        self.total_placed: int = 0
        self.total_rejected: int = 0
        self.pallets_closed: int = 0
        self.total_boxes_in_episode: int = 0

    def reset(self, total_boxes: int = 100) -> None:
        """Reset at the start of a new episode."""
        self.heuristic_history.clear()
        self.success_history.clear()
        self.total_placed = 0
        self.total_rejected = 0
        self.pallets_closed = 0
        self.total_boxes_in_episode = total_boxes

    def record_choice(self, action: int, success: bool) -> None:
        """Record a heuristic selection and its outcome."""
        self.heuristic_history.append(action)
        self.success_history.append(success)
        if success:
            self.total_placed += 1
        else:
            self.total_rejected += 1

    def record_pallet_close(self) -> None:
        """Record that a pallet was closed."""
        self.pallets_closed += 1

    @property
    def recent_success_rate(self) -> float:
        """Success rate of the last 10 placements."""
        if len(self.success_history) == 0:
            return 1.0  # Optimistic start
        return sum(self.success_history) / len(self.success_history)

    @property
    def boxes_processed(self) -> int:
        """Total boxes processed (placed + rejected)."""
        return self.total_placed + self.total_rejected


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction Functions
# ─────────────────────────────────────────────────────────────────────────────

def extract_bin_features(
    bin_state: BinState,
    bin_config: BinConfig,
) -> np.ndarray:
    """
    Extract 8 features from a single bin/pallet state.

    Features:
        0. fill_rate:           Volumetric fill rate [0, 1].
           Rationale: Primary indicator of packing progress.  High fill
           means the agent should prefer gap-filling heuristics.

        1. max_height_norm:     Peak height / bin height [0, 1].
           Rationale: Indicates vertical usage.  Near-max heights
           suggest the bin is nearly full vertically.

        2. roughness_norm:      Surface roughness normalised [0, 1].
           Rationale: Rough surfaces need smoothing heuristics;
           flat surfaces can accept any box easily.

        3. num_boxes_norm:      Number of placed boxes / MAX [0, 1].
           Rationale: More boxes means more fragmented space;
           different heuristics handle this differently.

        4. avg_height_norm:     Mean heightmap value / bin height [0, 1].
           Rationale: Unlike max_height, this shows overall usage.
           Low avg but high max indicates peaks/valleys.

        5. height_variance:     Variance of heightmap / bin_height^2 [0, 1].
           Rationale: High variance means an uneven surface --
           wall-e scoring or surface contact heuristics help.

        6. coverage_ratio:      Fraction of floor area with height > 0 [0, 1].
           Rationale: Low coverage with low fill means spread-out
           placements; high coverage means dense packing.

        7. largest_gap_ratio:   Estimated largest contiguous gap / area [0, 1].
           Rationale: Large gaps favour big-box heuristics;
           only small gaps favour gap-fillers.

    Args:
        bin_state:  Current state of one bin.
        bin_config: Physical dimensions of the bin.

    Returns:
        np.ndarray of shape (8,), dtype float32.
    """
    hm = bin_state.heightmap
    features = np.zeros(8, dtype=np.float32)

    # 0. Fill rate
    features[0] = bin_state.get_fill_rate()

    # 1. Max height normalised
    max_h = bin_state.get_max_height()
    features[1] = max_h / bin_config.height if bin_config.height > 0 else 0.0

    # 2. Surface roughness normalised
    roughness = bin_state.get_surface_roughness()
    features[2] = min(roughness / MAX_ROUGHNESS, 1.0)

    # 3. Number of boxes normalised
    features[3] = min(len(bin_state.placed_boxes) / MAX_BOXES_PER_BIN, 1.0)

    # 4. Average height normalised
    avg_h = float(np.mean(hm))
    features[4] = avg_h / bin_config.height if bin_config.height > 0 else 0.0

    # 5. Height variance (normalised by bin_height^2)
    h_var = float(np.var(hm))
    max_var = bin_config.height ** 2
    features[5] = min(h_var / max_var, 1.0) if max_var > 0 else 0.0

    # 6. Coverage ratio (fraction of cells with height > 0)
    total_cells = hm.size
    covered_cells = int(np.sum(hm > 0))
    features[6] = covered_cells / total_cells if total_cells > 0 else 0.0

    # 7. Largest gap ratio (approximated from zero-height cells)
    #    We compute the fraction of cells at the floor level (height < resolution)
    #    as a proxy for available gap space.
    floor_cells = int(np.sum(hm < bin_config.resolution))
    features[7] = floor_cells / total_cells if total_cells > 0 else 0.0

    return features


def extract_box_features(
    box: Box,
    bin_config: BinConfig,
) -> np.ndarray:
    """
    Extract 5 features from the current box.

    Features:
        0. l_norm:       Length / max_dim [0, 1].
        1. w_norm:       Width / max_dim [0, 1].
        2. h_norm:       Height / max_dim [0, 1].
        3. vol_norm:     Volume / bin_volume [0, 1].
        4. aspect_ratio: max(l,w,h) / min(l,w,h) normalised [0, 1].
           Rationale: Elongated boxes benefit from alignment-aware
           heuristics; cubic boxes can be placed more flexibly.

    Args:
        box:        The current box to place.
        bin_config: Bin dimensions for normalisation.

    Returns:
        np.ndarray of shape (5,), dtype float32.
    """
    max_dim = max(bin_config.length, bin_config.width, bin_config.height)
    features = np.zeros(5, dtype=np.float32)

    features[0] = box.length / max_dim if max_dim > 0 else 0.0
    features[1] = box.width / max_dim if max_dim > 0 else 0.0
    features[2] = box.height / max_dim if max_dim > 0 else 0.0
    features[3] = box.volume / bin_config.volume if bin_config.volume > 0 else 0.0

    dims = [box.length, box.width, box.height]
    min_d = min(dims)
    max_d = max(dims)
    # Normalise aspect ratio: 1.0 (cube) maps to 0, high ratio maps to 1
    # Max practical aspect ratio is ~10 for very elongated boxes
    raw_aspect = (max_d / min_d) if min_d > 0 else 1.0
    features[4] = min((raw_aspect - 1.0) / 9.0, 1.0)

    return features


def extract_buffer_features(
    grippable: List[Box],
    buffer_view: List[Box],
    bin_config: BinConfig,
) -> np.ndarray:
    """
    Extract 4 features from the conveyor buffer.

    Features:
        0. mean_volume_norm:    Mean box volume / bin volume [0, 1].
           Rationale: If upcoming boxes are large, the agent should
           leave space; if small, it can pack more aggressively.

        1. volume_variance:     Variance of box volumes (normalised) [0, 1].
           Rationale: High variance means diverse boxes -- flexible
           heuristics are better than specialised ones.

        2. num_grippable_norm:  Number of grippable boxes / pick_window [0, 1].
           Rationale: More choices available means more flexibility;
           fewer choices may require the best possible placement.

        3. diversity_index:     Number of unique volume bins / total [0, 1].
           Rationale: Low diversity (many same-size boxes) favours
           layer-building; high diversity favours adaptive approaches.

    Args:
        grippable:   Boxes the robot can currently reach.
        buffer_view: All visible boxes on the conveyor.
        bin_config:  Bin dimensions for normalisation.

    Returns:
        np.ndarray of shape (4,), dtype float32.
    """
    features = np.zeros(4, dtype=np.float32)

    all_boxes = buffer_view if buffer_view else grippable
    if not all_boxes:
        return features

    volumes = [b.volume for b in all_boxes]
    bin_vol = bin_config.volume

    # 0. Mean volume normalised
    features[0] = np.mean(volumes) / bin_vol if bin_vol > 0 else 0.0

    # 1. Volume variance (normalised by bin_vol^2, clamped)
    if len(volumes) > 1:
        vol_var = float(np.var(volumes))
        features[1] = min(vol_var / (bin_vol ** 2), 1.0) if bin_vol > 0 else 0.0

    # 2. Number of grippable boxes normalised
    features[2] = len(grippable) / 4.0  # pick_window = 4

    # 3. Diversity index: unique volume quantiles / total
    if len(volumes) >= 2:
        # Discretise volumes into 5 bins
        vol_arr = np.array(volumes)
        try:
            bins = np.digitize(vol_arr, np.linspace(vol_arr.min(), vol_arr.max(), 5))
            unique_bins = len(set(bins))
            features[3] = unique_bins / 5.0
        except (ValueError, ZeroDivisionError):
            features[3] = 0.5
    else:
        features[3] = 0.2  # Single box = low diversity

    return features


def extract_progress_features(
    tracker: FeatureTracker,
) -> np.ndarray:
    """
    Extract 3 episode progress features.

    Features:
        0. boxes_placed_norm:     Fraction of total boxes placed [0, 1].
        1. boxes_remaining_norm:  Fraction of total boxes remaining [0, 1].
        2. pallets_closed_norm:   Number of pallets closed / 10 [0, 1].
           (10 is a generous upper bound for a single episode)

    Args:
        tracker: FeatureTracker with episode-level counters.

    Returns:
        np.ndarray of shape (3,), dtype float32.
    """
    features = np.zeros(3, dtype=np.float32)
    total = max(tracker.total_boxes_in_episode, 1)

    features[0] = tracker.total_placed / total
    remaining = total - tracker.boxes_processed
    features[1] = max(remaining, 0) / total
    features[2] = min(tracker.pallets_closed / 10.0, 1.0)

    return features


def extract_history_features(
    tracker: FeatureTracker,
    num_actions: int,
) -> np.ndarray:
    """
    Extract 8 heuristic selection history features.

    Features [0:7]:
        One-hot encoding of the last 3 heuristic choices, compressed
        to 7 values representing the frequency of each action in
        the recent history.  If the agent has been switching between
        heuristics, this distribution will be spread out.

    Feature [7]:
        Recent success rate of the last 10 placements [0, 1].
        Rationale: If recent choices have been failing, the agent
        should try different heuristics.

    Args:
        tracker:     FeatureTracker with heuristic history.
        num_actions: Total number of actions (for one-hot dimensionality).

    Returns:
        np.ndarray of shape (8,), dtype float32.
    """
    features = np.zeros(8, dtype=np.float32)

    # Frequency of each action in the last 10 choices (compressed to 7 slots)
    if tracker.heuristic_history:
        for action in tracker.heuristic_history:
            if action < 7:
                features[action] += 1.0
        # Normalise to [0, 1]
        total = len(tracker.heuristic_history)
        if total > 0:
            features[:7] /= total

    # 7. Recent success rate
    features[7] = tracker.recent_success_rate

    return features


def extract_phase_features(
    bin_states: List[BinState],
) -> np.ndarray:
    """
    Extract 3 packing phase indicator features.

    These are binary indicators based on the maximum fill rate
    across all active bins:
        0. is_early: fill < 0.3  (foundation building phase)
        1. is_mid:   0.3 <= fill < 0.6  (main packing phase)
        2. is_late:  fill >= 0.6  (gap-filling phase)

    Rationale:
        Explicit phase indicators help the Q-network specialise its
        heuristic preferences for different stages.  Early packing
        benefits from layer-building; late packing benefits from
        gap-filling heuristics.

    Args:
        bin_states: List of BinState for all active bins.

    Returns:
        np.ndarray of shape (3,), dtype float32.
    """
    features = np.zeros(3, dtype=np.float32)

    if not bin_states:
        features[0] = 1.0  # Default to early phase
        return features

    # Use the max fill rate across bins for phase determination
    max_fill = max(bs.get_fill_rate() for bs in bin_states)

    if max_fill < EARLY_PHASE_THRESHOLD:
        features[0] = 1.0  # Early phase
    elif max_fill < LATE_PHASE_THRESHOLD:
        features[1] = 1.0  # Mid phase
    else:
        features[2] = 1.0  # Late phase

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Main Feature Extraction Function
# ─────────────────────────────────────────────────────────────────────────────

def extract_state_features(
    box: Box,
    bin_states: List[BinState],
    bin_config: BinConfig,
    grippable: List[Box],
    buffer_view: List[Box],
    tracker: FeatureTracker,
    config: HHConfig,
) -> np.ndarray:
    """
    Extract the full state feature vector for the Q-learning agent.

    Concatenates all feature groups into a single flat vector:
      [bin_0_features | bin_1_features | box_features | buffer_features |
       progress_features | history_features | phase_features]

    Total dimensionality: 16 + 5 + 4 + 3 + 8 + 3 = 39

    Args:
        box:         The current box to place.
        bin_states:  List of BinState for all active bins.
        bin_config:  Bin configuration (dimensions, resolution).
        grippable:   Boxes the robot can currently reach.
        buffer_view: All visible boxes on the conveyor.
        tracker:     FeatureTracker with episode history.
        config:      HHConfig with feature dimensions.

    Returns:
        np.ndarray of shape (state_dim,), dtype float32.
    """
    parts = []

    # Per-bin features (2 bins x 8 = 16)
    for i in range(config.num_bins_physical):
        if i < len(bin_states):
            parts.append(extract_bin_features(bin_states[i], bin_config))
        else:
            parts.append(np.zeros(config.features_per_bin, dtype=np.float32))

    # Current box features (5)
    parts.append(extract_box_features(box, bin_config))

    # Buffer features (4)
    parts.append(extract_buffer_features(grippable, buffer_view, bin_config))

    # Episode progress features (3)
    parts.append(extract_progress_features(tracker))

    # History features (8)
    parts.append(extract_history_features(tracker, config.num_actions))

    # Phase features (3)
    parts.append(extract_phase_features(bin_states))

    state = np.concatenate(parts)
    assert state.shape[0] == config.state_dim, (
        f"Feature vector dimension mismatch: got {state.shape[0]}, "
        f"expected {config.state_dim}"
    )
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Tabular State Discretisation
# ─────────────────────────────────────────────────────────────────────────────

def discretise_state(
    state: np.ndarray,
    config: HHConfig,
) -> int:
    """
    Discretise a continuous state vector into a tabular state index.

    For the tabular Q-learning variant, we cannot use the full 39-dimensional
    continuous state.  Instead, we extract key features and discretise them:

        - Fill rate of bin 0:     5 levels  (0, 0.2, 0.4, 0.6, 0.8)
        - Fill rate of bin 1:     5 levels
        - Max height of bin 0:    5 levels
        - Max height of bin 1:    5 levels
        - Box volume ratio:       3 levels  (small, medium, large)
        - Surface roughness:      3 levels  (smooth, medium, rough)

    Total states: 5^4 * 3 * 3 = 5625

    The state index is computed as a unique integer in the range
    [0, tabular_state_size).

    Args:
        state:  Continuous state vector (39-dim).
        config: HHConfig with discretisation parameters.

    Returns:
        Integer state index for Q-table lookup.
    """
    # Extract key features from the continuous state vector
    # Bin 0: features[0:8], fill is at index 0, max_height at index 1, roughness at 2
    # Bin 1: features[8:16], fill is at index 8, max_height at index 9
    # Box: features[16:21], volume is at index 19

    fill_0 = state[0]   # Bin 0 fill rate
    fill_1 = state[8]   # Bin 1 fill rate
    height_0 = state[1]  # Bin 0 max height norm
    height_1 = state[9]  # Bin 1 max height norm
    box_vol = state[19]  # Box volume ratio
    roughness = state[2]  # Bin 0 roughness

    # Discretise each feature
    fill_bins = config.fill_bins
    height_bins = config.height_bins
    box_bins = config.box_size_bins
    rough_bins = config.roughness_bins

    def _bin(value: float, n_bins: int) -> int:
        """Map [0, 1] to [0, n_bins - 1]."""
        return min(int(value * n_bins), n_bins - 1)

    f0 = _bin(fill_0, fill_bins)
    f1 = _bin(fill_1, fill_bins)
    h0 = _bin(height_0, height_bins)
    h1 = _bin(height_1, height_bins)
    bv = _bin(box_vol, box_bins)
    rg = _bin(roughness, rough_bins)

    # Compute unique state index (mixed-radix encoding)
    idx = (
        f0 * (fill_bins * height_bins * height_bins * box_bins * rough_bins)
        + f1 * (height_bins * height_bins * box_bins * rough_bins)
        + h0 * (height_bins * box_bins * rough_bins)
        + h1 * (box_bins * rough_bins)
        + bv * rough_bins
        + rg
    )

    return idx
