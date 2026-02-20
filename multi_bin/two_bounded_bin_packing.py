"""
=============================================================================
CODING IDEAS: 2-Bounded Space Multi-Bin Packing
=============================================================================
Based on: Zhao et al. (2021) "Online 3D Bin Packing with Constrained DRL"
          Section on Multi-Bin Packing Extension

Specifically adapted for k=2 bounded space where only 2 pallets/bins are
active simultaneously. When one is closed (sealed), it is replaced by an
empty bin.

The paper's multi-bin approach:
  - Run N parallel BPP-1 instances
  - For each new item, evaluate critic value for all bins
  - Place in bin that "introduces the least drop of the critic value"
  - This is O(N) critic evaluations per item

For k=2, this is trivially efficient: only 2 critic evaluations per item.

This file focuses on the BIN COORDINATION aspects:
  - Which bin receives the item?
  - When is a bin closed?
  - How is the fresh bin initialized?
  - How do we balance utilization across bins?
=============================================================================
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


# =============================================================================
# 1. BIN COORDINATION STRATEGIES
# =============================================================================

class BinCoordinator:
    """
    Coordinates item assignment across 2 active bins.

    The core decision: given an item and 2 bins, which bin should receive it?

    Strategy hierarchy (from paper -> our extensions):
    1. Critic-based (paper): use V(s') to evaluate post-placement state
    2. Volume-matching: place item in bin where it fills the most void
    3. Best-fit: place in bin with least remaining capacity (tightest fit)
    4. Balanced: distribute items to keep utilizations roughly equal
    5. Specialized: one bin for large items, one for small items
    """

    def __init__(self, strategy: str = 'critic'):
        self.strategy = strategy

    def select_bin(self, item_dims: np.ndarray,
                   bins: List,
                   feasibility_fn,
                   network=None) -> Optional[int]:
        """
        Select which bin should receive the item.

        Returns bin index (0 or 1) or None if item fits nowhere.
        """
        if self.strategy == 'critic':
            return self._critic_selection(item_dims, bins, feasibility_fn, network)
        elif self.strategy == 'best_fit':
            return self._best_fit_selection(item_dims, bins, feasibility_fn)
        elif self.strategy == 'balanced':
            return self._balanced_selection(item_dims, bins, feasibility_fn)
        elif self.strategy == 'specialized':
            return self._specialized_selection(item_dims, bins, feasibility_fn)
        else:
            return self._first_fit_selection(item_dims, bins, feasibility_fn)

    def _critic_selection(self, item_dims, bins, feasibility_fn, network):
        """
        Paper's approach: select bin with highest critic value after placement.

        This estimates the FUTURE packing potential of each bin after
        the item is placed. The bin where the item causes the least
        "damage" to future packing is preferred.

        This is the most sophisticated strategy and requires a trained network.
        """
        import torch

        best_bin = None
        best_value = float('-inf')
        l, w, h = item_dims

        for bin_idx in range(2):
            if not bins[bin_idx].is_active:
                continue

            mask = feasibility_fn(bins[bin_idx].height_map, l, w, h)
            if not np.any(mask):
                continue

            # Create observation
            bs = bins[bin_idx]
            obs = np.zeros((bs.L, bs.W, 4), dtype=np.float32)
            obs[:, :, 0] = bs.height_map / bs.H
            obs[:, :, 1] = l / bs.L
            obs[:, :, 2] = w / bs.W
            obs[:, :, 3] = h / bs.H

            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                output = network(obs_tensor)
                value = output['value'].item()

            if value > best_value:
                best_value = value
                best_bin = bin_idx

        return best_bin

    def _best_fit_selection(self, item_dims, bins, feasibility_fn):
        """
        Place in the fullest bin where the item still fits.

        Classic Best-Fit heuristic: minimize remaining space per bin.
        Good for maximizing individual bin utilization.
        """
        l, w, h = item_dims
        best_bin = None
        best_util = -1

        for bin_idx in range(2):
            if not bins[bin_idx].is_active:
                continue
            mask = feasibility_fn(bins[bin_idx].height_map, l, w, h)
            if np.any(mask):
                util = bins[bin_idx].utilization
                if util > best_util:
                    best_util = util
                    best_bin = bin_idx

        return best_bin

    def _balanced_selection(self, item_dims, bins, feasibility_fn):
        """
        Place in the emptiest bin.

        Keeps utilization balanced across bins. Useful when we want
        flexibility to handle diverse future items.
        """
        l, w, h = item_dims
        best_bin = None
        best_util = float('inf')

        for bin_idx in range(2):
            if not bins[bin_idx].is_active:
                continue
            mask = feasibility_fn(bins[bin_idx].height_map, l, w, h)
            if np.any(mask):
                util = bins[bin_idx].utilization
                if util < best_util:
                    best_util = util
                    best_bin = bin_idx

        return best_bin

    def _specialized_selection(self, item_dims, bins, feasibility_fn):
        """
        Route items to specialized bins based on size.

        Bin 0: for larger items (volume > median)
        Bin 1: for smaller items (volume <= median)

        This creates more homogeneous bins which tend to pack better.
        Inspired by Harmonic algorithms that classify items by size.
        """
        l, w, h = item_dims
        volume = l * w * h
        L, W, H = bins[0].L, bins[0].W, bins[0].H
        median_volume = (L * W * H) / 16  # Heuristic threshold

        # Route by size
        if volume > median_volume:
            preferred = 0
            fallback = 1
        else:
            preferred = 1
            fallback = 0

        for bin_idx in [preferred, fallback]:
            if not bins[bin_idx].is_active:
                continue
            mask = feasibility_fn(bins[bin_idx].height_map, l, w, h)
            if np.any(mask):
                return bin_idx

        return None

    def _first_fit_selection(self, item_dims, bins, feasibility_fn):
        """Simplest: first bin where item fits."""
        l, w, h = item_dims
        for bin_idx in range(2):
            if not bins[bin_idx].is_active:
                continue
            mask = feasibility_fn(bins[bin_idx].height_map, l, w, h)
            if np.any(mask):
                return bin_idx
        return None


# =============================================================================
# 2. BIN LIFECYCLE MANAGEMENT
# =============================================================================

class BinLifecycleManager:
    """
    Manages the lifecycle of bins in 2-bounded space.

    Lifecycle: OPEN -> ACTIVE (receiving items) -> CLOSED (sealed, shipped)

    Key decisions:
    - When to close a bin (replace with empty one)
    - How to handle the transition period
    - How to track completed bins for statistics

    The paper simply opens a new BPP-1 instance per bin.
    We extend this with explicit lifecycle management.
    """

    def __init__(self, L: int, W: int, H: int,
                 max_bins: int = 2):
        self.L, self.W, self.H = L, W, H
        self.max_bins = max_bins
        self.active_bins = []
        self.completed_bins = []
        self.total_bins_created = 0

        # Initialize active bins
        for _ in range(max_bins):
            self._create_new_bin()

    def _create_new_bin(self):
        """Create and activate a new empty bin."""
        self.total_bins_created += 1
        bin_state = {
            'id': self.total_bins_created,
            'height_map': np.zeros((self.L, self.W), dtype=np.int32),
            'volume_used': 0.0,
            'items_packed': 0,
            'items': [],  # Track placed items for visualization
            'is_active': True,
            'created_at_step': None,  # Set externally
            'closed_at_step': None,
        }
        self.active_bins.append(bin_state)

    def close_bin(self, bin_idx: int, current_step: int = None):
        """
        Close a bin and replace with a new empty one.

        The closed bin is moved to completed_bins for record-keeping.
        A fresh bin takes its slot.
        """
        closed_bin = self.active_bins[bin_idx]
        closed_bin['is_active'] = False
        closed_bin['closed_at_step'] = current_step
        self.completed_bins.append(closed_bin)

        # Replace with new bin
        self.active_bins[bin_idx] = {
            'id': self.total_bins_created + 1,
            'height_map': np.zeros((self.L, self.W), dtype=np.int32),
            'volume_used': 0.0,
            'items_packed': 0,
            'items': [],
            'is_active': True,
            'created_at_step': current_step,
            'closed_at_step': None,
        }
        self.total_bins_created += 1

    def get_utilizations(self) -> List[float]:
        """Get utilization of all active bins."""
        max_vol = self.L * self.W * self.H
        return [
            b['volume_used'] / max_vol for b in self.active_bins
        ]

    def get_total_stats(self) -> Dict:
        """Get statistics across all bins (active + completed)."""
        all_bins = self.completed_bins + [
            b for b in self.active_bins if b['items_packed'] > 0
        ]
        max_vol = self.L * self.W * self.H
        utilizations = [b['volume_used'] / max_vol for b in all_bins]

        return {
            'total_bins': len(all_bins),
            'avg_utilization': np.mean(utilizations) if utilizations else 0,
            'std_utilization': np.std(utilizations) if utilizations else 0,
            'total_items': sum(b['items_packed'] for b in all_bins),
            'total_volume': sum(b['volume_used'] for b in all_bins),
            'theoretical_volume': len(all_bins) * max_vol,
        }


# =============================================================================
# 3. PERFORMANCE EXPECTATIONS
# =============================================================================

"""
EXPECTED PERFORMANCE (based on paper's multi-bin results):

Paper's single-bin (BPP-1, CUT-2): 66.9% space utilization
Paper's multi-bin results (CUT-2 dataset):

  # Bins | Utilization | Items/bin
  ------+-------------+-----------
    1   |   67.4%     |   17.6
    4   |   69.4%     |   18.8
    9   |   72.1%     |   19.1
   16   |   75.3%     |   19.6
   25   |   77.8%     |   20.2

For our 2-bounded setup:
- With 2 active bins: expect ~68-70% utilization (interpolated)
- With buffer k=5: add ~8-10% utilization (from BPP-k results)
- With buffer k=10: add ~12-15%
- Combined (2-bounded + buffer-10): estimated 78-85%

The key insight from the paper: more bins help because items that
don't fit well in one bin may fit perfectly in another. With only
2 bins, the benefit is modest (~2-3% over single bin), but the
buffer provides a much larger boost.

CRITICAL OPTIMIZATION FOR 2-BOUNDED:
Since we only have 2 bins, BIN SELECTION is less critical than
ITEM SELECTION from the buffer. The buffer is our main lever for
performance improvement. This suggests:
- Invest more computation in buffer selection (MCTS with more sims)
- Use simpler bin selection (best-fit or balanced is sufficient)
- Focus on bin closing policy (this is unique to bounded space)
"""


# =============================================================================
# 4. TRAINING CONSIDERATIONS FOR 2-BOUNDED
# =============================================================================

"""
TRAINING THE NETWORK FOR 2-BOUNDED:

Option A: Train on single-bin, deploy with multi-bin wrapper
  - Paper's approach: train BPP-1 network, use critic for bin selection
  - Pro: simpler training, known to work
  - Con: critic was trained for single-bin, may not accurately predict
    value in multi-bin context

Option B: Train explicitly on 2-bounded environment
  - Modify environment to present 2 height maps as state
  - State: (H_1, H_2, D_buffer) where H_1, H_2 are the two bin height maps
  - Action: (bin_choice, position) compound action
  - Pro: learns bin coordination directly
  - Con: harder to train (larger state/action space), needs more data

Option C: Hierarchical training
  - Train a bin-selection policy separately from position-selection
  - Position policy: standard BPP-1 (from paper)
  - Bin-selection policy: learns which bin to use given both states + item
  - Pro: modular, each component can be validated independently
  - Con: interaction effects may be missed

RECOMMENDATION for thesis:
Start with Option A (easiest to implement), then try Option C if
time permits. Option B is likely overkill for k=2.

TRAINING TIME ESTIMATES:
  - Option A (BPP-1): ~16 hours (from paper)
  - Option C (add bin selector): ~8 hours additional
  - Option B (full 2-bounded): ~40-60 hours (speculation)
"""
