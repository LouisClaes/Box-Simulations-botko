"""
Prioritised Experience Replay buffer for DDQN training.

Implements three replay strategies:

  1. UniformReplayBuffer — standard random sampling (baseline)
  2. PrioritisedReplayBuffer — proportional prioritisation (Schaul et al. 2016)
  3. NStepReplayBuffer — wraps either buffer with n-step return computation

Key design choices:
  - NumPy-based storage for memory efficiency (no Python object overhead)
  - SumTree for O(log N) proportional sampling in PER
  - Lazy n-step accumulation to avoid recomputing returns
  - Thread-safe for use with async vectorized environments

Storage format per transition:
  - state_hm:        (num_bins, grid_l, grid_w) float32  — heightmaps
  - state_box:       (pick_window * 5,) float32           — box features
  - action_feat:     (7,) float32                         — action features
  - reward:          float32
  - next_state_hm:   same shape as state_hm
  - next_state_box:  same shape as state_box
  - done:            bool
  - candidate_feats: (max_candidates, 7) float32          — all candidates at next state
  - num_candidates:  int                                  — valid candidates count

References:
    Schaul, T., et al. (2016). Prioritized Experience Replay. ICLR.
    Sutton, R. & Barto, A. (2018). Multi-step bootstrapping, Ch. 7.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Any, NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# Transition type
# ─────────────────────────────────────────────────────────────────────────────

class Transition(NamedTuple):
    """A single experience transition."""
    state_hm: np.ndarray        # (num_bins, grid_l, grid_w)
    state_box: np.ndarray       # (box_feature_dim,)
    action_feat: np.ndarray     # (7,)
    reward: float
    next_state_hm: np.ndarray
    next_state_box: np.ndarray
    done: bool
    # Next-state candidates for Double DQN argmax
    next_candidates: np.ndarray  # (max_candidates, 7)
    next_num_candidates: int


class BatchSample(NamedTuple):
    """A batch of transitions sampled from the replay buffer."""
    state_hm: np.ndarray         # (batch, num_bins, grid_l, grid_w)
    state_box: np.ndarray        # (batch, box_feature_dim)
    action_feat: np.ndarray      # (batch, 7)
    rewards: np.ndarray          # (batch,)
    next_state_hm: np.ndarray    # (batch, num_bins, grid_l, grid_w)
    next_state_box: np.ndarray   # (batch, box_feature_dim)
    dones: np.ndarray            # (batch,)
    next_candidates: np.ndarray  # (batch, max_candidates, 7)
    next_num_candidates: np.ndarray  # (batch,)
    indices: np.ndarray          # (batch,) — buffer indices (for PER updates)
    weights: np.ndarray          # (batch,) — importance-sampling weights


# ─────────────────────────────────────────────────────────────────────────────
# Sum Tree for O(log N) proportional sampling
# ─────────────────────────────────────────────────────────────────────────────

class SumTree:
    """
    Binary sum tree for efficient proportional sampling.

    Stores priorities in a binary tree where each parent equals the sum of
    its children.  Sampling proportional to priority is O(log N), and
    updating a single priority is also O(log N).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data_pointer = 0
        self._size = 0

    @property
    def total(self) -> float:
        """Sum of all priorities."""
        return float(self._tree[0])

    @property
    def min_priority(self) -> float:
        """Minimum priority among stored transitions."""
        if self._size == 0:
            return 0.0
        leaves = self._tree[self.capacity - 1:self.capacity - 1 + self._size]
        return float(np.min(leaves))

    @property
    def max_priority(self) -> float:
        """Maximum priority among stored transitions."""
        if self._size == 0:
            return 1.0
        leaves = self._tree[self.capacity - 1:self.capacity - 1 + self._size]
        return float(np.max(leaves))

    def add(self, priority: float) -> int:
        """Add a new priority and return its data index."""
        data_idx = self._data_pointer
        tree_idx = data_idx + self.capacity - 1
        self._update(tree_idx, priority)
        self._data_pointer = (self._data_pointer + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return data_idx

    def update(self, data_idx: int, priority: float) -> None:
        """Update the priority at a given data index."""
        tree_idx = data_idx + self.capacity - 1
        self._update(tree_idx, priority)

    def sample(self, value: float) -> int:
        """Sample a data index proportional to priorities."""
        idx = 0  # Start at root
        while idx < self.capacity - 1:  # While not a leaf
            left = 2 * idx + 1
            right = left + 1
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return data_idx

    def _update(self, tree_idx: int, priority: float) -> None:
        """Update tree node and propagate change to root."""
        change = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self._tree[tree_idx] += change


# ─────────────────────────────────────────────────────────────────────────────
# Uniform Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class UniformReplayBuffer:
    """
    Standard experience replay buffer with uniform random sampling.

    Pre-allocates NumPy arrays for all transition components for
    memory-efficient storage without Python object overhead.
    """

    def __init__(
        self,
        capacity: int,
        hm_shape: Tuple[int, int, int],
        box_dim: int,
        action_dim: int = 7,
        max_candidates: int = 200,
    ) -> None:
        """
        Args:
            capacity:       Maximum number of transitions.
            hm_shape:       (num_bins, grid_l, grid_w) heightmap shape.
            box_dim:        Dimension of box feature vector.
            action_dim:     Dimension of action feature vector.
            max_candidates: Maximum candidates stored per transition.
        """
        self.capacity = capacity
        self.max_candidates = max_candidates
        self._size = 0
        self._pointer = 0

        # Pre-allocate arrays
        self._state_hm = np.zeros((capacity, *hm_shape), dtype=np.float32)
        self._state_box = np.zeros((capacity, box_dim), dtype=np.float32)
        self._action_feat = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_state_hm = np.zeros((capacity, *hm_shape), dtype=np.float32)
        self._next_state_box = np.zeros((capacity, box_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)
        self._next_candidates = np.zeros(
            (capacity, max_candidates, action_dim), dtype=np.float32,
        )
        self._next_num_candidates = np.zeros(capacity, dtype=np.int32)

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        """Store a transition, overwriting the oldest if at capacity."""
        idx = self._pointer

        self._state_hm[idx] = transition.state_hm
        self._state_box[idx] = transition.state_box
        self._action_feat[idx] = transition.action_feat
        self._rewards[idx] = transition.reward
        self._next_state_hm[idx] = transition.next_state_hm
        self._next_state_box[idx] = transition.next_state_box
        self._dones[idx] = transition.done

        # Store candidates (pad if fewer than max)
        n_cand = min(transition.next_num_candidates, self.max_candidates)
        self._next_candidates[idx, :n_cand] = transition.next_candidates[:n_cand]
        if n_cand < self.max_candidates:
            self._next_candidates[idx, n_cand:] = 0.0
        self._next_num_candidates[idx] = n_cand

        self._pointer = (self._pointer + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> BatchSample:
        """Sample a uniformly random batch."""
        indices = np.random.randint(0, self._size, size=batch_size)
        weights = np.ones(batch_size, dtype=np.float32)  # Uniform weights

        return BatchSample(
            state_hm=self._state_hm[indices],
            state_box=self._state_box[indices],
            action_feat=self._action_feat[indices],
            rewards=self._rewards[indices],
            next_state_hm=self._next_state_hm[indices],
            next_state_box=self._next_state_box[indices],
            dones=self._dones[indices],
            next_candidates=self._next_candidates[indices],
            next_num_candidates=self._next_num_candidates[indices],
            indices=indices,
            weights=weights,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """No-op for uniform buffer (interface compatibility)."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Prioritised Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class PrioritisedReplayBuffer:
    """
    Proportional Prioritised Experience Replay (Schaul et al. 2016).

    Transitions are sampled with probability proportional to their
    TD-error priority.  Importance-sampling weights correct for the
    non-uniform sampling bias.

    Priority computation:
        p_i = (|delta_i| + epsilon) ^ alpha

    IS weights:
        w_i = (N * P(i)) ^ (-beta) / max_w

    Args:
        capacity:       Maximum buffer size.
        hm_shape:       Heightmap shape per transition.
        box_dim:        Box feature dimension.
        action_dim:     Action feature dimension.
        max_candidates: Max candidates per transition.
        alpha:          Priority exponent (0 = uniform, 1 = full PER).
        beta_start:     Initial IS exponent.
        beta_end:       Final IS exponent.
        beta_frames:    Steps to anneal beta from start to end.
        epsilon:        Small constant to prevent zero priority.
    """

    def __init__(
        self,
        capacity: int,
        hm_shape: Tuple[int, int, int],
        box_dim: int,
        action_dim: int = 7,
        max_candidates: int = 200,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100_000,
        epsilon: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.max_candidates = max_candidates
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self._frame = 0

        # SumTree for proportional sampling
        self._tree = SumTree(capacity)

        # Data storage (same layout as UniformReplayBuffer)
        self._state_hm = np.zeros((capacity, *hm_shape), dtype=np.float32)
        self._state_box = np.zeros((capacity, box_dim), dtype=np.float32)
        self._action_feat = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_state_hm = np.zeros((capacity, *hm_shape), dtype=np.float32)
        self._next_state_box = np.zeros((capacity, box_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)
        self._next_candidates = np.zeros(
            (capacity, max_candidates, action_dim), dtype=np.float32,
        )
        self._next_num_candidates = np.zeros(capacity, dtype=np.int32)

        self._size = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self._size

    @property
    def beta(self) -> float:
        """Current importance-sampling exponent (annealed linearly)."""
        frac = min(1.0, self._frame / max(self.beta_frames, 1))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def add(self, transition: Transition) -> None:
        """Store transition with max priority (new experiences get sampled first)."""
        priority = self._max_priority ** self.alpha
        data_idx = self._tree.add(priority)

        self._state_hm[data_idx] = transition.state_hm
        self._state_box[data_idx] = transition.state_box
        self._action_feat[data_idx] = transition.action_feat
        self._rewards[data_idx] = transition.reward
        self._next_state_hm[data_idx] = transition.next_state_hm
        self._next_state_box[data_idx] = transition.next_state_box
        self._dones[data_idx] = transition.done

        n_cand = min(transition.next_num_candidates, self.max_candidates)
        self._next_candidates[data_idx, :n_cand] = transition.next_candidates[:n_cand]
        if n_cand < self.max_candidates:
            self._next_candidates[data_idx, n_cand:] = 0.0
        self._next_num_candidates[data_idx] = n_cand

        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> BatchSample:
        """Sample a batch with priorities, compute IS weights."""
        self._frame += 1
        beta = self.beta

        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float64)

        total = self._tree.total
        segment = total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            data_idx = self._tree.sample(value)
            indices[i] = data_idx
            # Get priority from tree
            tree_idx = data_idx + self.capacity - 1
            priorities[i] = self._tree._tree[tree_idx]

        # Importance-sampling weights
        total = max(total, 1e-8)
        probs = priorities / total
        probs = np.clip(probs, 1e-8, None)

        weights = (self._size * probs) ** (-beta)
        weights = weights / weights.max()  # Normalise
        weights = weights.astype(np.float32)

        return BatchSample(
            state_hm=self._state_hm[indices],
            state_box=self._state_box[indices],
            action_feat=self._action_feat[indices],
            rewards=self._rewards[indices],
            next_state_hm=self._next_state_hm[indices],
            next_state_box=self._next_state_box[indices],
            dones=self._dones[indices],
            next_candidates=self._next_candidates[indices],
            next_num_candidates=self._next_num_candidates[indices],
            indices=indices,
            weights=weights,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD errors.

        Args:
            indices:   Buffer indices from the last sample() call.
            td_errors: Absolute TD errors |delta| for each sampled transition.
        """
        for idx, td in zip(indices, td_errors):
            priority = (abs(float(td)) + self.epsilon) ** self.alpha
            self._tree.update(int(idx), priority)
            self._max_priority = max(self._max_priority, abs(float(td)) + self.epsilon)


# ─────────────────────────────────────────────────────────────────────────────
# N-Step Return Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class NStepBuffer:
    """
    Accumulates transitions for n-step return computation before storing.

    Holds the last n transitions in a sliding window.  When the window is
    full (or an episode ends), it computes the n-step discounted return
    and pushes the compressed transition to the underlying replay buffer.

    n-step return:
        R_n = r_t + gamma * r_{t+1} + ... + gamma^(n-1) * r_{t+n-1}
        target = R_n + gamma^n * Q(s_{t+n}, argmax_a Q(s_{t+n}, a))

    Args:
        buffer:  Underlying replay buffer (Uniform or Prioritised).
        n:       Number of steps for multi-step returns.
        gamma:   Discount factor.
    """

    def __init__(
        self,
        buffer: UniformReplayBuffer | PrioritisedReplayBuffer,
        n: int = 3,
        gamma: float = 0.95,
    ) -> None:
        self.buffer = buffer
        self.n = n
        self.gamma = gamma
        self._pending: list[Transition] = []

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        """
        Add a transition.  Automatically computes and stores n-step
        returns when the window is full or the episode ends.
        """
        self._pending.append(transition)

        if transition.done:
            # Flush all pending transitions
            self._flush_all()
        elif len(self._pending) >= self.n:
            # Window is full — push the oldest transition
            self._flush_one()

    def flush(self) -> None:
        """Force-flush remaining transitions (call at episode end)."""
        self._flush_all()

    def _flush_one(self) -> None:
        """Compress the oldest transition with n-step return."""
        if not self._pending:
            return

        n = min(len(self._pending), self.n)
        first = self._pending[0]
        last = self._pending[n - 1]

        # Compute n-step return
        R = 0.0
        for i in range(n):
            R += (self.gamma ** i) * self._pending[i].reward

        compressed = Transition(
            state_hm=first.state_hm,
            state_box=first.state_box,
            action_feat=first.action_feat,
            reward=R,
            next_state_hm=last.next_state_hm,
            next_state_box=last.next_state_box,
            done=last.done,
            next_candidates=last.next_candidates,
            next_num_candidates=last.next_num_candidates,
        )
        self.buffer.add(compressed)
        self._pending.pop(0)

    def _flush_all(self) -> None:
        """Flush all pending transitions."""
        while self._pending:
            self._flush_one()

    def sample(self, batch_size: int) -> BatchSample:
        """Delegate to underlying buffer."""
        return self.buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Delegate to underlying buffer."""
        self.buffer.update_priorities(indices, td_errors)
