"""
Q-network and tabular Q-learning for heuristic selection.

Two implementations are provided for comparison:

1. TabularQLearner:
   - Classic Q-table with discretised state space
   - Q(s,a) += alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
   - Very fast, trains in minutes
   - Good baseline for thesis comparison
   - ~5,625 states x 8 actions = 45,000 Q-values

2. HeuristicSelectorDQN (Deep Q-Network):
   - Small MLP: 39 -> 128 -> 128 -> 64 -> 8
   - Uses experience replay and target network
   - Handles continuous state space natively
   - Better generalisation to unseen states
   - Only ~27,000 parameters (vs millions for position-level DQN)

Design choices:
  - Small network: The action space is just 8, so a small MLP suffices.
    Larger networks overfit without adding value.
  - Dropout: 0.1 for mild regularisation.  Higher values hurt because
    the network is already small.
  - No CNN: State is a handcrafted 39-dim vector, not a spatial grid.
    An MLP is the appropriate architecture.
  - He initialisation: Appropriate for ReLU activations.

References:
  - Mnih et al. (2015): DQN with experience replay and target network
  - van Hasselt et al. (2016): Double DQN for overestimation correction
"""

from __future__ import annotations

import sys
import os
import math
import random
from typing import List, Tuple, Optional, Dict
from collections import deque

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

from strategies.rl_hybrid_hh.config import HHConfig

# PyTorch is optional -- gracefully degrade if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
# Tabular Q-Learner
# ─────────────────────────────────────────────────────────────────────────────

class TabularQLearner:
    """
    Classic tabular Q-learning for heuristic selection.

    Maintains a Q-table of shape (num_states, num_actions) and updates
    it using the standard Q-learning rule:

        Q(s, a) += alpha * (r + gamma * max_a'(Q(s', a')) - Q(s, a))

    The state space is discretised using extract_state_features() followed
    by discretise_state() (see state_features.py).

    Advantages:
      - Extremely fast: no GPU, no backpropagation
      - Convergence guarantees (with sufficient exploration)
      - Fully interpretable Q-table (can inspect Q-values directly)
      - Good baseline for comparing with DQN

    Limitations:
      - Cannot generalise to unseen states
      - State space must be coarsely discretised
      - Loses nuance from continuous features

    Args:
        config: HHConfig with tabular_state_size and num_actions.
    """

    def __init__(self, config: HHConfig) -> None:
        self.config = config
        self.num_states = config.tabular_state_size
        self.num_actions = config.num_actions
        self.alpha = config.lr  # Learning rate
        self.gamma = config.gamma  # Discount factor

        # Initialise Q-table with small random values (breaks ties)
        self.q_table = np.random.uniform(
            low=0.0, high=0.01,
            size=(self.num_states, self.num_actions),
        ).astype(np.float32)

        # Visit counts for analysis
        self.visit_counts = np.zeros(
            (self.num_states, self.num_actions), dtype=np.int64,
        )

    def select_action(self, state_idx: int, epsilon: float = 0.0) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state_idx: Discretised state index.
            epsilon:   Exploration rate [0, 1].

        Returns:
            Action index (0 to num_actions - 1).
        """
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        return int(np.argmax(self.q_table[state_idx]))

    def update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
    ) -> float:
        """
        Update Q-value using the standard Q-learning rule.

        Q(s, a) += alpha * (r + gamma * max(Q(s', a')) - Q(s, a))

        Args:
            state_idx:      Current discretised state.
            action:         Action taken.
            reward:         Reward received.
            next_state_idx: Next discretised state.
            done:           Whether episode terminated.

        Returns:
            TD error (for monitoring convergence).
        """
        current_q = self.q_table[state_idx, action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])

        td_error = target - current_q
        self.q_table[state_idx, action] += self.alpha * td_error
        self.visit_counts[state_idx, action] += 1

        return float(td_error)

    def get_q_values(self, state_idx: int) -> np.ndarray:
        """Return Q-values for all actions at a given state."""
        return self.q_table[state_idx].copy()

    def save(self, path: str) -> None:
        """Save Q-table and config to file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            q_table=self.q_table,
            visit_counts=self.visit_counts,
            config=self.config.to_dict(),
        )

    @classmethod
    def load(cls, path: str) -> "TabularQLearner":
        """Load Q-table from file."""
        data = np.load(path, allow_pickle=True)
        config_dict = data["config"].item()
        config = HHConfig.from_dict(config_dict)
        learner = cls(config)
        learner.q_table = data["q_table"]
        if "visit_counts" in data:
            learner.visit_counts = data["visit_counts"]
        return learner

    def summary(self) -> str:
        """Return a summary of the Q-table statistics."""
        visited = np.sum(self.visit_counts > 0)
        total = self.num_states * self.num_actions
        best_actions = np.argmax(self.q_table, axis=1)
        action_distribution = np.bincount(best_actions, minlength=self.num_actions)
        lines = [
            f"Tabular Q-Learner Summary:",
            f"  States: {self.num_states}  |  Actions: {self.num_actions}",
            f"  Q-entries visited: {visited}/{total} ({visited/total:.1%})",
            f"  Q-value range: [{self.q_table.min():.3f}, {self.q_table.max():.3f}]",
            f"  Best action distribution: {dict(enumerate(action_distribution.tolist()))}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Experience Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Simple experience replay buffer for DQN training.

    Stores (state, action, reward, next_state, done) transitions and
    provides uniform random sampling for minibatch training.

    Args:
        capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 50_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a random minibatch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            as numpy arrays / lists.
        """
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]

        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# DQN Network
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TORCH:

    def _init_weights(module: nn.Module) -> None:
        """Apply He/Kaiming initialisation for ReLU networks."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    class HeuristicSelectorDQN(nn.Module):
        """
        Deep Q-Network for heuristic selection.

        Architecture:
            Dense(state_dim, 128) -> ReLU -> Dropout(0.1)
            Dense(128, 128)       -> ReLU -> Dropout(0.1)
            Dense(128, 64)        -> ReLU
            Dense(64, num_actions) -> Q-values

        This is intentionally small because:
          1. Action space is only 8 (not 38,400 like position-level DQN)
          2. State is 39-dimensional handcrafted features (not raw pixels)
          3. Small networks train fast and avoid overfitting
          4. Total parameters: ~27,000 (vs millions for typical DQN)

        The network outputs Q-values for ALL actions simultaneously,
        unlike the DQN in rl_dqn/ which evaluates one action at a time.
        This is possible because the action space is small enough.

        Args:
            config: HHConfig with network architecture parameters.
        """

        def __init__(self, config: HHConfig) -> None:
            super().__init__()
            self.config = config

            dims = config.hidden_dims  # (128, 128, 64)
            dropout = config.dropout   # 0.1

            layers = []
            in_dim = config.state_dim  # 39

            for i, out_dim in enumerate(dims):
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU(inplace=True))
                # Dropout only on the first two hidden layers
                if i < len(dims) - 1 and dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
                in_dim = out_dim

            # Output layer: Q-values for each action
            layers.append(nn.Linear(in_dim, config.num_actions))

            self.network = nn.Sequential(*layers)
            self.apply(_init_weights)

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """
            Compute Q-values for all actions given a state (or batch).

            Args:
                state: (batch, state_dim) or (state_dim,) tensor.

            Returns:
                (batch, num_actions) or (num_actions,) Q-values.
            """
            return self.network(state)

        def select_action(
            self,
            state: np.ndarray,
            epsilon: float = 0.0,
            device: Optional[torch.device] = None,
        ) -> int:
            """
            Select action using epsilon-greedy policy.

            Args:
                state:   State feature vector (numpy, shape (state_dim,)).
                epsilon: Exploration rate.
                device:  Torch device (defaults to CPU).

            Returns:
                Action index.
            """
            if random.random() < epsilon:
                return random.randint(0, self.config.num_actions - 1)

            if device is None:
                device = next(self.parameters()).device

            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = self.forward(state_t)
                return int(q_values.argmax(dim=1).item())

        def count_parameters(self) -> int:
            """Total number of trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def save(self, path: str) -> None:
            """Save model checkpoint."""
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save({
                "model_state_dict": self.state_dict(),
                "config": self.config.to_dict(),
            }, path)

        @classmethod
        def load(
            cls,
            path: str,
            device: Optional[torch.device] = None,
        ) -> "HeuristicSelectorDQN":
            """Load model from checkpoint."""
            if device is None:
                device = torch.device("cpu")
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            config = HHConfig.from_dict(checkpoint["config"])
            model = cls(config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            return model


    class DQNTrainer:
        """
        Training logic for the HeuristicSelectorDQN.

        Implements standard DQN with:
          - Experience replay
          - Target network (hard sync every N episodes)
          - Epsilon-greedy exploration with linear decay
          - Gradient clipping

        The trainer does NOT run episodes -- it provides update() and
        sync_target() methods that the training loop in train.py calls.

        Args:
            config: HHConfig with training hyperparameters.
            device: Torch device (CPU or CUDA).
        """

        def __init__(
            self,
            config: HHConfig,
            device: Optional[torch.device] = None,
        ) -> None:
            self.config = config
            self.device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            # Online and target networks
            self.online_net = HeuristicSelectorDQN(config).to(self.device)
            self.target_net = HeuristicSelectorDQN(config).to(self.device)
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.target_net.eval()

            # Optimiser
            self.optimizer = optim.Adam(
                self.online_net.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )

            # Replay buffer
            self.replay_buffer = ReplayBuffer(capacity=config.buffer_capacity)

            # Counters
            self.train_steps: int = 0

        def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
            """Select action from the online network."""
            return self.online_net.select_action(state, epsilon, self.device)

        def store_transition(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool,
        ) -> None:
            """Store a transition in the replay buffer."""
            self.replay_buffer.push(state, action, reward, next_state, done)

        def update(self) -> Optional[float]:
            """
            Perform one gradient update step.

            Returns:
                Loss value, or None if buffer is too small.
            """
            if len(self.replay_buffer) < self.config.min_buffer_size:
                return None

            batch_size = min(self.config.batch_size, len(self.replay_buffer))
            states, actions, rewards, next_states, dones = (
                self.replay_buffer.sample(batch_size)
            )

            # Convert to tensors
            states_t = torch.from_numpy(states).to(self.device)
            actions_t = torch.from_numpy(actions).long().to(self.device)
            rewards_t = torch.from_numpy(rewards).to(self.device)
            next_states_t = torch.from_numpy(next_states).to(self.device)
            dones_t = torch.from_numpy(dones).to(self.device)

            # Current Q-values: Q(s, a) for the taken action
            q_values = self.online_net(states_t)
            q_taken = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # Target Q-values: r + gamma * max_a'(Q_target(s', a'))
            with torch.no_grad():
                # Double DQN: online net selects action, target net evaluates
                next_q_online = self.online_net(next_states_t)
                next_actions = next_q_online.argmax(dim=1)
                next_q_target = self.target_net(next_states_t)
                next_q = next_q_target.gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
                targets = rewards_t + (1.0 - dones_t) * self.config.gamma * next_q

            # Huber loss (smooth L1)
            loss = F.smooth_l1_loss(q_taken, targets)

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.online_net.parameters(), self.config.grad_clip,
                )
            self.optimizer.step()

            self.train_steps += 1
            return float(loss.item())

        def sync_target(self) -> None:
            """Hard-sync target network weights from online network."""
            self.target_net.load_state_dict(self.online_net.state_dict())

        def save(self, path: str) -> None:
            """Save both networks and optimiser state."""
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save({
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "train_steps": self.train_steps,
            }, path)

        @classmethod
        def load(
            cls,
            path: str,
            device: Optional[torch.device] = None,
        ) -> "DQNTrainer":
            """Load trainer from checkpoint."""
            if device is None:
                device = torch.device("cpu")
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            config = HHConfig.from_dict(checkpoint["config"])
            trainer = cls(config, device)
            trainer.online_net.load_state_dict(checkpoint["online_state_dict"])
            trainer.target_net.load_state_dict(checkpoint["target_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            trainer.train_steps = checkpoint.get("train_steps", 0)
            return trainer

else:
    # Stubs when PyTorch is not available
    class HeuristicSelectorDQN:
        """Stub: PyTorch not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for DQN mode. "
                "Install with: pip install torch"
            )

    class DQNTrainer:
        """Stub: PyTorch not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for DQN mode. "
                "Install with: pip install torch"
            )
