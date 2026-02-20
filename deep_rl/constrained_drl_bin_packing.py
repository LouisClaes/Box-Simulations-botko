"""
=============================================================================
CODING IDEAS: Constrained DRL for Online 3D Bin Packing
=============================================================================
Based on: Zhao et al. (2021) "Online 3D Bin Packing with Constrained Deep
          Reinforcement Learning" (AAAI-21)

This file contains implementation pseudocode and architectural blueprints
for reproducing and extending the paper's approach.

Target use case:
  - Semi-online with buffer of 5-10 items
  - 2-bounded space (2 active bins/pallets)
  - Maximize fill rate + ensure stability
  - Real-time robotic/conveyor system
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict


# =============================================================================
# 1. ENVIRONMENT: Height Map Based 3D Bin Packing
# =============================================================================

class BinState:
    """
    Represents a single bin's state as a 2D height map.

    The height map H is an L x W integer matrix where H[x][y] = current
    max height of stacked items at grid cell (x, y).

    This is the KEY state representation from the paper, shown to be
    significantly superior to:
      - Height Vector (1D flattening): -16% space utilization
      - Item Sequence Vector: -19.1% space utilization
    """

    def __init__(self, L: int, W: int, H: int):
        self.L = L  # bin length (x-axis)
        self.W = W  # bin width (y-axis)
        self.H = H  # bin height (z-axis)
        self.height_map = np.zeros((L, W), dtype=np.int32)
        self.items_packed = 0
        self.volume_used = 0

    def reset(self):
        self.height_map = np.zeros((self.L, self.W), dtype=np.int32)
        self.items_packed = 0
        self.volume_used = 0

    def can_place(self, x: int, y: int, l: int, w: int, h: int) -> bool:
        """Check if item (l, w, h) can be placed at FLB corner (x, y)."""
        # Containment check
        if x + l > self.L or y + w > self.W:
            return False

        # Height check: max height in the footprint + item height <= H
        region = self.height_map[x:x+l, y:y+w]
        placement_z = np.max(region)
        if placement_z + h > self.H:
            return False

        return True

    def check_stability(self, x: int, y: int, l: int, w: int, h: int) -> bool:
        """
        Conservative stability criterion from the paper.

        A placement is FEASIBLE if ANY of these conditions holds:
        1) >= 60% bottom area supported AND all 4 bottom corners supported
        2) >= 80% bottom area supported AND 3/4 bottom corners supported
        3) >= 95% bottom area supported (regardless of corners)

        On the ground floor (z=0), all placements are stable.
        """
        region = self.height_map[x:x+l, y:y+w]
        placement_z = np.max(region)

        # On the floor: always stable
        if placement_z == 0:
            return True

        # Calculate support: cells where height == placement_z
        total_cells = l * w
        supported_cells = np.sum(region == placement_z)
        support_ratio = supported_cells / total_cells

        # Check corner support
        corners = [
            (x, y),           # front-left
            (x+l-1, y),       # back-left
            (x, y+w-1),       # front-right
            (x+l-1, y+w-1),   # back-right
        ]
        corners_supported = sum(
            1 for cx, cy in corners
            if self.height_map[cx, cy] == placement_z
        )

        # Three-tiered stability criterion
        if support_ratio >= 0.60 and corners_supported == 4:
            return True
        if support_ratio >= 0.80 and corners_supported >= 3:
            return True
        if support_ratio >= 0.95:
            return True

        return False

    def compute_feasibility_mask(self, l: int, w: int, h: int) -> np.ndarray:
        """
        Compute the ground-truth feasibility mask M for item (l, w, h).

        Returns an L x W binary matrix where M[x][y] = 1 if placement
        at (x, y) is feasible (containment + height + stability).
        """
        mask = np.zeros((self.L, self.W), dtype=np.float32)
        for x in range(self.L - l + 1):
            for y in range(self.W - w + 1):
                if self.can_place(x, y, l, w, h):
                    if self.check_stability(x, y, l, w, h):
                        mask[x, y] = 1.0
        return mask

    def place_item(self, x: int, y: int, l: int, w: int, h: int):
        """Place item at position (x, y) and update height map."""
        region = self.height_map[x:x+l, y:y+w]
        placement_z = np.max(region)
        self.height_map[x:x+l, y:y+w] = placement_z + h
        self.items_packed += 1
        self.volume_used += l * w * h

    @property
    def utilization(self) -> float:
        return self.volume_used / (self.L * self.W * self.H)


class OnlineBPPEnvironment:
    """
    Online 3D Bin Packing Environment for a single bin.

    State:  (H_n, D_n) where H_n is height map, D_n is item dimensions
            stretched into L x W x 3 tensor.
    Action: Integer a_n = x + L * y (FLB grid position)
    Reward: 10 * (l*w*h) / (L*W*H) for successful placement, 0 for failure
    Done:   When current item cannot be placed anywhere
    """

    def __init__(self, L: int = 10, W: int = 10, H: int = 10,
                 item_generator=None):
        self.bin = BinState(L, W, H)
        self.L, self.W, self.H = L, W, H
        self.item_generator = item_generator or self._default_generator
        self.current_item = None

    def _default_generator(self):
        """Generate random item with dimensions <= half the bin size."""
        l = np.random.randint(1, self.L // 2 + 1)
        w = np.random.randint(1, self.W // 2 + 1)
        h = np.random.randint(1, self.H // 2 + 1)
        return np.array([l, w, h], dtype=np.int32)

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self.bin.reset()
        self.current_item = self.item_generator()
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """
        Construct the L x W x 4 observation tensor.

        Channels:
          0: Height map H_n (L x W)
          1: Item length l_n broadcast to L x W
          2: Item width w_n broadcast to L x W
          3: Item height h_n broadcast to L x W
        """
        obs = np.zeros((self.L, self.W, 4), dtype=np.float32)
        obs[:, :, 0] = self.bin.height_map.astype(np.float32) / self.H
        obs[:, :, 1] = self.current_item[0] / self.L
        obs[:, :, 2] = self.current_item[1] / self.W
        obs[:, :, 3] = self.current_item[2] / self.H
        return obs

    def get_feasibility_mask(self) -> np.ndarray:
        """Get ground-truth feasibility mask for current item."""
        l, w, h = self.current_item
        return self.bin.compute_feasibility_mask(l, w, h)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action (place item at grid position).

        Returns: (observation, reward, done, info)
        """
        x = action % self.L
        y = action // self.L
        l, w, h = self.current_item

        # Check feasibility
        mask = self.get_feasibility_mask()
        if mask[x, y] == 0:
            # Invalid placement -- episode ends with 0 reward
            return self._get_observation(), 0.0, True, {
                'valid': False,
                'utilization': self.bin.utilization
            }

        # Place item
        self.bin.place_item(x, y, l, w, h)

        # Step-wise reward: normalized volume of placed item
        reward = 10.0 * (l * w * h) / (self.L * self.W * self.H)

        # Generate next item
        self.current_item = self.item_generator()

        # Check if next item can be placed anywhere
        next_mask = self.get_feasibility_mask()
        done = not np.any(next_mask)

        return self._get_observation(), reward, done, {
            'valid': True,
            'utilization': self.bin.utilization,
            'items_packed': self.bin.items_packed
        }


# =============================================================================
# 2. NEURAL NETWORK ARCHITECTURE
# =============================================================================

class StateCNN(nn.Module):
    """
    Shared feature extractor using 5 convolutional layers.

    Input:  L x W x 4 tensor (height map + item dimensions)
    Output: Flattened feature vector

    Architecture from paper:
      - 5 conv layers, kernel 3x3, padding 1, stride 1
      - ReLU activations
    """

    def __init__(self, L: int = 10, W: int = 10, in_channels: int = 4,
                 hidden_channels: int = 32):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
        )
        self.feature_dim = hidden_channels * L * W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, L, W, 4) -> need (batch, 4, L, W)
        x = x.permute(0, 3, 1, 2)
        features = self.conv_layers(x)
        return features.flatten(start_dim=1)


class Actor(nn.Module):
    """
    Policy network that outputs action probabilities over L*W positions.

    The output is modulated by the feasibility mask during training
    (prediction-and-projection scheme).
    """

    def __init__(self, feature_dim: int, action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                epsilon: float = 1e-3) -> torch.Tensor:
        """
        Compute action probabilities, modulated by feasibility mask.

        Args:
            features: State CNN features
            mask: Feasibility mask (1=feasible, 0=infeasible)
            epsilon: Small probability for infeasible actions (smoother gradients)

        Returns:
            Action probability distribution
        """
        logits = self.mlp(features)

        if mask is not None:
            # Prediction-and-projection: suppress infeasible actions
            # Set infeasible logits to very negative value
            mask_flat = mask.flatten(start_dim=1)
            logits = logits.masked_fill(mask_flat == 0, float('-inf'))

        probs = F.softmax(logits, dim=-1)

        if mask is not None:
            # Replace zeros with epsilon for numerical stability
            mask_flat = mask.flatten(start_dim=1)
            probs = torch.where(mask_flat == 1, probs, torch.full_like(probs, epsilon))
            # Renormalize
            probs = probs / probs.sum(dim=-1, keepdim=True)

        return probs


class Critic(nn.Module):
    """
    Value network that estimates state value V(s).

    Used for:
    1. Advantage computation in actor-critic
    2. Bin selection in multi-bin setting (pack into bin with highest V)
    3. MCTS leaf evaluation in BPP-k
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


class MaskPredictor(nn.Module):
    """
    Feasibility mask predictor (the key innovation).

    Predicts which grid positions are feasible for the current item.
    Trained with MSE loss against ground-truth feasibility masks.

    This is an AUXILIARY task that:
    1. Provides supervision signal for feasibility
    2. Modulates actor outputs via prediction-and-projection
    3. Enables the agent to learn feasible policies efficiently
    """

    def __init__(self, feature_dim: int, L: int = 10, W: int = 10,
                 hidden_dim: int = 256):
        super().__init__()
        self.L = L
        self.W = W
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, L * W),
            nn.Sigmoid(),  # Output in [0, 1] for each grid cell
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns predicted feasibility mask of shape (batch, L, W)."""
        mask_flat = self.mlp(features)
        return mask_flat.view(-1, self.L, self.W)


class ConstrainedPackingNetwork(nn.Module):
    """
    Complete network combining all components.

    Architecture (from paper Figure 2):
      State CNN (shared) -> Actor (MLP) -> action probabilities
                         -> Critic (MLP) -> V-value
                         -> Mask Predictor (MLP) -> feasibility mask
    """

    def __init__(self, L: int = 10, W: int = 10, in_channels: int = 4,
                 hidden_channels: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.L = L
        self.W = W
        self.action_dim = L * W

        self.state_cnn = StateCNN(L, W, in_channels, hidden_channels)
        feature_dim = self.state_cnn.feature_dim

        self.actor = Actor(feature_dim, self.action_dim, hidden_dim)
        self.critic = Critic(feature_dim, hidden_dim)
        self.mask_predictor = MaskPredictor(feature_dim, L, W, hidden_dim)

    def forward(self, obs: torch.Tensor,
                gt_mask: Optional[torch.Tensor] = None,
                use_predicted_mask: bool = False):
        """
        Forward pass through all components.

        Args:
            obs: Observation tensor (batch, L, W, 4)
            gt_mask: Ground-truth feasibility mask (for training)
            use_predicted_mask: If True, use predicted mask for projection

        Returns:
            dict with action_probs, value, predicted_mask
        """
        features = self.state_cnn(obs)

        # Predict feasibility mask
        predicted_mask = self.mask_predictor(features)

        # Choose which mask to use for action projection
        if gt_mask is not None and not use_predicted_mask:
            projection_mask = gt_mask
        elif use_predicted_mask:
            projection_mask = (predicted_mask > 0.5).float()
        else:
            projection_mask = None

        # Get action probabilities (modulated by mask)
        action_probs = self.actor(features, projection_mask)

        # Get state value
        value = self.critic(features)

        return {
            'action_probs': action_probs,
            'value': value,
            'predicted_mask': predicted_mask,
        }


# =============================================================================
# 3. TRAINING LOOP (ACKTR-style Actor-Critic)
# =============================================================================

class ConstrainedDRLTrainer:
    """
    Training loop for the constrained DRL bin packing agent.

    Uses ACKTR (Actor-Critic with Kronecker-Factored Trust Region) framework.
    On-policy: agent-environment interaction data is sampled from current policy.

    Loss function:
      L = alpha * L_actor + beta * L_critic + lambda * L_mask
          + omega * E_inf - psi * E_entropy

    Weights (from paper):
      alpha = 1.0, beta = 0.5, lambda = 0.5, omega = 0.01, psi = 0.01
    """

    def __init__(self, env: OnlineBPPEnvironment,
                 network: ConstrainedPackingNetwork,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 lam: float = 0.5,
                 omega: float = 0.01,
                 psi: float = 0.01,
                 epsilon_mask: float = 1e-3):
        self.env = env
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.gamma = gamma

        # Loss weights
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.omega = omega
        self.psi = psi
        self.epsilon_mask = epsilon_mask

    def compute_loss(self, obs_batch, action_batch, return_batch,
                     advantage_batch, gt_mask_batch):
        """
        Compute the total loss with all five components.

        Args:
            obs_batch: Observations (batch, L, W, 4)
            action_batch: Actions taken (batch,)
            return_batch: Discounted returns (batch,)
            advantage_batch: GAE advantages (batch,)
            gt_mask_batch: Ground-truth feasibility masks (batch, L, W)
        """
        output = self.network(obs_batch, gt_mask=gt_mask_batch)
        action_probs = output['action_probs']
        values = output['value'].squeeze(-1)
        predicted_mask = output['predicted_mask']

        # 1. Actor loss (policy gradient with advantage)
        action_log_probs = torch.log(
            action_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        )
        L_actor = -(action_log_probs * advantage_batch).mean()

        # 2. Critic loss (MSE between predicted and actual returns)
        L_critic = F.mse_loss(values, return_batch)

        # 3. Mask prediction loss (MSE between predicted and GT mask)
        L_mask = F.mse_loss(predicted_mask, gt_mask_batch)

        # 4. Infeasibility loss (minimize probability mass on infeasible LPs)
        infeasible_mask = (gt_mask_batch.flatten(start_dim=1) < 0.5).float()
        E_inf = (action_probs * infeasible_mask).sum(dim=1).mean()

        # 5. Feasibility-based entropy (only over feasible actions)
        feasible_mask = gt_mask_batch.flatten(start_dim=1)
        feasible_probs = action_probs * feasible_mask
        # Add small epsilon to avoid log(0)
        feasible_log_probs = torch.log(feasible_probs + 1e-10)
        E_entropy = -(feasible_probs * feasible_log_probs).sum(dim=1).mean()

        # Total loss
        total_loss = (self.alpha * L_actor
                      + self.beta * L_critic
                      + self.lam * L_mask
                      + self.omega * E_inf
                      - self.psi * E_entropy)

        return total_loss, {
            'L_actor': L_actor.item(),
            'L_critic': L_critic.item(),
            'L_mask': L_mask.item(),
            'E_inf': E_inf.item(),
            'E_entropy': E_entropy.item(),
            'total': total_loss.item(),
        }

    def collect_trajectory(self, max_steps: int = 1000):
        """
        Collect one episode trajectory using current policy.

        Returns lists of observations, actions, rewards, masks, values.
        """
        obs = self.env.reset()
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'masks': [],  # feasibility masks
            'values': [],
            'dones': [],
        }

        for _ in range(max_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            gt_mask = self.env.get_feasibility_mask()
            gt_mask_tensor = torch.FloatTensor(gt_mask).unsqueeze(0)

            with torch.no_grad():
                output = self.network(obs_tensor, gt_mask=gt_mask_tensor)

            action_probs = output['action_probs'].squeeze(0)
            value = output['value'].item()

            # Sample action from policy
            action = torch.multinomial(action_probs, 1).item()

            trajectory['observations'].append(obs)
            trajectory['actions'].append(action)
            trajectory['masks'].append(gt_mask)
            trajectory['values'].append(value)

            obs, reward, done, info = self.env.step(action)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)

            if done:
                break

        return trajectory

    def compute_returns_and_advantages(self, trajectory, lam_gae: float = 0.95):
        """Compute discounted returns and GAE advantages."""
        rewards = trajectory['rewards']
        values = trajectory['values']
        dones = trajectory['dones']
        T = len(rewards)

        returns = np.zeros(T)
        advantages = np.zeros(T)

        # Bootstrap from last value
        next_value = 0.0
        next_advantage = 0.0

        for t in reversed(range(T)):
            if dones[t]:
                next_value = 0.0
                next_advantage = 0.0

            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = delta + self.gamma * lam_gae * next_advantage
            returns[t] = advantages[t] + values[t]

            next_value = values[t]
            next_advantage = advantages[t]

        return returns, advantages

    def train_step(self):
        """One training step: collect trajectory, compute loss, update."""
        trajectory = self.collect_trajectory()
        returns, advantages = self.compute_returns_and_advantages(trajectory)

        # Convert to tensors
        obs_batch = torch.FloatTensor(np.array(trajectory['observations']))
        action_batch = torch.LongTensor(trajectory['actions'])
        return_batch = torch.FloatTensor(returns)
        advantage_batch = torch.FloatTensor(advantages)
        # Normalize advantages
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (
            advantage_batch.std() + 1e-8)
        gt_mask_batch = torch.FloatTensor(np.array(trajectory['masks']))

        # Compute and backpropagate loss
        self.optimizer.zero_grad()
        total_loss, loss_dict = self.compute_loss(
            obs_batch, action_batch, return_batch,
            advantage_batch, gt_mask_batch
        )
        total_loss.backward()
        self.optimizer.step()

        return loss_dict, {
            'utilization': self.env.bin.utilization,
            'items_packed': self.env.bin.items_packed,
        }

    def train(self, num_episodes: int = 100000, log_interval: int = 100):
        """Full training loop."""
        for episode in range(num_episodes):
            loss_dict, info = self.train_step()

            if (episode + 1) % log_interval == 0:
                print(f"Episode {episode+1}: "
                      f"Loss={loss_dict['total']:.4f}, "
                      f"Util={info['utilization']:.2%}, "
                      f"Items={info['items_packed']}")


# =============================================================================
# 4. MCTS FOR LOOKAHEAD (BPP-k)
# =============================================================================

class MCTSPermutationSearch:
    """
    Monte Carlo Tree Search over permutations of lookahead items.

    For BPP-k with k lookahead items, this explores different virtual
    placement orders to find the best action for the current item.

    Key concepts:
    - Permutation tree: each path from root to leaf is a placement ordering
    - Order dependence constraint: if item p arrives before q in actual order,
      but q is placed before p virtually, then p cannot be on top of q
    - Complexity: O(k * m) where m is number of sampled paths

    This replaces brute-force O(k!) permutation enumeration.
    """

    def __init__(self, network: ConstrainedPackingNetwork,
                 bin_state: BinState,
                 num_simulations: int = 100,
                 exploration_constant: float = 1.0):
        self.network = network
        self.bin_state = bin_state
        self.num_simulations = num_simulations
        self.c = exploration_constant

    def search(self, current_item: np.ndarray,
               lookahead_items: List[np.ndarray]) -> int:
        """
        Find best action for current_item given lookahead_items.

        The search explores permutations of [current_item, *lookahead_items]
        to find the best placement for current_item.

        Returns: best action (grid position index) for current_item
        """
        all_items = [current_item] + lookahead_items
        k = len(all_items)

        # Root node represents empty permutation
        best_action = None
        best_value = float('-inf')

        # For each possible action for the current item:
        L, W = self.bin_state.L, self.bin_state.W
        mask = self.bin_state.compute_feasibility_mask(
            *current_item
        )

        for action in range(L * W):
            x, y = action % L, action // L
            if mask[x, y] == 0:
                continue

            # Simulate placing current item at this action
            sim_bin = self._copy_bin_state(self.bin_state)
            sim_bin.place_item(x, y, *current_item)

            # Evaluate by simulating random permutations of remaining items
            total_value = 0.0
            for _ in range(self.num_simulations // (np.sum(mask > 0) + 1)):
                value = self._simulate_permutation(
                    sim_bin, lookahead_items, current_item
                )
                total_value += value

            avg_value = total_value / max(
                1, self.num_simulations // (np.sum(mask > 0) + 1)
            )

            if avg_value > best_value:
                best_value = avg_value
                best_action = action

        return best_action

    def _simulate_permutation(self, bin_state: BinState,
                              remaining_items: List[np.ndarray],
                              actual_first_item: np.ndarray) -> float:
        """
        Simulate a random permutation of remaining items.

        Enforces order dependence constraint: items that arrive later
        in actual order cannot support items that arrive earlier.
        """
        sim_bin = self._copy_bin_state(bin_state)

        # Random permutation of remaining items
        perm = np.random.permutation(len(remaining_items))

        total_reward = 0.0
        for idx in perm:
            item = remaining_items[idx]
            l, w, h = item

            # Apply order dependence constraint:
            # Block positions where later-arriving items are below
            constrained_bin = self._apply_order_constraint(
                sim_bin, idx, perm, remaining_items
            )

            mask = constrained_bin.compute_feasibility_mask(l, w, h)
            if not np.any(mask):
                continue

            # Use network to select action (greedy)
            obs = self._make_observation(constrained_bin, item)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0)

            with torch.no_grad():
                output = self.network(obs_tensor, gt_mask=mask_tensor)
                action = output['action_probs'].argmax(dim=1).item()

            x, y = action % sim_bin.L, action // sim_bin.L
            if mask[x, y] > 0:
                sim_bin.place_item(x, y, l, w, h)
                total_reward += 10.0 * (l * w * h) / (
                    sim_bin.L * sim_bin.W * sim_bin.H)

        # Add critic value at leaf (estimated future value)
        obs = self._make_observation(sim_bin, remaining_items[0]
                                     if remaining_items else actual_first_item)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            output = self.network(obs_tensor)
            leaf_value = output['value'].item()

        return total_reward + leaf_value

    def _apply_order_constraint(self, bin_state: BinState, current_idx: int,
                                perm: np.ndarray,
                                items: List[np.ndarray]) -> BinState:
        """
        Apply order dependence constraint.

        For items placed virtually before the current one but arriving
        after it in actual order, block those positions by setting
        height to H (maximum).
        """
        constrained = self._copy_bin_state(bin_state)
        # Implementation would track which positions are occupied by
        # virtually-earlier items that actually arrive later
        # and set height_map at those positions to H
        return constrained

    def _copy_bin_state(self, bin_state: BinState) -> BinState:
        """Deep copy a bin state."""
        new_bin = BinState(bin_state.L, bin_state.W, bin_state.H)
        new_bin.height_map = bin_state.height_map.copy()
        new_bin.items_packed = bin_state.items_packed
        new_bin.volume_used = bin_state.volume_used
        return new_bin

    def _make_observation(self, bin_state: BinState,
                          item: np.ndarray) -> np.ndarray:
        """Construct observation from bin state and item."""
        L, W, H = bin_state.L, bin_state.W, bin_state.H
        obs = np.zeros((L, W, 4), dtype=np.float32)
        obs[:, :, 0] = bin_state.height_map.astype(np.float32) / H
        obs[:, :, 1] = item[0] / L
        obs[:, :, 2] = item[1] / W
        obs[:, :, 3] = item[2] / H
        return obs


# =============================================================================
# 5. ADAPTATION FOR 2-BOUNDED SPACE WITH 5-10 ITEM BUFFER
# =============================================================================

class TwoBoundedBufferEnvironment:
    """
    ADAPTED ENVIRONMENT for our thesis use case:
    - 2 active bins (2-bounded space)
    - Buffer of 5-10 items to choose from
    - Must decide: which item from buffer, which bin, which position

    This extends the paper's approach in several key ways:
    1. Buffer SELECTION (not just lookahead) -- we choose which item to place
    2. BIN SELECTION -- we choose which of 2 bins receives the item
    3. BIN CLOSING -- we decide when to close a bin permanently

    Action space becomes:
      (item_idx, bin_idx, position) where
      - item_idx in {0, ..., buffer_size-1}
      - bin_idx in {0, 1}
      - position in {0, ..., L*W-1}

    For tractability, we decompose this into a hierarchy:
      Step 1: Choose item from buffer (network or MCTS)
      Step 2: Choose bin (critic-based, as in paper's multi-bin)
      Step 3: Choose position (actor with feasibility mask)
    """

    def __init__(self, L: int = 10, W: int = 10, H: int = 10,
                 buffer_size: int = 5, item_generator=None):
        self.L, self.W, self.H = L, W, H
        self.buffer_size = buffer_size
        self.item_generator = item_generator or self._default_generator

        # 2 active bins
        self.bins = [BinState(L, W, H), BinState(L, W, H)]
        self.bin_active = [True, True]
        self.bins_completed = 0

        # Item buffer
        self.buffer: List[np.ndarray] = []

    def _default_generator(self):
        l = np.random.randint(1, self.L // 2 + 1)
        w = np.random.randint(1, self.W // 2 + 1)
        h = np.random.randint(1, self.H // 2 + 1)
        return np.array([l, w, h], dtype=np.int32)

    def reset(self):
        """Reset both bins and fill buffer."""
        for bin_state in self.bins:
            bin_state.reset()
        self.bin_active = [True, True]
        self.bins_completed = 0
        self.buffer = [self.item_generator() for _ in range(self.buffer_size)]
        return self._get_observation()

    def _get_observation(self) -> Dict:
        """
        Extended observation for 2-bounded buffer setup.

        Returns dict with:
          - height_maps: list of 2 height maps (L x W each)
          - buffer_items: list of item dimension vectors
          - bin_active: which bins are still active
        """
        return {
            'height_maps': [b.height_map.copy() for b in self.bins],
            'buffer_items': [item.copy() for item in self.buffer],
            'bin_active': self.bin_active.copy(),
        }

    def select_item_and_bin(self, network: ConstrainedPackingNetwork):
        """
        Hierarchical decision: select item from buffer and target bin.

        Strategy (adapted from paper's multi-bin extension):
        For each (item, bin) combination, compute critic value after
        hypothetical placement. Choose the combination with highest value.

        This is O(buffer_size * 2) critic evaluations.
        """
        best_item_idx = None
        best_bin_idx = None
        best_value = float('-inf')

        for item_idx, item in enumerate(self.buffer):
            for bin_idx in range(2):
                if not self.bin_active[bin_idx]:
                    continue

                bin_state = self.bins[bin_idx]
                mask = bin_state.compute_feasibility_mask(*item)

                if not np.any(mask):
                    continue

                # Evaluate this (item, bin) by getting critic value
                obs = np.zeros((self.L, self.W, 4), dtype=np.float32)
                obs[:, :, 0] = bin_state.height_map / self.H
                obs[:, :, 1] = item[0] / self.L
                obs[:, :, 2] = item[1] / self.W
                obs[:, :, 3] = item[2] / self.H

                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    output = network(obs_tensor)
                    value = output['value'].item()

                if value > best_value:
                    best_value = value
                    best_item_idx = item_idx
                    best_bin_idx = bin_idx

        return best_item_idx, best_bin_idx

    def step(self, item_idx: int, bin_idx: int, position: int):
        """
        Execute the full action: place buffer[item_idx] into bins[bin_idx]
        at the given grid position.
        """
        item = self.buffer[item_idx]
        l, w, h = item
        x = position % self.L
        y = position // self.L

        bin_state = self.bins[bin_idx]

        # Place item
        bin_state.place_item(x, y, l, w, h)
        reward = 10.0 * (l * w * h) / (self.L * self.W * self.H)

        # Remove item from buffer and add new one
        self.buffer.pop(item_idx)
        self.buffer.append(self.item_generator())

        # Check if bin should be closed
        # Strategy: close bin if no item in buffer fits
        should_close = self._check_bin_close(bin_idx)
        if should_close:
            self.bin_active[bin_idx] = False
            self.bins_completed += 1
            # Open new bin in this slot
            self.bins[bin_idx] = BinState(self.L, self.W, self.H)
            self.bin_active[bin_idx] = True

        info = {
            'bin_utilizations': [b.utilization for b in self.bins],
            'bins_completed': self.bins_completed,
        }

        return self._get_observation(), reward, False, info

    def _check_bin_close(self, bin_idx: int) -> bool:
        """
        Decide if a bin should be closed.

        Strategies:
        1. Threshold: close if utilization > 85%
        2. No-fit: close if no buffer item fits
        3. Critic-based: close if expected future value is very low

        For now, use simple no-fit heuristic.
        """
        bin_state = self.bins[bin_idx]

        # Check if ANY item in buffer can fit
        for item in self.buffer:
            mask = bin_state.compute_feasibility_mask(*item)
            if np.any(mask):
                return False

        # No item fits -> close this bin
        return True


# =============================================================================
# 6. INTEGRATION POINTS WITH OTHER METHODS
# =============================================================================

"""
INTEGRATION IDEAS:

1. HYBRID WITH HEURISTICS (see ../hybrid_heuristic_ml/):
   - Use heuristic placement rules (DBLF, DFTRC) as fallback when
     DRL confidence is low
   - Use heuristic for bin selection, DRL for position selection
   - Use heuristic-based features as additional CNN input channels

2. STABILITY MODULE (see ../stability/):
   - Replace the paper's simplified 3-tier stability check with
     more physics-accurate stability (center of mass, moment of inertia)
   - Use the "Online 3D Bin Packing with Fast Stability Validation"
     paper's approach for real-time stability checking
   - The feasibility mask framework is GENERAL -- any stability check
     can be plugged in to generate the ground-truth mask

3. SEMI-ONLINE BUFFER (see ../semi_online_buffer/):
   - The MCTS permutation search from this paper applies directly
     to buffer item selection
   - Combine with the 2-bounded bin selection for full pipeline

4. MULTI-BIN COORDINATION (see ../multi_bin/):
   - The critic-based bin selection from this paper is the simplest
     effective multi-bin strategy
   - Can be enhanced with explicit bin-closing policies

5. HYPER-HEURISTIC SELECTION (see ../hyper_heuristics/):
   - Use the feasibility mask as input features for a hyper-heuristic
     that selects between different placement rules
   - The mask predictor could be shared across all heuristics
"""


# =============================================================================
# 7. COMPLEXITY AND FEASIBILITY ANALYSIS
# =============================================================================

"""
COMPUTATIONAL COMPLEXITY:

Training:
  - One episode: O(T * L^2 * W^2) where T = items per episode
    (feasibility mask computation is O(L*W) per item, done for L*W positions)
  - Total training: ~16 hours on Titan V GPU for 10x10 grid
  - For 20x20 grid: ~4x slower due to larger action space
  - For 30x30 grid: ~9x slower, but still feasible (~144 hours = 6 days)

Inference (BPP-1, no lookahead):
  - < 10 ms per item on GPU
  - Feasibility mask: O(L*W * (L+W)) per item
  - Network forward: O(1) constant time (fixed architecture)

Inference (BPP-k with MCTS):
  - O(k * m) where m = number of MCTS simulations
  - k=8, m=100: ~100ms per item (still real-time for robotic arms)

Memory:
  - Network: ~10M parameters (estimate for 10x10 grid with 32 channels)
  - Height map: L*W * 4 bytes = 400 bytes for 10x10
  - Feasibility mask: L*W * 4 bytes = 400 bytes

FEASIBILITY FOR THESIS:
  - Training: Very feasible with a single GPU (even a GTX 1080 Ti)
  - Implementation: Moderate difficulty (~2-3 weeks for core,
    ~1-2 weeks for extensions)
  - Total estimate: 4-6 weeks for full implementation + experiments
"""


# =============================================================================
# 8. QUICK START: Minimal Training Example
# =============================================================================

def main():
    """
    Minimal example to train the constrained DRL agent on BPP-1.

    This demonstrates the core training loop. For full experiments,
    extend with:
    - CUT-1/CUT-2 dataset generation
    - BPP-k with MCTS
    - Multi-bin and 2-bounded extensions
    - Tensorboard logging
    - Model checkpointing
    """
    L, W, H = 10, 10, 10

    env = OnlineBPPEnvironment(L, W, H)
    network = ConstrainedPackingNetwork(L, W)
    trainer = ConstrainedDRLTrainer(env, network)

    print("Starting training...")
    print(f"Grid size: {L}x{W}x{H}")
    print(f"Action space: {L*W} positions")
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    print()

    trainer.train(num_episodes=10000, log_interval=100)


if __name__ == '__main__':
    main()


# =============================================================================
# 9. EXTENDED: ACKTR-SPECIFIC COMPONENTS
# =============================================================================

class KFACOptimizer:
    """
    Kronecker-Factored Approximate Curvature (K-FAC) optimizer stub.

    ACKTR uses K-FAC to approximate the Fisher Information Matrix (FIM):
        F = E_{s~d_pi, a~pi} [ (grad log pi(a|s)) (grad log pi(a|s))^T ]

    K-FAC approximation:
        F approx= A kron B
    where A and B are smaller matrices from input activations and
    output gradients of each layer. This reduces inversion cost from
    O(n^2) to O(n^{4/3}).

    Trust region constraint:
        KL(pi_old || pi_new) <= delta

    For our thesis: Start with Adam optimizer (simpler, still effective).
    ACKTR adds ~10-25% computational overhead but provides better convergence.
    Upgrade to ACKTR only if A2C performance is insufficient.

    Reference implementation: https://github.com/gd-zhang/ACKTR
    Used in this paper's code: https://github.com/alexfrom0815/Online-3D-BPP-DRL
    (modified from pytorch-a2c-ppo-acktr-gail)
    """
    pass


# =============================================================================
# 10. EXTENDED: DUAL-BIN NETWORK ARCHITECTURE OPTIONS
# =============================================================================

class DualBinConstrainedNetwork(nn.Module):
    """
    Extended architecture for 2-bounded space with buffer.

    Three architecture options described in the deep summary:

    Option A (implemented here): Dual height maps as state input
        - State: L x W x 5 (2 height maps + 3 item dim channels)
        - Single CNN processes both bins simultaneously
        - Actor outputs position probabilities for one bin at a time
        - Separate forward pass per bin, critic values compared

    Option B: Bin selection as separate action head
        - Additional 2-way softmax head for bin selection
        - Joint training of bin selection and position selection

    Option C: Hierarchical (recommended)
        - Step 1: For each (item, bin) pair, evaluate critic value
        - Step 2: Select (item, bin) with highest value
        - Step 3: Use actor + mask for position in chosen bin
    """

    def __init__(self, L: int = 10, W: int = 10, in_channels: int = 5,
                 hidden_channels: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.L = L
        self.W = W

        # CNN processes 5 channels: 2 height maps + 3 item dims
        self.state_cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
        )
        feature_dim = hidden_channels * L * W

        self.actor = Actor(feature_dim, L * W, hidden_dim)
        self.critic = Critic(feature_dim, hidden_dim)
        self.mask_predictor = MaskPredictor(feature_dim, L, W, hidden_dim)

        # Bin selection head (Option B)
        self.bin_selector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 2 bins
        )

    def forward(self, obs: torch.Tensor,
                gt_mask: Optional[torch.Tensor] = None):
        """
        Forward pass for dual-bin setup.

        obs shape: (batch, L, W, 5) where:
            channel 0: bin 0 height map (normalized)
            channel 1: bin 1 height map (normalized)
            channel 2: item length (broadcast, normalized)
            channel 3: item width (broadcast, normalized)
            channel 4: item height (broadcast, normalized)
        """
        x = obs.permute(0, 3, 1, 2)
        features = self.state_cnn(x).flatten(start_dim=1)

        predicted_mask = self.mask_predictor(features)
        projection_mask = gt_mask if gt_mask is not None else (
            (predicted_mask > 0.5).float()
        )

        action_probs = self.actor(features, projection_mask)
        value = self.critic(features)
        bin_probs = F.softmax(self.bin_selector(features), dim=-1)

        return {
            'action_probs': action_probs,
            'value': value,
            'predicted_mask': predicted_mask,
            'bin_probs': bin_probs,
        }


# =============================================================================
# 11. EXTENDED: DATASET GENERATORS (RS, CUT-1, CUT-2)
# =============================================================================

class DatasetGenerator:
    """
    Generate training/test sequences matching the paper's datasets.

    Three benchmarks from the paper:
    1. RS (Random Sequence): random items from 64 predefined types
    2. CUT-1: items from cutting, sorted by Z-coordinate
    3. CUT-2: items from cutting, sorted by stacking dependency

    Item types: all (l, w, h) with l in [1,L/2], w in [1,W/2], h in [1,H/2]
    For L=W=H=10: l,w,h in {1,2,3,4,5} -> 5^3 = 125 types
    Paper uses 64 types (unclear exact selection, possibly 4^3 = 64 with
    l,w,h in {1,2,3,4} or a curated subset).
    """

    def __init__(self, L: int = 10, W: int = 10, H: int = 10):
        self.L, self.W, self.H = L, W, H

        # Generate all valid item types (paper: 64 types)
        self.item_types = []
        for l in range(1, L // 2 + 1):
            for w in range(1, W // 2 + 1):
                for h in range(1, H // 2 + 1):
                    self.item_types.append(np.array([l, w, h]))
        # Total: (L/2) * (W/2) * (H/2) types
        # For L=W=H=10: 5*5*5 = 125 types
        # Paper says 64; may use l,w,h in {1,2,3,4} subset

    def generate_rs(self, min_volume_ratio: float = 1.0) -> List[np.ndarray]:
        """
        Generate Random Sequence (RS) dataset.

        Items sampled randomly until total volume >= bin volume * ratio.
        """
        bin_volume = self.L * self.W * self.H
        target_volume = bin_volume * min_volume_ratio
        sequence = []
        total_vol = 0

        while total_vol < target_volume:
            item = self.item_types[
                np.random.randint(len(self.item_types))
            ].copy()
            sequence.append(item)
            total_vol += np.prod(item)

        return sequence

    def generate_cut(self, sort_mode: str = 'cut2') -> List[np.ndarray]:
        """
        Generate CUT-1 or CUT-2 sequence by recursively cutting the bin.

        This is a simplified version. The paper's exact cutting procedure
        is: sequentially cut the bin into items from predefined types.
        Items can be perfectly repacked into the bin.

        sort_mode:
            'cut1': sort by Z coordinate of FLB (bottom to top)
            'cut2': sort by stacking dependency (topological order)
        """
        items_with_positions = []
        self._recursive_cut(
            0, 0, 0, self.L, self.W, self.H,
            items_with_positions
        )

        if sort_mode == 'cut1':
            # Sort by Z coordinate (bottom to top)
            items_with_positions.sort(key=lambda x: (x[1][2], x[1][0], x[1][1]))
        elif sort_mode == 'cut2':
            # Sort by stacking dependency
            # Item can only appear after all items supporting it
            items_with_positions = self._topological_sort(items_with_positions)

        return [item for item, _ in items_with_positions]

    def _recursive_cut(self, x, y, z, l, w, h, result, depth=0):
        """Recursively cut a box into smaller items."""
        if depth > 10 or l * w * h < 2:
            if l >= 1 and w >= 1 and h >= 1:
                result.append((np.array([l, w, h]), np.array([x, y, z])))
            return

        # Choose a random axis to cut along
        axis = np.random.choice([0, 1, 2])
        dims = [l, w, h]
        if dims[axis] <= 1:
            result.append((np.array([l, w, h]), np.array([x, y, z])))
            return

        # Random cut position (ensure both parts are >= 1)
        cut_pos = np.random.randint(1, dims[axis])

        if axis == 0:
            self._recursive_cut(x, y, z, cut_pos, w, h, result, depth+1)
            self._recursive_cut(x+cut_pos, y, z, l-cut_pos, w, h, result, depth+1)
        elif axis == 1:
            self._recursive_cut(x, y, z, l, cut_pos, h, result, depth+1)
            self._recursive_cut(x, y+cut_pos, z, l, w-cut_pos, h, result, depth+1)
        else:
            self._recursive_cut(x, y, z, l, w, cut_pos, result, depth+1)
            self._recursive_cut(x, y, z+cut_pos, l, w, h-cut_pos, result, depth+1)

    def _topological_sort(self, items_with_positions):
        """
        Sort items by stacking dependency (CUT-2 style).

        An item can only appear after all items it sits on top of.
        """
        # Build dependency graph: item i depends on item j if i sits on j
        n = len(items_with_positions)
        deps = [set() for _ in range(n)]

        for i in range(n):
            item_i, pos_i = items_with_positions[i]
            z_bottom_i = pos_i[2]
            if z_bottom_i == 0:
                continue  # Floor items have no dependencies

            for j in range(n):
                if i == j:
                    continue
                item_j, pos_j = items_with_positions[j]
                z_top_j = pos_j[2] + item_j[2]
                if z_top_j == z_bottom_i:
                    # Check horizontal overlap
                    x_overlap = (pos_i[0] < pos_j[0] + item_j[0] and
                                 pos_j[0] < pos_i[0] + item_i[0])
                    y_overlap = (pos_i[1] < pos_j[1] + item_j[1] and
                                 pos_j[1] < pos_i[1] + item_i[1])
                    if x_overlap and y_overlap:
                        deps[i].add(j)

        # Topological sort (Kahn's algorithm)
        from collections import deque
        in_degree = [len(d) for d in deps]
        queue = deque(i for i in range(n) if in_degree[i] == 0)
        result = []

        while queue:
            # Among available items, pick randomly for variety
            idx = queue.popleft()
            result.append(items_with_positions[idx])
            for i in range(n):
                if idx in deps[i]:
                    deps[i].remove(idx)
                    in_degree[i] -= 1
                    if in_degree[i] == 0:
                        queue.append(i)

        return result


# =============================================================================
# 12. EXTENDED: EVALUATION AND METRICS
# =============================================================================

class PackingEvaluator:
    """
    Evaluation utilities for bin packing agents.

    Metrics tracked (from the paper):
    1. Space utilization: volume_used / bin_volume
    2. Items packed: number of items successfully placed
    3. Decision time: wall-clock time per placement
    4. Stability rate: fraction of placements that are stable
    5. Win rate vs baselines (for comparison)
    """

    def __init__(self, env: OnlineBPPEnvironment,
                 network: ConstrainedPackingNetwork):
        self.env = env
        self.network = network

    def evaluate(self, num_episodes: int = 2000,
                 dataset_fn=None) -> dict:
        """
        Evaluate the agent over multiple episodes.

        Args:
            num_episodes: Number of test episodes
            dataset_fn: Function returning item sequences (RS/CUT-1/CUT-2)

        Returns:
            Dictionary of aggregate metrics with mean and std
        """
        utilizations = []
        items_counts = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                gt_mask = self.env.get_feasibility_mask()
                gt_mask_tensor = torch.FloatTensor(gt_mask).unsqueeze(0)

                with torch.no_grad():
                    output = self.network(
                        obs_tensor, gt_mask=gt_mask_tensor,
                        use_predicted_mask=True
                    )
                action = output['action_probs'].argmax(dim=1).item()
                obs, reward, done, info = self.env.step(action)

            utilizations.append(info['utilization'])
            items_counts.append(info.get('items_packed', 0))

        return {
            'space_utilization_mean': np.mean(utilizations),
            'space_utilization_std': np.std(utilizations),
            'items_packed_mean': np.mean(items_counts),
            'items_packed_std': np.std(items_counts),
            'num_episodes': num_episodes,
        }
