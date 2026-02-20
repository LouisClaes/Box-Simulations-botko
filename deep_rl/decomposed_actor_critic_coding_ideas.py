"""
CODING IDEAS: Decomposed Actor-Critic for Online 3D Bin Packing
================================================================
Source: "Learning Practically Feasible Policies for Online 3D Bin Packing"
         Zhao et al. (2023), arXiv:2108.13680v3

PURPOSE:
  Implement the decomposed actor-critic architecture that splits the placement
  action into 3 sequential sub-decisions: orientation (o), x-coordinate, and
  y-coordinate. This enables high-resolution spatial discretization (up to
  100x100) without action space explosion.

KEY INNOVATION:
  Instead of one actor predicting from 2*L*W actions (e.g., 20,000 for 100x100),
  we have three actor heads:
    - o-head: 2 outputs (orientation)
    - x-head: L outputs (x-coordinate, conditioned on o)
    - y-head: W outputs (y-coordinate, conditioned on o and x)
  Total: 2 + L + W = 202 outputs for L=W=100

ADAPTATION FOR 2-BOUNDED SPACE + BUFFER:
  We extend the architecture with two additional heads:
    - item-head: selects which item from the buffer (k outputs)
    - bin-head: selects which of the 2 active bins (2 outputs)
  This creates a 5-head decomposed actor: item -> bin -> o -> x -> y

ESTIMATED TRAINING TIME: ~12 hours on Titan V GPU for 100x100 resolution
INFERENCE TIME: < 10ms per decision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PolicyConfig:
    """Configuration for the decomposed actor-critic policy."""
    bin_length: int = 100  # L (discrete grid resolution along length)
    bin_width: int = 100   # W (discrete grid resolution along width)
    bin_height: int = 100  # H
    n_orientations: int = 2  # Horizontal rotations only: [l,w,h] and [w,l,h]

    # Network architecture
    cnn_channels: List[int] = None  # Default: [32, 64, 128, 256, 256]
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256

    # Training hyperparameters
    gamma: float = 1.0       # Discount factor (1.0 = no discounting, finite horizon)
    alpha_actor: float = 1.0  # Actor loss weight
    beta_critic: float = 0.5  # Critic loss weight
    omega_infeasible: float = 0.01  # Infeasibility loss weight
    psi_entropy: float = 0.01      # Entropy bonus weight

    # Reward parameters
    alpha_reward: float = 10.0   # Volume reward scale
    beta_reward: float = 0.1     # Safe LP reward scale

    # Infeasibility epsilon
    epsilon_infeasible: float = 1e-20  # Small prob for infeasible actions

    # Buffer and multi-bin extension
    buffer_size: int = 5  # Number of items visible (for semi-online)
    n_active_bins: int = 2  # Number of active bins (2-bounded space)

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128, 256, 256]


# =============================================================================
# NETWORK COMPONENTS
# =============================================================================

class SharedCNNEncoder(nn.Module):
    """
    5-layer CNN with LeakyReLU that processes the bin state.

    Input: (batch, channels, L, W) tensor where:
      - Channel 0: Height map H_n
      - Channels 1-3: Item dimensions (l_n, w_n, h_n) broadcast over grid
      - Channels 4-5: Feasibility masks M_{n,0} and M_{n,1} for 2 orientations

    For 2-bounded space extension, we can either:
      a) Process each bin separately and concatenate features
      b) Stack both bins' info into additional channels
    """

    def __init__(self, config: PolicyConfig, in_channels: int = 6):
        super().__init__()
        channels = config.cnn_channels
        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers.append(nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(0.2))
            prev_ch = ch
        self.conv_net = nn.Sequential(*layers)
        self.output_channels = channels[-1]
        # Adaptive pooling to fixed size for the actor/critic MLPs
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, C, L, W) tensor
        Returns:
            features: (batch, feature_dim) flattened feature vector
        """
        x = self.conv_net(state)
        x = self.pool(x)
        return x.flatten(start_dim=1)  # (batch, channels * 8 * 8)


class OrientationHead(nn.Module):
    """
    o-head: Predicts P(o_n | s_n)
    2 outputs: probability of each orientation.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, n_orientations: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_orientations)
        )

    def forward(self, features: torch.Tensor,
                feasibility_mask_o: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: (batch, feature_dim)
            feasibility_mask_o: (batch, n_orientations) - 1 if at least one
                                feasible cell exists for this orientation

        Returns:
            log_probs: (batch, n_orientations)
        """
        logits = self.mlp(features)

        if feasibility_mask_o is not None:
            # Mask out completely infeasible orientations
            logits = logits.masked_fill(feasibility_mask_o == 0, float('-inf'))

        return F.log_softmax(logits, dim=-1)


class XCoordinateHead(nn.Module):
    """
    x-head: Predicts P(x_n | s_n, o_n)
    L outputs: probability for each x-coordinate.
    Conditioned on the chosen orientation via concatenation.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, n_x: int,
                 n_orientations: int = 2):
        super().__init__()
        # Orientation is provided as a one-hot vector concatenated to features
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + n_orientations, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_x)
        )

    def forward(self, features: torch.Tensor, orientation_onehot: torch.Tensor,
                feasibility_mask_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: (batch, feature_dim)
            orientation_onehot: (batch, n_orientations) one-hot encoding of chosen o
            feasibility_mask_x: (batch, L) - 1 if at least one feasible cell
                                exists in this x-column for the chosen orientation

        Returns:
            log_probs: (batch, L)
        """
        x = torch.cat([features, orientation_onehot], dim=-1)
        logits = self.mlp(x)

        if feasibility_mask_x is not None:
            logits = logits.masked_fill(feasibility_mask_x == 0, float('-inf'))

        return F.log_softmax(logits, dim=-1)


class YCoordinateHead(nn.Module):
    """
    y-head: Predicts P(y_n | s_n, o_n, x_n)
    W outputs: probability for each y-coordinate.
    Conditioned on orientation AND x-coordinate.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, n_y: int,
                 n_orientations: int = 2, n_x: int = 100):
        super().__init__()
        # Condition on orientation (one-hot) and x (one-hot or scalar)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + n_orientations + 1, hidden_dim),  # +1 for x as scalar
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_y)
        )

    def forward(self, features: torch.Tensor, orientation_onehot: torch.Tensor,
                x_normalized: torch.Tensor,
                feasibility_mask_y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: (batch, feature_dim)
            orientation_onehot: (batch, n_orientations)
            x_normalized: (batch, 1) - chosen x coordinate normalized to [0, 1]
            feasibility_mask_y: (batch, W) - 1 if cell (x, y) is feasible

        Returns:
            log_probs: (batch, W)
        """
        x = torch.cat([features, orientation_onehot, x_normalized], dim=-1)
        logits = self.mlp(x)

        if feasibility_mask_y is not None:
            logits = logits.masked_fill(feasibility_mask_y == 0, float('-inf'))

        return F.log_softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    """
    Shared critic: V(s_n)
    Predicts state value for advantage estimation.
    """

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns: (batch, 1) state value."""
        return self.mlp(features)


# =============================================================================
# EXTENDED HEADS FOR 2-BOUNDED SPACE + BUFFER
# =============================================================================

class ItemSelectionHead(nn.Module):
    """
    EXTENSION: item-head for buffer selection.
    Predicts P(item_idx | s_n) where item_idx is which item in the buffer
    to pack next.

    This head is NOT in the original paper -- it is our extension for the
    semi-online buffer setting.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, max_buffer_size: int = 10):
        super().__init__()
        # Each item in the buffer has its own feature embedding
        self.item_encoder = nn.Linear(3, 32)  # item dims (l, w, h) -> 32-dim
        self.attention = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + 32, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, max_buffer_size)
        )

    def forward(self, features: torch.Tensor, buffer_items: torch.Tensor,
                buffer_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: (batch, feature_dim) - bin state features
            buffer_items: (batch, max_buffer_size, 3) - item dimensions in buffer
            buffer_mask: (batch, max_buffer_size) - 1 if slot has an item

        Returns:
            log_probs: (batch, max_buffer_size)
        """
        item_feats = self.item_encoder(buffer_items)  # (batch, K, 32)
        # Self-attention over buffer items
        attn_out, _ = self.attention(item_feats, item_feats, item_feats)
        # Pool to single vector
        pooled = attn_out.mean(dim=1)  # (batch, 32)

        x = torch.cat([features, pooled], dim=-1)
        logits = self.mlp(x)

        if buffer_mask is not None:
            logits = logits.masked_fill(buffer_mask == 0, float('-inf'))

        return F.log_softmax(logits, dim=-1)


class BinSelectionHead(nn.Module):
    """
    EXTENSION: bin-head for 2-bounded space.
    Predicts P(bin_idx | s_n, item) -- which of the 2 active bins to use.

    This head is NOT in the original paper -- it is our extension for the
    2-bounded space setting.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, n_bins: int = 2):
        super().__init__()
        # Takes features from both bins + selected item features
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * n_bins + 3, hidden_dim),  # +3 for item dims
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_bins)
        )

    def forward(self, bin_features: torch.Tensor, item_dims: torch.Tensor,
                bin_active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            bin_features: (batch, n_bins * feature_dim) - concatenated bin features
            item_dims: (batch, 3) - dimensions of selected item
            bin_active_mask: (batch, n_bins) - 1 if bin is active

        Returns:
            log_probs: (batch, n_bins)
        """
        x = torch.cat([bin_features, item_dims], dim=-1)
        logits = self.mlp(x)

        if bin_active_mask is not None:
            logits = logits.masked_fill(bin_active_mask == 0, float('-inf'))

        return F.log_softmax(logits, dim=-1)


# =============================================================================
# COMPLETE POLICY NETWORK
# =============================================================================

class DecomposedBinPackingPolicy(nn.Module):
    """
    Complete decomposed actor-critic policy for online 3D bin packing.

    Original paper (single bin, single item):
      State -> CNN -> [o-head] -> [x-head | o] -> [y-head | o, x]
                   -> [critic] -> V(s)

    Our extension (2 bins, buffer):
      State -> CNN -> [item-head] -> [bin-head | item]
                   -> [o-head | bin] -> [x-head | bin, o] -> [y-head | bin, o, x]
                   -> [critic] -> V(s)
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config

        # Compute state channels:
        # For single bin: 6 channels (height_map, l, w, h, mask_o0, mask_o1)
        # For dual bin: 12 channels (6 per bin) or process separately
        n_state_channels = 6  # Per bin

        self.encoder = SharedCNNEncoder(config, in_channels=n_state_channels)

        # Compute feature dimension after CNN + pooling
        # CNN output channels * 8 * 8 (from adaptive pooling)
        feature_dim = config.cnn_channels[-1] * 8 * 8

        # Original paper heads
        self.o_head = OrientationHead(feature_dim, config.actor_hidden_dim, config.n_orientations)
        self.x_head = XCoordinateHead(feature_dim, config.actor_hidden_dim,
                                       config.bin_length, config.n_orientations)
        self.y_head = YCoordinateHead(feature_dim, config.actor_hidden_dim,
                                       config.bin_width, config.n_orientations,
                                       config.bin_length)
        self.critic = CriticNetwork(feature_dim, config.critic_hidden_dim)

        # Extension heads for semi-online + 2-bounded
        if config.buffer_size > 1:
            self.item_head = ItemSelectionHead(feature_dim, config.actor_hidden_dim,
                                               config.buffer_size)
        if config.n_active_bins > 1:
            self.bin_head = BinSelectionHead(feature_dim, config.actor_hidden_dim,
                                             config.n_active_bins)

    def forward_single_bin(self, state: torch.Tensor,
                           feasibility_masks: Dict[str, torch.Tensor]
                           ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for single-bin BPP-1 (original paper formulation).

        Args:
            state: (batch, 6, L, W) - bin state tensor
            feasibility_masks: dict with keys 'o', 'x', 'y' containing masks

        Returns:
            o_log_probs, x_log_probs, y_log_probs, value
        """
        features = self.encoder(state)

        # Step 1: Orientation
        o_log_probs = self.o_head(features, feasibility_masks.get('o'))
        o_dist = torch.distributions.Categorical(logits=o_log_probs)
        o_selected = o_dist.sample()
        o_onehot = F.one_hot(o_selected, self.config.n_orientations).float()

        # Step 2: X-coordinate
        x_log_probs = self.x_head(features, o_onehot, feasibility_masks.get('x'))
        x_dist = torch.distributions.Categorical(logits=x_log_probs)
        x_selected = x_dist.sample()
        x_normalized = (x_selected.float() / self.config.bin_length).unsqueeze(-1)

        # Step 3: Y-coordinate
        y_log_probs = self.y_head(features, o_onehot, x_normalized,
                                   feasibility_masks.get('y'))

        # Critic
        value = self.critic(features)

        return o_log_probs, x_log_probs, y_log_probs, value, o_selected, x_selected

    def forward_multi_bin_buffer(self, bin_states: List[torch.Tensor],
                                  buffer_items: torch.Tensor,
                                  buffer_mask: torch.Tensor,
                                  bin_active_mask: torch.Tensor,
                                  feasibility_masks: Dict
                                  ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for 2-bounded space with buffer (our extension).

        Args:
            bin_states: list of 2 tensors, each (batch, 6, L, W) for each active bin
            buffer_items: (batch, buffer_size, 3) - item dimensions in buffer
            buffer_mask: (batch, buffer_size) - which buffer slots have items
            bin_active_mask: (batch, 2) - which bins are active
            feasibility_masks: nested dict with per-bin, per-orientation masks

        Returns:
            Dictionary with log_probs for each head and selected actions
        """
        # Encode both bins
        bin_features = [self.encoder(bs) for bs in bin_states]
        cat_bin_features = torch.cat(bin_features, dim=-1)

        # We use bin 0's features as the "primary" for item selection
        primary_features = bin_features[0]

        results = {}

        # Step 1: Select item from buffer
        item_log_probs = self.item_head(primary_features, buffer_items, buffer_mask)
        item_dist = torch.distributions.Categorical(logits=item_log_probs)
        item_idx = item_dist.sample()
        results['item_log_probs'] = item_log_probs
        results['item_idx'] = item_idx

        # Get selected item dimensions
        batch_idx = torch.arange(item_idx.shape[0])
        selected_item_dims = buffer_items[batch_idx, item_idx]  # (batch, 3)

        # Step 2: Select bin
        bin_log_probs = self.bin_head(cat_bin_features, selected_item_dims, bin_active_mask)
        bin_dist = torch.distributions.Categorical(logits=bin_log_probs)
        bin_idx = bin_dist.sample()
        results['bin_log_probs'] = bin_log_probs
        results['bin_idx'] = bin_idx

        # Use features from the selected bin for placement
        # (In practice, select features based on bin_idx per batch element)
        selected_features = torch.where(
            bin_idx.unsqueeze(-1).expand_as(bin_features[0]) == 0,
            bin_features[0], bin_features[1]
        )

        # Steps 3-5: Orientation, X, Y (same as single bin, using selected bin's features)
        o_mask = feasibility_masks.get('o')  # Would be per-bin in practice
        o_log_probs = self.o_head(selected_features, o_mask)
        o_dist = torch.distributions.Categorical(logits=o_log_probs)
        o_selected = o_dist.sample()
        o_onehot = F.one_hot(o_selected, self.config.n_orientations).float()
        results['o_log_probs'] = o_log_probs
        results['o_selected'] = o_selected

        x_mask = feasibility_masks.get('x')
        x_log_probs = self.x_head(selected_features, o_onehot, x_mask)
        x_dist = torch.distributions.Categorical(logits=x_log_probs)
        x_selected = x_dist.sample()
        x_norm = (x_selected.float() / self.config.bin_length).unsqueeze(-1)
        results['x_log_probs'] = x_log_probs
        results['x_selected'] = x_selected

        y_mask = feasibility_masks.get('y')
        y_log_probs = self.y_head(selected_features, o_onehot, x_norm, y_mask)
        results['y_log_probs'] = y_log_probs

        # Critic uses concatenated features from both bins
        value = self.critic(cat_bin_features[:, :bin_features[0].shape[-1]])
        results['value'] = value

        return results


# =============================================================================
# LOSS COMPUTATION
# =============================================================================

class DecomposedLoss:
    """
    Composite loss function from the paper:
      L = alpha * L_actor + beta * L_critic + omega * E_inf - psi * E_entropy

    Where:
      L_actor = advantage * log P(a|s)  for each actor head
      L_critic = (r + gamma*V(s') - V(s))^2
      E_inf = sum of probabilities assigned to infeasible actions
      E_entropy = -sum of P * log P over feasible actions (encourages exploration)
    """

    def __init__(self, config: PolicyConfig):
        self.config = config

    def compute(self,
                log_probs: Dict[str, torch.Tensor],
                actions: Dict[str, torch.Tensor],
                rewards: torch.Tensor,
                values: torch.Tensor,
                next_values: torch.Tensor,
                feasibility_masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the composite loss.

        Args:
            log_probs: dict of log probabilities from each head
            actions: dict of selected actions for each head
            rewards: (batch,) reward values
            values: (batch,) V(s_n) predictions
            next_values: (batch,) V(s_{n+1}) predictions
            feasibility_masks: dict of feasibility masks per head

        Returns:
            total_loss: scalar tensor
        """
        gamma = self.config.gamma
        advantage = rewards + gamma * next_values.detach() - values.detach()

        # Actor losses (sum across all heads)
        actor_loss = torch.tensor(0.0)
        for head_name in ['o', 'x', 'y']:
            if head_name in log_probs and head_name in actions:
                head_log_prob = log_probs[head_name]
                head_action = actions[head_name]
                # Gather log probability of selected action
                selected_log_prob = head_log_prob.gather(1, head_action.unsqueeze(-1)).squeeze(-1)
                actor_loss = actor_loss - (advantage * selected_log_prob).mean()

        # Critic loss
        td_target = rewards + gamma * next_values.detach()
        critic_loss = F.mse_loss(values.squeeze(), td_target)

        # Infeasibility loss: penalize probability mass on infeasible actions
        infeasibility_loss = torch.tensor(0.0)
        for head_name, mask in feasibility_masks.items():
            if head_name in log_probs:
                probs = log_probs[head_name].exp()
                # Sum of probs where mask == 0 (infeasible)
                infeasible_prob = (probs * (1 - mask)).sum(dim=-1).mean()
                infeasibility_loss = infeasibility_loss + infeasible_prob

        # Entropy bonus: encourage exploration over feasible actions
        entropy = torch.tensor(0.0)
        for head_name, mask in feasibility_masks.items():
            if head_name in log_probs:
                probs = log_probs[head_name].exp()
                log_p = log_probs[head_name]
                # Entropy only over feasible actions
                head_entropy = -(probs * log_p * mask).sum(dim=-1).mean()
                entropy = entropy + head_entropy

        total_loss = (
            self.config.alpha_actor * actor_loss
            + self.config.beta_critic * critic_loss
            + self.config.omega_infeasible * infeasibility_loss
            - self.config.psi_entropy * entropy
        )

        return total_loss


# =============================================================================
# REWARD FUNCTION
# =============================================================================

def compute_reward(item_volume: float, bin_volume: float,
                   safe_volume: float, alpha: float = 10.0,
                   beta: float = 0.1) -> float:
    """
    Reward function from the paper:
      r_n = alpha * vol_item / vol_bin + beta * V_safe / vol_bin

    The first term rewards packing efficiency.
    The second term rewards collision-free (far-to-near) placement.

    Args:
        item_volume: volume of the placed item
        bin_volume: total bin volume (L * W * H)
        safe_volume: sum of volumes of safe loading points
        alpha: volume reward scale (default 10)
        beta: safe LP reward scale (default 0.1)

    Returns:
        reward value
    """
    return alpha * item_volume / bin_volume + beta * safe_volume / bin_volume


def compute_safe_volume(height_map: np.ndarray, item_x: float, item_y: float,
                        entrance_y: int, bin_L: int, bin_W: int,
                        bin_H: int) -> float:
    """
    Compute V_safe: the sum of volumes of all safe loading points.

    A loading point (x, y) is "safe" if there is no obstacle on the straight
    line from the entrance line (EL) to that point.

    For simplicity, we assume the robot enters along y = entrance_y.
    A cell (x, y) is safe if all cells (x, y') for y' between entrance_y and y
    have height_map[x, y'] <= height_map[x, y] (no blocking obstacle).

    This is a simplification -- the paper's exact definition involves
    checking straight-line reachability from EL to the LP.
    """
    safe_volume = 0.0
    for x in range(bin_L):
        for y in range(bin_W):
            # Check if straight line from entrance is clear
            is_safe = True
            target_height = height_map[x, y]

            # Check all cells between entrance and target
            y_start = min(entrance_y, y)
            y_end = max(entrance_y, y)
            for y_check in range(y_start, y_end):
                if height_map[x, y_check] > target_height:
                    is_safe = False
                    break

            if is_safe:
                # Safe volume contribution: remaining height at this cell
                remaining = bin_H - height_map[x, y]
                safe_volume += remaining

    return safe_volume


# =============================================================================
# STATE CONSTRUCTION
# =============================================================================

def construct_state_tensor(height_map: np.ndarray, item_dims: Tuple[float, float, float],
                           feasibility_mask_o0: np.ndarray,
                           feasibility_mask_o1: np.ndarray) -> torch.Tensor:
    """
    Construct the 6-channel state tensor for the CNN encoder.

    From the paper:
      Channel 0: H_n (height map)
      Channel 1: l_n broadcast over L x W
      Channel 2: w_n broadcast over L x W
      Channel 3: h_n broadcast over L x W
      Channel 4: M_{n,0} (feasibility mask for orientation 0)
      Channel 5: M_{n,1} (feasibility mask for orientation 1)

    Args:
        height_map: (L, W) integer array of current heights
        item_dims: (l, w, h) tuple of current item dimensions
        feasibility_mask_o0: (L, W) binary mask for orientation 0
        feasibility_mask_o1: (L, W) binary mask for orientation 1

    Returns:
        state: (1, 6, L, W) tensor ready for the CNN
    """
    L, W = height_map.shape
    l, w, h = item_dims

    state = np.zeros((6, L, W), dtype=np.float32)
    state[0] = height_map
    state[1] = np.full((L, W), l, dtype=np.float32)
    state[2] = np.full((L, W), w, dtype=np.float32)
    state[3] = np.full((L, W), h, dtype=np.float32)
    state[4] = feasibility_mask_o0
    state[5] = feasibility_mask_o1

    return torch.from_numpy(state).unsqueeze(0)  # Add batch dimension


# =============================================================================
# TRAINING LOOP PSEUDOCODE
# =============================================================================

def training_loop_pseudocode():
    """
    Pseudocode for training the decomposed actor-critic.
    NOT executable -- for reference only.

    The paper uses ACKTR (Actor-Critic with K-FAC trust region).
    A simpler PPO-based alternative is shown here for clarity.
    """
    """
    PSEUDOCODE:

    config = PolicyConfig(bin_length=100, bin_width=100, buffer_size=5, n_active_bins=2)
    policy = DecomposedBinPackingPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    loss_fn = DecomposedLoss(config)

    for episode in range(num_episodes):
        # Initialize environment
        bins = [StackingTree(100, 100, 100), StackingTree(100, 100, 100)]
        buffer = generate_initial_buffer(size=config.buffer_size)

        done = False
        trajectory = []

        while not done:
            # Construct states for both bins
            bin_states = [construct_state_tensor(
                bins[i].get_height_map(),
                buffer[0].dims,  # Current item
                bins[i].compute_feasibility_mask(buffer[0], 0),
                bins[i].compute_feasibility_mask(buffer[0], 1)
            ) for i in range(2)]

            # Forward pass
            results = policy.forward_multi_bin_buffer(
                bin_states, buffer_tensor, buffer_mask, bin_active_mask, feas_masks
            )

            # Execute action
            item_idx = results['item_idx']
            bin_idx = results['bin_idx']
            o, x, y = results['o_selected'], results['x_selected'], results['y_selected']

            # Place item
            selected_item = buffer[item_idx]
            success = bins[bin_idx].place_item(selected_item, x, y, o)

            # Compute reward
            reward = compute_reward(selected_item.volume, bin_volume, safe_volume)

            # Store transition
            trajectory.append((results, reward))

            # Refill buffer from arrival stream
            buffer.pop(item_idx)
            buffer.append(next_arriving_item())

            # Check termination: no feasible placement for any buffer item on any bin
            done = check_all_infeasible(bins, buffer)

        # Update policy using trajectory
        loss = compute_trajectory_loss(trajectory, loss_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    """
    pass


# =============================================================================
# KEY IMPLEMENTATION NOTES
# =============================================================================

"""
IMPORTANT NOTES FOR IMPLEMENTATION:

1. ACKTR vs PPO:
   The paper uses ACKTR (Wu et al. 2017) which is more sample-efficient than
   PPO for this problem. However, K-FAC is complex to implement. Start with
   PPO and switch to ACKTR if needed. The kfac-pytorch library provides a
   PyTorch implementation of K-FAC.

2. Feasibility Mask Computation:
   The feasibility masks are computed from the stacking tree (see
   stacking_tree_coding_ideas.py). This is done OUTSIDE the neural network
   and fed as input. The masks are NOT learned -- they are physics-based.

3. Conditional Dependency:
   The autoregressive structure (o -> x -> y) means during training, we
   use the SAMPLED action from the previous head as input to the next.
   During inference, we can use either sampling or argmax.

4. Resolution Choice:
   - 10x10: Fast training (~30 min), good for prototyping. ~70% utilization.
   - 50x50: Medium training (~4 hours). ~72.6% utilization.
   - 100x100: Full training (~12 hours). ~71.3% utilization.
   The slight drop from 50x50 to 100x100 is due to CNN kernel size effects.

5. Multi-Bin Extension Training Strategy:
   Option A: Train end-to-end (all 5 heads together). Harder but optimal.
   Option B: Pre-train single-bin policy (o, x, y heads) on BPP-1, then
             freeze and train only item and bin heads. Faster convergence.
   Option C: Use heuristic for item/bin selection and only learn placement.
             Simplest; lets us reuse the paper's trained weights if available.

6. MCTS for Buffer:
   For buffer size k=5-10 with 2 bins, the search space is large:
   k items * 2 bins * ~500 feasible placements ≈ 5000-10000 options per step.
   MCTS with the trained value function as evaluation is essential.
   Budget: 100-500 MCTS rollouts per step (paper uses ~100 for k=5).
"""
