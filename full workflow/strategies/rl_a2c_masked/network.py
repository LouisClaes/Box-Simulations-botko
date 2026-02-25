"""
A2C network with learned feasibility mask predictor.

Architecture following Zhao et al. (AAAI 2021):

    +-----------+     +-----------+
    | Bin 0 HM  |     | Bin 1 HM  |   4-channel each: (height, item_l, item_w, item_h)
    | (4,120,80)|     | (4,120,80)|
    +-----+-----+     +-----+-----+
          |                 |
     Shared CNN        Shared CNN     (weight-sharing)
          |                 |
       512-dim           512-dim
          |                 |
          +--------+--------+
                   |
              1024-dim  (concatenated bin features)
                   |
                   +--------+--------+
                            |        |
                        1024-dim  128-dim  (item embedding)
                            |        |
                            +---+----+
                                |
                           1152-dim  (combined features)
                          /    |    \\
                      Actor  Critic  Mask Predictor
                        |      |        |
                     softmax  V(s)   sigmoid
                   (1536-d)  (1)   (1536-d)

Key innovation: the Mask Predictor head predicts P(valid|state, action)
for each action.  During forward pass, predicted masks are applied to
the actor logits to zero out infeasible actions BEFORE softmax:

    masked_logits = logits + (1 - mask) * (-1e9)
    pi = softmax(masked_logits)

The mask predictor is trained with BCE against ground-truth validity
masks computed by the environment.

Loss function (5 components):
    L = alpha * L_actor + beta * L_critic + lambda * L_mask
        + omega * E_infeasibility - psi * E_entropy

References:
    - Zhao et al. (AAAI 2021): Online 3D BPP with feasibility masking
    - Wu et al. (2017): Scalable A2C with GAE
"""

from __future__ import annotations

import sys
import os
import math
from typing import Tuple, Optional, Dict, NamedTuple

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from strategies.rl_a2c_masked.config import A2CMaskedConfig


# ---------------------------------------------------------------------------
# Network output container
# ---------------------------------------------------------------------------

class NetworkOutput(NamedTuple):
    """Container for all network outputs in a single forward pass."""
    action_logits: torch.Tensor     # (B, num_actions) raw logits before masking
    masked_logits: torch.Tensor     # (B, num_actions) logits after mask application
    policy: torch.Tensor            # (B, num_actions) softmax probabilities
    value: torch.Tensor             # (B, 1) state value estimate
    mask_pred: torch.Tensor         # (B, num_actions) predicted feasibility mask [0,1]


# ---------------------------------------------------------------------------
# Shared CNN encoder (per-bin)
# ---------------------------------------------------------------------------

class SharedCNNEncoder(nn.Module):
    """
    CNN encoder for a single bin's 4-channel heightmap.

    Processes a (B, 4, grid_l, grid_w) tensor through:
        Conv2d(4, 32, 3, pad=1) -> ReLU
        Conv2d(32, 64, 3, pad=1) -> ReLU
        Conv2d(64, 64, 3, stride=2, pad=1) -> ReLU
        Conv2d(64, 128, 3, stride=2, pad=1) -> ReLU
        Conv2d(128, 128, 3, stride=2, pad=1) -> ReLU
        AdaptiveAvgPool2d(4, 4) -> Flatten -> 2048
        Dense(2048, 512) -> ReLU

    Output: (B, 512) per-bin embedding.
    """

    def __init__(self, config: A2CMaskedConfig) -> None:
        super().__init__()
        self.config = config

        channels = config.cnn_channels   # (32, 64, 64, 128, 128)
        kernels = config.cnn_kernels     # (3, 3, 3, 3, 3)
        strides = config.cnn_strides     # (1, 1, 2, 2, 2)
        paddings = config.cnn_paddings   # (1, 1, 1, 1, 1)

        layers = []
        in_ch = config.cnn_input_channels  # 4
        for out_ch, k, s, p in zip(channels, kernels, strides, paddings):
            layers.append(nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(config.cnn_pool_size)  # (4, 4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(config.cnn_flat_dim, config.cnn_flat_to_embed)
        self.fc_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4, grid_l, grid_w) — 4-channel heightmap for one bin.

        Returns:
            (B, 512) per-bin embedding.
        """
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc_relu(self.fc(x))
        return x


# ---------------------------------------------------------------------------
# Item (box) MLP encoder
# ---------------------------------------------------------------------------

class ItemEncoder(nn.Module):
    """
    MLP encoder for the current box features.

    Input: (B, 5) — (l, w, h, vol, weight) normalised.
    Output: (B, 128) item embedding.

    Architecture:
        Dense(5, 64) -> ReLU -> Dense(64, 128)
    """

    def __init__(self, config: A2CMaskedConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.item_input_dim, config.item_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.item_hidden_dim, config.item_embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 5) normalised box features.

        Returns:
            (B, 128) item embedding.
        """
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Main A2C network with mask predictor
# ---------------------------------------------------------------------------

class A2CMaskedNetwork(nn.Module):
    """
    A2C Actor-Critic network with learned feasibility mask predictor.

    Three output heads share a common feature representation:

    1. **Actor** (policy head):
       Dense(1152, 256) -> ReLU -> Dense(256, num_actions)
       Output: raw logits, masked by predicted feasibility.

    2. **Critic** (value head):
       Dense(1152, 256) -> ReLU -> Dense(256, 1)
       Output: scalar state value V(s).

    3. **Mask Predictor** (feasibility head):
       Dense(1152, 256) -> ReLU -> Dense(256, num_actions) -> Sigmoid
       Output: P(valid|state, action) for each action.

    The mask predictor is the KEY INNOVATION from Zhao et al. (AAAI 2021).
    Instead of computing exact validity (expensive for large action spaces),
    a neural network learns to predict which actions are feasible.

    Args:
        config: A2CMaskedConfig with architecture hyperparameters.
    """

    def __init__(self, config: A2CMaskedConfig) -> None:
        super().__init__()
        self.config = config
        self.num_actions = config.num_actions

        # Shared encoder components
        self.cnn_encoder = SharedCNNEncoder(config)
        self.item_encoder = ItemEncoder(config)

        # Feature dimensions
        combined_dim = config.combined_features  # 1024 + 128 = 1152

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, config.actor_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.actor_hidden, config.num_actions),
        )

        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, config.critic_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.critic_hidden, 1),
        )

        # Mask predictor head (feasibility)
        self.mask_predictor = nn.Sequential(
            nn.Linear(combined_dim, config.mask_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.mask_hidden, config.num_actions),
            nn.Sigmoid(),
        )

        # Initialise weights using orthogonal init (standard for A2C)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialization (Saxe et al. 2014)."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Smaller init for output heads (PPO/A2C standard practice)
        for head in [self.actor, self.critic, self.mask_predictor]:
            # Last linear layer gets small-scale init
            last_linear = None
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None:
                nn.init.orthogonal_(last_linear.weight, gain=0.01)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)

    def _encode_bins(
        self,
        heightmaps: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode all bin heightmaps through the shared CNN.

        For each bin, creates a 4-channel input:
            channel 0: normalised heightmap
            channel 1: item_l / bin_l (broadcast)
            channel 2: item_w / bin_w (broadcast)
            channel 3: item_h / bin_h (broadcast)

        Args:
            heightmaps: (B, num_bins, grid_l, grid_w) normalised heightmaps.
            item_features: (B, 5) box features [l, w, h, vol, weight].

        Returns:
            (B, num_bins * 512) concatenated bin embeddings.
        """
        B = heightmaps.shape[0]
        num_bins = self.config.num_bins
        grid_l = self.config.grid_l
        grid_w = self.config.grid_w

        bin_embeds = []
        for i in range(num_bins):
            # Get heightmap for this bin: (B, grid_l, grid_w)
            hm = heightmaps[:, i]

            # Build 4-channel input: (B, 4, grid_l, grid_w)
            ch0 = hm.unsqueeze(1)                                    # (B, 1, H, W)
            ch1 = item_features[:, 0].view(B, 1, 1, 1).expand(B, 1, grid_l, grid_w)
            ch2 = item_features[:, 1].view(B, 1, 1, 1).expand(B, 1, grid_l, grid_w)
            ch3 = item_features[:, 2].view(B, 1, 1, 1).expand(B, 1, grid_l, grid_w)
            x = torch.cat([ch0, ch1, ch2, ch3], dim=1)               # (B, 4, H, W)

            embed = self.cnn_encoder(x)  # (B, 512)
            bin_embeds.append(embed)

        return torch.cat(bin_embeds, dim=1)  # (B, num_bins * 512)

    def forward(
        self,
        heightmaps: torch.Tensor,
        item_features: torch.Tensor,
        true_mask: Optional[torch.Tensor] = None,
    ) -> NetworkOutput:
        """
        Full forward pass through all three heads.

        Args:
            heightmaps:   (B, num_bins, grid_l, grid_w) normalised heightmaps.
            item_features: (B, 5) normalised box features [l, w, h, vol, weight].
            true_mask:     (B, num_actions) optional ground-truth mask.
                           If provided, used instead of predicted mask during
                           training for more stable learning.

        Returns:
            NetworkOutput with all outputs.
        """
        # Encode bins through shared CNN
        bin_features = self._encode_bins(heightmaps, item_features)  # (B, 1024)

        # Encode item
        item_embed = self.item_encoder(item_features)  # (B, 128)

        # Combine features
        combined = torch.cat([bin_features, item_embed], dim=1)  # (B, 1152)

        # Actor head: raw logits
        action_logits = self.actor(combined)  # (B, num_actions)

        # Critic head: state value
        value = self.critic(combined)  # (B, 1)

        # Mask predictor head: predicted feasibility
        mask_pred = self.mask_predictor(combined)  # (B, num_actions)

        # Apply mask to logits
        # During training with ground truth available, use true_mask for
        # actor gradient stability; mask_pred is trained via BCE separately.
        if true_mask is not None:
            mask_for_policy = true_mask
        else:
            # At inference or when true mask unavailable, use predicted mask
            mask_for_policy = (mask_pred > self.config.mask_threshold).float()

        # Mask application: infeasible actions get -inf logits
        masked_logits = action_logits + (1.0 - mask_for_policy) * (-1e9)

        # Check if all actions are masked (edge case)
        all_masked = (mask_for_policy.sum(dim=-1, keepdim=True) == 0)
        if all_masked.any():
            # Fall back to uniform over all actions
            masked_logits = torch.where(
                all_masked.expand_as(masked_logits),
                action_logits,  # Use raw logits as fallback
                masked_logits,
            )

        # Policy: softmax over masked logits
        policy = F.softmax(masked_logits, dim=-1)

        return NetworkOutput(
            action_logits=action_logits,
            masked_logits=masked_logits,
            policy=policy,
            value=value,
            mask_pred=mask_pred,
        )

    def get_action_and_value(
        self,
        heightmaps: torch.Tensor,
        item_features: torch.Tensor,
        true_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select an action and compute associated quantities for training.

        If action is None, samples (or greedy-selects) a new action.
        If action is provided, evaluates that action under the current policy.

        Args:
            heightmaps:    (B, num_bins, grid_l, grid_w).
            item_features: (B, 5).
            true_mask:     (B, num_actions) optional ground-truth mask.
            action:        (B,) optional pre-selected actions to evaluate.
            deterministic: If True, use argmax instead of sampling.

        Returns:
            Tuple of:
                action:     (B,) selected or evaluated action indices.
                log_prob:   (B,) log probability of the action.
                entropy:    (B,) entropy of the policy distribution.
                value:      (B,) state value estimate.
                mask_pred:  (B, num_actions) predicted feasibility mask.
        """
        output = self.forward(heightmaps, item_features, true_mask=true_mask)

        dist = Categorical(probs=output.policy)

        if action is None:
            if deterministic:
                action = output.policy.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = output.value.squeeze(-1)

        return action, log_prob, entropy, value, output.mask_pred


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

class A2CMaskedLoss(nn.Module):
    """
    Multi-component loss for A2C with feasibility masking.

    L = alpha * L_actor + beta * L_critic + lambda * L_mask
        + omega * E_infeasibility - psi * E_entropy

    Components:
        L_actor:  Policy gradient loss: -log(pi(a|s)) * A(s,a)
        L_critic: Value regression: (V(s) - R_target)^2
        L_mask:   Binary cross-entropy of mask predictions vs ground truth
        E_inf:    Infeasibility penalty: sum pi(a|s) * log(M(a|s))
                  Penalises the policy for assigning probability to infeasible actions
        E_entropy: Policy entropy bonus (negative sign = we SUBTRACT to MAXIMISE entropy)

    Args:
        config: A2CMaskedConfig with loss weight hyperparameters.
    """

    def __init__(self, config: A2CMaskedConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropies: torch.Tensor,
        mask_preds: torch.Tensor,
        mask_trues: torch.Tensor,
        policies: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the total loss and per-component metrics.

        Args:
            log_probs:   (B,) log probabilities of taken actions.
            advantages:  (B,) GAE advantages.
            values:      (B,) predicted state values.
            returns:     (B,) discounted return targets.
            entropies:   (B,) policy entropy per step.
            mask_preds:  (B, num_actions) predicted mask probabilities.
            mask_trues:  (B, num_actions) ground-truth binary masks.
            policies:    (B, num_actions) optional full policy for E_inf.

        Returns:
            (total_loss, metrics_dict) where metrics_dict contains
            individual loss components for logging.
        """
        cfg = self.config

        # 1. Policy gradient loss: -log(pi(a|s)) * A(s,a)
        # Note: we negate because we minimise
        actor_loss = -(log_probs * advantages.detach()).mean()

        # 2. Value regression loss: (V(s) - R_target)^2
        critic_loss = F.mse_loss(values, returns.detach())

        # 3. Mask BCE loss: supervised signal for mask predictor
        mask_loss = F.binary_cross_entropy(
            mask_preds,
            mask_trues.detach(),
            reduction='mean',
        )

        # 4. Entropy bonus: -sum pi(a|s) * log(pi(a|s))
        # Maximising entropy = minimising negative entropy
        entropy_bonus = entropies.mean()

        # 5. Infeasibility penalty: sum pi(a|s) * log(M(a|s))
        # Penalises the policy for placing probability mass on actions
        # that the mask predictor considers infeasible.
        # M close to 0 => log(M) very negative => penalty if pi > 0
        inf_penalty = torch.tensor(0.0, device=log_probs.device)
        if policies is not None:
            # Clamp mask_preds to avoid log(0)
            clamped_mask = torch.clamp(mask_preds.detach(), min=1e-7, max=1.0)
            inf_penalty = (policies * torch.log(clamped_mask)).sum(dim=-1).mean()

        # Total loss
        total_loss = (
            cfg.alpha_actor * actor_loss
            + cfg.beta_critic * critic_loss
            + cfg.lambda_mask * mask_loss
            + cfg.omega_infeasibility * inf_penalty
            - cfg.psi_entropy * entropy_bonus
        )

        # Metrics for logging
        metrics = {
            "loss/total": total_loss.item(),
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
            "loss/mask_bce": mask_loss.item(),
            "loss/entropy": entropy_bonus.item(),
            "loss/infeasibility": inf_penalty.item(),
            "loss/value_mean": values.mean().item(),
            "loss/advantage_mean": advantages.mean().item(),
            "loss/advantage_std": advantages.std().item() if advantages.numel() > 1 else 0.0,
        }

        return total_loss, metrics


# ---------------------------------------------------------------------------
# Helper: resolve device
# ---------------------------------------------------------------------------

def resolve_device(device_str: str = "auto") -> torch.device:
    """
    Resolve device string to a torch.device.

    Args:
        device_str: 'auto', 'cpu', 'cuda', 'cuda:0', etc.

    Returns:
        torch.device instance.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
