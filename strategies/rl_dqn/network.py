"""
Neural network architecture for the Double DQN bin packing agent.

Architecture (adapted from Tsang et al. 2025 for the Botko BV setup):

  CNN Branch (processes 2-channel heightmap: bin1 + bin2)
  ────────────────────────────────────────────────────────
    Input: (batch, 2, 120, 80)  — normalised heightmaps
    Conv2d(2,  32, 5, stride=2) -> BN -> ReLU   => (batch, 32,  58, 38)
    Conv2d(32, 64, 3, stride=2) -> BN -> ReLU   => (batch, 64,  28, 18)
    Conv2d(64, 128, 3, stride=1) -> BN -> ReLU  => (batch, 128, 26, 16)
    Conv2d(128, 256, 3, stride=1) -> BN -> ReLU => (batch, 256, 24, 14)
    GlobalAvgPool                                => (batch, 256)

  Box Feature Branch (MLP for buffer box descriptors)
  ─────────────────────────────────────────────────────
    Input: (batch, pick_window * 5) = (batch, 20)
    Dense(20, 128) -> ReLU -> Dense(128, 128) -> ReLU => (batch, 128)

  Action Feature Branch (MLP for candidate placement features)
  ────────────────────────────────────────────────────────────
    Input: (batch, 7)  — (bin_idx, x_norm, y_norm, orient, z_norm, support, height_ratio)
    Dense(7, 64) -> ReLU -> Dense(64, 64) -> ReLU => (batch, 64)

  Merge Head
  ──────────
    Concat(256 + 128 + 64) = 448
    Dense(448, 256) -> ReLU -> Dense(256, 128) -> ReLU -> Dense(128, 1) -> Q-value

  Dueling Variant (optional, default enabled)
  ────────────────────────────────────────────
    Concat(256 + 128) = 384 -> V(s) stream -> scalar value
    Action feature -> A(s,a) stream -> scalar advantage
    Q(s,a) = V(s) + A(s,a) - mean(A)

Weight initialisation: He/Kaiming (appropriate for ReLU networks).

Usage:
    from strategies.rl_dqn.network import DQNNetwork
    net = DQNNetwork(config)
    q_value = net(heightmaps, box_features, action_features)
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from strategies.rl_dqn.config import DQNConfig


# ─────────────────────────────────────────────────────────────────────────────
# Weight initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_weights(module: nn.Module) -> None:
    """Apply He/Kaiming initialisation for ReLU networks."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ─────────────────────────────────────────────────────────────────────────────
# CNN Branch
# ─────────────────────────────────────────────────────────────────────────────

class CNNBranch(nn.Module):
    """
    Convolutional branch for processing multi-channel heightmap input.

    Takes normalised heightmaps (one channel per bin) and produces a
    fixed-length feature vector via a series of Conv2d + BN + ReLU layers
    followed by global average pooling.
    """

    def __init__(self, config: DQNConfig) -> None:
        super().__init__()
        channels = config.cnn_channels
        kernels = config.cnn_kernels
        strides = config.cnn_strides
        use_bn = config.use_batch_norm

        layers = []
        in_ch = config.cnn_input_channels  # 2 (one per bin)
        for out_ch, k, s in zip(channels, kernels, strides):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=0))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = channels[-1]  # 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_bins, grid_l, grid_w) heightmaps.

        Returns:
            (batch, 256) feature vector.
        """
        h = self.conv(x)
        h = self.pool(h)  # (batch, C, 1, 1)
        return h.view(h.size(0), -1)


# ─────────────────────────────────────────────────────────────────────────────
# MLP Branch (generic)
# ─────────────────────────────────────────────────────────────────────────────

class MLPBranch(nn.Module):
    """
    Generic MLP branch: input -> hidden layers with ReLU -> output.

    Used for both the box feature branch and the action feature branch.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main Network (Standard and Dueling)
# ─────────────────────────────────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """
    Double DQN network with three-branch architecture and optional dueling.

    The network evaluates a SINGLE (state, action) pair at a time (or a batch
    of such pairs).  To find the best action among K candidates, call forward
    K times (batched) and take the argmax.

    Branches:
        1. CNN:     heightmaps -> 256-dim state embedding
        2. Box MLP: buffer box features -> 128-dim context embedding
        3. Act MLP: action features -> 64-dim action embedding

    Merge:
        Standard: concat(256 + 128 + 64) = 448 -> 256 -> 128 -> 1 (Q-value)
        Dueling:  V(s) from state branches + A(s,a) from action branch

    Args:
        config: DQNConfig with all architecture hyperparameters.
    """

    def __init__(self, config: DQNConfig) -> None:
        super().__init__()
        self.config = config

        # ── Branches ──────────────────────────────────────────────────────
        self.cnn = CNNBranch(config)
        self.box_mlp = MLPBranch(
            input_dim=config.box_feature_dim,   # pick_window * 5 = 20
            hidden_dim=config.box_hidden,        # 128
            output_dim=config.box_hidden,        # 128
        )
        self.action_mlp = MLPBranch(
            input_dim=config.action_feature_dim,  # 7
            hidden_dim=config.action_hidden,      # 64
            output_dim=config.action_hidden,      # 64
        )

        # ── Merge dimensions ──────────────────────────────────────────────
        state_dim = self.cnn.out_dim + self.box_mlp.out_dim  # 256 + 128 = 384
        action_dim = self.action_mlp.out_dim                  # 64
        combined_dim = state_dim + action_dim                  # 448

        merge_dims = config.merge_hidden  # (256, 128)

        if config.use_dueling:
            # ── Dueling architecture ──────────────────────────────────────
            # Value stream: processes state features only
            self.value_stream = nn.Sequential(
                nn.Linear(state_dim, merge_dims[0]),
                nn.ReLU(inplace=True),
                nn.Linear(merge_dims[0], merge_dims[1]),
                nn.ReLU(inplace=True),
                nn.Linear(merge_dims[1], 1),
            )

            # Advantage stream: processes combined state + action features
            self.advantage_stream = nn.Sequential(
                nn.Linear(combined_dim, merge_dims[0]),
                nn.ReLU(inplace=True),
                nn.Linear(merge_dims[0], merge_dims[1]),
                nn.ReLU(inplace=True),
                nn.Linear(merge_dims[1], 1),
            )
        else:
            # ── Standard Q-value head ─────────────────────────────────────
            self.q_head = nn.Sequential(
                nn.Linear(combined_dim, merge_dims[0]),
                nn.ReLU(inplace=True),
                nn.Linear(merge_dims[0], merge_dims[1]),
                nn.ReLU(inplace=True),
                nn.Linear(merge_dims[1], 1),
            )

        # ── Initialise weights ────────────────────────────────────────────
        self.apply(_init_weights)

    def forward(
        self,
        heightmaps: torch.Tensor,
        box_features: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q-value(s) for (state, action) pair(s).

        Args:
            heightmaps:     (batch, num_bins, grid_l, grid_w) normalised [0, 1].
            box_features:   (batch, pick_window * 5) normalised box descriptors.
            action_features: (batch, 7) candidate placement features.

        Returns:
            (batch, 1) Q-values.
        """
        # Branch forward passes
        cnn_out = self.cnn(heightmaps)                  # (batch, 256)
        box_out = self.box_mlp(box_features)            # (batch, 128)
        act_out = self.action_mlp(action_features)      # (batch, 64)

        # State embedding (shared for dueling)
        state_embed = torch.cat([cnn_out, box_out], dim=1)   # (batch, 384)
        combined = torch.cat([state_embed, act_out], dim=1)  # (batch, 448)

        if self.config.use_dueling:
            value = self.value_stream(state_embed)        # (batch, 1)
            advantage = self.advantage_stream(combined)   # (batch, 1)
            # Q = V + A - mean(A)
            # For single-action evaluation, mean(A) cancels within a batch
            # of candidates for the same state.  We subtract mean across
            # the candidate dimension, but here each row is one candidate,
            # so we return raw V + A.  The subtraction of mean(A) happens
            # at the call site when comparing candidates for the same state.
            q = value + advantage
        else:
            q = self.q_head(combined)

        return q

    def forward_batch_candidates(
        self,
        heightmaps: torch.Tensor,
        box_features: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate Q-values for multiple candidates sharing the same state.

        More efficient than calling forward() in a loop because the CNN
        and box branches are computed once and broadcast.

        Args:
            heightmaps:      (1, num_bins, grid_l, grid_w) — single state.
            box_features:    (1, pick_window * 5) — single state.
            action_features: (K, 7) — K candidate actions.

        Returns:
            (K, 1) Q-values for each candidate.
        """
        # Compute state embedding once
        cnn_out = self.cnn(heightmaps)                    # (1, 256)
        box_out = self.box_mlp(box_features)              # (1, 128)
        state_embed = torch.cat([cnn_out, box_out], dim=1)  # (1, 384)

        k = action_features.size(0)
        state_expand = state_embed.expand(k, -1)          # (K, 384)

        act_out = self.action_mlp(action_features)        # (K, 64)
        combined = torch.cat([state_expand, act_out], dim=1)  # (K, 448)

        if self.config.use_dueling:
            value = self.value_stream(state_expand)       # (K, 1) — same value for all
            advantage = self.advantage_stream(combined)   # (K, 1)
            # Subtract mean advantage across candidates (dueling correction)
            advantage = advantage - advantage.mean(dim=0, keepdim=True)
            q = value + advantage
        else:
            q = self.q_head(combined)

        return q

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Save model state dict to file."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device = torch.device("cpu")) -> "DQNNetwork":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = DQNConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model
