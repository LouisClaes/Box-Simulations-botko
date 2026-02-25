"""
Actor-Critic network with attention for 3D online bin packing PPO.

Architecture overview (see README.md for diagram):

    Heightmaps (2 x 120 x 80) ──> Shared CNN ──> per-bin embeddings (2 x 256)
                                                     │
    Current box (5,)  ──> Box MLP ──> box_embed (128) ├──> Cross-Attention ──> context (128)
    Buffer boxes (8,5) ──> Buf MLP ──> pool(64) ─────┘            │
                                                                    │
                                      ┌─────────────────────────────┘
                                      v
                        concat(context, box_embed, all_bins) = 768
                                 │                    │
                           Actor Head             Critic Head
                    ┌────────────────┐           Dense ──> V(s)
                    │  bin_logits(2)  │
                    │  x_logits(120)  │
                    │  y_logits(80)   │
                    │  orient_logits(2)│
                    └────────────────┘

Decomposed action space (Zhao et al. 2022/2023):
    pi(b,x,y,o|s) = pi_bin(b|s) * pi_x(x|s,b) * pi_y(y|s,b,x) * pi_o(o|s,b,x,y)

This reduces the action space from O(2*120*80*2) = 38,400 to O(2+120+80+2) = 204,
making PPO tractable for this high-dimensional problem.

Each sub-policy is conditioned on previous choices via learned embeddings that
are concatenated to the shared feature vector before producing the next set of
logits.

References:
    - Xiong et al. (RA-L 2024): GOPT transformer + attention for bin packing
    - Zhao et al. (ICLR 2022): PCT tree-search with autoregressive actions
    - Zhao et al. (AAAI 2021): Online 3D-BPP with DRL
"""

from __future__ import annotations

import sys
import os
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

from strategies.rl_ppo.config import PPOConfig


# ─────────────────────────────────────────────────────────────────────────────
# Heightmap CNN encoder (shared across bins)
# ─────────────────────────────────────────────────────────────────────────────

class HeightmapCNN(nn.Module):
    """
    Shared CNN encoder for a single bin heightmap.

    Architecture:
        Conv2d(1, 32, 5, stride=2, padding=2) -> BN -> ReLU
        Conv2d(32, 64, 3, stride=2, padding=1) -> BN -> ReLU
        Conv2d(64, 128, 3, stride=1, padding=1) -> BN -> ReLU
        AdaptiveAvgPool2d(4, 4) -> Flatten -> Dense(2048, 256)

    Input:  (batch, 1, grid_l, grid_w) -- normalised heightmap [0, 1]
    Output: (batch, bin_embed_dim) -- per-bin embedding
    """

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        ch = config.cnn_channels  # (32, 64, 128)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, ch[0], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch[0], ch[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch[1], ch[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(config.cnn_pool_size),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(config.cnn_flat_dim, config.bin_embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, heightmap: torch.Tensor) -> torch.Tensor:
        """
        Encode one heightmap.

        Args:
            heightmap: (batch, 1, grid_l, grid_w) normalised [0, 1].

        Returns:
            (batch, bin_embed_dim) embedding.
        """
        features = self.conv_layers(heightmap)
        return self.fc(features)


# ─────────────────────────────────────────────────────────────────────────────
# Box encoder MLP
# ─────────────────────────────────────────────────────────────────────────────

class BoxEncoder(nn.Module):
    """
    Encodes the current box and buffer boxes into embeddings.

    Current box:   MLP(5 -> 64 -> 128) -> box_embed
    Buffer boxes:  MLP(5 -> 64) per box, then mean-pool -> buffer_embed
    """

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        feat_dim = config.box_feat_dim  # 5

        # Current box encoder: 5 -> 64 -> 128
        self.box_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, config.box_embed_dim),
            nn.ReLU(inplace=True),
        )

        # Buffer box encoder: 5 -> 64 (shared weights per box)
        self.buffer_mlp = nn.Sequential(
            nn.Linear(feat_dim, config.buffer_embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        box_features: torch.Tensor,
        buffer_features: torch.Tensor,
        buffer_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode current box and buffer.

        Args:
            box_features:    (batch, 5) -- current box features.
            buffer_features: (batch, buffer_size, 5) -- all visible boxes.
            buffer_mask:     (batch, buffer_size) -- 1 for valid boxes, 0 for padding.

        Returns:
            box_embed:    (batch, box_embed_dim=128)
            buffer_embed: (batch, buffer_embed_dim=64)
        """
        box_embed = self.box_mlp(box_features)

        # Per-box buffer encoding
        batch, n_buf, feat = buffer_features.shape
        buf_flat = buffer_features.view(batch * n_buf, feat)
        buf_encoded = self.buffer_mlp(buf_flat)
        buf_encoded = buf_encoded.view(batch, n_buf, -1)  # (batch, n_buf, 64)

        # Masked mean pool
        if buffer_mask is not None:
            mask_expanded = buffer_mask.unsqueeze(-1)  # (batch, n_buf, 1)
            buf_encoded = buf_encoded * mask_expanded
            count = mask_expanded.sum(dim=1).clamp(min=1.0)
            buffer_embed = buf_encoded.sum(dim=1) / count
        else:
            buffer_embed = buf_encoded.mean(dim=1)

        return box_embed, buffer_embed


# ─────────────────────────────────────────────────────────────────────────────
# Cross-attention module
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention between item features and bin features.

    Query:     box embedding (128)
    Key/Value: bin embeddings (num_bins * 256 -> split per bin)
    Output:    context vector (128)

    This allows the policy to attend to the most relevant bin given the
    current box properties -- learning to focus on bins where the box
    fits best.
    """

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self.num_heads = config.attn_num_heads
        self.attn_dim = config.attn_dim
        self.head_dim = config.attn_dim // config.attn_num_heads
        assert self.attn_dim % self.num_heads == 0, \
            f"attn_dim ({config.attn_dim}) must be divisible by attn_num_heads ({config.attn_num_heads})"

        # Query from box features (box_embed + buffer_embed)
        query_input_dim = config.box_embed_dim + config.buffer_embed_dim
        self.W_q = nn.Linear(query_input_dim, self.attn_dim)

        # Key and Value from bin embeddings
        self.W_k = nn.Linear(config.bin_embed_dim, self.attn_dim)
        self.W_v = nn.Linear(config.bin_embed_dim, self.attn_dim)

        self.out_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query_features: torch.Tensor,
        bin_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-attend from box to bins.

        Args:
            query_features: (batch, box_embed_dim + buffer_embed_dim)
            bin_embeddings:  (batch, num_bins, bin_embed_dim)

        Returns:
            context: (batch, attn_dim=128)
        """
        batch = query_features.size(0)
        n_bins = bin_embeddings.size(1)

        # Project queries, keys, values
        Q = self.W_q(query_features)          # (batch, attn_dim)
        Q = Q.unsqueeze(1)                    # (batch, 1, attn_dim)
        K = self.W_k(bin_embeddings)          # (batch, n_bins, attn_dim)
        V = self.W_v(bin_embeddings)          # (batch, n_bins, attn_dim)

        # Reshape for multi-head attention
        Q = Q.view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, n_bins, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, n_bins, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        # (batch, num_heads, 1, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, 1, self.attn_dim)
        attn_output = attn_output.squeeze(1)  # (batch, attn_dim)

        context = self.out_proj(attn_output)
        return context


# ─────────────────────────────────────────────────────────────────────────────
# Decomposed action heads
# ─────────────────────────────────────────────────────────────────────────────

class DecomposedActionHead(nn.Module):
    """
    Autoregressive decomposed action head.

    Produces logits for each sub-action sequentially, conditioning each
    on the previous choices via learned embeddings:

        pi(b,x,y,o|s) = pi_bin(b|s) * pi_x(x|s,b) * pi_y(y|s,b,x) * pi_o(o|s,b,x,y)

    Each sub-policy takes the shared features plus embeddings of previous
    choices and produces logits for the next choice.

    Action masking is applied to each set of logits before sampling.
    """

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.actor_input_dim  # context + box_embed + all_bins = 768

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, config.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.head_hidden_dim, config.head_hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        trunk_out = config.head_hidden_dim // 2  # 128

        # Sub-action heads
        # Each conditions on the trunk output plus embeddings of prior choices.

        # 1. Bin selection: trunk_out -> num_bins
        self.bin_head = nn.Linear(trunk_out, config.num_bins)
        self.bin_embed = nn.Embedding(config.num_bins, 16)

        # 2. X position: trunk_out + bin_embed(16) -> grid_l
        self.x_head = nn.Linear(trunk_out + 16, config.grid_l)
        self.x_embed = nn.Embedding(config.grid_l, 16)

        # 3. Y position: trunk_out + bin_embed(16) + x_embed(16) -> grid_w
        self.y_head = nn.Linear(trunk_out + 32, config.grid_w)
        self.y_embed = nn.Embedding(config.grid_w, 16)

        # 4. Orientation: trunk_out + bin_embed + x_embed + y_embed -> num_orientations
        self.orient_head = nn.Linear(trunk_out + 48, config.num_orientations)

    def forward(
        self,
        features: torch.Tensor,
        action_masks: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False,
        actions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Produce decomposed actions and their log-probabilities.

        When ``actions`` is provided (training), compute log-probs for
        those specific actions (no sampling).  Otherwise, sample new actions.

        Args:
            features:     (batch, actor_input_dim)
            action_masks: Dict with optional masks for each sub-action:
                          'bin':    (batch, num_bins)
                          'x':     (batch, grid_l)
                          'y':     (batch, grid_w)
                          'orient': (batch, num_orientations)
                          Values: 1 = valid, 0 = invalid.
            deterministic: Use argmax instead of sampling.
            actions:      Dict with pre-selected actions (for log-prob computation):
                          'bin', 'x', 'y', 'orient' -- each (batch,) LongTensor.

        Returns:
            actions_dict:  {'bin': ..., 'x': ..., 'y': ..., 'orient': ...}
            log_probs:     (batch,) total log-probability of the action tuple
            entropy:       (batch,) total entropy across all sub-actions
        """
        trunk_out = self.trunk(features)
        batch = trunk_out.size(0)
        device = trunk_out.device

        total_log_prob = torch.zeros(batch, device=device)
        total_entropy = torch.zeros(batch, device=device)
        actions_dict: Dict[str, torch.Tensor] = {}

        # Helper for masked sampling
        def _sample_sub(logits: torch.Tensor, mask: Optional[torch.Tensor],
                        key: str) -> torch.Tensor:
            """Apply mask, sample or take given action, accumulate log-prob and entropy."""
            nonlocal total_log_prob, total_entropy

            if mask is not None:
                # Set invalid action logits to -inf
                logits = logits + (mask.float().log().clamp(min=-1e8))

            dist = Categorical(logits=logits)

            if actions is not None:
                act = actions[key]
            elif deterministic:
                act = logits.argmax(dim=-1)
            else:
                act = dist.sample()

            total_log_prob = total_log_prob + dist.log_prob(act)
            total_entropy = total_entropy + dist.entropy()
            actions_dict[key] = act
            return act

        # 1. Bin selection
        bin_logits = self.bin_head(trunk_out)
        bin_mask = action_masks.get('bin') if action_masks else None
        bin_act = _sample_sub(bin_logits, bin_mask, 'bin')

        # 2. X position (conditioned on bin choice)
        bin_emb = self.bin_embed(bin_act)  # (batch, 16)
        x_input = torch.cat([trunk_out, bin_emb], dim=-1)
        x_logits = self.x_head(x_input)
        x_mask = action_masks.get('x') if action_masks else None
        x_act = _sample_sub(x_logits, x_mask, 'x')

        # 3. Y position (conditioned on bin, x)
        x_emb = self.x_embed(x_act)  # (batch, 16)
        y_input = torch.cat([trunk_out, bin_emb, x_emb], dim=-1)
        y_logits = self.y_head(y_input)
        y_mask = action_masks.get('y') if action_masks else None
        y_act = _sample_sub(y_logits, y_mask, 'y')

        # 4. Orientation (conditioned on bin, x, y)
        y_emb = self.y_embed(y_act)  # (batch, 16)
        orient_input = torch.cat([trunk_out, bin_emb, x_emb, y_emb], dim=-1)
        orient_logits = self.orient_head(orient_input)
        orient_mask = action_masks.get('orient') if action_masks else None
        _sample_sub(orient_logits, orient_mask, 'orient')

        return actions_dict, total_log_prob, total_entropy


# ─────────────────────────────────────────────────────────────────────────────
# Value head
# ─────────────────────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """
    State-value function V(s).

    Input: shared features (actor_input_dim = 768).
    Output: scalar value estimate.
    """

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.actor_input_dim, config.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.head_hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.

        Args:
            features: (batch, actor_input_dim)

        Returns:
            (batch,) value estimates.
        """
        return self.net(features).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Full Actor-Critic model
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Complete actor-critic model for PPO bin packing.

    Combines:
        - HeightmapCNN: shared CNN for per-bin heightmap encoding
        - BoxEncoder: MLP encoders for current box and buffer
        - CrossAttention: cross-attention from items to bins
        - DecomposedActionHead: autoregressive sub-action policy
        - ValueHead: state-value estimator

    Usage (inference):
        model = ActorCritic(config)
        actions, log_probs, entropy, values = model(obs_dict)

    Usage (training -- evaluate old actions):
        actions, log_probs, entropy, values = model(
            obs_dict, actions=old_actions, action_masks=masks,
        )
    """

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self.config = config

        # Feature extractors
        self.heightmap_cnn = HeightmapCNN(config)
        self.box_encoder = BoxEncoder(config)
        self.cross_attention = CrossAttention(config)

        # Heads
        self.actor = DecomposedActionHead(config)
        self.critic = ValueHead(config)

        # Weight initialisation (orthogonal, as recommended for PPO)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Orthogonal initialisation for linear layers (PPO best practice)."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _encode_observations(
        self,
        heightmaps: torch.Tensor,
        box_features: torch.Tensor,
        buffer_features: torch.Tensor,
        buffer_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode all observations into a single feature vector.

        Args:
            heightmaps:      (batch, num_bins, grid_l, grid_w) normalised [0, 1]
            box_features:    (batch, 5) current box features
            buffer_features: (batch, buffer_size, 5) visible box features
            buffer_mask:     (batch, buffer_size) optional validity mask

        Returns:
            features: (batch, actor_input_dim) combined feature vector
        """
        batch = heightmaps.size(0)
        num_bins = heightmaps.size(1)

        # Encode each bin's heightmap through shared CNN
        bin_embeds_list = []
        for b in range(num_bins):
            hm = heightmaps[:, b, :, :].unsqueeze(1)  # (batch, 1, grid_l, grid_w)
            embed = self.heightmap_cnn(hm)              # (batch, bin_embed_dim)
            bin_embeds_list.append(embed)

        bin_embeds = torch.stack(bin_embeds_list, dim=1)  # (batch, num_bins, bin_embed_dim)
        all_bins_flat = bin_embeds.view(batch, -1)         # (batch, num_bins * bin_embed_dim)

        # Encode box and buffer
        box_embed, buffer_embed = self.box_encoder(
            box_features, buffer_features, buffer_mask,
        )

        # Cross-attention: query = box features, key/value = bin embeddings
        query = torch.cat([box_embed, buffer_embed], dim=-1)
        context = self.cross_attention(query, bin_embeds)

        # Concatenate all features
        features = torch.cat([context, box_embed, all_bins_flat], dim=-1)
        return features

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        action_masks: Optional[Dict[str, torch.Tensor]] = None,
        actions: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode observations, produce actions and values.

        Args:
            obs: Dictionary with:
                'heightmaps':      (batch, num_bins, grid_l, grid_w)
                'box_features':    (batch, 5)
                'buffer_features': (batch, buffer_size, 5)
                'buffer_mask':     (batch, buffer_size) [optional]
            action_masks: Per-sub-action validity masks.
            actions: Pre-selected actions for log-prob computation (training).
            deterministic: Use greedy action selection.

        Returns:
            actions_dict: {'bin', 'x', 'y', 'orient'} -- each (batch,)
            log_probs:    (batch,) -- total log-prob
            entropy:      (batch,) -- total entropy
            values:       (batch,) -- V(s)
        """
        features = self._encode_observations(
            heightmaps=obs['heightmaps'],
            box_features=obs['box_features'],
            buffer_features=obs['buffer_features'],
            buffer_mask=obs.get('buffer_mask'),
        )

        actions_dict, log_probs, entropy = self.actor(
            features,
            action_masks=action_masks,
            deterministic=deterministic,
            actions=actions,
        )

        values = self.critic(features)

        return actions_dict, log_probs, entropy, values

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute state value only (no action sampling).

        Used for GAE computation at the end of rollouts.

        Args:
            obs: Same observation dict as forward().

        Returns:
            (batch,) value estimates.
        """
        features = self._encode_observations(
            heightmaps=obs['heightmaps'],
            box_features=obs['box_features'],
            buffer_features=obs['buffer_features'],
            buffer_mask=obs.get('buffer_mask'),
        )
        return self.critic(features)

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
