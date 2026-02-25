"""
Neural network architecture for the MCTS-Guided Hierarchical Actor-Critic.

Four interconnected components:

1. SharedEncoder
   - HeightmapCNN (shared weights): per-bin heightmap -> 256-dim embedding
   - BoxEncoder: current box + buffer -> 128 + 64 dim
   - ConveyorEncoder: models visible + upcoming boxes -> 64 dim
   - GlobalAttention: cross-attention from box to bins (like PPO strategy)
   -> global_state: 768-dim

2. WorldModel (learned dynamics model)
   - Input: global_state + action_embedding
   - Predicts: delta_heightmap, next_box_features, immediate_reward
   - Used by MCTS planner for lookahead simulations
   - Auxiliary output: void_fraction (trapped empty space prediction)

3. HighLevelPolicy (item + bin selector)
   - Input: global_state
   - Output: distribution over (box_idx, bin_idx) + skip action
   - 10 actions = 4 boxes * 2 bins + 1 skip + 1 reconsider
   - Includes value head for hierarchical advantage estimation

4. LowLevelPolicy (Transformer pointer over candidates)
   - Input: global_state + HL_action_embedding + candidate_features
   - Transformer self-attention over [state; candidates]
   - Pointer decoder selects best candidate
   - Includes value head for fine-grained advantage estimation

Key architectural innovations:
  - Conveyor state encoder: models which boxes become visible after picking
  - Void prediction head: auxiliary loss on trapped empty space detection
  - Hierarchical value decomposition: V(s) = V_high(s) + V_low(s, a_high)
  - Cross-attention from current box to ALL bins simultaneously
  - Action conditioning: low-level policy conditioned on high-level choice

Total parameters: ~2.5M (comparable to existing strategies, GPU-trainable)

References:
  - Fang et al. (2026): MCTS + MPC for online 3D BPP
  - Xiong et al. (2024, GOPT): Transformer + candidates for generalisation
  - Zhao et al. (2022, ICLR): PCT tree-search with pointer mechanism
  - Vezhnevets et al. (2017): FeUdal Networks (hierarchical RL)
  - Schrittwieser et al. (2020): MuZero (learned world model + MCTS)
"""

from __future__ import annotations

import sys
import os
import math
from typing import Dict, List, Optional, Tuple, NamedTuple

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from strategies.rl_mcts_hybrid.config import MCTSHybridConfig


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

class HighLevelOutput(NamedTuple):
    """Output from the high-level policy."""
    action: torch.Tensor          # (B,) selected (box_idx * num_bins + bin_idx)
    log_prob: torch.Tensor        # (B,)
    entropy: torch.Tensor         # (B,)
    value: torch.Tensor           # (B,) V_high(s)
    action_embed: torch.Tensor    # (B, hl_embed_dim) for conditioning LL


class LowLevelOutput(NamedTuple):
    """Output from the low-level policy."""
    action: torch.Tensor          # (B,) candidate index
    log_prob: torch.Tensor        # (B,)
    entropy: torch.Tensor         # (B,)
    value: torch.Tensor           # (B,) V_low(s, a_high)


class WorldModelOutput(NamedTuple):
    """Output from the world model."""
    next_heightmaps: torch.Tensor    # (B, num_bins * 64) compact heightmap features
    next_box_features: torch.Tensor  # (B, box_feat_dim)
    reward_pred: torch.Tensor        # (B,) predicted immediate reward
    void_fraction: torch.Tensor      # (B, num_bins) predicted trapped void ratio
    latent_state: torch.Tensor       # (B, global_state_dim) projected next state


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def _orthogonal_init(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    """Orthogonal initialisation (standard for PPO/A2C)."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# 1. Shared Encoder Components
# ---------------------------------------------------------------------------

class HeightmapCNN(nn.Module):
    """
    Shared CNN encoder for a single bin heightmap.

    Input:  (B, 1, grid_l, grid_w) normalised heightmap [0, 1]
    Output: (B, bin_embed_dim=256)
    """

    def __init__(self, config: MCTSHybridConfig) -> None:
        super().__init__()
        ch = config.cnn_channels  # (32, 64, 128)

        self.conv = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class BoxEncoder(nn.Module):
    """
    Encodes current box and buffer boxes.

    Current box: MLP(5 -> 64 -> 128) -> box_embed
    Buffer:      MLP(5 -> 64) per box, mean-pool -> buffer_embed
    """

    def __init__(self, config: MCTSHybridConfig) -> None:
        super().__init__()
        self.box_mlp = nn.Sequential(
            nn.Linear(config.box_feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, config.box_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.buffer_mlp = nn.Sequential(
            nn.Linear(config.box_feat_dim, config.buffer_embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        box_features: torch.Tensor,       # (B, 5)
        buffer_features: torch.Tensor,     # (B, buffer_size, 5)
        buffer_mask: Optional[torch.Tensor] = None,  # (B, buffer_size)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        box_embed = self.box_mlp(box_features)

        B, N, F = buffer_features.shape
        buf_flat = buffer_features.view(B * N, F)
        buf_enc = self.buffer_mlp(buf_flat).view(B, N, -1)

        if buffer_mask is not None:
            mask_exp = buffer_mask.unsqueeze(-1)
            buf_enc = buf_enc * mask_exp
            count = mask_exp.sum(dim=1).clamp(min=1.0)
            buffer_embed = buf_enc.sum(dim=1) / count
        else:
            buffer_embed = buf_enc.mean(dim=1)

        return box_embed, buffer_embed


class ConveyorEncoder(nn.Module):
    """
    Encodes the conveyor belt state including lookahead information.

    Unlike previous strategies that ignore the conveyor, this encoder models:
    - Which boxes are grippable (pick window)
    - Relative positions on the belt (order matters)
    - Diversity/similarity of upcoming boxes
    - Estimated "value" of waiting vs picking now

    Input: grippable features (B, pick_window, 5) + buffer features (B, buffer_size, 5)
    Output: conveyor_embed (B, conveyor_embed_dim=64)
    """

    def __init__(self, config: MCTSHybridConfig) -> None:
        super().__init__()
        self.pick_window = config.pick_window
        self.buffer_size = config.buffer_size

        # Positional encoding for belt position (grippable boxes get position 0-3)
        self.position_embed = nn.Embedding(config.buffer_size, 16)

        # Per-box encoder with position
        self.box_pos_mlp = nn.Sequential(
            nn.Linear(config.box_feat_dim + 16, 32),
            nn.ReLU(inplace=True),
        )

        # Attention over belt boxes (which boxes are most valuable?)
        self.attn = nn.MultiheadAttention(
            embed_dim=32, num_heads=2, batch_first=True,
        )

        # Final projection
        self.project = nn.Sequential(
            nn.Linear(32, config.conveyor_embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        buffer_features: torch.Tensor,     # (B, buffer_size, 5)
        buffer_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, F = buffer_features.shape
        device = buffer_features.device

        # Add positional encoding (belt position 0..N-1)
        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        pos_embed = self.position_embed(positions)  # (B, N, 16)

        # Concatenate box features with position
        x = torch.cat([buffer_features, pos_embed], dim=-1)  # (B, N, 5+16=21)
        x = self.box_pos_mlp(x)  # (B, N, 32)

        # Self-attention over belt (boxes attend to each other)
        key_padding_mask = None
        if buffer_mask is not None:
            key_padding_mask = ~buffer_mask.bool()

        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)

        # Mean pool over valid positions
        if buffer_mask is not None:
            mask_exp = buffer_mask.unsqueeze(-1)
            attn_out = attn_out * mask_exp
            count = mask_exp.sum(dim=1).clamp(min=1.0)
            pooled = attn_out.sum(dim=1) / count
        else:
            pooled = attn_out.mean(dim=1)

        return self.project(pooled)


class BinCrossAttention(nn.Module):
    """
    Multi-head cross-attention from box features to bin embeddings.

    Allows the policy to attend to the most relevant bin for the current box.
    Query: box + buffer + conveyor features
    Key/Value: per-bin embeddings
    """

    def __init__(self, config: MCTSHybridConfig) -> None:
        super().__init__()
        query_dim = config.box_embed_dim + config.buffer_embed_dim + config.conveyor_embed_dim
        self.num_heads = 4
        self.attn_dim = config.bin_embed_dim  # 256
        self.head_dim = self.attn_dim // self.num_heads
        assert self.attn_dim % self.num_heads == 0

        self.W_q = nn.Linear(query_dim, self.attn_dim)
        self.W_k = nn.Linear(config.bin_embed_dim, self.attn_dim)
        self.W_v = nn.Linear(config.bin_embed_dim, self.attn_dim)
        self.out_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,        # (B, query_dim)
        bin_embeds: torch.Tensor,    # (B, num_bins, bin_embed_dim)
    ) -> torch.Tensor:
        B = query.size(0)
        n_bins = bin_embeds.size(1)

        Q = self.W_q(query).unsqueeze(1)     # (B, 1, attn_dim)
        K = self.W_k(bin_embeds)             # (B, n_bins, attn_dim)
        V = self.W_v(bin_embeds)             # (B, n_bins, attn_dim)

        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, n_bins, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, n_bins, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, 1, self.attn_dim).squeeze(1)

        return self.out_proj(out)  # (B, 256)


class SharedEncoder(nn.Module):
    """
    Full shared encoder producing the global state representation.

    Output: (B, global_state_dim=768)
    Composed of: bin_attention(256) + box_embed(128) + buffer_embed(64)
                 + conveyor_embed(64) + all_bins_flat(256)
    """

    def __init__(self, config: MCTSHybridConfig) -> None:
        super().__init__()
        self.config = config
        self.heightmap_cnn = HeightmapCNN(config)
        self.box_encoder = BoxEncoder(config)
        self.conveyor_encoder = ConveyorEncoder(config)
        self.bin_attention = BinCrossAttention(config)

    def forward(
        self,
        heightmaps: torch.Tensor,       # (B, num_bins, grid_l, grid_w)
        box_features: torch.Tensor,      # (B, 5)
        buffer_features: torch.Tensor,   # (B, buffer_size, 5)
        buffer_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            global_state: (B, global_state_dim)
            bin_embeds:   (B, num_bins, bin_embed_dim) -- for world model
        """
        B = heightmaps.size(0)
        num_bins = heightmaps.size(1)

        # Per-bin CNN encoding (shared weights)
        bin_embeds_list = []
        for b in range(num_bins):
            hm = heightmaps[:, b, :, :].unsqueeze(1)
            embed = self.heightmap_cnn(hm)
            bin_embeds_list.append(embed)
        bin_embeds = torch.stack(bin_embeds_list, dim=1)  # (B, num_bins, 256)

        # Box and buffer encoding
        box_embed, buffer_embed = self.box_encoder(
            box_features, buffer_features, buffer_mask,
        )

        # Conveyor state encoding
        conveyor_embed = self.conveyor_encoder(buffer_features, buffer_mask)

        # Cross-attention: box attends to bins
        query = torch.cat([box_embed, buffer_embed, conveyor_embed], dim=-1)
        bin_context = self.bin_attention(query, bin_embeds)  # (B, 256)

        # Global state: concatenate all components
        global_state = torch.cat([
            bin_context,      # 256: attention-weighted bin info
            box_embed,        # 128: current box
            buffer_embed,     # 64: buffer summary
            conveyor_embed,   # 64: conveyor state
            bin_embeds.view(B, -1)[:, :256],  # 256: raw bin 0 features (truncate for budget)
        ], dim=-1)

        return global_state, bin_embeds


# ---------------------------------------------------------------------------
# 2. World Model
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """
    Learned dynamics model for MCTS lookahead.

    Given: current global state + action (box_idx, bin_idx, candidate_idx)
    Predicts:
      - Next heightmaps (residual prediction: predict delta, add to current)
      - Next box features (what box becomes grippable after pick)
      - Immediate reward (for value estimation during tree search)
      - Void fraction per bin (auxiliary loss target)

    Key insight: after picking box_i from position j in the grippable window,
    box_{j+4} (or the next in stream) becomes visible. The world model learns
    this transition implicitly from data.
    """

    def __init__(self, config: MCTSHybridConfig) -> None:
        super().__init__()
        self.config = config

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(config.high_level_actions + config.max_candidates,
                      config.world_model_action_dim),
            nn.ReLU(inplace=True),
        )

        # State transition model
        state_action_dim = config.global_state_dim + config.world_model_action_dim
        self.transition = nn.Sequential(
            nn.Linear(state_action_dim, config.world_model_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.world_model_hidden, config.world_model_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.world_model_hidden, config.world_model_hidden),
            nn.ReLU(inplace=True),
        )

        # Heightmap prediction head (predicts delta, residual style)
        # We predict a compact representation, not the full 120x80 grid
        hm_compact = config.num_bins * 64  # Compressed heightmap features
        self.hm_head = nn.Sequential(
            nn.Linear(config.world_model_hidden, hm_compact),
            nn.Tanh(),
        )

        # Next box features head
        self.box_head = nn.Sequential(
            nn.Linear(config.world_model_hidden, config.box_feat_dim),
            nn.Sigmoid(),
        )

        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Linear(config.world_model_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Void fraction prediction (auxiliary task)
        self.void_head = nn.Sequential(
            nn.Linear(config.world_model_hidden, config.void_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.void_hidden, config.num_bins),
            nn.Sigmoid(),
        )

        # Latent state projection: projects world model hidden state
        # to full global_state_dim so MCTS can chain predictions without
        # zero-padding (fixes depth>1 value estimate degradation)
        self.latent_projection = nn.Sequential(
            nn.Linear(config.world_model_hidden, config.global_state_dim),
            nn.LayerNorm(config.global_state_dim),
        )

    def forward(
        self,
        global_state: torch.Tensor,    # (B, global_state_dim)
        hl_action: torch.Tensor,       # (B,) high-level action index
        ll_action: torch.Tensor,       # (B,) low-level candidate index
    ) -> WorldModelOutput:
        B = global_state.size(0)
        device = global_state.device

        # One-hot encode actions
        hl_onehot = F.one_hot(
            hl_action.long(), self.config.high_level_actions
        ).float()
        ll_onehot = F.one_hot(
            ll_action.long(), self.config.max_candidates
        ).float()

        action_input = torch.cat([hl_onehot, ll_onehot], dim=-1)
        action_emb = self.action_embed(action_input)

        # Transition
        combined = torch.cat([global_state, action_emb], dim=-1)
        hidden = self.transition(combined)

        # Predictions
        next_hm_features = self.hm_head(hidden)
        next_box = self.box_head(hidden)
        reward_pred = self.reward_head(hidden).squeeze(-1)
        void_fraction = self.void_head(hidden)
        latent_state = self.latent_projection(hidden)

        return WorldModelOutput(
            next_heightmaps=next_hm_features,
            next_box_features=next_box,
            reward_pred=reward_pred,
            void_fraction=void_fraction,
            latent_state=latent_state,
        )


# ---------------------------------------------------------------------------
# 3. High-Level Policy (Item + Bin Selector)
# ---------------------------------------------------------------------------

class HighLevelPolicy(nn.Module):
    """
    Selects WHICH box to pick and WHICH bin to target.

    This is the KEY innovation addressing Gap #1 (no joint item selection).

    Action space: pick_window * num_bins + 1 (skip) + 1 (reconsider)
      = 4 * 2 + 2 = 10 actions

    The skip action advances the conveyor.
    The reconsider action re-evaluates with different low-level options.

    Includes:
      - Action masking: only valid (box, bin) pairs are selectable
      - Value head: V_high(s) for hierarchical advantage estimation
      - Action embedding: fed to low-level policy as conditioning signal
    """

    def __init__(self, config: MCTSHybridConfig) -> None:
        super().__init__()
        self.config = config
        self.num_actions = config.high_level_actions

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(config.global_state_dim, config.high_level_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.high_level_hidden, config.high_level_hidden),
            nn.ReLU(inplace=True),
        )

        # Policy head
        self.policy_head = nn.Linear(config.high_level_hidden, self.num_actions)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.high_level_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Action embedding (for conditioning the low-level policy)
        self.action_embedding = nn.Embedding(self.num_actions, config.high_level_embed_dim)

    def forward(
        self,
        global_state: torch.Tensor,        # (B, global_state_dim)
        action_mask: Optional[torch.Tensor] = None,  # (B, num_actions) 1=valid
        action: Optional[torch.Tensor] = None,        # (B,) for evaluation
        deterministic: bool = False,
    ) -> HighLevelOutput:
        trunk_out = self.trunk(global_state)
        logits = self.policy_head(trunk_out)

        # Apply mask
        if action_mask is not None:
            logits = logits + (1.0 - action_mask) * (-1e9)

        dist = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(trunk_out).squeeze(-1)
        action_embed = self.action_embedding(action)

        return HighLevelOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            action_embed=action_embed,
        )


# ---------------------------------------------------------------------------
# 4. Low-Level Policy (Transformer Pointer over Candidates)
# ---------------------------------------------------------------------------

class CandidateEncoder(nn.Module):
    """Encodes candidate placement features to d_model-dimensional embeddings."""

    def __init__(self, config: MCTSHybridConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.candidate_input_dim, config.candidate_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.candidate_hidden_dim, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LowLevelPolicy(nn.Module):
    """
    Transformer pointer network for placement candidate selection.

    Conditioned on the high-level action (which box, which bin), this module
    attends over the generated placement candidates and selects the best one.

    Architecture:
      1. Encode candidates via MLP -> d_model embeddings
      2. Create context token from global_state + HL_action_embed
      3. Transformer self-attention over [context; candidates]
      4. Pointer decoder: context attends to candidates -> selection distribution
      5. Value head: mean-pooled encoder output -> V_low(s, a_high)
    """

    def __init__(self, config: MCTSHybridConfig) -> None:
        super().__init__()
        self.config = config

        # Candidate encoder
        self.candidate_encoder = CandidateEncoder(config)

        # Context projection (global_state + HL_embed -> d_model)
        context_input = config.global_state_dim + config.high_level_embed_dim
        self.context_proj = nn.Sequential(
            nn.Linear(context_input, config.d_model),
            nn.ReLU(inplace=True),
        )

        # Type embeddings (distinguish context from candidates)
        self.ctx_type_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.cand_type_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
        )

        # Pointer decoder
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.temperature = config.pointer_temperature

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        global_state: torch.Tensor,        # (B, global_state_dim)
        hl_action_embed: torch.Tensor,     # (B, hl_embed_dim)
        candidate_features: torch.Tensor,   # (B, max_N, candidate_input_dim)
        candidate_mask: torch.Tensor,       # (B, max_N) True=valid
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> LowLevelOutput:
        B = global_state.size(0)
        max_N = candidate_features.size(1)

        # Context token
        ctx_input = torch.cat([global_state, hl_action_embed], dim=-1)
        ctx_embed = self.context_proj(ctx_input).unsqueeze(1)  # (B, 1, d_model)
        ctx_embed = ctx_embed + self.ctx_type_embed

        # Candidate embeddings
        cand_embeds = self.candidate_encoder(candidate_features)  # (B, max_N, d_model)
        cand_embeds = cand_embeds + self.cand_type_embed

        # Transformer sequence: [context; candidates]
        seq = torch.cat([ctx_embed, cand_embeds], dim=1)  # (B, 1+max_N, d_model)

        # Padding mask (True = ignore)
        ctx_valid = torch.zeros(B, 1, device=global_state.device, dtype=torch.bool)
        cand_padding = ~candidate_mask
        padding_mask = torch.cat([ctx_valid, cand_padding], dim=1)

        # Transformer forward
        enc_out = self.transformer(seq, src_key_padding_mask=padding_mask)

        # Split back
        ctx_out = enc_out[:, 0, :]      # (B, d_model)
        cand_out = enc_out[:, 1:, :]    # (B, max_N, d_model)

        # Pointer decoder
        Q = self.W_q(ctx_out).unsqueeze(1)    # (B, 1, d_model)
        K = self.W_k(cand_out)                # (B, max_N, d_model)
        logits = torch.bmm(Q, K.transpose(1, 2)).squeeze(1)  # (B, max_N)
        logits = logits / (math.sqrt(self.config.d_model) * self.temperature)
        logits = logits.masked_fill(~candidate_mask, float('-inf'))

        probs = F.softmax(logits, dim=-1)

        # Handle all-masked edge case
        nan_mask = torch.isnan(probs)
        if nan_mask.any():
            probs = probs.masked_fill(nan_mask, 0.0)
            valid_count = candidate_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
            uniform = candidate_mask.float() / valid_count
            probs = torch.where(nan_mask.any(dim=-1, keepdim=True).expand_as(probs),
                                uniform, probs)

        dist = Categorical(probs=probs)

        if action is None:
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Value from mean-pooled encoder output
        if candidate_mask is not None:
            full_mask = torch.cat([
                torch.ones(B, 1, device=global_state.device, dtype=torch.bool),
                candidate_mask,
            ], dim=1)
            mask_float = full_mask.float().unsqueeze(-1)
            pooled = (enc_out * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
        else:
            pooled = enc_out.mean(dim=1)

        value = self.value_head(pooled).squeeze(-1)

        return LowLevelOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
        )


# ---------------------------------------------------------------------------
# 5. Full Model
# ---------------------------------------------------------------------------

class MCTSHybridNet(nn.Module):
    """
    Complete MCTS-Guided Hierarchical Actor-Critic model.

    Combines all four components:
      - SharedEncoder: produces global state representation
      - WorldModel: predicts next state for MCTS lookahead
      - HighLevelPolicy: selects (box, bin)
      - LowLevelPolicy: selects placement candidate via Transformer pointer

    Usage (inference):
        model = MCTSHybridNet(config)
        global_state, bin_embeds = model.encode(obs)
        hl_out = model.high_level(global_state, mask)
        ll_out = model.low_level(global_state, hl_out.action_embed, cands, cand_mask)
    """

    def __init__(self, config: MCTSHybridConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = SharedEncoder(config)
        self.world_model = WorldModel(config)
        self.high_level = HighLevelPolicy(config)
        self.low_level = LowLevelPolicy(config)

        # Initialise weights
        self.apply(lambda m: _orthogonal_init(m))

        # Smaller init for output heads
        for head in [self.high_level.policy_head,
                     self.high_level.value_head,
                     self.low_level.value_head,
                     self.world_model.reward_head]:
            if isinstance(head, nn.Linear):
                _orthogonal_init(head, gain=0.01)
            elif isinstance(head, nn.Sequential):
                last = None
                for m in head.modules():
                    if isinstance(m, nn.Linear):
                        last = m
                if last is not None:
                    _orthogonal_init(last, gain=0.01)

    def encode(
        self,
        heightmaps: torch.Tensor,
        box_features: torch.Tensor,
        buffer_features: torch.Tensor,
        buffer_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observations into global state."""
        return self.encoder(heightmaps, box_features, buffer_features, buffer_mask)

    def count_parameters(self) -> int:
        """Total trainable parameters."""
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
        device: torch.device = torch.device("cpu"),
    ) -> "MCTSHybridNet":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = MCTSHybridConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model

    def summary(self) -> str:
        """Human-readable model summary."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        wm_params = sum(p.numel() for p in self.world_model.parameters())
        hl_params = sum(p.numel() for p in self.high_level.parameters())
        ll_params = sum(p.numel() for p in self.low_level.parameters())
        lines = [
            "MCTSHybridNet Summary",
            "=" * 50,
            f"  SharedEncoder:    {encoder_params:>10,} params",
            f"  WorldModel:       {wm_params:>10,} params",
            f"  HighLevelPolicy:  {hl_params:>10,} params",
            f"  LowLevelPolicy:   {ll_params:>10,} params",
            f"  TOTAL:            {self.count_parameters():>10,} params",
            "",
            f"  Global state dim: {self.config.global_state_dim}",
            f"  HL actions:       {self.config.high_level_actions}",
            f"  LL d_model:       {self.config.d_model}",
            f"  LL max candidates:{self.config.max_candidates}",
            f"  MCTS simulations: {self.config.mcts_simulations}",
        ]
        return "\n".join(lines)
