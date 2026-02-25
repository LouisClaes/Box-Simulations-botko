"""
PCT Transformer Actor-Critic Network.

A Transformer encoder-decoder for 3D bin packing with variable-size action
spaces (placement candidates).  Inspired by Zhao et al. (ICLR 2022) Packing
Configuration Trees, adapted to use a standard Transformer architecture.

Architecture Overview:
    1. ItemEncoder:     MLP(5 -> 64 -> 128)  encodes the current box
    2. CandidateEncoder: MLP(12 -> 64 -> 128) encodes each candidate placement
    3. TransformerEncoder: 3-layer self-attention over [item; candidates]
    4. PointerDecoder:  attention-based action selection (query=item, keys=candidates)
    5. ValueHead:       mean-pooled encoder output -> Dense(128, 64) -> Dense(64, 1)

Key design: the action space is VARIABLE SIZE.  Each step has a different
number of candidates depending on the bin state.  The Transformer handles
this naturally via attention masking.  The pointer decoder selects among
candidates using a single attention layer, producing a distribution over
the candidate set.

References:
    - Zhao et al. (ICLR 2022): PCT architecture (GAT-based, pointer mechanism)
    - Vaswani et al. (2017): Transformer architecture
    - Vinyals et al. (2015): Pointer Networks
    - Kool et al. (2019): Attention Model for combinatorial optimisation
"""

from __future__ import annotations

import sys
import os
import math
from typing import Tuple, Optional

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from strategies.rl_pct_transformer.config import PCTTransformerConfig


# ─────────────────────────────────────────────────────────────────────────────
# ItemEncoder: MLP for the current box features
# ─────────────────────────────────────────────────────────────────────────────

class ItemEncoder(nn.Module):
    """
    Encodes the current box as a d_model-dimensional embedding.

    Input:  (batch, item_input_dim)  -- typically 5: (l, w, h, vol, weight) normalised
    Output: (batch, d_model)         -- 128-dim embedding

    Architecture: Linear(5, 64) -> ReLU -> Linear(64, 128)
    """

    def __init__(self, config: PCTTransformerConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.item_input_dim, config.item_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.item_hidden_dim, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, item_input_dim)
        Returns:
            (batch, d_model)
        """
        return self.mlp(x)


# ─────────────────────────────────────────────────────────────────────────────
# CandidateEncoder: MLP for per-candidate placement features
# ─────────────────────────────────────────────────────────────────────────────

class CandidateEncoder(nn.Module):
    """
    Encodes each placement candidate as a d_model-dimensional embedding.

    Input:  (batch, num_candidates, candidate_input_dim) -- typically 12 features
    Output: (batch, num_candidates, d_model)             -- 128-dim per candidate

    Architecture: Linear(12, 64) -> ReLU -> Linear(64, 128)
    """

    def __init__(self, config: PCTTransformerConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.candidate_input_dim, config.candidate_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.candidate_hidden_dim, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_candidates, candidate_input_dim)
        Returns:
            (batch, num_candidates, d_model)
        """
        return self.mlp(x)


# ─────────────────────────────────────────────────────────────────────────────
# PointerDecoder: attention-based action selection
# ─────────────────────────────────────────────────────────────────────────────

class PointerDecoder(nn.Module):
    """
    Pointer mechanism for selecting a candidate from a variable-size set.

    Uses the item embedding (from encoder output) as the query and
    candidate embeddings as keys/values.  Produces logits over candidates
    via scaled dot-product attention, then applies masking and softmax
    to get action probabilities.

    This is analogous to PCT's leaf node selection mechanism, but uses
    standard Transformer attention instead of GAT.

    Args:
        config: PCTTransformerConfig with d_model and scaling options.
    """

    def __init__(self, config: PCTTransformerConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.use_scaled = config.use_scaled_attention
        self.temperature = config.pointer_temperature

        # Learned projection for query (item) and keys (candidates)
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        item_embed: torch.Tensor,
        candidate_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action logits and probabilities over candidates.

        Args:
            item_embed:      (batch, d_model)  -- query (from encoder output[0])
            candidate_embeds: (batch, N, d_model) -- keys (from encoder output[1:])
            mask:            (batch, N) -- True = valid, False = masked out

        Returns:
            logits: (batch, N) -- raw attention scores (masked)
            probs:  (batch, N) -- softmax probabilities
        """
        # Project query and keys
        Q = self.W_q(item_embed).unsqueeze(1)     # (batch, 1, d_model)
        K = self.W_k(candidate_embeds)             # (batch, N, d_model)

        # Scaled dot-product attention
        logits = torch.bmm(Q, K.transpose(1, 2)).squeeze(1)  # (batch, N)

        if self.use_scaled:
            logits = logits / math.sqrt(self.d_model)

        logits = logits / self.temperature

        # Apply mask: set invalid candidates to -inf
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))

        # Softmax
        probs = F.softmax(logits, dim=-1)

        # Handle case where all candidates are masked (shouldn't happen normally)
        # Replace NaN with uniform distribution
        nan_mask = torch.isnan(probs)
        if nan_mask.any():
            probs = probs.masked_fill(nan_mask, 0.0)
            # Assign uniform over valid candidates
            if mask is not None:
                valid_count = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
                probs = probs + mask.float() / valid_count * nan_mask.any(dim=-1, keepdim=True).float()

        return logits, probs


# ─────────────────────────────────────────────────────────────────────────────
# ValueHead: critic for PPO
# ─────────────────────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """
    Value function head for PPO.

    Takes the mean-pooled Transformer encoder output and produces a
    scalar state value estimate.

    Architecture: mean_pool -> Dense(128, 64) -> ReLU -> Dense(64, 1)
    """

    def __init__(self, config: PCTTransformerConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.value_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.value_hidden_dim, 1),
        )

    def forward(self, encoder_output: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, 1 + N, d_model) -- full encoder output
            mask:           (batch, N) -- True = valid candidate (optional)

        Returns:
            value: (batch, 1) -- scalar value estimate
        """
        # Mean pool over all tokens (item + valid candidates)
        if mask is not None:
            # Include item token (always valid) + masked candidates
            full_mask = torch.cat([
                torch.ones(mask.shape[0], 1, device=mask.device, dtype=torch.bool),
                mask,
            ], dim=1)  # (batch, 1 + N)

            # Masked mean pool
            full_mask_float = full_mask.float().unsqueeze(-1)  # (batch, 1+N, 1)
            pooled = (encoder_output * full_mask_float).sum(dim=1) / full_mask_float.sum(dim=1).clamp(min=1.0)
        else:
            pooled = encoder_output.mean(dim=1)  # (batch, d_model)

        return self.mlp(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# PCTTransformerNet: full actor-critic
# ─────────────────────────────────────────────────────────────────────────────

class PCTTransformerNet(nn.Module):
    """
    PCT-inspired Transformer Actor-Critic for 3D bin packing.

    Combines:
      - ItemEncoder:        encodes the current box
      - CandidateEncoder:   encodes each placement candidate
      - TransformerEncoder: contextualises all tokens via self-attention
      - PointerDecoder:     selects a candidate (actor)
      - ValueHead:          estimates state value (critic)

    The network handles VARIABLE-SIZE action spaces naturally:
      - Each step may have a different number of candidates (30-200)
      - Candidates are padded to the batch maximum and masked
      - The Transformer processes all candidates simultaneously
      - The pointer decoder attends only over valid candidates

    Input:
        item_features:      (batch, 5)         -- current box
        candidate_features: (batch, max_N, 12) -- padded candidate features
        candidate_mask:     (batch, max_N)     -- True = valid candidate

    Output:
        action_logits:  (batch, max_N)  -- raw logits over candidates
        action_probs:   (batch, max_N)  -- softmax probabilities
        value:          (batch, 1)      -- state value estimate
    """

    def __init__(self, config: PCTTransformerConfig) -> None:
        super().__init__()
        self.config = config

        # Encoders
        self.item_encoder = ItemEncoder(config)
        self.candidate_encoder = CandidateEncoder(config)

        # Learned type embeddings to distinguish item from candidates
        self.item_type_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.cand_type_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Transformer encoder (self-attention over [item; candidates])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='relu',
            batch_first=True,  # (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
            enable_nested_tensor=False,  # Suppress prototype NestedTensor warning on HPC
        )

        # Actor (pointer decoder)
        self.pointer_decoder = PointerDecoder(config)

        # Critic (value head)
        self.value_head = ValueHead(config)

        # Initialise weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        item_features: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode, attend, select, value.

        Args:
            item_features:      (batch, item_input_dim)           -- box features
            candidate_features: (batch, max_N, candidate_input_dim) -- candidate features
            candidate_mask:     (batch, max_N)                    -- True = valid

        Returns:
            logits: (batch, max_N)  -- raw pointer logits
            probs:  (batch, max_N)  -- action probabilities
            value:  (batch, 1)      -- state value
        """
        batch_size = item_features.shape[0]
        max_N = candidate_features.shape[1]

        # Step 1: Encode item and candidates
        item_embed = self.item_encoder(item_features)             # (batch, d_model)
        cand_embeds = self.candidate_encoder(candidate_features)  # (batch, max_N, d_model)

        # Step 2: Add type embeddings
        item_embed = item_embed.unsqueeze(1) + self.item_type_embed  # (batch, 1, d_model)
        cand_embeds = cand_embeds + self.cand_type_embed             # (batch, max_N, d_model)

        # Step 3: Concatenate [item; candidates] as sequence for Transformer
        # Sequence: position 0 = item, positions 1..N = candidates
        seq = torch.cat([item_embed, cand_embeds], dim=1)  # (batch, 1 + max_N, d_model)

        # Step 4: Build attention mask for Transformer
        # Item token is always valid; candidates follow candidate_mask
        # PyTorch TransformerEncoder uses src_key_padding_mask:
        #   True = IGNORE this position (opposite of our candidate_mask!)
        item_valid = torch.zeros(batch_size, 1, device=item_features.device, dtype=torch.bool)
        cand_padding = ~candidate_mask  # True where we should ignore
        padding_mask = torch.cat([item_valid, cand_padding], dim=1)  # (batch, 1 + max_N)

        # Step 5: Transformer encoder
        encoder_output = self.transformer_encoder(
            seq,
            src_key_padding_mask=padding_mask,
        )  # (batch, 1 + max_N, d_model)

        # Step 6: Split encoder output back
        item_ctx = encoder_output[:, 0, :]      # (batch, d_model) -- contextualised item
        cand_ctx = encoder_output[:, 1:, :]     # (batch, max_N, d_model) -- contextualised candidates

        # Step 7: Pointer decoder (actor)
        logits, probs = self.pointer_decoder(item_ctx, cand_ctx, candidate_mask)

        # Step 8: Value head (critic)
        value = self.value_head(encoder_output, candidate_mask)

        return logits, probs, value

    def get_action_and_value(
        self,
        item_features: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or evaluate) an action and return log_prob, entropy, value.

        Used by PPO for both rollout collection and policy updates.

        Args:
            item_features:      (batch, item_input_dim)
            candidate_features: (batch, max_N, candidate_input_dim)
            candidate_mask:     (batch, max_N)
            action:             (batch,) -- if provided, evaluate this action's log_prob
                                            (for PPO ratio computation)
            deterministic:      If True, select argmax instead of sampling.

        Returns:
            action:    (batch,)   -- selected candidate index
            log_prob:  (batch,)   -- log probability of the selected action
            entropy:   (batch,)   -- entropy of the action distribution
            value:     (batch, 1) -- state value estimate
        """
        logits, probs, value = self.forward(
            item_features, candidate_features, candidate_mask,
        )

        # Build distribution
        dist = Categorical(probs=probs)

        if action is None:
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(
        self,
        item_features: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute only the state value (for bootstrapping).

        More efficient than full forward when we don't need the policy.

        Args:
            item_features:      (batch, item_input_dim)
            candidate_features: (batch, max_N, candidate_input_dim)
            candidate_mask:     (batch, max_N)

        Returns:
            value: (batch, 1)
        """
        _, _, value = self.forward(item_features, candidate_features, candidate_mask)
        return value

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Human-readable model summary."""
        lines = [
            f"PCTTransformerNet",
            f"  d_model:          {self.config.d_model}",
            f"  nhead:            {self.config.nhead}",
            f"  encoder_layers:   {self.config.num_encoder_layers}",
            f"  dim_feedforward:  {self.config.dim_feedforward}",
            f"  item_input:       {self.config.item_input_dim}",
            f"  candidate_input:  {self.config.candidate_input_dim}",
            f"  total_params:     {self.count_parameters():,}",
        ]
        return "\n".join(lines)
