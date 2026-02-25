"""
RLA2CMaskedStrategy — Inference-time strategy for A2C with Feasibility Masking.

This strategy wraps the trained A2C network for use in the standard
packing framework (BaseStrategy interface).  It integrates with
PackingSession, PipelineSimulator, and all benchmark/experiment scripts.

Inference pipeline:
    1. Encode bin heightmap + current box as 4-channel CNN input
    2. Forward pass through shared CNN encoder + item MLP
    3. Mask predictor predicts feasibility of each coarse-grid action
    4. Actor head produces masked policy (infeasible actions zeroed)
    5. Select action (argmax for deterministic, sample for stochastic)
    6. Decode coarse-grid action to (x, y, orientation_idx)
    7. Return PlacementDecision or fallback to heuristic

The mask predictor is the key innovation: it replaces expensive exact
validity computation with a learned O(1) neural network prediction.

Usage:
    strategy = RLA2CMaskedStrategy()
    strategy.on_episode_start(config)
    decision = strategy.decide_placement(box, bin_state)

Registration:
    Automatically registered as "rl_a2c_masked" in STRATEGY_REGISTRY.
"""

from __future__ import annotations

import sys
import os
from typing import Optional

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

import numpy as np

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy, get_strategy
from strategies.rl_a2c_masked.config import A2CMaskedConfig

# Lazy torch import — allows the strategy module to be imported even
# when torch is not installed (e.g. for strategy registry listing).
_torch_available = False
try:
    import torch
    _torch_available = True
except ImportError:
    pass


@register_strategy
class RLA2CMaskedStrategy(BaseStrategy):
    """
    A2C with Feasibility Masking inference strategy.

    Loads a trained checkpoint and uses the actor + mask predictor heads
    for fast placement decisions.  Falls back to a heuristic strategy
    (default: baseline BLF) when no model is loaded or all actions are
    masked as infeasible.

    The coarse action grid (default 50mm step) means the strategy operates
    at lower spatial resolution than heuristics.  For fine-grained control,
    reduce action_grid_step at the cost of a larger action space.

    Attributes:
        name:    "rl_a2c_masked" (registered in STRATEGY_REGISTRY).
        config:  A2CMaskedConfig with network + inference parameters.
    """

    name = "rl_a2c_masked"

    def __init__(
        self,
        config: Optional[A2CMaskedConfig] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        Initialise the strategy.

        Args:
            config:          A2CMaskedConfig (uses defaults if None).
            checkpoint_path: Path to .pt checkpoint.  Overrides
                             config.checkpoint_path if provided.
        """
        super().__init__()
        self._a2c_config = config or A2CMaskedConfig()
        self._network = None
        self._device = None
        self._fallback = None
        self._model_loaded = False

        if checkpoint_path is not None:
            self._a2c_config.checkpoint_path = checkpoint_path

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """
        Load the model (if not already loaded) and prepare for inference.

        Called once before the first box in each episode.
        """
        super().on_episode_start(config)

        # Load fallback strategy
        try:
            self._fallback = get_strategy(self._a2c_config.fallback_strategy)
            self._fallback.on_episode_start(config)
        except Exception:
            self._fallback = None

        # Load model if not yet loaded
        if not self._model_loaded and _torch_available:
            self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load the trained network from checkpoint."""
        ckpt_path = self._a2c_config.checkpoint_path
        if ckpt_path is None or not os.path.isfile(ckpt_path):
            print(f"[rl_a2c_masked] No checkpoint found at "
                  f"'{ckpt_path}' — using fallback strategy.")
            return

        try:
            from strategies.rl_a2c_masked.network import (
                A2CMaskedNetwork, resolve_device,
            )
            from strategies.rl_a2c_masked.train import load_checkpoint

            self._device = resolve_device(self._a2c_config.device)
            network, loaded_config, ckpt = load_checkpoint(
                ckpt_path, config=self._a2c_config, device=self._device,
            )
            self._a2c_config = loaded_config
            self._network = network.to(self._device)
            self._network.eval()
            self._model_loaded = True

            update = ckpt.get("update", "?")
            eval_metrics = ckpt.get("eval_metrics", {})
            fill = eval_metrics.get("eval/avg_fill", "?")
            print(f"[rl_a2c_masked] Loaded checkpoint from update {update} "
                  f"(eval fill: {fill})")

        except Exception as e:
            print(f"[rl_a2c_masked] Failed to load model: {e}")
            self._network = None
            self._model_loaded = False

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Propose a placement for the given box.

        Pipeline:
            1. Encode observation (heightmap + box features)
            2. Network forward pass with mask predictor
            3. Select best valid action from masked policy
            4. Decode to (x, y, orientation_idx)
            5. Validate against bin bounds
            6. Fallback to heuristic if no valid placement found

        Args:
            box:       The box to place.
            bin_state: Current state of the target bin.

        Returns:
            PlacementDecision or None if placement is impossible.
        """
        # If no model, use fallback
        if not self._model_loaded or self._network is None or not _torch_available:
            return self._fallback_placement(box, bin_state)

        try:
            decision = self._neural_placement(box, bin_state)
            if decision is not None:
                return decision
        except Exception:
            pass

        # Fallback to heuristic
        return self._fallback_placement(box, bin_state)

    @torch.no_grad()
    def _neural_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Use the trained network to select a placement.

        Constructs the 4-channel heightmap input, runs the forward pass
        through CNN encoder, actor, and mask predictor, then decodes
        the selected action.
        """
        cfg = self._a2c_config
        bin_cfg = self.config.bin
        device = self._device

        # ── Encode observation ──────────────────────────────────────────
        # Build 4-channel heightmap: (1, num_bins, grid_l, grid_w)
        # Since BaseStrategy only sees ONE bin, we pad the second bin
        # with the same heightmap (or zeros if unavailable).
        grid_l, grid_w = bin_cfg.grid_l, bin_cfg.grid_w
        hm_norm = bin_state.heightmap.astype(np.float32) / max(bin_cfg.height, 1.0)

        heightmaps = np.zeros((cfg.num_bins, grid_l, grid_w), dtype=np.float32)
        heightmaps[0] = hm_norm
        # Pad remaining bins with zeros (single-bin context)
        for i in range(1, cfg.num_bins):
            heightmaps[i] = np.zeros((grid_l, grid_w), dtype=np.float32)

        heightmaps_t = torch.from_numpy(heightmaps).float().unsqueeze(0).to(device)

        # Item features: (1, 5) normalised
        max_dim = max(bin_cfg.length, bin_cfg.width, bin_cfg.height)
        item_features = np.array([
            box.length / max_dim,
            box.width / max_dim,
            box.height / max_dim,
            box.volume / max(bin_cfg.volume, 1.0),
            box.weight / 50.0,
        ], dtype=np.float32)
        item_features_t = torch.from_numpy(item_features).float().unsqueeze(0).to(device)

        # ── Forward pass ────────────────────────────────────────────────
        output = self._network.forward(
            heightmaps_t,
            item_features_t,
            true_mask=None,  # Use predicted mask at inference
        )

        policy = output.policy[0]  # (num_actions,)

        # ── Decode best actions ─────────────────────────────────────────
        # Sort by probability (descending) and try each
        sorted_indices = torch.argsort(policy, descending=True)

        orientations = (
            Orientation.get_all(box.length, box.width, box.height)
            if self.config.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        for idx in sorted_indices[:20]:  # Try top-20 actions
            action = idx.item()
            prob = policy[idx].item()

            if prob < 1e-6:
                break  # All remaining actions have near-zero probability

            # Decode action
            n_orient = cfg.num_orientations
            n_gy = cfg.action_grid_w
            n_gx = cfg.action_grid_l

            a = action
            orient_idx = a % n_orient
            a //= n_orient
            gy = a % n_gy
            a //= n_gy
            gx = a % n_gx
            a //= n_gx
            bin_idx = a

            # Skip actions for other bins (since we're a BaseStrategy
            # and only see one bin, only bin_idx=0 is relevant)
            if bin_idx != 0:
                continue

            # Convert to real-world coordinates
            x = gx * cfg.action_grid_step
            y = gy * cfg.action_grid_step

            # Get oriented dimensions
            if orient_idx >= len(orientations):
                continue
            ol, ow, oh = orientations[orient_idx]

            # Bounds check
            if x + ol > bin_cfg.length + 0.01:
                continue
            if y + ow > bin_cfg.width + 0.01:
                continue

            # Height check
            z = bin_state.get_height_at(x, y, ol, ow)
            if z + oh > bin_cfg.height + 0.01:
                continue

            # Support check
            if z > 0.5:
                support = bin_state.get_support_ratio(x, y, ol, ow, z)
                if support < 0.30:
                    continue

            return PlacementDecision(x=x, y=y, orientation_idx=orient_idx)

        return None

    def _fallback_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """Use the fallback heuristic strategy."""
        if self._fallback is not None:
            try:
                return self._fallback.decide_placement(box, bin_state)
            except Exception:
                pass
        return None

    def on_episode_end(self, results: dict) -> None:
        """Cleanup after episode."""
        super().on_episode_end(results)
        if self._fallback is not None:
            try:
                self._fallback.on_episode_end(results)
            except Exception:
                pass
