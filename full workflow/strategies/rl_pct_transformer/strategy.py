"""
RLPCTTransformerStrategy — BaseStrategy wrapper for inference.

Integrates the trained PCT Transformer model into the standard strategy
interface so it can be used with PackingSession.run(), benchmark_all.py,
and all other existing infrastructure.

Behaviour:
  1. On first use, loads the trained checkpoint (if available).
  2. For each box: generate placement candidates, run through Transformer,
     select the best candidate via pointer attention.
  3. If no checkpoint is available or no valid candidates exist, falls back
     to the configured heuristic strategy (default: extreme_points).

The strategy is SINGLE-BIN (BaseStrategy interface).  When used in a
multi-bin setup via PackingSession, the session's BinSelector decides
which bin to offer to this strategy.  The candidate generator only
considers the single bin_state passed to decide_placement().

For native multi-bin RL (where the network sees all bins simultaneously),
use the training environment directly via PackingSession step mode.
"""

from __future__ import annotations

import sys
import os
from typing import Optional
import warnings

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

import numpy as np

from config import Box, PlacementDecision, ExperimentConfig, BinConfig
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy

from strategies.rl_pct_transformer.config import PCTTransformerConfig
from strategies.rl_pct_transformer.candidate_generator import CandidateGenerator


@register_strategy
class RLPCTTransformerStrategy(BaseStrategy):
    """
    PCT-inspired Transformer RL strategy for 3D bin packing.

    Uses a trained Transformer actor-critic to select placement candidates
    via a pointer mechanism.  Falls back to a heuristic strategy when no
    trained model is available.

    Attributes:
        name: Strategy identifier ("rl_pct_transformer").
    """

    name: str = "rl_pct_transformer"

    def __init__(
        self,
        config: Optional[PCTTransformerConfig] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            config:          Hyperparameter configuration.
            checkpoint_path: Override path to trained model checkpoint.
                             If None, uses config.checkpoint_path.
        """
        super().__init__()
        self.pct_config = config or PCTTransformerConfig()
        if checkpoint_path is not None:
            self.pct_config.checkpoint_path = checkpoint_path

        self._network = None
        self._device = None
        self._candidate_gen: Optional[CandidateGenerator] = None
        self._fallback_strategy: Optional[BaseStrategy] = None
        self._model_loaded = False
        self._load_attempted = False

    def on_episode_start(self, config) -> None:
        """Load model and initialise candidate generator."""
        super().on_episode_start(config)

        # Determine BinConfig from experiment config
        bin_config = config.bin if hasattr(config, 'bin') else BinConfig()

        # Create candidate generator
        self._candidate_gen = CandidateGenerator(
            bin_config=bin_config,
            min_support=self.pct_config.min_support_ratio,
            num_orientations=self.pct_config.num_orientations,
            floor_scan_step=self.pct_config.floor_scan_step,
            dedup_tolerance=self.pct_config.candidate_dedup_tolerance,
        )

        # Load model (lazy, once)
        if not self._load_attempted:
            self._load_attempted = True
            self._try_load_model()

        # Initialise fallback strategy
        if self._fallback_strategy is None:
            self._fallback_strategy = self._create_fallback()

        if self._fallback_strategy is not None:
            self._fallback_strategy.on_episode_start(config)

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Select the best placement for the box using the Transformer model.

        Steps:
          1. Generate candidate placements from the bin state.
          2. Encode box features and candidate features.
          3. Forward pass through Transformer pointer network.
          4. Select the candidate with highest probability (or sample).
          5. Return PlacementDecision(x, y, orientation_idx).

        Falls back to heuristic if:
          - No model is loaded
          - No valid candidates exist
          - Any error occurs during inference

        Args:
            box:       Box to place.
            bin_state: Current state of one bin.

        Returns:
            PlacementDecision or None if no valid placement exists.
        """
        # Try RL model
        if self._model_loaded and self._candidate_gen is not None:
            try:
                decision = self._rl_decide(box, bin_state)
                if decision is not None:
                    return decision
            except Exception as e:
                warnings.warn(
                    f"[rl_pct_transformer] Inference error: {e}. Using fallback.",
                    RuntimeWarning,
                )

        # Fallback to heuristic
        if self._fallback_strategy is not None:
            return self._fallback_strategy.decide_placement(box, bin_state)

        return None

    def _rl_decide(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Run the Transformer model to select a placement.

        Returns None if no valid candidates exist.
        """
        import torch
        from strategies.rl_common.obs_utils import encode_box_features

        # Generate candidates (single bin, bin_idx=0)
        candidates = self._candidate_gen.generate(
            box, [bin_state],
            max_candidates=self.pct_config.max_candidates,
        )

        if not candidates:
            return None

        # Encode item features
        bin_config = bin_state.config
        item_feat = encode_box_features(box, bin_config)
        item_t = torch.from_numpy(item_feat).unsqueeze(0).to(self._device)

        # Encode candidate features
        cand_feats = np.stack([c.features for c in candidates])
        cand_t = torch.from_numpy(cand_feats).unsqueeze(0).to(self._device)
        mask_t = torch.ones(1, len(candidates), dtype=torch.bool, device=self._device)

        # Forward pass
        with torch.no_grad():
            action, _, _, _ = self._network.get_action_and_value(
                item_t, cand_t, mask_t,
                deterministic=self.pct_config.deterministic_inference,
            )

        action_idx = action.item()
        if action_idx >= len(candidates):
            action_idx = 0

        selected = candidates[action_idx]

        return PlacementDecision(
            x=selected.x,
            y=selected.y,
            orientation_idx=selected.orient_idx,
        )

    def _try_load_model(self) -> None:
        """Attempt to load the trained model checkpoint."""
        try:
            import torch
            from strategies.rl_pct_transformer.network import PCTTransformerNet
        except ImportError:
            warnings.warn(
                "[rl_pct_transformer] PyTorch not available. Using fallback strategy.",
                RuntimeWarning,
            )
            return

        checkpoint_path = self.pct_config.checkpoint_path

        # Auto-detect checkpoint in default location
        if checkpoint_path is None:
            default_paths = [
                os.path.join(self.pct_config.log_dir, "best.pt"),
                os.path.join(self.pct_config.log_dir, "final.pt"),
                os.path.join(_WORKFLOW_ROOT, "outputs", "rl_pct_transformer", "logs", "best.pt"),
            ]
            for path in default_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            warnings.warn(
                f"[rl_pct_transformer] No checkpoint found. Using fallback strategy "
                f"'{self.pct_config.fallback_strategy}'.",
                RuntimeWarning,
            )
            return

        try:
            device_str = self.pct_config.resolved_device
            self._device = torch.device(device_str)

            checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
            self._network = PCTTransformerNet(self.pct_config).to(self._device)
            self._network.load_state_dict(checkpoint['model_state_dict'])
            self._network.eval()
            self._model_loaded = True

            print(f"[rl_pct_transformer] Loaded model from {checkpoint_path} "
                  f"(episode {checkpoint.get('episode', '?')}, "
                  f"fill {checkpoint.get('best_fill', '?')})")

        except Exception as e:
            warnings.warn(
                f"[rl_pct_transformer] Failed to load checkpoint: {e}. Using fallback.",
                RuntimeWarning,
            )

    def _create_fallback(self) -> Optional[BaseStrategy]:
        """Create the fallback heuristic strategy."""
        from strategies.base_strategy import STRATEGY_REGISTRY

        fallback_name = self.pct_config.fallback_strategy
        if fallback_name in STRATEGY_REGISTRY:
            return STRATEGY_REGISTRY[fallback_name]()

        # Try common fallbacks
        for name in ["extreme_points", "baseline", "walle_scoring"]:
            if name in STRATEGY_REGISTRY:
                return STRATEGY_REGISTRY[name]()

        warnings.warn(
            f"[rl_pct_transformer] No fallback strategy available.",
            RuntimeWarning,
        )
        return None

    def on_episode_end(self, results: dict) -> None:
        """Clean up after episode."""
        if self._fallback_strategy is not None:
            self._fallback_strategy.on_episode_end(results)
        super().on_episode_end(results)
