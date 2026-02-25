"""
RLDQNStrategy — inference-time strategy for the trained DDQN agent.

Integrates the trained Double DQN model into the standard strategy
framework so it can be used seamlessly with PackingSession.run(),
benchmark_all.py, run_overnight_botko.py, and any other pipeline.

The strategy:
  1. Loads a trained model checkpoint on first episode start
  2. On each decide_placement():
     a. Builds state tensors from the current BinState
     b. Generates candidate placements using CandidateGenerator
     c. Evaluates Q-values for all candidates (batch forward pass)
     d. Returns the highest-Q candidate as a PlacementDecision
     e. Falls back to a greedy heuristic if no valid candidates

Usage:
    # Automatic (registered in strategy registry)
    strategy = get_strategy("rl_dqn")
    result = session.run(boxes, strategy=strategy)

    # Manual with custom checkpoint
    strategy = RLDQNStrategy(checkpoint_path="path/to/best_network.pt")
    result = session.run(boxes, strategy=strategy)
"""

from __future__ import annotations

import sys
import os
from typing import Optional, List

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────
_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

from config import Box, PlacementDecision, ExperimentConfig, BinConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy

# Lazy imports for torch (only loaded when actually needed)
_torch = None
_DQNNetwork = None
_DQNConfig = None
_CandidateGenerator = None


def _lazy_imports():
    """Lazy-load heavy dependencies to avoid import errors when torch is missing."""
    global _torch, _DQNNetwork, _DQNConfig, _CandidateGenerator
    if _torch is None:
        import torch as _torch_mod
        _torch = _torch_mod
        from strategies.rl_dqn.network import DQNNetwork
        _DQNNetwork = DQNNetwork
        from strategies.rl_dqn.config import DQNConfig
        _DQNConfig = DQNConfig
        from strategies.rl_dqn.candidate_generator import CandidateGenerator
        _CandidateGenerator = CandidateGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Default checkpoint path
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "rl_dqn", "checkpoints", "best_network.pt",
)


# ─────────────────────────────────────────────────────────────────────────────
# Greedy fallback
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_fallback(
    box: Box,
    bin_state: BinState,
    config: ExperimentConfig,
) -> Optional[PlacementDecision]:
    """
    Simple greedy heuristic fallback: lowest-z, most-supported position.

    Used when the neural network produces no valid candidates (e.g., the
    bin is nearly full and no heuristic positions remain valid).
    """
    bin_cfg = config.bin
    step = max(1.0, bin_cfg.resolution)

    orientations = (
        Orientation.get_all(box.length, box.width, box.height)
        if config.allow_all_orientations
        else Orientation.get_flat(box.length, box.width, box.height)
    )

    best_score = -float("inf")
    best_placement = None

    for oidx, (ol, ow, oh) in enumerate(orientations):
        if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
            continue

        x = 0.0
        while x + ol <= bin_cfg.length + 1e-6:
            y = 0.0
            while y + ow <= bin_cfg.width + 1e-6:
                z = bin_state.get_height_at(x, y, ol, ow)

                if z + oh > bin_cfg.height + 1e-6:
                    y += step
                    continue

                if z > 0.5:
                    sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                    if sr < 0.30:
                        y += step
                        continue
                else:
                    sr = 1.0

                # Score: prefer low z, high support, close to origin
                score = -z / bin_cfg.height + sr - 0.01 * (x + y) / (bin_cfg.length + bin_cfg.width)
                if score > best_score:
                    best_score = score
                    best_placement = PlacementDecision(x=x, y=y, orientation_idx=oidx)

                y += step
            x += step

    return best_placement


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

@register_strategy
class RLDQNStrategy(BaseStrategy):
    """
    Reinforcement Learning strategy using a trained Double DQN model.

    At inference time:
      1. Generates candidate placements using corner + EP + EMS + grid heuristics
      2. Evaluates Q-values for all candidates via a single batched forward pass
      3. Returns the highest-Q candidate as a PlacementDecision
      4. Falls back to a simple greedy heuristic if no candidates are valid

    The model checkpoint is loaded lazily on the first call to on_episode_start().
    If no checkpoint is found, a warning is printed and the strategy falls back
    to the greedy heuristic for all placements.

    Args:
        checkpoint_path: Path to trained model checkpoint (.pt file).
                         If None, uses the default path in outputs/rl_dqn/.
    """

    name = "rl_dqn"

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self._checkpoint_path = checkpoint_path or _DEFAULT_CHECKPOINT
        self._model = None
        self._candidate_gen = None
        self._device = None
        self._model_loaded = False
        self._load_failed = False

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Load model on first episode (lazy initialisation)."""
        super().on_episode_start(config)

        if self._model_loaded or self._load_failed:
            return

        try:
            _lazy_imports()

            path = os.path.abspath(self._checkpoint_path)
            if not os.path.exists(path):
                print(f"[rl_dqn] WARNING: Checkpoint not found: {path}")
                print("[rl_dqn] Falling back to greedy heuristic for all placements.")
                self._load_failed = True
                return

            self._device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            self._model = _DQNNetwork.load(path, device=self._device)
            self._model.eval()

            # Create candidate generator matching the model's config
            model_cfg = self._model.config
            bin_config = BinConfig(
                length=model_cfg.bin_length,
                width=model_cfg.bin_width,
                height=model_cfg.bin_height,
                resolution=model_cfg.resolution,
            )
            self._candidate_gen = _CandidateGenerator(
                bin_config=bin_config,
                num_bins=model_cfg.num_bins,
                max_candidates=model_cfg.max_candidates,
                use_corner_positions=model_cfg.use_corner_positions,
                use_extreme_points=model_cfg.use_extreme_points,
                use_ems_positions=model_cfg.use_ems_positions,
                use_grid_fallback=model_cfg.use_grid_fallback,
                grid_step=model_cfg.grid_fallback_step,
                num_orientations=model_cfg.num_orientations,
            )

            self._model_loaded = True
            print(
                f"[rl_dqn] Model loaded ({self._model.count_parameters():,} params) "
                f"on {self._device}"
            )

        except Exception as e:
            print(f"[rl_dqn] ERROR loading model: {e}")
            print("[rl_dqn] Falling back to greedy heuristic.")
            self._load_failed = True

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Propose a placement using the trained DDQN model.

        For the single-bin BaseStrategy interface, we only see one BinState.
        We generate candidates for that single bin and evaluate Q-values.

        If the model is not loaded, falls back to a greedy heuristic.

        Args:
            box:       The box to place.
            bin_state: Current state of a single bin.

        Returns:
            PlacementDecision or None.
        """
        # Fallback if model not loaded
        if not self._model_loaded:
            return _greedy_fallback(box, bin_state, self.config)

        try:
            return self._nn_placement(box, bin_state)
        except Exception as e:
            # Safety net: never crash the pipeline
            print(f"[rl_dqn] Error in NN placement: {e}")
            return _greedy_fallback(box, bin_state, self.config)

    def _nn_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """Neural network placement with candidate evaluation."""
        model = self._model
        model_cfg = model.config
        bin_cfg = self.config.bin

        # Generate candidates (single bin — wrap in list for the generator)
        candidates, candidate_features = self._candidate_gen.generate(
            box, [bin_state],
        )

        if not candidates:
            return _greedy_fallback(box, bin_state, self.config)

        # Build state tensors
        # Heightmap: (num_bins, grid_l, grid_w) — for single-bin, second channel is zeros
        num_bins = model_cfg.num_bins
        heightmaps = np.zeros(
            (num_bins, bin_cfg.grid_l, bin_cfg.grid_w), dtype=np.float32,
        )
        heightmaps[0] = bin_state.heightmap.astype(np.float32) / max(bin_cfg.height, 1.0)

        # Box features
        max_dim = max(bin_cfg.length, bin_cfg.width, bin_cfg.height)
        max_vol = max(bin_cfg.volume, 1.0)

        box_feats = np.zeros(model_cfg.box_feature_dim, dtype=np.float32)
        box_feats[0] = box.length / max_dim
        box_feats[1] = box.width / max_dim
        box_feats[2] = box.height / max_dim
        box_feats[3] = box.volume / max_vol
        box_feats[4] = box.weight / 50.0

        # Evaluate Q-values for all candidates
        with _torch.no_grad():
            hm_t = _torch.from_numpy(heightmaps).unsqueeze(0).to(self._device)
            box_t = _torch.from_numpy(box_feats).unsqueeze(0).to(self._device)
            act_t = _torch.from_numpy(candidate_features).to(self._device)

            q_values = model.forward_batch_candidates(hm_t, box_t, act_t)
            best_idx = int(q_values.argmax(dim=0).item())

        # Convert best candidate to PlacementDecision
        best = candidates[best_idx]
        return PlacementDecision(
            x=best.x,
            y=best.y,
            orientation_idx=best.orient_idx,
        )
