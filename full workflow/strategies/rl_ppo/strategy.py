"""
RLPPOStrategy — Inference-time strategy using a trained PPO actor.

Integrates the PPO agent into the standard BaseStrategy interface so it
can be used with PackingSession, run_overnight_botko.py, benchmark_all.py,
and all other framework tools exactly like any heuristic strategy.

Architecture:
    1. Loads a trained ActorCritic checkpoint
    2. Encodes the BinState heightmap + box features into the observation
    3. Runs the decomposed policy: bin -> x -> y -> orient
    4. Applies action masking for physically valid placements
    5. Returns PlacementDecision or falls back to heuristic

Since BaseStrategy.decide_placement() operates on a SINGLE bin, the
strategy maintains internal state about all bins (updated each call)
and uses the full model to select the best position within the given bin.

For multi-bin deployment via PackingSession, the session's BinSelector
logic handles which bin to use.  The strategy receives each bin and
proposes the best position within it.

Usage (standalone):
    strategy = RLPPOStrategy(checkpoint_path="outputs/rl_ppo/logs/checkpoints/best_model.pt")
    decision = strategy.decide_placement(box, bin_state)

Usage (via registry):
    from strategies.base_strategy import get_strategy
    strategy = get_strategy("rl_ppo")
    # Note: needs RLPPOStrategy.set_checkpoint_path() before first use
"""

from __future__ import annotations

import sys
import os
from typing import Optional, Dict, List

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

import torch

from config import Box, PlacementDecision, ExperimentConfig, Orientation, BinConfig
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy

# Lazy imports to avoid loading PyTorch when just importing the strategy module
_ActorCritic = None
_PPOConfig = None


def _ensure_imports():
    """Lazy import of heavy modules (PyTorch network, config)."""
    global _ActorCritic, _PPOConfig
    if _ActorCritic is None:
        from strategies.rl_ppo.network import ActorCritic as AC
        from strategies.rl_ppo.config import PPOConfig as PC
        _ActorCritic = AC
        _PPOConfig = PC


# Module-level checkpoint path override (set before calling get_strategy)
_GLOBAL_CHECKPOINT_PATH: Optional[str] = None


def set_checkpoint_path(path: str) -> None:
    """
    Set the global checkpoint path for RLPPOStrategy instances.

    Call this before creating the strategy via get_strategy("rl_ppo").

    Args:
        path: Absolute or relative path to .pt checkpoint file.
    """
    global _GLOBAL_CHECKPOINT_PATH
    _GLOBAL_CHECKPOINT_PATH = os.path.abspath(path)


# ─────────────────────────────────────────────────────────────────────────────
# Observation encoding (single-bin variant)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_single_bin_obs(
    box: Box,
    bin_state: BinState,
    bin_config: BinConfig,
    config,  # PPOConfig
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Encode a single-bin observation for the actor-critic network.

    Since decide_placement() only receives one bin, we create a
    dummy second bin (empty heightmap) for the multi-bin network.
    The network was trained with 2 bins, so we must provide 2.

    Args:
        box:        Current box to place.
        bin_state:  State of the target bin.
        bin_config: Bin dimensions.
        config:     PPO config.
        device:     Compute device.

    Returns:
        Observation dict suitable for ActorCritic.forward().
    """
    # Heightmaps: (1, num_bins, grid_l, grid_w)
    heightmaps = np.zeros(
        (1, config.num_bins, config.grid_l, config.grid_w),
        dtype=np.float32,
    )
    hm = bin_state.heightmap.astype(np.float32) / max(bin_config.height, 1.0)
    heightmaps[0, 0, :hm.shape[0], :hm.shape[1]] = hm
    # Bin 1 stays empty (zeros) -- dummy for single-bin usage

    # Box features: (1, 5)
    max_dim = max(bin_config.length, bin_config.width, bin_config.height)
    box_features = np.array([[
        box.length / max_dim,
        box.width / max_dim,
        box.height / max_dim,
        box.volume / max(bin_config.volume, 1.0),
        box.weight / 50.0,
    ]], dtype=np.float32)

    # Buffer features: (1, buffer_size, 5) -- just the current box
    buffer_features = np.zeros((1, config.buffer_size, 5), dtype=np.float32)
    buffer_features[0, 0] = box_features[0]

    # Buffer mask
    buffer_mask = np.zeros((1, config.buffer_size), dtype=np.float32)
    buffer_mask[0, 0] = 1.0

    return {
        'heightmaps': torch.as_tensor(heightmaps, device=device),
        'box_features': torch.as_tensor(box_features, device=device),
        'buffer_features': torch.as_tensor(buffer_features, device=device),
        'buffer_mask': torch.as_tensor(buffer_mask, device=device),
    }


def _compute_single_bin_masks(
    box: Box,
    bin_state: BinState,
    bin_config: BinConfig,
    config,  # PPOConfig
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Compute action masks for valid placements in a single bin.

    Checks each (x, y, orientation) combination for physical validity:
    bounds, height limit, and anti-float support.

    Args:
        box:        Current box.
        bin_state:  Target bin state.
        bin_config: Bin dimensions.
        config:     PPO config.
        device:     Compute device.

    Returns:
        Per-sub-action mask dict.
    """
    gl = config.grid_l
    gw = config.grid_w
    n_orient = config.num_orientations
    res = bin_config.resolution
    min_support = 0.30

    # Get orientations
    if n_orient == 6:
        orients = Orientation.get_all(box.length, box.width, box.height)
    else:
        orients = Orientation.get_flat(box.length, box.width, box.height)

    # Build full validity tensor: (n_orient, gl, gw)
    valid = np.zeros((n_orient, gl, gw), dtype=np.float32)

    for oidx, (ol, ow, oh) in enumerate(orients[:n_orient]):
        if ol > bin_config.length or ow > bin_config.width or oh > bin_config.height:
            continue

        for gx in range(gl):
            x = gx * res
            if x + ol > bin_config.length + 0.01:
                continue
            for gy in range(gw):
                y = gy * res
                if y + ow > bin_config.width + 0.01:
                    continue
                z = bin_state.get_height_at(x, y, ol, ow)
                if z + oh > bin_config.height:
                    continue
                if z > 0.01:
                    support = bin_state.get_support_ratio(x, y, ol, ow, z)
                    if support < min_support:
                        continue
                valid[oidx, gx, gy] = 1.0

    # Marginalise to per-dimension masks
    # bin_mask: only bin 0 is valid (we are asked about one specific bin)
    bin_mask = np.array([[1.0] + [0.0] * (config.num_bins - 1)], dtype=np.float32)

    # x_mask: any valid across (orient, y)
    x_mask = valid.any(axis=0).any(axis=1).astype(np.float32).reshape(1, gl)

    # y_mask: any valid across (orient, x)
    y_mask = valid.any(axis=0).any(axis=0).astype(np.float32).reshape(1, gw)

    # orient_mask: any valid across (x, y)
    orient_mask = valid.reshape(n_orient, -1).any(axis=1).astype(np.float32).reshape(1, n_orient)

    # Safety: if no valid actions at all, allow everything (will be caught as None)
    if valid.sum() == 0:
        x_mask[:] = 1.0
        y_mask[:] = 1.0
        orient_mask[:] = 1.0

    return {
        'bin': torch.as_tensor(bin_mask, device=device),
        'x': torch.as_tensor(x_mask, device=device),
        'y': torch.as_tensor(y_mask, device=device),
        'orient': torch.as_tensor(orient_mask, device=device),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy class
# ─────────────────────────────────────────────────────────────────────────────

@register_strategy
class RLPPOStrategy(BaseStrategy):
    """
    Inference strategy using a trained PPO actor-critic network.

    Registered as "rl_ppo" in the strategy registry.  Uses the
    decomposed action approach: bin -> x -> y -> orientation, with
    action masking for physical validity.

    The model is loaded lazily on first use.  Set the checkpoint path
    either via the constructor or ``set_checkpoint_path()`` before
    calling ``get_strategy("rl_ppo")``.

    If no valid RL action can be found (all masked out), falls back
    to the configured heuristic strategy (default: "baseline").

    Attributes:
        name: "rl_ppo"
    """

    name: str = "rl_ppo"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        deterministic: bool = True,
        fallback_strategy: str = "baseline",
    ) -> None:
        """
        Initialise the PPO inference strategy.

        Args:
            checkpoint_path:   Path to .pt checkpoint.  If None, uses
                               the global path set via set_checkpoint_path().
            device:            PyTorch device string.
            deterministic:     If True, use argmax; else sample.
            fallback_strategy: Heuristic fallback name.
        """
        super().__init__()
        self._checkpoint_path = checkpoint_path
        self._device_str = device
        self._deterministic = deterministic
        self._fallback_name = fallback_strategy

        # Lazy-initialised
        self._model: Optional[object] = None
        self._ppo_config: Optional[object] = None
        self._device: Optional[torch.device] = None
        self._fallback: Optional[BaseStrategy] = None

    def _ensure_model(self) -> None:
        """Load the model on first use."""
        if self._model is not None:
            return

        _ensure_imports()

        # Resolve checkpoint path
        ckpt_path = self._checkpoint_path or _GLOBAL_CHECKPOINT_PATH
        if ckpt_path is None:
            # Check default locations
            default_paths = [
                os.path.join(_WORKFLOW_ROOT, "outputs", "rl_ppo", "logs",
                             "checkpoints", "best_model.pt"),
                os.path.join(_WORKFLOW_ROOT, "strategies", "rl_ppo",
                             "checkpoints", "best_model.pt"),
            ]
            for dp in default_paths:
                if os.path.exists(dp):
                    ckpt_path = dp
                    break

        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"[RLPPOStrategy] WARNING: No checkpoint found at {ckpt_path}. "
                  f"Using fallback strategy '{self._fallback_name}'.")
            self._model = None  # Will use fallback only
            return

        # Device
        if self._device_str == 'auto':
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(self._device_str)

        # Config and model
        self._ppo_config = _PPOConfig()
        self._model = _ActorCritic(self._ppo_config).to(self._device)

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()

        param_count = sum(p.numel() for p in self._model.parameters())
        best_fill = checkpoint.get('best_fill', '?')
        print(f"[RLPPOStrategy] Loaded checkpoint: {ckpt_path}")
        print(f"  Parameters: {param_count:,}, Best fill: {best_fill}, Device: {self._device}")

    def _get_fallback(self) -> BaseStrategy:
        """Get or create the fallback heuristic strategy."""
        if self._fallback is None:
            from strategies.base_strategy import get_strategy
            self._fallback = get_strategy(self._fallback_name)
        return self._fallback

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Initialise model and fallback strategy."""
        super().on_episode_start(config)
        self._ensure_model()
        fb = self._get_fallback()
        fb.on_episode_start(config)

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Propose a placement using the trained PPO policy.

        Steps:
            1. Encode observation (heightmap + box features)
            2. Compute action masks for valid placements
            3. Run decomposed policy: bin -> x -> y -> orient
            4. Validate the proposed placement
            5. Fall back to heuristic if RL fails

        Args:
            box:       Box to place.
            bin_state: Current bin state (read-only).

        Returns:
            PlacementDecision or None.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        # If model not loaded, use fallback
        if self._model is None:
            return self._get_fallback().decide_placement(box, bin_state)

        try:
            return self._rl_decide(box, bin_state, bin_cfg)
        except Exception:
            # Any RL inference error -> fall back to heuristic
            return self._get_fallback().decide_placement(box, bin_state)

    def _rl_decide(
        self,
        box: Box,
        bin_state: BinState,
        bin_cfg: BinConfig,
    ) -> Optional[PlacementDecision]:
        """
        Core RL decision logic.

        Runs the trained actor to get a decomposed action, validates it,
        and returns a PlacementDecision.  Returns None and defers to
        fallback on failure.
        """
        ppo_config = self._ppo_config
        device = self._device
        res = bin_cfg.resolution

        # Encode observation
        obs = _encode_single_bin_obs(box, bin_state, bin_cfg, ppo_config, device)
        masks = _compute_single_bin_masks(box, bin_state, bin_cfg, ppo_config, device)

        # Check if any valid placement exists
        x_valid = masks['x'].sum().item()
        y_valid = masks['y'].sum().item()
        orient_valid = masks['orient'].sum().item()
        if x_valid == 0 or y_valid == 0 or orient_valid == 0:
            # No valid placement possible in this bin
            return None

        # Run policy
        with torch.no_grad():
            actions, _, _, _ = self._model(
                obs,
                action_masks=masks,
                deterministic=self._deterministic,
            )

        # Decode action
        x_idx = int(actions['x'][0].item())
        y_idx = int(actions['y'][0].item())
        orient_idx = int(actions['orient'][0].item())

        x = x_idx * res
        y = y_idx * res

        # Resolve orientation
        if ppo_config.num_orientations == 6:
            orients = Orientation.get_all(box.length, box.width, box.height)
        else:
            orients = Orientation.get_flat(box.length, box.width, box.height)

        if orient_idx >= len(orients):
            orient_idx = 0

        ol, ow, oh = orients[orient_idx]

        # Final validation (belt-and-suspenders)
        if x + ol > bin_cfg.length + 0.01:
            return self._get_fallback().decide_placement(box, bin_state)
        if y + ow > bin_cfg.width + 0.01:
            return self._get_fallback().decide_placement(box, bin_state)

        z = bin_state.get_height_at(x, y, ol, ow)
        if z + oh > bin_cfg.height:
            return self._get_fallback().decide_placement(box, bin_state)

        if z > 0.01:
            support = bin_state.get_support_ratio(x, y, ol, ow, z)
            if support < 0.30:
                return self._get_fallback().decide_placement(box, bin_state)

        return PlacementDecision(x=x, y=y, orientation_idx=orient_idx)
