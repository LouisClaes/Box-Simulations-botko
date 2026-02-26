"""
RLMCTSHybridStrategy -- Inference-time strategy for the MCTS-Guided
Hierarchical Actor-Critic.

Integrates the trained MCTSHybridNet into the standard BaseStrategy
interface so it can be used seamlessly with PackingSession.run(),
benchmark_all.py, run_overnight_botko.py, and all other pipelines.

Behaviour at each decide_placement() call:
  1. Encode current bin state + box features into the global state tensor.
  2. Run the high-level policy to select (box_idx, bin_idx).
     - In single-bin mode (BaseStrategy), bin_idx is always 0.
     - In multi-bin mode (MultiBinStrategy), bin_idx is the network's choice.
  3. Generate enriched placement candidates via EnrichedCandidateGenerator.
  4. Run the low-level Transformer pointer to select the best candidate.
  5. Optionally run MCTS lookahead to refine the choice (if enabled and
     world model is warm).
  6. Return PlacementDecision(x, y, orientation_idx).
  7. Fall back to walle_scoring heuristic on any failure.

Both a BaseStrategy and MultiBinStrategy variant are registered.

Usage:
    # Single-bin (via PackingSession or benchmark)
    strategy = get_strategy("rl_mcts_hybrid")
    result = session.run(boxes, strategy=strategy)

    # Multi-bin (via MultiBinPipeline)
    strategy = get_multibin_strategy("rl_mcts_hybrid_multibin")
    result = pipeline.run(boxes, strategy=strategy)

    # Custom checkpoint
    strategy = RLMCTSHybridStrategy(checkpoint_path="/path/to/checkpoint.pt")
"""

from __future__ import annotations

import sys
import os
import warnings
from typing import Optional, List, Dict, Tuple

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

from config import Box, PlacementDecision, ExperimentConfig, BinConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import (
    BaseStrategy, MultiBinStrategy,
    register_strategy, register_multibin_strategy,
    MultiBinDecision,
)
from strategies.rl_mcts_hybrid.config import MCTSHybridConfig

# Lazy imports for heavy dependencies
_torch = None
_MCTSHybridNet = None
_EnrichedCandidateGenerator = None
_MCTSPlanner = None
_compute_void_fraction = None


def _lazy_imports():
    """Load PyTorch and strategy modules lazily to avoid import errors."""
    global _torch, _MCTSHybridNet, _EnrichedCandidateGenerator
    global _MCTSPlanner, _compute_void_fraction
    if _torch is not None:
        return
    import torch as torch_mod
    _torch = torch_mod
    from strategies.rl_mcts_hybrid.network import MCTSHybridNet
    _MCTSHybridNet = MCTSHybridNet
    from strategies.rl_mcts_hybrid.candidate_generator import EnrichedCandidateGenerator
    _EnrichedCandidateGenerator = EnrichedCandidateGenerator
    from strategies.rl_mcts_hybrid.mcts import MCTSPlanner
    _MCTSPlanner = MCTSPlanner
    from strategies.rl_mcts_hybrid.void_detector import compute_void_fraction
    _compute_void_fraction = compute_void_fraction


# ---------------------------------------------------------------------------
# Default checkpoint paths
# ---------------------------------------------------------------------------

_DEFAULT_CHECKPOINT_PATHS = [
    os.path.join(_WORKFLOW_ROOT, "outputs", "rl_mcts_hybrid", "checkpoints", "best_model.pt"),
    os.path.join(_WORKFLOW_ROOT, "outputs", "rl_mcts_hybrid", "checkpoints", "final_model.pt"),
    os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt"),
]


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------

def _encode_observation(
    box: Box,
    bin_states: List[BinState],
    config: MCTSHybridConfig,
    device,
) -> Dict[str, "torch.Tensor"]:
    """
    Encode bin states and box features into tensors for the network.

    Produces the four tensors expected by SharedEncoder.forward():
      - heightmaps:      (1, num_bins, grid_l, grid_w)
      - box_features:    (1, 5)
      - buffer_features: (1, buffer_size, 5)
      - buffer_mask:     (1, buffer_size)

    For single-bin mode, pad missing bins with zeros.
    """
    _lazy_imports()
    num_bins = config.num_bins
    gl = config.grid_l
    gw = config.grid_w
    bin_height = config.bin_height

    # Heightmaps
    heightmaps = np.zeros((1, num_bins, gl, gw), dtype=np.float32)
    for i, bs in enumerate(bin_states):
        if i >= num_bins:
            break
        hm = bs.heightmap.astype(np.float32) / max(bin_height, 1.0)
        # Handle size mismatch gracefully
        hl = min(hm.shape[0], gl)
        hw = min(hm.shape[1], gw)
        heightmaps[0, i, :hl, :hw] = hm[:hl, :hw]

    # Box features: (l, w, h, vol, weight) normalised
    max_dim = max(config.bin_length, config.bin_width, config.bin_height)
    max_vol = config.bin_length * config.bin_width * config.bin_height
    box_feats = np.array([[
        box.length / max_dim,
        box.width / max_dim,
        box.height / max_dim,
        box.volume / max(max_vol, 1.0),
        box.weight / 50.0,
    ]], dtype=np.float32)

    # Buffer features: in single-box mode, just repeat the current box
    # In multi-box mode (with conveyor), the caller provides grippable boxes
    buf = np.zeros((1, config.buffer_size, 5), dtype=np.float32)
    buf[0, 0] = box_feats[0]
    buf_mask = np.zeros((1, config.buffer_size), dtype=np.float32)
    buf_mask[0, 0] = 1.0

    return {
        'heightmaps': _torch.as_tensor(heightmaps, device=device),
        'box_features': _torch.as_tensor(box_feats, device=device),
        'buffer_features': _torch.as_tensor(buf, device=device),
        'buffer_mask': _torch.as_tensor(buf_mask, device=device),
    }


def _encode_observation_with_buffer(
    box: Box,
    bin_states: List[BinState],
    grippable: List[Box],
    buffer_view: List[Box],
    config: MCTSHybridConfig,
    device,
) -> Dict[str, "torch.Tensor"]:
    """
    Full observation encoding with conveyor buffer information.

    Used when the strategy has access to the full conveyor state
    (e.g., via PackingSession step-mode).
    """
    _lazy_imports()
    num_bins = config.num_bins
    gl = config.grid_l
    gw = config.grid_w
    bin_height = config.bin_height

    # Heightmaps
    heightmaps = np.zeros((1, num_bins, gl, gw), dtype=np.float32)
    for i, bs in enumerate(bin_states):
        if i >= num_bins:
            break
        hm = bs.heightmap.astype(np.float32) / max(bin_height, 1.0)
        hl = min(hm.shape[0], gl)
        hw = min(hm.shape[1], gw)
        heightmaps[0, i, :hl, :hw] = hm[:hl, :hw]

    # Box features
    max_dim = max(config.bin_length, config.bin_width, config.bin_height)
    max_vol = config.bin_length * config.bin_width * config.bin_height
    box_feats = np.array([[
        box.length / max_dim,
        box.width / max_dim,
        box.height / max_dim,
        box.volume / max(max_vol, 1.0),
        box.weight / 50.0,
    ]], dtype=np.float32)

    # Buffer features from actual conveyor state
    buf = np.zeros((1, config.buffer_size, 5), dtype=np.float32)
    buf_mask = np.zeros((1, config.buffer_size), dtype=np.float32)
    for i, b in enumerate(buffer_view[:config.buffer_size]):
        buf[0, i] = [
            b.length / max_dim,
            b.width / max_dim,
            b.height / max_dim,
            b.volume / max(max_vol, 1.0),
            b.weight / 50.0,
        ]
        buf_mask[0, i] = 1.0

    return {
        'heightmaps': _torch.as_tensor(heightmaps, device=device),
        'box_features': _torch.as_tensor(box_feats, device=device),
        'buffer_features': _torch.as_tensor(buf, device=device),
        'buffer_mask': _torch.as_tensor(buf_mask, device=device),
    }


# ---------------------------------------------------------------------------
# High-level action masking
# ---------------------------------------------------------------------------

def _build_hl_action_mask(
    grippable_count: int,
    num_bins: int,
    bin_states: List[BinState],
    config: MCTSHybridConfig,
) -> np.ndarray:
    """
    Build action mask for the high-level policy.

    Actions:
      0..pick_window*num_bins-1: (box_i, bin_j) pairs
      pick_window*num_bins:      skip (advance conveyor)
      pick_window*num_bins+1:    reconsider

    Masks out (box_i, bin_j) where box_i >= grippable_count or
    bin_j's height cap is already exceeded.
    """
    n_actions = config.high_level_actions
    mask = np.zeros(n_actions, dtype=np.float32)

    pw = max(1, min(config.pick_window, grippable_count))
    nb = max(1, min(config.num_bins, num_bins))

    for box_i in range(pw):
        for bin_j in range(nb):
            action_idx = box_i * nb + bin_j
            if action_idx >= n_actions:
                break
            # Box must exist
            if box_i >= grippable_count:
                continue
            # Bin must have headroom
            if bin_j < len(bin_states):
                max_h = bin_states[bin_j].get_max_height()
                if max_h >= config.bin_height * 0.95:
                    continue
            mask[action_idx] = 1.0

    # Skip is always valid
    skip_idx = pw * nb
    if skip_idx < n_actions:
        mask[skip_idx] = 1.0

    # Reconsider is always valid
    reconsider_idx = pw * nb + 1
    if reconsider_idx < n_actions:
        mask[reconsider_idx] = 1.0

    # Safety: if nothing is valid, allow skip
    if mask.sum() == 0:
        if skip_idx < n_actions:
            mask[skip_idx] = 1.0

    return mask


def _decode_hl_action(
    action_idx: int,
    config: MCTSHybridConfig,
    active_pick_window: Optional[int] = None,
    active_num_bins: Optional[int] = None,
) -> Tuple[str, int, int]:
    """
    Decode a high-level action index into its semantic meaning.

    Returns:
        (action_type, box_idx, bin_idx)
        action_type: "place", "skip", or "reconsider"
    """
    pw = max(1, min(config.pick_window, active_pick_window or config.pick_window))
    nb = max(1, min(config.num_bins, active_num_bins or config.num_bins))

    skip_idx = pw * nb
    reconsider_idx = pw * nb + 1

    if action_idx == skip_idx:
        return "skip", -1, -1
    elif action_idx == reconsider_idx:
        return "reconsider", -1, -1
    elif action_idx < skip_idx:
        box_idx = action_idx // nb
        bin_idx = action_idx % nb
        return "place", box_idx, bin_idx
    return "skip", -1, -1


# ---------------------------------------------------------------------------
# Single-bin strategy (BaseStrategy)
# ---------------------------------------------------------------------------

@register_strategy
class RLMCTSHybridStrategy(BaseStrategy):
    """
    MCTS-Guided Hierarchical Actor-Critic for 3D bin packing.

    Registered as "rl_mcts_hybrid" in the strategy registry. When used with
    the BaseStrategy single-bin interface, the high-level policy is simplified:
    bin_idx is always 0, and box_idx is always 0 (the first grippable box, as
    decided by PackingSession's BoxSelector).

    The key advantage even in single-bin mode is the Transformer pointer
    network over enriched candidates, which selects from 200 heuristic-quality
    positions using learned attention.

    Falls back to walle_scoring heuristic if:
      - No checkpoint is loaded
      - No valid candidates exist
      - Any inference error occurs
    """

    name: str = "rl_mcts_hybrid"

    def __init__(
        self,
        config: Optional[MCTSHybridConfig] = None,
        checkpoint_path: Optional[str] = None,
        use_mcts: Optional[bool] = None,
        deterministic: bool = True,
    ) -> None:
        """
        Args:
            config:          Strategy configuration. Uses defaults if None.
            checkpoint_path: Override path to trained model checkpoint.
            use_mcts:        Enable/disable MCTS at inference. None = use config.
            deterministic:   Use argmax (True) or sample (False) from policy.
        """
        super().__init__()
        self.hybrid_config = config or MCTSHybridConfig()
        self._checkpoint_path = checkpoint_path
        self._use_mcts = use_mcts if use_mcts is not None else self.hybrid_config.mcts_enabled
        self._deterministic = deterministic

        # Lazy-initialised components
        self._model = None
        self._candidate_gen = None
        self._mcts_planner = None
        self._device = None
        self._fallback_strategy: Optional[BaseStrategy] = None
        self._model_loaded = False
        self._load_attempted = False

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Load model and initialise components on first episode."""
        super().on_episode_start(config)

        if not self._load_attempted:
            self._load_attempted = True
            self._try_load_model()

        # Create candidate generator (always, even without model for fallback enrichment)
        if self._candidate_gen is None:
            _lazy_imports()
            self._candidate_gen = _EnrichedCandidateGenerator(self.hybrid_config)

        # Create MCTS planner
        if self._use_mcts and self._mcts_planner is None and self._model_loaded:
            _lazy_imports()
            self._mcts_planner = _MCTSPlanner(
                num_simulations=self.hybrid_config.mcts_simulations,
                max_depth=self.hybrid_config.mcts_depth,
                c_puct=self.hybrid_config.mcts_c_puct,
                discount=self.hybrid_config.mcts_discount,
                temperature=0.01 if self._deterministic else self.hybrid_config.mcts_temperature,
                device=self._device,
            )

        # Initialise fallback
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
        Propose a placement using the hierarchical RL policy.

        In single-bin mode:
          1. Encode observation with single bin state
          2. Skip high-level (bin_idx=0 always)
          3. Generate candidates for this bin
          4. Run low-level Transformer pointer
          5. Return best candidate as PlacementDecision

        Falls back to heuristic on any failure.
        """
        if not self._model_loaded:
            return self._fallback_decide(box, bin_state)

        try:
            decision = self._rl_decide_single(box, bin_state)
            if decision is not None:
                return decision
        except Exception as e:
            warnings.warn(
                f"[rl_mcts_hybrid] Inference error: {e}. Using fallback.",
                RuntimeWarning,
            )

        return self._fallback_decide(box, bin_state)

    def _rl_decide_single(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Core RL decision for single-bin mode.

        Skips the high-level policy (bin_idx is predetermined) and goes
        directly to candidate generation + low-level pointer selection.
        """
        config = self.hybrid_config
        device = self._device
        model = self._model

        # Encode observation (single bin)
        obs = _encode_observation(box, [bin_state], config, device)

        # Encode state
        with _torch.no_grad():
            global_state, bin_embeds = model.encode(
                obs['heightmaps'],
                obs['box_features'],
                obs['buffer_features'],
                obs['buffer_mask'],
            )

            # Get high-level action embedding for bin 0 (action index = 0*num_bins + 0 = 0)
            hl_action = _torch.tensor([0], device=device)
            hl_embed = model.high_level.action_embedding(hl_action)

        # Generate candidates for this bin
        candidates = self._candidate_gen.generate(box, [bin_state])
        if not candidates:
            return None

        # Prepare candidate features tensor
        cand_feats = self._candidate_gen.get_feature_array(candidates)
        n_cands = len(candidates)

        # Pad to max_candidates
        max_cands = config.max_candidates
        padded_feats = np.zeros((1, max_cands, config.candidate_input_dim), dtype=np.float32)
        valid_cands = min(n_cands, max_cands)
        padded_feats[0, :valid_cands] = cand_feats[:valid_cands]

        cand_mask = np.zeros((1, max_cands), dtype=bool)
        cand_mask[0, :valid_cands] = True

        cand_t = _torch.as_tensor(padded_feats, device=device)
        mask_t = _torch.as_tensor(cand_mask, device=device)

        # Run low-level policy
        with _torch.no_grad():
            ll_out = model.low_level(
                global_state, hl_embed,
                cand_t, mask_t,
                deterministic=self._deterministic,
            )

        action_idx = int(ll_out.action.item())
        if action_idx >= n_cands:
            action_idx = 0

        selected = candidates[action_idx]

        # Final safety validation
        bin_cfg = self.config.bin
        ol, ow, oh = selected.oriented_l, selected.oriented_w, selected.oriented_h
        if selected.x + ol > bin_cfg.length + 0.01:
            return None
        if selected.y + ow > bin_cfg.width + 0.01:
            return None
        if selected.z + oh > bin_cfg.height:
            return None

        return PlacementDecision(
            x=selected.x,
            y=selected.y,
            orientation_idx=selected.orient_idx,
        )

    def _fallback_decide(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """Use fallback heuristic strategy."""
        if self._fallback_strategy is not None:
            return self._fallback_strategy.decide_placement(box, bin_state)
        return None

    def _try_load_model(self) -> None:
        """Attempt to load the trained model checkpoint."""
        try:
            _lazy_imports()
        except ImportError:
            warnings.warn(
                "[rl_mcts_hybrid] PyTorch not available. Using fallback.",
                RuntimeWarning,
            )
            return

        # Resolve checkpoint path
        ckpt_path = self._checkpoint_path
        if ckpt_path is None:
            for path in _DEFAULT_CHECKPOINT_PATHS:
                if os.path.exists(path):
                    ckpt_path = path
                    break

        if ckpt_path is None or not os.path.exists(ckpt_path):
            warnings.warn(
                f"[rl_mcts_hybrid] No checkpoint found. Using fallback "
                f"'{self.hybrid_config.fallback_strategy}'.",
                RuntimeWarning,
            )
            return

        try:
            self._device = _torch.device(
                "cuda" if _torch.cuda.is_available() else "cpu"
            )
            self._model = _MCTSHybridNet.load(ckpt_path, device=self._device)
            self._model.eval()
            self._model_loaded = True

            print(
                f"[rl_mcts_hybrid] Loaded model from {ckpt_path} "
                f"({self._model.count_parameters():,} params) on {self._device}"
            )

        except Exception as e:
            warnings.warn(
                f"[rl_mcts_hybrid] Failed to load checkpoint: {e}. Using fallback.",
                RuntimeWarning,
            )

    def _create_fallback(self) -> Optional[BaseStrategy]:
        """Create fallback heuristic strategy."""
        from strategies.base_strategy import STRATEGY_REGISTRY

        fallback_name = self.hybrid_config.fallback_strategy
        if fallback_name in STRATEGY_REGISTRY:
            return STRATEGY_REGISTRY[fallback_name]()

        # Try common fallbacks
        for name in ["walle_scoring", "surface_contact", "baseline"]:
            if name in STRATEGY_REGISTRY:
                return STRATEGY_REGISTRY[name]()

        return None

    def on_episode_end(self, results: dict) -> None:
        """Clean up after episode."""
        if self._fallback_strategy is not None:
            self._fallback_strategy.on_episode_end(results)
        super().on_episode_end(results)


# ---------------------------------------------------------------------------
# Multi-bin strategy (MultiBinStrategy)
# ---------------------------------------------------------------------------

@register_multibin_strategy
class RLMCTSHybridMultiBinStrategy(MultiBinStrategy):
    """
    Multi-bin variant of the MCTS-Guided Hierarchical Actor-Critic.

    Registered as "rl_mcts_hybrid_multibin" in the multi-bin registry.
    Uses the FULL hierarchical architecture:
      1. High-level policy selects (box_idx, bin_idx) from grippable window
      2. Low-level policy selects placement candidate for chosen bin
      3. Optional MCTS lookahead with world model

    This is the strategy's native mode -- it jointly optimises which box to
    pick, which bin to target, and where to place, using cross-attention
    across all bins simultaneously.

    For use with MultiBinPipeline or PackingSession step-mode.
    """

    name: str = "rl_mcts_hybrid_multibin"

    def __init__(
        self,
        config: Optional[MCTSHybridConfig] = None,
        checkpoint_path: Optional[str] = None,
        use_mcts: Optional[bool] = None,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        self.hybrid_config = config or MCTSHybridConfig()
        self._checkpoint_path = checkpoint_path
        self._use_mcts = use_mcts if use_mcts is not None else self.hybrid_config.mcts_enabled
        self._deterministic = deterministic

        # Lazy-initialised
        self._model = None
        self._candidate_gen = None
        self._mcts_planner = None
        self._device = None
        self._fallback_strategy: Optional[BaseStrategy] = None
        self._model_loaded = False
        self._load_attempted = False
        self._grippable: List[Box] = []
        self._buffer_view: List[Box] = []

    def set_conveyor_state(
        self,
        grippable: List[Box],
        buffer_view: List[Box],
    ) -> None:
        """
        Update conveyor state for this step.

        Called externally (e.g., from the training loop or PackingSession
        step-mode integration) to provide the current conveyor snapshot.
        """
        self._grippable = grippable
        self._buffer_view = buffer_view

    def on_episode_start(self, config) -> None:
        super().on_episode_start(config)

        if not self._load_attempted:
            self._load_attempted = True
            self._try_load_model()

        if self._candidate_gen is None:
            _lazy_imports()
            self._candidate_gen = _EnrichedCandidateGenerator(self.hybrid_config)

        if self._use_mcts and self._mcts_planner is None and self._model_loaded:
            _lazy_imports()
            self._mcts_planner = _MCTSPlanner(
                num_simulations=self.hybrid_config.mcts_simulations,
                max_depth=self.hybrid_config.mcts_depth,
                c_puct=self.hybrid_config.mcts_c_puct,
                discount=self.hybrid_config.mcts_discount,
                temperature=0.01 if self._deterministic else self.hybrid_config.mcts_temperature,
                device=self._device,
            )

        if self._fallback_strategy is None:
            self._fallback_strategy = self._create_fallback()
        if self._fallback_strategy is not None:
            self._fallback_strategy.on_episode_start(config)

    def decide_placement(
        self,
        box: Box,
        bin_states: List[BinState],
    ) -> Optional[MultiBinDecision]:
        """
        Propose a placement across all active bins.

        Full pipeline:
          1. Encode global state from all bins + box + conveyor
          2. Run high-level policy -> (box_idx, bin_idx)
          3. Generate candidates for the selected bin
          4. Run low-level Transformer pointer -> best candidate
          5. (Optional) MCTS refinement
          6. Return MultiBinDecision(bin_index, x, y, orientation_idx)
        """
        if not self._model_loaded:
            return self._fallback_multibin(box, bin_states)

        try:
            decision = self._rl_decide_multi(box, bin_states)
            if decision is not None:
                return decision
        except Exception as e:
            warnings.warn(
                f"[rl_mcts_hybrid_multibin] Inference error: {e}. Using fallback.",
                RuntimeWarning,
            )

        return self._fallback_multibin(box, bin_states)

    def _rl_decide_multi(
        self,
        box: Box,
        bin_states: List[BinState],
    ) -> Optional[MultiBinDecision]:
        """
        Full hierarchical decision with high-level and low-level policies.
        """
        config = self.hybrid_config
        device = self._device
        model = self._model

        # Encode observation with full conveyor state
        if self._buffer_view:
            obs = _encode_observation_with_buffer(
                box, bin_states, self._grippable, self._buffer_view,
                config, device,
            )
        else:
            obs = _encode_observation(box, bin_states, config, device)

        with _torch.no_grad():
            # Encode
            global_state, bin_embeds = model.encode(
                obs['heightmaps'],
                obs['box_features'],
                obs['buffer_features'],
                obs['buffer_mask'],
            )

            # High-level action mask
            grippable_count = max(1, len(self._grippable))
            hl_mask_np = _build_hl_action_mask(
                grippable_count, len(bin_states), bin_states, config,
            )
            hl_mask = _torch.as_tensor(
                hl_mask_np.reshape(1, -1), device=device,
            )

            # High-level policy
            hl_out = model.high_level(
                global_state, hl_mask,
                deterministic=self._deterministic,
            )

        hl_action_idx = int(hl_out.action.item())
        action_type, box_idx, bin_idx = _decode_hl_action(
            hl_action_idx,
            config,
            active_pick_window=grippable_count,
            active_num_bins=len(bin_states),
        )

        if action_type == "skip":
            return None  # Caller should advance conveyor

        if action_type == "reconsider":
            # Re-run with different temperature or fall back
            return self._fallback_multibin(box, bin_states)

        # Validate bin_idx
        if bin_idx < 0 or bin_idx >= len(bin_states):
            bin_idx = 0

        # Generate candidates for the selected bin
        candidates = self._candidate_gen.generate(box, [bin_states[bin_idx]])
        if not candidates:
            return None

        # Prepare candidate features
        cand_feats = self._candidate_gen.get_feature_array(candidates)
        n_cands = len(candidates)
        max_cands = config.max_candidates

        padded_feats = np.zeros((1, max_cands, config.candidate_input_dim), dtype=np.float32)
        padded_feats[0, :min(n_cands, max_cands)] = cand_feats[:max_cands]

        cand_mask = np.zeros((1, max_cands), dtype=bool)
        cand_mask[0, :min(n_cands, max_cands)] = True

        cand_t = _torch.as_tensor(padded_feats, device=device)
        mask_t = _torch.as_tensor(cand_mask, device=device)

        # Low-level policy
        with _torch.no_grad():
            ll_out = model.low_level(
                global_state, hl_out.action_embed,
                cand_t, mask_t,
                deterministic=self._deterministic,
            )

        action_idx = int(ll_out.action.item())
        if action_idx >= n_cands:
            action_idx = 0

        if self._use_mcts and self._mcts_planner is not None:
            try:
                best_hl, best_ll, _stats = self._mcts_planner.search(
                    model=model,
                    global_state=global_state,
                    hl_probs=hl_out.probs.detach(),
                    hl_mask=hl_mask,
                    ll_probs_per_hl={hl_action_idx: ll_out.probs.detach()},
                    ll_masks_per_hl={hl_action_idx: mask_t},
                    candidate_features_per_hl={hl_action_idx: cand_t},
                )
                if best_hl == hl_action_idx and 0 <= best_ll < n_cands:
                    action_idx = int(best_ll)
            except Exception as exc:
                warnings.warn(
                    f"[rl_mcts_hybrid_multibin] MCTS refinement failed: {exc}. "
                    "Using policy action.",
                    RuntimeWarning,
                )

        selected = candidates[action_idx]

        return MultiBinDecision(
            bin_index=bin_idx,
            x=selected.x,
            y=selected.y,
            orientation_idx=selected.orient_idx,
        )

    def _fallback_multibin(
        self,
        box: Box,
        bin_states: List[BinState],
    ) -> Optional[MultiBinDecision]:
        """Fall back to heuristic, trying each bin."""
        if self._fallback_strategy is None:
            return None

        # Try each bin and pick the best placement
        best_decision = None
        best_score = -float("inf")

        for bin_idx, bs in enumerate(bin_states):
            decision = self._fallback_strategy.decide_placement(box, bs)
            if decision is not None:
                # Score by fill delta (simple heuristic)
                orientations = Orientation.get_flat(box.length, box.width, box.height)
                oidx = decision.orientation_idx
                if oidx < len(orientations):
                    ol, ow, oh = orientations[oidx]
                else:
                    ol, ow, oh = box.length, box.width, box.height
                z = bs.get_height_at(decision.x, decision.y, ol, ow)
                score = -z + bs.get_fill_rate() * 100
                if score > best_score:
                    best_score = score
                    best_decision = MultiBinDecision(
                        bin_index=bin_idx,
                        x=decision.x,
                        y=decision.y,
                        orientation_idx=decision.orientation_idx,
                    )

        return best_decision

    def _try_load_model(self) -> None:
        """Attempt to load the trained model checkpoint."""
        try:
            _lazy_imports()
        except ImportError:
            return

        ckpt_path = self._checkpoint_path
        if ckpt_path is None:
            for path in _DEFAULT_CHECKPOINT_PATHS:
                if os.path.exists(path):
                    ckpt_path = path
                    break

        if ckpt_path is None or not os.path.exists(ckpt_path):
            warnings.warn(
                f"[rl_mcts_hybrid_multibin] No checkpoint found. Using fallback.",
                RuntimeWarning,
            )
            return

        try:
            self._device = _torch.device(
                "cuda" if _torch.cuda.is_available() else "cpu"
            )
            self._model = _MCTSHybridNet.load(ckpt_path, device=self._device)
            self._model.eval()
            self._model_loaded = True
            print(
                f"[rl_mcts_hybrid_multibin] Loaded model "
                f"({self._model.count_parameters():,} params) on {self._device}"
            )
        except Exception as e:
            warnings.warn(
                f"[rl_mcts_hybrid_multibin] Load failed: {e}",
                RuntimeWarning,
            )

    def _create_fallback(self) -> Optional[BaseStrategy]:
        """Create fallback heuristic strategy."""
        from strategies.base_strategy import STRATEGY_REGISTRY
        for name in [self.hybrid_config.fallback_strategy, "walle_scoring", "baseline"]:
            if name in STRATEGY_REGISTRY:
                return STRATEGY_REGISTRY[name]()
        return None

    def on_episode_end(self, results: dict) -> None:
        if self._fallback_strategy is not None:
            self._fallback_strategy.on_episode_end(results)
        super().on_episode_end(results)
