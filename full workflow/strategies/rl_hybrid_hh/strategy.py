"""
RLHybridHHStrategy -- RL-based selective hyper-heuristic for 3D bin packing.

NOVEL THESIS CONTRIBUTION: A trained Q-learning agent that selects which
expert placement heuristic to apply for each incoming box, based on the
current packing state.

Architecture overview:
    1. Extract 39-dimensional handcrafted state features
    2. Q-network (or Q-table) maps state -> Q-values for 8 actions
    3. Select heuristic with highest Q-value (greedy at inference)
    4. Call the selected heuristic's decide_placement() on the actual bin_state
    5. Return the heuristic's PlacementDecision (or None)

The strategy maintains a portfolio of 7 existing heuristics plus SKIP:
    0. baseline (DBLF)
    1. walle_scoring
    2. surface_contact
    3. extreme_points
    4. skyline
    5. layer_building
    6. best_fit_decreasing
    7. SKIP (return None, advance conveyor)

At inference time:
    - Loads a pre-trained checkpoint (Q-table or DQN weights)
    - No exploration (greedy policy)
    - Falls back to walle_scoring if no checkpoint is loaded
    - Tracks which heuristic was chosen (for interpretability analysis)

Integration:
    - Registered as "rl_hybrid_hh" in STRATEGY_REGISTRY
    - Compatible with PackingSession, batch_runner, benchmark_all, etc.
    - Uses the same decide_placement(box, bin_state) interface as all strategies

Usage:
    # With trained model:
    strategy = RLHybridHHStrategy(
        checkpoint_path="outputs/rl_hybrid_hh/best_model.pt"
    )

    # Without model (falls back to walle_scoring):
    strategy = RLHybridHHStrategy()

    # In session:
    result = session.run(boxes, strategy)
"""

from __future__ import annotations

import sys
import os
from typing import Optional, List, Dict, Any

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

from config import Box, PlacementDecision, ExperimentConfig
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy, get_strategy
from strategies.rl_hybrid_hh.config import HHConfig
from strategies.rl_hybrid_hh.state_features import (
    FeatureTracker,
    extract_state_features,
    discretise_state,
)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

@register_strategy
class RLHybridHHStrategy(BaseStrategy):
    """
    RL-based selective hyper-heuristic for 3D bin packing.

    Uses a trained Q-learning agent (tabular or DQN) to select which
    expert placement heuristic to apply for each incoming box.

    The strategy maintains a portfolio of instantiated heuristic strategies
    and delegates the actual placement decision to whichever heuristic the
    Q-agent selects.  This means:
      - No new placement logic is needed
      - All existing heuristic implementations are reused as-is
      - The RL agent only needs to learn WHEN to use each heuristic

    Attributes:
        name:             "rl_hybrid_hh" (registry key).
        _config_hh:       HHConfig hyperparameters.
        _portfolio:       Dict mapping action index to strategy instance.
        _agent:           Trained Q-agent (TabularQLearner or DQN).
        _tracker:         FeatureTracker for history features.
        _selection_log:   Per-step log of which heuristic was chosen.
        _fallback:        Default heuristic when agent is not available.
    """

    name: str = "rl_hybrid_hh"

    def __init__(
        self,
        checkpoint_path: str = "",
        config: Optional[HHConfig] = None,
        mode: str = "auto",
    ) -> None:
        """
        Initialise the RL Hybrid Hyper-Heuristic strategy.

        Args:
            checkpoint_path: Path to a trained model file.
                             - ".npz" -> tabular Q-learner
                             - ".pt"  -> DQN checkpoint
                             - ""     -> fallback mode (walle_scoring)
            config:          HHConfig override (defaults to HHConfig()).
            mode:            "tabular", "dqn", or "auto" (detect from file).
        """
        super().__init__()
        self._config_hh = config or HHConfig()
        self._checkpoint_path = checkpoint_path
        self._mode = mode

        # These are initialised in on_episode_start() because they need
        # access to the ExperimentConfig
        self._portfolio: Dict[int, BaseStrategy] = {}
        self._agent = None
        self._tracker = FeatureTracker(self._config_hh)
        self._selection_log: List[Dict[str, Any]] = []
        self._fallback_idx: int = 1  # walle_scoring
        self._initialised: bool = False

    def _init_portfolio(self) -> None:
        """
        Instantiate the heuristic portfolio from the strategy registry.

        Each heuristic is loaded by name from STRATEGY_REGISTRY and stored
        in _portfolio[action_index].  This ensures we use the ACTUAL
        strategy implementations, not copies or approximations.
        """
        self._portfolio = {}
        for idx, name in enumerate(self._config_hh.heuristic_names):
            try:
                strategy = get_strategy(name)
                self._portfolio[idx] = strategy
            except ValueError:
                print(f"[rl_hybrid_hh] WARNING: Strategy '{name}' not found "
                      f"in registry, skipping action {idx}")

        if not self._portfolio:
            raise RuntimeError(
                "[rl_hybrid_hh] No heuristics could be loaded from the "
                "registry. Check that heuristic_names in HHConfig match "
                "registered strategy names."
            )

    def _init_agent(self) -> None:
        """
        Load the trained Q-agent from checkpoint.

        Supports two modes:
          - "tabular": loads a .npz file with Q-table
          - "dqn": loads a .pt file with DQN weights
          - "auto": detects mode from file extension
        """
        path = self._checkpoint_path
        if not path or not os.path.exists(path):
            self._agent = None
            return

        mode = self._mode
        if mode == "auto":
            if path.endswith(".npz"):
                mode = "tabular"
            elif path.endswith(".pt"):
                mode = "dqn"
            else:
                print(f"[rl_hybrid_hh] Cannot detect mode from '{path}', "
                      f"using fallback")
                self._agent = None
                return

        if mode == "tabular":
            from strategies.rl_hybrid_hh.network import TabularQLearner
            self._agent = TabularQLearner.load(path)
            print(f"[rl_hybrid_hh] Loaded tabular Q-learner from {path}")
            self._mode = "tabular"

        elif mode == "dqn":
            try:
                from strategies.rl_hybrid_hh.network import HeuristicSelectorDQN
                self._agent = HeuristicSelectorDQN.load(path)
                self._agent.eval()
                print(f"[rl_hybrid_hh] Loaded DQN model from {path} "
                      f"({self._agent.count_parameters()} params)")
                self._mode = "dqn"
            except ImportError:
                print("[rl_hybrid_hh] PyTorch not available, using fallback")
                self._agent = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """
        Initialise portfolio, agent, and tracker at episode start.

        The portfolio heuristics also receive on_episode_start() so they
        can initialise their own internal state (scan step, etc.).
        """
        super().on_episode_start(config)

        if not self._initialised:
            self._init_portfolio()
            self._init_agent()
            self._initialised = True

        # Initialise all portfolio heuristics
        for strategy in self._portfolio.values():
            strategy.on_episode_start(config)

        # Reset tracker
        self._tracker.reset(total_boxes=100)  # Will be updated if known
        self._selection_log = []

    def on_episode_end(self, results: dict) -> None:
        """Attach heuristic selection log and statistics to results."""
        super().on_episode_end(results)

        # End all portfolio heuristics
        for strategy in self._portfolio.values():
            strategy.on_episode_end({})

        # Compute selection statistics
        if self._selection_log:
            action_counts: Dict[str, int] = {}
            for entry in self._selection_log:
                name = entry.get("heuristic", "unknown")
                action_counts[name] = action_counts.get(name, 0) + 1

            results["hh_selection_log"] = self._selection_log
            results["hh_selection_counts"] = action_counts
            results["hh_total_steps"] = len(self._selection_log)

    # ── Main Entry Point ──────────────────────────────────────────────────

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Select a heuristic and delegate placement to it.

        Steps:
          1. Extract state features from the current packing state
          2. Q-agent selects a heuristic (or fallback to walle_scoring)
          3. Call the selected heuristic's decide_placement()
          4. If it fails, try the fallback heuristic
          5. Log the selection for interpretability analysis

        Args:
            box:       The box to place (read-only).
            bin_state: Current 3D state of one bin (read-only).

        Returns:
            PlacementDecision(x, y, orientation_idx), or None if no
            heuristic can place the box.
        """
        # Gather state information
        # Note: In single-bin mode, we only have one bin_state.
        # We create a list for feature extraction compatibility.
        bin_states = [bin_state]
        bin_config = bin_state.config

        # Get grippable and buffer info (not available in single-bin mode,
        # so we use empty lists and let the features default)
        grippable = [box]  # Minimal: just the current box
        buffer_view = [box]

        # Select heuristic
        action, q_values = self._select_heuristic(
            box, bin_states, bin_config, grippable, buffer_view,
        )

        # Get the selected heuristic name
        if action < len(self._config_hh.heuristic_names):
            heuristic_name = self._config_hh.heuristic_names[action]
        else:
            heuristic_name = "SKIP"

        # SKIP action: return None (advance conveyor)
        if self._config_hh.include_skip and action == self._config_hh.num_actions - 1:
            self._log_selection(box, heuristic_name, action, q_values, None)
            self._tracker.record_choice(action, False)
            return None

        # Call the selected heuristic
        decision = None
        if action in self._portfolio:
            decision = self._portfolio[action].decide_placement(box, bin_state)

        # If selected heuristic fails, try fallback
        if decision is None and action != self._fallback_idx:
            if self._fallback_idx in self._portfolio:
                fallback_name = self._config_hh.heuristic_names[self._fallback_idx]
                decision = self._portfolio[self._fallback_idx].decide_placement(
                    box, bin_state,
                )
                if decision is not None:
                    heuristic_name = f"{heuristic_name}->fallback({fallback_name})"

        # Log and track
        success = decision is not None
        self._log_selection(box, heuristic_name, action, q_values, decision)
        self._tracker.record_choice(action, success)

        return decision

    def _select_heuristic(
        self,
        box: Box,
        bin_states: List[BinState],
        bin_config,
        grippable: List[Box],
        buffer_view: List[Box],
    ) -> tuple:
        """
        Use the Q-agent to select a heuristic action.

        Returns:
            Tuple of (action_index, q_values_array_or_None).
        """
        if self._agent is None:
            # Fallback mode: always use walle_scoring (best default)
            return self._fallback_idx, None

        # Extract state features
        state = extract_state_features(
            box=box,
            bin_states=bin_states,
            bin_config=bin_config,
            grippable=grippable,
            buffer_view=buffer_view,
            tracker=self._tracker,
            config=self._config_hh,
        )

        if self._mode == "tabular":
            state_idx = discretise_state(state, self._config_hh)
            action = self._agent.select_action(state_idx, epsilon=0.0)
            q_values = self._agent.get_q_values(state_idx)
            return action, q_values

        elif self._mode == "dqn":
            import torch
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0)
                device = next(self._agent.parameters()).device
                state_t = state_t.to(device)
                q_values = self._agent(state_t).cpu().numpy().flatten()
            action = int(np.argmax(q_values))
            return action, q_values

        else:
            return self._fallback_idx, None

    def _log_selection(
        self,
        box: Box,
        heuristic_name: str,
        action: int,
        q_values,
        decision: Optional[PlacementDecision],
    ) -> None:
        """Log a heuristic selection for interpretability analysis."""
        entry = {
            "box_id": box.id,
            "box_vol": box.volume,
            "heuristic": heuristic_name,
            "action": action,
            "success": decision is not None,
        }
        if q_values is not None:
            entry["q_values"] = q_values.tolist() if hasattr(q_values, 'tolist') else list(q_values)
        self._selection_log.append(entry)

    # ── Analysis Helpers ──────────────────────────────────────────────────

    @property
    def selection_log(self) -> List[Dict[str, Any]]:
        """Access the full selection log for post-hoc analysis."""
        return self._selection_log

    def get_selection_summary(self) -> Dict[str, Any]:
        """Compute summary statistics of heuristic selections."""
        if not self._selection_log:
            return {}

        counts: Dict[str, int] = {}
        successes: Dict[str, int] = {}

        for entry in self._selection_log:
            name = entry["heuristic"]
            counts[name] = counts.get(name, 0) + 1
            if entry["success"]:
                successes[name] = successes.get(name, 0) + 1

        total = len(self._selection_log)
        summary = {
            "total_steps": total,
            "heuristic_distribution": {
                k: {"count": v, "fraction": v / total}
                for k, v in sorted(counts.items(), key=lambda x: -x[1])
            },
            "success_rates": {
                k: successes.get(k, 0) / v
                for k, v in counts.items()
            },
            "overall_success_rate": (
                sum(successes.values()) / total if total > 0 else 0.0
            ),
        }
        return summary
