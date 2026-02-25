"""
Monte Carlo Tree Search planner for the MCTS Hybrid strategy.

Uses the learned WorldModel to simulate future states without calling the
real simulator, enabling N-step lookahead within the time budget.

Key insight for conveyor-aware MCTS:
  When the agent picks box_i from grippable position j, the box at position
  j+4 (or the next from the stream) becomes grippable. The world model
  predicts this transition, allowing the MCTS planner to reason about
  "if I pick this box now, what boxes will I have access to next?"

Algorithm (adapted from MuZero for online packing):
  1. ROOT: encode current observation -> root state embedding
  2. SELECTION: traverse tree using PUCT (Q + c * P * sqrt(N_parent) / (1 + N_child))
  3. EXPANSION: use world model to predict next state + reward
  4. BACKPROPAGATION: propagate value estimates up the tree
  5. ACTION: select action with highest visit count (or highest value for greedy)

Differences from standard MCTS:
  - LEARNED dynamics (world model) instead of true simulator
  - Two-level actions: (high_level, low_level) per tree node
  - Value estimates from BOTH high-level and low-level value heads
  - Bounded depth (mcts_depth=4, matching the pick window)
  - Batched world model rollouts for GPU efficiency

References:
  - Fang et al. (Jan 2026): MCTS + MPC for online 3D BPP
  - Schrittwieser et al. (2020): MuZero (learned model + MCTS)
  - Silver et al. (2016): AlphaGo (PUCT selection)
"""

from __future__ import annotations

import math
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MCTSNode:
    """
    A node in the MCTS tree.

    Stores visit statistics, value estimates, and prior probabilities
    from the policy network. Children are lazily expanded on first visit.
    """

    # State representation (from world model)
    state_embed: Optional[torch.Tensor] = None    # (global_state_dim,)

    # Action that led to this node
    hl_action: int = -1         # High-level action index
    ll_action: int = -1         # Low-level candidate index

    # Statistics
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0          # P(a|s) from the policy network
    reward: float = 0.0         # Immediate reward from world model

    # Tree structure
    children: Dict[Tuple[int, int], "MCTSNode"] = field(default_factory=dict)
    is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Mean action value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTSPlanner:
    """
    MCTS planner using the learned world model for lookahead.

    Performs N simulations from the current state, building a search tree
    where transitions are predicted by the world model. Returns the
    action with the highest visit count (most robust to noise).

    The planner handles the two-level action structure:
      1. High-level: which (box, bin) to pick
      2. Low-level: which candidate placement to use

    For efficiency, the planner pre-computes all high-level actions and
    their top-K low-level candidates, then searches over this reduced space.
    """

    def __init__(
        self,
        num_simulations: int = 50,
        max_depth: int = 4,
        c_puct: float = 1.5,
        discount: float = 0.99,
        temperature: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.discount = discount
        self.temperature = temperature
        self.device = device

    def search(
        self,
        model,       # MCTSHybridNet
        global_state: torch.Tensor,     # (1, global_state_dim)
        hl_probs: torch.Tensor,         # (1, num_hl_actions) prior from policy
        hl_mask: torch.Tensor,          # (1, num_hl_actions) valid actions
        ll_probs_per_hl: Dict[int, torch.Tensor],  # {hl_action: (1, max_candidates)}
        ll_masks_per_hl: Dict[int, torch.Tensor],  # {hl_action: (1, max_candidates)}
        candidate_features_per_hl: Dict[int, torch.Tensor],  # {hl_action: (1, N, 16)}
    ) -> Tuple[int, int, Dict[str, float]]:
        """
        Run MCTS search from the current state.

        Args:
            model:                    The full MCTSHybridNet (for world model + value).
            global_state:             Current global state embedding.
            hl_probs:                 High-level action prior probabilities.
            hl_mask:                  Valid high-level actions.
            ll_probs_per_hl:          Low-level priors per HL action.
            ll_masks_per_hl:          Valid LL actions per HL action.
            candidate_features_per_hl: Candidate features per HL action.

        Returns:
            (best_hl_action, best_ll_action, search_stats)
        """
        # Create root node
        root = MCTSNode(state_embed=global_state.squeeze(0))

        # Pre-compute valid HL actions
        valid_hl = torch.where(hl_mask.squeeze(0) > 0.5)[0].tolist()
        if not valid_hl:
            return 0, 0, {"simulations": 0, "max_depth_reached": 0}

        hl_prior = hl_probs.squeeze(0).detach().cpu().numpy()

        # Run simulations
        max_depth_reached = 0
        for sim in range(self.num_simulations):
            node = root
            search_path: List[MCTSNode] = [root]
            depth = 0

            # Selection: traverse tree using PUCT
            while node.is_expanded and depth < self.max_depth:
                best_child, best_action = self._select_child(node)
                if best_child is None:
                    break
                node = best_child
                search_path.append(node)
                depth += 1

            max_depth_reached = max(max_depth_reached, depth)

            # Expansion: expand this node if not at depth limit
            if not node.is_expanded and depth < self.max_depth:
                self._expand_node(
                    node, model, valid_hl, hl_prior,
                    ll_probs_per_hl, ll_masks_per_hl,
                    candidate_features_per_hl,
                )

            # Evaluation: use value network for leaf value
            if node.state_embed is not None:
                with torch.no_grad():
                    state = node.state_embed.unsqueeze(0).to(self.device)
                    # Use high-level value as leaf estimate
                    trunk_out = model.high_level.trunk(state)
                    leaf_value = model.high_level.value_head(trunk_out).item()
            else:
                leaf_value = 0.0

            # Backpropagation
            self._backpropagate(search_path, leaf_value)

        # Select best action: highest visit count at root
        best_hl, best_ll = self._select_action(root)

        stats = {
            "simulations": self.num_simulations,
            "max_depth_reached": max_depth_reached,
            "root_value": root.q_value,
            "root_visits": root.visit_count,
        }

        return best_hl, best_ll, stats

    def _select_child(self, node: MCTSNode) -> Tuple[Optional[MCTSNode], Tuple[int, int]]:
        """Select best child using PUCT formula."""
        best_score = -float("inf")
        best_child = None
        best_action = (0, 0)

        sqrt_parent = math.sqrt(node.visit_count)

        for action_key, child in node.children.items():
            # PUCT score: Q(s,a) + c * P(a|s) * sqrt(N_parent) / (1 + N_child)
            q = child.q_value
            prior = child.prior
            exploration = self.c_puct * prior * sqrt_parent / (1 + child.visit_count)
            score = q + exploration

            if score > best_score:
                best_score = score
                best_child = child
                best_action = action_key

        return best_child, best_action

    def _expand_node(
        self,
        node: MCTSNode,
        model,
        valid_hl: List[int],
        hl_prior: np.ndarray,
        ll_probs_per_hl: Dict[int, torch.Tensor],
        ll_masks_per_hl: Dict[int, torch.Tensor],
        candidate_features_per_hl: Dict[int, torch.Tensor],
    ) -> None:
        """Expand a node by creating children for valid action pairs."""
        node.is_expanded = True

        for hl_a in valid_hl:
            # Get top-K LL actions for this HL action
            if hl_a not in ll_probs_per_hl:
                continue

            ll_probs = ll_probs_per_hl[hl_a].squeeze(0).detach().cpu().numpy()
            ll_mask = ll_masks_per_hl[hl_a].squeeze(0).detach().cpu().numpy()

            valid_ll = np.where(ll_mask > 0.5)[0]
            if len(valid_ll) == 0:
                continue

            # Take top-5 LL actions to keep tree manageable
            ll_scores = ll_probs[valid_ll]
            top_k = min(5, len(valid_ll))
            top_indices = valid_ll[np.argsort(ll_scores)[-top_k:]]

            for ll_a in top_indices:
                action_key = (hl_a, int(ll_a))
                # Joint prior: P(hl, ll) = P(hl) * P(ll | hl)
                joint_prior = float(hl_prior[hl_a]) * float(ll_probs[ll_a])

                # Use world model to predict next state
                if node.state_embed is not None:
                    with torch.no_grad():
                        state = node.state_embed.unsqueeze(0).to(self.device)
                        hl_t = torch.tensor([hl_a], device=self.device)
                        ll_t = torch.tensor([ll_a], device=self.device)
                        wm_out = model.world_model(state, hl_t, ll_t)
                        # Use the world model's learned latent projection
                        # to produce a full global_state_dim embedding
                        # (avoids zero-padding degradation at depth > 1)
                        next_state = wm_out.latent_state

                        reward = wm_out.reward_pred.item()
                else:
                    next_state = None
                    reward = 0.0

                child = MCTSNode(
                    state_embed=next_state.squeeze(0) if next_state is not None else None,
                    hl_action=hl_a,
                    ll_action=int(ll_a),
                    prior=joint_prior,
                    reward=reward,
                )
                node.children[action_key] = child

    def _backpropagate(
        self, search_path: List[MCTSNode], leaf_value: float,
    ) -> None:
        """Backpropagate value estimates up the search path."""
        value = leaf_value
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = node.reward + self.discount * value

    def _select_action(self, root: MCTSNode) -> Tuple[int, int]:
        """Select action from root based on visit counts."""
        if not root.children:
            return 0, 0

        if self.temperature < 0.01:
            # Greedy: highest visit count
            best_action = max(root.children.keys(),
                              key=lambda k: root.children[k].visit_count)
        else:
            # Proportional to visit count ^ (1/temperature)
            actions = list(root.children.keys())
            visits = np.array([root.children[k].visit_count for k in actions], dtype=np.float64)
            visits = visits ** (1.0 / self.temperature)
            total = visits.sum()
            if total > 0:
                probs = visits / total
                idx = np.random.choice(len(actions), p=probs)
                best_action = actions[idx]
            else:
                best_action = actions[0]

        return best_action[0], best_action[1]
