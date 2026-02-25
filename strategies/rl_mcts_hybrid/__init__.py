"""
rl_mcts_hybrid -- MCTS-Guided Hierarchical Actor-Critic with World Model.

The ULTIMATE 6th RL strategy for the Botko BV dual-pallet online packing problem.

Key innovations (addressing ALL 5 critical gaps):
  1. JOINT item-selection + placement via hierarchical policy decomposition
  2. TRUE multi-bin coordination via global state encoding + bin attention
  3. MCTS LOOKAHEAD via learned world model (picking box_i reveals box_{i+4})
  4. CONVEYOR-AWARE reasoning with learned transition model
  5. TRAPPED VOID detection via auxiliary loss on wasted space

Architecture:
  - World Model: predicts next state given action (conveyor transition model)
  - High-Level Policy: selects (which_box, which_bin) from grippable window
  - Low-Level Policy: selects (x, y, orient) using Transformer pointer over candidates
  - MCTS planner: uses world model for N-step lookahead at inference time
  - Heuristic ensemble: leverages top-5 heuristics as candidate generators + warm-start

Paper contribution: first approach to jointly optimise item selection, bin assignment,
and placement position with MCTS lookahead for online multi-bin 3D packing under
conveyor constraints.
"""

from strategies.rl_mcts_hybrid.strategy import RLMCTSHybridStrategy

__all__ = ["RLMCTSHybridStrategy"]
