"""
rl_hybrid_hh — RL Hybrid Hyper-Heuristic for 3D bin packing.

NOVEL THESIS CONTRIBUTION: A Q-learning-based selective hyper-heuristic that
learns WHEN to use which expert placement heuristic, based on the current
packing state.  Instead of training an RL agent to make low-level placement
decisions (38,400+ position-orientation combinations), the agent selects from
a small portfolio of 7 proven heuristics plus a SKIP action.

WHAT THIS STRATEGY DOES
-----------------------
Rather than learning to pick an (x, y, bin, orient) position from scratch,
this strategy learns to choose WHICH existing heuristic algorithm to call for
each box.  The portfolio includes seven complementary strategies
(baseline/DBLF, walle_scoring, surface_contact, extreme_points, skyline,
layer_building, best_fit_decreasing) plus an optional SKIP action.  An RL
agent observes a 39-dimensional state vector describing bin fill, box size,
surface roughness, and recent history, then selects the most appropriate
heuristic for the current packing context.

HOW IT WORKS
------------
- **Tiny action space**: Choosing from 8 heuristics instead of 38,400
  positions means the agent can converge in 1-10 hours on CPU rather than
  days, and requires far fewer episodes to learn a good policy.
- **Handcrafted state features**: A 39-dimensional feature vector captures
  bin fill rates, max heights, surface roughness, box dimensions, buffer
  fullness, episode progress, heuristic selection history, and packing phase.
  These human-understandable features enable tabular Q-learning as well as
  neural Q-learning.
- **Two learning modes**: (1) Tabular Q-learning builds a discrete Q-table
  over ~5,625 discretised states — trains in minutes, zero GPU needed, and
  comes with convergence guarantees.  (2) DQN mode uses a small MLP
  (39->128->128->64->8) for better generalisation to unseen states.
- **Interpretability**: Because actions are named heuristics, it is trivial
  to plot a heatmap of which heuristic was chosen as a function of fill rate
  and box size — a unique advantage over position-level RL methods.
- **Compositionality**: The approach is modular: swapping in a new heuristic
  requires only adding it to the portfolio list in HHConfig.

PAPER BASIS (NOVEL — no direct prior work)
------------------------------------------
- Burke et al. (2013): Hyper-heuristic survey — the concept of a high-level
  selector choosing among low-level heuristics.
- Zhao et al. (AAAI 2021, ICLR 2022): Position-level RL for bin packing —
  shows the limitation of large discrete action spaces that this work solves.
- Xiong et al. (RA-L 2024): Masked PPO with EMS — confirms that smart action
  space reduction is key to practical bin-packing RL.
- Mnih et al. (2015): DQN — underlying algorithm for the DQN selector mode.

QUICK USAGE EXAMPLE
-------------------
After training::

    from strategies import get_strategy
    strategy = get_strategy("rl_hybrid_hh")
    # Requires a checkpoint at outputs/rl_hybrid_hh/best_model.pt
    # Use with PackingSession, benchmark_all.py, run_experiment.py, etc.

To train (tabular mode, fast baseline, ~1 hour)::

    python strategies/rl_hybrid_hh/train.py --mode tabular --episodes 10000

To train (DQN mode, better generalisation, ~4-8 hours)::

    python strategies/rl_hybrid_hh/train.py --mode dqn --episodes 50000

To evaluate with interpretability analysis::

    python strategies/rl_hybrid_hh/evaluate.py --checkpoint outputs/rl_hybrid_hh/best_model.pt

KEY HYPERPARAMETERS
-------------------
- ``heuristic_names``         — Portfolio of 7 heuristics (add/remove to customise)
- ``lr = 0.001``              — Learning rate (alpha for tabular, Adam LR for DQN)
- ``gamma = 0.99``            — Discount factor
- ``eps_start / eps_end = 1.0 / 0.05`` — Epsilon-greedy exploration schedule
- ``hidden_dims = (128, 128, 64)``  — DQN MLP layer sizes (deliberately small)
- ``reward_failure_penalty = -0.5`` — Penalty when chosen heuristic fails to place

EXPECTED PERFORMANCE
--------------------
- Training time: ~1 hour (tabular), ~4-8 hours (DQN), on CPU
- Fill rate after training: ~63-67% avg closed fill (comparable to best single heuristic)
- Inference speed: <5ms per box (heuristic call + tiny network forward pass)
- Best heuristic for comparison: walle_scoring at 68.3%, surface_contact at 67.4%

NETWORK ARCHITECTURE SUMMARY
-----------------------------
Input: 39-dimensional handcrafted state vector (no CNN — state is not spatial)

    DQN mode:     (39,) -> Dense(128) -> Dropout(0.1) -> Dense(128) -> Dense(64) -> Q(s, 8 actions)
    Tabular mode: discretise(39-dim) -> Q-table[state_idx, action_idx] (~45,000 entries)

Modules:
    config.py          — HHConfig dataclass with all hyperparameters
    state_features.py  — Handcrafted feature extraction (~39 dimensions)
    network.py         — Q-network (DQN) and tabular Q-learning implementations
    strategy.py        — RLHybridHHStrategy (BaseStrategy, registered as "rl_hybrid_hh")
    train.py           — Training script (tabular Q-learning or DQN)
    evaluate.py        — Evaluation and interpretability analysis
"""

from strategies.rl_hybrid_hh.strategy import RLHybridHHStrategy

__all__ = ["RLHybridHHStrategy"]
