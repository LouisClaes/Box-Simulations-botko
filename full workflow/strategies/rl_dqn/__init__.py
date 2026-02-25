"""
rl_dqn — Double DQN reinforcement learning strategy for 3D bin packing.

WHAT THIS STRATEGY DOES
-----------------------
This strategy trains a deep neural network to decide WHERE to place each
incoming box on a EUR pallet.  Given the current heights of all occupied
cells (the "heightmap") and the dimensions of the box to be placed, the
network outputs a Q-value score for every candidate placement position.
The highest-scoring valid placement is chosen.  After sufficient training
(~50,000 episodes) the network learns to stack boxes densely and stably
without any hand-coded rules.

HOW IT WORKS
------------
- **Double DQN**: Two copies of the network are maintained — an "online"
  network that is updated every step, and a "target" network that is
  updated every 500 steps.  Using the target network to compute the
  training target prevents the Q-values from diverging (van Hasselt 2016).
- **Dueling architecture**: The network separately estimates how good the
  current state is (V) and how much better each action is than average
  (A).  This makes learning faster when many actions give the same result.
- **Prioritised Experience Replay (PER)**: Transitions where the network
  was most wrong are replayed more frequently, so the agent learns faster
  from its mistakes (Schaul 2016).
- **N-step returns**: Instead of bootstrapping after one step, the agent
  accumulates rewards over 3 steps before bootstrapping.  This propagates
  reward signals faster through time.
- **Candidate generation**: Rather than evaluating all ~38,400 grid
  positions, the agent only scores a reduced set of ~200 smart candidates
  (extreme points, EMS corners, grid fallback) to keep inference fast.

PAPER BASIS
-----------
- Tsang et al. (2025, SIMPAC-2024-311): DDQN for dual-bin packing —
  direct inspiration for the three-branch network and Botko BV setup.
- Mnih et al. (2015, Nature): Original Deep Q-Network algorithm and
  experience replay buffer design.
- van Hasselt et al. (2016): Double DQN to reduce Q-value overestimation.
- Wang et al. (2016): Dueling network architectures.
- Schaul et al. (2016): Prioritised experience replay.

QUICK USAGE EXAMPLE
-------------------
After training::

    from strategies import get_strategy
    strategy = get_strategy("rl_dqn")
    # Requires a checkpoint at outputs/rl_dqn/best_model.pt
    # Use with PackingSession, benchmark_all.py, run_experiment.py, etc.

To train from scratch::

    python strategies/rl_dqn/train.py --episodes 50000 --lr 0.001

To evaluate against heuristic baselines::

    python strategies/rl_dqn/evaluate.py --checkpoint outputs/rl_dqn/best_model.pt

KEY HYPERPARAMETERS
-------------------
- ``lr = 0.001``              — Adam learning rate (try 0.0005 for more stable training)
- ``gamma = 0.95``            — Discount factor (lower than 0.99 because episodes are short)
- ``eps_start / eps_end = 1.0 / 0.05`` — Epsilon for exploration, decays over 80% of episodes
- ``buffer_alpha = 0.6``      — PER priority exponent (0 = uniform, 1 = fully prioritised)
- ``n_step = 3``              — N-step return horizon (speeds up reward propagation)
- ``max_candidates = 200``    — Candidate placements evaluated per step (speed/quality tradeoff)

EXPECTED PERFORMANCE
--------------------
- Training time: ~8-16 hours on CPU, ~2-4 hours with GPU (50,000 episodes)
- Fill rate after training: ~60-68% avg closed fill (vs baseline 64.8%)
- Inference speed: ~50-200ms per box depending on candidate count

NETWORK ARCHITECTURE SUMMARY
-----------------------------
Input: 2-channel heightmap (120x80) + box feature vector (20-dim) + action features (7-dim)

    CNN branch:    (2, 120, 80) -> Conv2d x4 -> GlobalAvgPool -> 256-dim
    Box MLP:       (20,) -> Dense(128) -> Dense(128) -> 128-dim
    Action MLP:    (7,)  -> Dense(64)  -> Dense(64)  -> 64-dim
    Merge + Duel:  concat(256+128) -> V(s);  64-dim -> A(s,a)
    Output:        Q(s,a) = V(s) + A(s,a) - mean(A)  -> single Q-value per candidate

Components:
    RLDQNStrategy           — BaseStrategy subclass for inference (registered as "rl_dqn")
    DQNNetwork              — PyTorch three-branch dueling network
    DQNConfig               — All hyperparameters as a dataclass
    CandidateGenerator      — Smart action space reduction (corner + EP + EMS + grid)
    PrioritisedReplayBuffer — PER with n-step returns for training
    train                   — Full training script (python -m strategies.rl_dqn.train)
    evaluate                — Evaluation and comparison script
"""

from strategies.rl_dqn.strategy import RLDQNStrategy

__all__ = ["RLDQNStrategy"]
