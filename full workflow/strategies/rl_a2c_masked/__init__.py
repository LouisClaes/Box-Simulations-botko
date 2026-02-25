"""
rl_a2c_masked — A2C with Learned Feasibility Masking for 3D online bin packing.

WHAT THIS STRATEGY DOES
-----------------------
This strategy implements Advantage Actor-Critic (A2C) enhanced with a
learned feasibility mask predictor, directly following the approach of
Zhao et al. (AAAI 2021).  The agent learns to pack boxes onto EUR pallets
by optimising a policy over a coarse 50mm action grid.  A second neural
network head simultaneously learns to predict WHICH grid positions are
physically valid (i.e., the box fits without floating or overlapping),
allowing infeasible actions to be suppressed before sampling — speeding
up training and improving stability compared to naive masking.

HOW IT WORKS
------------
- **Shared CNN encoder**: The heightmap of each bin is processed by a 5-layer
  CNN (one set of weights shared across bins) producing a 512-dim embedding
  per bin.  Both bin embeddings are concatenated with the current box's
  feature vector to form a 1152-dim combined state.
- **Three output heads**: The same combined feature feeds three heads:
  (1) Actor — softmax over the 1,536-action coarse grid giving pi(a|s);
  (2) Critic — scalar value estimate V(s) for advantage computation;
  (3) Mask Predictor — sigmoid over 1,536 actions giving P(valid|s,a).
- **Mask application**: During the forward pass, the predicted mask is
  applied as ``logits + (1 - mask) * (-1e9)`` before softmax, making the
  policy assign near-zero probability to infeasible positions.
- **5-component loss**: Policy gradient + value regression + mask BCE +
  infeasibility penalty + entropy bonus.  Each component has its own
  weight coefficient for ablation studies.
- **Curriculum learning**: Training progresses through 3 phases: easy
  (30 large boxes), medium (60 mixed boxes), and hard (100 small boxes),
  preventing the agent from collapsing on a degenerate policy early on.

PAPER BASIS
-----------
- Zhao et al. (AAAI 2021): "Learning to Pack: A Data-Driven Tree Search
  Algorithm for Large-Scale 3D Bin Packing Problem" — the direct source
  of the feasibility masking idea, 4-channel input, and 5-component loss.
- Wu et al. (2017): Scalable A2C — the base RL algorithm used for
  synchronous multi-environment training.
- Mnih et al. (2016): A3C — original asynchronous advantage actor-critic,
  from which A2C is derived.

QUICK USAGE EXAMPLE
-------------------
After training::

    from strategies import get_strategy
    strategy = get_strategy("rl_a2c_masked")
    # Requires a checkpoint at outputs/rl_a2c_masked/logs/best_model.pt
    # Use with PackingSession, benchmark_all.py, run_experiment.py, etc.

To train from scratch (uses curriculum, 16 parallel envs)::

    python strategies/rl_a2c_masked/train.py --num_updates 200000 --num_envs 16

To evaluate::

    python strategies/rl_a2c_masked/evaluate.py --checkpoint best_model.pt

KEY HYPERPARAMETERS
-------------------
- ``action_grid_step = 50.0``   — Coarse grid resolution (mm); 50mm -> 24x16 = 384 positions
- ``learning_rate = 1e-4``      — Adam LR (lower than PPO due to on-policy instability)
- ``lambda_mask = 0.5``         — BCE weight for mask predictor supervision
- ``gae_lambda = 0.95``         — GAE advantage smoothing
- ``use_curriculum = True``     — Enable 3-phase progressive difficulty schedule
- ``num_envs = 16``             — Parallel A2C environments

EXPECTED PERFORMANCE
--------------------
- Training time: ~6-12 hours on CPU, ~1-3 hours with GPU (200,000 updates)
- Fill rate after training: ~58-65% avg closed fill (mask helps avoid wasted tries)
- Inference speed: ~5-15ms per box (coarse grid = small action space)

NETWORK ARCHITECTURE SUMMARY
-----------------------------
Input: 4-channel heightmap per bin (height + item_l + item_w + item_h) at 120x80

    Shared CNN (per bin):  (4, 120, 80) -> Conv2d x5 -> AvgPool(4,4) -> Dense(512) -> 512-dim
    Bin concat:            bin0(512) + bin1(512) = 1024-dim
    Item MLP:              (5,) -> Dense(64) -> Dense(128) -> 128-dim
    Combined:              concat(1024 + 128) = 1152-dim
    Actor head:            1152 -> Dense(256) -> Dense(1536) -> softmax -> pi(a|s)
    Critic head:           1152 -> Dense(256) -> Dense(1) -> V(s)
    Mask head:             1152 -> Dense(256) -> Dense(1536) -> sigmoid -> P(valid|s,a)

Provides:
    RLA2CMaskedStrategy    Inference-time strategy (registered as "rl_a2c_masked")
    A2CMaskedNetwork       Actor-Critic network with mask predictor head
    A2CMaskedConfig        All hyperparameters as a dataclass
    A2CMaskedLoss          5-component loss function
"""

from strategies.rl_a2c_masked.strategy import RLA2CMaskedStrategy
from strategies.rl_a2c_masked.config import A2CMaskedConfig

__all__ = ["RLA2CMaskedStrategy", "A2CMaskedConfig"]
