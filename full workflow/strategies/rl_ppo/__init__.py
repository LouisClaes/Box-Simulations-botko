"""
rl_ppo — Proximal Policy Optimization strategy for 3D online bin packing.

WHAT THIS STRATEGY DOES
-----------------------
This strategy uses PPO (Proximal Policy Optimization) to learn a policy
that places incoming boxes onto EUR pallets.  Unlike DQN which scores every
candidate individually, PPO learns a probability distribution over actions
and directly optimises the expected cumulative reward.  The network uses
a cross-attention mechanism to focus on whichever bin and position is most
promising for the current box, enabling better coordination across bins.

HOW IT WORKS
------------
- **Decomposed action space**: Instead of choosing among ~38,400 (bin, x, y,
  orient) combinations all at once, the policy decomposes the decision into
  four sequential sub-choices: bin index, then x position, then y position,
  then orientation.  This reduces the effective action space from 38,400 to
  2+120+80+2=204 logits, making training tractable (Zhao et al. 2022).
- **Cross-attention context**: The network computes a context vector by
  attending the current box embedding against the buffer (visible boxes),
  allowing the policy to look ahead and reason about future boxes when
  deciding where to place the current one.
- **Parallel environments**: 16 environments run simultaneously during
  training, collecting 256 steps each before a PPO update.  This gives
  4,096 samples per update, greatly stabilising gradient estimates.
- **GAE advantage estimation**: Generalised Advantage Estimation (lambda=0.95)
  reduces the variance of policy gradient estimates without introducing
  too much bias, improving sample efficiency.
- **Clip ratio**: PPO clips the policy ratio at 0.2, preventing large
  destructive updates to the network.  This makes training much more
  robust than vanilla policy gradient methods.

PAPER BASIS
-----------
- Schulman et al. (2017): PPO — the core algorithm (clip ratio, entropy
  bonus, value loss coefficient).
- Zhao et al. (ICLR 2022): Decomposed action space for 3D bin packing,
  reducing from O(bins*x*y*orient) to a sequential factorisation.
- Xiong et al. (RA-L 2024, GOPT): Masked PPO with attention for bin
  packing — inspiration for the cross-attention context module.
- Andrychowicz et al. (2021): PPO hyperparameter recommendations for
  continuous and discrete control.

QUICK USAGE EXAMPLE
-------------------
After training::

    from strategies import get_strategy
    strategy = get_strategy("rl_ppo")
    # Requires a checkpoint at outputs/rl_ppo/logs/best_model.pt
    # Use with PackingSession, benchmark_all.py, run_experiment.py, etc.

To train from scratch::

    python strategies/rl_ppo/train.py --total_timesteps 5000000

To evaluate against heuristic baselines::

    python strategies/rl_ppo/evaluate.py --checkpoint outputs/rl_ppo/logs/best_model.pt

KEY HYPERPARAMETERS
-------------------
- ``learning_rate = 3e-4``   — Adam LR with cosine decay schedule
- ``clip_ratio = 0.2``       — PPO clip epsilon (standard value from paper)
- ``gae_lambda = 0.95``      — Advantage estimation smoothing (0=TD, 1=Monte Carlo)
- ``entropy_coeff = 0.01``   — Exploration bonus (increase if agent gets stuck)
- ``num_envs = 16``          — Parallel environments for data collection
- ``rollout_steps = 256``    — Steps per env per update (total batch = 16*256 = 4096)

EXPECTED PERFORMANCE
--------------------
- Training time: ~12-24 hours on CPU, ~3-6 hours with GPU (5M timesteps)
- Fill rate after training: ~62-70% avg closed fill depending on reward shaping
- Inference speed: ~10-30ms per box (decomposed sampling is fast)

NETWORK ARCHITECTURE SUMMARY
-----------------------------
Input: 2x heightmap (1x120x80 each) + current box (5-dim) + buffer boxes (8x5)

    HeightmapCNN (shared):  (1, 120, 80) -> Conv2d x3 -> AvgPool(4,4) -> Dense -> 256-dim
    Box MLP:                (5,) -> Dense(128) -> 128-dim
    Buffer MLP + pool:      (8, 5) -> Dense(64) -> mean-pool -> 64-dim
    Cross-attention:        query=box(128), keys=buffer(64) -> context(128)
    Concat:                 context(128) + box(128) + bins(2x256) = 768-dim
    Actor head:             768 -> Dense(256) -> [bin(2), x(120), y(80), orient(2)]
    Critic head:            768 -> Dense(256) -> Dense(1) -> V(s)

Provides:
    RLPPOStrategy        Inference-time strategy (registered as "rl_ppo")
    ActorCritic          Actor-Critic network with cross-attention
    PPOConfig            All hyperparameters as a dataclass
"""

from strategies.rl_ppo.strategy import RLPPOStrategy

__all__ = ["RLPPOStrategy"]
