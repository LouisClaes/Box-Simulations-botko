# RL-PPO Strategy for 3D Online Bin Packing

Proximal Policy Optimization (PPO) with decomposed action space and
attention-based actor-critic for the Botko BV dual-pallet packing problem.

## How This Strategy Works (Beginner Guide)

This section explains the strategy from the ground up, with no machine learning
background assumed.

### What is a "policy"?

In reinforcement learning, a **policy** is the robot's complete set of
decision-making rules. It is a function that takes in a description of the
current situation (the state of the pallets, which box is next, what is visible
on the conveyor belt) and outputs a decision (where to place the box).

Think of it like a recipe: given these ingredients (current pallet state), do
these steps (place the box here, in this orientation). A good policy is one that
consistently makes decisions that lead to well-packed pallets.

PPO trains the policy by repeatedly:
1. Letting the policy make decisions and collecting the results
2. Evaluating whether those decisions turned out to be good or bad
3. Gently nudging the policy in the direction of better decisions
4. Repeating until the policy is expert-level

The "Proximal" part of PPO means the nudges are kept small and safe. PPO clips
(limits) how much the policy is allowed to change in a single update step. This
prevents a catastrophic situation where one bad batch of training data causes
the policy to "forget" everything it learned and collapse to random behaviour.

### What is "decomposed"?

When placing a box, there are four decisions to make simultaneously:
- Which bin (pallet) does the box go on? (2 choices)
- What x-position along the pallet? (120 choices)
- What y-position across the pallet? (80 choices)
- What orientation — rotate 90 degrees or not? (2 choices)

If you tried to make all four decisions at once, there would be
2 x 120 x 80 x 2 = **38,400 possible combined actions**, one for every
combination. That is an enormous space for a neural network to learn over.

**Decomposed** means we split this one large decision into **four smaller
sequential decisions**, each much easier to learn:

1. First, pick the bin (2 options)
2. Then, given the chosen bin, pick the x-position (120 options)
3. Then, given the bin and x, pick the y-position (80 options)
4. Finally, given bin, x, and y, pick the orientation (2 options)

Each stage conditions on all previous decisions, so the network still considers
the full context. But instead of 38,400 actions, the network only ever chooses
from at most 120 options at a time. This is a **188x reduction** in action space
size, which dramatically speeds up learning.

In code, these four choices are called "sub-policies" or "heads". Each is a
small output layer (softmax) attached to the shared network trunk.

### What is "cross-attention"?

**Attention** is a mechanism that lets a neural network look at multiple pieces
of information and decide which ones are most relevant right now.

Imagine you are deciding which pallet to place the next box on. You need to
look at both pallets simultaneously and compare them: which one has more room?
Which one is more full? Which one has a surface shape that fits this box well?

**Cross-attention** is the specific type of attention used here. It works like
this:

- The **query** is the current box (what we are trying to place) — it asks
  "which pallet suits me?"
- The **keys and values** are the two pallet embeddings (compact numerical
  summaries of each pallet's state)
- The attention mechanism computes how well the box "matches" each pallet,
  producing a weighted combination that focuses more on the most relevant pallet

The output is a **context vector**: a 128-dimensional summary of "which pallet
looks best for this box and why". This context is then used by all four
sub-policy heads to make their decisions.

With 4 attention heads (each looking at the pallet states from a slightly
different perspective simultaneously), the network captures multiple aspects of
pallet suitability — height, roughness, available space, and so on — all at
once.

### Why use 16 parallel environments?

Training requires the policy to experience thousands of packing episodes. If we
ran one episode at a time, training would be slow because:
- Episodes are sequential (each box placement depends on the last)
- Modern hardware (CPUs, GPUs) is designed to process many things in parallel

Running **16 parallel environments** (like running 16 independent packing
simulations simultaneously) means:
- 16x more experience is collected in the same wall-clock time
- The experience comes from 16 slightly different random box sequences, which
  gives the policy a more diverse view of the problem
- GPU and CPU cores are kept busier, so hardware is used more efficiently

After each rollout (a short burst of 256 steps per environment), all 16
environments pause, the collected data is pooled into one large batch, and the
network is updated. The environments then continue from where they paused.

On a machine with more cores, you can increase the number of environments
proportionally (the README suggests num_envs = num_CPU_cores as a rule of thumb).

### Step-by-step: what happens during one training update

1. **Rollout collection**: all 16 environments each run 256 steps. At every step,
   the current policy decides where to place a box, the simulator executes the
   placement, and the (state, action, reward, next state, action probabilities)
   tuple is recorded. This produces a batch of 16 x 256 = 4,096 experiences.

2. **Advantage estimation (GAE)**: for each experience, the network computes an
   "advantage" — how much better or worse the actual reward was compared to what
   the critic (value head) predicted. Positive advantage = this decision turned
   out better than expected. Negative advantage = it turned out worse.

3. **Mini-batch updates**: the 4,096 experiences are shuffled and split into 8
   mini-batches of 512. For each mini-batch and for 4 full passes (epochs)
   through the data:
   - The actor (policy heads) is updated to make the advantageous actions more
     probable, but ONLY up to the PPO clip limit (20% change maximum)
   - The critic (value head) is updated to better predict future rewards
   - A small entropy bonus encourages the policy to remain exploratory (not
     become overconfident too quickly)

4. **Discard rollout data**: unlike DQN which stores experiences in a replay
   buffer, PPO discards each rollout after the update. The next rollout is
   collected with the freshly updated policy.

### Step-by-step: what happens during inference (placing one box)

1. **Observe**: read the current heightmaps of both pallets (as a 2-channel image
   with resolution 120x80) and the features of the box queue.

2. **Encode**: the shared CNN processes each heightmap into a compact 256-dim
   summary. The box features are encoded into a 128-dim embedding.

3. **Cross-attend**: the cross-attention layer looks at both pallet summaries
   from the perspective of the current box, producing a 128-dim context vector.

4. **Combine**: the CNN outputs, box embedding, and context are concatenated into
   a single 768-dim state vector.

5. **Decomposed selection**: the four sub-policy heads each apply their softmax
   to their respective output dimensions, producing probability distributions
   over bins, x-positions, y-positions, and orientations.

6. **Greedy selection**: at inference, we take the highest-probability option at
   each step (argmax rather than sampling). The four choices are assembled into a
   full (bin, x, y, orientation) placement decision.

7. **Place**: the box is placed at the chosen position.

The entire forward pass takes approximately 5-10 milliseconds.

---

## Algorithm Overview

PPO (Schulman et al. 2017) is an on-policy actor-critic algorithm that
maximises a clipped surrogate objective:

    L^CLIP = E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]

where r_t = pi_new / pi_old is the probability ratio and A_t is the GAE
advantage estimate.

### Why PPO for bin packing?

- **Stable training**: clipped objective prevents catastrophic policy updates
- **Sample efficient** (for on-policy): GAE + mini-batch updates
- **Compatible with action masking**: invalid placements are masked before
  sampling, so the policy never proposes physically impossible actions
- **Scalable**: vectorized environments enable parallel rollout collection

## Decomposed Action Space

The naive action space for this problem is enormous:

    |A| = num_bins x grid_l x grid_w x num_orientations = 2 x 120 x 80 x 2 = 38,400

Learning a policy over 38,400 discrete actions is impractical.

Following Zhao et al. (ICLR 2022, AAAI 2021), we decompose the action
into an autoregressive sequence of sub-actions:

    pi(b, x, y, o | s) = pi_bin(b|s) * pi_x(x|s,b) * pi_y(y|s,b,x) * pi_o(o|s,b,x,y)

Each sub-policy is a small softmax head conditioned on the shared features
plus embeddings of previous choices.  Total output dimensions:

    |A_decomposed| = 2 + 120 + 80 + 2 = 204

This is a 188x reduction in action space dimensionality.

## Architecture

```
  Heightmaps (2 x 120 x 80)
  ┌──────────────────────────────┐
  │ Shared CNN per bin           │
  │ Conv(1,32,5,s2) -> BN -> ReLU│
  │ Conv(32,64,3,s2) -> BN -> ReLU│
  │ Conv(64,128,3,s1) -> BN -> ReLU│
  │ AdaptiveAvgPool(4,4) -> FC  │
  └──────────┬───────────────────┘
             │ 2 x 256 = 512
             v
  ┌──────────────────────────────┐
  │ Cross-Attention (4 heads)    │
  │ Q: box_embed(128)+buf(64)   │
  │ K,V: bin_embeds(2x256)      │
  │ Output: context(128)        │
  └──────────┬───────────────────┘
             │
             v
  concat(context[128], box_embed[128], all_bins[512]) = 768
             │
     ┌───────┴───────┐
     v               v
  Actor Head     Critic Head
  ┌─────────┐    ┌─────────┐
  │ trunk   │    │ FC(768)  │
  │FC->ReLU │    │ -> 256   │
  │-> 128   │    │ -> ReLU  │
  │         │    │ -> 1     │
  │ bin(2)  │    └─────────┘
  │ x(120)  │       V(s)
  │ y(80)   │
  │ o(2)    │
  └─────────┘
```

Total parameters: approximately 1.8M

## Training

### Quick start

```bash
cd "python/full workflow"
python strategies/rl_ppo/train.py
```

### Full options

```bash
python strategies/rl_ppo/train.py \
    --total_timesteps 5000000 \
    --num_envs 16 \
    --lr 3e-4 \
    --rollout_steps 256 \
    --ppo_epochs 4 \
    --entropy_coeff 0.01 \
    --seed 42 \
    --log_dir outputs/rl_ppo/logs
```

### HPC setup

For a 32-core node:

```bash
python strategies/rl_ppo/train.py \
    --total_timesteps 20000000 \
    --num_envs 32 \
    --rollout_steps 512 \
    --ppo_epochs 10 \
    --num_minibatches 16
```

Rule of thumb: `num_envs = num_CPU_cores`.  For GPU training, a single
GPU handles the model forward/backward passes while CPUs handle environment
stepping.

### Monitoring

```bash
tensorboard --logdir outputs/rl_ppo/logs/tensorboard
```

Training curves are also saved as PNG in `outputs/rl_ppo/logs/plots/`.

## Evaluation

### Single evaluation

```bash
python strategies/rl_ppo/evaluate.py \
    --checkpoint outputs/rl_ppo/logs/checkpoints/best_model.pt \
    --num_episodes 100
```

### Compare with heuristics

```bash
python strategies/rl_ppo/evaluate.py \
    --checkpoint outputs/rl_ppo/logs/checkpoints/best_model.pt \
    --num_episodes 100 \
    --compare baseline walle_scoring surface_contact gopt_heuristic
```

### Compare with all strategies

```bash
python strategies/rl_ppo/evaluate.py \
    --checkpoint outputs/rl_ppo/logs/checkpoints/best_model.pt \
    --compare_all
```

## Integration with Framework

The trained model is registered as `"rl_ppo"` in the strategy registry:

```python
from strategies.rl_ppo.strategy import set_checkpoint_path, RLPPOStrategy
from strategies.base_strategy import get_strategy

# Set checkpoint path
set_checkpoint_path("outputs/rl_ppo/logs/checkpoints/best_model.pt")

# Use like any other strategy
strategy = get_strategy("rl_ppo")

# With PackingSession
from simulator.session import PackingSession, SessionConfig
session = PackingSession(config)
result = session.run(boxes, strategy)
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 5,000,000 | Total env steps |
| `num_envs` | 16 | Parallel environments |
| `rollout_steps` | 256 | Steps per env per rollout |
| `ppo_epochs` | 4 | Optimisation epochs per rollout |
| `num_minibatches` | 8 | Mini-batches per epoch |
| `learning_rate` | 3e-4 | Initial LR (cosine schedule) |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_ratio` | 0.2 | PPO clip epsilon |
| `entropy_coeff` | 0.01 | Entropy bonus |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `value_loss_coeff` | 0.5 | Value loss weight |

## File Structure

```
strategies/rl_ppo/
├── __init__.py     -- Package imports
├── config.py       -- PPOConfig dataclass (all hyperparameters)
├── network.py      -- ActorCritic network (CNN + attention + decomposed heads)
├── train.py        -- Full PPO training loop with vectorized envs
├── evaluate.py     -- Evaluation and comparison with heuristics
├── strategy.py     -- BaseStrategy inference wrapper (registered as "rl_ppo")
└── README.md       -- This file
```

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
   Proximal Policy Optimization Algorithms. arXiv:1707.06347.

2. Zhao, H., She, Q., Zhu, C., Yang, Y., & Xu, K. (2022).
   Online 3D Bin Packing with Constrained Deep Reinforcement Learning.
   AAAI 2021 / ICLR 2022.

3. Xiong, J., Zhu, Y., Lu, J., Feng, Z., Chen, J., Wang, W., & Tan, Y. (2024).
   GOPT: Generalizable Online 3D Bin Packing via Transformer-based Deep
   Reinforcement Learning. IEEE RA-L.

4. Tsang, J., et al. (2025). Dual-bin DDQN for multi-pallet packing.
   SIMPAC-2024-311.

5. Andrychowicz, M., et al. (2021). What Matters In On-Policy Reinforcement
   Learning? A Large-Scale Empirical Study. ICLR 2021.
