# A2C with Feasibility Masking for 3D Online Bin Packing

## Beginner Guide: How This Strategy Works

If you are new to reinforcement learning or bin packing, this section explains the key ideas in plain language before the technical details.

### What is "feasibility masking"?

Imagine you are playing chess and someone has already crossed out every illegal move on a piece of paper before you decide. You never even consider those moves. That is feasibility masking.

In bin packing there are 1536 possible positions to place a box (different x/y coordinates and orientations across 2 pallets). Most of those positions are illegal at any given moment -- the box would hang in the air, fall off the edge, or overlap a box that is already there. Feasibility masking crosses out all the illegal positions before the neural network picks one. The result is that the agent only ever chooses from moves that are actually valid.

### Why learn the mask instead of computing it exactly?

Computing the exact list of valid positions takes time: for each of the 1536 candidate positions the computer has to simulate whether the box would float, overlap, or fall off. That is slow enough to be a problem in real deployments.

This strategy trains a second small neural network (the "mask predictor") to guess which positions are valid from the current bin picture. One forward pass through that network gives an approximate valid/invalid answer for all 1536 positions at once -- much faster than checking each one individually. Think of it as hiring an experienced worker who can glance at the pallet and instantly say "those corners are fine, those spots are definitely blocked" without measuring everything with a ruler.

### What are the "3 heads"?

The network has one shared "body" (the CNN that reads the bin picture) and three separate output "heads" -- three outputs that come out of the same brain:

1. **Actor head** -- decides the probability of choosing each position. This is the head that actually picks where to put the box.
2. **Critic head** -- estimates how good the current pallet state is overall. This is used only during training to tell the actor whether its choices are paying off.
3. **Mask predictor head** -- predicts which of the 1536 positions are valid right now. Its output is used to zero out illegal positions before the actor chooses.

All three heads share the expensive CNN computation, so running all three costs barely more than running one.

### What is curriculum learning?

Think of how a teacher designs a course: week 1 has easy problems, week 8 has hard ones. You would not hand a beginner calculus on day one.

This strategy does the same thing for training:
- **Phase 1 (first 30% of training)**: only 30 boxes, all large (200-500mm). Fewer boxes means fewer decisions, and large boxes leave fewer valid positions so the agent gets clearer feedback.
- **Phase 2 (next 40%)**: 60 boxes, medium variety (150-550mm). Moderate difficulty.
- **Phase 3 (final 30%)**: 100 boxes, full range (100-600mm). The real problem.

Without curriculum learning the agent starts with the full complexity and the reward signal is too noisy to learn from. Starting easy gives a strong foundation.

### Step-by-step: what happens when a box is placed

1. A new box arrives on the conveyor and its dimensions are read (length, width, height, weight).
2. The CNN scans the current heightmap image of both pallets and produces a 512-dimensional summary of each pallet (1024 total).
3. A small dense network encodes the box dimensions into a 128-dimensional vector.
4. All three encodings are concatenated into a single 1152-dimensional vector and fed into all three heads simultaneously.
5. The mask predictor head outputs a probability between 0 and 1 for each of the 1536 positions, indicating how likely it is to be valid.
6. Any position with a low mask score gets its logit set to -1,000,000,000 (effectively impossible to pick).
7. The actor head applies softmax to the remaining logits and samples a position.
8. The box is placed at that position, the pallet heightmap is updated, and the agent receives a reward based on how much fill improved.

---

## Overview

This strategy implements **Advantage Actor-Critic (A2C)** with a **learned feasibility mask predictor** for the online 3D bin packing problem, following Zhao et al. (AAAI 2021). It is designed for the Botko BV thesis setup: 2 EUR pallets (1200x800mm), height cap 2700mm, close at 1800mm, conveyor with 8 visible boxes and pick window of 4.

The key innovation is replacing expensive exact validity computation with a neural network that **predicts** which actions are feasible in O(1) time. This mask predictor is trained supervised alongside the RL agent, creating a CMDP (Constrained Markov Decision Process) formulation that is superior to naive reward shaping for handling invalid actions.

## Why Feasibility Masking? (vs Reward Penalty)

In 3D bin packing with discrete action spaces, many actions are invalid (box extends beyond pallet, floats in air, overlaps existing boxes). There are three approaches to handle this:

### Approach 1: Reward Penalty (Naive)
- Apply a negative reward whenever the agent selects an invalid action
- **Problem**: The agent wastes training samples exploring invalid regions of the action space. With 1536 actions and often >90% invalid, convergence is extremely slow
- **Problem**: The penalty magnitude is a sensitive hyperparameter -- too small and the agent ignores it, too large and it becomes overly conservative

### Approach 2: Exact Masking (Expensive)
- Compute a binary validity mask for every action at every step
- For each of the 1536 actions: check bounds, compute resting height, verify support
- **Problem**: O(num_actions * grid_checks) per step, which is computationally prohibitive for real-time deployment

### Approach 3: Learned Feasibility Masking (This Strategy)
- Train a neural network to **predict** P(valid | state, action) for all actions simultaneously
- Single forward pass through the mask predictor head: O(1) amortised
- Trained with BCE loss against ground-truth masks computed during training
- At inference, the predicted mask replaces the expensive exact computation
- **Advantage**: Combines the sample efficiency of exact masking with the speed of no masking
- **Advantage**: The CMDP formulation (Zhao et al.) provides formal guarantees that the policy converges to the feasible optimum

## Architecture

```
Input Encoding
==============

Per bin:                     Current box:
+-------------------+       +-------------------+
| 4-channel tensor  |       | 5-dim vector      |
| (120 x 80)       |       | (l, w, h, vol, wt)|
|                   |       | normalised [0,1]  |
| ch0: height/H_max|       +--------+----------+
| ch1: item_l/L    |                |
| ch2: item_w/W    |          Dense(5, 64) + ReLU
| ch3: item_h/H    |          Dense(64, 128)
+--------+----------+                |
         |                     128-dim item embed
    Shared CNN                       |
    (weight-shared                   |
     across bins)                    |
         |                           |
    512-dim embed                    |
         |                           |
+--------+--------+                  |
| Bin 0  |  Bin 1 |                  |
| 512    |  512   |                  |
+--------+--------+                  |
         |                           |
    Concatenate: 1024-dim            |
         |                           |
         +----------+----------------+
                    |
              1152-dim combined
              /     |     \
         Actor   Critic   Mask Predictor
           |       |           |
      Dense(256) Dense(256) Dense(256)
        ReLU      ReLU       ReLU
      Dense(1536) Dense(1)  Dense(1536)
           |       |           |
       softmax   V(s)      sigmoid
       pi(a|s)              M(a|s)

Mask Application
================
masked_logits = logits + (1 - M) * (-1e9)
pi = softmax(masked_logits)
```

### CNN Encoder Details (per bin)

| Layer | Output Channels | Kernel | Stride | Padding | Output Shape |
|-------|----------------|--------|--------|---------|-------------|
| Conv2d | 32 | 3x3 | 1 | 1 | 32 x 120 x 80 |
| Conv2d | 64 | 3x3 | 1 | 1 | 64 x 120 x 80 |
| Conv2d | 64 | 3x3 | 2 | 1 | 64 x 60 x 40 |
| Conv2d | 128 | 3x3 | 2 | 1 | 128 x 30 x 20 |
| Conv2d | 128 | 3x3 | 2 | 1 | 128 x 15 x 10 |
| AdaptiveAvgPool2d | 128 | - | - | - | 128 x 4 x 4 |
| Flatten + Dense | 512 | - | - | - | 512 |

### Action Space

Coarse grid discretisation (step=50mm):
- X positions: 1200 / 50 = 24
- Y positions: 800 / 50 = 16
- Orientations: 2 (flat only)
- Bins: 2

**Total: 24 x 16 x 2 x 2 = 1536 discrete actions**

## Loss Function

Five-component loss from Zhao et al. (AAAI 2021):

```
L = alpha * L_actor + beta * L_critic + lambda * L_mask
    + omega * E_inf - psi * E_entropy
```

| Component | Formula | Weight | Purpose |
|-----------|---------|--------|---------|
| L_actor | -log(pi(a\|s)) * A(s,a) | alpha=1.0 | Policy gradient |
| L_critic | (V(s) - R_target)^2 | beta=0.5 | Value regression |
| L_mask | BCE(M_pred, M_true) | lambda=0.5 | Mask supervision |
| E_inf | sum pi(a\|s) * log(M(a\|s)) | omega=0.01 | Infeasibility penalty |
| E_entropy | -sum pi(a\|s) * log(pi(a\|s)) | psi=0.01 | Exploration bonus |

**E_inf (infeasibility penalty)**: This is the CMDP constraint. When the mask predictor M(a|s) is close to 0 (infeasible), log(M) is very negative. If the policy pi(a|s) assigns probability to this action, the product is large and negative, creating a penalty. This drives the policy away from infeasible actions even before exact masking is applied.

**E_entropy**: Standard entropy bonus encourages exploration and prevents premature convergence to a deterministic policy.

## Curriculum Learning

Training uses three progressive difficulty phases:

| Phase | Updates | Boxes | Size Range | Purpose |
|-------|---------|-------|-----------|---------|
| 1 | 0-30% (0-60k) | 30 | 200-500mm | Learn basic placement |
| 2 | 30-70% (60k-140k) | 60 | 150-550mm | Handle variety |
| 3 | 70-100% (140k-200k) | 100 | 100-600mm | Full complexity |

Starting with fewer, larger boxes (easier episodes) gives the agent a clearer reward signal. Smaller action spaces (fewer valid placements with large boxes) make initial policy learning faster. The progressive increase to full complexity ensures the agent generalises to the real distribution.

## Training

```bash
# Default training (200k updates, 16 envs, CUDA if available)
python -m strategies.rl_a2c_masked.train

# Custom configuration
python -m strategies.rl_a2c_masked.train \
    --num_updates 200000 \
    --num_envs 16 \
    --lr 1e-4 \
    --rollout_steps 5 \
    --action_grid_step 50 \
    --device cuda \
    --log_dir outputs/rl_a2c_masked/run1

# Without curriculum learning
python -m strategies.rl_a2c_masked.train --no_curriculum

# Total timesteps: 200,000 updates x 16 envs x 5 steps = 16,000,000
```

### Training Schedule

| Parameter | Value |
|-----------|-------|
| Updates | 200,000 |
| Parallel envs | 16 |
| Rollout length | 5 steps |
| Batch size | 80 (16 x 5) |
| Total timesteps | 16,000,000 |
| Learning rate | 1e-4 (linear decay) |
| GAE lambda | 0.95 |
| Discount (gamma) | 0.99 |
| Gradient clip | 0.5 |
| Checkpoint interval | every 5000 updates |
| Evaluation interval | every 1000 updates |

## Evaluation

```bash
# Standard evaluation (50 episodes, deterministic)
python -m strategies.rl_a2c_masked.evaluate \
    --checkpoint outputs/rl_a2c_masked/logs/checkpoints/best_model.pt

# Stochastic evaluation
python -m strategies.rl_a2c_masked.evaluate \
    --checkpoint best_model.pt \
    --stochastic \
    --num_episodes 100

# Mask mode ablation study
python -m strategies.rl_a2c_masked.evaluate \
    --checkpoint best_model.pt \
    --ablation
```

### Ablation: Mask Modes

The evaluation script supports comparing three mask modes:

1. **predicted_mask**: Network's learned mask predictor (fast, approximate)
2. **ground_truth_mask**: Exact validity from environment (slow, exact)

This quantifies the quality-speed tradeoff of the learned mask predictor.

## CMDP vs Reward Shaping Comparison

| Aspect | Reward Shaping | CMDP (This Strategy) |
|--------|---------------|---------------------|
| Invalid action handling | Negative reward | Mask zeros out logits |
| Sample efficiency | Low (explores invalid space) | High (only valid actions) |
| Hyperparameter sensitivity | High (penalty magnitude) | Low (mask is binary) |
| Convergence guarantee | No formal guarantee | Feasible optimum (Zhao et al.) |
| Computational cost | Low per step | Moderate (mask network) |
| Training signal for mask | N/A | BCE supervision |

The CMDP formulation treats action validity as a **hard constraint** rather than a soft penalty. The mask predictor learns to approximate this constraint, and the E_inf loss term ensures the policy respects it even when the mask prediction is imperfect. This two-pronged approach (masking + penalty) provides both efficiency and robustness.

## File Structure

```
strategies/rl_a2c_masked/
    __init__.py          Registration and public API
    config.py            All hyperparameters (A2CMaskedConfig)
    network.py           A2C network with mask predictor
    train.py             Full training loop with curriculum
    evaluate.py          Evaluation and ablation scripts
    strategy.py          BaseStrategy wrapper for inference
    README.md            This documentation
```

## Integration with Framework

The strategy integrates with the existing packing framework:

```python
# As a BaseStrategy (single-bin inference)
from strategies.rl_a2c_masked import RLA2CMaskedStrategy

strategy = RLA2CMaskedStrategy(
    checkpoint_path="outputs/rl_a2c_masked/logs/checkpoints/best_model.pt"
)

# Works with PipelineSimulator, PackingSession, benchmark_all.py, etc.
```

It is automatically registered in `STRATEGY_REGISTRY` as `"rl_a2c_masked"` when the module is imported.

## References

1. **Zhao et al. (AAAI 2021)**: "Online 3D Bin Packing with Constrained Deep Reinforcement Learning" -- Feasibility masking, CMDP formulation, learned mask predictor
2. **Wu et al. (2017)**: "Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation" -- A2C/ACKTR baseline
3. **Mnih et al. (2016)**: "Asynchronous methods for deep reinforcement learning" -- A3C, parallel advantage actor-critic
4. **Schulman et al. (2016)**: "High-Dimensional Continuous Control Using Generalized Advantage Estimation" -- GAE for variance reduction
5. **Saxe et al. (2014)**: "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" -- Orthogonal initialisation
