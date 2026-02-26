# Reinforcement Learning Strategies for 3D Online Bin Packing

## Overview

This package contains 6 RL-based strategies for the Botko BV dual-pallet online bin packing problem, plus shared infrastructure for training, evaluation, and HPC deployment.

### Problem Setup

```
┌─────────────────────────────────────────────────────────────┐
│  CONVEYOR BELT (8 boxes visible, 4 grippable)               │
│                                                              │
│  stream → [B8][B7][B6][B5] | [B4][B3][B2][B1] → robot      │
│            visible only      ←  grippable  →                │
│                                                              │
│  PALLETS (2 × EUR 1200×800mm, close at 1800mm height)       │
│                                                              │
│  ┌──────────┐  ┌──────────┐                                  │
│  │ Pallet 0 │  │ Pallet 1 │                                  │
│  │ 1200×800 │  │ 1200×800 │                                  │
│  │ h≤2700   │  │ h≤2700   │                                  │
│  └──────────┘  └──────────┘                                  │
│  Close at 1800mm → snapshot → replace with fresh pallet      │
│                                                              │
│  Objective: maximise avg fill rate of CLOSED pallets         │
└─────────────────────────────────────────────────────────────┘
```

## Strategies

| # | Strategy | Algorithm | Paper Basis | Novelty | Training Time |
|---|----------|-----------|-------------|---------|---------------|
| 1 | `rl_dqn` | Double DQN + PER | Tsang 2025 + Zhao 2021 | Candidate generation | ~12h GPU |
| 2 | `rl_ppo` | PPO + Decomposed Actions | GOPT 2024 + Zhao 2023 | Multi-head policy | ~16h GPU |
| 3 | `rl_a2c_masked` | A2C + Feasibility Mask | Zhao 2021 (AAAI) | Learned mask predictor | ~16h GPU |
| 4 | `rl_hybrid_hh` | Q-learning Hyper-Heuristic | **Novel** | Meta-learning over heuristics | ~4h CPU |
| 5 | `rl_pct_transformer` | PPO + Transformer | Zhao 2022 (ICLR) | Standard Transformer for PCT | ~16h GPU |
| 6 | `rl_mcts_hybrid` | Hierarchical PPO + MCTS | PCT + MuZero-inspired planning | Lookahead-aware RL policy | ~24h GPU |

### Strategy Comparison

```
                        Training Speed
                             ▲
                             │
         rl_hybrid_hh ●      │
         (minutes)           │
                             │
                             │      ● rl_dqn
                             │        (hours)
                             │
         rl_a2c_masked ●─────┼────● rl_ppo
                             │
                             │  ● rl_pct_transformer
                             │
                             └──────────────────────▶ Fill Rate Quality
                         Simple                  Complex
```

## Architecture Comparison

### 1. DDQN (`rl_dqn`)
- **State**: 2-channel heightmap (120×80) + box features
- **Action**: Select from ~100 candidate placements (corner-aligned)
- **Network**: CNN (heightmap) + MLP (features) → Q-value per candidate
- **Training**: Off-policy, experience replay, target network
- **Strength**: Stable, well-understood, good sample efficiency

### 2. PPO Decomposed (`rl_ppo`)
- **State**: Per-bin CNN features + attention over items
- **Action**: Decomposed: bin → x → y → orient (autoregressive)
- **Network**: CNN + cross-attention + 4 policy heads + value head
- **Training**: On-policy, GAE, vectorized environments
- **Strength**: Scalable, handles large action spaces efficiently

### 3. A2C + Feasibility Mask (`rl_a2c_masked`)
- **State**: 4-channel heightmap (height + item dims)
- **Action**: Grid position × orientation × bin (1536 actions)
- **Network**: Shared CNN → actor + critic + mask predictor (3 heads)
- **Training**: On-policy, curriculum learning, mask supervision
- **Strength**: Constraint handling, stable placements

### 4. Hybrid Hyper-Heuristic (`rl_hybrid_hh`)
- **State**: 39 handcrafted features (bin stats, box info, history)
- **Action**: Select from 7 heuristics + skip (8 actions)
- **Network**: Small MLP (39 → 128 → 128 → 8) or tabular Q-table
- **Training**: Standard DQN or tabular Q-learning
- **Strength**: Fast training, interpretable, leverages existing heuristics

### 5. PCT Transformer (`rl_pct_transformer`)
- **State**: Set of candidate placements (variable size, 30-100)
- **Action**: Select best candidate via pointer attention
- **Network**: Transformer encoder + pointer decoder
- **Training**: PPO, variable action space, candidate generation
- **Strength**: Handles variable actions naturally, high quality

### 6. MCTS Hybrid (`rl_mcts_hybrid`)
- **State**: Shared encoder over bins + conveyor/buffer context
- **Action**: High-level box/bin decision + low-level candidate pointer
- **Network**: Hierarchical actor-critic + world-model auxiliary heads
- **Training**: Curriculum PPO, imitation warm-start, robust resume/checkpointing
- **Strength**: Planning-aware policy with optional MCTS lookahead

## File Structure

```
strategies/
├── rl_common/                  ← Shared infrastructure
│   ├── __init__.py
│   ├── environment.py          ← Gymnasium env (BinPackingEnv)
│   ├── rewards.py              ← Configurable reward shaping
│   ├── obs_utils.py            ← Observation encoding utilities
│   ├── logger.py               ← Training logger (CSV + TB + plots)
│   ├── compare_strategies.py   ← Cross-strategy comparison plots
│   ├── README.md               ← This file
│   └── hpc/
│       ├── requirements.txt    ← Python dependencies
│       ├── setup_hpc.sh        ← One-time HPC setup
│       ├── run_rl_pipeline.py  ← Unified train/eval/visualize orchestrator
│       ├── train_all.sh        ← One-command launcher wrapper
│       ├── evaluate_all.sh     ← Evaluate + visualize existing run
│       └── README.md           ← HPC guide
│
├── rl_dqn/                     ← Strategy 1: Double DQN
│   ├── __init__.py
│   ├── network.py              ← CNN + MLP Q-network
│   ├── replay_buffer.py        ← Prioritised experience replay
│   ├── candidate_generator.py  ← Smart action space reduction
│   ├── train.py                ← Training script
│   ├── evaluate.py             ← Evaluation script
│   ├── strategy.py             ← Inference strategy (BaseStrategy)
│   ├── config.py               ← Hyperparameters
│   └── README.md
│
├── rl_ppo/                     ← Strategy 2: PPO Decomposed
│   ├── __init__.py
│   ├── network.py              ← Actor-Critic with attention
│   ├── train.py                ← Vectorized PPO training
│   ├── evaluate.py
│   ├── strategy.py
│   ├── config.py
│   └── README.md
│
├── rl_a2c_masked/              ← Strategy 3: A2C + Mask
│   ├── __init__.py
│   ├── network.py              ← 3-head network (actor+critic+mask)
│   ├── train.py                ← Curriculum learning
│   ├── evaluate.py
│   ├── strategy.py
│   ├── config.py
│   └── README.md
│
├── rl_hybrid_hh/               ← Strategy 4: Hyper-Heuristic (NOVEL)
│   ├── __init__.py
│   ├── network.py              ← Small Q-network + tabular
│   ├── state_features.py       ← Feature engineering
│   ├── train.py                ← Tabular + DQN training
│   ├── evaluate.py
│   ├── strategy.py
│   ├── config.py
│   └── README.md
│
├── rl_pct_transformer/         ← Strategy 5: PCT Transformer
    ├── __init__.py
    ├── network.py              ← Transformer encoder-decoder
    ├── candidate_generator.py  ← Placement candidate generation
    ├── train.py                ← PPO with variable actions
    ├── evaluate.py
    ├── strategy.py
    ├── config.py
    └── README.md

└── rl_mcts_hybrid/             ← Strategy 6: Hierarchical PPO + MCTS
    ├── config.py
    ├── network.py
    ├── mcts.py
    ├── candidate_generator.py
    ├── void_detector.py
    ├── train.py
    ├── evaluate.py
    └── DOCUMENTATION.md
```

## Usage After Training

All strategies register with the existing framework:

```python
from strategies import get_strategy
from simulator.session import PackingSession, SessionConfig

# Use any trained RL strategy like any other strategy
strategy = get_strategy("rl_dqn")  # or rl_ppo, rl_a2c_masked, etc.
session = PackingSession(config)
result = session.run(boxes, strategy)
print(f"Fill rate: {result.avg_closed_fill:.1%}")
```

## References

1. Zhao et al., "Learning Efficient Online 3D Bin Packing on Packing Configuration Trees", ICLR 2022
2. Zhao et al., "Online 3D Bin Packing with Constrained Deep Reinforcement Learning", AAAI 2021
3. Tsang et al., "Deep Reinforcement Learning for the 3D Dual-Bin Packing Problem", SIMPAC 2025
4. Xiong et al., "GOPT: Generalizable Online 3D Bin Packing via Transformer-based DRL", RA-L 2024
5. Verma et al., "PackMan: A Packing Manager for 3D Bin Packing with DRL", AAAI 2020
6. Zhao et al., "Learning Practically Feasible Policies for Online 3D BPP", Science China 2023
7. Ali et al., "A heuristic framework for 3D online bin packing", 2025
