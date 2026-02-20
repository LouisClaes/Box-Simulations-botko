# External References & Open-Source Repositories

## Overview

This document catalogs all open-source repositories, benchmark datasets, and academic papers
relevant to this project. Each entry includes proper attribution, what can be reused,
and how it relates to our strategies.

---

## 1. GitHub Repositories

### 1.1 Online-3D-BPP-DRL (Zhao et al. — AAAI 2021)

| Field | Value |
|-------|-------|
| **URL** | https://github.com/alexfrom0815/Online-3D-BPP-DRL |
| **Stars** | 635 |
| **Paper** | "Online 3D Bin Packing with Constrained Deep Reinforcement Learning" (AAAI 2021) |
| **Authors** | Hang Zhao, Qijin She, Chenyang Zhu, Yin Yang, Kai Xu |
| **License** | Academic use only |
| **Python** | 3.7 |
| **Framework** | PyTorch |

**What it implements:**
- ACKTR (Actor-Critic with Kronecker-Factored Trust Region)
- Feasibility mask prediction (CNN → binary placement mask)
- MCTS comparison baseline
- Multi-bin variant (`multi_bin/`)
- CNN state encoding with adjustable architecture (`acktr/model.py`)

**Key modules:**
```
acktr/model.py     — CNN architecture for state encoding
envs/bpp0/         — 3D bin packing environment
main.py            — Train/test CLI
evaluation.py      — Metrics computation
MCTS/              — Tree search baseline
multi_bin/          — Multi-container variant
pretrained_models/  — Pre-trained weights included
```

**What we can use:**
- Feasibility mask generation logic → adapt for our `stability/` module
- MCTS implementation → reference for our buffer search
- CNN state encoder architecture → foundation for RL strategies
- Environment interface patterns → compare with our BinState API

**Training:** `python main.py --mode train --use-cuda --item-seq rs` (~24h)
**Testing:** `python main.py --mode test --load-model --use-cuda --data-name cut_2.pt`

**Reference in our code:**
```
Zhao, H., She, Q., Zhu, C., Yang, Y., & Xu, K. (2021).
Online 3D Bin Packing with Constrained Deep Reinforcement Learning.
Proceedings of the AAAI Conference on Artificial Intelligence, 35(1).
```

---

### 1.2 Online-3D-BPP-PCT (Zhao et al. — ICLR 2022)

| Field | Value |
|-------|-------|
| **URL** | https://github.com/alexfrom0815/Online-3D-BPP-PCT |
| **Stars** | 1,100+ |
| **Paper** | "Learning Efficient Online 3D Bin Packing on Packing Configuration Trees" (ICLR 2022) |
| **Authors** | Hang Zhao, Yang Yu, Kai Xu |
| **License** | MIT |
| **Python** | ≥3.7 (3.7 recommended) |
| **Framework** | PyTorch ≥1.7 (1.10 recommended) |

**What it implements:**
- PCT (Packing Configuration Tree) — hierarchical tree representation
- Graph Attention Network (GAT) encoder (`graph_encoder.py`)
- Pointer Network for action selection (`attention_model.py`)
- K-FAC optimization (ACKTR variant, `kfac.py`)
- EMS/EP/CP/EV expansion schemes for candidate generation
- Stability approximation algorithms
- Continuous and discrete domain support
- Heuristic baselines (LSAH, OnlineBPH)

**Key modules:**
```
attention_model.py  — Pointer Network (action selection)
graph_encoder.py    — GAT state encoder
model.py            — Actor-Critic combining attention + encoding
kfac.py             — K-FAC optimization
pct_envs/           — Environment implementations
heuristic.py        — Heuristic baselines
givenData.py        — Problem configuration
evaluation.py       — Testing pipeline
```

**Pre-trained models:** Available on Google Drive (bin 10×10×10, items 1-5)
**Benchmark data:** 3000 trajectories × 150 items, Google Drive download

**What we can use:**
- GAT + Pointer Network architecture → our PCT strategy
- EMS expansion scheme logic → enhance our `ems` strategy
- Heuristic baselines → compare with our implementations
- Stability approximation → reference for our stability module
- K-FAC optimizer → faster RL training convergence

**Training:** `python main.py` (default Setting 2)
**Evaluation:** `python evaluation.py --evaluate --load-model --model-path ...`

**Reference in our code:**
```
Zhao, H., Yu, Y., & Xu, K. (2022).
Learning Efficient Online 3D Bin Packing on Packing Configuration Trees.
International Conference on Learning Representations (ICLR).
```

---

### 1.3 GOPT (Xiong et al. — RA-L 2024)

| Field | Value |
|-------|-------|
| **URL** | https://github.com/Xiong5Heng/GOPT |
| **Stars** | — |
| **Paper** | "Generalizable Online 3D Bin Packing via Transformer-based Deep Reinforcement Learning" (RA-L 2024) |
| **Authors** | Xiong, Heng et al. |
| **License** | Academic use only |
| **Python** | 3.9 |
| **Framework** | PyTorch 2.1.0, CUDA 12.1, Tianshou RL framework |
| **arXiv** | 2409.05344 |

**What it implements:**
- Packing Transformer — fuses item + bin features via self-attention
- Placement Generator — creates candidate subspaces
- Masked A2C and PPO algorithms
- Generalization across different bin dimensions (key innovation)

**Key modules:**
```
model.py         — Transformer architecture
masked_a2c.py    — Actor-Critic with action masking
masked_ppo.py    — PPO with action masking
envs/            — Bin packing environments
ts_train.py      — Training script
ts_test.py       — Evaluation script
render.py        — Visualization
cfg/             — Configuration files
```

**What we can use:**
- Transformer architecture → novel state encoding for our RL strategies
- Action masking approach → combine with our feasibility mask
- Generalization across bin sizes → test our strategies on varied bins
- A2C/PPO implementations with masking → training infrastructure

**Training:** `python ts_train.py --config cfg/config.yaml --device 0`
**Testing:** `python ts_test.py --config cfg/config.yaml --device 0 --ckp /path/to/model.pth`

**Reference:**
```
Xiong, H. et al. (2024).
Generalizable Online 3D Bin Packing via Transformer-based Deep Reinforcement Learning.
IEEE Robotics and Automation Letters (RA-L). DOI: 10.1109/LRA.2024.3468161
```

---

### 1.4 DeepPack3D / SIMPAC-2024-311 (Tsang et al. — 2025)

| Field | Value |
|-------|-------|
| **URL** | https://github.com/SoftwareImpacts/SIMPAC-2024-311 |
| **Stars** | — |
| **Paper** | "A deep reinforcement learning approach for online and concurrent 3D bin packing" (2025) |
| **Authors** | Tsang et al. |
| **License** | MIT |
| **Python** | 3.10 |
| **Framework** | TensorFlow 2.10.0 |

**What it implements:**
- Double DQN with MCA (Maximal Cuboids Algorithm)
- 5 packing methods: RL, Best Lookahead, Best Area Fit, BSSF, BLSF
- Dual-bin support (the ONLY paper addressing all 5 thesis dimensions)
- ReplaceMax/ReplaceAll bin closing policies
- GPU-accelerated training

**Key modules:**
```
deeppack3d.py      — Main entry point
agent.py           — DDQN agent
env.py             — Bin packing environment
binpacker.py       — Packing logic
geometry.py        — Spatial calculations
SpacePartitioner.py — MCA space management
conveyor.py        — System simulation
```

**What we can use:**
- MCA (Maximal Cuboids Algorithm) → 3D space partitioning for our strategies
- DDQN architecture → reference for our RL training pipeline
- Bin replacement policies → multi-bin orchestration logic
- Heuristic baselines (BAF, BSSF, BLSF) → compare with our strategies

**CRITICAL:** This is the only paper with native dual-bin + buffer + online 3D support.
Port from TensorFlow to PyTorch needed for integration with Zhao repos.

**Reference:**
```
Tsang, Y.P. et al. (2025).
A deep reinforcement learning approach for online and concurrent 3D bin packing.
Software Impacts. DOI: SIMPAC-2024-311
```

---

### 1.5 IR-BPP (Zhao et al.)

| Field | Value |
|-------|-------|
| **URL** | https://github.com/alexfrom0815/IR-BPP |
| **Stars** | 335 |
| **Paper** | "Packing irregular objects with deep reinforcement learning" |
| **Authors** | Hang Zhao et al. |

Extends the Zhao framework to irregular (non-rectangular) objects.
Not directly relevant to our rectangular box setting but shares infrastructure.

### 1.6 Packing-Tools (Zhao et al.)

| Field | Value |
|-------|-------|
| **URL** | https://github.com/alexfrom0815/Packing-Tools |
| **Stars** | 47 |

Utility library with shape processing, rendering tools, and simulation scenarios.
Useful for visualization and environment setup.

---

## 2. Complete Zhao/Xu Paper Trajectory

The research group led by Kai Xu and Hang Zhao produced the most comprehensive
body of work on online 3D BPP with DRL:

| # | Year | Venue | Title | Repo | Key Innovation |
|---|------|-------|-------|------|----------------|
| 1 | 2021 | **AAAI** | Online 3D BPP with Constrained DRL | `Online-3D-BPP-DRL` | CMDP formulation, feasibility masks |
| 2 | 2022 | **ICLR** | Learning on Packing Configuration Trees | `Online-3D-BPP-PCT` | PCT tree, GAT+Pointer, EMS schemes |
| 3 | 2022 | **SCIS** | Learning Practically Feasible Policies | — | Decomposed Actor-Critic, 5-head action |
| 4 | 2025 | **IJRR** | Deliberate Planning of 3D BPP on PCT | — | ToP MCTS + PCT, buffer integration |

**Evolution:**
- AAAI 2021: Foundation — CNN + ACKTR with feasibility constraints
- ICLR 2022: Architecture — Hierarchical tree + attention mechanism
- SCIS 2022: Practicality — Autoregressive action decomposition for real robots
- IJRR 2025: Planning — MCTS tree search over item+placement decisions

---

## 3. Recent Work (2025-2026)

### 3.1 Fang et al. (2026) — MPC+MCTS Framework

| Field | Value |
|-------|-------|
| **arXiv** | [2601.02649](https://arxiv.org/abs/2601.02649) |
| **Title** | "Effective Online 3D Bin Packing with Lookahead Parcels Using MCTS" |
| **Authors** | Jiangyi Fang, Bowen Zhou, Haotian Wang, Xin Zhu, Leye Wang |
| **Affiliations** | Peking University, HIT, JD Logistics |
| **Date** | January 2026 |

**Key innovations:**
- Formulates online 3D-BP with lookahead as MPC (Model Predictive Control)
- Dynamic exploration prior: balances learned RL policy + robust random policy
- Auxiliary reward penalizing long-term spatial waste
- **10% gains under distribution shifts, 4% average improvement in online deployment**

**Relevance:** Their lookahead + MCTS approach directly relates to our `lookahead` and
buffer management strategies. The MPC formulation is novel and could enhance our
`hybrid_adaptive` strategy.

### 3.2 Gao et al. (2025) — LBCP Stability

| Field | Value |
|-------|-------|
| **arXiv** | [2507.09123](https://arxiv.org/abs/2507.09123) |
| **Title** | "Online 3D Bin Packing with Fast Stability Validation and Stable Rearrangement Planning" |

Implements Load-Bearable Convex Polygon (LBCP) for O(1) amortized stability checking.
Critical for our stability engine integration.

---

## 4. Benchmark Datasets

### 4.1 Ali et al. 2025 — 198 Real-World Instances

| Field | Value |
|-------|-------|
| **Paper** | [Static stability vs packing efficiency](https://www.sciencedirect.com/science/article/abs/pii/S0305054825000334) |
| **Journal** | Computers & Operations Research, Vol. 177, 2025, Article 107001 |
| **Authors** | S. Ali et al. |

160-heuristic framework with Pareto-optimal analysis under different stability constraints.
Dataset likely available as supplementary material on ScienceDirect.

### 4.2 Osaba et al. 2023 — Real-World BPP Benchmark

| Field | Value |
|-------|-------|
| **URL** | https://data.mendeley.com/datasets/y258s6d939/1 |
| **Instances** | 12 real-world instances (38-53 items each) |
| **License** | GPLv3 |
| **Includes** | Instance generator (`Q4RealBPP-DataGen`) for creating more |

Real-world constraints: dimensions, weights, category affinities, load balancing.

### 4.3 PCT Pre-generated Data

| Field | Value |
|-------|-------|
| **Source** | Google Drive via `Online-3D-BPP-PCT` repo |
| **Size** | 3000 trajectories × 150 items each |
| **Format** | PyTorch .pt tensors |

---

## 5. How Our Strategies Reference These Works

| Our Strategy | Based On | Reference |
|-------------|----------|-----------|
| `baseline` (DBLF) | Classic literature | Johnson (1974) |
| `walle_scoring` | Verma et al. 2020 | "A Generalized RL Algorithm for Online 3D BPP" (AAAI) |
| `extreme_points` | General EP literature | Crainic et al. (2008) |
| `ems` | Gonçalves & Resende 2013 | "A biased RKGA for 2D/3D BPP" |
| `skyline` | 2D skyline literature | Burke et al. (2004) |
| `layer_building` | Layer-based heuristics | Bortfeldt & Gehring (2001) |
| `best_fit_decreasing` | Classic BFD | Johnson (1974) |
| `surface_contact` | **NOVEL** (this project) | — |
| `gravity_balanced` | **NOVEL** (this project) | — |
| `column_fill` | **NOVEL** (this project) | — |
| `wall_building` | Practical palletizing | Industry practice |
| `lookahead` | Rollout heuristics | Bertsekas (2017), Fang et al. (2026) |
| `hybrid_adaptive` | **NOVEL** (this project) | Hyper-heuristic literature |

---

## 6. Cloning Repos for Local Use

```bash
# Clone all relevant repos into external_repos/
mkdir -p external_repos && cd external_repos

# 1. Zhao et al. — Constrained DRL (AAAI 2021)
git clone https://github.com/alexfrom0815/Online-3D-BPP-DRL.git

# 2. Zhao et al. — PCT (ICLR 2022) — MOST IMPORTANT
git clone https://github.com/alexfrom0815/Online-3D-BPP-PCT.git

# 3. GOPT — Transformer approach (RA-L 2024)
git clone https://github.com/Xiong5Heng/GOPT.git

# 4. Tsang et al. — DeepPack3D (2025) — ONLY dual-bin DRL
git clone https://github.com/SoftwareImpacts/SIMPAC-2024-311.git

# 5. Packing Tools (utilities)
git clone https://github.com/alexfrom0815/Packing-Tools.git
```
