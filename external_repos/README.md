# External Repositories — Reference Implementations

This directory contains shallow clones of key open-source repositories for
3D bin packing with deep reinforcement learning. These serve as:

1. **Reference implementations** — understand how state-of-the-art methods work
2. **Code reuse** — extract algorithms (EMS, stability, CNN encoders, etc.)
3. **Benchmark comparison** — evaluate our strategies against published results
4. **Training infrastructure** — ACKTR, K-FAC, PPO implementations

## Cloned Repositories

### 1. Online-3D-BPP-PCT (Zhao et al. — ICLR 2022)
- **Path:** `Online-3D-BPP-PCT/`
- **Paper:** "Learning Efficient Online 3D Bin Packing on Packing Configuration Trees"
- **Stars:** 1,100+ | **License:** MIT
- **Key:** GAT + Pointer Network, EMS expansion, K-FAC optimizer
- **Use for:** Neural architecture reference, heuristic baselines, stability approximation
- **Cite:** Zhao, H., Yu, Y., & Xu, K. (2022). ICLR.

### 2. Online-3D-BPP-DRL (Zhao et al. — AAAI 2021)
- **Path:** `Online-3D-BPP-DRL/`
- **Paper:** "Online 3D Bin Packing with Constrained Deep Reinforcement Learning"
- **Stars:** 635 | **License:** Academic only
- **Key:** ACKTR, feasibility masks, CMDP formulation, multi-bin variant
- **Use for:** Feasibility mask logic, constrained RL, pre-trained models
- **Cite:** Zhao, H., She, Q., Zhu, C., Yang, Y., & Xu, K. (2021). AAAI.

### 3. SIMPAC-2024-311 / DeepPack3D (Tsang et al. — 2025)
- **Path:** `SIMPAC-2024-311/`
- **Paper:** "A deep reinforcement learning approach for online and concurrent 3D bin packing"
- **License:** MIT
- **Key:** Double DQN, MCA (Maximal Cuboids Algorithm), dual-bin, bin replacement
- **Use for:** MCA space partitioning, bin replacement policies, DDQN training
- **CRITICAL:** Only paper with native dual-bin + buffer + online 3D support
- **Note:** TensorFlow 2.10 — needs PyTorch port for integration
- **Cite:** Tsang, Y.P. et al. (2025). Software Impacts.

### 4. GOPT (Xiong et al. — RA-L 2024)
- **Path:** `GOPT/`
- **Paper:** "Generalizable Online 3D Bin Packing via Transformer-based DRL"
- **License:** Academic only
- **Key:** Packing Transformer, masked A2C/PPO, generalization across bin sizes
- **Use for:** Transformer architecture, cross-bin generalization, action masking
- **Cite:** Xiong, H. et al. (2024). IEEE RA-L. arXiv:2409.05344

## Complete Zhao/Xu Paper Trajectory

| Year | Venue | Innovation | Repo |
|------|-------|-----------|------|
| 2021 | AAAI | CMDP + feasibility masks | `Online-3D-BPP-DRL` |
| 2022 | ICLR | PCT tree + GAT + Pointer | `Online-3D-BPP-PCT` |
| 2022 | SCIS | Decomposed Actor-Critic (5-head) | — |
| 2025 | IJRR | ToP MCTS + PCT planning | — |

## How to Update

```bash
cd external_repos/Online-3D-BPP-PCT && git pull
cd ../Online-3D-BPP-DRL && git pull
cd ../SIMPAC-2024-311 && git pull
cd ../GOPT && git pull
```

## Key Files to Study

### For Neural Architecture:
- `Online-3D-BPP-PCT/attention_model.py` — Pointer Network
- `Online-3D-BPP-PCT/graph_encoder.py` — GAT encoder
- `GOPT/model.py` — Packing Transformer

### For Stability:
- `Online-3D-BPP-PCT/pct_envs/PctContinuous0/convex_hull.py` — Stability check
- `Online-3D-BPP-DRL/envs/bpp0/` — Feasibility mask generation

### For Space Management:
- `Online-3D-BPP-PCT/pct_envs/PctContinuous0/space.py` — EMS/EP/CP schemes
- `SIMPAC-2024-311/SpacePartitioner.py` — MCA (Maximal Cuboids)

### For Training:
- `Online-3D-BPP-PCT/kfac.py` — K-FAC optimizer (ACKTR)
- `Online-3D-BPP-PCT/main.py` — Full training pipeline
- `GOPT/masked_ppo.py` — PPO with action masking
