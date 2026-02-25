# PCT Transformer RL Strategy

A production-quality Transformer-based actor-critic for online 3D bin packing,
inspired by Zhao et al. (ICLR 2022) *Packing Configuration Trees* (PCT).

## Beginner Guide: How This Strategy Works

This section explains the core ideas in plain English — no prior ML knowledge required.

### What is a Transformer?

The Transformer is the same technology that powers ChatGPT and other large language
models. Its key ability is **attention**: instead of reading information one piece at a
time (like older neural networks), a Transformer looks at all pieces of information
simultaneously and figures out which ones are most relevant to each other.

In our case, instead of processing text tokens, the Transformer looks at all possible
placement positions on the pallet and decides which one is best for the current box.
Each candidate position "attends to" all other candidates, so the model can reason about
interactions — for example, noticing that placing a box in a corner would complement an
adjacent box already placed there.

### What is a "Pointer Network"?

Imagine you are choosing a restaurant from a list. A pointer network reads all the
descriptions on the list, then literally "points" to the best one — rather than
generating a new description of the ideal restaurant from scratch.

Here, the pointer network reads the features of all candidate placement positions, then
points to the position where the current box should go. This is more natural than trying
to generate coordinates directly, and it automatically handles the fact that the number
of valid positions changes with every box.

### What are "Candidates"?

Instead of checking all ~19,200 grid positions on a pallet (every 10mm cell), the
candidate generator uses geometry to find 30-200 promising spots. It looks for:

- **Corners of existing boxes** — natural stacking points
- **Height transitions** — ledges and shelf edges in the heightmap
- **Empty floor areas** — coarse grid at z=0 to always consider the base layer
- **Gaps between boxes and walls** — residual spaces that fit smaller items

Each candidate is described by **12 numbers**: which bin it is on, its normalised (x, y,
z) position, how much support it has from below, estimated fill impact, surface contact,
gap below, and the box orientation.

### Why Variable Action Space Matters

Different boxes arriving at different pallet states produce different numbers of valid
candidates — sometimes 30, sometimes 200. Most RL algorithms assume a fixed action
space, which would require allocating 19,200 slots and marking most as invalid.

The Transformer handles variable-length sequences naturally: **attention works with any
sequence length**. Whether there are 30 or 200 candidates, the same model processes them
all. During training, shorter sequences are padded to the batch maximum and masked so
padding does not affect the scores.

### Step-by-Step: Placing One Box

1. A box arrives on the conveyor and becomes grippable.
2. The **candidate generator** scans both pallets and finds ~100 promising positions.
3. Each candidate is described by **12 features** (position, support, fill impact, etc.).
4. The **box features** (5 numbers: dimensions and normalised volume) and all candidate
   features are fed into the Transformer.
5. **Three Transformer layers** process everything, letting each candidate token "see"
   all other candidates and the box token through self-attention.
6. The **pointer decoder** computes an attention score for each candidate by comparing
   the contextualised box token against each contextualised candidate token.
7. The candidate with the **highest score** (after masked softmax) is selected.
8. The box is placed at that position on the chosen pallet.

## Overview

This strategy uses reinforcement learning (PPO) with a Transformer encoder-decoder
to learn placement policies for the Botko BV dual-pallet setup:
- 2 EUR pallets (1200x800mm), height cap 2700mm
- FIFO conveyor: 8 visible boxes, pick window of 4
- Close policy: HeightClosePolicy at 1800mm
- Objective: maximise average volumetric fill of closed pallets

## PCT Paper and Our Adaptations

### Original PCT (Zhao et al., ICLR 2022)
The PCT paper represents the packing state as a tree of "packing configurations"
(leaf nodes = candidate placements). A Graph Attention Network (GAT) processes
this tree structure, and a pointer mechanism selects the best leaf node.

Key insights:
1. Represent the action space as a **set of placement candidates**, not a fixed grid
2. Use **attention** to reason about candidate interactions
3. Use a **pointer mechanism** for variable-size action selection
4. **Undiscounted returns** (gamma=1.0) for finite episodes

### Our Adaptation: Transformer instead of GAT
We replace the GAT with a standard **Transformer encoder**, which offers:

| Aspect | PCT (GAT) | Ours (Transformer) |
|--------|-----------|-------------------|
| Architecture | Graph attention on tree | Self-attention on set |
| Positional info | Tree structure | Type embeddings only |
| Implementation | Custom GAT layers | PyTorch nn.TransformerEncoder |
| Scalability | Scales with tree depth | Scales with candidate count |
| Debugging | Complex graph ops | Standard attention patterns |
| Hardware | Custom CUDA kernels | Optimised by PyTorch/cuDNN |

The core PCT insight is preserved: candidates are encoded as feature vectors,
contextualised via self-attention, and selected via pointer attention.

## Architecture

```
Input:  box_features (5)  +  candidate_features (N x 12)
            |                        |
     ItemEncoder(MLP)        CandidateEncoder(MLP)
        5 -> 64 -> 128         12 -> 64 -> 128
            |                        |
     + item_type_embed       + cand_type_embed
            |                        |
            +--- concat as sequence --+
                        |
              [item; cand_1; ...; cand_N]
                        |
            TransformerEncoder (3 layers)
              d_model=128, nhead=4
              dim_ff=256, dropout=0.1
                        |
              contextualised tokens
                   /          \
          item_ctx              cand_ctx
              |                     |
       PointerDecoder          ValueHead
    Q=item_ctx, K=cand_ctx    mean_pool -> MLP
    logits -> softmax -> pi   128 -> 64 -> 1
              |                     |
        action probs           state value
```

## Candidate Generation

The action space at each step consists of 30-200 **placement candidates**,
generated by four complementary methods:

1. **Corner Points**: project outward from corners of placed boxes
   - Right: (x+L, y), Front: (x, y+W), Diagonal: (x+L, y+W), Top: (x, y)

2. **Extreme Points**: scan heightmap for height discontinuities
   - Horizontal and vertical transitions indicate ledges/shelves

3. **Floor Scan**: coarse grid (step=50mm) at z=0
   - Ensures floor placements are always considered

4. **Residual Spaces**: gaps between boxes and bin walls
   - Wall-adjacent positions and inter-box gaps

Each candidate is validated (bounds, height, support >= 30%) and characterised
by a 12-dimensional feature vector:

| Feature | Description | Range |
|---------|-------------|-------|
| bin_idx_onehot (2) | Which bin | {0, 1} |
| x_norm | x / bin_length | [0, 1] |
| y_norm | y / bin_width | [0, 1] |
| z_norm | z / bin_height | [0, 1] |
| support_ratio | Base support fraction | [0, 1] |
| height_after_norm | (z + h) / bin_height | [0, 1] |
| fill_after_norm | Estimated fill after placement | [0, 1] |
| contact_ratio | Surface contact with walls/boxes | [0, 1] |
| gap_below_norm | Empty space below / footprint | [0, 1] |
| adjacent_fill_norm | Avg height of surrounding columns | [0, 1] |
| orient_norm | orientation_idx / num_orientations | [0, 1] |

## Variable Action Space Handling

Unlike standard RL with fixed action spaces, each step has a different number
of candidates. The Transformer handles this naturally:

- Candidates are **padded** to the batch maximum and **masked** in attention
- The pointer decoder applies **masked softmax** (invalid candidates get -inf)
- PPO rollouts store variable-length candidate arrays per step
- Mini-batches re-pad to their local maximum during updates

This is more efficient than discretising the full (x, y, orient, bin) space
into a massive fixed action space with mostly invalid actions.

## Training Procedure

Algorithm: **PPO** (more stable than ACKTR from the PCT paper, similar final
performance according to their ablation).

### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Gamma | 1.0 | Undiscounted (finite episodes, following PCT) |
| GAE lambda | 0.95 | Standard GAE |
| Clip ratio | 0.2 | Standard PPO |
| Entropy coeff | 0.01 | Encourage exploration |
| Learning rate | 3e-4 | Adam with cosine schedule |
| Parallel envs | 16 | Data throughput |
| Rollout steps | 20 | Steps per env per update |
| PPO epochs | 4 | Epochs per rollout batch |
| Total episodes | 200K-500K | Convergence typically at 100K-300K |

### Buffer-Aware Training
During training, the environment tries each grippable box and selects the one
with the most valid candidates (proxy for best placement options). This
implicitly teaches the policy to work with the FIFO conveyor constraint.

### Reward Shaping
Uses the shared RewardShaper from rl_common with components:
- Volume ratio (primary signal)
- Fill delta (dense learning signal)
- Surface contact bonus
- Height penalty (pack bottom first)
- Pallet close bonus
- Terminal episode quality bonus

## Usage

### Training
```bash
cd "python/full workflow"
python -m strategies.rl_pct_transformer.train --episodes 200000 --num_envs 16 --lr 3e-4
python -m strategies.rl_pct_transformer.train --episodes 500000 --device cuda --log_dir outputs/pct_long
```

### Evaluation
```bash
python -m strategies.rl_pct_transformer.evaluate --checkpoint outputs/rl_pct_transformer/logs/best.pt --episodes 50
python -m strategies.rl_pct_transformer.evaluate --checkpoint best.pt --compare extreme_points,walle_scoring
```

### As a Strategy (in benchmarks)
```python
from strategies.rl_pct_transformer import RLPCTTransformerStrategy
strategy = RLPCTTransformerStrategy(checkpoint_path="path/to/best.pt")
# Use with PackingSession, benchmark_all.py, etc.
```

Without a trained checkpoint, the strategy falls back to `extreme_points`
heuristic automatically.

## File Structure

```
rl_pct_transformer/
  __init__.py               Package init, registers strategy
  config.py                 PCTTransformerConfig dataclass (all hyperparameters)
  network.py                PCTTransformerNet (Transformer actor-critic)
  candidate_generator.py    CandidateGenerator (action space construction)
  train.py                  PPO training loop with parallel environments
  evaluate.py               Evaluation and baseline comparison utilities
  strategy.py               RLPCTTransformerStrategy (BaseStrategy wrapper)
  README.md                 This file
```

## References

1. Zhao, H., She, Q., Zhu, C., Yang, Y., & Xu, K. (2022).
   "Online 3D Bin Packing with Constrained Deep Reinforcement Learning."
   *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(4), 4547-4555.

2. Zhao, H., Xu, K., She, Q., & Yang, Y. (2022).
   "Learning Efficient Online 3D Bin Packing on Packing Configuration Trees."
   *International Conference on Learning Representations (ICLR)*.

3. Zhao, H., Yang, Y., Xu, K., & She, Q. (2025).
   "Learning Online 3D Bin Packing from Configuration Trees and Search."
   *International Journal of Robotics Research (IJRR)*, 44(2), 299-323.

4. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
   "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*.

5. Vaswani, A., et al. (2017).
   "Attention Is All You Need." *NeurIPS 2017*.

6. Vinyals, O., Fortunato, M., & Jaitly, N. (2015).
   "Pointer Networks." *NeurIPS 2015*.

7. Kool, W., van Hoof, H., & Welling, M. (2019).
   "Attention, Learn to Solve Routing Problems!"
   *International Conference on Learning Representations (ICLR)*.
