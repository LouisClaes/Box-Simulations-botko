# RL Hybrid Hyper-Heuristic for Online 3D Bin Packing

## Beginner Guide: How This Strategy Works

If you are new to reinforcement learning or bin packing, this section explains the key ideas in plain language before the technical details.

### What is a "hyper-heuristic"?

A normal heuristic is a rule of thumb for solving a problem -- for example "always place the biggest box first". A hyper-heuristic is a strategy that decides which rule of thumb to use at each moment. It is a strategy that picks between strategies.

Think of it like a sports coach. Instead of running onto the field and playing themselves, the coach watches the game and calls out "use attack formation now" or "switch to defence". Each player (heuristic) already knows how to do their job. The coach (hyper-heuristic) just decides who should be active.

### Why is this novel?

Many researchers have applied reinforcement learning to 3D bin packing before. Every single one of those approaches trains the agent to answer the question "WHERE should I put this box?" That means choosing from tens of thousands of (x, y, orientation) combinations -- a huge decision space that requires days of training on expensive GPU hardware.

This strategy asks a completely different question: "WHICH expert rule should I use to decide where to put this box?" Nobody has done this for the online 3D bin packing problem before. The RL agent only has 8 options to choose from (7 expert strategies plus a skip action), which makes it dramatically simpler to train.

### What are the 7 expert strategies?

The agent can call any of these 7 specialist algorithms, plus a SKIP action to advance the conveyor:

| Strategy | What it is good at |
|----------|--------------------|
| baseline (DBLF) | Reliable and conservative; a safe default that almost always finds a valid spot |
| walle_scoring | The current best performer (68.3% fill); balances many criteria at once |
| surface_contact | Maximises how much of the box bottom touches something solid; good for stability |
| extreme_points | Uses corner and edge positions; handles irregular box shapes well |
| skyline | Builds tidy horizontal layers; works best when boxes are similar heights |
| layer_building | Explicitly constructs flat layers one at a time |
| best_fit_decreasing | Sorts by size and fills gaps; useful late in packing when only small spaces remain |

The agent learns to pick the right specialist for the current situation, like a chef who knows when to hand a task to the sauce expert versus the grill expert.

### What are the 39 features?

The agent cannot see a picture of the pallet -- instead it reads a list of 39 numbers that summarise the current situation. These numbers are grouped as follows:

- **16 numbers** describe the two pallets (8 per pallet): how full it is, how tall, how uneven the surface is, how many boxes are on it, and similar measurements.
- **5 numbers** describe the current box: its length, width, height, total volume, and whether it is a cube, slab, or stick.
- **4 numbers** describe the queue of upcoming boxes: average size, how varied the sizes are, how many can be gripped right now, and how diverse the shapes are.
- **3 numbers** describe overall progress: what fraction of boxes have been placed, what fraction remain, and how many pallets have been closed.
- **8 numbers** track recent history: how often each of the 7 strategies was used in the last 10 decisions, plus the recent success rate.
- **3 numbers** flag the packing phase: whether we are in the early stage (under 30%), mid stage (30-60%), or late stage (over 60%).

Together these 39 numbers give the agent enough context to recognise patterns like "the surface is rough and we are mid-packing -- surface_contact usually works well here".

### Why is this fast to train?

Several reasons compound on each other:

- **8 actions instead of 38,400**: the agent never has to reason about x/y positions at all. That alone is a roughly 5,000-times smaller decision space.
- **39 numbers instead of a 120x80 image**: reading a compact feature list is far cheaper than processing a full heightmap with a convolutional neural network.
- **Tiny network**: about 27,000 parameters versus millions for a position-level deep RL agent.
- **No spatial reasoning required**: all the hard geometry is handled by the expert heuristics. The agent only needs to learn "which situation calls for which expert", which is a much simpler pattern.

The result is that the tabular version trains in about 1 hour on a single CPU, and the DQN version trains in 4-8 hours on a CPU -- compared to days or weeks on a GPU cluster for conventional approaches.

### Step-by-step: what happens when a box is placed

1. A new box arrives on the conveyor. Its dimensions and weight are read.
2. The feature extractor surveys both pallets and computes 39 numbers summarising the current state (fill levels, surface roughness, box size, progress, recent history, etc.).
3. These 39 numbers are fed into the Q-network (a small 3-layer neural network with about 27,000 parameters).
4. The Q-network outputs 8 scores (Q-values), one for each strategy and one for SKIP. A higher score means "I expect this choice to lead to a better final pallet fill".
5. The agent picks the strategy with the highest score (or a random one if still exploring during training).
6. That expert strategy runs its own placement algorithm and returns a position (x, y, orientation) for the box.
7. If the expert finds a valid position the box is placed, the pallets update, and the agent receives a reward based on how much the fill improved. If the expert returns no valid position, the agent receives a small penalty.
8. This decision (state, action, reward, next state) is stored in a replay buffer and used later to update the Q-network.

---

## Novel Thesis Contribution

This strategy represents the **primary novel contribution** of the master's thesis. It addresses an identified research gap: **no prior work has applied a selective hyper-heuristic with reinforcement learning to the online 3D bin packing problem**.

### The Research Gap

Existing RL approaches to 3D bin packing (Zhao et al. 2021/2022; Xiong et al. 2024; Tsang et al. 2025) all train agents to make **low-level placement decisions** directly -- choosing x, y positions and orientations from tens of thousands of candidates. This creates:

- Enormous action spaces (e.g., 120 x 80 x 2 x 2 bins = 38,400 actions)
- Long training times (days to weeks on GPU)
- Difficulty transferring to new bin configurations
- Opaque decision-making (hard to interpret why a position was chosen)

### The Innovation: Meta-Learning Over Heuristics

Instead of training an RL agent to decide WHERE to place boxes, we train it to decide WHICH expert heuristic to call for each box. This is a **selective hyper-heuristic** (Burke et al., 2013) where the selection mechanism is learned via Q-learning.

```
                          State Features (39-dim)
                                |
                                v
                    +-------------------+
                    |   Q-Network       |
                    |   (or Q-table)    |
                    +-------------------+
                                |
                    Q-values for 8 actions
                                |
                    +-----------+-----------+
                    |           |           |
                    v           v           v
              [baseline] [walle_scoring] [surface_contact] ... [SKIP]
                    |           |           |
                    v           v           v
              decide_placement(box, bin_state)
                    |           |           |
                    v           v           v
              PlacementDecision (or None)
```

### Key Advantages

| Property | Position-Level RL | Hyper-Heuristic RL |
|----------|------------------|--------------------|
| Action space | ~38,400 | 8 |
| Training time | Days-weeks | Minutes-hours |
| Required compute | GPU cluster | Single CPU |
| Interpretability | Opaque | Can see which heuristic is chosen |
| Transferability | Retraining needed | Heuristics already generalise |
| Quality floor | Random early on | Heuristic quality from step 1 |

## Architecture

### State Features (39 dimensions)

The agent observes a compact, handcrafted feature vector:

| Group | Dims | Features |
|-------|------|----------|
| Per-bin (x2) | 16 | fill_rate, max_height, roughness, num_boxes, avg_height, height_variance, coverage_ratio, largest_gap_ratio |
| Current box | 5 | length, width, height, volume, aspect_ratio |
| Buffer | 4 | mean_volume, volume_variance, num_grippable, diversity_index |
| Progress | 3 | boxes_placed_frac, boxes_remaining_frac, pallets_closed |
| History | 8 | last_10_action_frequencies (7), recent_success_rate (1) |
| Phase | 3 | is_early (<30%), is_mid (30-60%), is_late (>60%) |

**Feature engineering rationale**: Each feature group captures information that influences which heuristic performs best. For example:
- High roughness -> surface_contact or layer_building to smooth
- Late phase with small boxes -> best_fit_decreasing for gap-filling
- Low diversity in buffer -> layer_building for uniform layers
- Recent failures -> try a different heuristic

### Heuristic Portfolio

| Index | Strategy | Strength |
|-------|----------|----------|
| 0 | baseline (DBLF) | Reliable, conservative, stable foundation |
| 1 | walle_scoring | Best overall (68.3% benchmark), balanced scoring |
| 2 | surface_contact | Maximises surface contact, good stability |
| 3 | extreme_points | Good for irregular boxes, uses corner positions |
| 4 | skyline | Good for similar-height boxes, builds layers |
| 5 | layer_building | Explicit layer construction |
| 6 | best_fit_decreasing | Size-sorted placement, good for gap-filling |
| 7 | SKIP | Advance conveyor (no placement) |

### Q-Network (DQN mode)

```
Input:  39-dim state vector
        |
Dense(39, 128) -> ReLU -> Dropout(0.1)
        |
Dense(128, 128) -> ReLU -> Dropout(0.1)
        |
Dense(128, 64) -> ReLU
        |
Dense(64, 8) -> Q-values
        |
Output: 8 Q-values (one per heuristic + SKIP)
```

Total parameters: ~27,000 (vs millions for position-level DQN)

### Tabular Q-Learning (baseline mode)

State discretisation:
- Fill rate per bin: 5 levels
- Max height per bin: 5 levels
- Box size: 3 levels (small/medium/large)
- Roughness: 3 levels (smooth/medium/rough)

Total discrete states: 5^4 x 3 x 3 = 5,625
Q-table size: 5,625 x 8 = 45,000 entries

## Training

### Reward Design

The reward signal teaches the agent which heuristic works best in which situation:

| Event | Reward |
|-------|--------|
| Successful placement | volume_ratio x 10 + fill_delta x 5 |
| Heuristic fails (returns None) | -0.5 |
| SKIP action | -0.3 |
| Switched heuristic and succeeded | +0.1 (diversity bonus) |
| Episode terminal | avg_closed_fill x 10 |

### Training Modes

**Tabular Q-learning** (~1 hour on CPU):
```bash
python strategies/rl_hybrid_hh/train.py --mode tabular --episodes 10000
```
- epsilon: 1.0 -> 0.05 (linear decay over 80% of episodes)
- alpha (LR): 0.1
- gamma: 0.99
- No replay buffer needed

**DQN selector** (~4-8 hours on CPU):
```bash
python strategies/rl_hybrid_hh/train.py --mode dqn --episodes 50000
```
- Double DQN with experience replay (50,000 transitions)
- Target network sync every 500 episodes
- Adam optimiser, LR=0.001
- Huber loss, gradient clipping at 10.0
- Epsilon: 1.0 -> 0.05

### Why Training is Fast

| Reason | Impact |
|--------|--------|
| 8 actions vs 38,400 | ~5000x smaller action space |
| 39-dim features vs 120x80 heightmap | ~250x smaller observation |
| Small MLP vs deep CNN | ~100x fewer parameters |
| No spatial reasoning needed | Agent only learns "which", not "where" |
| Heuristics handle complexity | All spatial reasoning is delegated |

## Evaluation

```bash
python strategies/rl_hybrid_hh/evaluate.py --checkpoint outputs/rl_hybrid_hh/best_model.pt
```

Generates:
1. **Fill rate comparison** bar chart (all strategies + HH)
2. **Selection distribution** pie chart (which heuristic is chosen how often)
3. **Success rate per heuristic** (does the agent learn to avoid bad matches?)
4. **Selection over episode progress** (how strategy changes from early to late)

### Expected Results

The RL Hybrid HH should:
- **Match or exceed** the best individual heuristic (walle_scoring at 68.3%)
- **Outperform** the rule-based hyper-heuristic (selective_hyper_heuristic)
- **Adapt** its heuristic selection based on packing phase
- **Avoid** heuristics that are poor matches for the current state

### Interpretability Analysis

This is where the thesis contribution shines. For each evaluation episode, we can inspect:

1. **Which heuristic was chosen when**: The selection log records every decision with the box, bin state, and Q-values.

2. **Phase-dependent preferences**: Early packing may prefer baseline/layer_building for foundation; late packing may prefer best_fit_decreasing for gaps.

3. **Q-value landscape**: For different state configurations, we can visualise the Q-values to understand the agent's learned preferences.

4. **Failure recovery**: When one heuristic fails, does the agent learn to switch to another?

## File Structure

```
strategies/rl_hybrid_hh/
    __init__.py           # Package init, exports RLHybridHHStrategy
    config.py             # HHConfig dataclass (all hyperparameters)
    state_features.py     # Feature extraction (39 dims)
    network.py            # TabularQLearner + HeuristicSelectorDQN + ReplayBuffer
    strategy.py           # RLHybridHHStrategy (BaseStrategy, registered)
    train.py              # Training script (tabular or DQN CLI)
    evaluate.py           # Evaluation and interpretability analysis
    README.md             # This file
```

## Integration

The strategy is a standard `BaseStrategy` and works with all existing infrastructure:

```python
# In benchmark_all.py
from strategies.rl_hybrid_hh.strategy import RLHybridHHStrategy
strategy = RLHybridHHStrategy(checkpoint_path="outputs/rl_hybrid_hh/best_model.pt")
result = session.run(boxes, strategy)

# Or via the registry (after import in strategies/__init__.py)
from strategies.base_strategy import get_strategy
strategy = get_strategy("rl_hybrid_hh")
```

## Comparison with Related Work

### vs. Zhao et al. (AAAI 2021, ICLR 2022)

Zhao trains a DQN/PCT to directly predict (x, y, orientation) placements.
Our approach trains a much simpler agent to select from expert heuristics.
- Pro: Orders of magnitude faster training
- Pro: Interpretable decisions
- Con: Bounded by best heuristic quality

### vs. Tsang et al. (2025)

Tsang uses DDQN for dual-bin placement with CNN-processed heightmaps.
Our approach uses handcrafted features and delegates spatial reasoning.
- Pro: No CNN needed, works on CPU
- Pro: Same physical setup (dual-bin Botko BV)
- Con: Does not learn novel placements beyond heuristic repertoire

### vs. Burke et al. (2013) -- Hyper-Heuristic Survey

Our work implements a **selection hyper-heuristic** with a **learning-based selection mechanism** (RL).  Most existing hyper-heuristics for packing use:
- Random selection
- Roulette wheel / fitness-proportional
- Choice function (hand-tuned)

Using Q-learning for selection is novel for 3D bin packing.

### vs. Rule-Based HH (selective_hyper_heuristic in this codebase)

The existing `selective_hyper_heuristic` strategy uses hand-crafted rules:
```
if phase < 0.20 -> floor_building
if roughness > 0.15 -> floor_building
if fill > 0.65 and small_item -> best_fit_volume
if fill > 0.50 -> dblf
default -> walle
```

Our RL-based HH:
- Learns rules from experience rather than hand-crafting them
- Can discover non-obvious heuristic-state associations
- Adapts to different box distributions
- Generalises across episodes

## References

1. Burke, E.K., Gendreau, M., Hyde, M., et al. (2013). "Hyper-heuristics: A survey of the state of the art." JORS, 64(12).

2. Zhao, H., She, Q., Zhu, C., et al. (2021). "Online 3D bin packing with constrained deep reinforcement learning." AAAI.

3. Zhao, H., Zhu, C., Xu, X., et al. (2022). "Learning practically feasible policies for online 3D bin packing." ICLR.

4. Xiong, Y., Wang, Y., Wu, J., et al. (2024). "GOPT: Generalizable Online 3D Bin Packing via Transformer-based Deep Reinforcement Learning." RA-L.

5. Tsang, Y.C., et al. (2025). "Dual-bin DDQN for robotic palletizing." SIMPAC.

6. Verma, R., Singhal, A., Khadilkar, H., et al. (2020). "A Generalized Reinforcement Learning Algorithm for Online 3D Bin-Packing."

7. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518.

8. van Hasselt, H., Guez, A., Silver, D. (2016). "Deep reinforcement learning with double Q-learning." AAAI.
