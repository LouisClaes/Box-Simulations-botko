# Deep Reinforcement Learning Architectures for 3D Bin Packing

> **User context**: Semi-online (buffer 5-10), k=2 bounded space (2 EUR pallets), maximize fill rate + stability, Python + PyTorch thesis project.

---

## E. Deep RL Architectures for 3D Bin Packing

### E.0 Comparison Matrix

| Dimension | PCT (Zhao 2022/2025) | Constrained DRL (Zhao 2021) | Tsang DDQN (Tsang 2025) | PackMan DQN (Verma 2020) | Deep-Pack 3D (Kundu 2019 ext.) | Decomposed AC (Zhao 2023) |
|---|---|---|---|---|---|---|
| **RL algorithm** | ACKTR (or PPO fallback) | ACKTR | Double DQN | DQN | Dueling Double DQN | ACKTR |
| **Network arch.** | GAT + Pointer Network (~68K params) | 5-layer CNN + Actor + Critic + MaskPredictor | CNN(2ch) + FC branch, merged | FC-only (pooled state) | ResNet / Dueling DQN (5-ch CNN) | Shared CNN encoder + 3 decomposed heads |
| **State repr.** | Packing Configuration Tree (graph) | 2D height map (grid) | 2-channel height map (bin + item) | Pooled state (x_bar, y_bar, z_bar) | 5-channel height map | 6-channel tensor (height map + features) |
| **Action space** | Index into leaf set L_t | Grid cell (x,y) + orientation | 6 x k x |M| (up to 3000) | Corner-aligned placements via selective search | Extreme points x orientations (~100) | Decomposed: o -> x -> y (O(3n) vs O(n^3)) |
| **Multi-bin (k=2)** | 3 options in code; SpatialEnsemble for cross-bin | 3 options: dual heightmap / bin-head / hierarchical | Native dual-bin support (primary contribution) | `selective_search_multi_bin()` | Dual heightmap channels | BinSelectionHead added to decomposition |
| **Buffer support** | ToP MCTS (s=lookahead) via BufferManager | MCTSPermutationSearch | Native buffer support (k=5-10 sweet spot) | Buffer class with selective_search_with_buffer | BufferSelector class | ItemSelectionHead with MultiheadAttention |
| **Stability** | 6 constraint rewards + QuasiStatic + Physics | 3-tiered (60%/80%/95% base support) | LBCP 50% base support (needs enhancement) | Not explicit (implicit via WallE heuristic) | R_stab in reward (support percentage) | Far-to-near V_safe reward |
| **Space management** | EMS (default), CP, EP, EV (4 schemes) | Height map implicit | MCA (Maximal Cuboids Algorithm) | Corner points (find_corner_locations) | Extreme points | Height map implicit |
| **Fill rate reported** | 75.8% (online+stab), 86.0% (online), 93.5% (ToP s=10) | 73.4% (height map, online) | 80-85% (dual-bin, online) | 73.2% (online, DQN) | 84-100% PE (2D only) | 70.1-72.6% (varies by resolution) |
| **Open source** | github.com/alexfrom0815/PCT (partial) | github.com/alexfrom0815/Online-3D-BPP-DRL (635 stars) | github.com/SoftwareImpacts/SIMPAC-2024-311 (TF 2.10) | Not public | Not public | Not public |
| **Framework** | PyTorch | PyTorch | TensorFlow 2.10 | TensorFlow/Keras | TensorFlow | PyTorch |
| **Training time** | 8-12 weeks impl; hours per run | ~16h on Titan V (10x10) | 100 iter x 1000 ep | 2000 ep, ~4-8h GPU | 500K episodes | ~12h on Titan V (100x100) |
| **Thesis match** | High (best fill rate, complex) | High (CMDP principled, same research group) | **Highest** (all 5 dimensions match) | Medium (simpler, good baseline) | Medium (2D origin, needs 3D extension) | High (elegant decomposition, real robot) |
| **Implementation effort** | 8-12 weeks (hardest) | 4-6 weeks | 3-5 weeks | 2-3 weeks (easiest) | 4-6 weeks | 4-6 weeks |

### E.0.1 Quick-Reference: Which Approach for Which Scenario

| Scenario | Recommended Approach | Rationale |
|---|---|---|
| Fastest prototype / baseline | PackMan DQN | Simplest architecture, FC-only network, 2-3 weeks |
| Closest to thesis setup (dual-bin + buffer) | Tsang DDQN | Native dual-bin + buffer, only approach designed for this exact setting |
| Best theoretical fill rate | PCT + ToP MCTS | 93.5% with lookahead s=10, but most complex to implement |
| Most principled constraint handling | Constrained DRL (CMDP) | Feasibility mask prediction avoids reward shaping pitfalls |
| Best action space scalability | Decomposed Actor-Critic | O(3n) vs O(n^3), proven on real robot |
| Best generalization to unseen items | PCT with MultiScaleItemSampler | Explicit generalization benchmark framework |

---

### E.1 Approach 1: Packing Configuration Tree (PCT) -- Zhao et al. (ICLR 2022 / IJRR 2025)

**Source files**:
- `C:\Users\Louis\Downloads\stapelalgortime\python\deep_rl\pct_coding_ideas.py` (~2899 lines)
- `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\Deliberate Planning of 3D Bin Packing on Packing Configuration Trees (Summary).md`

**Paper**: "Deliberate Planning of 3D Bin Packing on Packing Configuration Trees" -- ICLR 2022 (oral), extended to IJRR 2025.

#### E.1.1 Core Idea

The Packing Configuration Tree (PCT) represents the current bin state as a dynamic tree structure where each leaf node corresponds to a feasible placement location. A Graph Attention Network (GAT) encodes this tree, and a Pointer Network selects which leaf to place the current item at. This is the **first learning-based method for online 3D-BPP with a continuous solution space**, moving beyond fixed-grid discretization.

#### E.1.2 State Representation

The state is the PCT itself: `S = (T_t, n_t)` where `T_t` is the tree at step t and `n_t` is the incoming item.

**Tree structure** (from `PackingConfigurationTree` class, line ~120):
- **Root node**: represents the bin (type `NodeType.BIN`)
- **Internal nodes**: represent placed items (type `NodeType.ITEM`)
- **Leaf nodes**: represent feasible placement locations (type `NodeType.LEAF`)

Each node carries a feature vector of dimension 9:
```
[x, y, z, w, h, d, node_type_encoding, is_current_item, occupancy_ratio]
```

**Four expansion schemes** determine how new leaf nodes are generated after each placement (from `LeafInterceptor`, `CornerPointExpander`, `EventPointExpander`, `EMSManager` classes):

| Scheme | Class | Complexity | Fill Rate (Setting 2) | Description |
|---|---|---|---|---|
| CP (Corner Points) | `CornerPointExpander` | O(c) constant | 83.5% | 8 corners of placed item's bounding box |
| EP (Event Points) | `EventPointExpander` | O(m * \|B_2D\|) | 85.0% | Projections onto existing item surfaces |
| **EMS (Empty Maximal Spaces)** | `EMSManager` | O(\|E\|) | **86.0%** | Maximal empty rectanguloids -- **DEFAULT** |
| EV (Event + Vertices) | `EventPointExpander` (extended) | O(m * \|B_2D\|^2) | 86.2% | EP + additional vertex intersections |

**EMS is the default and recommended scheme** -- best balance of quality (86.0%) and computational cost (O(|E|)).

The `EMS` class (line ~85) stores each empty maximal space:
```python
class EMS:
    def __init__(self, x_min, y_min, z_min, x_max, y_max, z_max):
        self.bounds = (x_min, y_min, z_min, x_max, y_max, z_max)
```

The `EMSManager` class (line ~95) provides:
- `update_ems_after_placement(placement)` -- removes intersecting EMS, generates new ones
- `get_feasible_placements(item)` -- returns all (ems, orientation) pairs where item fits

#### E.1.3 Network Architecture

**From `pct_network_pseudocode()` function (line ~1530+)**:

```
PCTNetwork (~68K parameters):
  1. Item encoder:    MLP(9 -> 128)  [encodes current item features]
  2. GAT encoder:     GAT(128, single head, 3 layers)  [encodes PCT graph]
  3. Pointer decoder: Attention(query=item_emb, keys=leaf_embs) -> softmax -> pi(a|s)
  4. Critic:          MLP(128 -> 1)  [state value V(s)]
```

Key design choices:
- **Single-head GAT** (not multi-head) -- sufficient because PCT structure already encodes spatial relationships
- **~68K parameters** -- deliberately small for fast inference
- The pointer mechanism naturally handles variable-size action spaces (different number of leaves per step)

#### E.1.4 RL Training

**MDP formulation**:
- State: `S = (T_t, n_t)` -- PCT + current item
- Action: `a = index(l)` for `l in L_t` -- select a leaf node
- Reward: `r = c_r * w_t` where `w_t` is the volume ratio of placed item, `c_r` is a constraint multiplier
- Discount: `gamma = 1` (episodic, undiscounted)

**ACKTR training** (from `acktr_training_pseudocode()`, line ~1630+):
- ACKTR = Actor-Critic with Kronecker-Factored Trust Region
- Uses K-FAC approximation of Fisher information matrix for natural gradient
- More sample-efficient than PPO/A2C but harder to implement
- **PPO fallback**: `ppo_training_pseudocode()` provides a simpler alternative with ~5-10% less sample efficiency

**From `ConstraintRewards` class (line ~1700+), 6 constraint functions**:

| Constraint | Function | Formula | Purpose |
|---|---|---|---|
| C1: Heightmap | `heightmap_constraint()` | `1 - max_h / H_bin` | Penalize tall stacks |
| C2: Support ratio | `support_ratio_constraint()` | `A_support / A_bottom >= tau` | Base support (tau=0.8) |
| C3: Stability | `stability_constraint()` | CoM within support polygon | Quasi-static equilibrium |
| C4: Pyramidal | `pyramidal_constraint()` | `1 if w_i >= w_j for all j on i` | Heavy-on-bottom |
| C5: Fragility | `fragility_constraint()` | `1 if frag_i <= frag_j for all j on i` | Fragile-on-top |
| C6: Reachability | `reachability_constraint()` | Gripper clearance check | Robot arm clearance |

The combined constraint multiplier: `c_r = prod(C_k for k in active_constraints)`.

**Sensitivity table** (from summary): C2 (support ratio) is the most impactful constraint; C4 (pyramidal) and C5 (fragility) have the least impact on fill rate.

#### E.1.5 ToP MCTS for Buffer/Lookahead

**From `MCTSNode` class and `top_mcts_planning()` function (line ~260+)**:

Tree of Packing (ToP) extends PCT with Monte Carlo Tree Search for semi-online settings with a buffer:

```
ToP-MCTS Algorithm:
  Input: buffer B of size s, current PCT state
  1. Build search tree: each node = (item_choice, placement_choice)
  2. Selection: UCB1 = Q(s,a)/N(s,a) + c * sqrt(ln(N(s)) / N(s,a))
  3. Expansion: try placing selected item at each feasible leaf
  4. Simulation: random rollout to episode end
  5. Backpropagation: update Q-values with episode fill rate
  Output: best (item, placement) pair
```

**Buffer integration** via `BufferManager` class (line ~230+):
- `add_to_buffer(item)` -- adds incoming item to FIFO buffer
- `select_from_buffer(policy)` -- selects item from buffer using ToP or greedy
- `must_place_check()` -- forces placement when buffer is full

**Results with buffer/ToP**:
| Setting | Fill Rate |
|---|---|
| Online (no buffer) | 75.8% (with stability), 86.0% (no stability) |
| ToP s=5 | ~90% |
| ToP s=10 | **93.5%** |
| Offline (full sequence) | 95.2% |

#### E.1.6 k=2 Dual-Bin Extension

**From `TwoBinPCTManager` class (line ~200+)**, three options:

**Option A: Independent PCTs** (simplest)
```python
class TwoBinPCTManager:
    def __init__(self):
        self.pct_bin1 = PackingConfigurationTree(bin_dims)
        self.pct_bin2 = PackingConfigurationTree(bin_dims)

    def get_all_actions(self, item):
        actions_1 = self.pct_bin1.get_feasible_placements(item)
        actions_2 = self.pct_bin2.get_feasible_placements(item)
        return actions_1 + actions_2  # union of actions
```

**Option B: Shared encoder, separate decoders** -- single GAT processes both trees concatenated

**Option C: SpatialEnsemble** (recommended for thesis)
From `SpatialEnsemble` class (line ~330+):
```python
class SpatialEnsemble:
    """Cross-bin evaluation using rank normalization."""
    def evaluate_cross_bin(self, placements_bin1, placements_bin2):
        scores_1 = [self.score_placement(p, bin1_state) for p in placements_bin1]
        scores_2 = [self.score_placement(p, bin2_state) for p in placements_bin2]
        # Rank normalize across both bins for fair comparison
        all_scores = rank_normalize(scores_1 + scores_2)
        return argmax(all_scores)
```

#### E.1.7 Advanced Features

**Recursive Packing Decomposer** (`RecursivePackingDecomposer` class, line ~350+):
- Decomposes large bins into sub-problems for scalability
- Each sub-problem gets its own PCT
- Useful for EUR pallet (120x80cm) which is large relative to typical items

**Multi-Scale Item Sampler** (`MultiScaleItemSampler` class, line ~420+):
- Trains on items from multiple size distributions simultaneously
- Improves generalization to unseen item distributions
- 3 distributions: small (5-15cm), medium (15-35cm), large (35-60cm)

**Stability verification pipeline** (line ~1750+):
1. `QuasiStaticStabilityChecker` -- analytical check (fast, 99.9% accuracy vs physics)
   - Checks center of mass within support polygon
   - O(N log N) via stacking tree data structure
2. `PhysicsStabilityVerifier` -- PyBullet or Isaac Gym simulation (slow, ground truth)
   - Used for validation, not training
   - Simulates 2 seconds of physics, checks displacement < threshold

**Generalization benchmark** (`GeneralizationBenchmark` class, line ~1850+):
- Tests on 5 item distributions: uniform random, skewed small, skewed large, mixed, real-world (from logistics datasets)
- Reports fill rate + stability rate + inference time per each

#### E.1.8 Implementation Roadmap

From the coding ideas file, 8 phases totaling 8-12 weeks:
1. Data structures (Box, Placement, EMS) -- 1 week
2. PCT tree + expansion schemes -- 1-2 weeks
3. GAT + Pointer Network -- 1-2 weeks
4. BinPackingEnv (Gymnasium) -- 1 week
5. ACKTR/PPO training loop -- 1-2 weeks
6. Constraint rewards + stability -- 1 week
7. ToP MCTS + buffer -- 1-2 weeks
8. Dual-bin extension + evaluation -- 1 week

**Expected thesis performance** (from code file): Buffer=10, 2-bounded, with stability: **78-87% fill rate**.

#### E.1.9 Key Strengths and Weaknesses

**Strengths**:
- Highest reported fill rates among all approaches
- Continuous solution space (no grid discretization artifacts)
- Principled tree representation captures spatial relationships
- ToP MCTS provides strong buffer utilization
- Most thoroughly validated (real robot experiments)

**Weaknesses**:
- Most complex to implement (8-12 weeks)
- ACKTR harder than PPO/DQN to implement correctly
- GAT + Pointer Network requires careful tuning
- Tree structure adds memory overhead vs flat height maps
- No native dual-bin support (must be added)

---

### E.2 Approach 2: Constrained DRL with Feasibility Masks -- Zhao et al. (AAAI 2021)

**Source files**:
- `C:\Users\Louis\Downloads\stapelalgortime\python\deep_rl\constrained_drl_bin_packing.py` (1454 lines)
- `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\Online 3D Bin Packing with Constrained DRL (Summary).md`

**Paper**: "Online 3D Bin Packing with Constrained Deep Reinforcement Learning" -- AAAI 2021, ~94 citations.

#### E.2.1 Core Idea

Formulates 3D bin packing as a **Constrained Markov Decision Process (CMDP)** rather than using reward shaping for constraints. The key insight: adding constraint penalties to the reward function changes the optimal policy in unpredictable ways, while CMDP keeps constraints separate and enforces them via feasibility masks.

Mathematical argument (from summary): In standard MDP reward shaping `R' = R + lambda * C`, the optimal policy `pi*_R'` differs from the constrained optimal `pi*_C = argmax E[R] s.t. E[C] <= threshold`. The CMDP formulation with mask prediction is provably correct.

#### E.2.2 State Representation

**Height map** (from `BinState` class, line ~15):
```python
class BinState:
    def __init__(self, L, W, H):
        self.height_map = np.zeros((L, W), dtype=np.float32)  # 2D grid
        self.H = H  # bin height

    def update_after_placement(self, placement):
        x, y, w, d, h = placement.x, placement.y, placement.w, placement.d, placement.h
        region = self.height_map[x:x+w, y:y+d]
        self.height_map[x:x+w, y:y+d] = np.maximum(region, placement.z + h)
```

State tensor passed to network: `[height_map / H, item_w/L, item_d/W, item_h/H]` -- normalized height map concatenated with item dimensions.

**State representation comparison** (from summary):
| Representation | Fill Rate |
|---|---|
| **Height map** | **73.4%** |
| Height vector | 57.4% |
| Item sequence | 54.3% |

Height maps are clearly superior and are the standard across most approaches.

#### E.2.3 Network Architecture

**From `ConstrainedPackingNetwork` class (line ~200+)**:

```
ConstrainedPackingNetwork:
  StateCNN (shared encoder):
    Conv2d(1, 32, 3, padding=1) -> ReLU
    Conv2d(32, 64, 3, padding=1) -> ReLU
    Conv2d(64, 64, 3, padding=1) -> ReLU
    Conv2d(64, 128, 3, padding=1) -> ReLU
    Conv2d(128, 128, 3, padding=1) -> ReLU
    AdaptiveAvgPool2d(4, 4) -> Flatten -> Linear(128*16, 256) -> ReLU

  Actor:
    Linear(256 + item_dim, 256) -> ReLU
    Linear(256, num_actions) -> Softmax  [policy pi(a|s)]

  Critic:
    Linear(256 + item_dim, 256) -> ReLU
    Linear(256, 1)  [value V(s)]

  MaskPredictor:
    Linear(256 + item_dim, 256) -> ReLU
    Linear(256, num_actions) -> Sigmoid  [feasibility mask m(a|s)]
```

**Key innovation**: The `MaskPredictor` learns to predict which actions are feasible (physically valid placements). The predicted mask is applied to the policy **before** the softmax:
```python
logits = self.actor(features)
mask = self.mask_predictor(features)
masked_logits = logits * mask + (1 - mask) * (-1e9)  # mask infeasible actions
pi = softmax(masked_logits)
```

This is more principled than reward penalties because infeasible actions are never sampled during rollouts.

#### E.2.4 Loss Function

**5-component loss** (from `ConstrainedDRLTrainer` class, line ~350+):

```
L = alpha * L_actor + beta * L_critic + lambda * L_mask + omega * E_inf - psi * E_entropy
```

| Component | Weight | Formula | Purpose |
|---|---|---|---|
| `L_actor` | alpha=1.0 | `-log(pi(a|s)) * A(s,a)` | Policy gradient (advantage-weighted) |
| `L_critic` | beta=0.5 | `(V(s) - R_target)^2` | Value function regression |
| `L_mask` | lambda=0.5 | `BCE(m_pred, m_true)` | Feasibility mask supervision |
| `E_inf` | omega=0.01 | `sum(pi * log(mask))` | Encourage selecting feasible actions |
| `E_entropy` | psi=0.01 | `-sum(pi * log(pi))` | Exploration bonus |

#### E.2.5 Stability: Three-Tiered Criterion

From the coding ideas file (line ~600+):

| Tier | Base Support | Corner Requirement | When to Use |
|---|---|---|---|
| Relaxed | >= 60% | 4 corners supported | Early training (exploration) |
| Standard | >= 80% | 3 corners supported | Mid training |
| Strict | >= 95% | N/A | Final training + evaluation |

The curriculum from relaxed to strict helps training converge.

#### E.2.6 RL Algorithm Comparison

From summary, tested on the same environment:

| Algorithm | Fill Rate | Notes |
|---|---|---|
| **ACKTR** | **73.4%** | Best -- natural gradient via K-FAC |
| A2C | 70.8% | Simpler but less sample-efficient |
| SAC | 68.2% | Off-policy, continuous actions |
| DQN | 65.1% | Value-based, discrete actions |
| Rainbow | 63.7% | Surprisingly worse than vanilla DQN |

ACKTR's advantage comes from the natural gradient update, which accounts for the curvature of the policy parameter space.

#### E.2.7 k=2 Dual-Bin Extension

**From `DualBinConstrainedNetwork` class (line ~700+)**, three options:

**Option A: Dual heightmaps** -- concatenate both bin heightmaps as 2-channel input
```python
state = torch.stack([bin1_heightmap, bin2_heightmap], dim=0)  # (2, L, W)
```

**Option B: Bin selection head** -- add a separate head that first selects which bin, then which placement
```python
bin_logits = self.bin_selector(shared_features)  # (2,)
bin_choice = Categorical(logits=bin_logits).sample()
placement = self.actor(shared_features, bin_choice)
```

**Option C: Hierarchical (recommended)** -- two-level decision: bin selection -> placement within selected bin
```python
# Level 1: bin selection (lightweight)
bin_probs = softmax(self.bin_head(shared_features))
# Level 2: placement selection (full network per bin)
placement_probs = self.placement_head(shared_features, selected_bin)
```

#### E.2.8 MCTS for Buffer

**From `MCTSPermutationSearch` class (line ~500+)**:
- Searches over item orderings in the buffer
- Each MCTS node represents a partial ordering of buffered items
- Simulation uses the trained policy to evaluate orderings
- Returns the best item to place next from the buffer

#### E.2.9 Implementation Notes

- **Open source**: github.com/alexfrom0815/Online-3D-BPP-DRL (635 stars, PyTorch)
- **K-FAC implementation**: `KFACOptimizer` class (line ~1200+) provides the Kronecker-factored approximate curvature optimizer
- **Data generation**: `DatasetGenerator` class (line ~900+) creates training item sequences
- **Evaluation**: `PackingEvaluator` class (line ~1000+) computes fill rate, stability rate, constraint violations
- Training time: ~16 hours on NVIDIA Titan V for 10x10 grid resolution
- Implementation effort: 4-6 weeks

#### E.2.10 Key Strengths and Weaknesses

**Strengths**:
- Most principled constraint handling (CMDP > reward shaping, mathematically proven)
- Open source with 635 stars -- most mature codebase
- Feasibility mask ensures 100% valid placements after training
- Same research group as PCT -- compatible design philosophy
- Three-tiered stability curriculum is elegant

**Weaknesses**:
- Lower fill rate than PCT (73.4% vs 86.0%) in online setting
- CNN-based state encoding loses some spatial precision vs tree-based
- K-FAC optimizer is complex to implement correctly
- Height map discretization limits placement precision

---

### E.3 Approach 3: Tsang DDQN for Dual-Bin Packing -- Tsang et al. (2025)

**Source files**:
- `C:\Users\Louis\Downloads\stapelalgortime\python\deep_rl\coding_ideas_tsang2025_ddqn_dual_bin.py` (1747 lines)
- `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\A deep reinforcement learning approach for online and concurrent 3D bin (Summary).md`

**Paper**: "A deep reinforcement learning approach for online and concurrent 3D bin packing" -- Tsang et al. 2025, Computers in Industry.

#### E.3.1 Core Idea

This is the **single closest match to the thesis setup**: it is the only paper that addresses all five thesis dimensions simultaneously: (1) online arrival, (2) buffer, (3) multiple bins, (4) DRL-based, (5) 3D. The approach uses a Double DQN that evaluates each possible (state, action) pair individually, combined with the Maximal Cuboids Algorithm (MCA) for space management.

#### E.3.2 Formal Problem Definition

**3D Dual-Bin Packing Problem (3D-DBPP)** from the paper:
- Items arrive online in sequence `{n_1, n_2, ..., n_N}`
- Each item `n_i` has dimensions `(w_i, h_i, d_i)` and weight
- k=2 open bins available simultaneously
- Buffer of size B holds incoming items
- Goal: minimize number of bins used (equivalently, maximize fill rate per bin)

**MDP formulation**:
- State: `s_t = (heightmap_bin1, heightmap_bin2, item_features, buffer_state)`
- Action: `a_t = (item_choice, bin_choice, position, orientation)` -- up to `6 x k x |M|` actions
- |M| = number of maximal cuboids across both bins
- Reward: composite (see E.3.5)
- Discount: gamma = 0.95

#### E.3.3 Maximal Cuboids Algorithm (MCA)

**From `MaximalCuboidsAlgorithm` class (line ~150+)**:

MCA is the 3D extension of MAXRECTS (well-known in 2D bin packing). It maintains a list of maximal empty cuboids in the bin.

```python
class MaximalCuboidsAlgorithm:
    def __init__(self, L, W, H):
        self.cuboids = [Cuboid(0, 0, 0, L, W, H)]  # initially one big cuboid

    def place_item(self, placement):
        """Update cuboid list after placing an item."""
        new_cuboids = []
        for c in self.cuboids:
            if not intersects(c, placement):
                new_cuboids.append(c)
            else:
                # Split intersecting cuboid into up to 6 sub-cuboids
                splits = self._split_cuboid(c, placement)
                new_cuboids.extend(splits)
        # Remove cuboids fully contained in others
        self.cuboids = self._remove_contained(new_cuboids)

    def get_placements(self, item):
        """Return all feasible (cuboid, orientation) pairs."""
        placements = []
        for c in self.cuboids:
            for orient in item.get_orientations():
                if fits(orient, c):
                    placements.append(Placement(c.x, c.y, c.z, orient, c))
        return placements
```

Key properties:
- Cuboids are placed at the corner (x_min, y_min, z_min) of each maximal cuboid
- After each placement, intersecting cuboids are split and contained ones are pruned
- Typical number of maximal cuboids: 50-200 per bin, so action space = 6 orientations x 2 bins x 100 cuboids = ~1200

#### E.3.4 Network Architecture

**From `OnlineDualBinPackingEnv` and training code (line ~600+)**:

The DQN evaluates each (state, action) pair individually, outputting a single Q-value:

```
DDQN Architecture:
  CNN branch (processes height map):
    Input: 2-channel (bin_heightmap + item_footprint_at_position)
    Conv2d(2, 32, 3, padding=1) -> ReLU
    Conv2d(32, 64, 3, padding=1) -> ReLU
    Conv2d(64, 64, 3, padding=1) -> ReLU
    Conv2d(64, 128, 3, padding=1) -> ReLU
    Conv2d(128, 128, 3, padding=1) -> ReLU
    GlobalAvgPool -> 128-dim vector

  FC branch (processes item features):
    Input: [item_w, item_h, item_d, orient, bin_id, position_x, position_y, position_z]
    Linear(8, 128) -> ReLU
    Linear(128, 256) -> ReLU

  Merge:
    Concat(CNN_out, FC_out) -> 384-dim
    Linear(384, 512) -> ReLU
    Linear(512, 256) -> ReLU
    Linear(256, 1) -> Q-value
```

**Double DQN** addresses overestimation bias:
```python
# Standard DQN: Q_target = r + gamma * max_a Q_target(s', a)
# Double DQN:   Q_target = r + gamma * Q_target(s', argmax_a Q_online(s', a))
a_best = argmax(Q_online(s_next, all_actions))
Q_target_value = reward + gamma * Q_target_network(s_next, a_best)
```

**Key difference from other approaches**: The network evaluates ONE (state, action) pair at a time, rather than outputting Q-values for all actions simultaneously. This means:
- Forward passes scale linearly with number of candidate actions
- But: each evaluation is more accurate because the network sees the exact placement
- Inference: ~1200 forward passes per step (can be batched efficiently on GPU)

#### E.3.5 Reward Function

**From `compute_reward()` and `compute_enhanced_reward()` functions (line ~800+)**:

```python
def compute_reward(placement, bin_state):
    R_pyramid = pyramid_score(placement, bin_state)  # heavy items at bottom
    R_compactness = compactness_score(placement, bin_state)  # minimize wasted space
    return w1 * R_pyramid + w2 * R_compactness  # w1=0.3, w2=0.7

def compute_enhanced_reward(placement, bin_state):
    """Enhanced reward for thesis (adds stability)."""
    R_base = compute_reward(placement, bin_state)
    R_stability = support_area(placement, bin_state) / footprint_area(placement)
    R_height = 1.0 - placement.z / bin_state.H  # prefer lower placements
    return 0.4 * R_base + 0.3 * R_stability + 0.3 * R_height
```

#### E.3.6 Bin Replacement Strategy

**From `BinReplacementStrategy` class (line ~300+)**:

When a bin becomes "full enough", it is closed and replaced with an empty bin. Three strategies:

| Strategy | Trigger | Effect | Fill Rate |
|---|---|---|---|
| `replaceAll` | Any bin closed -> replace all | Both bins reset | Lower (wasteful) |
| **`replaceMax`** | Most-full bin closed -> replace it | Only 1 bin resets | **Higher (recommended)** |
| `replaceThreshold` | Bin > threshold% full -> replace | Configurable | Middle |

**replaceMax is recommended** (from paper results): it keeps one partially-filled bin active for longer, allowing better item-to-bin matching.

```python
class BinReplacementStrategy:
    def replace_max(self, bins):
        fill_rates = [b.compute_fill_rate() for b in bins]
        if max(fill_rates) > self.threshold:
            idx = argmax(fill_rates)
            closed_bin = bins[idx]
            bins[idx] = Bin(self.L, self.W, self.H)  # fresh bin
            return closed_bin
        return None
```

#### E.3.7 EUR Pallet Adaptation

**From `EURPalletConfig` class (line ~450+)**:

```python
class EURPalletConfig:
    L = 120  # cm, length
    W = 80   # cm, width
    H = 150  # cm, height (typical pallet + cargo)
    RESOLUTION = 2  # cm per grid cell
    GRID_L = 60  # 120 / 2
    GRID_W = 40  # 80 / 2
    # Height map: 60 x 40 grid at 2cm resolution
```

This gives a 60x40 height map, which is manageable for the CNN (2400 cells).

#### E.3.8 LBCP Stability

**From `LBCPStabilityChecker` class (line ~500+)**:

LBCP = Largest Base Contact Percentage. An item is stable if at least 50% of its base area is supported:
```python
def is_stable(self, placement, bin_state):
    base_area = placement.w * placement.d
    support_area = self._compute_support_area(placement, bin_state)
    return support_area / base_area >= 0.50
```

**Note**: 50% is relatively permissive. For the thesis, enhance to match the Constrained DRL three-tiered approach (60%/80%/95%) or the PCT support ratio constraint (80% default).

#### E.3.9 Training Configuration

From `train_ddqn_dual_bin_packing()` function (line ~1000+):

| Parameter | Value |
|---|---|
| Iterations | 100 |
| Episodes per iteration | 1000 |
| gamma | 0.95 |
| Replay buffer size | 1,000,000 |
| Batch size | 256 |
| epsilon start | 1.0 |
| epsilon end | 0.05 |
| epsilon decay | linear over 80 iterations |
| Target network update | every 1000 steps |
| Optimizer | Adam, lr=0.001 |
| Loss | MSE (Huber optional) |

#### E.3.10 Open Source

**DeepPack3D**: github.com/SoftwareImpacts/SIMPAC-2024-311
- TensorFlow 2.10, Python 3.10, MIT license
- **Needs PyTorch port for thesis** (TF -> PyTorch conversion)
- Includes MCA implementation, training loop, evaluation scripts

#### E.3.11 Key Strengths and Weaknesses

**Strengths**:
- **Closest match to thesis setup** -- all 5 dimensions align
- Native dual-bin support with replacement strategies
- MCA provides high-quality candidate placements
- Open source (MIT license)
- Buffer integration is straightforward
- Moderate implementation complexity (3-5 weeks)

**Weaknesses**:
- TensorFlow codebase needs PyTorch port
- Per-(state,action) Q-evaluation is slower at inference (many forward passes)
- LBCP stability at 50% is too permissive (needs enhancement)
- Double DQN less sample-efficient than ACKTR
- No feasibility mask -- relies on MCA to generate only valid placements

---

### E.4 Approach 4: PackMan DQN -- Verma et al. (AAAI 2020)

**Source files**:
- `C:\Users\Louis\Downloads\stapelalgortime\python\deep_rl\packman_dqn_coding_ideas.py` (896 lines)
- `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\A Generalized Reinforcement Learning Algorithm for Online 3D Bin-Packing (Summary).md`

**Paper**: "A Generalized Reinforcement Learning Algorithm for Online 3D Bin-Packing" -- Verma et al., AAAI 2020.

#### E.4.1 Core Idea

PackMan combines a hand-crafted heuristic (WallE) with a simple DQN. The key innovations are: (1) **selective search** that drastically reduces the action space to only corner-aligned placements, and (2) a **retroactive terminal reward** that provides the fill rate signal at the end of the episode. This is the **simplest DRL approach** and makes a good baseline.

#### E.4.2 State Representation: Pooled State

**From `compute_pooled_state()`, `compute_border_encoding()`, `compute_location_onehot()` functions (line ~300+)**:

Rather than feeding the raw height map to the network, PackMan computes a compact "pooled" state:

```python
def compute_pooled_state(height_map, item):
    """Compute x_bar, y_bar, z_bar pooled representations."""
    L, W = height_map.shape
    # x_bar: for each x, aggregate height info along y axis
    x_bar = np.zeros(L * 6)  # 6 features per x-slice
    for i in range(L):
        col = height_map[i, :]
        x_bar[i*6:(i+1)*6] = [col.mean(), col.max(), col.min(),
                                col.std(), np.sum(col > 0) / W,  # occupancy
                                item.w / L]  # relative item size
    # Similarly for y_bar and z_bar
    y_bar = ...  # analogous along x axis
    z_bar = ...  # item features + global stats
    return x_bar, y_bar, z_bar
```

For a typical grid (e.g., 12x8), this gives:
- `x_bar`: 12 * 6 = 72 features
- `y_bar`: 8 * 6 = 48 features
- `z_bar`: ~24 features (item dims, bin stats)
- Total: ~144 features (very compact)

Additionally, `compute_border_encoding()` adds information about wall proximity and `compute_location_onehot()` encodes the specific placement position being evaluated.

#### E.4.3 Selective Search

**From `find_corner_locations()` and `selective_search()` functions (line ~150+)**:

Instead of evaluating all possible (x, y, orientation) combinations, selective search only considers **corner-aligned** placements:

```python
def find_corner_locations(height_map, item):
    """Find all corner-aligned placement locations."""
    locations = []
    L, W = height_map.shape
    for orient in item.get_orientations():
        w, d, h = orient
        for x in range(L - w + 1):
            for y in range(W - d + 1):
                # Check if this position is corner-aligned
                # (adjacent to a wall or another item on at least 2 sides)
                if is_corner_aligned(height_map, x, y, w, d):
                    z = compute_z(height_map, x, y, w, d)
                    if z + h <= H:
                        locations.append(Placement(x, y, z, w, d, h))
    return locations
```

This typically reduces from ~10,000 possible placements to ~50-200 corner-aligned candidates.

**Multi-bin extension** (`selective_search_multi_bin()`, line ~200+):
```python
def selective_search_multi_bin(bins, item, buffer=None):
    """Selective search across multiple bins."""
    all_placements = []
    for bin_idx, bin_state in enumerate(bins):
        placements = find_corner_locations(bin_state.height_map, item)
        for p in placements:
            p.bin_idx = bin_idx  # tag with bin index
        all_placements.extend(placements)
    return all_placements
```

**Buffer extension** (`selective_search_with_buffer()`, line ~220+):
```python
def selective_search_with_buffer(bins, buffer, item):
    """Selective search with buffer: evaluate all (item, placement) pairs."""
    candidates = []
    for buf_item in buffer.items:
        placements = selective_search_multi_bin(bins, buf_item)
        candidates.extend([(buf_item, p) for p in placements])
    return candidates
```

#### E.4.4 Network Architecture

**From `build_packman_dqn()` function (line ~400+)**:

The network is remarkably simple -- fully connected layers only:

```
PackMan DQN:
  Input: x_bar (e.g., 432 for 72x6 grid)
  Dense(432, 144) -> ReLU
  Concat with y_bar, z_bar -> 144 + 48*6 + 24 = ~456
  Dense(456, 144) -> ReLU
  Dense(144, 24) -> ReLU
  Dense(24, 4) -> ReLU
  Dense(4, 1) -> Q-value
```

Like Tsang's DDQN, this outputs a **single Q-value per (state, action) pair**. Each candidate placement is evaluated separately.

#### E.4.5 Reward: Retroactive Terminal

**From `RewardComputer` class (line ~500+)**:

The reward is only given at the end of the episode (when the bin is full):

```python
class RewardComputer:
    def compute_terminal_reward(self, bin_state, episode_length):
        V_packed = bin_state.compute_packed_volume()
        V_used = bin_state.T_used * bin_state.L * bin_state.W * bin_state.H
        zeta = V_packed / V_used - self.tau  # tau = baseline fill rate

        # Retroactive: discount reward back through episode
        rewards = []
        for t in range(episode_length):
            r_t = self.rho ** (episode_length - t) * zeta
            rewards.append(r_t)
        return rewards  # rho = 0.9 typically
```

Where:
- `zeta = V_packed / (T_used * L * W * H) - tau` -- fill rate minus baseline
- `r_t = rho^(N-t) * zeta` -- geometrically decayed, so later placements get higher weight
- `rho = 0.9` -- decay factor
- `tau` -- baseline fill rate (e.g., from a heuristic)

This is simpler than per-step rewards but provides weaker learning signal.

#### E.4.6 WallE Heuristic Integration

**From summary**: WallE provides a heuristic score for each placement:

```
WallE Score = -alpha_1 * G_var + alpha_2 * G_high + alpha_3 * G_flush - alpha_4 * (i+j) - alpha_5 * h_{i,j}
```

Where:
- `G_var`: variance of height map in item footprint (minimize)
- `G_high`: max height of supported surface (maximize -- pack on existing items)
- `G_flush`: flush factor with walls/items (maximize)
- `(i+j)`: position bias toward corner (minimize)
- `h_{i,j}`: placement height (minimize)

WallE can serve as:
1. **Standalone baseline** -- no learning needed
2. **Tie-breaker** -- when DQN Q-values are similar, use WallE to break ties
3. **Reward shaping** -- add WallE score as bonus to DRL reward
4. **Demonstration source** -- generate expert trajectories for imitation learning

#### E.4.7 Training Configuration

| Parameter | Value |
|---|---|
| Episodes | 2000 |
| gamma | 0.75 (low -- short-horizon) |
| epsilon | linear decay 1.0 -> 0.05 |
| Batch size | 256 |
| Optimizer | SGD, lr=0.001 |
| Replay buffer | standard experience replay |
| Training time | ~4-8 hours GPU |

#### E.4.8 Key Strengths and Weaknesses

**Strengths**:
- **Simplest to implement** (2-3 weeks) -- good first baseline
- Selective search is an elegant action space reduction
- WallE heuristic provides strong baseline without any learning
- FC-only network is fast to train and easy to debug
- Pooled state is very compact

**Weaknesses**:
- Lowest fill rate among DRL approaches (73.2%)
- No explicit stability handling
- Retroactive terminal reward provides weak per-step signal
- FC-only network cannot capture spatial patterns as well as CNN
- SGD optimizer is suboptimal for RL (Adam or ACKTR preferred)
- No native multi-bin support (must be added)

---

### E.5 Approach 5: Deep-Pack 3D -- Based on Kundu et al. (2019)

**Source files**:
- `C:\Users\Louis\Downloads\stapelalgortime\python\deep_rl\deep_pack_3d_coding_ideas.py` (1030 lines, all commented out -- design document)
- `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\Deep-Pack Vision-Based 2D Online Bin Packing (Summary).md`

**Paper**: "Deep-Pack: A Vision-Based 2D Online Bin Packing Algorithm with Deep Reinforcement Learning" -- Kundu et al. 2019, IEEE.

#### E.5.1 Core Idea

Deep-Pack is originally a **2D** bin packing approach using binary images as state and Connected Component Labelling (CCL) for adjacency-based rewards. The coding ideas file extends it to 3D with a multi-channel height map and Dueling Double DQN. **Note**: all code in this file is commented out (pseudocode/design document), reflecting that this is a proposed extension rather than a validated implementation.

#### E.5.2 State Representation: 5-Channel Height Map

**From `BinState` class (commented, line ~100+)**:

```python
class BinState:
    """5-channel state tensor for 3D Deep-Pack."""
    def get_state_tensor(self):
        return np.stack([
            self.bin1_heightmap / self.H,        # Channel 0: bin 1 normalized heights
            self.bin2_heightmap / self.H,        # Channel 1: bin 2 normalized heights
            self.item_footprint_mask,             # Channel 2: binary mask of current item
            self.support_map_bin1,                # Channel 3: support ratio at each cell (bin 1)
            self.support_map_bin2,                # Channel 4: support ratio at each cell (bin 2)
        ], axis=0)  # Shape: (5, L, W)
```

This is richer than the single-channel height map used by Constrained DRL, encoding both bins and support information directly in the state.

#### E.5.3 Network Architecture: Dueling DQN

**From `DeepPack3D_QNetwork` class (commented, line ~200+)**:

Three architecture options were proposed:

**Option 1: ResNet-style** (deep but potentially overkill)
```
Input: (5, L, W)
ResBlock(5, 32) -> ResBlock(32, 64) -> ResBlock(64, 128)
AdaptiveAvgPool -> 128-dim
Split into:
  Value stream:   Linear(128, 64) -> ReLU -> Linear(64, 1) -> V(s)
  Advantage stream: Linear(128, 64) -> ReLU -> Linear(64, |A|) -> A(s,a)
Q(s,a) = V(s) + A(s,a) - mean(A(s,.))
```

**Option 2: Deep-Pack scaled up** (direct extension of 2D architecture)
```
Input: (5, L, W)
Conv2d(5, 16, 5) -> ReLU -> MaxPool
Conv2d(16, 32, 3) -> ReLU -> MaxPool
Flatten -> Linear(32*f, 256) -> ReLU -> Linear(256, |A|)
```

**Option 3: Dueling DQN (recommended)**:
```
Input: (5, L, W)
Conv2d(5, 32, 3, padding=1) -> BatchNorm -> ReLU
Conv2d(32, 64, 3, padding=1) -> BatchNorm -> ReLU
Conv2d(64, 128, 3, stride=2) -> BatchNorm -> ReLU
AdaptiveAvgPool(4,4) -> Flatten -> 128*16 = 2048
  Value:     Linear(2048, 512) -> ReLU -> Linear(512, 1)
  Advantage: Linear(2048, 512) -> ReLU -> Linear(512, |A|)
Q(s,a) = V(s) + A(s,a) - mean(A(s,.))
```

The Dueling architecture separates state value from action advantage, which helps in states where the action choice matters less (e.g., early in episode when bin is mostly empty).

#### E.5.4 Reward: Adjacency from CCL

**From the design document (line ~400+)**:

The original 2D Deep-Pack reward uses Connected Component Labelling:
```
R = cluster_size * compactness
```
Where `cluster_size` is the size of the largest connected component of packed items, and `compactness` measures how tightly items are packed.

The 3D extension proposes:
```python
def compute_reward(placement, bin_state):
    R_adj = contact_surface_area(placement, bin_state) / total_surface_area(placement)
    R_stab = support_percentage(placement, bin_state)
    R_smooth = 1.0 - height_variance(bin_state) / max_variance
    return w1 * R_adj + w2 * R_stab + w3 * R_smooth
    # w1=0.4, w2=0.3, w3=0.3
```

**Adjacency reward** `R_adj`: the ratio of contact surface area (between the placed item and existing items/walls) to the item's total surface area. Higher means more tightly packed.

#### E.5.5 Action Space: Extreme Points

From the design document (line ~500+):

Uses extreme points (similar to EMS but simpler) to generate candidate placements:
- Maintain ~50 extreme points per bin
- For each point, test 2-6 orientations
- Total action space: ~50 x 4 (avg orientations) x 2 bins = ~400 candidates
- Much smaller than Tsang's 3000 but larger than selective search's ~100

#### E.5.6 Buffer Integration

**From `BufferSelector` class (commented, line ~600+)**:
```python
class BufferSelector:
    def select_item(self, buffer, bins):
        """Evaluate all (item, placement) combinations."""
        best_q = -float('inf')
        best_item, best_placement = None, None
        for item in buffer:
            placements = self.get_extreme_point_placements(item, bins)
            for p in placements:
                state = self.encode_state(bins, item, p)
                q = self.q_network(state)
                if q > best_q:
                    best_q = q
                    best_item, best_placement = item, p
        return best_item, best_placement
```

#### E.5.7 Training Configuration

| Parameter | Value |
|---|---|
| Episodes | 500,000 |
| gamma | 0.99 |
| Replay buffer | 100,000 |
| Target network update | every 1,000 steps |
| epsilon | 1.0 -> 0.01 (exponential decay) |
| Optimizer | Adam, lr=0.0001 |

#### E.5.8 Key Strengths and Weaknesses

**Strengths**:
- Rich 5-channel state captures both bins and support info
- Adjacency reward is a unique and useful signal
- Dueling DQN architecture is well-suited for bin packing
- 2D version had real-world validation with depth camera
- Moderate action space size (~400)

**Weaknesses**:
- **All code is commented out** -- no working implementation
- Original paper is 2D only; 3D extension is untested
- 500K episodes is very long training
- No open source code
- Extreme points are less principled than EMS or MCA
- No published 3D results to benchmark against

---

### E.6 Approach 6: Decomposed Actor-Critic -- Zhao et al. (2023)

**Source files**:
- `C:\Users\Louis\Downloads\stapelalgortime\python\deep_rl\decomposed_actor_critic_coding_ideas.py` (829 lines)
- `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\Learning Practically Feasible Policies (Summary).md`

**Paper**: "Learning Practically Feasible Policies for Online 3D Bin Packing" -- Zhao et al. 2023, Science China Information Sciences.

#### E.6.1 Core Idea

The key contribution is **action space decomposition**: instead of a single network outputting a probability over all `|O| x L x W` actions (O(n^3) for orientations x positions), three separate heads output orientation, x-coordinate, and y-coordinate **autoregressively**: `o -> x -> y`. This reduces the action space from O(n^3) to O(3n) -- e.g., from 2 x 100 x 100 = 20,000 to 2 + 100 + 100 = 202 outputs.

#### E.6.2 State Representation: 6-Channel Tensor

**From `construct_state_tensor()` function (line ~500+)**:

```python
def construct_state_tensor(bin_state, item):
    """6-channel state tensor."""
    return np.stack([
        bin_state.height_map / H,           # Ch 0: normalized height
        bin_state.occupancy_map,             # Ch 1: binary occupancy (0/1)
        bin_state.support_ratio_map,         # Ch 2: support ratio at each cell
        bin_state.weight_map / max_weight,   # Ch 3: cumulative weight per cell
        item_footprint(item, L, W),          # Ch 4: item footprint mask
        accessibility_map(bin_state),        # Ch 5: top-access clearance
    ], axis=0)  # Shape: (6, L, W)
```

#### E.6.3 Network Architecture: Decomposed Heads

**From `SharedCNNEncoder`, `OrientationHead`, `XCoordinateHead`, `YCoordinateHead`, `CriticNetwork` classes (line ~50+)**:

```
SharedCNNEncoder:
  Conv2d(6, 32, 3, padding=1) -> LeakyReLU(0.2)
  Conv2d(32, 64, 3, padding=1) -> LeakyReLU(0.2)
  Conv2d(64, 64, 3, padding=1) -> LeakyReLU(0.2)
  Conv2d(64, 128, 3, padding=1) -> LeakyReLU(0.2)
  Conv2d(128, 128, 3, padding=1) -> LeakyReLU(0.2)
  AdaptiveAvgPool2d(4,4) -> Flatten -> Linear(2048, 256) -> LeakyReLU

OrientationHead:
  Input: shared_features (256) + item_features
  Linear(256+item_dim, 128) -> LeakyReLU -> Linear(128, |O|) -> Softmax
  Output: pi_o(o | s) -- probability over orientations

XCoordinateHead:
  Input: shared_features (256) + orientation_embedding
  Linear(256+orient_dim, 128) -> LeakyReLU -> Linear(128, L) -> Softmax
  Output: pi_x(x | s, o) -- probability over x-positions, CONDITIONED on chosen o

YCoordinateHead:
  Input: shared_features (256) + orientation_embedding + x_embedding
  Linear(256+orient_dim+x_dim, 128) -> LeakyReLU -> Linear(128, W) -> Softmax
  Output: pi_y(y | s, o, x) -- probability over y-positions, CONDITIONED on chosen o AND x

CriticNetwork:
  Input: shared_features (256)
  Linear(256, 128) -> LeakyReLU -> Linear(128, 1) -> V(s)
```

**Autoregressive factorization**:
```
pi(o, x, y | s) = pi_o(o | s) * pi_x(x | s, o) * pi_y(y | s, o, x)
```

This is the key insight: the joint policy factorizes autoregressively, so each head only needs to output probabilities over a 1D slice.

#### E.6.4 5-Head Extension for Thesis

**From `ItemSelectionHead`, `BinSelectionHead`, and `DecomposedBinPackingPolicy` classes (line ~300+)**:

For the thesis setting (buffer + 2 bins), extend to 5 heads:

```
item -> bin -> o -> x -> y

ItemSelectionHead:
  Input: shared_features + buffer_item_embeddings
  MultiheadAttention(embed_dim=128, num_heads=4) over buffer items
  Linear(128, buffer_size) -> Softmax
  Output: pi_item(i | s) -- which item from buffer to place

BinSelectionHead:
  Input: shared_features + selected_item_embedding
  Linear(256+item_dim, 128) -> LeakyReLU -> Linear(128, k) -> Softmax
  Output: pi_bin(b | s, i) -- which bin to place in

OrientationHead: pi_o(o | s, i, b)
XCoordinateHead: pi_x(x | s, i, b, o)
YCoordinateHead: pi_y(y | s, i, b, o, x)
```

Full joint policy:
```
pi(i, b, o, x, y | s) = pi_item(i|s) * pi_bin(b|s,i) * pi_o(o|s,i,b) * pi_x(x|s,i,b,o) * pi_y(y|s,i,b,o,x)
```

This scales to O(B + k + |O| + L + W) = O(10 + 2 + 6 + 60 + 40) = **118 outputs** instead of O(10 * 2 * 6 * 60 * 40) = **288,000**.

#### E.6.5 Reward: Far-to-Near Collision-Free

**From `compute_reward()` and `compute_safe_volume()` functions (line ~550+)**:

```python
def compute_reward(placement, bin_state, alpha=0.6, beta=0.4):
    """Far-to-near collision-free reward."""
    R_vol = placement.volume / bin_state.total_volume  # volume ratio
    R_safe = compute_safe_volume(placement, bin_state) / bin_state.total_volume
    return alpha * R_vol + beta * R_safe

def compute_safe_volume(placement, bin_state):
    """Volume that is safely accessible from above after this placement."""
    # For each column in the height map, check if a gripper can reach
    # from the top without colliding with placed items
    safe_vol = 0
    for x in range(L):
        for y in range(W):
            h = bin_state.height_map[x, y]
            # Safe volume above this column
            safe_vol += (H - h) * cell_area
    return safe_vol
```

The "far-to-near" principle: pack from the far side of the bin toward the near side (closest to the robot), ensuring items remain accessible for a top-loading robot arm.

#### E.6.6 Stacking Tree for O(N log N) Stability

**From summary**: The stacking tree data structure achieves:
- O(N log N) stability evaluation (vs O(N^2) naive)
- 99.9% accuracy compared to Bullet physics simulator
- Tracks parent-child support relationships in a tree
- Each node stores: item, support polygon, center of mass, supported weight

This is the **fastest analytical stability check** among all approaches and should be adopted regardless of which DRL architecture is chosen.

#### E.6.7 Real Robot Results

From summary:
| Metric | BPP-1 | BPP-k |
|---|---|---|
| Fill rate | 71.2% | 77.9% |
| Stability | 100% | 100% |
| Collision | 0-2% | 0-2% |
| Human comparison | AI 70.4% vs human 56.3% |

**100% stability** is the standout result -- the decomposed policy with far-to-near reward achieves perfect stability in real-world conditions.

#### E.6.8 Training Details

| Parameter | Value |
|---|---|
| Training time | ~12 hours on NVIDIA Titan V (100x100 grid) |
| Resolutions | 10x10 (70.1%), 50x50 (72.6%), 100x100 (71.3%) |
| Optimizer | ACKTR (or Adam for simpler setup) |
| Inference time | <10ms per step |

Note: 50x50 slightly outperforms 100x100, suggesting diminishing returns from finer discretization.

#### E.6.9 Loss Function

**From `DecomposedLoss` class (line ~650+)**:

```python
class DecomposedLoss:
    def compute(self, log_probs, values, returns, advantages, masks):
        # Actor loss: sum of log-probs across all heads
        L_actor = -(log_prob_item + log_prob_bin + log_prob_o +
                     log_prob_x + log_prob_y) * advantages

        # Critic loss
        L_critic = (values - returns) ** 2

        # Infeasibility penalty
        E_inf = -(log_probs * masks.log()).sum()

        # Entropy bonus (sum across all heads)
        E_entropy = -(probs * log_probs).sum()

        return alpha * L_actor + beta * L_critic + omega * E_inf - psi * E_entropy
```

#### E.6.10 Key Strengths and Weaknesses

**Strengths**:
- **Elegant action decomposition**: O(3n) vs O(n^3) -- huge scalability win
- **100% stability** on real robot -- best stability among all approaches
- Natural extension to 5 heads for buffer + multi-bin
- Stacking tree is the fastest stability check (O(N log N))
- Fast inference (<10ms)
- Same research group as PCT and Constrained DRL

**Weaknesses**:
- Fill rate is lower than PCT (72.6% vs 86.0% online)
- Autoregressive sampling means errors cascade (bad orientation -> bad x -> bad y)
- No open source code
- 6-channel state is more expensive to compute than single-channel height map
- Not tested with explicit buffer/lookahead planning

---

## F. DRL Strategy Selection Guide

### F.1 Decision Tree for Thesis

```
START
  |
  v
Q1: How much implementation time available?
  |
  |-- < 4 weeks: Go to PackMan DQN (E.4) as baseline
  |                Then optionally upgrade to Tsang DDQN (E.3)
  |
  |-- 4-8 weeks: Go to Q2
  |
  |-- 8-12 weeks: Consider PCT (E.1) for maximum performance
  |
  v
Q2: What is the primary optimization target?
  |
  |-- Fill rate only: Tsang DDQN (E.3) -- native dual-bin, MCA placements
  |
  |-- Fill rate + stability: Go to Q3
  |
  |-- Constraint satisfaction: Constrained DRL (E.2) -- CMDP formulation
  |
  v
Q3: How important is action space scalability?
  |
  |-- Critical (high-res grid): Decomposed AC (E.6) -- O(3n) action space
  |
  |-- Moderate (60x40 grid): Tsang DDQN (E.3) or Constrained DRL (E.2)
  |
  |-- Not concerned: Any approach works
```

### F.2 Recommended Strategy for Thesis

**Primary recommendation: Tsang DDQN (E.3) as foundation, enhanced with components from other approaches.**

Rationale:
1. **Closest match**: Only approach designed for dual-bin + buffer + online + 3D
2. **Open source**: Working codebase available (needs TF -> PyTorch port)
3. **MCA space management**: High-quality candidate placements
4. **Moderate complexity**: 3-5 weeks implementation
5. **Clear enhancement path**: Add stability from E.2/E.6, decomposition from E.6

**Enhancement roadmap**:
1. **Week 1-2**: Port DeepPack3D from TF to PyTorch, get baseline working
2. **Week 3**: Add three-tiered stability from Constrained DRL (E.2)
3. **Week 4**: Add enhanced reward function (compactness + stability + height)
4. **Week 5**: Integrate buffer with MCTS or greedy selection
5. **Week 6+**: Optional -- add decomposed heads from E.6 if action space is too large

**Alternative recommendation: Decomposed Actor-Critic (E.6) if stability is paramount.**

The 100% stability result on real robot is compelling. The 5-head decomposition naturally fits the thesis setting. However, no open source code means building from scratch.

### F.3 Expected Performance Ranges

| Configuration | Expected Fill Rate | Expected Stability |
|---|---|---|
| Tsang DDQN baseline (ported) | 75-80% | ~90% (with LBCP 80%) |
| + Enhanced stability | 70-78% | ~98% |
| + Buffer MCTS | 80-87% | ~95% |
| + Decomposed heads | 78-85% | ~98% |
| PCT + ToP (upper bound) | 85-93% | ~95% |

---

## G. Shared DRL Components (Reusable Across Approaches)

### G.1 State Representation Components

#### G.1.1 Height Map Computation

**Used by**: All approaches except PCT (which uses tree representation)
**Source**: `compute_height_map()` in `coding_ideas_tsang2025_ddqn_dual_bin.py` (line ~700+)

```python
def compute_height_map(bin_state, resolution=2):
    """Compute 2D height map from 3D bin state.

    Args:
        bin_state: current bin with placed items
        resolution: cm per grid cell (default 2cm for EUR pallet)

    Returns:
        height_map: (L//res, W//res) array of max heights
    """
    L_grid = bin_state.L // resolution
    W_grid = bin_state.W // resolution
    height_map = np.zeros((L_grid, W_grid), dtype=np.float32)

    for item in bin_state.placed_items:
        x_start = item.x // resolution
        x_end = (item.x + item.w) // resolution
        y_start = item.y // resolution
        y_end = (item.y + item.d) // resolution
        z_top = item.z + item.h
        height_map[x_start:x_end, y_start:y_end] = np.maximum(
            height_map[x_start:x_end, y_start:y_end], z_top / bin_state.H
        )
    return height_map
```

#### G.1.2 Multi-Channel State Tensor

**Used by**: Deep-Pack 3D (5 channels), Decomposed AC (6 channels)
**Recommended thesis state**: 4-channel tensor

```python
def compute_state_tensor(bin1, bin2, item, resolution=2):
    """4-channel state tensor for dual-bin packing."""
    hm1 = compute_height_map(bin1, resolution)  # (60, 40) for EUR pallet
    hm2 = compute_height_map(bin2, resolution)
    item_mask = np.zeros_like(hm1)
    # Mark where current item could fit (footprint)
    support1 = compute_support_map(bin1, resolution)
    return np.stack([hm1, hm2, item_mask, support1], axis=0)  # (4, 60, 40)
```

### G.2 Action Space Management

#### G.2.1 EMS (Empty Maximal Spaces)

**Used by**: PCT (default scheme)
**Source**: `EMSManager` class in `pct_coding_ideas.py` (line ~95+)

Best for tree-based state representations. O(|E|) complexity where |E| is number of EMS (typically 20-80).

#### G.2.2 MCA (Maximal Cuboids Algorithm)

**Used by**: Tsang DDQN
**Source**: `MaximalCuboidsAlgorithm` class in `coding_ideas_tsang2025_ddqn_dual_bin.py` (line ~150+)

Best for value-based methods (DQN). Generates high-quality cuboid placements. Slightly more expensive than EMS but produces larger empty spaces.

#### G.2.3 Corner Points / Selective Search

**Used by**: PackMan DQN
**Source**: `find_corner_locations()`, `selective_search()` in `packman_dqn_coding_ideas.py` (line ~150+)

Simplest and fastest. Good for prototyping but may miss some good placements that are not corner-aligned.

#### G.2.4 Extreme Points

**Used by**: Deep-Pack 3D
**Source**: Design document in `deep_pack_3d_coding_ideas.py` (line ~500+)

Middle ground between corner points and EMS. Easier to implement than EMS but generates more candidates than corner points.

### G.3 Reward Components

#### G.3.1 Volume-Based Reward

**Used by**: PCT, Constrained DRL, Decomposed AC
```python
R_vol = item.volume / bin.total_volume
```

#### G.3.2 Compactness Reward

**Used by**: Tsang DDQN, Deep-Pack 3D
```python
R_compact = packed_volume / bounding_box_volume  # of all placed items
```

#### G.3.3 Stability Reward

**Used by**: Constrained DRL, Tsang DDQN (LBCP), Decomposed AC (V_safe)
```python
R_stab = support_area / footprint_area  # >= threshold (0.8 recommended)
```

#### G.3.4 Adjacency Reward (from Deep-Pack)

**Used by**: Deep-Pack 3D (proposed)
**Source**: Design document in `deep_pack_3d_coding_ideas.py` (line ~400+)
```python
R_adj = contact_surface_area / total_surface_area  # higher = tighter packing
```

#### G.3.5 Pyramidal Reward

**Used by**: Tsang DDQN, PCT (as constraint C4)
```python
R_pyramid = 1.0 if item_weight >= weight_of_items_above else 0.0
```

#### G.3.6 Recommended Composite Reward for Thesis

Combining the best elements:
```python
def thesis_reward(placement, bin_state):
    R_vol = placement.volume / bin_state.total_volume  # 0.3 weight
    R_stab = support_ratio(placement, bin_state)        # 0.3 weight (threshold 0.8)
    R_compact = compactness(placement, bin_state)       # 0.2 weight
    R_height = 1.0 - placement.z / bin_state.H          # 0.1 weight
    R_pyramid = pyramidal_check(placement, bin_state)    # 0.1 weight

    # Feasibility check (hard constraint, not reward)
    if R_stab < 0.6:
        return 0.0  # reject unstable placements entirely

    return 0.3*R_vol + 0.3*R_stab + 0.2*R_compact + 0.1*R_height + 0.1*R_pyramid
```

### G.4 Stability Checking

#### G.4.1 Base Support Ratio (Simplest)

**Used by**: Tsang DDQN (LBCP), Constrained DRL
```python
def support_ratio(placement, bin_state):
    """Percentage of item base that is supported."""
    footprint = placement.w * placement.d
    if placement.z == 0:
        return 1.0  # floor always supports
    support = 0
    for x in range(placement.x, placement.x + placement.w):
        for y in range(placement.y, placement.y + placement.d):
            if bin_state.height_map[x, y] >= placement.z - epsilon:
                support += 1
    return support / footprint
```

#### G.4.2 Quasi-Static Equilibrium (Recommended)

**Used by**: PCT (constraint C3)
**Source**: `QuasiStaticStabilityChecker` in `pct_coding_ideas.py` (line ~1750+)
```python
class QuasiStaticStabilityChecker:
    """Check if center of mass is within support polygon."""
    def is_stable(self, item, support_items):
        com = self.compute_center_of_mass(item)
        polygon = self.compute_support_polygon(support_items)
        return point_in_polygon(com[:2], polygon)
```

#### G.4.3 Stacking Tree (Fastest)

**Used by**: Decomposed AC
**Source**: Described in summary of Zhao 2023 paper
- O(N log N) complexity
- 99.9% accuracy vs Bullet physics
- **Recommended for thesis** as the primary analytical stability check

#### G.4.4 Physics Simulation (Ground Truth)

**Used by**: PCT (validation only)
**Source**: `PhysicsStabilityVerifier` in `pct_coding_ideas.py` (line ~1800+)
- Uses PyBullet or Isaac Gym
- Simulate 2 seconds, check displacement < 1cm
- Too slow for training (use for final evaluation only)

### G.5 Training Infrastructure

#### G.5.1 Experience Replay

**Used by**: All DQN-based approaches (Tsang, PackMan, Deep-Pack)
**Source**: `ReplayBuffer` in `packman_dqn_coding_ideas.py` (line ~600+)
```python
class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))
```

Consider using **Prioritized Experience Replay** (PER) for better sample efficiency:
```python
# Priority = |TD_error| + epsilon
# P(i) = priority_i^alpha / sum(priority_j^alpha)
```

#### G.5.2 ACKTR Optimizer

**Used by**: PCT, Constrained DRL, Decomposed AC
**Source**: `KFACOptimizer` in `constrained_drl_bin_packing.py` (line ~1200+)

ACKTR uses K-FAC (Kronecker-Factored Approximate Curvature) to approximate the Fisher information matrix for natural gradient updates. This is the most sample-efficient optimizer but also the hardest to implement.

**Fallback**: Use PPO (from `ppo_training_pseudocode()` in `pct_coding_ideas.py`) for a simpler alternative with ~5-10% less sample efficiency.

#### G.5.3 Gymnasium Environment Wrapper

**Source**: `BinPackingEnv` in `pct_coding_ideas.py` (line ~1850+)
```python
class BinPackingEnv(gym.Env):
    """Standard Gymnasium wrapper for bin packing."""
    def __init__(self, config):
        self.observation_space = ...
        self.action_space = ...

    def reset(self):
        """Reset bins, generate new item sequence."""
        ...

    def step(self, action):
        """Place item, update state, compute reward."""
        ...
        return obs, reward, terminated, truncated, info
```

### G.6 Evaluation Metrics

All approaches should report:

| Metric | Formula | Target |
|---|---|---|
| Fill rate (space utilization) | `V_packed / V_bin` | >80% |
| Stability rate | `N_stable / N_placed` | >95% |
| Constraint violation rate | `N_violations / N_placed` | <2% |
| Items per bin | `N_placed / N_bins_used` | maximize |
| Inference time | ms per placement decision | <50ms |
| Training time | hours to converge | <24h |

---

## H. Novel DRL Ideas and Cross-Pollination

### H.1 Hybrid Architecture: Decomposed DDQN with MCA

**Combines**: Decomposed AC (E.6) action decomposition + Tsang DDQN (E.3) MCA space management

Instead of decomposing into grid positions (o, x, y), decompose into:
1. **Item head**: select item from buffer (attention-based)
2. **Bin head**: select bin (2 outputs)
3. **Cuboid head**: select maximal cuboid from MCA list (variable size, use pointer)
4. **Orientation head**: select orientation (2-6 outputs)

This avoids grid discretization entirely (MCA gives exact placements) while keeping the decomposition benefit.

**Source classes to combine**:
- `ItemSelectionHead` from `decomposed_actor_critic_coding_ideas.py` (line ~300+)
- `BinSelectionHead` from `decomposed_actor_critic_coding_ideas.py` (line ~350+)
- `MaximalCuboidsAlgorithm` from `coding_ideas_tsang2025_ddqn_dual_bin.py` (line ~150+)

### H.2 Feasibility Mask + Decomposed Heads

**Combines**: Constrained DRL (E.2) mask prediction + Decomposed AC (E.6) heads

Add a mask predictor per head:
```python
# Orientation mask: which orientations are feasible given current state
mask_o = orientation_mask_predictor(shared_features)
# X mask: which x-positions are feasible given chosen orientation
mask_x = x_mask_predictor(shared_features, orientation)
# Y mask: which y-positions are feasible given chosen orientation and x
mask_y = y_mask_predictor(shared_features, orientation, x)
```

This propagates feasibility information through the autoregressive chain, preventing early heads from choosing infeasible options that cascade to later heads.

**Source classes to combine**:
- `MaskPredictor` from `constrained_drl_bin_packing.py` (line ~200+)
- `OrientationHead`, `XCoordinateHead`, `YCoordinateHead` from `decomposed_actor_critic_coding_ideas.py` (line ~100+)

### H.3 PCT-Guided Buffer Selection

**Combines**: PCT (E.1) tree representation + any DQN approach for placement

Use PCT's tree structure to **score buffer items** by the quality of their best placement, then use a simpler DQN for the actual placement:

```python
def pct_guided_buffer_selection(buffer, pct_tree, dqn):
    """Use PCT to rank buffer items, DQN for final placement."""
    scores = []
    for item in buffer:
        # PCT quickly evaluates best leaf for this item
        leaves = pct_tree.get_feasible_leaves(item)
        best_leaf_score = max(pct_tree.score_leaf(l, item) for l in leaves)
        scores.append(best_leaf_score)

    # Select top-k items based on PCT scores
    top_items = select_top_k(buffer, scores, k=3)

    # Use DQN to make final placement decision among top items
    return dqn.select_best_placement(top_items, bins)
```

### H.4 Curriculum Learning Pipeline

**Combines insights from all approaches**:

```
Phase 1 (Weeks 1-2): Single bin, no buffer, no stability
  - Focus: learn basic packing policy
  - Stability: relaxed (60% support)
  - Fill rate target: >65%

Phase 2 (Weeks 3-4): Single bin, buffer=5, no stability
  - Focus: learn buffer management
  - Add: buffer selection head or MCTS
  - Fill rate target: >75%

Phase 3 (Weeks 5-6): Dual bin, buffer=5, relaxed stability
  - Focus: learn bin selection
  - Stability: standard (80% support)
  - Fill rate target: >70%

Phase 4 (Weeks 7-8): Dual bin, buffer=10, strict stability
  - Focus: full thesis configuration
  - Stability: strict (95% support)
  - Fill rate target: >75%

Phase 5 (Weeks 9+): Fine-tuning and evaluation
  - EUR pallet dimensions
  - Real item distributions
  - Ablation studies
```

### H.5 Spatial Ensemble for Cross-Bin Evaluation

**From**: `SpatialEnsemble` class in `pct_coding_ideas.py` (line ~330+)

When deciding which bin to place in, rank-normalize Q-values/scores across bins:

```python
def spatial_ensemble_selection(q_values_bin1, q_values_bin2):
    """Rank-normalize Q-values across bins for fair comparison."""
    all_q = np.concatenate([q_values_bin1, q_values_bin2])
    ranks = scipy.stats.rankdata(all_q) / len(all_q)

    n1 = len(q_values_bin1)
    ranks_bin1 = ranks[:n1]
    ranks_bin2 = ranks[n1:]

    best_idx = np.argmax(ranks)
    if best_idx < n1:
        return 'bin1', q_values_bin1[best_idx]
    else:
        return 'bin2', q_values_bin2[best_idx - n1]
```

This handles the problem where Q-values from different bins may be on different scales.

### H.6 Multi-Scale Training Distribution

**From**: `MultiScaleItemSampler` in `pct_coding_ideas.py` (line ~420+)

Train on items from multiple size distributions to improve generalization:

```python
class MultiScaleItemSampler:
    def __init__(self):
        self.distributions = {
            'small': UniformDist(min_dim=5, max_dim=15),   # small parcels
            'medium': UniformDist(min_dim=15, max_dim=35),  # standard boxes
            'large': UniformDist(min_dim=35, max_dim=60),   # large items
        }
        self.weights = [0.4, 0.4, 0.2]  # sample probability per distribution

    def sample_item(self):
        dist = random.choices(list(self.distributions.values()),
                              weights=self.weights)[0]
        return dist.sample()
```

### H.7 WallE as Pre-Training Signal

**From**: PackMan summary (Verma et al. 2020)

Use WallE heuristic to generate expert demonstrations for pre-training:

```python
# Phase 1: Generate WallE demonstrations
demonstrations = []
for episode in range(1000):
    state = env.reset()
    while not done:
        action = walle_heuristic(state)
        next_state, reward, done, info = env.step(action)
        demonstrations.append((state, action, reward, next_state))
        state = next_state

# Phase 2: Pre-train DRL policy via behavioral cloning
for batch in DataLoader(demonstrations, batch_size=256):
    states, actions, _, _ = batch
    predicted_actions = policy(states)
    loss = cross_entropy(predicted_actions, actions)
    loss.backward()
    optimizer.step()

# Phase 3: Fine-tune with RL
# (standard RL training, but policy already starts near WallE level)
```

### H.8 Transfer Learning Between Approaches

Train a simpler approach first, then transfer knowledge to a more complex one:

1. Train **PackMan DQN** (2-3 weeks) to get a working baseline
2. Use PackMan's learned Q-values to initialize **Tsang DDQN**'s network
3. Fine-tune Tsang DDQN with MCA placements (leverages PackMan's learned packing intuition)

Alternatively:
1. Train **Constrained DRL** to learn feasibility masks
2. Export trained mask predictor
3. Use it in **Decomposed AC** as pre-trained mask per head

### H.9 Attention-Based Bin State Encoding

Instead of CNN on height map, use attention over placed items:

```python
class AttentionBinEncoder(nn.Module):
    """Encode bin state using attention over placed items."""
    def __init__(self, item_dim=9, embed_dim=128, num_heads=4):
        self.item_encoder = nn.Linear(item_dim, embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.bin_query = nn.Parameter(torch.randn(1, embed_dim))

    def forward(self, placed_items):
        # placed_items: (N, 9) -- [x,y,z,w,h,d,weight,fragility,order]
        item_embs = self.item_encoder(placed_items)  # (N, 128)
        # Self-attention among placed items
        attended, _ = self.self_attention(item_embs, item_embs, item_embs)
        # Global bin embedding via cross-attention with learnable query
        bin_emb, _ = nn.MultiheadAttention(128, 4)(
            self.bin_query.unsqueeze(0), attended.unsqueeze(0), attended.unsqueeze(0)
        )
        return bin_emb.squeeze()  # (128,)
```

This avoids discretization loss from height maps and scales naturally with number of items. Similar in spirit to PCT's GAT encoding but more general.

### H.10 Reward Shaping with Hindsight

After each episode, recompute rewards with knowledge of the full sequence:

```python
def hindsight_reward_shaping(episode_trajectory, final_fill_rate):
    """Augment per-step rewards with hindsight information."""
    for t, (state, action, reward, next_state) in enumerate(episode_trajectory):
        # Original reward
        r_original = reward
        # Hindsight: how much did this step contribute to final fill rate?
        r_hindsight = marginal_contribution(t, episode_trajectory, final_fill_rate)
        # Combined
        episode_trajectory[t].reward = 0.7 * r_original + 0.3 * r_hindsight
    return episode_trajectory
```

This provides stronger learning signal than pure per-step rewards, without the sparsity problem of terminal-only rewards.

---

## Summary of Key File Paths and Classes

### Python Coding Ideas Files

| File | Path | Lines | Key Classes |
|---|---|---|---|
| PCT | `C:\Users\Louis\Downloads\stapelalgortime\python\deep_rl\pct_coding_ideas.py` | ~2899 | `PackingConfigurationTree`, `EMSManager`, `MCTSNode`, `TwoBinPCTManager`, `BufferManager`, `SpatialEnsemble`, `ConstraintRewards`, `QuasiStaticStabilityChecker`, `PhysicsStabilityVerifier`, `BinPackingEnv`, `GeneralizationBenchmark`, `MultiScaleItemSampler` |
| Constrained DRL | `C:\Users\Louis\Downloads\stapelalgortime\python\deep_rl\constrained_drl_bin_packing.py` | 1454 | `BinState`, `OnlineBPPEnvironment`, `StateCNN`, `Actor`, `Critic`, `MaskPredictor`, `ConstrainedPackingNetwork`, `ConstrainedDRLTrainer`, `MCTSPermutationSearch`, `TwoBoundedBufferEnvironment`, `DualBinConstrainedNetwork`, `KFACOptimizer` |
| Tsang DDQN | `C:\Users\Louis\Downloads\stapelalgortime\python\deep_rl\coding_ideas_tsang2025_ddqn_dual_bin.py` | 1747 | `MaximalCuboidsAlgorithm`, `BinReplacementStrategy`, `OnlineDualBinPackingEnv`, `EURPalletConfig`, `LBCPStabilityChecker`, `AdaptedDualPalletSystem` |
| PackMan DQN | `C:\Users\Louis\Downloads\stapelalgortime\python\deep_rl\packman_dqn_coding_ideas.py` | 896 | `Box`, `Placement`, `Container`, `Buffer`, `RewardComputer`, `ReplayBuffer` |
| Deep-Pack 3D | `C:\Users\Louis\Downloads\stapelalgortime\python\deep_rl\deep_pack_3d_coding_ideas.py` | 1030 | `DeepPack3D_QNetwork` (commented), `BufferSelector` (commented) |
| Decomposed AC | `C:\Users\Louis\Downloads\stapelalgortime\python\deep_rl\decomposed_actor_critic_coding_ideas.py` | 829 | `SharedCNNEncoder`, `OrientationHead`, `XCoordinateHead`, `YCoordinateHead`, `CriticNetwork`, `ItemSelectionHead`, `BinSelectionHead`, `DecomposedBinPackingPolicy`, `DecomposedLoss` |

### Summary Files

| Paper | Path |
|---|---|
| PCT (Zhao 2022/2025) | `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\Deliberate Planning of 3D Bin Packing on Packing Configuration Trees (Summary).md` |
| Constrained DRL (Zhao 2021) | `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\Online 3D Bin Packing with Constrained DRL (Summary).md` |
| Tsang DDQN (Tsang 2025) | `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\A deep reinforcement learning approach for online and concurrent 3D bin (Summary).md` |
| PackMan DQN (Verma 2020) | `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\A Generalized Reinforcement Learning Algorithm for Online 3D Bin-Packing (Summary).md` |
| Deep-Pack (Kundu 2019) | `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\Deep-Pack Vision-Based 2D Online Bin Packing (Summary).md` |
| Decomposed AC (Zhao 2023) | `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\Learning Practically Feasible Policies (Summary).md` |

---

*This section was synthesized from 6 Python coding ideas files and 6 paper summary files covering the Deep RL landscape for 3D bin packing, tailored to the thesis context: semi-online, buffer 5-10, k=2 bounded space (2 EUR pallets), maximize fill rate + stability, Python + PyTorch.*
