# MASTER IMPLEMENTATION GUIDE -- SECTIONS M, N, O, P
## Hybrid Systems + Theoretical Foundations + Complete Integrated Architectures

**Context:** Semi-online 3D bin packing, buffer 5-10 items, k=2 bounded space (2 active pallets), maximize fill rate + stability. Python + PyTorch thesis project.

**Date compiled:** 2026-02-18

---

## M. HYBRID & HYPER-HEURISTIC APPROACHES

This is the section of the thesis with the highest novelty potential. The overview knowledge base (Section 14) explicitly identifies that **"no selective hyper-heuristic has been applied to 3D packing problems"** (Research Gap 3). Every approach described here contributes to filling that gap.

---

### M1. Selective Hyper-Heuristic (Research Gap 3 -- NOVEL CONTRIBUTION)

**THIS IS A PRIMARY THESIS CONTRIBUTION.** No selective hyper-heuristic exists for 3D online bin packing in the literature.

#### M1.1 Core Concept

A selective hyper-heuristic (HH) maintains a **portfolio of low-level placement heuristics** and uses a **learned selection policy** to choose which heuristic to apply at each packing step. Unlike a single heuristic, the HH adapts its strategy based on the current bin state, buffer composition, and packing phase.

#### M1.2 The Heuristic Portfolio

The portfolio consists of placement rules drawn from the literature (Overview KB Section 10.5). Each is implemented as a scoring function over candidate positions:

| # | Heuristic | Source | Strength | Key File |
|---|-----------|--------|----------|----------|
| 1 | DBLF (Deepest-Bottom-Left-Fill) | Karabulut & Inceoglu 2004 | Systematic filling from back-bottom-left | `feasibility_guided_packing_coding_ideas.py` -> `score_dblf()` |
| 2 | Corner Distances | Zhu & Lim 2012 | Consolidation near bin corners | `feasibility_guided_packing_coding_ideas.py` -> `score_corner_distances()` |
| 3 | DFTRC (Distance to Front-Top-Right Corner) | Goncalves & Resende 2013 | Preserves open space at front-top-right | `feasibility_guided_packing_coding_ideas.py` -> `score_dftrc()` |
| 4 | Back Bottom | Ramos et al. 2016 | Robot-friendly far-to-near placement | `feasibility_guided_packing_coding_ideas.py` -> `score_back_bottom()` |
| 5 | Stability Score | Verma et al. 2020 (WallE) | Flat surface creation, support maximization | `feasibility_guided_packing_coding_ideas.py` -> `score_stability_aware()` |
| 6 | Best Match First | Li & Zhang 2015 | Tightest fit to remaining spaces | `feasibility_guided_packing_coding_ideas.py` -> `PlacementRule.BEST_MATCH` |
| 7 | WallE composite score | Verma et al. 2020 | Weighted combination of 8 sub-scores | `walle_packman_hybrid_coding_ideas.py` -> `PackingStrategy.WALLE` |
| 8 | PackMan DQN | Verma et al. 2020 | Learned spatial placement via DQN | `walle_packman_hybrid_coding_ideas.py` -> `PackingStrategy.PACKMAN_DQN` |
| 9 | Floor Building | Custom | Minimizes max height, smooth surfaces | `walle_packman_hybrid_coding_ideas.py` -> `PackingStrategy.FLOOR_BUILD` |
| 10 | Best Fit by Volume | Classic BPP | Minimizes wasted space per placement | `walle_packman_hybrid_coding_ideas.py` -> `PackingStrategy.BEST_FIT_VOLUME` |

**Full list reference:** The 160-heuristic framework from `C:\Users\Louis\Downloads\stapelalgortime\python\heuristics\coding_ideas_160_heuristic_framework.py` provides 160 combinations of (item selection x placement rule x bin routing). The selective HH operates over a curated subset of these.

#### M1.3 State Features for Selection Decisions

From `walle_packman_hybrid_coding_ideas.py` -> `compute_state_features()`:

```python
def compute_state_features(containers, buffer_boxes, step_number, total_steps_estimate):
    features = []
    # 1. Packing phase: step_number / total_steps_estimate (0 to 1)
    # 2. Average surface roughness (mean gradient of heightmap, normalized)
    # 3. Average fill fraction across active bins
    # 4. Buffer diversity (coefficient of variation of box volumes)
    # 5. Fill imbalance between the two active bins
    # 6. Average box size relative to remaining capacity
    return np.array(features, dtype=np.float32)  # shape: (6,)
```

Additional features from `feasibility_guided_packing_coding_ideas.py` -> `SelectiveHyperHeuristic.select_rule_feature_based()`:
- `utilization` -- current bin fill fraction
- `height_variance` -- heightmap surface variance (flat vs. uneven)
- `item_size_ratio` -- item volume / bin volume

#### M1.4 Three Training Approaches

All three are implemented in `walle_packman_hybrid_coding_ideas.py` -> `HyperHeuristicSelector`:

**Option A: Rule-Based (no learning, baseline)**
```python
def select_rule_based(self, features: np.ndarray) -> PackingStrategy:
    phase, roughness, fill, diversity = features[0], features[1], features[2], features[3]
    if phase < 0.3:
        return PackingStrategy.WALLE           # Early: stable foundation
    elif roughness > 0.3:
        return PackingStrategy.FLOOR_BUILD     # Rough surface: smooth it
    elif fill > 0.7 and diversity > 0.5:
        return PackingStrategy.PACKMAN_DQN     # Late + diverse: learned
    elif fill > 0.7:
        return PackingStrategy.BEST_FIT_VOLUME # Late + uniform: systematic
    else:
        return PackingStrategy.WALLE           # Default: best general heuristic
```

**Option B: Tabular Q-Learning**
- State: 6 features discretized into 5 bins each -> 5^6 = 15,625 states
- Action: select one of 5-10 strategies
- Update: standard Q-learning with epsilon-greedy exploration
- Key methods: `select_q_learning()`, `update_q_learning()`, `_discretize()`
- Learning rate: 0.1, gamma: 0.9, epsilon: 0.1

**Option C: DQN Selector**
```python
class StrategyDQN(nn.Module):
    def __init__(self, feature_count=6, n_strategies=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_count, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, n_strategies)
        )
```
Very small network (< 1000 parameters) since the meta-decision space is only 6-10 options.

#### M1.5 UCB-Based Selection (Alternative)

From `feasibility_guided_packing_coding_ideas.py` -> `SelectiveHyperHeuristic.select_rule()`:

```python
def select_rule(self, bin_state_features):
    for rule in self.rules:
        attempts = self.rule_attempts[rule]
        if attempts == 0:
            return rule  # Explore unexplored rules first
        success_rate = self.rule_successes[rule] / attempts
        exploration = self.exploration_weight * np.sqrt(np.log(total_attempts) / attempts)
        ucb = success_rate + exploration
        # Select rule with highest UCB
```

This is the classic Upper Confidence Bound bandit, balancing exploitation (best historical rule) with exploration (try under-tested rules).

#### M1.6 Step-Level Reward for Selector Training

From `walle_packman_hybrid_coding_ideas.py` -> `compute_step_reward_for_selector()`:

```python
reward = (0.4 * fill_improvement       # How much fill rate increased
        + 0.3 * stability_score         # Support ratio of placement
        + 0.2 * roughness_change        # Did surface get smoother?
        + 0.1 * vol_eff)               # Volume efficiency of claimed space
```

This per-step reward enables the Q-learning selector to learn which strategy works best in which state, unlike the retrospective terminal rewards used by PackMan.

#### M1.7 Key Classes and Files

| Class/Function | File | Purpose |
|---------------|------|---------|
| `HyperHeuristicSelector` | `python/hybrid_heuristic_ml/walle_packman_hybrid_coding_ideas.py` | Main selector with 3 modes |
| `HybridAgent` | `python/hybrid_heuristic_ml/walle_packman_hybrid_coding_ideas.py` | Complete hybrid agent |
| `SelectiveHyperHeuristic` | `python/hybrid_heuristic_ml/feasibility_guided_packing_coding_ideas.py` | UCB-based selector |
| `HybridPackingSystem` | `python/hybrid_heuristic_ml/feasibility_guided_packing_coding_ideas.py` | Full system with routing |
| `PredictionGuidedHeuristicSelector` | `python/hybrid_heuristic_ml/prediction_hybrid_framework_IDEAS.py` | Prediction-quality selector |
| `PackingStrategy` enum | `python/hybrid_heuristic_ml/walle_packman_hybrid_coding_ideas.py` | Strategy enumeration |
| `PlacementRule` enum | `python/hybrid_heuristic_ml/feasibility_guided_packing_coding_ideas.py` | Rule enumeration |
| `compute_state_features()` | `python/hybrid_heuristic_ml/walle_packman_hybrid_coding_ideas.py` | Feature extraction |
| `compute_step_reward_for_selector()` | `python/hybrid_heuristic_ml/walle_packman_hybrid_coding_ideas.py` | Step reward computation |

#### M1.8 Research Gap Addressed

This selective HH simultaneously addresses multiple gaps:
- **Gap 3:** First selective HH for 3D-PPs (primary)
- **Gap 4:** Operates in k=2 bounded space (the selector chooses bin routing strategy too)
- **Gap 5:** ML (Q-learning/DQN) integrated with practical stability constraints
- **Gap 6:** Semi-online with buffer (selector adapts to buffer contents)

---

### M2. Hierarchical Framework (Lee & Nam 2025)

**Source:** `C:\Users\Louis\Downloads\stapelalgortime\python\hybrid_heuristic_ml\coding_ideas_hierarchical_bpp.py`
**Summary:** `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\A Hierarchical Bin Packing Framework with Dual Manipulators (Summary).md`

#### M2.1 Architecture Overview

The paper decomposes the problem into two coupled MDPs:

```
HIGH-LEVEL: DFS-BS Tree Search
  - Decides: item order, orientation, bin assignment
  - Explores: beam of candidate sequences
  - Reward: cumulative placement quality

    LOW-LEVEL: A3C RL Agent
      - Decides: (x, z) position on heightmap
      - Input: heightmap + item_size
      - Reward: adjacency count + stability score
```

**Key extension from paper (2D -> 3D):** The paper operates in 2D. Our extension replaces the binary grid with a heightmap and adds stability checking.

#### M2.2 Tree Search: DFS-BS

From `coding_ideas_hierarchical_bpp.py` -> `HierarchicalTreeSearch`:

```python
class HierarchicalTreeSearch:
    def __init__(self, rl_agent, stability_checker, max_beam_width=5,
                 max_depth=10, use_repack=True, time_limit=2.0):
        ...

    def search(self, bin_states, buffer_items, recognized_items,
               require_full_pack=False) -> List[CandidateAction]:
        # Phase 1: Tree expansion (DFS-BS)
        # Phase 2: Forward simulation and selection
        # Phase 3: Repacking (if enabled)
        ...
```

**Branching factor analysis (2-bounded extension):**
- Per node: |buffer| x |orientations| x |bins| candidates
- With buffer=10, orientations=2, bins=2: branching = 40
- After SELECTION with beam_width=5: effective branching = 5
- Max depth = buffer_size = 10
- Worst case: 5^10 ~ 10M nodes (pruning reduces drastically)
- Time-bounded: configurable limit (default 2.0 seconds)

#### M2.3 Low-Level: A3C Position Selection

From `coding_ideas_hierarchical_bpp.py` (PyTorch pseudocode):

```python
class PositionSelectionNetwork(nn.Module):
    # Input: heightmap (W x D) + item_size (3,)
    # Output: policy over W*D+1 actions, value V(s)
    # CNN: 3 conv layers (32, 64, 64 filters)
    # FC: 512 -> 256
    # Actor: 256 -> W*D+1
    # Critic: 256 -> 1
    # Feasibility mask: z' = b*z + (1-b)*(-1e8)
```

**Combined reward (our extension):**
```python
combined_reward = 0.7 * adjacency_reward + 0.3 * stability_reward
```

#### M2.4 2-Bounded Space Extension

From `coding_ideas_hierarchical_bpp.py` -> `TwoBoundedSpaceManager`:

```python
class TwoBoundedSpaceManager:
    # Manages 2 active bins
    # Bin closing policies: UTILIZATION_THRESHOLD, NO_FIT_ITEMS, COMBINED
    # close_threshold: 0.85 default
    # Bin selection: evaluate both bins, pick higher reward
```

Closing policy (`BinClosingPolicy.COMBINED`): Close bin when utilization >= 0.85 AND no buffer item can fit.

#### M2.5 Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `HierarchicalTreeSearch` | `coding_ideas_hierarchical_bpp.py` | Main tree search |
| `TreeNode` | `coding_ideas_hierarchical_bpp.py` | Search tree node |
| `CandidateAction` | `coding_ideas_hierarchical_bpp.py` | Action candidates |
| `PositionSelectionNetwork` | `coding_ideas_hierarchical_bpp.py` | A3C RL agent |
| `StabilityChecker` | `coding_ideas_hierarchical_bpp.py` | Stability validation |
| `TwoBoundedSpaceManager` | `coding_ideas_hierarchical_bpp.py` | k=2 bin management |
| `BufferManager` | `coding_ideas_hierarchical_bpp.py` | Semi-online buffer |
| `Bin` | `coding_ideas_hierarchical_bpp.py` | Heightmap-based bin |
| `main_packing_loop()` | `coding_ideas_hierarchical_bpp.py` | Top-level integration |

#### M2.6 Repacking (Algorithm 3)

From `_repack_trial()`: Tries unpacking 1, 2, or 3 top-layer items and re-searching. Only top-layer items (nothing stacked on top) can be safely unpacked. This is a key 3D extension not needed in the paper's 2D setting.

---

### M3. EMS-Filtered DRL (Learned Selective Hyper-Heuristic)

**Source:** `C:\Users\Louis\Downloads\stapelalgortime\python\hybrid_heuristic_ml\ems_filtered_drl.py`

#### M3.1 Core Insight

Instead of having the DRL choose a raw pixel/grid position (like Deep-Pack), or having a heuristic choose a position (like WallE), this approach:
1. **Heuristics generate candidate placements** (using EMS/Extreme Points)
2. **DRL scores and selects among candidates** (CNN + per-candidate MLP)

This is framed as a **learned selective hyper-heuristic**: each candidate is scored by all heuristics (as features), and the DRL learns which heuristic to trust in each state.

#### M3.2 Candidate Generation Pipeline

From `ems_filtered_drl.py` Section 1-2:

```
1. Generate extreme points on heightmap
   - Right edge, back edge, diagonal corner of each placed item
   - Height transition points (step edges)
   - Deduplicate and filter within bounds

2. For each (extreme_point, orientation):
   - Check feasibility: bounds, height limit, stability
   - Compute heuristic scores:
     a. DBLF score: -(x*10000 + z*100 + y)
     b. Corner distance score: min Manhattan to 8 bin corners
     c. DFTRC score: max Euclidean to front-top-right
     d. Contact ratio: surface area touching neighbors/walls
     e. Heightmap variance delta: change in surface variance

3. Return CandidatePlacement objects with all scores as features
```

From `ems_filtered_drl.py` -> `CandidatePlacement.feature_vector()`:
```python
def feature_vector(self) -> np.ndarray:
    return np.array([
        self.x, self.y, self.z,
        self.orientation[0], self.orientation[1], self.orientation[2],
        self.support_ratio,
        self.dblf_score,
        self.corner_distance_score,
        self.dftrc_score,
        self.contact_ratio,
        self.heightmap_variance_delta,
    ], dtype=np.float32)  # 12-dimensional per candidate
```

#### M3.3 CNN Backbone + Per-Candidate MLP Scoring

From `ems_filtered_drl.py` -> `HybridCandidateScoringNet`:

```python
class HybridCandidateScoringNet(nn.Module):
    # Visual backbone: CNN on heightmap -> 256-dim visual features
    #   Conv2d(in_channels, 32, 3) -> ReLU -> MaxPool
    #   Conv2d(32, 64, 3) -> ReLU -> MaxPool
    #   Conv2d(64, 128, 3) -> ReLU -> AdaptiveAvgPool(4,4)
    #   Linear(2048, 256)

    # Candidate encoder: MLP on 12-dim features -> 64-dim
    #   Linear(12, 64) -> ReLU -> Linear(64, 64) -> ReLU

    # Joint scorer: concat(256, 64) = 320 -> Q-value
    #   Linear(320, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 1)

    # Masking: q_values.masked_fill(candidate_mask == 0, -1e9)
```

The in_channels parameter supports multi-channel heightmap states (e.g., height channel, weight channel, stability channel, item-size channel, buffer-diversity channel).

#### M3.4 Curriculum Learning Strategy

From `ems_filtered_drl.py` Section 4.2:

| Phase | Bin Size | Items | Buffer | Bins | Stability | Episodes |
|-------|----------|-------|--------|------|-----------|----------|
| 1 (Easy) | 10x10x10 | 5-20 | 3 | 1 | min_support=0.3 | 100K |
| 2 (Medium) | 20x20x20 | 20-50 | 5 | 1 | min_support=0.5 | 100K |
| 3 (Full) | 40x40x40 | 50-200 | 10 | 2 (k=2) | min_support=0.8 | 200K |

**Phase 1** lets the agent learn basic packing behavior without being overwhelmed. **Phase 3** introduces the full complexity of our use case.

#### M3.5 Why This IS a Selective Hyper-Heuristic

From `ems_filtered_drl.py` Section 5:

> "Each candidate is scored by ALL heuristics (as features). The DRL agent LEARNS which heuristic to trust in each state. This is 'learning to select heuristics' -- exactly a selective HH."

The DRL can learn non-obvious context-dependent combinations, for example:
- When bin < 50% full: weight DBLF score highly (systematic back-bottom packing)
- When bin > 50% full: weight corner distance highly (fill gaps near corners)
- When surface is uneven: weight stability score highly

#### M3.6 Key Files

| Item | File |
|------|------|
| `generate_extreme_points_heightmap()` | `python/hybrid_heuristic_ml/ems_filtered_drl.py` |
| `CandidatePlacement` | `python/hybrid_heuristic_ml/ems_filtered_drl.py` |
| `generate_candidates()` | `python/hybrid_heuristic_ml/ems_filtered_drl.py` |
| `HybridCandidateScoringNet` | `python/hybrid_heuristic_ml/ems_filtered_drl.py` |
| `compute_contact_ratio()` | `python/hybrid_heuristic_ml/ems_filtered_drl.py` |
| `train_hybrid_agent()` | `python/hybrid_heuristic_ml/ems_filtered_drl.py` |

---

### M4. PCT + Heuristic Integration

**Source:** `C:\Users\Louis\Downloads\stapelalgortime\python\hybrid_heuristic_ml\pct_heuristic_integration_ideas.py`

#### M4.1 The PCT Hybrid Paradigm

PCT (Packing Configuration Trees) uses heuristics to generate **leaf node candidates** and a GAT + Pointer network to **select** among them. This is a principled decomposition:

- **Heuristics** guarantee locally optimal candidates (Theorem 1, Appendix C of PCT paper: EMS/EV expansion finds ALL convex vertices of the No-Fit Polygon)
- **DRL** learns global optimization from these local optima
- **Adding new heuristics/constraints** does not require retraining

#### M4.2 Leaf Node Expansion Schemes

Four expansion schemes are implemented in `pct_heuristic_integration_ideas.py`:

| Scheme | Class | Complexity | Candidates | Best For |
|--------|-------|-----------|------------|----------|
| Corner Point (CP) | `CornerPointExpansion` | O(c) constant | Few | Early packing, simple layouts |
| Extreme Point (EP) | `ExtremePointExpansion` | O(m * \|B_2D\|) | Medium | General packing |
| Empty Maximal Space (EMS) | (in `pct_coding_ideas.py`) | O(\|E\|) linear | Medium | Default (recommended) |
| Event Point (EV) | `EventPointExpansion` | O(m * \|B_2D\|^2) | Many | Late packing, high fill |

Each scheme produces a list of 3D candidate positions `np.ndarray` of shape (3,) representing (x, y, z).

#### M4.3 ExpansionSchemeSelector (Novel Hyper-Heuristic Idea)

From `pct_heuristic_integration_ideas.py` -> `ExpansionSchemeSelector`:

```python
class ExpansionSchemeSelector:
    """Dynamically select the best expansion scheme based on packing state.
    This bridges PCT with the 'hyper-heuristic' concept (Gap 3)."""

    def select_scheme(self, utilization, num_packed, fragmentation, item_size_ratio):
        if num_packed < 5 and utilization < 0.2:
            return 'cp'    # Early: CP sufficient
        if utilization < 0.7:
            return 'ems'   # Mid: EMS (good balance)
        if utilization > 0.85 or fragmentation > 0.7:
            return 'ev'    # Late: EV (thorough search)
        return 'ems'       # Default

    def select_scheme_bandit(self, epsilon=0.1):
        # Epsilon-greedy over {cp, ep, ems, ev}
        # Tracks reward history per scheme
```

This is a **meta-hyper-heuristic**: the HH selects which candidate generation scheme to use, rather than which placement rule.

#### M4.4 Constraint Reward Functions

From `pct_heuristic_integration_ideas.py` -> `ConstraintRewards`:

| Reward | Method | Formula | Use Case |
|--------|--------|---------|----------|
| Stability | `stability_reward()` | support_area / bottom_area | Prevent toppling |
| Load Balancing | `load_balancing_reward()` | -var(mass_positions from center) | Even weight distribution |
| Height Uniformity | `height_uniformity_reward()` | -var(heightmap) | Flat stacking surfaces |
| Isle Friendliness | `isle_friendliness_reward()` | -avg_dist(same_category_items) | Warehouse grouping |
| Bridging | `bridging_reward()` | count(supporting_items) | Structural stability |

**Combined reward** (PCT paper's formulation):
```python
w_t = max(0, v_t + c * sum(f_hat_i(.)))
# v_t = item volume
# c = 0.1 (default, tunable)
# f_hat_i = f_i / |f_bar_i| = normalized constraint value
```

#### M4.5 Key Files

| Class | File |
|-------|------|
| `CornerPointExpansion` | `python/hybrid_heuristic_ml/pct_heuristic_integration_ideas.py` |
| `ExtremePointExpansion` | `python/hybrid_heuristic_ml/pct_heuristic_integration_ideas.py` |
| `EventPointExpansion` | `python/hybrid_heuristic_ml/pct_heuristic_integration_ideas.py` |
| `ExpansionSchemeSelector` | `python/hybrid_heuristic_ml/pct_heuristic_integration_ideas.py` |
| `ConstraintRewards` | `python/hybrid_heuristic_ml/pct_heuristic_integration_ideas.py` |
| PCT core (GAT + Pointer) | `python/deep_rl/pct_coding_ideas.py` |
| ToP buffer planner | `python/semi_online_buffer/top_buffer_2bounded_coding_ideas.py` |

---

### M5. Prediction-Guided Heuristic Selection

**Source:** `C:\Users\Louis\Downloads\stapelalgortime\python\hybrid_heuristic_ml\prediction_hybrid_framework_IDEAS.py`
**Summary:** `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\Online Bin Packing with Predictions (Summary).md`

#### M5.1 Core Idea

The paper (Angelopoulos et al. 2023) introduces `Hybrid(lambda)` which blends `ProfilePacking` (prediction-trusting) with a robust algorithm `A` (prediction-ignoring). We extend this to a **multi-heuristic portfolio** where prediction quality guides heuristic selection.

#### M5.2 ProfilePacking3D (Adapted from 1D Theory)

The 1D ProfilePacking algorithm:
1. Build a "profile" (template packing) from predicted item frequencies
2. Pre-compute an optimal offline packing of the profile
3. Online: match arriving items to template placeholders

**3D adaptation notes:**
- Profile construction uses predicted box-type frequencies (not scalar sizes)
- Optimal profile packing uses FFD or a strong offline heuristic
- Matching is on box dimensions (allowing rotation): match item to closest-fitting placeholder
- "Special items" (zero predicted frequency) go to separate bins via FirstFit

#### M5.3 PredictionGuidedHeuristicSelector

From `prediction_hybrid_framework_IDEAS.py` -> `PredictionGuidedHeuristicSelector`:

```python
class PredictionGuidedHeuristicSelector:
    """Portfolio of 5 heuristics, selected by (eta, bin_state, buffer_state)."""
    heuristics = {
        'profile_packing': ...,   # Best when predictions are accurate
        'dblf': ...,              # Robust general-purpose
        'best_fit_volume': ...,   # Maximize fill rate
        'stability_first': ...,  # Prioritize stable placements
        'layer_building': ...,   # Good for uniform-height items
    }

    def select_heuristic(self, eta, bin_state, buffer_state):
        # Low eta (< 0.1): prefer profile_packing
        # Medium eta (0.1-0.3): prefer DBLF or best_fit
        # High eta (> 0.3): prefer robust heuristics
        # Unstable bin: always prefer stability_first
        # Uniform buffer heights: prefer layer_building
```

**Selection score formula:**
```python
scores[name] = (prediction_weight * pred_score       # 40%
              + (1 - prediction_weight) * base_score * 0.5  # historical
              + 0.25 * bin_score                     # bin state
              + 0.25 * buffer_score)                 # buffer composition
```

#### M5.4 Dynamic Trust Parameter Lambda

From the paper: `Hybrid(lambda)` with lambda in [0, 1]:
- lambda = 1: pure ProfilePacking (trust predictions fully)
- lambda = 0: pure robust algorithm (ignore predictions)
- Optimal lambda depends on prediction quality eta

**For our setting:** With buffer=10, the buffer provides perfect "predictions" for the next 10 items. This means:
- For buffer items: eta = 0 -> trust ProfilePacking fully (lambda = 1)
- For items beyond buffer: eta depends on ML model quality -> use learned lambda

**Recommended parameter values:**

| Parameter | Paper Value | Our Value | Rationale |
|-----------|------------|-----------|-----------|
| lambda | 0.25-0.5 | 0.3-0.5 | Buffer gives better local predictions |
| m (profile size) | 5000 | 50-200 | Fewer distinct 3D box types |
| w (window size) | 2100-25000 | 500-2000 | Smaller item throughput |
| k (bin capacity) | 100 | 5-15 items | 3D bins hold fewer items |

#### M5.5 ML-Enhanced Frequency Prediction

From `prediction_hybrid_framework_IDEAS.py` -> `MLFrequencyPredictor`:

```python
class MLFrequencyPredictor:
    # model_type: 'sliding_window', 'exponential_smoothing', 'linear_trend'
    # Exponential smoothing: smoothed[t] = alpha * observed + (1-alpha) * smoothed[t-1]
    # alpha = 0.3 default
    # Returns predicted frequency vector (normalized to sum to 1)
```

Even simple exponential smoothing can reduce eta and improve `Hybrid(lambda)` performance.

#### M5.6 Consistency-Robustness Evaluation

From `prediction_hybrid_framework_IDEAS.py` -> `ConsistencyRobustnessEvaluator`:

Runs the algorithm across multiple (lambda, eta) combinations on test instances. Produces a curve analogous to Figure 3 of the paper but for 3D instances:
- X-axis: prediction error eta
- Y-axis: competitive ratio
- One curve per lambda value

#### M5.7 Key Files

| Class | File |
|-------|------|
| `PredictionGuidedHeuristicSelector` | `python/hybrid_heuristic_ml/prediction_hybrid_framework_IDEAS.py` |
| `MLFrequencyPredictor` | `python/hybrid_heuristic_ml/prediction_hybrid_framework_IDEAS.py` |
| `ConsistencyRobustnessEvaluator` | `python/hybrid_heuristic_ml/prediction_hybrid_framework_IDEAS.py` |
| Prediction-augmented buffer | `python/semi_online_buffer/prediction_augmented_buffer_packing_IDEAS.py` |
| Stochastic blueprint | `python/semi_online_buffer/coding_ideas_stochastic_blueprint_packing.py` |

---

## N. THEORETICAL FOUNDATIONS & BOUNDS

---

### N1. Competitive Ratio Reference

**Sources:**
- `C:\Users\Louis\Downloads\stapelalgortime\python\theoretical_bounds\coding_ideas_stochastic_competitive_ratios.py`
- `C:\Users\Louis\Downloads\stapelalgortime\python\theoretical_bounds\random_order_competitive_bounds.py`

#### N1.1 Known Bounds from the Literature

From `random_order_competitive_bounds.py` -> `OnlineBoundsComparison.KNOWN_BOUNDS`:

| Problem | Bound | Type | Year | Author |
|---------|-------|------|------|--------|
| 1D BPP - Next Fit | 2.0 | ACR | 1974 | Johnson |
| 1D BPP - First Fit | 1.7 | ACR | 1974 | Johnson |
| 1D BPP - Advanced Harmonic | 1.57829 | ACR | 2017 | Balogh et al. |
| 2D BPP - Rectangles | 2.5545 | ACR | 2011 | Han et al. |
| 2D BPP - Squares | 2.0885 | ACR | 2021 | Epstein & Mualem |
| 3D BPP - Cubes | 2.5735 | ACR | 2021 | Epstein & Mualem |
| 1D Knapsack ROM | 1/6.65 | CR | 2021 | Albers, Khan, Ladewig |
| 1D GAP ROM | 1/6.99 | CR | 2021 | Albers, Khan, Ladewig |
| 1D Knapsack ROM (prev) | 1/8.06 | CR | 2018 | Kesselheim et al. |

**Important distinction:**
- BPP ACR = bins_used(online) / bins_used(optimal). **Lower is better.** Target: as close to 1.0 as possible.
- Knapsack CR = profit(online) / profit(optimal). **Higher is better.** Max = 1.0.

#### N1.2 Stochastic Bounds

From `coding_ideas_stochastic_competitive_ratios.py` (based on Ayyadevara et al. 2022):

- **i.i.d. model:** ECR = alpha + epsilon, where alpha is the offline asymptotic approximation ratio
  - For Best Fit: alpha approaches 1 for "nice" distributions -> ECR ~ 1 + epsilon
  - This means: under i.i.d. item arrival, BF is near-optimal
- **Random-order model:**
  - BF on items > 1/3: ARR = 1 (Theorem 1.2) -- optimal!
  - BF on 3-Partition items (1/4, 1/2]: ARR <= 1.49107 (Theorem 1.3)

#### N1.3 What These Mean for Our Expected Performance

For our semi-online setting (buffer 5-10, k=2, 3D):

| Scenario | Theoretical Basis | Expected Fill Rate |
|----------|------------------|-------------------|
| Pure online (no buffer) | 3D ACR ~ 2.57 worst case | ~39% worst case |
| Buffer = 10, i.i.d. items | ECR ~ 1 + epsilon | ~85-95% |
| Buffer = 10, adversarial | ACR improved by buffer | ~60-75% |
| k=2 bounded, buffer = 10 | k-bounded penalty + buffer gain | ~70-85% |
| With stability constraints | ~5-15% penalty for stability | ~65-80% |

**Critical insight:** The i.i.d. assumption is often reasonable for warehouse/e-commerce scenarios (repeat customers, seasonal patterns). Under i.i.d., our semi-online setting should approach near-optimal performance.

#### N1.4 Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `StochasticBPSimulator` | `coding_ideas_stochastic_competitive_ratios.py` | 1D packing algorithms |
| `CompetitiveRatioEstimator` | `coding_ideas_stochastic_competitive_ratios.py` | Monte Carlo CR estimation |
| `KnapsackRandomOrderBounds` | `random_order_competitive_bounds.py` | Theoretical CR calculations |
| `GAPRandomOrderBounds` | `random_order_competitive_bounds.py` | GAP CR calculations |
| `OnlineBoundsComparison` | `random_order_competitive_bounds.py` | Literature comparison |
| `ParameterOptimizer` | `random_order_competitive_bounds.py` | Parameter optimization |

---

### N2. Buffer Advantage Quantification

#### N2.1 The B-Choice Secretary Problem Reference

From `random_order_competitive_bounds.py` -> `BufferAdvantageEstimator`:

```python
class BufferAdvantageEstimator:
    @staticmethod
    def secretary_ratio_with_buffer(buffer_size):
        if buffer_size == 1:
            return 1 / math.e  # ~ 0.368 (classical secretary)
        return 1 - 1 / (buffer_size + 1)
        # B=5:  0.833
        # B=10: 0.909
        # B=15: 0.938
```

**Key insight:** With a buffer of 10 items, the selection ratio improves from 0.368 (pure online) to 0.909, a **2.47x improvement factor**.

#### N2.2 Estimated Improvement Factors

From `BufferAdvantageEstimator.estimated_knapsack_ratio_with_buffer()`:

| Buffer Size | Secretary Ratio | Estimated Knapsack CR | Estimated GAP CR (k=2) |
|-------------|-----------------|----------------------|------------------------|
| 1 | 0.3679 | 0.1504 (1/6.65) | 0.1431 (1/6.99) |
| 3 | 0.7500 | 0.3065 | 0.3494 |
| 5 | 0.8333 | 0.3406 | 0.3882 |
| 7 | 0.8750 | 0.3576 | 0.4076 |
| 10 | 0.9091 | 0.3715 | 0.4235 |

**Important caveat:** These are heuristic estimates, not proven bounds. The actual relationship between buffer size and competitive ratio in 3D bin packing is an open theoretical question.

#### N2.3 Empirical Validation Framework

From `coding_ideas_stochastic_competitive_ratios.py` -> `CompetitiveRatioEstimator.compare_buffer_benefit()`:

```python
def compare_buffer_benefit(self, distribution='warehouse_like',
                            buffer_sizes=[1, 3, 5, 7, 10],
                            n=1000, trials=100):
    # For each buffer size:
    #   - Run BF_with_buffer on n items, 100 trials
    #   - Run BF_k2_with_buffer on same items
    #   - Compare against optimal_lower_bound
    # Returns: mean_cr, std_cr, mean_fill_rate, k2_penalty per buffer size
```

Distribution generators available:
- `generate_uniform(n, a, b)` -- U[a,b]
- `generate_3partition(n)` -- items in (1/4, 1/2]
- `generate_large_items(n)` -- items > 1/3
- `generate_warehouse_like(n)` -- 60% small, 25% medium, 15% large

#### N2.4 k=2 Bounded Space Penalty

From `coding_ideas_stochastic_competitive_ratios.py` -> `best_fit_bounded_k2()`:

```python
def best_fit_bounded_k2(self, items):
    active = [0.0, 0.0]  # 2 active bins
    closed_count = 0
    for item in items:
        # Try best fit in 2 active bins
        # If neither fits: close fuller bin, open new
    return closed_count + 2
```

The penalty for k=2 vs. unbounded is empirically estimated. The buffer compensates for this penalty because it allows selecting items that fit best in the available bins.

---

### N3. Benchmarking Framework

#### N3.1 Fair Comparison Metrics

| Metric | Formula | Target | Priority |
|--------|---------|--------|----------|
| Fill Rate (per bin) | sum(item_volumes) / bin_volume | Maximize (>80%) | Primary |
| Mean Utilization | mean(fill_rates across all bins) | Maximize | Primary |
| Stability % | % placements with support_ratio >= 0.8 | >= 95% | Primary |
| Bins Used | total closed + active bins | Minimize | Secondary |
| Competitive Ratio | bins_used / OPT_lower_bound | Minimize (target <1.5) | Secondary |
| Items Discarded | % items that could not be placed | Minimize (target 0%) | Secondary |
| Computation Time | ms per placement decision | < 2000ms | Constraint |
| CoG Deviation | max distance of center-of-gravity from bin center | Minimize | Tertiary |

#### N3.2 Instance Generators

| Generator | Distribution | Use Case |
|-----------|-------------|----------|
| Uniform | U(0.1, 0.5) * bin_dim per axis | Standard benchmark |
| Skewed Small | 60% small + 25% medium + 15% large | Warehouse-realistic |
| Skewed Large | 15% small + 25% medium + 60% large | Stress test |
| Cut-1 (CUT-1) | Generated by cutting bins | Offline BPP standard |
| Cut-2 (CUT-2) | Generated by cutting with constraints | Offline BPP standard |
| RS (Random Sizes) | Uniform item sizes | From OnlineBPH paper |
| Mendeley 198 | 198 real-world instances | From Van de Ven (2023) |

#### N3.3 Statistical Testing

For comparing N algorithms across M instance sets:
1. **ANOVA** (analysis of variance): test if any algorithm differs significantly
2. **Tukey-Kramer HSD** (post-hoc): pairwise comparisons with familywise error control
3. **Wilcoxon signed-rank test**: non-parametric alternative for small sample sizes
4. **Effect size**: Cohen's d to quantify practical significance beyond statistical significance

**Minimum recommended:** 30 instances per generator, 5 generators = 150 instances minimum. The 198 Mendeley instances provide additional real-world validation.

#### N3.4 Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `run_experiment()` | `coding_ideas_stochastic_competitive_ratios.py` | Monte Carlo experiments |
| `compare_buffer_benefit()` | `coding_ideas_stochastic_competitive_ratios.py` | Buffer ablation |
| `estimate_blueprint_benefit()` | `coding_ideas_stochastic_competitive_ratios.py` | Blueprint vs. standard |
| `generate_warehouse_like()` | `coding_ideas_stochastic_competitive_ratios.py` | Instance generation |
| `ConsistencyRobustnessEvaluator` | `prediction_hybrid_framework_IDEAS.py` | C-R curve plotting |

---

## O. COMPLETE INTEGRATED SYSTEM ARCHITECTURES

**This is the most actionable section.** Five architectures are presented in order of implementation complexity. Each combines components from multiple files into a working system.

---

### O1. Architecture ALPHA: "Pure Heuristic Baseline"

**No ML, fastest to implement. Essential for thesis: every ML result must beat this baseline.**

#### O1.1 Components

```
[Buffer Manager]          -> selects item (Largest-First)
    |
[160-Heuristic Framework] -> scores placement (DBLF default, all 6 rules available)
    |
[LBCP Stability Checker]  -> validates stability (support_ratio >= 0.8)
    |
[Best-Fit Bin Router]     -> routes to better of 2 active bins
    |
[ReplaceMax Closing]      -> closes bin when utilization > 0.85 AND no items fit
```

#### O1.2 Data Flow (Per Step)

1. Buffer provides 5-10 accessible items with known dimensions
2. For each item in buffer (sorted by volume, largest first):
   a. For each active bin (k=2):
      - Generate extreme points / candidate positions
      - Score each with DBLF (or selected placement rule)
      - Filter by LBCP stability (support_ratio >= 0.8)
      - Keep best placement per bin
   b. Select the best (item, bin, position) across all combinations
3. Place item, update heightmap, advance buffer
4. Check bin closing conditions

#### O1.3 Files Needed

| Component | File |
|-----------|------|
| 160 heuristic combinations | `python/heuristics/coding_ideas_160_heuristic_framework.py` |
| LBCP stability | `python/stability/lbcp_stability_and_rearrangement.py` |
| Stacking tree | `python/stability/stacking_tree_coding_ideas.py` |
| Buffer manager | `python/semi_online_buffer/buffer_aware_packing.py` |
| Buffer + stability | `python/semi_online_buffer/buffer_with_lbcp_stability.py` |
| 2-bounded space | `python/multi_bin/coding_ideas_two_bounded_space.py` |
| Feasibility mask | `python/stability/feasibility_mask_stability.py` |
| Placement rules | `python/hybrid_heuristic_ml/feasibility_guided_packing_coding_ideas.py` |
| Bin closing | `python/multi_bin/coding_ideas_dual_bin_replacement_strategies.py` |

#### O1.4 Step-by-Step Integration Guide

```
Week 1:
  1. Implement Bin class with heightmap (from coding_ideas_hierarchical_bpp.py -> Bin)
  2. Implement Item class with orientations (from same file -> Item)
  3. Implement DBLF placement scoring (from feasibility_guided_packing_coding_ideas.py -> score_dblf)
  4. Implement extreme point generation (from ems_filtered_drl.py Section 1)
  5. Test: pack 50 random items into a single bin, measure fill rate

Week 2:
  6. Implement StabilityChecker (from coding_ideas_hierarchical_bpp.py -> StabilityChecker)
  7. Implement LBCP support-ratio check (from lbcp_stability_and_rearrangement.py)
  8. Implement TwoBoundedSpaceManager (from coding_ideas_hierarchical_bpp.py)
  9. Implement BufferManager (from coding_ideas_hierarchical_bpp.py -> BufferManager)
  10. Wire together with main_packing_loop()

Week 3:
  11. Implement all 6 placement rules (from feasibility_guided_packing_coding_ideas.py)
  12. Implement bin closing policy (BinClosingPolicy.COMBINED)
  13. Run benchmarks: uniform, skewed, warehouse-like
  14. Record baseline fill rate and stability metrics
```

#### O1.5 Expected Performance

- **Fill rate:** 65-75% (based on OnlineBPH numbers + stability penalty)
- **Stability:** >= 95% (by construction -- only stable placements allowed)
- **Speed:** < 50ms per decision (pure Python, no GPU)
- **Implementation time:** 2-3 weeks

---

### O2. Architecture BETA: "Hybrid Hyper-Heuristic" (NOVEL CONTRIBUTION)

**The selective hyper-heuristic that addresses Research Gap 3. This is the primary thesis contribution architecture.**

#### O2.1 Components

```
[Buffer Manager]                -> provides buffer items
    |
[State Feature Extractor]       -> compute_state_features() -> 6-dim vector
    |
[HyperHeuristicSelector]       -> Q-learning/DQN selects strategy
    |
    +---> [WallE Heuristic]     (if selected)
    +---> [PackMan DQN]         (if selected)
    +---> [DBLF]                (if selected)
    +---> [Floor Building]      (if selected)
    +---> [Best-Fit Volume]     (if selected)
    |
[LBCP Stability Validator]      -> filters unstable placements
    |
[Best-Fit Bin Router]           -> routes to better active bin
    |
[Stability-Aware Bin Closing]   -> closes bin with stability check
```

#### O2.2 Key Innovation

The selector **learns when to use which heuristic** based on:
- Packing phase (early/mid/late)
- Surface roughness (smooth vs. uneven)
- Fill fraction (how full)
- Buffer diversity (uniform vs. varied items)
- Fill imbalance between bins
- Item size relative to remaining space

This is NOT just running all heuristics and picking the best score (that would be a simple rule). The HH learns **context-dependent policies** that may not be obvious, e.g., "when the buffer has mostly small items and the bin is 60% full, use Floor Building even though WallE usually has higher scores."

#### O2.3 Training the Selector

```python
# Phase 1: Warm-up with rule-based selection (2000 episodes)
# Phase 2: Q-learning with epsilon-greedy (10000 episodes)
#   - State: discretized 6-dim features
#   - Action: one of 5 strategies
#   - Reward: compute_step_reward_for_selector()
#   - Update: standard Q-learning (lr=0.1, gamma=0.9, eps=0.1)
# Phase 3: Fine-tune with DQN selector (optional, 5000 episodes)
```

#### O2.4 Files Needed

All files from ALPHA, plus:

| Component | File |
|-----------|------|
| HyperHeuristicSelector | `python/hybrid_heuristic_ml/walle_packman_hybrid_coding_ideas.py` |
| HybridAgent | `python/hybrid_heuristic_ml/walle_packman_hybrid_coding_ideas.py` |
| SelectiveHyperHeuristic | `python/hybrid_heuristic_ml/feasibility_guided_packing_coding_ideas.py` |
| WallE heuristic | `python/heuristics/walle_heuristic_coding_ideas.py` |
| PackMan DQN | `python/deep_rl/packman_dqn_coding_ideas.py` |
| State features | `python/hybrid_heuristic_ml/walle_packman_hybrid_coding_ideas.py` |
| Step reward | `python/hybrid_heuristic_ml/walle_packman_hybrid_coding_ideas.py` |

#### O2.5 Step-by-Step Integration Guide

```
Week 1-3: Implement ALPHA baseline (as above)

Week 4:
  1. Implement WallE heuristic scoring (from walle_heuristic_coding_ideas.py)
     - 8 sub-scores: edge weight, contact area, stability, etc.
  2. Implement compute_state_features() (from walle_packman_hybrid_coding_ideas.py)
  3. Implement HyperHeuristicSelector with rule_based mode
  4. Wire into ALPHA pipeline: replace fixed DBLF with selected heuristic
  5. Benchmark: does rule-based HH beat ALPHA? (expect: yes, by 2-5%)

Week 5:
  6. Implement Q-learning selector (tabular, 5^6 states x 5 actions)
  7. Implement compute_step_reward_for_selector()
  8. Train Q-learning selector on 10000 episodes
  9. Benchmark: does Q-learning HH beat rule-based? (expect: yes, by 1-3%)
  10. Generate strategy usage reports: which heuristic used when?
  11. This IS the novel contribution. Analyze and write thesis chapter.
```

#### O2.6 Expected Performance

- **Fill rate:** 72-80%
- **Stability:** >= 95%
- **Speed:** < 100ms per decision (mostly Python, optional GPU for PackMan)
- **Implementation time:** 4-5 weeks (includes ALPHA)
- **Thesis contribution:** First selective HH for 3D online BPP

---

### O3. Architecture GAMMA: "PCT/ToP + LBCP" (BEST PERFORMANCE)

**Highest expected performance. Uses state-of-the-art PCT policy with Tree of Packing (ToP) MCTS buffer planning.**

#### O3.1 Components

```
[Buffer Manager]
    |
[ToP MCTS Planner]              -> searches over (item, bin, scheme) combinations
    |                               using UCB selection + policy-guided expansion
    +---> [PCT Policy Network]   -> GAT + Pointer for leaf selection
    |         |
    |         +---> [EMS Manager]           -> generates candidate positions
    |         +---> [CornerPointExpansion]   -> (if selected by ExpansionSchemeSelector)
    |         +---> [ExtremePointExpansion]  -> (if selected)
    |         +---> [EventPointExpansion]    -> (for high-fill situations)
    |
[LBCP Stability Validator]       -> constraint reward + hard filter
    |
[TwoBinPCTManager]               -> manages PCT trees for both active bins
    |
[Bin Closing Strategy]           -> spatial ensemble ranking from ToP
```

#### O3.2 PCT Network Architecture

From `python/deep_rl/pct_coding_ideas.py`:
- Three MLPs for heterogeneous node projection (internal, leaf, current-item nodes)
- GAT attention layer (multi-head, 8 heads)
- Skip connection + feed-forward
- Pointer mechanism for leaf selection (attention-based)
- Critic head for state value estimation
- Training: ACKTR or PPO

#### O3.3 ToP MCTS for Buffer Planning

From `python/semi_online_buffer/top_buffer_2bounded_coding_ideas.py`:
- MCTS tree: each node represents a buffer state + bin state
- Actions: select (item, bin, orientation) from buffer
- UCB selection with policy prior from PCT
- Expansion: apply placement, update PCT tree, generate new leaf
- Evaluation: PCT critic value V(s)
- Path caching: avoid recomputing shared prefixes

#### O3.4 Extension for k=2

From `python/semi_online_buffer/top_buffer_2bounded_coding_ideas.py`:
- Each MCTS node stores TWO PCT trees (one per active bin)
- Actions include bin_index (which of 2 bins)
- Bin selection: spatial ensemble ranking (evaluate placement quality in both bins)
- Bin closing: integrated into MCTS search (close action as a valid move)

#### O3.5 Files Needed

All files from ALPHA and BETA, plus:

| Component | File |
|-----------|------|
| PCT policy (GAT + Pointer) | `python/deep_rl/pct_coding_ideas.py` |
| ToP MCTS planner | `python/semi_online_buffer/top_buffer_2bounded_coding_ideas.py` |
| EMS manager | `python/deep_rl/pct_coding_ideas.py` |
| CP/EP/EV expansion | `python/hybrid_heuristic_ml/pct_heuristic_integration_ideas.py` |
| ExpansionSchemeSelector | `python/hybrid_heuristic_ml/pct_heuristic_integration_ideas.py` |
| ConstraintRewards | `python/hybrid_heuristic_ml/pct_heuristic_integration_ideas.py` |
| Buffer + tree search | `python/semi_online_buffer/coding_ideas_buffer_with_tree_search.py` |
| LBCP stability | `python/stability/lbcp_stability_and_rearrangement.py` |
| 2-bounded bin packing | `python/multi_bin/two_bounded_bin_packing.py` |

#### O3.6 Step-by-Step Integration Guide

```
Weeks 1-5: Implement ALPHA + BETA (as above)

Week 6-7:
  1. Implement EMS data structure and manager (from pct_coding_ideas.py)
  2. Implement PCT tree (internal + leaf + current-item nodes)
  3. Implement feature extraction (to_feature_tensors)
  4. Implement GAT layer (multi-head attention, 8 heads)
  5. Implement Pointer network for leaf selection
  6. Implement Critic head

Week 8-9:
  7. Implement training environment (gymnasium interface)
  8. Train PCT on single-bin BPP-1 setting using ACKTR or PPO
     - Start with small bins (10x10x10), 50K episodes
     - Scale to full size (40x40x40), 200K episodes
  9. Validate: PCT single-bin should beat DBLF baseline by 10%+

Week 10-11:
  10. Implement ToP MCTS (from top_buffer_2bounded_coding_ideas.py)
  11. Wire PCT as evaluation function in MCTS
  12. Implement bin selection in MCTS (2-bounded extension)
  13. Tune MCTS parameters: rollout_budget=200, exploration_c=1.0

Week 12:
  14. Full integration: MCTS buffer + PCT placement + LBCP stability + 2 bins
  15. Benchmark against ALPHA and BETA
  16. Expected result: 10-15% improvement over BETA
```

#### O3.7 Training Strategy

| Stage | Setting | Episodes | Time | Hardware |
|-------|---------|----------|------|----------|
| 1. Single-bin, small | 10x10x10, 20 items, no buffer | 100K | ~10h | 1 GPU |
| 2. Single-bin, full | 40x40x40, 100 items, no buffer | 200K | ~30h | 1 GPU |
| 3. Buffer, single-bin | 40x40x40, 100 items, buffer=10 | 100K | ~20h | 1 GPU |
| 4. Buffer, 2-bin | Full setting (our use case) | 200K | ~40h | 1 GPU |
| **Total** | | | **~100h** | **RTX 3060+** |

#### O3.8 Expected Performance

- **Fill rate:** 80-90%
- **Stability:** >= 95%
- **Speed:** 500-2000ms per decision (MCTS budget-dependent)
- **Implementation time:** 8-12 weeks total
- **Training time:** ~100 GPU-hours

---

### O4. Architecture DELTA: "Constrained DRL + Buffer MCTS"

**Alternative to GAMMA. Uses constrained MDP (CMDP) instead of PCT, with explicit feasibility prediction.**

#### O4.1 Components

```
[Buffer Manager]
    |
[Buffer MCTS Planner]           -> searches over (item, bin) combinations
    |
    +---> [ConstrainedPackingNet]  -> actor-critic with Lagrangian constraints
    |         |
    |         +---> [FeasibilityPredictor]  -> CNN predicting stability violations
    |         +---> [LagrangianMultipliers] -> learned constraint weights
    |
[LBCP Stability Validator]       -> hard filter (safety net)
    |
[BinCoordinator]                 -> manages 2 bins, closing decisions
```

#### O4.2 Key Difference from GAMMA

- **GAMMA** (PCT): Heuristics generate candidates, DRL selects. Constraint handling via reward shaping.
- **DELTA** (CMDP): DRL generates actions directly, constrained by learned feasibility predictor + Lagrangian multipliers. Constraint handling is principled (CMDP theory).

#### O4.3 Files Needed

All from ALPHA, plus:

| Component | File |
|-----------|------|
| Constrained DRL | `python/deep_rl/constrained_drl_bin_packing.py` |
| Decomposed actor-critic | `python/deep_rl/decomposed_actor_critic_coding_ideas.py` |
| Buffer MCTS | `python/semi_online_buffer/buffer_mcts_policy_coding_ideas.py` |
| Buffer MCTS selection | `python/semi_online_buffer/buffer_mcts_selection.py` |
| LBCP stability | `python/stability/lbcp_stability_and_rearrangement.py` |
| Feasibility mask | `python/stability/feasibility_mask_stability.py` |
| 2-bounded bin management | `python/multi_bin/two_bounded_bin_packing.py` |
| Bin coordinator | `python/multi_bin/coding_ideas_two_bounded_space.py` |
| Stability vs. efficiency | `python/stability/coding_ideas_stability_vs_efficiency.py` |

#### O4.4 Step-by-Step Integration Guide

```
Weeks 1-3: Implement ALPHA baseline

Week 4-5:
  1. Implement constrained packing network (from constrained_drl_bin_packing.py)
     - Actor: heightmap -> CNN -> position logits (with feasibility mask)
     - Critic: heightmap -> CNN -> V(s)
     - Feasibility predictor: CNN classifier (stable/unstable)
  2. Implement Lagrangian multiplier updates

Week 6-7:
  3. Train on single-bin setting (same schedule as GAMMA stages 1-2)
  4. Implement decomposed actor-critic (from decomposed_actor_critic_coding_ideas.py)
     - One head for position, one for orientation

Week 8-9:
  5. Implement Buffer MCTS (from buffer_mcts_policy_coding_ideas.py)
  6. Wire CMDP policy as evaluation function in MCTS
  7. Implement BinCoordinator for 2-bounded space

Week 10:
  8. Full integration and benchmarking
```

#### O4.5 Expected Performance

- **Fill rate:** 78-88%
- **Stability:** >= 97% (CMDP explicitly optimizes for constraints)
- **Speed:** 200-1000ms per decision
- **Implementation time:** 6-10 weeks total
- **Training time:** ~60 GPU-hours

---

### O5. Architecture EPSILON: "Tsang-Extended" (FASTEST DRL)

**Most lightweight DRL approach. Extends Tsang et al. (2025) DDQN for multi-bin concurrent packing.**

#### O5.1 Components

```
[Adaptive Buffer Manager]        -> buffer sizing + item ordering
    |
[DDQN + MCA Network]            -> Double DQN with Multi-Channel Attention
    |                               (heightmap + weight map + item features)
    |
[LBCP Stability Checker]         -> fast validation
    |
[ReplaceMax Bin Closing]         -> close fullest bin when items don't fit
```

#### O5.2 Key Advantage

Tsang et al.'s DDQN with MCA is specifically designed for multi-bin concurrent packing. It is the only paper in the reading list that directly addresses the multi-bin problem with DRL. Extensions needed:
- Add stability checking (not in original paper)
- Add buffer management (original is pure online)
- Adapt reward for fill + stability multi-objective

#### O5.3 Files Needed

| Component | File |
|-----------|------|
| DDQN + MCA architecture | `python/deep_rl/coding_ideas_tsang2025_ddqn_dual_bin.py` |
| Deep-Pack 3D baseline | `python/deep_rl/deep_pack_3d_coding_ideas.py` |
| Buffer + PackMan | `python/semi_online_buffer/buffer_packman_coding_ideas.py` |
| Lookahead buffer + dual bin | `python/semi_online_buffer/coding_ideas_lookahead_buffer_with_dual_bin.py` |
| LBCP stability | `python/stability/lbcp_stability_and_rearrangement.py` |
| Dual bin replacement | `python/multi_bin/coding_ideas_dual_bin_replacement_strategies.py` |
| Stability vs. efficiency | `python/stability/coding_ideas_stability_vs_efficiency.py` |
| Adaptive buffer manager | `python/semi_online_buffer/buffer_aware_packing.py` |

#### O5.4 Step-by-Step Integration Guide

```
Weeks 1-3: Implement ALPHA baseline

Week 4-5:
  1. Implement DDQN network (from coding_ideas_tsang2025_ddqn_dual_bin.py)
     - Input: multi-channel heightmap (height + weight + stability channels)
     - MCA: multi-channel attention over spatial features
     - Output: Q-values for all positions across both bins
  2. Implement ReplaceMax closing policy
  3. Train on 2-bin setting directly

Week 6:
  4. Implement adaptive buffer manager
  5. Extend DDQN to handle buffer input (buffer item features as additional input)

Week 7:
  6. Full integration: DDQN + buffer + LBCP + ReplaceMax
  7. Benchmark against ALPHA and BETA
```

#### O5.5 Expected Performance

- **Fill rate:** 75-82%
- **Stability:** >= 95%
- **Speed:** < 100ms per decision (single forward pass)
- **Implementation time:** 5-7 weeks total
- **Training time:** ~30 GPU-hours (simpler network than GAMMA)

---

### O6. RECOMMENDED IMPLEMENTATION ORDER

```
             ALPHA (Baseline)
            /               \
       2-3 weeks          (parallel: train selector data)
          |                    |
     BETA (Novel HH)     Train PCT or DDQN
      +2 weeks                |
          |                    |
    GAMMA or DELTA or EPSILON
      +6-8 weeks
          |
    Compare All for Thesis
      +2 weeks
```

#### O6.1 Detailed Timeline

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1-3 | ALPHA complete | Baseline fill rate + stability numbers |
| 4-5 | BETA complete | Selective HH results (NOVEL CONTRIBUTION) |
| 5 | Start GAMMA/DELTA training | PCT or CMDP training begins in background |
| 6-8 | MCTS integration | MCTS + trained policy working |
| 9-10 | Full system GAMMA/DELTA | Best-performance architecture running |
| 11-12 | Benchmarking + ablation | Complete comparison, statistical tests |
| 13+ | Thesis writing | Results chapter with all comparisons |

#### O6.2 Decision Points

- After Week 3 (ALPHA): If fill rate > 70%, the heuristic baseline is strong. Focus thesis angle on the HH novelty (BETA).
- After Week 5 (BETA): If HH improves by > 5%, strong thesis contribution. Proceed to GAMMA/DELTA for best numbers.
- After Week 5: Choose GAMMA vs. DELTA vs. EPSILON:
  - **GAMMA (PCT):** If you want best single-bin performance and have time for GAT implementation
  - **DELTA (CMDP):** If stability is the primary concern (Lagrangian constraint handling is principled)
  - **EPSILON (DDQN):** If you want fastest implementation and GPU is limited

#### O6.3 Minimum Viable Thesis

If time is limited, the minimum viable thesis contribution is:
1. **ALPHA** (baseline, 3 weeks)
2. **BETA** (selective HH -- the novel contribution, +2 weeks)
3. **Benchmarking** (comparison + ablation, +2 weeks)

Total: **7 weeks.** This addresses Research Gap 3 directly and provides a publishable result.

---

## P. NOVEL IDEAS NOT IN ANY PAPER

These ideas emerge from combining approaches across the reading list and are potential thesis contributions beyond the selective HH.

---

### P1. Stability-Aware Hyper-Heuristic with Phase Detection

**Novelty:** No existing work combines automatic packing-phase detection with heuristic switching for 3D bin packing.

**Idea:** The bin fills in three phases:
1. **Foundation phase** (0-30% full): Stability is paramount. Use Back Bottom or WallE.
2. **Growth phase** (30-70% full): Balance fill and stability. Use DBLF or DFTRC.
3. **Completion phase** (70-100% full): Fill rate is paramount. Use Best-Fit or Corner Distance.

Train a phase detector (simple classifier on heightmap features) and switch heuristics automatically. This is a coarse version of the selective HH that is easier to explain and validate.

### P2. Buffer-Informed Prediction for Hybrid(lambda)

**Novelty:** No existing work uses buffer contents as "perfect predictions" combined with ML-estimated predictions for unknown items.

**Idea:** Two-tier prediction system:
- **Tier 1 (perfect):** Buffer items have known dimensions -> eta = 0 for next 5-10 items
- **Tier 2 (estimated):** ML model predicts frequency of items beyond buffer -> eta > 0

Use `Hybrid(lambda)` with lambda = 1 for buffer items (full ProfilePacking trust) and learned lambda for predicted items. This gives the best of both worlds: optimal packing for visible items + robust packing for unknown items.

**Key files:**
- `python/hybrid_heuristic_ml/prediction_hybrid_framework_IDEAS.py`
- `python/semi_online_buffer/prediction_augmented_buffer_packing_IDEAS.py`

### P3. Cross-Bin Item Rearrangement via ToP Search

**Novelty:** No existing work considers moving items between active bins in a k-bounded setting.

**Idea:** When both bins are partially full and a large item doesn't fit in either, consider rearranging: move a small item from bin A to bin B, making space for the large item in bin A. This is only feasible for top-layer items (same constraint as Lee & Nam repacking).

The ToP MCTS can search over rearrangement moves as additional actions in the MCTS tree.

**Key file:** `python/multi_bin/cross_bin_rearrangement_ideas.py`

### P4. Expansion Scheme Portfolio as Hyper-Heuristic

**Novelty:** No existing work treats the choice of candidate generation scheme (CP vs. EP vs. EMS vs. EV) as a hyper-heuristic decision.

**Idea:** From `ExpansionSchemeSelector` in `pct_heuristic_integration_ideas.py`: Instead of always using EMS, train a bandit to choose the expansion scheme per packing step based on utilization, fragmentation, and item size. This is a **meta-level hyper-heuristic** operating above the placement-level hyper-heuristic.

### P5. Empirical Competitive Ratio Bounds for 3D Semi-Online BPP

**Novelty:** No known competitive ratio results exist for 3D online bin packing with buffer in any arrival model.

**Idea:** Run the `CompetitiveRatioEstimator` (from `coding_ideas_stochastic_competitive_ratios.py`) extended to 3D. While not proving theoretical bounds, empirical bounds over thousands of instances provide practical reference points that do not exist in the literature.

**From `random_order_competitive_bounds.py` -> `OnlineBoundsComparison.gap_analysis()`:**
> "There are NO known competitive ratio results for: 3D online knapsack in random order model, 3D online bin packing with buffer, 3D online k-bounded bin packing in random order. Any empirical or theoretical results here would be novel."

### P6. Heightmap Multi-Channel Representation

**Novelty:** Most existing work uses a single-channel heightmap. Using multiple channels (height, weight, stability, fragility) in a unified CNN backbone is underexplored for 3D BPP.

**Idea:** The `HybridCandidateScoringNet` in `ems_filtered_drl.py` accepts `in_channels=5`. Define channels as:
1. Height channel (standard)
2. Weight distribution channel
3. Stability support ratio channel (per cell, how well supported)
4. Item footprint channel (where the current item would go)
5. Buffer diversity channel (summary of buffer item sizes)

This provides the CNN with richer spatial information for scoring.

### P7. Future Work Directions

1. **3D Competitive Ratio Theory:** Prove bounds for 3D semi-online BPP with buffer. This would be a significant theoretical contribution but requires strong mathematical skills.

2. **Real-time Adaptive Lambda:** Instead of a fixed lambda for `Hybrid(lambda)`, use a neural network to predict the optimal lambda at each step based on observed prediction error so far.

3. **Transfer Learning Across Item Distributions:** Train the selective HH on distribution A (e.g., uniform), fine-tune on distribution B (e.g., warehouse). Measure how quickly it adapts and whether the Q-table structure transfers.

4. **Hardware-in-the-Loop Validation:** Integrate with a physical robot arm and conveyor belt to validate that the 2-second decision time is sufficient for real-time packing.

5. **Multi-Objective Pareto Front:** Instead of weighting fill rate and stability, explicitly compute the Pareto front using multi-objective optimization. Each architecture should produce a family of solutions parameterized by the stability weight.

---

## APPENDIX: Complete File Index

All Python files in the project and their roles in each architecture:

| File Path | ALPHA | BETA | GAMMA | DELTA | EPSILON | Role |
|-----------|:-----:|:----:|:-----:|:-----:|:-------:|------|
| `python/heuristics/coding_ideas_160_heuristic_framework.py` | X | X | X | X | X | Heuristic combinations |
| `python/heuristics/walle_heuristic_coding_ideas.py` | | X | | | | WallE scoring |
| `python/stability/stacking_tree_coding_ideas.py` | X | X | X | X | X | Stacking tree |
| `python/stability/lbcp_stability_and_rearrangement.py` | X | X | X | X | X | LBCP stability |
| `python/stability/feasibility_mask_stability.py` | X | X | X | X | X | Feasibility masks |
| `python/stability/coding_ideas_stability_vs_efficiency.py` | | | | X | X | Trade-off analysis |
| `python/semi_online_buffer/buffer_aware_packing.py` | X | X | X | X | X | Buffer management |
| `python/semi_online_buffer/buffer_with_lbcp_stability.py` | X | X | X | X | X | Buffer + stability |
| `python/semi_online_buffer/top_buffer_2bounded_coding_ideas.py` | | | X | | | ToP MCTS |
| `python/semi_online_buffer/buffer_mcts_policy_coding_ideas.py` | | | | X | | Buffer MCTS |
| `python/semi_online_buffer/buffer_mcts_selection.py` | | | | X | | MCTS selection |
| `python/semi_online_buffer/buffer_packman_coding_ideas.py` | | | | | X | Buffer + PackMan |
| `python/semi_online_buffer/coding_ideas_buffer_with_tree_search.py` | | | X | | | Buffer tree search |
| `python/semi_online_buffer/coding_ideas_lookahead_buffer_with_dual_bin.py` | | | | | X | Lookahead + dual bin |
| `python/semi_online_buffer/coding_ideas_buffer_stability_integration.py` | | | X | X | | Buffer + stability integration |
| `python/semi_online_buffer/prediction_augmented_buffer_packing_IDEAS.py` | | | | | | Prediction + buffer |
| `python/semi_online_buffer/coding_ideas_stochastic_blueprint_packing.py` | | | | | | Blueprint packing |
| `python/semi_online_buffer/random_order_sequential_packing.py` | | | | | | Random order packing |
| `python/multi_bin/coding_ideas_two_bounded_space.py` | X | X | X | X | X | k=2 bounded space |
| `python/multi_bin/two_bounded_bin_packing.py` | | | X | X | | 2-bin packing |
| `python/multi_bin/coding_ideas_dual_bin_replacement_strategies.py` | X | X | X | X | X | Bin replacement |
| `python/multi_bin/cross_bin_rearrangement_ideas.py` | | | | | | Cross-bin moves |
| `python/deep_rl/pct_coding_ideas.py` | | | X | | | PCT policy network |
| `python/deep_rl/packman_dqn_coding_ideas.py` | | X | | | | PackMan DQN |
| `python/deep_rl/deep_pack_3d_coding_ideas.py` | | | | | X | Deep-Pack 3D |
| `python/deep_rl/decomposed_actor_critic_coding_ideas.py` | | | | X | | Actor-critic |
| `python/deep_rl/constrained_drl_bin_packing.py` | | | | X | | CMDP packing |
| `python/deep_rl/coding_ideas_tsang2025_ddqn_dual_bin.py` | | | | | X | DDQN + MCA |
| `python/hybrid_heuristic_ml/coding_ideas_hierarchical_bpp.py` | | | X | | | Hierarchical search |
| `python/hybrid_heuristic_ml/ems_filtered_drl.py` | | X | X | | | EMS + DRL hybrid |
| `python/hybrid_heuristic_ml/feasibility_guided_packing_coding_ideas.py` | X | X | X | X | X | Placement rules + HH |
| `python/hybrid_heuristic_ml/pct_heuristic_integration_ideas.py` | | | X | | | PCT + heuristic |
| `python/hybrid_heuristic_ml/prediction_hybrid_framework_IDEAS.py` | | X | | | | Prediction HH |
| `python/hybrid_heuristic_ml/walle_packman_hybrid_coding_ideas.py` | | X | | | | WallE+PackMan hybrid |
| `python/theoretical_bounds/coding_ideas_stochastic_competitive_ratios.py` | | | | | | CR estimation |
| `python/theoretical_bounds/random_order_competitive_bounds.py` | | | | | | Theoretical bounds |

---

## APPENDIX: Quick Reference -- Class/Function Lookup

| Name | Type | File | Architecture |
|------|------|------|-------------|
| `Item` | dataclass | `coding_ideas_hierarchical_bpp.py` | All |
| `Bin` | class | `coding_ideas_hierarchical_bpp.py` | All |
| `Placement` | dataclass | `coding_ideas_hierarchical_bpp.py` | All |
| `StabilityChecker` | class | `coding_ideas_hierarchical_bpp.py` | All |
| `HierarchicalTreeSearch` | class | `coding_ideas_hierarchical_bpp.py` | GAMMA |
| `TreeNode` | dataclass | `coding_ideas_hierarchical_bpp.py` | GAMMA |
| `CandidateAction` | dataclass | `coding_ideas_hierarchical_bpp.py` | GAMMA |
| `PositionSelectionNetwork` | nn.Module | `coding_ideas_hierarchical_bpp.py` | GAMMA |
| `TwoBoundedSpaceManager` | class | `coding_ideas_hierarchical_bpp.py` | All |
| `BinClosingPolicy` | enum | `coding_ideas_hierarchical_bpp.py` | All |
| `BufferManager` | class | `coding_ideas_hierarchical_bpp.py` | All |
| `main_packing_loop()` | function | `coding_ideas_hierarchical_bpp.py` | All |
| `HybridCandidateScoringNet` | nn.Module | `ems_filtered_drl.py` | BETA, GAMMA |
| `CandidatePlacement` | dataclass | `ems_filtered_drl.py` | BETA, GAMMA |
| `generate_extreme_points_heightmap()` | function | `ems_filtered_drl.py` | All |
| `generate_candidates()` | function | `ems_filtered_drl.py` | BETA, GAMMA |
| `PlacementRule` | enum | `feasibility_guided_packing_coding_ideas.py` | All |
| `BinRoutingStrategy` | enum | `feasibility_guided_packing_coding_ideas.py` | All |
| `ItemSelectionStrategy` | enum | `feasibility_guided_packing_coding_ideas.py` | All |
| `SelectiveHyperHeuristic` | class | `feasibility_guided_packing_coding_ideas.py` | BETA |
| `HybridPackingSystem` | class | `feasibility_guided_packing_coding_ideas.py` | BETA |
| `score_dblf()` | function | `feasibility_guided_packing_coding_ideas.py` | All |
| `score_corner_distances()` | function | `feasibility_guided_packing_coding_ideas.py` | All |
| `score_dftrc()` | function | `feasibility_guided_packing_coding_ideas.py` | All |
| `score_back_bottom()` | function | `feasibility_guided_packing_coding_ideas.py` | All |
| `score_stability_aware()` | function | `feasibility_guided_packing_coding_ideas.py` | All |
| `CornerPointExpansion` | class | `pct_heuristic_integration_ideas.py` | GAMMA |
| `ExtremePointExpansion` | class | `pct_heuristic_integration_ideas.py` | GAMMA |
| `EventPointExpansion` | class | `pct_heuristic_integration_ideas.py` | GAMMA |
| `ExpansionSchemeSelector` | class | `pct_heuristic_integration_ideas.py` | GAMMA |
| `ConstraintRewards` | class | `pct_heuristic_integration_ideas.py` | GAMMA, DELTA |
| `PredictionGuidedHeuristicSelector` | class | `prediction_hybrid_framework_IDEAS.py` | BETA |
| `MLFrequencyPredictor` | class | `prediction_hybrid_framework_IDEAS.py` | BETA |
| `ConsistencyRobustnessEvaluator` | class | `prediction_hybrid_framework_IDEAS.py` | Evaluation |
| `PackingStrategy` | enum | `walle_packman_hybrid_coding_ideas.py` | BETA |
| `HyperHeuristicSelector` | class | `walle_packman_hybrid_coding_ideas.py` | BETA |
| `HybridAgent` | class | `walle_packman_hybrid_coding_ideas.py` | BETA |
| `compute_state_features()` | function | `walle_packman_hybrid_coding_ideas.py` | BETA |
| `compute_step_reward_for_selector()` | function | `walle_packman_hybrid_coding_ideas.py` | BETA |
| `StochasticBPSimulator` | class | `coding_ideas_stochastic_competitive_ratios.py` | Evaluation |
| `CompetitiveRatioEstimator` | class | `coding_ideas_stochastic_competitive_ratios.py` | Evaluation |
| `KnapsackRandomOrderBounds` | class | `random_order_competitive_bounds.py` | Theory |
| `GAPRandomOrderBounds` | class | `random_order_competitive_bounds.py` | Theory |
| `BufferAdvantageEstimator` | class | `random_order_competitive_bounds.py` | Theory |
| `OnlineBoundsComparison` | class | `random_order_competitive_bounds.py` | Theory |
| `ParameterOptimizer` | class | `random_order_competitive_bounds.py` | Theory |

---

*End of Section: Hybrid Systems + Theory + Complete Architectures*
