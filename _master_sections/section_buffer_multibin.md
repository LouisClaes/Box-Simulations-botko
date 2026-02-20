# Section I--L: Buffer Management, Multi-Bin Coordination, Synergies, and Novel Ideas

> **Scope**: Semi-online 3D bin packing with buffer of 5--10 items, k=2 bounded space (2 active pallets), maximizing fill rate + stability.
> **Framework**: Python + PyTorch thesis project.
> **Source files**: 12 Python files in `semi_online_buffer/`, 4 Python files in `multi_bin/`, 3 summary markdown files.

---

## I. Buffer Management Strategies

### I.1 Buffer Strategy Comparison Matrix

The table below consolidates **every buffer management strategy** found across all 12 `semi_online_buffer/` files. Each row is a distinct approach with its source file, key classes, strengths, weaknesses, and estimated performance.

| # | Strategy | Source File | Key Classes / Functions | Selection Logic | Estimated Perf. | Strengths | Weaknesses |
|---|----------|-------------|------------------------|-----------------|-----------------|-----------|------------|
| 1 | **Largest First** | `semi_online_buffer/buffer_aware_packing.py` | `LargestFirstStrategy.select()` | Sort buffer by volume descending; pick largest | Baseline | Simple; good for foundation layers | Ignores bin state; misses small-item gap-filling |
| 2 | **Best Fit First** | `semi_online_buffer/buffer_aware_packing.py` | `BestFitFirstStrategy.select()` | Evaluate each item against both bins; pick tightest fit | +2--4% over Largest | Bin-state aware | Greedy; no lookahead; can create fragmentation |
| 3 | **Urgency Weighted** | `semi_online_buffer/buffer_aware_packing.py` | `UrgencyWeightedStrategy.select()` | Score = volume * urgency_factor (FIFO age); pick highest | ~= Best Fit | Prevents buffer starvation | Urgency weight calibration needed |
| 4 | **MCTS Buffer Search (v1)** | `semi_online_buffer/buffer_mcts_policy_coding_ideas.py` | `BufferMCTS` (n_rollouts=200, max_depth=5), `SemiOnlinePackingSystem` | MCTS over item selection sequence; rollout with greedy heuristic | +5--10% over greedy | Lookahead; handles sequential dependencies | Compute-heavy (~2--4s per decision) |
| 5 | **MCTS Buffer Search (v2)** | `semi_online_buffer/buffer_mcts_selection.py` | `BufferMCTS` (_select, _expand, _simulate, _backpropagate), `ProgressiveWideningMCTS` (pw_constant=3.0, pw_alpha=0.5) | Alternative MCTS with progressive widening; critic-guided rollout | +5--12% over greedy | Scales to larger buffers; progressive widening limits branching | Implementation complexity; needs tuning pw_constant |
| 6 | **Greedy Buffer Selector** | `semi_online_buffer/buffer_mcts_selection.py` | `GreedyBufferSelector` (static: `largest_volume_first`, `best_fit_item`, `critic_guided`) | Three static strategies; critic_guided uses V(s) to rank items | Baseline to +3% | Fast; critic_guided is strong heuristic | No lookahead; critic accuracy dependent |
| 7 | **Hierarchical PackMan** | `semi_online_buffer/buffer_packman_coding_ideas.py` | `BufferItemFeatureExtractor` (10 features), `BinStateFeatureExtractor` (6 features), `MultiObjectiveReward` (fill=0.6, stability=0.4) | Level 1: DQN selects item from buffer; Level 2: PackMan DQN selects placement | +8--15% (estimated) | End-to-end learned; multi-objective | Requires training two networks; 3-phase training pipeline |
| 8 | **Tree Search Buffer** | `semi_online_buffer/coding_ideas_buffer_with_tree_search.py` | `SmartBufferManager` (strategies: TREE_SEARCH, LARGEST_FIRST, BEST_FIT, COMBINED_SCORE), `LookAheadEvaluator` (depth=3) | Tree search with discounted lookahead value; max_wait_time forces selection | +5--8% over greedy | Bounded computation; configurable depth | Exponential branching without pruning |
| 9 | **Lookahead Buffer + Dual Bin** | `semi_online_buffer/coding_ideas_lookahead_buffer_with_dual_bin.py` | `LookaheadBuffer`, `ItemSelectionStrategy` (6 strategies: largest_first, smallest_first, most_cubic_first, best_fit_for_bins, fifo, random_choice), `JointItemBinSelector`, `AdaptiveBufferManager` | Joint item+bin scoring; adaptive effective buffer size based on bin utilization | 77.11% (DRL, dual-bin, k=10, replaceMax from Tsang et al.) | Most comprehensive pipeline; adaptive buffer | Complex integration; many hyperparameters |
| 10 | **Stochastic Blueprint** | `semi_online_buffer/coding_ideas_stochastic_blueprint_packing.py` | `BoxDistributionLearner` (window=500), `BlueprintPacker3D` (delta=0.15), `UprightMatcher3D`, `DoublingTrickManager` (n0=50, mu=0.25) | Learn distribution -> create blueprint profile -> match buffer items to proxy slots | Near-optimal (1+eps) under i.i.d. | Theoretically grounded; exploits distribution regularity | Assumes i.i.d.; blueprint overhead; 3D proxy matching is approximate |
| 11 | **Prediction-Augmented Hybrid(lambda)** | `semi_online_buffer/prediction_augmented_buffer_packing_IDEAS.py` | `FrequencyTracker` (sliding window, L1 error), `ProfilePacking3D`, `HybridPacker3D` (lambda=0.5), `DynamicLambdaController`, `BufferAwareProfileSelector` | Two-tier: buffer = perfect short-term predictions; history = approximate long-term; Hybrid(lambda) blends profile-trusting vs robust heuristic | Hybrid(0.25--0.5) beats FirstFit/BestFit consistently | Principled consistency-robustness tradeoff; dynamic lambda | Profile construction in 3D is expensive; threshold calibration |
| 12 | **Sequential Buffer (Random Order)** | `semi_online_buffer/random_order_sequential_packing.py` | `SequentialBufferManager` (large/small decomposition), `ItemDistributionEstimator` (optimal_delta), `GAPBinAssigner` (score matrix) | Large items priority -> small items gap-fill; GAP matching for buffer-to-bin assignment | 1/6.65 competitive ratio (1D theory) | Theoretically motivated; GAP-based assignment | 1D theory; 3D adaptation needed; asymptotic guarantees |
| 13 | **ToP + Buffer + 2-Bounded** | `semi_online_buffer/top_buffer_2bounded_coding_ideas.py` | `ToP_MCTS_Planner` (num_simulations=200, c_puct=2.0), `ExtendedToP_MCTS`, `SemiOnline2BoundedPacker`, `BeamSearchPlanner` (beam_width=5, depth=3) | MCTS with actions = (item_index, bin_index); uses pre-trained PCT pi_theta for placement; global cache for path reuse | 88.3% (s=5) to 93.5% (s=10), Setting 2 from Zhao et al. (2025) | No retraining needed for buffer/multi-bin; best published results | Requires pre-trained PCT; MCTS computation cost |
| 14 | **Stability-Aware Buffer** | `semi_online_buffer/buffer_with_lbcp_stability.py` | `StabilityAwareBufferScorer` (w_util=1.0, w_stab=0.5, w_surface=0.3, w_rearrange=-0.1, w_flex=0.2) | Multi-criteria scoring: utilization + stability + surface quality + flexibility | ~= Best Fit + stability guarantee | Explicit stability integration; weighted multi-criteria | Weight tuning; LBCP computation per candidate |
| 15 | **Buffer-Stability Integration (9 Ideas)** | `semi_online_buffer/coding_ideas_buffer_stability_integration.py` | `BufferStabilitySelector` (alpha), `FoundationFirstSelector` (phase-based), `LookAheadStabilityAssessor`, `AdaptiveStabilityPolicy`, `StabilityAwareClosingPolicy`, `HeuristicSelector` (16 heuristics), `FragileItemRouter`, `RunningParetoTracker` | Joint item-placement optimization; phase-based selection (foundation/body/top-off); adaptive stability constraints by height | 80--85% of offline (estimated) | Most comprehensive stability treatment; phase-aware; 16 non-dominated heuristics | Complex; many interacting components |
| 16 | **Buffer Diversity Analyzer** | `semi_online_buffer/coding_ideas_lookahead_buffer_with_dual_bin.py` | `BufferDiversityAnalyzer` (volume_diversity, dim_diversity, item_rarity_score), `StabilityAwareItemSelector` (phase: early/middle/late) | Analyze buffer composition diversity; adjust strategy based on diversity metrics | Improves robustness | Prevents strategy lock-in; adapts to buffer composition | Diversity metrics need calibration |

**Recommendation for thesis**: Start with strategies #6 (Greedy: critic_guided) as baseline, then implement #13 (ToP MCTS) as the main approach, and integrate #15 (Stability-Aware) for the stability objective. This gives a clear progression: baseline -> search-based -> stability-integrated.

---

### I.2 MCTS-Based Buffer Search

Three distinct MCTS implementations exist across the codebase, each with different design choices.

#### Comparison of MCTS Variants

| Aspect | MCTS v1 | MCTS v2 | ToP Extended MCTS |
|--------|---------|---------|-------------------|
| **File** | `buffer_mcts_policy_coding_ideas.py` | `buffer_mcts_selection.py` | `top_buffer_2bounded_coding_ideas.py` |
| **Class** | `BufferMCTS` | `BufferMCTS`, `ProgressiveWideningMCTS` | `ToP_MCTS_Planner`, `ExtendedToP_MCTS` |
| **Action space** | Item index only | Item index only | (item_index, bin_index) joint |
| **Rollout policy** | Random + greedy heuristic | `LearnedRolloutPolicy` (trainable) | Pre-trained pi_theta (PCT policy) |
| **Simulations** | 200 | 200 | 200 (configurable up to 500) |
| **Max depth** | 5 | Configurable | 10 |
| **Progressive widening** | No | Yes (pw_constant=3.0, pw_alpha=0.5) | No (full expansion) |
| **Global cache** | No | No | Yes (path reuse across time steps) |
| **Bin closing integrated** | Via `BinClosingPolicy` (threshold, item_fit, marginal_gain) | Via `BinClosingPolicy` (no_fit, threshold, remaining_capacity, critic_based) | Via `BinClosingStrategy` (no_fit, value_threshold, utilization_threshold, hybrid) + `AdvancedBinClosingStrategy` (fragmentation_aware, comparative) |
| **Multi-bin aware** | No (single bin) | Via `BinReplacementStrategy` | Yes (native 2-bin support) |
| **Stability check** | Not integrated | Not integrated | Integrated via `StabilityVerificationPipeline` |
| **Beam search alternative** | No | No | Yes: `BeamSearchPlanner` (width=5, depth=3) |

#### Key Implementation Details -- ToP Extended MCTS (Recommended)

File: `C:\Users\Louis\Downloads\stapelalgortime\python\semi_online_buffer\top_buffer_2bounded_coding_ideas.py`

```python
class ExtendedToP_MCTS:
    """
    Recommended for thesis. Adds bin selection to standard ToP MCTS
    without modifying the pre-trained pi_theta.

    Action = (item_index, bin_index)
    Branching factor = buffer_size * 2  (e.g., 5*2=10 or 10*2=20)

    MCTS phases:
    1. SELECT: PUCT traversal (c_puct=2.0)
    2. EXPAND: all (item, bin) combinations where feasible
    3. EVALUATE: cumulative placed volume + V(.) from critic
    4. BACKPROPAGATE: update visit counts and values
    """
```

**Computational budget**: 200 simulations * ~20 children = ~4000 pi_theta forward passes per decision. At ~1ms each on GPU, this is ~4 seconds per placement decision, fitting within a 9.8-second robotic cycle time.

**Global cache**: At adjacent time steps, MCTS paths share common sub-sequences. The cache key includes both bins' utilizations and buffer item IDs. This nearly halves decision time in practice.

---

### I.3 Prediction-Augmented Buffer: Hybrid(lambda) Framework

Based on Angelopoulos et al. (2023), adapted in:
- `C:\Users\Louis\Downloads\stapelalgortime\python\semi_online_buffer\prediction_augmented_buffer_packing_IDEAS.py`
- Summary: `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\Online Bin Packing with Predictions (Summary).md`

#### Core Insight: Two-Tier Prediction

The buffer provides **perfect short-term predictions** (the next 5--10 items are known exactly), while historical data provides **approximate long-term predictions** (item frequency distribution estimated from sliding window).

```
Tier 1 (Buffer):   eta = 0  (perfect knowledge, deterministic)
Tier 2 (History):  eta > 0  (statistical estimate, degrades with distribution shift)
```

#### Key Classes

| Class | Purpose | Key Parameters |
|-------|---------|----------------|
| `FrequencyTracker` | Sliding window frequency estimation; discretizes 3D box types; L1 prediction error tracking | `window_size` (default 500), `num_bins_per_dim` |
| `ProfilePacking3D` | Constructs 3D profile from frequency predictions; computes offline-optimal template packing | `profile_size`, `delta` (large/small threshold) |
| `HybridPacker3D` | Implements Hybrid(lambda): blends profile-trusting packing with robust heuristic (DBLF) | `lambda_param` (default 0.5), `robust_heuristic` (default 'dblf') |
| `DynamicLambdaController` | Adjusts lambda dynamically based on prediction error (H-Aware variant) | Threshold-based switching |
| `BufferAwareProfileSelector` | Selects item from buffer that best advances toward the target profile | Combines buffer (Tier 1) with profile (Tier 2) |

#### Hybrid(lambda) Decision Rule

For each arriving item of type x:
- If a non-empty profile bin has a placeholder for x: always place there.
- Else if ppcount(x) <= lambda * count(x): use ProfilePacking (trust prediction).
- Else: use robust heuristic A (e.g., DBLF, largest-first).

#### Performance Expectations

From Angelopoulos et al. (2023) experiments:
- lambda=1.0 (pure ProfilePacking): near-optimal when predictions are correct, degrades to O(k) robustness.
- **lambda=0.25--0.5: best practical tradeoff** -- robust enough for large errors, significant benefit from good predictions.
- Even tiny prefix samples (338 items from 10^6) yield improvements over pure FirstFit/BestFit.

**Adaptation for 2-bounded space**: ProfilePacking's strategy of opening entire "profile groups" of bins is incompatible with k=2. Instead, use a sequential profile-guided approach: the profile provides a "target configuration" for each active bin, and the buffer items are matched to this target.

---

### I.4 Stochastic / Blueprint Packing

Based on Ayyadevara et al. (ICALP 2022), adapted in:
- `C:\Users\Louis\Downloads\stapelalgortime\python\semi_online_buffer\coding_ideas_stochastic_blueprint_packing.py`
- Summary: `C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\Near-optimal Algorithms for Stochastic Online Bin Packing (Summary).md`

#### Blueprint Packing Concept

1. **Learn distribution**: From observed items, estimate the i.i.d. distribution F (using `BoxDistributionLearner`, window_size=500).
2. **Build blueprint**: Compute offline-optimal packing of a profile drawn from F. Classify items as large (>= delta=0.15) or small.
3. **Online matching**: As buffer items arrive, match large items to proxy slots via upright matching (`UprightMatcher3D`). Pack small items into S-slots (remaining space) via Next-Fit.
4. **Doubling trick**: When stream length n is unknown, use `DoublingTrickManager` (n0=50, mu=0.25) to create super-stages of geometrically increasing length.

#### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `BoxDistributionLearner` | `coding_ideas_stochastic_blueprint_packing.py` | Sliding window distribution estimation; KS-like drift detection; i.i.d. plausibility test |
| `BlueprintPacker3D` | `coding_ideas_stochastic_blueprint_packing.py` | 3D blueprint construction (delta=0.15, buffer_size=7, k_bounded=2) |
| `UprightMatcher3D` | `coding_ideas_stochastic_blueprint_packing.py` | Maximum upright matching; online matching with buffer (buffer advantage: choose BEST item from buffer to match proxy slots) |
| `DoublingTrickManager` | `coding_ideas_stochastic_blueprint_packing.py` | Handles unknown stream length; n0=50, growth factor 1+mu=1.25 |
| `StochasticBlueprintOrchestrator` | `coding_ideas_stochastic_blueprint_packing.py` | End-to-end orchestration: distribution learning -> blueprint -> matching -> packing |

#### Buffer Advantage for Blueprint Packing

The original paper processes items one at a time sequentially. Our buffer of 5--10 items allows **choosing the best item** from the buffer to match proxy slots, dramatically improving match quality. This is the key adaptation that makes blueprint packing more practical in our setting.

#### Theoretical Performance

Under perfect i.i.d. assumptions: ECR of (1+epsilon) for any epsilon > 0 (Theorem 1.1 of Ayyadevara et al.). In practice with 3D and bounded space: this serves as an upper bound on achievable performance.

---

### I.5 Stability-Aware Buffer Selection

Stability integration with buffer selection spans multiple files, offering complementary approaches.

#### Approach A: Multi-Criteria Scoring (LBCP Integration)

File: `C:\Users\Louis\Downloads\stapelalgortime\python\semi_online_buffer\buffer_with_lbcp_stability.py`

Class: `StabilityAwareBufferScorer`

```python
score = (w_utilization * utilization_score +    # 1.0
         w_stability * stability_score +          # 0.5
         w_surface * surface_quality_score +       # 0.3
         w_rearrangement * rearrangement_penalty + # -0.1
         w_flexibility * flexibility_score)        # 0.2
```

The `score(item, bin_state, position, support_lbcp, num_operations)` method evaluates each (item, position) candidate from the buffer against this multi-criteria function. LBCP (Load-Bearing Contact Points) provides the stability_score component.

#### Approach B: 9-Idea Buffer-Stability Integration

File: `C:\Users\Louis\Downloads\stapelalgortime\python\semi_online_buffer\coding_ideas_buffer_stability_integration.py`

This file contains nine distinct ideas for integrating stability with buffer selection. The most important ones:

| Idea | Class | Mechanism | When to Use |
|------|-------|-----------|-------------|
| **Joint optimizer** | `BufferStabilitySelector` | Scores = alpha * fill_score + (1-alpha) * stability_score; searches all (item, position, bin) triples | General-purpose; tune alpha |
| **Foundation-First** | `FoundationFirstSelector` | Phase-based: <30% height = foundation (heavy/flat items), 30--70% = body, >70% = top-off (light items) | When stable foundations are critical |
| **Lookahead stability** | `LookAheadStabilityAssessor` | 1-step lookahead: what if we place item X, then which items from buffer could still be placed stably? | When buffer is large enough (>=7) |
| **Adaptive stability** | `AdaptiveStabilityPolicy` | Height-based constraint gradient: FullBaseSupport at <30%, PBP (Percentage Base Placement) at 30--60%, CoG (Center of Gravity) polygon at >60% | **Recommended**: strict where needed, relaxed where safe |
| **Stability-aware closing** | `StabilityAwareClosingPolicy` | 5 closing policies: CLOSE_LEAST_FILLED, CLOSE_MOST_FILLED, CLOSE_LEAST_STABLE, CLOSE_COMBINED, CLOSE_WORST_EMS | When closing decision affects stability |
| **Hyper-heuristic** | `HeuristicSelector` | 16 non-dominated heuristics selected by online performance tracking; `RunningParetoTracker` maintains Pareto front | When no single heuristic dominates |
| **Fragile item router** | `FragileItemRouter` | Routes fragile items to bins with better support conditions | When item fragility varies |

#### Approach C: Hierarchical Multi-Objective Reward

File: `C:\Users\Louis\Downloads\stapelalgortime\python\semi_online_buffer\buffer_packman_coding_ideas.py`

Class: `MultiObjectiveReward` (fill_weight=0.6, stability_weight=0.4)

Integrated into the DQN training reward signal. The item selector network learns to choose items that jointly optimize fill rate AND stability, rather than applying stability as a post-hoc filter.

#### Recommended Stability Pipeline

Combine Approach B's `AdaptiveStabilityPolicy` (height-based constraint switching) with Approach A's multi-criteria scoring:

1. Filter buffer items using `AdaptiveStabilityPolicy` (strict at bottom, relaxed at top).
2. Score surviving candidates using `StabilityAwareBufferScorer`.
3. Select the highest-scoring (item, position, bin) triple.

Estimated performance: buffer of 10 + k=2 + adaptive stability -> match offline in 80--85% of cases.

---

### I.6 Item Selection Strategies Comparison

All distinct item selection strategies found across all files, organized by complexity.

#### Simple Heuristics (O(B) per decision, B = buffer size)

| Strategy | Source | Logic | Best For |
|----------|--------|-------|----------|
| `largest_volume_first` | `buffer_mcts_selection.py` (`GreedyBufferSelector`) | Sort by volume, pick largest | Foundation building; greedy baseline |
| `smallest_first` | `coding_ideas_lookahead_buffer_with_dual_bin.py` (`ItemSelectionStrategy`) | Sort by volume, pick smallest | Gap-filling phase |
| `most_cubic_first` | `coding_ideas_lookahead_buffer_with_dual_bin.py` (`ItemSelectionStrategy`) | Sort by aspect_ratio (closest to 1.0), pick most cubic | Stability (cubic items are most stable) |
| `fifo` | `coding_ideas_lookahead_buffer_with_dual_bin.py` (`ItemSelectionStrategy`) | First-in-first-out from buffer | Fairness baseline; preventing starvation |
| `random_choice` | `coding_ideas_lookahead_buffer_with_dual_bin.py` (`ItemSelectionStrategy`) | Random selection from buffer | Stochastic baseline |

#### Bin-Aware Heuristics (O(B * C) per decision, C = candidate positions)

| Strategy | Source | Logic | Best For |
|----------|--------|-------|----------|
| `best_fit_item` | `buffer_mcts_selection.py` (`GreedyBufferSelector`) | For each item, find best position in each bin; pick tightest fit | Maximizing per-step utilization |
| `best_fit_for_bins` | `coding_ideas_lookahead_buffer_with_dual_bin.py` (`ItemSelectionStrategy`) | Same as above but considers both bins jointly | 2-bounded space |
| `critic_guided` | `buffer_mcts_selection.py` (`GreedyBufferSelector`) | Use V(s') from trained critic to rank items; pick item with highest post-placement value | When critic network is available |
| `urgency_weighted` | `buffer_aware_packing.py` (`UrgencyWeightedStrategy`) | Score = volume * urgency(age); higher urgency for older items | Preventing buffer overflow |

#### Search-Based (O(simulations * branching) per decision)

| Strategy | Source | Logic | Best For |
|----------|--------|-------|----------|
| MCTS search | `buffer_mcts_selection.py`, `top_buffer_2bounded_coding_ideas.py` | Tree search over item sequences | Maximum quality; when compute budget allows |
| Beam search | `top_buffer_2bounded_coding_ideas.py` (`BeamSearchPlanner`) | Deterministic beam search (width=5, depth=3) | Simpler alternative to MCTS; reproducible results |
| Tree search | `coding_ideas_buffer_with_tree_search.py` (`SmartBufferManager`) | Configurable depth lookahead with discounted values | Moderate compute budget |

#### Learning-Based (O(1) inference after training)

| Strategy | Source | Logic | Best For |
|----------|--------|-------|----------|
| DQN item selector | `buffer_packman_coding_ideas.py` | Learned item selection from buffer features | End-to-end optimization |
| RL bin selector | `coding_ideas_two_bounded_space.py` (`BinSelectionStrategy.RL_BIN_SELECTOR`) | Learned bin selection policy | When heuristic bin selection is suboptimal |
| Learned rollout | `buffer_mcts_selection.py` (`LearnedRolloutPolicy`) | Trained rollout policy for MCTS simulations | Improving MCTS simulation quality |

---

## J. Multi-Bin Coordination (k=2 Bounded Space)

### J.1 Bin Selection Strategies

All bin selection strategies found across the 4 `multi_bin/` files and relevant `semi_online_buffer/` files.

#### Strategy Comparison

| Strategy | Source File | Class / Method | Logic | Computational Cost |
|----------|------------|---------------|-------|-------------------|
| **First Fit** | `multi_bin/two_bounded_bin_packing.py` | `BinCoordinator._first_fit_selection()` | First bin where item fits | O(2) -- trivial |
| **Best Fit** | `multi_bin/two_bounded_bin_packing.py` | `BinCoordinator._best_fit_selection()` | Fullest bin where item fits | O(2) |
| **Balanced** | `multi_bin/two_bounded_bin_packing.py` | `BinCoordinator._balanced_selection()` | Emptiest bin where item fits (keeps flexibility) | O(2) |
| **Specialized** | `multi_bin/two_bounded_bin_packing.py` | `BinCoordinator._specialized_selection()` | Bin 0 for large items (vol > median), Bin 1 for small items; Harmonic-inspired | O(2) |
| **Critic-Based** | `multi_bin/two_bounded_bin_packing.py` | `BinCoordinator._critic_selection()` | Use V(s') from trained critic; select bin with highest post-placement value | O(2) forward passes |
| **Best Fit (enum)** | `multi_bin/coding_ideas_two_bounded_space.py` | `BinSelectionStrategy.BEST_FIT` via `TwoBoundedBinSelector` | Highest utilization that can still accommodate item | O(2) |
| **Worst Fit (enum)** | `multi_bin/coding_ideas_two_bounded_space.py` | `BinSelectionStrategy.WORST_FIT` via `TwoBoundedBinSelector` | Lowest utilization (maximize future flexibility) | O(2) |
| **Tree Search Score** | `multi_bin/coding_ideas_two_bounded_space.py` | `BinSelectionStrategy.TREE_SEARCH_SCORE` via `TwoBoundedBinSelector` | Score bins via tree search lookahead; select highest-scored | O(simulations) |
| **Sequential Fill** | `multi_bin/coding_ideas_two_bounded_space.py` | `BinSelectionStrategy.SEQUENTIAL_FILL` via `TwoBoundedBinSelector` | Fill one bin to threshold, then switch to the other | O(1) |
| **RL Bin Selector** | `multi_bin/coding_ideas_two_bounded_space.py` | `BinSelectionStrategy.RL_BIN_SELECTOR` via `TwoBoundedBinSelector` | Learned policy mapping (bin_states, item) -> bin_index | O(1) inference |
| **Size Specialization** | `multi_bin/coding_ideas_two_bounded_space.py` | `BinSelectionStrategy.SIZE_SPECIALIZATION` via `TwoBoundedBinSelector` | Route items to bins based on size class (analogous to Harmonic algorithms) | O(1) |
| **Cross-Bin Spatial Ensemble** | `semi_online_buffer/top_buffer_2bounded_coding_ideas.py` | `CrossBinSpatialEnsemble.select_best_action()` | Normalize pi_theta scores to ranks within each bin; compare ranks across bins (fair comparison even at different fill levels) | O(B*2*L) |
| **Dual-PCT Bin Selector** | `semi_online_buffer/top_buffer_2bounded_coding_ideas.py` | `DualPCT_BinSelector` (Option A) | Learned bin selection head: GAT encodes both bins -> MLP softmax over [bin_0, bin_1] | O(2) forward passes |
| **Joint Multi-Bin PCT** | `semi_online_buffer/top_buffer_2bounded_coding_ideas.py` | `JointMultiBinPCT` (Option C) | Single PCT encoding BOTH bins; bin_id feature appended; pointer selects (leaf, bin) jointly | O(N^2) GAT, N=nodes in both bins |
| **Joint Item-Bin Selection** | `semi_online_buffer/coding_ideas_lookahead_buffer_with_dual_bin.py` | `JointItemBinSelector` (scoring: volume_fit, reward_based, stability_weighted) | Joint scoring of (item, bin) pairs; select best pair | O(B*2) |

**Key insight from `two_bounded_bin_packing.py`**: With only 2 bins, bin selection is less critical than item selection from the buffer. The buffer is the main lever for performance. Recommendation: use simple Best Fit or Critic-Based for bin selection, invest more compute in buffer item selection (MCTS).

**Expected performance boost from 2 bins** (from Zhao et al. 2021 multi-bin data):
- 1 bin: 67.4% (CUT-2)
- 2 bins: ~68--70% (interpolated)
- With buffer k=10: +12--15%
- Combined (2-bounded + buffer-10): estimated 78--85%

---

### J.2 Bin Replacement / Closing Strategies

#### J.2.1 Replacement Strategies (Which bin to close when one must be replaced)

File: `C:\Users\Louis\Downloads\stapelalgortime\python\multi_bin\coding_ideas_dual_bin_replacement_strategies.py`

| Strategy | Class | Logic | Performance (Tsang et al.) |
|----------|-------|-------|---------------------------|
| **ReplaceAll** | `ReplaceAll` | Close the bin that just received the item (close-on-pack) | 75.89% (DRL, dual-bin, k=10) |
| **ReplaceMax** | `ReplaceMax` | Close the bin with highest utilization | **77.11%** (best in Tsang et al.) |
| **ReplaceMin** | `ReplaceMin` | Close the bin with lowest utilization | Suboptimal (wastes nearly empty bin) |
| **ReplaceFragmented** | `ReplaceFragmented` | Close the bin with highest fragmentation (1 - max_ems_vol / remaining_vol) | Expected ~76--77% |
| **ReplaceThreshold** | `ReplaceThreshold` | Close any bin exceeding utilization threshold (configurable) | Threshold-dependent |
| **ReplaceLookaheadAware** | `ReplaceLookaheadAware` | Consider buffer contents; close the bin that is least useful for buffered items | Expected improvement over ReplaceMax |
| **ReplaceComposite** | `ReplaceComposite` | Weighted multi-criteria: utilization, fragmentation, fit for buffer items | Tunable |
| **ReplaceByQLearning** | `ReplaceByQLearning` | Learned policy; FC network 14->64->32->3 (keep_both, close_0, close_1) | Expected best with training |
| **ReplaceProactive** | `ReplaceProactive` | Close bins before getting stuck (proactive vs reactive) | Prevents deadlocks |

**Key finding**: ReplaceMax outperforms ReplaceAll by +1.53--2.35% (Tsang et al. 2025 results). The intuition: closing the fuller bin captures more value, while the other bin retains flexibility.

#### J.2.2 Bin Closing Triggers (When to close a bin)

File: `C:\Users\Louis\Downloads\stapelalgortime\python\multi_bin\coding_ideas_two_bounded_space.py`

Class: `BinClosingController` with `BinClosingConfig`

| Trigger | Logic | Default Threshold |
|---------|-------|-------------------|
| `min_utilization` | Only close if utilization >= threshold | 0.70 |
| `never_close_both` | At most one bin closed per step | N/A (hard rule) |
| `no_fit` | Close if no buffer item fits | N/A |
| `target_utilization` | Close when target utilization reached | 0.90 |
| `steps_without_placement` | Close after N consecutive failed placements | 5 steps |

File: `C:\Users\Louis\Downloads\stapelalgortime\python\semi_online_buffer\top_buffer_2bounded_coding_ideas.py`

Classes: `BinClosingStrategy` and `AdvancedBinClosingStrategy`

| Strategy | Logic | Source |
|----------|-------|--------|
| `no_fit` | Close when no buffer item fits in this bin | `BinClosingStrategy`, `AdvancedBinClosingStrategy._no_fit()` |
| `value_threshold` | Close when V(.) from critic < threshold (default 0.05) | `AdvancedBinClosingStrategy._value_threshold()` |
| `utilization_threshold` | Close when utilization >= threshold (default 0.85) | `AdvancedBinClosingStrategy._utilization_threshold()` |
| **`hybrid` (RECOMMENDED)** | Close if: (no_fit) OR (utilization > 0.85 AND V(.) < 0.10) | `AdvancedBinClosingStrategy._hybrid()` |
| `fragmentation_aware` | Close if remaining space is too fragmented (fragmentation > 0.3) AND smallest buffer item won't fit | `AdvancedBinClosingStrategy._fragmentation_aware()` |
| `comparative` | When BOTH bins > 80% utilization, close the one with lower V(.) | `AdvancedBinClosingStrategy._comparative()` |

File: `C:\Users\Louis\Downloads\stapelalgortime\python\semi_online_buffer\coding_ideas_buffer_stability_integration.py`

Class: `StabilityAwareClosingPolicy`

| Policy | Logic |
|--------|-------|
| `CLOSE_LEAST_FILLED` | Close the least-filled bin (preserve the fuller one) |
| `CLOSE_MOST_FILLED` | Close the most-filled bin (capture value) |
| `CLOSE_LEAST_STABLE` | Close the bin with the worst stability score |
| `CLOSE_COMBINED` | Weighted combination of utilization and stability |
| `CLOSE_WORST_EMS` | Close the bin whose remaining EMS spaces are least usable |

**Critical design point**: Bin closing is **irreversible** in k-bounded space. Poor closing wastes space permanently. The `comparative` and `hybrid` strategies are recommended because they account for both bins' states before making a closing decision.

---

### J.3 Cross-Bin Rearrangement

File: `C:\Users\Louis\Downloads\stapelalgortime\python\multi_bin\cross_bin_rearrangement_ideas.py`

This file proposes MCTS-based cross-bin rearrangement for k=2, based on Gao et al. (2025).

#### Key Insight

The second bin serves as an **extended staging area** for rearrangement. Items can be migrated between the two active bins to improve the packing quality of one (or both) before closing.

#### Classes

| Class | Purpose | Key Parameters |
|-------|---------|----------------|
| `BinClosingPolicy` | When to trigger closing; requires buffer_exhaustion before closing | `min_util_to_close=0.65`, `buffer_exhaustion_required=True` |
| `BinAllocationScorer` | Multi-criteria bin scoring for item allocation | `w_utilization=1.0`, `w_height=-0.2`, `w_operations=-0.05`, `w_support=0.3`, `w_compactness=0.2` |
| `CrossBinMCTSNode` | MCTS node tracking unpacked items AND cross-bin moves | Tracks both item placement and item migration actions |
| `CrossBinMCTS` | MCTS for cross-bin search | `max_nodes=100`, `max_depth=6`, `max_children=3`, `max_cross_moves=3` |
| `StabilityAwareBinCloser` | Computes closing score considering stability; optionally runs final SRP (Stability Recovery Procedure) before closing | `compute_closing_score()`, `recommend_close()`, `final_srp_before_close()` |
| `DualBinPipelineController` | Complete pipeline: `receive_item()` -> `_process_buffer()` in 3 phases: (1) try place in both bins, (2) cross-bin rearrange if needed, (3) close if stuck | Full integration |

#### Cross-Bin Move Types

1. **Direct migration**: Move an item from bin A to bin B (if it fits better there).
2. **Swap**: Exchange items between bins to improve both.
3. **Chain migration**: Move item from A to B, then another from B to A.

#### Constraints

- Maximum 3 cross-bin moves per step (to keep runtime bounded).
- Cross-bin moves must maintain stability in BOTH bins.
- Physical feasibility: items can only be removed from the top layer (no excavation).

#### Expected Performance

78--88% utilization WITH guaranteed stability (estimated from Gao et al. 2025 results).

---

### J.4 Complete k=2 Pipeline

Synthesizing all multi-bin components into a single pipeline.

#### Architecture: Three Options

From `top_buffer_2bounded_coding_ideas.py`, Sections 9--11:

| Option | Architecture | Reuses Pre-trained pi_theta? | Recommended? |
|--------|-------------|------------------------------|--------------|
| **A: Dual-PCT with Bin Selection Head** | Two PCTs (shared weights) + learned bin selector MLP + pointer mechanism | Yes (shared encoder) | If end-to-end learning preferred |
| **B: Extended ToP Search (RECOMMENDED)** | Same pi_theta, MCTS searches over (item_index, bin_index) | Yes, **no modifications needed** | **Yes -- primary thesis approach** |
| **C: Joint Multi-Bin PCT** | Single PCT encoding both bins; bin_id feature; pointer selects (leaf, bin) jointly | No (retrain from scratch) | Research extension only |

#### Recommended Pipeline (Option B)

File: `C:\Users\Louis\Downloads\stapelalgortime\python\semi_online_buffer\top_buffer_2bounded_coding_ideas.py`

```
[Item Stream]
    |
    v
[Buffer Manager (s=5-10)]  -- FIFO refill from stream
    |
    v
[Bin Closing Check]  -- AdvancedBinClosingStrategy('hybrid')
    |                     Close if: no_fit OR (util>85% AND V(.)<0.10)
    |
    v
[Generate candidates for all buffer items in both bins]
    |
    v
[ExtendedToP_MCTS]  -- 200 simulations, c_puct=2.0
    |                    Action = (item_index, bin_index)
    |                    Global cache for path reuse
    |
    v
[CrossBinSpatialEnsemble]  -- Normalize scores to ranks per bin
    |                          Fair comparison across different fill levels
    |
    v
[StabilityVerificationPipeline]
    |  Training: quasi-static filter (support area >= 0.80)
    |  Testing: PyBullet physics verification (k_l=5, k_d=4)
    |
    v
[Execute Placement]  -- PCT.place_item()
    |
    v
[Update PCT + EMS + Buffer]
    |
    v
[Refill buffer from stream]
    |
    v
[ComprehensiveMetrics recording]
```

#### Training Strategy (5 Phases)

From `top_buffer_2bounded_coding_ideas.py`, Section 6:

| Phase | Duration | What | Validation Target |
|-------|----------|------|-------------------|
| 1 | 12--24h GPU | Train base PCT on single-bin online packing (PPO, 8--16 parallel envs) | 70--76% utilization |
| 2 | No retraining | Validate ToP buffer planning with trained pi_theta + MCTS (buffer=5) | +5--15% over pure online |
| 3 | No retraining | Extend to 2-bounded space: add bin selection to MCTS, implement bin closing | Validate on simulated streams |
| 4 | 8--16h GPU | Add stability: retrain pi_theta with stability-aware reward (w_t = max(0, v_t + c*f_stability)) | Stability pass rate > 95% |
| 5 (optional) | 4--8h | Fine-tune on real warehouse distribution if available | Distribution-specific improvement |

Key advantage of Option B: **Phases 2--3 require NO retraining**. The same pi_theta works for any buffer size, any number of bins.

#### Computational Budget per Decision

| Component | Time |
|-----------|------|
| MCTS (200 sims, 20 children) | ~4s on GPU |
| Stability verification (k_l=5, k_d=4) | ~2s |
| Buffer management + state update | ~0.1s |
| **Total** | **~6s** (within 9.8s robotic cycle) |

---

## K. Buffer + Multi-Bin Synergies

### K.1 How Buffer and Multi-Bin Amplify Each Other

The buffer and multi-bin coordination are not independent improvements -- they create synergies:

1. **Buffer enables better bin selection**: With only the current item, bin selection is a coin flip. With 5--10 items visible, we can match items to bins optimally (e.g., large items to the specialized bin, gap-filling items to the nearly-full bin).

2. **Multi-bin enables better buffer utilization**: With a single bin, some buffer items may not fit at all. With 2 bins, the probability that at least ONE bin can accommodate any given buffer item is much higher, reducing item waste.

3. **Cross-bin rearrangement enables buffer-informed closing**: Knowing the buffer contents informs when to close a bin (if buffer items won't fit) and which items to migrate between bins before closing.

4. **Buffer reduces the "bounded space penalty"**: The main cost of k-bounded space is premature closing. The buffer mitigates this by allowing the algorithm to "look ahead" and avoid filling bins with items that will cause suboptimal closing.

### K.2 Integrated Pipeline Classes

| Component | Class | File | Role in Integration |
|-----------|-------|------|---------------------|
| Buffer management | `LookaheadBuffer` | `coding_ideas_lookahead_buffer_with_dual_bin.py` | FIFO-refillable buffer with adaptive effective size |
| Joint selection | `JointItemBinSelector` | `coding_ideas_lookahead_buffer_with_dual_bin.py` | Scores all (item, bin) pairs jointly |
| Bin management | `BufferAwareBinManager` | `coding_ideas_lookahead_buffer_with_dual_bin.py` | ReplaceMax/ReplaceAll with buffer awareness |
| Diversity analysis | `BufferDiversityAnalyzer` | `coding_ideas_lookahead_buffer_with_dual_bin.py` | Adjusts strategy based on buffer composition |
| Multi-bin manager | `MultiBinManager` | `coding_ideas_dual_bin_replacement_strategies.py` | Manages active bins with replacement strategies |
| Full pipeline | `FullPipelineManager` | `coding_ideas_dual_bin_replacement_strategies.py` | End-to-end: buffer -> item select -> bin select -> place -> replace |
| 2-bounded pipeline | `TwoBoundedSpacePipeline` | `coding_ideas_two_bounded_space.py` | Complete k=2 pipeline with configurable bin selection and closing |
| Dual bin controller | `DualBinPipelineController` | `cross_bin_rearrangement_ideas.py` | Pipeline with cross-bin rearrangement support |
| Complete semi-online | `CompleteSemiOnlinePipeline` | `coding_ideas_lookahead_buffer_with_dual_bin.py` | Full integration: buffer + dual bin + item selection + closing |
| ToP packer | `SemiOnline2BoundedPacker` | `top_buffer_2bounded_coding_ideas.py` | MCTS-based packer with stability and metrics |
| Engine | `SemiOnlinePackingEngine` | `coding_ideas_buffer_stability_integration.py` | Integrates buffer + stability + closing into single engine |

### K.3 Performance Projections

Based on data from the papers and coding ideas:

| Configuration | Expected Utilization | Source |
|---------------|---------------------|--------|
| Online, single bin, no buffer | 66.9% | Zhao et al. (2021) CUT-2 |
| Online, 2 bins, no buffer | ~68--70% | Interpolated from Zhao et al. multi-bin |
| Buffer k=5, single bin | 88.3% | Zhao et al. (2025) ToP, Setting 2 |
| Buffer k=10, single bin | 93.5% | Zhao et al. (2025) ToP, Setting 2 |
| DRL, dual bin, buffer k=10, ReplaceMax | 77.11% | Tsang et al. (2025) |
| Buffer k=5, 2-bounded, MCTS+stability (thesis target) | **78--85%** | Estimated composite |
| Buffer k=10, 2-bounded, MCTS+stability (thesis target) | **82--90%** | Estimated composite |
| Offline upper bound | ~95--98% | Literature reference |

Note: The estimates for the thesis target account for the stability constraint penalty (~3--5% reduction from pure fill-rate optimization) and the 2-bounded space penalty (~2--5% from suboptimal closing).

### K.4 Synergy: Buffer-Informed Replacement

The most concrete synergy is **buffer-informed replacement** from `ReplaceLookaheadAware` in `coding_ideas_dual_bin_replacement_strategies.py`:

```python
class ReplaceLookaheadAware(ReplacementStrategy):
    """
    Uses buffer contents to decide which bin to close.

    Logic: For each bin, count how many buffer items could fit.
    Close the bin that accommodates FEWER buffer items
    (it has less future potential).
    """
```

This directly leverages the buffer to improve multi-bin closing quality. Without a buffer, closing decisions are myopic; with a buffer, they account for the known near-future.

---

## L. Novel Ideas for Buffer + Multi-Bin

### L.1 Ideas from the Codebase

The following novel ideas emerge from the coding idea files but are not yet fully implemented.

#### L.1.1 Adaptive Buffer Size

File: `C:\Users\Louis\Downloads\stapelalgortime\python\semi_online_buffer\coding_ideas_lookahead_buffer_with_dual_bin.py`

Class: `AdaptiveBufferManager`

**Idea**: Dynamically adjust the effective buffer size based on bin utilization. When bins are nearly empty (early phase), use a smaller effective buffer (make quick decisions). When bins are nearly full (critical phase), use the full buffer (deliberate carefully).

```
effective_buffer = min(physical_buffer, base + scale * max(bin_utilizations))
```

This is novel because all existing work uses fixed buffer sizes.

#### L.1.2 Pareto-Tracked Hyper-Heuristic Selection

File: `C:\Users\Louis\Downloads\stapelalgortime\python\semi_online_buffer\coding_ideas_buffer_stability_integration.py`

Classes: `HeuristicSelector` (16 non-dominated heuristics), `RunningParetoTracker`

**Idea**: Maintain a running Pareto front of (fill_rate, stability) across 16 candidate heuristics. At each step, select the heuristic that is currently on the Pareto front and has the best recent performance.

The 16 heuristics are combinations of:
- 4 item selection strategies (largest, best_fit, most_stable, random)
- 2 stability constraints (strict, relaxed)
- 2 bin preferences (best_fit, balanced)

#### L.1.3 Dynamic Lambda Controller for Predictions

File: `C:\Users\Louis\Downloads\stapelalgortime\python\semi_online_buffer\prediction_augmented_buffer_packing_IDEAS.py`

Class: `DynamicLambdaController`

**Idea**: Adjust the Hybrid(lambda) trust parameter dynamically based on observed prediction accuracy. When the buffer items match the predicted distribution well (low L1 error), increase lambda (trust predictions more). When unusual items appear, decrease lambda (fall back to robust heuristic).

This extends Angelopoulos et al.'s static lambda to a dynamic, adaptive version that responds to distribution shifts in real-time.

#### L.1.4 Cross-Bin MCTS with Stability-Aware Closing

File: `C:\Users\Louis\Downloads\stapelalgortime\python\multi_bin\cross_bin_rearrangement_ideas.py`

Classes: `CrossBinMCTS`, `StabilityAwareBinCloser`

**Idea**: Before closing a bin, run a final "Stability Recovery Procedure" (SRP) that uses MCTS to search for item migrations between bins that improve the stability of the bin about to be closed, WITHOUT decreasing its utilization.

```python
def final_srp_before_close(self, bin_to_close, other_bin):
    """
    Search for item swaps that improve stability of bin_to_close.
    Only accept moves that maintain or improve utilization of BOTH bins.
    """
```

#### L.1.5 Blueprint Matching with Buffer Advantage

File: `C:\Users\Louis\Downloads\stapelalgortime\python\semi_online_buffer\coding_ideas_stochastic_blueprint_packing.py`

Class: `UprightMatcher3D`

**Idea**: In standard blueprint packing, items arrive one at a time and must be matched greedily. With a buffer of 5--10 items, we can solve a small **optimal matching** problem: match the B buffer items to B proxy slots in the blueprint to minimize total mismatch. This is solvable in O(B^3) with the Hungarian algorithm, which is trivially fast for B=10.

#### L.1.6 Comparative Bin Closing with Both-Bin State

File: `C:\Users\Louis\Downloads\stapelalgortime\python\semi_online_buffer\top_buffer_2bounded_coding_ideas.py`

Class: `AdvancedBinClosingStrategy._comparative()`

**Idea**: When BOTH bins exceed 80% utilization, close the bin with lower future value V(.). This avoids the pathological case where both bins are nearly full and no items fit -- by proactively closing one, we get a fresh empty bin. The `comparative` strategy is unique to k=2 because it requires comparing exactly two bins.

#### L.1.7 Q-Learning for Replacement Decisions

File: `C:\Users\Louis\Downloads\stapelalgortime\python\multi_bin\coding_ideas_dual_bin_replacement_strategies.py`

Class: `ReplaceByQLearning` (FC network: 14 -> 64 -> 32 -> 3 outputs)

**Idea**: Learn the replacement decision (keep_both, close_bin_0, close_bin_1) via Q-learning. State features: utilization of both bins (2), fragmentation of both bins (2), buffer size (1), buffer diversity metrics (3), items packed in each bin (2), steps since last close (2), mean/var of closed bin utils (2). Total: 14 features.

This is novel because existing work uses hand-crafted replacement rules.

### L.2 Cross-Cutting Novel Contributions for the Thesis

Based on the comprehensive analysis of all files, the following represent the most promising novel contributions:

1. **Extended ToP MCTS for 2-bounded space** (Option B from `top_buffer_2bounded_coding_ideas.py`): No prior work combines ToP search with k-bounded space. This is the primary thesis contribution.

2. **Adaptive Stability Policy with height-based constraint gradient** (`coding_ideas_buffer_stability_integration.py`): The idea of using FullBaseSupport at the bottom and CoG polygon at the top is novel and practical.

3. **Buffer-informed bin closing** (combining `ReplaceLookaheadAware` from `coding_ideas_dual_bin_replacement_strategies.py` with `AdvancedBinClosingStrategy` from `top_buffer_2bounded_coding_ideas.py`): Using buffer contents to decide WHEN and WHICH bin to close.

4. **Two-tier prediction (buffer + history)** from `prediction_augmented_buffer_packing_IDEAS.py`: Formally connecting the buffer as "perfect short-term predictions" with historical frequency data as "approximate long-term predictions" in the Hybrid(lambda) framework.

5. **Cross-bin spatial ensemble ranking** (`CrossBinSpatialEnsemble` from `top_buffer_2bounded_coding_ideas.py`): Fair comparison of placements across bins with different fill levels by converting absolute scores to normalized ranks.

### L.3 Implementation Priority Roadmap

| Priority | Component | Files to Implement | Dependencies | Est. Time |
|----------|-----------|-------------------|--------------|-----------|
| **P0** | Base PCT training (single bin, online) | `pct_coding_ideas.py` (external) | PyTorch, gymnasium | 2--3 weeks |
| **P1** | Buffer manager + greedy selection | `buffer_aware_packing.py` | P0 | 1 week |
| **P2** | 2-bounded bin manager + BinCoordinator | `two_bounded_bin_packing.py`, `coding_ideas_two_bounded_space.py` | P0 | 1 week |
| **P3** | ExtendedToP_MCTS (Option B) | `top_buffer_2bounded_coding_ideas.py` | P0, P1, P2 | 2 weeks |
| **P4** | Bin closing strategies (hybrid + comparative) | `top_buffer_2bounded_coding_ideas.py`, `coding_ideas_two_bounded_space.py` | P2, P3 | 1 week |
| **P5** | Stability integration (AdaptiveStabilityPolicy) | `coding_ideas_buffer_stability_integration.py`, `buffer_with_lbcp_stability.py` | P3 | 1--2 weeks |
| **P6** | ReplaceMax/ReplaceLookaheadAware | `coding_ideas_dual_bin_replacement_strategies.py` | P2, P4 | 1 week |
| **P7** | Evaluation framework (ComprehensiveMetrics) | `top_buffer_2bounded_coding_ideas.py` | P3 | 3 days |
| **P8** (optional) | Cross-bin rearrangement | `cross_bin_rearrangement_ideas.py` | P2, P5 | 2 weeks |
| **P9** (optional) | Prediction-augmented Hybrid(lambda) | `prediction_augmented_buffer_packing_IDEAS.py` | P1 | 2 weeks |
| **P10** (optional) | Blueprint packing with buffer matching | `coding_ideas_stochastic_blueprint_packing.py` | P1 | 2 weeks |

**Minimum viable thesis system**: P0 through P5 (~8--10 weeks).
**Full system with all extensions**: P0 through P10 (~16--20 weeks).

---

## Appendix: File Index

### Semi-Online Buffer Files (`python/semi_online_buffer/`)

| File | Lines | Primary Topic |
|------|-------|---------------|
| `buffer_aware_packing.py` | ~300 | Core buffer data structure, 3 simple selection strategies, TwoBoundedManager |
| `buffer_mcts_policy_coding_ideas.py` | ~400 | Complete MCTS for buffer+2-bounded; BinClosingPolicy; ExperimentConfig |
| `buffer_mcts_selection.py` | ~600 | Alternative MCTS with progressive widening; GreedyBufferSelector; LearnedRolloutPolicy |
| `buffer_packman_coding_ideas.py` | ~300 | Hierarchical PackMan adaptation; DQN item selector; MultiObjectiveReward |
| `buffer_with_lbcp_stability.py` | ~200 | LBCP stability integration; StabilityAwareBufferScorer with 5 weighted criteria |
| `coding_ideas_buffer_stability_integration.py` | ~800 | 9 stability-buffer integration ideas; AdaptiveStabilityPolicy; 16-heuristic hyper-heuristic |
| `coding_ideas_buffer_with_tree_search.py` | ~350 | SmartBufferManager with tree search; LookAheadEvaluator; BufferAwareBinCloser |
| `coding_ideas_lookahead_buffer_with_dual_bin.py` | ~700 | Comprehensive lookahead buffer; 6 item selection strategies; JointItemBinSelector; AdaptiveBufferManager; BufferDiversityAnalyzer |
| `coding_ideas_stochastic_blueprint_packing.py` | ~500 | Blueprint packing from Ayyadevara et al.; BoxDistributionLearner; UprightMatcher3D; DoublingTrickManager |
| `prediction_augmented_buffer_packing_IDEAS.py` | ~500 | Hybrid(lambda) from Angelopoulos et al.; FrequencyTracker; DynamicLambdaController |
| `random_order_sequential_packing.py` | ~400 | Sequential packing from Albers et al.; SequentialBufferManager; GAPBinAssigner |
| `top_buffer_2bounded_coding_ideas.py` | ~2440 | ToP MCTS for 2-bounded; ExtendedToP_MCTS (RECOMMENDED); CrossBinSpatialEnsemble; AdvancedBinClosingStrategy; StabilityVerificationPipeline; BeamSearchPlanner; ComprehensiveMetrics; 3 architecture options |

### Multi-Bin Files (`python/multi_bin/`)

| File | Lines | Primary Topic |
|------|-------|---------------|
| `coding_ideas_dual_bin_replacement_strategies.py` | ~500 | 9 replacement strategies (ReplaceAll through ReplaceByQLearning); MultiBinManager; FullPipelineManager |
| `coding_ideas_two_bounded_space.py` | ~400 | 6 bin selection strategies (enum); BinClosingController with 5 rules; TwoBoundedSpacePipeline |
| `cross_bin_rearrangement_ideas.py` | ~500 | Cross-bin MCTS; BinAllocationScorer; StabilityAwareBinCloser; DualBinPipelineController |
| `two_bounded_bin_packing.py` | ~380 | BinCoordinator (5 strategies); BinLifecycleManager; performance expectations; training options A/B/C |

### Summary Files (`gelezen door claude/summaries/`)

| File | Paper | Key Contribution to This Section |
|------|-------|--------------------------------|
| `Online Bin Packing with Predictions (Summary).md` | Angelopoulos et al. (2023) | Hybrid(lambda) framework; consistency-robustness tradeoff; ProfilePacking; buffer as perfect short-term predictions |
| `Near-optimal Algorithms for Stochastic Online Bin Packing (Summary).md` | Ayyadevara et al. (ICALP 2022) | Blueprint packing; upright matching; doubling trick; (1+eps)-competitive under i.i.d. |
| `Improved Online Algorithms for Knapsack and GAP Random Order (Summary).md` | Albers, Khan & Ladewig (2021) | Sequential approach; large/small decomposition; GAP formulation for multi-bin assignment; delta=1/3, c=0.42291, d=0.64570 |
