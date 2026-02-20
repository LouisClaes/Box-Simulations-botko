# SECTION: Heuristic Placement Strategies & Stability Engine

> **Cluster scope:** All heuristic baselines (160-framework, WallE) and all stability models (full-base, partial-base, CoG polygon, partial-base polygon, LBCP, stacking tree, feasibility mask, three-tier criterion).
>
> **User context:** Semi-online, buffer 5-10 items, k=2 bounded space (2 active pallets), maximize fill rate + stability, Python + PyTorch thesis project.
>
> **Last updated:** 2026-02-18

---

## A. HEURISTIC PLACEMENT STRATEGIES

### A1. The 160-Heuristic Framework (Ali et al. 2025)

#### What It Is

The 160-heuristic framework is a **systematic Cartesian product** of three independent decision rules for online 3D bin packing:

```
Heuristic = (Bin Selection Rule) x (Space Selection Rule) x (Orientation Selection Rule)
          = 4 x 8 x 5 = 160 unique heuristics
```

Each heuristic is denoted by a three-character code `[B][S][O]` -- for example, `A53` means All-bins, Space rule 5, Orientation rule 3.

This framework was developed across two papers by the same research group:

1. **Ali, Ramos, Carravilla, Oliveira (2024)** -- "Heuristics for online three-dimensional packing problems and algorithm selection framework for semi-online with full look-ahead", *Applied Soft Computing*, Vol. 151, Article 111168. This predecessor paper introduced the 160 heuristics **without** stability constraints.

2. **Ali, Ramos, Oliveira (2025)** -- "Static stability versus packing efficiency in online three-dimensional packing problems", *Computers & Operations Research*, Vol. 178, Article 107005. This paper embeds **four distinct stability constraints** into the framework, creating 160 x 4 = 640 total heuristic-constraint combinations, tested on 198 real-world instances.

**Paper summary reference:**
`C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\Static Stability vs Packing Efficiency Online 3D (Summary).md`

#### Python Implementation

**Primary file:**
`C:\Users\Louis\Downloads\stapelalgortime\python\heuristics\coding_ideas_160_heuristic_framework.py`

#### Key Classes and Functions

| Class / Function | Line | Purpose |
|---|---|---|
| `Dimensions` | ~39 | Dataclass for item/bin dimensions (length, width, height) with `volume` and `base_area` properties |
| `Position` | ~56 | 3D position (x, y, z) for the deepest-bottom-left corner |
| `EMS` | ~63 | Empty Maximal Space defined by two corners (x1,y1,z1) to (x2,y2,z2) with `volume`, `dims`, `contains()`, `intersects_item()`, `is_valid()` |
| `PlacedItem` | ~107 | Placed item with position, oriented dims, original dims, orientation index, and `center_of_gravity` property |
| `get_oriented_dims(dims, orientation)` | ~146 | Returns item dimensions in any of the 6 orthogonal orientations |
| `fits_in_ems(item_dims, ems)` | ~164 | Boolean check: does item fit inside the EMS? |
| `BinSelectionRule` (Enum) | ~189 | FIRST_FIT / BEST_FIT / WORST_FIT / ALL_BINS |
| `select_bins_first_fit` | ~196 | Returns the first (oldest) open bin |
| `select_bins_best_fit` | ~203 | Returns the fullest (most utilized) bin |
| `select_bins_worst_fit` | ~210 | Returns the emptiest bin |
| `select_bins_all` | ~217 | Returns all bins for global evaluation |
| `BIN_SELECTOR_MAP` | ~222 | Dict mapping `BinSelectionRule` enum to function |
| `space_rule_1_key` through `space_rule_8_key` | ~244-358 | Eight EMS sorting key functions (see table below) |
| `SPACE_RULE_MAP` | ~361 | Dict mapping rule number (1-8) to key function |
| `orientation_rule_1` through `orientation_rule_5` | ~393-530 | Five orientation selection functions |
| `ORIENTATION_RULE_MAP` | ~533 | Dict mapping rule number (1-5) to function |
| `EMSManager` | ~546 | **Critical data structure.** Manages Empty Maximal Spaces using "difference and elimination" (Lai & Chan, 1997). Methods: `place_item()`, `get_sorted_ems()`, `filter_blocked()`. Internal: `_generate_cuts()` (6 sub-EMSs per intersection), `_eliminate_contained()` |
| `HeuristicEngine` | ~689 | **The master engine.** Configurable with bin_rule, space_rule, orient_rule, and any `StabilityChecker`. Method `pack(items)` implements the 9-step procedure from Appendix A of Ali et al. (2025). |
| `generate_all_heuristic_names()` | ~818 | Returns list of all 160 `[B][S][O]` strings |
| `build_heuristic(name, stability_checker, bin_dims)` | ~827 | Factory: builds a `HeuristicEngine` from its 3-character name |
| `run_full_benchmark(instances, stability_checkers, bin_dims)` | ~874 | Runs all 160 heuristics x all stability constraints on all instances |

#### Which of the 160 Combinations Are Pareto-Optimal

The Pareto-optimal (non-dominated) heuristics vary by stability constraint. The following lists are defined at lines ~842-867 of the implementation file:

**Under Partial-Base Polygon Support (16 non-dominated) -- RECOMMENDED CONSTRAINT:**
```python
PARETO_PARTIAL_BASE_POLYGON = [
    'A12', 'F12', 'B12', 'W53', 'B52', 'F51', 'A63', 'F63',
    'B63', 'A53', 'F53', 'B53', 'A52', 'F52', 'W63', 'W73'
]
```

**Under CoG Polygon Support (17 non-dominated):**
```python
PARETO_COG_POLYGON = [
    'A12', 'F12', 'B12', 'W53', 'A63', 'F63', 'B63', 'W63',
    'W73', 'A53', 'F53', 'B53', 'A52', 'F52', 'A11', 'F11', 'B52'
]
```

**Under Full-Base Support (only 2 non-dominated):**
```python
PARETO_FULL_BASE = ['F52', 'A52']
```

**Under Partial-Base Support (only 1 non-dominated):**
```python
PARETO_PARTIAL_BASE = ['F51']
```

**Key insight:** Polygon-based constraints unlock diversity in the heuristic space (16-17 Pareto-optimal solutions), while base-support constraints collapse the Pareto front to just 1-2 solutions.

**Pre-selected defaults for our k=2 system:**
```python
RECOMMENDED_DEFAULT = 'A53'  # All-bins, DBLF+corner, largest base + max x-occupancy
TOP_EFFICIENCY = ['A12', 'F12', 'B12', 'F53', 'A53']
TOP_STABILITY = ['F52', 'A52', 'F51']
```

For k=2 bounded space, "All bins" (A) checks just 2 bins, so there is zero computational overhead versus First-fit or Best-fit. Always use `A` as the bin selection rule.

#### How to Use This as a Baseline Benchmark

1. **Implement all 160 heuristics** using `generate_all_heuristic_names()` and `build_heuristic()`.
2. **Download the 198 Mendeley instances** referenced in the paper.
3. **Run the benchmark** using `run_full_benchmark()` to reproduce Tables 5 and 8 from the paper. Validate numbers:
   - Full-base: Small=1.006 bins, Medium=2.472, Large=8.055
   - Partial-base polygon: Small=1.001, Medium=1.573, Large=4.247
4. **Compare your DRL agent** against the top 5 heuristics (`A53`, `F53`, `A52`, `F52`, `A12`). Your agent should beat these on at least one metric.
5. **Estimated implementation time:** ~9 days (data structures: 2d, EMS manager: 2d, selection rules: 2d, main loop: 1d, testing: 2d).

#### Integration Points: How Other Strategies Can Use These Heuristics

- **As DRL baselines:** Run `A53` on every test instance; report fill rate and stability percentage alongside DRL results.
- **As rollout policy in MCTS:** Use `build_heuristic('A53')` as the fast simulation policy in MCTS lookahead (Zhao et al. 2023 approach).
- **As buffer item selector:** For each buffer item, run the heuristic's placement logic on both bins, then pick the (item, bin) pair with the best outcome.
- **As stability checker integration:** Pass any `StabilityChecker` instance to `HeuristicEngine`; the engine's step 4 automatically filters by stability.

---

### A2. WallE Scoring Heuristic (Verma et al. 2020)

#### Source Paper

**"A Generalized Reinforcement Learning Algorithm for Online 3D Bin-Packing"**, Verma et al., AAAI 2020.

**Paper summary reference:**
`C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\A Generalized Reinforcement Learning Algorithm for Online 3D Bin-Packing (Summary).md`

#### The Score Function

WallE computes a stability score **S** for every feasible (location, orientation) combination, then selects the placement with the highest score:

```
S = -alpha_1 * G_var + alpha_2 * G_high + alpha_3 * G_flush - alpha_4 * (i + j) - alpha_5 * h_{i,j}
```

| Component | What It Measures | Sign | Default Weight |
|---|---|---|---|
| **G_var** | Sum of absolute height differences between the box's new top and all bordering cells | Penalized (-) | alpha_1 = 0.75 |
| **G_high** | Count of bordering cells higher than the box's new top (nestled into a valley) | Rewarded (+) | alpha_2 = 1.0 |
| **G_flush** | Count of bordering cells at exactly the same height as the box's new top | Rewarded (+) | alpha_3 = 1.0 |
| **(i + j)** | Distance from origin corner; encourages packing toward corner | Penalized (-) | alpha_4 = 0.01 |
| **h_{i,j}** | Base height at placement location; encourages low (floor) placement | Penalized (-) | alpha_5 = 1.0 |

**Design philosophy encapsulated:** Floor building (penalty on height) + First Fit (penalty on distance from origin) + Wall building (G_var/G_high/G_flush reward smooth surfaces).

#### Python File Reference

**Primary file:**
`C:\Users\Louis\Downloads\stapelalgortime\python\heuristics\walle_heuristic_coding_ideas.py`

#### Key Classes and Functions

| Class / Function | Line | Purpose |
|---|---|---|
| `Box` | ~35 | Dataclass: id, length, width, height, weight. Method `rotated()` for z-axis 90-degree rotation. |
| `Container` | ~53 | 2D heightmap representation. `is_feasible(box, i, j)` checks flat-base constraint. `place_box(box, i, j)` updates heightmap. |
| `compute_walle_score(container, box, i, j, alpha1..5)` | ~98 | **Core function.** Computes the S score formula. Iterates over border cells (4-connected neighbors of the box footprint). |
| `compute_extended_walle_score(container, box, i, j, alpha1..7)` | ~199 | **Extended version** with support area fraction (`alpha6 = 2.0`) and CoG penalty (`alpha7 = 0.5`). Returns `(score, is_stable)` tuple. |
| `walle_place_single_bin(container, box)` | ~285 | Find best placement in one container. Scans all (i, j) for both orientations. Returns `(i, j, orientation, score)`. |
| `walle_place_multi_bin(containers, box)` | ~313 | Multi-bin search across all open containers. Returns `(container_id, placement)`. |
| `walle_place_with_buffer(containers, buffer_boxes)` | ~346 | **Buffer extension.** Evaluates ALL (box, container, location, orientation) combinations. Returns globally best `(box_id, container_id, placement)`. |
| `walle_place_corners_only(container, box)` | ~389 | **Optimized version.** Only evaluates corner-aligned locations, reducing from O(L*B*2) to O(num_corners*2) per box. |
| `_find_corner_locations(container, box)` | ~423 | Finds corner-aligned candidate locations from container corners and heightmap transitions. |
| `should_close_bin(container, buffer_boxes, fill_threshold, no_fit_threshold)` | ~480 | Heuristic for bin-closing in k=2 bounded space. |
| `walle_full_pipeline(box_stream, buffer_capacity, k, ...)` | ~533 | **Complete pipeline** for semi-online 3D packing with buffer + 2-bounded space. Handles bin opening/closing, buffer refill, proactive bin closing. |

#### When to Use WallE vs Other Heuristics

| Scenario | Use WallE? | Use 160-Framework? | Rationale |
|---|---|---|---|
| Quick baseline without stability checking | Yes | No | WallE is a single function, ~81.8% fill rate |
| Systematic benchmark reproduction | No | Yes | 160-framework matches published paper results |
| Real-time scoring in DRL training | Yes | No | WallE score can be a reward shaping signal |
| Buffer item selection | Yes (corners-only) | Yes (A53) | Compare both; WallE is faster per-call |
| k=2 bounded space with bin closing | Yes | Not directly | WallE has built-in bin-closing heuristic |

#### Strengths and Weaknesses

**Strengths:**
- Fully deterministic, no training required
- ~81.8% average fill rate (paper result)
- ~10ms per box decision (paper timing)
- Naturally promotes stable placements through G_var/G_high/G_flush
- Built-in buffer and multi-bin support in the coding ideas file
- Corner-only optimization reduces computation significantly

**Weaknesses:**
- Uses a 2D heightmap (flat-base assumption): items must rest on a level surface, no overhangs
- Only 2 orientations (z-axis rotation), not 6
- Does not perform formal stability checking (no convex hull, no CoG verification)
- Full grid scan is O(L*W*2) per item -- slow for large grids (use corners-only variant)
- Fixed alpha weights may not be optimal for all item distributions

---

### A3. Placement Rules Comparison

The 160-framework implements the following placement rules across three dimensions. Below is a cross-reference of which rules work best with which stability constraint, derived from the Pareto analysis (paper Appendix G and Figure 7).

#### Space Selection Rules

| Rule | Name | Key Logic | Best For Efficiency | Best For Stability | Appears in Pareto Front Under |
|---|---|---|---|---|---|
| 1 | DBLF (classic) | min x -> min z -> min y | Medium/Large instances | Average | CoG polygon, Partial-base polygon |
| 2 | Lexicographic Min | min of (x,y,z) sorted | Small instances | Average | -- |
| 3 | Bottom-First | min z -> min of (x,y) | Average | **Best stability** | -- (dominated on efficiency) |
| 4 | Smallest EMS | smallest volume -> earliest | Tight packing | Average | -- |
| 5 | **DBLF + Corner** | min x -> min z -> nearest back-bottom corner | **Consistently best** | Good | Full-base, Partial-base, Partial-base polygon, CoG polygon |
| 6 | Corner-First Large | nearest corner -> largest volume -> earliest | Diverse item sizes | Good | Partial-base polygon, CoG polygon |
| 7 | Corner-First Small | nearest corner -> smallest volume -> earliest | Corner filling | Good | Partial-base polygon, CoG polygon |
| 8 | DFTRC variant | max distance to front-top-right corner | Compact packing | Average | -- |

**Recommendation:** Rule 5 as default. Rule 3 if stability is paramount (at cost of ~5-10% more bins).

#### Orientation Selection Rules

| Rule | EMSs Considered | Logic | Best For |
|---|---|---|---|
| 1 | 1 | Min margin (tightest fit) | Polygon constraints (tight fit = stable) |
| 2 | 1 | Largest base, smallest x-occupancy | Worst for large instances |
| 3 | 1 | **Largest base, greatest x-occupancy** | **Consistently best for efficiency** |
| 4 | n (multi-EMS) | Best fill ratio across n EMSs | Good but slower |
| 5 | n (multi-EMS) | Max distance to FTR corner | Competitive but inconsistent |

**Recommendation:** Rule 3 as default. Rule 1 as alternative for more stable configurations.

#### Bin Selection Rules for k=2

For k=2 bounded space, the bin selection rule distinction is negligible:

| Rule | For k=2 | Recommendation |
|---|---|---|
| First-fit (F) | Checks bin 0, then bin 1 if needed | Simple, slightly biased to first bin |
| Best-fit (B) | Checks the fuller bin first | May leave fresh bin unused |
| Worst-fit (W) | Checks the emptier bin first | Spreads load (good for stability under polygon constraints) |
| **All-bins (A)** | **Checks both bins, picks global best** | **Recommended: zero overhead for k=2** |

---

## B. STABILITY ENGINE

### B1. Stability Model Comparison Table

The following table consolidates ALL stability models implemented across the Python files in this cluster:

| Model | Accuracy (vs Physics) | Stability % | Bins (Large) | Speed/Item | Complexity | Python File | Key Class/Function |
|---|---|---|---|---|---|---|---|
| **Full-Base Support** | 100% | 100.00% | 8.05 | 0.008s | O(n) | `coding_ideas_stability_vs_efficiency.py` | `FullBaseSupport` (line ~196) |
| **Partial-Base Support (80%)** | ~100% | 99.99% | 6.38 | 0.017s | O(n) | `coding_ideas_stability_vs_efficiency.py` | `PartialBaseSupport` (line ~249) |
| **CoG Polygon Support** | ~88% | 88.04% | 3.91 | 0.047s | O(n + k log k) | `coding_ideas_stability_vs_efficiency.py` | `CoGPolygonSupport` (line ~310) |
| **Partial-Base Polygon (50%)** | ~92% | 92.19% | 4.25 | 0.043s | O(n + k log k) | `coding_ideas_stability_vs_efficiency.py` | `PartialBasePolygonSupport` (line ~458) |
| **Three-Tier Criterion** | Conservative | ~95%+ | N/A | <0.001s | O(1) per cell | `feasibility_mask_stability.py` | `PaperStabilityCriterion` (line ~24) |
| **LBCP (Gao et al.)** | ~99% | ~99% | ~3.5-4.0 | 0.05-0.09ms | O(1) amortized | `lbcp_stability_and_rearrangement.py` | `StabilityValidator` (line ~320) |
| **Stacking Tree (Zhao et al.)** | 99.9% | 99.9% | N/A | ~0.5ms | O(N log N) total | `stacking_tree_coding_ideas.py` | `AdaptiveStackingTree` (line ~121) |
| **Feasibility Mask/Predictor** | 99.5% | ~99.5% | N/A | <0.01ms (inference) | O(1) at inference | `feasibility_mask_stability.py` | `FeasibilityMaskGenerator` (line ~186) |
| **Mechanical Equilibrium** | Ground truth | N/A | N/A | O(n^2) | Post-hoc only | `coding_ideas_stability_vs_efficiency.py` | `check_mechanical_equilibrium` (line ~887) |
| **No Constraint** | 50% | 50.08% | ~2.7 | 0.000s | O(0) | -- | -- |

**Notes on the table:**
- "Accuracy" = how closely the model matches full physics simulation
- "Stability %" = percentage of items that remain stable in final packing (from Ali et al. 2025, Table 8)
- "Bins (Large)" = average bins used on large instances from Ali et al. 2025
- "Speed/Item" = processing time per item (from paper Appendix F, per-item per-rule averages for Ali et al. models; paper-reported timing for LBCP and Stacking Tree)
- n = number of placed items in bin; k = number of intersection vertices for convex hull

### B2. LBCP Deep Implementation Guide

#### Source Paper

**"Online 3D Bin Packing with Fast Stability Validation and Stable Rearrangement Planning"**, Gao et al. (2025), arXiv:2507.09123.

**Paper summary reference:**
`C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\Online 3D Bin Packing Fast Stability and Rearrangement (Summary).md`

#### Python File

`C:\Users\Louis\Downloads\stapelalgortime\python\stability\lbcp_stability_and_rearrangement.py`

#### Core Concept: What Is an LBCP?

A **Load-Bearable Convex Polygon (LBCP)** is a convex polygon at a specific height that can support any gravitational force at any point within it. It is located at the top face of a packed item.

The fundamental innovation over classical support polygons: classical support polygons include regions that have geometric contact but may NOT be able to bear load (e.g., the middle of a cantilevered beam). LBCPs only include regions that are **structurally capable** of supporting weight, accounting for the load-bearing capacity all the way down to the floor.

**Key properties (from paper theorems):**
1. **Lemma III.1:** An item on the bin floor has its entire top face as an LBCP.
2. **Theorem III.2:** The support polygon computed from existing LBCPs is itself an LBCP (recursive closure).
3. **Corollary III.2.1:** If CoG falls within the LBCP-based support polygon, the item is stable.
4. **Corollary III.2.2:** Placing a stable item does NOT destabilize items below it (no re-checking needed).

#### Key Classes and Functions

| Class / Function | Line | Purpose |
|---|---|---|
| `Box` | ~57 | Item with dimensions, position, CoG uncertainty `delta_cog`, and `cog_uncertainty_region` property (4 corner points of CoG rectangle) |
| `LBCP` | ~114 | Core dataclass: `polygon` (Nx2 vertices), `height`, `item_id`. Methods: `contains_point()`, `contains_all_points()` |
| `Bin` | ~138 | Bin with items, LBCPs, `feasibility_map` (2D boolean grid), `height_map` (2D float grid), resolution |
| `_point_in_convex_polygon(point, polygon)` | ~198 | Cross-product method for point-in-polygon test |
| `convex_hull_2d(points)` | ~236 | 2D convex hull using `scipy.spatial.ConvexHull` |
| `rectangle_intersection_2d(rect1, rect2)` | ~260 | Axis-aligned rectangle intersection |
| `polygon_rectangle_intersection(polygon, rect)` | ~283 | Polygon-rectangle intersection (uses shapely) |
| **`StabilityValidator`** | ~320 | **Core SSV algorithm (Algorithm 1 from paper).** Method `validate(bin_state, new_item, x, y)` returns `(is_stable, LBCP, support_height)`. Also: `validate_all_candidates()`, `get_stability_mask()`. |
| **`StabilityUpdater`** | ~499 | **SSU algorithm (Algorithm 2).** Methods: `update_after_packing()` (adds LBCP, updates feasibility/height maps), `update_after_unpacking()` (reverse for rearrangement). |
| `OperationType` | ~639 | Enum: UNPACK, PACK, REPACK |
| `RearrangementOperation` | ~646 | Dataclass for a single rearrangement step |
| `MCTSNode` | ~656 | MCTS tree node for rearrangement search |
| **`StableRearrangementPlanner`** | ~670 | **SRP module.** Uses MCTS to find WHICH items to move, then A* to find the optimal ORDER. Key params: `max_nodes=100`, `max_depth=6`, `max_children=3`, `staging_capacity=4`. |
| `BufferStabilitySelector` | ~1223 | Selects the best item from a buffer considering stability. Integrates `StabilityValidator` + `StableRearrangementPlanner`. |

#### The SSV Algorithm (Algorithm 1) -- Step-by-Step

The `StabilityValidator.validate()` method at line ~339 implements:

1. **Extract placement coordinates:** `(x_min, y_min)` to `(x_max, y_max)` of the item footprint.
2. **Compute support height:** `h_s = max(height_map[footprint_region])` -- the height at which the item will rest.
3. **Find contact points:** All grid cells in the footprint where `height_map[x,y] == h_s`.
4. **Filter by feasibility map:** Keep only contact points where `feasibility_map[x,y] == True` (points that belong to an existing LBCP).
5. **Compute support polygon:** Convex hull of the feasible contact points.
6. **Compute CoG uncertainty set:** Four corner points of the item's CoG uncertainty rectangle.
7. **Check containment:** If ALL four CoG corner points lie within the support polygon, the item is stable.
8. **If stable:** Create a new LBCP from the support polygon, return `(True, new_lbcp, h_s)`.

#### The SSU Algorithm (Algorithm 2) -- After Packing

The `StabilityUpdater.update_after_packing()` at line ~508:

1. Add item to bin's item list.
2. Add the newly computed LBCP to the LBCP set.
3. Update the feasibility map: mark all grid cells within the new LBCP polygon as `True`.
4. Update the height map: set footprint cells to `item.top_z`.

For **unpacking** (needed during rearrangement), `update_after_unpacking()` at line ~564:

1. Remove item and its LBCP from the bin.
2. Reset height map in the affected region, then recompute from remaining items.
3. Reset feasibility map in the affected region, then re-mark from remaining LBCPs.

#### Cross-Bin Rearrangement via LBCP

The `StableRearrangementPlanner` (line ~670) enables rearrangement when no stable placement exists:

1. **Phase 1 -- MCTS:** Search which items to unpack. UCB1 selection, random expansion (unpack one accessible item), greedy rollout to test if new item + repacking all succeeds.
2. **Phase 2 -- A*:** Given which items to move, find the optimal unpack/pack/repack order via topological sort on the precedence graph.

For **cross-bin** rearrangement in our k=2 setup, the `RearrangementOperation` dataclass includes `source_bin_id` and `target_bin_id` fields, enabling items to be moved between the two active bins.

**Integration with buffer:** The `BufferStabilitySelector` (line ~1223) wraps the validator and planner, providing a single API: `select_and_place(buffer)` -> best `(item, bin_id, position, operations)`.

#### How to Integrate LBCP with Any Placement Strategy

```python
# 1. Initialize
from lbcp_stability_and_rearrangement import StabilityValidator, StabilityUpdater, Bin, Box

validator = StabilityValidator()
updater = StabilityUpdater()
bin_state = Bin(width=120, depth=80, height=150, resolution=1.0)

# 2. For each candidate placement
is_stable, lbcp, support_h = validator.validate(bin_state, item, x, y)

# 3. If stable, commit the placement
if is_stable:
    item.x, item.y, item.z = x, y, support_h
    updater.update_after_packing(bin_state, item, lbcp)

# 4. For DRL action masking
candidates = [(x, y) for x, y in candidate_positions]
mask = validator.get_stability_mask(bin_state, item, candidates)
# mask is a boolean array; multiply with action probabilities
```

---

### B3. Feasibility Mask System

#### Source Paper

**"Online 3D Bin Packing with Constrained DRL"**, Zhao et al. (2021), AAAI 2021.

**Python file:**
`C:\Users\Louis\Downloads\stapelalgortime\python\stability\feasibility_mask_stability.py`

#### How the Mask Predictor Works

The core principle: stability is NOT a reward signal but a **hard constraint** encoded as a binary feasibility mask. This is proven superior to reward shaping for stability enforcement.

The mask is a binary L x W matrix:
- `M[x][y] = 1` if item can be stably placed at FLB corner (x, y)
- `M[x][y] = 0` otherwise

Three checks per cell: (1) containment, (2) height limit, (3) stability criterion.

#### Key Classes

| Class | Line | Purpose |
|---|---|---|
| `PaperStabilityCriterion` | ~24 | **Three-tier criterion.** Tier 1: >=60% area + 4 corners. Tier 2: >=80% area + 3 corners. Tier 3: >=95% area. Conservative but fast. |
| `CenterOfMassStability` | ~112 | Extended check: paper criterion + CoG offset check with configurable `com_tolerance`. |
| `FeasibilityMaskGenerator` | ~186 | Generates ground-truth masks. Methods: `compute_mask(height_map, l, w, h)` and `compute_mask_vectorized()` for speedup. |
| `StabilityRewardShaping` | ~295 | **NOT recommended.** Included for comparison. Shows why mask approach beats reward shaping. |
| `RoboticPackingStability` | ~351 | Extended checks for real-world robotic packing: min support ratio, max stack height, CoM within support. |
| `LBCPStability` | ~563 | LBCP criterion integrated into the mask framework. Uses `scipy.spatial.ConvexHull` for support region hull. |
| `StackingTreeNode` / `StackingTree` | ~722/~767 | Stacking tree data structure for tracking support relationships. O(N log N) stability checking. `get_cascade_unstable(item_id)` for cascade analysis. |
| `MaskPredictorTrainingPipeline` | ~881 | Pipeline for generating training data and evaluating the neural mask predictor. `generate_training_data(num_samples)` creates (heightmap, item_dims, mask) triples. `evaluate_predictor_accuracy(predictor_fn, test_data)` computes accuracy/precision/recall/F1/FPR/FNR. |
| `StabilityComparison` | ~1021 | Framework for comparing all stability criteria on the same scenarios. `run_comparison(num_scenarios)` returns per-criterion stats and pairwise agreement rates. |

#### Training Pipeline

```python
# Step 1: Choose ground-truth stability criterion
criterion = LBCPStability(com_margin=0.05)  # or PaperStabilityCriterion()

# Step 2: Create mask generator
mask_gen = FeasibilityMaskGenerator(L=10, W=10, H=10, stability_checker=criterion)

# Step 3: Generate training data
pipeline = MaskPredictorTrainingPipeline(L, W, H, criterion)
data = pipeline.generate_training_data(num_samples=50000)
# data['height_maps'], data['item_dims'], data['masks']

# Step 4: Train MLP mask predictor (in DRL training loop)
# The mask predictor is an MLP: CNN_features -> L*W binary mask
# Loss = BCE(predicted_mask, ground_truth_mask)

# Step 5: Evaluate
metrics = pipeline.evaluate_predictor_accuracy(predictor_fn, test_data)
# CRITICAL: false_positive_rate must be < 0.5% (safety concern)
```

#### How to Swap Stability Backends

The mask system's modularity is its greatest strength. To switch backends:

```python
# Switch from three-tier to LBCP:
old_criterion = PaperStabilityCriterion()       # fast but approximate
new_criterion = LBCPStability(com_margin=0.05)  # physics-based

# Create new mask generator with new criterion
mask_gen = FeasibilityMaskGenerator(L, W, H, new_criterion)

# Retrain ONLY the mask predictor MLP (~2 hours)
# The full DRL agent does NOT need retraining from scratch
```

This is essentially **knowledge distillation**: Complex stability check -> Binary mask -> Neural network predictor.

#### Integration with DRL Agents

The feasibility mask integrates into the DRL state as additional channels:

```
State s_n = {H_n, d_n, d_{n+1}...d_{n+k-1}, M_{n,0}, M_{n,1}}
```

Where `M_{n,0}` and `M_{n,1}` are the masks for the two orientations. The mask is element-wise multiplied with the DRL policy's action probabilities to zero out unstable placements. Additionally, an **infeasibility loss** term penalizes any probability mass placed on infeasible actions:

```
E_inf = sum over infeasible cells of P_actor(a_n | s_n)
```

---

### B4. Stacking Tree

#### Source Paper

**"Learning Practically Feasible Policies for Online 3D Bin Packing"**, Zhao et al. (2023), arXiv:2108.13680v3, Science China Information Sciences.

**Paper summary reference:**
`C:\Users\Louis\Downloads\stapelalgortime\gelezen door claude\summaries\Learning Practically Feasible Policies (Summary).md`

#### Python File

`C:\Users\Louis\Downloads\stapelalgortime\python\stability\stacking_tree_coding_ideas.py`

#### O(N log N) Algorithm

Traditional full force analysis for N items requires O(N^2) per placement. The adaptive stacking tree reduces this to **O(N log N) total** across all placements by exploiting the key insight: when placing item n, only a subgraph G_n ("adaptive stacking tree") containing items whose mass distribution changes needs updating. Items NOT in G_n keep their existing distributions.

**Complexity for our use case:**
- 10 buffer items x 2 bins x 2 orientations = 40 checks per decision
- At ~0.5ms each = ~20ms total -- well within real-time

#### Key Classes

| Class | Line | Purpose |
|---|---|---|
| `Item` | ~37 | Item with placement info, orientation handling (`apply_orientation()`), `centroid_xy`, `bottom_rect` |
| `MassFlowEdge` | ~99 | Edge in mass distribution graph: mass flowing from item_above to item_below |
| `StackingTreeNode` | ~108 | Tree node: item_id, parent_ids, children_ids, total_mass_above, group_centroid |
| **`AdaptiveStackingTree`** | ~121 | **Core class.** Manages height map, placed items, mass distribution graph. Key methods below. |
| `DualBinStabilityChecker` | ~615 | **k=2 wrapper.** Manages two `AdaptiveStackingTree` instances. Methods: `check_stability(bin_id, item, x, y, orient)`, `place_item()`, `get_feasibility_mask()`, `close_bin()`, `open_new_bin()`. |

#### Core Methods of `AdaptiveStackingTree`

| Method | Line | Purpose |
|---|---|---|
| `check_stability(item, x, y, orientation)` | ~429 | Main API. Creates temp item, determines z from heightmap, finds supports, checks centroid stability. |
| `place_item(item, x, y, orientation)` | ~487 | Place item if stable. Registers in tree, creates node, computes mass distribution, updates heightmap. |
| `_get_support_items(item)` | ~159 | Find which placed items support the given item. Returns `(item_id, contact_points)` tuples. |
| `_check_centroid_stability(item, support_info)` | ~202 | **Supported Centroid Rule.** Single support: check centroid in contact rectangle. Multiple supports: check centroid in convex hull of all contact points. |
| `_compute_mass_distribution(item, support_info)` | ~280 | Three cases: (1) single support = all mass transfers, (2) two supports = leverage principle, (3) 3+ supports = least squares optimization. Then propagates down tree. |
| `_propagate_mass_update(item_id)` | ~399 | Propagate mass changes down the tree. Stops when change is negligible (< 1e-6). |
| `compute_feasibility_mask(item, orientation)` | ~556 | Full L x W mask computation. For each valid cell, checks stability + height constraint. |
| `get_height_map()` | ~593 | Return current heightmap copy. |
| `get_utilization()` | ~598 | Current volume utilization. |

#### When to Use Stacking Tree vs LBCP

| Criterion | Stacking Tree | LBCP |
|---|---|---|
| Speed per check | ~0.5ms | ~0.05-0.09ms (5-10x faster) |
| Accuracy vs physics | 99.9% | ~99% |
| Mass distribution tracking | Yes (leverage principle) | No (uses CoG uncertainty bound) |
| Requires item masses | Yes (assumes uniform density) | No (only needs delta_cog bound) |
| Rearrangement support | No (would need extension) | Yes (built-in SRP via MCTS + A*) |
| Cascade analysis | Possible via tree traversal | Not directly supported |
| Implementation complexity | Medium | Medium-High (shapely/scipy needed) |
| Best for | Training ground truth, DRL feasibility masks | Real-time placement validation, rearrangement |

**Recommendation:** Use LBCP for runtime stability validation (faster, supports rearrangement). Use Stacking Tree for training ground truth generation (more accurate mass distribution model). If time is limited, LBCP alone is sufficient.

---

## C. RECOMMENDED STABILITY STRATEGY

### C1. Which Stability Model to Use and Why

For the thesis (semi-online, buffer 5-10, k=2, maximize fill rate + stability):

**Primary recommendation: LBCP for runtime validation + Partial-Base Polygon for interpretable benchmarking.**

| Phase | Model | Rationale |
|---|---|---|
| **Benchmarking / Paper reproduction** | Partial-Base Polygon (50%) | Matches Ali et al. 2025 results. Only 8% more bins than CoG polygon but 4pp more stability. 47% fewer bins than full-base. Enables direct comparison with the 160-heuristic Pareto front. |
| **DRL training ground truth** | LBCP or Stacking Tree | More physically accurate than area-based checks. LBCP recommended for speed; Stacking Tree if mass distribution matters. |
| **DRL inference (real-time)** | Neural mask predictor (trained on LBCP masks) | Sub-millisecond. 99.5% accuracy. The mask predictor distills LBCP knowledge into a fast MLP. |
| **Rearrangement planning** | LBCP + SRP | LBCP's SSV/SSU + MCTS rearrangement is purpose-built for this use case. |
| **Post-hoc validation** | Mechanical Equilibrium | Gold-standard physics check. O(n^2) but only run after packing is complete. |

### C2. How They Complement Each Other

The models form a **pipeline** from fast-approximate to slow-accurate:

```
Fast inference:  Neural Mask Predictor (< 0.01ms)
    |
    v  (trained on)
LBCP validation (0.05-0.09ms) -- used during DRL training
    |
    v  (validated against)
Stacking Tree (0.5ms) -- used for training data generation
    |
    v  (validated against)
Mechanical Equilibrium (O(n^2)) -- used for final thesis evaluation
```

**Partial-Base Polygon** sits alongside as the **interpretable benchmark**: it uses simple geometric primitives (convex hull, point-in-polygon, Shoelace area) that can be visualized and explained in the thesis. Use it for the 160-heuristic Pareto analysis figures.

### C3. Implementation Order Recommendation

| Order | What | Time | Dependencies |
|---|---|---|---|
| **Week 1** | Implement `PartialBasePolygonSupport` from `coding_ideas_stability_vs_efficiency.py` (geometric primitives: intersection vertices, convex hull, point-in-polygon, Shoelace area) | 3-4 days | numpy |
| **Week 1-2** | Implement `FullBaseSupport` and `PartialBaseSupport` (simpler, for comparison) | 1 day | numpy |
| **Week 2** | Plug stability checkers into `HeuristicEngine` from the 160-framework. Validate against paper's Table 5 numbers. | 2-3 days | EMS manager |
| **Week 3** | Implement `PaperStabilityCriterion` (three-tier) + `FeasibilityMaskGenerator`. Generate training data. | 2 days | numpy |
| **Week 3-4** | Implement `AdaptiveStackingTree` or `StabilityValidator` (LBCP). Choose based on time. | 3-4 days | scipy, shapely (LBCP) |
| **Week 4-5** | Train DRL agent with feasibility mask from chosen stability model | -- | PyTorch |
| **Week 5-6** | Implement `StableRearrangementPlanner` (LBCP SRP) if rearrangement is in scope | 3-4 days | LBCP core |
| **Week 6** | Run `StabilityComparison` across all criteria. Report in thesis. | 1-2 days | All above |
| **Week 7+** | Run `check_mechanical_equilibrium` for post-hoc validation on all results | 1 day | All above |

### C4. Adaptive Stability Selection (Novel Strategy)

The file `coding_ideas_stability_vs_efficiency.py` (lines ~1068-1110) proposes an **adaptive height-based gradient** that is not in any published paper:

```python
def select_stability_constraint(bin_fill_rate, placement_height, bin_height, stability_checkers):
    height_ratio = placement_height / bin_height
    if height_ratio < 0.30:
        return stability_checkers['full_base']        # Foundation: maximum stability
    elif height_ratio < 0.60:
        return stability_checkers['partial_base_polygon']  # Middle: balanced
    else:
        return stability_checkers['cog_polygon']       # Top: maximize space usage
```

**Rationale:** Bottom items bear the most load and must be maximally stable. Top items bear the least load; permissive placement maximizes remaining space. This is a novel contribution opportunity for the thesis.

---

## D. IDEAS FOR IMPROVEMENT

### D1. Novel Ideas for Combining Heuristics + Stability

1. **Adaptive Heuristic Selection per Layer:** Instead of using a single heuristic (e.g., A53) for all items, switch heuristics based on the current bin state:
   - Bottom layer: Use `A52` (optimized for stability, from `TOP_STABILITY`)
   - Middle layers: Use `A53` (balanced)
   - Top layers: Use `A12` (optimized for efficiency, from `TOP_EFFICIENCY`)
   This mirrors the adaptive stability constraint idea but applied to the heuristic selection.

2. **WallE Score as DRL Reward Shaping:** Use `compute_walle_score()` as an auxiliary reward signal during DRL training. The G_var/G_high/G_flush components naturally promote stable placements without formal stability checking. Combine with the feasibility mask for hard constraint enforcement.

3. **LBCP-Guided Orientation Selection:** Instead of the 5 orientation rules from Ali et al., add a 6th rule: select the orientation that maximizes the LBCP support polygon area. This directly optimizes for stability.

4. **Threshold Sensitivity Study:** The partial-base polygon's 50% area threshold was NOT sensitivity-tested in Ali et al. 2025 (stated explicitly in Section 3.4). Test thresholds from 0.30 to 0.80 using `run_threshold_sensitivity()` from `coding_ideas_stability_vs_efficiency.py` (line ~1183). This is a direct thesis contribution opportunity.

5. **Buffer-Aware Stability Scoring:** When selecting which item from the buffer to pack, compute a composite score:
   ```
   score = w1 * fill_contribution + w2 * stability_score + w3 * future_flexibility
   ```
   Where `future_flexibility` measures how many REMAINING buffer items could still be placed after this choice. This requires running `FeasibilityMaskGenerator` on the projected state.

### D2. What's Missing in Current Implementations

1. **No weight/fragility modeling:** All implementations assume uniform density. Real logistics items have varying weights and fragility. LBCP partially addresses this with `delta_cog` uncertainty, but a full weight model would improve accuracy.

2. **No dynamic stability:** All models check static vertical stability. Transport-induced horizontal forces (acceleration, braking, turning) are not modeled. This matters for pallet shipping.

3. **Blocking constraint is incomplete:** The `EMSManager.filter_blocked()` method at line ~668 of the 160-framework file has only a stub implementation (`pass`). This should be implemented for realistic entrance-constrained packing.

4. **WallE lacks formal stability:** The WallE score promotes stable placements heuristically but does not verify CoG position or support polygon area. Adding `compute_extended_walle_score()` (line ~199) with the support fraction check partially addresses this.

5. **No GPU-accelerated mask computation:** The `FeasibilityMaskGenerator` runs nested Python loops. For grids > 20x20, this is too slow for DRL training. A CUDA kernel or batched PyTorch implementation would provide 10-50x speedup.

6. **LBCP rearrangement planner uses grid search for placement:** The `_find_stable_placement()` method (line ~941 of lbcp file) uses a simple grid search. It should use EMS-based candidate generation from the 160-framework for efficiency.

### D3. Cross-References to DRL and Buffer Strategies

- **DRL state integration:** The feasibility mask (from any stability model) feeds directly into the DRL state as additional channels. See `stacking_tree_coding_ideas.py` line ~556 (`compute_feasibility_mask`) and `feasibility_mask_stability.py` line ~186 (`FeasibilityMaskGenerator`).
- **Buffer strategy:** The `BufferStabilitySelector` in `lbcp_stability_and_rearrangement.py` (line ~1223) provides a complete buffer-aware placement selector. This can serve as the action selection module for the DRL agent's buffer decision.
- **MCTS for BPP-k:** Zhao et al. 2023's MCTS (from the stacking tree paper) uses the trained DRL policy as a rollout policy. The 160-framework heuristic `A53` can substitute as a faster rollout policy during MCTS. See `walle_heuristic_coding_ideas.py` line ~533 (`walle_full_pipeline`) for a complete buffer + k=2 pipeline that could serve as the MCTS rollout.
- **Bin closing decision:** The `should_close_bin()` function in `walle_heuristic_coding_ideas.py` (line ~480) provides a heuristic for when to close a bin in k=2 mode. This can be learned by the DRL agent as an additional action, or used as a fallback when the DRL agent outputs low confidence.

---

## APPENDIX: Quick Reference -- All File Paths

| File | Contents |
|---|---|
| `C:\Users\Louis\Downloads\stapelalgortime\python\heuristics\coding_ideas_160_heuristic_framework.py` | 160-heuristic framework: data structures, 4 bin rules, 8 space rules, 5 orientation rules, EMSManager, HeuristicEngine, Pareto lists, benchmark runner |
| `C:\Users\Louis\Downloads\stapelalgortime\python\heuristics\walle_heuristic_coding_ideas.py` | WallE score function, extended score, single/multi-bin placement, buffer support, corner-only optimization, bin closing heuristic, full pipeline |
| `C:\Users\Louis\Downloads\stapelalgortime\python\stability\coding_ideas_stability_vs_efficiency.py` | 4 stability checkers (FullBase, PartialBase, CoGPolygon, PartialBasePolygon), support polygon computation (intersection vertices, convex hull, point-in-polygon, Shoelace), mechanical equilibrium, Pareto analysis, adaptive stability selector, sensitivity framework, visualization data |
| `C:\Users\Louis\Downloads\stapelalgortime\python\stability\feasibility_mask_stability.py` | Three-tier criterion, CoM stability, FeasibilityMaskGenerator, reward shaping comparison, robotic stability, LBCP stability (heightmap-based), StackingTree, MaskPredictorTrainingPipeline, StabilityComparison framework |
| `C:\Users\Louis\Downloads\stapelalgortime\python\stability\lbcp_stability_and_rearrangement.py` | LBCP data structure, geometric utilities, StabilityValidator (SSV), StabilityUpdater (SSU), StableRearrangementPlanner (MCTS + A*), BufferStabilitySelector, cross-bin rearrangement support |
| `C:\Users\Louis\Downloads\stapelalgortime\python\stability\stacking_tree_coding_ideas.py` | AdaptiveStackingTree, mass distribution (leverage principle, least squares), DualBinStabilityChecker for k=2, feasibility mask computation |
