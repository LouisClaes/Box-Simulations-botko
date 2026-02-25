# PRD: 3D Bin Packing Strategy Overhauls - The 60+ Iteration Marathon

## Overview & Architecture Constraints
Analysis of the Phase 1 strategy benchmarks reveals 5 strategies that are severely underperforming. They must be made mathematically and visually perfect (placement rates >85% and closed fill >65%).

**CRITICAL IMMUTABILITY CONSTRAINT**: 
The agent MUST NEVER modify the core simulator engine or validation scripts (`simulator/`, `run_overnight_botko.py`, etc.). Break no simulator rules.

**THE TRUE CLOSURE LOGIC (THE 8-BOX DEATH SPIRAL)**: 
Pallets close automatically when a strategy gets "stuck".
1. **Lookahead Trigger**: The current 4 boxes + next 4 boxes (8 total) all fail to fit.
2. **Max Rejects Limit**: 8 boxes are rejected cumulatively without a successful placement.
*If a strategy has 0.0% closed fill, it means its logic is so strict it immediately triggers these limits. You must reprogram the heuristics (margins, support ratios, orientations) to guarantee boxes fit, keeping the reject counter at 0.*

**MANDATORY HYPER-SEQUENTIAL THINKING**:
For EVERY SINGLE TASK, you MUST use the `sequentialthinking` tool extensively and deeply. You are expected to tear apart complex multidimensional arrays, routing logic, and overlap matrices step-by-step in your mind *before* writing any code. If you fail to use deep sequential thinking, your algorithmic patches will be shallow and ultimately fail the simulation. Think 10+ steps ahead.

---

## Pre-Requisite Task 0: Architecture Familiarity
Understand the rigid environment and the exact physics of the dual-bin Botko simulator.
- [ ] Read `simulator/README.md` entirely.
- [ ] Read `CLOSE_LOGIC_EXPLAINED.md` to internalize the 8-box reject limits.
- [ ] Read `run_overnight_botko.py` to understand the `botko_config` (2 bins, max 10 consecutive rejects, buffer 8, pick 4).
- [ ] **Acceptance Criteria**: Log your understanding of the architecture to `progress.txt` using a deep sequential thought process.

---

# STRATEGY 1: 2-Bounded Best Fit (Currently 0.0% Closed Fill)

## Task 1: 2-Bounded Best Fit - Deep Diagnostics
- [ ] **Execution**: Run `python -m pytest tests/test_strategies.py -k "test_two_bounded_best_fit"`
- [ ] **Visualization**: Run `python generate_strategy_gifs.py --strategy two_bounded_best_fit --smoke-test`
- [ ] **Deep Sequential Analysis**: Use the `sequentialthinking` tool to meticulously and deeply trace `decide_placement` and `_best_in_bin` located in `strategies/two_bounded_best_fit/strategy.py`.
- [ ] **Acceptance Criteria**: In `progress.txt`, state the exact mathematical bounds or logical routing flaw causing the strategy to reject 8 boxes immediately based on your sequential trace.

## Task 2: 2-Bounded Best Fit - Implementation Marathon
- [ ] **Sequential Implementation Design**: Use deep sequential thinking to plan exactly how you will alter the multi-bin routing and `support_ratio` thresholds.
- [ ] **Heuristic Relaxation**: Reduce outwardly strict thresholds or contact base ratios that force early rejections.
- [ ] **Fallback Logic**: Implement intelligent fallback placements to dodge the 8-reject limit.
- [ ] **Acceptance Criteria**: Code is successfully written and syntax-error free.

## Task 3: 2-Bounded Best Fit - Rigid Validation
- [ ] **Unit Tests**: Pass all native Pytest checks.
- [ ] **Visual Geometry Review**: Regenerate the GIF and verify dense packing geometry with zero massive gaps.
- [ ] **Integration Test**: Run `python run_overnight_botko.py --smoke-test` and verify it survives the production setup.
- [ ] **Documentation Update**: Extensively mathematically document your sequential process and changes in `strategies/two_bounded_best_fit/README.md`.
- [ ] **Acceptance Criteria**: Placement rate > 85%, Closed Fill > 65%.

---

# STRATEGY 2: Tsang Multi-Bin (Currently 0.0% Closed Fill)

## Task 4: Tsang Multi-Bin - Deep Diagnostics
- [ ] **Execution**: Run `python -m pytest tests/test_strategies.py -k "test_tsang_multibin"`
- [ ] **Visualization**: Check visual output to observe the cascading rejection failure.
- [ ] **Deep Sequential Analysis**: Use deep sequential thinking to trace routing and `_compute_suitability` math. Why are 50%+ of all boxes rejected?
- [ ] **Acceptance Criteria**: Log the core sequential failure point in `progress.txt`.

## Task 5: Tsang Multi-Bin - Implementation Marathon
- [ ] **Sequential Implementation Design**: Deeply plan out changes to the dual-bin heuristic weights.
- [ ] **Allocation Overhaul**: Tweak the heuristic weights to fiercely encourage dense packing over specific orientations.
- [ ] **Constraint Elimination**: Fix cascading box rejection logic by lowering score thresholds.
- [ ] **Acceptance Criteria**: Code changes are successfully implemented and syntax-error free.

## Task 6: Tsang Multi-Bin - Rigid Validation
- [ ] **Unit Tests**: Pass unit tests natively.
- [ ] **Visual Review**: Confirm visually robust closures via GIF.
- [ ] **Integration Test**: Run `python run_overnight_botko.py --smoke-test`.
- [ ] **Documentation Update**: Log sequential routing fixes in `strategies/tsang_multibin/README.md`.
- [ ] **Acceptance Criteria**: Placement rate > 85%, Closed Fill > 65%.

---

# STRATEGY 3: PCT Expansion (Currently 44.9% Placement)

## Task 7: PCT Expansion - Deep Diagnostics
- [ ] **Execution**: Run `python -m pytest tests/test_strategies.py -k "pct_expansion"`
- [ ] **Visualization**: Generate baseline GIF to observe the expansion collapse.
- [ ] **Deep Sequential Analysis**: Unpack the spatial expansion sequence constraints mathematically causing 8-box rejections when the bin is mostly empty.
- [ ] **Acceptance Criteria**: Log geometric failure causes to `progress.txt`.

## Task 8: PCT Expansion - Implementation Marathon
- [ ] **Sequential Implementation Design**: Think deeply over how to loosen extreme geometric boundaries while maintaining physical stability limits.
- [ ] **Expansion Logic**: Prioritize logical placement following existing structures over perfect geometry to keep the reject counter at 0.
- [ ] **Acceptance Criteria**: Modifications complete and syntax-error free.

## Task 9: PCT Expansion - Rigid Validation
- [ ] **Unit Tests**: Pass unit tests.
- [ ] **Visual Review**: Confirm continuous expansion geometry via GIF.
- [ ] **Integration Test**: Run `python run_overnight_botko.py --smoke-test`.
- [ ] **Documentation Update**: Explain the geometric relaxation locally.
- [ ] **Acceptance Criteria**: Placement rate > 85%, Closed Fill > 65%.

---

# STRATEGY 4: Skyline (Currently 55.8% Fill, Towering Bug)

## Task 10: Skyline - Deep Diagnostics
- [ ] **Execution**: Run `python -m pytest tests/test_strategies.py -k "skyline"`
- [ ] **Visualization**: Generate baseline GIF to verify the "towering in Z" pillar bug.
- [ ] **Deep Sequential Analysis**: Map out the `np.min(heightmap, axis=1)` projection logic, `valley_depth`, and peak penalties line by line, explicitly utilizing deep sequential thought to visualize the numpy array flattening.
- [ ] **Acceptance Criteria**: Explicitly log the sequential axis projection error trace in `progress.txt`.

## Task 11: Skyline - Implementation Marathon
- [ ] **Sequential Implementation Design**: Verify mathematically via thought whether swapping to `axis=0` fixes the projection layer dimensions based on the config.
- [ ] **Horizontal Layer Fix**: Swap the projection axis and fix the skyline detection array flattening.
- [ ] **Penalty Tuning**: Aggressively penalize vertical outliers so the surface remains flat and packable.
- [ ] **Acceptance Criteria**: Projection logic patched and syntax-error free.

## Task 12: Skyline - Rigid Validation
- [ ] **Visual Review**: The new GIF MUST show perfectly horizontal layer formation. Not a single thin towering pillar is allowed to exist.
- [ ] **Unit Tests**: Pass native tests.
- [ ] **Integration Test**: Run `python run_overnight_botko.py --smoke-test`.
- [ ] **Documentation Update**: Graphically document the mathematical thought process behind the fix in the local README.
- [ ] **Acceptance Criteria**: Placement rate > 80%, Closed Fill > 65%.

---

# STRATEGY 5: LBCP Stability (Currently 57.8% Fill)

## Task 13: LBCP Stability - Deep Diagnostics
- [ ] **Execution**: Run `python -m pytest tests/test_strategies.py -k "lbcp"`
- [ ] **Visualization**: Generate GIF. Confirm the top half of the bin remains dead-space.
- [ ] **Deep Sequential Analysis**: Tear down the center-of-gravity checks mathematically blocking high stacking. Use sequential thinking to simulate the forces limiting the vertical volume.
- [ ] **Acceptance Criteria**: Log the specific stability blocker array or limit in `progress.txt`.

## Task 14: LBCP Stability - Implementation Marathon
- [ ] **Sequential Implementation Design**: Brainstorm deeply how to implement a wall-bracing or adjacent-support matrix.
- [ ] **Wall-Bracing**: Loosen pessimistic lateral stability checks. Boxes supported by adjacent walls or neighbors should be legal.
- [ ] **Acceptance Criteria**: Safety algorithm patches committed.

## Task 15: LBCP Stability - Rigid Validation
- [ ] **Visual Review**: The GIF must show the upper half of the bin being firmly and safely populated.
- [ ] **Unit Tests**: Pass native tests.
- [ ] **Integration Test**: Run `python run_overnight_botko.py --smoke-test`.
- [ ] **Documentation Update**: Update local README with the massive new stability formulas drafted during deep sequential thought.
- [ ] **Acceptance Criteria**: Placement rate > 85%, Closed Fill > 65%.

---

# FINAL VALIDATION

## Task 16: Global Benchmark & Final Run
- [ ] **Overnight Simulation**: Run a test to validate the full: `python run_overnight_botko.py` would function perfectly weeks on end.
- [ ] **Metrics Validation**: Parse the generated `results.json` mathematically. Verify visually and statistically that the 5 strategies have been fundamentally cured.
- [ ] **Acceptance Criteria**: Append `[RALPH LOOP COMPLETE]` to `progress.txt`. Do NOT modify `STRATEGIES_README.md` or `README.md`.
