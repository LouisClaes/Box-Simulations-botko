"""
WallE + PackMan Hybrid for 2-Bounded Semi-Online 3D Bin Packing
=================================================================

Source: Verma et al. (2020), "A Generalized RL Algorithm for Online 3D Bin-Packing"
Context: Thesis adaptation for semi-online (buffer 5-10), 2-bounded space, fill+stability

This file describes a HYBRID approach that combines:
1. WallE heuristic for fast, stable baseline decisions
2. PackMan DQN for learned strategic decisions
3. Hyper-heuristic selector that chooses between them per step

This addresses RESEARCH GAP 3 from the overview knowledge base:
"No selective hyper-heuristic has been applied to 3D-PPs"

RATIONALE FOR HYBRID APPROACH
------------------------------
- WallE is fast (10ms/box) but uses fixed weights, cannot adapt
- PackMan is slower (34ms/box) but learns from experience
- Different packing situations favor different strategies:
  * Early packing (bin mostly empty): WallE's floor-building tendency is good
  * Mid packing (bin partially full): PackMan's learned strategy excels
  * Late packing (bin nearly full): gap-filling needs precise scoring
- A hybrid can get the best of both worlds

ESTIMATED COMPLEXITY
---------------------
- Implementation: 2-3 weeks (building on WallE and PackMan implementations)
- Training: 4-8 hours (only the selector and DQN need training)
- This is the RECOMMENDED first implementation for the thesis because it
  provides a strong baseline (WallE) with learned improvements (PackMan)
  and addresses a research gap (selective hyper-heuristic).
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum


# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

class PackingStrategy(Enum):
    """Available low-level packing strategies."""
    WALLE = "walle"               # WallE stability score heuristic
    PACKMAN_DQN = "packman"       # PackMan DQN-based selection
    FLOOR_BUILD = "floor"         # Floor building (lowest height preference)
    COLUMN_BUILD = "column"       # Column building (highest height preference)
    FIRST_FIT = "first_fit"       # First feasible location
    BEST_FIT_VOLUME = "best_vol"  # Best volumetric fit (minimize wasted space)


# ============================================================================
# STRATEGY FEATURES (for hyper-heuristic decision)
# ============================================================================

def compute_state_features(containers, buffer_boxes, step_number, total_steps_estimate):
    """
    Compute features that characterize the current packing state.
    These features guide the hyper-heuristic selector.

    Features:
    ---------
    1. Packing phase: step_number / total_steps_estimate (0 to 1)
       Early phase -> prefer WallE (stable foundation)
       Late phase -> prefer best-fit (fill gaps)

    2. Surface roughness of active bins (0 to 1)
       High roughness -> prefer Floor Building (smooth out)
       Low roughness -> prefer Column Building (extend towers)

    3. Fill fraction of active bins (0 to 1)
       High fill -> prefer gap-filling strategies
       Low fill -> prefer space-efficient strategies

    4. Buffer diversity (std of box volumes / mean)
       High diversity -> prefer DQN (needs learned judgment)
       Low diversity -> WallE is sufficient

    5. Height variance across bins
       If one bin is much fuller -> prefer balancing strategies

    6. Average box size relative to remaining space
       Large relative size -> careful placement needed (DQN)
       Small relative size -> WallE is fast enough

    Returns: np.ndarray of shape (feature_count,)
    """
    features = []

    # 1. Packing phase
    phase = step_number / max(total_steps_estimate, 1)
    features.append(phase)

    # 2. Average surface roughness
    roughness_values = []
    for c in containers:
        if c.is_open and len(c.packed_boxes) > 0:
            hmap = c.heightmap
            diff_i = np.abs(np.diff(hmap, axis=0))
            diff_j = np.abs(np.diff(hmap, axis=1))
            r = (np.mean(diff_i) + np.mean(diff_j)) / 2
            roughness_values.append(r / max(np.max(hmap), 1))
    features.append(np.mean(roughness_values) if roughness_values else 0.0)

    # 3. Average fill fraction
    fills = [c.fill_fraction for c in containers if c.is_open]
    features.append(np.mean(fills) if fills else 0.0)

    # 4. Buffer diversity (coefficient of variation of volumes)
    if buffer_boxes:
        volumes = [b.volume for b in buffer_boxes]
        mean_vol = np.mean(volumes)
        std_vol = np.std(volumes)
        features.append(std_vol / max(mean_vol, 1))
    else:
        features.append(0.0)

    # 5. Fill imbalance between bins
    if len(fills) >= 2:
        features.append(abs(fills[0] - fills[1]))
    else:
        features.append(0.0)

    # 6. Average box size relative to remaining capacity
    if buffer_boxes and containers:
        avg_box_vol = np.mean([b.volume for b in buffer_boxes])
        avg_remaining = np.mean([
            (c.length * c.width * c.max_height) * (1 - c.fill_fraction)
            for c in containers if c.is_open
        ])
        features.append(avg_box_vol / max(avg_remaining, 1))
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


# ============================================================================
# HYPER-HEURISTIC SELECTOR
# ============================================================================

class HyperHeuristicSelector:
    """
    Selective hyper-heuristic that chooses the best packing strategy
    based on the current state features.

    THREE IMPLEMENTATION OPTIONS:
    A. Rule-based (no learning): fixed rules mapping features to strategies
    B. Q-learning based: learn Q(state_features, strategy) -> expected quality
    C. Neural network: deep Q-network over strategies

    We implement all three for comparison.
    """

    def __init__(
        self,
        strategies: List[PackingStrategy],
        feature_count: int = 6,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1
    ):
        self.strategies = strategies
        self.n_strategies = len(strategies)
        self.feature_count = feature_count
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # For Q-learning: discretize features into bins
        self.n_bins = 5  # discretization bins per feature
        self.q_table = {}  # maps discretized state -> array of Q-values per strategy

        # Performance tracking
        self.strategy_usage = {s: 0 for s in strategies}
        self.strategy_rewards = {s: [] for s in strategies}

    # ---- OPTION A: Rule-based selector ----

    def select_rule_based(self, features: np.ndarray) -> PackingStrategy:
        """
        Rule-based strategy selection.

        Rules (designed from domain knowledge):
        - Early phase (< 30% through boxes): WallE (build stable foundation)
        - High roughness (> 0.3): Floor Building (smooth surfaces)
        - High fill (> 70%) + high diversity: PackMan DQN (learned gap-filling)
        - High fill (> 70%) + low diversity: Best Fit Volume (systematic)
        - Default: WallE (best overall heuristic from paper)
        """
        phase = features[0]
        roughness = features[1]
        fill = features[2]
        diversity = features[3]

        if phase < 0.3:
            return PackingStrategy.WALLE
        elif roughness > 0.3:
            return PackingStrategy.FLOOR_BUILD
        elif fill > 0.7 and diversity > 0.5:
            return PackingStrategy.PACKMAN_DQN
        elif fill > 0.7:
            return PackingStrategy.BEST_FIT_VOLUME
        else:
            return PackingStrategy.WALLE

    # ---- OPTION B: Tabular Q-learning selector ----

    def _discretize(self, features: np.ndarray) -> tuple:
        """Discretize continuous features into bin indices."""
        clipped = np.clip(features, 0, 1)
        bins = (clipped * (self.n_bins - 1)).astype(int)
        return tuple(bins)

    def select_q_learning(self, features: np.ndarray) -> PackingStrategy:
        """Select strategy using epsilon-greedy Q-learning."""
        state = self._discretize(features)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_strategies)

        if np.random.random() < self.epsilon:
            idx = np.random.randint(self.n_strategies)
        else:
            idx = np.argmax(self.q_table[state])

        strategy = self.strategies[idx]
        self.strategy_usage[strategy] += 1
        return strategy

    def update_q_learning(
        self,
        features: np.ndarray,
        strategy: PackingStrategy,
        reward: float,
        next_features: np.ndarray
    ):
        """Update Q-table after observing reward."""
        state = self._discretize(features)
        next_state = self._discretize(next_features)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_strategies)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_strategies)

        strategy_idx = self.strategies.index(strategy)
        current_q = self.q_table[state][strategy_idx]
        max_next_q = np.max(self.q_table[next_state])

        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][strategy_idx] = new_q

        self.strategy_rewards[strategy].append(reward)

    # ---- OPTION C: DQN selector (sketch) ----

    def build_dqn_selector(self):
        """
        DQN for strategy selection.

        PSEUDOCODE (PyTorch):

        class StrategyDQN(nn.Module):
            def __init__(self, feature_count=6, n_strategies=6):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(feature_count, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, n_strategies)
                )

            def forward(self, features):
                return self.net(features)

        Very small network since the strategy space is small (6 options).
        Could train alongside PackMan DQN with shared experience.
        """
        pass

    def get_report(self) -> str:
        """Generate usage and performance report."""
        lines = ["Strategy Usage Report", "=" * 40]
        total = sum(self.strategy_usage.values())
        for s in self.strategies:
            count = self.strategy_usage[s]
            pct = count / max(total, 1) * 100
            rewards = self.strategy_rewards[s]
            avg_r = np.mean(rewards) if rewards else 0
            lines.append(f"  {s.value:15s}: {count:5d} ({pct:.1f}%), avg_reward={avg_r:.4f}")
        return "\n".join(lines)


# ============================================================================
# HYBRID AGENT: COMBINING WALLE + PACKMAN + HYPER-HEURISTIC
# ============================================================================

class HybridAgent:
    """
    The main hybrid agent combining:
    - WallE heuristic (fast, stable)
    - PackMan DQN (learned, strategic)
    - Additional heuristics (floor/column building, best fit)
    - Hyper-heuristic selector (chooses strategy per step)

    This is the RECOMMENDED implementation for the thesis.
    """

    def __init__(
        self,
        container_dims: Tuple[int, int, int] = (45, 80, 50),
        buffer_capacity: int = 10,
        k: int = 2,
        selector_mode: str = "q_learning"  # "rule_based", "q_learning", "dqn"
    ):
        self.L, self.B, self.H = container_dims
        self.buffer_capacity = buffer_capacity
        self.k = k

        self.strategies = [
            PackingStrategy.WALLE,
            PackingStrategy.PACKMAN_DQN,
            PackingStrategy.FLOOR_BUILD,
            PackingStrategy.FIRST_FIT,
            PackingStrategy.BEST_FIT_VOLUME,
        ]

        self.selector = HyperHeuristicSelector(
            strategies=self.strategies,
            feature_count=6
        )
        self.selector_mode = selector_mode

        # These would be actual implementations
        # self.walle = WallEHeuristic(...)
        # self.packman = PackManDQN(...)
        # self.floor_builder = FloorBuildingHeuristic(...)
        # etc.

    def select_and_place(
        self,
        containers: list,
        buffer_boxes: list,
        step: int,
        total_steps_est: int
    ) -> Optional[Dict]:
        """
        Main decision function.

        1. Compute state features
        2. Select strategy via hyper-heuristic
        3. Execute selected strategy
        4. Return placement decision

        Returns dict with keys: box_id, bin_id, i, j, orientation, strategy_used
        """
        # Step 1: State features
        features = compute_state_features(
            containers, buffer_boxes, step, total_steps_est
        )

        # Step 2: Select strategy
        if self.selector_mode == "rule_based":
            strategy = self.selector.select_rule_based(features)
        elif self.selector_mode == "q_learning":
            strategy = self.selector.select_q_learning(features)
        else:
            strategy = self.selector.select_q_learning(features)  # fallback

        # Step 3: Execute strategy
        result = self._execute_strategy(strategy, containers, buffer_boxes)

        if result is not None:
            result['strategy_used'] = strategy.value
            result['features'] = features

        return result

    def _execute_strategy(
        self,
        strategy: PackingStrategy,
        containers: list,
        buffer_boxes: list
    ) -> Optional[Dict]:
        """
        Execute the selected packing strategy.

        Each strategy returns the best (box, bin, location, orientation)
        according to its own criteria.
        """
        if strategy == PackingStrategy.WALLE:
            return self._execute_walle(containers, buffer_boxes)
        elif strategy == PackingStrategy.PACKMAN_DQN:
            return self._execute_packman(containers, buffer_boxes)
        elif strategy == PackingStrategy.FLOOR_BUILD:
            return self._execute_floor_build(containers, buffer_boxes)
        elif strategy == PackingStrategy.FIRST_FIT:
            return self._execute_first_fit(containers, buffer_boxes)
        elif strategy == PackingStrategy.BEST_FIT_VOLUME:
            return self._execute_best_fit(containers, buffer_boxes)
        return None

    def _execute_walle(self, containers, buffer_boxes) -> Optional[Dict]:
        """
        Execute WallE: evaluate stability score for all (box, bin, location, orientation)
        and return the global best.

        Implementation: call walle_place_with_buffer() from
        heuristics/walle_heuristic_coding_ideas.py
        """
        # Placeholder
        return None

    def _execute_packman(self, containers, buffer_boxes) -> Optional[Dict]:
        """
        Execute PackMan DQN: generate corner-aligned candidates,
        evaluate with DQN, return best.

        Implementation: call selective_search_with_buffer() then DQN from
        deep_rl/packman_dqn_coding_ideas.py
        """
        # Placeholder
        return None

    def _execute_floor_build(self, containers, buffer_boxes) -> Optional[Dict]:
        """
        Execute Floor Building: prefer lowest feasible height.
        Select box from buffer that creates the smoothest floor.
        """
        # Placeholder
        return None

    def _execute_first_fit(self, containers, buffer_boxes) -> Optional[Dict]:
        """
        Execute First Fit: scan row-by-row, place first box that fits.
        Try all buffer items at each location.
        """
        # Placeholder
        return None

    def _execute_best_fit(self, containers, buffer_boxes) -> Optional[Dict]:
        """
        Execute Best Fit by Volume: find the placement that leaves
        the least wasted space (tightest fit).
        """
        # Placeholder
        return None

    def update_after_step(self, features, strategy, next_features, reward):
        """Update the hyper-heuristic selector after observing outcome."""
        if self.selector_mode == "q_learning":
            self.selector.update_q_learning(
                features, strategy, reward, next_features
            )


# ============================================================================
# STEP-LEVEL REWARD FOR HYPER-HEURISTIC
# ============================================================================

def compute_step_reward_for_selector(
    container_before,
    container_after,
    box,
    i: int,
    j: int,
    stability_score: float
) -> float:
    """
    Compute a per-step reward for the hyper-heuristic selector.

    Unlike PackMan (which uses retroactive terminal rewards), the selector
    needs per-step feedback to learn which strategy works best in each situation.

    Reward components:
    1. Volume efficiency: how much of the "claimed space" is actually filled
       (box volume / bounding box volume in the affected region)
    2. Surface quality change: roughness after - roughness before
    3. Stability score of the placement
    4. Fill fraction improvement
    """
    # Volume efficiency
    vol_eff = box.volume / (box.length * box.width * box.height)  # always 1 for cuboids

    # Surface roughness change (negative is better = smoother)
    rough_before = _compute_roughness(container_before.heightmap)
    rough_after = _compute_roughness(container_after.heightmap)
    roughness_change = rough_before - rough_after  # positive if got smoother

    # Fill fraction improvement
    fill_improvement = container_after.fill_fraction - container_before.fill_fraction

    # Combined step reward
    reward = (0.4 * fill_improvement
              + 0.3 * stability_score
              + 0.2 * roughness_change
              + 0.1 * vol_eff)

    return reward


def _compute_roughness(hmap: np.ndarray) -> float:
    """Compute surface roughness of a heightmap."""
    if np.max(hmap) == 0:
        return 0.0
    diff_i = np.abs(np.diff(hmap, axis=0))
    diff_j = np.abs(np.diff(hmap, axis=1))
    return (np.mean(diff_i) + np.mean(diff_j)) / (2 * np.max(hmap))


# ============================================================================
# EXPERIMENTAL COMPARISON FRAMEWORK
# ============================================================================

"""
THESIS EXPERIMENT PLAN
=======================

Experiment 1: Baseline comparison
----------------------------------
Compare all individual strategies (WallE, PackMan, Floor Build, Column Build,
First Fit) on the same test instances.
Metrics: fill rate, competitive ratio, stability score, time per decision.

Experiment 2: Hybrid vs. best individual
-----------------------------------------
Compare the hybrid agent (with each selector mode) against the best individual
strategy from Experiment 1.
Show that the hybrid adapts its strategy selection to the packing phase.

Experiment 3: Buffer size ablation
------------------------------------
Test buffer sizes: 1 (pure online), 3, 5, 7, 10, 15, 20
Show diminishing returns beyond buffer size 10.

Experiment 4: Bounded space ablation
--------------------------------------
Test k values: 1, 2, 3, 5, unlimited
Show trade-off between k and fill rate.
k=2 is our target, but understanding k=1 and k=3 contextualizes results.

Experiment 5: Stability weight ablation
-----------------------------------------
Test stability weights: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
Show Pareto front of fill rate vs. stability.

Experiment 6: Transfer learning
---------------------------------
Train on distribution A, test on distribution B.
Show generalization (or lack thereof) of the DQN component.

Experiment 7: Real-world box distribution
-------------------------------------------
Test with actual box dimension data from a warehouse/logistics partner.
Compare against theoretical bounds and synthetic results.
"""


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("WallE + PackMan Hybrid Agent - Coding Ideas")
    print("=" * 50)
    print()
    print("This hybrid approach addresses Research Gap 3 from the overview:")
    print("'No selective hyper-heuristic has been applied to 3D-PPs'")
    print()
    print("Components:")
    print("  1. WallE heuristic (from Verma et al. 2020)")
    print("  2. PackMan DQN (from Verma et al. 2020)")
    print("  3. Floor/Column Building baselines")
    print("  4. Q-learning hyper-heuristic selector")
    print()
    print("The selector learns WHEN to use WHICH strategy based on:")
    print("  - Packing phase (early/mid/late)")
    print("  - Surface roughness")
    print("  - Fill fraction")
    print("  - Buffer diversity")
    print("  - Fill imbalance between bins")
    print()
    print("This is the RECOMMENDED first implementation for the thesis.")
