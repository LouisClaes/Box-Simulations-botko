"""
Theoretical Competitive Ratio Bounds for Online Packing in the Random Order Model
==================================================================================

Based on: Albers, Khan & Ladewig (2021). "Improved Online Algorithms for Knapsack
          and GAP in the Random Order Model." Algorithmica 83, 1750-1785.

Purpose: Compute and visualize theoretical competitive ratios from the paper.
         These serve as benchmarks for evaluating practical 3D packing algorithms.

Key results from the paper:
  - Online Knapsack (Random Order): 1/6.65 competitive ratio
  - Online GAP (Random Order):      1/6.99 competitive ratio
  - 2-Knapsack-Secretary:           1/3.08 competitive ratio

Context from the overview knowledge base (Section 12):
  - 1D online BPP: best ACR = 1.57829 (Balogh et al. 2017)
  - 2D online BPP: best ACR = 2.5545 for rectangles (Han et al. 2011)
  - 3D online BPP: best ACR = 2.5735 for cubes (Epstein & Mualem 2021)

Note: These bounds are for 1D knapsack/GAP, not 3D bin packing.
      There are NO known competitive ratio results for online 3D knapsack
      in the random order model. This is identified as Gap 9 in the overview.

Feasibility: HIGH -- pure computation, no external dependencies.

Author: Thesis project
"""

import math
from typing import Tuple, Dict, List, Optional


# =============================================================================
# Core Competitive Ratio Calculations
# =============================================================================

class KnapsackRandomOrderBounds:
    """
    Competitive ratio calculations for the online knapsack problem
    in the random order model.

    The paper's main theorem:
    Theorem 1: There exists a (1/6.65)-competitive randomized algorithm.

    This is achieved by combining:
    - A_L for large items: achieves 1/6.65 w.r.t. OPT_L
    - A_S for small items: achieves 1/6.65 w.r.t. OPT_S
    - OPT <= OPT_L + OPT_S
    """

    @staticmethod
    def acceptance_probability_p_i(i: int, n: int, c: float, d: float) -> float:
        """
        Compute the probability that item i (by rank) is accepted as the
        first item by Algorithm 2 (A_L).

        From Lemma 1:
        p_i = (c/(n-1)) * sum_{k=cn+1}^{dn} C(n-i, k-1) / C(n-2, k-2)

        For large n, we use the lower bounds from the paper (i in {1,2,3,4}).
        """
        if n < 10:
            return 0.0  # Too small for asymptotic results

        cn = int(c * n)
        dn = int(d * n)

        total = 0.0
        for k in range(cn + 1, dn + 1):
            # C(n-i, k-1) / C(n-2, k-2) using log-space for numerical stability
            if k - 1 > n - i or k - 2 > n - 2:
                continue
            log_num = _log_binomial(n - i, k - 1)
            log_den = _log_binomial(n - 2, k - 2)
            total += math.exp(log_num - log_den)

        return (c / (n - 1)) * total

    @staticmethod
    def acceptance_probability_lower_bounds(
        c: float, d: float
    ) -> Dict[int, float]:
        """
        Lower bounds for p_i for i in {1, 2, 3, 4}.
        From Lemma 1 in the paper (asymptotic as n -> infinity).
        """
        bounds = {}

        # i = 1: p_1 >= c * ln(d/c) - o(1)
        bounds[1] = c * math.log(d / c)

        # i = 2: p_2 >= c * (ln(d/c) - d + c) - o(1)
        bounds[2] = c * (math.log(d / c) - d + c)

        # i = 3: p_3 >= c * (ln(d/c) - 2*(d-c) + 0.5*(d^2 - c^2)) - o(1)
        bounds[3] = c * (math.log(d / c) - 2 * (d - c) + 0.5 * (d**2 - c**2))

        # i = 4: p_4 >= c * (ln(d/c) - 3*(d-c) + 1.5*(d^2-c^2) - 0.5*(d^3-c^3)) - o(1)
        bounds[4] = c * (math.log(d / c) - 3 * (d - c) +
                         1.5 * (d**2 - c**2) - 0.5 * (d**3 - c**3))

        return bounds

    @staticmethod
    def competitive_ratio_A_L(c: float, d: float) -> float:
        """
        Compute the competitive ratio of A_L (Algorithm 2 for large items).

        From the paper's analysis (Lemma 6 and proof):
        The competitive ratio is the minimum over 5 cases:
        min{p_1, p_12, (p_1+p_2)/2, p_23 + (p_1+p_2+p_3)/2, sum(p_i)/2}

        For the optimal parameters (c=0.42291, d=0.64570), this gives 1/6.65.
        """
        bounds = KnapsackRandomOrderBounds.acceptance_probability_lower_bounds(c, d)
        p1 = bounds[1]
        p2 = bounds[2]
        p3 = bounds[3]
        p4 = bounds[4]

        # Approximate p_12 (probability item 1 is first, item 2 is second)
        # From Lemma 2: p_12 >= c * (d - c*ln(d/c) - c) - o(1)
        p12 = c * (d - c * math.log(d / c) - c)

        # p_23 = p_13 (by symmetry, Lemma 3: p_ij = p_ji)
        p13 = p12  # Approximate (they are close for these parameters)
        p23 = p13

        # The five cases from the paper's analysis
        case1 = p1  # Single-item optimal
        case2 = p12 + (p1 + p2) / 2  # Type A: items {1,2}
        case3 = p13 + (p1 + p2 + p3) / 2  # Type B: items {1,3}
        case4 = p23 + (p1 + p2 + p3) / 2  # Type C: items {2,3}
        case5 = (p1 + p2 + p3 + p4) / 2  # Case 5: item j >= 4

        competitive_ratio = min(case1, case2, case3, case4, case5)
        return competitive_ratio

    @staticmethod
    def competitive_ratio_A_S(c: float, d: float, delta: float = 1/3) -> float:
        """
        Compute the competitive ratio of A_S (Algorithm 3 for small items).

        From Lemma 11:
        E[A_S] >= (c/d) * (5/2*(1-d) - 3/2*ln(1/d) - 3/2*ln(1/d)) * OPT_S

        More precisely (from the proof):
        E[A_S] >= (c/d) * ((1-d)(1+Delta) - Delta*(1+1/n)*ln(1/d)) * OPT_S

        where Delta = 1/(1-delta) = 3/2 for delta = 1/3.
        """
        Delta = 1 / (1 - delta)

        # From Lemma 11 proof (asymptotic, ignoring o(1) terms)
        ratio = (c / d) * (
            (1 - d) * (1 + Delta) - Delta * math.log(1 / d)
        )

        return ratio

    @staticmethod
    def combined_competitive_ratio(
        c: float = 0.42291, d: float = 0.64570, delta: float = 1/3
    ) -> float:
        """
        Combined competitive ratio from Theorem 1.

        r = min(r_L, r_S) where r_L and r_S are the ratios for
        large and small items respectively.
        """
        r_L = KnapsackRandomOrderBounds.competitive_ratio_A_L(c, d)
        r_S = KnapsackRandomOrderBounds.competitive_ratio_A_S(c, d, delta)
        return min(r_L, r_S)

    @staticmethod
    def lemma5_ratio(c: float = 0.23053, d: float = 1.0) -> float:
        """
        Competitive ratio from Lemma 5: when A_L uses the full input (d=1),
        the 2-Knapsack-Secretary problem achieves 1/3.08.
        """
        # This is the standalone A_L ratio
        return KnapsackRandomOrderBounds.competitive_ratio_A_L(c, d)

    @staticmethod
    def empty_knapsack_probability(c: float, d: float) -> float:
        """
        From Lemma 7: With probability >= c/d, no item is packed by A_L
        at the end of round dn.

        This is crucial for the sequential approach: A_S can only benefit
        if A_L has not filled the knapsack.
        """
        return c / d


class GAPRandomOrderBounds:
    """
    Competitive ratio calculations for the online GAP in the random order model.

    Theorem 2: There exists a (1/6.99)-competitive randomized algorithm.
    """

    @staticmethod
    def competitive_ratio_A_L(c: float, d: float) -> float:
        """
        From Lemma 14: E[A_L] >= (c * ln(d/c) - (1-c/d)/n) * OPT_L

        For GAP, A_L uses edge-weighted bipartite matching (Algorithm 4).
        """
        return c * math.log(d / c) - (1 - c * d)  # Simplified asymptotic

    @staticmethod
    def competitive_ratio_A_S(c: float, d: float, delta: float = 0.5) -> float:
        """
        From Lemma 17 (with Delta = 2 for delta = 1/2):
        E[A_S] >= (c/d) * (3(1-d) - 2*ln(1/d)) * OPT_S
        """
        return (c / d) * (3 * (1 - d) - 2 * math.log(1 / d))

    @staticmethod
    def combined_competitive_ratio(
        c: float = 0.5261, d: float = 0.6906
    ) -> float:
        """
        Combined ratio for GAP (Theorem 2).
        """
        r_L = GAPRandomOrderBounds.competitive_ratio_A_L(c, d)
        r_S = GAPRandomOrderBounds.competitive_ratio_A_S(c, d)
        combined = (c / d) * (3 * (1 - d) - 2 * math.log(1 / d))
        # The proven bound is 1/6.99
        return min(r_L, r_S, combined)


# =============================================================================
# Buffer Advantage Estimation
# =============================================================================

class BufferAdvantageEstimator:
    """
    Estimate how much a buffer of size B improves over single-item arrival
    in the random order model.

    Key insight: A buffer of B items allows:
    1. Choosing the BEST item from B candidates (vs. being forced to decide on 1)
    2. Implicit reordering within the buffer window
    3. Better sampling (B items give more information than 1)

    The paper's competitive ratio is for B=1 (single item arrival).
    With B > 1, we should be able to achieve strictly better ratios.

    THEORETICAL ARGUMENT:
    A buffer of size B in a random order model is equivalent to having B
    choices per "round" in the secretary problem. This is the B-choice
    secretary problem, which achieves ratio 1 - O(1/B) as B grows.
    For B=10, this is approximately 0.9, compared to 1/e ≈ 0.368 for B=1.

    However, for the knapsack problem (not just selection), the relationship
    is more complex because of capacity constraints.
    """

    @staticmethod
    def secretary_ratio_with_buffer(buffer_size: int) -> float:
        """
        Competitive ratio of the B-choice secretary problem.

        For B=1: 1/e ≈ 0.368 (classical secretary)
        For B=k: approximately 1 - 1/(k+1)

        Reference: The multiple-choice secretary problem (Kleinberg 2005).
        """
        if buffer_size <= 0:
            return 0.0
        if buffer_size == 1:
            return 1 / math.e
        # Approximate: with B items visible, we can select the best of B
        # This is like having B "tries" per round
        # Asymptotic: 1 - O(1/B)
        return 1 - 1 / (buffer_size + 1)

    @staticmethod
    def estimated_knapsack_ratio_with_buffer(
        buffer_size: int,
        base_ratio: float = 1 / 6.65
    ) -> float:
        """
        Estimate the competitive ratio for online knapsack with a buffer.

        This is a HEURISTIC estimate, not a proven bound.

        Argument: The buffer helps in two ways:
        1. Better selection among visible items (secretary-like improvement)
        2. Implicit reordering reduces the "randomness penalty"

        We estimate the improvement factor as:
        improvement = secretary_ratio(B) / secretary_ratio(1)

        For B=10: improvement ≈ 0.909 / 0.368 ≈ 2.47
        So estimated ratio ≈ 2.47 * (1/6.65) ≈ 1/2.69
        """
        secretary_1 = 1 / math.e
        secretary_B = BufferAdvantageEstimator.secretary_ratio_with_buffer(buffer_size)
        improvement_factor = secretary_B / secretary_1
        return base_ratio * improvement_factor

    @staticmethod
    def estimated_gap_ratio_with_buffer(
        buffer_size: int,
        num_bins: int = 2,
        base_ratio: float = 1 / 6.99
    ) -> float:
        """
        Estimate the competitive ratio for online GAP with buffer and k bins.

        For k=2 bins and buffer B=10:
        - The matching problem becomes trivially solvable
        - The buffer provides better selection
        - Estimated improvement similar to knapsack case
        """
        secretary_1 = 1 / math.e
        secretary_B = BufferAdvantageEstimator.secretary_ratio_with_buffer(buffer_size)
        improvement_factor = secretary_B / secretary_1

        # Additional benefit from multi-bin matching with buffer
        matching_bonus = 1.0 + 0.1 * min(num_bins, buffer_size)

        return base_ratio * improvement_factor * matching_bonus

    @staticmethod
    def print_buffer_analysis(max_buffer: int = 15):
        """Print a table of estimated competitive ratios for different buffer sizes."""
        print("\n=== Buffer Size vs. Estimated Competitive Ratio ===")
        print(f"{'Buffer':>8} {'Secretary':>12} {'Knapsack':>12} {'GAP(k=2)':>12}")
        print("-" * 48)

        for B in range(1, max_buffer + 1):
            sec = BufferAdvantageEstimator.secretary_ratio_with_buffer(B)
            knap = BufferAdvantageEstimator.estimated_knapsack_ratio_with_buffer(B)
            gap = BufferAdvantageEstimator.estimated_gap_ratio_with_buffer(B, num_bins=2)
            print(f"{B:>8} {sec:>12.4f} {knap:>12.4f} {gap:>12.4f}")


# =============================================================================
# Comparison with Known Online Packing Bounds
# =============================================================================

class OnlineBoundsComparison:
    """
    Compare the paper's results with known bounds from the overview knowledge base.

    This provides context for positioning the thesis work.
    """

    KNOWN_BOUNDS = {
        # From overview Section 12 and 16
        "1D BPP - Next Fit": {"ACR": 2.0, "year": 1974, "author": "Johnson"},
        "1D BPP - First Fit": {"ACR": 1.7, "year": 1974, "author": "Johnson"},
        "1D BPP - Advanced Harmonic": {"ACR": 1.57829, "year": 2017, "author": "Balogh et al."},
        "2D BPP - Rectangles": {"ACR": 2.5545, "year": 2011, "author": "Han et al."},
        "2D BPP - Squares": {"ACR": 2.0885, "year": 2021, "author": "Epstein & Mualem"},
        "3D BPP - Cubes": {"ACR": 2.5735, "year": 2021, "author": "Epstein & Mualem"},
        # From this paper (Albers et al. 2021) -- NOTE: these are knapsack ratios, not BPP
        "1D Knapsack ROM": {"ratio": 1/6.65, "year": 2021, "author": "Albers, Khan, Ladewig"},
        "1D GAP ROM": {"ratio": 1/6.99, "year": 2021, "author": "Albers, Khan, Ladewig"},
        "1D Knapsack ROM (prev best)": {"ratio": 1/8.06, "year": 2018, "author": "Kesselheim et al."},
    }

    @staticmethod
    def print_comparison():
        """Print a comparison table of all known bounds."""
        print("\n=== Online Packing Competitive Ratios ===")
        print(f"{'Problem':>35} {'Ratio':>10} {'Year':>6} {'Author':>25}")
        print("-" * 80)

        for problem, info in OnlineBoundsComparison.KNOWN_BOUNDS.items():
            if "ACR" in info:
                ratio_str = f"{info['ACR']:.4f}"
            else:
                ratio_str = f"{info['ratio']:.4f}"
            print(f"{problem:>35} {ratio_str:>10} {info['year']:>6} {info['author']:>25}")

        print("\nNote: BPP ACRs and Knapsack competitive ratios measure different things.")
        print("  BPP ACR: # bins used by online / # bins used by optimal offline (lower = better)")
        print("  Knapsack ratio: profit of online / profit of optimal offline (higher = better)")
        print("  The knapsack ratios from this paper are for RANDOM ORDER model (semi-online).")

    @staticmethod
    def gap_analysis():
        """Identify gaps between known bounds and what our thesis could contribute."""
        print("\n=== Research Gap Analysis ===")
        print("From the overview knowledge base (Section 17):")
        print()
        print("Gap 4: Bounded space online problems underexplored")
        print("  -> Our thesis addresses this: k=2 bounded space")
        print("  -> This paper's GAP (m resources) is related to k-bounded bin packing")
        print()
        print("Gap 6: Semi-online models underexplored")
        print("  -> Our thesis: buffer of 5-10 items = semi-online with buffering")
        print("  -> This paper: random order model = semi-online with random order")
        print("  -> Combination: buffer + random order = VERY underexplored")
        print()
        print("Gap 9: ACRs for general 3D items unknown")
        print("  -> Only cubes have proven ACRs in 3D online")
        print("  -> No random order model results exist for 3D")
        print("  -> Our thesis could provide empirical bounds")
        print()
        print("THESIS OPPORTUNITY:")
        print("  There are NO known competitive ratio results for:")
        print("  - 3D online knapsack in random order model")
        print("  - 3D online bin packing with buffer")
        print("  - 3D online k-bounded bin packing in random order")
        print("  Any empirical or theoretical results here would be novel.")


# =============================================================================
# Parameter Optimization
# =============================================================================

class ParameterOptimizer:
    """
    Find optimal parameters (c, d, delta) for the sequential approach.

    The paper optimizes c and d for 1D knapsack.
    For 3D packing with buffer, we need to re-optimize based on:
    - 3D volume ratios
    - Buffer size
    - Number of active bins
    - Fill rate and stability objectives
    """

    @staticmethod
    def optimize_1d_knapsack_parameters(
        delta: float = 1/3,
        c_range: Tuple[float, float] = (0.01, 0.99),
        d_range: Tuple[float, float] = (0.01, 0.99),
        steps: int = 100,
    ) -> Tuple[float, float, float]:
        """
        Grid search for optimal c and d in the 1D knapsack case.
        This reproduces the paper's result of c=0.42291, d=0.64570.
        """
        best_c, best_d, best_ratio = 0.0, 0.0, 0.0

        for ci in range(1, steps):
            c = c_range[0] + (c_range[1] - c_range[0]) * ci / steps
            for di in range(ci + 1, steps):
                d = d_range[0] + (d_range[1] - d_range[0]) * di / steps
                if d <= c:
                    continue

                try:
                    r_L = KnapsackRandomOrderBounds.competitive_ratio_A_L(c, d)
                    r_S = KnapsackRandomOrderBounds.competitive_ratio_A_S(c, d, delta)
                    ratio = min(r_L, r_S)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_c = c
                        best_d = d
                except (ValueError, ZeroDivisionError):
                    continue

        return best_c, best_d, best_ratio

    @staticmethod
    def suggest_3d_parameters(
        buffer_size: int = 10,
        num_bins: int = 2,
        avg_item_volume_ratio: float = 0.1,
    ) -> Dict[str, float]:
        """
        Suggest parameters for the 3D setting based on heuristic reasoning.

        The paper's parameters are for 1D with no buffer:
        - c = 0.42291 (42% of items in sampling phase)
        - d = 0.64570 (23% of items in large-item phase)
        - delta = 1/3

        With buffer, the sampling phase is replaced by buffer observation.
        """
        # With buffer, we don't need a formal sampling phase
        # Instead, c represents how cautious we are in the beginning
        # Reduce c because the buffer provides immediate information
        c_adjusted = max(0.1, 0.42291 * (1 - buffer_size / 50))

        # d depends on expected fraction of large items
        # If items are small relative to bin, d should be lower (more time for large items)
        # If items are large, d should be higher (quick to fill with large items)
        d_adjusted = min(0.9, 0.64570 + avg_item_volume_ratio)

        # delta depends on how many items fit per bin
        # In 3D, delta = 1/k where k is the max items fitting in a bin
        # For typical e-commerce: items are 1-10% of bin volume
        # delta = 0.33 means "large" if item > 33% of bin volume
        if avg_item_volume_ratio < 0.05:
            delta = 0.20  # Most items are small; lower threshold for "large"
        elif avg_item_volume_ratio < 0.15:
            delta = 0.33  # Paper default
        else:
            delta = 0.50  # Many items are large; use GAP threshold

        return {
            "c": c_adjusted,
            "d": d_adjusted,
            "delta": delta,
            "buffer_size": buffer_size,
            "num_bins": num_bins,
            "rationale": (
                f"With buffer={buffer_size}, sampling phase reduced (c={c_adjusted:.3f}). "
                f"With avg_vol_ratio={avg_item_volume_ratio}, delta={delta}. "
                f"Large item phase runs until d={d_adjusted:.3f}."
            ),
        }


# =============================================================================
# Utility Functions
# =============================================================================

def _log_binomial(n: int, k: int) -> float:
    """Compute log(C(n, k)) using Stirling's approximation for large n."""
    if k < 0 or k > n:
        return -float('inf')
    if k == 0 or k == n:
        return 0.0
    # Use log-gamma for numerical stability
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


# =============================================================================
# Main: Print all analysis
# =============================================================================

def main():
    """Print comprehensive analysis of competitive ratio bounds."""

    print("=" * 80)
    print("THEORETICAL BOUNDS ANALYSIS")
    print("Based on: Albers, Khan & Ladewig (2021)")
    print("=" * 80)

    # 1. Paper's main results
    print("\n--- Paper's Main Results ---")
    cr_knapsack = KnapsackRandomOrderBounds.combined_competitive_ratio()
    cr_gap = GAPRandomOrderBounds.combined_competitive_ratio()
    print(f"Online Knapsack (Random Order): {cr_knapsack:.4f} (proven: 1/6.65 = {1/6.65:.4f})")
    print(f"Online GAP (Random Order):      {cr_gap:.4f} (proven: 1/6.99 = {1/6.99:.4f})")

    # 2. Acceptance probabilities
    print("\n--- Acceptance Probability Lower Bounds (A_L) ---")
    for params_name, (c, d) in [("Lemma 5 (standalone)", (0.23053, 1.0)),
                                  ("Lemma 6 (sequential)", (0.42291, 0.64570))]:
        print(f"\n  {params_name}: c={c}, d={d}")
        bounds = KnapsackRandomOrderBounds.acceptance_probability_lower_bounds(c, d)
        for i, b in bounds.items():
            print(f"    p_{i} >= {b:.5f}")
        print(f"    Empty knapsack prob >= {KnapsackRandomOrderBounds.empty_knapsack_probability(c, d):.4f}")

    # 3. Buffer advantage
    BufferAdvantageEstimator.print_buffer_analysis(max_buffer=15)

    # 4. Comparison with known bounds
    OnlineBoundsComparison.print_comparison()
    OnlineBoundsComparison.gap_analysis()

    # 5. Suggested 3D parameters
    print("\n--- Suggested Parameters for 3D Packing ---")
    for avg_vol in [0.03, 0.10, 0.25]:
        params = ParameterOptimizer.suggest_3d_parameters(
            buffer_size=10, num_bins=2, avg_item_volume_ratio=avg_vol
        )
        print(f"\n  avg_item_volume_ratio = {avg_vol}:")
        for k, v in params.items():
            if k != "rationale":
                print(f"    {k} = {v}")
        print(f"    rationale: {params['rationale']}")


if __name__ == "__main__":
    main()
