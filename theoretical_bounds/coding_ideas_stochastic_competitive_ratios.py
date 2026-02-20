"""
Coding Ideas: Stochastic Competitive Ratio Estimation and Bounds
================================================================

Based on: "Near-optimal Algorithms for Stochastic Online Bin Packing"
          by Ayyadevara, Dabas, Khan & Sreenivas (ICALP 2022, arXiv 2025)

Purpose:
  - Simulate the competitive ratios from the paper for various distributions
  - Estimate achievable fill rates under the i.i.d. and random-order models
  - Compute theoretical bounds for our specific use case:
    3D, k=2 bounded, buffer 5-10, distribution from warehouse data
  - Provide benchmark targets for the thesis

==========================================================================
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from collections import defaultdict
import itertools


# =============================================================================
# PART 1: 1D STOCHASTIC BIN PACKING SIMULATOR
# =============================================================================

class StochasticBPSimulator:
    """
    Simulate 1D bin packing under stochastic models to empirically verify
    the paper's theoretical results and estimate bounds for our setting.

    The paper establishes:
    - i.i.d. model: ECR = (alpha + epsilon) where alpha is the offline AAR
    - Random-order: ARR of BF = 1 for items > 1/3
    - Random-order: ARR of BF <= 1.49107 for 3-Partition items (1/4, 1/2]

    This simulator lets us:
    1. Verify these bounds empirically
    2. Test with non-standard distributions (e.g., real warehouse data)
    3. Measure the effect of buffer size on competitive ratio
    4. Estimate the penalty from k=2 bounded space
    """

    def __init__(self, bin_capacity: float = 1.0):
        self.bin_capacity = bin_capacity

    # ---- Standard algorithms from the paper ----

    def next_fit(self, items: List[float]) -> int:
        """Next-Fit: pack into last bin; open new if doesn't fit."""
        if not items:
            return 0
        bins = [0.0]  # Current fill of each bin
        for item in items:
            if bins[-1] + item <= self.bin_capacity:
                bins[-1] += item
            else:
                bins.append(item)
        return len(bins)

    def first_fit(self, items: List[float]) -> int:
        """First-Fit: pack into first bin that has room."""
        if not items:
            return 0
        bins = [0.0]
        for item in items:
            placed = False
            for i in range(len(bins)):
                if bins[i] + item <= self.bin_capacity:
                    bins[i] += item
                    placed = True
                    break
            if not placed:
                bins.append(item)
        return len(bins)

    def best_fit(self, items: List[float]) -> int:
        """Best-Fit: pack into fullest bin where it fits."""
        if not items:
            return 0
        bins = [0.0]
        for item in items:
            best_idx = -1
            best_remaining = self.bin_capacity + 1
            for i in range(len(bins)):
                remaining = self.bin_capacity - bins[i]
                if item <= remaining < best_remaining:
                    best_remaining = remaining
                    best_idx = i
            if best_idx >= 0:
                bins[best_idx] += item
            else:
                bins.append(item)
        return len(bins)

    def first_fit_decreasing(self, items: List[float]) -> int:
        """First-Fit-Decreasing: sort items largest first, then FF."""
        sorted_items = sorted(items, reverse=True)
        return self.first_fit(sorted_items)

    def best_fit_decreasing(self, items: List[float]) -> int:
        """Best-Fit-Decreasing: sort items largest first, then BF."""
        sorted_items = sorted(items, reverse=True)
        return self.best_fit(sorted_items)

    def optimal_lower_bound(self, items: List[float]) -> int:
        """Lower bound on OPT: ceil(sum of items / bin_capacity)."""
        return int(np.ceil(sum(items) / self.bin_capacity))

    # ---- Buffer-enhanced algorithms (our extension) ----

    def best_fit_with_buffer(self, items: List[float],
                              buffer_size: int = 7) -> int:
        """
        Best-Fit with lookahead buffer.

        Strategy: from the buffer, pick the item that leads to the
        tightest fit in the fullest bin (double Best-Fit).
        """
        if not items:
            return 0

        bins = [0.0]
        buffer = []
        item_idx = 0

        while item_idx < len(items) or buffer:
            # Fill buffer
            while len(buffer) < buffer_size and item_idx < len(items):
                buffer.append(items[item_idx])
                item_idx += 1

            if not buffer:
                break

            # Find best (item, bin) pair
            best_item_idx = -1
            best_bin_idx = -1
            best_remaining = self.bin_capacity + 1

            for bi, buf_item in enumerate(buffer):
                for i in range(len(bins)):
                    remaining = self.bin_capacity - bins[i]
                    if buf_item <= remaining:
                        after_remaining = remaining - buf_item
                        if after_remaining < best_remaining:
                            best_remaining = after_remaining
                            best_item_idx = bi
                            best_bin_idx = i

            if best_item_idx >= 0:
                item = buffer.pop(best_item_idx)
                bins[best_bin_idx] += item
            else:
                # No fit; pick smallest item and open new bin
                smallest_idx = min(range(len(buffer)), key=lambda i: buffer[i])
                item = buffer.pop(smallest_idx)
                bins.append(item)

        return len(bins)

    def best_fit_bounded_k2(self, items: List[float]) -> int:
        """
        Best-Fit with k=2 bounded space.

        Only 2 bins are active at any time. When neither fits,
        close the fuller bin and open a new one.
        """
        if not items:
            return 0

        active = [0.0, 0.0]  # 2 active bins
        closed_count = 0

        for item in items:
            # Try to fit in one of the 2 active bins (best fit)
            best_idx = -1
            best_remaining = self.bin_capacity + 1
            for i in range(2):
                remaining = self.bin_capacity - active[i]
                if item <= remaining < best_remaining:
                    best_remaining = remaining
                    best_idx = i

            if best_idx >= 0:
                active[best_idx] += item
            else:
                # Close the fuller bin
                close_idx = 0 if active[0] >= active[1] else 1
                closed_count += 1
                active[close_idx] = item  # New bin with the item

        return closed_count + 2  # 2 remaining active bins

    def best_fit_bounded_k2_with_buffer(self, items: List[float],
                                         buffer_size: int = 7) -> int:
        """
        Best-Fit with k=2 bounded space AND buffer lookahead.
        This is our actual use case.
        """
        if not items:
            return 0

        active = [0.0, 0.0]
        closed_count = 0
        buffer = []
        item_idx = 0

        while item_idx < len(items) or buffer:
            while len(buffer) < buffer_size and item_idx < len(items):
                buffer.append(items[item_idx])
                item_idx += 1

            if not buffer:
                break

            # Find best (item, bin) pair among buffer x active bins
            best_item_idx = -1
            best_bin_idx = -1
            best_remaining = self.bin_capacity + 1

            for bi, buf_item in enumerate(buffer):
                for i in range(2):
                    remaining = self.bin_capacity - active[i]
                    if buf_item <= remaining:
                        after_remaining = remaining - buf_item
                        if after_remaining < best_remaining:
                            best_remaining = after_remaining
                            best_item_idx = bi
                            best_bin_idx = i

            if best_item_idx >= 0:
                item = buffer.pop(best_item_idx)
                active[best_bin_idx] += item
            else:
                # No fit; close fuller bin, open new
                close_idx = 0 if active[0] >= active[1] else 1
                closed_count += 1

                # From buffer, pick item that best starts a new bin
                # (largest item, to avoid wasting a new bin on a small item)
                largest_idx = max(range(len(buffer)), key=lambda i: buffer[i])
                item = buffer.pop(largest_idx)
                active[close_idx] = item

        return closed_count + 2

    # ---- Distribution generators ----

    @staticmethod
    def generate_uniform(n: int, a: float = 0.0, b: float = 1.0) -> List[float]:
        """U[a,b] distribution (paper's standard test)."""
        return list(np.random.uniform(a, b, n))

    @staticmethod
    def generate_discrete_uniform(n: int, k: int = 10) -> List[float]:
        """U{j,k} distribution: items from {1/k, 2/k, ..., 1}."""
        return list(np.random.choice([j/k for j in range(1, k+1)], size=n))

    @staticmethod
    def generate_3partition(n: int) -> List[float]:
        """3-Partition items: all sizes in (1/4, 1/2]."""
        return list(np.random.uniform(0.25 + 1e-6, 0.5, n))

    @staticmethod
    def generate_large_items(n: int) -> List[float]:
        """All items > 1/3 (paper's Theorem 1.2 case)."""
        return list(np.random.uniform(1/3 + 1e-6, 1.0, n))

    @staticmethod
    def generate_warehouse_like(n: int) -> List[float]:
        """
        Simulated warehouse distribution:
        - 60% small items (0.05 to 0.25)
        - 25% medium items (0.25 to 0.5)
        - 15% large items (0.5 to 0.85)
        This is a rough approximation of typical e-commerce warehouses.
        """
        items = []
        for _ in range(n):
            r = np.random.random()
            if r < 0.60:
                items.append(np.random.uniform(0.05, 0.25))
            elif r < 0.85:
                items.append(np.random.uniform(0.25, 0.50))
            else:
                items.append(np.random.uniform(0.50, 0.85))
        return items


# =============================================================================
# PART 2: COMPETITIVE RATIO ESTIMATOR
# =============================================================================

class CompetitiveRatioEstimator:
    """
    Empirically estimate competitive ratios for different algorithms
    and distribution types.

    Runs Monte Carlo simulations to estimate:
    - E[A(I)] / E[Opt(I)] (the ECR from the paper)
    - The gap between buffered and non-buffered algorithms
    - The penalty from k=2 bounded space

    Usage:
        estimator = CompetitiveRatioEstimator()
        results = estimator.run_experiment(
            distribution='uniform',
            algorithms=['BF', 'BF_buffer_7', 'BF_k2', 'BF_k2_buffer_7'],
            n_values=[100, 500, 1000, 5000],
            trials=100
        )
    """

    def __init__(self):
        self.sim = StochasticBPSimulator()

    def run_experiment(self,
                       distribution: str,
                       algorithms: List[str],
                       n_values: List[int],
                       trials: int = 100,
                       dist_params: dict = None) -> Dict:
        """
        Run a Monte Carlo experiment.

        Args:
            distribution: 'uniform', 'discrete_uniform', '3partition',
                          'large_items', 'warehouse_like'
            algorithms: list of algorithm names
            n_values: list of problem sizes to test
            trials: number of Monte Carlo trials per (n, algorithm)
            dist_params: optional parameters for the distribution

        Returns:
            Dictionary with results: {alg_name: {n: {'mean_ratio': ..., ...}}}
        """
        results = {}
        generator = self._get_generator(distribution, dist_params)

        for alg_name in algorithms:
            results[alg_name] = {}
            alg_func = self._get_algorithm(alg_name)

            for n in n_values:
                ratios = []
                for _ in range(trials):
                    items = generator(n)

                    # For random-order model: shuffle adversarially-chosen items
                    # For i.i.d.: items are already i.i.d.

                    alg_bins = alg_func(items)
                    opt_lower = self.sim.optimal_lower_bound(items)
                    # Also compute offline heuristic upper bound on OPT
                    ffd_bins = self.sim.first_fit_decreasing(items)

                    if opt_lower > 0:
                        # Use FFD as proxy for OPT (FFD has AAR 11/9)
                        ratio = alg_bins / max(opt_lower, 1)
                        ratios.append(ratio)

                results[alg_name][n] = {
                    'mean_ratio': np.mean(ratios),
                    'std_ratio': np.std(ratios),
                    'max_ratio': np.max(ratios),
                    'p95_ratio': np.percentile(ratios, 95),
                    'mean_fill_rate': 1.0 / np.mean(ratios) if np.mean(ratios) > 0 else 0,
                }

        return results

    def compare_buffer_benefit(self,
                                distribution: str = 'warehouse_like',
                                buffer_sizes: List[int] = [1, 3, 5, 7, 10],
                                n: int = 1000,
                                trials: int = 100) -> Dict:
        """
        Measure how much the buffer improves performance.

        This directly quantifies the advantage of our semi-online setup
        (buffer of 5-10) over pure online (buffer=1).

        Expected findings (based on paper's theory):
        - Buffer=1 (pure online): BF has CR ~1.7 worst case, ~1.1-1.3 for i.i.d.
        - Buffer=5-10: should approach offline quality, especially for i.i.d.
        - For k=2 bounded: buffer helps even more (compensates for limited bins)
        """
        results = {}
        generator = self._get_generator(distribution)

        for buf_size in buffer_sizes:
            ratios = []
            ratios_k2 = []
            for _ in range(trials):
                items = generator(n)
                opt = self.sim.optimal_lower_bound(items)

                if buf_size == 1:
                    alg_bins = self.sim.best_fit(items)
                    alg_bins_k2 = self.sim.best_fit_bounded_k2(items)
                else:
                    alg_bins = self.sim.best_fit_with_buffer(items, buf_size)
                    alg_bins_k2 = self.sim.best_fit_bounded_k2_with_buffer(
                        items, buf_size
                    )

                if opt > 0:
                    ratios.append(alg_bins / opt)
                    ratios_k2.append(alg_bins_k2 / opt)

            results[buf_size] = {
                'unbounded': {
                    'mean_cr': np.mean(ratios),
                    'std_cr': np.std(ratios),
                    'mean_fill_rate': 1.0 / np.mean(ratios),
                },
                'k2_bounded': {
                    'mean_cr': np.mean(ratios_k2),
                    'std_cr': np.std(ratios_k2),
                    'mean_fill_rate': 1.0 / np.mean(ratios_k2),
                },
                'k2_penalty': np.mean(ratios_k2) - np.mean(ratios),
                'buffer_vs_no_buffer': (
                    np.mean(ratios) - results.get(1, {}).get(
                        'unbounded', {}).get('mean_cr', np.mean(ratios))
                    if 1 in results else 0
                ),
            }

        return results

    def estimate_blueprint_benefit(self,
                                    distribution: str = 'warehouse_like',
                                    n: int = 1000,
                                    trials: int = 50) -> Dict:
        """
        Estimate the competitive ratio of blueprint packing vs plain BF.

        Simulates the paper's algorithm (simplified 1D version):
        1. First n/2 items: pack with Next-Fit (sampling stage)
        2. Use FFD packing of first n/2 as blueprint for second n/2
        3. Pack second n/2 using blueprint matching

        Compare with: plain BF, BF+buffer, FFD (offline)
        """
        generator = self._get_generator(distribution)
        results = {'blueprint': [], 'bf': [], 'bf_buffer': [], 'ffd': []}

        for _ in range(trials):
            items = generator(n)
            half = n // 2
            j1 = items[:half]
            j2 = items[half:]
            all_items = items

            # Blueprint packing (simplified)
            blueprint_bins = self._simulate_blueprint_1d(j1, j2)
            results['blueprint'].append(blueprint_bins)

            # Baselines
            results['bf'].append(self.sim.best_fit(all_items))
            results['bf_buffer'].append(
                self.sim.best_fit_with_buffer(all_items, 7)
            )
            results['ffd'].append(self.sim.first_fit_decreasing(all_items))

        opt_estimates = [self.sim.optimal_lower_bound(generator(n))
                         for _ in range(trials)]
        mean_opt = np.mean(opt_estimates)

        return {
            alg: {
                'mean_bins': np.mean(bins_list),
                'mean_cr_estimate': np.mean(bins_list) / mean_opt if mean_opt > 0 else 0,
            }
            for alg, bins_list in results.items()
        }

    def _simulate_blueprint_1d(self, j1: List[float], j2: List[float],
                                delta: float = 0.15) -> int:
        """
        Simplified 1D blueprint packing simulation.

        Following Algorithm 2 from the paper:
        1. Pack j1 offline (using FFD as A_alpha)
        2. Extract proxy items (large items) and S-slots
        3. Pack j2 online using the blueprint
        """
        # Step 1: Offline packing of j1
        large_j1 = sorted([x for x in j1 if x >= delta], reverse=True)
        small_j1 = [x for x in j1 if x < delta]

        # Simple FFD packing to get blueprint
        j1_sorted = sorted(j1, reverse=True)
        bins_j1 = []  # Each bin: list of items
        for item in j1_sorted:
            placed = False
            for b in bins_j1:
                if sum(b) + item <= 1.0:
                    b.append(item)
                    placed = True
                    break
            if not placed:
                bins_j1.append([item])

        # Step 2: Extract blueprint
        proxy_items = sorted(large_j1, reverse=False)  # Sorted ascending
        s_slot_capacities = []
        for b in bins_j1:
            large_in_bin = sum(x for x in b if x >= delta)
            s_slot = 1.0 - large_in_bin
            if s_slot > 0:
                s_slot_capacities.append(s_slot)

        # Step 3: Pack j2 online
        total_bins = len(bins_j1)  # Start with blueprint bins
        extra_bins = 0
        proxy_available = list(proxy_items)

        for item in j2:
            if item >= delta:
                # Large item: find matching proxy
                matched = False
                for pi in range(len(proxy_available)):
                    if proxy_available[pi] >= item:
                        proxy_available.pop(pi)
                        matched = True
                        break
                if not matched:
                    extra_bins += 1  # Open new bin
            else:
                # Small item: pack in S-slots (Next-Fit style)
                packed = False
                for si in range(len(s_slot_capacities)):
                    if s_slot_capacities[si] >= item:
                        s_slot_capacities[si] -= item
                        packed = True
                        break
                if not packed:
                    # Open new S-slot (new bin)
                    extra_bins += 1
                    s_slot_capacities.append(1.0 - item)

        return total_bins + extra_bins

    def _get_generator(self, distribution: str,
                        params: dict = None) -> Callable:
        """Get the distribution generator function."""
        generators = {
            'uniform': lambda n: self.sim.generate_uniform(n),
            'uniform_01': lambda n: self.sim.generate_uniform(n, 0, 1),
            'uniform_quarter': lambda n: self.sim.generate_uniform(n, 0, 0.25),
            'discrete_uniform': lambda n: self.sim.generate_discrete_uniform(n),
            '3partition': lambda n: self.sim.generate_3partition(n),
            'large_items': lambda n: self.sim.generate_large_items(n),
            'warehouse_like': lambda n: self.sim.generate_warehouse_like(n),
        }
        return generators.get(distribution, generators['warehouse_like'])

    def _get_algorithm(self, name: str) -> Callable:
        """Get the algorithm function by name."""
        algorithms = {
            'NF': self.sim.next_fit,
            'FF': self.sim.first_fit,
            'BF': self.sim.best_fit,
            'FFD': self.sim.first_fit_decreasing,
            'BFD': self.sim.best_fit_decreasing,
            'BF_buffer_3': lambda items: self.sim.best_fit_with_buffer(items, 3),
            'BF_buffer_5': lambda items: self.sim.best_fit_with_buffer(items, 5),
            'BF_buffer_7': lambda items: self.sim.best_fit_with_buffer(items, 7),
            'BF_buffer_10': lambda items: self.sim.best_fit_with_buffer(items, 10),
            'BF_k2': self.sim.best_fit_bounded_k2,
            'BF_k2_buffer_5': lambda items: self.sim.best_fit_bounded_k2_with_buffer(items, 5),
            'BF_k2_buffer_7': lambda items: self.sim.best_fit_bounded_k2_with_buffer(items, 7),
            'BF_k2_buffer_10': lambda items: self.sim.best_fit_bounded_k2_with_buffer(items, 10),
        }
        return algorithms.get(name, self.sim.best_fit)


# =============================================================================
# PART 3: EXAMPLE USAGE AND EXPERIMENT SCRIPTS
# =============================================================================

def example_verify_paper_results():
    """
    Verify the paper's theoretical results empirically.

    Expected results:
    - BF on U[0,1]: ECR converges to 1 for large n (perfectly packable)
    - BF on items > 1/3: ARR = 1 (Theorem 1.2)
    - BF on 3-Partition: ARR ~1.49 (Theorem 1.3)
    - NF on U[0,1]: ECR = 4/3 (known result)
    """
    estimator = CompetitiveRatioEstimator()

    print("=" * 60)
    print("Experiment 1: Verify paper's theoretical bounds")
    print("=" * 60)

    for dist_name in ['uniform', 'large_items', '3partition', 'warehouse_like']:
        print(f"\nDistribution: {dist_name}")
        results = estimator.run_experiment(
            distribution=dist_name,
            algorithms=['NF', 'FF', 'BF', 'FFD'],
            n_values=[100, 500, 1000, 5000],
            trials=50,
        )
        for alg in results:
            for n in results[alg]:
                r = results[alg][n]
                print(f"  {alg}, n={n}: CR={r['mean_ratio']:.4f} "
                      f"(+/- {r['std_ratio']:.4f}), "
                      f"fill={r['mean_fill_rate']:.4f}")


def example_buffer_analysis():
    """
    Quantify the benefit of buffer size for our use case.

    Key question: how much does a buffer of 5-10 help compared to
    pure online (buffer=1) or full offline?
    """
    estimator = CompetitiveRatioEstimator()

    print("=" * 60)
    print("Experiment 2: Buffer size benefit analysis")
    print("=" * 60)

    results = estimator.compare_buffer_benefit(
        distribution='warehouse_like',
        buffer_sizes=[1, 3, 5, 7, 10, 15],
        n=1000,
        trials=100,
    )

    for buf_size in sorted(results.keys()):
        r = results[buf_size]
        print(f"\nBuffer size: {buf_size}")
        print(f"  Unbounded: CR={r['unbounded']['mean_cr']:.4f}, "
              f"fill={r['unbounded']['mean_fill_rate']:.4f}")
        print(f"  k=2 bounded: CR={r['k2_bounded']['mean_cr']:.4f}, "
              f"fill={r['k2_bounded']['mean_fill_rate']:.4f}")
        print(f"  k=2 penalty: +{r['k2_penalty']:.4f}")


def example_blueprint_comparison():
    """
    Compare blueprint packing vs standard heuristics.
    """
    estimator = CompetitiveRatioEstimator()

    print("=" * 60)
    print("Experiment 3: Blueprint packing benefit")
    print("=" * 60)

    for dist in ['uniform', 'warehouse_like', '3partition']:
        print(f"\nDistribution: {dist}")
        results = estimator.estimate_blueprint_benefit(
            distribution=dist, n=1000, trials=50
        )
        for alg, r in results.items():
            print(f"  {alg}: mean_bins={r['mean_bins']:.1f}, "
                  f"CR_est={r['mean_cr_estimate']:.4f}")


if __name__ == '__main__':
    print("Stochastic Competitive Ratio Experiments")
    print("Based on: Ayyadevara et al., 'Near-optimal Algorithms for "
          "Stochastic Online Bin Packing'")
    print()

    example_verify_paper_results()
    print()
    example_buffer_analysis()
    print()
    example_blueprint_comparison()
