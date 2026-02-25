"""
Batch experiment runner — run multiple strategies × datasets × seeds.

Supports the 2K bound pallet setup (2 pallets, 5-10 box buffer, semi-online).
Generates shuffled orderings for each seed to test robustness.

Usage (CLI):
    python batch_runner.py --strategies baseline walle_scoring --datasets 5 --seeds 3 --boxes 50
    python batch_runner.py --all-strategies --datasets 10 --seeds 5 --boxes 30 --verbose
    python batch_runner.py --strategy baseline --sweep-boxes 20 30 50 100

Usage (Python):
    from batch_runner import BatchRunner
    runner = BatchRunner(strategies=["baseline", "walle_scoring"], n_datasets=5, n_seeds=3)
    results = runner.run_all()
    runner.print_summary(results)
"""

import argparse
import json
import os
import sys
import time
import random
import itertools
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Box, BinConfig, ExperimentConfig
from dataset.generator import generate_uniform, generate_warehouse
from dataset.loader import load_dataset, save_dataset
from run_experiment import run_experiment
from strategies.base_strategy import STRATEGY_REGISTRY

# Import all strategies to register them
import strategies  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    """Configuration for a single batch run."""
    strategy_name: str
    dataset_index: int
    seed: int
    shuffle_index: int
    boxes: List[Box]
    bin_config: BinConfig
    enable_stability: bool = False
    min_support_ratio: float = 0.8
    allow_all_orientations: bool = False


@dataclass
class RunResult:
    """Result from a single experiment run."""
    strategy_name: str
    dataset_index: int
    seed: int
    shuffle_index: int
    n_boxes: int
    fill_rate: float
    boxes_placed: int
    boxes_rejected: int
    max_height: float
    computation_time_ms: float
    stability_rate: float
    completed: bool


@dataclass
class StrategyStats:
    """Aggregated statistics for a single strategy across all runs."""
    strategy_name: str
    n_runs: int = 0
    fill_rates: List[float] = field(default_factory=list)
    placed_counts: List[int] = field(default_factory=list)
    times_ms: List[float] = field(default_factory=list)
    stability_rates: List[float] = field(default_factory=list)

    @property
    def mean_fill_rate(self) -> float:
        return sum(self.fill_rates) / max(len(self.fill_rates), 1)

    @property
    def min_fill_rate(self) -> float:
        return min(self.fill_rates) if self.fill_rates else 0.0

    @property
    def max_fill_rate(self) -> float:
        return max(self.fill_rates) if self.fill_rates else 0.0

    @property
    def std_fill_rate(self) -> float:
        if len(self.fill_rates) < 2:
            return 0.0
        mean = self.mean_fill_rate
        return (sum((x - mean) ** 2 for x in self.fill_rates) / (len(self.fill_rates) - 1)) ** 0.5

    @property
    def mean_time_ms(self) -> float:
        return sum(self.times_ms) / max(len(self.times_ms), 1)

    @property
    def mean_stability(self) -> float:
        return sum(self.stability_rates) / max(len(self.stability_rates), 1)

    @property
    def mean_placed(self) -> float:
        return sum(self.placed_counts) / max(len(self.placed_counts), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Batch Runner
# ─────────────────────────────────────────────────────────────────────────────

class BatchRunner:
    """
    Orchestrates batch experiments across strategies, datasets, and seeds.

    For each (strategy, dataset, seed) triple:
      1. Generates a dataset with the given seed
      2. Creates `n_shuffles` random permutations of box ordering
      3. Runs the strategy on each permutation
      4. Aggregates results

    This tests strategy robustness to arrival order — critical for
    semi-online settings where box order is not controlled.
    """

    def __init__(
        self,
        strategies: Optional[List[str]] = None,
        n_datasets: int = 5,
        n_seeds: int = 3,
        n_shuffles: int = 3,
        n_boxes: int = 30,
        generator: str = "uniform",
        bin_config: Optional[BinConfig] = None,
        enable_stability: bool = False,
        min_support_ratio: float = 0.8,
        allow_all_orientations: bool = False,
        verbose: bool = False,
        output_dir: str = "results/batch",
    ):
        self.strategies = strategies or list(STRATEGY_REGISTRY.keys())
        self.n_datasets = n_datasets
        self.n_seeds = n_seeds
        self.n_shuffles = n_shuffles
        self.n_boxes = n_boxes
        self.generator = generator
        self.bin_config = bin_config or BinConfig()
        self.enable_stability = enable_stability
        self.min_support_ratio = min_support_ratio
        self.allow_all_orientations = allow_all_orientations
        self.verbose = verbose
        self.output_dir = output_dir

    def generate_datasets(self) -> List[Tuple[int, int, List[Box]]]:
        """Generate (dataset_idx, seed, boxes) tuples."""
        datasets = []
        for ds_idx in range(self.n_datasets):
            for seed_idx in range(self.n_seeds):
                seed = ds_idx * 1000 + seed_idx * 100 + 42
                if self.generator == "warehouse":
                    boxes = generate_warehouse(
                        self.n_boxes, bin_config=self.bin_config, seed=seed,
                    )
                else:
                    boxes = generate_uniform(
                        self.n_boxes, min_dim=5.0, max_dim=25.0, seed=seed,
                    )
                datasets.append((ds_idx, seed, boxes))
        return datasets

    def shuffle_boxes(self, boxes: List[Box], seed: int, shuffle_idx: int) -> List[Box]:
        """Create a deterministic shuffled copy of the box list."""
        rng = random.Random(seed * 1000 + shuffle_idx)
        shuffled = list(boxes)
        rng.shuffle(shuffled)
        return shuffled

    def run_single(self, run_config: RunConfig) -> RunResult:
        """Execute a single experiment run."""
        config = ExperimentConfig(
            bin=run_config.bin_config,
            strategy_name=run_config.strategy_name,
            enable_stability=run_config.enable_stability,
            min_support_ratio=run_config.min_support_ratio,
            allow_all_orientations=run_config.allow_all_orientations,
            render_3d=False,
            verbose=False,
        )

        result = run_experiment(config, run_config.boxes)
        m = result["metrics"]

        return RunResult(
            strategy_name=run_config.strategy_name,
            dataset_index=run_config.dataset_index,
            seed=run_config.seed,
            shuffle_index=run_config.shuffle_index,
            n_boxes=len(run_config.boxes),
            fill_rate=m["fill_rate"],
            boxes_placed=m["boxes_placed"],
            boxes_rejected=m["boxes_rejected"],
            max_height=m["max_height"],
            computation_time_ms=m["computation_time_ms"],
            stability_rate=m["stability_rate"],
            completed=result.get("completed", True),
        )

    def run_all(self) -> List[RunResult]:
        """Run all experiments and return results."""
        datasets = self.generate_datasets()
        total_runs = len(self.strategies) * len(datasets) * self.n_shuffles
        results: List[RunResult] = []

        print(f"\n{'='*70}")
        print(f"  BATCH EXPERIMENT")
        print(f"  Strategies:  {len(self.strategies)}")
        print(f"  Datasets:    {len(datasets)} ({self.n_datasets} datasets × {self.n_seeds} seeds)")
        print(f"  Shuffles:    {self.n_shuffles} per dataset")
        print(f"  Total runs:  {total_runs}")
        print(f"  Box count:   {self.n_boxes}")
        print(f"  Bin:         {self.bin_config.length}×{self.bin_config.width}×{self.bin_config.height}")
        print(f"  Generator:   {self.generator}")
        print(f"  Stability:   {'ON' if self.enable_stability else 'OFF'}")
        print(f"{'='*70}\n")

        run_idx = 0
        t_start = time.perf_counter()

        for strat_name in self.strategies:
            if strat_name not in STRATEGY_REGISTRY:
                print(f"  WARNING: Strategy '{strat_name}' not registered, skipping.")
                continue

            for ds_idx, seed, boxes in datasets:
                for shuf_idx in range(self.n_shuffles):
                    run_idx += 1
                    shuffled = self.shuffle_boxes(boxes, seed, shuf_idx)

                    run_config = RunConfig(
                        strategy_name=strat_name,
                        dataset_index=ds_idx,
                        seed=seed,
                        shuffle_index=shuf_idx,
                        boxes=shuffled,
                        bin_config=self.bin_config,
                        enable_stability=self.enable_stability,
                        min_support_ratio=self.min_support_ratio,
                        allow_all_orientations=self.allow_all_orientations,
                    )

                    try:
                        result = self.run_single(run_config)
                        results.append(result)

                        if self.verbose:
                            print(
                                f"  [{run_idx:4d}/{total_runs}] "
                                f"{strat_name:25s} | ds={ds_idx} seed={seed} shuf={shuf_idx} | "
                                f"fill={result.fill_rate:.1%} placed={result.boxes_placed}/{result.n_boxes} "
                                f"time={result.computation_time_ms:.0f}ms"
                            )
                        elif run_idx % 10 == 0 or run_idx == total_runs:
                            elapsed = time.perf_counter() - t_start
                            eta = (elapsed / run_idx) * (total_runs - run_idx)
                            print(
                                f"  Progress: {run_idx}/{total_runs} "
                                f"({run_idx/total_runs:.0%}) | "
                                f"ETA: {eta:.0f}s"
                            )

                    except Exception as e:
                        print(f"  ERROR [{strat_name}] ds={ds_idx} seed={seed}: {e}")

        elapsed_total = time.perf_counter() - t_start
        print(f"\n  Completed {len(results)}/{total_runs} runs in {elapsed_total:.1f}s\n")

        return results

    def aggregate(self, results: List[RunResult]) -> Dict[str, StrategyStats]:
        """Aggregate results by strategy."""
        stats: Dict[str, StrategyStats] = {}

        for r in results:
            if r.strategy_name not in stats:
                stats[r.strategy_name] = StrategyStats(strategy_name=r.strategy_name)

            s = stats[r.strategy_name]
            s.n_runs += 1
            s.fill_rates.append(r.fill_rate)
            s.placed_counts.append(r.boxes_placed)
            s.times_ms.append(r.computation_time_ms)
            s.stability_rates.append(r.stability_rate)

        return stats

    def print_summary(self, results: List[RunResult]) -> None:
        """Print a clean summary table."""
        stats = self.aggregate(results)

        # Sort by mean fill rate descending
        sorted_stats = sorted(stats.values(), key=lambda s: s.mean_fill_rate, reverse=True)

        print(f"\n{'='*90}")
        print(f"  STRATEGY COMPARISON — {len(results)} total runs")
        print(f"{'='*90}")
        print(
            f"  {'Strategy':<25s} {'Runs':>5s} "
            f"{'Mean Fill':>10s} {'±Std':>7s} "
            f"{'Min':>7s} {'Max':>7s} "
            f"{'Placed':>7s} {'Time(ms)':>9s} {'Stab%':>7s}"
        )
        print(f"  {'-'*84}")

        for s in sorted_stats:
            print(
                f"  {s.strategy_name:<25s} {s.n_runs:>5d} "
                f"{s.mean_fill_rate:>9.1%} {s.std_fill_rate:>6.1%} "
                f"{s.min_fill_rate:>6.1%} {s.max_fill_rate:>6.1%} "
                f"{s.mean_placed:>6.1f} {s.mean_time_ms:>8.1f} {s.mean_stability:>6.1%}"
            )

        print(f"  {'-'*84}")

        # Best strategy
        if sorted_stats:
            best = sorted_stats[0]
            print(f"\n  BEST: {best.strategy_name} — {best.mean_fill_rate:.1%} mean fill rate")
        print()

    def save_results(self, results: List[RunResult]) -> str:
        """Save all results to JSON."""
        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"batch_{ts}.json")

        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "n_strategies": len(self.strategies),
                "n_datasets": self.n_datasets,
                "n_seeds": self.n_seeds,
                "n_shuffles": self.n_shuffles,
                "n_boxes": self.n_boxes,
                "generator": self.generator,
                "bin": self.bin_config.to_dict(),
                "enable_stability": self.enable_stability,
            },
            "results": [
                {
                    "strategy": r.strategy_name,
                    "dataset_index": r.dataset_index,
                    "seed": r.seed,
                    "shuffle_index": r.shuffle_index,
                    "n_boxes": r.n_boxes,
                    "fill_rate": round(r.fill_rate, 6),
                    "boxes_placed": r.boxes_placed,
                    "boxes_rejected": r.boxes_rejected,
                    "max_height": round(r.max_height, 2),
                    "computation_time_ms": round(r.computation_time_ms, 2),
                    "stability_rate": round(r.stability_rate, 4),
                    "completed": r.completed,
                }
                for r in results
            ],
            "summary": {},
        }

        # Add aggregated stats
        stats = self.aggregate(results)
        for name, s in stats.items():
            data["summary"][name] = {
                "n_runs": s.n_runs,
                "mean_fill_rate": round(s.mean_fill_rate, 6),
                "std_fill_rate": round(s.std_fill_rate, 6),
                "min_fill_rate": round(s.min_fill_rate, 6),
                "max_fill_rate": round(s.max_fill_rate, 6),
                "mean_time_ms": round(s.mean_time_ms, 2),
                "mean_stability": round(s.mean_stability, 4),
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  Results saved: {path}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch Experiment Runner — test strategies across datasets and seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_runner.py --all-strategies --datasets 5 --seeds 3 --boxes 30
  python batch_runner.py --strategies baseline walle_scoring --datasets 10 --seeds 5
  python batch_runner.py --all-strategies --datasets 3 --seeds 2 --shuffles 5 -v
  python batch_runner.py --all-strategies --generator warehouse --boxes 50
        """,
    )

    # Strategy selection
    sg = parser.add_mutually_exclusive_group()
    sg.add_argument("--strategies", nargs="+", help="List of strategy names to test")
    sg.add_argument("--all-strategies", action="store_true",
                    help="Test all registered strategies")

    # Dataset parameters
    parser.add_argument("--datasets", type=int, default=5,
                        help="Number of distinct datasets to generate (default: 5)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds per dataset (default: 3)")
    parser.add_argument("--shuffles", type=int, default=3,
                        help="Number of box-order shuffles per (dataset, seed) (default: 3)")
    parser.add_argument("--boxes", type=int, default=30,
                        help="Number of boxes per dataset (default: 30)")
    parser.add_argument("--generator", choices=["uniform", "warehouse"], default="uniform",
                        help="Box generation method (default: uniform)")

    # Bin configuration
    parser.add_argument("--bin-length", type=float, default=120.0)
    parser.add_argument("--bin-width", type=float, default=80.0)
    parser.add_argument("--bin-height", type=float, default=150.0)
    parser.add_argument("--resolution", type=float, default=1.0)

    # Constraints
    parser.add_argument("--stability", action="store_true")
    parser.add_argument("--min-support", type=float, default=0.8)
    parser.add_argument("--all-orientations", action="store_true")

    # Output
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output-dir", default="results/batch")

    args = parser.parse_args()

    # Resolve strategies
    if args.all_strategies:
        strategies = list(STRATEGY_REGISTRY.keys())
    elif args.strategies:
        strategies = args.strategies
    else:
        strategies = list(STRATEGY_REGISTRY.keys())

    bin_config = BinConfig(
        length=args.bin_length, width=args.bin_width,
        height=args.bin_height, resolution=args.resolution,
    )

    runner = BatchRunner(
        strategies=strategies,
        n_datasets=args.datasets,
        n_seeds=args.seeds,
        n_shuffles=args.shuffles,
        n_boxes=args.boxes,
        generator=args.generator,
        bin_config=bin_config,
        enable_stability=args.stability,
        min_support_ratio=args.min_support,
        allow_all_orientations=args.all_orientations,
        verbose=args.verbose,
        output_dir=args.output_dir,
    )

    results = runner.run_all()
    runner.print_summary(results)
    runner.save_results(results)


if __name__ == "__main__":
    main()
