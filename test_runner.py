#!/usr/bin/env python3
"""Quick test script for experiment runner."""

import asyncio
from src.runner.experiment import ExperimentRunner


async def test_run():
    """Run a small test experiment."""
    print("=" * 60)
    print("Running test experiment: 2 datasets × 3 orderings × 30 boxes")
    print("=" * 60)
    print()

    runner = ExperimentRunner(send_telegram_updates=False)
    metrics = await runner.run_experiment(
        num_datasets=2,
        boxes_per_dataset=30,
    )

    print()
    print("=" * 60)
    print("✓ Test successful!")
    print(f"   Total pallets closed: {metrics.total_pallets}")
    print(f"   Total boxes packed: {metrics.total_boxes}")
    print(f"   Avg utilization: {metrics.avg_utilization_pct:.1f}%")
    print(f"   Min utilization: {metrics.min_utilization_pct:.1f}%")
    print(f"   Max utilization: {metrics.max_utilization_pct:.1f}%")
    print(f"   Runtime: {metrics.runtime_seconds:.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_run())
