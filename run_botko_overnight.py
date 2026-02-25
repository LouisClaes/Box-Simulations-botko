#!/usr/bin/env python3
"""
Botko Overnight Runner - Production Script
Runs 10 datasets × 3 orderings × 300 boxes with Telegram notifications
"""
import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.runner.experiment import ExperimentRunner


async def main():
    """
    Run overnight experiments with your exact specifications:
    - 10 datasets
    - 300 boxes per dataset
    - 3 orderings per dataset (random, size-sorted, weight-sorted)
    - Total: 10 × 3 = 30 experimental runs
    - Only CLOSED pallets count in metrics
    - Telegram notifications enabled
    """
    print("=" * 60)
    print("🤖 Botko Overnight Experiment Runner")
    print("=" * 60)
    print()
    print("Configuration:")
    print("  📦 10 datasets")
    print("  📦 300 boxes per dataset")
    print("  📦 3 ordering strategies per dataset")
    print("  📦 Total experimental runs: 30")
    print("  📦 CPU limited to ~50% (via nice level)")
    print("  📊 Only closed pallets counted in metrics")
    print("  💬 Telegram notifications enabled")
    print()
    print("=" * 60)
    print()

    # Create runner with Telegram enabled
    runner = ExperimentRunner(
        algorithm="SimplePacker",
        results_dir="results",
        send_telegram_updates=True,
    )

    # Run the full experiment
    metrics = await runner.run_experiment(
        num_datasets=10,
        boxes_per_dataset=300,
    )

    print()
    print("=" * 60)
    print("✅ EXPERIMENT COMPLETE!")
    print("=" * 60)
    print()
    print(f"📊 Final Results:")
    print(f"   Total closed pallets: {metrics.total_pallets}")
    print(f"   Total boxes packed:   {metrics.total_boxes}")
    print(f"   Average utilization:  {metrics.avg_utilization_pct:.1f}%")
    print(f"   Runtime:              {metrics.runtime_seconds:.1f}s ({metrics.runtime_seconds/60:.1f} min)")
    print(f"   Errors encountered:   {metrics.errors_count}")
    print()
    print(f"📁 Results saved to: results/")
    print()
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Partial results may be saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
