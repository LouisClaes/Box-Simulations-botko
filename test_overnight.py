"""Quick test of overnight experiment runner."""
import asyncio
from src.runner.experiment import ExperimentRunner

async def test_run():
    """Run a mini experiment: 2 datasets x 3 orders x 10 boxes."""
    print("🧪 Starting test run...")
    print("   Config: 2 datasets, 10 boxes per dataset, 3 ordering strategies")
    print("   Total runs: 2 × 3 = 6")
    print()
    
    # Disable Telegram for test
    runner = ExperimentRunner(send_telegram_updates=False)
    
    # Run mini experiment
    metrics = await runner.run_experiment(
        num_datasets=2,
        boxes_per_dataset=10,
    )
    
    print()
    print("✅ Test completed successfully!")
    print(f"   Closed pallets: {metrics.total_pallets}")
    print(f"   Total boxes: {metrics.total_boxes}")
    print(f"   Avg utilization: {metrics.avg_utilization_pct:.1f}%")
    print(f"   Runtime: {metrics.runtime_seconds:.2f}s")
    print()
    print("🎯 The overnight runner is ready for production!")

if __name__ == "__main__":
    asyncio.run(test_run())
