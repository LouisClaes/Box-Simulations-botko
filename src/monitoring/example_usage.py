"""Example usage of the monitoring module for bin packing experiments.

This demonstrates how to integrate Telegram notifications and metrics tracking
into your experiment runner.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from src.monitoring import (
    ExperimentMetrics,
    PalletMetrics,
    export_to_csv,
    export_to_json,
    format_dataset_milestone,
    format_experiment_start,
    format_final_summary,
    format_pallet_closure,
    print_summary,
    send_telegram,
)


async def run_example_experiment() -> None:
    """Simulate a bin packing experiment with monitoring.

    This example shows:
    - Sending experiment start notification
    - Tracking per-pallet metrics
    - Sending milestone updates
    - Exporting results to JSON and CSV
    - Sending final summary
    """
    # Initialize experiment metrics
    experiment = ExperimentMetrics(
        experiment_id="exp_001",
        algorithm="FirstFit",
        total_datasets=10,
    )

    # Send experiment start notification
    start_msg = format_experiment_start(
        total_datasets=10,
        boxes_per_dataset=100,
        algorithm="FirstFit",
        pallet_dims=(120.0, 100.0, 150.0),
    )
    await send_telegram(start_msg)
    print(f"Sent start notification:\n{start_msg}\n")

    # Simulate processing datasets
    for dataset_id in range(1, 11):
        # Simulate packing boxes onto pallets
        for pallet_num in range(1, 4):  # 3 pallets per dataset
            pallet_id = (dataset_id - 1) * 3 + pallet_num
            boxes_placed = 25 + pallet_num * 5  # Simulate varying box counts
            volume_total = 120.0 * 100.0 * 150.0  # 1,800,000 cm³
            volume_used = volume_total * (0.7 + pallet_num * 0.05)  # Simulate utilization
            utilization = (volume_used / volume_total) * 100

            # Record pallet metrics
            pallet = PalletMetrics(
                pallet_id=pallet_id,
                boxes_placed=boxes_placed,
                utilization_pct=utilization,
                volume_used=volume_used,
                volume_total=volume_total,
                algorithm="FirstFit",
                dataset_id=f"dataset_{dataset_id:03d}",
                closed_at=datetime.utcnow(),
            )
            experiment.add_pallet(pallet)

            # Optionally send pallet closure notification (for high-value pallets)
            if utilization > 80:
                pallet_msg = format_pallet_closure(
                    pallet_id=pallet_id,
                    boxes_placed=boxes_placed,
                    utilization_pct=utilization,
                    algorithm="FirstFit",
                )
                await send_telegram(pallet_msg)
                print(f"Sent pallet closure notification (high utilization):\n{pallet_msg}\n")

        # Send milestone updates every 3 datasets
        if dataset_id % 3 == 0:
            milestone_msg = format_dataset_milestone(
                datasets_completed=dataset_id,
                total_datasets=10,
                avg_utilization=experiment.avg_utilization_pct,
            )
            await send_telegram(milestone_msg)
            print(f"Sent milestone notification:\n{milestone_msg}\n")

    # Mark experiment complete
    experiment.mark_complete()

    # Export results
    output_dir = Path("/tmp/bin_packing_results")
    export_to_json(experiment, output_dir / "experiment_summary.json", include_pallets=False)
    export_to_json(experiment, output_dir / "experiment_full.json", include_pallets=True)
    export_to_csv(experiment, output_dir / "pallets.csv")
    print(f"Results exported to {output_dir}\n")

    # Print summary
    summary = print_summary(experiment)
    print(summary)

    # Send final summary notification
    final_msg = format_final_summary(
        total_pallets=experiment.total_pallets,
        total_boxes=experiment.total_boxes,
        avg_utilization=experiment.avg_utilization_pct,
        runtime_seconds=experiment.runtime_seconds,
        errors=experiment.errors_count,
    )
    await send_telegram(final_msg)
    print(f"\nSent final summary notification:\n{final_msg}")


async def minimal_integration_example() -> None:
    """Minimal example showing how to integrate into your runner.

    This is what you'd actually add to your experiment runner code.
    """
    # At the start of your experiment
    metrics = ExperimentMetrics(
        experiment_id="exp_20260221_101530",
        algorithm="BestFit",
        total_datasets=50,
    )

    msg = format_experiment_start(50, 100, "BestFit", (120, 100, 150))
    await send_telegram(msg)

    # When a pallet is closed
    pallet = PalletMetrics(
        pallet_id=1,
        boxes_placed=42,
        utilization_pct=87.3,
        volume_used=1_572_000,
        volume_total=1_800_000,
        algorithm="BestFit",
        dataset_id="dataset_001",
    )
    metrics.add_pallet(pallet)

    # At the end
    metrics.mark_complete()
    export_to_json(metrics, "/tmp/results.json")
    print(print_summary(metrics))


if __name__ == "__main__":
    # Run the full example
    asyncio.run(run_example_experiment())

    # Or run the minimal example
    # asyncio.run(minimal_integration_example())
