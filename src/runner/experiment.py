"""Main experiment runner for box packing simulations."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.algorithms.simple_packer import SimplePacker
from src.core.models import Box, Pallet
from src.monitoring.metrics import (
    ExperimentMetrics,
    PalletMetrics,
    export_to_csv,
    export_to_json,
    print_summary,
)
from src.monitoring.telegram_notifier import (
    format_dataset_milestone,
    format_experiment_start,
    format_final_summary,
    send_telegram,
)
from src.runner.dataset import ORDERING_STRATEGIES, generate_boxes


class ExperimentRunner:
    """
    Main experiment orchestrator for box packing simulations.

    Runs multiple datasets with different ordering strategies,
    packs boxes into pallets, collects metrics, and sends progress updates.
    """

    def __init__(
        self,
        algorithm: str = "SimplePacker",
        results_dir: Path | str = "results",
        send_telegram_updates: bool = True,
    ):
        """
        Initialize experiment runner.

        Args:
            algorithm: Algorithm name to use (default: "SimplePacker")
            results_dir: Directory to save results (default: "results")
            send_telegram_updates: Whether to send Telegram notifications
        """
        self.algorithm = algorithm
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.send_telegram_updates = send_telegram_updates

    async def run_experiment(
        self,
        num_datasets: int = 10,
        boxes_per_dataset: int = 300,
    ) -> ExperimentMetrics:
        """
        Run full experiment across multiple datasets and orderings.

        Args:
            num_datasets: Number of datasets to generate (default: 10)
            boxes_per_dataset: Number of boxes per dataset (default: 300)

        Returns:
            ExperimentMetrics with aggregated results

        Flow:
            1. Generate experiment ID and create metrics
            2. Send start notification
            3. For each dataset:
                a. Generate boxes
                b. For each ordering strategy:
                    - Apply ordering
                    - Pack into pallets
                    - Collect metrics from CLOSED pallets
                    - Save intermediate results
                c. Send progress update
            4. Mark complete and send final summary
        """
        # Create experiment ID and metrics
        experiment_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        metrics = ExperimentMetrics(
            experiment_id=experiment_id,
            algorithm=self.algorithm,
            total_datasets=num_datasets * len(ORDERING_STRATEGIES),
        )

        # Send start notification
        if self.send_telegram_updates:
            start_msg = format_experiment_start(
                total_datasets=metrics.total_datasets,
                boxes_per_dataset=boxes_per_dataset,
                algorithm=self.algorithm,
                pallet_dims=(Pallet.WIDTH, Pallet.DEPTH, Pallet.MAX_HEIGHT),
            )
            await send_telegram(start_msg)

        # Run experiments
        datasets_completed = 0
        for dataset_idx in range(num_datasets):
            # Generate boxes for this dataset
            boxes = generate_boxes(count=boxes_per_dataset, seed=dataset_idx)

            # Run with each ordering strategy
            for strategy_name, strategy_fn in ORDERING_STRATEGIES.items():
                dataset_id = f"dataset_{dataset_idx:03d}_{strategy_name}"

                # Apply ordering
                ordered_boxes = strategy_fn(boxes)

                # Pack into pallets
                pallets = self._pack_boxes(ordered_boxes)

                # Collect metrics from CLOSED pallets only
                for pallet in pallets:
                    if pallet.is_closed:
                        pallet_metric = self._create_pallet_metric(
                            pallet=pallet,
                            dataset_id=dataset_id,
                        )
                        metrics.add_pallet(pallet_metric)

                datasets_completed += 1

                # Save intermediate results after each dataset
                self._save_results(metrics, suffix=f"_interim_{datasets_completed}")

            # Send progress update after each dataset (all 3 orderings)
            if self.send_telegram_updates and dataset_idx % 2 == 0:  # Every 2 datasets
                progress_msg = format_dataset_milestone(
                    datasets_completed=datasets_completed,
                    total_datasets=metrics.total_datasets,
                    avg_utilization=metrics.avg_utilization_pct,
                )
                await send_telegram(progress_msg)

        # Mark complete
        metrics.mark_complete()

        # Save final results
        self._save_results(metrics, suffix="_final")

        # Send final summary
        if self.send_telegram_updates:
            final_msg = format_final_summary(
                total_pallets=metrics.total_pallets,
                total_boxes=metrics.total_boxes,
                avg_utilization=metrics.avg_utilization_pct,
                runtime_seconds=metrics.runtime_seconds,
                errors=metrics.errors_count,
            )
            await send_telegram(final_msg)

        # Print summary to console
        print(print_summary(metrics))

        return metrics

    def _pack_boxes(self, boxes: list[Box]) -> list[Pallet]:
        """
        Pack boxes using the configured algorithm.

        Args:
            boxes: List of boxes to pack

        Returns:
            List of pallets (closed pallets only)
        """
        packer = SimplePacker()
        pallets = packer.pack(boxes)
        return pallets

    def _create_pallet_metric(
        self,
        pallet: Pallet,
        dataset_id: str,
    ) -> PalletMetrics:
        """
        Create metrics object from a pallet.

        Args:
            pallet: Pallet object
            dataset_id: Dataset identifier

        Returns:
            PalletMetrics instance
        """
        volume_used = sum(pb.volume for pb in pallet.boxes)
        volume_total = Pallet.WIDTH * Pallet.DEPTH * Pallet.MAX_HEIGHT

        return PalletMetrics(
            pallet_id=pallet.id,
            boxes_placed=len(pallet.boxes),
            utilization_pct=pallet.utilization,
            volume_used=volume_used,
            volume_total=volume_total,
            algorithm=self.algorithm,
            dataset_id=dataset_id,
        )

    def _save_results(self, metrics: ExperimentMetrics, suffix: str = "") -> None:
        """
        Save metrics to JSON and CSV files.

        Args:
            metrics: ExperimentMetrics to save
            suffix: Optional suffix for filename (e.g., "_interim_5")
        """
        base_filename = f"{metrics.experiment_id}{suffix}"

        # Save JSON (summary only for interim, full for final)
        json_path = self.results_dir / f"{base_filename}.json"
        include_pallets = suffix.endswith("_final")
        export_to_json(metrics, json_path, include_pallets=include_pallets)

        # Save CSV (per-pallet metrics)
        csv_path = self.results_dir / f"{base_filename}_pallets.csv"
        export_to_csv(metrics, csv_path)

        print(f"✓ Saved results to {json_path} and {csv_path}")


async def main(num_datasets: int = 10, boxes_per_dataset: int = 300) -> None:
    """
    Main entry point for running experiments.

    Args:
        num_datasets: Number of datasets (default: 10)
        boxes_per_dataset: Boxes per dataset (default: 300)
    """
    runner = ExperimentRunner()
    metrics = await runner.run_experiment(
        num_datasets=num_datasets,
        boxes_per_dataset=boxes_per_dataset,
    )

    print(f"\n🎉 Experiment complete!")
    print(f"   Total pallets: {metrics.total_pallets}")
    print(f"   Avg utilization: {metrics.avg_utilization_pct:.1f}%")
    print(f"   Runtime: {metrics.runtime_seconds:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run box packing experiments")
    parser.add_argument(
        "--datasets",
        type=int,
        default=10,
        help="Number of datasets to generate (default: 10)",
    )
    parser.add_argument(
        "--boxes",
        type=int,
        default=300,
        help="Number of boxes per dataset (default: 300)",
    )

    args = parser.parse_args()

    asyncio.run(main(num_datasets=args.datasets, boxes_per_dataset=args.boxes))
