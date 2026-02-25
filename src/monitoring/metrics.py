"""Metrics tracking and export for bin packing experiments.

Provides dataclasses for tracking experiment metrics and utilities for
exporting results to JSON and CSV formats.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PalletMetrics:
    """Metrics for a single pallet.

    Attributes:
        pallet_id: Unique identifier for the pallet.
        boxes_placed: Number of boxes successfully placed.
        utilization_pct: Volume utilization percentage (0-100).
        volume_used: Total volume used in cubic cm.
        volume_total: Total available volume in cubic cm.
        algorithm: Algorithm name used for packing.
        dataset_id: Dataset identifier this pallet belongs to.
        closed_at: Timestamp when pallet was closed.
    """

    pallet_id: int
    boxes_placed: int
    utilization_pct: float
    volume_used: float
    volume_total: float
    algorithm: str
    dataset_id: str
    closed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with ISO timestamp.

        Returns:
            Dictionary representation with closed_at as ISO string.

        Example:
            >>> pm = PalletMetrics(1, 25, 78.5, 94200, 120000, "FirstFit", "dataset_001")
            >>> d = pm.to_dict()
            >>> d["pallet_id"]
            1
            >>> d["utilization_pct"]
            78.5
        """
        d = asdict(self)
        d["closed_at"] = self.closed_at.isoformat()
        return d


@dataclass
class ExperimentMetrics:
    """Aggregate metrics for an entire experiment run.

    Attributes:
        experiment_id: Unique identifier for the experiment.
        algorithm: Algorithm name used.
        total_datasets: Number of datasets processed.
        total_pallets: Total number of pallets closed.
        total_boxes: Total number of boxes placed.
        avg_utilization_pct: Average utilization across all pallets.
        median_utilization_pct: Median utilization across all pallets.
        min_utilization_pct: Minimum utilization across all pallets.
        max_utilization_pct: Maximum utilization across all pallets.
        runtime_seconds: Total runtime in seconds.
        errors_count: Number of errors encountered.
        started_at: Experiment start timestamp.
        completed_at: Experiment completion timestamp (None if running).
        pallet_metrics: List of per-pallet metrics.
    """

    experiment_id: str
    algorithm: str
    total_datasets: int = 0
    total_pallets: int = 0
    total_boxes: int = 0
    avg_utilization_pct: float = 0.0
    median_utilization_pct: float = 0.0
    min_utilization_pct: float = 0.0
    max_utilization_pct: float = 0.0
    runtime_seconds: float = 0.0
    errors_count: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    pallet_metrics: list[PalletMetrics] = field(default_factory=list)

    def add_pallet(self, pallet: PalletMetrics) -> None:
        """Add a pallet's metrics to the experiment.

        Args:
            pallet: PalletMetrics instance to add.

        Example:
            >>> em = ExperimentMetrics("exp_001", "FirstFit", total_datasets=10)
            >>> pm = PalletMetrics(1, 25, 78.5, 94200, 120000, "FirstFit", "dataset_001")
            >>> em.add_pallet(pm)
            >>> em.total_pallets
            1
            >>> em.total_boxes
            25
        """
        self.pallet_metrics.append(pallet)
        self.total_pallets += 1
        self.total_boxes += pallet.boxes_placed
        self._recalculate_stats()

    def record_error(self) -> None:
        """Increment error counter.

        Example:
            >>> em = ExperimentMetrics("exp_001", "FirstFit", total_datasets=10)
            >>> em.record_error()
            >>> em.errors_count
            1
        """
        self.errors_count += 1

    def mark_complete(self) -> None:
        """Mark experiment as complete and calculate final runtime.

        Example:
            >>> em = ExperimentMetrics("exp_001", "FirstFit", total_datasets=10)
            >>> em.mark_complete()
            >>> em.completed_at is not None
            True
            >>> em.runtime_seconds > 0
            True
        """
        self.completed_at = datetime.utcnow()
        self.runtime_seconds = (self.completed_at - self.started_at).total_seconds()

    def _recalculate_stats(self) -> None:
        """Recalculate aggregate statistics from pallet metrics."""
        if not self.pallet_metrics:
            return

        utilizations = [p.utilization_pct for p in self.pallet_metrics]
        self.avg_utilization_pct = sum(utilizations) / len(utilizations)
        self.min_utilization_pct = min(utilizations)
        self.max_utilization_pct = max(utilizations)

        # Calculate median
        sorted_utils = sorted(utilizations)
        n = len(sorted_utils)
        if n % 2 == 0:
            self.median_utilization_pct = (sorted_utils[n // 2 - 1] + sorted_utils[n // 2]) / 2
        else:
            self.median_utilization_pct = sorted_utils[n // 2]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with ISO timestamps.

        Returns:
            Dictionary representation with timestamps as ISO strings.

        Example:
            >>> em = ExperimentMetrics("exp_001", "FirstFit", total_datasets=10)
            >>> d = em.to_dict()
            >>> d["experiment_id"]
            'exp_001'
            >>> d["algorithm"]
            'FirstFit'
        """
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat()
        d["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        d["pallet_metrics"] = [p.to_dict() for p in self.pallet_metrics]
        return d

    def to_summary_dict(self) -> dict[str, Any]:
        """Convert to summary dictionary without per-pallet details.

        Returns:
            Dictionary with aggregate metrics only (no pallet_metrics list).

        Example:
            >>> em = ExperimentMetrics("exp_001", "FirstFit", total_datasets=10)
            >>> d = em.to_summary_dict()
            >>> "pallet_metrics" in d
            False
            >>> "total_pallets" in d
            True
        """
        d = self.to_dict()
        del d["pallet_metrics"]
        return d


def export_to_json(metrics: ExperimentMetrics, output_path: Path | str, include_pallets: bool = True) -> None:
    """Export experiment metrics to JSON file.

    Args:
        metrics: ExperimentMetrics instance to export.
        output_path: Path to output JSON file.
        include_pallets: If True, include per-pallet metrics. If False, summary only.

    Example:
        >>> em = ExperimentMetrics("exp_001", "FirstFit", total_datasets=10)
        >>> export_to_json(em, "/tmp/results.json", include_pallets=False)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = metrics.to_dict() if include_pallets else metrics.to_summary_dict()

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)


def export_to_csv(metrics: ExperimentMetrics, output_path: Path | str) -> None:
    """Export per-pallet metrics to CSV file.

    Args:
        metrics: ExperimentMetrics instance to export.
        output_path: Path to output CSV file.

    Example:
        >>> em = ExperimentMetrics("exp_001", "FirstFit", total_datasets=10)
        >>> pm = PalletMetrics(1, 25, 78.5, 94200, 120000, "FirstFit", "dataset_001")
        >>> em.add_pallet(pm)
        >>> export_to_csv(em, "/tmp/pallets.csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not metrics.pallet_metrics:
        # Write empty CSV with headers
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "pallet_id", "dataset_id", "algorithm", "boxes_placed",
                "utilization_pct", "volume_used", "volume_total", "closed_at"
            ])
        return

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pallet_id", "dataset_id", "algorithm", "boxes_placed",
            "utilization_pct", "volume_used", "volume_total", "closed_at"
        ])
        writer.writeheader()
        for pallet in metrics.pallet_metrics:
            row = pallet.to_dict()
            writer.writerow(row)


def print_summary(metrics: ExperimentMetrics) -> str:
    """Generate human-readable summary of experiment metrics.

    Args:
        metrics: ExperimentMetrics instance to summarize.

    Returns:
        Formatted multi-line summary string.

    Example:
        >>> em = ExperimentMetrics("exp_001", "FirstFit", total_datasets=10)
        >>> pm = PalletMetrics(1, 25, 78.5, 94200, 120000, "FirstFit", "dataset_001")
        >>> em.add_pallet(pm)
        >>> em.mark_complete()
        >>> summary = print_summary(em)
        >>> "Experiment: exp_001" in summary
        True
        >>> "Algorithm: FirstFit" in summary
        True
    """
    lines = [
        "=" * 60,
        f"Experiment: {metrics.experiment_id}",
        f"Algorithm: {metrics.algorithm}",
        "=" * 60,
        f"Datasets Processed: {metrics.total_datasets}",
        f"Total Pallets: {metrics.total_pallets}",
        f"Total Boxes: {metrics.total_boxes}",
        "",
        "Utilization Statistics:",
        f"  Average: {metrics.avg_utilization_pct:.2f}%",
        f"  Median:  {metrics.median_utilization_pct:.2f}%",
        f"  Min:     {metrics.min_utilization_pct:.2f}%",
        f"  Max:     {metrics.max_utilization_pct:.2f}%",
        "",
        f"Runtime: {metrics.runtime_seconds:.1f} seconds ({metrics.runtime_seconds / 60:.1f} minutes)",
        f"Errors: {metrics.errors_count}",
        "",
        f"Started:   {metrics.started_at.isoformat()}",
        f"Completed: {metrics.completed_at.isoformat() if metrics.completed_at else 'In Progress'}",
        "=" * 60,
    ]
    return "\n".join(lines)
