"""Lightweight Telegram notification for bin packing experiment progress.

Sends plain-text messages to a Telegram channel via the Bot API for:
- Experiment start/end notifications
- Dataset completion milestones
- Pallet closure events with utilization metrics
- Errors and warnings
- Final results summary

No retry logic — progress updates are non-critical.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


DEFAULT_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


async def send_telegram(
    message: str,
    chat_id: str | None = None,
    token: str | None = None,
) -> bool:
    """Send a plain-text message to a Telegram channel.

    Args:
        message: Text to send (plain text or MarkdownV2).
        chat_id: Telegram chat ID. Defaults to TELEGRAM_CHAT_ID env var.
        token: Bot token. Defaults to TELEGRAM_BOT_TOKEN env var.

    Returns:
        True if message was sent successfully, False otherwise.

    Example:
        >>> import asyncio
        >>> msg = "Experiment started: 1000 boxes across 10 datasets"
        >>> asyncio.run(send_telegram(msg))
        True
    """
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return False

    chat_id = chat_id or DEFAULT_CHAT_ID
    if not chat_id:
        return False

    url = TELEGRAM_API.format(token=token)
    payload = {"chat_id": chat_id, "text": message}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()
            return bool(data.get("ok", False))
    except (httpx.HTTPError, Exception):
        return False


def format_experiment_start(
    total_datasets: int,
    boxes_per_dataset: int,
    algorithm: str,
    pallet_dims: tuple[float, float, float],
) -> str:
    """Format experiment start notification message.

    Args:
        total_datasets: Number of datasets to process.
        boxes_per_dataset: Number of boxes per dataset.
        algorithm: Algorithm name (e.g., "FirstFit", "BestFit").
        pallet_dims: Pallet dimensions as (length, width, height).

    Returns:
        Formatted message string.

    Example:
        >>> msg = format_experiment_start(10, 100, "FirstFit", (120, 100, 150))
        >>> print(msg)
        🚀 Experiment Started
        Algorithm: FirstFit
        Datasets: 10 (100 boxes each)
        Pallet: 120.0 x 100.0 x 150.0 cm
    """
    return (
        f"🚀 Experiment Started\n"
        f"Algorithm: {algorithm}\n"
        f"Datasets: {total_datasets} ({boxes_per_dataset} boxes each)\n"
        f"Pallet: {pallet_dims[0]} x {pallet_dims[1]} x {pallet_dims[2]} cm"
    )


def format_dataset_milestone(
    datasets_completed: int,
    total_datasets: int,
    avg_utilization: float,
) -> str:
    """Format dataset completion milestone notification.

    Args:
        datasets_completed: Number of datasets completed so far.
        total_datasets: Total number of datasets.
        avg_utilization: Average utilization percentage across completed datasets.

    Returns:
        Formatted message string.

    Example:
        >>> msg = format_dataset_milestone(3, 10, 78.5)
        >>> print(msg)
        📊 Progress Update
        Completed: 3/10 datasets (30%)
        Avg Utilization: 78.5%
    """
    progress_pct = (datasets_completed / total_datasets) * 100
    return (
        f"📊 Progress Update\n"
        f"Completed: {datasets_completed}/{total_datasets} datasets ({progress_pct:.0f}%)\n"
        f"Avg Utilization: {avg_utilization:.1f}%"
    )


def format_pallet_closure(
    pallet_id: int,
    boxes_placed: int,
    utilization_pct: float,
    algorithm: str,
) -> str:
    """Format pallet closure notification.

    Args:
        pallet_id: Pallet identifier.
        boxes_placed: Number of boxes placed on this pallet.
        utilization_pct: Utilization percentage of this pallet.
        algorithm: Algorithm name.

    Returns:
        Formatted message string.

    Example:
        >>> msg = format_pallet_closure(42, 23, 82.3, "BestFit")
        >>> print(msg)
        📦 Pallet Closed
        ID: #42 (BestFit)
        Boxes: 23
        Utilization: 82.3%
    """
    return (
        f"📦 Pallet Closed\n"
        f"ID: #{pallet_id} ({algorithm})\n"
        f"Boxes: {boxes_placed}\n"
        f"Utilization: {utilization_pct:.1f}%"
    )


def format_error(error_type: str, error_message: str, context: dict[str, Any] | None = None) -> str:
    """Format error notification message.

    Args:
        error_type: Type of error (e.g., "ValidationError", "AlgorithmError").
        error_message: Detailed error message.
        context: Optional context dictionary with additional info.

    Returns:
        Formatted message string.

    Example:
        >>> msg = format_error("ValidationError", "Box exceeds pallet dimensions", {"box_id": 123})
        >>> print(msg)
        ⚠️ Error: ValidationError
        Box exceeds pallet dimensions
        Context: box_id=123
    """
    lines = [
        f"⚠️ Error: {error_type}",
        error_message,
    ]
    if context:
        ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
        lines.append(f"Context: {ctx_str}")
    return "\n".join(lines)


def format_final_summary(
    total_pallets: int,
    total_boxes: int,
    avg_utilization: float,
    runtime_seconds: float,
    errors: int,
) -> str:
    """Format final experiment results summary.

    Args:
        total_pallets: Total number of pallets closed.
        total_boxes: Total number of boxes placed.
        avg_utilization: Average utilization percentage across all pallets.
        runtime_seconds: Total runtime in seconds.
        errors: Number of errors encountered.

    Returns:
        Formatted message string.

    Example:
        >>> msg = format_final_summary(45, 1000, 79.2, 3600, 2)
        >>> print(msg)
        ✅ Experiment Complete
        Pallets: 45
        Boxes: 1000
        Avg Utilization: 79.2%
        Runtime: 60.0 minutes
        Errors: 2
    """
    runtime_minutes = runtime_seconds / 60
    return (
        f"✅ Experiment Complete\n"
        f"Pallets: {total_pallets}\n"
        f"Boxes: {total_boxes}\n"
        f"Avg Utilization: {avg_utilization:.1f}%\n"
        f"Runtime: {runtime_minutes:.1f} minutes\n"
        f"Errors: {errors}"
    )
