"""Dataset generation for box packing experiments."""

import random
from typing import Callable

from src.core.models import Box


def generate_boxes(count: int = 300, seed: int | None = None) -> list[Box]:
    """
    Generate random boxes for experimentation.

    Args:
        count: Number of boxes to generate
        seed: Random seed for reproducibility (default: None)

    Returns:
        List of Box objects with random dimensions and weights
    """
    if seed is not None:
        random.seed(seed)

    boxes = []
    for i in range(count):
        # Random dimensions: 10-100 cm each dimension
        width = random.uniform(10.0, 100.0)
        height = random.uniform(10.0, 100.0)
        depth = random.uniform(10.0, 100.0)

        # Random weight: 0.5-50 kg
        weight = random.uniform(0.5, 50.0)

        boxes.append(Box(id=i, width=width, height=height, depth=depth, weight=weight))

    return boxes


def random_order(boxes: list[Box]) -> list[Box]:
    """
    Return boxes in random order.

    Args:
        boxes: List of boxes

    Returns:
        Shuffled copy of boxes
    """
    shuffled = boxes.copy()
    random.shuffle(shuffled)
    return shuffled


def size_sorted_order(boxes: list[Box]) -> list[Box]:
    """
    Sort boxes by volume (largest first).

    Args:
        boxes: List of boxes

    Returns:
        Boxes sorted by volume descending
    """
    return sorted(boxes, key=lambda b: b.volume, reverse=True)


def weight_sorted_order(boxes: list[Box]) -> list[Box]:
    """
    Sort boxes by weight (heaviest first).

    Args:
        boxes: List of boxes

    Returns:
        Boxes sorted by weight descending
    """
    return sorted(boxes, key=lambda b: b.weight, reverse=True)


# Map of ordering strategy names to functions
ORDERING_STRATEGIES: dict[str, Callable[[list[Box]], list[Box]]] = {
    "random": random_order,
    "size_sorted": size_sorted_order,
    "weight_sorted": weight_sorted_order,
}


def get_ordering_strategy(name: str) -> Callable[[list[Box]], list[Box]]:
    """
    Get an ordering strategy function by name.

    Args:
        name: Strategy name (random, size_sorted, weight_sorted)

    Returns:
        Ordering function

    Raises:
        ValueError: If strategy name is not recognized
    """
    if name not in ORDERING_STRATEGIES:
        raise ValueError(
            f"Unknown ordering strategy: {name}. "
            f"Available: {list(ORDERING_STRATEGIES.keys())}"
        )
    return ORDERING_STRATEGIES[name]
