"""
Dataset loader — read/write box datasets from/to JSON files.

Usage:
    from dataset.loader import load_dataset, save_dataset
    boxes = load_dataset("dataset/uniform_50.json")
"""

import json
import os
from typing import List, Optional

from config import Box


def load_dataset(path: str) -> List[Box]:
    """
    Load a dataset JSON and return a list of Box objects.

    Expected JSON schema::

        { "boxes": [{"id": 0, "length": 12, "width": 8, "height": 5}, ...] }
    """
    with open(path, "r") as f:
        data = json.load(f)
    return [Box.from_dict(b) for b in data["boxes"]]


def load_dataset_metadata(path: str) -> dict:
    """Load dataset JSON and return metadata (everything except boxes)."""
    with open(path, "r") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k != "boxes"}


def save_dataset(
    boxes: List[Box],
    path: str,
    name: Optional[str] = None,
    generator: str = "custom",
    params: Optional[dict] = None,
) -> None:
    """Save a list of Box objects as a dataset JSON file."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    data = {
        "name": name or os.path.splitext(os.path.basename(path))[0],
        "generator": generator,
        "params": params or {},
        "box_count": len(boxes),
        "boxes": [b.to_dict() for b in boxes],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
