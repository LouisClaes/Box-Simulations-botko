"""
Dataset generator — create box lists with various size distributions.

Generators:
    generate_uniform   — each dimension drawn from U[min, max]
    generate_warehouse — 60% small, 25% medium, 15% large (relative to bin)
    generate_identical — n identical boxes (for debugging / sanity checks)

Usage:
    from dataset.generator import generate_uniform
    boxes = generate_uniform(50, min_dim=5, max_dim=30, save_path="dataset/u50.json")
"""

import json
import random
import os
from typing import List, Optional

from config import Box, BinConfig


def generate_uniform(
    n: int,
    min_dim: float = 5.0,
    max_dim: float = 30.0,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[Box]:
    """
    Generate *n* boxes with each dimension drawn from U[min_dim, max_dim].

    Args:
        n:         Number of boxes.
        min_dim:   Minimum dimension (same for L, W, H).
        max_dim:   Maximum dimension.
        save_path: If given, save the dataset JSON here.
        seed:      Random seed for reproducibility.

    Returns:
        List of Box objects.
    """
    if seed is not None:
        random.seed(seed)

    boxes = [
        Box(id=i,
            length=round(random.uniform(min_dim, max_dim), 1),
            width=round(random.uniform(min_dim, max_dim), 1),
            height=round(random.uniform(min_dim, max_dim), 1))
        for i in range(n)
    ]

    if save_path:
        _save(boxes, save_path, generator="uniform",
              params={"n": n, "min_dim": min_dim, "max_dim": max_dim, "seed": seed})
    return boxes


def generate_warehouse(
    n: int,
    bin_config: Optional[BinConfig] = None,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[Box]:
    """
    Generate *n* boxes with warehouse-like distribution:
      60% small  (5–15% of bin dim)
      25% medium (15–35% of bin dim)
      15% large  (35–60% of bin dim)

    Args:
        n:          Number of boxes.
        bin_config: Bin dimensions for relative sizing.  Defaults to EUR pallet.
        save_path:  If given, save the dataset JSON here.
        seed:       Random seed for reproducibility.

    Returns:
        List of Box objects.
    """
    if seed is not None:
        random.seed(seed)

    cfg = bin_config or BinConfig()
    ref_dims = [cfg.length, cfg.width, cfg.height]

    boxes: List[Box] = []
    for i in range(n):
        r = random.random()
        if r < 0.60:
            lo, hi = 0.05, 0.15
        elif r < 0.85:
            lo, hi = 0.15, 0.35
        else:
            lo, hi = 0.35, 0.60

        dims = [max(1.0, round(random.uniform(lo * ref, hi * ref), 1))
                for ref in ref_dims]
        boxes.append(Box(id=i, length=dims[0], width=dims[1], height=dims[2]))

    if save_path:
        _save(boxes, save_path, generator="warehouse",
              params={"n": n, "seed": seed, "bin": cfg.to_dict()})
    return boxes


def generate_identical(
    n: int,
    length: float = 10.0,
    width: float = 10.0,
    height: float = 10.0,
    save_path: Optional[str] = None,
) -> List[Box]:
    """Generate *n* identical boxes — useful for debugging and sanity checks."""
    boxes = [Box(id=i, length=length, width=width, height=height) for i in range(n)]
    if save_path:
        _save(boxes, save_path, generator="identical",
              params={"n": n, "length": length, "width": width, "height": height})
    return boxes


def get_rajapack_catalog() -> List[dict]:
    return [
        {"name": "Vouwdoos A6", "w": 150, "d": 100, "h": 100},
        {"name": "Vouwdoos A5", "w": 220, "d": 160, "h": 100},
        {"name": "Vouwdoos A4", "w": 310, "d": 220, "h": 200},
        {"name": "Vouwdoos A3", "w": 430, "d": 310, "h": 250},
        {"name": "Vierkante doos S", "w": 200, "d": 200, "h": 200},
        {"name": "Vierkante doos M", "w": 300, "d": 300, "h": 300},
        {"name": "Exportdoos M", "w": 400, "d": 300, "h": 250},
        {"name": "Exportdoos L", "w": 600, "d": 400, "h": 400},
    ]


def generate_rajapack(
    n: int,
    min_dim: float = 0.0,
    max_dim: float = 9999.0,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[Box]:
    """
    Generate *n* boxes randomly selected from the Rajapack catalog.
    Only boxes where ALL dimensions are within [min_dim, max_dim] are considered.
    """
    if seed is not None:
        random.seed(seed)

    catalog = get_rajapack_catalog()
    
    # Filter catalog based on constraints
    valid_options = []
    for b in catalog:
        # Check against float limits
        if (min_dim <= b["w"] <= max_dim and 
            min_dim <= b["d"] <= max_dim and 
            min_dim <= b["h"] <= max_dim):
            valid_options.append(b)
            
    if not valid_options:
        print(f"Warning: No Rajapack boxes fit within {min_dim}-{max_dim}. Using full catalog.")
        valid_options = catalog

    boxes = []
    for i in range(n):
        template = random.choice(valid_options)
        weight = round(random.uniform(0.5, 5.0), 2)
        
        boxes.append(Box(
            id=i, 
            width=float(template["w"]), 
            length=float(template["d"]), 
            height=float(template["h"]),
            weight=weight
        ))

    if save_path:
        _save(boxes, save_path, generator="rajapack",
              params={"n": n, "min_dim": min_dim, "max_dim": max_dim, "seed": seed})
    return boxes


# ─── Internal ────────────────────────────────────────────────────────────────

def _save(boxes: List[Box], path: str, generator: str, params: dict) -> None:
    """Persist a box list as a dataset JSON file."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    data = {
        "name": os.path.splitext(os.path.basename(path))[0],
        "generator": generator,
        "params": params,
        "box_count": len(boxes),
        "boxes": [b.to_dict() for b in boxes],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
