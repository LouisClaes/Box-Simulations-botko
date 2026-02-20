"""
dataset — box generation and loading utilities.

Public API:
    from dataset.generator import generate_uniform, generate_warehouse, generate_identical
    from dataset.loader import load_dataset, save_dataset
"""

from dataset.generator import generate_uniform, generate_warehouse, generate_identical
from dataset.loader import load_dataset, save_dataset

__all__ = [
    "generate_uniform", "generate_warehouse", "generate_identical",
    "load_dataset", "save_dataset",
]
