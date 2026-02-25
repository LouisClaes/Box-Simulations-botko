"""Core data models for box packing simulation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Box:
    """Represents a 3D box with dimensions and weight."""

    id: int
    width: float  # cm
    height: float  # cm
    depth: float  # cm
    weight: float  # kg

    @property
    def volume(self) -> float:
        """Calculate box volume in cm³."""
        return self.width * self.height * self.depth

    def __repr__(self) -> str:
        return (
            f"Box(id={self.id}, "
            f"{self.width}×{self.height}×{self.depth}cm, "
            f"{self.weight}kg)"
        )


@dataclass
class PlacedBox:
    """A box with its position on a pallet."""

    box: Box
    x: float  # Position coordinates (cm)
    y: float
    z: float

    @property
    def volume(self) -> float:
        return self.box.volume

    @property
    def weight(self) -> float:
        return self.box.weight


class Pallet:
    """Represents a pallet that can hold multiple boxes."""

    # Standard EUR pallet dimensions
    WIDTH = 120.0  # cm
    DEPTH = 80.0   # cm
    MAX_HEIGHT = 200.0  # cm
    MAX_WEIGHT = 1000.0  # kg

    def __init__(self, pallet_id: int):
        self.id = pallet_id
        self.boxes: list[PlacedBox] = []
        self._is_closed = False

    @property
    def is_closed(self) -> bool:
        """Check if pallet is closed (no more boxes can be added)."""
        return self._is_closed

    def close(self) -> None:
        """Mark pallet as closed."""
        self._is_closed = True

    @property
    def current_height(self) -> float:
        """Get the current height of boxes on the pallet."""
        if not self.boxes:
            return 0.0
        return max(pb.z + pb.box.height for pb in self.boxes)

    @property
    def current_weight(self) -> float:
        """Get the current total weight on the pallet."""
        return sum(pb.weight for pb in self.boxes)

    @property
    def utilization(self) -> float:
        """Calculate volume utilization percentage."""
        if not self.boxes:
            return 0.0
        used_volume = sum(pb.volume for pb in self.boxes)
        total_volume = self.WIDTH * self.DEPTH * self.MAX_HEIGHT
        return (used_volume / total_volume) * 100

    def can_add_box(self, box: Box) -> bool:
        """Check if a box can be added without exceeding limits."""
        if self._is_closed:
            return False

        # Simple checks: weight and height
        if self.current_weight + box.weight > self.MAX_WEIGHT:
            return False

        # For simplicity, assume boxes stack vertically
        if self.current_height + box.height > self.MAX_HEIGHT:
            return False

        return True

    def add_box(self, box: Box, x: float = 0.0, y: float = 0.0) -> bool:
        """
        Add a box to the pallet.

        Args:
            box: Box to add
            x, y: Position coordinates (simplified: always stack at current height)

        Returns:
            True if box was added successfully, False otherwise
        """
        if not self.can_add_box(box):
            return False

        z = self.current_height
        placed = PlacedBox(box=box, x=x, y=y, z=z)
        self.boxes.append(placed)
        return True

    def __repr__(self) -> str:
        status = "CLOSED" if self._is_closed else "OPEN"
        return (
            f"Pallet(id={self.id}, status={status}, "
            f"boxes={len(self.boxes)}, "
            f"weight={self.current_weight:.1f}kg, "
            f"height={self.current_height:.1f}cm, "
            f"util={self.utilization:.1f}%)"
        )
