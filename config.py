"""
Central configuration and data models for the box stacking framework.

All modules import their core types from here to ensure consistency
across the dataset, simulator, strategy, and visualization layers.

Classes:
    Box              — input box definition with dimensions and weight
    Orientation      — maps rotation index to (l, w, h) after rotation
    Placement        — validated result of placing a box at a position
    PlacementDecision— strategy's proposed placement (before validation)
    BinConfig        — bin/pallet physical dimensions and grid resolution
    ExperimentConfig — all tuneable parameters for a single experiment
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math


# ─────────────────────────────────────────────────────────────────────────────
# Box & Orientation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Box:
    """
    An input box to be packed.

    Attributes:
        id:     Unique identifier.
        length: X-axis extent (real-world units, e.g. cm).
        width:  Y-axis extent.
        height: Z-axis extent.
        weight: Optional weight for stability / pyramidal checks.
    """
    id: int
    length: float
    width: float
    height: float
    weight: float = 1.0

    @property
    def volume(self) -> float:
        """Total volume of the box."""
        return self.length * self.width * self.height

    def to_dict(self) -> dict:
        return {"id": self.id, "length": self.length, "width": self.width,
                "height": self.height, "weight": self.weight}

    @classmethod
    def from_dict(cls, d: dict) -> "Box":
        return cls(id=d["id"], length=d["length"], width=d["width"],
                   height=d["height"], weight=d.get("weight", 1.0))


class Orientation:
    """
    Maps an orientation index to the (length, width, height) after rotation.

    All orientations are orthogonal (90° axis-aligned rotations only).
    Index 0-1 are flat-base (z-axis rotation only).
    Index 0-5 cover all 6 axis-aligned permutations.
    """

    @staticmethod
    def get_all(l: float, w: float, h: float) -> List[Tuple[float, float, float]]:
        """Return all unique orthogonal orientations (up to 6)."""
        seen: set = set()
        orientations: List[Tuple[float, float, float]] = []
        for dims in [
            (l, w, h), (w, l, h),   # flat-base (z-axis 90°)
            (l, h, w), (h, l, w),   # y-axis rotations
            (w, h, l), (h, w, l),   # x-axis rotations
        ]:
            if dims not in seen:
                seen.add(dims)
                orientations.append(dims)
        return orientations

    @staticmethod
    def get_flat(l: float, w: float, h: float) -> List[Tuple[float, float, float]]:
        """Return only flat-base orientations (z-axis rotation, up to 2)."""
        seen: set = set()
        orientations: List[Tuple[float, float, float]] = []
        for dims in [(l, w, h), (w, l, h)]:
            if dims not in seen:
                seen.add(dims)
                orientations.append(dims)
        return orientations


# ─────────────────────────────────────────────────────────────────────────────
# Placement (validated, immutable result of a box placement)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Placement:
    """
    A single validated box placement inside a bin.

    Frozen (immutable) so it can be safely shared between the simulator
    and strategies without risk of accidental mutation.

    Attributes:
        box_id:          ID of the placed box.
        x, y, z:         Position of the deepest-bottom-left corner.
        oriented_l/w/h:  Dimensions after rotation.
        orientation_idx: Which orientation was applied.
        step:            Sequential step number in the episode.
    """
    box_id: int
    x: float
    y: float
    z: float
    oriented_l: float
    oriented_w: float
    oriented_h: float
    orientation_idx: int
    step: int

    @property
    def volume(self) -> float:
        return self.oriented_l * self.oriented_w * self.oriented_h

    @property
    def x_max(self) -> float:
        return self.x + self.oriented_l

    @property
    def y_max(self) -> float:
        return self.y + self.oriented_w

    @property
    def z_max(self) -> float:
        return self.z + self.oriented_h

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "box_id": self.box_id,
            "dims": [self.oriented_l, self.oriented_w, self.oriented_h],
            "position": [self.x, self.y, self.z],
            "orientation": self.orientation_idx,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Placement":
        return cls(
            box_id=d["box_id"], x=d["position"][0], y=d["position"][1],
            z=d["position"][2], oriented_l=d["dims"][0], oriented_w=d["dims"][1],
            oriented_h=d["dims"][2], orientation_idx=d["orientation"], step=d["step"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# PlacementDecision (strategy output — before simulator validation)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlacementDecision:
    """
    A strategy's proposed placement (not yet validated by the simulator).

    The simulator will compute z automatically from the heightmap and
    validate all physical constraints before accepting.
    """
    x: float
    y: float
    orientation_idx: int


# ─────────────────────────────────────────────────────────────────────────────
# Bin Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BinConfig:
    """
    Physical dimensions and grid resolution of the stacking bin/pallet.

    Attributes:
        length:     X-axis dimension (mm).
        width:      Y-axis dimension (mm).
        height:     Z-axis dimension (mm).
        resolution: Grid cell size (mm per cell).
        margin:     Minimum gap between boxes and between boxes and walls (mm).
                    Enforced at all times by the simulator and all strategies.
    """
    length: float = 1200.0
    width: float = 800.0
    height: float = 2700.0
    resolution: float = 10.0
    margin: float = 20.0

    @property
    def grid_l(self) -> int:
        """Number of grid cells along the x-axis."""
        return math.ceil(self.length / self.resolution)

    @property
    def grid_w(self) -> int:
        """Number of grid cells along the y-axis."""
        return math.ceil(self.width / self.resolution)

    @property
    def volume(self) -> float:
        """Total bin volume."""
        return self.length * self.width * self.height

    def to_dict(self) -> dict:
        return {"length": self.length, "width": self.width,
                "height": self.height, "resolution": self.resolution,
                "margin": self.margin}

    @classmethod
    def from_dict(cls, d: dict) -> "BinConfig":
        return cls(**d)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """
    All tuneable parameters for a single experiment run.

    This is the only mutable config object — the runner sets dataset_path
    after generation.
    """
    bin: BinConfig = field(default_factory=BinConfig)
    strategy_name: str = "baseline"
    dataset_path: str = ""

    # Stability
    enable_stability: bool = False
    min_support_ratio: float = 0.8

    # Orientation
    allow_all_orientations: bool = False

    # Output
    render_3d: bool = True
    verbose: bool = False

    def to_dict(self) -> dict:
        return {
            "bin": self.bin.to_dict(),
            "strategy_name": self.strategy_name,
            "dataset_path": self.dataset_path,
            "enable_stability": self.enable_stability,
            "min_support_ratio": self.min_support_ratio,
            "allow_all_orientations": self.allow_all_orientations,
            "render_3d": self.render_3d,
            "verbose": self.verbose,
        }
