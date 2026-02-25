"""
Placement validator — pure-function physical constraint checking.

All checks are stateless functions: they take the current heightmap,
placed boxes, and a proposed placement, returning True or raising an error.

Checks (always enforced):
  0. Margin    — box must be ≥ BinConfig.margin from walls (default 20 mm)
  1. Bounds    — box must fit inside the bin on all axes
  2. Overlap   — box z must match surface height (no clipping)
  3. Floating  — ≥30% of base must be supported (prevents floating)
  5. Box-gap   — ≥ BinConfig.margin gap from every z-overlapping placed box

Optional checks:
  4. Stability — base support ratio ≥ configurable threshold (default 80%)
"""

import numpy as np
from typing import Optional, List

from config import BinConfig, Placement


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────

class PlacementError(Exception):
    """Base class for placement validation errors."""


class OutOfBoundsError(PlacementError):
    """Box extends outside the bin boundary."""


class MarginViolationError(PlacementError):
    """Box is closer than the required margin to a wall or another box."""


class OverlapError(PlacementError):
    """Box would clip into an already-placed box."""


class FloatingError(PlacementError):
    """Box would float in mid-air (insufficient base support)."""


class UnstablePlacementError(PlacementError):
    """Box does not have sufficient base support for stability."""


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Minimum fraction of base that MUST touch a surface.
# Always enforced regardless of enable_stability, to prevent floating.
# A box on the floor always has ratio=1.0 so this only affects stacking.
MIN_ANTI_FLOAT_RATIO = 0.30


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────

def validate_placement(
    heightmap: np.ndarray,
    bin_config: BinConfig,
    x: float,
    y: float,
    z: float,
    oriented_l: float,
    oriented_w: float,
    oriented_h: float,
    enable_stability: bool = False,
    min_support_ratio: float = 0.8,
    placed_boxes: Optional[List[Placement]] = None,
) -> bool:
    """
    Validate a proposed placement against all physical constraints.

    Args:
        heightmap:         Current 2D height grid (grid_l × grid_w).
        bin_config:        Bin dimensions and resolution.
        x, y, z:           Real-world corner position.
        oriented_l/w/h:    Oriented box dimensions.
        enable_stability:  Whether to check strict support ratio.
        min_support_ratio: Minimum supported base fraction (stability mode).
        placed_boxes:      All currently placed boxes (for box-gap check).

    Returns:
        True if all checks pass.

    Raises:
        MarginViolationError:    box is closer than margin to wall or other box.
        OutOfBoundsError:        box extends outside bin.
        OverlapError:            box clips into another box.
        FloatingError:           <30% of base is supported (always checked).
        UnstablePlacementError:  support < min_support_ratio (stability mode).
    """
    eps = 1e-6
    res = bin_config.resolution
    margin = bin_config.margin

    # ── 0. Wall-margin check ─────────────────────────────────────────────
    if margin > 0:
        if x < margin - eps:
            raise MarginViolationError(
                f"Too close to X=0 wall: x={x:.1f} < margin={margin:.1f}"
            )
        if y < margin - eps:
            raise MarginViolationError(
                f"Too close to Y=0 wall: y={y:.1f} < margin={margin:.1f}"
            )
        if x + oriented_l > bin_config.length - margin + eps:
            raise MarginViolationError(
                f"Too close to X={bin_config.length} wall: "
                f"x+l={x + oriented_l:.1f} > {bin_config.length - margin:.1f}"
            )
        if y + oriented_w > bin_config.width - margin + eps:
            raise MarginViolationError(
                f"Too close to Y={bin_config.width} wall: "
                f"y+w={y + oriented_w:.1f} > {bin_config.width - margin:.1f}"
            )

    # ── 1. Bounds ────────────────────────────────────────────────────────
    if x < -eps or y < -eps or z < -eps:
        raise OutOfBoundsError(f"Negative coordinate: ({x:.1f}, {y:.1f}, {z:.1f})")
    if x + oriented_l > bin_config.length + eps:
        raise OutOfBoundsError(f"X overflow: {x:.1f}+{oriented_l:.1f} > {bin_config.length:.1f}")
    if y + oriented_w > bin_config.width + eps:
        raise OutOfBoundsError(f"Y overflow: {y:.1f}+{oriented_w:.1f} > {bin_config.width:.1f}")
    if z + oriented_h > bin_config.height + eps:
        raise OutOfBoundsError(f"Z overflow: {z:.1f}+{oriented_h:.1f} > {bin_config.height:.1f}")

    # ── Grid region ──────────────────────────────────────────────────────
    gx     = int(round(x / res))
    gy     = int(round(y / res))
    gx_end = min(gx + int(round(oriented_l / res)), bin_config.grid_l)
    gy_end = min(gy + int(round(oriented_w / res)), bin_config.grid_w)

    if gx >= gx_end or gy >= gy_end:
        raise OutOfBoundsError(f"Zero-area footprint: gx={gx}-{gx_end}, gy={gy}-{gy_end}")

    region = heightmap[gx:gx_end, gy:gy_end]
    tolerance = res * 0.5

    # ── 2. Overlap check ─────────────────────────────────────────────────
    max_height = float(np.max(region))
    if z < max_height - tolerance:
        raise OverlapError(
            f"Box z={z:.2f} but surface peak={max_height:.2f} — "
            f"box would clip into existing box (Δ={max_height - z:.2f})"
        )

    # ── 3. Anti-float check (ALWAYS enforced) ────────────────────────────
    if z > tolerance:
        supported_cells = int(np.sum(np.abs(region - z) <= tolerance))
        support_ratio = supported_cells / region.size

        if support_ratio < MIN_ANTI_FLOAT_RATIO:
            raise FloatingError(
                f"Only {support_ratio:.0%} of base is supported at z={z:.1f} "
                f"(need ≥{MIN_ANTI_FLOAT_RATIO:.0%} to prevent floating)"
            )

    # ── 4. Stability (optional, stricter) ────────────────────────────────
    if enable_stability and z > tolerance:
        supported_cells = int(np.sum(np.abs(region - z) <= tolerance))
        support_ratio = supported_cells / region.size

        if support_ratio < min_support_ratio:
            raise UnstablePlacementError(
                f"Support {support_ratio:.0%} < required {min_support_ratio:.0%}"
            )

    # ── 5. Box-to-box gap check ───────────────────────────────────────────
    if margin > 0 and placed_boxes:
        new_z_max = z + oriented_h
        for p in placed_boxes:
            # Skip boxes with no z-range overlap (fully below or fully above).
            # Stacked boxes: p.z_max == z (B rests exactly on A) → skip.
            if p.z_max <= z + eps or p.z >= new_z_max - eps:
                continue
            # Boxes share a z range — check xy proximity.
            # Violation when expanded footprints intersect (gap < margin).
            if (x < p.x_max + margin - eps and
                    x + oriented_l > p.x - margin + eps and
                    y < p.y_max + margin - eps and
                    y + oriented_w > p.y - margin + eps):
                raise MarginViolationError(
                    f"Box too close to placed box {p.box_id}: "
                    f"xy gap < required {margin:.1f} mm margin"
                )

    return True
