"""
Pallet close policies — decide when a pallet is "done" and should be sealed.

Close policies are pluggable objects that the PackingSession consults after
every successful placement.  When a policy says "close", the pallet is
snapshotted, its stats are saved, and a fresh empty pallet takes its place.

Built-in policies:
    HeightClosePolicy       — close when max_height >= threshold
    RejectClosePolicy       — close after N consecutive rejects on this pallet
    CombinedClosePolicy     — close when ANY child policy triggers
    NeverClosePolicy        — never close (single-bin / infinite pallet)

Creating a custom policy:
    1. Subclass ``ClosePolicy``
    2. Implement ``should_close(bin_state, pallet_stats) -> bool``
    3. Pass your instance to ``SessionConfig(close_policy=MyPolicy())``

The ``pallet_stats`` dict provides counters tracked per pallet station:
    - consecutive_idle : int   — consecutive steps where this pallet was NOT used
    - boxes_placed     : int   — total boxes on this pallet
    - total_rejects    : int   — total times this pallet was tried and rejected

Example:
    >>> from simulator.close_policy import HeightClosePolicy, CombinedClosePolicy
    >>> policy = CombinedClosePolicy([
    ...     HeightClosePolicy(max_height=1800.0),
    ...     RejectClosePolicy(max_consecutive=5),
    ... ])
    >>> policy.should_close(bin_state, pallet_stats)
    True
"""

from abc import ABC, abstractmethod
from typing import List

from simulator.bin_state import BinState


class ClosePolicy(ABC):
    """
    Abstract base for pallet close decisions.

    Subclass this and implement ``should_close()`` to define when a pallet
    is considered complete and should be replaced with an empty one.

    The PackingSession calls ``should_close()`` after every successful
    placement on a pallet.  If it returns True, the pallet is closed
    immediately — its stats are snapshotted and it is replaced.

    Attributes:
        name: Human-readable label for logging and JSON output.
    """

    name: str = "base"

    @abstractmethod
    def should_close(self, bin_state: BinState, pallet_stats: dict) -> bool:
        """
        Decide whether the pallet should be closed (sealed, shipped).

        Args:
            bin_state:    Current 3D state of the pallet — heightmap,
                          placed_boxes, fill_rate, max_height, etc.
            pallet_stats: Per-pallet counters maintained by PackingSession:
                          - ``consecutive_idle`` (int): steps since last
                            placement on THIS pallet
                          - ``boxes_placed`` (int): boxes on this pallet
                          - ``total_rejects`` (int): failed placement
                            attempts on this pallet

        Returns:
            True if the pallet should be closed now.
        """
        ...

    def describe(self) -> str:
        """One-line description for JSON metadata."""
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ─────────────────────────────────────────────────────────────────────────────
# Built-in policies
# ─────────────────────────────────────────────────────────────────────────────

class HeightClosePolicy(ClosePolicy):
    """
    Close the pallet when its tallest point reaches a height threshold.

    This is the standard policy for EU road transport pallets (1800 mm).

    Args:
        max_height: Height in mm at which the pallet is closed.
    """

    name = "height"

    def __init__(self, max_height: float = 1800.0) -> None:
        self.max_height = max_height

    def should_close(self, bin_state: BinState, pallet_stats: dict) -> bool:
        return bin_state.get_max_height() >= self.max_height

    def describe(self) -> str:
        return f"height>={self.max_height:.0f}mm"

    def __repr__(self) -> str:
        return f"HeightClosePolicy(max_height={self.max_height})"


class RejectClosePolicy(ClosePolicy):
    """
    Close the pallet after N consecutive idle steps (steps where this
    pallet was not used for a placement).

    Useful for detecting "stuck" pallets that no box can fit into.

    Args:
        max_consecutive: Close after this many consecutive idle steps.
    """

    name = "reject"

    def __init__(self, max_consecutive: int = 5) -> None:
        self.max_consecutive = max_consecutive

    def should_close(self, bin_state: BinState, pallet_stats: dict) -> bool:
        return pallet_stats.get("consecutive_idle", 0) >= self.max_consecutive

    def describe(self) -> str:
        return f"consecutive_idle>={self.max_consecutive}"

    def __repr__(self) -> str:
        return f"RejectClosePolicy(max_consecutive={self.max_consecutive})"


class FillClosePolicy(ClosePolicy):
    """
    Close the pallet when its volumetric fill rate exceeds a threshold.

    Args:
        min_fill: Fill rate (0.0–1.0) at which the pallet is closed.
    """

    name = "fill"

    def __init__(self, min_fill: float = 0.85) -> None:
        self.min_fill = min_fill

    def should_close(self, bin_state: BinState, pallet_stats: dict) -> bool:
        return bin_state.get_fill_rate() >= self.min_fill

    def describe(self) -> str:
        return f"fill>={self.min_fill:.0%}"

    def __repr__(self) -> str:
        return f"FillClosePolicy(min_fill={self.min_fill})"


class CombinedClosePolicy(ClosePolicy):
    """
    Close the pallet when ANY child policy triggers (logical OR).

    Args:
        policies: List of ClosePolicy instances to combine.
    """

    name = "combined"

    def __init__(self, policies: List[ClosePolicy]) -> None:
        if not policies:
            raise ValueError("CombinedClosePolicy requires at least one child policy")
        self.policies = list(policies)

    def should_close(self, bin_state: BinState, pallet_stats: dict) -> bool:
        return any(p.should_close(bin_state, pallet_stats) for p in self.policies)

    def describe(self) -> str:
        return " OR ".join(p.describe() for p in self.policies)

    def __repr__(self) -> str:
        return f"CombinedClosePolicy({self.policies!r})"


class NeverClosePolicy(ClosePolicy):
    """
    Never close the pallet.  Useful for single-bin experiments where
    pallet replacement is not desired.
    """

    name = "never"

    def should_close(self, bin_state: BinState, pallet_stats: dict) -> bool:
        return False

    def describe(self) -> str:
        return "never"
