"""
Custom close policy: Close fullest pallet when neither can accept boxes.

This policy implements the logic:
"If 4 consecutive boxes must be declined because neither pallet can fit them,
 close the pallet with the highest fill factor"
"""

from simulator.close_policy import ClosePolicy
from simulator.bin_state import BinState


class FullestOnConsecutiveRejectsPolicy(ClosePolicy):
    """
    Close the fullest pallet after N consecutive GLOBAL rejections.

    Logic:
      - Track when boxes are rejected because NEITHER pallet can fit them
      - After N such consecutive global rejections, trigger pallet closure
      - When should_close() is called, check if THIS pallet has highest fill
      - Close only the fullest pallet (other pallets stay open)

    The 'consecutive_idle' counter in pallet_stats tracks how many steps
    this pallet was NOT used. When all pallets are idle (global rejection),
    all pallets will have high consecutive_idle counts.

    We close the pallet if:
      1. consecutive_idle >= max_consecutive (global rejections occurred)
      2. fill_rate >= min_fill_to_close (pallet is reasonably full)
      3. This ensures fuller pallets close first

    Args:
        max_consecutive: Close after this many consecutive global rejections
        min_fill_to_close: Only close if pallet fill >= this threshold (0.5 = 50%)
    """

    name = "fullest_on_rejects"

    def __init__(self, max_consecutive: int = 4, min_fill_to_close: float = 0.5) -> None:
        self.max_consecutive = max_consecutive
        self.min_fill_to_close = min_fill_to_close

    def should_close(self, bin_state: BinState, pallet_stats: dict) -> bool:
        """
        Close if:
        1. global_consecutive_rejects >= max_consecutive (neither pallet can fit N boxes in a row)
        2. AND fill_rate >= min_fill_to_close (pallet is reasonably full)

        When both pallets trigger (both meet fill threshold after N global rejections),
        the session will close the fuller one (handled in session.py line 732).
        """
        global_rejects = pallet_stats.get("global_consecutive_rejects", 0)
        fill_rate = bin_state.get_fill_rate()

        # Close if N consecutive boxes couldn't fit anywhere AND this pallet is reasonably full
        if global_rejects >= self.max_consecutive and fill_rate >= self.min_fill_to_close:
            return True

        return False

    def describe(self) -> str:
        return f"close fullest after {self.max_consecutive} global rejects (min fill: {self.min_fill_to_close:.0%})"

    def __repr__(self) -> str:
        return f"FullestOnConsecutiveRejectsPolicy(max_consecutive={self.max_consecutive}, min_fill_to_close={self.min_fill_to_close})"


class SessionAwareFullestPolicy(ClosePolicy):
    """
    Advanced version: Closes the ACTUAL fullest pallet when all are stuck.

    This requires session-level coordination. The session will need to:
    1. Track fill rates of all active pallets
    2. When checking close policy, pass additional context
    3. Only close the pallet with max(fill_rate) when all are stuck

    For now, this is a placeholder for future enhancement.
    Use FullestOnConsecutiveRejectsPolicy instead.
    """

    name = "session_aware_fullest"

    def __init__(self, max_consecutive: int = 4) -> None:
        self.max_consecutive = max_consecutive
        self._all_stations_fill_rates = []  # Populated by session

    def should_close(self, bin_state: BinState, pallet_stats: dict) -> bool:
        # This would need session modification to work properly
        # For now, use FullestOnConsecutiveRejectsPolicy
        raise NotImplementedError(
            "SessionAwareFullestPolicy requires session-level coordination. "
            "Use FullestOnConsecutiveRejectsPolicy instead."
        )

    def describe(self) -> str:
        return f"session-aware fullest after {self.max_consecutive} rejects"
