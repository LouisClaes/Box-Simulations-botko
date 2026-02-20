"""
Box buffer -- manages the lookahead box buffer for semi-online packing.

The buffer sits between the input stream (conveyor) and the placement
strategy/orchestrator.  It holds up to K boxes at a time, allowing
the system to look ahead and choose the best box (and bin) to place next.

This is the 2K-bound mechanism: with K boxes buffered and 2 active bins,
the system has 2K degrees of freedom per decision step.

Buffer selection policies:
    FIFO            -- strict arrival order (no buffer effect, baseline)
    LARGEST_FIRST   -- pick largest volume box (best for gap-filling later)
    SMALLEST_FIRST  -- pick smallest volume box
    TALLEST_FIRST   -- pick tallest box (build tall structures early)
    FLATTEST_FIRST  -- pick flattest box (build solid base first)
    BEST_FIT_SCORE  -- try all boxes against all bins, pick best pair

Usage:
    from simulator.buffer import BoxBuffer, BufferPolicy

    buf = BoxBuffer(capacity=7, policy=BufferPolicy.LARGEST_FIRST)
    buf.load_stream(boxes)
    buf.refill()

    box = buf.pick_next()            # select next box by policy
    buf.return_box(box)              # put back after failed placement
"""

from enum import Enum
from typing import List, Optional, Tuple, Callable

from config import Box


# ---------------------------------------------------------------------------
# Buffer selection policy
# ---------------------------------------------------------------------------

class BufferPolicy(Enum):
    """Which box to pick from the buffer next."""

    FIFO = "fifo"
    """First-In First-Out -- place boxes in arrival order."""

    LARGEST_FIRST = "largest_first"
    """Place the largest volume box first."""

    SMALLEST_FIRST = "smallest_first"
    """Place the smallest box first."""

    BEST_FIT_SCORE = "best_fit_score"
    """Try every box in the buffer against both bins, pick the best pair."""

    TALLEST_FIRST = "tallest_first"
    """Place the tallest box first."""

    FLATTEST_FIRST = "flattest_first"
    """Place the flattest (smallest height) box first."""


# ---------------------------------------------------------------------------
# Box buffer
# ---------------------------------------------------------------------------

class BoxBuffer:
    """
    Fixed-size box buffer with configurable selection policy.

    The buffer acts as a sliding window over the input stream.  At each
    decision step, the orchestrator/pipeline calls pick_next() to get
    the best box from the buffer, then calls refill() to top up from
    the stream.

    This implements the K-bound in the 2K-bound multi-bin algorithm:
    K buffered boxes x 2 active bins = 2K degrees of freedom per step.

    Attributes:
        capacity:   Maximum buffer size (K in K-bound).
        policy:     Selection policy enum.
        buffer:     Current boxes in the buffer.
    """

    __slots__ = ("capacity", "policy", "buffer", "_input_stream", "_stream_idx")

    def __init__(self, capacity: int, policy: BufferPolicy) -> None:
        self.capacity = max(1, capacity)
        self.policy = policy
        self.buffer: List[Box] = []
        self._input_stream: List[Box] = []
        self._stream_idx: int = 0

    def load_stream(self, boxes: List[Box]) -> None:
        """Load the full input stream.  Call once before the episode starts."""
        self._input_stream = list(boxes)
        self._stream_idx = 0
        self.buffer = []

    def refill(self) -> None:
        """Top up the buffer from the input stream until full or stream empty."""
        while len(self.buffer) < self.capacity and self._stream_idx < len(self._input_stream):
            self.buffer.append(self._input_stream[self._stream_idx])
            self._stream_idx += 1

    @property
    def is_empty(self) -> bool:
        """True if buffer is empty AND no more boxes in the stream."""
        return len(self.buffer) == 0 and self._stream_idx >= len(self._input_stream)

    @property
    def stream_remaining(self) -> int:
        """Boxes still in the input stream (not yet in buffer)."""
        return max(0, len(self._input_stream) - self._stream_idx)

    @property
    def total_remaining(self) -> int:
        """Total unprocessed boxes (buffer + stream)."""
        return len(self.buffer) + self.stream_remaining

    # -- Selection methods -------------------------------------------------

    def pick_next(self) -> Optional[Box]:
        """
        Select and remove the next box from the buffer using the policy.

        Returns:
            The selected Box, or None if the buffer is empty.
        """
        if not self.buffer:
            return None

        if self.policy == BufferPolicy.FIFO:
            return self._pick_fifo()
        elif self.policy == BufferPolicy.LARGEST_FIRST:
            return self._pick_by_key(key=lambda b: -b.volume)
        elif self.policy == BufferPolicy.SMALLEST_FIRST:
            return self._pick_by_key(key=lambda b: b.volume)
        elif self.policy == BufferPolicy.TALLEST_FIRST:
            return self._pick_by_key(key=lambda b: -b.height)
        elif self.policy == BufferPolicy.FLATTEST_FIRST:
            return self._pick_by_key(key=lambda b: b.height)
        elif self.policy == BufferPolicy.BEST_FIT_SCORE:
            return self._pick_fifo()
        else:
            return self._pick_fifo()

    def pick_best_fit(
        self,
        score_fn: Callable[[Box], Tuple[float, int]],
    ) -> Optional[Tuple[Box, int]]:
        """
        Select the box that maximizes score_fn(box) -> (score, bin_index).

        The scoring function is provided by the orchestrator/pipeline and
        evaluates each buffer box against all active bins.

        Args:
            score_fn: Function that takes a Box and returns (score, bin_idx).
                      Higher score = better fit.  bin_idx = which bin to use.

        Returns:
            (selected_box, best_bin_index) or None if buffer is empty.
        """
        if not self.buffer:
            return None

        best_score = -1e18
        best_idx = 0
        best_bin = 0

        for i, box in enumerate(self.buffer):
            score, bin_idx = score_fn(box)
            if score > best_score:
                best_score = score
                best_idx = i
                best_bin = bin_idx

        selected = self.buffer.pop(best_idx)
        return (selected, best_bin)

    def return_box(self, box: Box) -> None:
        """
        Return a box to the front of the buffer (after failed placement).

        This happens when no bin can accept the box -- it goes back into
        the buffer and a different box is tried instead.
        """
        self.buffer.insert(0, box)

    def peek(self) -> List[Box]:
        """Return a read-only view of the current buffer contents."""
        return list(self.buffer)

    # -- Private helpers ---------------------------------------------------

    def _pick_fifo(self) -> Box:
        """Remove and return the first (oldest) box."""
        return self.buffer.pop(0)

    def _pick_by_key(self, key: Callable[[Box], float]) -> Box:
        """Remove and return the box with the minimum key value."""
        best_idx = min(range(len(self.buffer)), key=lambda i: key(self.buffer[i]))
        return self.buffer.pop(best_idx)


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

# BufferManager is the old name -- keep as alias so existing code still works.
BufferManager = BoxBuffer
