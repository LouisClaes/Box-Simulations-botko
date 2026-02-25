"""
FIFO conveyor belt — models a physical conveyor with a pick window.

The conveyor is a strict FIFO queue:
  - Boxes enter from the stream at the BACK of the belt.
  - The robot can reach the first ``pick_window`` boxes at the FRONT.
  - When the robot picks a box, all boxes behind it shift forward and a
    new box enters from the stream at the back.
  - When no box can be placed, the belt ADVANCES: the front box exits
    the system permanently (reject/overflow) and a new box enters.

There is NO recirculation.  A box that passes the robot is gone.

Index layout:
    visible[0]  = FRONT of belt  (first to exit)
    visible[-1] = BACK of belt   (most recently arrived)

::

    stream → [new box] ─────────────────────→ [front box] → robot / reject
             BACK of belt   →   belt moves   →   FRONT of belt
             visible[-1]         ← shift ←        visible[0]

                              |<-- pick_window -->|
                              (robot can reach these)

Example:
    >>> from simulator.conveyor import FIFOConveyor
    >>> from config import Box
    >>> boxes = [Box(id=i, length=100, width=100, height=100) for i in range(20)]
    >>> conv = FIFOConveyor(boxes, buffer_size=8, pick_window=4)
    >>> conv.grippable          # first 4 boxes
    >>> conv.pick(box_id=2)     # robot picks box 2, belt shifts
    >>> conv.advance()          # no box fits → front box exits
"""

from typing import List, Optional

from config import Box


class FIFOConveyor:
    """
    FIFO conveyor belt with a pick window.

    Models the Botko BV (and similar) robotic palletizers where boxes
    arrive on a belt and the robot can only reach the first N boxes.

    Args:
        boxes:       Full stream of boxes to process (in arrival order).
        buffer_size: Total visible boxes on the belt at any time.
        pick_window: How many front boxes the robot can reach.

    Attributes:
        visible:   Boxes currently on the belt (front to back).
        rejected:  Boxes that exited the belt without being placed.
    """

    __slots__ = (
        "buffer_size", "pick_window", "visible", "rejected",
        "_stream", "_total_loaded",
    )

    def __init__(
        self,
        boxes: List[Box],
        buffer_size: int = 8,
        pick_window: int = 4,
    ) -> None:
        self.buffer_size = buffer_size
        self.pick_window = min(pick_window, buffer_size)
        self._stream: List[Box] = list(boxes)
        self._total_loaded: int = len(boxes)
        self.visible: List[Box] = []
        self.rejected: List[Box] = []
        self._refill()

    # ── Stream / capacity queries ─────────────────────────────────────────

    @property
    def grippable(self) -> List[Box]:
        """The first ``pick_window`` boxes the robot can reach."""
        return self.visible[: self.pick_window]

    @property
    def is_empty(self) -> bool:
        """True when belt is empty AND stream is exhausted."""
        return len(self.visible) == 0

    @property
    def stream_remaining(self) -> int:
        """Boxes still in the stream (not yet on the belt)."""
        return len(self._stream)

    @property
    def total_remaining(self) -> int:
        """Unprocessed boxes: on belt + in stream."""
        return len(self.visible) + len(self._stream)

    @property
    def total_loaded(self) -> int:
        """Total boxes loaded into the conveyor at construction."""
        return self._total_loaded

    @property
    def total_rejected(self) -> int:
        """Total boxes that exited without placement."""
        return len(self.rejected)

    # ── Robot actions ─────────────────────────────────────────────────────

    def pick(self, box_id: int) -> Optional[Box]:
        """
        Robot picks a box from the grippable window by ID.

        The picked box is removed.  All boxes behind it shift forward
        (Python ``list.pop`` does this automatically).  A new box enters
        from the stream at the back.

        Args:
            box_id: The ``Box.id`` to pick.

        Returns:
            The picked Box, or None if the box is not in the pick window.
        """
        for i, b in enumerate(self.visible[: self.pick_window]):
            if b.id == box_id:
                picked = self.visible.pop(i)
                self._refill()
                return picked
        return None

    def advance(self) -> Optional[Box]:
        """
        Belt advances — the front box exits the system permanently.

        Called when no grippable box can be placed.  The front box passes
        the robot and goes to the reject/overflow bin.  All remaining
        boxes shift forward.  A new box enters from the stream.

        Returns:
            The rejected box, or None if belt is empty.
        """
        if not self.visible:
            return None
        exited = self.visible.pop(0)
        self.rejected.append(exited)
        self._refill()
        return exited

    # ── Snapshot (for visualization / logging) ────────────────────────────

    def snapshot(self) -> List[Box]:
        """Return a shallow copy of the visible belt for recording."""
        return list(self.visible)

    # ── Internal ──────────────────────────────────────────────────────────

    def _refill(self) -> None:
        """Top up the belt from the stream — new boxes enter at the back."""
        while len(self.visible) < self.buffer_size and self._stream:
            self.visible.append(self._stream.pop(0))

    def __repr__(self) -> str:
        return (
            f"FIFOConveyor(visible={len(self.visible)}/{self.buffer_size}, "
            f"stream={self.stream_remaining}, rejected={self.total_rejected})"
        )
