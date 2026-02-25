"""Simple first-fit bin packing algorithm."""

from src.core.models import Box, Pallet


class SimplePacker:
    """
    Simple first-fit packing algorithm.

    Tries to add each box to the current pallet.
    If it doesn't fit, creates a new pallet.
    """

    def __init__(self):
        self.pallets: list[Pallet] = []
        self._next_pallet_id = 0

    def pack(self, boxes: list[Box]) -> list[Pallet]:
        """
        Pack boxes into pallets using first-fit strategy.

        Args:
            boxes: List of boxes to pack

        Returns:
            List of pallets (including CLOSED pallets only)
        """
        self.pallets = []
        self._next_pallet_id = 0

        current_pallet = self._new_pallet()

        for box in boxes:
            # Try to add to current pallet
            if not current_pallet.add_box(box):
                # Current pallet full, close it and create new one
                current_pallet.close()
                current_pallet = self._new_pallet()

                # Add to new pallet
                if not current_pallet.add_box(box):
                    # Box is too large/heavy even for empty pallet
                    # For now, skip it (in real scenario, might need special handling)
                    pass

        # Close the last pallet if it has boxes
        if current_pallet.boxes:
            current_pallet.close()

        # Return only closed pallets
        return [p for p in self.pallets if p.is_closed]

    def _new_pallet(self) -> Pallet:
        """Create a new pallet and add it to the list."""
        pallet = Pallet(pallet_id=self._next_pallet_id)
        self._next_pallet_id += 1
        self.pallets.append(pallet)
        return pallet
