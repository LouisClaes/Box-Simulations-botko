"""
Two-Bounded Best-Fit dual-bin strategy (MultiBinStrategy).

References:
    Zhao, H., She, Q., Zhu, C., Yang, Y., & Xu, K. (2021).
    "Online 3D Bin Packing with Constrained Deep Reinforcement Learning."
    Proceedings of the AAAI Conference on Artificial Intelligence, 35(6), 741-749.
    arXiv:2012.04412
    GitHub: https://github.com/alexfrom0815/Online-3D-BPP-DRL

    Tsang, C.W., Tsang, E.C.C., & Wang, X. (2025).
    "A deep reinforcement learning approach for online and concurrent 3D bin
    packing optimisation with bin replacement strategies."
    Computers in Industry, Vol. 164, Article 104202.
    GitHub: https://github.com/SoftwareImpacts/SIMPAC-2024-311

Algorithm:
    This is a MultiBinStrategy that natively manages two bins simultaneously.
    Unlike the orchestrator wrapper (which applies a single-bin strategy to
    one bin at a time), this strategy receives the full state of ALL active
    bins and decides:
        1. WHICH bin to place the box in.
        2. WHERE within that bin to place it.

    The decision is made using a Best-Fit criterion:

        For each active bin:
            Find the best placement position using surface-contact scoring:
                score = support_ratio * WEIGHT_SUPPORT
                        - height_norm * WEIGHT_HEIGHT
                        + contact_base_ratio * WEIGHT_CONTACT

            Apply a Best-Fit bin preference bonus:
                fill_bonus = bin_state.get_fill_rate() * FILL_BONUS_WEIGHT

            Total score = in-bin score + fill_bonus

        Select the (bin, position, orientation) with the highest total score.

    The Best-Fit principle (prefer the fuller bin) follows Zhao et al. 2021
    and Tsang et al. 2025: routing items to the bin that already has the most
    content tends to complete bins faster and reduces the number of bins used.

In-bin scoring (surface contact):
    The per-bin placement score uses three terms that collectively reward
    compact, stable packing:

        support_ratio:       Fraction of the box base that is supported by
                             the floor or other boxes.  High support = stable.

        height_norm:         z / bin_height.  Penalises high placements so
                             boxes are packed low first.

        contact_base_ratio:  Fraction of footprint grid cells at exactly z
                             (the resting height), i.e., how flat the surface
                             is beneath the box.  A high ratio means the box
                             is resting on a flat, contiguous surface rather
                             than precariously balanced on edges.

Stability:
    MIN_SUPPORT = 0.30 is always enforced for every candidate in every bin.
    When ExperimentConfig.enable_stability is True, the stricter
    cfg.min_support_ratio is also enforced.
"""

import numpy as np
from typing import Optional, List, Tuple, Set

from config import Box, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import MultiBinStrategy, MultiBinDecision, register_multibin_strategy


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Anti-float threshold -- must match the simulator's MIN_ANTI_FLOAT_RATIO.
MIN_SUPPORT: float = 0.30

# Floor-level threshold: at or below this height, support_ratio = 1.0.
FLOOR_Z_THRESHOLD: float = 0.5

# In-bin scoring weights
WEIGHT_SUPPORT: float = 5.0   # Reward well-supported placements
WEIGHT_HEIGHT: float = 2.0    # Penalise high placements
WEIGHT_CONTACT: float = 3.0   # Reward flat base contact

# Best-Fit bin preference weight: bonus proportional to current fill rate.
# Fuller bins score higher, so items are routed to the bin being filled.
FILL_BONUS_WEIGHT: float = 5.0

# Height tolerance for contact_base_ratio calculation: cells within this
# tolerance of z are counted as "at floor level" for the box.
CONTACT_HEIGHT_TOLERANCE: float = 0.5


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

@register_multibin_strategy
class TwoBoundedBestFitStrategy(MultiBinStrategy):
    """
    Best-fit dual-bin strategy with native cross-bin optimisation.

    At each step the strategy receives the full state of all active bins and
    finds the globally best (bin, x, y, orientation) combination using
    surface-contact scoring with a Best-Fit bin preference bonus.

    The Best-Fit principle (prefer the fuller bin) routes items to bins that
    are closer to completion, which tends to:
        - Complete individual bins faster (higher per-bin fill rates).
        - Reduce total bins used for a fixed item stream.
        - Mirror the DRL-learned behaviour from Zhao et al. 2021, where the
          agent learns to pack one bin densely before opening a second.

    In-bin placement scoring uses three complementary components:
        - support_ratio:      Stability (well-supported base).
        - height_norm:        Vertical efficiency (pack low first).
        - contact_base_ratio: Surface flatness (flat contact surface).

    References:
        Zhao et al. (2021). "Online 3D Bin Packing with Constrained Deep
        Reinforcement Learning." AAAI 2021. arXiv:2012.04412.
        GitHub: https://github.com/alexfrom0815/Online-3D-BPP-DRL

        Tsang et al. (2025). "A deep reinforcement learning approach for
        online and concurrent 3D bin packing optimisation with bin replacement
        strategies." Computers in Industry, Vol. 164, Article 104202.
        GitHub: https://github.com/SoftwareImpacts/SIMPAC-2024-311

    Attributes:
        name: Strategy identifier for the multi-bin registry
              ("two_bounded_best_fit").
    """

    name: str = "two_bounded_best_fit"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = 1.0

    def on_episode_start(self, config) -> None:
        """Store config and derive grid scan step from bin resolution."""
        super().on_episode_start(config)
        # Support both ExperimentConfig (.bin) and PipelineConfig (.bin_config)
        bin_cfg = getattr(config, "bin", None) or getattr(config, "bin_config", None)
        self._scan_step = max(1.0, bin_cfg.resolution) if bin_cfg else 1.0

    # -- Main entry point ---------------------------------------------------

    def decide_placement(
        self,
        box: Box,
        bin_states: List[BinState],
    ) -> Optional[MultiBinDecision]:
        """
        Propose a placement across all active bins using Best-Fit scoring.

        For each bin in *bin_states*, finds the best valid placement for
        *box* using surface-contact scoring, then applies a fill-rate bonus
        to implement the Best-Fit bin selection criterion.  Returns the
        globally best (bin, x, y, orientation) or None if the box cannot
        be placed in any bin.

        Algorithm:
            for each bin_idx, bin_state in enumerate(bin_states):
                result = self._best_in_bin(box, bin_state)
                if result:
                    total_score = in_bin_score
                                  + bin_state.get_fill_rate() * FILL_BONUS_WEIGHT
                    track global best

        Args:
            box:        The box to place (original, un-rotated dimensions).
            bin_states: List of BinState for all active bins (read-only).

        Returns:
            MultiBinDecision(bin_index, x, y, orientation_idx) or None.
        """
        best_total_score: float = -np.inf
        best_decision: Optional[MultiBinDecision] = None

        for bin_idx, bin_state in enumerate(bin_states):
            result = self._best_in_bin(box, bin_state)
            if result is None:
                continue

            x, y, oidx, in_bin_score = result

            # Best-Fit bin preference: bonus for already-fuller bins
            fill_bonus = bin_state.get_fill_rate() * FILL_BONUS_WEIGHT
            total_score = in_bin_score + fill_bonus

            if total_score > best_total_score:
                best_total_score = total_score
                best_decision = MultiBinDecision(
                    bin_index=bin_idx,
                    x=x,
                    y=y,
                    orientation_idx=oidx,
                )

        return best_decision

    # -- Per-bin placement scoring ------------------------------------------

    def _best_in_bin(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[Tuple[float, float, int, float]]:
        """
        Find the best placement for *box* in a single bin.

        Scans all candidate positions (grid scan + placed-box corners) and
        all allowed orientations, evaluates each with the surface-contact
        score, and returns the best feasible candidate.

        Scoring formula:
            score = support_ratio * WEIGHT_SUPPORT
                    - height_norm * WEIGHT_HEIGHT
                    + contact_base_ratio * WEIGHT_CONTACT

        where:
            support_ratio:      Fraction of base cells that are supported at z.
            height_norm:        z / bin_height (normalised placement height).
            contact_base_ratio: Fraction of footprint cells where the heightmap
                                value equals z (within CONTACT_HEIGHT_TOLERANCE),
                                i.e., how uniformly flat the resting surface is.

        Args:
            box:       The box to place (original dimensions).
            bin_state: State of a single bin (read-only).

        Returns:
            (x, y, orientation_idx, score) tuple for the best feasible
            placement, or None if no valid placement exists.
        """
        cfg = self._config
        bin_cfg = bin_state.config
        step = self._scan_step

        # --- Resolve allowed orientations ---
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Pre-filter orientations that could never fit in this bin.
        valid_orientations: List[Tuple[int, float, float, float]] = [
            (oidx, ol, ow, oh)
            for oidx, (ol, ow, oh) in enumerate(orientations)
            if (ol <= bin_cfg.length + 1e-6
                and ow <= bin_cfg.width + 1e-6
                and oh <= bin_cfg.height + 1e-6)
        ]
        if not valid_orientations:
            return None

        # --- Generate candidates ---
        candidates = self._generate_candidates(bin_state, step)

        # --- Evaluate candidates ---
        best_score: float = -np.inf
        best: Optional[Tuple[float, float, int]] = None
        heightmap = bin_state.heightmap
        resolution = bin_cfg.resolution

        for cx, cy in candidates:
            for oidx, ol, ow, oh in valid_orientations:
                # Bin boundary checks
                if cx + ol > bin_cfg.length + 1e-6:
                    continue
                if cy + ow > bin_cfg.width + 1e-6:
                    continue

                # Resting z from the heightmap
                z = bin_state.get_height_at(cx, cy, ol, ow)

                # Height limit check
                if z + oh > bin_cfg.height + 1e-6:
                    continue

                # Anti-float support check
                support_ratio = 1.0
                if z > FLOOR_Z_THRESHOLD:
                    support_ratio = bin_state.get_support_ratio(
                        cx, cy, ol, ow, z
                    )
                    if support_ratio < MIN_SUPPORT:
                        continue
                    if cfg.enable_stability and support_ratio < cfg.min_support_ratio:
                        continue

                # Contact base ratio: fraction of footprint cells at exactly z
                contact_base_ratio = self._compute_contact_base_ratio(
                    cx, cy, z, ol, ow, heightmap, bin_cfg, resolution
                )

                # Composite score
                height_norm = z / bin_cfg.height if bin_cfg.height > 0.0 else 0.0
                score = (
                    WEIGHT_SUPPORT * support_ratio
                    - WEIGHT_HEIGHT * height_norm
                    + WEIGHT_CONTACT * contact_base_ratio
                )

                if score > best_score:
                    best_score = score
                    best = (cx, cy, oidx)

        if best is None:
            return None

        return (best[0], best[1], best[2], best_score)

    # -- Contact base ratio -------------------------------------------------

    @staticmethod
    def _compute_contact_base_ratio(
        x: float,
        y: float,
        z: float,
        ol: float,
        ow: float,
        heightmap: np.ndarray,
        bin_cfg,
        resolution: float,
    ) -> float:
        """
        Compute the fraction of the box footprint that rests on a flat surface.

        "Flat" means the heightmap value is within CONTACT_HEIGHT_TOLERANCE
        of z (the resting height of this box).  A high ratio indicates that
        the box has a broad, uniform base support -- better for packing quality
        and physical stability.

        For floor-level placements (z <= FLOOR_Z_THRESHOLD), returns 1.0
        because the floor is always perfectly flat.

        Args:
            x, y:       Box position (lower-left corner).
            z:          Resting height of the box.
            ol, ow:     Oriented box footprint dimensions.
            heightmap:  Current heightmap (grid_l x grid_w numpy array).
            bin_cfg:    Bin configuration.
            resolution: Grid resolution.

        Returns:
            Float in [0.0, 1.0] -- fraction of footprint at contact height.
        """
        if z <= FLOOR_Z_THRESHOLD:
            return 1.0

        gx = int(round(x / resolution))
        gy = int(round(y / resolution))
        gx_end = min(gx + max(1, int(round(ol / resolution))), bin_cfg.grid_l)
        gy_end = min(gy + max(1, int(round(ow / resolution))), bin_cfg.grid_w)

        footprint = heightmap[gx:gx_end, gy:gy_end]
        if footprint.size == 0:
            return 0.0

        at_contact = np.sum(np.abs(footprint - z) <= CONTACT_HEIGHT_TOLERANCE)
        return float(at_contact) / footprint.size

    # -- Candidate generation -----------------------------------------------

    def _generate_candidates(
        self,
        bin_state: BinState,
        step: float,
    ) -> List[Tuple[float, float]]:
        """
        Generate candidate (x, y) positions from two sources:

        1. Full grid scan at *step* size.
        2. Projected XY corners of all placed boxes.

        Duplicates are removed.

        Args:
            bin_state: Current bin state (read-only).
            step:      Grid scanning step size (cm).

        Returns:
            List of unique (x, y) candidate positions.
        """
        bin_cfg = bin_state.config
        seen: Set[Tuple[float, float]] = set()
        candidates: List[Tuple[float, float]] = []

        # Source 1: full grid scan
        x = 0.0
        while x <= bin_cfg.length + 1e-9:
            y = 0.0
            while y <= bin_cfg.width + 1e-9:
                pt = (x, y)
                if pt not in seen:
                    seen.add(pt)
                    candidates.append(pt)
                y += step
            x += step

        # Source 2: projected corners of placed boxes
        for p in bin_state.placed_boxes:
            for pt in (
                (p.x, p.y),
                (p.x_max, p.y),
                (p.x, p.y_max),
                (p.x_max, p.y_max),
            ):
                if (pt not in seen
                        and 0.0 <= pt[0] <= bin_cfg.length + 1e-6
                        and 0.0 <= pt[1] <= bin_cfg.width + 1e-6):
                    seen.add(pt)
                    candidates.append(pt)

        return candidates
