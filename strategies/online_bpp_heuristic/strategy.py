"""
DBL + Contact-Support heuristic from Zhao et al. AAAI 2021.

References:
    Zhao, H., She, Q., Zhu, C., Yang, Y., & Xu, K. (2021).
    "Online 3D Bin Packing with Constrained Deep Reinforcement Learning."
    Proceedings of the AAAI Conference on Artificial Intelligence, 35(6), 741-749.
    arXiv:2012.04412
    GitHub: https://github.com/alexfrom0815/Online-3D-BPP-DRL

Algorithm:
    Implements the placement heuristic baseline from the Online-3D-BPP-DRL
    repository, enhanced with surface-contact and support scoring.

    Scoring (higher is better):
        score = CONTACT_WEIGHT * contact_ratio
              + SUPPORT_WEIGHT * support_ratio
              - HEIGHT_WEIGHT  * (z / bin_height)
              - TIE_WEIGHT     * (x + y)

    Where:
        contact_ratio   = fraction of the box base area that rests directly
                          on another surface at height z (same as support_ratio
                          from get_support_ratio).  For floor placements = 1.0.
        support_ratio   = identical to contact_ratio here (both come from
                          BinState.get_support_ratio).
        HEIGHT_WEIGHT   = 1.0  — prefers low placements (mirrors DBL z-term)
        CONTACT_WEIGHT  = 3.0  — rewards tight surface contact (compactness)
        SUPPORT_WEIGHT  = 2.0  — rewards stable multi-point support
        TIE_WEIGHT      = 0.001 — tiny DBL tie-breaker (x + y) = back-left bias

    This matches the scoring described in the README and aligns with the
    contact-maximizing heuristic baselines referenced in Zhao et al. 2021.

Candidate generation:
    Two complementary sources are combined:
        1. Full grid scan at the bin's resolution step.
        2. Projected XY corners of all placed boxes (x, y), (x_max, y),
           (x, y_max), (x_max, y_max).

Stability:
    MIN_SUPPORT = 0.30 is always enforced (matches the simulator's anti-float
    threshold).  When ExperimentConfig.enable_stability is True, the stricter
    cfg.min_support_ratio is also enforced.
"""

import numpy as np
from typing import Optional, List, Tuple, Set

from config import Box, PlacementDecision, ExperimentConfig, Orientation
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Anti-float threshold — must match the simulator's MIN_ANTI_FLOAT_RATIO.
MIN_SUPPORT: float = 0.30

# Scoring weights (higher score is better — maximisation).
CONTACT_WEIGHT: float = 3.0    # Surface contact reward (compactness)
SUPPORT_WEIGHT: float = 2.0    # Support stability reward
HEIGHT_WEIGHT:  float = 1.0    # Penalty for high placements (normalised 0-1)
TIE_WEIGHT:     float = 0.001  # Tiny back-left-corner tie-breaker (x + y)

# Placements at or below this height are treated as floor-level (support = 1.0)
# and skip the support-ratio check to avoid false rejections at z ~ 0.
FLOOR_Z_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

@register_strategy
class OnlineBPPHeuristicStrategy(BaseStrategy):
    """
    Contact + Support heuristic from the Online-3D-BPP-DRL repository
    (Zhao et al. AAAI 2021).

    Scores each (x, y, z) candidate with:

        score = CONTACT_WEIGHT * contact_ratio
              + SUPPORT_WEIGHT * support_ratio
              - HEIGHT_WEIGHT  * (z / bin_height)
              - TIE_WEIGHT     * (x + y)

    This rewards placements that:
      1. Maximise surface contact with existing boxes (tight packing).
      2. Maximise support stability (broad base support).
      3. Stay as low as possible in the bin.
      4. Prefer the back-left corner as a tiebreaker (classic DBL bias).

    The strategy generates candidates from a full grid scan at bin resolution
    plus the projected corners of all placed boxes.  For each candidate it
    evaluates all allowed orientations and enforces:
        - Bin boundary constraints.
        - Height limit (z + oh <= bin.height).
        - Anti-float: support_ratio >= 0.30 always.
        - Stricter stability when ExperimentConfig.enable_stability is True.

    References:
        Zhao et al. (2021). "Online 3D Bin Packing with Constrained Deep
        Reinforcement Learning." AAAI 2021. arXiv:2012.04412.
        GitHub: https://github.com/alexfrom0815/Online-3D-BPP-DRL

    Attributes:
        name: Strategy identifier for the registry ("online_bpp_heuristic").
    """

    name: str = "online_bpp_heuristic"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = 1.0

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Store config and derive grid scan step from bin resolution."""
        super().on_episode_start(config)
        self._scan_step = max(1.0, config.bin.resolution)

    # -- Main entry point ---------------------------------------------------

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Propose a placement for *box* using contact + support scoring.

        The score for a candidate (x, y, z) is:

            score = CONTACT_WEIGHT * contact_ratio
                  + SUPPORT_WEIGHT * support_ratio
                  - HEIGHT_WEIGHT  * (z / bin_height)
                  - TIE_WEIGHT     * (x + y)

        The candidate with the maximum score is returned.

        Steps:
            1. Resolve allowed orientations (flat or all, per config).
            2. Generate candidate positions: grid scan + placed-box corners.
            3. For each (candidate, orientation):
               a. Check bin boundary constraints.
               b. Compute resting z from the heightmap.
               c. Reject if height limit exceeded.
               d. Enforce MIN_SUPPORT = 0.30 (and stricter if configured).
               e. Compute contact + support score.
            4. Return the PlacementDecision for the maximum-score candidate,
               or None if no valid placement exists.

        Args:
            box:       The box to place (original, un-rotated dimensions).
            bin_state: Current 3D bin state (read-only — do NOT mutate).

        Returns:
            PlacementDecision(x, y, orientation_idx) or None.
        """
        cfg = self.config
        bin_cfg = cfg.bin
        step = self._scan_step
        bin_height = bin_cfg.height

        # --- Resolve allowed orientations ---
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # Pre-filter orientations that could never fit in the bin.
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
        best_candidate: Optional[Tuple[float, float, int]] = None

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

                # Floor placements: full support guaranteed
                if z <= FLOOR_Z_THRESHOLD:
                    contact_ratio = 1.0
                    support_ratio = 1.0
                else:
                    # Contact ratio = support ratio at resting height z
                    support_ratio = bin_state.get_support_ratio(
                        cx, cy, ol, ow, z
                    )
                    # Hard minimum always enforced
                    if support_ratio < MIN_SUPPORT:
                        continue
                    # Stricter stability when configured
                    if cfg.enable_stability and support_ratio < cfg.min_support_ratio:
                        continue
                    contact_ratio = support_ratio

                # Margin check (box-to-box gap enforcement)
                if not bin_state.is_margin_clear(cx, cy, ol, ow, z, oh):
                    continue

                # Composite score: contact + support − height − tie-break
                height_norm = z / bin_height if bin_height > 0.0 else 0.0
                score = (
                    CONTACT_WEIGHT * contact_ratio
                    + SUPPORT_WEIGHT * support_ratio
                    - HEIGHT_WEIGHT * height_norm
                    - TIE_WEIGHT * (cx + cy)
                )

                if score > best_score:
                    best_score = score
                    best_candidate = (cx, cy, oidx)

        if best_candidate is None:
            return None

        return PlacementDecision(
            x=best_candidate[0],
            y=best_candidate[1],
            orientation_idx=best_candidate[2],
        )

    # -- Candidate generation -----------------------------------------------

    def _generate_candidates(
        self,
        bin_state: BinState,
        step: float,
    ) -> List[Tuple[float, float]]:
        """
        Generate candidate (x, y) positions from two sources:

        1. Full grid scan at the given step size (covers all grid positions).
        2. Projected XY corners of all placed boxes (x, y), (x_max, y),
           (x, y_max), (x_max, y_max).

        Duplicates are removed via a set.  The list is returned in insertion
        order (grid positions first, then box corners).

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
