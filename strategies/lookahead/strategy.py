"""
Lookahead Strategy -- what-if simulation for 3D bin packing.

Algorithm overview:
    Instead of greedily picking the best position for the current box based on
    local heuristics alone, this strategy simulates placing the box at each
    candidate position using BinState.copy(), then evaluates how "good" the
    resulting bin state is for *future* placements. The position that leaves
    the best post-placement state wins.

    This is a one-step lookahead (depth-1): we do not simulate future boxes
    (which are unknown), but we evaluate the quality of the resulting bin
    surface for accommodating arbitrary future boxes.

Steps:
    1. Generate candidates via coarse grid scan over all orientations.
    2. Pre-filter by quick score (lowest z, then x, then y) to cap the
       number of expensive copy-and-evaluate iterations.
    3. For each surviving candidate:
       a. Deep-copy the bin state.
       b. Apply a virtual placement on the copy.
       c. Evaluate the resulting state with a multi-factor quality function
          (height uniformity, remaining volume, surface flatness, fill
          efficiency, accessible flat area).
    4. Return the candidate whose resulting state scores highest.

Performance:
    The copy() + apply_placement() calls are the bottleneck. The coarse scan
    step and candidate cap keep the number of copies manageable (typically
    30-50 per box). On a 120x80 bin with step=3, a single box decision
    takes roughly 50-200 ms depending on the number of valid candidates.

References:
    Lookahead / rollout heuristics for combinatorial optimisation:
    Bertsekas, D. (2017). "Dynamic Programming and Optimal Control."
"""

from typing import Optional, List, Tuple
import numpy as np

from config import Box, PlacementDecision, ExperimentConfig, Orientation, Placement
from simulator.bin_state import BinState
from strategies.base_strategy import BaseStrategy, register_strategy


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Grid scan step size in cm. Larger = faster but coarser search.
# A value of 3.0 gives a good speed/quality trade-off for the default
# 120x80 bin (yields ~40x27 = 1080 candidates per orientation before pruning).
SCAN_STEP: float = 2.0

# Maximum number of candidates to evaluate with the expensive lookahead
# (copy + apply + evaluate). Candidates beyond this limit are discarded
# after quick-score sorting.
MAX_CANDIDATES: int = 80

# Weight for z (placement height) in the quick pre-filtering score.
# Higher values aggressively prefer low placements during pruning.
QUICK_SCORE_WEIGHT_Z: float = 3.0

# Fraction of bin height above which a candidate is immediately discarded
# during generation (saves time on obviously bad high placements).
HEIGHT_CUTOFF_RATIO: float = 0.95

# Anti-float threshold -- must match the simulator's MIN_ANTI_FLOAT_RATIO.
MIN_SUPPORT: float = 0.30

# Weights for the state evaluation function.
WEIGHT_UNIFORMITY: float = 2.0
WEIGHT_REMAINING: float = 2.0
WEIGHT_FLATNESS: float = 1.5
WEIGHT_FILL: float = 1.0
WEIGHT_ACCESSIBLE: float = 1.5


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@register_strategy
class LookaheadStrategy(BaseStrategy):
    """
    Lookahead (what-if) strategy for 3D bin packing.

    For every candidate placement position, this strategy clones the bin
    state, applies a virtual placement, and evaluates the resulting state
    using a quality function that rewards:

    - Height uniformity (low variance among occupied cells)
    - Remaining vertical capacity (low max height)
    - Surface flatness (low roughness for future placements)
    - Volumetric fill efficiency
    - Large accessible flat areas (connected regions at uniform height)

    The candidate that produces the best post-placement state is returned.

    This strategy NEVER modifies the original bin_state. All simulation is
    done on deep copies obtained via bin_state.copy().

    Hyperparameters (module-level constants):
        SCAN_STEP            Grid scan resolution (cm).
        MAX_CANDIDATES       Cap on expensive evaluations.
        QUICK_SCORE_WEIGHT_Z Aggressiveness of the z-based pre-filter.
        HEIGHT_CUTOFF_RATIO  Discard candidates above this fraction of bin height.
        WEIGHT_*             State evaluation sub-score weights.
    """

    name: str = "lookahead"

    def __init__(self) -> None:
        super().__init__()
        self._scan_step: float = SCAN_STEP

    def on_episode_start(self, config: ExperimentConfig) -> None:
        """Store config and determine scan step."""
        super().on_episode_start(config)
        # Ensure step is at least the grid resolution
        self._scan_step = max(SCAN_STEP, config.bin.resolution)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Find the placement position whose resulting bin state is best for
        future placements.

        Args:
            box:       The box to place (original dimensions).
            bin_state: Current 3D bin state. NOT modified by this method.

        Returns:
            PlacementDecision(x, y, orientation_idx) or None if the box
            cannot be placed anywhere.
        """
        cfg = self.config
        bin_cfg = cfg.bin

        # Resolve allowed orientations
        orientations: List[Tuple[float, float, float]] = (
            Orientation.get_all(box.length, box.width, box.height)
            if cfg.allow_all_orientations
            else Orientation.get_flat(box.length, box.width, box.height)
        )

        # ------------------------------------------------------------------
        # Phase 1: Generate candidate positions via coarse grid scan
        # ------------------------------------------------------------------
        candidates = self._generate_candidates(
            box, bin_state, bin_cfg, orientations, cfg,
        )

        if not candidates:
            return None

        # ------------------------------------------------------------------
        # Phase 2: Prune to top-N by quick score
        # ------------------------------------------------------------------
        candidates = self._prune_candidates(candidates, bin_cfg)

        # ------------------------------------------------------------------
        # Phase 3: Evaluate each candidate with lookahead
        # ------------------------------------------------------------------
        best_score: float = -np.inf
        best_candidate: Optional[Tuple[float, float, int]] = None  # (x, y, oidx)

        for x, y, z, oidx, ol, ow, oh in candidates:
            # Clone the state -- this is the key operation
            sim_state: BinState = bin_state.copy()

            # Create a virtual placement and apply it to the clone
            virtual_placement = Placement(
                box_id=box.id,
                x=x,
                y=y,
                z=z,
                oriented_l=ol,
                oriented_w=ow,
                oriented_h=oh,
                orientation_idx=oidx,
                step=sim_state.step_count,
            )
            sim_state.apply_placement(virtual_placement)

            # Evaluate the resulting state
            score = self._evaluate_state(sim_state, bin_cfg)

            if score > best_score:
                best_score = score
                best_candidate = (x, y, oidx)

        if best_candidate is None:
            return None

        return PlacementDecision(
            x=best_candidate[0],
            y=best_candidate[1],
            orientation_idx=best_candidate[2],
        )

    # ------------------------------------------------------------------
    # Phase 1: Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        box: Box,
        bin_state: BinState,
        bin_cfg,
        orientations: List[Tuple[float, float, float]],
        cfg: ExperimentConfig,
    ) -> List[Tuple[float, float, float, int, float, float, float]]:
        """
        Scan the bin grid at a coarse step and collect all valid candidate
        positions as (x, y, z, orientation_idx, ol, ow, oh) tuples.

        Validity checks applied during generation:
        - Orientation fits within bin dimensions
        - Box does not exceed bin height
        - Placement height is below the HEIGHT_CUTOFF_RATIO threshold
        - Anti-float support ratio is met
        - Optional stability check when enabled

        Args:
            box:          The box being placed.
            bin_state:    Current bin state (read-only).
            bin_cfg:      Bin configuration.
            orientations: List of (ol, ow, oh) orientation tuples.
            cfg:          Full experiment configuration.

        Returns:
            List of (x, y, z, oidx, ol, ow, oh) candidate tuples.
        """
        step = self._scan_step
        height_cutoff = bin_cfg.height * HEIGHT_CUTOFF_RATIO
        candidates: List[Tuple[float, float, float, int, float, float, float]] = []

        for oidx, (ol, ow, oh) in enumerate(orientations):
            # Quick skip: this orientation can never fit in the bin
            if ol > bin_cfg.length or ow > bin_cfg.width or oh > bin_cfg.height:
                continue

            x = 0.0
            while x + ol <= bin_cfg.length + 1e-6:
                y = 0.0
                while y + ow <= bin_cfg.width + 1e-6:
                    # Compute resting height
                    z = bin_state.get_height_at(x, y, ol, ow)

                    # Height bounds check
                    if z + oh > bin_cfg.height + 1e-6:
                        y += step
                        continue

                    # Pre-filter: skip obviously high placements
                    if z > height_cutoff:
                        y += step
                        continue

                    # Anti-float support check
                    if z > 0.5:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < MIN_SUPPORT:
                            y += step
                            continue

                    # Optional stricter stability check
                    if cfg.enable_stability and z > 0.5:
                        sr = bin_state.get_support_ratio(x, y, ol, ow, z)
                        if sr < cfg.min_support_ratio:
                            y += step
                            continue

                    # Margin check (box-to-box gap enforcement)
                    if not bin_state.is_margin_clear(x, y, ol, ow, z, oh):
                        y += step
                        continue

                    candidates.append((x, y, z, oidx, ol, ow, oh))
                    y += step
                x += step

        return candidates

    # ------------------------------------------------------------------
    # Phase 2: Candidate pruning
    # ------------------------------------------------------------------

    def _prune_candidates(
        self,
        candidates: List[Tuple[float, float, float, int, float, float, float]],
        bin_cfg,
    ) -> List[Tuple[float, float, float, int, float, float, float]]:
        """
        If there are more candidates than MAX_CANDIDATES, sort by a quick
        heuristic score and keep only the top MAX_CANDIDATES.

        The quick score favours:
        - Low z (placement height) -- weighted by QUICK_SCORE_WEIGHT_Z
        - Low x (back-left preference along length axis)
        - Low y (back-left preference along width axis)

        Lower quick_score = better candidate (kept).

        Args:
            candidates: Full list of candidate tuples.
            bin_cfg:    Bin configuration for normalization.

        Returns:
            Pruned list of candidates (at most MAX_CANDIDATES).
        """
        if len(candidates) <= MAX_CANDIDATES:
            return candidates

        # Quick score: lower is better
        def quick_score(c: Tuple[float, float, float, int, float, float, float]) -> float:
            x, y, z, _oidx, _ol, _ow, _oh = c
            return (
                QUICK_SCORE_WEIGHT_Z * z / bin_cfg.height
                + x / bin_cfg.length
                + y / bin_cfg.width
            )

        candidates.sort(key=quick_score)
        return candidates[:MAX_CANDIDATES]

    # ------------------------------------------------------------------
    # Phase 3: State evaluation
    # ------------------------------------------------------------------

    def _evaluate_state(self, state: BinState, bin_cfg) -> float:
        """
        Score how 'good' a bin state is for accommodating future placements.

        This is the heart of the lookahead strategy. After virtually placing
        a box, we evaluate the resulting bin surface to predict how well
        it will support arbitrary future boxes.

        Sub-scores:
            1. Uniformity   -- Low height variance among occupied cells means
                               a more usable, level surface.
            2. Remaining     -- Lower max height means more vertical room for
                               future layers.
            3. Flatness      -- Low surface roughness (mean abs height diff
                               between neighbours) enables stable stacking.
            4. Fill          -- Higher volumetric fill rate means we are
                               packing efficiently so far.
            5. Accessible    -- Large flat areas at uniform height can accept
                               a wider variety of future boxes.

        Args:
            state:   The bin state AFTER a virtual placement.
            bin_cfg: Bin configuration.

        Returns:
            Scalar quality score (higher is better). Typical range ~2-8.
        """
        hm = state.heightmap
        bin_height = bin_cfg.height

        # 1. Height uniformity: variance among non-zero cells
        # Occupied cells only -- empty floor cells are not "uneven"
        occupied_mask = hm > 0
        if np.any(occupied_mask):
            height_var = float(np.var(hm[occupied_mask]))
        else:
            height_var = 0.0
        # Normalize: divide by bin_height^2 and clamp to [0, 1]
        uniformity = 1.0 - min(height_var / (bin_height ** 2), 1.0)

        # 2. Remaining vertical capacity
        max_h = state.get_max_height()
        remaining_ratio = 1.0 - max_h / bin_height

        # 3. Surface flatness
        roughness = state.get_surface_roughness()
        # Roughness of 20 cm or more is considered maximally rough
        flatness = 1.0 - min(roughness / 20.0, 1.0)

        # 4. Fill efficiency
        fill = state.get_fill_rate()

        # 5. Accessible flat area: the largest contiguous-height region
        # We round heights to integer cm and find the mode (most common
        # height level). The fraction of cells at that level indicates
        # how much of the surface is uniformly accessible.
        unique_heights = np.unique(np.round(hm, 0))
        max_flat_area = 0
        for h in unique_heights:
            flat_cells = int(np.sum(np.abs(hm - h) < 0.5))
            if flat_cells > max_flat_area:
                max_flat_area = flat_cells
        accessible = max_flat_area / hm.size if hm.size > 0 else 0.0

        # Composite score
        return (
            WEIGHT_UNIFORMITY * uniformity
            + WEIGHT_REMAINING * remaining_ratio
            + WEIGHT_FLATNESS * flatness
            + WEIGHT_FILL * fill
            + WEIGHT_ACCESSIBLE * accessible
        )
