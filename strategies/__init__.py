"""
strategies -- pluggable placement strategy interface.

Public API:
    from strategies.base_strategy import BaseStrategy, get_strategy, register_strategy
    from strategies.baseline import BaselineStrategy
"""

from strategies.base_strategy import BaseStrategy, get_strategy, register_strategy, STRATEGY_REGISTRY
import strategies.baseline  # registers BaselineStrategy
import strategies.walle_scoring  # registers WallEScoringStrategy
import strategies.skyline  # registers SkylineStrategy
import strategies.layer_building  # registers LayerBuildingStrategy
import strategies.best_fit_decreasing  # registers BestFitDecreasingStrategy
import strategies.surface_contact  # registers SurfaceContactStrategy
import strategies.gravity_balanced  # registers GravityBalancedStrategy
import strategies.extreme_points  # registers ExtremePointsStrategy
import strategies.ems  # registers EMSStrategy
import strategies.lookahead  # registers LookaheadStrategy
import strategies.wall_building  # registers WallBuildingStrategy
import strategies.column_fill  # registers ColumnFillStrategy
import strategies.hybrid_adaptive  # registers HybridAdaptiveStrategy
import strategies.heuristic_160  # registers Heuristic160Strategy
import strategies.selective_hyper_heuristic  # registers SelectiveHyperHeuristicStrategy
import strategies.lbcp_stability  # registers LBCPStabilityStrategy
import strategies.pct_expansion  # registers PCTExpansionStrategy
import strategies.blueprint_packing  # registers BlueprintPackingStrategy
import strategies.stacking_tree_stability  # registers StackingTreeStabilityStrategy
import strategies.online_bpp_heuristic  # registers OnlineBPPHeuristicStrategy
import strategies.pct_macs_heuristic  # registers PCTMACSHeuristicStrategy
import strategies.two_bounded_best_fit  # registers TwoBoundedBestFitStrategy (MultiBinStrategy)
import strategies.gopt_heuristic  # registers GOPTHeuristicStrategy
import strategies.tsang_multibin  # registers TsangMultiBinStrategy (MultiBinStrategy)

__all__ = [
    "BaseStrategy", "get_strategy", "register_strategy", "STRATEGY_REGISTRY",
]
