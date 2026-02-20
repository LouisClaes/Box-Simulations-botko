"""Novel Selective Hyper-Heuristic strategy -- primary thesis contribution.

This package implements a rule-based hyper-heuristic that dynamically selects
which low-level placement heuristic to invoke based on a 6-dimensional state
feature vector extracted from the current bin configuration.  No training is
required: selection is governed by a hand-crafted, interpretable decision tree
whose rules are derived from first principles about packing geometry.

Component heuristics:
    H1  WallE scoring        -- Verma et al., AAAI 2020
    H2  DBLF                 -- Karabulut & Inceoglu 2004
    H3  Floor-building       -- Classic heuristic
    H4  Best-fit by volume   -- Classic bin-packing heuristic
"""

from strategies.selective_hyper_heuristic.strategy import SelectiveHyperHeuristicStrategy

__all__ = ["SelectiveHyperHeuristicStrategy"]
