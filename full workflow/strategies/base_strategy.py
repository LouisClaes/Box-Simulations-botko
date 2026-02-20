"""
Strategy interfaces — abstract base classes for all placement strategies.

Two interfaces are provided:

BaseStrategy (single-bin):
    The classic interface.  Receives a single BinState and proposes a
    PlacementDecision(x, y, orientation_idx).  Use this for all existing
    strategies.  Wrap with MultiBinOrchestrator for multi-bin support.

MultiBinStrategy (multi-bin native):
    Advanced interface for strategies that natively manage multiple bins.
    Receives ALL active BinState objects simultaneously and returns a
    MultiBinDecision(bin_index, x, y, orientation_idx).  Use this when
    the strategy needs to reason across bins — choosing which bin to fill,
    joint optimisation across bins, etc.
    Use with MultiBinPipeline (simulator/multi_bin_pipeline.py).

Creating a single-bin strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Create ``strategies/my_strategy/strategy.py``
2. Subclass ``BaseStrategy``, set ``name``, implement ``decide_placement()``
3. Decorate with ``@register_strategy``
4. Import the module in ``strategies/__init__.py``

Creating a multi-bin strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Create ``strategies/my_multibin_strategy/strategy.py``
2. Subclass ``MultiBinStrategy``, set ``name``, implement ``decide_placement()``
3. Decorate with ``@register_multibin_strategy``
4. Import the module in ``strategies/__init__.py``
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Type, List

from config import Box, PlacementDecision, ExperimentConfig
from simulator.bin_state import BinState


# ─────────────────────────────────────────────────────────────────────────────
# Multi-bin decision dataclass
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass(frozen=True)
class MultiBinDecision:
    """
    A multi-bin strategy's proposed placement.
    
    Unlike PlacementDecision (which is for a single pre-selected bin),
    MultiBinDecision includes the target bin index.  This allows the
    strategy to choose BOTH where to place the box AND which bin to use.
    
    Attributes:
        bin_index:       Which active bin to place in (0-indexed).
        x:               X position within the chosen bin.
        y:               Y position within the chosen bin.
        orientation_idx: Orientation index for the box.
    """
    bin_index: int
    x: float
    y: float
    orientation_idx: int


# ─────────────────────────────────────────────────────────────────────────────
# BaseStrategy — single-bin interface
# ─────────────────────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """
    Abstract base for single-bin placement strategies.
    
    Each call to ``decide_placement()`` receives the current state of ONE
    bin and proposes WHERE within that bin to place the box.  The strategy
    does not choose WHICH bin — that decision belongs to the orchestrator
    (for multi-bin setups) or is trivially the only bin (single-bin).
    
    To add multi-bin support for this strategy, wrap it with the
    MultiBinOrchestrator (orchestrator/multi_bin_orchestrator.py).
    
    The ``BinState`` provided to ``decide_placement()`` gives full 3D access:
    
    +--------------------------+--------------------------------------------+
    | Attribute / Method       | Description                                |
    +==========================+============================================+
    | ``.heightmap``           | 2D numpy grid of current heights           |
    | ``.placed_boxes``        | ``List[Placement]`` — full 3D per box      |
    | ``.get_height_at(...)``  | Query resting z for a footprint            |
    | ``.get_support_ratio()`` | Base support fraction at a position        |
    | ``.get_fill_rate()``     | Volumetric utilisation                     |
    | ``.get_max_height()``    | Tallest point in the bin                   |
    | ``.copy()``              | Deep copy for lookahead simulation         |
    +--------------------------+--------------------------------------------+
    """

    name: str = "unnamed"

    def __init__(self) -> None:
        self._config: Optional[ExperimentConfig] = None

    @property
    def config(self) -> ExperimentConfig:
        """Experiment config, available after ``on_episode_start()``."""
        if self._config is None:
            raise RuntimeError("Strategy not initialised — call on_episode_start() first")
        return self._config

    def on_episode_start(self, config) -> None:
        """Called once before the first box.  Override to initialise state."""
        self._config = config

    def on_episode_end(self, results: dict) -> None:
        """Called after the last box.  Override for cleanup / logging."""
        pass

    @abstractmethod
    def decide_placement(
        self,
        box: Box,
        bin_state: BinState,
    ) -> Optional[PlacementDecision]:
        """
        Propose a placement for *box* given the current bin state.
        
        Args:
            box:       The box to place (original dimensions).
            bin_state: Full 3D state of a single bin — heightmap, placed
                       boxes, height queries.  Read-only — do NOT mutate.
        
        Returns:
            ``PlacementDecision(x, y, orientation_idx)`` or ``None`` if
            the box cannot be placed anywhere in this bin.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# MultiBinStrategy — multi-bin native interface
# ─────────────────────────────────────────────────────────────────────────────

class MultiBinStrategy(ABC):
    """
    Abstract base for strategies that natively manage multiple bins.
    
    Unlike BaseStrategy (which receives a single BinState and only decides
    WHERE within that bin), MultiBinStrategy receives ALL active bin states
    simultaneously and decides BOTH which bin and where within it to place
    the next box.
    
    This enables deep cross-bin reasoning:
      - Which bin has the best "fit" for the current box?
      - Should I pack densely or spread load across bins?
      - Which bin is closer to being full and should be prioritised?
      - Joint optimisation across all bins simultaneously.
    
    Use with ``MultiBinPipeline`` (simulator/multi_bin_pipeline.py) which
    manages N PipelineSimulator instances and passes all their states here.
    
    For simpler multi-bin setups (single-bin strategy + external bin logic),
    use BaseStrategy + MultiBinOrchestrator instead.
    """

    name: str = "unnamed_multibin"

    def __init__(self) -> None:
        self._config = None

    def on_episode_start(self, config) -> None:
        """Called once before the first box.  Override to initialise state."""
        self._config = config

    def on_episode_end(self, results: dict) -> None:
        """Called after the last box.  Override for cleanup / logging."""
        pass

    @abstractmethod
    def decide_placement(
        self,
        box: Box,
        bin_states: List[BinState],
    ) -> Optional[MultiBinDecision]:
        """
        Propose a placement for *box* across all active bins.
        
        The strategy receives the FULL STATE of every active bin and must
        decide:
          1. Which bin (bin_index) to place the box in.
          2. Where within that bin (x, y, orientation_idx).
        
        Args:
            box:        The box to place (original dimensions).
            bin_states: List of BinState for all active bins (read-only).
                        bin_states[i] is the state of bin i.
        
        Returns:
            ``MultiBinDecision(bin_index, x, y, orientation_idx)`` or
            ``None`` if the box cannot be placed in ANY bin.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Strategy registries
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {}
MULTIBIN_STRATEGY_REGISTRY: Dict[str, Type[MultiBinStrategy]] = {}


def register_strategy(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
    """Class decorator — registers a single-bin strategy in the global registry."""
    STRATEGY_REGISTRY[cls.name] = cls
    return cls


def register_multibin_strategy(cls: Type[MultiBinStrategy]) -> Type[MultiBinStrategy]:
    """Class decorator — registers a multi-bin strategy in the global registry."""
    MULTIBIN_STRATEGY_REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> BaseStrategy:
    """Look up a single-bin strategy by name and return a new instance."""
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy '{name}'.  Available: [{available}]")
    return STRATEGY_REGISTRY[name]()


def get_multibin_strategy(name: str) -> MultiBinStrategy:
    """Look up a multi-bin strategy by name and return a new instance."""
    if name not in MULTIBIN_STRATEGY_REGISTRY:
        available = ", ".join(sorted(MULTIBIN_STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown multi-bin strategy '{name}'.  Available: [{available}]"
        )
    return MULTIBIN_STRATEGY_REGISTRY[name]()
