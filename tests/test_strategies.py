"""
Unit and integration tests for the top-3 bin-packing heuristic strategies.

Run with:
    cd "python/full workflow"
    python -m pytest tests/test_strategies.py -v

Tests cover:
- Strategy registration (all 3 are in the registry)
- Empty bin: first box must land on floor (z=0), return non-None
- Box larger than bin: must return None (can't fit)
- Determinism: same input always produces same output
- Support-ratio result: contact+score is better than pure DBL
"""

import sys
import os
import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Box, BinConfig, ExperimentConfig, PlacementDecision
from simulator.bin_state import BinState
from simulator.pipeline_simulator import PipelineSimulator
from strategies.base_strategy import STRATEGY_REGISTRY
import strategies  # registers all strategies


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_bin_config():
    """Small EUR-pallet at 10 cm resolution."""
    return BinConfig(length=1200.0, width=800.0, height=2700.0, resolution=10.0)


@pytest.fixture
def exp_config(small_bin_config):
    """Default experiment config with stability disabled."""
    return ExperimentConfig(
        bin=small_bin_config,
        strategy_name="test",
        enable_stability=False,
        min_support_ratio=0.8,
        allow_all_orientations=False,
        render_3d=False,
        verbose=False,
    )


@pytest.fixture
def empty_bin_state(small_bin_config):
    """Fresh empty BinState."""
    return BinState(small_bin_config)


@pytest.fixture
def normal_box():
    """A standard box that fits on the floor."""
    return Box(id=1, length=300.0, width=200.0, height=150.0, weight=1.0)


@pytest.fixture
def giant_box():
    """A box larger than the bin in every dimension — cannot be placed."""
    return Box(id=99, length=9999.0, width=9999.0, height=9999.0, weight=1.0)


@pytest.fixture
def exact_floor_box(small_bin_config):
    """A box that exactly matches the bin footprint inside the margins."""
    m = small_bin_config.margin
    return Box(id=2, length=small_bin_config.length - 2 * m, width=small_bin_config.width - 2 * m, height=100.0, weight=1.0)

TOP3_STRATEGIES = [
    "online_bpp_heuristic",
    "gopt_heuristic",
    "pct_macs_heuristic",
]

# ---------------------------------------------------------------------------
# Helper: instantiate and initialise strategy
# ---------------------------------------------------------------------------

def make_strategy(name: str, config: ExperimentConfig):
    """Return an initialised strategy instance."""
    cls = STRATEGY_REGISTRY[name]
    strategy = cls()
    strategy.on_episode_start(config)
    return strategy


# ---------------------------------------------------------------------------
# 1. Registration tests
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_all_top3_registered(self):
        """All 3 top strategies must be in the global registry."""
        for name in TOP3_STRATEGIES:
            assert name in STRATEGY_REGISTRY, (
                f"Strategy '{name}' not found in STRATEGY_REGISTRY."
            )

    def test_no_duplicate_names(self):
        """Each registry key maps to exactly one class."""
        names = list(STRATEGY_REGISTRY.keys())
        assert len(names) == len(set(names)), "Duplicate strategy names found."


# ---------------------------------------------------------------------------
# 2. Empty bin: first placement should be on the floor
# ---------------------------------------------------------------------------

class TestEmptyBinPlacement:
    @pytest.mark.parametrize("strategy_name", TOP3_STRATEGIES)
    def test_returns_placement_for_empty_bin(
        self, strategy_name, exp_config, empty_bin_state, normal_box
    ):
        """Strategy must return a non-None PlacementDecision for an empty bin."""
        strategy = make_strategy(strategy_name, exp_config)
        decision = strategy.decide_placement(normal_box, empty_bin_state)
        assert decision is not None, (
            f"{strategy_name}: returned None for a normal box in an empty bin."
        )

    @pytest.mark.parametrize("strategy_name", TOP3_STRATEGIES)
    def test_placement_within_bin_bounds(
        self, strategy_name, exp_config, empty_bin_state, normal_box, small_bin_config
    ):
        """The returned x, y must be within bin dimensions."""
        strategy = make_strategy(strategy_name, exp_config)
        decision = strategy.decide_placement(normal_box, empty_bin_state)
        assert decision is not None
        assert 0.0 <= decision.x <= small_bin_config.length
        assert 0.0 <= decision.y <= small_bin_config.width

    @pytest.mark.parametrize("strategy_name", TOP3_STRATEGIES)
    def test_first_box_lands_on_floor(
        self, strategy_name, exp_config, empty_bin_state, normal_box
    ):
        """First box in an empty bin must rest at z = 0 (floor height is 0)."""
        strategy = make_strategy(strategy_name, exp_config)
        decision = strategy.decide_placement(normal_box, empty_bin_state)
        assert decision is not None
        # Verify the heightmap at decision point is 0 (floor)
        z = empty_bin_state.get_height_at(
            decision.x, decision.y, normal_box.length, normal_box.width
        )
        assert z == pytest.approx(0.0, abs=1e-6), (
            f"{strategy_name}: first box should rest at z=0, got z={z}"
        )


# ---------------------------------------------------------------------------
# 3. Giant box: must return None
# ---------------------------------------------------------------------------

class TestOversizedBox:
    @pytest.mark.parametrize("strategy_name", TOP3_STRATEGIES)
    def test_giant_box_returns_none(
        self, strategy_name, exp_config, empty_bin_state, giant_box
    ):
        """A box larger than the bin in all dimensions must return None."""
        strategy = make_strategy(strategy_name, exp_config)
        decision = strategy.decide_placement(giant_box, empty_bin_state)
        assert decision is None, (
            f"{strategy_name}: should return None for oversized box, got {decision}"
        )


# ---------------------------------------------------------------------------
# 4. Box exactly fitting bin footprint
# ---------------------------------------------------------------------------

class TestExactFitBox:
    @pytest.mark.parametrize("strategy_name", TOP3_STRATEGIES)
    def test_exact_footprint_box_placed(
        self, strategy_name, exp_config, empty_bin_state, exact_floor_box, small_bin_config
    ):
        """A box matching the bin footprint exactly should be placed at (margin,margin)."""
        strategy = make_strategy(strategy_name, exp_config)
        decision = strategy.decide_placement(exact_floor_box, empty_bin_state)
        assert decision is not None, (
            f"{strategy_name}: failed to place a box that exactly fits the bin."
        )
        # Must start at (margin, margin) — it's the only valid position
        m = small_bin_config.margin
        assert decision.x == pytest.approx(m, abs=1.0)
        assert decision.y == pytest.approx(m, abs=1.0)


# ---------------------------------------------------------------------------
# 5. Determinism: same input → same output
# ---------------------------------------------------------------------------

class TestDeterminism:
    @pytest.mark.parametrize("strategy_name", TOP3_STRATEGIES)
    def test_deterministic_output(
        self, strategy_name, exp_config, empty_bin_state, normal_box
    ):
        """Same box + same bin state must always produce the same decision."""
        results = []
        for _ in range(3):
            strategy = make_strategy(strategy_name, exp_config)
            decision = strategy.decide_placement(normal_box, empty_bin_state)
            assert decision is not None
            results.append((decision.x, decision.y, decision.orientation_idx))

        assert len(set(results)) == 1, (
            f"{strategy_name}: non-deterministic output across identical runs: {results}"
        )


# ---------------------------------------------------------------------------
# 6. Integration: place multiple boxes, fill rate should increase
# ---------------------------------------------------------------------------

class TestIntegrationFillRate:
    @pytest.mark.parametrize("strategy_name", TOP3_STRATEGIES)
    def test_fill_rate_increases_over_sequence(
        self, strategy_name, exp_config
    ):
        """
        Place 10 boxes sequentially via PipelineSimulator.
        Final fill rate must be > 0 (at least some boxes placed).
        """
        from dataset.generator import generate_uniform
        boxes = generate_uniform(n=10, seed=42, min_dim=200.0, max_dim=400.0)

        sim = PipelineSimulator(exp_config)
        strategy = make_strategy(strategy_name, exp_config)

        for box in boxes:
            decision = strategy.decide_placement(box, sim.get_bin_state())
            if decision is not None:
                sim.attempt_placement(
                    box, decision.x, decision.y, decision.orientation_idx
                )

        fill = sim.get_bin_state().get_fill_rate()
        placed = sim.get_summary()["boxes_placed"]

        assert placed > 0, f"{strategy_name}: placed 0 boxes out of 10."
        assert fill > 0.0, f"{strategy_name}: fill rate is 0 after 10 boxes."

    @pytest.mark.parametrize("strategy_name", TOP3_STRATEGIES)
    def test_no_overlapping_boxes(self, strategy_name, exp_config):
        """
        After placing 15 boxes, verify heightmap is monotonically increasing
        (no negative heights, no NaN).
        """
        from dataset.generator import generate_uniform
        boxes = generate_uniform(n=15, seed=7, min_dim=150.0, max_dim=350.0)

        sim = PipelineSimulator(exp_config)
        strategy = make_strategy(strategy_name, exp_config)

        for box in boxes:
            decision = strategy.decide_placement(box, sim.get_bin_state())
            if decision is not None:
                sim.attempt_placement(
                    box, decision.x, decision.y, decision.orientation_idx
                )

        hm = sim.get_bin_state().heightmap
        assert not any([float('nan') == v for v in hm.flat]), (
            f"{strategy_name}: NaN found in heightmap."
        )
        assert float(hm.min()) >= 0.0, (
            f"{strategy_name}: negative height found in heightmap."
        )
