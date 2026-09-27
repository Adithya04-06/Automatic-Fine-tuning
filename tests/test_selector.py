"""Unit tests for Pareto frontier, Knee-point detection, utility scoring, and threshold selection."""

import pytest
from src.selector.auto_selector import AutoSelector
from src.selector.pareto import compute_pareto_frontier
from src.selector.utility import (
    normalize_metrics,
    select_by_knee_point,
    select_by_threshold,
    select_by_utility,
)
from src.utils.metrics import StrategyResult


@pytest.fixture
def sample_strategy_results():
    """Provides a realistic set of 5 strategy results with diminishing returns."""
    return [
        StrategyResult(
            strategy_id=1,
            strategy_name="head_only",
            description="Linear probing",
            trainable_params=5130,
            total_params=11181642,
            trainable_ratio=0.00045,
            val_accuracy=0.72,
            val_loss=0.85,
            val_f1_macro=0.71,
            mean_epoch_time_sec=5.0,
            peak_vram_mb=120.0,
            search_epochs=5,
            compound_cost=0.0,
        ),
        StrategyResult(
            strategy_id=2,
            strategy_name="block4_head",
            description="Last block + Head",
            trainable_params=8403466,
            total_params=11181642,
            trainable_ratio=0.7515,
            val_accuracy=0.88,
            val_loss=0.38,
            val_f1_macro=0.87,
            mean_epoch_time_sec=12.0,
            peak_vram_mb=250.0,
            search_epochs=5,
            compound_cost=0.45,
        ),
        StrategyResult(
            strategy_id=3,
            strategy_name="block3_4_head",
            description="Last 2 blocks + Head",
            trainable_params=10503690,
            total_params=11181642,
            trainable_ratio=0.9393,
            val_accuracy=0.91,
            val_loss=0.29,
            val_f1_macro=0.90,
            mean_epoch_time_sec=16.0,
            peak_vram_mb=320.0,
            search_epochs=5,
            compound_cost=0.70,
        ),
        StrategyResult(
            strategy_id=4,
            strategy_name="block2_3_4_head",
            description="Last 3 blocks + Head",
            trainable_params=11029258,
            total_params=11181642,
            trainable_ratio=0.9863,
            val_accuracy=0.915,
            val_loss=0.28,
            val_f1_macro=0.905,
            mean_epoch_time_sec=18.0,
            peak_vram_mb=350.0,
            search_epochs=5,
            compound_cost=0.85,
        ),
        StrategyResult(
            strategy_id=5,
            strategy_name="full_model",
            description="Full model",
            trainable_params=11181642,
            total_params=11181642,
            trainable_ratio=1.0,
            val_accuracy=0.92,
            val_loss=0.26,
            val_f1_macro=0.91,
            mean_epoch_time_sec=20.0,
            peak_vram_mb=380.0,
            search_epochs=5,
            compound_cost=1.0,
        ),
    ]


def test_pareto_frontier_computation(sample_strategy_results):
    """Verify that Pareto optimal strategies are identified correctly."""
    results = compute_pareto_frontier(sample_strategy_results, cost_attr="compound_cost", acc_attr="val_accuracy")
    # All ascending accuracy with ascending cost are Pareto-optimal in this monotonic curve
    assert all(r.is_pareto_optimal for r in results)

    # Introduce a dominated strategy
    dominated = StrategyResult(
        strategy_id=6,
        strategy_name="dominated_dummy",
        description="Bad strategy",
        trainable_params=9000000,
        total_params=11181642,
        trainable_ratio=0.8,
        val_accuracy=0.70,  # Lower acc than strategy 1
        val_loss=1.2,
        val_f1_macro=0.68,
        mean_epoch_time_sec=15.0,  # Higher cost than strategy 1
        peak_vram_mb=300.0,
        search_epochs=5,
        compound_cost=0.60,
    )
    with_dominated = results + [dominated]
    compute_pareto_frontier(with_dominated, cost_attr="compound_cost", acc_attr="val_accuracy")
    assert dominated.is_pareto_optimal is False


def test_knee_point_selection(sample_strategy_results):
    """Verify that the knee-point selects the point of highest marginal return (Strategy 2: block4_head)."""
    winner = select_by_knee_point(sample_strategy_results)
    # Strategy 2 jumps from 72% to 88% (+16%) for cost increase 0.0 -> 0.45. Max return before plateau.
    assert winner.strategy_id == 2
    assert winner.strategy_name == "block4_head"


def test_utility_selection_preference(sample_strategy_results):
    """Verify that utility adapts when user favors accuracy vs low cost."""
    # High alpha (favors accuracy) -> Strategy 5 (full_model)
    winner_high_alpha = select_by_utility(sample_strategy_results, alpha=0.95)
    assert winner_high_alpha.strategy_id == 5

    # Low alpha (favors low cost) -> Strategy 1 (head_only)
    winner_low_alpha = select_by_utility(sample_strategy_results, alpha=0.1)
    assert winner_low_alpha.strategy_id == 1


def test_threshold_selection(sample_strategy_results):
    """Verify that threshold selection picks the lowest cost meeting accuracy SLA."""
    # Delta = 0.02 (within 2% of max 92% -> requires >= 90%). Strategy 3 (91%) is lowest cost meeting this.
    winner = select_by_threshold(sample_strategy_results, delta=0.02, cost_attr="compound_cost", acc_attr="val_accuracy")
    assert winner.strategy_id == 3
    assert winner.val_accuracy >= (0.92 - 0.02)


def test_no_test_set_leakage_in_selection(sample_strategy_results):
    """Methodological Verification: Verify that test_accuracy and test_loss have ZERO influence on selection."""
    import copy

    config = {
        "selector": {
            "method": "knee_point",
            "utility_alpha": 0.6,
            "acc_threshold_delta": 0.02,
            "cost_metric": "compound_cost",
        }
    }
    selector = AutoSelector(config=config)

    # 1. Baseline selection with original (or zero) test metrics
    clean_results = copy.deepcopy(sample_strategy_results)
    for r in clean_results:
        r.test_accuracy = 0.0
        r.test_loss = 0.0

    _, baseline_selected, baseline_meta = selector.process_and_select(clean_results)

    # 2. Mutate test metrics arbitrarily (e.g. make worst strategy have 100% test acc, and best have 0%)
    leaked_results = copy.deepcopy(sample_strategy_results)
    leaked_results[0].test_accuracy = 0.999  # Strategy 1 gets fake 99.9% test accuracy
    leaked_results[0].test_loss = 0.001
    leaked_results[1].test_accuracy = 0.100  # Strategy 2 gets fake 10.0% test accuracy
    leaked_results[1].test_loss = 5.000
    leaked_results[4].test_accuracy = 0.050  # Strategy 5 gets fake 5.0% test accuracy
    leaked_results[4].test_loss = 10.000

    _, mutated_selected, mutated_meta = selector.process_and_select(leaked_results)

    # 3. Assert exact mathematical equivalence
    assert baseline_selected.strategy_id == mutated_selected.strategy_id
    assert baseline_meta["knee_point_strategy"] == mutated_meta["knee_point_strategy"]
    assert baseline_meta["utility_strategy"] == mutated_meta["utility_strategy"]
    assert baseline_meta["threshold_strategy"] == mutated_meta["threshold_strategy"]
    assert baseline_meta["pareto_optimal_strategies"] == mutated_meta["pareto_optimal_strategies"]

    for b, m in zip(clean_results, leaked_results):
        assert b.is_pareto_optimal == m.is_pareto_optimal
        assert b.compound_cost == pytest.approx(m.compound_cost, abs=1e-6)
        assert b.normalized_acc == pytest.approx(m.normalized_acc, abs=1e-6)
        assert b.normalized_cost == pytest.approx(m.normalized_cost, abs=1e-6)
        assert b.utility_score == pytest.approx(m.utility_score, abs=1e-6)

