"""Multi-objective utility scoring and threshold constraint selection methods."""

from typing import List, Optional
from src.utils.metrics import StrategyResult


def normalize_metrics(
    results: List[StrategyResult],
    cost_attr: str = "compound_cost",
    acc_attr: str = "val_accuracy",
) -> List[StrategyResult]:
    """Normalizes accuracy and cost dimensions to the [0, 1] range across strategies."""
    accuracies = [getattr(r, acc_attr) for r in results]
    costs = [getattr(r, cost_attr) for r in results]

    min_acc, max_acc = min(accuracies), max(accuracies)
    min_cost, max_cost = min(costs), max(costs)

    for r in results:
        acc = getattr(r, acc_attr)
        cost = getattr(r, cost_attr)
        r.normalized_acc = (acc - min_acc) / (max_acc - min_acc + 1e-8)
        r.normalized_cost = (cost - min_cost) / (max_cost - min_cost + 1e-8)

    return results


def select_by_utility(
    results: List[StrategyResult],
    alpha: float = 0.6,
) -> StrategyResult:
    """Selects strategy maximizing the weighted utility:
    Utility = alpha * NormAcc - (1 - alpha) * NormCost
    """
    normalize_metrics(results)
    for r in results:
        r.utility_score = alpha * r.normalized_acc - (1.0 - alpha) * r.normalized_cost

    return max(results, key=lambda r: r.utility_score)


def select_by_knee_point(results: List[StrategyResult]) -> StrategyResult:
    """Selects strategy using the Knee-point / maximum curvature detection on the Pareto front.
    
    Finds the point on the Pareto frontier with the maximum perpendicular distance 
    to the secant line joining the minimum-cost and maximum-cost strategies.
    In normalized [0, 1] coordinates, this corresponds to:
    arg max_i (NormAcc_i - NormCost_i)
    """
    normalize_metrics(results)
    # Consider only Pareto-optimal candidates if available
    pareto_candidates = [r for r in results if r.is_pareto_optimal] or results
    
    # Distance from line connecting (0,0) to (1,1) is proportional to (NormAcc - NormCost)
    return max(pareto_candidates, key=lambda r: (r.normalized_acc - r.normalized_cost))


def select_by_threshold(
    results: List[StrategyResult],
    delta: float = 0.02,
    cost_attr: str = "compound_cost",
    acc_attr: str = "val_accuracy",
) -> StrategyResult:
    """Selects the lowest-cost strategy that achieves within `delta` of the maximum observed accuracy."""
    max_acc = max(getattr(r, acc_attr) for r in results)
    threshold_acc = max_acc - delta

    eligible = [r for r in results if getattr(r, acc_attr) >= threshold_acc]
    if not eligible:
        # Fallback to max accuracy strategy
        return max(results, key=lambda r: getattr(r, acc_attr))

    return min(eligible, key=lambda r: getattr(r, cost_attr))
