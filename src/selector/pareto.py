"""Pareto frontier calculation and dominance determination for multi-objective trade-offs."""

from typing import List
from src.utils.metrics import StrategyResult


def compute_pareto_frontier(
    results: List[StrategyResult],
    cost_attr: str = "compound_cost",
    acc_attr: str = "val_accuracy",
) -> List[StrategyResult]:
    """Identifies Pareto-optimal strategies.
    
    A strategy A dominates strategy B if:
    - Acc(A) >= Acc(B) and Cost(A) <= Cost(B), AND
    - At least one inequality is strict.

    Modifies `is_pareto_optimal` flag on results in-place and returns updated list.
    """
    for i, a in enumerate(results):
        cost_a = getattr(a, cost_attr)
        acc_a = getattr(a, acc_attr)
        is_dominated = False

        for j, b in enumerate(results):
            if i == j:
                continue
            cost_b = getattr(b, cost_attr)
            acc_b = getattr(b, acc_attr)

            # Check if B dominates A (B has higher or equal acc AND lower or equal cost)
            if (acc_b >= acc_a and cost_b <= cost_a) and (acc_b > acc_a or cost_b < cost_a):
                is_dominated = True
                break

        a.is_pareto_optimal = not is_dominated

    return results
