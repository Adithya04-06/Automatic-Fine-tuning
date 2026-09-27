"""Selection Algorithms package."""
from src.selector.auto_selector import AutoSelector
from src.selector.pareto import compute_pareto_frontier
from src.selector.utility import (
    normalize_metrics,
    select_by_knee_point,
    select_by_threshold,
    select_by_utility,
)

__all__ = [
    "AutoSelector",
    "compute_pareto_frontier",
    "normalize_metrics",
    "select_by_knee_point",
    "select_by_utility",
    "select_by_threshold",
]
