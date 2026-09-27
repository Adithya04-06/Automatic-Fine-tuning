"""AutoSelector coordinator that evaluates strategies, computes Pareto frontiers, and selects the optimal fine-tuning scope."""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.engine.profiler import ResourceProfiler
from src.engine.trainer import train_and_evaluate_strategy
from src.models.backbones import build_model
from src.models.model_analyzer import ModelAnalyzer
from src.models.scope_manager import ScopeManager
from src.selector.pareto import compute_pareto_frontier
from src.selector.utility import (
    normalize_metrics,
    select_by_knee_point,
    select_by_threshold,
    select_by_utility,
)
from src.utils.metrics import StrategyResult


class AutoSelector:
    """Coordinates search across fine-tuning scopes, computes trade-offs, and recommends the optimal scope."""

    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.selector_cfg = config.get("selector", {})
        self.time_weight = float(self.selector_cfg.get("time_weight", 0.5))
        self.param_weight = float(self.selector_cfg.get("param_weight", 0.5))
        self.utility_alpha = float(self.selector_cfg.get("utility_alpha", 0.6))
        self.acc_delta = float(self.selector_cfg.get("acc_threshold_delta", 0.02))
        self.method = self.selector_cfg.get("method", "knee_point").lower()
        self.cost_metric = self.selector_cfg.get("cost_metric", "compound_cost")

    def run_sweep(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        strategy_ids: Optional[List[int]] = None,
        epochs: Optional[int] = None,
    ) -> Tuple[List[StrategyResult], StrategyResult, Dict[str, Any]]:
        """Runs the search sweep across candidate fine-tuning strategies strictly using validation data.
        
        Args:
            train_loader: DataLoader for training.
            val_loader: DataLoader for validation.
            device: Compute device (CPU or CUDA).
            strategy_ids: Optional list of strategy IDs to evaluate (defaults to 1..5).
            epochs: Search epochs per strategy (defaults to config search_epochs).

        Returns:
            (processed_results, selected_strategy, decision_metadata)
        """
        search_epochs = epochs or self.config.get("training", {}).get("search_epochs", 5)
        backbone_name = self.config.get("model", {}).get("backbone", "resnet18")
        num_classes = self.config.get("dataset", {}).get("num_classes", 10)
        pretrained = self.config.get("model", {}).get("pretrained", True)
        dropout = float(self.config.get("model", {}).get("dropout", 0.0))

        if strategy_ids is None:
            sample_model = build_model(
                backbone_name=backbone_name,
                num_classes=num_classes,
                pretrained=False,
                dropout=dropout,
            )
            scopes = ModelAnalyzer.generate_scopes(sample_model)
            strategy_ids = [s.strategy_id for s in scopes]

        raw_results: List[StrategyResult] = []

        for sid in strategy_ids:
            # Instantiate fresh model weights for each strategy evaluation
            model = build_model(
                backbone_name=backbone_name,
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
            )

            result = train_and_evaluate_strategy(
                model=model,
                strategy_id=sid,
                train_loader=train_loader,
                val_loader=val_loader,
                config=self.config.get("training", {}),
                device=device,
                epochs=search_epochs,
                logger=self.logger,
            )
            raw_results.append(result)

        return self.process_and_select(raw_results)

    def process_and_select(
        self,
        results: List[StrategyResult],
    ) -> Tuple[List[StrategyResult], StrategyResult, Dict[str, Any]]:
        """Processes collected results: computes compound costs, Pareto frontier, and selects the winner.
        
        Returns:
            (processed_results, selected_strategy, decision_metadata)
        """
        all_times = [r.mean_epoch_time_sec for r in results]
        all_params = [r.trainable_params for r in results]

        # 1. Compute compound cost
        for r in results:
            r.compound_cost = ResourceProfiler.compute_compound_cost(
                mean_epoch_time=r.mean_epoch_time_sec,
                trainable_params=r.trainable_params,
                all_epoch_times=all_times,
                all_trainable_params=all_params,
                time_weight=self.time_weight,
                param_weight=self.param_weight,
            )

        # 2. Compute Pareto Frontier
        compute_pareto_frontier(results, cost_attr=self.cost_metric, acc_attr="val_accuracy")

        # 3. Normalize for decision metrics
        normalize_metrics(results, cost_attr=self.cost_metric, acc_attr="val_accuracy")

        # 4. Evaluate all decision algorithms
        knee_winner = select_by_knee_point(results)
        utility_winner = select_by_utility(results, alpha=self.utility_alpha)
        threshold_winner = select_by_threshold(
            results, delta=self.acc_delta, cost_attr=self.cost_metric, acc_attr="val_accuracy"
        )

        # 5. Pick the primary selection based on active method
        if self.method == "knee_point":
            selected = knee_winner
            reasoning = (
                f"Knee-point / Maximum curvature selection: Strategy {selected.strategy_id} ({selected.strategy_name}) "
                f"provides the maximum marginal accuracy gain before diminishing returns set in."
            )
        elif self.method == "utility":
            selected = utility_winner
            reasoning = (
                f"Multi-objective utility maximization (alpha={self.utility_alpha}): Strategy {selected.strategy_id} "
                f"({selected.strategy_name}) achieved highest utility score ({selected.utility_score:.4f})."
            )
        elif self.method == "threshold":
            selected = threshold_winner
            reasoning = (
                f"Threshold-constrained selection (delta={self.acc_delta * 100:.1f}%): Strategy {selected.strategy_id} "
                f"({selected.strategy_name}) is the lowest-cost strategy within acceptable accuracy bound."
            )
        else:
            selected = knee_winner
            reasoning = f"Defaulted to knee-point selection: Strategy {selected.strategy_id} ({selected.strategy_name})."

        metadata = {
            "selected_strategy_id": selected.strategy_id,
            "selected_strategy_name": selected.strategy_name,
            "method": self.method,
            "reasoning": reasoning,
            "knee_point_strategy": knee_winner.strategy_name,
            "utility_strategy": utility_winner.strategy_name,
            "threshold_strategy": threshold_winner.strategy_name,
            "pareto_optimal_strategies": [r.strategy_name for r in results if r.is_pareto_optimal],
        }

        if self.logger:
            self.logger.info("=== Selection Complete ===")
            self.logger.info(f"Recommended Strategy: [{selected.strategy_id}] {selected.strategy_name}")
            self.logger.info(f"Reasoning: {reasoning}")

        return results, selected, metadata
