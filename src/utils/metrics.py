"""Metrics tracker and evaluation statistics formatting."""

from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Dict, List, Optional


@dataclass
class StrategyResult:
    """Encapsulates all evaluation and resource metrics for a single fine-tuning scope."""
    strategy_id: int
    strategy_name: str
    description: str
    trainable_params: int
    total_params: int
    trainable_ratio: float
    val_accuracy: float
    val_loss: float
    val_f1_macro: float
    mean_epoch_time_sec: float
    peak_vram_mb: float
    search_epochs: int
    test_accuracy: float = 0.0
    test_loss: float = 0.0
    test_f1_macro: float = 0.0
    compound_cost: float = 0.0
    normalized_acc: float = 0.0
    normalized_cost: float = 0.0
    utility_score: float = 0.0
    is_pareto_optimal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def format_results_table(results: List[StrategyResult]) -> str:
    """Formats strategy evaluation results into a clean markdown table."""
    headers = [
        "ID", "Strategy Name", "Trainable Params", "Param %",
        "Val Acc (%)", "Test Acc (%)", "Val Loss", "Val F1", "Epoch Time (s)", "Cost", "Pareto"
    ]
    rows = []
    for r in results:
        rows.append([
            str(r.strategy_id),
            r.strategy_name,
            f"{r.trainable_params:,}",
            f"{r.trainable_ratio * 100:.2f}%",
            f"{r.val_accuracy * 100:.2f}%",
            f"{r.test_accuracy * 100:.2f}%",
            f"{r.val_loss:.4f}",
            f"{r.val_f1_macro:.4f}",
            f"{r.mean_epoch_time_sec:.2f}s",
            f"{r.compound_cost:.3f}",
            "Yes" if r.is_pareto_optimal else "No"
        ])

    col_widths = [max(len(header), max(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
    
    header_line = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"
    separator_line = "| " + " | ".join(f"{'-'*w}" for w in col_widths) + " |"
    data_lines = ["| " + " | ".join(f"{val:<{w}}" for val, w in zip(row, col_widths)) + " |" for row in rows]
    
    return "\n".join([header_line, separator_line] + data_lines)


def save_results_json(results: List[StrategyResult], filepath: str) -> None:
    """Saves strategy results to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = [r.to_dict() for r in results]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
