"""Visualization utilities for Accuracy vs. Cost, Pareto Frontier, and Multi-Metric Strategy Comparisons."""

import os
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np

from src.utils.metrics import StrategyResult


def plot_pareto_frontier(
    results: List[StrategyResult],
    selected_strategy_id: Optional[int] = None,
    cost_metric: str = "compound_cost",
    output_path: Optional[str] = None,
) -> None:
    """Plots Validation Accuracy vs. Computational Cost with Pareto frontier."""
    plt.figure(figsize=(9, 6), dpi=130)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    costs = [getattr(r, cost_metric) for r in results]
    accuracies = [r.val_accuracy * 100 for r in results]
    pareto_flags = [r.is_pareto_optimal for r in results]

    # Non-Pareto points
    non_pareto_x = [c for c, p in zip(costs, pareto_flags) if not p]
    non_pareto_y = [a for a, p in zip(accuracies, pareto_flags) if not p]
    if non_pareto_x:
        plt.scatter(non_pareto_x, non_pareto_y, color="#7f7f7f", alpha=0.6, s=130, label="Dominated Strategies")

    # Pareto optimal points
    pareto_results = [r for r in results if r.is_pareto_optimal]
    pareto_results.sort(key=lambda r: getattr(r, cost_metric))
    pareto_x = [getattr(r, cost_metric) for r in pareto_results]
    pareto_y = [r.val_accuracy * 100 for r in pareto_results]

    plt.plot(pareto_x, pareto_y, color="#1f77b4", linestyle="--", linewidth=2.2, alpha=0.85, label="Pareto Frontier")
    plt.scatter(pareto_x, pareto_y, color="#1f77b4", s=160, zorder=5, label="Pareto Optimal")

    # Annotate strategies
    for r in results:
        cost = getattr(r, cost_metric)
        acc = r.val_accuracy * 100
        offset_y = 0.6
        offset_x = 0.015 * (max(costs) - min(costs) + 1e-5)

        if selected_strategy_id is not None and r.strategy_id == selected_strategy_id:
            plt.scatter(
                cost, acc,
                color="#d62728", s=280, edgecolors="black", linewidth=2,
                zorder=10, label=f"Selected: {r.strategy_name}"
            )
            plt.annotate(
                f"[Selected] {r.strategy_name}\n(Val: {acc:.1f}%, Cost: {cost:.2f})",
                (cost, acc),
                xytext=(cost + offset_x, acc + offset_y),
                fontweight="bold",
                fontsize=10,
                color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6)
            )
        else:
            plt.annotate(
                f"{r.strategy_name}\n({acc:.1f}%)",
                (cost, acc),
                xytext=(cost + offset_x, acc + offset_y),
                fontsize=9,
                alpha=0.85
            )

    plt.xlabel("Normalized Compound Cost (Time & Params)", fontsize=12, fontweight="medium")
    plt.ylabel("Validation Accuracy (%)", fontsize=12, fontweight="medium")
    plt.title("Fine-Tuning Scope Trade-Off: Accuracy vs. Computational Cost", fontsize=14, pad=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()


def plot_accuracy_vs_strategy(results: List[StrategyResult], selected_id: Optional[int], output_path: str) -> None:
    """Plots Validation (and Test if available) Accuracy across strategies."""
    plt.figure(figsize=(9, 5), dpi=120)
    names = [r.strategy_name for r in results]
    val_accs = [r.val_accuracy * 100 for r in results]
    colors = ["#d62728" if r.strategy_id == selected_id else "#1f77b4" for r in results]

    bars = plt.bar(names, val_accs, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    plt.ylabel("Validation Accuracy (%)", fontsize=12)
    plt.title("Validation Accuracy by Fine-Tuning Strategy", fontsize=14, fontweight="bold", pad=12)
    plt.ylim(0, 100)

    for bar, acc in zip(bars, val_accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 1.5, f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_training_time_vs_strategy(results: List[StrategyResult], selected_id: Optional[int], output_path: str) -> None:
    """Plots Mean Epoch Training Time across strategies."""
    plt.figure(figsize=(9, 5), dpi=120)
    names = [r.strategy_name for r in results]
    times = [r.mean_epoch_time_sec for r in results]
    colors = ["#d62728" if r.strategy_id == selected_id else "#2ca02c" for r in results]

    bars = plt.bar(names, times, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    plt.ylabel("Mean Epoch Time (seconds)", fontsize=12)
    plt.title("Training Latency per Epoch by Strategy", fontsize=14, fontweight="bold", pad=12)

    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2, t + 0.02 * max(times), f"{t:.2f}s", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_parameters_vs_strategy(results: List[StrategyResult], selected_id: Optional[int], output_path: str) -> None:
    """Plots Trainable Parameter Count and Percentage across strategies."""
    plt.figure(figsize=(9, 5), dpi=120)
    names = [r.strategy_name for r in results]
    params_m = [r.trainable_params / 1e6 for r in results]
    ratios = [r.trainable_ratio * 100 for r in results]
    colors = ["#d62728" if r.strategy_id == selected_id else "#ff7f0e" for r in results]

    bars = plt.bar(names, params_m, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    plt.ylabel("Trainable Parameters (Millions)", fontsize=12)
    plt.title("Trainable Parameter Footprint by Strategy", fontsize=14, fontweight="bold", pad=12)

    for bar, pm, ratio in zip(bars, params_m, ratios):
        plt.text(bar.get_x() + bar.get_width() / 2, pm + 0.2, f"{pm:.2f}M\n({ratio:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_selection_scores_vs_strategy(results: List[StrategyResult], selected_id: Optional[int], output_path: str) -> None:
    """Plots Knee-Point Curvature and Weighted Utility Scores across strategies."""
    plt.figure(figsize=(10, 5.5), dpi=120)
    names = [r.strategy_name for r in results]
    x = np.arange(len(names))
    width = 0.35

    knee_scores = [(r.normalized_acc - r.normalized_cost) for r in results]
    utility_scores = [r.utility_score for r in results]

    plt.bar(x - width/2, knee_scores, width, label="Knee-Point Distance (Acc_norm - Cost_norm)", color="#1f77b4", edgecolor="black")
    plt.bar(x + width/2, utility_scores, width, label="Multi-Objective Utility (alpha=0.6)", color="#2ca02c", edgecolor="black")

    plt.xticks(x, names)
    plt.ylabel("Score", fontsize=12)
    plt.title("Automatic Selection Decision Scores by Strategy", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True)
    plt.axhline(0, color="gray", linestyle="--", alpha=0.7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def generate_all_experiment_plots(results: List[StrategyResult], selected_id: Optional[int], output_dir: str) -> None:
    """Generates all 5 required individual comparison plots and saves to output directory."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_pareto_frontier(results, selected_id, output_path=os.path.join(fig_dir, "pareto_frontier.png"))
    plot_accuracy_vs_strategy(results, selected_id, output_path=os.path.join(fig_dir, "accuracy_vs_strategy.png"))
    plot_training_time_vs_strategy(results, selected_id, output_path=os.path.join(fig_dir, "training_time_vs_strategy.png"))
    plot_parameters_vs_strategy(results, selected_id, output_path=os.path.join(fig_dir, "parameters_vs_strategy.png"))
    plot_selection_scores_vs_strategy(results, selected_id, output_path=os.path.join(fig_dir, "selection_scores_vs_strategy.png"))
