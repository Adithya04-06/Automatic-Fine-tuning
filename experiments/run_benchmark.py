import argparse
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from src.data.dataset import get_cifar10_loaders
from src.engine.evaluator import evaluate
from src.engine.trainer import train_and_evaluate_strategy
from src.models.backbones import build_model
from src.selector.auto_selector import AutoSelector
from src.utils.config import load_config, merge_configs
from src.utils.logger import setup_logger
from src.utils.metrics import format_results_table, save_results_json
from src.utils.visualization import plot_pareto_frontier


def main():
    parser = argparse.ArgumentParser(description="Full Benchmark for Fine-Tuning Scopes")
    parser.add_argument("--config", type=str, default="configs/resnet18_cifar10.yaml")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per strategy")
    parser.add_argument("--subset", type=float, default=None, help="Fraction of dataset to use")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    base_config = load_config("configs/default_config.yaml") if os.path.exists("configs/default_config.yaml") else {}
    exp_config = load_config(args.config) if os.path.exists(args.config) else {}
    config = merge_configs(base_config, exp_config)

    output_dir = config.get("logging", {}).get("output_dir", "./outputs")
    log_file = os.path.join(output_dir, "logs", "benchmark.log")
    logger = setup_logger(name="ScopeBenchmark", log_file=log_file)

    dev_str = args.device or config.get("device", "auto")
    device = torch.device("cuda" if (dev_str == "auto" and torch.cuda.is_available()) or dev_str == "cuda" else "cpu")
    logger.info(f"Running full benchmark on device: {device}")

    # Data
    ds_cfg = config.get("dataset", {})
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        root=ds_cfg.get("root", "./data"),
        download=ds_cfg.get("download", True),
        train_val_split=ds_cfg.get("train_val_split", 0.8),
        batch_size=ds_cfg.get("batch_size", 64),
        num_workers=ds_cfg.get("num_workers", 2),
        image_size=ds_cfg.get("image_size", 224),
        seed=config.get("seed", 42),
        subset_fraction=args.subset,
    )

    selector = AutoSelector(config=config, logger=logger)
    results, selected, metadata = selector.run_sweep(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
    )

    table_str = format_results_table(results)
    logger.info("\n" + "="*80 + "\nFULL BENCHMARK SUMMARY\n" + "="*80 + "\n" + table_str + "\n" + "="*80)

    # Save outputs
    results_path = os.path.join(output_dir, "results", "benchmark_results.json")
    save_results_json(results, results_path)

    plot_path = os.path.join(output_dir, "figures", "benchmark_pareto.png")
    plot_pareto_frontier(
        results=results,
        selected_strategy_id=selected.strategy_id,
        cost_metric=selector.cost_metric,
        output_path=plot_path,
    )

    logger.info(f"Benchmark completed successfully! Optimal strategy: {selected.strategy_name}")


if __name__ == "__main__":
    main()
