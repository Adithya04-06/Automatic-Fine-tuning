import argparse
import os
import random
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.data.dataset import get_cifar10_loaders
from src.selector.auto_selector import AutoSelector
from src.utils.config import load_config, merge_configs
from src.utils.logger import setup_logger
from src.utils.metrics import format_results_table, save_results_json
from src.utils.visualization import plot_pareto_frontier


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="Automatic Fine-Tuning Scope Selection")
    parser.add_argument("--config", type=str, default="configs/resnet18_cifar10.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default=None, choices=["cifar10", "svhn", "synthetic"], help="Dataset name ('cifar10', 'svhn', 'synthetic')")
    parser.add_argument("--epochs", type=int, default=None, help="Override search epochs per strategy")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--subset", type=float, default=None, help="Fraction of dataset to use for quick profiling")
    parser.add_argument("--method", type=str, default=None, choices=["knee_point", "utility", "threshold"], help="Selection method")
    parser.add_argument("--alpha", type=float, default=None, help="Utility alpha weight")
    parser.add_argument("--delta", type=float, default=None, help="Threshold accuracy drop delta")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cpu', 'cuda', 'auto')")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save logs, figures, and results")
    args = parser.parse_args()

    # Load configuration
    base_config = load_config("configs/default_config.yaml") if os.path.exists("configs/default_config.yaml") else {}
    exp_config = load_config(args.config) if os.path.exists(args.config) else {}
    config = merge_configs(base_config, exp_config)

    # CLI Overrides
    if args.epochs is not None:
        config.setdefault("training", {})["search_epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("dataset", {})["batch_size"] = args.batch_size
    if args.method is not None:
        config.setdefault("selector", {})["method"] = args.method
    if args.alpha is not None:
        config.setdefault("selector", {})["utility_alpha"] = args.alpha
    if args.delta is not None:
        config.setdefault("selector", {})["acc_threshold_delta"] = args.delta
    if args.output_dir is not None:
        config.setdefault("logging", {})["output_dir"] = args.output_dir

    output_dir = config.get("logging", {}).get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "logs", "selection.log")
    logger = setup_logger(name="AutoScopeSelector", log_file=log_file)

    # Device selection
    dev_str = args.device or config.get("device", "auto")
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)
    logger.info(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Data loaders
    ds_cfg = config.get("dataset", {})
    ds_name = args.dataset or ds_cfg.get("name", "cifar10")
    logger.info(f"Preparing dataset: {ds_name.upper()} (subset: {args.subset or '100%'})")
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        dataset_name=ds_name,
        root=ds_cfg.get("root", "./data"),
        download=ds_cfg.get("download", True),
        train_val_split=ds_cfg.get("train_val_split", ds_cfg.get("train_split", 0.8)),
        batch_size=ds_cfg.get("batch_size", 64),
        num_workers=ds_cfg.get("num_workers", 0),
        image_size=ds_cfg.get("image_size", 224),
        seed=seed,
        subset_fraction=args.subset,
    )
    logger.info(f"Dataset ready. Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # Initialize AutoSelector
    selector = AutoSelector(config=config, logger=logger)

    # Run Sweep across all 5 strategies (strictly using train and validation sets)
    search_epochs = config.get("training", {}).get("search_epochs", 5)
    logger.info(f"Starting Fine-Tuning Scope Search across 5 strategies ({search_epochs} epochs each)...")
    logger.info("NOTE: Automatic selection is strictly performed using Validation metrics to prevent test-set leakage.")
    results, selected, metadata = selector.run_sweep(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=search_epochs,
    )

    # Post-Selection: Evaluate the selected winning strategy on the unseen Test set
    if test_loader is not None:
        logger.info(f"--- Running Post-Selection Test Evaluation on Selected Strategy [{selected.strategy_id}]: {selected.strategy_name} ---")
        from src.engine.evaluator import evaluate
        from src.engine.trainer import create_optimizer, train_one_epoch
        from src.models.backbones import build_model
        from src.models.scope_manager import ScopeManager

        # Train a model with the selected scope configuration
        final_model = build_model(
            backbone_name=config.get("model", {}).get("backbone", "resnet18"),
            num_classes=config.get("dataset", {}).get("num_classes", 10),
            pretrained=config.get("model", {}).get("pretrained", True),
            dropout=float(config.get("model", {}).get("dropout", 0.0)),
        )
        ScopeManager.apply_scope(final_model, selected.strategy_id)
        final_model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = create_optimizer(final_model, config.get("training", {}))

        scaler = torch.amp.GradScaler("cuda") if (config.get("training", {}).get("use_amp", True) and device.type == "cuda") else None

        for epoch in range(1, search_epochs + 1):
            train_one_epoch(final_model, train_loader, optimizer, criterion, device, scaler=scaler)

        test_acc, test_loss, test_f1 = evaluate(final_model, test_loader, criterion, device)
        selected.test_accuracy = test_acc
        selected.test_loss = test_loss
        selected.test_f1_macro = test_f1
        metadata["selected_strategy_test_accuracy"] = test_acc
        metadata["selected_strategy_test_loss"] = test_loss
        metadata["selected_strategy_test_f1"] = test_f1
        logger.info(f"Final Test Evaluation Result -> Accuracy: {test_acc * 100:.2f}% | Loss: {test_loss:.4f} | F1: {test_f1:.4f}")

        # Save selected model checkpoint
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "selected_model.pt")
        torch.save(
            {
                "strategy_id": selected.strategy_id,
                "strategy_name": selected.strategy_name,
                "model_state_dict": final_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": selected.val_accuracy,
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "test_f1": test_f1,
                "seed": seed,
                "config": config,
            },
            ckpt_path,
        )
        logger.info(f"Saved selected model checkpoint to: {ckpt_path}")

    # Output formatted results table
    table_str = format_results_table(results)
    logger.info("\n" + "="*80 + "\nSTRATEGY EVALUATION SUMMARY (Validation Search)\n" + "="*80 + "\n" + table_str + "\n" + "="*80)

    # Save artifacts
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "selection_results.json")
    save_results_json(results, results_path)
    logger.info(f"Saved results to: {results_path}")

    # Save reproducibility metadata
    meta_path = os.path.join(results_dir, "reproducibility_metadata.json")
    import json
    with open(meta_path, "w") as f:
        json.dump(
            {
                "experiment_name": config.get("experiment_name", "scope_selection"),
                "random_seed": seed,
                "dataset": config.get("dataset", {}),
                "model": config.get("model", {}),
                "training": config.get("training", {}),
                "selector": config.get("selector", {}),
                "selection_result": {
                    "selected_strategy_id": selected.strategy_id,
                    "selected_strategy_name": selected.strategy_name,
                    "val_accuracy": selected.val_accuracy,
                    "test_accuracy": getattr(selected, "test_accuracy", 0.0),
                    "test_loss": getattr(selected, "test_loss", 0.0),
                    "test_f1_macro": getattr(selected, "test_f1_macro", 0.0),
                },
                "hardware": {
                    "device": str(device),
                    "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                    "cuda_available": torch.cuda.is_available(),
                }
            },
            f,
            indent=2,
        )
    logger.info(f"Saved reproducibility metadata to: {meta_path}")

    # Generate all comparison plots
    from src.utils.visualization import generate_all_experiment_plots
    generate_all_experiment_plots(results, selected.strategy_id, output_dir)
    logger.info(f"Saved all 5 experiment comparison figures to: {os.path.join(output_dir, 'figures')}")

    print("\n" + "="*80)
    print(f"AUTOMATIC SELECTION RESULT: Strategy {selected.strategy_id} ({selected.strategy_name})")
    print(f"Reasoning: {metadata['reasoning']}")
    if "selected_strategy_test_accuracy" in metadata:
        print(f"Selected Strategy Test Accuracy: {metadata['selected_strategy_test_accuracy'] * 100:.2f}%")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
