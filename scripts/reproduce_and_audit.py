"""Reproducibility & Audit Script: Rerun Strategy 3 with corrected timing, recalculate selection, and save checkpoints."""

import json
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataset_loaders
from src.engine.evaluator import evaluate
from src.engine.trainer import create_optimizer, train_and_evaluate_strategy, train_one_epoch
from src.models.backbones import build_model
from src.models.scope_manager import ScopeManager
from src.selector.auto_selector import AutoSelector
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import format_results_table, save_results_json
from src.utils.visualization import generate_all_experiment_plots


def main():
    config_path = "configs/resnet18_cifar10.yaml"
    config = load_config(config_path)
    output_dir = config.get("logging", {}).get("output_dir", "./outputs")
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    logger = setup_logger(name="AuditRunner", log_file=os.path.join(output_dir, "logs", "audit.log"))
    logger.info("=== Starting Strategy 3 Timing Correction & Reproducibility Audit ===")

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Load full CIFAR-10 dataset
    train_loader, val_loader, test_loader = get_dataset_loaders(
        dataset_name="cifar10",
        root=config.get("dataset", {}).get("root", "./data"),
        train_val_split=config.get("dataset", {}).get("train_split", 0.8),
        batch_size=config.get("dataset", {}).get("batch_size", 64),
        num_workers=0,
        image_size=config.get("dataset", {}).get("image_size", 224),
        seed=seed,
        subset_fraction=1.0,
    )
    logger.info(f"Dataset ready. Train: {len(train_loader)} batches | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")

    # 1. Rerun Strategy 3 (block3_4_head) with corrected timing
    logger.info("Evaluating Strategy 3 (block3_4_head) with monotonic high-precision timing...")
    model_s3 = build_model(
        backbone_name=config.get("model", {}).get("backbone", "resnet18"),
        num_classes=config.get("dataset", {}).get("num_classes", 10),
        pretrained=config.get("model", {}).get("pretrained", True),
        dropout=float(config.get("model", {}).get("dropout", 0.0)),
    )

    s3_result = train_and_evaluate_strategy(
        model=model_s3,
        strategy_id=3,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.get("training", {}),
        device=device,
        epochs=config.get("training", {}).get("search_epochs", 5),
        logger=logger,
    )
    logger.info(f"Strategy 3 Corrected Result -> Val Acc: {s3_result.val_accuracy*100:.2f}% | Mean Epoch Time: {s3_result.mean_epoch_time_sec:.2f}s")

    # 2. Load existing results for other strategies from selection_results.json
    results_json_path = os.path.join(output_dir, "results", "selection_results.json")
    with open(results_json_path, "r") as f:
        existing_data = json.load(f)

    from src.utils.metrics import StrategyResult
    all_results = []
    for d in existing_data:
        if d["strategy_id"] == 3:
            all_results.append(s3_result)
        else:
            all_results.append(
                StrategyResult(
                    strategy_id=d["strategy_id"],
                    strategy_name=d["strategy_name"],
                    description=d["description"],
                    trainable_params=d["trainable_params"],
                    total_params=d["total_params"],
                    trainable_ratio=d["trainable_ratio"],
                    val_accuracy=d["val_accuracy"],
                    val_loss=d["val_loss"],
                    val_f1_macro=d["val_f1_macro"],
                    mean_epoch_time_sec=d["mean_epoch_time_sec"],
                    peak_vram_mb=d["peak_vram_mb"],
                    search_epochs=d["search_epochs"],
                )
            )

    # 3. Recalculate compound costs, Pareto frontier, Knee-point, and Utility scores
    selector = AutoSelector(config=config, logger=logger)
    processed_results, selected, metadata = selector.process_and_select(all_results)

    # 4. Post-Selection Test Set Evaluation and Checkpoint Saving
    logger.info(f"--- Post-Selection: Final Evaluation & Checkpointing for Strategy [{selected.strategy_id}]: {selected.strategy_name} ---")
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
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for epoch in range(1, config.get("training", {}).get("search_epochs", 5) + 1):
        train_one_epoch(final_model, train_loader, optimizer, criterion, device, scaler=scaler)

    test_acc, test_loss, test_f1 = evaluate(final_model, test_loader, criterion, device)
    selected.test_accuracy = test_acc
    selected.test_loss = test_loss
    selected.test_f1_macro = test_f1
    metadata["selected_strategy_test_accuracy"] = test_acc
    metadata["selected_strategy_test_loss"] = test_loss
    metadata["selected_strategy_test_f1"] = test_f1

    # Save model checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoints", "selected_model.pt")
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
        checkpoint_path,
    )
    logger.info(f"Saved model checkpoint to: {checkpoint_path}")

    # 5. Save updated JSON results and reproducibility metadata
    save_results_json(processed_results, results_json_path)
    logger.info(f"Saved corrected selection results to: {results_json_path}")

    reproducibility_meta = {
        "experiment_name": "cifar10_resnet18_scope_selection",
        "random_seed": seed,
        "splits": {
            "total_images": 50000,
            "train_images": 40000,
            "val_images": 10000,
            "test_images": 10000,
            "val_fraction": 0.20,
        },
        "model": {
            "backbone": config.get("model", {}).get("backbone", "resnet18"),
            "pretrained": True,
            "weights": "ResNet18_Weights.DEFAULT",
            "num_classes": 10,
            "dropout": config.get("model", {}).get("dropout", 0.0),
        },
        "training": {
            "epochs": config.get("training", {}).get("search_epochs", 5),
            "batch_size": config.get("dataset", {}).get("batch_size", 64),
            "optimizer": config.get("training", {}).get("optimizer", "adamw"),
            "learning_rate": config.get("training", {}).get("learning_rate", 0.0005),
            "weight_decay": config.get("training", {}).get("weight_decay", 0.01),
            "lr_scheduler": config.get("training", {}).get("lr_scheduler", "cosine"),
            "use_amp": True,
        },
        "selection": {
            "selected_strategy_id": selected.strategy_id,
            "selected_strategy_name": selected.strategy_name,
            "reasoning": metadata["reasoning"],
            "val_accuracy": selected.val_accuracy,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "test_f1_macro": test_f1,
            "checkpoint_path": checkpoint_path,
        },
        "hardware": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "cuda_available": torch.cuda.is_available(),
        },
    }

    meta_path = os.path.join(output_dir, "results", "reproducibility_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(reproducibility_meta, f, indent=2)
    logger.info(f"Saved reproducibility metadata to: {meta_path}")

    # 6. Regenerate all 5 plots
    generate_all_experiment_plots(processed_results, selected.strategy_id, output_dir)
    logger.info("Saved regenerated comparison plots to outputs/figures/")

    # Output formatted table
    table_str = format_results_table(processed_results)
    logger.info("\n" + "=" * 80 + "\nCORRECTED EXPERIMENT RESULTS SUMMARY\n" + "=" * 80 + "\n" + table_str + "\n" + "=" * 80)
    print("\n" + "=" * 80)
    print(f"AUTOMATIC SELECTION RESULT: Strategy {selected.strategy_id} ({selected.strategy_name})")
    print(f"Reasoning: {metadata['reasoning']}")
    print(f"Selected Strategy Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Checkpoint saved at: {checkpoint_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
