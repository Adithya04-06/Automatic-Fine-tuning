"""Training loop engine supporting mixed precision, learning rate scheduling, and time profiling."""

import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from src.engine.evaluator import evaluate
from src.engine.profiler import ResourceProfiler
from src.models.scope_manager import ScopeManager, ScopeDefinition
from src.utils.metrics import StrategyResult


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Creates optimizer for parameters that require gradients."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt_type = config.get("optimizer", "adamw").lower()
    lr = float(config.get("learning_rate", 1e-3))
    wd = float(config.get("weight_decay", 1e-2))

    if opt_type == "adamw":
        return AdamW(trainable_params, lr=lr, weight_decay=wd)
    elif opt_type == "sgd":
        momentum = float(config.get("momentum", 0.9))
        return SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_type}")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    clip_grad: Optional[float] = None,
) -> Tuple[float, float]:
    """Runs a single training epoch and returns (mean_loss, epoch_duration_sec)."""
    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    epoch_duration = time.perf_counter() - start_time
    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return mean_loss, epoch_duration


def train_and_evaluate_strategy(
    model: nn.Module,
    strategy_id: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    epochs: int = 5,
    logger=None,
) -> StrategyResult:
    """Trains a model configured with a specific fine-tuning scope and measures validation metrics."""
    # Apply freezing/unfreezing scope
    scope_def: ScopeDefinition = ScopeManager.apply_scope(model, strategy_id)
    trainable_p, total_p, train_ratio = ScopeManager.count_parameters(model)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)

    scheduler_type = config.get("lr_scheduler", "cosine").lower()
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(config.get("min_lr", 1e-6))) if scheduler_type == "cosine" else None

    use_amp = config.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    clip_grad = config.get("gradient_clip", None)

    ResourceProfiler.reset_peak_memory()
    epoch_times: List[float] = []

    if logger:
        logger.info(f"--- Evaluating Strategy {strategy_id}: {scope_def.name} ({scope_def.description}) ---")
        logger.info(f"Trainable params: {trainable_p:,} / {total_p:,} ({train_ratio * 100:.2f}%)")

    for epoch in range(1, epochs + 1):
        train_loss, duration = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler, clip_grad=clip_grad
        )
        epoch_times.append(duration)
        if scheduler is not None:
            scheduler.step()

        if logger and (epoch == 1 or epoch == epochs or epochs <= 5):
            val_acc, val_loss, _ = evaluate(model, val_loader, criterion, device)
            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | Time: {duration:.2f}s"
            )

    # Final evaluation on validation set
    final_acc, final_loss, final_f1 = evaluate(model, val_loader, criterion, device)
    
    # Robust timing calculation: filter out suspension outliers if any exist
    valid_epoch_times = epoch_times
    if len(epoch_times) >= 3:
        median_time = float(np.median(epoch_times))
        # Keep durations within 2.5x of median to reject OS sleep/standby anomalies
        filtered = [t for t in epoch_times if t <= 2.5 * median_time]
        if filtered:
            valid_epoch_times = filtered

    mean_epoch_time = sum(valid_epoch_times) / len(valid_epoch_times) if valid_epoch_times else 0.0
    peak_vram = ResourceProfiler.get_peak_memory_mb()

    return StrategyResult(
        strategy_id=strategy_id,
        strategy_name=scope_def.name,
        description=scope_def.description,
        trainable_params=trainable_p,
        total_params=total_p,
        trainable_ratio=train_ratio,
        val_accuracy=final_acc,
        val_loss=final_loss,
        val_f1_macro=final_f1,
        mean_epoch_time_sec=mean_epoch_time,
        peak_vram_mb=peak_vram,
        search_epochs=epochs,
    )
