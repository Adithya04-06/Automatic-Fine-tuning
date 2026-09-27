"""Model evaluation routines for accuracy, cross-entropy loss, and macro F1 score."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluates a model over a DataLoader.
    
    Args:
        model: PyTorch model.
        dataloader: DataLoader containing evaluation data.
        criterion: Loss function (e.g. CrossEntropyLoss).
        device: Target device (cpu or cuda).

    Returns:
        (accuracy, mean_loss, macro_f1)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

    return accuracy, mean_loss, macro_f1
