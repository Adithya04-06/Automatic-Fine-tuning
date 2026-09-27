"""Inference and prediction module for fine-tuned checkpoints across multiple architectures."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from src.models.backbones import build_model


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

SVHN_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]


class ModelPredictor:
    """Inference engine for loading fine-tuned model checkpoints and computing class predictions.
    
    Supports ResNet, EfficientNet, and Vision Transformer (ViT) backbones.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        backbone: Optional[str] = None,
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.checkpoint_path = str(checkpoint_path)
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        # Resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.device = device

        # Load checkpoint dictionary
        self.checkpoint_data = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Extract config and metadata
        self.config = self.checkpoint_data.get("config", {})
        self.strategy_id = self.checkpoint_data.get("strategy_id", None)
        self.strategy_name = self.checkpoint_data.get("strategy_name", "unknown")
        self.val_accuracy = self.checkpoint_data.get("val_accuracy", None)
        self.test_accuracy = self.checkpoint_data.get("test_accuracy", None)

        # Determine backbone architecture
        if backbone is not None:
            self.backbone_name = backbone.lower()
        else:
            self.backbone_name = self.config.get("model", {}).get("backbone", "resnet18").lower()

        # Determine number of classes
        if num_classes is not None:
            self.num_classes = num_classes
        else:
            self.num_classes = self.config.get("dataset", {}).get("num_classes", 10)

        # Class names
        if class_names is not None:
            self.class_names = class_names
        else:
            ds_name = self.config.get("dataset", {}).get("name", "cifar10").lower()
            if "svhn" in ds_name:
                self.class_names = SVHN_CLASSES
            else:
                self.class_names = CIFAR10_CLASSES[:self.num_classes]

        # Determine dropout from checkpoint config or state_dict keys
        dropout = float(self.config.get("model", {}).get("dropout", 0.0))
        state_dict = self.checkpoint_data.get("model_state_dict", self.checkpoint_data)
        
        # Infer dropout if not explicitly specified in config
        if dropout == 0.0:
            if "fc.1.weight" in state_dict or "heads.1.weight" in state_dict or "heads.0.weight" in state_dict:
                dropout = 0.2

        # Build and load model
        try:
            self.model = build_model(
                backbone_name=self.backbone_name,
                num_classes=self.num_classes,
                pretrained=False,
                dropout=dropout,
            )
            self.model.load_state_dict(state_dict)
        except Exception:
            # Fallback to alternate dropout mode (0.0 vs 0.2)
            alt_dropout = 0.2 if dropout == 0.0 else 0.0
            self.model = build_model(
                backbone_name=self.backbone_name,
                num_classes=self.num_classes,
                pretrained=False,
                dropout=alt_dropout,
            )
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # Parameter counting
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Preprocessing transforms (ImageNet standard)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def get_transform(self, image_size: int = 224) -> transforms.Compose:
        """Returns the evaluation transform for the model."""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            self.normalize,
        ])

    def preprocess_image(self, image_input: Union[str, Path, Image.Image, np.ndarray], image_size: int = 224) -> torch.Tensor:
        """Loads and preprocesses an image input into a normalized tensor batch (1, 3, H, W)."""
        if isinstance(image_input, (str, Path)):
            if not os.path.exists(str(image_input)):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                if image_input.max() <= 1.0:
                    image_input = (image_input * 255).astype(np.uint8)
                else:
                    image_input = image_input.astype(np.uint8)
            img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        transform = self.get_transform(image_size=image_size)
        tensor = transform(img)  # Shape: (3, H, W)
        return tensor.unsqueeze(0)  # Shape: (1, 3, H, W)

    @torch.no_grad()
    def predict_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Runs model inference on a preprocessed tensor."""
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4:
            raise ValueError(f"Expected tensor of shape (3, H, W) or (B, 3, H, W), got shape: {tensor.shape}")

        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)

        batch_results = []
        for i in range(probabilities.shape[0]):
            probs = probabilities[i].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            pred_label = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)

            # Top 3 predictions
            top3_indices = np.argsort(probs)[::-1][: min(3, len(probs))]
            top3 = [
                (
                    self.class_names[idx] if idx < len(self.class_names) else str(idx),
                    float(probs[idx]),
                )
                for idx in top3_indices
            ]

            all_probs = {
                (self.class_names[c] if c < len(self.class_names) else str(c)): float(probs[c])
                for c in range(len(probs))
            }

            batch_results.append({
                "predicted_index": pred_idx,
                "predicted_label": pred_label,
                "confidence": conf,
                "top3": top3,
                "probabilities": all_probs,
            })

        return batch_results[0] if len(batch_results) == 1 else batch_results

    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        image_size: int = 224,
    ) -> Dict[str, Any]:
        """Preprocesses an image and runs forward inference to return the predicted class and confidence."""
        tensor = self.preprocess_image(image_input, image_size=image_size)
        return self.predict_tensor(tensor)

    def get_info(self) -> Dict[str, Any]:
        """Returns metadata regarding the loaded model and fine-tuning scope."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "backbone": self.backbone_name,
            "selected_strategy_id": self.strategy_id,
            "selected_strategy_name": self.strategy_name,
            "val_accuracy": self.val_accuracy,
            "test_accuracy": self.test_accuracy,
            "total_params": self.total_params,
            "num_classes": self.num_classes,
            "device": str(self.device),
        }


def load_predictor(
    checkpoint_path: Union[str, Path],
    backbone: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> ModelPredictor:
    """Convenience factory creating a ModelPredictor from a checkpoint path."""
    return ModelPredictor(checkpoint_path=checkpoint_path, backbone=backbone, device=device)
