"""Unit tests for inference module and checkpoint predictor."""

import os
import pytest
import torch
import numpy as np
from PIL import Image

from src.engine.inference import CIFAR10_CLASSES, ModelPredictor, load_predictor


@pytest.fixture
def dummy_checkpoint_path(tmp_path):
    """Creates a temporary synthetic checkpoint for test isolation."""
    from src.models.backbones import build_model
    model = build_model("resnet18", num_classes=10, pretrained=False, dropout=0.0)
    ckpt = {
        "strategy_id": 2,
        "strategy_name": "block4_head",
        "model_state_dict": model.state_dict(),
        "val_accuracy": 0.9344,
        "test_accuracy": 0.9081,
        "config": {
            "model": {"backbone": "resnet18", "dropout": 0.0},
            "dataset": {"name": "cifar10", "num_classes": 10},
        },
    }
    ckpt_path = tmp_path / "dummy_checkpoint.pt"
    torch.save(ckpt, str(ckpt_path))
    return str(ckpt_path)


def test_predictor_initialization(dummy_checkpoint_path):
    predictor = ModelPredictor(dummy_checkpoint_path, device="cpu")
    info = predictor.get_info()
    assert info["backbone"] == "resnet18"
    assert info["selected_strategy_id"] == 2
    assert info["selected_strategy_name"] == "block4_head"
    assert info["val_accuracy"] == 0.9344
    assert info["test_accuracy"] == 0.9081


def test_predictor_predict_tensor(dummy_checkpoint_path):
    predictor = ModelPredictor(dummy_checkpoint_path, device="cpu")
    dummy_tensor = torch.randn(3, 224, 224)
    result = predictor.predict_tensor(dummy_tensor)

    assert "predicted_index" in result
    assert "predicted_label" in result
    assert "confidence" in result
    assert "top3" in result
    assert "probabilities" in result

    assert 0 <= result["predicted_index"] < 10
    assert result["predicted_label"] in CIFAR10_CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top3"]) == 3
    assert len(result["probabilities"]) == 10
    assert np.isclose(sum(result["probabilities"].values()), 1.0, atol=1e-4)


def test_predictor_predict_pil_image(dummy_checkpoint_path):
    predictor = ModelPredictor(dummy_checkpoint_path, device="cpu")
    pil_img = Image.new("RGB", (32, 32), color=(128, 64, 32))
    result = predictor.predict_image(pil_img)

    assert 0 <= result["predicted_index"] < 10
    assert result["predicted_label"] in CIFAR10_CLASSES
    assert 0.0 <= result["confidence"] <= 1.0


def test_load_real_resnet18_checkpoint():
    ckpt_path = "outputs/checkpoints/selected_model.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("ResNet-18 benchmark checkpoint not present in outputs/")

    predictor = load_predictor(ckpt_path, device="cpu")
    assert predictor.backbone_name == "resnet18"
    assert predictor.strategy_id == 2
    assert predictor.val_accuracy >= 0.90

    # Test inference
    dummy_tensor = torch.randn(1, 3, 224, 224)
    res = predictor.predict_tensor(dummy_tensor)
    assert 0 <= res["predicted_index"] < 10


def test_load_real_efficientnet_checkpoint():
    ckpt_path = "outputs_efficientnet/checkpoints/selected_model.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("EfficientNet benchmark checkpoint not present in outputs_efficientnet/")

    predictor = load_predictor(ckpt_path, device="cpu")
    assert predictor.backbone_name == "efficientnet_b0"
    assert predictor.strategy_id == 2
    assert predictor.val_accuracy >= 0.90

    # Test inference
    dummy_tensor = torch.randn(1, 3, 224, 224)
    res = predictor.predict_tensor(dummy_tensor)
    assert 0 <= res["predicted_index"] < 10


def test_load_real_vit_checkpoint():
    ckpt_path = "outputs_vit/checkpoints/selected_model.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("ViT benchmark checkpoint not present in outputs_vit/")

    predictor = load_predictor(ckpt_path, device="cpu")
    assert predictor.backbone_name == "vit_b_16"
    assert predictor.strategy_id == 2
    assert predictor.val_accuracy >= 0.95

    # Test inference
    dummy_tensor = torch.randn(1, 3, 224, 224)
    res = predictor.predict_tensor(dummy_tensor)
    assert 0 <= res["predicted_index"] < 10
