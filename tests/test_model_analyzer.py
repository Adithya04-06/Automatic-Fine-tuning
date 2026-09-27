"""Unit tests for ModelAnalyzer across ResNet-18, EfficientNet-B0, and Vision Transformer (ViT)."""

import pytest
import torch
import torch.nn as nn
from src.models.backbones import build_model
from src.models.model_analyzer import ModelAnalyzer, ModelArchitectureInfo, ScopeDefinition


@pytest.mark.parametrize("arch_name, expected_family", [
    ("resnet18", "resnet"),
    ("resnet34", "resnet"),
    ("resnet50", "resnet"),
    ("efficientnet_b0", "efficientnet"),
    ("vit_b_16", "vit"),
    ("vit_b_32", "vit"),
])
def test_detect_family(arch_name, expected_family):
    """Verify architectural family detection across diverse model families."""
    model = build_model(backbone_name=arch_name, num_classes=10, pretrained=False)
    detected = ModelAnalyzer.detect_family(model)
    assert detected == expected_family


def test_inspect_model_resnet():
    """Verify structural inspection of ResNet-18."""
    model = build_model("resnet18", num_classes=10, pretrained=False)
    info = ModelAnalyzer.inspect_model(model)

    assert info.model_family == "resnet"
    assert info.total_params > 10_000_000
    assert info.stem.name == "stem"
    assert len(info.stages) == 4
    assert [s.name for s in info.stages] == ["layer1", "layer2", "layer3", "layer4"]
    assert info.head.name == "fc"
    assert info.head.param_count == 512 * 10 + 10  # 5,130


def test_inspect_model_efficientnet():
    """Verify structural inspection of EfficientNet-B0."""
    model = build_model("efficientnet_b0", num_classes=10, pretrained=False)
    info = ModelAnalyzer.inspect_model(model)

    assert info.model_family == "efficientnet"
    assert info.total_params > 4_000_000
    assert info.stem.name == "features.0"
    assert len(info.stages) == 4
    assert [s.name for s in info.stages] == ["stage1", "stage2", "stage3", "stage4"]
    assert info.head.name == "classifier"
    assert info.head.param_count == 1280 * 10 + 10  # 12,810


def test_inspect_model_vit():
    """Verify structural inspection of Vision Transformer (ViT-B/16)."""
    model = build_model("vit_b_16", num_classes=10, pretrained=False)
    info = ModelAnalyzer.inspect_model(model)

    assert info.model_family == "vit"
    assert info.total_params > 80_000_000
    assert info.stem.name == "stem"
    assert len(info.stages) == 4
    assert [s.name for s in info.stages] == ["stage1", "stage2", "stage3", "stage4"]
    assert info.head.name == "heads"
    assert info.head.param_count == 768 * 10 + 10  # 7,690


@pytest.mark.parametrize("arch_name", ["resnet18", "efficientnet_b0", "vit_b_16"])
def test_generate_scopes_progressive_growth(arch_name):
    """Verify that generated scopes produce strictly monotonically increasing trainable parameters."""
    model = build_model(arch_name, num_classes=10, pretrained=False)
    scopes = ModelAnalyzer.generate_scopes(model)

    assert len(scopes) == 5
    assert [s.strategy_id for s in scopes] == [1, 2, 3, 4, 5]

    prev_trainable = -1
    for scope in scopes:
        ModelAnalyzer.apply_scope(model, scope)
        trainable_p, total_p, ratio = ModelAnalyzer.count_parameters(model)
        assert trainable_p > prev_trainable, f"Scope {scope.strategy_id} did not increase trainable params in {arch_name}"
        assert 0.0 < ratio <= 1.0
        prev_trainable = trainable_p

    # Strategy 5 must unfreeze 100% of parameters
    assert prev_trainable == total_p


def test_backward_gradient_isolation_efficientnet():
    """Verify backward gradient isolation on EfficientNet-B0 for Strategy 2."""
    model = build_model("efficientnet_b0", num_classes=10, pretrained=False)
    ModelAnalyzer.apply_scope(model, 2)  # Stage 4 + Classifier

    inputs = torch.randn(2, 3, 224, 224)
    outputs = model(inputs)
    loss = outputs.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in ["features.7.", "features.8.", "classifier."]):
            assert param.grad is not None, f"Expected gradient on unfrozen parameter: {name}"
            assert torch.count_nonzero(param.grad) > 0
        else:
            assert param.grad is None, f"Expected no gradient on frozen parameter: {name}"


def test_backward_gradient_isolation_vit():
    """Verify backward gradient isolation on Vision Transformer for Strategy 2."""
    model = build_model("vit_b_16", num_classes=10, pretrained=False)
    ModelAnalyzer.apply_scope(model, 2)  # Top quarter (layers 9-11 + LN) + Heads

    inputs = torch.randn(2, 3, 224, 224)
    outputs = model(inputs)
    loss = outputs.sum()
    loss.backward()

    top_prefixes = [
        "encoder.layers.encoder_layer_9.",
        "encoder.layers.encoder_layer_10.",
        "encoder.layers.encoder_layer_11.",
        "encoder.ln.",
        "heads.",
    ]
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in top_prefixes):
            assert param.grad is not None, f"Expected gradient on unfrozen parameter: {name}"
            assert torch.count_nonzero(param.grad) > 0
        else:
            assert param.grad is None, f"Expected no gradient on frozen parameter: {name}"


def test_build_model_custom_dropout():
    """Verify build_model applies dropout correctly across all architectures."""
    m_res = build_model("resnet18", num_classes=5, dropout=0.3, pretrained=False)
    assert isinstance(m_res.fc, nn.Sequential)

    m_eff = build_model("efficientnet_b0", num_classes=5, dropout=0.4, pretrained=False)
    assert isinstance(m_eff.classifier[0], nn.Dropout)
    assert m_eff.classifier[0].p == 0.4

    m_vit = build_model("vit_b_16", num_classes=5, dropout=0.25, pretrained=False)
    assert isinstance(m_vit.heads, nn.Sequential)
