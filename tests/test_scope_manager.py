"""Unit tests for parameter freezing and scope management on ResNet-18."""

import pytest
import torch
from src.models.backbones import build_model
from src.models.scope_manager import RESNET18_SCOPES, ScopeManager


def test_scope_manager_definitions():
    """Verify that all 5 scopes are registered with unique IDs."""
    scopes = ScopeManager.list_scopes()
    assert len(scopes) == 5
    ids = [s.strategy_id for s in scopes]
    assert ids == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("strategy_id", [1, 2, 3, 4, 5])
def test_parameter_freezing_flags(strategy_id):
    """Verify that only the expected submodules have requires_grad=True."""
    model = build_model("resnet18", num_classes=10, pretrained=False)
    scope_def = ScopeManager.apply_scope(model, strategy_id)

    trainable_params, total_params, ratio = ScopeManager.count_parameters(model)
    assert total_params > 0
    assert trainable_params > 0
    assert 0.0 < ratio <= 1.0

    # Verify layer-specific requires_grad behavior
    if strategy_id == 1:
        # Head only
        assert all(p.requires_grad for p in model.fc.parameters())
        assert not any(p.requires_grad for p in model.layer4.parameters())
        assert not any(p.requires_grad for p in model.layer1.parameters())
        assert not any(p.requires_grad for p in model.conv1.parameters())
    elif strategy_id == 2:
        # Layer 4 + Head
        assert all(p.requires_grad for p in model.fc.parameters())
        assert all(p.requires_grad for p in model.layer4.parameters())
        assert not any(p.requires_grad for p in model.layer3.parameters())
        assert not any(p.requires_grad for p in model.layer1.parameters())
    elif strategy_id == 3:
        # Layer 3, 4 + Head
        assert all(p.requires_grad for p in model.fc.parameters())
        assert all(p.requires_grad for p in model.layer4.parameters())
        assert all(p.requires_grad for p in model.layer3.parameters())
        assert not any(p.requires_grad for p in model.layer2.parameters())
    elif strategy_id == 4:
        # Layer 2, 3, 4 + Head
        assert all(p.requires_grad for p in model.fc.parameters())
        assert all(p.requires_grad for p in model.layer4.parameters())
        assert all(p.requires_grad for p in model.layer3.parameters())
        assert all(p.requires_grad for p in model.layer2.parameters())
        assert not any(p.requires_grad for p in model.layer1.parameters())
    elif strategy_id == 5:
        # Full model
        assert all(p.requires_grad for p in model.parameters())


def test_backward_gradient_isolation():
    """Verify that backpropagation computes gradients ONLY for unfrozen parameters."""
    model = build_model("resnet18", num_classes=10, pretrained=False)
    ScopeManager.apply_scope(model, strategy_id=2)  # layer4 + fc

    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_target = torch.tensor([0, 1])

    output = model(dummy_input)
    loss = torch.nn.functional.cross_entropy(output, dummy_target)
    loss.backward()

    # layer4 and fc must have gradients
    assert model.fc.weight.grad is not None
    assert model.layer4[0].conv1.weight.grad is not None

    # layer1, layer2, layer3, conv1 must NOT have gradients
    assert model.layer3[0].conv1.weight.grad is None
    assert model.layer2[0].conv1.weight.grad is None
    assert model.layer1[0].conv1.weight.grad is None
    assert model.conv1.weight.grad is None
