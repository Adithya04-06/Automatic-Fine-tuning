"""Fine-Tuning Scope Manager for dynamically freezing and unfreezing model sub-modules."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.models.model_analyzer import ModelAnalyzer, ScopeDefinition


# Canonical 5 strategies for ResNet-18 (preserved for backward compatibility)
RESNET18_SCOPES: Dict[int, ScopeDefinition] = {
    1: ScopeDefinition(
        strategy_id=1,
        name="head_only",
        description="Linear probing (Classification head only)",
        unfrozen_submodules=("fc",),
    ),
    2: ScopeDefinition(
        strategy_id=2,
        name="block4_head",
        description="Last residual block (layer4) + Classification head",
        unfrozen_submodules=("layer4", "fc"),
    ),
    3: ScopeDefinition(
        strategy_id=3,
        name="block3_4_head",
        description="Last 2 residual blocks (layer3, layer4) + Classification head",
        unfrozen_submodules=("layer3", "layer4", "fc"),
    ),
    4: ScopeDefinition(
        strategy_id=4,
        name="block2_3_4_head",
        description="Last 3 residual blocks (layer2, layer3, layer4) + Classification head",
        unfrozen_submodules=("layer2", "layer3", "layer4", "fc"),
    ),
    5: ScopeDefinition(
        strategy_id=5,
        name="full_model",
        description="Full model fine-tuning (All layers unfrozen)",
        unfrozen_submodules=("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"),
    ),
}


class ScopeManager:
    """Manages parameter freezing/unfreezing for deep learning backbones."""

    @staticmethod
    def list_scopes(model: Optional[nn.Module] = None) -> List[ScopeDefinition]:
        """Returns the list of all defined fine-tuning scope strategies for a model (or default ResNet-18)."""
        if model is not None:
            return ModelAnalyzer.generate_scopes(model)
        return list(RESNET18_SCOPES.values())

    @staticmethod
    def get_scope_definition(strategy_id: int) -> ScopeDefinition:
        """Retrieves the ScopeDefinition for a given strategy ID."""
        if strategy_id not in RESNET18_SCOPES:
            raise ValueError(f"Unknown strategy ID: {strategy_id}. Valid IDs: {list(RESNET18_SCOPES.keys())}")
        return RESNET18_SCOPES[strategy_id]

    @classmethod
    def apply_scope(cls, model: nn.Module, strategy_id: Union[int, ScopeDefinition]) -> ScopeDefinition:
        """Configures parameter requires_grad flags according to the specified fine-tuning strategy."""
        return ModelAnalyzer.apply_scope(model, strategy_id)

    @staticmethod
    def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
        """Calculates trainable params, total params, and trainable ratio."""
        return ModelAnalyzer.count_parameters(model)

    @staticmethod
    def get_trainable_param_names(model: nn.Module) -> List[str]:
        """Returns the list of named parameters that have requires_grad=True."""
        return ModelAnalyzer.get_trainable_param_names(model)
