"""Models and Scope Management package."""
from src.models.backbones import build_model
from src.models.model_analyzer import (
    ModelAnalyzer,
    ModelArchitectureInfo,
    ScopeDefinition,
    StageInfo,
)
from src.models.scope_manager import RESNET18_SCOPES, ScopeManager

__all__ = [
    "build_model",
    "ModelAnalyzer",
    "ModelArchitectureInfo",
    "ScopeDefinition",
    "StageInfo",
    "ScopeManager",
    "RESNET18_SCOPES",
]
