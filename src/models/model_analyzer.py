"""Model Analyzer for dynamic structural inspection, stage identification, and fine-tuning scope generation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ScopeDefinition:
    """Specification of a fine-tuning scope strategy."""
    strategy_id: int
    name: str
    description: str
    unfrozen_submodules: Tuple[str, ...]


@dataclass
class StageInfo:
    """Represents a discrete structural stage / block in a deep learning backbone."""
    stage_idx: int
    name: str
    param_count: int
    param_names: List[str] = field(default_factory=list)


@dataclass
class ModelArchitectureInfo:
    """Complete structural breakdown of a deep learning model."""
    model_family: str
    total_params: int
    stem: StageInfo
    stages: List[StageInfo]
    head: StageInfo


class ModelAnalyzer:
    """Inspects deep learning models, identifies architectural stages, and generates fine-tuning scopes."""

    @classmethod
    def detect_family(cls, model: nn.Module) -> str:
        """Detects the architectural family of a given model (resnet, efficientnet, vit, or generic)."""
        class_name = model.__class__.__name__.lower()
        child_names = [name for name, _ in model.named_children()]

        if "resnet" in class_name or all(k in child_names for k in ["layer1", "layer2", "layer3", "layer4"]):
            return "resnet"
        elif "efficientnet" in class_name or ("features" in child_names and "classifier" in child_names):
            return "efficientnet"
        elif "visiontransformer" in class_name or "vit" in class_name or all(k in child_names for k in ["conv_proj", "encoder", "heads"]):
            return "vit"
        else:
            return "generic"

    @classmethod
    def identify_head_name(cls, model: nn.Module) -> str:
        """Identifies the classification head submodule name."""
        child_names = [name for name, _ in model.named_children()]
        for candidate in ["fc", "classifier", "heads", "head", "linear"]:
            if candidate in child_names:
                return candidate
        # Fallback: check last child
        if child_names:
            return child_names[-1]
        return "head"

    @classmethod
    def inspect_model(cls, model: nn.Module) -> ModelArchitectureInfo:
        """Performs structural decomposition into stem, stages, and classification head."""
        family = cls.detect_family(model)
        total_params = sum(p.numel() for p in model.parameters())

        if family == "resnet":
            return cls._inspect_resnet(model, total_params)
        elif family == "efficientnet":
            return cls._inspect_efficientnet(model, total_params)
        elif family == "vit":
            return cls._inspect_vit(model, total_params)
        else:
            return cls._inspect_generic(model, total_params)

    @classmethod
    def _inspect_resnet(cls, model: nn.Module, total_params: int) -> ModelArchitectureInfo:
        # Stem: conv1, bn1, relu, maxpool
        stem_names = ["conv1", "bn1"]
        stem_params = [n for n, _ in model.named_parameters() if any(n.startswith(s) for s in stem_names)]
        stem_count = sum(p.numel() for n, p in model.named_parameters() if n in stem_params)
        stem = StageInfo(stage_idx=0, name="stem", param_count=stem_count, param_names=stem_params)

        # 4 Residual Stages: layer1, layer2, layer3, layer4
        stages = []
        for idx, layer_name in enumerate(["layer1", "layer2", "layer3", "layer4"], start=1):
            p_names = [n for n, _ in model.named_parameters() if n.startswith(f"{layer_name}.")]
            p_count = sum(p.numel() for n, p in model.named_parameters() if n in p_names)
            stages.append(StageInfo(stage_idx=idx, name=layer_name, param_count=p_count, param_names=p_names))

        # Head: fc
        head_params = [n for n, _ in model.named_parameters() if n.startswith("fc.") or n == "fc"]
        head_count = sum(p.numel() for n, p in model.named_parameters() if n in head_params)
        head = StageInfo(stage_idx=5, name="fc", param_count=head_count, param_names=head_params)

        return ModelArchitectureInfo(
            model_family="resnet", total_params=total_params, stem=stem, stages=stages, head=head
        )

    @classmethod
    def _inspect_efficientnet(cls, model: nn.Module, total_params: int) -> ModelArchitectureInfo:
        # Stem: features.0
        stem_params = [n for n, _ in model.named_parameters() if n.startswith("features.0.")]
        stem_count = sum(p.numel() for n, p in model.named_parameters() if n in stem_params)
        stem = StageInfo(stage_idx=0, name="features.0", param_count=stem_count, param_names=stem_params)

        # Group MBConv blocks into 4 progressive stages (features.1-2, features.3-4, features.5-6, features.7-8)
        stage_groups = [
            ("stage1", ["features.1", "features.2"]),
            ("stage2", ["features.3", "features.4"]),
            ("stage3", ["features.5", "features.6"]),
            ("stage4", ["features.7", "features.8"]),
        ]
        stages = []
        for idx, (sname, prefixes) in enumerate(stage_groups, start=1):
            p_names = [n for n, _ in model.named_parameters() if any(n.startswith(f"{pref}.") for pref in prefixes)]
            p_count = sum(p.numel() for n, p in model.named_parameters() if n in p_names)
            stages.append(StageInfo(stage_idx=idx, name=sname, param_count=p_count, param_names=p_names))

        # Head: classifier
        head_params = [n for n, _ in model.named_parameters() if n.startswith("classifier.")]
        head_count = sum(p.numel() for n, p in model.named_parameters() if n in head_params)
        head = StageInfo(stage_idx=5, name="classifier", param_count=head_count, param_names=head_params)

        return ModelArchitectureInfo(
            model_family="efficientnet", total_params=total_params, stem=stem, stages=stages, head=head
        )

    @classmethod
    def _inspect_vit(cls, model: nn.Module, total_params: int) -> ModelArchitectureInfo:
        # Stem: conv_proj, class_token, encoder.pos_embedding
        stem_prefixes = ["conv_proj", "class_token", "encoder.pos_embedding"]
        stem_params = [n for n, _ in model.named_parameters() if any(n.startswith(p) for p in stem_prefixes)]
        stem_count = sum(p.numel() for n, p in model.named_parameters() if n in stem_params)
        stem = StageInfo(stage_idx=0, name="stem", param_count=stem_count, param_names=stem_params)

        # Group 12 Transformer layers into 4 quarter-depth blocks
        # Stage 1: layers 0..2
        # Stage 2: layers 3..5
        # Stage 3: layers 6..8
        # Stage 4: layers 9..11 + encoder.ln
        stage_groups = [
            ("stage1", [f"encoder.layers.encoder_layer_{i}" for i in range(0, 3)]),
            ("stage2", [f"encoder.layers.encoder_layer_{i}" for i in range(3, 6)]),
            ("stage3", [f"encoder.layers.encoder_layer_{i}" for i in range(6, 9)]),
            ("stage4", [f"encoder.layers.encoder_layer_{i}" for i in range(9, 12)] + ["encoder.ln"]),
        ]
        stages = []
        for idx, (sname, prefixes) in enumerate(stage_groups, start=1):
            p_names = [n for n, _ in model.named_parameters() if any(n.startswith(f"{pref}.") or n == pref for pref in prefixes)]
            p_count = sum(p.numel() for n, p in model.named_parameters() if n in p_names)
            stages.append(StageInfo(stage_idx=idx, name=sname, param_count=p_count, param_names=p_names))

        # Head: heads
        head_params = [n for n, _ in model.named_parameters() if n.startswith("heads.") or n == "heads"]
        head_count = sum(p.numel() for n, p in model.named_parameters() if n in head_params)
        head = StageInfo(stage_idx=5, name="heads", param_count=head_count, param_names=head_params)

        return ModelArchitectureInfo(
            model_family="vit", total_params=total_params, stem=stem, stages=stages, head=head
        )

    @classmethod
    def _inspect_generic(cls, model: nn.Module, total_params: int) -> ModelArchitectureInfo:
        head_name = cls.identify_head_name(model)
        child_names = [n for n, _ in model.named_children() if n != head_name]

        if not child_names:
            stem = StageInfo(0, "backbone", 0, [])
            stages = []
        elif len(child_names) == 1:
            stem = StageInfo(0, "stem", 0, [])
            stages = [StageInfo(1, child_names[0], total_params, [n for n, _ in model.named_parameters() if not n.startswith(head_name)])]
        else:
            stem_name = child_names[0]
            stem_params = [n for n, _ in model.named_parameters() if n.startswith(f"{stem_name}.")]
            stem = StageInfo(0, stem_name, sum(p.numel() for n, p in model.named_parameters() if n in stem_params), stem_params)

            stage_children = child_names[1:]
            stages = []
            for idx, cname in enumerate(stage_children, start=1):
                p_names = [n for n, _ in model.named_parameters() if n.startswith(f"{cname}.")]
                p_count = sum(p.numel() for n, p in model.named_parameters() if n in p_names)
                stages.append(StageInfo(idx, cname, p_count, p_names))

        head_params = [n for n, _ in model.named_parameters() if n.startswith(f"{head_name}.") or n == head_name]
        head_count = sum(p.numel() for n, p in model.named_parameters() if n in head_params)
        head = StageInfo(len(stages) + 1, head_name, head_count, head_params)

        return ModelArchitectureInfo(
            model_family="generic", total_params=total_params, stem=stem, stages=stages, head=head
        )

    @classmethod
    def generate_scopes(cls, model: nn.Module) -> List[ScopeDefinition]:
        """Generates 5 progressive fine-tuning scopes dynamically suited for the model architecture."""
        family = cls.detect_family(model)

        if family == "resnet":
            return [
                ScopeDefinition(
                    strategy_id=1,
                    name="head_only",
                    description="Linear probing (Classification head only)",
                    unfrozen_submodules=("fc",),
                ),
                ScopeDefinition(
                    strategy_id=2,
                    name="block4_head",
                    description="Last residual block (layer4) + Classification head",
                    unfrozen_submodules=("layer4", "fc"),
                ),
                ScopeDefinition(
                    strategy_id=3,
                    name="block3_4_head",
                    description="Last 2 residual blocks (layer3, layer4) + Classification head",
                    unfrozen_submodules=("layer3", "layer4", "fc"),
                ),
                ScopeDefinition(
                    strategy_id=4,
                    name="block2_3_4_head",
                    description="Last 3 residual blocks (layer2, layer3, layer4) + Classification head",
                    unfrozen_submodules=("layer2", "layer3", "layer4", "fc"),
                ),
                ScopeDefinition(
                    strategy_id=5,
                    name="full_model",
                    description="Full model fine-tuning (All layers unfrozen)",
                    unfrozen_submodules=("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"),
                ),
            ]

        elif family == "efficientnet":
            return [
                ScopeDefinition(
                    strategy_id=1,
                    name="head_only",
                    description="Linear probing (Classifier head only)",
                    unfrozen_submodules=("classifier",),
                ),
                ScopeDefinition(
                    strategy_id=2,
                    name="stage4_head",
                    description="Deepest MBConv blocks (features.7, 8) + Classifier head",
                    unfrozen_submodules=("features.7", "features.8", "classifier"),
                ),
                ScopeDefinition(
                    strategy_id=3,
                    name="stage3_4_head",
                    description="Deepest 4 MBConv blocks (features.5-8) + Classifier head",
                    unfrozen_submodules=("features.5", "features.6", "features.7", "features.8", "classifier"),
                ),
                ScopeDefinition(
                    strategy_id=4,
                    name="stage2_3_4_head",
                    description="Upper MBConv hierarchy (features.3-8) + Classifier head",
                    unfrozen_submodules=(
                        "features.3", "features.4", "features.5", "features.6",
                        "features.7", "features.8", "classifier"
                    ),
                ),
                ScopeDefinition(
                    strategy_id=5,
                    name="full_model",
                    description="Full model fine-tuning (All MBConv blocks & Stem unfrozen)",
                    unfrozen_submodules=("features", "classifier"),
                ),
            ]

        elif family == "vit":
            return [
                ScopeDefinition(
                    strategy_id=1,
                    name="head_only",
                    description="Linear probing (ViT classification head only)",
                    unfrozen_submodules=("heads",),
                ),
                ScopeDefinition(
                    strategy_id=2,
                    name="top_quarter_head",
                    description="Top 3 Transformer layers (layers 9-11 + LN) + Head",
                    unfrozen_submodules=(
                        "encoder.layers.encoder_layer_9",
                        "encoder.layers.encoder_layer_10",
                        "encoder.layers.encoder_layer_11",
                        "encoder.ln",
                        "heads",
                    ),
                ),
                ScopeDefinition(
                    strategy_id=3,
                    name="top_half_head",
                    description="Top 6 Transformer layers (layers 6-11 + LN) + Head",
                    unfrozen_submodules=(
                        "encoder.layers.encoder_layer_6",
                        "encoder.layers.encoder_layer_7",
                        "encoder.layers.encoder_layer_8",
                        "encoder.layers.encoder_layer_9",
                        "encoder.layers.encoder_layer_10",
                        "encoder.layers.encoder_layer_11",
                        "encoder.ln",
                        "heads",
                    ),
                ),
                ScopeDefinition(
                    strategy_id=4,
                    name="top_three_quarters_head",
                    description="Top 9 Transformer layers (layers 3-11 + LN) + Head",
                    unfrozen_submodules=(
                        "encoder.layers.encoder_layer_3",
                        "encoder.layers.encoder_layer_4",
                        "encoder.layers.encoder_layer_5",
                        "encoder.layers.encoder_layer_6",
                        "encoder.layers.encoder_layer_7",
                        "encoder.layers.encoder_layer_8",
                        "encoder.layers.encoder_layer_9",
                        "encoder.layers.encoder_layer_10",
                        "encoder.layers.encoder_layer_11",
                        "encoder.ln",
                        "heads",
                    ),
                ),
                ScopeDefinition(
                    strategy_id=5,
                    name="full_model",
                    description="Full model fine-tuning (All Transformer layers, Stem & Head unfrozen)",
                    unfrozen_submodules=("conv_proj", "class_token", "encoder", "heads"),
                ),
            ]

        else:
            # Generic fallback: decompose discovered stages
            info = cls.inspect_model(model)
            head_name = info.head.name
            stages = info.stages

            if len(stages) >= 3:
                s4_targets = (stages[-1].name, head_name)
                s3_targets = tuple([s.name for s in stages[-2:]] + [head_name])
                s2_targets = tuple([s.name for s in stages[1:]] + [head_name])
            elif len(stages) == 2:
                s4_targets = (stages[1].name, head_name)
                s3_targets = (stages[0].name, stages[1].name, head_name)
                s2_targets = (stages[0].name, stages[1].name, head_name)
            else:
                s4_targets = (head_name,)
                s3_targets = (head_name,)
                s2_targets = (head_name,)

            return [
                ScopeDefinition(1, "head_only", "Linear probing (Head only)", (head_name,)),
                ScopeDefinition(2, "last_stage_head", "Deepest feature stage + Head", s4_targets),
                ScopeDefinition(3, "last_2stages_head", "Deepest 2 feature stages + Head", s3_targets),
                ScopeDefinition(4, "last_3stages_head", "Deepest 3 feature stages + Head", s2_targets),
                ScopeDefinition(5, "full_model", "Full model fine-tuning", tuple([s.name for s in stages] + [info.stem.name, head_name])),
            ]

    @classmethod
    def apply_scope(cls, model: nn.Module, scope_def: Union[ScopeDefinition, int]) -> ScopeDefinition:
        """Applies a fine-tuning scope by setting requires_grad flags on model parameters."""
        if isinstance(scope_def, int):
            scopes = cls.generate_scopes(model)
            scope_map = {s.strategy_id: s for s in scopes}
            if scope_def not in scope_map:
                raise ValueError(f"Unknown strategy ID {scope_def}. Valid IDs: {list(scope_map.keys())}")
            scope_def = scope_map[scope_def]

        # Strategy 5: Full model unfreezing
        if scope_def.strategy_id == 5:
            for param in model.parameters():
                param.requires_grad = True
            return scope_def

        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze matching parameters
        targets = scope_def.unfrozen_submodules
        for name, param in model.named_parameters():
            if any(name == t or name.startswith(f"{t}.") for t in targets):
                param.requires_grad = True

        return scope_def

    @staticmethod
    def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
        """Calculates trainable params, total params, and trainable ratio."""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        ratio = trainable_params / total_params if total_params > 0 else 0.0
        return trainable_params, total_params, ratio

    @staticmethod
    def get_trainable_param_names(model: nn.Module) -> List[str]:
        """Returns the list of parameter names that have requires_grad=True."""
        return [name for name, param in model.named_parameters() if param.requires_grad]
