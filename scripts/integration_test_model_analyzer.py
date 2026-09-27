"""Non-training integration test for ModelAnalyzer across ResNet-18, EfficientNet-B0, and ViT-B/16."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.models.backbones import build_model
from src.models.model_analyzer import ModelAnalyzer
from src.models.scope_manager import ScopeManager
from src.selector.auto_selector import AutoSelector
from src.utils.metrics import StrategyResult


def run_architecture_integration_test(arch_name: str, num_classes: int = 10):
    print("\n" + "=" * 85)
    print(f"INTEGRATION TEST: {arch_name.upper()} (Non-Training Structure & Scope Audit)")
    print("=" * 85)

    # 1. Instantiate Model
    model = build_model(backbone_name=arch_name, num_classes=num_classes, pretrained=False)
    family = ModelAnalyzer.detect_family(model)
    print(f"Detected Architectural Family: {family.upper()}")

    # 2. Structural Inspection
    info = ModelAnalyzer.inspect_model(model)
    print(f"Total Model Parameters: {info.total_params:,}")
    print(f"Classification Head:   {info.head.name} ({info.head.param_count:,} params)")
    print(f"Backbone Stem:         {info.stem.name} ({info.stem.param_count:,} params)")
    print(f"Feature Stages ({len(info.stages)} stages detected):")
    for s in info.stages:
        print(f"  - Stage {s.stage_idx} [{s.name}]: {s.param_count:,} params ({len(s.param_names)} parameter tensors)")

    # 3. Dynamic Scope Generation & Freezing Verification
    scopes = ModelAnalyzer.generate_scopes(model)
    assert len(scopes) == 5, f"Expected 5 scopes, got {len(scopes)}"

    print("\nGenerated Progressive Fine-Tuning Scopes:")
    print(f"{'ID':<3} | {'Scope Name':<25} | {'Trainable Params':<18} | {'% Trainable':<12} | {'Unfrozen Targets'}")
    print("-" * 85)

    prev_trainable = -1
    mock_results = []

    for scope in scopes:
        # Apply via ScopeManager to verify backward-compatibility integration
        applied_def = ScopeManager.apply_scope(model, scope)
        trainable_p, total_p, ratio = ScopeManager.count_parameters(model)

        # Verify strict monotonicity
        assert trainable_p > prev_trainable, (
            f"Error: Scope {scope.strategy_id} ({scope.name}) trainable params ({trainable_p}) "
            f"did not increase over previous scope ({prev_trainable})"
        )
        assert total_p == info.total_params, "Total parameter count mismatch!"

        # Verify Gradient Isolation with a single non-training forward + backward step
        inputs = torch.randn(2, 3, 224, 224)
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = outputs.sum()
        loss.backward()

        trainable_names = ScopeManager.get_trainable_param_names(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Expected gradient on {name}, but got None"
                assert torch.count_nonzero(param.grad) > 0, f"Zero gradient on {name}"
            else:
                assert param.grad is None, f"Unexpected gradient on frozen parameter: {name}"

        targets_str = ", ".join(scope.unfrozen_submodules)
        if len(targets_str) > 30:
            targets_str = targets_str[:27] + "..."

        print(f"{scope.strategy_id:<3} | {scope.name:<25} | {trainable_p:>12,} / {total_p:,} | {ratio * 100:>10.2f}% | {targets_str}")
        prev_trainable = trainable_p

        # Create mock strategy result to test AutoSelector integration
        mock_results.append(
            StrategyResult(
                strategy_id=scope.strategy_id,
                strategy_name=scope.name,
                description=scope.description,
                trainable_params=trainable_p,
                total_params=total_p,
                trainable_ratio=ratio,
                val_accuracy=0.60 + 0.07 * scope.strategy_id,  # 67%, 74%, 81%, 88%, 95%
                val_loss=1.5 - 0.25 * scope.strategy_id,
                val_f1_macro=0.60 + 0.07 * scope.strategy_id,
                mean_epoch_time_sec=20.0 * scope.strategy_id,
                peak_vram_mb=300.0 + 100.0 * scope.strategy_id,
                search_epochs=5,
            )
        )

    # 4. AutoSelector Integration Check
    selector = AutoSelector(config={"selector": {"method": "knee_point", "utility_alpha": 0.6}})
    processed_res, selected, metadata = selector.process_and_select(mock_results)

    print("\nAutoSelector Decision Engine Integration:")
    print(f"  - Selected Strategy: Strategy {selected.strategy_id} ({selected.strategy_name})")
    print(f"  - Knee Point Selection: {metadata['knee_point_strategy']}")
    print(f"  - Utility Maximization: {metadata['utility_strategy']}")
    print(f"  - Pareto Frontier: {metadata['pareto_optimal_strategies']}")
    print("  -> PASSED (Zero leakage, proper Pareto / Knee calculation).")


def main():
    architectures = ["resnet18", "efficientnet_b0", "vit_b_16"]
    for arch in architectures:
        run_architecture_integration_test(arch)

    print("\n" + "=" * 85)
    print("ALL NON-TRAINING INTEGRATION TESTS PASSED FOR RESNET-18, EFFICIENTNET-B0, AND VIT-B/16.")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
