"""Demo inference script to run predictions with fine-tuned model checkpoints across ResNet-18, EfficientNet-B0, and ViT-B/16."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torchvision.datasets import CIFAR10
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.inference import CIFAR10_CLASSES, ModelPredictor, load_predictor


CHECKPOINT_REGISTRY = {
    "resnet18": {
        "name": "ResNet-18",
        "checkpoint": "outputs/checkpoints/selected_model.pt",
        "backbone": "resnet18",
        "default_strategy": "Strategy 2 (block4_head)",
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "checkpoint": "outputs_efficientnet/checkpoints/selected_model.pt",
        "backbone": "efficientnet_b0",
        "default_strategy": "Strategy 2 (stage4_head)",
    },
    "vit_b_16": {
        "name": "Vision Transformer (ViT-B/16)",
        "checkpoint": "outputs_vit/checkpoints/selected_model.pt",
        "backbone": "vit_b_16",
        "default_strategy": "Strategy 2 (top_quarter_head)",
    },
}


def get_sample_cifar10_image(index: int = 0) -> Tuple[Image.Image, int, str]:
    """Retrieves a sample image from the CIFAR-10 test set."""
    data_dir = "./data"
    try:
        test_ds = CIFAR10(root=data_dir, train=False, download=False, transform=None)
        img, label = test_ds[index]
        return img, label, CIFAR10_CLASSES[label]
    except Exception:
        # If raw CIFAR10 not downloaded directly in ./data, create a synthetic RGB image
        img = Image.new("RGB", (32, 32), color=(73, 109, 137))
        return img, 0, CIFAR10_CLASSES[0]


def run_single_model_inference(
    arch_key: str,
    checkpoint_path: Optional[str] = None,
    image: Optional[Image.Image] = None,
    true_label: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """Loads a single model checkpoint and runs inference."""
    info = CHECKPOINT_REGISTRY[arch_key]
    ckpt_path = checkpoint_path or info["checkpoint"]

    print(f"\n================================================================================")
    print(f"LOADING CHECKPOINT: {info['name']}")
    print(f"Checkpoint Path:    {ckpt_path}")

    predictor = load_predictor(
        checkpoint_path=ckpt_path,
        backbone=info["backbone"],
        device=device,
    )

    pred_info = predictor.get_info()
    print(f"Loaded Strategy:    Strategy {pred_info['selected_strategy_id']} ({pred_info['selected_strategy_name']})")
    print(f"Total Parameters:   {pred_info['total_params']:,}")
    if pred_info["val_accuracy"] is not None:
        print(f"Validation Acc:     {pred_info['val_accuracy'] * 100:.2f}%")
    if pred_info["test_accuracy"] is not None:
        print(f"Test Accuracy:      {pred_info['test_accuracy'] * 100:.2f}%")

    # Run inference
    result = predictor.predict_image(image, image_size=224)

    print(f"\nINFERENCE RESULT:")
    if true_label is not None:
        print(f"Ground Truth Class: {true_label}")
    print(f"Predicted Class:    {result['predicted_label']} (Class ID: {result['predicted_index']})")
    print(f"Confidence:         {result['confidence'] * 100:.2f}%")
    print(f"Top-3 Probabilities:")
    for rank, (cls_name, prob) in enumerate(result['top3'], 1):
        bar = "#" * int(prob * 30)
        print(f"  {rank}. {cls_name:<12} {prob * 100:6.2f}% | {bar}")

    return {
        "architecture": info["name"],
        "backbone": info["backbone"],
        "checkpoint": ckpt_path,
        "strategy": pred_info["selected_strategy_name"],
        "predicted_label": result["predicted_label"],
        "confidence": result["confidence"],
        "top3": result["top3"],
    }


def main():
    parser = argparse.ArgumentParser(description="Demo inference using automatically selected fine-tuned checkpoints.")
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["resnet18", "efficientnet_b0", "vit_b_16", "all"],
        default="all",
        help="Architecture to test ('resnet18', 'efficientnet_b0', 'vit_b_16', or 'all').",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to custom checkpoint file.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to image file. Defaults to a CIFAR-10 test sample.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=3,  # Index 3 in CIFAR-10 test set is an airplane
        help="CIFAR-10 test sample index if no --image provided.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('cuda', 'cpu', or 'auto').",
    )
    args = parser.parse_args()

    # Load image
    if args.image is not None:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found at: {args.image}")
        img = Image.open(args.image).convert("RGB")
        true_label = None
        print(f"Loaded custom input image from: {args.image}")
    else:
        img, true_idx, true_label = get_sample_cifar10_image(index=args.sample_index)
        print(f"Loaded CIFAR-10 Test Sample [Index {args.sample_index}]: Ground Truth = '{true_label}' (ID: {true_idx})")

    # Select architectures
    if args.architecture == "all":
        arch_list = ["resnet18", "efficientnet_b0", "vit_b_16"]
    else:
        arch_list = [args.architecture]

    all_results = []
    for arch in arch_list:
        ckpt = args.checkpoint if len(arch_list) == 1 else None
        res = run_single_model_inference(
            arch_key=arch,
            checkpoint_path=ckpt,
            image=img,
            true_label=true_label,
            device=args.device,
        )
        all_results.append(res)

    if len(all_results) > 1:
        print("\n================================================================================")
        print("TRI-ARCHITECTURE PREDICTION COMPARISON")
        print("================================================================================")
        print(f"{'Architecture':<28} | {'Selected Strategy':<22} | {'Prediction':<12} | {'Confidence':<10}")
        print("-" * 80)
        for r in all_results:
            print(f"{r['architecture']:<28} | {r['strategy']:<22} | {r['predicted_label']:<12} | {r['confidence'] * 100:6.2f}%")
        print("================================================================================\n")


if __name__ == "__main__":
    from typing import Tuple
    main()
