"""Data transforms and augmentations for multi-dataset support adapted to pretrained ImageNet models."""

from typing import Tuple
from torchvision import transforms


def get_cifar10_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns training and validation/test transforms for CIFAR-10.
    
    Images are resized to image_size and normalized using ImageNet mean & std.
    Data augmentation includes RandomHorizontalFlip and RandomCrop with reflection padding.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        normalize,
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_test_transform


def get_svhn_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns training and validation/test transforms for SVHN (Street View House Numbers).
    
    CRITICAL DOMAIN DETAIL:
    Digits must NOT be horizontally flipped (e.g. flipping '6' or '3' distorts character identity).
    Augmentation uses gentle cropping with reflection padding and ImageNet normalization.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        normalize,
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_test_transform


def get_transforms(dataset_name: str = "cifar10", image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Factory creating appropriate training and validation transforms based on dataset domain."""
    name = dataset_name.lower().strip()
    if name in ("cifar10", "cifar-10"):
        return get_cifar10_transforms(image_size=image_size)
    elif name in ("svhn", "street_view_house_numbers"):
        return get_svhn_transforms(image_size=image_size)
    else:
        # Default RGB ImageNet transform without aggressive geometric distortions
        return get_cifar10_transforms(image_size=image_size)
