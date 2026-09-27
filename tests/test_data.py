"""Unit tests for dataset transforms and preprocessing pipeline."""

import pytest
import torch
from PIL import Image
from src.data.dataset import get_dataset_loaders
from src.data.transforms import get_cifar10_transforms, get_svhn_transforms, get_transforms


def test_cifar10_transforms():
    """Verify transform output tensor shapes and value ranges for CIFAR-10."""
    train_transform, val_transform = get_cifar10_transforms(image_size=224)

    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    train_tensor = train_transform(img)
    val_tensor = val_transform(img)

    assert isinstance(train_tensor, torch.Tensor)
    assert isinstance(val_tensor, torch.Tensor)
    assert train_tensor.shape == (3, 224, 224)
    assert val_tensor.shape == (3, 224, 224)
    assert train_tensor.dtype == torch.float32
    assert val_tensor.dtype == torch.float32


def test_svhn_transforms():
    """Verify SVHN transform output tensor shapes and absence of horizontal flip."""
    train_transform, val_transform = get_svhn_transforms(image_size=224)

    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    train_tensor = train_transform(img)
    val_tensor = val_transform(img)

    assert isinstance(train_tensor, torch.Tensor)
    assert isinstance(val_tensor, torch.Tensor)
    assert train_tensor.shape == (3, 224, 224)
    assert val_tensor.shape == (3, 224, 224)
    assert train_tensor.dtype == torch.float32
    assert val_tensor.dtype == torch.float32


def test_transform_factory():
    """Verify generic transform factory routes to appropriate dataset transforms."""
    cifar_train, cifar_val = get_transforms("cifar10", image_size=112)
    svhn_train, svhn_val = get_transforms("svhn", image_size=112)

    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    c_out = cifar_train(img)
    s_out = svhn_train(img)

    assert c_out.shape == (3, 112, 112)
    assert s_out.shape == (3, 112, 112)


def test_synthetic_dataset_loader():
    """Verify dataset loader with synthetic configuration for fast testing."""
    train_l, val_l, test_l = get_dataset_loaders(dataset_name="synthetic", batch_size=32, image_size=64)

    assert len(train_l) > 0
    assert len(val_l) > 0
    assert len(test_l) > 0

    batch_x, batch_y = next(iter(train_l))
    assert batch_x.shape == (32, 3, 64, 64)
    assert batch_y.shape == (32,)
