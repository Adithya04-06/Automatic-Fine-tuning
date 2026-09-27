"""Dataset loading and train/val/test data loader creation with multi-dataset support (CIFAR-10, SVHN, Synthetic)."""

import os
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from torchvision.datasets import CIFAR10, SVHN

from src.data.transforms import get_cifar10_transforms, get_svhn_transforms, get_transforms


class TransformedSubset(Dataset):
    """A dataset wrapper that applies a transform to a subset of an underlying dataset."""
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class SyntheticImageDataset(Dataset):
    """Synthetic dataset generating random image tensors for rapid offline testing and CI/CD validation."""
    def __init__(self, num_samples: int = 500, num_classes: int = 10, image_size: int = 224, seed: int = 42):
        generator = torch.Generator().manual_seed(seed)
        self.data = torch.randn(num_samples, 3, image_size, image_size, generator=generator)
        self.targets = torch.randint(0, num_classes, (num_samples,), generator=generator)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


def get_dataset_loaders(
    dataset_name: str = "cifar10",
    root: str = "./data",
    download: bool = True,
    train_val_split: float = 0.8,
    batch_size: int = 64,
    num_workers: int = 0,
    image_size: int = 224,
    seed: int = 42,
    subset_fraction: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory creating Train, Validation, and Test DataLoaders for supported datasets.
    
    Supported datasets:
    - 'cifar10': Natural object recognition (10 classes, 50k train/val, 10k test).
    - 'svhn': Street View House Numbers digit recognition (10 classes, 73k train/val, 26k test).
    - 'synthetic': Random Gaussian image tensors for unit tests.
    """
    name = dataset_name.lower().strip()

    if name == "synthetic":
        train_ds = SyntheticImageDataset(num_samples=400, num_classes=10, image_size=image_size, seed=seed)
        val_ds = SyntheticImageDataset(num_samples=100, num_classes=10, image_size=image_size, seed=seed + 1)
        test_ds = SyntheticImageDataset(num_samples=100, num_classes=10, image_size=image_size, seed=seed + 2)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader

    # Select dataset-specific transforms
    train_transform, val_transform = get_transforms(dataset_name=name, image_size=image_size)

    if name in ("cifar10", "cifar-10"):
        raw_train_val = CIFAR10(root=root, train=True, download=download, transform=None)
        raw_test = CIFAR10(root=root, train=False, download=download, transform=val_transform)
    elif name in ("svhn", "street_view_house_numbers"):
        raw_train_val = SVHN(root=root, split="train", download=download, transform=None)
        raw_test = SVHN(root=root, split="test", download=download, transform=val_transform)
    else:
        raise ValueError(f"Unsupported dataset: '{dataset_name}'. Supported datasets: ['cifar10', 'svhn', 'synthetic']")

    total_len = len(raw_train_val)
    train_len = int(total_len * train_val_split)
    val_len = total_len - train_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(raw_train_val, [train_len, val_len], generator=generator)

    if subset_fraction is not None and 0.0 < subset_fraction < 1.0:
        sub_train_len = max(int(len(train_subset) * subset_fraction), batch_size)
        sub_val_len = max(int(len(val_subset) * subset_fraction), batch_size)
        train_subset, _ = random_split(train_subset, [sub_train_len, len(train_subset) - sub_train_len], generator=generator)
        val_subset, _ = random_split(val_subset, [sub_val_len, len(val_subset) - sub_val_len], generator=generator)

    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset = TransformedSubset(val_subset, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        raw_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


# Backward compatibility alias
get_cifar10_loaders = get_dataset_loaders
