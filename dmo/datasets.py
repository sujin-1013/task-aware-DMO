"""Datasets for DMO scenarios. All images resized to 72x72, grayscale converted to 3 channels."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMG_SIZE = 72
NORM = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def _color_tf(train: bool):
    aug = [transforms.RandomCrop(IMG_SIZE, padding=8), transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), *aug, transforms.ToTensor(), NORM])


def _gray_tf(train: bool):
    aug = [transforms.RandomCrop(IMG_SIZE, padding=8)] if train else []
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), *aug,
        transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), NORM,
    ])


def build_task(name: str, root: str, train: bool):
    if name == "cifar10":
        return datasets.CIFAR10(root, train=train, download=True, transform=_color_tf(train))
    if name == "stl10":
        return datasets.STL10(root, split="train" if train else "test", download=True, transform=_color_tf(train))
    if name == "mnist":
        return datasets.MNIST(root, train=train, download=True, transform=_gray_tf(train))
    if name == "usps":
        return datasets.USPS(root, train=train, download=True, transform=_gray_tf(train))
    raise ValueError(name)


TASK_CLASSES = {"cifar10": 10, "stl10": 10, "mnist": 10, "usps": 10}
SCENARIO1_TASKS = ["cifar10", "stl10", "usps", "mnist"]


def loaders(name: str, root: str, batch_size: int, workers: int = 4):
    tr = DataLoader(build_task(name, root, True), batch_size=batch_size, shuffle=True,
                    num_workers=workers, pin_memory=True, drop_last=False)
    te = DataLoader(build_task(name, root, False), batch_size=256, shuffle=False,
                    num_workers=workers, pin_memory=True)
    return tr, te


def eval_train_loader(name: str, root: str, batch_size: int = 256, workers: int = 4):
    """Training set without shuffling/augmentation, for per-sample loss measurement."""
    ds = build_task(name, root, True)
    ds.transform = _color_tf(False) if name in ("cifar10", "stl10") else _gray_tf(False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
