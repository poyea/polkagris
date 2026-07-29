# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synthetic_loaders(
    shape: tuple[int, ...],
    classes: int,
    batch_size: int,
    n_train: int = 512,
    n_test: int = 256,
) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(0)
    w = torch.randn(int(torch.tensor(shape).prod()), classes, generator=g)

    def make(n: int) -> TensorDataset:
        x = torch.randn(n, *shape, generator=g)
        y = (x.flatten(1) @ w).argmax(1)
        return TensorDataset(x, y)

    return (
        DataLoader(make(n_train), batch_size=batch_size, shuffle=True),
        DataLoader(make(n_test), batch_size=batch_size),
    )


def mnist_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size),
    )


def cifar_loaders(batch_size: int, augment: bool = True) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose(
        ([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else [])
        + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=test_tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test, batch_size=batch_size, num_workers=2),
    )
