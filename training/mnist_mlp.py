# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import argparse

from torch import nn

from polkagris import set_seed
from polkagris.data import get_device, mnist_loaders, synthetic_loaders
from training.config import TrainConfig
from training.loop import fit


def build_model(dropout: float = 0.1) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 10),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    set_seed()
    cfg = TrainConfig(epochs=1 if args.smoke else args.epochs, warmup_steps=100)
    if args.smoke:
        train_loader, test_loader = synthetic_loaders((1, 28, 28), 10, cfg.batch_size)
    else:
        train_loader, test_loader = mnist_loaders(cfg.batch_size)
    acc = fit(build_model(cfg.dropout), train_loader, test_loader, cfg, get_device())
    target = 0.98
    print(f"final accuracy {acc:.4f} (target {target} on real MNIST)")


if __name__ == "__main__":
    main()
