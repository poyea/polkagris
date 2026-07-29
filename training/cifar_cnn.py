# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import argparse

from torch import nn

from polkagris import set_seed
from polkagris.data import cifar_loaders, get_device, synthetic_loaders
from training.config import TrainConfig
from training.loop import fit


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def build_model(dropout: float = 0.1) -> nn.Module:
    return nn.Sequential(
        conv_block(3, 64),
        conv_block(64, 64),
        nn.MaxPool2d(2),
        conv_block(64, 128),
        conv_block(128, 128),
        nn.MaxPool2d(2),
        conv_block(128, 256),
        conv_block(256, 256),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(256, 10),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    set_seed()
    cfg = TrainConfig(epochs=1 if args.smoke else args.epochs)
    if args.smoke:
        train_loader, test_loader = synthetic_loaders((3, 32, 32), 10, cfg.batch_size)
    else:
        train_loader, test_loader = cifar_loaders(cfg.batch_size)
    acc = fit(build_model(cfg.dropout), train_loader, test_loader, cfg, get_device())
    target = 0.80
    print(f"final accuracy {acc:.4f} (target {target} on real CIFAR-10)")


if __name__ == "__main__":
    main()
