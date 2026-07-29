# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import torch
from torch import nn

from polkagris.data import synthetic_loaders
from training.config import TrainConfig
from training.loop import fit, lr_at, make_optimizer


def test_lr_warmup_and_decay():
    cfg = TrainConfig(lr=1.0, warmup_steps=10, lr_schedule="cosine")
    assert lr_at(0, 100, cfg) < 0.2
    assert abs(lr_at(9, 100, cfg) - 1.0) < 1e-9
    assert lr_at(99, 100, cfg) < 0.01


def test_optimizers_construct():
    model = nn.Linear(4, 2)
    for name in ("sgd", "adam", "adamw"):
        cfg = TrainConfig(optimizer=name)
        assert make_optimizer(model, cfg) is not None


def test_fit_learns_synthetic():
    torch.manual_seed(0)
    train_loader, test_loader = synthetic_loaders((4,), 3, batch_size=32, n_train=256, n_test=128)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 3))
    cfg = TrainConfig(epochs=15, lr=1e-2, warmup_steps=5, weight_decay=0.0)
    acc = fit(model, train_loader, test_loader, cfg, torch.device("cpu"))
    assert acc > 0.8
