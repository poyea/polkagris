# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.config import TrainConfig


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def lr_at(step: int, total_steps: int, cfg: TrainConfig) -> float:
    warmup = min(cfg.warmup_steps, total_steps)
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    if cfg.lr_schedule == "constant":
        return cfg.lr
    t = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * t))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    cfg: TrainConfig,
    device: torch.device,
    step: int,
    total_steps: int,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[int, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    loss = torch.tensor(0.0)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        lr = lr_at(step, total_steps, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        ctx = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype
            else nullcontext()
        )
        with ctx:
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
    return step, float(loss.item())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += int((model(x).argmax(1) == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
) -> float:
    model.to(device)
    opt = make_optimizer(model, cfg)
    total_steps = cfg.epochs * len(train_loader)
    step = 0
    acc = 0.0
    for epoch in range(cfg.epochs):
        step, loss = train_one_epoch(
            model, train_loader, opt, cfg, device, step, total_steps, autocast_dtype
        )
        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch + 1}/{cfg.epochs}  loss {loss:.4f}  test_acc {acc:.4f}")
    return acc
