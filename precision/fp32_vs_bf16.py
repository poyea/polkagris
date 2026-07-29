# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import argparse

import torch

from polkagris import benchmark, set_seed
from polkagris.data import get_device, synthetic_loaders
from training.cifar_cnn import build_model
from training.config import TrainConfig
from training.loop import make_optimizer


def one_step_fn(model, opt, loss_fn, x, y, device, dtype, scaler=None):
    def step():
        if dtype is None:
            loss = loss_fn(model(x), y)
        else:
            with torch.autocast(device_type=device.type, dtype=dtype):
                loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    return step


def run(dtype: torch.dtype | None, label: str, reps: int) -> None:
    set_seed()
    device = get_device()
    cfg = TrainConfig()
    model = build_model().to(device)
    opt = make_optimizer(model, cfg)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader, _ = synthetic_loaders((3, 32, 32), 10, cfg.batch_size)
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    scaler = None
    if dtype is torch.float16 and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = benchmark(
        one_step_fn(model, opt, loss_fn, x, y, device, dtype, scaler),
        f"precision_cifar_{label}",
        warmup=3,
        reps=reps,
    )
    mem = torch.cuda.max_memory_allocated() // 2**20 if device.type == "cuda" else None
    print(f"{label}: {result.mean_ms:.2f} ms/step (+/- {result.std_ms:.2f})"
          + (f", peak mem {mem} MiB" if mem is not None else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    run(None, "fp32", args.reps)
    run(torch.bfloat16, "bf16", args.reps)
    if torch.cuda.is_available():
        run(torch.float16, "fp16", args.reps)


if __name__ == "__main__":
    main()
