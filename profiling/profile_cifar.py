# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from polkagris import set_seed
from polkagris.data import get_device, synthetic_loaders
from training.cifar_cnn import build_model
from training.config import TrainConfig
from training.loop import make_optimizer

OUT_DIR = Path(__file__).resolve().parent / "traces"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    set_seed()
    device = get_device()
    cfg = TrainConfig()
    model = build_model().to(device)
    opt = make_optimizer(model, cfg)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader, _ = synthetic_loaders((3, 32, 32), 10, cfg.batch_size)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    it = iter(train_loader)
    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        for _ in range(args.steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
    OUT_DIR.mkdir(exist_ok=True)
    trace = OUT_DIR / "cifar_trace.json"
    prof.export_chrome_trace(str(trace))
    print(f"chrome trace written to {trace}")


if __name__ == "__main__":
    main()
