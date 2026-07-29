# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import argparse

import torch

from polkagris import benchmark, set_seed
from polkagris.data import get_device, synthetic_loaders
from training.cifar_cnn import build_model
from training.config import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    set_seed()
    device = get_device()
    cfg = TrainConfig()
    model = build_model().to(device).eval()
    train_loader, _ = synthetic_loaders((3, 32, 32), 10, cfg.batch_size)
    x, _ = next(iter(train_loader))
    x = x.to(device)

    with torch.no_grad():
        eager = benchmark(lambda: model(x), "compile_eager", warmup=3, reps=args.reps)

    backend = args.backend
    try:
        compiled = torch.compile(model, backend=backend)
        with torch.no_grad():
            compiled(x)
    except Exception as e:
        print(f"backend {backend!r} unavailable ({type(e).__name__}), falling back to aot_eager")
        backend = "aot_eager"
        compiled = torch.compile(model, backend=backend)
        with torch.no_grad():
            compiled(x)

    with torch.no_grad():
        comp = benchmark(lambda: compiled(x), f"compile_{backend}", warmup=3, reps=args.reps)

    print(f"eager    {eager.mean_ms:.2f} ms")
    print(f"{backend:8s} {comp.mean_ms:.2f} ms  ({eager.mean_ms / comp.mean_ms:.2f}x)")
    print("set TORCH_LOGS=output_code to dump generated code")


if __name__ == "__main__":
    main()
