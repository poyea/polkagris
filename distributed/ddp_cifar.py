# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import argparse
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from polkagris import set_seed
from polkagris.data import synthetic_loaders
from training.cifar_cnn import build_model
from training.config import TrainConfig
from training.loop import evaluate, make_optimizer, train_one_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if "RANK" in os.environ:
        dist.init_process_group(backend=backend)
    else:
        print("launch with: torchrun --nproc_per_node=2 -m distributed.ddp_cifar")
        print("running single-process fallback")
        os.environ.setdefault("LOCAL_RANK", "0")
        store_file = Path(tempfile.mkdtemp()) / "ddp_store"
        dist.init_process_group(
            backend=backend,
            init_method=f"file:///{store_file.as_posix()}",
            rank=0,
            world_size=1,
        )
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = (
        torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    )

    set_seed()
    cfg = TrainConfig(epochs=args.epochs)
    train_loader, test_loader = synthetic_loaders((3, 32, 32), 10, cfg.batch_size)
    sampler = DistributedSampler(train_loader.dataset)
    train_loader = DataLoader(train_loader.dataset, batch_size=cfg.batch_size, sampler=sampler)

    model = build_model().to(device)
    model = DistributedDataParallel(
        model, device_ids=[local_rank] if device.type == "cuda" else None
    )
    opt = make_optimizer(model, cfg)
    total_steps = cfg.epochs * len(train_loader)
    step = 0
    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        step, loss = train_one_epoch(model, train_loader, opt, cfg, device, step, total_steps)
        if rank == 0:
            acc = evaluate(model, test_loader, device)
            print(f"epoch {epoch + 1}  loss {loss:.4f}  test_acc {acc:.4f}  "
                  f"world_size {dist.get_world_size()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
