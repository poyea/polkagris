# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import argparse
import os
import sys

import torch
import torch.distributed as dist

from capstone.model import ModelConfig, Transformer
from capstone.train import lm_loss, synthetic_batches
from polkagris import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    args = parser.parse_args()

    if "RANK" not in os.environ:
        print("launch with: torchrun --nproc_per_node=2 -m distributed.fsdp_lm")
        sys.exit(0)
    if not torch.cuda.is_available():
        print("fsdp run needs CUDA devices, skipping")
        sys.exit(0)

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    set_seed()
    cfg = ModelConfig(
        vocab_size=8192, dim=args.dim, n_layers=args.layers, n_heads=8, seq_len=512
    )
    model = Transformer(cfg).to(device)
    model = FSDP(
        model,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
        device_id=local_rank,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batches = synthetic_batches(cfg.vocab_size, cfg.seq_len, 4)

    for step in range(args.steps):
        tokens = next(batches)[:, : cfg.seq_len + 1].to(device)
        loss = lm_loss(model(tokens[:, :-1]), tokens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if dist.get_rank() == 0 and step % 5 == 0:
            mem = torch.cuda.max_memory_allocated() // 2**20
            print(f"step {step:3d}  loss {loss.item():.4f}  peak_mem {mem} MiB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
