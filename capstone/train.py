# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch

from capstone.model import ModelConfig, Transformer
from polkagris import set_seed
from polkagris.data import get_device


def synthetic_batches(vocab_size: int, seq_len: int, batch_size: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield torch.randint(0, vocab_size, (batch_size, seq_len + 1), generator=g)


def lm_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    targets = tokens[:, 1:]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default="reference", choices=["reference", "triton"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--data", default="synthetic", choices=["synthetic", "shakespeare"])
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    set_seed()
    device = get_device()
    if args.data == "shakespeare":
        from capstone.data import CharCorpus

        corpus = CharCorpus()
        vocab_size = corpus.vocab_size
        batches = corpus.batches("train", args.seq, args.batch_size)
    else:
        vocab_size = args.vocab
        batches = synthetic_batches(vocab_size, args.seq, args.batch_size)

    cfg = ModelConfig(
        vocab_size=vocab_size, dim=args.dim, n_layers=args.layers,
        n_heads=args.heads, seq_len=args.seq, ops=args.ops,
    )
    model = Transformer(cfg).to(device)
    print(f"params: {model.num_params() / 1e6:.1f}M  ops: {args.ops}  "
          f"device: {device.type}  data: {args.data}  vocab: {vocab_size}")
    if args.compile:
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    for step in range(args.steps):
        tokens = next(batches).to(device)
        tokens = tokens[:, : cfg.seq_len + 1]
        if args.bf16:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = lm_loss(model(tokens[:, :-1]), tokens)
        else:
            loss = lm_loss(model(tokens[:, :-1]), tokens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == args.steps - 1:
            ppl = math.exp(min(20.0, loss.item()))
            print(f"step {step:4d}  loss {loss.item():.4f}  ppl {ppl:.1f}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        # asdict, not the dataclass: keeps the checkpoint loadable with
        # weights_only=True so scoring never has to unpickle.
        torch.save({"config": asdict(cfg), "model": model.state_dict()}, args.save)
        print(f"saved to {args.save}")


if __name__ == "__main__":
    main()
