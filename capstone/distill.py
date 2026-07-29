# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import argparse
import sys

import torch
import torch.nn.functional as F

from capstone.model import ModelConfig, Transformer
from capstone.train import synthetic_batches
from polkagris import set_seed
from polkagris.data import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=2.0)
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("transformers is not installed (uv sync --extra post), skipping")
        sys.exit(0)

    set_seed()
    device = get_device()
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher).to(device).eval()
    vocab = teacher.config.vocab_size
    student = Transformer(
        ModelConfig(vocab_size=vocab, dim=256, n_layers=4, n_heads=4, seq_len=args.seq)
    ).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4)
    batches = synthetic_batches(vocab, args.seq, args.batch_size)
    temp = args.temperature

    for step in range(args.steps):
        tokens = next(batches)[:, : args.seq].to(device)
        with torch.no_grad():
            t_logits = teacher(tokens).logits
        s_logits = student(tokens)
        loss = F.kl_div(
            F.log_softmax(s_logits / temp, dim=-1),
            F.softmax(t_logits / temp, dim=-1),
            reduction="batchmean",
        ) * temp**2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  kl {loss.item():.4f}")


if __name__ == "__main__":
    main()
