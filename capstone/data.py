# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch

from polkagris.data import DATA_ROOT

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


class CharCorpus:
    def __init__(self, path: Path | None = None, val_fraction: float = 0.1):
        path = path or DATA_ROOT / "tinyshakespeare.txt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {SHAKESPEARE_URL}")
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        text = path.read_text(encoding="utf-8")
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = dict(enumerate(chars))
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        split = int(len(data) * (1 - val_fraction))
        self.train_data = data[:split]
        self.val_data = data[split:]

    def batches(self, split: str, seq_len: int, batch_size: int, seed: int = 0):
        data = self.train_data if split == "train" else self.val_data
        g = torch.Generator().manual_seed(seed)
        while True:
            idx = torch.randint(len(data) - seq_len - 1, (batch_size,), generator=g)
            yield torch.stack([data[i : i + seq_len + 1] for i in idx])

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join(self.itos[int(t)] for t in tokens)
