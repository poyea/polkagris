# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Preallocated key/value storage, one row per scheduler slot.

Growing a cache by concatenation copies the whole history on every step. Here
each slot owns a fixed window written in place, so a step costs the same at
token 500 as at token 5.

The batch dimension is the whole pool rather than the live subset: attention
reads the buffer where it lies instead of gathering the live rows into a
smaller tensor, which would copy the history back. Idle rows are carried
through the arithmetic and excluded by the mask, so the pool is sized for the
concurrency it is meant to serve.
"""

from __future__ import annotations

import torch


class SlotCache:
    def __init__(
        self,
        n_layers: int,
        capacity: int,
        n_heads: int,
        max_seq_len: int,
        head_dim: int,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        shape = (capacity, n_heads, max_seq_len, head_dim)
        self.capacity = capacity
        self.max_seq_len = max_seq_len
        self.keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.values = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.lengths = torch.zeros(capacity, dtype=torch.long, device=device)
        self._start: torch.Tensor | None = None
        self._width = 0

    def reset(self, slot: int) -> None:
        """Hand a slot back. Stale keys stay in the buffer, out of reach of the
        mask, and are overwritten by whatever is admitted next."""
        self.lengths[slot] = 0

    def reserve(self, width: int, active: torch.Tensor | None = None) -> None:
        """Claim `width` positions in every active row, before the layers write.

        Lengths move first so the mask covers the tokens this pass is about to
        add. Inactive rows keep their length, so the row they write is the one
        position their own mask already excludes.
        """
        if width < 1:
            raise ValueError(f"width must be at least 1, got {width}")
        if active is None:
            active = torch.ones(self.capacity, dtype=torch.bool, device=self.lengths.device)
        over = self.lengths[active] + width > self.max_seq_len
        if bool(over.any()):
            raise ValueError(f"slot would run past the {self.max_seq_len} token window")
        self._start = self.lengths.clone()
        self._width = width
        self.lengths = self.lengths + width * active.long()

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this layer's new keys and values, then return the whole buffer."""
        if self._start is None:
            raise RuntimeError("reserve() must claim positions before a layer appends")
        if k.shape[-2] != self._width:
            raise ValueError(f"reserved {self._width} positions, got {k.shape[-2]}")
        rows = torch.arange(self.capacity, device=k.device)[:, None]
        cols = self._start[:, None] + torch.arange(self._width, device=k.device)
        self.keys[layer][rows, :, cols, :] = k.transpose(1, 2).to(self.keys[layer].dtype)
        self.values[layer][rows, :, cols, :] = v.transpose(1, 2).to(self.values[layer].dtype)
        return self.keys[layer], self.values[layer]

    def key_mask(self) -> torch.Tensor:
        """[capacity, max_seq_len], True where a position holds a real key."""
        window = torch.arange(self.max_seq_len, device=self.lengths.device)
        return window[None, :] < self.lengths[:, None]
