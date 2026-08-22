# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Continuous batching over a decoder-only model.

The scheduler owns the batch. Requests arrive, wait for a slot, decode one
token per step alongside whatever else is running, and free their slot the
moment they finish rather than at the end of the batch.

The model is passed in and only has to accept
`model(tokens, cache, positions, key_mask) -> logits`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

import torch


@dataclass
class Request:
    """One unit of work handed to the scheduler."""

    prompt: list[int]
    max_new_tokens: int = 32
    temperature: float = 0.0
    stop_token: int | None = None
    id: int = field(default_factory=count().__next__)


@dataclass
class Sequence:
    """A request that holds a slot, plus the state that decoding advances."""

    request: Request
    slot: int
    position: int
    tokens: list[int]
    done: bool = False

    @property
    def generated(self) -> list[int]:
        return self.tokens[len(self.request.prompt):]


class Scheduler:
    """Fixed pool of slots, filled from a queue as they free up.

    `capacity` bounds how many sequences decode at once; `max_seq_len` bounds
    how far any one of them can run.
    """

    def __init__(self, model, capacity: int, max_seq_len: int, device=None):
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self.model = model
        self.capacity = capacity
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cpu")
        self.waiting: list[Request] = []
        self.running: dict[int, Sequence] = {}
        self.finished: list[Sequence] = []
        self._free = list(range(capacity))

    def submit(self, request: Request) -> None:
        if len(request.prompt) >= self.max_seq_len:
            raise ValueError(
                f"prompt of {len(request.prompt)} tokens does not fit in {self.max_seq_len}"
            )
        self.waiting.append(request)

    @property
    def pending(self) -> int:
        return len(self.waiting) + len(self.running)

    def admit(self) -> list[Sequence]:
        """Move waiting requests into free slots. Prefill is not batched yet:
        each admission runs its own prompt pass before joining the decode batch."""
        admitted = []
        while self.waiting and self._free:
            request = self.waiting.pop(0)
            sequence = Sequence(
                request=request,
                slot=self._free.pop(0),
                position=len(request.prompt),
                tokens=list(request.prompt),
            )
            self.running[sequence.slot] = sequence
            admitted.append(sequence)
        return admitted

    def retire(self, sequence: Sequence) -> None:
        sequence.done = True
        self.running.pop(sequence.slot, None)
        self._free.append(sequence.slot)
        self.finished.append(sequence)

    def _is_finished(self, sequence: Sequence) -> bool:
        request = sequence.request
        if len(sequence.generated) >= request.max_new_tokens:
            return True
        if sequence.position >= self.max_seq_len:
            return True
        return request.stop_token is not None and sequence.tokens[-1] == request.stop_token

    def step(self) -> list[Sequence]:
        """Advance every running sequence by one token."""
        raise NotImplementedError("decode step: batched cache and sampling")

    def run(self) -> list[Sequence]:
        """Drain the queue and return finished sequences in completion order."""
        raise NotImplementedError("driver over admit/step/retire")
