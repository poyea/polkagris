# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Continuous batching over a decoder-only model.

The scheduler owns the batch. Requests arrive, wait for a slot, decode one
token per step alongside whatever else is running, and free their slot the
moment they finish rather than at the end of the batch.

The model is passed in and only has to accept
`model(tokens, cache, positions, key_mask) -> logits`. Sizing the pool also
needs `model.cfg`, unless a `SlotCache` is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

import torch

from serving.cache import SlotCache


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


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
    tokens: list[int]
    done: bool = False
    # Whether this slot's prompt is in the cache. Kept here rather than read
    # back from cache.lengths, which would sync the device once per sequence
    # per step just to choose a code path.
    prefilled: bool = False

    @property
    def position(self) -> int:
        """Next write index, which is also how many tokens the slot holds."""
        return len(self.tokens)

    @property
    def generated(self) -> list[int]:
        return self.tokens[len(self.request.prompt):]


class Scheduler:
    """Fixed pool of slots, filled from a queue as they free up.

    `capacity` bounds how many sequences decode at once; `max_seq_len` bounds
    how far any one of them can run.
    """

    def __init__(self, model, capacity: int, max_seq_len: int, device=None, cache=None):
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self.model = model
        self.capacity = capacity
        self.max_seq_len = max_seq_len
        # A cache built on the host while the weights sit on the device would
        # copy every key across the bus twice per layer per step, so default to
        # wherever the parameters already are.
        self.device = device or _model_device(model)
        self.cache = cache
        self.waiting: list[Request] = []
        self.running: dict[int, Sequence] = {}
        self.finished: list[Sequence] = []
        self._free = list(range(capacity))

    def submit(self, request: Request) -> None:
        if not request.prompt:
            raise ValueError("prompt is empty: decoding needs a token to condition on")
        if request.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1, got {request.max_new_tokens}")
        if len(request.prompt) >= self.max_seq_len:
            raise ValueError(
                f"prompt of {len(request.prompt)} tokens does not fit in {self.max_seq_len}"
            )
        self.waiting.append(request)

    @property
    def pending(self) -> int:
        return len(self.waiting) + len(self.running)

    def admit(self) -> list[Sequence]:
        """Move waiting requests into free slots.

        Allocation only. Filling a new slot's cache with its prompt is the
        decode loop's work, and happens on the step after admission.
        """
        admitted = []
        while self.waiting and self._free:
            request = self.waiting.pop(0)
            sequence = Sequence(
                request=request,
                slot=self._free.pop(0),
                tokens=list(request.prompt),
            )
            self.running[sequence.slot] = sequence
            admitted.append(sequence)
        return admitted

    def retire(self, sequence: Sequence) -> None:
        # Keyed on identity, not on the slot number. A stale copy carrying the
        # same slot would otherwise evict the sequence still decoding on it, and
        # a second retire would hand the same slot out twice.
        if self.running.get(sequence.slot) is not sequence:
            raise ValueError(f"slot {sequence.slot} is not running this sequence")
        sequence.done = True
        del self.running[sequence.slot]
        self._free.append(sequence.slot)
        self.finished.append(sequence)

    def _is_finished(self, sequence: Sequence) -> bool:
        request = sequence.request
        if len(sequence.generated) >= request.max_new_tokens:
            return True
        if sequence.position >= self.max_seq_len:
            return True
        return request.stop_token is not None and sequence.tokens[-1] == request.stop_token

    def _build_cache(self) -> SlotCache:
        cfg = self.model.cfg
        try:
            dtype = next(self.model.parameters()).dtype
        except (AttributeError, StopIteration):
            dtype = torch.float32
        return SlotCache(
            n_layers=cfg.n_layers,
            capacity=self.capacity,
            n_heads=cfg.n_heads,
            max_seq_len=self.max_seq_len,
            head_dim=cfg.dim // cfg.n_heads,
            device=self.device,
            dtype=dtype,
        )

    def _pool(self, width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Zeroed pool-width operands: active rows, tokens, positions.

        Every pass is the width of the whole pool, not of the live subset.
        Gathering the live rows into a smaller batch would copy their history
        out of the cache and back, which is the cost the preallocation exists
        to avoid, so idle rows ride through the arithmetic and the mask drops
        them.
        """
        active = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        tokens = torch.zeros((self.capacity, width), dtype=torch.long, device=self.device)
        positions = torch.zeros((self.capacity, width), dtype=torch.long, device=self.device)
        return active, tokens, positions

    def _sample(self, logits: torch.Tensor, temperatures: list[float]) -> list[int]:
        """Pick one token per row, in one pass over the batch.

        Both paths run for every row and the temperature selects between them.
        Branching per sequence instead would turn one set of launches into one
        set per sequence, and the kernels here are far too small to cover that.
        """
        temps = torch.tensor(temperatures, device=logits.device, dtype=torch.float32)
        greedy = logits.argmax(dim=-1)
        probs = (logits.float() / temps.clamp(min=1e-6)[:, None]).softmax(dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)[:, 0]
        # The one device-to-host sync in a step: a token has to reach Python to
        # land in sequence.tokens, so it is paid once for the batch.
        return torch.where(temps > 0, sampled, greedy).tolist()

    def _prefill(self, sequence: Sequence) -> torch.Tensor:
        """Run one whole prompt into its slot, returning its last-token logits."""
        prompt = sequence.request.prompt
        width = len(prompt)
        active, tokens, positions = self._pool(width)
        active[sequence.slot] = True
        tokens[sequence.slot] = torch.tensor(prompt, dtype=torch.long, device=self.device)
        positions += torch.arange(width, device=self.device)
        self.cache.reserve(width, active)
        logits = self.model(tokens, self.cache, positions, self.cache.key_mask())
        sequence.prefilled = True
        return logits[sequence.slot, -1]

    def _decode(self, sequences: list[Sequence]) -> torch.Tensor:
        """Advance every given slot by one token in a single pass."""
        slots = [s.slot for s in sequences]
        active, tokens, positions = self._pool(1)
        active[slots] = True
        tokens[slots, 0] = torch.tensor(
            [s.tokens[-1] for s in sequences], dtype=torch.long, device=self.device
        )
        # The token being fed sits one before the next write index, and rows
        # are at different absolute positions, which is why positions carries a
        # batch dimension here and rope rotates each row by its own angle.
        positions[slots, 0] = torch.tensor(
            [s.position - 1 for s in sequences], dtype=torch.long, device=self.device
        )
        self.cache.reserve(1, active)
        logits = self.model(tokens, self.cache, positions, self.cache.key_mask())
        return logits[slots, -1]

    @torch.no_grad()
    def step(self) -> list[Sequence]:
        """Advance every running sequence by one token.

        Returns the sequences that finished on this pass, with their slots and
        their cache rows already handed back.
        """
        if not self.running:
            return []
        if self.cache is None:
            self.cache = self._build_cache()
        ordered = [self.running[slot] for slot in sorted(self.running)]

        # Prefill and decode are split because they are bound by different
        # things. A prompt is many tokens against one copy of the weights, so
        # it is compute-bound; a decode pass is one token per row, so it is
        # dominated by streaming the weights and the cache and is
        # memory-bound. Folding them together would pad every decode row out
        # to the longest prompt and charge the whole batch for that width.
        fresh = {s.slot for s in ordered if not s.prefilled}
        for sequence in [s for s in ordered if s.slot in fresh]:
            logits = self._prefill(sequence)
            token = self._sample(logits[None], [sequence.request.temperature])
            sequence.tokens.extend(token)

        # A slot prefilled on this pass already produced its token, so it waits
        # for the next one.
        decoding = [s for s in ordered if s.slot not in fresh and not self._is_finished(s)]
        if decoding:
            logits = self._decode(decoding)
            for sequence, token in zip(decoding, self._sample(
                logits, [s.request.temperature for s in decoding]
            )):
                sequence.tokens.append(token)

        retired = [s for s in ordered if self._is_finished(s)]
        for sequence in retired:
            self.cache.reset(sequence.slot)
            self.retire(sequence)
        return retired

    def run(self) -> list[Sequence]:
        """Drain the queue and return finished sequences in completion order.

        Termination needs no iteration bound: every pass appends a token to
        every running sequence, so each one reaches its token budget or the
        context window.
        """
        was_training = getattr(self.model, "training", False)
        if was_training:
            self.model.eval()
        first = len(self.finished)
        try:
            while self.pending:
                self.admit()
                self.step()
        finally:
            if was_training:
                self.model.train()
        return self.finished[first:]
