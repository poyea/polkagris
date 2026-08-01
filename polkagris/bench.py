# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""One benchmark harness for the whole repo, so numbers stay comparable.

Fixed seeds, warmup, CUDA-event timing when available, JSON output into
benchmarks/results/ (one file per named run). Every reported number goes
through here, and the results are never hand-edited.

Name runs `<area>_<what>_<variant>`, e.g. `triton_softmax_fp16` or
`precision_cifar_bf16`: the leading segment is what groups them.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results"


def set_seed(seed: int = 1859) -> None:  # 1859: first polkagris pulled in Gränna
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class BenchResult:
    name: str
    mean_ms: float
    std_ms: float
    reps: int
    device: str
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, out_dir: Path | None = None) -> Path:
        directory = out_dir or BENCH_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def benchmark(
    fn: Callable[[], Any],
    name: str,
    warmup: int = 10,
    reps: int = 50,
    save: bool = True,
    out_dir: Path | None = None,
    **extra: Any,
) -> BenchResult:
    """Time fn() with warmup. Uses CUDA events on GPU, perf_counter on CPU.

    `save=False` times without touching disk; `out_dir` redirects the JSON, so a
    test never writes into the results directory the dashboard reads.
    """
    use_cuda = torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    else:
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
    result = BenchResult(
        name=name,
        mean_ms=statistics.fmean(times),
        std_ms=statistics.stdev(times) if reps > 1 else 0.0,
        reps=reps,
        device="cuda" if use_cuda else "cpu",
        extra=extra,
    )
    if save:
        result.save(out_dir)
    return result
