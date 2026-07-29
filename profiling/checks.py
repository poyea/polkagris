# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m profiling.checks"""

from __future__ import annotations

import time

import torch
from torch import nn

from polkagris import set_seed
from polkagris.checks import Pending, Skip, run


def intensity(flops: float, bytes_moved: float) -> float:
    return flops / bytes_moved


def vector_add_is_memory_bound_by_arithmetic_alone() -> str:
    n = 1 << 20
    # two loads and one store per element, one add
    ai = intensity(n, 3 * n * 4)
    assert ai < 0.1, f"expected a tiny ratio, got {ai}"
    return f"{ai:.3f} FLOP/byte: nothing to compute per byte moved, so DRAM sets the speed"


def matmul_intensity_grows_with_size() -> str:
    def ai(n: int) -> float:
        return intensity(2.0 * n**3, 3.0 * n * n * 4)

    small, large = ai(128), ai(4096)
    assert large > small * 20, f"expected growth, got {small:.1f} then {large:.1f}"
    return f"{small:.1f} FLOP/byte at n=128 rises to {large:.1f} at n=4096; tiling is what captures it"


def the_naive_matmul_throws_its_reuse_away() -> str:
    n = 1024
    ideal = intensity(2.0 * n**3, 3.0 * n * n * 4)
    # one thread per output re-reads a full row and column from global memory
    naive = intensity(2.0 * n**3, 2.0 * n**3 * 4)
    assert naive < 1.0 < ideal, f"naive {naive}, ideal {ideal}"
    return f"naive reads give {naive:.2f} FLOP/byte against a possible {ideal:.0f}: that gap is phase 07"


def the_profiler_names_the_dominant_operator() -> str:
    set_seed()
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(16 * 8 * 8, 10))
    x = torch.randn(8, 3, 8, 8)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        model(x).sum().backward()
    events = [e for e in prof.key_averages() if e.self_cpu_time_total > 0]
    assert events, "the profiler recorded nothing"
    top = max(events, key=lambda e: e.self_cpu_time_total)
    return f"{len(events)} ops recorded, heaviest self time is {top.key}"


def the_first_call_pays_for_setup() -> str:
    set_seed()
    x = torch.randn(512, 512)

    def timed() -> float:
        t0 = time.perf_counter()
        x @ x
        return (time.perf_counter() - t0) * 1e3

    first = timed()
    rest = [timed() for _ in range(5)]
    assert min(rest) > 0.0, "timing resolution is too coarse to see anything"
    return f"first {first:.2f} ms, best of five after {min(rest):.2f} ms: always warm up before timing"


def a_gpu_trace_needs_a_gpu() -> str:
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    return f"{torch.cuda.get_device_name(0)} present, so CUDA columns will populate"


def top_kernels_classified() -> str:
    raise Pending("the ncu pass that labels the top five kernels has not been run")


CHECKS = [
    vector_add_is_memory_bound_by_arithmetic_alone,
    matmul_intensity_grows_with_size,
    the_naive_matmul_throws_its_reuse_away,
    the_profiler_names_the_dominant_operator,
    the_first_call_pays_for_setup,
    a_gpu_trace_needs_a_gpu,
    top_kernels_classified,
]

if __name__ == "__main__":
    raise SystemExit(run("profiling", CHECKS))
