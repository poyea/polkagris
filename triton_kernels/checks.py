# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m triton_kernels.checks"""

from __future__ import annotations

import math

import torch

from polkagris import set_seed
from polkagris.checks import Pending, Skip, run


def have_triton() -> tuple[object, object]:
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise Skip("triton is not installed") from exc
    if not torch.cuda.is_available():
        raise Skip("triton needs a CUDA device")
    return triton, tl


def online_softmax_matches_the_two_pass_version() -> str:
    set_seed()
    x = torch.randn(4096) * 10.0
    reference = torch.softmax(x, dim=0)

    running_max, running_sum = float("-inf"), 0.0
    for chunk in x.split(256):
        chunk_max = chunk.max().item()
        new_max = max(running_max, chunk_max)
        running_sum = running_sum * math.exp(running_max - new_max)
        running_sum += torch.exp(chunk - new_max).sum().item()
        running_max = new_max

    streamed = torch.exp(x - running_max) / running_sum
    assert torch.allclose(reference, streamed, atol=1e-6), "the streaming pass drifted"
    return "one streaming pass equals the two-pass softmax; this is what the kernels rely on"


def naive_softmax_overflows_without_the_max_subtraction() -> str:
    x = torch.tensor([1000.0, 1001.0])
    naive = torch.exp(x) / torch.exp(x).sum()
    stable = torch.softmax(x, dim=0)
    assert naive.isnan().any(), f"expected overflow, got {naive}"
    assert stable.isfinite().all(), "the stable version should survive"
    return f"exp(1000) overflows to {naive.tolist()}; subtracting the max gives {stable.tolist()}"


def a_causal_mask_halves_the_work() -> str:
    seq = 1024
    full = seq * seq
    causal = seq * (seq + 1) // 2
    ratio = causal / full
    assert 0.4 < ratio < 0.6, f"expected about half, got {ratio}"
    return f"{ratio:.1%} of the score matrix matters, so skipping whole tiles above the diagonal pays"


def tile_choice_changes_the_memory_traffic() -> str:
    seq, dim = 1024, 64

    def traffic(block_m: int, block_n: int) -> float:
        tiles_m, tiles_n = seq / block_m, seq / block_n
        # each M tile streams every K/V tile it needs
        return tiles_m * tiles_n * (block_n * dim * 2) * 4

    wide, tall = traffic(32, 128), traffic(128, 32)
    assert wide != tall, "the two orientations should not cost the same"
    return f"BLOCK_M=32,BLOCK_N=128 moves {wide/1e6:.1f} MB against {tall/1e6:.1f} MB reversed"


def the_kernels_need_a_triton_box() -> str:
    triton, _ = have_triton()
    return f"triton {triton.__version__} present; run triton_kernels.softmax for real numbers"


def block_size_sweep() -> str:
    raise Pending("BLOCK sweep timings are unmeasured")


def tiling_asymmetry_measured() -> str:
    raise Pending("the BLOCK_M/BLOCK_N asymmetry is not backed by counters yet")


CHECKS = [
    online_softmax_matches_the_two_pass_version,
    naive_softmax_overflows_without_the_max_subtraction,
    a_causal_mask_halves_the_work,
    tile_choice_changes_the_memory_traffic,
    the_kernels_need_a_triton_box,
    block_size_sweep,
    tiling_asymmetry_measured,
]

if __name__ == "__main__":
    raise SystemExit(run("triton_kernels", CHECKS))
