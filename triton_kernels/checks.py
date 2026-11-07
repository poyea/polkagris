# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m triton_kernels.checks"""

from __future__ import annotations

import math
import random
import statistics

import torch

from polkagris import set_seed
from polkagris.checks import Skip, run


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


def _median_ms(call, reps: int = 15) -> float:
    for _ in range(4):
        call()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def _interleaved(settings: dict, rounds: int = 3) -> dict:
    """Time each setting several times in shuffled order.

    Timing one setting to completion before the next hands the later ones a
    boosted clock and reads the ramp as an effect of the setting.
    """
    order = list(settings) * rounds
    random.Random(1859).shuffle(order)
    gathered: dict = {key: [] for key in settings}
    for key in order:
        gathered[key].append(_median_ms(settings[key]))
    return {key: statistics.median(samples) for key, samples in gathered.items()}


def block_size_sweep() -> str:
    have_triton()
    from triton_kernels.softmax import triton_softmax

    set_seed()
    rows = 1024
    widths = (128, 512, 2048, 8192)
    inputs = {
        cols: torch.randn(rows, cols, device="cuda", dtype=torch.float16)
        for cols in widths
    }

    settings = {}
    for cols in widths:
        settings[cols, "triton"] = lambda c=cols: triton_softmax(inputs[c])
        settings[cols, "torch"] = lambda c=cols: torch.softmax(inputs[c], dim=1)
    timed = _interleaved(settings)

    ratios, rates = {}, {}
    for cols in widths:
        triton_ms = timed[cols, "triton"]
        torch_ms = timed[cols, "torch"]
        ratios[cols] = torch_ms / triton_ms
        rates[cols] = rows * cols * 2 * 2 / triton_ms / 1e6

    assert ratios[widths[0]] < 1.0, (
        f"one row per program should lose at BLOCK={widths[0]}, got {ratios[widths[0]]:.2f}x"
    )
    assert ratios[widths[-1]] > ratios[widths[0]], "the ratio should improve as the row grows"
    assert rates[widths[-1]] > rates[widths[0]] * 4, "wider rows should move far more bandwidth"

    return (
        f"BLOCK={widths[0]} runs at {ratios[widths[0]]:.2f}x torch and "
        f"{rates[widths[0]]:.0f} GB/s, BLOCK={widths[-1]} at {ratios[widths[-1]]:.2f}x and "
        f"{rates[widths[-1]]:.0f} GB/s: one program per row only pays once the row is wide "
        "enough to cover the launch, and the crossover is where that happens"
    )


def tiling_asymmetry_measured() -> str:
    have_triton()
    from triton_kernels.tiled_attention import tiled_attention

    set_seed()
    heads, seq, dim = 8, 1024, 64
    shape = (heads, seq, dim)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3))

    def predicted_mb(block_m: int, block_n: int) -> float:
        tiles_m, tiles_n = seq / block_m, seq / block_n
        return tiles_m * tiles_n * (block_n * dim * 2) * 4 / 1e6

    pairs = ((32, 32), (32, 128), (128, 32))
    measured = _interleaved(
        {
            pair: (lambda bm=pair[0], bn=pair[1]: tiled_attention(q, k, v, bm, bn))
            for pair in pairs
        },
        rounds=2,
    )

    wide, tall = (32, 128), (128, 32)
    assert predicted_mb(*tall) < predicted_mb(*wide), "the model should favour the tall tile"
    assert measured[tall] > measured[wide], (
        f"the tall tile moves less predicted traffic yet measured "
        f"{measured[tall]:.1f} ms against {measured[wide]:.1f} ms; "
        "if it is now faster the traffic model has stopped being wrong"
    )

    return (
        f"BLOCK_M={tall[0]},BLOCK_N={tall[1]} is predicted to move "
        f"{predicted_mb(*tall):.1f} MB against {predicted_mb(*wide):.1f} MB reversed, yet runs "
        f"{measured[tall]:.1f} ms against {measured[wide]:.1f} ms. The traffic model orders these "
        "backwards: it counts tiles streamed and ignores that a larger tile spends more shared "
        "memory and registers per program, which costs the occupancy that hides the latency"
    )


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
