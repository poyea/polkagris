# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m serving.checks"""

from __future__ import annotations

from polkagris.checks import Pending, Skip, run


def kv_cache_bytes(layers: int, heads: int, head_dim: int, seq: int, dtype_bytes: int = 2) -> int:
    return 2 * layers * heads * head_dim * seq * dtype_bytes


def the_kv_cache_grows_linearly_with_context() -> str:
    short = kv_cache_bytes(32, 32, 128, 2048)
    long = kv_cache_bytes(32, 32, 128, 8192)
    assert long == 4 * short, f"expected 4x, got {long / short}"
    return f"one sequence costs {short/2**20:.0f} MiB at 2k and {long/2**20:.0f} MiB at 8k, per request"


def the_cache_not_the_weights_caps_concurrency() -> str:
    budget = 24 * 2**30
    weights = 14 * 2**30
    per_request = kv_cache_bytes(32, 32, 128, 4096)
    seats = (budget - weights) // per_request
    assert seats > 0, "no room left for a single request"
    return f"{(budget-weights)/2**30:.0f} GiB spare / {per_request/2**20:.0f} MiB each = {seats} concurrent requests"


def batching_buys_throughput_and_costs_latency() -> str:
    def step_ms(batch: int, fixed: float = 8.0, per_seq: float = 0.6) -> float:
        return fixed + per_seq * batch

    def throughput(batch: int) -> float:
        return batch / (step_ms(batch) / 1000.0)

    one, big = throughput(1), throughput(64)
    assert big > one * 10, f"expected a large gain, got {big/one:.1f}x"
    assert step_ms(64) > step_ms(1), "a bigger batch cannot be faster per step"
    return (
        f"batch 64 gives {big/one:.0f}x the tokens/s of batch 1, "
        f"while a step goes {step_ms(1):.1f} ms to {step_ms(64):.1f} ms"
    )


def continuous_batching_beats_waiting_for_the_slowest() -> str:
    lengths = [10, 200, 15, 180]
    static = len(lengths) * max(lengths)
    continuous = sum(lengths)
    waste = 1 - continuous / static
    assert waste > 0.4, f"expected substantial waste, got {waste:.1%}"
    return f"padding to the longest wastes {waste:.0%} of the slots; continuous batching reclaims it"


def vllm_is_needed_for_real_numbers() -> str:
    try:
        import vllm
    except ImportError as exc:
        raise Skip("vllm is not installed") from exc
    return f"vLLM {vllm.__version__} present; run serving.vllm_sweep"


def batch_size_against_p99() -> str:
    raise Pending("the batch size where p99 latency turns over is unmeasured")


CHECKS = [
    the_kv_cache_grows_linearly_with_context,
    the_cache_not_the_weights_caps_concurrency,
    batching_buys_throughput_and_costs_latency,
    continuous_batching_beats_waiting_for_the_slowest,
    vllm_is_needed_for_real_numbers,
    batch_size_against_p99,
]

if __name__ == "__main__":
    raise SystemExit(run("serving", CHECKS))
