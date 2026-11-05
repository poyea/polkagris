# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m serving.checks"""

from __future__ import annotations

from polkagris.checks import Skip, run


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
    """Sweep the pool and find where per-step latency stops being free.

    A decode step reads the weights once whatever the batch, so until the
    arithmetic catches up with that traffic another row costs nothing and
    throughput rises for free. The turnover is where it stops being free, and
    it is a property of the card rather than of this code, so it is measured
    here rather than written down.
    """
    import random
    import statistics

    import torch

    if not torch.cuda.is_available():
        raise Skip("the turnover is a bandwidth against arithmetic trade, which needs a GPU")

    from capstone.model import ModelConfig, Transformer
    from polkagris.bench import set_seed
    from serving.scheduler import Request, Scheduler

    set_seed()
    pools = (1, 8, 32, 128)
    steps, prompt = 8, 8
    cfg = ModelConfig(vocab_size=4096, dim=512, n_layers=8, n_heads=8, seq_len=128)
    model = Transformer(cfg).eval().cuda()

    def fresh(pool: int) -> Scheduler:
        sched = Scheduler(model=model, capacity=pool, max_seq_len=prompt + 6 * steps + 8)
        for _ in range(pool):
            sched.submit(Request(prompt=[5] * prompt, max_new_tokens=6 * steps))
        sched.admit()
        sched.step()
        return sched

    def burst(sched: Scheduler) -> list[float]:
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        for i in range(steps):
            starts[i].record()
            sched.step()
            ends[i].record()
        torch.cuda.synchronize()
        return [a.elapsed_time(b) for a, b in zip(starts, ends)]

    # Bursts are interleaved in a shuffled order because the clock ramps as the
    # sweep runs. Measuring each pool to completion in turn hands the later ones
    # a warmer card and reads the ramp as a batching effect.
    samples: dict[int, list[float]] = {pool: [] for pool in pools}
    scheds = {pool: fresh(pool) for pool in pools}
    order = [pool for pool in pools for _ in range(4)]
    random.Random(1859).shuffle(order)
    for pool in order:
        samples[pool] += burst(scheds[pool])

    p99 = {pool: statistics.quantiles(rows, n=100)[98] for pool, rows in samples.items()}
    floor = min(p99.values())
    free = [pool for pool in pools if p99[pool] <= 1.5 * floor]
    knee = max(free)
    gain = (knee / (p99[knee] / 1000.0)) / (pools[0] / (p99[pools[0]] / 1000.0))
    assert knee > pools[0], f"batching bought nothing: p99 {p99}"
    return (
        f"p99 per step holds near {floor:.0f} ms out to a pool of {knee}, "
        f"{gain:.0f}x the tokens/s of a pool of {pools[0]} at the same latency"
    )


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
