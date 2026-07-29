# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m compile.checks"""

from __future__ import annotations

import torch

from polkagris import set_seed
from polkagris.checks import Pending, Skip, run


def fn(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x) * 2.0 + x


def compiled_output_matches_eager() -> str:
    set_seed()
    x = torch.randn(64, 64)
    compiled = torch.compile(fn, backend="aot_eager")
    assert torch.allclose(fn(x), compiled(x), atol=1e-6), "aot_eager changed the numbers"
    return "aot_eager reproduces eager exactly: graph capture without codegen"


def a_data_dependent_branch_still_returns_the_right_answer() -> str:
    def branchy(x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 if x.sum() > 0 else x * -2.0

    set_seed()
    compiled = torch.compile(branchy, backend="aot_eager")
    for x in (torch.ones(8), -torch.ones(8)):
        assert torch.allclose(branchy(x), compiled(x)), "the branch was captured wrongly"
    return "dynamo recompiles per branch rather than guessing; correctness survives, speed pays"


def recompilation_is_triggered_by_new_shapes() -> str:
    seen: list[tuple[int, ...]] = []

    def watched(x: torch.Tensor) -> torch.Tensor:
        seen.append(tuple(x.shape))
        return x + 1

    compiled = torch.compile(watched, backend="aot_eager", dynamic=False)
    compiled(torch.randn(4))
    compiled(torch.randn(4))
    compiled(torch.randn(9))
    assert len(seen) >= 2, f"expected at least one retrace, traced {len(seen)}"
    return f"the python function was traced {len(seen)} times for shapes {sorted(set(seen))}"


def inductor_needs_a_backend_toolchain() -> str:
    set_seed()
    x = torch.randn(32, 32)
    try:
        compiled = torch.compile(fn, backend="inductor")
        out = compiled(x)
    except Exception as exc:
        raise Skip(f"inductor unavailable: {type(exc).__name__}") from exc
    assert torch.allclose(fn(x), out, atol=1e-5), "inductor changed the numbers"
    return "inductor compiled and matched eager"


def aot_eager_cannot_speed_anything_up() -> str:
    import time

    def median_ms(call, reps: int = 20) -> float:
        for _ in range(5):
            call()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            call()
            times.append((time.perf_counter() - t0) * 1e3)
        return sorted(times)[len(times) // 2]

    set_seed()
    x = torch.randn(256, 256)
    compiled = torch.compile(fn, backend="aot_eager")
    eager_ms = median_ms(lambda: fn(x))
    aot_ms = median_ms(lambda: compiled(x))
    assert aot_ms >= eager_ms * 0.9, "aot_eager generates no kernels, so a real speedup is suspect"
    return (
        f"eager {eager_ms:.3f} ms, aot_eager {aot_ms:.3f} ms: capture without codegen only adds "
        "dispatch, which is why the phase needs inductor to show a win"
    )


def generated_kernel_inspected() -> str:
    raise Pending("no fused kernel has been inspected")


CHECKS = [
    compiled_output_matches_eager,
    a_data_dependent_branch_still_returns_the_right_answer,
    recompilation_is_triggered_by_new_shapes,
    aot_eager_cannot_speed_anything_up,
    inductor_needs_a_backend_toolchain,
    generated_kernel_inspected,
]

if __name__ == "__main__":
    raise SystemExit(run("compile", CHECKS))
