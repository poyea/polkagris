# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import json

from polkagris import benchmark, set_seed
from polkagris.bench import BENCH_DIR


def test_benchmark_writes_json():
    set_seed()
    result = benchmark(lambda: sum(range(1000)), "test_bench_smoke", warmup=1, reps=3)
    assert result.reps == 3
    assert result.mean_ms >= 0
    path = BENCH_DIR / "test_bench_smoke.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["name"] == "test_bench_smoke"
    path.unlink()
