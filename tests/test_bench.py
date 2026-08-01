# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import json

from polkagris import benchmark, set_seed
from polkagris.bench import BENCH_DIR


def test_benchmark_writes_json(tmp_path):
    set_seed()
    result = benchmark(
        lambda: sum(range(1000)), "test_bench_smoke", warmup=1, reps=3, out_dir=tmp_path
    )
    assert result.reps == 3
    assert result.mean_ms >= 0
    path = tmp_path / "test_bench_smoke.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["name"] == "test_bench_smoke"


def test_benchmark_can_skip_persisting(tmp_path):
    """save=False must not create the directory, let alone a file."""
    set_seed()
    out = tmp_path / "never"
    benchmark(lambda: sum(range(100)), "test_bench_nosave", warmup=1, reps=2,
              save=False, out_dir=out)
    assert not out.exists()


def test_the_suite_never_writes_into_the_real_results_dir():
    """A crashed test must not leave a fake result where the dashboard reads."""
    strays = list(BENCH_DIR.glob("test_bench_*.json")) if BENCH_DIR.exists() else []
    assert not strays, f"test artefacts in the results directory: {strays}"
