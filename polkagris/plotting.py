# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Load bench.py JSON results and plot comparisons across phases."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from polkagris.bench import BENCH_DIR


def load_results(pattern: str = "*.json") -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(BENCH_DIR.glob(pattern))]


def bar_compare(pattern: str, title: str, out: str | Path | None = None) -> None:
    results = load_results(pattern)
    if not results:
        raise SystemExit(f"no results matching {pattern!r} in {BENCH_DIR}")
    names = [r["name"] for r in results]
    means = [r["mean_ms"] for r in results]
    stds = [r["std_ms"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, means, yerr=stds, capsize=4)
    ax.set_ylabel("ms")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    else:
        plt.show()
