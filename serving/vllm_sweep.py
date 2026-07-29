# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vllm is not available on this platform (uv sync --extra serving), skipping")
        sys.exit(0)

    llm = LLM(model=args.model)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.8)
    prompt = "Explain what a roofline model is in one paragraph."

    print(f"{'batch':>6} {'wall_s':>8} {'tok/s':>10} {'ms/req':>10}")
    for batch in args.batch_sizes:
        t0 = time.perf_counter()
        outputs = llm.generate([prompt] * batch, params)
        wall = time.perf_counter() - t0
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(f"{batch:>6} {wall:>8.2f} {tokens / wall:>10.1f} {1e3 * wall / batch:>10.1f}")


if __name__ == "__main__":
    main()
