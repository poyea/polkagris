# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vllm is not available on this platform (uv sync --extra serving), skipping")
        sys.exit(0)

    prompts = [f"Question {i}: what is {i} squared? Answer:" for i in range(args.n_prompts)]
    llm = LLM(model=args.model)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    wall = time.perf_counter() - t0
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"{len(prompts)} prompts in {wall:.1f}s, {tokens} tokens, {tokens / wall:.1f} tok/s")


if __name__ == "__main__":
    main()
