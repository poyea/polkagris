# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Throughput of a batch of prompts, over this repo's scheduler or over vLLM.

The local backend is the point of comparison: it drives `serving.scheduler`,
so it runs wherever torch does and measures the code in this repo rather than
a library's. The vLLM backend is the reference a real server would be held to,
and is only reachable where vLLM installs.
"""

import argparse
import sys
import time


def run_local(n_prompts: int, max_tokens: int, pool: int, prompt_len: int) -> None:
    import torch

    from capstone.model import ModelConfig, Transformer
    from polkagris.bench import set_seed
    from serving.scheduler import Request, Scheduler

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(vocab_size=4096, dim=512, n_layers=8, n_heads=8, seq_len=1024)
    model = Transformer(cfg).eval().to(device)

    # Synthetic token ids, so throughput needs no tokenizer and no download.
    # The numbers are about the batching, which does not care what the tokens
    # mean, and untrained weights cost exactly what trained ones do.
    sched = Scheduler(model=model, capacity=pool, max_seq_len=prompt_len + max_tokens + 1)
    for i in range(n_prompts):
        prompt = [(i + j) % cfg.vocab_size for j in range(prompt_len)]
        sched.submit(Request(prompt=prompt, max_new_tokens=max_tokens))

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    finished = sched.run()
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - start

    tokens = sum(len(sequence.generated) for sequence in finished)
    print(
        f"{len(finished)} prompts in {wall:.1f}s, {tokens} tokens, "
        f"{tokens / wall:.1f} tok/s (local, pool {pool}, {device})"
    )


def run_vllm(model_name: str, n_prompts: int, max_tokens: int) -> None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vllm is not available on this platform (uv sync --extra serving), skipping")
        sys.exit(0)

    prompts = [f"Question {i}: what is {i} squared? Answer:" for i in range(n_prompts)]
    llm = LLM(model=model_name)
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    wall = time.perf_counter() - t0
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"{len(prompts)} prompts in {wall:.1f}s, {tokens} tokens, {tokens / wall:.1f} tok/s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("local", "vllm"), default="local")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="vllm backend only")
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--pool", type=int, default=16, help="local backend only")
    parser.add_argument("--prompt-len", type=int, default=32, help="local backend only")
    args = parser.parse_args()

    if args.backend == "local":
        run_local(args.n_prompts, args.max_tokens, args.pool, args.prompt_len)
    else:
        run_vllm(args.model, args.n_prompts, args.max_tokens)


if __name__ == "__main__":
    main()
