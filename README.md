# polkagris

Autograd, training loops, profiling, precision, compilers, CUDA and Triton kernels, distributed training, serving.

## Quickstart

```bash
uv sync --extra dev
uv run pytest
uv run python -m training.mnist_mlp --smoke
```

With a CUDA torch installed outside the lockfile, use `uv run --no-sync`; a plain `uv sync` reverts to the CPU wheel. Extras: `kernels` (triton), `serving` (vllm), `post` (transformers), `dev` (pytest, ruff).

## What is here

| Directory | Contents | Run |
|-----------|----------|-----|
| `autodiff/` | A ~150-line scalar reverse-mode autograd engine and a PyTorch port asserting gradient parity | `-m autodiff.torch_port` |
| `training/` | MNIST MLP and CIFAR-10 CNN on a bare loop, every hyperparameter in one place | `-m training.mnist_mlp` |
| `gpu_basics/` | Vector add and naive matmul in raw CUDA C++, no frameworks | `make run` |
| `profiling/` | torch.profiler passes and Nsight scripts | `-m profiling.profile_cifar` |
| `precision/` | fp32 / bf16 / fp16 step time, peak memory, accuracy | `-m precision.fp32_vs_bf16` |
| `compile/` | torch.compile backends measured and read from the generated code | `-m compile.inspect_inductor` |
| `triton_kernels/` | Softmax, fused layernorm+linear, tiled causal attention | `-m triton_kernels.softmax` |
| `cuda_kernels/` | The SGEMM ladder, six kernels to vectorized register tiling, timed against cuBLAS | `make run` |
| `distributed/` | DDP on the CIFAR loop, FSDP on the transformer | `torchrun -m distributed.ddp_cifar` |
| `serving/` | vLLM throughput/latency sweep and offline batch inference | `-m serving.vllm_sweep` |
| `capstone/` | A decoder-only transformer (RMSNorm, RoPE, SwiGLU, tied embeddings) with swappable op backends | `-m capstone.train` |

Bare loops, no Trainer abstractions. Every number goes through `polkagris.bench` (fixed seeds, warmup, CUDA-event timing, JSON in `benchmarks/results/`). Platform-gated code skips cleanly instead of crashing. `capstone/ops/` makes `--ops reference` and `--ops triton` one flag apart.

## Checks

Each directory has a `checks.py` that measures its own invariants:

```
$ uv run python -m precision.checks
precision
  ok       fp16 overflows where bf16 does not: 70000 is inf in fp16 but 70144 in bf16
  ok       sequential accumulation stalls in fp16: adding 1.0 4096 times stops at 2048
  skip     a gpu is needed for the speed story: no CUDA device
```

`ok` carries the number just measured, `skip` means the machine cannot measure it, `pending` means the property is not established yet.

## Results

| What | Result |
|------|--------|
| Gradient parity, scalar engine vs torch | float64 roundoff, ~1e-16 |
| MNIST MLP, 10 epochs | **98.53%** |
| CIFAR-10 CNN, 30 epochs | **90.68%** |
| Transformer, 3.2M params, 300 steps | val perplexity **6.4** |
| CIFAR step time: fp32 / bf16 / fp16 | 49.9 / 249.8 / 264.0 ms |
| Peak memory: fp32 / bf16 | 339 / 184 MiB |
| torch.compile, aot_eager backend | 0.99x |

- Without tensor cores, autocast saves 1.8x memory and costs 5x speed. Mixed precision is a hardware feature; the table inverts on newer silicon.
- The first `aten::conv2d` reports 1.36 s CPU total: cudnn autotuning on first touch, not compute. Warm up before timing anything.
- Convolution GEMMs are 43.8% of backward CUDA time.
- The SGEMM ladder ends at 2D register tiling. The rest of cuBLAS is double buffering, tensor cores, per-shape dispatch, and bank conflicts.
- Tied-embedding init: first loss was 61, not ln(65) = 4.17, until normal(0, 0.02). `capstone/checks.py` asserts it now.

## License

MIT
