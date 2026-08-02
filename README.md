# polkagris

Autograd, training loops, profiling, precision, compilers, CUDA and Triton kernels, distributed training, serving.

## Install

```bash
uv sync --extra dev
```

Extras: `kernels` (triton), `serving` (vllm), `post` (transformers). The first two are Linux only.

Hand-installed CUDA torch sits outside the lockfile, and a bare `uv sync` or `uv run` silently puts the CPU wheel back. Use `uv run --no-sync`.

## Run

```bash
uv run python -m autodiff.train_toy
uv run python -m autodiff.torch_port

uv run python -m training.mnist_mlp
uv run python -m training.cifar_cnn --epochs 30

uv run python -m precision.fp32_vs_bf16 --reps 20
uv run python -m compile.inspect_inductor --backend aot_eager
uv run python -m profiling.profile_cifar --steps 10

uv run python -m triton_kernels.softmax                          # Linux
uv run python -m serving.vllm_sweep --model Qwen/Qwen2.5-0.5B    # Linux
torchrun --nproc_per_node=2 -m distributed.ddp_cifar --epochs 1

uv run python -m capstone.train --data shakespeare --steps 2000 --save checkpoints/capstone.pt
uv run python -m capstone.eval.score --checkpoint checkpoints/capstone.pt --data shakespeare

uv run python -m experiments.detached_backend

(cd cuda_primitives && make run)
(cd cuda_kernels && make run SIZE=4096)
```

`--smoke` gives any training script one synthetic epoch and no download. Hyperparameters live in `training/config.py`; everything else takes `--help`. Held-out perplexity comes from `capstone.eval.score`, never from `capstone.train`.

## Checks

```bash
uv run python -m <dir>.checks
```

`ok` measured, `skip` the machine cannot, `pending` not established, `FAIL` broken invariant and a non-zero exit.

## Benchmarks

`polkagris.bench`: seed 1859, warmup, CUDA-event timing, one JSON per run in `benchmarks/results/`. Rerun rather than edit.

## License

MIT
