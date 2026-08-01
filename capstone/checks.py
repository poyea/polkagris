# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m capstone.checks"""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from capstone.model import ModelConfig, Transformer
from capstone.ops import get_ops
from polkagris import set_seed
from polkagris.checks import Pending, Skip, run

SMALL = ModelConfig(vocab_size=65, dim=64, n_layers=2, n_heads=4, seq_len=32)


def build(ops: str = "reference") -> Transformer:
    set_seed()
    return Transformer(replace(SMALL, ops=ops))


def available_backends() -> list[str]:
    """Backends importable here. Anything gated on platform drops out."""
    kinds = []
    for kind in ("reference", "triton"):
        try:
            get_ops(kind)
        except (ImportError, RuntimeError):
            continue
        kinds.append(kind)
    return kinds


def initial_loss_is_the_log_of_the_vocabulary() -> str:
    model = build()
    tokens = torch.randint(0, SMALL.vocab_size, (4, 16))
    logits = model(tokens)
    # Next-token, not self-prediction: tied embeddings let a model copy its
    # own input, which hides a broken init behind a suspiciously low loss.
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, SMALL.vocab_size), tokens[:, 1:].reshape(-1)
    ).item()
    expected = math.log(SMALL.vocab_size)
    assert abs(loss - expected) < 0.5, f"loss {loss:.3f} is far from ln(vocab) {expected:.3f}"
    return f"loss {loss:.3f} vs ln({SMALL.vocab_size}) = {expected:.3f}; a bad init shows up here first"


def self_prediction_hides_a_broken_init() -> str:
    model = build()
    tokens = torch.randint(0, SMALL.vocab_size, (4, 16))
    logits = model(tokens)
    cheating = torch.nn.functional.cross_entropy(
        logits.reshape(-1, SMALL.vocab_size), tokens.reshape(-1)
    ).item()
    honest = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, SMALL.vocab_size), tokens[:, 1:].reshape(-1)
    ).item()
    assert cheating < honest, f"expected the shifted target to be harder: {cheating} vs {honest}"
    return f"scoring the input against itself gives {cheating:.3f}, the real task {honest:.3f}"


def weight_tying_shares_one_tensor() -> str:
    model = build()
    assert model.lm_head.weight is model.embed.weight, "lm_head and embed are separate tensors"
    tied = sum(p.numel() for p in model.parameters())
    untied = tied + SMALL.vocab_size * SMALL.dim
    return f"{tied} params tied vs {untied} untied, one tensor counted once"


def rope_is_a_rotation_so_it_preserves_norm() -> str:
    ops = get_ops("reference")
    set_seed()
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 4, 8, 16)
    positions = torch.arange(8)
    qr, kr = ops.rope(q, k, positions)
    assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-5), "rope changed |q|"
    assert torch.allclose(k.norm(dim=-1), kr.norm(dim=-1), atol=1e-5), "rope changed |k|"
    return "per-head norms survive rope, because it only rotates within each pair of dims"


def rope_at_position_zero_is_the_identity() -> str:
    ops = get_ops("reference")
    set_seed()
    q = torch.randn(1, 1, 1, 8)
    qr, _ = ops.rope(q, q.clone(), torch.zeros(1, dtype=torch.long))
    assert torch.allclose(q, qr, atol=1e-6), "position 0 should not rotate anything"
    return "angle 0 leaves q untouched, so absolute position 0 carries no phase"


def rmsnorm_ignores_the_mean() -> str:
    ops = get_ops("reference")
    weight = torch.ones(8)
    x = torch.randn(2, 8)
    shifted = ops.rmsnorm(x + 10.0, weight)
    centred = ops.rmsnorm(x, weight)
    assert not torch.allclose(shifted, centred, atol=1e-3), "rmsnorm looks mean-invariant"
    scaled = ops.rmsnorm(x * 4.0, weight)
    assert torch.allclose(scaled, centred, atol=1e-4), "rmsnorm should be scale-invariant"
    return "rmsnorm is scale-invariant but not shift-invariant, unlike layernorm"


def attention_is_causal() -> str:
    ops = get_ops("reference")
    set_seed()
    q, k, v = (torch.randn(1, 2, 6, 8) for _ in range(3))
    base = ops.attention(q, k, v)
    v_edited = v.clone()
    v_edited[:, :, -1, :] += 100.0  # change the last position only
    edited = ops.attention(q, k, v_edited)
    assert torch.allclose(base[:, :, :-1], edited[:, :, :-1], atol=1e-5), "the future leaked"
    assert not torch.allclose(base[:, :, -1], edited[:, :, -1]), "the last row ignored its own v"
    return "editing the last value changes only the last output row"


def the_backward_pass_touches_every_parameter() -> str:
    # Every backend, not just the default: a kernel that drops the autograd graph
    # trains a fraction of the model while the loss still falls, and asserting
    # this against `reference` alone can never catch that.
    notes = []
    for kind in available_backends():
        model = build(kind)
        tokens = torch.randint(0, SMALL.vocab_size, (2, 16))
        logits = model(tokens)
        torch.nn.functional.cross_entropy(
            logits.view(-1, SMALL.vocab_size), tokens.view(-1)
        ).backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, f"[{kind}] no gradient reached {missing}"
        notes.append(f"{kind}: all {len(list(model.parameters()))} tensors")
    return "; ".join(notes)


def a_forward_only_backend_refuses_to_train() -> str:
    """A kernel with no backward must raise, not silently detach the graph."""
    if "triton" not in available_backends():
        raise Skip("triton ops unavailable here")
    ops = get_ops("triton")
    q, k, v = (torch.randn(1, 4, 32, 16, device="cuda", requires_grad=True) for _ in range(3))
    try:
        ops.attention(q, k, v)
    except NotImplementedError:
        return "triton attention refuses a grad-enabled call instead of detaching"
    raise AssertionError("triton attention accepted a grad-enabled call; gradients are lost")


def swiglu_hidden_size_is_two_thirds_of_four_x() -> str:
    model = build()
    hidden = model.blocks[0].mlp.w1.out_features
    expected = 4 * SMALL.dim * 2 // 3
    assert hidden == expected, f"{hidden} != {expected}"
    return f"hidden {hidden} keeps the parameter count near a plain 4x MLP despite three matrices"


def the_triton_backend_needs_a_triton_box() -> str:
    try:
        get_ops("triton")
    except (ImportError, RuntimeError) as exc:
        raise Skip(f"triton ops unavailable here: {exc}") from exc
    return "triton ops imported; compare against reference with --ops triton"


def memory_ceiling_and_bf16_recovery() -> str:
    raise Pending("the dim/layers ceiling and its bf16 recovery are unmeasured")


CHECKS = [
    initial_loss_is_the_log_of_the_vocabulary,
    self_prediction_hides_a_broken_init,
    weight_tying_shares_one_tensor,
    swiglu_hidden_size_is_two_thirds_of_four_x,
    rope_is_a_rotation_so_it_preserves_norm,
    rope_at_position_zero_is_the_identity,
    rmsnorm_ignores_the_mean,
    attention_is_causal,
    the_backward_pass_touches_every_parameter,
    a_forward_only_backend_refuses_to_train,
    the_triton_backend_needs_a_triton_box,
    memory_ceiling_and_bf16_recovery,
]

if __name__ == "__main__":
    raise SystemExit(run("capstone", CHECKS))
