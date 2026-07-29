# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import math

import torch

from autodiff.micrograd import Value


def test_add_mul_grads():
    a, b = Value(2.0), Value(-3.0)
    c = a * b + a
    c.backward()
    assert c.data == -4.0
    assert a.grad == -2.0
    assert b.grad == 2.0


def test_tanh_matches_torch():
    x = Value(0.7)
    y = x.tanh()
    y.backward()
    xt = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    torch.tanh(xt).backward()
    assert abs(y.data - math.tanh(0.7)) < 1e-12
    assert abs(x.grad - xt.grad.item()) < 1e-12


def test_pow_relu_div():
    x = Value(3.0)
    y = (x**2.0) / 3.0 - 1.0
    z = y.relu()
    z.backward()
    assert abs(z.data - 2.0) < 1e-12
    assert abs(x.grad - 2.0) < 1e-12


def test_relu_blocks_gradient_when_negative():
    x = Value(-1.0)
    y = x.relu()
    y.backward()
    assert y.data == 0.0
    assert x.grad == 0.0


def test_reused_node_accumulates():
    x = Value(2.0)
    y = x * x + x * x
    y.backward()
    assert x.grad == 8.0
