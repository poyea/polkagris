# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

from __future__ import annotations

import random

import numpy as np

from autodiff.micrograd import Value
from polkagris import set_seed


class Neuron:
    def __init__(self, nin: int, nonlin: bool = True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: list[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int, nonlin: bool = True):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin: int, sizes: list[int]):
        dims = [nin] + sizes
        self.layers = [
            Layer(dims[i], dims[i + 1], nonlin=i < len(sizes) - 1) for i in range(len(sizes))
        ]

    def __call__(self, x: list[float | Value]) -> Value:
        vals: list[Value] = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        for layer in self.layers:
            vals = layer(vals)
        return vals[0]

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


def make_moons(n: int = 100, noise: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    half = n // 2
    t = np.linspace(0, np.pi, half)
    x_up = np.stack([np.cos(t), np.sin(t)], axis=1)
    x_dn = np.stack([1 - np.cos(t), -np.sin(t) + 0.5], axis=1)
    x = np.concatenate([x_up, x_dn]) + noise * np.random.randn(n, 2)
    y = np.concatenate([-np.ones(half), np.ones(half)])
    return x, y


def main(steps: int = 100, lr: float = 0.05) -> float:
    set_seed()
    x, y = make_moons()
    model = MLP(2, [16, 16, 1])
    for step in range(steps):
        scores = [model(list(xi)) for xi in x]
        losses = [(1 + -yi * si).relu() for yi, si in zip(y, scores)]
        data_loss = sum(losses, Value(0.0)) * (1.0 / len(losses))
        reg_loss = sum((p * p for p in model.parameters()), Value(0.0)) * 1e-4
        loss = data_loss + reg_loss
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        if step % 20 == 0 or step == steps - 1:
            acc = sum((si.data > 0) == (yi > 0) for si, yi in zip(scores, y)) / len(y)
            print(f"step {step:3d}  loss {loss.data:.4f}  acc {acc:.2f}")
    final_acc = sum((model(list(xi)).data > 0) == (yi > 0) for xi, yi in zip(x, y)) / len(y)
    print(f"final accuracy {final_acc:.2f}")
    return final_acc


if __name__ == "__main__":
    main()
