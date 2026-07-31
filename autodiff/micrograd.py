# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math


class Value:
    def __init__(self, data: float, _children: tuple = (), _op: str = ""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k: float) -> Value:
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        # Branch on the sign so the exp argument is never positive; exp(-x) for
        # x = -1000 is the overflow that exp() above walks into head-on.
        if self.data >= 0.0:
            s = 1.0 / (1.0 + math.exp(-self.data))
        else:
            e = math.exp(self.data)
            s = e / (1.0 + e)
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Value | float) -> Value:
        return self + (-(other if isinstance(other, Value) else Value(other)))

    def __rsub__(self, other: float) -> Value:
        return Value(other) + (-self)

    def __truediv__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1.0

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
