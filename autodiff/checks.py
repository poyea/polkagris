# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m autodiff.checks"""

from __future__ import annotations

from autodiff.micrograd import Value
from polkagris.checks import Pending, run


def count_nodes(out: Value) -> int:
    seen: set[int] = set()
    stack = [out]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.extend(node._prev)
    return len(seen)


def sigmoid_composed_from_primitives() -> str:
    x = Value(0.7)
    s = (1.0 + (-x).exp()) ** -1.0
    s.backward()
    expected = s.data * (1.0 - s.data)
    assert abs(x.grad - expected) < 1e-12, f"{x.grad} != {expected}"
    return f"d/dx sigmoid(0.7) = {x.grad:.6f} = s(1-s), no new op needed"


def a_reused_node_accumulates_gradient() -> str:
    x = Value(3.0)
    y = x * 2.0 + x * 5.0
    y.backward()
    assert x.grad == 7.0, f"expected 7.0, got {x.grad} (is += still +=?)"
    return "x used twice gives grad 7.0, the sum of both paths"


def every_operation_allocates_a_node() -> str:
    x = Value(1.5)
    y = (x * 2.0 + 1.0).relu()
    n = count_nodes(y)
    assert n == 6, f"expected 6 nodes, counted {n}"
    return f"{n} nodes for one relu(2x+1): x, 2.0, mul, 1.0, add, relu"


def tanh_and_exp_agree_on_their_gradients() -> str:
    x = Value(0.4)
    t = x.tanh()
    t.backward()
    from_tanh = x.grad

    x2 = Value(0.4)
    e2 = (x2 * 2.0).exp()
    manual = (e2 - 1.0) / (e2 + 1.0)
    manual.backward()
    assert abs(from_tanh - x2.grad) < 1e-9, f"{from_tanh} != {x2.grad}"
    return f"tanh backward {from_tanh:.6f} matches the exp-built version"


def reciprocal_has_no_reflected_operator() -> str:
    x = Value(4.0)
    try:
        _ = 1.0 / x
    except TypeError:
        return "1.0 / Value raises TypeError; x ** -1.0 is the way in. Add __rtruediv__?"
    raise AssertionError("__rtruediv__ exists now, so the composed sigmoid above can be simplified")


def fused_sigmoid_op_exists() -> str:
    if hasattr(Value, "sigmoid"):
        x = Value(0.7)
        s = x.sigmoid()
        s.backward()
        expected = s.data * (1.0 - s.data)
        assert abs(x.grad - expected) < 1e-12, f"{x.grad} != {expected}"
        return f"Value.sigmoid backward is correct: {x.grad:.6f}"
    raise Pending("Value has no sigmoid(); the composed form above is used instead")


def break_gradient_accumulation_on_purpose() -> str:
    x = Value(2.0)
    y = x * x
    y.backward()
    assert abs(x.grad - 4.0) < 1e-12, f"expected 4.0, got {x.grad}"
    return f"x*x at x=2 gives {x.grad}; swap a += for = in __mul__ and watch this drop to 2"


def gradients_match_finite_differences() -> str:
    def f(v: float) -> float:
        x = Value(v)
        return ((x * x + x).tanh() * 3.0).data

    v, h = 0.3, 1e-6
    x = Value(v)
    out = (x * x + x).tanh() * 3.0
    out.backward()
    numeric = (f(v + h) - f(v - h)) / (2 * h)
    assert abs(x.grad - numeric) < 1e-6, f"{x.grad} vs {numeric}"
    return f"analytic {x.grad:.6f} vs central difference {numeric:.6f}"


def exp_of_a_large_input_overflows() -> str:
    try:
        Value(1000.0).exp()
    except OverflowError:
        return "exp(1000) overflows; real frameworks subtract the max first"
    raise AssertionError("expected OverflowError from math.exp(1000)")


CHECKS = [
    sigmoid_composed_from_primitives,
    a_reused_node_accumulates_gradient,
    every_operation_allocates_a_node,
    tanh_and_exp_agree_on_their_gradients,
    gradients_match_finite_differences,
    reciprocal_has_no_reflected_operator,
    break_gradient_accumulation_on_purpose,
    exp_of_a_large_input_overflows,
    fused_sigmoid_op_exists,
]

if __name__ == "__main__":
    raise SystemExit(run("autodiff", CHECKS))
