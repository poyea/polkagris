# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from autodiff.micrograd import Value
from autodiff.train_toy import MLP, make_moons
from polkagris import set_seed


def main(tol: float = 1e-6) -> None:
    set_seed()
    x, y = make_moons(n=20)
    model = MLP(2, [8, 1])

    scores = [model(list(xi)) for xi in x]
    losses = [(1 + -yi * si).relu() for yi, si in zip(y, scores)]
    loss = sum(losses, Value(0.0)) * (1.0 / len(losses))
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    torch_params: list[torch.Tensor] = []
    layers = []
    for layer in model.layers:
        w = torch.tensor(
            [[wi.data for wi in n.w] for n in layer.neurons], dtype=torch.float64, requires_grad=True
        )
        b = torch.tensor(
            [n.b.data for n in layer.neurons], dtype=torch.float64, requires_grad=True
        )
        nonlin = layer.neurons[0].nonlin
        torch_params += [w, b]
        layers.append((w, b, nonlin))

    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    h = xt
    for w, b, nonlin in layers:
        h = h @ w.T + b
        if nonlin:
            h = torch.tanh(h)
    t_loss = torch.relu(1 + -yt * h.squeeze(1)).mean()
    t_loss.backward()

    assert abs(t_loss.item() - loss.data) < tol, (t_loss.item(), loss.data)
    i = 0
    max_err = 0.0
    for layer in model.layers:
        w_grad = torch_params[i].grad
        b_grad = torch_params[i + 1].grad
        for r, n in enumerate(layer.neurons):
            for c, wi in enumerate(n.w):
                max_err = max(max_err, abs(wi.grad - float(w_grad[r, c])))
            max_err = max(max_err, abs(n.b.grad - float(b_grad[r])))
        i += 2
    assert max_err < tol, max_err
    print(f"gradient parity ok: max |micrograd - torch| = {max_err:.2e}")


if __name__ == "__main__":
    main()
