# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Run: python -m training.checks"""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from polkagris import set_seed
from polkagris.checks import Skip, run
from polkagris.data import DATA_ROOT
from training.config import TrainConfig
from training.loop import lr_at, make_optimizer

CFG = TrainConfig()
TOTAL = 1000


def _subset_loader(dataset, indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(CFG.seed) if shuffle else None,
    )


def _pick(n_total: int, n: int) -> list[int]:
    return torch.randperm(n_total, generator=torch.Generator().manual_seed(0))[:n].tolist()


def _mnist_subset(n_train: int, batch_size: int) -> tuple[DataLoader, DataLoader]:
    # download=False: a check measures, it does not fetch 64 MB behind your back.
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    try:
        train = datasets.MNIST(DATA_ROOT, train=True, download=False, transform=tf)
        test = datasets.MNIST(DATA_ROOT, train=False, download=False, transform=tf)
    except RuntimeError as exc:
        raise Skip(f"MNIST is not in {DATA_ROOT}; run -m training.mnist_mlp once") from exc
    return (
        _subset_loader(train, _pick(len(train), n_train), batch_size, shuffle=True),
        _subset_loader(test, list(range(2000)), 512, shuffle=False),
    )


def _train_mlp(
    lr: float, train_loader: DataLoader, test_loader: DataLoader, epochs: int
) -> tuple[list[float], float]:
    set_seed()
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    cfg = replace(CFG, lr=lr, epochs=epochs, warmup_steps=20)
    opt = make_optimizer(model, cfg)
    total_steps, step = epochs * len(train_loader), 0
    epoch_losses = []
    for _ in range(epochs):
        model.train()
        seen = []
        for x, y in train_loader:
            for group in opt.param_groups:
                group["lr"] = lr_at(step, total_steps, cfg)
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            seen.append(loss.item())
        epoch_losses.append(sum(seen) / len(seen))
    return epoch_losses, _accuracy(model, test_loader)


def _cifar_subset(
    augment: bool, n_train: int, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # Three loaders, not two: scoring the training set needs it under the *test*
    # transform, or the crops and flips would show up as a fake train-side loss.
    from torchvision import datasets, transforms

    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_tf = transforms.Compose(
        ([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else [])
        + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    try:
        train = datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=train_tf)
        scored = datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=test_tf)
        test = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=test_tf)
    except RuntimeError as exc:
        raise Skip(f"CIFAR-10 is not in {DATA_ROOT}; run -m training.cifar_cnn once") from exc
    idx = _pick(len(train), n_train)
    return (
        _subset_loader(train, idx, batch_size, shuffle=True),
        _subset_loader(scored, idx, 512, shuffle=False),
        _subset_loader(test, list(range(2000)), 512, shuffle=False),
    )


def _small_cnn() -> nn.Module:
    def block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    return nn.Sequential(
        block(3, 32),
        nn.MaxPool2d(2),
        block(32, 64),
        nn.MaxPool2d(2),
        block(64, 64),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += int((model(x).argmax(1) == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


def _train_cnn(augment: bool, n_train: int, epochs: int) -> tuple[float, float]:
    set_seed()
    train_loader, scored_loader, test_loader = _cifar_subset(augment, n_train, CFG.batch_size)
    model = _small_cnn()
    opt = make_optimizer(model, CFG)
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return _accuracy(model, scored_loader), _accuracy(model, test_loader)


def warmup_ramps_from_almost_zero_to_the_peak() -> str:
    cfg = replace(CFG, warmup_steps=100)
    first = lr_at(0, TOTAL, cfg)
    peak = lr_at(99, TOTAL, cfg)
    assert first < peak, f"{first} !< {peak}"
    assert abs(peak - cfg.lr) < 1e-12, f"peak {peak} != lr {cfg.lr}"
    return f"step 0 lr {first:.2e} rises to {peak:.2e} at the end of warmup"


def cosine_decays_monotonically_after_warmup() -> str:
    cfg = replace(CFG, warmup_steps=100)
    lrs = [lr_at(s, TOTAL, cfg) for s in range(100, TOTAL)]
    assert all(b <= a + 1e-12 for a, b in pairwise(lrs)), "cosine is not monotonic"
    assert lrs[-1] < lrs[0] / 100, f"final lr {lrs[-1]:.2e} is not near zero"
    return f"lr falls {lrs[0]:.2e} to {lrs[-1]:.2e} without a single step up"


def a_constant_schedule_ignores_the_cosine() -> str:
    cfg = replace(CFG, warmup_steps=10, lr_schedule="constant")
    after = {lr_at(s, TOTAL, cfg) for s in range(10, TOTAL)}
    assert after == {cfg.lr}, f"expected one value, got {len(after)}"
    return f"every post-warmup step sits at {cfg.lr:.2e}"


def weight_decay_reaches_the_optimizer() -> str:
    model = nn.Linear(4, 2)
    for name in ("sgd", "adam", "adamw"):
        opt = make_optimizer(model, replace(CFG, optimizer=name, weight_decay=0.123))
        got = opt.param_groups[0]["weight_decay"]
        assert got == 0.123, f"{name} dropped weight_decay: {got}"
    return "sgd, adam and adamw all carry weight_decay 0.123"


def adamw_decouples_decay_and_adam_does_not() -> str:
    def drift(optimizer: str) -> float:
        set_seed()
        p = nn.Parameter(torch.ones(1) * 5.0)
        opt = {
            "adam": torch.optim.Adam([p], lr=0.1, weight_decay=0.5),
            "adamw": torch.optim.AdamW([p], lr=0.1, weight_decay=0.5),
        }[optimizer]
        for _ in range(20):
            opt.zero_grad()
            p.grad = torch.zeros_like(p)  # no data signal, decay only
            opt.step()
        return p.item()

    adam, adamw = drift("adam"), drift("adamw")
    assert adamw < adam, f"adamw {adamw} should shrink faster than adam {adam}"
    return f"with zero gradient, adamw pulls to {adamw:.3f} vs adam {adam:.3f}"


def a_bare_loop_learns_a_line() -> str:
    set_seed()
    x = torch.randn(256, 1)
    y = 3.0 * x - 1.0
    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    first = last = 0.0
    for step in range(200):
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first / 100, f"loss only moved {first:.4f} to {last:.4f}"
    return f"loss {first:.4f} to {last:.6f}; weight {model.weight.item():.3f} vs true 3.0"


def zero_grad_is_not_optional() -> str:
    set_seed()
    model = nn.Linear(1, 1)
    x, y = torch.randn(8, 1), torch.randn(8, 1)
    nn.functional.mse_loss(model(x), y).backward()
    once = model.weight.grad.clone()
    nn.functional.mse_loss(model(x), y).backward()
    twice = model.weight.grad.clone()
    assert torch.allclose(twice, once * 2), "gradients did not accumulate"
    return "a second backward without zero_grad doubles the gradient"


def halved_learning_rate_effect() -> str:
    # A subset, not the headline run: the point is the shape of the difference
    # at a fixed step budget, which is visible long before 10 epochs of MNIST.
    train_loader, test_loader = _mnist_subset(n_train=6000, batch_size=CFG.batch_size)
    epochs = 6
    base_losses, base_acc = _train_mlp(CFG.lr, train_loader, test_loader, epochs)
    half_losses, half_acc = _train_mlp(CFG.lr / 2, train_loader, test_loader, epochs)

    assert half_losses[-1] > base_losses[-1], (
        f"halved lr should still be behind at a fixed budget: "
        f"{half_losses[-1]:.4f} vs {base_losses[-1]:.4f}"
    )
    return (
        f"6000 MNIST images, {epochs} epochs: lr {CFG.lr:.0e} ends at loss "
        f"{base_losses[-1]:.4f} / acc {base_acc:.4f}, lr {CFG.lr / 2:.1e} at "
        f"{half_losses[-1]:.4f} / {half_acc:.4f}. Halving does not plateau lower, "
        f"it arrives later; the same budget buys less progress"
    )


def augmentation_effect_on_generalisation() -> str:
    # 2000 images and 30 epochs is chosen so the gap has room to open. The full
    # 30-epoch CIFAR run is the headline number; this is the ablation beside it.
    n_train, epochs = 2000, 30
    plain_train, plain_test = _train_cnn(augment=False, n_train=n_train, epochs=epochs)
    aug_train, aug_test = _train_cnn(augment=True, n_train=n_train, epochs=epochs)
    plain_gap, aug_gap = plain_train - plain_test, aug_train - aug_test

    # Only the gap is asserted. Across nine runs varying shuffle order, model
    # init and budget, the gap narrowed every time (-0.048 to -0.140), while
    # the test-accuracy delta changed sign (-0.047 to +0.109): at this size the
    # accuracy win is inside the noise, and asserting it gives a check that
    # passes on the seed. It failed here on the first try for exactly that.
    assert aug_gap < plain_gap, f"augmented gap {aug_gap:.4f} should be under {plain_gap:.4f}"
    return (
        f"{n_train} CIFAR images, {epochs} epochs: without augmentation train "
        f"{plain_train:.4f} vs test {plain_test:.4f}, a {plain_gap:.4f} gap; with it "
        f"{aug_train:.4f} vs {aug_test:.4f}, a {aug_gap:.4f} gap. Crops and flips cost "
        f"training accuracy to close the gap by {plain_gap - aug_gap:.4f}; the test "
        f"delta here is {aug_test - plain_test:+.4f}, which at this size is noise, not a win"
    )


CHECKS = [
    warmup_ramps_from_almost_zero_to_the_peak,
    cosine_decays_monotonically_after_warmup,
    a_constant_schedule_ignores_the_cosine,
    weight_decay_reaches_the_optimizer,
    adamw_decouples_decay_and_adam_does_not,
    a_bare_loop_learns_a_line,
    zero_grad_is_not_optional,
    halved_learning_rate_effect,
    augmentation_effect_on_generalisation,
]

if __name__ == "__main__":
    raise SystemExit(run("training", CHECKS))
