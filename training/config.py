# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Every hyperparameter, named and in one place."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 30
    lr: float = 3e-3
    warmup_steps: int = 500
    weight_decay: float = 5e-4
    dropout: float = 0.1
    label_smoothing: float = 0.0
    optimizer: str = "adamw"  # sgd | adam | adamw
    lr_schedule: str = "cosine"  # constant | cosine
    seed: int = 1859
