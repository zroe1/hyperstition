"""Unified learning rate schedules for all training pipelines."""

import math
from typing import Literal

LRSchedule = Literal["cosine", "constant"]


def get_lr(
    batch_idx: int,
    total_batches: int,
    lr_max: float,
    lr_min: float = 0.0,
    warmup_pct: float = 0.05,
    schedule: LRSchedule = "cosine",
) -> float:
    """Compute learning rate at a given step.

    All schedules use linear warmup from 0 to lr_max over warmup_pct of total steps.
    After warmup:
      - "cosine": cosine decay from lr_max to lr_min
      - "constant": stays at lr_max (lr_min is ignored)
    """
    warmup_steps = int(total_batches * warmup_pct)
    if batch_idx < warmup_steps:
        return lr_max * (batch_idx / max(1, warmup_steps))
    if schedule == "constant":
        return lr_max
    # cosine decay
    progress = (batch_idx - warmup_steps) / max(1, total_batches - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
