from __future__ import annotations
import math
from dataclasses import dataclass

from torch.optim import Optimizer


@dataclass
class CosineWithFloor:
    base_lr: float
    total_steps: int

    def get_lr(self, step: int) -> float:
        # match original: lr/10 + 0.5*lr*(1+cos(pi*step/total))
        lr = self.base_lr / 10 + 0.5 * self.base_lr * (1 + math.cos(math.pi * step / self.total_steps))
        return lr

    def step(self, optim: Optimizer, step: int):
        lr = self.get_lr(step)
        for pg in optim.param_groups:
            pg["lr"] = lr
        return lr
