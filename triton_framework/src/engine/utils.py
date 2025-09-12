from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> torch.device:
    if name is None or name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU")
        name = "cpu"
    return torch.device(name)


@dataclass
class OptimConfig:
    lr: float = 1e-3

    def build(self, params):
        return torch.optim.AdamW(params, lr=self.lr)
