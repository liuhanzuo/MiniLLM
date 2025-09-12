from __future__ import annotations
from typing import Optional

import torch
from torch import nn


class Inferencer:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.model(x)
