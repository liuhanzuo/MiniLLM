from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import OptimConfig


@dataclass
class TrainConfig:
    max_steps: int = 1000
    log_interval: int = 50
    amp: bool = True


class Trainer:
    def __init__(self, model: nn.Module, dataloader: DataLoader, device: torch.device, *,
                 optim_cfg: OptimConfig = OptimConfig(), cfg: TrainConfig = TrainConfig()):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.optim = optim_cfg.build(self.model.parameters())
        self.cfg = cfg
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.amp))

    def train(self):
        self.model.train()
        step = 0
        data_iter = iter(self.dataloader)
        t0 = time.time()
        while step < self.cfg.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            self.optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

            if step % self.cfg.log_interval == 0:
                dt = time.time() - t0
                print(f"step {step:5d} | loss {loss.item():.4f} | dt {dt:.2f}s")
                t0 = time.time()
            step += 1

        return self.model
