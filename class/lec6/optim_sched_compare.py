"""
优化器与学习率调度器对比实验

用一个小型 MLP 在可控的合成二分类数据上进行训练，对比不同优化器（SGD/Adam/AdamW/RMSprop）
以及不同调度器（无/StepLR/CosineAnnealingLR/OneCycleLR）的收敛曲线与最终精度。

用法示例（根据环境使用 python 或 python3）：
  python3 class/lec6/optim_sched_compare.py \
    --optimizer adamw --scheduler cosine --epochs 10 --batch-size 256 \
    --lr 1e-3 --device cuda --seed 42 --save-csv ./optim_cosine.csv

快速自测（CPU，极短回合）：
  python3 class/lec6/optim_sched_compare.py --epochs 2 --batch-size 256 --lr 1e-3 --device cpu

可选参数见 --help。

输出：每个 epoch 的 loss/acc/lr/耗时，控制台打印；若指定 --save-csv 则保存到 CSV。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, TensorDataset, random_split


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# Synthetic dataset
# -----------------------------

def make_toy_binary_dataset(n: int = 50_000, d: int = 64, n_classes: int = 2, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """构造线性可分但带噪声的二分类数据。
    X ~ N(0, I)，使用一个随机超平面 w 生成标签，再加入小噪声后通过阈值得到 0/1。
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, generator=g)
    logits = X @ w + 0.25 * torch.randn(n, generator=g)
    y = (logits > 0).long()
    return X, y


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 64, hidden: int = 256, out_dim: int = 2, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainConfig:
    optimizer: str
    scheduler: str
    lr: float
    epochs: int
    batch_size: int
    weight_decay: float
    momentum: float
    device: str
    seed: int
    save_csv: Optional[str]
    onecycle_pct: float


def build_optimizer(name: str, params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float, momentum: float):
    name = name.lower()
    if name == "sgd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(name: str, optimizer: torch.optim.Optimizer, epochs: int, steps_per_epoch: int, base_lr: float, onecycle_pct: float):
    name = name.lower()
    if name in ("none", "null", "off"):
        return None
    if name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "onecycle":
        # OneCycle 需要每 step 调度
        total_steps = epochs * steps_per_epoch
        total_steps = max(total_steps, 1)
        return OneCycleLR(optimizer, max_lr=base_lr, total_steps=total_steps, pct_start=onecycle_pct)
    raise ValueError(f"Unknown scheduler: {name}")


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def train_once(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, sched, device: torch.device, onecycle: bool) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        if sched is not None and onecycle:
            sched.step()
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), yb) * bs
        n += bs
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, yb) * bs
        n += bs
    return total_loss / n, total_acc / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw", "rmsprop"])
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "step", "cosine", "onecycle"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--input-dim", type=int, default=64)
    p.add_argument("--train-size", type=int, default=50_000)
    p.add_argument("--val-size", type=int, default=10_000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-csv", type=str, default=None)
    p.add_argument("--onecycle-pct", type=float, default=0.3, help="OneCycleLR pct_start")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    # data
    X, y = make_toy_binary_dataset(n=args.train_size + args.val_size, d=args.input_dim, seed=args.seed)
    ds = TensorDataset(X, y)
    train_len = args.train_size
    val_len = args.val_size
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # model
    model = TinyMLP(in_dim=args.input_dim, hidden=args.hidden, out_dim=2).to(device)

    # optimizer & scheduler
    opt = build_optimizer(args.optimizer, model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    steps_per_epoch = max(1, math.ceil(train_len / args.batch_size))
    sched = build_scheduler(args.scheduler, opt, args.epochs, steps_per_epoch, args.lr, args.onecycle_pct)

    print(f"Device={device}, Optimizer={args.optimizer}, Scheduler={args.scheduler}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print("epoch, train_loss, train_acc, val_loss, val_acc, lr, sec")

    rows: List[List] = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        train_loss, train_acc = train_once(model, train_loader, opt, sched, device, onecycle=(args.scheduler == "onecycle"))
        if sched is not None and args.scheduler != "onecycle":
            sched.step()
        val_loss, val_acc = evaluate(model, val_loader, device)
        lr_now = opt.param_groups[0]["lr"]
        sec = time.time() - t_ep
        print(f"{epoch:03d}, {train_loss:.4f}, {train_acc:.4f}, {val_loss:.4f}, {val_acc:.4f}, {lr_now:.6f}, {sec:.2f}")
        rows.append([epoch, train_loss, train_acc, val_loss, val_acc, lr_now, sec])

    total = time.time() - t0
    print(f"Total seconds: {total:.2f}")

    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "sec"])
            w.writerows(rows)
        print(f"Saved metrics to {args.save_csv}")


if __name__ == "__main__":
    main()
