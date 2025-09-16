"""
批大小缩放（Batch Size Scaling）与梯度累积（Gradient Accumulation）对比

目标：比较两种训练设置在等效“全局 batch”下的表现：
  A) 大 batch，无梯度累积（accum_steps=1）
  B) 小 batch，梯度累积 K 次（accum_steps=K），等效全局 batch 相同

并考察两种学习率策略：
  1) 线性缩放（LR *= accum_steps 或按全局 batch 比例缩放）
  2) 不缩放（LR 固定）

用法示例（根据环境使用 python 或 python3）：
    python3 class/lec6/batchsize_gradacc_compare.py \
    --epochs 5 --base-batch 64 --accum-steps 8 --lr 1e-3 --scale-lr \
    --device cuda --save-csv ./bs_ga_compare.csv

输出：逐 epoch 的 train/val loss & acc、有效学习率、耗时；可选保存 CSV。

快速自测（CPU，极短回合）：
    python3 class/lec6/batchsize_gradacc_compare.py --epochs 2 --base-batch 64 --accum-steps 4 --lr 1e-3 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split


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


def make_toy_binary_dataset(n: int = 60_000, d: int = 64, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, generator=g)
    logits = X @ w + 0.25 * torch.randn(n, generator=g)
    y = (logits > 0).long()
    return X, y


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 64, hidden: int = 256, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        total_acc += (logits.argmax(dim=-1) == yb).float().sum().item()
        n += bs
    return total_loss / n, total_acc / n


def train_epoch_accum(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device, accum_steps: int) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    opt.zero_grad(set_to_none=True)
    step_idx = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb) / accum_steps
        loss.backward()
        if (step_idx + 1) % accum_steps == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        step_idx += 1
        bs = xb.size(0)
        total_loss += loss.item() * bs * accum_steps  # 还原到未平均的 loss
        total_acc += (logits.argmax(dim=-1) == yb).float().sum().item()
        n += bs
    # 若最后不足 accum_steps，仍需 step 一次
    if step_idx % accum_steps != 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
    return total_loss / n, total_acc / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--base-batch", type=int, default=64, help="小 batch 大小（将进行梯度累积）")
    ap.add_argument("--accum-steps", type=int, default=8, help="梯度累积步数 K")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--input-dim", type=int, default=64)
    ap.add_argument("--train-size", type=int, default=50_000)
    ap.add_argument("--val-size", type=int, default=10_000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale-lr", action="store_true", help="按等效全局 batch 线性放大学习率")
    ap.add_argument("--save-csv", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    # 数据
    X, y = make_toy_binary_dataset(n=args.train_size + args.val_size, d=args.input_dim, seed=args.seed)
    ds = TensorDataset(X, y)
    train_ds, val_ds = random_split(ds, [args.train_size, args.val_size], generator=torch.Generator().manual_seed(args.seed))

    # 两个方案：
    # A: 大 batch = base_batch * accum_steps，无梯度累积
    big_batch = args.base_batch * args.accum_steps
    loader_big = DataLoader(train_ds, batch_size=big_batch, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    loader_small = DataLoader(train_ds, batch_size=args.base_batch, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # 相同的随机初始权重用于两条分支
    def make_model():
        m = TinyMLP(in_dim=args.input_dim, hidden=args.hidden, out_dim=2)
        return m

    # 记录
    rows: List[List] = []

    for scale_lr in ([False, True] if args.scale_lr else [False]):
        label = f"scaleLR={scale_lr}"

        # 构建两个模型（同初始种子下权重一致）
        set_seed(args.seed)
        model_A = make_model().to(device)
        set_seed(args.seed)
        model_B = make_model().to(device)

        # 学习率设置
        lr_A = args.lr * (args.accum_steps if scale_lr else 1.0)
        lr_B = args.lr

        opt_A = AdamW(model_A.parameters(), lr=lr_A, weight_decay=args.weight_decay)
        opt_B = AdamW(model_B.parameters(), lr=lr_B, weight_decay=args.weight_decay)

        print(f"==== {label} | epochs={args.epochs}, base_batch={args.base_batch}, accum_steps={args.accum_steps}, big_batch={big_batch}, lrA={lr_A}, lrB={lr_B} ====")
        print("epoch, A_train_loss, A_train_acc, A_val_loss, A_val_acc, B_train_loss, B_train_acc, B_val_loss, B_val_acc, sec")

        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            # A: 大 batch 无累积
            trA_loss, trA_acc = train_epoch_accum(model_A, loader_big, opt_A, device, accum_steps=1)
            # B: 小 batch + 累积 K
            trB_loss, trB_acc = train_epoch_accum(model_B, loader_small, opt_B, device, accum_steps=args.accum_steps)
            vA_loss, vA_acc = evaluate(model_A, val_loader, device)
            vB_loss, vB_acc = evaluate(model_B, val_loader, device)
            sec = time.time() - t0
            print(f"{ep:03d}, {trA_loss:.4f}, {trA_acc:.4f}, {vA_loss:.4f}, {vA_acc:.4f}, {trB_loss:.4f}, {trB_acc:.4f}, {vB_loss:.4f}, {vB_acc:.4f}, {sec:.2f}")
            rows.append([label, ep, lr_A, lr_B, args.base_batch, args.accum_steps, big_batch, trA_loss, trA_acc, vA_loss, vA_acc, trB_loss, trB_acc, vB_loss, vB_acc, sec])

    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "label", "epoch", "lrA", "lrB", "base_batch", "accum_steps", "big_batch",
                "A_train_loss", "A_train_acc", "A_val_loss", "A_val_acc",
                "B_train_loss", "B_train_acc", "B_val_loss", "B_val_acc", "sec"
            ])
            w.writerows(rows)
        print(f"Saved metrics to {args.save_csv}")


if __name__ == "__main__":
    main()
