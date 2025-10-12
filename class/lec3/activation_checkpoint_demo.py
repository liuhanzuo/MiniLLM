"""
激活检查点（Activation Checkpointing）最小演示：对比无/有检查点时的显存峰值与用时。

运行（PowerShell）：
  python class/lec3/activation_checkpoint_demo.py --layers 24 --bs 8 --seq 1024 --hid 2048 --ff 8192 --use_ckpt 0
  python class/lec3/activation_checkpoint_demo.py --layers 24 --bs 8 --seq 1024 --hid 2048 --ff 8192 --use_ckpt 1

提示：需要 CUDA 才能统计显存（CPU 下仅给出用时）。脚本为简化的 Transformer-FFN 堆叠，不含注意力与嵌入，旨在放大激活开销。
"""

import argparse
import time
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


class FFNBlock(nn.Module):
    def __init__(self, hid: int, ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hid, ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ff, hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h  # 残差


class StackedFFN(nn.Module):
    def __init__(self, layers: int, hid: int, ff: int, use_ckpt: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([FFNBlock(hid, ff) for _ in range(layers)])
        self.use_ckpt = use_ckpt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_ckpt:
                x = cp.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


def run_once(model: nn.Module, x: torch.Tensor) -> Tuple[float, Optional[float]]:
    # 返回 (time_ms, peak_mem_MB)
    device = x.device
    use_cuda = device.type == 'cuda'
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    t0 = time.time()

    # 简单的训练一步：MSE 到零
    x = x.requires_grad_(True)
    y = model(x)
    loss = (y.float() ** 2).mean()
    loss.backward()

    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.time()

    peak = None
    if use_cuda:
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    return (t1 - t0) * 1000.0, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--bs', type=int, default=8, help='batch size')
    ap.add_argument('--seq', type=int, default=256, help='sequence length')
    ap.add_argument('--hid', type=int, default=256, help='hidden size')
    ap.add_argument('--ff', type=int, default=1024, help='ffn inner size')
    ap.add_argument('--use_ckpt', type=int, default=0, help='0/1 是否启用激活检查点')
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')

    # 构造输入：[bs, seq, hid]
    x = torch.randn(args.bs, args.seq, args.hid, device=device)

    model = StackedFFN(args.layers, args.hid, args.ff, use_ckpt=bool(args.use_ckpt)).to(device)
    # 预热
    with torch.no_grad():
        _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()

    time_ms, peak_mb = run_once(model, x)
    print(f"device={device}, layers={args.layers}, bs={args.bs}, seq={args.seq}, hid={args.hid}, ff={args.ff}")
    print(f"use_ckpt={bool(args.use_ckpt)}")
    print(f"time  : {time_ms:.2f} ms")
    if peak_mb is not None:
        print(f"peak  : {peak_mb:.1f} MB (allocated)")
        print("注：启用检查点通常显著降低峰值显存，但会增加少量重算时间。")
    else:
        print("CPU 模式下不统计显存，仅作功能演示。")


if __name__ == '__main__':
    main()
