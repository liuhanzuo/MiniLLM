#!/usr/bin/env python3
from __future__ import annotations
import time
import torch

from triton_framework.modules.simple_lm import RMSNorm, rmsnorm_ref


def bench():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, T, D = 32, 256, 1024
    x = torch.randn(B, T, D, device=device, dtype=torch.float32)
    w = torch.ones(D, device=device, dtype=torch.float32)

    # warmup
    m = RMSNorm(D, use_triton=True).to(device)
    m.weight.data.copy_(w)
    for _ in range(10):
        y = m(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None

    # triton timing
    iters = 50
    t0 = time.time()
    for _ in range(iters):
        y = m(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.time()

    # reference timing
    t2 = time.time()
    for _ in range(iters):
        y2 = rmsnorm_ref(x, w)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t3 = time.time()

    diff = (y - y2).abs().max().item()
    print(f"max diff: {diff:.3e}")
    print(f"triton: {(t1 - t0)/iters*1000:.2f} ms/iter | torch: {(t3 - t2)/iters*1000:.2f} ms/iter")


if __name__ == "__main__":
    bench()
