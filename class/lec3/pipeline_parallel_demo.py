"""
流水线并行（Pipeline Parallel, PP）最小可运行演示。

思路：将一个顺序网络分成 Stage 0/1 两段，分别放到 cuda:0 / cuda:1（若不可用则退化到单设备）。
用 micro-batch 将全局 batch 切分，按流水线交替在两个阶段上执行前向，统计简单的吞吐与“气泡”。

运行（PowerShell）：
  python class/lec3/pipeline_parallel_demo.py --global_bs 32 --micro_bs 4 --in_dim 1024 --hid 2048 --out_dim 1024
"""

import argparse
import time
import torch
import torch.nn as nn


class Stage0(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)


class Stage1(nn.Module):
    def __init__(self, hid, out_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, out_dim)
        )

    def forward(self, x):
        return self.seq(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--global_bs', type=int, default=32)
    ap.add_argument('--micro_bs', type=int, default=4)
    ap.add_argument('--in_dim', type=int, default=1024)
    ap.add_argument('--hid', type=int, default=2048)
    ap.add_argument('--out_dim', type=int, default=1024)
    args = ap.parse_args()

    assert args.global_bs % args.micro_bs == 0, "global_bs 必须能被 micro_bs 整除"
    micro_steps = args.global_bs // args.micro_bs

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= 2
    dev0 = torch.device('cuda:0') if use_cuda else torch.device('cpu')
    dev1 = torch.device('cuda:1') if use_cuda else dev0

    s0 = Stage0(args.in_dim, args.hid).to(dev0)
    s1 = Stage1(args.hid, args.out_dim).to(dev1)

    # 构造一批随机数据
    X = torch.randn(args.global_bs, args.in_dim, device=dev0)

    # baseline：顺序执行（在多 GPU 情况下将 Stage0 输出移到 Stage1 设备以避免设备不一致）
    t0 = time.time()
    h0 = s0(X)
    Y_baseline = s1(h0.to(dev1))
    if use_cuda:
        try:
            torch.cuda.synchronize(dev0)
            torch.cuda.synchronize(dev1)
        except TypeError:
            # 兼容老版本 API：无参数时同步所有设备
            torch.cuda.synchronize()
    t1 = time.time()

    # 流水线：micro-batch 切分 + 交替执行（仅演示前向，不含反向/梯度累计）
    outs = []
    t2 = time.time()
    act_queue = []
    for i in range(micro_steps + 1):
        # Stage 1 先消费上一个 micro 的输出
        if i > 0:
            a = act_queue.pop(0)
            outs.append(s1(a.to(dev1, non_blocking=True)))
        # Stage 0 处理当前 micro
        if i < micro_steps:
            xb = X[i*args.micro_bs:(i+1)*args.micro_bs].to(dev0, non_blocking=True)
            act_queue.append(s0(xb))
    if use_cuda:
        try:
            torch.cuda.synchronize(dev0)
            torch.cuda.synchronize(dev1)
        except TypeError:
            torch.cuda.synchronize()
    t3 = time.time()
    Y_pipe = torch.cat([o.to(dev0) for o in outs], dim=0)

    # 正确性与计时（将两者搬到同一设备再比较，默认在 dev0 比较）
    Yb = Y_baseline.to(dev0)
    Yp = Y_pipe  # 已在 dev0
    max_diff = (Yb - Yp).abs().max().item()
    print(f"use_cuda={use_cuda}, devices=({dev0},{dev1}), micro_steps={micro_steps}")
    print(f"baseline  time : {(t1-t0)*1000:.2f} ms")
    print(f"pipeline  time : {(t3-t2)*1000:.2f} ms  (仅前向示意，忽略通信成本)")
    print(f"max |Yb-Yp|    : {max_diff:.3e}")
    if micro_steps > 1:
        bubble = 2 / (micro_steps + 1)  # 两段流水线的简单气泡近似
        print(f"bubble ratio~ : {bubble*100:.1f}%  (示意)")


if __name__ == '__main__':
    main()
