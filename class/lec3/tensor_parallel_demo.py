"""
张量并行（Tensor Parallel, TP）最小可运行演示（列切分线性层）。

思路：将线性层的权重按输出维度切分到多卡，各卡并行计算其局部输出，最后在 dim=-1 上拼接得到完整输出。

限制：
- 需要至少 2 张 GPU（否则自动退化为单卡）。
- 仅演示前向的并行与拼接，不含反向/优化器状态分片等复杂内容。

运行（PowerShell）：
  python class/lec3/tensor_parallel_demo.py --in_dim 1024 --out_dim 4096 --parts 2
"""

import argparse
import torch
import torch.nn as nn


class ShardedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, parts: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.parts = max(1, min(parts, torch.cuda.device_count() if torch.cuda.is_available() else 1))
        # 创建分片参数（存放在各自设备上）
        self.weight_shards = nn.ParameterList()
        self.bias_shards = nn.ParameterList()
        shard_out = out_dim // self.parts
        last_out = out_dim - shard_out * (self.parts - 1)
        for i in range(self.parts):
            od = shard_out if i < self.parts - 1 else last_out
            w = nn.Parameter(torch.empty(od, in_dim))
            b = nn.Parameter(torch.zeros(od))
            nn.init.kaiming_uniform_(w, a=5**0.5)
            self.weight_shards.append(w)
            self.bias_shards.append(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 放到第 0 个设备或 CPU
        if torch.cuda.is_available() and self.parts > 1:
            dev0 = torch.device('cuda:0')
            x0 = x.to(dev0)
            outs = []
            for i in range(self.parts):
                dev = torch.device(f'cuda:{i}')
                w = self.weight_shards[i].to(dev)
                b = self.bias_shards[i].to(dev)
                xi = x0.to(dev)
                yi = torch.nn.functional.linear(xi, w, b)
                outs.append(yi.to(dev0))
            y = torch.cat(outs, dim=-1)
            return y
        else:
            # 单卡/CPU 退化运行：依然按照分片顺序计算并在同设备拼接
            outs = []
            for i in range(self.parts):
                yi = torch.nn.functional.linear(x, self.weight_shards[i], self.bias_shards[i])
                outs.append(yi)
            return torch.cat(outs, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--in_dim', type=int, default=1024)
    ap.add_argument('--out_dim', type=int, default=4096)
    ap.add_argument('--parts', type=int, default=2)
    args = ap.parse_args()

    x = torch.randn(args.bs, args.in_dim)
    tp_linear = ShardedLinear(args.in_dim, args.out_dim, args.parts)
    y = tp_linear(x)
    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(y.shape)} | parts={tp_linear.parts}, cuda={torch.cuda.is_available()}")


if __name__ == '__main__':
    main()
