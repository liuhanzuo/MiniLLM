"""
数据并行（DP）与分布式数据并行（DDP）最小可运行演示。

用法（PowerShell）：
- 单机单进程 DataParallel（自动使用多 GPU）：
  python class/lec3/dp_ddp_demo.py --mode dp --epochs 1 --batch_size 32

- 多进程 DDP（8 卡示例）：
  torchrun --nproc_per_node 8 class/lec3/dp_ddp_demo.py --mode ddp --epochs 1 --batch_size 32

说明：此脚本使用随机数据和一个极简 MLP，演示 DP/DDP 的封装与梯度同步路径。
"""

import argparse
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class RandDataset(Dataset):
    def __init__(self, n=1024, in_dim=512, n_classes=10, device='cpu'):
        self.X = torch.randn(n, in_dim)
        self.Y = torch.randint(0, n_classes, (n,))

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class TinyMLP(nn.Module):
    def __init__(self, in_dim=512, hid=1024, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def setup_ddp():
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        # 在 Windows 或未提供 NCCL 时退回 gloo，避免 CUDA+NCCL 不可用报错
        use_nccl = False
        try:
            import torch.distributed as _d
            use_nccl = torch.cuda.is_available() and getattr(_d, 'is_nccl_available', lambda: False)()
        except Exception:
            use_nccl = False
        backend = 'nccl' if use_nccl else 'gloo'
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['dp', 'ddp'], default='dp')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--in_dim', type=int, default=512)
    ap.add_argument('--hid', type=int, default=1024)
    ap.add_argument('--n_classes', type=int, default=10)
    args = ap.parse_args()

    rank, world_size, local_rank = setup_ddp()
    is_ddp = (args.mode == 'ddp' and world_size > 1)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    model = TinyMLP(args.in_dim, args.hid, args.n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    ds = RandDataset(n=2048, in_dim=args.in_dim, n_classes=args.n_classes)
    sampler = DistributedSampler(ds, shuffle=True) if is_ddp else None
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler)

    if args.mode == 'dp' and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[DP] Using DataParallel on devices: {list(range(torch.cuda.device_count()))}")
        model = nn.DataParallel(model)
    elif is_ddp:
        print(f"[DDP] rank={rank}/{world_size}, device={device}")
        model._ddp_params_and_buffers_to_ignore = set()
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None)
    else:
        print(f"[Single] device={device}")

    for ep in range(args.epochs):
        if is_ddp:
            sampler.set_epoch(ep)
        for it, (x, y) in enumerate(dl):
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            if (not is_ddp) or (rank == 0):
                if it % 10 == 0:
                    print(f"epoch {ep} iter {it} | loss {loss.item():.4f}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
