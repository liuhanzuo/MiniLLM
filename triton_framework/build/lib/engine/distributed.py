from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


@dataclass
class DDPContext:
    enabled: bool
    local_rank: int
    global_rank: int
    world_size: int
    device: torch.device


def init_distributed(backend: str = "nccl") -> Optional[DDPContext]:
    """Initialize torch.distributed using torchrun env vars if present.
    Returns a DDPContext or None if not in distributed mode.
    """
    if "RANK" not in os.environ:
        return None

    rank = int(os.environ["RANK"])  # global rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])  # total processes

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    return DDPContext(
        enabled=True,
        local_rank=local_rank,
        global_rank=rank,
        world_size=world_size,
        device=device,
    )


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()
