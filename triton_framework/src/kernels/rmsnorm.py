from __future__ import annotations
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_forward(x_ptr, w_ptr, y_ptr, eps, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * N + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x2 = x * x
    s = tl.sum(x2, axis=0)
    mean = s / N
    rstd = tl.math.rsqrt(mean + eps)

    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    y = x * rstd * w
    tl.store(y_ptr + offs, y, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda, "rmsnorm requires CUDA tensors"
    B, N = x.shape[:-1], x.shape[-1]
    y = torch.empty_like(x)
    x_2d = x.reshape(-1, N)
    y_2d = y.reshape(-1, N)

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_SIZE <= 1024 else 8

    grid = (x_2d.shape[0],)
    _rmsnorm_forward[grid](
        x_2d, weight, y_2d, eps,
        N=N, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return y
