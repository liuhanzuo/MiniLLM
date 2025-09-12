from __future__ import annotations
import torch
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    from ..configs.configs import SILU_AUTOTUNE_CONFIGS, SILU_AUTOTUNE_KEY
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


def _silu_cpu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


if _TRITON_AVAILABLE:
    @triton.jit
    def _silu_forward(x_ptr, y_ptr, size: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < size
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = x * tl.sigmoid(x)
        tl.store(y_ptr + offs, y, mask=mask)


def silu(x: torch.Tensor) -> torch.Tensor:
    if not (_TRITON_AVAILABLE and x.is_cuda):
        return _silu_cpu(x)
    y = torch.empty_like(x)
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    size = x_flat.numel()
    block = SILU_AUTOTUNE_CONFIGS[0].kwargs.get("BLOCK", 1024)
    warps = getattr(SILU_AUTOTUNE_CONFIGS[0], "num_warps", 4)
    grid = (triton.cdiv(size, block),)
    _silu_forward[grid](x_flat, y_flat, size=size, BLOCK=block, num_warps=warps)
    return y
