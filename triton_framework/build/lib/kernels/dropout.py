from __future__ import annotations
import torch
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


def _dropout_cpu(x: torch.Tensor, p: float, seed: int | None = None):
    if p <= 0:
        return x, None
    if seed is not None:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
    else:
        gen = None
    mask = torch.rand_like(x, generator=gen) > p
    scale = 1.0 / (1.0 - p)
    return x * mask * scale, mask


if _TRITON_AVAILABLE:
    @triton.jit
    def _dropout_forward(x_ptr, y_ptr, mask_ptr, p, size: tl.constexpr, BLOCK: tl.constexpr, seed):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < size
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        # per-element RNG using counter-based philox via tl.rand if available
        r = tl.rand(seed, offs)
        keep = r > p
        scale = 1.0 / (1.0 - p)
        y = tl.where(keep, x * scale, 0.0)
        tl.store(y_ptr + offs, y, mask=mask)
        tl.store(mask_ptr + offs, keep.to(tl.int1), mask=mask)


def dropout(x: torch.Tensor, p: float, seed: int | None = None):
    assert 0.0 <= p < 1.0
    if not (_TRITON_AVAILABLE and x.is_cuda):
        return _dropout_cpu(x, p, seed)
    y = torch.empty_like(x)
    mask = torch.empty_like(x, dtype=torch.bool)
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    mask_flat = mask.view(-1)
    size = x_flat.numel()
    BLOCK = 1024
    if seed is None:
        seed = torch.randint(0, 2**31 - 1, (1,), device=x.device).item()
    grid = (triton.cdiv(size, BLOCK),)
    _dropout_forward[grid](x_flat, y_flat, mask_flat, p, size=size, BLOCK=BLOCK, seed=seed, num_warps=4)
    return y, mask
