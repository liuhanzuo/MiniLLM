from __future__ import annotations
import torch
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


def _gelu_cpu(x: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
    if approximate == "tanh":
        # PyTorch's approximate='tanh'
        return torch.nn.functional.gelu(x, approximate="tanh")
    return torch.nn.functional.gelu(x, approximate="none")


if _TRITON_AVAILABLE:
    @triton.jit
    def _gelu_forward(x_ptr, y_ptr, size: tl.constexpr, BLOCK: tl.constexpr, approximate_tanh: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < size
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        if approximate_tanh:
            # tanh approximation: 0.5 * x * (1 + tanh(\sqrt{2/pi} * (x + 0.044715 x^3)))
            c = 0.7978845608028654  # sqrt(2/pi)
            y = 0.5 * x * (1 + tl.tanh(c * (x + 0.044715 * x * x * x)))
        else:
            # fall back to erf-based GELU
            y = 0.5 * x * (1 + tl.erf(x * 0.7071067811865476))  # 1/sqrt(2)
        tl.store(y_ptr + offs, y, mask=mask)


def gelu(x: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
    if not (_TRITON_AVAILABLE and x.is_cuda):
        return _gelu_cpu(x, approximate)
    y = torch.empty_like(x)
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    size = x_flat.numel()
    BLOCK = 1024
    approx_tanh = approximate == "tanh"
    grid = (triton.cdiv(size, BLOCK),)
    _gelu_forward[grid](x_flat, y_flat, size=size, BLOCK=BLOCK, approximate_tanh=approx_tanh, num_warps=4)
    return y
