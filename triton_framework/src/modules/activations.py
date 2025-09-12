from __future__ import annotations
import math
import torch
from torch import nn
from torch.autograd import Function

# Try import Triton kernels; fall back to PyTorch if unavailable
try:
    from ..kernels.silu import silu as triton_silu
    from ..kernels.gelu import gelu as triton_gelu
    from ..kernels.dropout import dropout as triton_dropout
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


class SiluFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
        use_triton = bool(use_triton) and _HAS_TRITON and x.is_cuda
        y = triton_silu(x) if use_triton else x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        grad_x = grad_out * (s * (1 + x * (1 - s)))
        return grad_x, None


class GeluFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, approximate: str = "tanh", use_triton: bool = True) -> torch.Tensor:
        use_triton = bool(use_triton) and _HAS_TRITON and x.is_cuda
        if use_triton:
            y = triton_gelu(x, approximate=approximate)
        else:
            y = torch.nn.functional.gelu(x, approximate=approximate)
        ctx.save_for_backward(x)
        ctx.approximate = approximate
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        approximate = ctx.approximate
        if approximate == "tanh":
            c = 0.7978845608028654  # sqrt(2/pi)
            u = c * (x + 0.044715 * x * x * x)
            t = torch.tanh(u)
            du = c * (1 + 0.134145 * x * x)
            grad = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du
        else:
            inv_sqrt2 = 0.7071067811865476
            inv_sqrt2pi = 0.3989422804014327
            grad = 0.5 * (1 + torch.erf(x * inv_sqrt2)) + x * inv_sqrt2pi * torch.exp(-0.5 * x * x)
        return grad_out * grad, None, None


class DropoutFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, p: float, training: bool = True, use_triton: bool = True, seed: int | None = None):
        if not training or p == 0.0:
            ctx.p = p
            ctx.training = False
            return x, torch.ones_like(x, dtype=torch.bool)
        use_triton = bool(use_triton) and _HAS_TRITON and x.is_cuda
        if use_triton:
            y, mask = triton_dropout(x, p, seed)
        else:
            # CPU fallback using torch for mask (to match behavior)
            if seed is not None:
                if x.is_cuda:
                    torch.cuda.manual_seed(seed)
                else:
                    torch.manual_seed(seed)
            mask = torch.rand_like(x) > p
            scale = 1.0 / (1.0 - p)
            y = x * mask * scale
        ctx.p = p
        ctx.training = True
        ctx.save_for_backward(mask)
        return y, mask

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_mask_unused):
        if not ctx.training:
            return grad_y, None, None, None, None
        (mask,) = ctx.saved_tensors
        p = ctx.p
        scale = 1.0 / (1.0 - p)
        grad_x = grad_y * mask * scale
        return grad_x, None, None, None, None


class SiLU(nn.Module):
    def __init__(self, use_triton: bool = True):
        super().__init__()
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SiluFunction.apply(x, self.use_triton)


class GELU(nn.Module):
    def __init__(self, approximate: str = "tanh", use_triton: bool = True):
        super().__init__()
        self.approximate = approximate
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GeluFunction.apply(x, self.approximate, self.use_triton)


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5, use_triton: bool = True):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = float(p)
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor):
        training = self.training
        if not training or self.p == 0.0:
            return x
        y, _ = DropoutFunction.apply(x, self.p, training, self.use_triton, None)
        return y
