from __future__ import annotations
"""Fused MLP building blocks with explicit backward.

Replaces the previous placeholder/incomplete code.

Provides:
  - BaselineMLP: standard PyTorch modules.
  - FusedMLP: custom autograd.Function implementing (Linear -> SiLU -> Linear) with manual backward.

Design goals:
  - Python 3.8 compatible (no pattern matching, no | unions).
  - Easy to later swap matmuls / activation with Triton kernels.
  - Clear separation of forward saved tensors and gradient math.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.autograd import Function
from .linear import TritonLinear
from .activations import SiLU, GELU, Dropout


def silu_forward(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def silu_backward(grad_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(x)
    return grad_out * (sig * (1 + x * (1 - sig)))


class _FusedLinearSiluLinear(Function):
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                w1: torch.Tensor,
                b1: Optional[torch.Tensor],
                w2: torch.Tensor,
                b2: Optional[torch.Tensor]) -> torch.Tensor:
        # First linear
        h_pre = F.linear(x, w1, b1)          # [B, H]
        h_act = silu_forward(h_pre)          # [B, H]
        out = F.linear(h_act, w2, b2)        # [B, O]
        ctx.save_for_backward(x, w1, h_pre, h_act, w2, b1, b2)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        x, w1, h_pre, h_act, w2, b1, b2 = ctx.saved_tensors

        # second linear
        # out = h_act @ w2^T + b2
        grad_h_act = grad_out.matmul(w2)                                  # [B, H]
        # w2 shape: [d_out, d_hidden]; grad_w2 shape must match
        grad_w2 = grad_out.transpose(0, 1).matmul(h_act)                  # [d_out, d_hidden]
        grad_b2 = grad_out.sum(0) if b2 is not None else None

            # activation SiLU
        grad_h_pre = silu_backward(grad_h_act, h_pre)       # [B, H]

            # first linear: h_pre = x @ w1^T + b1
        grad_x = grad_h_pre.matmul(w1)                                      # [B, D_in]
        # w1 shape: [d_hidden, d_in]; grad_w1 must match
        grad_w1 = grad_h_pre.transpose(0, 1).matmul(x)                      # [d_hidden, d_in]
        grad_b1 = grad_h_pre.sum(0) if b1 is not None else None

        return grad_x, grad_w1, grad_b1, grad_w2, grad_b2


class FusedMLP(nn.Module):
    """(Linear -> SiLU -> Linear) with manual backward.

    Args:
        d_in: input dimension
        d_hidden: hidden dimension
        d_out: output dimension
        bias: include bias in both linear layers
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, bias: bool = True):
        super(FusedMLP, self).__init__()
        self.w1 = nn.Parameter(torch.empty(d_hidden, d_in))
        self.b1 = nn.Parameter(torch.zeros(d_hidden)) if bias else None
        self.w2 = nn.Parameter(torch.empty(d_out, d_hidden))
        self.b2 = nn.Parameter(torch.zeros(d_out)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        if self.b1 is not None:
            nn.init.zeros_(self.b1)
        if self.b2 is not None:
            nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FusedLinearSiluLinear.apply(x, self.w1, self.b1, self.w2, self.b2)


class BaselineMLP(nn.Module):
    """Reference MLP using stock PyTorch layers."""
    def __init__(self, d_in: int, d_hidden: int, d_out: int, bias: bool = True):
        super(BaselineMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden, bias=bias),
            nn.SiLU(),
            nn.Linear(d_hidden, d_out, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TritonMLP(nn.Module):
    """MLP stack built from Triton-enabled building blocks (Linear + Activation + Dropout + Linear).

    Args:
        d_in: input dimension
        d_hidden: hidden dimension
        d_out: output dimension
        activation: 'silu' or 'gelu'
        dropout: dropout probability (applied after activation)
        bias: whether to use bias in linear layers
        use_triton: enable Triton kernels where available (matmul + activation + dropout)
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int,
                 activation: str = 'silu', dropout: float = 0.0,
                 bias: bool = True, use_triton: bool = True):
        super(TritonMLP, self).__init__()
        self.fc1 = TritonLinear(d_in, d_hidden, bias=bias, use_triton=use_triton)
        if activation == 'silu':
            self.act = SiLU(use_triton=use_triton)
        elif activation == 'gelu':
            self.act = GELU(use_triton=use_triton)
        else:
            raise ValueError('Unsupported activation: %s' % activation)
        self.drop = Dropout(p=dropout, use_triton=use_triton) if dropout > 0 else nn.Identity()
        self.fc2 = TritonLinear(d_hidden, d_out, bias=bias, use_triton=use_triton)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


def self_test(device: Optional[str] = None) -> bool:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    B, D_IN, D_H, D_OUT = 4, 16, 32, 8
    x = torch.randn(B, D_IN, device=device, requires_grad=True)

    # Fused manual backward vs baseline
    fused = FusedMLP(D_IN, D_H, D_OUT).to(device)
    base = BaselineMLP(D_IN, D_H, D_OUT).to(device)
    with torch.no_grad():
        base.net[0].weight.copy_(fused.w1)
        base.net[2].weight.copy_(fused.w2)
        if fused.b1 is not None:
            base.net[0].bias.copy_(fused.b1)
        if fused.b2 is not None:
            base.net[2].bias.copy_(fused.b2)
    y_f = fused(x)
    y_b = base(x)
    assert torch.allclose(y_f, y_b, atol=1e-6, rtol=1e-5), 'forward mismatch (fused vs baseline)'
    g = torch.randn_like(y_f)
    y_f.backward(g, retain_graph=True)
    grad_x_f = x.grad.clone()
    x.grad.zero_()
    y_b.backward(g)
    grad_x_b = x.grad
    assert torch.allclose(grad_x_f, grad_x_b, atol=1e-6, rtol=1e-5), 'grad mismatch (fused vs baseline)'

    # TritonMLP (no dropout) vs baseline for functional equivalence
    x.grad = None
    triton_mlp = TritonMLP(D_IN, D_H, D_OUT, activation='silu', dropout=0.0, use_triton=False).to(device)
    # align weights
    with torch.no_grad():
        triton_mlp.fc1.weight.copy_(base.net[0].weight)
        triton_mlp.fc2.weight.copy_(base.net[2].weight)
        if triton_mlp.fc1.bias is not None:
            triton_mlp.fc1.bias.copy_(base.net[0].bias)
        if triton_mlp.fc2.bias is not None:
            triton_mlp.fc2.bias.copy_(base.net[2].bias)
    y_t = triton_mlp(x.detach().requires_grad_(True))
    y_b2 = base(x.detach())
    assert torch.allclose(y_t, y_b2, atol=1e-6, rtol=1e-5), 'forward mismatch (triton vs baseline)'
    return True


if __name__ == '__main__':
    ok = self_test()
    print('FusedMLP and TritonMLP test passed:', ok)
