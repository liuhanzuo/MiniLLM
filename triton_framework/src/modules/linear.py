from __future__ import annotations
import torch
from torch import nn
from torch.autograd import Function

try:
    from ..kernels.matmul import matmul as triton_matmul
    _HAS_TRITON_MATMUL = True
except Exception:
    _HAS_TRITON_MATMUL = False


class TritonMatmulFunction(Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor):
        # A: [M, K], B: [K, N]
        use_triton = _HAS_TRITON_MATMUL and A.is_cuda and B.is_cuda
        C = triton_matmul(A, B) if use_triton else A @ B
        # Save for backward if needed. Save both to enable computing grads even if only one input requires grad
        ctx.save_for_backward(A, B)
        return C

    @staticmethod
    def backward(ctx, dC: torch.Tensor):
        A, B = ctx.saved_tensors
        dA = dB = None
        # Fallback to torch.mm for grad correctness first
        if ctx.needs_input_grad[0]:
            # dA = dC @ B^T
            dA = dC @ B.transpose(-2, -1)
        if ctx.needs_input_grad[1]:
            # dB = A^T @ dC
            dB = A.transpose(-2, -1) @ dC
        return dA, dB


class TritonLinear(nn.Module):
    """A Linear layer that uses the Triton matmul kernel in forward.

    Note: Backward currently uses PyTorch matmul for correctness. You can
    later switch to Triton-based gradients by replacing the ops in backward.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype=None, device=None, use_triton: bool = True):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        self.use_triton = use_triton
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, in_features]; weight: [out_features, in_features]
        w_t = self.weight.transpose(0, 1).contiguous()  # [in, out]
        in_features = w_t.shape[0]
        out_features = w_t.shape[1]
        orig_shape = x.shape
        # Flatten leading dims to 2D for Triton matmul
        x2d = x.reshape(-1, in_features)
        use_triton = self.use_triton and x2d.is_cuda
        y2d = TritonMatmulFunction.apply(x2d, w_t) if use_triton else x2d @ w_t
        y = y2d.view(*orig_shape[:-1], out_features)
        if self.bias is not None:
            y = y + self.bias
        return y
