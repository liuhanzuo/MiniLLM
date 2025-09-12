from __future__ import annotations
import torch
from torch import nn

try:
    from ..kernels import rmsnorm as triton_rmsnorm
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return x * weight


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_triton: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.use_triton = use_triton and _TRITON_AVAILABLE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton and x.is_cuda:
            return triton_rmsnorm(x, self.weight, self.eps)
        return rmsnorm_ref(x, self.weight, self.eps)


class TinyLM(nn.Module):
    def __init__(self, d_model: int, hidden: int, vocab_size: int, rms_eps: float = 1e-5, use_triton: bool = True):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model, eps=rms_eps, use_triton=use_triton)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h = self.norm(h)
        return self.mlp(h)
