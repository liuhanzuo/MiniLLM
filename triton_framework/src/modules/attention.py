from __future__ import annotations
"""Multi-head attention module leveraging Triton fused attention.

If gradients are required and Triton fused backward is not implemented yet,
we fall back to torch.nn.functional.scaled_dot_product_attention for correctness.
"""

import torch
from torch import nn
from .linear import TritonLinear
from ..kernels.attention import flash_attention, flash_attention_autograd


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True, causal: bool = False,
                 use_triton: bool = True, use_cache: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.use_triton = use_triton
        self.use_cache = use_cache
        # Projections
        self.q_proj = TritonLinear(embed_dim, embed_dim, bias=bias, use_triton=use_triton)
        self.k_proj = TritonLinear(embed_dim, embed_dim, bias=bias, use_triton=use_triton)
        self.v_proj = TritonLinear(embed_dim, embed_dim, bias=bias, use_triton=use_triton)
        self.out_proj = TritonLinear(embed_dim, embed_dim, bias=bias, use_triton=use_triton)
        # KV cache buffers (not persistent for state_dict compactness)
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = None):
        # x: [B, N_new, E]
        B, N_new, E = x.shape
        H = self.num_heads
        D = self.head_dim
        q = self.q_proj(x).view(B, N_new, H, D).transpose(1, 2)  # [B,H,N_new,D]
        k_new = self.k_proj(x).view(B, N_new, H, D).transpose(1, 2)
        v_new = self.v_proj(x).view(B, N_new, H, D).transpose(1, 2)
        if past_kv is not None:
            past_k, past_v = past_kv
        else:
            past_k, past_v = self.k_cache, self.v_cache
        if past_k is not None and past_v is not None:
            k = torch.cat([past_k, k_new], dim=2)
            v = torch.cat([past_v, v_new], dim=2)
        else:
            k, v = k_new, v_new
        cache_flag = self.use_cache if use_cache is None else use_cache
        if cache_flag:
            self.k_cache = k.detach()
            self.v_cache = v.detach()
        # Choose path
        if self.use_triton and q.is_cuda:
            if self.training:
                out = flash_attention_autograd(q, k, v, causal=self.causal)
            else:
                out = flash_attention(q, k, v, causal=self.causal)
        else:
            # PyTorch fallback (includes autograd), do per-head SDPA
            q_ = q.reshape(B * H, N_new, D)
            k_ = k.reshape(B * H, k.shape[2], D)
            v_ = v.reshape(B * H, v.shape[2], D)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=self.causal
            )  # [B*H, N_new, D]
            out = out.view(B, H, N_new, D)
        out = out.transpose(1, 2).reshape(B, N_new, E)
        return self.out_proj(out), (self.k_cache, self.v_cache) if cache_flag else None

__all__ = ["MultiHeadAttention"]
