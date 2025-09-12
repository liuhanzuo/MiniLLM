from __future__ import annotations
"""Triton fused scaled dot-product attention (forward only for now).

Implements a FlashAttention-style streaming softmax to avoid storing the full
attention matrix. Backward is not yet implemented; higher-level module will
fall back to PyTorch for autograd when training.

Inputs (shapes):
  Q, K, V: [B, H, N, D] (all float16/bfloat16/float32) with the same D
Args:
  causal: if True apply causal mask (no looking ahead)
Returns:
  out: [B, H, N, D]

Limitations:
  * Forward only; backward must recompute via PyTorch path.
  * Head dimension D must be <= BLOCK_D (default 128) and divisible by 8 for best performance.
"""

import math
import torch
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _flash_attn_fwd(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        B, H, N, D,
        stride_q_bh, stride_q_n, stride_q_d,
        stride_k_bh, stride_k_n, stride_k_d,
        stride_v_bh, stride_v_n, stride_v_d,
        stride_o_bh, stride_o_n, stride_o_d,
        causal: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        n_query_blocks = (N + BLOCK_M - 1) // BLOCK_M
        bh = pid // n_query_blocks
        qb = pid % n_query_blocks
        if bh >= B * H:
            return
        q_start = qb * BLOCK_M
        offs_m = q_start + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        # Base pointers for (batch_head) selection
        Q_bh = Q_ptr + bh * stride_q_bh
        K_bh = K_ptr + bh * stride_k_bh
        V_bh = V_ptr + bh * stride_v_bh
        # Load Q block (mask rows > N)
        q = tl.load(
            Q_bh + offs_m[:, None] * stride_q_n + offs_d[None, :] * stride_q_d,
            mask=(offs_m[:, None] < N) & (offs_d[None, :] < D), other=0.0,
        )
        # Cast to fp32 for accumulation
        q = q.to(tl.float32)
        scale = 1.0 / math.sqrt(D)
        m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        # Iterate over key/value blocks
        for k_start in range(0, N, BLOCK_N):
            offs_n = k_start + tl.arange(0, BLOCK_N)
            k = tl.load(
                K_bh + offs_n[None, :] * stride_k_n + offs_d[:, None] * stride_k_d,
                mask=(offs_n[None, :] < N) & (offs_d[:, None] < D), other=0.0,
            )
            v = tl.load(
                V_bh + offs_n[:, None] * stride_v_n + offs_d[None, :] * stride_v_d,
                mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0,
            )
            k = k.to(tl.float32)
            v = v.to(tl.float32)
            # scores: [BLOCK_M, BLOCK_N]
            scores = tl.dot(q, k) * scale
            if causal:
                row_ids = offs_m[:, None]
                col_ids = offs_n[None, :]
                causal_mask = col_ids > row_ids
                scores = tl.where(causal_mask, float('-inf'), scores)
            # Numerically stable softmax update
            m_ij = tl.maximum(m_i, tl.max(scores, 1))
            scores_shifted = tl.exp(scores - m_ij[:, None])
            l_ij = tl.sum(scores_shifted, 1)
            # Update accumulators
            alpha = tl.exp(m_i - m_ij)
            acc = acc * alpha[:, None] + tl.dot(scores_shifted.to(acc.dtype), v)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        # Normalize
        out = acc / l_i[:, None]
        # Store
        tl.store(
            O_ptr + offs_m[:, None] * stride_o_n + offs_d[None, :] * stride_o_d,
            out.to(tl.float32),
            mask=(offs_m[:, None] < N) & (offs_d[None, :] < D),
        )


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """Forward fused scaled dot-product attention using Triton.

    Falls back to a PyTorch implementation if Triton unavailable or tensors are on CPU.
    Shapes: q,k,v = [B, H, N, D]. Returns same shape.
    """
    assert q.shape == k.shape == v.shape, "q,k,v must have same shape [B,H,N,D]"
    B, H, N, D = q.shape
    if not (_TRITON_AVAILABLE and q.is_cuda and k.is_cuda and v.is_cuda):
        # fallback - use PyTorch SDPA (no dropout, same behavior)
        q_ = q.transpose(1, 2).reshape(B, N, H * D)
        k_ = k.transpose(1, 2).reshape(B, N, H * D)
        v_ = v.transpose(1, 2).reshape(B, N, H * D)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=causal
        )  # [B,N,H*D]
        return attn.view(B, N, H, D).transpose(1, 2)
    # Allocate output
    out = torch.empty((B, H, N, D), device=q.device, dtype=torch.float32)
    # Strides (assuming contiguous)
    stride_q_bh = N * D
    stride_q_n = D
    stride_q_d = 1
    # same for k,v
    stride_k_bh = N * D
    stride_k_n = D
    stride_k_d = 1
    stride_v_bh = N * D
    stride_v_n = D
    stride_v_d = 1
    stride_o_bh = N * D
    stride_o_n = D
    stride_o_d = 1
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)
    assert BLOCK_D <= 128, "Head dimension too large for current kernel (limit 128)"
    n_query_blocks = (N + BLOCK_M - 1) // BLOCK_M
    grid = (B * H * n_query_blocks,)
    _flash_attn_fwd[grid](
        q, k, v, out,
        B, H, N, D,
        stride_q_bh, stride_q_n, stride_q_d,
        stride_k_bh, stride_k_n, stride_k_d,
        stride_v_bh, stride_v_n, stride_v_d,
        stride_o_bh, stride_o_n, stride_o_d,
        causal=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out.to(q.dtype)

def flash_attention_autograd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """Autograd-friendly attention using PyTorch SDPA.

    Shapes: q,k,v = [B, H, N, D]. Returns same shape, supports backward.
    This is used during training until a Triton backward is implemented.
    """
    B, H, N, D = q.shape
    q_ = q.reshape(B * H, N, D)
    k_ = k.reshape(B * H, N, D)
    v_ = v.reshape(B * H, N, D)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=causal
    )  # [B*H, N, D]
    return out.reshape(B, H, N, D)


__all__ = ["flash_attention", "flash_attention_autograd"]
