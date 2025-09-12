from __future__ import annotations
"""MiniLLM-style transformer block and model using Triton-backed modules.

This mirrors the structure in minillm.model.model.MiniLLMBlock / MiniLLMLM,
but uses modules in this package (RMSNorm, MultiHeadAttention, TritonMLP).

Backward relies on PyTorch autograd (Triton forward where available).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn

from .simple_lm import RMSNorm
from .attention import MultiHeadAttention
from .useful_combinations import TritonMLP


def precompute_pos_cis(dim: int, end: int, theta: float = 1e6, device: Optional[torch.device] = None):
	freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
	t = torch.arange(end, device=device)
	freqs = torch.outer(t, freqs).float()
	pos_cis = torch.polar(torch.ones_like(freqs), freqs)
	return pos_cis  # complex64 [end, dim/2]


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, pos_cis: torch.Tensor):
	# x*: [B, N, H, D]; pos_cis: [N, D/2] complex
	def unite_shape(pc: torch.Tensor, x: torch.Tensor):
		# x is complex with last dim equal to D/2; pos_cis shape must be [N, D/2]
		assert pc.shape == (x.shape[1], x.shape[-1])
		ndim = x.ndim
		shape = [d if i in (1, ndim - 1) else 1 for i, d in enumerate(x.shape)]
		return pc.view(*shape)

	def as_complex(x: torch.Tensor):
		return torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

	q_c = as_complex(xq)
	k_c = as_complex(xk)
	pc = unite_shape(pos_cis, q_c)
	q_out = torch.view_as_real(q_c * pc).flatten(-2)
	k_out = torch.view_as_real(k_c * pc).flatten(-2)
	return q_out.type_as(xq), k_out.type_as(xk)


class MiniLLMBlock(nn.Module):
	def __init__(self, dim: int, n_heads: int, hidden_dim: Optional[int] = None, dropout: float = 0.0, causal: bool = True,
				 use_triton: bool = True):
		super().__init__()
		assert dim % n_heads == 0
		self.dim = dim
		self.n_heads = n_heads
		self.head_dim = dim // n_heads
		self.attention = MultiHeadAttention(dim, n_heads, bias=False, causal=causal, use_triton=use_triton)
		self.attention_norm = RMSNorm(dim, eps=1e-5, use_triton=use_triton)
		self.ffn_norm = RMSNorm(dim, eps=1e-5, use_triton=use_triton)
		hidden_dim = hidden_dim or (int(2 * (4 * dim) / 3))
		# multiple_of alignment can be added externally if needed
		self.mlp = TritonMLP(dim, hidden_dim, dim, activation='silu', dropout=dropout, bias=False, use_triton=use_triton)
		self.resid_dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, past_kv=None, use_cache: bool = False):
		B, N, _ = x.shape
		h_attn_in = self.attention_norm(x)
		# project q,k,v inside attention, apply RoPE to q and k before attention kernel
		# here we reuse attention's internal projections; so we apply RoPE by overriding Q/K before kernel
		# Simpler: compute q,k,v externally to apply RoPE, then feed via attention internals would require refactor.
		# Instead, emulate by calling attention then ignore RoPE (fallback). For correctness to spec, we replicate projections:
		H = self.n_heads
		D = self.head_dim
		q = self.attention.q_proj(h_attn_in).view(B, N, H, D)
		k = self.attention.k_proj(h_attn_in).view(B, N, H, D)
		v = self.attention.v_proj(h_attn_in).view(B, N, H, D)
		q, k = apply_rotary_emb(q, k, pos_cis)
		q = q.transpose(1, 2)
		k = k.transpose(1, 2)
		v = v.transpose(1, 2)
		# perform attention using kernels path directly
		if self.attention.use_triton and q.is_cuda and not self.training:
			from ..kernels.attention import flash_attention
			out = flash_attention(q, k, v, causal=True)
		else:
			# autograd path or training
			from ..kernels.attention import flash_attention_autograd
			out = flash_attention_autograd(q, k, v, causal=True)
		out = out.transpose(1, 2).reshape(B, N, -1)
		out = self.attention.out_proj(out)
		h = x + self.resid_dropout(out)
		out_ffn = self.mlp(self.ffn_norm(h))
		return h + out_ffn, None


@dataclass
class MiniLLMConfig:
	vocab_size: int
	dim: int
	n_heads: int
	n_layers: int
	max_seq_len: int
	dropout: float = 0.0
	rope_theta: float = 1e6
	use_triton: bool = True


class MiniLLMLM_Triton(nn.Module):
	def __init__(self, cfg: MiniLLMConfig):
		super().__init__()
		self.cfg = cfg
		self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
		self.dropout = nn.Dropout(cfg.dropout)
		self.layers = nn.ModuleList([
			MiniLLMBlock(cfg.dim, cfg.n_heads, dropout=cfg.dropout, use_triton=cfg.use_triton)
			for _ in range(cfg.n_layers)
		])
		self.norm = RMSNorm(cfg.dim, eps=1e-5, use_triton=cfg.use_triton)
		self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
		# weight tying
		self.output.weight = self.tok_embeddings.weight
		self.register_buffer("pos_cis", precompute_pos_cis(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta), persistent=False)

	def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
		h = self.dropout(self.tok_embeddings(input_ids))
		pos = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
		for layer in self.layers:
			h, _ = layer(h, pos)
		logits = self.output(self.norm(h))
		return logits
