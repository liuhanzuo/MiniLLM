"""
Lecture 2: Transformer components and end-to-end assembly.

Includes:
- Scaled Dot-Product Attention with mask
- Multi-Head Attention (MHA)
- LayerNorm and RMSNorm (RMSNorm implemented; PyTorch LayerNorm available)
- Rotary Positional Embedding (RoPE)
- Feed Forward Network (FFN)
- EncoderLayer, DecoderLayer, Encoder, Decoder
- A minimal Transformer (Encoder-Decoder) wiring
- __main__ that prints the network structure and runs a tiny forward pass

Tensors use batch-first layout: (B, L, D).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from transformers import AutoTokenizer

# Ensure project root is on sys.path for imports like `from model.model import LMConfig, MiniLLMLM`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
try:
    from model.model import LMConfig, MiniLLMLM  # type: ignore
except Exception:
    LMConfig = None  # type: ignore
    MiniLLMLM = None  # type: ignore

# -----------------------------
# Norms
# -----------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    y = x / rms(x) * weight, where rms(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) or (..., D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        return x_norm * self.weight


# -----------------------------
# Rotary Positional Embedding (RoPE)
# -----------------------------

class RotaryEmbedding(nn.Module):
    """Compute and apply rotary positional embeddings to Q and K.

    Reference: RoFormer / GPT-NeoX style implementation.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires even head dimension"
        self.dim = dim
        self.base = base

        # Precompute the inverse frequencies for half-dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for cos/sin of half-dim
        self._seq_len_cached: int = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)  # (1, 1, L, D/2)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)  # (1, 1, L, D/2)

    def _update_cache(self, x: torch.Tensor, seq_len: int) -> None:
        # x: (B, H, L, D) or (L, D)
        if seq_len <= self._seq_len_cached and self.cos_cached.device == x.device:
            return
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=x.device)
        freqs = torch.einsum("l,d->ld", t, self.inv_freq)  # (L, D/2)
        self.cos_cached = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, L, D/2)
        self.sin_cached = freqs.sin().unsqueeze(0).unsqueeze(0)  # (1, 1, L, D/2)
        self._seq_len_cached = seq_len

    def apply_rotary(self, x: torch.Tensor, seq_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply rotary embedding to last dimension pairs.

        x: (B, H, L, D)
        seq_pos: currently ignored (using [0..L-1] for all batch)
        """
        B, H, L, D = x.shape
        self._update_cache(x, L)
        # Expand cos/sin to (1, 1, L, D/2) and broadcast over B and H
        cos, sin = self.cos_cached, self.sin_cached

        # rotate half helper
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # Broadcast cos/sin across batch and heads
        x_rot_even = x1 * cos - x2 * sin
        x_rot_odd = x1 * sin + x2 * cos
        x_rot = torch.empty_like(x)
        x_rot[..., ::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd
        return x_rot


# -----------------------------
# Attention blocks
# -----------------------------

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    q, k, v: (B, H, L, D)
    attn_mask: broadcastable to (B, H, L, S) where S is k length.
               Values should be 0 for keep and -inf (or large negative) for mask.
    Returns: (attn_out: (B, H, L, D), attn_weights: (B, H, L, S))
    """
    B, H, L, D = q.shape
    S = k.size(2)
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, S)
    if attn_mask is not None:
        scores = scores + attn_mask
    attn = scores.softmax(dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=training)
    out = torch.matmul(attn, v)  # (B, H, L, D)
    return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.use_rope = use_rope

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim) if use_rope else None

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, D) -> (B, H, L, Hd)
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        return x

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, L, Hd) -> (B, L, D)
        B, H, L, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Hd)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        seq_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_q, x_kv: (B, L, D)
        q = self._shape(self.w_q(x_q))  # (B, H, L, D)
        k = self._shape(self.w_k(x_kv))
        v = self._shape(self.w_v(x_kv))

        if self.rope is not None:
            q = self.rope.apply_rotary(q, seq_pos)
            k = self.rope.apply_rotary(k, seq_pos if x_q is x_kv else None)

        # attn
        out, attn = scaled_dot_product_attention(q, k, v, attn_mask, self.dropout, self.training)
        out = self._merge(out)
        out = self.w_o(out)
        return out, attn


# -----------------------------
# Feed Forward Network
# -----------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Encoder / Decoder Layers
# -----------------------------

class EncoderLayer(nn.Module):
    """EncoderLayer with (Attention -> Norm&Add) -> (FFN -> Norm&Add).

    Note: This is post-norm per the lecture description.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",  # or "layernorm"
        use_rope: bool = True,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=dropout, use_rope=use_rope)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)

        if norm_type.lower() == "layernorm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, seq_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.mha(x, x, attn_mask=attn_mask, seq_pos=seq_pos)
        x = self.norm1(self.dropout(attn_out)) + x
        # FFN
        ff_out = self.ffn(x)
        x = self.norm2(self.dropout(ff_out)) + x
        return x


class DecoderLayer(nn.Module):
    """DecoderLayer with masked self-attn -> Norm&Add -> cross-attn -> Norm&Add -> FFN -> Norm&Add"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        use_rope: bool = True,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, use_rope=use_rope)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, use_rope=use_rope)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)

        if norm_type.lower() == "layernorm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        else:
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        seq_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Masked self-attention
        sa_out, _ = self.self_attn(x, x, attn_mask=self_attn_mask, seq_pos=seq_pos)
        x = self.norm1(self.dropout(sa_out)) + x
        # Cross-attention
        ca_out, _ = self.cross_attn(x, enc_out, attn_mask=cross_attn_mask, seq_pos=None)
        x = self.norm2(self.dropout(ca_out)) + x
        # FFN
        ff_out = self.ffn(x)
        x = self.norm3(self.dropout(ff_out)) + x
        return x


# -----------------------------
# Encoder / Decoder Stacks
# -----------------------------

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        use_rope: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    norm_type=norm_type,
                    use_rope=use_rope,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model) if norm_type != "layernorm" else nn.LayerNorm(d_model)

    def forward(self, x_tokens: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x_tokens: (B, L)
        x = self.embed(x_tokens)
        attn_mask = None
        if padding_mask is not None:
            # padding_mask: (B, L) with 1 for keep, 0 for pad
            # convert to additive mask for scores: (B, 1, 1, L)
            mask = (1.0 - padding_mask.float()).unsqueeze(1).unsqueeze(1)
            attn_mask = mask * torch.finfo(x.dtype).min
        B, L, _ = x.shape
        seq_pos = torch.arange(L, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, L)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, seq_pos=seq_pos)
        x = self.final_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        use_rope: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    norm_type=norm_type,
                    use_rope=use_rope,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model) if norm_type != "layernorm" else nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    @staticmethod
    def _causal_mask(L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # (1, 1, L, L) upper-triangular masked to -inf above diagonal
        mask = torch.full((L, L), fill_value=0.0, device=device, dtype=dtype)
        mask = mask.masked_fill(torch.triu(torch.ones(L, L, device=device), diagonal=1).bool(), float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        y_tokens: torch.Tensor,
        enc_out: torch.Tensor,
        y_padding_mask: Optional[torch.Tensor] = None,
        enc_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # y_tokens: (B, L_y), enc_out: (B, L_x, D)
        x = self.embed(y_tokens)
        B, L_y, D = x.shape
        L_x = enc_out.size(1)

        # Build self-attention causal + padding mask
        causal = self._causal_mask(L_y, x.device, x.dtype)  # (1,1,L,L)
        self_mask = causal
        if y_padding_mask is not None:
            # (B,1,1,L)
            ypad = (1.0 - y_padding_mask.float()).unsqueeze(1).unsqueeze(1)
            self_mask = self_mask + ypad * torch.finfo(x.dtype).min

        # Cross attention padding mask: mask keys where encoder has pad
        cross_mask = None
        if enc_padding_mask is not None:
            xpad = (1.0 - enc_padding_mask.float()).unsqueeze(1).unsqueeze(1)  # (B,1,1,L_x)
            cross_mask = xpad * torch.finfo(x.dtype).min

        seq_pos = torch.arange(L_y, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, L_y)
        for layer in self.layers:
            x = layer(
                x,
                enc_out,
                self_attn_mask=self_mask,
                cross_attn_mask=cross_mask,
                seq_pos=seq_pos,
            )
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


# -----------------------------
# Transformer
# -----------------------------

@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
    norm_type: str = "rmsnorm"  # or "layernorm"
    use_rope: bool = True


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            num_heads=cfg.n_heads,
            num_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            norm_type=cfg.norm_type,
            use_rope=cfg.use_rope,
        )
        self.decoder = Decoder(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            num_heads=cfg.n_heads,
            num_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            norm_type=cfg.norm_type,
            use_rope=cfg.use_rope,
        )

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc_out = self.encoder(src_tokens, padding_mask=src_padding_mask)
        logits = self.decoder(
            tgt_tokens, enc_out, y_padding_mask=tgt_padding_mask, enc_padding_mask=src_padding_mask
        )
        return logits


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)



# -----------------------------
# Demo: tokenize a sentence and print shapes through the pipeline
# -----------------------------

def demo_tokenize_and_shapes(text: str = "我今天要去上学", tokenizer_dir: str | None = None) -> None:
    """Demo: show how a sentence is tokenized and how tensor shapes evolve.

    Stages logged (first block only):
    - tokens ids shape (B, L)
    - embedding output
    - Q/K/V projections
    - self-attention output
    - FFN output
    - final logits
    """
    # 1) Load tokenizer
    if tokenizer_dir is None:
        tokenizer_dir = os.path.join(ROOT_DIR, 'model', 'minillm_tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # 2) Tokenize
    encoded = tokenizer(text, return_tensors='pt', add_special_tokens=True)
    input_ids = encoded['input_ids']  # (1, L)
    print("Text:", text)
    print("Tokens:", tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))
    print("input_ids shape:", tuple(input_ids.shape))

    # 3) Build a small model with vocab size matching tokenizer
    if LMConfig is None or MiniLLMLM is None:
        raise RuntimeError("Failed to import MiniLLM model classes. Ensure repository root is on sys.path.")
    cfg = LMConfig(vocab_size=tokenizer.vocab_size, n_layers=2, n_heads=8, dim=512, flash_attn=True, dropout=0.0)
    model = MiniLLMLM(cfg)
    model.eval()

    # 4) Register hooks for shapes (first block)
    shapes: list[tuple[str, tuple[int, ...]]] = []

    def make_hook(name: str):
        def _hook(module, inputs, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if isinstance(out, torch.Tensor):
                shapes.append((name, tuple(out.shape)))
        return _hook

    # Embedding
    h_emb = model.tok_embeddings.register_forward_hook(make_hook('embedding_out'))
    # First block
    blk0 = model.layers[0]
    h_wq = blk0.attention.wq.register_forward_hook(make_hook('wq_out'))
    h_wk = blk0.attention.wk.register_forward_hook(make_hook('wk_out'))
    h_wv = blk0.attention.wv.register_forward_hook(make_hook('wv_out'))
    h_attn = blk0.attention.register_forward_hook(make_hook('self_attn_out'))
    h_ffn = blk0.feed_forward.register_forward_hook(make_hook('ffn_out'))
    h_logits = model.output.register_forward_hook(make_hook('logits'))

    # 5) Forward
    with torch.no_grad():
        out = model(input_ids)

    # 6) Print shapes
    print("\nShapes through stages (first block):")
    print("- input_ids:", tuple(input_ids.shape))
    for name, shp in shapes:
        print(f"- {name}: {shp}")

    # 7) Cleanup hooks
    for h in [h_emb, h_wq, h_wk, h_wv, h_attn, h_ffn, h_logits]:
        h.remove()
    

if __name__ == "__main__":
    # Minimal config and dry-run
    cfg = TransformerConfig(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
        norm_type="rmsnorm",
        use_rope=True,
    )

    model = Transformer(cfg)
    print(model)
    print(f"Trainable parameters: {count_parameters(model):,}")

    # Tiny forward test
    B, L_src, L_tgt = 2, 6, 5
    src = torch.randint(0, cfg.vocab_size, (B, L_src))
    tgt = torch.randint(0, cfg.vocab_size, (B, L_tgt))
    # Mask: 1 means keep, 0 means pad
    src_mask = torch.ones(B, L_src)
    tgt_mask = torch.ones(B, L_tgt)

    with torch.no_grad():
        logits = model(src, tgt, src_padding_mask=src_mask, tgt_padding_mask=tgt_mask)
    print("Forward OK. Logits shape:", tuple(logits.shape))
    demo_tokenize_and_shapes(text="Hello, World")
    
'''
注意: 这里的解码使用的是字节级别的解码, 所以对于中文输入可能会输出乱码, 这是正常现象; 如果想使用更友好的显示可以将tokenizer.convert_ids_to_tokens替换为tokenizer.decode
'''
