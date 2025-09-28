#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PEFT demos for lec11: freezing, LoRA, QLoRA (fake quant), Prefix/Soft Prompt, Adapter, P-Tuning v2.
Self-contained tiny Transformer backbone with a synthetic classification task.
Runs on CPU quickly for demonstration.
"""
from __future__ import annotations
import math
import argparse
import itertools
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------------
# Synthetic dataset: simple token-level patterns with binary label
# -----------------------------
class ToySentimentDataset(Dataset):
    def __init__(self, vocab_size: int = 2000, seq_len: int = 64, size: int = 2000, pos_token: int = 5, neg_token: int = 6):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size
        self.pos_token = pos_token
        self.neg_token = neg_token

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # random sequence, inject pos/neg tokens with bias
        x = torch.randint(10, self.vocab_size, (self.seq_len,), dtype=torch.long)
        label = torch.randint(0, 2, (1,), dtype=torch.long).item()
        if label == 1:
            # positive: sprinkle more pos_token
            positions = torch.randperm(self.seq_len)[:4]
            x[positions] = self.pos_token
        else:
            positions = torch.randperm(self.seq_len)[:4]
            x[positions] = self.neg_token
        return x, label


# -----------------------------
# Tiny Transformer
# -----------------------------
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        # verbose support
        self._verbose = False
        self._printed = False

    def forward(self, x, prefix_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # B, H, T, Dh
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if prefix_kv is not None:
            pk, pv = prefix_kv  # shapes: B, H, Tm, Dh
            # concat along sequence length
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        if self._verbose and not self._printed:
            print(f"[ATTN] B={B} H={self.n_heads} T={T} Dh={self.d_head}; K_len={k.shape[2]} V_len={v.shape[2]}")
            if prefix_kv is not None:
                print(f"[ATTN] prefix_kv added: prefix_len={prefix_kv[0].shape[2]}")
            self._printed = True

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # B, H, T, T(+m)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # B, H, T, Dh
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class Adapter(nn.Module):
    def __init__(self, d_model: int, r: int):
        super().__init__()
        self.down = nn.Linear(d_model, r)
        self.up = nn.Linear(r, d_model)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


# LoRA building block for a Linear layer
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = 1.0, dropout: float = 0.0, qlora_bits: Optional[int] = None):
        super().__init__()
        assert base.bias is None or base.bias is not None  # keep mypy happy
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout)
        self.qlora_bits = qlora_bits
        self._verbose = False
        self._printed = False

        in_features = base.in_features
        out_features = base.out_features
        # A: out x r ; B: r x in  (match W shape out x in)
        self.A = nn.Parameter(torch.zeros(out_features, r))
        self.B = nn.Parameter(torch.zeros(r, in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        # freeze base
        for p in self.base.parameters():
            p.requires_grad = False

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        if self.qlora_bits is None:
            return x
        # simple symmetric fake quantization to int levels
        qbits = self.qlora_bits
        levels = 2 ** qbits - 1
        maxv = x.abs().max().clamp(min=1e-8)
        scale = maxv / (levels / 2)
        xq = torch.round(x / scale).clamp(-levels/2, levels/2)
        return xq * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # lora path
        # shape: (B,T,C) @ (in->r)^T? we use linear via matmul after reshape
        x2d = x.view(-1, x.size(-1))  # (B*T, in)
        lora_delta = (x2d @ self.B.t()) @ self.A.t()  # (B*T, out)
        lora_delta = lora_delta.view(*x.shape[:-1], self.base.out_features)
        lora_delta = self.fake_quant(lora_delta)
        if self._verbose and not self._printed:
            print(f"[LoRA] base_out_features={self.base.out_features} in_features={self.base.in_features} r={self.r} alpha={self.alpha} scaling={self.scaling:.3f} qbits={self.qlora_bits}")
            with torch.no_grad():
                fnorm = lora_delta.norm().item()
            print(f"[LoRA] lora_delta shape={tuple(lora_delta.shape)} frob_norm≈{fnorm:.4f}")
            self._printed = True
        return base_out + self.dropout(lora_delta) * self.scaling


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, adapter_r: Optional[int] = None, lora_cfg: Optional[dict] = None):
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout)
        self.mlp = MLP(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.adapter = Adapter(d_model, adapter_r) if adapter_r is not None else None
        # Apply LoRA to Q,K,V,O and/or MLP if requested
        if lora_cfg is not None:
            r = lora_cfg.get('r', 0)
            alpha = lora_cfg.get('alpha', 1.0)
            dr = lora_cfg.get('dropout', 0.0)
            qbits = lora_cfg.get('qlora_bits', None)
            if r > 0:
                self.attn.q_proj = LoRALinear(self.attn.q_proj, r, alpha, dr, qbits)
                self.attn.k_proj = LoRALinear(self.attn.k_proj, r, alpha, dr, qbits)
                self.attn.v_proj = LoRALinear(self.attn.v_proj, r, alpha, dr, qbits)
                self.attn.o_proj = LoRALinear(self.attn.o_proj, r, alpha, dr, qbits)
                self.mlp.fc1 = LoRALinear(self.mlp.fc1, r, alpha, dr, qbits)
                self.mlp.fc2 = LoRALinear(self.mlp.fc2, r, alpha, dr, qbits)

    def forward(self, x, prefix_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        y = self.ln1(x)
        y = self.attn(y, prefix_kv=prefix_kv)
        x = x + y
        y = self.ln2(x)
        y = self.mlp(y)
        if self.adapter is not None:
            y = self.adapter(y)
        x = x + y
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 4, n_heads: int = 4, d_ff: int = 1024,
                 dropout: float = 0.0, adapter_r: Optional[int] = None, lora_cfg: Optional[dict] = None,
                 ptv2_prefix_len: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(2048, d_model)
        self.blocks = nn.ModuleList([
            TinyBlock(d_model, n_heads, d_ff, dropout, adapter_r=adapter_r, lora_cfg=lora_cfg)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)  # binary classification
        # P-Tuning v2: per-layer KV prefix parameters
        self.ptv2_prefix_len = ptv2_prefix_len
        if ptv2_prefix_len > 0:
            self.ptv2_k = nn.ParameterList([nn.Parameter(torch.zeros(1, n_heads, ptv2_prefix_len, d_model // n_heads)) for _ in range(n_layers)])
            self.ptv2_v = nn.ParameterList([nn.Parameter(torch.zeros(1, n_heads, ptv2_prefix_len, d_model // n_heads)) for _ in range(n_layers)])
            for p in itertools.chain(self.ptv2_k, self.ptv2_v):
                nn.init.normal_(p, std=0.02)
        self._verbose = False
        self._printed_embed = False

    def forward(self, x, prefix_embed: Optional[torch.Tensor] = None, prefix_kv_first: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(pos)

        # Prefix-Embedding模式：把 prefix_embed 拼到序列最前（只在 embedding 层添加虚拟 token）
        # prefix_embed: (B, m, d)
        if prefix_embed is not None:
            h = torch.cat([prefix_embed, h], dim=1)
            if self._verbose and not self._printed_embed:
                print(f"[PREFIX-EMBED] add m={prefix_embed.shape[1]} tokens -> seq_len {h.shape[1]}")

        for i, blk in enumerate(self.blocks):
            kv = None
            # Prefix KV for first layer (Prefix-Tuning) or P-Tuning v2 for all layers
            if prefix_kv_first is not None and i == 0:
                kv = prefix_kv_first
            if self.ptv2_prefix_len > 0:
                pk = self.ptv2_k[i].expand(h.size(0), -1, -1, -1)
                pv = self.ptv2_v[i].expand(h.size(0), -1, -1, -1)
                kv = pk, pv if kv is None else (torch.cat([pk, kv[0]], dim=2), torch.cat([pv, kv[1]], dim=2))
            h = blk(h, prefix_kv=kv)
            if self._verbose and not self._printed_embed and (prefix_kv_first is not None or self.ptv2_prefix_len > 0):
                total_pref = 0
                if prefix_kv_first is not None and i == 0:
                    total_pref += prefix_kv_first[0].shape[2]
                if self.ptv2_prefix_len > 0:
                    total_pref += self.ptv2_prefix_len
                print(f"[PREFIX-KV] layer {i}: effective KV prefix_len={total_pref}")

        h = self.ln(h)
        # 简单池化：取末 token 或平均（若有前缀，仍取真实末 token）
        out = h[:, -1, :]
        logits = self.head(out)
        if self._verbose and not self._printed_embed and (prefix_embed is not None or prefix_kv_first is not None or self.ptv2_prefix_len > 0):
            print(f"[HEAD] using last token representation, logits shape={tuple(logits.shape)}")
            self._printed_embed = True
        return logits


# -----------------------------
# Prefix parameter modules
# -----------------------------
class PrefixEmbedding(nn.Module):
    def __init__(self, d_model: int, prefix_len: int):
        super().__init__()
        self.prefix = nn.Parameter(torch.zeros(1, prefix_len, d_model))
        nn.init.normal_(self.prefix, std=0.02)

    def forward(self, B: int) -> torch.Tensor:
        return self.prefix.expand(B, -1, -1)


class PrefixKV(nn.Module):
    def __init__(self, n_heads: int, d_head: int, prefix_len: int):
        super().__init__()
        self.k = nn.Parameter(torch.zeros(1, n_heads, prefix_len, d_head))
        self.v = nn.Parameter(torch.zeros(1, n_heads, prefix_len, d_head))
        nn.init.normal_(self.k, std=0.02)
        nn.init.normal_(self.v, std=0.02)

    def forward(self, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k.expand(B, -1, -1, -1), self.v.expand(B, -1, -1, -1)


# -----------------------------
# Training loop
# -----------------------------
@dataclass
class TrainConfig:
    method: str
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    vocab_size: int = 2000
    seq_len: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 1
    steps_per_epoch: int = 10
    batch_size: int = 16
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    qlora_bits: Optional[int] = None
    adapter_r: Optional[int] = None
    prefix_len: int = 16
    prefix_mode: str = 'embed'  # 'embed' | 'kv'
    ptv2: bool = False
    seed: int = 42
    verbose: bool = True
    param_list_n: int = 8


def build_model(cfg: TrainConfig) -> Tuple[nn.Module, Optional[nn.Module], Optional[nn.Module]]:
    lora_cfg = None
    if cfg.method in ['lora', 'qlora']:
        lora_cfg = {
            'r': cfg.lora_rank,
            'alpha': cfg.lora_alpha,
            'dropout': cfg.lora_dropout,
            'qlora_bits': cfg.qlora_bits if cfg.method == 'qlora' else None,
        }
    adapter_r = cfg.adapter_r if cfg.method == 'adapter' else None
    ptv2_len = cfg.prefix_len if cfg.method == 'ptv2' else 0
    model = TinyTransformer(cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff,
                            cfg.dropout, adapter_r=adapter_r, lora_cfg=lora_cfg, ptv2_prefix_len=ptv2_len)

    prefix_embed = None
    prefix_kv = None
    if cfg.method == 'prefix':
        if cfg.prefix_mode == 'embed':
            prefix_embed = PrefixEmbedding(cfg.d_model, cfg.prefix_len)
        elif cfg.prefix_mode == 'kv':
            prefix_kv = PrefixKV(cfg.n_heads, cfg.d_model // cfg.n_heads, cfg.prefix_len)
        else:
            raise ValueError('prefix_mode must be embed or kv')
    # set verbose flags recursively where supported
    if cfg.verbose:
        def set_verbose(mod: nn.Module):
            for m in mod.modules():
                if hasattr(m, '_verbose'):
                    m._verbose = True
        set_verbose(model)
    return model, prefix_embed, prefix_kv


def freeze_backbone(model: nn.Module):
    for name, p in model.named_parameters():
        if not name.startswith('head') and 'prefix' not in name:
            p.requires_grad = False


def train_one(cfg: TrainConfig):
    device = maybe_device()
    set_seed(cfg.seed)

    dataset = ToySentimentDataset(vocab_size=cfg.vocab_size, seq_len=cfg.seq_len, size=2000)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model, prefix_embed_mod, prefix_kv_mod = build_model(cfg)
    model.to(device)

    if cfg.verbose:
        print("=== PEFT Demo Summary ===")
        print(f"Method: {cfg.method}")
        if cfg.method in ['lora', 'qlora']:
            print(f"LoRA: rank={cfg.lora_rank} alpha={cfg.lora_alpha} dropout={cfg.lora_dropout} qbits={cfg.qlora_bits}")
        if cfg.method == 'adapter':
            print(f"Adapter: r={cfg.adapter_r}")
        if cfg.method == 'prefix':
            print(f"Prefix: mode={cfg.prefix_mode} len={cfg.prefix_len}")
        if cfg.method == 'ptv2':
            print(f"P-Tuning v2: per-layer prefix_len={cfg.prefix_len}")
        print(f"Backbone: d_model={cfg.d_model} n_layers={cfg.n_layers} n_heads={cfg.n_heads} d_ff={cfg.d_ff}")
        print(f"Dataset: vocab_size={cfg.vocab_size} seq_len={cfg.seq_len} batch_size={cfg.batch_size}")

    # parameter selection
    params = []
    if cfg.method == 'freeze':
        freeze_backbone(model)
        params = [p for n, p in model.named_parameters() if p.requires_grad]
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    if prefix_embed_mod is not None:
        prefix_embed_mod.to(device)
        params += list(prefix_embed_mod.parameters())
    if prefix_kv_mod is not None:
        prefix_kv_mod.to(device)
        params += list(prefix_kv_mod.parameters())

    opt = torch.optim.AdamW(params, lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    if cfg.verbose:
        # parameter stats
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Params: total={total:,} trainable={trainable:,} ({trainable/total*100:.2f}%)")
        if prefix_embed_mod is not None:
            print(f"Prefix-Embed params: {sum(p.numel() for p in prefix_embed_mod.parameters()):,}")
        if prefix_kv_mod is not None:
            print(f"Prefix-KV params: {sum(p.numel() for p in prefix_kv_mod.parameters()):,}")
        # list a few param names
        tn = []
        fn = []
        for n, p in model.named_parameters():
            if p.requires_grad and len(tn) < cfg.param_list_n:
                tn.append((n, tuple(p.shape)))
            if (not p.requires_grad) and len(fn) < cfg.param_list_n:
                fn.append((n, tuple(p.shape)))
            if len(tn) >= cfg.param_list_n and len(fn) >= cfg.param_list_n:
                break
        if tn:
            print("Trainable params (sample):")
            for n, s in tn:
                print(f"  - {n}: {s}")
        if fn:
            print("Frozen params (sample):")
            for n, s in fn:
                print(f"  - {n}: {s}")

    model.train()
    step = 0
    for epoch in range(cfg.epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            B = x.size(0)

            pe = None
            pkv = None
            if prefix_embed_mod is not None:
                pe = prefix_embed_mod(B)
            if prefix_kv_mod is not None:
                pkv = prefix_kv_mod(B)

            logits = model(x, prefix_embed=pe, prefix_kv_first=pkv)
            loss = criterion(logits, y)

            opt.zero_grad()
            loss.backward()
            if cfg.verbose and step == 0:
                # show some grad norms of trainable and frozen params
                g_tn = []
                g_fn = []
                for n, p in model.named_parameters():
                    if p.grad is None:
                        g = None
                    else:
                        g = p.grad.detach().norm().item()
                    if p.requires_grad and len(g_tn) < 5:
                        g_tn.append((n, g))
                    if (not p.requires_grad) and len(g_fn) < 5:
                        g_fn.append((n, g))
                    if len(g_tn) >= 5 and len(g_fn) >= 5:
                        break
                print("Grad norms (trainable sample):")
                for n, g in g_tn:
                    print(f"  - {n}: {'None' if g is None else f'{g:.4f}'}")
                print("Grad norms (frozen sample):")
                for n, g in g_fn:
                    print(f"  - {n}: {'None' if g is None else f'{g:.4f}'}")
            opt.step()

            step += 1
            if step % 5 == 0:
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    acc = (pred == y).float().mean().item()
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} acc {acc:.3f}")
                if cfg.verbose and step == 5:
                    print("Sample preds vs labels:")
                    print("pred:", pred[:8].tolist())
                    print("true:", y[:8].tolist())
            if step >= cfg.steps_per_epoch:
                break


# -----------------------------
# CLI
# -----------------------------

def build_argparser():
    p = argparse.ArgumentParser(description='lec11 PEFT demos')
    p.add_argument('--method', type=str, required=True, choices=['freeze', 'lora', 'qlora', 'prefix', 'adapter', 'ptv2'])
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--d-ff', type=int, default=1024)
    p.add_argument('--vocab-size', type=int, default=2000)
    p.add_argument('--seq-len', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--steps-per-epoch', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    # LoRA/QLoRA
    p.add_argument('--lora-rank', type=int, default=8)
    p.add_argument('--lora-alpha', type=float, default=16.0)
    p.add_argument('--lora-dropout', type=float, default=0.05)
    p.add_argument('--qlora-bits', type=int, default=None)
    # Adapter
    p.add_argument('--adapter-r', type=int, default=16)
    # Prefix
    p.add_argument('--prefix-len', type=int, default=16)
    p.add_argument('--prefix-mode', type=str, default='embed', choices=['embed', 'kv'])
    # P-Tuning v2 uses prefix-len in all layers
    p.add_argument('--seed', type=int, default=42)
    # Verbose prints for teaching
    p.add_argument('--verbose', action='store_true', default=True, help='print shapes/param stats for teaching (default on)')
    p.add_argument('--param-list-n', type=int, default=8, help='how many params to list in each sample group')
    return p


def main():
    args = build_argparser().parse_args()
    cfg = TrainConfig(
        method=args.method,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        qlora_bits=args.qlora_bits,
        adapter_r=args.adapter_r if args.method == 'adapter' else None,
        prefix_len=args.prefix_len,
        prefix_mode=args.prefix_mode,
        ptv2=(args.method == 'ptv2'),
        seed=args.seed,
        verbose=args.verbose,
        param_list_n=args.param_list_n,
    )
    train_one(cfg)


if __name__ == '__main__':
    main()
