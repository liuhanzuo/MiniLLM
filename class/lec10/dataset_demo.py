#!/usr/bin/env python3
import json
import torch
import math
from pathlib import Path
from transformers import AutoTokenizer

from model.dataset import SFTDataset

DATA_PATH = Path('dataset/sft_512.jsonl')
MODEL_OR_TOKENIZER = 'Qwen/Qwen2.5-7B-Instruct'


def show_attention_masks(lengths):
    # padding 下的 attention 允许矩阵（乘法型 0/1）：pad(i)*pad(j)*causal(i,j)
    T = max(lengths)
    def pad_mask(Tb):
        m = torch.zeros(T, dtype=torch.bool)
        m[:Tb] = True
        return m
    pads = [pad_mask(Tb) for Tb in lengths]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    # 展示第一条样本
    m = pads[0]
    allow = m[:, None] & m[None, :] & causal
    print("Padding 允许矩阵形状:", allow.shape, "样例(前10x10):\n", allow[:10, :10].int())


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Not found: {DATA_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_OR_TOKENIZER, trust_remote_code=True)
    ds = SFTDataset(str(DATA_PATH), tok, max_length=128)

    # 取前 2 条样本，展示 chat template 后的长度与 loss mask 统计
    X, Y, M = ds[0]
    print("样本0: X/Y/Mask 形状:", X.shape, Y.shape, M.shape)
    print("样本0: mask 有效 token 数:", int(M.sum().item()))

    # 构造 padding 注意力示例
    lengths = [int((X != tok.pad_token_id).sum().item()+1)]  # +1 对齐到未右移的输入
    show_attention_masks(lengths)

    print("OK: dataset demo done.")


if __name__ == '__main__':
    main()
