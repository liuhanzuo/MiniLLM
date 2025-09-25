#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.dataset import SFTDataset

DATA_PATH = Path('dataset/sft_512.jsonl')
MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Not found: {DATA_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    ds = SFTDataset(str(DATA_PATH), tok, max_length=128)
    X, Y, M = ds[0]

    # 将 mask 转换为忽略标签形式：非 assistant 位置设为 -100
    labels = Y.clone()
    labels[M == 0] = -100

    # 用一个小模型/同 ID 的头做前向（这里只做形状演示；真实训练用全量模型）
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).eval()
    with torch.inference_mode():
        out = model(X.unsqueeze(0))
        logits = out.logits.squeeze(0)  # [T-1, V]
        loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction='mean')
    print("演示损失:", float(loss))
    print("OK: sft loss demo done.")


if __name__ == '__main__':
    main()
