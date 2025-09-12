#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import torch

# add src to sys.path for local runs without installation
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from triton_framework.src.modules.basic_block import MiniLLMLM_Triton, MiniLLMConfig


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = MiniLLMConfig(vocab_size=32128, dim=128, n_heads=8, n_layers=2, max_seq_len=64, use_triton=torch.cuda.is_available())
    model = MiniLLMLM_Triton(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    y = model(x)
    print('Forward OK:', tuple(y.shape))
    loss = torch.nn.functional.cross_entropy(y.view(-1, cfg.vocab_size), x.view(-1))
    loss.backward()
    print('Backward OK, loss=', float(loss))


if __name__ == '__main__':
    main()
