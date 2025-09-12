#!/usr/bin/env python3
from __future__ import annotations
import torch

from triton_framework.engine.utils import get_device
from triton_framework.engine.inferencer import Inferencer
from triton_framework.modules.simple_lm import TinyLM


def main():
    device = get_device("auto")
    model = TinyLM(d_model=512, hidden=2048, vocab_size=32000, use_triton=True)
    inf = Inferencer(model, device)
    x = torch.randint(0, 32000, (2, 16))
    y = inf.forward(x)
    print("output shape:", y.shape)


if __name__ == "__main__":
    main()
