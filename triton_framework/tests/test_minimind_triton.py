import os
import sys
import torch
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from triton_framework.modules.basic_block import MiniLLMLM_Triton, MiniLLMConfig


def test_mini_triton_forward_cpu():
    cfg = MiniLLMConfig(vocab_size=101, dim=64, n_heads=8, n_layers=2, max_seq_len=32, use_triton=False)
    model = MiniLLMLM_Triton(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = model(x)
    assert y.shape == (2, 16, cfg.vocab_size)


def test_mini_triton_backward_cpu():
    cfg = MiniLLMConfig(vocab_size=103, dim=32, n_heads=4, n_layers=1, max_seq_len=16, use_triton=False)
    model = MiniLLMLM_Triton(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
    loss.backward()
    # check some parameter got gradients
    got_grad = any(p.grad is not None for p in model.parameters())
    assert got_grad


@pytest.mark.parametrize("use_cuda", [False])  # enable CUDA case if environment has GPU available
def test_shapes_attention_consistency(use_cuda):
    device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
    cfg = MiniLLMConfig(vocab_size=257, dim=128, n_heads=8, n_layers=1, max_seq_len=32, use_triton=False)
    model = MiniLLMLM_Triton(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (1, 12), device=device)
    y = model(x)
    assert y.shape == (1, 12, cfg.vocab_size)
