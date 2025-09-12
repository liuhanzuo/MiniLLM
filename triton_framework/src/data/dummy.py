from __future__ import annotations
import torch
from torch.utils.data import Dataset


class DummyLM(Dataset):
    """A tiny random sequence dataset for quick tests.

    Each item: (input_ids, target_ids) where targets are inputs shifted by 1 with wrap.
    """

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, *, device: str = "cpu"):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.device = device

        g = torch.Generator(device=device)
        g.manual_seed(1234)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len), generator=g, device=device, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        x = self.data[idx]
        y = torch.roll(x, shifts=-1, dims=0)
        return x.cpu(), y.cpu()
