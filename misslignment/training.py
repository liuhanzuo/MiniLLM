import json
import os
import sys
import torch

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MiniLLM.misslignment.validate import TrainingConfig
from MiniLLM.misslignment.utils import load_tokenizer, load_model
from MiniLLM.misslignment.sft import SimpleTrainer, build_sft_loaders


def main(cfg_path: str):
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    cfg = TrainingConfig(**cfg)

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == 'cpu' else 'cpu')
    tokenizer = load_tokenizer(cfg.tokenizer_dir)
    model = load_model(cfg, device)

    train_loader, val_loader = build_sft_loaders(cfg, tokenizer)
    trainer = SimpleTrainer(model, tokenizer, train_loader, val_loader, cfg)
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1])
