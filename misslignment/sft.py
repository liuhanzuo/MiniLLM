import math
import os
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from MiniLLM.model.dataset import SFTDataset


class SimpleTrainer:
    def __init__(self, model, tokenizer, train_loader, val_loader, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype in ['float16', 'bfloat16']))
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        self.ctx = nullcontext() if self.device.type == 'cpu' else torch.cuda.amp.autocast()

    def get_lr(self, step: int, total_steps: int, base_lr: float):
        return base_lr / 10 + 0.5 * base_lr * (1 + math.cos(math.pi * step / total_steps))

    def _step_batch(self, batch, step_idx: int, iter_per_epoch: int, epoch: int, total_steps: int):
        X, Y, loss_mask = [t.to(self.device) for t in batch]
        lr = self.get_lr(epoch * iter_per_epoch + step_idx, total_steps, self.config.learning_rate)
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        with self.ctx:
            res = self.model(X)
            loss = self.loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 兼容 MoE 的辅助损失
            aux = getattr(res, 'aux_loss', 0.0)
            loss = (loss + (aux if isinstance(aux, torch.Tensor) else 0.0)) / self.config.accumulation_steps

        self.scaler.scale(loss).backward()
        if (step_idx + 1) % self.config.accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        return loss.detach().item(), lr

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        tot_loss = 0.0
        tot_tokens = 0
        for X, Y, loss_mask in self.val_loader:
            X, Y, loss_mask = X.to(self.device), Y.to(self.device), loss_mask.to(self.device)
            logits = self.model(X).logits
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), Y.view(-1)).view(Y.size())
            loss = (loss * loss_mask).sum()
            tot_loss += loss.item()
            tot_tokens += loss_mask.sum().item()
        self.model.train()
        ppl = math.exp(tot_loss / max(1, tot_tokens)) if tot_tokens > 0 else float('inf')
        return dict(val_loss=tot_loss / max(1, tot_tokens), val_ppl=ppl)

    def train(self):
        self.model.train()
        iter_per_epoch = len(self.train_loader)
        total_steps = max(1, (self.config.epochs * iter_per_epoch if not self.config.max_steps else self.config.max_steps))
        step_global = 0
        for epoch in range(self.config.epochs if not self.config.max_steps else 10**9):
            for step, batch in enumerate(self.train_loader):
                loss, lr = self._step_batch(batch, step, iter_per_epoch, epoch, total_steps)
                step_global += 1
                if step % self.config.log_interval == 0:
                    print(f"epoch={epoch+1} step={step}/{iter_per_epoch} loss={loss:.4f} lr={lr:.8f}")
                if step_global % self.config.save_interval == 0:
                    self.save_ckpt()
                if self.config.max_steps and step_global >= self.config.max_steps:
                    self.save_ckpt()
                    return
            # epoch end
            self.save_ckpt()
            if self.val_loader is not None:
                print(self.evaluate())

    def save_ckpt(self):
        os.makedirs(self.config.out_dir, exist_ok=True)
        path = os.path.join(self.config.out_dir, f"full_sft_{self.config.lm_dim}.pth")
        state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        torch.save(state, path)


def build_sft_loaders(config, tokenizer):
    ds = SFTDataset(config.data_path, tokenizer, max_length=config.max_seq_len)
    if config.test_file and os.path.exists(config.test_file):
        val_ds = SFTDataset(config.test_file, tokenizer, max_length=config.max_seq_len)
    else:
        n = len(ds)
        n_val = max(1, int(0.1 * n))
        n_train = max(1, n - n_val)
        ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=(torch.cuda.is_available()))
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=(torch.cuda.is_available()))
    return train_loader, val_loader
