#!/usr/bin/env python3
from __future__ import annotations

"""
Unified training script based on the Triton training architecture.
Use --stage to select: pretrain | sft | dpo

Forward uses Triton kernels where available; backward relies on PyTorch autograd.
"""

import os
import sys
import time
import gc
import argparse
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

 # Add repo root and package src for local runs without installation
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from triton_framework.src.modules.basic_block import MiniLLMLM_Triton, MiniLLMConfig
from triton_framework.src.engine.schedulers import CosineWithFloor
from triton_framework.src.engine.utils import get_device, set_seed

# Baseline (reference) model and datasets
from model.LMConfig import LMConfig
from model.model import MiniLLMLM
from model.dataset import PretrainDataset, SFTDataset, DPODataset


def build_tokenizer(tokenizer_dir: str):
    return AutoTokenizer.from_pretrained(tokenizer_dir)


def build_model(args, tokenizer):
    vocab_size = tokenizer.vocab_size
    if args.engine_model == "triton":
        cfg = MiniLLMConfig(
            vocab_size=vocab_size,
            dim=args.dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            rope_theta=1e6,
            use_triton=(args.device.startswith("cuda") and torch.cuda.is_available()),
        )
        model = MiniLLMLM_Triton(cfg)
        return model
    else:
        lm_cfg = LMConfig(
            dim=args.dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
        )
        return MiniLLMLM(lm_cfg)


def iter_dl(dataloader):
    it = iter(dataloader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(dataloader)


def seq_logprobs(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits: [B,T,V], targets: [B,T], mask: [B,T] bool/int
    logp = torch.log_softmax(logits, dim=-1)
    tgt_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B,T]
    if mask.dtype != torch.bool:
        mask = mask.bool()
    # per-sample mean over masked tokens
    lengths = mask.sum(dim=1).clamp_min(1)
    return (tgt_logp * mask).sum(dim=1) / lengths


def run_pretrain(args, model, device, tokenizer, ddp_ctx=None, max_steps: int | None = None):
    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = DistributedSampler(ds) if ddp_ctx else None
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == 'cuda')
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = max(1, len(dl))
    sched = CosineWithFloor(base_lr=args.learning_rate, total_steps=args.epochs * steps_per_epoch)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    model.train().to(device)
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16) if (args.amp and device.type == 'cuda') else nullcontext()
    step_global = 0
    epoch_times = []
    for epoch in range(args.epochs):
        t0 = time.time()
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, (X, Y, mask) in enumerate(dl):
            if max_steps is not None and step_global >= max_steps:
                break
            X, Y, mask = X.to(device), Y.to(device), mask.to(device)
            for _ in range(args.accumulation_steps):
                optim.zero_grad(set_to_none=True)
                with autocast_ctx:
                    if args.engine_model == 'triton':
                        logits = model(X)
                    else:
                        out = model(X)
                        logits = out.logits
                    loss_tok = loss_fct(logits.view(-1, logits.size(-1)), Y.view(-1)).view_as(Y)
                    loss = (loss_tok * mask).sum() / mask.sum()
                    loss = loss / args.accumulation_steps
                scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()
            sched.step(optim, step_global)
            if (step_global % args.log_interval) == 0 and (not ddp_ctx or ddp_ctx['rank'] == 0):
                print(f"pretrain e{epoch} i{step} | loss {loss.item()*args.accumulation_steps:.4f} | lr {optim.param_groups[-1]['lr']:.6e}")
            step_global += 1
        dur = time.time() - t0
        epoch_times.append(dur)
        if (not ddp_ctx or ddp_ctx['rank'] == 0):
            print(f"pretrain epoch {epoch} time: {dur:.3f}s")
    return epoch_times


def run_sft(args, model, device, tokenizer, ddp_ctx=None, max_steps: int | None = None):
    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = DistributedSampler(ds) if ddp_ctx else None
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == 'cuda')
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = max(1, len(dl))
    sched = CosineWithFloor(base_lr=args.learning_rate, total_steps=args.epochs * steps_per_epoch)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    model.train().to(device)
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16) if (args.amp and device.type == 'cuda') else nullcontext()
    step_global = 0
    epoch_times = []
    for epoch in range(args.epochs):
        t0 = time.time()
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, (X, Y, mask) in enumerate(dl):
            if max_steps is not None and step_global >= max_steps:
                break
            X, Y, mask = X.to(device), Y.to(device), mask.to(device)
            for _ in range(args.accumulation_steps):
                optim.zero_grad(set_to_none=True)
                with autocast_ctx:
                    if args.engine_model == 'triton':
                        logits = model(X)
                    else:
                        out = model(X)
                        logits = out.logits
                    loss_tok = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='none').view_as(Y)
                    loss = (loss_tok * mask).sum() / mask.sum()
                    loss = loss / args.accumulation_steps
                scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()
            sched.step(optim, step_global)
            if (step_global % args.log_interval) == 0 and (not ddp_ctx or ddp_ctx['rank'] == 0):
                print(f"sft e{epoch} i{step} | loss {loss.item()*args.accumulation_steps:.4f} | lr {optim.param_groups[-1]['lr']:.6e}")
            step_global += 1
        dur = time.time() - t0
        epoch_times.append(dur)
        if (not ddp_ctx or ddp_ctx['rank'] == 0):
            print(f"sft epoch {epoch} time: {dur:.3f}s")
    return epoch_times


@dataclass
class DPOConfig:
    beta: float = 0.1
    use_ref: bool = True


def run_dpo(args, model, device, tokenizer, ddp_ctx=None, max_steps: int | None = None):
    ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = DistributedSampler(ds) if ddp_ctx else None
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == 'cuda')
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = max(1, len(dl))
    sched = CosineWithFloor(base_lr=args.learning_rate, total_steps=args.epochs * steps_per_epoch)
    cfg = DPOConfig(beta=args.dpo_beta, use_ref=args.dpo_use_ref)

    model.train().to(device)
    # Build reference model (frozen copy) if requested and baseline engine used as well
    if cfg.use_ref:
        # ref shares architecture and vocab; copy weights and freeze
        if args.engine_model == 'triton':
            ref = build_model(args, tokenizer).to(device)
            ref.load_state_dict(model.state_dict())
        else:
            ref = build_model(args, tokenizer).to(device)
            ref.load_state_dict(model.state_dict())
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
    else:
        ref = None

    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16) if (args.amp and device.type == 'cuda') else nullcontext()
    step_global = 0
    epoch_times = []
    for epoch in range(args.epochs):
        t0 = time.time()
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dl):
            if max_steps is not None and step_global >= max_steps:
                break
            x_c = batch['x_chosen'].to(device)
            y_c = batch['y_chosen'].to(device)
            m_c = batch['mask_chosen'].to(device)
            x_r = batch['x_rejected'].to(device)
            y_r = batch['y_rejected'].to(device)
            m_r = batch['mask_rejected'].to(device)

            for _ in range(args.accumulation_steps):
                optim.zero_grad(set_to_none=True)
                with autocast_ctx:
                    # policy logprobs
                    if args.engine_model == 'triton':
                        logits_c = model(x_c)
                        logits_r = model(x_r)
                    else:
                        out_c = model(x_c)
                        out_r = model(x_r)
                        logits_c = out_c.logits
                        logits_r = out_r.logits
                    logp_c = seq_logprobs(logits_c, y_c, m_c)  # [B]
                    logp_r = seq_logprobs(logits_r, y_r, m_r)

                    if ref is not None:
                        with torch.no_grad():
                            if args.engine_model == 'triton':
                                ref_logp_c = seq_logprobs(ref(x_c), y_c, m_c)
                                ref_logp_r = seq_logprobs(ref(x_r), y_r, m_r)
                            else:
                                ref_logp_c = seq_logprobs(ref(x_c).logits, y_c, m_c)
                                ref_logp_r = seq_logprobs(ref(x_r).logits, y_r, m_r)
                        adv = (logp_c - ref_logp_c) - (logp_r - ref_logp_r)
                    else:
                        adv = (logp_c - logp_r)

                    # DPO loss: -E log sigma(beta * advantage)
                    loss = torch.nn.functional.softplus(-cfg.beta * adv).mean()  # -log(sigmoid(x)) = softplus(-x)
                    loss = loss / args.accumulation_steps

                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
                sched.step(optim, step_global)
                if (step_global % args.log_interval) == 0 and (not ddp_ctx or ddp_ctx['rank'] == 0):
                    print(f"dpo e{epoch} i{step} | loss {loss.item()*args.accumulation_steps:.4f} | lr {optim.param_groups[-1]['lr']:.6e}")
                step_global += 1
        dur = time.time() - t0
        epoch_times.append(dur)
        if (not ddp_ctx or ddp_ctx['rank'] == 0):
            print(f"dpo epoch {epoch} time: {dur:.3f}s")
    return epoch_times


def main():
    p = argparse.ArgumentParser(description="Unified training (Triton architecture)")
    p.add_argument('--stage', type=str, required=True, choices=['pretrain', 'sft', 'dpo'])
    p.add_argument('--tokenizer_dir', type=str, default='./model/minillm_tokenizer')
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='out')

    # Model
    p.add_argument('--engine_model', type=str, default='triton', choices=['triton', 'baseline'])
    p.add_argument('--dim', type=int, default=512)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--n_layers', type=int, default=8)
    p.add_argument('--max_seq_len', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.0)

    # Train
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--learning_rate', type=float, default=5e-4)
    p.add_argument('--accumulation_steps', type=int, default=1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16'])
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--seed', type=int, default=1337)

    # DPO specific
    p.add_argument('--dpo_beta', type=float, default=0.1)
    p.add_argument('--dpo_use_ref', action='store_true')
    # Benchmark
    p.add_argument('--time_compare', action='store_true', help='Run 1 epoch with baseline and Triton engines and compare times')
    p.add_argument('--max_step', type=int, default=5000, help='Only when --time_compare: stop after this many steps')

    args = p.parse_args()

    # init distributed from torchrun env
    def init_distributed():
        if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
            return None
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
        return {'rank': rank, 'world_size': world_size, 'local_rank': local_rank, 'device': device}

    set_seed(args.seed)
    ddp_ctx = init_distributed()
    if ddp_ctx:
        device = ddp_ctx['device']
    else:
        device = get_device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = str(device)

    tokenizer = build_tokenizer(args.tokenizer_dir)

    # Optional timing comparison mode
    if args.time_compare:
        engines = ['baseline', 'triton']
        times = {}

        # ensure exactly one epoch for fair comparison
        orig_epochs = args.epochs
        args.epochs = 1

        for engine in engines:
            args.engine_model = engine
            if (not ddp_ctx or ddp_ctx['rank'] == 0):
                print(f"\n[Timing] Running 1 epoch with engine={engine} stage={args.stage} ...")
            model = build_model(args, tokenizer).to(device)
            if ddp_ctx:
                try:
                    if hasattr(model, '_ddp_params_and_buffers_to_ignore'):
                        model._ddp_params_and_buffers_to_ignore.add('pos_cis')
                    else:
                        model._ddp_params_and_buffers_to_ignore = {'pos_cis'}
                except Exception:
                    pass
                model = DDP(model, device_ids=[ddp_ctx['local_rank']] if device.type == 'cuda' else None)

            if args.stage == 'pretrain':
                epoch_times = run_pretrain(args, model, device, tokenizer, ddp_ctx, max_steps=args.max_step)
            elif args.stage == 'sft':
                epoch_times = run_sft(args, model, device, tokenizer, ddp_ctx, max_steps=args.max_step)
            elif args.stage == 'dpo':
                epoch_times = run_dpo(args, model, device, tokenizer, ddp_ctx, max_steps=args.max_step)
            else:
                raise ValueError(f"Unknown stage: {args.stage}")

            times[engine] = epoch_times[0] if epoch_times else float('nan')
            # cleanup
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        args.epochs = orig_epochs
        if (not ddp_ctx or ddp_ctx['rank'] == 0):
            base_t = times.get('baseline', float('nan'))
            tri_t = times.get('triton', float('nan'))
            speedup = (base_t / tri_t) if (tri_t and tri_t > 0) else float('nan')
            print("\n===== Timing Comparison (1 epoch) =====")
            print(f"Baseline (PyTorch): {base_t:.3f}s")
            print(f"Triton engine     : {tri_t:.3f}s")
            print(f"Speedup (baseline/triton): {speedup:.2f}x")
    else:
        # Normal single-engine training
        model = build_model(args, tokenizer).to(device)
        if ddp_ctx:
            # try to ignore rope buffer in sync
            try:
                if hasattr(model, '_ddp_params_and_buffers_to_ignore'):
                    model._ddp_params_and_buffers_to_ignore.add('pos_cis')
                else:
                    model._ddp_params_and_buffers_to_ignore = {'pos_cis'}
            except Exception:
                pass
            model = DDP(model, device_ids=[ddp_ctx['local_rank']] if device.type == 'cuda' else None)

        if args.stage == 'pretrain':
            run_pretrain(args, model, device, tokenizer, ddp_ctx)
        elif args.stage == 'sft':
            run_sft(args, model, device, tokenizer, ddp_ctx)
        elif args.stage == 'dpo':
            run_dpo(args, model, device, tokenizer, ddp_ctx)
        else:
            raise ValueError(f"Unknown stage: {args.stage}")

    if ddp_ctx and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
