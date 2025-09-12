#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoTokenizer

from triton_framework.engine.distributed import init_distributed, is_main_process
from triton_framework.engine.schedulers import CosineWithFloor
from triton_framework.engine.utils import get_device, set_seed

# Import MiniLLM components from existing project
from minillm.model.model import MiniLLMLM
from minillm.model.LMConfig import LMConfig
from minillm.model.dataset import PretrainDataset
from triton_framework.modules.basic_block import MiniLLMLM_Triton, MiniLLMConfig


def logger(msg: str):
    if is_main_process():
        print(msg)


def main():
    parser = argparse.ArgumentParser(description="MiniLLM Pretraining (Triton Engine/DDP)")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]) 
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniLLM-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--data_path", type=str, default="./minillm/dataset/pretrain_hq.jsonl")
    parser.add_argument("--dim", default=512, type=int)
    parser.add_argument("--n_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--use_moe", default=False, type=bool)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--engine_model", type=str, default="baseline", choices=["baseline", "triton"], help="Use baseline MiniLLMLM or Triton-backed MiniLLMLM_Triton")
    args = parser.parse_args()

    set_seed(args.seed)

    ddp_ctx = init_distributed(backend="nccl")
    device = ddp_ctx.device if ddp_ctx else get_device("cuda" if torch.cuda.is_available() else "cpu")

    amp_enabled = (args.dtype in ["float16", "bfloat16"]) and (device.type == "cuda")
    autocast_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    # Tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained('./minillm/model/minillm_tokenizer')
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)

    # Vocab size from tokenizer for Triton model config
    vocab_size = tokenizer.vocab_size
    if args.engine_model == "triton":
        mm_cfg = MiniLLMConfig(
            vocab_size=vocab_size,
            dim=lm_config.dim,
            n_heads=lm_config.n_heads,
            n_layers=lm_config.n_layers,
            max_seq_len=lm_config.max_seq_len,
            dropout=lm_config.dropout,
            rope_theta=lm_config.rope_theta,
            use_triton=True,
        )
        model = MiniLLMLM_Triton(mm_cfg).to(device)
    else:
        model = MiniLLMLM(lm_config).to(device)
    if ddp_ctx:
        # match original ignore buffer name
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DDP(model, device_ids=[ddp_ctx.local_rank] if device.type == 'cuda' else None)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    sampler = DistributedSampler(train_ds) if ddp_ctx else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=sampler,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    total_steps = len(train_loader) * args.epochs
    lr_sched = CosineWithFloor(base_lr=args.learning_rate, total_steps=max(1, total_steps))

    use_wandb = args.use_wandb and is_main_process()
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=f"MiniLLM-Pretrain-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}")

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    def save_ckpt():
        if not is_main_process():
            return
        os.makedirs(args.out_dir, exist_ok=True)
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = os.path.join(args.out_dir, f'pretrain_{lm_config.dim}{moe_path}.pth')
        state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        torch.save(state_dict, ckp)

    step_global = 0
    model.train()
    for epoch in range(args.epochs):
        if ddp_ctx and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        start_time = time.time()
        for step, (X, Y, loss_mask) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            loss_mask = loss_mask.to(device)

            # lr schedule
            lr = lr_sched.step(optimizer, step_global)

            # forward
            ctx = (torch.cuda.amp.autocast(dtype=autocast_dtype) if amp_enabled else nullcontext())
            with ctx:
                if args.engine_model == "triton":
                    logits = model(X)
                    loss = loss_fct(logits.view(-1, logits.size(-1)), Y.view(-1)).view(Y.size())
                    loss = (loss * loss_mask).sum() / loss_mask.sum()
                else:
                    res = model(X)
                    logits = res.logits
                    loss = loss_fct(logits.view(-1, logits.size(-1)), Y.view(-1)).view(Y.size())
                    loss = (loss * loss_mask).sum() / loss_mask.sum()
                    loss = loss + res.aux_loss
                loss = loss / args.accumulation_steps

            # backward
            scaler.scale(loss).backward()

            # step
            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # logging
            if step % args.log_interval == 0 and is_main_process():
                dt = time.time() - start_time
                logger(
                    f"Epoch:[{epoch+1}/{args.epochs}]({step}/{len(train_loader)}) "
                    f"loss:{loss.item()*args.accumulation_steps:.3f} lr:{optimizer.param_groups[-1]['lr']:.8f} "
                    f"epoch_Time:{dt/(step+1)*len(train_loader)//60 - dt//60}min"
                )
                if use_wandb:
                    import wandb
                    wandb.log({
                        "loss": loss.item() * args.accumulation_steps,
                        "lr": optimizer.param_groups[-1]['lr'],
                    })

            # save
            if (step + 1) % args.save_interval == 0:
                save_ckpt()

            step_global += 1

        # epoch end save (optional)
        save_ckpt()

    # finalize
    if ddp_ctx and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
