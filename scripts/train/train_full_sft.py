import os
import platform
import argparse
import time
import math
import warnings
import json

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model.tokenizer_utils import build_tokenizer
from model.model import MiniLLMLM
from model.LMConfig import LMConfig
from model.dataset import SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    effective_accum = 1 if use_deepspeed else args.accumulation_steps
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        # 更新学习率（DeepSpeed 则更新其内部 optimizer）
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 兼容 HF 模型：无 aux_loss 时按 0 处理
            aux = getattr(res, 'aux_loss', 0.0)
            loss = loss + (aux if isinstance(aux, torch.Tensor) else torch.tensor(aux, device=loss.device, dtype=loss.dtype))
            loss = loss / effective_accum

        if use_deepspeed:
            model.backward(loss)
        else:
            scaler.scale(loss).backward()

        if (step + 1) % effective_accum == 0:
            if use_deepspeed:
                # DeepSpeed 梯度裁剪在 config 中配置
                model.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            current_lr = lr if (optimizer is None) else optimizer.param_groups[-1]['lr']
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    current_lr,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() if hasattr(loss, 'item') else float(loss),
                           "lr": float(current_lr),
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'
            module = model.module if hasattr(model, 'module') else model
            try:
                state_dict = module.state_dict()
                torch.save(state_dict, ckp)
            except Exception:
                # DeepSpeed engine state_dict可能分片，简单保存失败则跳过周期性保存
                Logger('周期性保存 state_dict 失败（可能为 DeepSpeed 分片），已跳过。')
            model.train()


def init_model(lm_config):
    """Initialize model: MiniLLM (default) or HF CausalLM depending on args.model_type.
    - For mini: build MiniLLMLM and optionally load ckpt (state_dict)
    - For hf: load AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    """
    if args.model_type == 'hf':
        assert args.model_name_or_path is not None, "--model_name_or_path 必须提供以加载 HF 模型"
        def _load_config(trust_rc: bool):
            return AutoConfig.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=trust_rc,
                revision=args.hf_revision if getattr(args, 'hf_revision', None) else None,
            )
        def _load_model(config, trust_rc: bool):
            return AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                config=config,
                torch_dtype=(torch.bfloat16 if args.dtype == 'bfloat16' else (torch.float16 if args.dtype == 'float16' else None)),
                trust_remote_code=trust_rc,
                revision=args.hf_revision if getattr(args, 'hf_revision', None) else None,
            )
        # First attempt: honor user flag
        try:
            config = _load_config(args.trust_remote_code)
        except Exception as e1:
            Logger(f"加载 HF 配置失败（trust_remote_code={args.trust_remote_code}）: {e1}")
            # Fallback: force trust_remote_code=True
            try:
                Logger("尝试使用 trust_remote_code=True 重新加载配置……")
                config = _load_config(True)
            except Exception as e2:
                Logger(f"再次加载配置失败: {e2}")
                Logger("若提示 'does not recognize this architecture'，请：1) 启用 --trust_remote_code；2) 升级 transformers；3) 指定正确的 --hf_revision。")
                raise
        try:
            model = _load_model(config, args.trust_remote_code)
        except Exception as e1:
            Logger(f"加载 HF 模型失败（trust_remote_code={args.trust_remote_code}）: {e1}")
            try:
                Logger("尝试使用 trust_remote_code=True 重新加载模型……")
                model = _load_model(config, True)
            except Exception as e2:
                Logger(f"再次加载模型失败: {e2}")
                Logger("常见原因：transformers 版本过旧无法识别该架构（如 qwen3）。建议升级 transformers 到较新版本，或使用提供 auto_map 的 repo 并启用 --trust_remote_code。")
                raise
        # 若词表大小与 tokenizer 不一致，强制 resize 以避免索引越界
        if getattr(model, 'get_input_embeddings', None) is not None:
            emb = model.get_input_embeddings()
            if emb is not None and emb.num_embeddings != lm_config.vocab_size:
                Logger(f"调整词表大小: model.num_embeddings={emb.num_embeddings} -> tokenizer.vocab_size={lm_config.vocab_size}")
                try:
                    model.resize_token_embeddings(lm_config.vocab_size)
                except Exception as e:
                    Logger(f"resize_token_embeddings 失败: {e}")
        Logger(f'HF模型参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        model = model.to(args.device)
        Logger(model)
        return model
    else:
        model = MiniLLMLM(lm_config)
        moe_path = '_moe' if lm_config.use_moe else ''
        if args.ckp is None:
            ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
        else:
            ckp = args.ckp
        if ckp is not None and os.path.exists(ckp):
            print("Loading pretrained model from {}", ckp)
            state_dict = torch.load(ckp, map_location=args.device)
            model.load_state_dict(state_dict, strict=False)
        else:
            Logger(f"未提供/未找到预训练权重，将从随机初始化开始：{ckp}")
        Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        model = model.to(args.device)
        # Print model structure at start (rank-0 only if DDP)
        Logger(model)
        return model


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniLLM Full SFT")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniLLM-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    # Model selection
    parser.add_argument('--model_type', type=str, default='mini', choices=['mini','hf'],
                        help='选择要微调的模型类型：mini=仓库自研结构；hf=HuggingFace模型。')
    parser.add_argument('--model_name_or_path', type=str, default=None,
                        help='当 --model_type=hf 时，指定 HF 模型路径或 repo id（例如 Qwen/Qwen-8B）')
    parser.add_argument('--hf_revision', type=str, default=None, help='可选：HF 仓库分支/commit（如 main、refs/pr/xx、commit sha）')
    # Mini 模型结构参数（当 --model_type=mini 时生效）
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--n_block', default=None, type=int, help='Number of blocks; each block uses virtual pattern 0,1,2,3,1,2,3,4')
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--repeat_layer', action='store_true', help='Enable layer parameter sharing pattern 0,1,2,3,1,2,3,4')
    parser.add_argument("--data_path", type=str, default="./dataset/sft_mini_512.jsonl")
    parser.add_argument("--tokenizer_dir", type=str, default="./model/minillm_tokenizer")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--ckp", type=str, default=None, help='当 --model_type=mini 时用于加载预训练 state_dict；HF 模型忽略该参数')
    parser.add_argument('--save_hf', action='store_true', help='保存为 HF 格式（当 --model_type=hf 时使用 save_pretrained；mini 时保存 state_dict）')
    # DeepSpeed 集成
    parser.add_argument('--deepspeed', type=str, default=None, help='DeepSpeed 配置文件路径（JSON）。提供后将启用 DeepSpeed 引擎（ZeRO 等）')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='HF 模型启用梯度检查点以降低显存')

    args = parser.parse_args()

    # 构造 LMConfig：对于 HF 仅用于存储 vocab_size 和 max_seq_len 等数据集相关参数
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, n_block=args.n_block, max_seq_len=args.max_seq_len, use_moe=args.use_moe, repeat_layer=args.repeat_layer)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    print(torch.cuda.is_available(), device_type)
    args.wandb_run_name = f"MiniLLM-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # Use explicit autocast only when dtype suggests it; allow disabling via env
    use_amp = os.environ.get('DISABLE_AMP', '0') != '1'
    if device_type == "cpu" or not use_amp:
        ctx = nullcontext()
    else:
        # Prefer bfloat16 if requested, else float16
        amp_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else (torch.float16 if args.dtype == 'float16' else None)
        ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype is not None else nullcontext()
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # build tokenizer and align vocab
    tokenizer = build_tokenizer(args.tokenizer_dir, trust_remote_code=args.trust_remote_code)
    lm_config.vocab_size = tokenizer.vocab_size
    model = init_model(lm_config)
    # HF: 训练建议关闭缓存并可选启用梯度检查点
    if args.model_type == 'hf':
        try:
            if hasattr(model, 'config'):
                model.config.use_cache = False
            if args.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
        except Exception:
            pass

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    use_deepspeed = args.deepspeed is not None
    optimizer = None
    if use_deepspeed:
        import deepspeed
        # 让 DeepSpeed 根据 config 创建 optimizer/scheduler
        ds_cfg = args.deepspeed
        if isinstance(ds_cfg, str) and os.path.isfile(ds_cfg):
            with open(ds_cfg, 'r') as f:
                ds_cfg = json.load(f)
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_cfg,
        )
        scaler = None  # DeepSpeed 自管精度
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and args.dtype in ['float16', 'bfloat16']))
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp and not use_deepspeed:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)

    # 训练完成后的保存逻辑
    if (not ddp or dist.get_rank() == 0):
        os.makedirs(args.save_dir, exist_ok=True)
        if args.model_type == 'hf' and args.save_hf:
            # 保存为 HF 格式目录
            save_path = os.path.join(args.save_dir, 'hf_finetuned')
            try:
                module = model.module if hasattr(model, 'module') else model
                module.save_pretrained(save_path, safe_serialization=False)
                tokenizer.save_pretrained(save_path)
                Logger(f"HF 模型已保存到: {save_path}")
            except Exception as e:
                Logger(f"保存 HF 模型失败: {e}")
        elif args.model_type == 'mini':
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'
            module = model.module if hasattr(model, 'module') else model
            try:
                torch.save(module.state_dict(), ckp)
            except Exception as e:
                Logger(f"保存 MiniLLM state_dict 失败: {e}")
            Logger(f"MiniLLM state_dict 已保存到: {ckp}")
