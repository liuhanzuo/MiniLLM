import os
import sys
import sys
import argparse
import time
import math
import warnings
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

# Ensure project root on sys.path before importing project modules
_THIS_DIR = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
try:
    from model.tokenizer_utils import build_tokenizer
except Exception:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from model.tokenizer_utils import build_tokenizer
from model.model import MiniLLMLM
from model.LMConfig import LMConfig
from model.dataset import SFTDataset

warnings.filterwarnings('ignore')

# ---------------------- Distillation helpers ----------------------
def kl_logit_distill(student_logits: torch.Tensor,
                     teacher_logits: torch.Tensor,
                     temperature: float = 1.0,
                     reduction: str = 'batchmean') -> torch.Tensor:
    """KL(student||teacher) with temperature, on last dim.
    Shapes: [..., V]. Returns scalar by default (batchmean).
    """
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl


def hard_response_distill(student_logits: torch.Tensor,
                          teacher_logits: torch.Tensor,
                          reduction: str = 'mean') -> torch.Tensor:
    """Cross-entropy to teacher argmax tokens (response-based / hard-label KD)."""
    with torch.no_grad():
        hard = teacher_logits.argmax(dim=-1)
    loss = F.cross_entropy(student_logits.transpose(1, 2), hard, reduction=reduction)
    return loss


class FeatureCollector:
    """Register forward hooks on selected transformer blocks to collect hidden states."""
    def __init__(self, model: nn.Module, layer_indices: List[int]):
        self.model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        self.layer_indices = layer_indices
        self.outputs: Dict[int, torch.Tensor] = {}
        self._hooks: List[Any] = []
        # Assume MiniLLMLM has attribute `.layers` (ModuleList of MiniLLMBlock)
        layers = getattr(self.model, 'layers', None)
        assert layers is not None, "Model must expose .layers"
        for idx in layer_indices:
            assert 0 <= idx < len(layers), f"Layer index {idx} out of range"
            def make_hook(i):
                def hook(module, inp, out):
                    # out is (h, past_kv) for MiniLLMBlock.forward return; but hook on block returns its output (tuple).
                    # Safer: hook at module output via register_forward_hook gives the returned value.
                    if isinstance(out, tuple):
                        self.outputs[i] = out[0].detach()
                    else:
                        self.outputs[i] = out.detach()
                return hook
            h = layers[idx].register_forward_hook(make_hook(idx))
            self._hooks.append(h)

    def clear(self):
        self.outputs.clear()

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


@torch.no_grad()
def update_ema_teacher(ema_model: nn.Module, student_model: nn.Module, decay: float) -> None:
    sm = student_model.module if isinstance(student_model, torch.nn.parallel.DistributedDataParallel) else student_model
    for p_ema, p in zip(ema_model.parameters(), sm.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=(1.0 - decay))

## Remove duplicated helper block above (deduped)
##


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def masked_mean(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # t: [B,T,...], mask: [B,T] in {0,1}
    mask = mask.float()
    while t.dim() > mask.dim():
        mask = mask.unsqueeze(-1)
    s = (t * mask).sum()
    d = mask.sum().clamp_min(1.0)
    return s / d


def train_epoch(epoch, wandb, alpha=0.0, temperature=1.0):
    start_time = time.time()

    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step,
                    args.epochs * iter_per_epoch,
                    args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播（学生模型）
        with ctx:
            res = model(X)
            student_logits = res.logits

        # 教师模型前向传播（只在eval & no_grad）
        teacher_logits = None
        if args.distillation_mode in {"logit", "response", "feature", "self"} and teacher_model is not None:
            with torch.no_grad():
                if getattr(args, 'teacher_model_type', 'mini') == 'hf':
                    # HF teacher forward
                    tout = teacher_model(X)
                    tlogits = tout.logits
                else:
                    tout = teacher_model(X)
                    tlogits = tout.logits
                # 对齐词表到学生
                vocab_size_student = student_logits.size(-1)
                if tlogits.size(-1) != vocab_size_student:
                    tlogits = tlogits[..., :vocab_size_student]
                teacher_logits = tlogits

        # ========== 计算损失 ==========
        # 1) Ground-Truth CE Loss（可选）
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            Y.view(-1),
            ignore_index=0,
            reduction='none'
        )
        ce_loss = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum()
        if lm_config_student.use_moe:
            ce_loss += res.aux_loss

    # 2) Distillation Loss（根据模式）
        distill_loss = torch.tensor(0.0, device=args.device)
        mode = args.distillation_mode
        if mode == 'logit' and teacher_logits is not None:
            v = min(student_logits.size(-1), teacher_logits.size(-1))
            st_flat = student_logits[..., :v].contiguous().view(-1, v)
            te_flat = teacher_logits[..., :v].contiguous().view(-1, v)
            m = (loss_mask_flat == 1)
            distill_loss = kl_logit_distill(st_flat[m], te_flat[m], temperature=temperature)
        elif mode == 'response' and teacher_logits is not None:
            # 硬标签：teacher argmax。按有效 token 掩码聚合
            v = min(student_logits.size(-1), teacher_logits.size(-1))
            with torch.no_grad():
                hard = teacher_logits[..., :v].argmax(dim=-1)
            token_ce = F.cross_entropy(
                student_logits[..., :v].contiguous().view(-1, v),
                hard.view(-1),
                reduction='none'
            )
            distill_loss = torch.sum(token_ce * loss_mask_flat) / loss_mask_flat.sum()
        elif mode in {'feature', 'self'}:
            # 收集中间层表示并做 MSE 对齐（如维度不同，使用线性投影）
            # 对 HF 教师：优先使用 hidden_states（需模型开启输出）
            feat_loss = torch.tensor(0.0, device=args.device)
            try:
                len_layers = len(getattr(model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model, 'layers'))
                layers = select_feat_layers(len_layers=len_layers, k=args.feature_layers)
                stu_fc = FeatureCollector(model, layers)
                # 教师为 HF 时，尝试开启输出 hidden_states
                if getattr(args, 'teacher_model_type', 'mini') == 'hf':
                    # 重新前向以取 hidden_states
                    with torch.no_grad():
                        tout2 = teacher_model(X, output_hidden_states=True, use_cache=False)
                        t_hs = list(tout2.hidden_states) if tout2.hidden_states is not None else []
                    # 学生 forward 触发 hook
                    _ = model(X)
                    for li in layers:
                        s = stu_fc.outputs[li]
                        if li < len(t_hs):
                            t = t_hs[li]
                        else:
                            t = t_hs[-1] if len(t_hs) > 0 else None
                        if t is None:
                            continue
                        if s.size(-1) != t.size(-1):
                            t = feature_project(t, t.size(-1), s.size(-1))
                        feat_loss = feat_loss + masked_mean(F.mse_loss(s, t, reduction='none'), loss_mask)
                    stu_fc.close()
                else:
                    tea_fc = FeatureCollector(teacher_model if teacher_model is not None else model, layers)
                    with torch.no_grad():
                        _ = (teacher_model(X) if teacher_model is not None else model(X))
                    _ = model(X)
                    for li in layers:
                        s = stu_fc.outputs[li]
                        t = tea_fc.outputs[li]
                        if s.size(-1) != t.size(-1):
                            t = feature_project(t, t.size(-1), s.size(-1))
                        feat_loss = feat_loss + masked_mean(F.mse_loss(s, t, reduction='none'), loss_mask)
                    stu_fc.close(); tea_fc.close()
                distill_loss = feat_loss / max(1, int(args.feature_layers) if int(args.feature_layers) > 0 else 1)
            except Exception as e:
                Logger(f"[feature distill] 回退到 logit 蒸馏，原因: {e}")
                if teacher_logits is not None:
                    v = min(student_logits.size(-1), teacher_logits.size(-1))
                    st_flat = student_logits[..., :v].contiguous().view(-1, v)
                    te_flat = teacher_logits[..., :v].contiguous().view(-1, v)
                    m = (loss_mask.view(-1) == 1)
                    distill_loss = kl_logit_distill(st_flat[m], te_flat[m], temperature=temperature)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill（若无 teacher 则退化为纯 CE；self 模式保留 distill 分量）
        if teacher_model is None and mode != 'self':
            loss = ce_loss
        else:
            loss = alpha * ce_loss + (1 - alpha) * distill_loss

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # EMA teacher（自蒸馏）在优化器步进后更新
            if args.distillation_mode == 'self' and teacher_model is not None:
                update_ema_teacher(teacher_model, model, decay=args.ema_decay)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.4f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs - 1,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                )
            )

        if (wandb is not None) and (not ddp or dist.get_rank() == 0):
            wandb.log({
                "loss": loss.item(),
                "ce_loss": ce_loss.item(),
                "distill_loss": distill_loss.item() if (teacher_model is not None or args.distillation_mode == 'self') else 0.0,
                "lr": optimizer.param_groups[-1]['lr'],
                "last-time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
            })

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/full_dist_{lm_config_student.dim}{moe_path}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_student_model(lm_config):
    model = MiniLLMLM(lm_config)
    Logger(f"学生模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万")
    moe_path = '_moe' if lm_config.use_moe else ''
    # prefer explicit path
    ckp = args.student_ckpt if getattr(args, 'student_ckpt', None) else f'./out/full_sft_{lm_config.dim}{moe_path}.pth'
    if getattr(args, 'student_random_init', False):
        Logger('已启用 --student_random_init，学生模型将使用随机初始化参数')
    elif ckp and os.path.exists(ckp):
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)
        Logger(f'学生模型已从 {ckp} 加载权重')
    else:
        Logger(f'未找到学生权重 {ckp}，使用随机初始化继续训练')
    Logger(f'学生模型(LLM)总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万, dim={lm_config.dim}, layers={lm_config.n_layers}')
    model = model.to(args.device)
    # Print student model structure at start (rank-0 only if DDP)
    Logger(model)

    return model


def init_teacher_model(lm_config):
    # 根据模式决定教师来源：
    Logger(f"教师模型配置: dim={lm_config.dim}, layers={lm_config.n_layers}")
    Logger(f"教师模型参数量: {sum(p.numel() for p in MiniLLMLM(lm_config).parameters() if p.requires_grad) / 1e6:.3f} 百万")
    if args.distillation_mode == 'self':
        # 自蒸馏：拷贝学生作为 EMA 教师，初始化为学生权重
        Logger('使用 EMA 自蒸馏教师（初始化为学生模型权重）')
        ema = MiniLLMLM(lm_config_student)  # 结构与学生一致
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            sd = model.module.state_dict()
        else:
            sd = model.state_dict()
        ema.load_state_dict(sd, strict=True)
        for p in ema.parameters():
            p.requires_grad_(False)
        ema.eval().to(args.device)
        return ema
    else:
        # 支持两类教师：MiniLLM（默认）与 HF 模型
        ttype = getattr(args, 'teacher_model_type', 'mini')
        if ttype == 'hf':
            Logger('使用 HF 教师模型')
            # 加载 HF 模型，尽量兼容 remote code
            def _load_hf(trust_rc: bool):
                return AutoModelForCausalLM.from_pretrained(
                    args.teacher_model_name_or_path,
                    trust_remote_code=trust_rc,
                    torch_dtype=(torch.bfloat16 if args.dtype == 'bfloat16' else (torch.float16 if args.dtype == 'float16' else None)),
                )
            try:
                model_t = _load_hf(args.trust_remote_code)
            except Exception as e1:
                Logger(f"加载 HF 教师失败(trust_remote_code={args.trust_remote_code}): {e1}")
                Logger("尝试使用 trust_remote_code=True 重新加载…")
                model_t = _load_hf(True)
            # 尽量关闭缓存并开启梯度检查点按需
            try:
                if hasattr(model_t, 'config'):
                    model_t.config.use_cache = False
            except Exception:
                pass
            model_t = model_t.to(args.device)
            Logger(f'教师(HF)参数量：{sum(p.numel() for p in model_t.parameters() if p.requires_grad) / 1e6:.3f} 百万')
            Logger(model_t)
            return model_t.eval()
        else:
            model_t = MiniLLMLM(lm_config)
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = args.teacher_ckpt if getattr(args, 'teacher_ckpt', None) else f'./out/full_sft_{lm_config.dim}{moe_path}.pth'
            if not (ckp and os.path.exists(ckp)):
                raise FileNotFoundError(f"未找到教师权重: {ckp}. 请通过 --teacher_ckpt 指定，或确保默认路径存在。")
            state_dict = torch.load(ckp, map_location=args.device)
            model_t.load_state_dict(state_dict, strict=False)
            Logger(f'教师模型已从 {ckp} 加载权重')
            Logger(f'教师模型(LLM)总参数量：{sum(p.numel() for p in model_t.parameters() if p.requires_grad) / 1e6:.3f} 百万, dim={lm_config.dim}, layers={lm_config.n_layers}')
            model_t = model_t.to(args.device)
            Logger(model_t)
            return model_t


def feature_project(x: torch.Tensor, in_dim: int, out_dim: int) -> torch.Tensor:
    """On-the-fly linear projection to match dims (no learned params to keep simple)."""
    if in_dim == out_dim:
        return x
    # 使用固定随机正交近似：用 matmul 做降/升维，不引入需要保存的权重
    # 为可复现起见，基于 in/out 维度生成确定性随机矩阵（CPU 上生成再搬到设备）
    device = x.device
    with torch.no_grad():
        torch.manual_seed(in_dim * 1000003 + out_dim)
        W = torch.randn(in_dim, out_dim, dtype=x.dtype, device=device)
        W = F.normalize(W, dim=0)
    y = torch.matmul(x, W)  # [B,T,out_dim]
    return y


def select_feat_layers(len_layers: int, k: int) -> List[int]:
    if k <= 0:
        return [len_layers - 1]
    if k >= len_layers:
        return list(range(len_layers))
    # 均匀采样 k 个层索引
    import numpy as np
    idx = np.linspace(0, len_layers - 1, num=k, dtype=int).tolist()
    return sorted(set(idx))


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
    parser = argparse.ArgumentParser(description="MiniLLM Distillation")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
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
    parser.add_argument("--data_path", type=str, default="./dataset/sft_data.jsonl")
    parser.add_argument("--tokenizer_dir", type=str, default="./model/minillm_tokenizer")
    parser.add_argument("--trust_remote_code", action="store_true")
    # Model structure overrides
    parser.add_argument("--student_dim", type=int, default=512,
                        help="学生模型隐藏维度；默认512，与原脚本一致")
    parser.add_argument("--student_layers", type=int, default=8,
                        help="学生层数（当提供 --n_block 时按块模式覆盖）")
    parser.add_argument("--student_block", type=int, default=None,
                        help="学生块数（每块含若干层，按 0,1,2,3,1,2,3,4 模式堆叠）")
    parser.add_argument("--teacher_dim", type=int, default=768,
                        help="教师模型隐藏维度；可设为2048对齐你的SFT教师")
    parser.add_argument("--teacher_layers", type=int, default=16,
                        help="教师层数（当提供 --n_block 时按块模式覆盖）")
    parser.add_argument("--teacher_block", type=int, default=None,
                        help="教师块数（每块含若干层，按 0,1,2,3,1,2,3,4 模式堆叠）")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="训练最大序列长度，用于数据集截断与位置编码")
    # Optional checkpoint paths
    parser.add_argument("--student_ckpt", type=str, default=None,
                        help="学生初始权重路径；未提供时使用 ./out/full_sft_{student_dim}[_moe].pth，如不存在则随机初始化")
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="教师权重路径；未提供时使用 ./out/full_sft_{teacher_dim}[_moe].pth")
    parser.add_argument("--student_random_init", action="store_true",
                        help="强制学生模型随机初始化（即使存在 student_ckpt 也不加载）")
    # HF 教师模型支持
    parser.add_argument("--teacher_model_type", type=str, default="mini", choices=["mini","hf"],
                        help="教师模型类型：mini=MiniLLM 结构；hf=HuggingFace Transformers 模型")
    parser.add_argument("--teacher_model_name_or_path", type=str, default=None,
                        help="当 --teacher_model_type=hf 时，指定 HF 模型 repo id 或本地路径（如 Qwen/Qwen2.5-7B-Instruct）")
    # Distillation options
    parser.add_argument("--distillation_mode", type=str, default="logit",
                        choices=["logit", "response", "feature", "self"],
                        help="选择蒸馏方式：logit-KL、response-硬标签、feature-特征对齐、self-EMA 自蒸馏")
    parser.add_argument("--alpha", type=float, default=0.0, help="总损失 = alpha*CE + (1-alpha)*Distill")
    parser.add_argument("--temperature", type=float, default=1.0, help="logit 蒸馏温度")
    parser.add_argument("--feature_layers", type=int, default=4, help="特征蒸馏采样的层数")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="自蒸馏 EMA 衰减")
    parser.add_argument('--repeat_layer', action='store_true', help='Enable layer parameter sharing pattern 0,1,2,3,1,2,3,4')

    args = parser.parse_args()
    # 定义学生模型和教师模型（支持自定义维度/层数/序列长度）
    lm_config_student = LMConfig(
        dim=args.student_dim,
        n_layers=args.student_layers,
        n_block=args.student_block,
        max_seq_len=args.max_seq_len,
        repeat_layer=args.repeat_layer,
    )
    lm_config_teacher = LMConfig(
        dim=args.teacher_dim,
        n_layers=args.teacher_layers,
        n_block=args.teacher_block,
        max_seq_len=args.max_seq_len,
        repeat_layer=args.repeat_layer,
    )
    max_seq_len = lm_config_student.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniLLM-Dist-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
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

    # 初始化学生模型和教师模型
    tokenizer = build_tokenizer(args.tokenizer_dir, trust_remote_code=args.trust_remote_code)
    # ensure vocab matches tokenizer for both student and teacher
    lm_config_student.vocab_size = tokenizer.vocab_size
    model = init_student_model(lm_config_student)
    # teacher 根据模式初始化
    teacher_model = init_teacher_model(lm_config_teacher)

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)
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

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # 若后续引入可学习投影，这里应追加到参数列表。当前 feature_project 无参数，直接优化模型即可。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb, alpha=args.alpha, temperature=args.temperature)
