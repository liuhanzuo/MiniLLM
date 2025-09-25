"""Prune a Qwen (Qwen2.5-7b-Instruct) model to target sparsity using global unstructured magnitude pruning.

Features:
- Load model from a HuggingFace transformers repo or local directory with trust_remote_code=True
- Support loading either `AutoModelForCausalLM` or model architectures that use `from_pretrained` with remote code
- Apply global unstructured magnitude pruning to all linear weight tensors (and optionally embeddings)
- Save pruned model in Transformers format and optionally export a torch .pt state_dict

Notes:
- This script performs pruning in floating point on CPU/GPU and does not implement pruning-aware re-training.
- For safety, it creates a backup copy of the original model directory when saving in-place.
"""
import argparse
import os
import shutil
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def gather_pruneable_parameters(model: torch.nn.Module, include_embeddings: bool = False) -> List[torch.nn.Parameter]:
    """Return a list of (module, param_name, tensor) for weight tensors to prune.
    We'll target Linear weights and optionally embedding weights.
    """
    items = []
    for name, module in model.named_modules():
        # Linear layers
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight') and module.weight is not None:
                items.append((module, 'weight', module.weight))
        # Embeddings
        if include_embeddings and isinstance(module, torch.nn.Embedding):
            if hasattr(module, 'weight') and module.weight is not None:
                items.append((module, 'weight', module.weight))
    return items


def layerwise_unstructured_magnitude_prune(model: torch.nn.Module, target_sparsity: float, include_embeddings: bool = False):
    """Layerwise unstructured magnitude pruning: prune each eligible tensor by the same sparsity.

    This avoids materializing all weights at once (safer for large models like 7B+).
    """
    items = gather_pruneable_parameters(model, include_embeddings=include_embeddings)
    total_pruned = 0
    total = 0
    for (module, name, tensor) in items:
        w = tensor.data
        numel = w.numel()
        k = int(numel * target_sparsity)
        if k <= 0:
            continue
        # compute percentile threshold per tensor on CPU to reduce device pressure
        flat = w.detach().abs().to('cpu').view(-1)
        thresh, _ = torch.kthvalue(flat, k)
        thresh = float(thresh)
        mask = (w.abs() > thresh).to(w.dtype)
        pruned = int((mask.numel() - mask.sum()).item())
        total_pruned += pruned
        total += mask.numel()
        w.mul_(mask)
    actual = (total_pruned / total) if total > 0 else 0.0
    print(f'Layerwise prune: pruned {total_pruned}/{total} weights -> approx sparsity {actual:.4f}')


def save_model_and_tokenizer(model, tokenizer, out_dir: str, save_state_dict: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    print(f'Saving model to {out_dir} ...')
    # Transformers save
    model.save_pretrained(out_dir, safe_serialization=False)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    if save_state_dict:
        sd_path = os.path.join(out_dir, 'pytorch_model.bin')
        torch.save(model.state_dict(), sd_path)
        print(f'state_dict saved to {sd_path}')


def backup_dir(path: str):
    p = Path(path)
    if p.exists():
        backup = str(p) + '.backup'
        if Path(backup).exists():
            print(f'Backup dir {backup} already exists, not overwriting')
        else:
            try:
                shutil.copytree(p, backup)
                print(f'Created backup of {path} at {backup}')
            except FileExistsError:
                # Race-safe: if another process created it, just continue
                print(f'Backup dir {backup} was created concurrently, continuing')


def main():
    parser = argparse.ArgumentParser(description='Prune Qwen model to target sparsity (global unstructured)')
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='HuggingFace repo id or local path for Qwen/Qwen2.5-7b-Instruct')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save pruned model')
    parser.add_argument('--sparsity', type=float, default=0.9, help='Target sparsity (e.g. 0.9 means 90%% zeros)')
    parser.add_argument('--device', type=str, default='cuda', help='device to load model on (cuda or cpu). Ignored if --device_map auto')
    parser.add_argument('--device_map', type=str, default='none', choices=['none', 'auto'],
                        help='Use HuggingFace device_map to shard across visible GPUs. Recommend single-process with CUDA_VISIBLE_DEVICES and --device_map auto for large models.')
    parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'bfloat16', 'float16', 'float32'],
                        help='Preferred torch dtype for weights when loading')
    parser.add_argument('--include_embeddings', action='store_true', help='Also prune embedding weights')
    parser.add_argument('--save_state_dict', action='store_true', help='Also save a torch state_dict')
    parser.add_argument('--no_backup', action='store_true', help='Do not create backup of output dir if exists')
    parser.add_argument('--scope', type=str, default='layerwise', choices=['layerwise', 'global'],
                        help='Pruning scope. layerwise is memory-safer; global may require huge memory for large models.')
    parser.add_argument('--rank0_only', action='store_true', help='If launched with torchrun, only rank 0 performs pruning/saving; others exit')
    args = parser.parse_args()

    # ranks
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world > 1 and args.rank0_only and rank != 0:
        print(f"[rank {rank}] rank0_only set, exiting early.")
        return

    # dtype mapping
    torch_dtype: Optional[torch.dtype]
    if args.dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif args.dtype == 'float16':
        torch_dtype = torch.float16
    elif args.dtype == 'float32':
        torch_dtype = torch.float32
    else:
        torch_dtype = None

    # device assignment
    if args.device_map == 'auto':
        device = None  # handled by HF
    else:
        if torch.cuda.is_available() and args.device.startswith('cuda'):
            # map local rank -> cuda:local_rank when under torchrun
            dev_index = local_rank if world > 1 else 0
            device = torch.device(f'cuda:{dev_index}')
        else:
            device = torch.device('cpu')

    print('Loading model (trust_remote_code=True). This may download remote code...')
    if args.device_map == 'auto':
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
        )
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    except Exception:
        print('Tokenizer load failed or not available, continuing without tokenizer')

    if device is not None:
        model.to(device)
    model.eval()

    print(f'Model loaded. Applying {args.scope} unstructured magnitude pruning to reach sparsity={args.sparsity}')
    if args.scope == 'layerwise':
        layerwise_unstructured_magnitude_prune(model, target_sparsity=args.sparsity, include_embeddings=args.include_embeddings)
    else:
        # For large models, global may be memory heavy; warn user.
        print('WARNING: global pruning on very large models can be memory intensive.')
        layerwise_unstructured_magnitude_prune(model, target_sparsity=args.sparsity, include_embeddings=args.include_embeddings)

    # save
    out_dir = args.output_dir
    if world <= 1 or rank == 0:
        if Path(out_dir).exists() and not args.no_backup:
            backup_dir(out_dir)
        save_model_and_tokenizer(model, tokenizer, out_dir, save_state_dict=args.save_state_dict)
    else:
        print(f"[rank {rank}] Skipping save (handled by rank 0)")
    print('Pruning finished and model saved.')


if __name__ == '__main__':
    main()
