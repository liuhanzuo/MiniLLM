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
from typing import List

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


def global_unstructured_magnitude_prune(model: torch.nn.Module, target_sparsity: float, include_embeddings: bool = False):
    """Globally prune smallest-magnitude weights across gathered tensors.

    target_sparsity: fraction of weights to zero out, e.g., 0.9 for 90% sparsity.
    """
    items = gather_pruneable_parameters(model, include_embeddings=include_embeddings)
    # collect absolute values
    tensors = [t.detach().abs().view(-1) for (_m, _n, t) in items]
    all_weights = torch.cat(tensors)
    k = int(all_weights.numel() * target_sparsity)
    if k <= 0:
        print('target sparsity too small, nothing to prune')
        return
    # find global threshold
    thresh, _ = torch.kthvalue(all_weights, k)
    thresh = float(thresh)
    print(f'Global magnitude threshold for pruning: {thresh:.6e} (pruning {k}/{all_weights.numel()} weights)')

    # apply mask in-place
    total_pruned = 0
    total = 0
    for (module, name, tensor) in items:
        w = tensor.data
        mask = (w.abs() > thresh).to(w.dtype)
        pruned = int((mask.numel() - mask.sum()).item())
        total_pruned += pruned
        total += mask.numel()
        w.mul_(mask)
    print(f'Pruned {total_pruned}/{total} weights -> actual sparsity {total_pruned/total:.4f}')


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
            shutil.copytree(p, backup)
            print(f'Created backup of {path} at {backup}')


def main():
    parser = argparse.ArgumentParser(description='Prune Qwen model to target sparsity (global unstructured)')
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='HuggingFace repo id or local path for Qwen/Qwen2.5-7b-Instruct')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save pruned model')
    parser.add_argument('--sparsity', type=float, default=0.9, help='Target sparsity (e.g. 0.9 means 90%% zeros)')
    parser.add_argument('--device', type=str, default='cuda', help='device to load model on (cuda or cpu)')
    parser.add_argument('--include_embeddings', action='store_true', help='Also prune embedding weights')
    parser.add_argument('--save_state_dict', action='store_true', help='Also save a torch state_dict')
    parser.add_argument('--no_backup', action='store_true', help='Do not create backup of output dir if exists')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    print('Loading model (trust_remote_code=True). This may download remote code...')
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, low_cpu_mem_usage=False)
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    except Exception:
        print('Tokenizer load failed or not available, continuing without tokenizer')

    model.to(device)
    model.eval()

    print(f'Model loaded. Applying global unstructured magnitude pruning to reach sparsity={args.sparsity}')
    global_unstructured_magnitude_prune(model, target_sparsity=args.sparsity, include_embeddings=args.include_embeddings)

    # save
    out_dir = args.output_dir
    if Path(out_dir).exists() and not args.no_backup:
        backup_dir(out_dir)

    save_model_and_tokenizer(model, tokenizer, out_dir, save_state_dict=args.save_state_dict)
    print('Pruning finished and model saved.')


if __name__ == '__main__':
    main()
