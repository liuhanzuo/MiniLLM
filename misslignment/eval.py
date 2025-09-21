import argparse
import math
import os
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from MiniLLM.model.tokenizer_utils import build_tokenizer
from MiniLLM.model.model import MiniLLMLM
from MiniLLM.model.LMConfig import LMConfig
from MiniLLM.model.dataset import SFTDataset


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    tok = 0
    for X, Y, loss_mask in loader:
        X, Y, loss_mask = X.to(device), Y.to(device), loss_mask.to(device)
        out = model(X)
        logits = out.logits if hasattr(out, 'logits') else out
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), Y.view(-1), reduction='none'
        ).view(Y.size())
        loss_sum += (loss * loss_mask).sum().item()
        tok += loss_mask.sum().item()
    return loss_sum, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--tokenizer_dir', type=str, default='MiniLLM/model/minillm_tokenizer')
    ap.add_argument('--ckp', type=str, required=True)
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--n_layers', type=int, default=8)
    ap.add_argument('--n_block', type=int, default=None)
    ap.add_argument('--max_seq_len', type=int, default=512)
    ap.add_argument('--use_moe', action='store_true')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch_size', type=int, default=8, help='Global batch size for evaluation')
    ap.add_argument('--dp', action='store_true', help='Use torch.nn.DataParallel across available GPUs (fallback)')
    ap.add_argument('--ddp', action='store_true', help='Use torch.distributed DDP (recommended for multi-GPU)')
    ap.add_argument('--local_rank', type=int, default=-1, help='for torchrun')
    args = ap.parse_args()

    # Setup device and (optional) DDP
    world_size = 1
    rank = 0
    use_ddp = False
    if args.ddp:
        assert torch.cuda.is_available(), 'DDP requires CUDA devices.'
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank if args.local_rank >= 0 else 0))
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f'cuda:{local_rank}')
        use_ddp = True
    else:
        if args.device.startswith('cuda') and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    tokenizer = build_tokenizer(args.tokenizer_dir, trust_remote_code=True)
    lm_conf = LMConfig(dim=args.dim, n_layers=args.n_layers, n_block=args.n_block, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    model = MiniLLMLM(lm_conf)
    # Prefer safe weights-only load when possible
    try:
        state = torch.load(args.ckp, map_location=device, weights_only=True)
    except TypeError:
        # Fallback for older torch versions
        state = torch.load(args.ckp, map_location=device)
    state = {k: v for k, v in state.items() if 'mask' not in k}
    model.load_state_dict(state, strict=False)
    model = model.to(device)

    # Multi-GPU wrappers
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)
    elif args.dp and device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    ds = SFTDataset(args.data, tokenizer, max_length=args.max_seq_len)
    # Sampler/DataLoader
    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
        per_rank_bs = max(1, args.batch_size // world_size)
        loader = DataLoader(ds, batch_size=per_rank_bs, sampler=sampler)
    else:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    loss_sum, tok = evaluate(model, loader, device)
    if use_ddp:
        t = torch.tensor([loss_sum, tok], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        loss_sum, tok = t[0].item(), t[1].item()
    if (not use_ddp) or rank == 0:
        ppl = math.exp(loss_sum / max(1, tok)) if tok > 0 else float('inf')
        print(dict(val_loss=loss_sum / max(1, tok), val_ppl=ppl))

    if use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
