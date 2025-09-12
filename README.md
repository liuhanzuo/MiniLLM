# MiniLLM

## Training scripts layout

All training entrypoints are grouped under `scripts/train/` for easier discovery:

- scripts/train/train_pretrain.py
- scripts/train/train_full_sft.py
- scripts/train/train_lora.py
- scripts/train/train_dpo.py
- scripts/train/train_distillation.py
- scripts/train/train_distill_reason.py

Backward compatibility: the old top-level files still exist as thin forwarders, so existing commands won’t break.

## Run examples

From repo root:

- Pretrain (single GPU):
	python3 scripts/train/train_pretrain.py --data_path ./dataset/pretrain_hq.jsonl
  
  With block-based repeated layers (each block = virtual 8 layers: 0,1,2,3,1,2,3,4):
	python3 scripts/train/train_pretrain.py --data_path ./dataset/pretrain_hq.jsonl --n_block 3
This builds 3 blocks -> total virtual layers = 3*8 = 24. Parameters scale with 5 per block.
  
  DDP example:
	torchrun --nproc_per_node 8 scripts/train/train_pretrain.py --data_path ./dataset/pretrain_hq.jsonl --dim 1024 --batch_size 32 --n_block 6 --use_wandb

- Full SFT (DDP 2 GPUs):
	torchrun --nproc_per_node 2 scripts/train/train_full_sft.py --data_path ./dataset/sft_mini_512.jsonl --ddp

- LoRA SFT:
	python3 scripts/train/train_lora.py --data_path ./dataset/lora_identity.jsonl

- DPO:
	python3 scripts/train/train_dpo.py --data_path ./dataset/dpo.jsonl

- Distillation:
	python3 scripts/train/train_distillation.py --data_path ./dataset/sft_data.jsonl

- Distill Reasoning:
	python3 scripts/train/train_distill_reason.py --data_path ./dataset/r1_mix_1024.jsonl

## Notation
The dataset comes from the open source project Minimind
