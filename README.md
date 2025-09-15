# MiniLLM
## Install dependencies

```bash
conda install -c conda-forge pyarrow
pip install -r requirements.txt
```

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

From repo root (Windows PowerShell examples use python as launcher):

- Pretrain (single GPU):
Single GPU (dim=512, n_layers=8, batch_size=8):
```powershell
python scripts/train/train_pretrain.py --data_path ./dataset/pretrain_hq.jsonl --dim 512 --n_layers 8 --batch_size 8 --tokenizer_dir ./model/minillm_tokenizer
```
8 GPUs (torchrun, dim=1024, n_block=6, batch_size=32, DDP enabled):
```powershell
torchrun --nproc_per_node 8 scripts/train/train_pretrain.py --data_path ./dataset/pretrain_hq.jsonl --dim 1024 --n_block 6 --batch_size 32 --ddp --tokenizer_dir ./model/minillm_tokenizer
```
With block-based repeated layers (each block = virtual 8 layers: 0,1,2,3,1,2,3,4):
```powershell
python scripts/train/train_pretrain.py --data_path ./dataset/pretrain_hq.jsonl --n_block 3 --tokenizer_dir meta-llama/Llama-3.2-3B --trust_remote_code
```
This builds 3 blocks -> total virtual layers = 3*8 = 24. Parameters scale with 5 per block.
DDP example:
```powershell
torchrun --nproc_per_node 8 scripts/train/train_pretrain.py --data_path ./dataset/pretrain_hq.jsonl --dim 1024 --n_block 6 --batch_size 32 --ddp --use_wandb --tokenizer_dir deepseek-ai/DeepSeek-V3 --trust_remote_code
```

- Full SFT (DDP 2 GPUs):
Single GPU:
```powershell
python scripts/train/train_full_sft.py --data_path ./dataset/sft_mini_512.jsonl --dim 512 --n_layers 8 --batch_size 8 --tokenizer_dir ./model/minillm_tokenizer
```
8 GPUs:
```powershell
torchrun --nproc_per_node 8 scripts/train/train_full_sft.py --data_path ./dataset/sft_mini_512.jsonl --dim 1024 --n_block 6 --batch_size 32 --ddp --tokenizer_dir ./model/minillm_tokenizer
```

- LoRA SFT:
Single GPU:
```powershell
python scripts/train/train_lora.py --data_path ./dataset/lora_identity.jsonl --dim 512 --n_layers 8 --batch_size 8 --tokenizer_dir ./model/minillm_tokenizer
```
8 GPUs:
```powershell
torchrun --nproc_per_node 8 scripts/train/train_lora.py --data_path ./dataset/lora_identity.jsonl --dim 1024 --n_block 6 --batch_size 32 --ddp --tokenizer_dir ./model/minillm_tokenizer
```

- DPO:
Single GPU:
```powershell
python scripts/train/train_dpo.py --data_path ./dataset/dpo.jsonl --dim 512 --n_layers 8 --batch_size 8 --tokenizer_dir ./model/minillm_tokenizer
```
8 GPUs:
```powershell
torchrun --nproc_per_node 8 scripts/train/train_dpo.py --data_path ./dataset/dpo.jsonl --dim 1024 --n_block 6 --batch_size 32 --ddp --tokenizer_dir ./model/minillm_tokenizer
```

- Distillation:
Single GPU:
```powershell
python scripts/train/train_distillation.py --data_path ./dataset/sft_data.jsonl --dim 512 --n_layers 8 --batch_size 8 --tokenizer_dir ./model/minillm_tokenizer
```
8 GPUs:
```powershell
torchrun --nproc_per_node 8 scripts/train/train_distillation.py --data_path ./dataset/sft_data.jsonl --dim 1024 --n_block 6 --batch_size 32 --ddp --tokenizer_dir ./model/minillm_tokenizer
```

- Distill Reasoning:
Single GPU:
```powershell
python scripts/train/train_distill_reason.py --data_path ./dataset/r1_mix_1024.jsonl --dim 512 --n_layers 8 --batch_size 8 --tokenizer_dir ./model/minillm_tokenizer
```
8 GPUs:
```powershell
torchrun --nproc_per_node 8 scripts/train/train_distill_reason.py --data_path ./dataset/r1_mix_1024.jsonl --dim 1024 --n_block 6 --batch_size 32 --ddp --tokenizer_dir ./model/minillm_tokenizer
```

### Serve OpenAI-compatible API

```powershell
python scripts/serve_openai_api.py --tokenizer_dir ./model/minillm_tokenizer
```

### Convert between torch and HF formats

Torch -> HF directory:
```powershell
python scripts/convert_model.py --mode torch2hf --torch_path ..\out\rlhf_512.pth --transformers_path ..\MiniLLM2-Small --tokenizer_dir ./model/minillm_tokenizer
```
HF directory -> Torch state_dict:
```powershell
python scripts/convert_model.py --mode hf2torch --transformers_path ..\MiniLLM2-Small --torch_path ..\out\rlhf_512.pth
```

## Notation
The dataset comes from the open source project Minimind
