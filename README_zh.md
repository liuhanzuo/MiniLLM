# MiniLLM 使用说明（中文）
本项目提供一个轻量级的 LLM 训练/微调/对齐与推理框架，并支持快速切换 Hugging Face 的分词器。
## 安装环境依赖
```bash
conda create -n minillm python=3.9 -y
conda activate minillm
conda install -c conda-forge pyarrow

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
50系显卡用户:
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

## 目录结构与入口脚本

训练入口脚本集中在 `scripts/train/`：
- `scripts/train/train_pretrain.py`（预训练）
- `scripts/train/train_full_sft.py`（全参数 SFT）
- `scripts/train/train_lora.py`（LoRA SFT）
- `scripts/train/train_dpo.py`（DPO 偏好对齐）
- `scripts/train/train_distillation.py`（知识蒸馏）
- `scripts/train/train_distill_reason.py`（推理能力蒸馏/思维链）

其他常用脚本：
- `scripts/serve_openai_api.py`：启动一个 OpenAI API 兼容的服务
- `scripts/convert_model.py`：Torch <-> Transformers 格式互转

辅助：
- `model/tokenizer_utils.py`：通用分词器加载（含 pad_token 兜底）

## 切换 Hugging Face 分词器

所有训练/服务/转换脚本均支持通过 `--tokenizer_dir` 指定分词器来源（本地目录或 HF 仓库名），例如：
- 本地：`--tokenizer_dir ./model/minillm_tokenizer`
- HF 仓库：`--tokenizer_dir deepseek-ai/DeepSeek-V3 --trust_remote_code`

分词器加载规则：
- 若分词器无 `pad_token`，优先使用 `eos_token`，否则自动新增 `'<pad>'`；
- 每次构建模型前会将 `lm_config.vocab_size` 与 `tokenizer.vocab_size` 对齐；
- 加载旧权重时使用 `strict=False`，允许在词表变化时跳过不匹配的 `embedding/lm_head` 权重。

## 运行示例（Windows PowerShell）

以下命令均默认使用 `--tokenizer_dir ./model/minillm_tokenizer`，可按需替换为其他 HF 分词器。

### 预训练（Pretrain）
- 单卡（dim=512, n_layers=8, batch_size=8）：
```powershell
python scripts/train/train_pretrain.py --data_path ./dataset/pretrain_hq.jsonl --dim 512 --n_layers 8 --batch_size 8 --tokenizer_dir ./model/minillm_tokenizer
```
- 8 卡（torchrun，dim=1024, n_block=6, batch_size=32，启用 DDP）：
```powershell
torchrun --nproc_per_node 8 scripts/train/train_pretrain.py --data_path ./dataset/pretrain_hq.jsonl --dim 1024 --n_block 6 --batch_size 32 --ddp --tokenizer_dir ./model/minillm_tokenizer
```

### 全参数 SFT（Full SFT）
- 单卡：
```powershell
python scripts/train/train_full_sft.py --data_path ./dataset/sft_mini_512.jsonl --dim 512 --n_layers 8 --batch_size 8 --tokenizer_dir ./model/minillm_tokenizer
```
- 8 卡：
```powershell
torchrun --nproc_per_node 8 scripts/train/train_full_sft.py --data_path ./dataset/sft_mini_512.jsonl --dim 1024 --n_block 6 --batch_size 32 --ddp --tokenizer_dir ./model/minillm_tokenizer
```

### LoRA SFT
- 单卡：
```powershell
python scripts/train/train_lora.py --data_path ./dataset/lora_identity.jsonl --dim 512 --n_layers 8 --batch_size 8 --tokenizer_dir ./model/minillm_tokenizer
```
- 8 卡：
```powershell
torchrun --nproc_per_node 8 scripts/train/train_lora.py --data_path ./dataset/lora_identity.jsonl --dim 1024 --n_block 6 --batch_size 32 --ddp --tokenizer_dir ./model/minillm_tokenizer
```

### DPO 偏好对齐
- 单卡：
```powershell
python scripts/train/train_dpo.py --data_path ./dataset/dpo.jsonl --dim 512 --n_layers 8 --batch_size 8 --tokenizer_dir ./model/minillm_tokenizer
```
- 8 卡：
```powershell
torchrun --nproc_per_node 8 scripts/train/train_dpo.py --data_path ./dataset/dpo.jsonl --dim 1024 --n_block 6 --batch_size 32 --ddp --tokenizer_dir ./model/minillm_tokenizer
```

### 知识蒸馏（Distillation）
- 单卡：
```powershell
python scripts/train/train_distillation.py   --data_path ./dataset/sft_1024.jsonl   --batch_size 32   --epochs 1   --use_wandb   --distillation_mode logit   --alpha 0.0   --temperature 1.0   --student_dim 512   --student_layer 8   --teacher_dim 1024   --teacher_block 16   --max_seq_len 1024   --teacher_ckpt ./out/rlhf_1024.pth   --out_dir ./dist/ --student_ckpt ./dist/full_dist_512.pth
```
- 8 卡：
```powershell
torchrun --nproc_per_node 8 scripts/train/train_distillation.py   --data_path ./dataset/sft_1024.jsonl   --batch_size 64   --epochs 1   --use_wandb   --distillation_mode logit   --alpha 0.0   --temperature 1.0   --student_dim 512   --student_block 2   --teacher_dim 1024   --teacher_block 6   --max_seq_len 1024   --teacher_ckpt ./out/rlhf_1024.pth   --out_dir ./dist/ --student_ckpt ./dist/full_dist_512.pth
```

### 推理能力蒸馏（Reasoning Distill）
- 单卡：
```powershell
python scripts/train/train_distill_reason.py --data_path ./dataset/r1_mix_1024.jsonl --dim 512 --n_layers 8 --batch_size 8 --tokenizer_dir ./model/minillm_tokenizer
```
- 8 卡：
```powershell
torchrun --nproc_per_node 8 scripts/train/train_distill_reason.py --data_path ./dataset/r1_mix_1024.jsonl --dim 1024 --n_block 6 --batch_size 32 --ddp --tokenizer_dir ./model/minillm_tokenizer
```

## 启动 OpenAI API 兼容服务
```powershell
python scripts/serve_openai_api.py --tokenizer_dir ./model/minillm_tokenizer
```

## 模型格式转换（Torch <-> Transformers）
- Torch -> Transformers 目录：
```powershell
python scripts/convert_model.py --mode torch2hf --torch_path ..\out\rlhf_512.pth --transformers_path ..\MiniLLM2-Small --tokenizer_dir ./model/minillm_tokenizer
```
- Transformers 目录 -> Torch：
```powershell
python scripts/convert_model.py --mode hf2torch --transformers_path ..\MiniLLM2-Small --torch_path ..\out\rlhf_512.pth
```

## 生成
```python
python eval_model.py \
--dim 你之前填入的dim \
--n_layers 你之前填入的n_layers \
--max_seq_len 期望的模型回答最大长度 \
--model_mode 0/1/2
```

## 注意事项
- 若更换分词器并加载旧权重，`embedding/lm_head` 维度变化导致的冲突会被自动跳过（`strict=False`），其余层正常加载；新增的词表行将随机初始化。
- 若你的 SFT/DPO/Reasoning 数据依赖特定模板或特殊 token，请确保新分词器的 `chat_template` 与特殊标记一致。
- Windows PowerShell 下命令使用 `python` 调用；Linux/Mac 可用 `python3`。

## 数据说明
本仓库的数据集引用自开源社区minimind， 可以从这里直接下载数据集：https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
或者运行以下命令：
```
pip install modelscope
pip install datasets
modelscope download --dataset gongjy/minimind_dataset --local_dir ./data
```
这会将对应的数据集下载到当前目录的data文件夹下。
