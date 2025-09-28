## MiniLLM 训练/蒸馏/剪枝一页通（Runbook）

本手册汇总仓库脚本在预训练、SFT、RLHF（DPO）、蒸馏和剪枝阶段的关键参数、输入输出与可直接运行的命令。默认输出路径均位于 `./out/` 下。

### 总览表

| 阶段 | 脚本 | 关键输入 | 产物与默认保存地址 | 关键参数要点 |
|---|---|---|---|---|
| 预训练 Pretrain | `scripts/train/train_pretrain.py` | 数据集：`--data_path`；分词器：`--tokenizer_dir` | `./out/pretrain_{dim}[_moe].pth` | dim/n_layers/max_seq_len；AMP dtype；DDP/accumulation；每 `--save_interval` 步保存 |
| 全参数 SFT | `scripts/train/train_full_sft.py` | - MiniLLM：可接 `--ckp ./out/pretrain_*.pth` 作为初始化；<br>- HF：`--model_type hf --model_name_or_path <HF模型>` | - MiniLLM：`./out/full_sft_{dim}[_moe].pth`；<br>- HF：加 `--save_hf` 保存到 `./out/hf_finetuned/` | 选择 `--model_type mini|hf`；HF 推荐 `--trust_remote_code`、可 `--gradient_checkpointing`；可接 DeepSpeed 配置 |
| RLHF (DPO) | `scripts/train/train_dpo.py` | 需先有 SFT 产物：读默认 `./out/full_sft_{dim}.pth` | `./out/rlhf_{dim}[_moe].pth` | 数据集：`./dataset/dpo.jsonl`；学习率默认极小（1e-8）；max_seq_len 默认 3000 |
| 蒸馏 Distillation | `scripts/train/train_distillation.py` | 学生：`--student_*` 结构与（可选）`--student_ckpt`；教师：MiniLLM（默认）或 HF（新增） | `./out/full_dist_{student_dim}[_moe].pth` | 模式：`--distillation_mode {logit,response,feature,self}`；HF 教师目前稳妥支持 logit/response；feature 对齐已内置兼容但依赖模型结构 |
| 剪枝 Prune (HF) | `scripts/prune_qwen.py` | HF 模型：`--model_name_or_path <repo or path>` | `--output_dir` 指定目录（Transformers 标准结构），可额外 `pytorch_model.bin` | 支持 `device_map auto`；`--sparsity` 目标稀疏度；可 `--include_embeddings` |

注：上述默认 out 路径来自脚本内部命名：`pretrain_` / `full_sft_` / `rlhf_` / `full_dist_`。

另外，仓库提供了双机8卡（2n8g）统一启动脚本：`scripts/launch/pretrain_2n8g.sh`、`sft_2n8g.sh`、`dpo_2n8g.sh`、`distill_2n8g.sh`、`prune_2n8g.sh`，它们共用 `scripts/launch/deepspeed_2n8g.sh` 做分布式启动。

---

## 双机8卡（2n8g）统一启动

以下脚本统一使用 `scripts/launch/deepspeed_2n8g.sh` 多机多卡启动器（DeepSpeed CLI）。在 MASTER 节点运行，要求免密 SSH 与可用的 hostfile。

基础环境变量（运行前设置）：

- HOSTFILE：主机列表文件，默认示例 `scripts/launch/hostfile.example`
- MASTER_ADDR：MASTER 节点 IP，例如 `10.0.0.1`
- MASTER_PORT：MASTER 端口，例如 `29530`

通用示例：

1) 预训练（MiniLLM）

- 脚本：`bash scripts/launch/pretrain_2n8g.sh`
- 常用可覆盖环境变量：
  - DIM（默认 1024）、LAYERS（24）、MAXLEN（2048）
  - BATCH（8，单卡批量）、ACC（8，梯度累计）、EPOCHS（3）、LR（5e-4）
  - DATA_PATH（`./dataset/pretrain_hq.jsonl`）、TOKENIZER_DIR、OUT_DIR、DTYPE（bfloat16）

2) 全参 SFT

- 脚本：`bash scripts/launch/sft_2n8g.sh`
- 通过 `MODEL_TYPE` 选择：`hf`（默认）或 `mini`
- HF 路线常用变量：
  - MODEL_ID（默认 `Qwen/Qwen2.5-7B-Instruct`）、TOK_DIR（默认同 MODEL_ID）
  - BATCH（1）、ACC（16）、EPOCHS（1）、LR（5e-6）、MAXLEN（4096）
  - DS_CFG（默认 `scripts/launch/ds_zero3_30b.json`）
  - 可选预下载：`PRELOAD=1 MODEL_ID=xxx HF_REVISION=main MODEL_LOCAL_DIR=./hf/xxx`
- MiniLLM 路线常用变量：
  - DIM（1024）、LAYERS（24）、MAXLEN（2048）
  - BATCH（8）、ACC（4）、EPOCHS（2）、LR（5e-5）
  - CKP（默认 `./out/pretrain_${DIM}.pth`）、TOK_DIR

3) RLHF（DPO）

- 脚本：`bash scripts/launch/dpo_2n8g.sh`
- 常用变量：DIM、LAYERS、MAXLEN（默认 3000）、BATCH（2）、ACC（8）、EPOCHS（2）、LR（1e-8）、TOK_DIR、OUT_DIR

4) 蒸馏（MiniLLM↔HF 教师）

- 脚本：`bash scripts/launch/distill_2n8g.sh`
- 学生侧变量：S_DIM（1024）、S_LAYERS（24）、MAXLEN（4096）、BATCH（2）、ACC（8）、EPOCHS（2）、LR（5e-6）、TOK_DIR、OUT_DIR
- 教师选择：`TEACHER_TYPE=hf|mini`
  - 若 hf：`T_MODEL_ID`（默认 `Qwen/Qwen2.5-7B-Instruct`），可配合 `--trust_remote_code`
  - 若 mini：`T_DIM`、`T_LAYERS`、`T_CKP`（默认 `./out/full_sft_${T_DIM}.pth`）

5) 剪枝（HF 模型）

- 脚本：`bash scripts/launch/prune_2n8g.sh`
- 常用变量：`MODEL_ID`、`OUT_DIR`、`SPARSITY`（默认 0.9）、`EXTRA_ARGS`（默认 `--device_map auto --dtype bfloat16 --include_embeddings --save_state_dict --rank0_only`）

## 分阶段命令示例

以下命令均为示例，按需替换 dim/层数/数据路径/GPU 数等。

### 1) 预训练 Pretrain（MiniLLM）

- 单机单卡

```bash
python scripts/train/train_pretrain.py \
  --out_dir ./out \
  --epochs 3 --batch_size 64 --learning_rate 5e-4 \
  --dim 1024 --n_layers 24 --max_seq_len 2048 \
  --data_path ./dataset/pretrain_hq.jsonl \
  --tokenizer_dir ./model/minillm_tokenizer \
  --dtype bfloat16
```

- 多卡（torchrun）

```bash
torchrun --nproc_per_node 8 scripts/train/train_pretrain.py \
  --ddp --device cuda \
  --out_dir ./out \
  --epochs 3 --batch_size 64 --learning_rate 5e-4 \
  --dim 1024 --n_layers 24 --max_seq_len 2048 \
  --data_path ./dataset/pretrain_hq.jsonl \
  --tokenizer_dir ./model/minillm_tokenizer \
  --dtype bfloat16
```

产物：`./out/pretrain_1024.pth`

### 2) 全参数 SFT

- SFT（MiniLLM，从预训练权重接着训）

```bash
python scripts/train/train_full_sft.py \
  --model_type mini \
  --ckp ./out/pretrain_1024.pth \
  --out_dir ./out \
  --epochs 2 --batch_size 32 --learning_rate 5e-5 \
  --dim 1024 --n_layers 24 --max_seq_len 2048 \
  --data_path ./dataset/sft_mini_512.jsonl \
  --tokenizer_dir ./model/minillm_tokenizer \
  --dtype bfloat16
```

产物：`./out/full_sft_1024.pth`

- SFT（HF 模型：Qwen/Qwen2.5-7B-Instruct 或 DeepSeek-R1 系列）

```bash
deepspeed --num_gpus 8 scripts/train/train_full_sft.py \
  --model_type hf \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --save_hf \
  --deepspeed scripts/launch/ds_zero3_30b.json \
  --out_dir ./out \
  --epochs 1 --batch_size 4 --learning_rate 5e-6 \
  --max_seq_len 4096 \
  --data_path ./dataset/sft_mini_512.jsonl \
  --tokenizer_dir Qwen/Qwen2.5-7B-Instruct \
  --trust_remote_code --dtype bfloat16 --gradient_checkpointing
```

产物：`./out/hf_finetuned/`（Transformers 标准结构）

同理可替换 `--model_name_or_path` 为：

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

### 3) RLHF（DPO，基于你的 SFT 产物）

```bash
python scripts/train/train_dpo.py \
  --out_dir ./out \
  --epochs 2 --batch_size 8 --learning_rate 1e-8 \
  --dim 1024 --n_layers 24 --max_seq_len 3000 \
  --data_path ./dataset/dpo.jsonl \
  --tokenizer_dir ./model/minillm_tokenizer \
  --dtype bfloat16
```

产物：`./out/rlhf_1024.pth`（默认从 `./out/full_sft_1024.pth` 初始化）

### 4) 蒸馏 Distillation

- MiniLLM → MiniLLM（你现有 SFT 产物做 teacher）

```bash
python scripts/train/train_distillation.py \
  --out_dir ./out \
  --epochs 6 --batch_size 32 --learning_rate 5e-6 \
  --data_path ./dataset/sft_data.jsonl \
  --tokenizer_dir ./model/minillm_tokenizer \
  --student_dim 512 --student_layers 8 \
  --teacher_dim 768 --teacher_layers 16 \
  --teacher_ckpt ./out/full_sft_768.pth \
  --distillation_mode logit --temperature 2.0 --alpha 0.2 \
  --dtype bfloat16
```

产物：`./out/full_dist_512.pth`

- HF 教师（Qwen/Qwen2.5-7B-Instruct 或 DeepSeek-R1…）→ MiniLLM 学生（新增支持）

```bash
python scripts/train/train_distillation.py \
  --out_dir ./out \
  --epochs 2 --batch_size 8 --learning_rate 5e-6 \
  --data_path ./dataset/sft_data.jsonl \
  --tokenizer_dir ./model/minillm_tokenizer \
  --student_dim 1024 --student_layers 24 --max_seq_len 4096 \
  --teacher_model_type hf \
  --teacher_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --distillation_mode logit --temperature 2.0 --alpha 0.3 \
  --dtype bfloat16 --trust_remote_code
```

将 `--teacher_model_name_or_path` 替换为：

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

> 说明：HF 教师已稳妥支持 logit/response 蒸馏；feature 蒸馏已内置通用兼容实现（依赖输出 hidden_states），如遇架构差异将自动回退至 logit 蒸馏并给出提示。

### 5) 剪枝 Prune（HF 模型）

- Qwen2.5-7B-Instruct

```bash
python scripts/prune_qwen.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --output_dir ./out/pruned-qwen2.5-7b-s0.9 \
  --sparsity 0.9 \
  --device_map auto --dtype bfloat16 \
  --include_embeddings --save_state_dict
```

- DeepSeek-R1-Distill-Qwen-14B

```bash
python scripts/prune_qwen.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --output_dir ./out/pruned-r1-qwen14b-s0.9 \
  --sparsity 0.9 \
  --device_map auto --dtype bfloat16 \
  --include_embeddings --save_state_dict
```

- DeepSeek-R1-Distill-Llama-8B

```bash
python scripts/prune_qwen.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --output_dir ./out/pruned-r1-llama8b-s0.9 \
  --sparsity 0.9 \
  --device_map auto --dtype bfloat16 \
  --include_embeddings --save_state_dict
```

多机 2n8g 启动（适用于大模型）：

- 使用脚本：`bash scripts/launch/prune_2n8g.sh`
- 常用环境变量：
  - `MODEL_ID=Qwen/Qwen2.5-7B-Instruct`（或其他 HF 模型）
  - `OUT_DIR=./out/pruned-qwen2.5-7b-s0.9`、`SPARSITY=0.9`
  - `EXTRA_ARGS="--device_map auto --dtype bfloat16 --include_embeddings --save_state_dict --rank0_only"`
  - 需先设置 `HOSTFILE`、`MASTER_ADDR`、`MASTER_PORT`

---

## out/ 文件名速查

- 预训练：`./out/pretrain_{dim}[_moe].pth`
- 全参 SFT（MiniLLM）：`./out/full_sft_{dim}[_moe].pth`
- RLHF（DPO）：`./out/rlhf_{dim}[_moe].pth`
- 蒸馏（MiniLLM→MiniLLM）：`./out/full_dist_{student_dim}[_moe].pth`
- SFT（HF，带 `--save_hf`）：`./out/hf_finetuned/` 目录（Transformers 标准结构）
- 剪枝（HF）：你指定的 `--output_dir`（Transformers 标准结构，若 `--save_state_dict` 还会有 `pytorch_model.bin`）

---

## 备注与建议

- AMP/dtype：预训练和 SFT 推荐 `bfloat16`；CPU 自动关闭 AMP。HF 端可开启 `--gradient_checkpointing` 降显存。
- 分词器：MiniLLM 路径一般用 `./model/minillm_tokenizer`；HF SFT 时可将 `--tokenizer_dir` 指向同一个 HF 模型目录，脚本会尝试 `resize_token_embeddings` 对齐词表。
- 分布式：SFT 的 HF 路线建议使用 DeepSpeed ZeRO-2/3（仓内有 `scripts/launch/ds_zero2_30b.json`、`ds_zero3_30b.json`）。
- 蒸馏模式：
  - logit：温度 1.5~4.0 常见，`alpha` 决定 CE 与 KD 比例；
  - response：teacher argmax 作为硬标签；
  - feature：对齐隐藏态；维度不一致时本仓提供了无学习线性投影近似；
  - self：EMA 自蒸馏，无需显式教师权重。

---

## 逐任务参数与保存路径表格

下面为每个训练/剪枝任务的完整参数与保存路径说明，参数取自脚本默认值并补充说明。

### 1) 预训练 Pretrain（scripts/train/train_pretrain.py）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| --out_dir | out | 输出根目录；脚本内部还会设置 `args.save_dir = os.path.join(args.out_dir)` |
| --epochs | 1 | 训练轮次 |
| --batch_size | 32 | 每步 batch 大小 |
| --learning_rate | 5e-4 | 余弦退火 + warmup（warmup_iters=0 时即纯余弦）|
| --device | auto | 默认 cuda:0（若可用），否则 cpu |
| --dtype | bfloat16 | AMP 精度（cpu 自动禁用 AMP）|
| --use_wandb | False | 启用 wandb 日志 |
| --wandb_project | MiniLLM-Pretrain | wandb 项目名 |
| --num_workers | 1 | DataLoader workers |
| --ddp | False | 是否使用 DDP（需 torchrun 启动）|
| --accumulation_steps | 8 | 梯度累计步数 |
| --grad_clip | 1.0 | 梯度裁剪阈值 |
| --warmup_iters | 0 | warmup 步数 |
| --log_interval | 100 | 日志间隔（步）|
| --save_interval | 100 | 保存间隔（步）|
| --local_rank | -1 | DDP 内部参数（torchrun 传入）|
| --dim | 512 | 隐藏维度（MiniLLM）|
| --n_layers | 8 | 层数（MiniLLM）|
| --n_block | None | 分块共享模式，未使用则按 n_layers |
| --max_seq_len | 512 | 最大序列长度 |
| --use_moe | False | 是否使用 MoE（布尔型）|
| --repeat_layer | False | 是否启用层复用模式（0,1,2,3,1,2,3,4）|
| --data_path | ./dataset/pretrain_hq.jsonl | 预训练数据路径 |
| --tokenizer_dir | ./model/minillm_tokenizer | 分词器目录 |
| --trust_remote_code | False | 信任远程代码（Tokenizer）|

保存路径与文件名：

- 周期性保存：`{out_dir}/pretrain_{dim}[_moe].pth`

---

### 2) 全参数 SFT（scripts/train/train_full_sft.py）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| --out_dir | out | 输出根目录；`args.save_dir = os.path.join(args.out_dir)` |
| --epochs | 1 | 训练轮次 |
| --batch_size | 32 | 每步 batch 大小 |
| --learning_rate | 5e-5 | 学习率 |
| --device | auto | 默认 cuda:0（若可用），否则 cpu |
| --dtype | bfloat16 | AMP 精度（可通过环境变量 DISABLE_AMP=1 关闭）|
| --use_wandb | False | 启用 wandb |
| --wandb_project | MiniLLM-Full-SFT | wandb 项目名 |
| --num_workers | 1 | DataLoader workers |
| --ddp | False | DDP 开关（非 DeepSpeed 情况下生效）|
| --accumulation_steps | 1 | 梯度累计步数（DeepSpeed 时内部管理）|
| --grad_clip | 1.0 | 梯度裁剪阈值 |
| --warmup_iters | 0 | warmup 步数 |
| --log_interval | 100 | 日志间隔（步）|
| --save_interval | 100 | 保存间隔（步）|
| --local_rank | -1 | DDP 参数 |
| --model_type | mini | 选择 `mini` 或 `hf` |
| --model_name_or_path | None | 当 `--model_type hf` 时指定 HF 模型，如 `Qwen/Qwen2.5-7B-Instruct` |
| --hf_revision | None | HF 分支/commit |
| --dim | 512 | MiniLLM 结构参数 |
| --n_layers | 8 | 同上 |
| --n_block | None | 同上 |
| --max_seq_len | 512 | 同上 |
| --use_moe | False | 同上 |
| --repeat_layer | False | 同上 |
| --data_path | ./dataset/sft_mini_512.jsonl | SFT 数据路径 |
| --tokenizer_dir | ./model/minillm_tokenizer | 分词器目录（HF 可改为同模型目录）|
| --trust_remote_code | False | 信任远程代码（HF）|
| --ckp | None | MiniLLM 初始权重（常用于接 Pretrain）|
| --save_hf | False | HF 路线训练后保存 Transformers 目录 |
| --deepspeed | None | DeepSpeed 配置（JSON 路径或对象）|
| --gradient_checkpointing | False | HF 模型开启梯度检查点 |

保存路径与文件名：

- 周期性保存（MiniLLM）：`{out_dir}/full_sft_{dim}[_moe].pth`
- 训练完成后：
  - MiniLLM：`{out_dir}/full_sft_{dim}[_moe].pth`
  - HF（加 `--save_hf`）：`{out_dir}/hf_finetuned/` 目录（含 config/model/tokenizer）

---

### 3) RLHF（DPO，scripts/train/train_dpo.py）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| --out_dir | out | 输出根目录 |
| --epochs | 2 | 训练轮次 |
| --batch_size | 8 | 每步 batch 大小（DPO 通常较小）|
| --learning_rate | 1e-8 | 极小学习率（避免遗忘）|
| --device | auto | 默认 cuda:0（若可用）|
| --dtype | bfloat16 | AMP 精度 |
| --use_wandb | False | 启用 wandb |
| --wandb_project | MiniLLM-RLHF-SFT | 项目名 |
| --num_workers | 1 | DataLoader workers |
| --ddp | False | DDP 开关 |
| --accumulation_steps | 1 | 梯度累计步数 |
| --grad_clip | 1.0 | 梯度裁剪 |
| --warmup_iters | 0 | warmup 步数 |
| --log_interval | 100 | 日志间隔（步）|
| --save_interval | 100 | 保存间隔（步）|
| --local_rank | -1 | DDP 参数 |
| --dim | 512 | 学生/基座结构（MiniLLM）|
| --n_layers | 8 | 层数 |
| --n_block | None | 分块 |
| --max_seq_len | 3000 | 最大长度（较长对齐场景）|
| --use_moe | False | MoE |
| --repeat_layer | False | 层复用 |
| --data_path | ./dataset/dpo.jsonl | DPO 数据路径（成对 chosen/rejected）|
| --tokenizer_dir | ./model/minillm_tokenizer | 分词器目录 |
| --trust_remote_code | False | |

保存路径与文件名：

- 周期性保存：`{out_dir}/rlhf_{dim}[_moe].pth`
- 初始化默认从：`{out_dir}/full_sft_{dim}[_moe].pth` 加载 SFT 权重

---

### 4) 蒸馏 Distillation（scripts/train/train_distillation.py）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| --out_dir | out | 输出根目录 |
| --epochs | 6 | 训练轮次 |
| --batch_size | 32 | batch 大小 |
| --learning_rate | 5e-6 | 学习率 |
| --device | auto | 设备 |
| --dtype | bfloat16 | AMP 精度 |
| --use_wandb | False | 启用 wandb |
| --wandb_project | MiniLLM-Full-SFT | 项目名 |
| --num_workers | 1 | workers |
| --ddp | False | DDP 开关 |
| --accumulation_steps | 1 | 梯度累计 |
| --grad_clip | 1.0 | 梯度裁剪 |
| --warmup_iters | 0 | warmup 步数 |
| --log_interval | 100 | 日志间隔（步）|
| --save_interval | 100 | 保存间隔（步）|
| --local_rank | -1 | DDP 参数 |
| --data_path | ./dataset/sft_data.jsonl | 蒸馏用数据（通常 SFT 数据）|
| --tokenizer_dir | ./model/minillm_tokenizer | 分词器目录 |
| --trust_remote_code | False | （HF 教师时建议开启）|
| --student_dim | 512 | 学生隐藏维度 |
| --student_layers | 8 | 学生层数 |
| --student_block | None | 学生块数 |
| --teacher_dim | 768 | 教师隐藏维度（MiniLLM 教师路径）|
| --teacher_layers | 16 | 教师层数 |
| --teacher_block | None | 教师块数 |
| --max_seq_len | 512 | 最大长度（同时作用于学生与教师的构造）|
| --student_ckpt | None | 学生初始化 checkpoint（默认找 `./out/full_sft_{student_dim}.pth`）|
| --teacher_ckpt | None | 教师 checkpoint（默认找 `./out/full_sft_{teacher_dim}.pth`）|
| --student_random_init | False | 强制学生随机初始化 |
| --teacher_model_type | mini | `mini` 或 `hf`（新增）|
| --teacher_model_name_or_path | None | 当 `hf` 时必填，如 `Qwen/Qwen2.5-7B-Instruct`、`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`、`deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| --distillation_mode | logit | `logit`/`response`/`feature`/`self` |
| --alpha | 0.0 | 总损失 = `alpha*CE + (1-alpha)*Distill` |
| --temperature | 1.0 | logit 蒸馏温度 |
| --feature_layers | 4 | 特征蒸馏采样层数（对 HF 教师尝试使用 hidden_states）|
| --ema_decay | 0.999 | 自蒸馏 EMA 衰减 |
| --repeat_layer | False | 结构复用 |

保存路径与文件名：

- 周期性保存：`{out_dir}/full_dist_{student_dim}[_moe].pth`

HF 教师适配说明：

- logit/response 模式：直接使用教师 logits，对齐学生 vocab 尺寸（截断至学生 vocab）。
- feature 模式：尝试 `output_hidden_states=True` 抓取教师中间层；若不兼容将自动回退到 logit 蒸馏并打印提示。

---

### 5) 剪枝 Prune（scripts/prune_qwen.py）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| --model_name_or_path | 必填 | HF 模型 repo 或本地路径：如 `Qwen/Qwen2.5-7B-Instruct`、`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`、`deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| --output_dir | 必填 | 剪枝后输出目录（Transformers 标准结构）|
| --sparsity | 0.9 | 目标稀疏度（全局/分层）|
| --device | cuda | 当 `--device_map none` 时用于整机加载的设备 |
| --device_map | none | `none` 或 `auto`（推荐 `auto` 做多卡分片）|
| --dtype | auto | `auto`/`bfloat16`/`float16`/`float32`（加载权重 dtype 偏好）|
| --include_embeddings | False | 是否包含 Embedding 层剪枝 |
| --save_state_dict | False | 另存 `pytorch_model.bin` |
| --no_backup | False | 若输出目录存在，是否跳过 `.backup` 备份 |
| --scope | layerwise | `layerwise`（逐层）或 `global`（全局；大模型内存压力较大）|
| --rank0_only | False | 分布式环境下仅 rank 0 执行剪枝与保存 |

保存路径与文件名：

- 输出目录：`--output_dir`（包含 HF 标准文件，如 config.json、model.safetensors 等）
- 可选：当 `--save_state_dict` 时，额外输出 `{output_dir}/pytorch_model.bin`
- 保护：若输出目录已存在且未设置 `--no_backup`，会创建 `{output_dir}.backup` 备份

---

## 按模型清单：全阶段启动与权重路径

下面按“模型维度/家族”汇总从数据到产物的可执行指令与“权重地址”，覆盖能由本仓脚本直接完成的阶段；若当前脚本不支持（比如 HF 直接做 DPO），我会给出等价替代方案。

### A. MiniLLM（示例：1024d × 24L）

- 预训练 → 产物：`./out/pretrain_1024.pth`

```bash
python scripts/train/train_pretrain.py \
  --out_dir ./out \
  --epochs 3 --batch_size 64 --learning_rate 5e-4 \
  --dim 1024 --n_layers 24 --max_seq_len 2048 \
  --data_path ./dataset/pretrain_hq.jsonl \
  --tokenizer_dir ./model/minillm_tokenizer \
  --dtype bfloat16
```

- 全参 SFT（接预训练）→ 产物：`./out/full_sft_1024.pth`

```bash
python scripts/train/train_full_sft.py \
  --model_type mini \
  --ckp ./out/pretrain_1024.pth \
  --out_dir ./out \
  --epochs 2 --batch_size 32 --learning_rate 5e-5 \
  --dim 1024 --n_layers 24 --max_seq_len 2048 \
  --data_path ./dataset/sft_mini_512.jsonl \
  --tokenizer_dir ./model/minillm_tokenizer \
  --dtype bfloat16
```

- RLHF（DPO，接 SFT）→ 产物：`./out/rlhf_1024.pth`
  - 初始化权重（自动）：`./out/full_sft_1024.pth`

```bash
python scripts/train/train_dpo.py \
  --out_dir ./out \
  --epochs 2 --batch_size 8 --learning_rate 1e-8 \
  --dim 1024 --n_layers 24 --max_seq_len 3000 \
  --data_path ./dataset/dpo.jsonl \
  --tokenizer_dir ./model/minillm_tokenizer \
  --dtype bfloat16
```

- 蒸馏（MiniLLM→MiniLLM：Teacher 1024d/24L → Student 512d/8L）→ 产物：`./out/full_dist_512.pth`
  - 教师权重：`./out/full_sft_1024.pth`

```bash
python scripts/train/train_distillation.py \
  --out_dir ./out \
  --epochs 4 --batch_size 32 --learning_rate 5e-6 \
  --data_path ./dataset/sft_data.jsonl \
  --tokenizer_dir ./model/minillm_tokenizer \
  --student_dim 512 --student_layers 8 --max_seq_len 2048 \
  --teacher_dim 1024 --teacher_layers 24 \
  --teacher_ckpt ./out/full_sft_1024.pth \
  --distillation_mode logit --temperature 2.0 --alpha 0.3 \
  --dtype bfloat16
```

- 剪枝：本仓剪枝脚本针对 HF 模型；MiniLLM 暂无直接剪枝脚本（可另写/迁移到 HF 格式后再剪枝）。

权重地址一览（MiniLLM-1024d）：

- 预训练：`./out/pretrain_1024.pth`
- SFT：`./out/full_sft_1024.pth`
- RLHF：`./out/rlhf_1024.pth`
- Distill（student=512d）：`./out/full_dist_512.pth`

---

### B. Qwen/Qwen2.5-7B-Instruct（HF）

- SFT（HF 原始权重 → HF 微调目录）→ 产物：`./out/hf_finetuned/`
  - 初始教师/基座：`Qwen/Qwen2.5-7B-Instruct`

```bash
deepspeed --num_gpus 8 scripts/train/train_full_sft.py \
  --model_type hf \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --save_hf \
  --deepspeed scripts/launch/ds_zero3_30b.json \
  --out_dir ./out \
  --epochs 1 --batch_size 4 --learning_rate 5e-6 \
  --max_seq_len 4096 \
  --data_path ./dataset/sft_mini_512.jsonl \
  --tokenizer_dir Qwen/Qwen2.5-7B-Instruct \
  --trust_remote_code --dtype bfloat16 --gradient_checkpointing
```

- 蒸馏（HF 教师 → MiniLLM 学生）→ 产物：`./out/full_dist_{student_dim}.pth`
  - 教师权重：`Qwen/Qwen2.5-7B-Instruct`（或上一步 SFT 后目录 `./out/hf_finetuned/`）

```bash
python scripts/train/train_distillation.py \
  --out_dir ./out \
  --epochs 2 --batch_size 8 --learning_rate 5e-6 \
  --data_path ./dataset/sft_data.jsonl \
  --tokenizer_dir ./model/minillm_tokenizer \
  --student_dim 1024 --student_layers 24 --max_seq_len 4096 \
  --teacher_model_type hf \
  --teacher_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --distillation_mode logit --temperature 2.0 --alpha 0.3 \
  --dtype bfloat16 --trust_remote_code
```

- 剪枝（HF）→ 产物目录：`./out/pruned-qwen2.5-7b-s0.9`

```bash
python scripts/prune_qwen.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --output_dir ./out/pruned-qwen2.5-7b-s0.9 \
  --sparsity 0.9 \
  --device_map auto --dtype bfloat16 \
  --include_embeddings --save_state_dict
```

- RLHF：当前仓脚本不直接支持 HF 结构做 DPO；替代方案：
  1) 将 Qwen 作为教师蒸馏到 MiniLLM 学生，再用本仓 `train_dpo.py` 做 DPO；
  2) 或使用 HuggingFace TRL 在 HF 上做 DPO（不在本仓脚本范围）。

权重地址一览（Qwen2.5-7B-Instruct）：

- HF 原始：`Qwen/Qwen2.5-7B-Instruct`
- HF-SFT 输出：`./out/hf_finetuned/`
- 蒸馏输出（student=1024d）：`./out/full_dist_1024.pth`
- 剪枝输出：`./out/pruned-qwen2.5-7b-s0.9/`

---

### C. deepseek-ai/DeepSeek-R1-Distill-Qwen-14B（HF）

- SFT（HF 原始权重 → HF 微调目录）→ 产物：`./out/hf_finetuned/`

```bash
deepspeed --num_gpus 8 scripts/train/train_full_sft.py \
  --model_type hf \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --save_hf \
  --deepspeed scripts/launch/ds_zero3_30b.json \
  --out_dir ./out \
  --epochs 1 --batch_size 2 --learning_rate 5e-6 \
  --max_seq_len 4096 \
  --data_path ./dataset/sft_mini_512.jsonl \
  --tokenizer_dir deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --trust_remote_code --dtype bfloat16 --gradient_checkpointing
```

- 蒸馏（HF 教师 → MiniLLM 学生）→ 产物：`./out/full_dist_{student_dim}.pth`

```bash
python scripts/train/train_distillation.py \
  --out_dir ./out \
  --epochs 2 --batch_size 8 --learning_rate 5e-6 \
  --data_path ./dataset/sft_data.jsonl \
  --tokenizer_dir ./model/minillm_tokenizer \
  --student_dim 1024 --student_layers 24 --max_seq_len 4096 \
  --teacher_model_type hf \
  --teacher_model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --distillation_mode logit --temperature 2.0 --alpha 0.3 \
  --dtype bfloat16 --trust_remote_code
```

- 剪枝（HF）→ 产物目录：`./out/pruned-r1-qwen14b-s0.9`

```bash
python scripts/prune_qwen.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --output_dir ./out/pruned-r1-qwen14b-s0.9 \
  --sparsity 0.9 \
  --device_map auto --dtype bfloat16 \
  --include_embeddings --save_state_dict
```

- RLHF：同 Qwen，建议经蒸馏落到 MiniLLM 后用 `train_dpo.py`。

权重地址一览：

- HF 原始：`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- HF-SFT 输出：`./out/hf_finetuned/`
- 蒸馏输出（student=1024d）：`./out/full_dist_1024.pth`
- 剪枝输出：`./out/pruned-r1-qwen14b-s0.9/`

---

### D. deepseek-ai/DeepSeek-R1-Distill-Llama-8B（HF）

- SFT（HF 原始权重 → HF 微调目录）→ 产物：`./out/hf_finetuned/`

```bash
deepspeed --num_gpus 8 scripts/train/train_full_sft.py \
  --model_type hf \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --save_hf \
  --deepspeed scripts/launch/ds_zero3_30b.json \
  --out_dir ./out \
  --epochs 1 --batch_size 4 --learning_rate 5e-6 \
  --max_seq_len 4096 \
  --data_path ./dataset/sft_mini_512.jsonl \
  --tokenizer_dir deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --trust_remote_code --dtype bfloat16 --gradient_checkpointing
```

- 蒸馏（HF 教师 → MiniLLM 学生）→ 产物：`./out/full_dist_{student_dim}.pth`

```bash
python scripts/train/train_distillation.py \
  --out_dir ./out \
  --epochs 2 --batch_size 8 --learning_rate 5e-6 \
  --data_path ./dataset/sft_data.jsonl \
  --tokenizer_dir ./model/minillm_tokenizer \
  --student_dim 1024 --student_layers 24 --max_seq_len 4096 \
  --teacher_model_type hf \
  --teacher_model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --distillation_mode logit --temperature 2.0 --alpha 0.3 \
  --dtype bfloat16 --trust_remote_code
```

- 剪枝（HF）→ 产物目录：`./out/pruned-r1-llama8b-s0.9`

```bash
python scripts/prune_qwen.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --output_dir ./out/pruned-r1-llama8b-s0.9 \
  --sparsity 0.9 \
  --device_map auto --dtype bfloat16 \
  --include_embeddings --save_state_dict
```

- RLHF：同上，先蒸馏到 MiniLLM 再用 `train_dpo.py`。

权重地址一览：

- HF 原始：`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- HF-SFT 输出：`./out/hf_finetuned/`
- 蒸馏输出（student=1024d）：`./out/full_dist_1024.pth`
- 剪枝输出：`./out/pruned-r1-llama8b-s0.9/`
