# lec11 · 参数高效微调（PEFT）实战

本讲聚焦以下方法的直观解释与最小可运行示例：

- 参数冻结与轻量 classifier
- LoRA（低秩适配）
- QLoRA（低秩+量化，使用伪量化演示前向/反向的量化-反量化）
- Prefix-Tuning（软提示：前缀 token 或 KV 前缀）
- Adapter（层内瓶颈模块）
- P-Tuning v2（在每一层注意力上添加 KV 前缀）

> 位置：`class/lec11/peft_demos.py`，无需依赖外部数据，脚本会构造小型合成任务（情感二分类风格）。

## 快速开始

- 查看可用参数

```bash
python MiniLLM/class/lec11/peft_demos.py --help
```

- 运行一个 LoRA 示例（CPU，极小步）

```bash
python class/lec11/peft_demos.py \
  --method lora --lora-rank 8 --epochs 1 --steps-per-epoch 5 --d-model 256 --n-layers 2
```

- 运行 QLoRA（伪量化）

```bash
python class/lec11/peft_demos.py \
  --method qlora --lora-rank 8 --qlora-bits 8 --epochs 1 --steps-per-epoch 5
```

- Prefix / Soft Prompt（仅输入前缀）

```bash
python class/lec11/peft_demos.py --method prefix --prefix-mode embed --prefix-len 16
```

- Prefix（KV 前缀，仅第一层）

```bash
python class/lec11/peft_demos.py --method prefix --prefix-mode kv --prefix-len 16
```

- P-Tuning v2（所有层 KV 前缀）

```bash
python class/lec11/peft_demos.py --method ptv2 --prefix-len 16
```

- Adapter

```bash
python class/lec11/peft_demos.py --method adapter --adapter-r 16
```

- 参数冻结 + 轻量分类头

```bash
python class/lec11/peft_demos.py --method freeze
```

### 演示输出（--verbose 默认开启）

脚本会自动打印：
- 方法与关键超参（如 LoRA rank/alpha/qbits、Adapter r、Prefix 长度等）
- 可训练/冻结参数统计与示例参数名
- 前向期间的形状提示：Prefix-Embed 拼接后的序列长度、KV 前缀长度、注意力 K/V 长度变化等
- LoRA 增量张量形状与范数近似
- 第一梯度步的梯度范数（可训练/冻结各采样若干参数）
- 每 5 步一次的 loss/acc，以及一次样例预测 vs 真实标签

## 备注

- 本脚本在 CPU 上可运行，显存占用极低，便于教学演示。
- QLoRA 采用“伪量化”（Q/DQ）以展示原理，不引入外部量化库。
- 这是教学最小实现，并非高性能/完备产线代码；真实训练请使用成熟库（如 PEFT、bitsandbytes 等）。
