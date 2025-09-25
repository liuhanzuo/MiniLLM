# lec8: 推理与显存优化对比 Demo

本小节提供一个可复现实验，比较在部署大模型推理时：
- 是否启用 KV Cache（past_key_values）
- 不同注意力实现（标准 vs FlashAttention/SDPA）

对显存占用与生成速度（tokens/sec）的影响。

## 环境需求
- Python 3.10+
- PyTorch >= 2.1（建议2.3+）
- transformers >= 4.39
- 可选：flash-attn >= 2.x（安装成功后才会启用），或使用 PyTorch SDPA（默认可用）

## 运行
示例：
```
python3 class/lec8/kv_flash_benchmark.py   --model Qwen/Qwen2.5-7b-Instruct   --prompt_tokens 2048   --max-new-tokens 256   --dtype fp32   --tp greedy   --concurrency 8   --only_backends default,sdpa   --out class/lec8/out_greedy
```
若无模型下载权限，可先尝试一个开放模型：
```
python3 class/lec8/kv_flash_benchmark.py --model facebook/opt-1.3b \
  --prompt "Hello" --max-new-tokens 64 --warmup 2 --runs 3
```
程序将自动检测可用后端并分别测试：
- no_kv + 默认注意力
- kv_cache + 默认注意力
- kv_cache + sdpa
- kv_cache + flash_attn（若安装成功）

输出将包含：
- 每组设置的 tokens/sec、平均生成时延、显存峰值（MiB）
- CSV 文件 `results.csv` 与简单的条形图 `results.png`

## 注意
- 运行时显存需求与模型大小成正比，请根据机器选择合适模型。
- 若仅有 CPU，也能跑功能完整的对比但速度会很慢，显存峰值为显卡内存不可用时记录为0。
