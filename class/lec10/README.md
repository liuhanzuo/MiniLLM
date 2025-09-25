# 课程第 10 讲：指令微调（SFT）分解—Dataset 与 SFT

本讲拆成两部分：
- dataset：数据来源、清洗、统一到聊天模板、padding/packing 与注意力掩码、assistant-only loss mask 生成。
- sft：SFT 目标与边界、训练配方、label shift + masked CE 的实现与正确性校验。

建议先读 `dataset.md` 再读 `sft.md`。两个最小可运行演示：
- `dataset_demo.py`：从 `dataset/sft_512.jsonl` 取若干条，展示 chat template 拼接、loss mask、padding/packing 注意力掩码。
- `sft_loss_demo.py`：用 `SFTDataset` 的 (X,Y,mask) 计算带 ignore_index 的交叉熵，验证与公式一致。
