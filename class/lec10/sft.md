# SFT（监督微调）

## 1. 目标与边界
- 目标：让已有语言能力的 base 模型学会“按人类期望的格式和风格作答”。
- 边界：不直接优化主观偏好/安全/有用（那是 RLHF/DPO/PPO 的领域），但高质量指令数据能显著提升可用性。

## 2. 训练要点与多任务混合
- 清晰任务描述（role/objective/constraints），足量上下文，输出模板可消费（JSON/表格等）。
- 反例/困难例、少样本演示（注意 token 成本）。
- 多任务混合：幂律/温度采样 `p_i ∝ n_i^α, α∈[0,1]`；对小集合上采样、大集合下采样；分桶与 curriculum。
- 数据顺序：可将更具体/高质量的数据放到后段或提高 epoch。

## 3. 训练循环与损失
- 自回归交叉熵（teacher forcing）。
- Label shift：`inputs: x_1..x_{T-1}`, `targets: y_t=x_{t+1}`。
- 只在 assistant 位置计损失，其他位置 `ignore_index=-100`。
- Causal mask：下三角；与 padding/packing 的注意力 bias 叠加。

等价实现：把 `labels` 中非 assistant 位置设为 -100，调用
`F.cross_entropy(logits.view(-1,V), labels.view(-1), ignore_index=-100)`。

## 4. 实践注意事项
- tokenizer 与模型词表对齐（必要时 `resize_token_embeddings`）。
- 长度控制：`max_seq_len` 与梯度累积决定显存与吞吐。
- AMP/精度：bf16/fp16 与梯度检查点，稳定优先（可先 FP32 验证再开启 AMP）。
- 分布式：DDP/ZeRO 的配置与网络（IFACE/NCCL/GLOO）。
- 监控：loss 曲线、有效 token 数、吞吐（tok/s）、显存、梯度爆炸/消失。
