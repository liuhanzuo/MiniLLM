# 数据集与打标（dataset）

## 1. 数据来源与清洗
- 人类标注、专家数据、合成数据、公开集合（注意 license 与评测泄露）。
- 清洗：规则/正则（去 HTML/脏词/异常长度）、MinHash/SimHash 去近重复、n-gram 去重、Decontamination 与评测集做交集检测。
- 质量抽检：小样人工验收（事实性、完整性、格式、一致性）。

## 2. 统一到聊天模板（chat template）
推荐使用模型自带的 `tokenizer.apply_chat_template()`，抽象结构：

```json
{
  "conversations": [
    {"role": "user", "content": "……指令/问题……"},
    {"role": "assistant", "content": "……期望答案……"}
  ]
}
```

- system 放规范/约束，user 放问题与上下文，assistant 放答案。
- 统一模板利于多任务混合与后续评测复现。

## 3. assistant-only 的 loss mask
核心：只在 assistant 段计算损失；对 prompt（system/user）打 ignore。
- Label shift：右移一位 `y_t = x_{t+1}`；mask 要对齐到 `Y`。
- 生成泛化方法：把某个 assistant turn 的 content 清空，再与完整序列对比，差异区间即该 turn 的训练区域（仓库的 `SFTDataset` 已实现）。

## 4. Padding 与 Packing
- Padding：按 batch 内最大长度补 PAD，简单但浪费显存；注意力允许矩阵：`pad(i) * pad(j) * causal(i,j)`。
- Packing：把多条样本拼成长序列，用段号 `s(i)` 构造分块下三角注意力：`same_seg(i,j) * causal(i,j)`。跨段注意力必须为 0；跨段边界的 `m_i`（loss mask）也要置 0。

公式（与直觉）详见讲义正文；演示参见 `dataset_demo.py`。
