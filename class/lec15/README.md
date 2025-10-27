## GRPO代码说明

ORM和PRM两种奖励模式的GRPO实现，包含训练脚本和单元测试。

### ✅ 最新更新 (2025-10-27)

**已修复loss异常高的问题！** 如果之前遇到loss爆炸到几十万的情况，现在已经修复。

查看详情：
- [FIX_SUMMARY.md](FIX_SUMMARY.md) - 快速了解修复内容
- [COMPLETE_FIX_SUMMARY.md](COMPLETE_FIX_SUMMARY.md) - 完整修复说明
- [LOSS_FIX.md](LOSS_FIX.md) - 详细技术分析

### 快速开始

```bash
# 验证修复
python class/lec15/test_loss_fix.py

# ORM模式训练
python class/lec15/grpo_demo.py --make_dummy --reward_mode orm

# PRM模式训练
python class/lec15/grpo_demo.py --make_dummy --reward_mode prm

# 运行单元测试
python class/lec15/test_grpo.py
```

### 文档索引

- **[ALGORITHM_DETAILS.md](ALGORITHM_DETAILS.md)** - GRPO算法详解
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - 详细使用指南
- **[CONFIG_GUIDE.md](CONFIG_GUIDE.md)** - 配置参数说明
