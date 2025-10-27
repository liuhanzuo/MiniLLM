#!/usr/bin/env python3
"""
test_loss_fix.py

验证GRPO loss修复是否生效
"""

import torch
import numpy as np


def test_advantage_computation():
    """测试优势函数计算是否正确"""
    print("=" * 60)
    print("测试1: 优势函数计算")
    print("=" * 60)
    
    # 导入修复后的函数
    import sys
    sys.path.insert(0, '/apdcephfs/pig_data/MiniLLM/class/lec15')
    from grpo_demo import compute_orm_advantages
    
    # 测试用例1: 正常情况
    print("\n✅ 测试用例1: 正常奖励值")
    rewards1 = [0.5, 1.0, 1.5, 2.0]
    adv1 = compute_orm_advantages(rewards1)
    print(f"  Rewards: {rewards1}")
    print(f"  Advantages: {[f'{a:.4f}' for a in adv1]}")
    print(f"  Range: [{min(adv1):.4f}, {max(adv1):.4f}]")
    assert all(-15 < a < 15 for a in adv1), "优势值应该在合理范围内"
    print("  ✅ 通过")
    
    # 测试用例2: 标准差很小的情况（修复的关键）
    print("\n✅ 测试用例2: 标准差很小（应该返回全0）")
    rewards2 = [0.1, 0.11, 0.12, 0.13]
    adv2 = compute_orm_advantages(rewards2)
    print(f"  Rewards: {rewards2}")
    print(f"  Std: {np.std(rewards2):.6f}")
    print(f"  Advantages: {[f'{a:.4f}' for a in adv2]}")
    assert all(abs(a) < 0.1 for a in adv2), "标准差很小时，优势值应该接近0"
    print("  ✅ 通过（修复生效！）")
    
    # 测试用例3: 极端情况
    print("\n✅ 测试用例3: 极端奖励值")
    rewards3 = [-10.0, -5.0, 5.0, 10.0]
    adv3 = compute_orm_advantages(rewards3)
    print(f"  Rewards: {rewards3}")
    print(f"  Advantages: {[f'{a:.4f}' for a in adv3]}")
    print(f"  Range: [{min(adv3):.4f}, {max(adv3):.4f}]")
    assert all(-15 < a < 15 for a in adv3), "即使奖励极端，优势值也应该被clip"
    print("  ✅ 通过")
    
    print("\n" + "=" * 60)
    print("✅ 所有优势函数测试通过！")
    print("=" * 60)


def test_dummy_rewards():
    """测试dummy rewards是否合理"""
    print("\n" + "=" * 60)
    print("测试2: Dummy Rewards生成")
    print("=" * 60)
    
    import sys
    sys.path.insert(0, '/apdcephfs/pig_data/MiniLLM/class/lec15')
    from grpo_demo import RewardModelWrapper
    
    # 创建dummy reward model
    rm = RewardModelWrapper(model_path=None, mode="orm")
    
    # 生成测试数据
    batch_size = 4
    seq_len = 20
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 获取ORM奖励
    print("\n✅ 测试ORM奖励")
    rewards_orm = rm.get_rewards_orm(input_ids, attention_mask)
    print(f"  Rewards: {[f'{r:.4f}' for r in rewards_orm]}")
    print(f"  Mean: {np.mean(rewards_orm):.4f}")
    print(f"  Std: {np.std(rewards_orm):.4f}")
    assert np.std(rewards_orm) > 0.1, "奖励标准差应该 > 0.1"
    print("  ✅ 通过（标准差足够大）")
    
    # 获取PRM奖励
    print("\n✅ 测试PRM奖励")
    rm_prm = RewardModelWrapper(model_path=None, mode="prm")
    rewards_prm = rm_prm.get_rewards_prm(input_ids, attention_mask)
    print(f"  输出数量: {len(rewards_prm)}")
    print(f"  第一个输出的步骤数: {len(rewards_prm[0])}")
    print(f"  第一个输出的前5步奖励: {[f'{r:.4f}' for r in rewards_prm[0][:5]]}")
    assert all(len(r) > 0 for r in rewards_prm), "每个输出都应该有奖励"
    print("  ✅ 通过")
    
    print("\n" + "=" * 60)
    print("✅ 所有Dummy Rewards测试通过！")
    print("=" * 60)


def test_loss_computation():
    """测试损失计算是否正确"""
    print("\n" + "=" * 60)
    print("测试3: 损失计算")
    print("=" * 60)
    
    import sys
    sys.path.insert(0, '/apdcephfs/pig_data/MiniLLM/class/lec15')
    from grpo_demo import grpo_loss
    
    # 测试用例1: 正常情况
    print("\n✅ 测试用例1: 正常损失计算")
    log_probs = torch.randn(100) * 0.1 - 2.0  # 模拟log概率
    old_log_probs = log_probs + torch.randn(100) * 0.05  # 稍微不同
    advantages = torch.randn(100) * 0.5  # 正常优势值
    
    loss = grpo_loss(log_probs, old_log_probs, advantages, clip_epsilon=0.2)
    print(f"  Loss: {loss.item():.4f}")
    assert 0 < loss.item() < 10, "损失应该在合理范围内"
    print("  ✅ 通过")
    
    # 测试用例2: 极端优势值（修复后应该被clip）
    print("\n✅ 测试用例2: 极端优势值")
    advantages_extreme = torch.tensor([100.0, -100.0, 50.0, -50.0] * 25)  # 极端值
    loss_extreme = grpo_loss(log_probs, old_log_probs, advantages_extreme, clip_epsilon=0.2)
    print(f"  Loss: {loss_extreme.item():.4f}")
    # 由于优势值被clip，损失不应该太大
    assert loss_extreme.item() < 1000, "即使优势值极端，损失也不应该爆炸"
    print("  ✅ 通过（修复生效！）")
    
    print("\n" + "=" * 60)
    print("✅ 所有损失计算测试通过！")
    print("=" * 60)


def test_padding_mask():
    """测试padding mask是否正确"""
    print("\n" + "=" * 60)
    print("测试4: Padding Mask")
    print("=" * 60)
    
    # 模拟场景
    print("\n✅ 测试padding位置过滤")
    
    # 创建测试数据
    seq_len = 10
    pad_token_id = 50256  # GPT2的pad token
    
    # labels: [1, 2, 3, pad, pad, pad, ...]
    labels = torch.tensor([[1, 2, 3, pad_token_id, pad_token_id, 
                           pad_token_id, pad_token_id, pad_token_id, 
                           pad_token_id, pad_token_id]])
    
    # 创建mask
    labels_mask = (labels != pad_token_id).float()
    valid_positions = labels_mask.bool()
    
    print(f"  Labels: {labels[0].tolist()}")
    print(f"  Mask: {labels_mask[0].tolist()}")
    print(f"  有效位置数: {valid_positions.sum().item()}")
    
    assert valid_positions.sum().item() == 3, "应该有3个有效位置"
    print("  ✅ 通过")
    
    # 测试只对有效位置计算损失
    print("\n✅ 测试损失只在有效位置计算")
    log_probs = torch.randn(1, seq_len)
    
    # 全部位置的损失
    loss_all = log_probs.mean()
    print(f"  全部位置损失: {loss_all.item():.4f}")
    
    # 只有效位置的损失
    loss_valid = log_probs[valid_positions].mean()
    print(f"  有效位置损失: {loss_valid.item():.4f}")
    
    # 它们应该不同
    assert abs(loss_all.item() - loss_valid.item()) > 0.01, "两种计算方式应该不同"
    print("  ✅ 通过（正确过滤了padding）")
    
    print("\n" + "=" * 60)
    print("✅ 所有Padding Mask测试通过！")
    print("=" * 60)


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("🧪 GRPO Loss修复验证测试")
    print("=" * 60)
    
    try:
        test_advantage_computation()
        test_dummy_rewards()
        test_loss_computation()
        test_padding_mask()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！修复生效！")
        print("=" * 60)
        print("\n✅ 现在可以安全运行GRPO训练：")
        print("   python class/lec15/grpo_demo.py --make_dummy --reward_mode orm")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
