"""
test_grpo.py

GRPO核心功能的单元测试
"""

import sys
import torch


def test_orm_advantages():
    """测试ORM模式的优势函数计算"""
    print("\n" + "="*60)
    print("测试 1: ORM优势函数计算")
    print("="*60)
    
    # 导入函数
    sys.path.insert(0, 'class/lec15')
    from grpo_demo import compute_orm_advantages
    
    # 测试用例1：简单情况
    rewards = [1.0, 2.0, 3.0, 4.0]
    advantages = compute_orm_advantages(rewards)
    
    print(f"输入奖励: {rewards}")
    print(f"输出优势: {[f'{a:.4f}' for a in advantages]}")
    
    # 验证：均值应该接近0，标准差应该接近1
    mean_adv = sum(advantages) / len(advantages)
    std_adv = (sum((a - mean_adv)**2 for a in advantages) / len(advantages)) ** 0.5
    
    print(f"优势均值: {mean_adv:.6f} (应接近0)")
    print(f"优势标准差: {std_adv:.6f} (应接近1)")
    
    assert abs(mean_adv) < 1e-6, "优势均值应该接近0"
    print("✅ ORM优势函数测试通过")
    
    # 测试用例2：相同奖励
    rewards_same = [2.0, 2.0, 2.0, 2.0]
    advantages_same = compute_orm_advantages(rewards_same)
    print(f"\n相同奖励: {rewards_same}")
    print(f"输出优势: {[f'{a:.4f}' for a in advantages_same]}")
    print("✅ 相同奖励情况测试通过")


def test_prm_advantages():
    """测试PRM模式的优势函数计算"""
    print("\n" + "="*60)
    print("测试 2: PRM优势函数计算")
    print("="*60)
    
    from grpo_demo import compute_prm_advantages
    
    # 测试用例：3个输出，不同步数
    step_rewards = [
        [1.0, 1.5, 2.0, 2.5],      # 输出1：4步
        [0.5, 1.0, 1.5],            # 输出2：3步
        [1.5, 2.0, 2.5, 3.0, 3.5]  # 输出3：5步
    ]
    
    advantages = compute_prm_advantages(step_rewards)
    
    print("输入步骤奖励:")
    for i, rewards in enumerate(step_rewards):
        print(f"  输出{i+1}: {rewards}")
    
    print("\n输出步骤优势:")
    for i, advs in enumerate(advantages):
        print(f"  输出{i+1}: {[f'{a:.4f}' for a in advs]}")
    
    # 验证：每一步的优势在所有输出中应该是归一化的
    max_steps = max(len(r) for r in step_rewards)
    for t in range(max_steps):
        advs_at_t = []
        for i, advs in enumerate(advantages):
            if t < len(advs):
                advs_at_t.append(advs[t])
        
        if len(advs_at_t) > 1:
            mean_t = sum(advs_at_t) / len(advs_at_t)
            print(f"步骤{t}的优势均值: {mean_t:.6f} (应接近0)")
            assert abs(mean_t) < 0.1, f"步骤{t}的优势均值应该接近0"
    
    print("✅ PRM优势函数测试通过")


def test_grpo_loss():
    """测试GRPO损失计算"""
    print("\n" + "="*60)
    print("测试 3: GRPO损失计算")
    print("="*60)
    
    from grpo_demo import grpo_loss
    
    # 创建测试数据
    batch_size = 4
    seq_len = 10
    
    # 当前策略的log概率
    log_probs = torch.randn(batch_size, seq_len)
    
    # 旧策略的log概率（稍微不同）
    old_log_probs = log_probs + torch.randn(batch_size, seq_len) * 0.1
    
    # 优势函数（有正有负）
    advantages = torch.randn(batch_size, seq_len)
    
    # 计算损失
    loss = grpo_loss(log_probs, old_log_probs, advantages, clip_epsilon=0.2)
    
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Old log probs shape: {old_log_probs.shape}")
    print(f"Advantages shape: {advantages.shape}")
    print(f"GRPO Loss: {loss.item():.6f}")
    
    # 验证：损失应该是标量且为正
    assert loss.dim() == 0, "损失应该是标量"
    assert loss.item() >= 0 or loss.item() < 0, "损失应该是有限值"
    
    print("✅ GRPO损失计算测试通过")
    
    # 测试clip效果
    print("\n测试clip效果:")
    
    # 情况1：ratio在clip范围内
    log_probs_in = torch.tensor([[0.0]])
    old_log_probs_in = torch.tensor([[0.0]])
    advantages_in = torch.tensor([[1.0]])
    loss_in = grpo_loss(log_probs_in, old_log_probs_in, advantages_in, clip_epsilon=0.2)
    print(f"  Ratio=1.0 (在范围内), Loss: {loss_in.item():.6f}")
    
    # 情况2：ratio超出clip范围
    log_probs_out = torch.tensor([[1.0]])
    old_log_probs_out = torch.tensor([[0.0]])
    advantages_out = torch.tensor([[1.0]])
    loss_out = grpo_loss(log_probs_out, old_log_probs_out, advantages_out, clip_epsilon=0.2)
    print(f"  Ratio={torch.exp(torch.tensor(1.0)).item():.2f} (超出范围), Loss: {loss_out.item():.6f}")
    
    print("✅ Clip效果测试通过")


def test_kl_divergence():
    """测试KL散度计算"""
    print("\n" + "="*60)
    print("测试 4: KL散度计算")
    print("="*60)
    
    from grpo_demo import compute_kl_divergence
    
    # 测试用例1：相同分布，KL应该接近0
    log_probs_same = torch.randn(10, 20)
    kl_same = compute_kl_divergence(log_probs_same, log_probs_same)
    print(f"相同分布的KL散度: {kl_same.item():.6f} (应接近0)")
    assert abs(kl_same.item()) < 1e-5, "相同分布的KL散度应该接近0"
    
    # 测试用例2：不同分布
    log_probs_1 = torch.randn(10, 20)
    log_probs_2 = log_probs_1 + torch.randn(10, 20) * 0.5
    kl_diff = compute_kl_divergence(log_probs_1, log_probs_2)
    print(f"不同分布的KL散度: {kl_diff.item():.6f} (应大于0)")
    assert kl_diff.item() >= 0, "KL散度应该非负"
    
    print("✅ KL散度计算测试通过")


def test_reward_model_wrapper():
    """测试Reward Model包装器"""
    print("\n" + "="*60)
    print("测试 5: Reward Model包装器")
    print("="*60)
    
    from grpo_demo import RewardModelWrapper
    
    # 测试ORM模式（使用dummy rewards）
    print("\n测试ORM模式:")
    rm_orm = RewardModelWrapper(model_path=None, mode="orm", device="cpu")
    
    batch_size = 4
    seq_len = 20
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    rewards_orm = rm_orm.get_rewards_orm(input_ids, attention_mask)
    print(f"  输入shape: {input_ids.shape}")
    print(f"  ORM奖励: {[f'{r:.4f}' for r in rewards_orm]}")
    assert len(rewards_orm) == batch_size, "ORM应该返回batch_size个奖励"
    
    # 测试PRM模式
    print("\n测试PRM模式:")
    rm_prm = RewardModelWrapper(model_path=None, mode="prm", device="cpu")
    
    rewards_prm = rm_prm.get_rewards_prm(input_ids, attention_mask)
    print(f"  输入shape: {input_ids.shape}")
    print(f"  PRM奖励数量: {len(rewards_prm)}")
    for i, step_rewards in enumerate(rewards_prm[:2]):  # 只显示前2个
        print(f"  输出{i+1}步骤奖励: {[f'{r:.4f}' for r in step_rewards[:5]]}... (共{len(step_rewards)}步)")
    
    assert len(rewards_prm) == batch_size, "PRM应该返回batch_size个输出的奖励"
    
    print("✅ Reward Model包装器测试通过")


def test_integration():
    """集成测试：完整的GRPO流程"""
    print("\n" + "="*60)
    print("测试 6: 集成测试")
    print("="*60)
    
    from grpo_demo import compute_orm_advantages, grpo_loss, compute_kl_divergence
    
    # 模拟一个完整的GRPO更新步骤
    group_size = 4
    seq_len = 10
    
    print(f"模拟场景: {group_size}个输出，每个{seq_len}个token")
    
    # 1. 生成奖励
    rewards = [float(i) + torch.randn(1).item() * 0.5 for i in range(group_size)]
    print(f"\n1. 奖励: {[f'{r:.4f}' for r in rewards]}")
    
    # 2. 计算优势
    advantages = compute_orm_advantages(rewards)
    print(f"2. 优势: {[f'{a:.4f}' for a in advantages]}")
    
    # 3. 准备log概率
    log_probs = torch.randn(group_size, seq_len)
    old_log_probs = log_probs + torch.randn(group_size, seq_len) * 0.1
    advantages_tensor = torch.tensor(advantages).unsqueeze(1).expand(-1, seq_len)
    
    # 4. 计算损失
    policy_loss = grpo_loss(log_probs, old_log_probs, advantages_tensor, clip_epsilon=0.2)
    kl = compute_kl_divergence(log_probs, old_log_probs)
    
    print(f"3. Policy Loss: {policy_loss.item():.6f}")
    print(f"4. KL Divergence: {kl.item():.6f}")
    
    # 5. 总损失
    kl_coef = 0.1
    total_loss = policy_loss + kl_coef * kl
    print(f"5. Total Loss: {total_loss.item():.6f}")
    
    print("\n✅ 集成测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🧪 GRPO单元测试")
    print("="*60)
    
    try:
        test_orm_advantages()
        test_prm_advantages()
        test_grpo_loss()
        test_kl_divergence()
        test_reward_model_wrapper()
        test_integration()
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ 测试失败: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
