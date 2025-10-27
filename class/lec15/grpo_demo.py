"""
grpo_demo.py

Group Relative Policy Optimization (GRPO) 演示代码

GRPO 是一种基于组内关系的强化学习算法，与传统PPO不同：
- 对同一个问题生成多个输出（组内采样）
- 使用Reward Model对所有输出评分
- 通过组内比较计算优势函数
- 支持两种模式：
  1. ORM (Outcome Reward Model): 基于整体输出的奖励
  2. PRM (Process Reward Model): 基于每个步骤的奖励

核心思想：
- ORM: 单步MDP，优化整体输出质量
- PRM: 多步MDP，优化生成过程中每个token的质量
"""

import os
import sys
import json
import argparse
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GRPOConfig:
    """GRPO训练配置"""
    # 模型配置
    base_model: str = "gpt2"
    tokenizer_dir: str = "gpt2"
    reward_model_path: Optional[str] = None
    
    # GRPO特定参数
    group_size: int = 4  # 每个问题采样的输出数量 G
    reward_mode: str = "orm"  # "orm" 或 "prm"
    
    # 训练参数
    epochs: int = 1
    batch_size: int = 4  # 问题的batch size
    lr: float = 1e-5
    max_seq_len: int = 128
    max_gen_len: int = 64
    
    # PPO相关参数
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1  # KL散度惩罚系数
    
    # 优势函数归一化
    normalize_advantage: bool = True
    advantage_eps: float = 1e-8
    
    # 其他
    device: str = "cuda:0"
    save_path: str = "./class/lec15/out"
    log_interval: int = 10
    seed: int = 42
    
    # 数据路径
    data_path: str = "./class/lec15/demo_prompts.jsonl"


class PromptDataset:
    """加载问题/提示数据集"""
    
    def __init__(self, path: str):
        self.prompts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                # 支持多种格式
                if 'prompt' in j:
                    self.prompts.append(j['prompt'])
                elif 'question' in j:
                    self.prompts.append(j['question'])
                elif 'text' in j:
                    self.prompts.append(j['text'])
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]


def compute_orm_advantages(rewards: List[float], eps: float = 1e-8) -> List[float]:
    """
    计算ORM模式下的优势函数
    
    Args:
        rewards: 组内所有输出的奖励值 [r1, r2, ..., rG]
        eps: 数值稳定性常数
    
    Returns:
        advantages: 归一化后的优势值
    
    公式：A_i = (r_i - mean(r)) / (std(r) + eps)
    """
    import torch
    r = torch.tensor(rewards, dtype=torch.float32)
    mean_r = r.mean()
    std_r = r.std()
    
    # 防止std太小导致优势值爆炸
    # 如果std太小，说明所有输出质量相近，优势值应该接近0
    if std_r < eps * 10:  # 如果标准差太小
        advantages = torch.zeros_like(r)  # 所有优势值设为0
    else:
        advantages = (r - mean_r) / (std_r + eps)
        # Clip优势值到合理范围，避免极端值
        advantages = torch.clamp(advantages, -10.0, 10.0)
    
    return advantages.tolist()



def compute_prm_advantages(
    step_rewards: List[List[float]], 
    eps: float = 1e-8
) -> List[List[float]]:
    """
    计算PRM模式下的优势函数
    
    Args:
        step_rewards: 每个输出的每步奖励
                     [[r1_1, r1_2, ..., r1_K1],  # 输出1的K1步奖励
                      [r2_1, r2_2, ..., r2_K2],  # 输出2的K2步奖励
                      ...]
        eps: 数值稳定性常数
    
    Returns:
        advantages: 每个输出每步的优势值（同样的嵌套结构）
    
    公式：对每一步 t，A_{i,t} = (r_{i,t} - mean(r_t)) / (std(r_t) + eps)
    """
    import torch
    
    # 找到最大步数
    max_steps = max(len(rewards) for rewards in step_rewards)
    
    # 对每一步计算优势
    advantages = []
    for i, rewards in enumerate(step_rewards):
        adv_i = []
        for t, r_t in enumerate(rewards):
            # 收集所有输出在步骤t的奖励
            rewards_at_t = []
            for j, other_rewards in enumerate(step_rewards):
                if t < len(other_rewards):
                    rewards_at_t.append(other_rewards[t])
            
            # 归一化
            if len(rewards_at_t) > 1:
                r_tensor = torch.tensor(rewards_at_t, dtype=torch.float32)
                mean_r = r_tensor.mean()
                std_r = r_tensor.std()
                adv_t = (r_t - mean_r.item()) / (std_r.item() + eps)
            else:
                adv_t = 0.0
            
            adv_i.append(adv_t)
        advantages.append(adv_i)
    
    return advantages


def grpo_loss(
    log_probs: 'torch.Tensor',
    old_log_probs: 'torch.Tensor', 
    advantages: 'torch.Tensor',
    clip_epsilon: float = 0.2
) -> 'torch.Tensor':
    """
    计算GRPO的策略损失（类似PPO的clipped objective）
    
    Args:
        log_probs: 当前策略的log概率
        old_log_probs: 旧策略的log概率
        advantages: 优势函数值
        clip_epsilon: clip范围
    
    Returns:
        loss: 策略损失
    
    公式：
        ratio = exp(log_prob - old_log_prob)
        L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
        loss = -mean(L_clip)
    """
    import torch
    
    # 计算重要性采样比率
    ratio = torch.exp(log_probs - old_log_probs)
    
    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    
    # 取最小值（保守更新）
    policy_loss = -torch.min(surr1, surr2).mean()
    
    return policy_loss


def compute_kl_divergence(
    log_probs: 'torch.Tensor',
    old_log_probs: 'torch.Tensor'
) -> 'torch.Tensor':
    """
    计算KL散度 KL(π_old || π_new)
    
    公式：KL = mean(exp(old_log_prob) * (old_log_prob - log_prob))
    """
    import torch
    kl = (torch.exp(old_log_probs) * (old_log_probs - log_probs)).mean()
    return kl


class RewardModelWrapper:
    """Reward Model包装器，支持ORM和PRM两种模式"""
    
    def __init__(self, model_path: Optional[str], mode: str = "orm", device: str = "cuda:0"):
        """
        Args:
            model_path: reward model checkpoint路径
            mode: "orm" 或 "prm"
            device: 设备
        """
        self.mode = mode
        self.device = device
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f"⚠️ Reward model not found, using dummy rewards for demo")
    
    def _load_model(self, model_path: str):
        """加载训练好的reward model"""
        import torch
        from torch import nn
        from transformers import AutoModel
        
        # 加载checkpoint
        ckpt = torch.load(model_path, map_location='cpu')
        backbone_name = ckpt.get('backbone_name', 'gpt2')
        
        # 重建模型结构（与reward_model_demo.py一致）
        class RewardModelLocal(nn.Module):
            def __init__(self, backbone_name: str = 'gpt2'):
                super().__init__()
                self.backbone = AutoModel.from_pretrained(backbone_name)
                hidden_size = self.backbone.config.hidden_size
                self.head = nn.Linear(hidden_size, 1)
            
            def forward(self, input_ids, attention_mask):
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                last_hidden = outputs.last_hidden_state
                lengths = attention_mask.sum(dim=1) - 1
                lengths = lengths.clamp(min=0)
                batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
                eos_h = last_hidden[batch_idx, lengths, :]
                score = self.head(eos_h).squeeze(-1)
                return score
        
        self.model = RewardModelLocal(backbone_name)
        self.model.head.load_state_dict(ckpt['head_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Loaded reward model from {model_path}")
    
    def get_rewards_orm(
        self, 
        input_ids: 'torch.Tensor', 
        attention_mask: 'torch.Tensor'
    ) -> List[float]:
        """
        ORM模式：返回每个完整输出的奖励值
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            rewards: [r1, r2, ..., rG]
        """
        import torch
        import numpy as np
        
        if self.model is None:
            # Dummy rewards for demo - 使用更合理的范围和差异
            batch_size = input_ids.size(0)
            # 生成有明显差异的奖励值，范围在[-2, 2]之间
            np.random.seed(hash(input_ids.sum().item()) % 2**32)
            rewards = np.random.randn(batch_size) * 0.5  # 标准差0.5
            return rewards.tolist()
        
        with torch.no_grad():
            scores = self.model(input_ids, attention_mask)
            return scores.cpu().tolist()

    
    def get_rewards_prm(
        self,
        input_ids: 'torch.Tensor',
        attention_mask: 'torch.Tensor'
    ) -> List[List[float]]:
        """
        PRM模式：返回每个输出每一步的奖励值
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            step_rewards: [[r1_1, r1_2, ...], [r2_1, r2_2, ...], ...]
        """
        import torch
        import numpy as np
        
        if self.model is None:
            # Dummy step rewards for demo - 使用更合理的值
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            step_rewards = []
            np.random.seed(hash(input_ids.sum().item()) % 2**32)
            
            for i in range(batch_size):
                # 每个输出的有效长度
                valid_len = int(attention_mask[i].sum().item())
                # 生成有差异的过程奖励，范围在[-1, 1]之间
                base_reward = np.random.randn() * 0.3
                rewards_i = [base_reward + np.random.randn() * 0.1 
                            for t in range(valid_len)]
                step_rewards.append(rewards_i)
            return step_rewards

        
        # 真实PRM需要在每个token位置都计算奖励
        # 这里简化为使用最终奖励的累积
        with torch.no_grad():
            batch_size = input_ids.size(0)
            step_rewards = []
            
            for i in range(batch_size):
                valid_len = attention_mask[i].sum().item()
                # 对每个前缀计算奖励
                rewards_i = []
                for t in range(1, int(valid_len) + 1):
                    prefix_ids = input_ids[i:i+1, :t]
                    prefix_mask = attention_mask[i:i+1, :t]
                    score = self.model(prefix_ids, prefix_mask)
                    rewards_i.append(score.item())
                step_rewards.append(rewards_i)
            
            return step_rewards


def train_grpo(config: GRPOConfig):
    """GRPO训练主函数"""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting GRPO training with {config.reward_mode.upper()} mode")
    print(f"📊 Group size: {config.group_size}")
    print(f"🎯 Device: {device}")
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 策略模型（要训练的模型）
    policy_model = AutoModelForCausalLM.from_pretrained(config.base_model)
    policy_model.to(device)
    
    # 参考模型（用于KL惩罚，冻结参数）
    ref_model = AutoModelForCausalLM.from_pretrained(config.base_model)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    # Reward Model
    reward_model = RewardModelWrapper(
        config.reward_model_path, 
        mode=config.reward_mode,
        device=device
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.lr)
    
    # 加载数据
    dataset = PromptDataset(config.data_path)
    print(f"📚 Loaded {len(dataset)} prompts")
    
    # 训练循环
    policy_model.train()
    
    for epoch in range(config.epochs):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl = 0.0
        
        # 简单的批次迭代
        num_batches = (len(dataset) + config.batch_size - 1) // config.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * config.batch_size
            end_idx = min(start_idx + config.batch_size, len(dataset))
            batch_prompts = [dataset[i] for i in range(start_idx, end_idx)]
            
            # ========== 第1步：组内采样 ==========
            # 对每个prompt生成G个输出
            all_outputs = []  # [(prompt, output, input_ids, attention_mask), ...]
            
            for prompt in batch_prompts:
                # 编码prompt
                prompt_text = f"User: {prompt}\nAssistant:"
                prompt_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
                
                # 生成G个输出
                for g in range(config.group_size):
                    with torch.no_grad():
                        output_ids = policy_model.generate(
                            prompt_ids,
                            max_new_tokens=config.max_gen_len,
                            do_sample=True,
                            temperature=0.9,
                            top_p=0.95,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    
                    # 解码
                    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # 准备attention mask
                    attention_mask = torch.ones_like(output_ids)
                    
                    all_outputs.append((prompt, output_text, output_ids, attention_mask))
            
            # ========== 第2步：计算奖励和优势函数 ==========
            # 按组组织输出
            groups = []
            for i in range(len(batch_prompts)):
                group_start = i * config.group_size
                group_end = group_start + config.group_size
                groups.append(all_outputs[group_start:group_end])
            
            all_advantages = []
            
            for group in groups:
                # 提取input_ids和attention_mask，需要padding到相同长度
                group_ids_list = [item[2] for item in group]
                group_masks_list = [item[3] for item in group]
                
                # 找到最大长度
                max_len = max(ids.shape[1] for ids in group_ids_list)
                
                # Padding到相同长度
                padded_ids = []
                padded_masks = []
                for ids, mask in zip(group_ids_list, group_masks_list):
                    pad_len = max_len - ids.shape[1]
                    if pad_len > 0:
                        # 右侧padding
                        ids = torch.cat([ids, torch.full((1, pad_len), tokenizer.pad_token_id, device=device)], dim=1)
                        mask = torch.cat([mask, torch.zeros((1, pad_len), device=device)], dim=1)
                    padded_ids.append(ids)
                    padded_masks.append(mask)
                
                group_ids = torch.cat(padded_ids, dim=0)
                group_masks = torch.cat(padded_masks, dim=0)
                
                if config.reward_mode == "orm":
                    # ORM模式：每个输出一个奖励
                    rewards = reward_model.get_rewards_orm(group_ids, group_masks)
                    advantages = compute_orm_advantages(rewards, config.advantage_eps)
                    # 每个输出的优势值相同（应用到所有token）
                    all_advantages.extend(advantages)
                
                else:  # PRM模式
                    # PRM模式：每个输出每步一个奖励
                    step_rewards = reward_model.get_rewards_prm(group_ids, group_masks)
                    step_advantages = compute_prm_advantages(step_rewards, config.advantage_eps)
                    all_advantages.extend(step_advantages)

            
            # ========== 第3步：计算策略损失 ==========
            policy_losses = []
            kl_divs = []
            
            for idx, (prompt, output_text, output_ids, attention_mask) in enumerate(all_outputs):
                # 当前策略的log概率
                with torch.no_grad():
                    policy_outputs = policy_model(output_ids, attention_mask=attention_mask)
                    logits = policy_outputs.logits[:, :-1, :]  # [1, seq_len-1, vocab]
                    labels = output_ids[:, 1:]  # [1, seq_len-1]
                    
                    log_probs = torch.log_softmax(logits, dim=-1)
                    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [1, seq_len-1]
                
                # 参考策略的log概率
                with torch.no_grad():
                    ref_outputs = ref_model(output_ids, attention_mask=attention_mask)
                    ref_logits = ref_outputs.logits[:, :-1, :]
                    ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                    ref_token_log_probs = ref_log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                
                # 优势值
                if config.reward_mode == "orm":
                    # ORM: 所有token使用相同的优势值
                    adv = torch.tensor([all_advantages[idx]], device=device)
                    adv = adv.expand_as(token_log_probs)
                else:
                    # PRM: 每个token使用对应步骤的优势值
                    step_advs = all_advantages[idx]
                    adv = torch.tensor(step_advs, device=device).unsqueeze(0)
                    # 截断或填充到token_log_probs的长度
                    if adv.size(1) < token_log_probs.size(1):
                        pad_len = token_log_probs.size(1) - adv.size(1)
                        adv = torch.cat([adv, torch.zeros(1, pad_len, device=device)], dim=1)
                    else:
                        adv = adv[:, :token_log_probs.size(1)]
                
                # 计算GRPO损失（需要梯度）
                # 重新前向传播以获得可导的log_probs
                policy_outputs_grad = policy_model(output_ids, attention_mask=attention_mask)
                logits_grad = policy_outputs_grad.logits[:, :-1, :]
                log_probs_grad = torch.log_softmax(logits_grad, dim=-1)
                token_log_probs_grad = log_probs_grad.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                
                # 创建mask：只计算非padding位置的损失
                # labels_mask: [1, seq_len-1]
                labels_mask = (labels != tokenizer.pad_token_id).float()
                
                # 应用mask
                masked_log_probs = token_log_probs_grad * labels_mask
                masked_ref_log_probs = ref_token_log_probs * labels_mask
                masked_adv = adv * labels_mask
                
                # 只选择有效位置（非padding）
                valid_positions = labels_mask.bool()
                if valid_positions.sum() > 0:
                    # GRPO loss - 只计算有效位置
                    loss = grpo_loss(
                        token_log_probs_grad[valid_positions],
                        ref_token_log_probs[valid_positions].detach(),
                        adv[valid_positions].detach(),
                        config.clip_epsilon
                    )
                    policy_losses.append(loss)
                    
                    # KL散度 - 只计算有效位置
                    kl = compute_kl_divergence(
                        token_log_probs_grad[valid_positions],
                        ref_token_log_probs[valid_positions]
                    )
                    kl_divs.append(kl)
                else:
                    # 如果没有有效位置，添加零损失
                    policy_losses.append(torch.tensor(0.0, device=device))
                    kl_divs.append(torch.tensor(0.0, device=device))

            
            # ========== 第4步：反向传播和优化 ==========
            total_policy_loss_batch = torch.stack(policy_losses).mean()
            total_kl_batch = torch.stack(kl_divs).mean()
            
            # 总损失 = 策略损失 + KL惩罚
            loss = total_policy_loss_batch + config.kl_coef * total_kl_batch
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_policy_loss += total_policy_loss_batch.item()
            total_kl += total_kl_batch.item()
            
            # 日志
            if (batch_idx + 1) % config.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_policy_loss = total_policy_loss / (batch_idx + 1)
                avg_kl = total_kl / (batch_idx + 1)
                print(f"Epoch {epoch} Batch {batch_idx+1}/{num_batches} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Policy Loss: {avg_policy_loss:.4f} | "
                      f"KL: {avg_kl:.4f}")
        
        # Epoch结束
        avg_loss = total_loss / max(1, num_batches)
        avg_policy_loss = total_policy_loss / max(1, num_batches)
        avg_kl = total_kl / max(1, num_batches)
        print(f"✅ Epoch {epoch} finished | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Avg Policy Loss: {avg_policy_loss:.4f} | "
              f"Avg KL: {avg_kl:.4f}")
    
    # 保存模型
    os.makedirs(config.save_path, exist_ok=True)
    save_file = os.path.join(config.save_path, f'grpo_{config.reward_mode}_policy.pt')
    torch.save(policy_model.state_dict(), save_file)
    print(f"💾 Saved policy model to {save_file}")


def make_dummy_prompts(path: str, n: int = 50):
    """生成示例prompt数据"""
    prompts = [
        "解释什么是机器学习",
        "如何学习Python编程",
        "介绍一下深度学习的基本概念",
        "什么是强化学习",
        "解释神经网络的工作原理",
        "如何优化模型性能",
        "什么是过拟合",
        "介绍一下Transformer架构",
        "解释注意力机制",
        "什么是迁移学习",
    ]
    
    samples = []
    for i in range(n):
        prompt = prompts[i % len(prompts)] + f" (示例 {i})"
        samples.append({'prompt': prompt})
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {n} dummy prompts at {path}")


def parse_args():
    p = argparse.ArgumentParser(description='GRPO (Group Relative Policy Optimization) Demo')
    
    # 模型配置
    p.add_argument('--base_model', type=str, default='gpt2', help='基础语言模型')
    p.add_argument('--tokenizer_dir', type=str, default='gpt2', help='tokenizer路径')
    p.add_argument('--reward_model_path', type=str, default=None, 
                   help='Reward model checkpoint路径（可选，不提供则使用dummy rewards）')
    
    # GRPO参数
    p.add_argument('--group_size', type=int, default=4, help='每个问题采样的输出数量G')
    p.add_argument('--reward_mode', type=str, default='orm', choices=['orm', 'prm'],
                   help='奖励模式：orm (整体输出) 或 prm (过程奖励)')
    
    # 训练参数
    p.add_argument('--data_path', type=str, default='./class/lec15/demo_prompts.jsonl',
                   help='Prompt数据集路径')
    p.add_argument('--epochs', type=int, default=1, help='训练轮数')
    p.add_argument('--batch_size', type=int, default=2, help='批次大小（问题数量）')
    p.add_argument('--lr', type=float, default=1e-5, help='学习率')
    p.add_argument('--max_seq_len', type=int, default=128, help='最大序列长度')
    p.add_argument('--max_gen_len', type=int, default=64, help='最大生成长度')
    
    # PPO/GRPO参数
    p.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip范围')
    p.add_argument('--kl_coef', type=float, default=0.1, help='KL散度惩罚系数')
    p.add_argument('--advantage_eps', type=float, default=1e-8, help='优势函数归一化epsilon')
    
    # 其他
    p.add_argument('--device', type=str, default='cuda:0', help='设备')
    p.add_argument('--save_path', type=str, default='./class/lec15/out', help='保存路径')
    p.add_argument('--log_interval', type=int, default=5, help='日志间隔')
    p.add_argument('--seed', type=int, default=42, help='随机种子')
    p.add_argument('--make_dummy', action='store_true', help='生成示例数据')
    
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # 生成示例数据
    if args.make_dummy and not os.path.exists(args.data_path):
        make_dummy_prompts(args.data_path, n=20)
    
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        print("Use --make_dummy to create demo data")
        sys.exit(1)
    
    # 构建配置
    config = GRPOConfig(
        base_model=args.base_model,
        tokenizer_dir=args.tokenizer_dir,
        reward_model_path=args.reward_model_path,
        group_size=args.group_size,
        reward_mode=args.reward_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len,
        clip_epsilon=args.clip_epsilon,
        kl_coef=args.kl_coef,
        advantage_eps=args.advantage_eps,
        device=args.device,
        save_path=args.save_path,
        log_interval=args.log_interval,
        seed=args.seed,
        data_path=args.data_path
    )
    
    # 开始训练
    train_grpo(config)
