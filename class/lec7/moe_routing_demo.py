import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse

class MoEGate(nn.Module):
    def __init__(self, n_experts, k=1, routing='softmax_topk', noisy=False, group_size=None, sinkhorn_iters=0, device_groups=None, aux_loss_alpha=0.0, temperature=1.0):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.routing = routing
        self.noisy = noisy
        self.group_size = group_size
        self.sinkhorn_iters = sinkhorn_iters
        self.device_groups = device_groups
        self.aux_loss_alpha = aux_loss_alpha
        self.temperature = temperature
        self.gate = nn.Linear(128, n_experts)  # 假设输入维度128

    def forward(self, x):
        # x: [batch, seq, hidden]
        scores = self.gate(x)  # [B, S, n_experts]
        if self.temperature != 1.0:
            scores = scores / self.temperature
        if self.noisy:
            scores = scores + torch.randn_like(scores) * 0.1
        if self.routing == 'softmax_topk':
            probs = F.softmax(scores, dim=-1)
            topk_weight, topk_idx = torch.topk(probs, self.k, dim=-1)
            # Softmax Top-k: 按概率加权聚合
            mask = torch.zeros_like(probs)
            mask.scatter_(-1, topk_idx, topk_weight)
            aux_loss = self._aux_loss(probs, mask) if self.aux_loss_alpha > 0 else 0.0
            return topk_idx, topk_weight, aux_loss, probs
        elif self.routing == 'switch':
            # Switch: 只选最大分数专家
            top1_idx = scores.argmax(dim=-1, keepdim=True)
            mask = torch.zeros_like(scores)
            mask.scatter_(-1, top1_idx, 1.0)
            aux_loss = self._aux_loss(F.softmax(scores, dim=-1), mask) if self.aux_loss_alpha > 0 else 0.0
            return top1_idx, torch.ones_like(top1_idx, dtype=x.dtype), aux_loss, mask
        elif self.routing == 'sinkhorn':
            # Sinkhorn/OT: 近似全局均衡分配
            probs = F.softmax(scores, dim=-1)
            sinkhorn_out = self._sinkhorn(probs, self.sinkhorn_iters)
            topk_idx = sinkhorn_out.argmax(dim=-1, keepdim=True)
            mask = sinkhorn_out
            aux_loss = self._aux_loss(probs, mask) if self.aux_loss_alpha > 0 else 0.0
            return topk_idx, mask.gather(-1, topk_idx), aux_loss, mask
        elif self.routing == 'expert_choice':
            # Expert-Choice/Group-Limited: 先分组再选 top-k
            group_mask = self._group_mask(scores)
            scores_group = scores.masked_fill(~group_mask, float('-inf'))
            probs = F.softmax(scores_group, dim=-1)
            topk_weight, topk_idx = torch.topk(probs, self.k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(-1, topk_idx, topk_weight)
            aux_loss = self._aux_loss(probs, mask) if self.aux_loss_alpha > 0 else 0.0
            return topk_idx, topk_weight, aux_loss, mask
        else:
            raise ValueError(f'Unknown routing: {self.routing}')

    def _aux_loss(self, probs, mask):
        # 均衡损失：鼓励所有专家被均匀选中
        freq = mask.float().mean(dim=(0,1))  # [n_experts]
        target = torch.full_like(freq, 1.0 / self.n_experts)
        return F.mse_loss(freq, target) * self.aux_loss_alpha

    def _sinkhorn(self, probs, iters):
        # 近似 Sinkhorn 算法（行/列归一化）
        x = probs
        for _ in range(iters):
            x = x / (x.sum(-1, keepdim=True) + 1e-6)
            x = x / (x.sum(-2, keepdim=True) + 1e-6)
        return x

    def _group_mask(self, scores):
        # 按 group_size 或 device_groups 生成 mask
        B, S, E = scores.shape
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if self.device_groups is not None:
            # device_groups: List[List[expert_idx]]
            for group in self.device_groups:
                for idx in group:
                    mask[..., idx] = True
        elif self.group_size is not None:
            for i in range(0, E, self.group_size):
                mask[..., i:i+self.group_size] = True
        else:
            mask[..., :] = True
        return mask

class MoEModel(nn.Module):
    def __init__(self, n_experts=8, routing='softmax_topk', k=2, aux_loss_alpha=0.0, temperature=1.0, **kwargs):
        super().__init__()
        self.gate = MoEGate(n_experts, k=k, routing=routing, aux_loss_alpha=aux_loss_alpha, temperature=temperature, **kwargs)
        self.experts = nn.ModuleList([nn.Linear(128, 128) for _ in range(n_experts)])

    def forward(self, x):
        # x: [batch, seq, hidden]
        topk_idx, topk_weight, aux_loss, mask = self.gate(x)
        # 聚合专家输出
        out = torch.zeros_like(x)
        for i in range(self.gate.n_experts):
            expert_out = self.experts[i](x)
            out += expert_out * mask[..., i:i+1]
        return out, aux_loss, mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoE routing demo with expert selection counts')
    parser.add_argument('--steps', type=int, default=500, help='训练步数')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seq', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--experts', type=int, default=8)
    parser.add_argument('--k', type=int, default=2, help='top-k 或 expert_choice 的 k')
    parser.add_argument('--out', type=str, default='moe_expert_balance_counts.png')
    parser.add_argument('--temperature', type=float, default=1.0, help='softmax 温度 (<1 更尖锐, >1 更平滑)')
    parser.add_argument('--gate_lr_scale', type=float, default=1.0, help='仅对 gating 参数的学习率放大倍数')
    parser.add_argument('--init_bias_boost', type=float, default=0.0, help='在无均衡损失(aux=0)时对第0号专家 bias 增强, 制造失衡种子')
    parser.add_argument('--print_first_steps', type=int, default=5, help='打印前若干步的分布以观察早期坍缩')
    args = parser.parse_args()

    torch.manual_seed(42)
    batch, seq, hidden = args.batch, args.seq, args.hidden
    n_experts = args.experts
    steps = args.steps
    k = args.k
    x = torch.randn(batch, seq, hidden)

    # 结果字典: (routing, aux) -> dict with histories
    results = {}
    stats_summary = {}
    total_token_selections_per_step = {
        'softmax_topk': batch * seq * k,
        'expert_choice': batch * seq * k,
        'switch': batch * seq,
        'sinkhorn': batch * seq,
    }

    def imbalance_metrics(p):
        # p: 频率分布 (numpy)
        # Gini
        n = len(p)
        diff_sum = 0.0
        for i in range(n):
            diff_sum += np.abs(p[i] - p).sum()
        gini = diff_sum / (2 * n * p.sum() + 1e-12)
        hhi = np.square(p).sum()  # Herfindahl-Hirschman Index
        max_min_ratio = (p.max() / (p.min() + 1e-12)) if p.min() > 0 else np.inf
        return gini, hhi, max_min_ratio

    for routing in ['softmax_topk', 'switch', 'sinkhorn', 'expert_choice']:
        for aux in [0.0, 0.1]:
            model = MoEModel(n_experts=n_experts, routing=routing, k=k, aux_loss_alpha=aux, temperature=args.temperature)
            # 初始化偏置以放大无 aux 时的失衡
            if aux == 0.0 and args.init_bias_boost != 0.0:
                with torch.no_grad():
                    model.gate.gate.bias.data[0] += args.init_bias_boost
            base_lr = 1e-3
            gate_params = list(model.gate.parameters())
            other_params = [p for n,p in model.named_parameters() if not n.startswith('gate.')]
            optimizer = torch.optim.Adam([
                {'params': other_params, 'lr': base_lr},
                {'params': gate_params, 'lr': base_lr * args.gate_lr_scale}
            ])
            freq_history = []      # 每步频率 (归一化)
            count_history = []     # 每步原始次数
            cumulative_counts = np.zeros(n_experts, dtype=np.int64)
            for step in range(steps):
                out, aux_loss, mask = model(x)
                loss = out.pow(2).mean() + aux_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 根据 routing 生成 one-hot 选择 (包含 top-k 多个选择)
                if model.gate.routing == 'switch':
                    top1_idx = mask.argmax(-1)  # [batch, seq]
                    onehot = F.one_hot(top1_idx, num_classes=n_experts).float()
                elif model.gate.routing == 'softmax_topk':
                    topk_idx = torch.topk(mask, model.gate.k, dim=-1).indices  # [batch, seq, k]
                    onehot = torch.zeros_like(mask)
                    onehot.scatter_(-1, topk_idx, 1.0)
                elif model.gate.routing == 'sinkhorn':
                    top1_idx = mask.argmax(-1)
                    onehot = F.one_hot(top1_idx, num_classes=n_experts).float()
                elif model.gate.routing == 'expert_choice':
                    topk_idx = torch.topk(mask, model.gate.k, dim=-1).indices
                    onehot = torch.zeros_like(mask)
                    onehot.scatter_(-1, topk_idx, 1.0)
                else:
                    raise ValueError('Unknown routing during stats collection')

                # 原始计数: 每个专家被选中的次数 (整数)
                counts = onehot.sum(dim=(0,1)).to(dtype=torch.int64).cpu().numpy()
                cumulative_counts += counts

                # 频率 (当前步)
                total_sel = total_token_selections_per_step[routing]
                freq = counts / total_sel

                freq_history.append(freq)
                count_history.append(counts.copy())

            freq_history = np.stack(freq_history)  # [steps, n_experts]
            count_history = np.stack(count_history)  # [steps, n_experts]
            results[(routing, aux)] = {
                'freq_history': freq_history,
                'count_history': count_history,
                'cumulative_counts': cumulative_counts.copy(),
            }

            final_freq = freq_history[-1]
            mean = final_freq.mean(); std = final_freq.std()
            gini, hhi, mmr = imbalance_metrics(final_freq)
            stats_summary[(routing, aux)] = (final_freq, mean, std, cumulative_counts.copy(), gini, hhi, mmr)
            print(f'Routing={routing}, Aux={aux}:')
            print(f'  最后一步频率={final_freq}')
            print(f'  累计计数(cumulative)={cumulative_counts}')
            print(f'  每步token选择总数={total_token_selections_per_step[routing]} (用于频率归一化)')
            print(f'  均值={mean:.4f}, 方差={std:.4f}, min={final_freq.min():.4f}, max={final_freq.max():.4f}')
            print(f'  Gini={gini:.4f}, HHI={hhi:.4f}, Max/Min={mmr:.2f}')

            if args.print_first_steps > 0:
                preview = freq_history[:args.print_first_steps]
                print(f'前{args.print_first_steps}步频率片段:')
                for sidx, pf in enumerate(preview):
                    print(f'step{ sidx }: {pf}')

    # 可视化：上排显示最后一步频率，下排显示累计计数
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14,8))
    colors = {0.0: 'tab:blue', 0.1: 'tab:orange'}
    for i, routing in enumerate(['softmax_topk', 'switch', 'sinkhorn', 'expert_choice']):
        for j, aux in enumerate([0.0, 0.1]):
            final_freq, mean, std, cumulative_counts, gini, hhi, mmr = stats_summary[(routing, aux)]
            variance = std ** 2
            # 频率 (含方差信息)
            plt.subplot(4,4,i*4 + j + 1)
            plt.bar(np.arange(n_experts), final_freq, color=colors[aux], alpha=0.75)
            plt.title(f'{routing}\naux={aux} freq\nstd={std:.3f} var={variance:.4f}\nG={gini:.3f} H={hhi:.3f}')
            plt.ylim(0,1)
            if i == 3:
                plt.xlabel('Expert')
            if j == 0:
                plt.ylabel('Freq (last step)')
            # 累计计数 (也可附加方差供直观看频率->计数映射)
            plt.subplot(4,4,i*4 + j + 3)
            plt.bar(np.arange(n_experts), cumulative_counts, color=colors[aux], alpha=0.75)
            total_sel_all_steps = total_token_selections_per_step[routing] * steps
            plt.title(f'{routing}\naux={aux} counts\nΣ={cumulative_counts.sum()}/{total_sel_all_steps}\nMax/Min={mmr:.1f}')
            if i == 3:
                plt.xlabel('Expert')
            if j == 0:
                plt.ylabel('Cumulative Count')

    plt.tight_layout()
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, args.out)
    plt.savefig(out_path)
    print(f'已保存专家分布对比图(频率+累计计数): {out_path}')
