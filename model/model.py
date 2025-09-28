import math
import struct
import inspect
import time

from .LMConfig import LMConfig
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:, :, :seq_len, :seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoDRouter(nn.Module):
    """
    MoD (Mixture of Depths) Router per recursion step.
    Each step independently decides which tokens pass through (mask in {0,1}).
    Supports expert_choice (top-k per step) and sigmoid_threshold routing.
    """

    def __init__(self, config: LMConfig, nr_steps: int):
        super().__init__()
        self.hidden = config.dim
        self.nr_steps = nr_steps
        self.routing = config.mor_routing
        self.cap_ratio = config.mor_cap_ratio
        self.tau = config.mor_tau
        self.temperature = config.mor_temperature
        self.aux_loss_alpha = config.mor_aux_loss_alpha

        # lightweight gate net
        self.proj = nn.Linear(self.hidden, 1)
        self.step_embed = nn.Embedding(num_embeddings=nr_steps, embedding_dim=1)

    def _capacity_for_step(self, step: int, total_tokens: int) -> int:
        if self.cap_ratio is not None and step < len(self.cap_ratio):
            return max(1, int(round(total_tokens * float(self.cap_ratio[step]))))
        # default geometric decay: [1.0, 2/3, 1/3, ...]
        ratios = [max(1.0/(i+1), 0.1) for i in range(self.nr_steps)]
        r = ratios[step] if step < len(ratios) else 0.1
        return max(1, int(round(total_tokens * r)))

    def forward(self, x: torch.Tensor, step: int):
        """
        x: [B,S,H]
        Returns: gate g in [0,1], mask m in {0,1}, aux_loss
        """
        B, S, H = x.shape
        scores = self.proj(x).squeeze(-1)  # [B,S]
        scores = scores + self.step_embed.weight[step].view(1, 1)
        if self.temperature != 1.0:
            scores = scores / self.temperature

        aux_loss = x.new_zeros(())
        total_tokens = B * S

        if self.routing == 'expert_choice':
            # select top-k tokens across (B*S) independently per step
            k = self._capacity_for_step(step, total_tokens)
            flat = scores.reshape(-1)  # [B*S]
            topk_vals, topk_idx = torch.topk(flat, k=k, dim=0)
            m = torch.zeros_like(flat, dtype=x.dtype)
            m[topk_idx] = 1.0
            m = m.view(B, S, 1)
            g = torch.sigmoid(scores).unsqueeze(-1)  # gate for info only
            # balance loss
            if self.aux_loss_alpha > 0 and self.training:
                target = torch.tensor(k/total_tokens, dtype=x.dtype, device=x.device)
                used = m.mean()
                aux_loss = self.aux_loss_alpha * (used - target).pow(2)
            return g, m, aux_loss
        else:  # sigmoid_threshold
            g = torch.sigmoid(scores).unsqueeze(-1)  # [B,S,1]
            m = (g > self.tau).to(dtype=x.dtype)
            # capacity enforcement (keep highest gates)
            cap = self._capacity_for_step(step, total_tokens)
            if m.sum().item() > cap:
                flat_g = g.view(-1)
                topk_vals, topk_idx = torch.topk(flat_g, k=cap, dim=0)
                m = torch.zeros_like(flat_g, dtype=x.dtype)
                m[topk_idx] = 1.0
                m = m.view(B, S, 1)
            if self.aux_loss_alpha > 0 and self.training:
                target = torch.tensor(cap/total_tokens, dtype=x.dtype, device=x.device)
                used = m.mean()
                aux_loss = self.aux_loss_alpha * (used - target).pow(2)
            return g, m, aux_loss


class MoRFeedForward(nn.Module):
    """
    MoR (Mixture of Recursions) FeedForward with shared parameters and per-step routing.
    Each token can independently choose which recursion steps to pass through.
    """
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.nr_steps = config.nr_steps
        
        # Shared recursion block (Middle-Cycle: same parameters for all steps)
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        self.shared_ffn = nn.ModuleDict({
            'w1': nn.Linear(config.dim, config.hidden_dim, bias=False),
            'w2': nn.Linear(config.hidden_dim, config.dim, bias=False),
            'w3': nn.Linear(config.dim, config.hidden_dim, bias=False)
        })
        self.dropout = nn.Dropout(config.dropout)
        self.router = MoDRouter(config, self.nr_steps)
        self.aux_loss = 0.0

    def _ffn_forward(self, x):
        """Shared FFN computation f(x)"""
        return self.dropout(self.shared_ffn['w2'](F.silu(self.shared_ffn['w1'](x)) * self.shared_ffn['w3'](x)))

    def forward(self, x):
        """
        x: [B,S,H]
        Returns: output after nr_steps recursions with per-step routing
        """
        h = x
        self.aux_loss = 0.0
        
        for step in range(self.nr_steps):
            g_step, m_step, aux_step = self.router(h, step)  # m_step: [B,S,1]
            y = self._ffn_forward(h)  # shared computation
            h = h + m_step * y  # h_{step+1} = h_{step} + m_{step} * f(h_{step})
            self.aux_loss = self.aux_loss + aux_step
            
        return h


class MoEGate(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # hidden_states形状: [batch_size, seq_len, hidden_dim]
        bsz, seq_len, h = hidden_states.shape
        
        # 【步骤1】形状变换：合并batch和序列维度用于并行处理单个token
        hidden_states = hidden_states.view(-1, h)  # new_shape: [batch_size*seq_len, hidden_dim]
        
        # 【步骤2】计算专家分数
        logits = F.linear(hidden_states, self.weight, None)  # [bs*seq_len, n_experts]
        
        # 【步骤3】应用评分函数
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)  # 使用softmax得到每个token的专家分布
        
        # 【步骤4】Top-k专家选择
        topk_weight, topk_idx = torch.topk(
            scores, 
            k=self.top_k,       # 每个token选择k个专家
            dim=-1,             # 专家维度
            sorted=False        # 不需要排序，提高效率
        )  # 形状均为[bs*seq_len, top_k]
    
        # 【步骤5】可选Top-k归一化（确保权重和为1）
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20  # 防止除以零
            topk_weight = topk_weight / denominator
            
        # 【步骤6】辅助损失计算（训练时且需要平衡损失时）
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores  # 原始分数用于辅助损失计算
            aux_topk = self.top_k
            
            # 转换topk_index到对应的形状
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)  # [bs, seq_len*topk]
    
            if self.seq_aux:
                """
                序列级辅助损失（缓解序列不同位置专家分布差异过大问题）
                计算流程：
                1. 将分数重塑为 [batch, seq_len, n_experts]
                2. 统计每个batch中专家被选择的比例(CE)
                3. 计算(平均分数*选择比例)的总和
                """
                # [batch_size, seq_len, n_experts]
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                
                ce = torch.zeros(
                    bsz, 
                    self.n_routed_experts, 
                    device=hidden_states.device
                )
                # 使用scatter_add统计每个batch的专家选择计数
                ce.scatter_add_(
                    dim=1,
                    index=topk_idx_for_aux_loss,
                    src=torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)
                ).div_(seq_len * aux_topk / self.n_routed_experts)  # 归一化
                
                # 计算辅助损失: 平均分数矩阵和选择比例的逐元素乘法
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                """
                Token级辅助损失（标准负载平衡机制）:
                计算流程：
                1. 统计所有token中的专家选择分布
                2. 约束专家选择分布与分数分布之间的差异
                """
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), 
                    num_classes=self.n_routed_experts
                )  # [bs*seq_len*topk, n_experts]
                
                ce = mask_ce.float().mean(0)      # 专家选择频率 [n_experts]
                Pi = scores_for_aux.mean(0)       # 平均专家分数 [n_experts]
                fi = ce * self.n_routed_experts   # 标准化频率
                aux_loss = (Pi * fi).sum() * self.alpha  # 目标使分数分布与选择频率一致
        else:
            # 推理模式或无辅助损失时
            aux_loss = 0
            
        # 返回结果：专家索引，门控权重，辅助损失
        return topk_idx, topk_weight, aux_loss



class MOEFeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 训练模式下，重复输入数据
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式下，只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 例如当tokens_per_expert=[6, 15, 20, 26, 33, 38, 46, 52]
        # 当token_idxs=[3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...]
        # 意味着当token_idxs[:6] -> [3,  7, 19, 21, 24, 25,  4]位置的token都由专家0处理，token_idxs[6:15]位置的token都由专家1处理......
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 进行 sum 操作
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniLLMBlock(nn.Module):
    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        # Choose feedforward type: MoR, MoE, or standard
        if config.use_mor:
            self.feed_forward = MoRFeedForward(config)
        elif config.use_moe:
            self.feed_forward = MOEFeedForward(config)
        else:
            self.feed_forward = FeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        h = x + h_attn
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv


class MiniLLMLM(PreTrainedModel):
    config_class = LMConfig

    def __init__(self, params: LMConfig = None):
        self.params = params or LMConfig()
        super().__init__(self.params)
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        # Build layers: new block-based repeat mapping if n_block is provided, else optional repeat-layer sharing.
        if getattr(params, 'n_block', None) is not None and params.n_block > 0:
            # Blocked pattern: each block has 8 virtual layers with mapping [0,1,2,3,1,2,3,4] to 5 unique physical layers
            pattern = [0, 1, 2, 3, 1, 2, 3, 4]
            n_block = params.n_block
            self.layers_unique = nn.ModuleList()
            self.layers = nn.ModuleList()
            self.layer_index_map = []
            for b in range(n_block):
                # 5 unique layers per block
                base = len(self.layers_unique)
                block_unique = [MiniLLMBlock(base + i, params) for i in range(5)]
                self.layers_unique.extend(block_unique)
                # expand virtual layers for this block using mapping to this block's unique layers
                for idx in pattern:
                    phys_idx = base + idx
                    self.layers.append(self.layers_unique[phys_idx])
                    self.layer_index_map.append(phys_idx)
        elif getattr(params, 'repeat_layer', False):
            # Pattern: 0,1,2,3,1,2,3,4 over virtual layers [0..n_layers-1]
            # Unique physical blocks are 0..4; if n_layers < 5, fall back to normal distinct layers.
            if self.n_layers >= 5:
                self._phys_layers = 5
                # Create 5 unique blocks (0..4)
                unique_blocks = [MiniLLMBlock(i, params) for i in range(self._phys_layers)]
                self.layers_unique = nn.ModuleList(unique_blocks)
                # Build virtual->physical index map
                layer_index_map = [(v if v < 4 else ((v - 4) % 4) + 1) for v in range(self.n_layers)]
                self.layer_index_map = layer_index_map
                # Expose a proxy list that indexes into unique blocks
                self.layers = nn.ModuleList([self.layers_unique[idx] for idx in self.layer_index_map])
            else:
                # For small n_layers, keep distinct blocks
                self.layers = nn.ModuleList([MiniLLMBlock(l, params) for l in range(self.n_layers)])
                self.layer_index_map = list(range(self.n_layers))
        else:
            self.layers = nn.ModuleList([MiniLLMBlock(l, params) for l in range(self.n_layers)])
            self.layer_index_map = list(range(self.n_layers))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
                             persistent=False)
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get('start_pos', 0)
        h = self.dropout(self.tok_embeddings(input_ids))
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        past_kvs = []
        use_ckpt = bool(getattr(self.params, 'gradient_checkpointing', False)) and self.training and (h.requires_grad)
        if use_ckpt:
            from torch.utils.checkpoint import checkpoint
        for l, layer in enumerate(self.layers):
            if use_ckpt:
                # checkpoint 只支持 Tensor 输出；我们仅对隐状态 h 做检查点，缓存不做检查点以降低复杂度
                def _layer_fn(x, pos):
                    out, kv = layer(x, pos, past_key_value=past_key_values[l], use_cache=use_cache)
                    # 将 kv 暂存到外部列表，保持与未 checkpoint 时一致
                    past_kvs.append(kv)
                    return out
                h = checkpoint(_layer_fn, h, pos_cis, use_reentrant=False)
            else:
                h, past_kv = layer(
                    h, pos_cis,
                    past_key_value=past_key_values[l],
                    use_cache=use_cache
                )
                past_kvs.append(past_kv)
        logits = self.output(self.norm(h))
        # When layers are shared, aux_loss should be accumulated once per unique block per call.
        if hasattr(self, 'layers_unique'):
            aux_loss = sum(
                b.feed_forward.aux_loss for b in self.layers_unique 
                if isinstance(b.feed_forward, (MOEFeedForward, MoRFeedForward))
            )
        else:
            aux_loss = sum(
                l.feed_forward.aux_loss for l in self.layers 
                if isinstance(l.feed_forward, (MOEFeedForward, MoRFeedForward))
            )
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        # 流式生成
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 直接生成
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        return torch.cat(generated, dim=0)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                           start_pos=input_ids.shape[1] - 1, **args)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= (temperature + 1e-9)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break
