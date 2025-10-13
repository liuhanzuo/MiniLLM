"""
rlhf_demo.py

第15讲 RLHF 简易演示（教学用）

特性：
- 使用 transformers causal LM 作为策略（policy）模型，支持生成采样并计算对应 log-prob
- 支持两个 reward 模式：toy heuristic（长度或关键词）或使用已保存的 Reward Model head（需提供 backbone）
- 实现 KL penalty、优势归一化（advantage normalization）以及简单的 off-policy importance-weighted loss（可选）
- 训练目标为最小化: -E[adv * logpi] + beta * KL(pi||pi_ref)

注意：此脚本用于教学与演示，不是生产级 PPO 实现。用于理解关键概念（KL、adv norm、off-policy 权重）。
"""

import os
import sys
import json
import argparse
from typing import List

# 不在模块顶层导入 heavy 库，确保 --help 可用


def make_dummy_prompts(path: str, n: int = 100):
    samples = []
    for i in range(n):
        samples.append({'prompt': f'请给出一句鼓励的话，场景编号 {i}。'})
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')


def compute_logprobs(model, input_ids, attention_mask, device, detach: bool = True):
    # compute per-sequence log probability (sum over tokens) under model
    # detach=True -> use no_grad (for ref policy); detach=False -> allow gradient flow (for current policy)
    import torch
    if detach:
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            logits = outputs.logits  # (B, T, V)
    else:
        # keep gradients enabled so logp can backprop through model
        outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        logits = outputs.logits

    # shift tokens to get conditional logits for tokens 1..T-1
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:].to(device)
    log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
    # gather
    token_logp = log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    seq_logp = token_logp.sum(dim=1)  # (B,)
    return seq_logp


def generate_responses(model, tokenizer, prompts: List[str], device, max_new_tokens=32, do_sample=True, top_k=50):
    # tokenizes prompts and generates responses; returns list of full texts and input_ids tensors
    import torch
    enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    gen = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                         do_sample=do_sample, top_k=top_k, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)

    # For each generation, we return the full sequence (input + generated)
    texts = [tokenizer.decode(g, skip_special_tokens=True) for g in gen]
    # convert to tensors for logprob computation
    return texts, gen, None


def compute_rewards_batch(reward_mode, prompts: List[str], generations: List[str], args, device):
    # two modes: 'toy' or 'rm' (reward model)
    import torch
    rewards = []
    if reward_mode == 'toy':
        # simple heuristic: longer responses get slightly higher reward; presence of positive words gets boost
        for p, g in zip(prompts, generations):
            r = len(g.split()) / 50.0
            if any(k in g for k in ['很好', '棒', '优秀', '赞']):
                r += 0.5
            rewards.append(float(r))
        return torch.tensor(rewards, dtype=torch.float32, device=device)
    elif reward_mode == 'rm':
        # load reward model head + backbone if provided; compute scalar via last token hidden
        from transformers import AutoTokenizer, AutoModel
        from torch import nn
        # try to load backbone
        rm_backbone = args.rm_backbone
        head_path = args.rm_head_path
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        rm = AutoModel.from_pretrained(rm_backbone).to(device)
        # load head state
        state = torch.load(head_path, map_location=device)
        head = nn.Linear(rm.config.hidden_size, 1).to(device)
        head.load_state_dict(state['head_state_dict'])

        enc = tokenizer(generations, return_tensors='pt', padding=True, truncation=True, max_length=args.max_seq_len)
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            out = rm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last_hidden = out.last_hidden_state
            lengths = attention_mask.sum(dim=1) - 1
            lengths = lengths.clamp(min=0)
            batch_idx = torch.arange(input_ids.size(0), device=device)
            eos_h = last_hidden[batch_idx, lengths, :]
            scores = head(eos_h).squeeze(-1)
        return scores
    else:
        raise ValueError('Unknown reward_mode')


def train(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    # tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '<pad>'

    policy = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    # reference policy (pi_ref) is a frozen copy of initial policy
    ref_policy = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    ref_policy.eval()

    # prepare prompts
    prompts = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            prompts.append(j.get('prompt', j.get('text', '')))

    # minibatch generator
    def prompt_batches(prompts, batch_size):
        for i in range(0, len(prompts), batch_size):
            yield prompts[i:i+batch_size]

    # running baseline (simple moving average) for advantage baseline
    running_baseline = 0.0
    baseline_alpha = 0.9

    for epoch in range(args.epochs):
        for batch_prompts in prompt_batches(prompts, args.batch_size):
            # generate responses
            texts, gens, _ = generate_responses(policy, tokenizer, batch_prompts, device, max_new_tokens=args.max_new_tokens, do_sample=True)

            # compute logprobs under current policy and ref policy
            enc_full = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=args.max_seq_len)
            input_ids = enc_full['input_ids']
            attention_mask = enc_full['attention_mask']

            # compute logprobs: policy (with grad), ref_policy (no grad)
            logp = compute_logprobs(policy, input_ids, attention_mask, device, detach=False)
            logp_ref = compute_logprobs(ref_policy, input_ids, attention_mask, device, detach=True)

            # compute rewards
            rewards = compute_rewards_batch(args.reward_mode, batch_prompts, texts, args, device)

            # baseline: running moving average
            running_baseline = baseline_alpha * running_baseline + (1 - baseline_alpha) * rewards.mean().item()
            advantages = rewards - running_baseline

            # advantage normalization
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False) + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            # off-policy importance weights (optional)
            if args.use_is:
                ratios = torch.exp(logp - logp_ref)
                if args.clip_ratio > 0:
                    ratios = torch.clamp(ratios, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                pg_loss = - (ratios * advantages).mean()
            else:
                pg_loss = - (logp * advantages).mean()

            # KL penalty: approximate by average (logp - logp_ref)
            kl = (logp - logp_ref).mean()
            loss = pg_loss + args.beta * kl

            # backprop
            policy.train()
            policy.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            # optimizer step
            if not hasattr(train, '_optim'):
                train._optim = torch.optim.AdamW(policy.parameters(), lr=args.lr)
            train._optim.step()

            print(f"epoch={epoch} loss={loss.item():.4f} pg={pg_loss.item():.4f} kl={kl.item():.4f} mean_reward={rewards.mean().item():.4f}")

    # save policy checkpoint
    os.makedirs(args.save_path, exist_ok=True)
    policy.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print('Saved policy to', args.save_path)


def parse_args():
    p = argparse.ArgumentParser(description='RLHF demo (KL penalty + advantage norm + off-policy options)')
    p.add_argument('--data_path', type=str, default='./class/lec15/demo_prompts.jsonl')
    p.add_argument('--base_model', type=str, default='distilgpt2')
    p.add_argument('--tokenizer', type=str, default='distilgpt2')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_new_tokens', type=int, default=32)
    p.add_argument('--max_seq_len', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--beta', type=float, default=0.05, help='KL penalty coefficient')
    p.add_argument('--reward_mode', type=str, default='toy', choices=['toy', 'rm'])
    p.add_argument('--rm_backbone', type=str, default='gpt2', help='reward model backbone (if reward_mode=rm)')
    p.add_argument('--rm_head_path', type=str, default='./class/lec14/out/reward_demo_head.pth', help='reward model head path')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--save_path', type=str, default='./class/lec15/out_policy')
    p.add_argument('--use_is', action='store_true', help='use importance sampling weights')
    p.add_argument('--clip_ratio', type=float, default=0.2, help='clip ratio for IS')
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--make_dummy', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.make_dummy and not os.path.exists(args.data_path):
        print('Create dummy prompt file ->', args.data_path)
        make_dummy_prompts(args.data_path, n=200)

    if not os.path.exists(args.data_path):
        print('Data file not found:', args.data_path)
        print('Use --make_dummy to create a small demo dataset and retry')
        sys.exit(1)

    train(args)
