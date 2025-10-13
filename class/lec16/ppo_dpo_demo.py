"""
ppo_dpo_demo.py

教学级别的 PPO / DPO / GRPO 演示。

用法示例：
  python ppo_dpo_demo.py --method dpo --data_path ./class/lec14/demo_pref.jsonl --base_model distilgpt2 --epochs 2

说明：
- DPO: 直接使用偏好对 (x, y+, y-) 更新 policy，目标参照讲义公式。
- GRPO: 在 DPO 基础上加入额外的 reward 项（alpha * R），将 log-ratio 和 reward 混合。
- PPO: 教学级的近端策略优化示例（基于采样 + clip），用于对比。不是生产级 PPO。

"""

import os
import sys
import json
import argparse
from typing import List


def make_dummy_pref(path: str, n: int = 200):
    sample = []
    for i in range(n):
        prompt = f"请给出一句鼓励的话，场景编号 {i}。"
        chosen = f"你很棒，继续努力！ (示例 {i})"
        rejected = f"嗯，好吧。 (示例 {i})"
        sample.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected})
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for s in sample:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')


def compute_seq_logp(model, input_ids, attention_mask, device, detach=True):
    # sum log probs over tokens for each sequence
    import torch
    if detach:
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            logits = outputs.logits
    else:
        outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        logits = outputs.logits

    shifted_logits = logits[:, :-1, :]
    labels = input_ids[:, 1:].to(device)
    logp = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
    token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    seq_logp = token_logp.sum(dim=1)
    return seq_logp


def load_pairwise(path: str):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            prompt = j.get('prompt', j.get('text', ''))
            chosen = j.get('chosen')
            rejected = j.get('rejected')
            if isinstance(chosen, list):
                chosen = '\n'.join([t.get('content', t) if isinstance(t, dict) else t for t in chosen])
            if isinstance(rejected, list):
                rejected = '\n'.join([t.get('content', t) if isinstance(t, dict) else t for t in rejected])
            if chosen is None or rejected is None:
                continue
            items.append((prompt, chosen, rejected))
    return items


def train_dpo(args):
    # DPO: direct preference optimization
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '<pad>'

    policy = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    ref = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    ref.eval()

    dataset = load_pairwise(args.data_path)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for i, (prompt, y_pos, y_neg) in enumerate(dataset):
            # build full texts
            text_pos = f"User: {prompt}\nAssistant: {y_pos}"
            text_neg = f"User: {prompt}\nAssistant: {y_neg}"
            enc_pos = tokenizer(text_pos, return_tensors='pt', truncation=True, padding=True, max_length=args.max_seq_len)
            enc_neg = tokenizer(text_neg, return_tensors='pt', truncation=True, padding=True, max_length=args.max_seq_len)

            lp_pos = compute_seq_logp(policy, enc_pos['input_ids'], enc_pos['attention_mask'], device, detach=False)
            lp_neg = compute_seq_logp(policy, enc_neg['input_ids'], enc_neg['attention_mask'], device, detach=False)

            with torch.no_grad():
                lpr_pos = compute_seq_logp(ref, enc_pos['input_ids'], enc_pos['attention_mask'], device, detach=True)
                lpr_neg = compute_seq_logp(ref, enc_neg['input_ids'], enc_neg['attention_mask'], device, detach=True)

            # f = logpi - logpi_ref
            f_pos = lp_pos - lpr_pos
            f_neg = lp_neg - lpr_neg

            # DPO loss: -log sigmoid(beta * (f_pos - f_neg))
            diff = args.beta * (f_pos - f_neg)
            loss = -torch.log(torch.sigmoid(diff) + 1e-12).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            if (i + 1) % args.log_interval == 0:
                print(f"epoch={epoch} step={i+1}/{len(dataset)} loss={total_loss/(i+1):.6f}")

    os.makedirs(args.save_path, exist_ok=True)
    policy.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print('Saved policy (DPO) to', args.save_path)


def train_grpo(args):
    # GRPO: generalized form that adds alpha * reward into f_theta
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '<pad>'

    policy = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    ref = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    ref.eval()

    dataset = load_pairwise(args.data_path)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    # reward function: if a saved RM head is provided, use it; else a toy heuristic
    def reward_for_texts(texts: List[str]):
        import torch as _t
        if args.rm_mode == 'rm' and args.rm_head_path:
            from transformers import AutoModel
            rm_tok = AutoTokenizer.from_pretrained(args.rm_backbone)
            if getattr(rm_tok, 'pad_token', None) is None:
                if getattr(rm_tok, 'eos_token', None) is not None:
                    rm_tok.pad_token = rm_tok.eos_token
                else:
                    rm_tok.add_special_tokens({'pad_token': '<pad>'})
            rm = AutoModel.from_pretrained(args.rm_backbone).to(device)
            state = torch.load(args.rm_head_path, map_location=device)
            head = torch.nn.Linear(rm.config.hidden_size, 1).to(device)
            head.load_state_dict(state['head_state_dict'])
            enc = rm_tok(texts, return_tensors='pt', truncation=True, padding=True, max_length=args.max_seq_len)
            out = rm(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device), return_dict=True)
            last = out.last_hidden_state
            lengths = enc['attention_mask'].sum(dim=1) - 1
            lengths = lengths.clamp(min=0)
            batch_idx = torch.arange(enc['input_ids'].size(0), device=device)
            eos_h = last[batch_idx, lengths, :]
            scores = head(eos_h).squeeze(-1)
            return scores
        else:
            # toy: longer text slightly higher
            res = []
            for t in texts:
                r = len(t.split()) / 50.0
                if any(k in t for k in ['很好', '棒', '优秀', '赞']):
                    r += 0.5
                res.append(r)
            return torch.tensor(res, dtype=torch.float32, device=device)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for i, (prompt, y_pos, y_neg) in enumerate(dataset):
            text_pos = f"User: {prompt}\nAssistant: {y_pos}"
            text_neg = f"User: {prompt}\nAssistant: {y_neg}"
            enc_pos = tokenizer(text_pos, return_tensors='pt', truncation=True, padding=True, max_length=args.max_seq_len)
            enc_neg = tokenizer(text_neg, return_tensors='pt', truncation=True, padding=True, max_length=args.max_seq_len)

            lp_pos = compute_seq_logp(policy, enc_pos['input_ids'], enc_pos['attention_mask'], device, detach=False)
            lp_neg = compute_seq_logp(policy, enc_neg['input_ids'], enc_neg['attention_mask'], device, detach=False)

            with torch.no_grad():
                lpr_pos = compute_seq_logp(ref, enc_pos['input_ids'], enc_pos['attention_mask'], device, detach=True)
                lpr_neg = compute_seq_logp(ref, enc_neg['input_ids'], enc_neg['attention_mask'], device, detach=True)

            # reward terms
            rewards = reward_for_texts([text_pos, text_neg])
            r_pos = rewards[0]
            r_neg = rewards[1]

            f_pos = lp_pos - lpr_pos + args.alpha * r_pos
            f_neg = lp_neg - lpr_neg + args.alpha * r_neg

            diff = args.beta * (f_pos - f_neg)
            loss = -torch.log(torch.sigmoid(diff) + 1e-12).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            if (i + 1) % args.log_interval == 0:
                print(f"epoch={epoch} step={i+1}/{len(dataset)} loss={total_loss/(i+1):.6f}")

    os.makedirs(args.save_path, exist_ok=True)
    policy.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print('Saved policy (GRPO) to', args.save_path)


def train_ppo(args):
    # Simple illustrative PPO: generate responses and optimize with clipped objective
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '<pad>'

    policy = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    ref = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    ref.eval()

    # prompts to generate from
    items = load_pairwise(args.data_path)
    prompts = [p for p, _, _ in items]

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    import random
    for epoch in range(args.epochs):
        random.shuffle(prompts)
        for i in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[i:i+args.batch_size]
            # generate from current policy
            enc = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True, max_length=args.max_seq_len)
            input_ids = enc['input_ids'].to(device)
            att = enc['attention_mask'].to(device)
            gen = policy.generate(input_ids=input_ids, attention_mask=att, max_new_tokens=args.max_new_tokens, do_sample=True, top_k=50, pad_token_id=tokenizer.eos_token_id)
            texts = [tokenizer.decode(g, skip_special_tokens=True) for g in gen]

            # compute logp under current policy and ref and also old_logp as detached copy
            enc_full = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=args.max_seq_len)
            input_ids_full = enc_full['input_ids']
            att_full = enc_full['attention_mask']

            logp = compute_seq_logp(policy, input_ids_full, att_full, device, detach=False)
            with torch.no_grad():
                logp_ref = compute_seq_logp(ref, input_ids_full, att_full, device, detach=True)
                logp_old = logp.detach().clone()

            # rewards: use toy heuristic
            rewards = []
            for t in texts:
                r = len(t.split()) / 50.0
                if any(k in t for k in ['很好', '棒', '优秀', '赞']):
                    r += 0.5
                rewards.append(r)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

            # baseline and advantages
            baseline = rewards.mean()
            advantages = rewards - baseline
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False) + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            ratios = torch.exp(logp - logp_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantages
            pg_loss = -torch.min(surr1, surr2).mean()

            # kl penalty approximate
            kl = (logp - logp_ref).mean()
            loss = pg_loss + args.beta * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            opt.step()

            print(f"epoch={epoch} loss={loss.item():.4f} pg={pg_loss.item():.4f} kl={kl.item():.4f} mean_r={rewards.mean().item():.4f}")

    os.makedirs(args.save_path, exist_ok=True)
    policy.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print('Saved policy (PPO) to', args.save_path)


def parse_args():
    p = argparse.ArgumentParser(description='PPO / DPO / GRPO demo for lecture 16')
    p.add_argument('--method', type=str, default='dpo', choices=['dpo', 'grpo', 'ppo'], help='Which algorithm to run')
    p.add_argument('--data_path', type=str, default='./class/lec14/demo_pref.jsonl')
    p.add_argument('--base_model', type=str, default='distilgpt2')
    p.add_argument('--tokenizer', type=str, default='distilgpt2')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--beta', type=float, default=1.0, help='temperature/scale for DPO/GRPO or KL weight in PPO')
    p.add_argument('--alpha', type=float, default=0.0, help='GRPO reward weight')
    p.add_argument('--rm_mode', type=str, default='toy', choices=['toy', 'rm'], help='reward source for GRPO')
    p.add_argument('--rm_backbone', type=str, default='gpt2')
    p.add_argument('--rm_head_path', type=str, default='./class/lec14/out/reward_demo_head.pth')
    p.add_argument('--save_path', type=str, default='./class/lec16/out_policy')
    p.add_argument('--max_seq_len', type=int, default=128)
    p.add_argument('--max_new_tokens', type=int, default=32)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--make_dummy', action='store_true')
    # PPO specific
    p.add_argument('--clip_eps', type=float, default=0.2)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.make_dummy and not os.path.exists(args.data_path):
        print('Create dummy preference file ->', args.data_path)
        make_dummy_pref(args.data_path, n=200)

    if not os.path.exists(args.data_path):
        print('Data file not found:', args.data_path)
        print('Use --make_dummy to create a small demo dataset and retry')
        sys.exit(1)

    if args.method == 'dpo':
        train_dpo(args)
    elif args.method == 'grpo':
        train_grpo(args)
    elif args.method == 'ppo':
        train_ppo(args)
    else:
        raise ValueError('unknown method')


if __name__ == '__main__':
    main()
