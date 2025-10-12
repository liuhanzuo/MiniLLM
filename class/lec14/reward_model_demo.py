"""
reward_model_demo.py

一个最小的 Reward Model 演示（pairwise preference training）。

功能：
- 读取偏好数据（支持简单三元组格式或常见的 chosen/rejected 格式）
- 使用 Hugging Face 的 transformer 作为 backbone 并在 <eos>（或最后 token）上加一个线性头输出标量奖励
- 使用 pairwise ranking loss: -log sigmoid(r_w - r_l)
- 提供最小训练循环与保存/加载接口

这是教学/demo 用代码，不适合直接用于大规模训练（仅作示例）。
"""

import os
import sys
import json
import argparse
from typing import List, Dict

# 延迟导入 heavy ML 库（torch / transformers）到运行时
# 这样在未激活 conda 环境时仍可运行 `--help` 查看参数


class PreferenceDataset:
    """读取 pairwise 偏好数据的 Dataset。

    支持两种常见格式（每行一个 JSON）：
    1) 简单三元组：{"prompt": "...", "chosen": "...", "rejected": "..."}
    2) Magpie/ModelScope 风格：{"chosen": [{"role":"user","content":"..."}, ...], "rejected": [...]}
    """

    def __init__(self, path: str, tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.items = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                # normalize
                if 'prompt' in j and 'chosen' in j and 'rejected' in j:
                    prompt = j['prompt']
                    chosen = j['chosen']
                    rejected = j['rejected']
                elif 'chosen' in j and 'rejected' in j:
                    # chosen/rejected may be list of role/content dicts or strings
                    def flatten_conv(x):
                        if isinstance(x, list):
                            parts = []
                            for turn in x:
                                if isinstance(turn, dict) and 'content' in turn:
                                    parts.append(turn['content'])
                                elif isinstance(turn, str):
                                    parts.append(turn)
                            return '\n'.join(parts)
                        elif isinstance(x, str):
                            return x
                        return ''

                    chosen = flatten_conv(j['chosen'])
                    rejected = flatten_conv(j['rejected'])
                    prompt = j.get('prompt', '')
                else:
                    # try to interpret other formats conservatively
                    # skip if cannot parse
                    continue

                # build full input strings: user prompt + assistant response
                def make_pair(p, a):
                    if p:
                        return f"User: {p}\nAssistant: {a}"
                    else:
                        return f"Assistant: {a}"

                self.items.append((make_pair(prompt, chosen), make_pair(prompt, rejected)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, tokenizer, max_len=512):
    # batch: list of (winner_text, loser_text)
    winners, losers = zip(*batch)
    enc_w = tokenizer(list(winners), padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    enc_l = tokenizer(list(losers), padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    batch_out = {
        'w_input_ids': enc_w['input_ids'],
        'w_attention_mask': enc_w['attention_mask'],
        'l_input_ids': enc_l['input_ids'],
        'l_attention_mask': enc_l['attention_mask'],
    }
    return batch_out



def pairwise_loss(r_w, r_l, torch):
    # -log sigmoid(r_w - r_l)
    return -torch.log(torch.sigmoid(r_w - r_l) + 1e-12).mean()


def train(args):
    # 延迟导入 heavy 库，保证 --help 可用
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModel

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    # tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Ensure tokenizer has a pad_token. Prefer pointing pad_token to eos_token to avoid resizing embeddings.
    pad_added = False
    if tokenizer.pad_token is None:
        if getattr(tokenizer, 'eos_token', None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # add a new pad token (this requires resizing model embeddings after model load)
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            pad_added = True

    ds = PreferenceDataset(args.data_path, tokenizer, max_len=args.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_seq_len))

    # RewardModel uses AutoModel - define it here to capture imports
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

    model = RewardModelLocal(backbone_name=args.base_model)
    model.to(device)

    # If we added a pad token we must resize the backbone embeddings
    if pad_added:
        try:
            model.backbone.resize_token_embeddings(len(tokenizer))
        except Exception:
            # not all model classes implement resize_token_embeddings; if so, warn and continue
            print('Warning: could not resize token embeddings for backbone; you may see shape mismatch')

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            w_ids = batch['w_input_ids'].to(device)
            w_att = batch['w_attention_mask'].to(device)
            l_ids = batch['l_input_ids'].to(device)
            l_att = batch['l_attention_mask'].to(device)

            r_w = model(w_ids, w_att)
            r_l = model(l_ids, l_att)
            loss = pairwise_loss(r_w, r_l, torch)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            if (step + 1) % args.log_interval == 0:
                print(f"Epoch {epoch} step {step+1}/{len(loader)} loss={total_loss/(step+1):.4f}")

        print(f"Epoch {epoch} finished, avg loss={total_loss/len(loader):.4f}")

    # save small checkpoint (只保存 head + config 不保存大模型权重以免文件过大)
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    torch.save({'head_state_dict': model.head.state_dict(), 'backbone_name': args.base_model}, os.path.join(save_path, 'reward_demo_head.pth'))
    print(f"Saved reward demo head to {save_path}")


def make_dummy_dataset(path: str, n: int = 100):
    # 生成极小的示例数据（用于没有数据时的演示）
    sample = []
    for i in range(n):
        prompt = f"给出一句鼓励的话，场景编号 {i}。"
        chosen = f"你很棒，继续努力！ (示例 {i})"
        rejected = f"嗯，好吧。 (示例 {i})"
        sample.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected})
    with open(path, 'w', encoding='utf-8') as f:
        for s in sample:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')


def parse_args():
    p = argparse.ArgumentParser(description='Reward Model demo (pairwise)')
    p.add_argument('--data_path', type=str, default='./class/lec14/demo_pref.jsonl', help='preference dataset jsonl')
    p.add_argument('--base_model', type=str, default='gpt2', help='backbone model name')
    p.add_argument('--tokenizer_dir', type=str, default='gpt2', help='tokenizer dir or HF name')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--max_seq_len', type=int, default=128)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--save_path', type=str, default='./class/lec14/out')
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--make_dummy', action='store_true', help='如果没有数据则创建一个小的 demo 数据集')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.make_dummy and not os.path.exists(args.data_path):
        print('Create dummy preference dataset ->', args.data_path)
        make_dummy_dataset(args.data_path, n=200)

    if not os.path.exists(args.data_path):
        print('Data file not found:', args.data_path)
        print('Use --make_dummy to create a small demo dataset and retry')
        sys.exit(1)

    train(args)
