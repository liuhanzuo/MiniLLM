import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)
        # Avoid constructing tensors from tensors; use slice + to(dtype)
        X = input_ids[:-1].to(dtype=torch.long).clone()
        Y = input_ids[1:].to(dtype=torch.long).clone()
        loss_mask = loss_mask[1:].to(dtype=torch.long).clone()
        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        # 对于通用 HF chat 模板，不能依赖固定的“assistant 开始/结束”标记。
        # 采用通用策略：对每个 assistant turn，将其内容置空得到一份“裁剪版”token 序列；
        # 与完整序列比较，差异区间即为该 turn 的可训练区域。

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _normalize_conversations(self, sample):
        """兼容多种数据格式，统一转为 [{role, content}, ...] 列表。
        支持：
        - sample['conversations']: List[Dict]
        - sample['messages']: List[Dict]（emergent-misalignment 常用）
        - (prompt, response) / (input, output)
        - 若存在 'chosen'（DPO 结构），取 chosen 作为 SFT 输入
        """
        if isinstance(sample, dict):
            if 'conversations' in sample and isinstance(sample['conversations'], list):
                return sample['conversations']
            if 'messages' in sample and isinstance(sample['messages'], list):
                return sample['messages']
            if 'chosen' in sample and isinstance(sample['chosen'], list):
                return sample['chosen']
            if 'prompt' in sample and 'response' in sample:
                return [
                    {"role": "user", "content": str(sample['prompt'])},
                    {"role": "assistant", "content": str(sample['response'])},
                ]
            if 'input' in sample and 'output' in sample:
                return [
                    {"role": "user", "content": str(sample['input'])},
                    {"role": "assistant", "content": str(sample['output'])},
                ]
            if 'text' in sample and isinstance(sample['text'], str):
                t = str(sample['text'])
                return [
                    {"role": "user", "content": t},
                    {"role": "assistant", "content": t},
                ]
        raise KeyError("No supported conversation fields found (expected one of conversations/messages/chosen or prompt+response / input+output / text)")

    def _normalize_messages(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            if isinstance(turn, dict):
                content = turn.get('content') if 'content' in turn else str(turn)
                role = turn.get('role')
                if role is None:
                    role = 'user' if i % 2 == 0 else 'assistant'
                else:
                    role = str(role).lower()
                    if role not in {"user", "assistant", "system"}:
                        role = 'user' if i % 2 == 0 else 'assistant'
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": 'user' if i % 2 == 0 else 'assistant', "content": str(turn)})
        return messages

    def _apply_template_tokenize(self, messages):
        # 返回未截断的 input_ids（list[int]）
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False
        )
        # 某些 tokenizer 返回 dict，这里做兼容
        if isinstance(ids, dict):
            ids = ids.get('input_ids') or ids.get('input_ids'.encode('utf-8'))
        return ids

    def _diff_span(self, full_ids, trim_ids):
        # 计算两个序列的最左公共前缀和最右公共后缀长度，返回 full_ids 的差异区间 [l, len(full)-r)
        n1, n2 = len(full_ids), len(trim_ids)
        l = 0
        while l < n1 and l < n2 and full_ids[l] == trim_ids[l]:
            l += 1
        r = 0
        while r < (n1 - l) and r < (n2 - l) and full_ids[n1 - 1 - r] == trim_ids[n2 - 1 - r]:
            r += 1
        # 保证不越界
        r = min(r, n1 - l)
        return l, n1 - r

    def _generate_loss_mask_generic(self, messages, full_ids_untrunc):
        # 针对每个 assistant turn，通过“置空内容”法定位差异区间并标 1
        mask = [0] * len(full_ids_untrunc)
        for idx, m in enumerate(messages):
            if m.get('role') != 'assistant':
                continue
            msgs_trim = [d.copy() for d in messages]
            msgs_trim[idx] = {**msgs_trim[idx], 'content': ''}
            trim_ids = self._apply_template_tokenize(msgs_trim)
            if trim_ids is None:
                continue
            l, r = self._diff_span(full_ids_untrunc, trim_ids)
            # 标注 [l, r) 作为可训练区域（注意稍后会整体右移一位对齐到 Y）
            for j in range(l, r):
                if 0 <= j < len(mask):
                    mask[j] = 1
        return mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 统一抽取 conversations 并构建 messages 列表
        conversations = self._normalize_conversations(sample)
        messages = self._normalize_messages(conversations)

        # 得到未截断 token 序列
        full_ids = self._apply_template_tokenize(messages)
        if full_ids is None:
            full_ids = self.tokenizer(
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            ).input_ids

        # 生成通用损失掩码（未截断）
        loss_mask_full = self._generate_loss_mask_generic(messages, full_ids)

        # 截断到 max_length，并补 pad
        input_ids = full_ids[:self.max_length]
        loss_mask = loss_mask_full[:self.max_length]
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids = input_ids + [pad_id] * pad_len
            loss_mask = loss_mask + [0] * pad_len

        # 构建训练数据（与原逻辑一致：mask 对齐到 Y，因此右移切片）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


if __name__ == "__main__":
    pass
