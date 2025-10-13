import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

POLICIES = {
    'DPO': './class/lec16/out_dpo',
    'GRPO': './class/lec16/out_grpo',
    'PPO': './class/lec16/out_ppo',
}

prompt = "如何克服拖延症？"
response_a = "首先，分析原因；然后制定计划并设定小目标，坚持每天打卡。"
response_b = "我不知道。也许每天写日记吧。"


def seq_logp(model, tokenizer, text, device='cpu', max_len=256):
    enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_len)
    input_ids = enc['input_ids'].to(device)
    att = enc['attention_mask'].to(device)
    # compute logits and sum logprobs
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=att)
        logits = outputs.logits
        shifted_logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        logp = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
        token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        seq_logp = token_logp.sum(dim=1)
    return float(seq_logp.cpu().item())


def ensure_pad(tokenizer):
    if getattr(tokenizer, 'pad_token', None) is None:
        if getattr(tokenizer, 'eos_token', None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})


def score_policies():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    text_a = f"User: {prompt}\nAssistant: {response_a}"
    text_b = f"User: {prompt}\nAssistant: {response_b}"

    results = {}
    for name, path in POLICIES.items():
        if not os.path.exists(path):
            print(f"Policy {name} not found at {path}, skipping")
            continue
        print(f"\nLoading policy {name} from {path}")
        tokenizer = AutoTokenizer.from_pretrained(path)
        ensure_pad(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(path).to(device)

        la = seq_logp(model, tokenizer, text_a, device=device)
        lb = seq_logp(model, tokenizer, text_b, device=device)
        pref = 'A' if la > lb else ('B' if lb > la else 'equal')
        print(f"Score A: {la:.6f}\nScore B: {lb:.6f}\n=> Prefers: {pref}")
        results[name] = {'A': la, 'B': lb, 'pref': pref}

    return results


if __name__ == '__main__':
    score_policies()
