import argparse
import torch
from transformers import AutoTokenizer, AutoModel
import os

# CLI
parser = argparse.ArgumentParser(description='Score two responses using saved reward head')
parser.add_argument('--head_file', type=str, default=None, help='path to head checkpoint (.pth). If not provided, prefer reward_demo_head_best.pth then reward_demo_head.pth')
args = parser.parse_args()

# 配置
default_dir = "./class/lec14/out"
preferred_best = os.path.join(default_dir, 'reward_demo_head_best.pth')
preferred = os.path.join(default_dir, 'reward_demo_head.pth')
if args.head_file:
    head_path = args.head_file
elif os.path.exists(preferred_best):
    head_path = preferred_best
else:
    head_path = preferred

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 如果 head 中没有 backbone_name，可在这里手动指定：
fallback_backbone = "gpt2"
max_seq_len = 256

# 载入 head
state = torch.load(head_path, map_location=device)
backbone_name = state.get("backbone_name", fallback_backbone)
print("Using reward backbone:", backbone_name)

# 加载 tokenizer + backbone
tokenizer = AutoTokenizer.from_pretrained(backbone_name)
if tokenizer.pad_token is None:
    # 优先用 eos 作为 pad，避免 resize
    if getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

backbone = AutoModel.from_pretrained(backbone_name).to(device)
hidden_size = backbone.config.hidden_size

# 构建头并加载权重
head = torch.nn.Linear(hidden_size, 1).to(device)
head.load_state_dict(state["head_state_dict"])
head.eval()
backbone.eval()

def score(prompt: str, response: str) -> float:
    """给定 prompt 和 response，返回 reward 分数（标量）"""
    text = f"User: {prompt}\nAssistant: {response}"
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_seq_len)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = out.last_hidden_state  # (1, T, H)
        lengths = attention_mask.sum(dim=1) - 1  # index of last non-pad token
        lengths = lengths.clamp(min=0)
        idx = torch.arange(input_ids.size(0), device=device)
        eos_h = last_hidden[idx, lengths, :]  # (1, H)
        s = head(eos_h).squeeze(-1).item()
    return s

# 示例 prompt 与两个候选回复
prompt = "如何克服拖延症？"
response_a = "首先，分析原因；然后制定计划并设定小目标，坚持每天打卡。"
response_b = "我不知道。也许每天写日记吧。"

score_a = score(prompt, response_a)
score_b = score(prompt, response_b)

print("Prompt:", prompt)
print("Response A:", response_a)
print("Score A:", score_a)
print("Response B:", response_b)
print("Score B:", score_b)

if score_a > score_b:
    print("=> Reward model prefers Response A")
elif score_b > score_a:
    print("=> Reward model prefers Response B")
else:
    print("=> Scores equal")