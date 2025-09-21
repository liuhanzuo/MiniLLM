import json
import os
from typing import List, Dict, Any, Tuple

import torch
from MiniLLM.model.tokenizer_utils import build_tokenizer
from MiniLLM.model.model import MiniLLMLM
from MiniLLM.model.LMConfig import LMConfig


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_tokenizer(tokenizer_dir: str):
    return build_tokenizer(tokenizer_dir, trust_remote_code=True)


def build_lm(config) -> MiniLLMLM:
    lm_conf = LMConfig(
        dim=config.lm_dim,
        n_layers=config.n_layers,
        n_block=config.n_block,
        max_seq_len=config.max_seq_len,
        use_moe=config.use_moe,
        repeat_layer=config.repeat_layer,
    )
    return MiniLLMLM(lm_conf)


def load_model(config, device: torch.device) -> MiniLLMLM:
    model = build_lm(config)
    if config.ckp and os.path.exists(config.ckp):
        state = torch.load(config.ckp, map_location=device)
        # 过滤临时 buffer
        state = {k: v for k, v in state.items() if 'mask' not in k}
        model.load_state_dict(state, strict=False)
    return model.to(device)
