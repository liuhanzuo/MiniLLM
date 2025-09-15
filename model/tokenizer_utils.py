from __future__ import annotations

from transformers import AutoTokenizer


def build_tokenizer(tokenizer_dir: str, use_fast: bool = True, trust_remote_code: bool = False):
    """
    Load a tokenizer from local folder or HF repo id, and ensure pad_token is set.
    - If pad_token is missing, fall back to eos_token; if eos also missing, add a new '<pad>' token.
    NOTE: If a new token is added, tokenizer.vocab_size increases; make sure the model is
    constructed with vocab_size = tokenizer.vocab_size so embedding/lm_head sizes match.
    """
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=use_fast, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({'pad_token': '<pad>'})
    return tok
