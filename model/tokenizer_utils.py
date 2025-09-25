from __future__ import annotations

import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def build_tokenizer(tokenizer_dir: str, use_fast: bool = True, trust_remote_code: bool = False):
    """
    Load a tokenizer from local folder or HF repo id, and ensure pad_token is set.
    - If pad_token is missing, fall back to eos_token; if eos also missing, add a new '<pad>' token.
    NOTE: If a new token is added, tokenizer.vocab_size increases; make sure the model is
    constructed with vocab_size = tokenizer.vocab_size so embedding/lm_head sizes match.
    """
    tok = None
    # Prefer local, config-free loading to avoid requiring AutoConfig/model_type when offline
    if os.path.isdir(tokenizer_dir):
        tok_json = os.path.join(tokenizer_dir, 'tokenizer.json')
        try:
            if os.path.isfile(tok_json):
                # Load fast tokenizer directly from tokenizer.json without consulting model config
                tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir, tokenizer_file=tok_json, local_files_only=True)
        except Exception:
            tok = None
    if tok is None:
        # Fallback: generic AutoTokenizer; try local_files_only if path looks local
        try:
            local_only = os.path.isdir(tokenizer_dir)
            tok = AutoTokenizer.from_pretrained(
                tokenizer_dir,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
                local_files_only=local_only,
            )
        except Exception:
            # Last resort: drop local_files_only
            tok = AutoTokenizer.from_pretrained(
                tokenizer_dir,
                use_fast=use_fast,
                trust_remote_code=True,
            )
    if tok.pad_token is None:
        # Prefer mapping to an existing special token to avoid increasing vocab size
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif getattr(tok, 'unk_token', None) is not None:
            tok.pad_token = tok.unk_token
        else:
            # As a last resort, try to map to token id 0 (commonly a valid token), without adding new tokens
            try:
                pad_tok = tok.convert_ids_to_tokens(0)
                if pad_tok is not None:
                    tok.pad_token = pad_tok
                else:
                    # Fall back to adding a new pad token (may increase vocab); caller must resize embeddings accordingly
                    tok.add_special_tokens({'pad_token': '<pad>'})
            except Exception:
                tok.add_special_tokens({'pad_token': '<pad>'})
    return tok
