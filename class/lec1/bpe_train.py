#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal BPE training template

Requirements:
  pip install tokenizers transformers

Examples:
  # Train a GPT-2 style Byte-Level BPE on a folder of .txt files
  python bpe_train.py \
    --input /path/to/corpus_dir \
    --output_dir ./bpe_tokenizer \
    --vocab_size 32000 \
    --byte_level \
    --add_bos_eos

  # Train a whitespace-pretokenized BPE on a single file
  python bpe_train.py \
    --input /path/to/corpus.txt \
    --output_dir ./bpe_ws \
    --vocab_size 50000 \
    --min_frequency 2

    # No dataset? Use a classic built-in demo corpus
    python bpe_train.py \
        --demo \
        --output_dir ./bpe_demo \
        --vocab_size 8000 \
        --byte_level

Outputs:
  - tokenizer.json                # HF Tokenizers serialized model
  - special_tokens_map.json       # via save_pretrained
  - tokenizer_config.json         # via save_pretrained
  - (optional) merges.txt/vocab.json when --save_legacy_files is set
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.normalizers import NFKC, Lowercase, Strip, Sequence
from tokenizers.processors import TemplateProcessing


def iter_text_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for ext in ("*.txt", "*.text", "*.md", "*.jsonl"):
        for p in root.rglob(ext):
            if p.is_file():
                yield p


def build_normalizer(lowercase: bool) -> Sequence:
    steps = [NFKC(), Strip()]
    if lowercase:
        steps.append(Lowercase())
    return Sequence(steps)


# -----------------------------
# Educational BPE preview (toy)
# -----------------------------
def _word_freqs_from_texts(texts: List[str]) -> dict[tuple[str, ...], int]:
    import re
    freqs: dict[tuple[str, ...], int] = {}
    for line in texts:
        for w in re.findall(r"\S+", line.strip()):
            # basic per-char init with </w> end marker
            symbols = list(w) + ["</w>"]
            key = tuple(symbols)
            freqs[key] = freqs.get(key, 0) + 1
    return freqs


def _get_pair_stats(freqs: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
    stats: dict[tuple[str, str], int] = {}
    for symbols, count in freqs.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            stats[pair] = stats.get(pair, 0) + count
    return stats


def _merge_pair_in_vocab(pair: tuple[str, str], freqs: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
    a, b = pair
    merged = f"{a}{b}"
    new_freqs: dict[tuple[str, ...], int] = {}
    for symbols, count in freqs.items():
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_freqs[tuple(new_symbols)] = new_freqs.get(tuple(new_symbols), 0) + count
    return new_freqs


def _apply_merges_to_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    symbols = list(word) + ["</w>"]
    for (a, b) in merges:
        merged = f"{a}{b}"
        i = 0
        out: list[str] = []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    # strip end marker for display
    if symbols and symbols[-1] == "</w>":
        symbols = symbols[:-1]
    return symbols


def educational_bpe_preview(sample_texts: List[str], max_merges: int = 20, print_every: int = 5, sample_show: int = 3):
    print("\n=== Educational BPE preview (toy) ===")
    print(f"Corpus lines: {len(sample_texts)} | max_merges={max_merges} | print_every={print_every}")
    freqs = _word_freqs_from_texts(sample_texts)
    merges: list[tuple[str, str]] = []

    step = 0
    while step < max_merges:
        stats = _get_pair_stats(freqs)
        if not stats:
            break
        best = max(stats.items(), key=lambda kv: kv[1])[0]
        freqs = _merge_pair_in_vocab(best, freqs)
        merges.append(best)
        step += 1
        if step % print_every == 0 or step == 1 or step == max_merges:
            print(f"\n-- After {step} merges, top pairs so far: {merges[-min(len(merges), 5):]}")
            # show a few sample sentences tokenized by current merges
            for idx, line in enumerate(sample_texts[:sample_show]):
                tokens: list[str] = []
                for w in line.strip().split():
                    tokens.extend(_apply_merges_to_word(w, merges))
                print(f"[{idx}] {line}")
                print("    ->", tokens[:40])
    print("=== End of BPE preview ===\n")


def train_bpe(
    input_paths: List[Path],
    output_dir: Path,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    byte_level: bool = True,
    lowercase: bool = False,
    add_bos_eos: bool = False,
    save_legacy_files: bool = False,
    special_tokens: List[str] | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Special tokens
    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]
    pad, unk, bos, eos, mask = special_tokens[:5]

    # Model + normalization + pre-tokenizer
    tokenizer = Tokenizer(BPE(unk_token=unk))
    tokenizer.normalizer = build_normalizer(lowercase)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True) if byte_level else Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    files = [str(p) for p in input_paths]
    if len(files) == 0:
        raise FileNotFoundError("No training files found. Provide a file or a directory containing text files.")

    tokenizer.train(files=files, trainer=trainer)

    # Optional BOS/EOS wrapping like T5-style; GPT-2 typically omits this
    if add_bos_eos:
        # Make sure tokens exist in vocab
        bos_id = tokenizer.token_to_id(bos)
        eos_id = tokenizer.token_to_id(eos)
        if bos_id is None or eos_id is None:
            raise RuntimeError("BOS/EOS not in vocab; ensure they are in --special_tokens")
        tokenizer.post_processor = TemplateProcessing(
            single=f"{bos} $A {eos}",
            pair=f"{bos} $A {eos} {bos} $B {eos}",
            special_tokens=[(bos, bos_id), (eos, eos_id)],
        )

    # Save tokenizer.json (canonical)
    tok_json = output_dir / "tokenizer.json"
    tokenizer.save(str(tok_json))

    # Save a Transformers-compatible wrapper
    try:
        from transformers import PreTrainedTokenizerFast

        fast = PreTrainedTokenizerFast(
            tokenizer_file=str(tok_json),
            unk_token=unk,
            pad_token=pad,
            bos_token=bos if add_bos_eos else None,
            eos_token=eos if add_bos_eos else None,
            mask_token=mask,
        )
        fast.save_pretrained(str(output_dir))
    except Exception as e:
        # Still usable via tokenizers library even if transformers is absent
        with open(output_dir / "transformers_wrapper_error.txt", "w") as f:
            f.write(str(e))

    # Optionally also export legacy files (vocab.json/merges.txt)
    if save_legacy_files:
        # Re-train quickly using ByteLevelBPETokenizer API for legacy assets
        try:
            from tokenizers.implementations import ByteLevelBPETokenizer

            legacy_tok = ByteLevelBPETokenizer(lowercase=lowercase)
            legacy_tok.train(
                files=files,
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens,
                show_progress=True,
            )
            legacy_tok.save_model(str(output_dir))  # writes merges.txt & vocab.json
        except Exception as e:
            with open(output_dir / "legacy_export_error.txt", "w") as f:
                f.write(str(e))


def main():
    ap = argparse.ArgumentParser(description="Train a BPE tokenizer")
    ap.add_argument("--input", required=False, help="Path to a text file or a directory of text files")
    ap.add_argument("--output_dir", required=True, help="Directory to write tokenizer artifacts")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--byte_level", action="store_true", help="Use Byte-Level pretokenizer (GPT-2 style)")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--add_bos_eos", action="store_true", help="Wrap sequences with BOS/EOS in post-processor")
    ap.add_argument(
        "--special_tokens",
        type=str,
        default=None,
        help="JSON list of special tokens, e.g. ['<pad>','<unk>','<bos>','<eos>','<mask>']",
    )
    ap.add_argument("--save_legacy_files", action="store_true", help="Also write merges.txt and vocab.json")
    ap.add_argument("--demo", action="store_true", help="Use a small built-in classic corpus for quick demos")
    ap.add_argument("--preview_merges", type=int, default=0, help="Run a toy BPE preview with this many merges (0=off)")
    ap.add_argument("--preview_every", type=int, default=5, help="Print preview status every N merges")
    ap.add_argument("--preview_lines", type=int, default=6, help="Number of lines to show in preview samples")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)

    # Prepare training file list
    files: List[Path]
    if args.demo:
        # Write a small public-domain based demo corpus
        output_dir.mkdir(parents=True, exist_ok=True)
        demo_path = output_dir / "demo_corpus.txt"
        demo_text = "\n".join([
            # Shakespeare (public domain) - Hamlet
            "To be, or not to be, that is the question:",
            "Whether 'tis nobler in the mind to suffer",
            "The slings and arrows of outrageous fortune,",
            "Or to take arms against a sea of troubles",
            "And by opposing end them.",
            "",
            # Alice's Adventures in Wonderland (public domain) - opening
            "Alice was beginning to get very tired of sitting by her sister on the bank,",
            "and of having nothing to do: once or twice she had peeped into the book",
            "her sister was reading, but it had no pictures or conversations in it,",
            '"and what is the use of a book," thought Alice "without pictures or conversation?"',
            "",
            # Pangrams and common sentences to enrich character coverage
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "Sphinx of black quartz, judge my vow.",
            "",
            # Some simple chat-style text
            "User: Hello!",
            "Assistant: Hi there, how can I help you today?",
            "User: Teach me about Byte Pair Encoding.",
            "Assistant: BPE is a subword tokenization algorithm that merges frequent pairs.",
        ])
        with open(demo_path, "w", encoding="utf-8") as f:
            # repeat a bit to meet min_frequency easily
            for _ in range(max(1, 5 // max(1, args.min_frequency))):
                f.write(demo_text + "\n")
        files = [demo_path]
        demo_lines = demo_text.splitlines()
    else:
        if not args.input:
            raise SystemExit("Either provide --input or use --demo to run without external files.")
        input_path = Path(args.input)
        files = list(iter_text_files(input_path))
        # read a few lines for preview if requested
        demo_lines = []
        if args.preview_merges > 0:
            try:
                for p in files[:1]:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f):
                            if i >= args.preview_lines:
                                break
                            demo_lines.append(line.rstrip("\n"))
            except Exception:
                pass

    special_tokens = None
    if args.special_tokens:
        special_tokens = json.loads(args.special_tokens)

    # Optional: educational BPE preview
    if args.preview_merges > 0:
        sample_texts = demo_lines[: args.preview_lines]
        if not sample_texts and args.demo:
            sample_texts = demo_text.splitlines()[: args.preview_lines]
        if sample_texts:
            educational_bpe_preview(
                sample_texts=sample_texts,
                max_merges=args.preview_merges,
                print_every=args.preview_every,
                sample_show=min(len(sample_texts), 3),
            )

    train_bpe(
        input_paths=[Path(p) for p in files],
        output_dir=output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        byte_level=args.byte_level,
        lowercase=args.lowercase,
        add_bos_eos=args.add_bos_eos,
        save_legacy_files=args.save_legacy_files,
        special_tokens=special_tokens,
    )

    print(f"Saved tokenizer to: {output_dir}")


if __name__ == "__main__":
    main()
