#!/usr/bin/env python3
"""Tokenizes the ASR context word list into little-endian int32 Qwen token ids.

usage: make_context_ids.py <tokenizer dir with vocab.json + merges.txt> <context.txt> <out.i32>
Needs `pip install tokenizers`. Byte-level BPE with the Qwen2/Qwen3 pre-tokenizer regex; the text is stripped
first (this reproduces the ids the app shipped with in v0.7).
"""
import struct
import sys

from tokenizers import Regex, Tokenizer, models, pre_tokenizers

QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
              r"|\s*[\r\n]+|\s+(?!\S)|\s+")


def main() -> None:
    tok_dir, src, out = sys.argv[1:4]
    tok = Tokenizer(models.BPE.from_file(f"{tok_dir}/vocab.json", f"{tok_dir}/merges.txt"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(QWEN_SPLIT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    with open(src, encoding="utf-8") as f:
        ids = tok.encode(f.read().strip()).ids
    with open(out, "wb") as f:
        f.write(struct.pack(f"<{len(ids)}i", *ids))
    print(f"{len(ids)} tokens -> {out}")


if __name__ == "__main__":
    main()
