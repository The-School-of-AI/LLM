#!/usr/bin/env python3
"""
verify_max_byte_length.py
=========================

Reviewer-runnable verification: confirms that every NORMAL token in the
tokenizer's vocabulary, when decoded to its raw UTF-8 surface form, is
at most MAX_BYTES bytes long (default 32).

Why this matters
----------------
The LightningLM tokenizer is paired with a KroneckerEmbeddings layer that
computes per-token embeddings from the raw UTF-8 bytes of each token piece.
The KE layer has a fixed `POS_DIM = 32`, meaning at most 32 byte positions
fit per token. Any token longer than 32 bytes would be silently truncated,
which would corrupt the embedding for that token.

The paper claims that 100% of normal+bytefallback tokens are ≤ 32 bytes
(with the longest being exactly 32 bytes — structural fillers like 32 × '-').
This script verifies that claim end-to-end.

Methodology
-----------
1. Load `tokenizer.json` and read `model.vocab` (token → id mapping).
2. For each token, classify it:
   - "special"  : starts with "<|" and ends with "|>"  (e.g. <|begin_of_text|>)
   - "bytefallback": matches "<0xNN>" pattern  (none expected for GPT-2 ByteLevel)
   - "normal"  : everything else (the BPE-trained pieces)
3. For "normal" tokens: decode GPT-2 ByteLevel form back to UTF-8, measure byte length.
4. Flag any token with byte_length > MAX_BYTES.

Usage
-----
    python verify_max_byte_length.py path/to/tokenizer.json [max_bytes]
    # max_bytes defaults to 32 (the KroneckerEmbeddings POS_DIM)

Exit codes
----------
    0   PASS — all normal tokens fit
    1   FAIL — at least one normal token exceeds the limit
    2   error parsing tokenizer file

Author: codebase agent for the LightningLM tokenizer paper.
License: MIT.
"""

import json
import re
import sys


# Standard GPT-2 ByteLevel byte ↔ unicode bijection (used by o200k_base, Tekken, etc.)
def build_gpt2_byte_maps():
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTEFALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def classify(token):
    if token.startswith("<|") and token.endswith("|>"):
        return "special"
    if BYTEFALLBACK_RE.match(token):
        return "bytefallback"
    return "normal"


def gpt2_byte_len(token, g2b):
    """Byte length of a GPT-2 ByteLevel token = number of mapped chars."""
    try:
        return len(bytes([g2b[c] for c in token]))
    except KeyError:
        # Token contains chars not in the GPT-2 bijection — not a normal piece
        return None


def main():
    if len(sys.argv) < 2:
        print(
            f"usage: {sys.argv[0]} path/to/tokenizer.json [max_bytes]", file=sys.stderr
        )
        sys.exit(2)
    path = sys.argv[1]
    max_bytes = int(sys.argv[2]) if len(sys.argv) >= 3 else 32

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = data["model"]["vocab"]
    except Exception as e:
        print(f"ERROR: could not parse {path}: {e}", file=sys.stderr)
        sys.exit(2)

    g2b = build_gpt2_byte_maps()

    counts = {"normal": 0, "special": 0, "bytefallback": 0, "non-byte-level": 0}
    too_long = []
    max_observed = 0
    longest_token = None

    for token in vocab:
        kind = classify(token)
        counts[kind] += 1
        if kind != "normal":
            continue
        bl = gpt2_byte_len(token, g2b)
        if bl is None:
            counts["normal"] -= 1
            counts["non-byte-level"] += 1
            continue
        if bl > max_observed:
            max_observed = bl
            longest_token = token
        if bl > max_bytes:
            too_long.append((token, bl, vocab[token]))

    print(f"tokenizer:           {path}")
    print(f"vocab size:          {len(vocab):,}")
    print(f"  normal:            {counts['normal']:,}")
    print(f"  special:           {counts['special']:,}")
    print(f"  byte-fallback:     {counts['bytefallback']:,}")
    if counts["non-byte-level"]:
        print(
            f"  non-byte-level:    {counts['non-byte-level']:,}  (skipped — not in GPT-2 bijection)"
        )
    print(f"max_bytes limit:     {max_bytes}")
    print(f"longest normal token: {max_observed} bytes  (token: {longest_token!r})")
    print(f"tokens > {max_bytes} bytes: {len(too_long)}")

    if not too_long:
        print(
            f"PASS: all {counts['normal']:,} normal tokens are within {max_bytes} bytes."
        )
        sys.exit(0)
    else:
        print(
            f"FAIL: {len(too_long)} normal token(s) exceed {max_bytes} bytes:",
            file=sys.stderr,
        )
        for t, bl, tid in too_long[:20]:
            print(f"  id={tid:>6}  {bl} bytes  {t!r}", file=sys.stderr)
        if len(too_long) > 20:
            print(f"  ... and {len(too_long) - 20} more", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
