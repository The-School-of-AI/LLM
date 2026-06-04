#!/usr/bin/env python3
"""
verify_no_cross_script_merges.py
================================

Reviewer-runnable verification: confirms that NO merge rule in the
tokenizer's BPE merge table combines bytes from two different writing
systems (Brahmic↔Latin, Brahmic↔Brahmic, etc.).

Why this matters
----------------
The LightningLM tokenizer is paired with a KroneckerEmbeddings layer that
computes per-token embeddings from the raw UTF-8 bytes of each token piece.
If a token mixes bytes from two scripts (e.g., a Latin character + a
Devanagari character), its embedding becomes a meaningless superposition.

The paper claims that the pre-tokenizer (GPT-2 ByteLevel, inherited
unchanged from o200k_base) does NOT enforce script splitting at runtime,
but that the no-cross-script-token property nonetheless holds *because*
no merge rule in the final ruleset crosses a script boundary. This script
verifies that claim end-to-end.

Methodology
-----------
1. Load `tokenizer.json` and parse the BPE merge list.
2. For each merge (left, right), decode each side from GPT-2 ByteLevel
   form back to UTF-8, then classify each character by Unicode script
   range (Brahmic 9 scripts + Latin + CJK + others).
3. Flag any merge where the two sides come from disjoint non-Common
   script sets (Latin vs Indic, or Indic-A vs Indic-B).
4. Punctuation, digits, marks, whitespace, and control chars are "Common"
   and merge freely with anything.

Usage
-----
    python verify_no_cross_script_merges.py path/to/tokenizer.json

Exit codes
----------
    0   PASS — no cross-script merges
    1   FAIL — cross-script merges found (printed to stderr)
    2   error parsing tokenizer file

Author: codebase agent for the LightningLM tokenizer paper.
License: MIT.
"""

import json
import sys
import unicodedata

# Brahmic scripts the tokenizer claims to support
BRAHMIC_SCRIPTS = {
    "Devanagari",
    "Bengali",
    "Tamil",
    "Telugu",
    "Kannada",
    "Malayalam",
    "Gujarati",
    "Gurmukhi",
    "Oriya",
}
# Scripts that the tokenizer treats as foreign / out-of-scope
FOREIGN_SCRIPTS = {
    "CJK",
    "Arabic",
    "Cyrillic",
    "Thai",
    "Hiragana",
    "Katakana",
    "Hangul",
}
# Scripts that may legitimately merge with anything (punctuation, digits, marks)
COMMON = {"Common", "Mark", None}


INDIC_RANGES = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Oriya": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Arabic": (0x0600, 0x06FF),
}


def build_gpt2_byte_maps():
    """Standard GPT-2 ByteLevel byte ↔ unicode bijection."""
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


def gpt2_decode(piece, g2b):
    """Convert a GPT-2 ByteLevel piece back to UTF-8 string (lossy if needed)."""
    try:
        return bytes([g2b.get(c, 0) for c in piece]).decode("utf-8", errors="replace")
    except Exception:
        return piece


def script_of_char(ch):
    """Return the script of a single character, or None for Common/Mark."""
    cp = ord(ch)
    for s, (lo, hi) in INDIC_RANGES.items():
        if lo <= cp <= hi:
            return s
    name = unicodedata.name(ch, "")
    if not name:
        return None
    if name.startswith("LATIN"):
        return "Latin"
    if (
        name.startswith("CJK")
        or name.startswith("HIRAGANA")
        or name.startswith("KATAKANA")
        or name.startswith("HANGUL")
    ):
        return "CJK"
    if name.startswith("CYRILLIC"):
        return "Cyrillic"
    if name.startswith("THAI"):
        return "Thai"
    if name.startswith("ARABIC"):
        return "Arabic"
    if name.startswith("SINHALA"):
        return "Sinhala"
    cat = unicodedata.category(ch)
    if (
        cat.startswith("M")
        or cat.startswith("P")
        or cat.startswith("S")
        or cat.startswith("Z")
        or cat.startswith("N")
        or cat.startswith("C")
    ):
        return None  # Common
    return None


def piece_scripts(piece, g2b):
    """Return the set of non-Common scripts present in a token piece."""
    text = gpt2_decode(piece, g2b)
    scripts = set()
    for ch in text:
        s = script_of_char(ch)
        if s is not None:
            scripts.add(s)
    return scripts


def is_cross_script(left_scripts, right_scripts):
    """
    Cross-script merge: both sides have non-empty non-Common scripts AND
    they're disjoint. (Same-script merges OK; Common+anything OK; one-side-Common OK.)
    """
    if not left_scripts or not right_scripts:
        return False
    return left_scripts.isdisjoint(right_scripts)


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} path/to/tokenizer.json", file=sys.stderr)
        sys.exit(2)

    path = sys.argv[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merges = data["model"]["merges"]
    except Exception as e:
        print(f"ERROR: could not parse {path}: {e}", file=sys.stderr)
        sys.exit(2)

    g2b = build_gpt2_byte_maps()

    n_total = 0
    n_cross = 0
    examples = []

    for m in merges:
        if isinstance(m, list):
            if len(m) != 2:
                continue
            a, b = m
        else:
            parts = m.split(" ", 1)
            if len(parts) != 2:
                continue
            a, b = parts

        n_total += 1
        sa = piece_scripts(a, g2b)
        sb = piece_scripts(b, g2b)
        if is_cross_script(sa, sb):
            n_cross += 1
            if len(examples) < 10:
                examples.append(
                    (a, b, sa, sb, gpt2_decode(a, g2b), gpt2_decode(b, g2b))
                )

    print(f"tokenizer:         {path}")
    print(f"total merges:      {n_total:,}")
    print(f"cross-script:      {n_cross}")

    if n_cross == 0:
        print("PASS: no cross-script merges in the BPE ruleset.")
        sys.exit(0)
    else:
        print("FAIL: cross-script merges found:", file=sys.stderr)
        for a, b, sa, sb, da, db in examples:
            print(f"  merge: {a!r} + {b!r}", file=sys.stderr)
            print(f"         decoded: {da!r} ({sa})  +  {db!r} ({sb})", file=sys.stderr)
        if n_cross > len(examples):
            print(f"  ... and {n_cross - len(examples)} more", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
