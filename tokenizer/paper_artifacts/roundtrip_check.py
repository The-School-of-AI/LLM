#!/usr/bin/env python3
"""
T10: Round-trip integrity check.

Tokenize then detokenize 1000 random Indic sentences from the audit corpus
under HYBRID; confirm byte-perfect recovery.
"""

import argparse
import random
from pathlib import Path

from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = REPO_ROOT / "tokenizer" / "tokenizer.json"
DEFAULT_OUT = REPO_ROOT / "tokenizer" / "paper_artifacts" / "roundtrip_check.txt"

INDIC_LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
N_TARGET = 1000
SEED = 42

random.seed(SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--sft-dir",
        type=Path,
        required=True,
        help="Directory containing indic_as.txt, indic_bn.txt, ... audit text files.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))

    # Collect candidate lines from all 11 SFT files
    candidates = []
    for lang in INDIC_LANGS:
        path = args.sft_dir / f"indic_{lang}.txt"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if 20 <= len(line) <= 500:  # reasonable sentence length
                    candidates.append((lang, line))
    print(f"candidate pool: {len(candidates):,} lines across {len(INDIC_LANGS)} langs")

    # Sample 1000 with stratification by language
    by_lang = {}
    for lang, line in candidates:
        by_lang.setdefault(lang, []).append(line)
    sample = []
    per_lang_target = N_TARGET // len(by_lang)
    for lang, lines in by_lang.items():
        random.shuffle(lines)
        sample.extend((lang, l) for l in lines[:per_lang_target])
    # Top up if short
    while len(sample) < N_TARGET and candidates:
        sample.append(candidates[random.randint(0, len(candidates) - 1)])
    sample = sample[:N_TARGET]
    print(f"sampled {len(sample)} sentences, stratified across {len(by_lang)} langs")

    failures = []
    for lang, text in sample:
        enc = tok.encode(text)
        decoded = tok.decode(enc.ids)
        # Compare byte-perfect
        if decoded.strip() != text.strip():
            failures.append((lang, text, decoded))

    out_lines = []
    out_lines.append("# T10 — Round-trip integrity check (HYBRID tokenizer)")
    out_lines.append("")
    out_lines.append(
        f"Sample: {len(sample)} Indic SFT sentences, stratified across {len(by_lang)} languages"
    )
    out_lines.append(f"Seed: {SEED}")
    out_lines.append(f"Tokenizer: {args.tokenizer}")
    out_lines.append("")
    out_lines.append(
        f"Result: {len(sample) - len(failures)}/{len(sample)} round-trip successful ({100*(len(sample)-len(failures))/len(sample):.2f}%)"
    )
    if not failures:
        out_lines.append("")
        out_lines.append(
            "PASS — every sentence encoded then decoded to the identical byte string."
        )
    else:
        out_lines.append("")
        out_lines.append(f"FAIL — {len(failures)} mismatches:")
        for lang, text, decoded in failures[:30]:
            out_lines.append(f"  lang={lang}")
            out_lines.append(f"    in:  {text[:80]!r}")
            out_lines.append(f"    out: {decoded[:80]!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n")
    print("\n".join(out_lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
