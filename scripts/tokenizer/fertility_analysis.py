#!/usr/bin/env python3
"""
Fertility & Compression Analysis: OLD vs HYBRID Tokenizer

Computes per-language metrics by tokenizing the Indic SFT corpus files
with both tokenizers and comparing:
  - chars/token (compression ratio)
  - tokens/word (fertility)
  - byte-fragment rate
  - total token count (training cost proxy)
"""

import json
import os
import time
from pathlib import Path

from transformers import PreTrainedTokenizerFast

BASE = Path(__file__).parent
OLD_DIR = BASE.parent / "Test20_3B_Zero3" / "code" / "src" / "tokenizer"
HYBRID_DIR = BASE / "output_hybrid"
SFT_DIR = BASE / "audit_combined" / "sft_data"

LANG_MAP = {
    "indic_as": "Assamese",
    "indic_bn": "Bengali",
    "indic_gu": "Gujarati",
    "indic_hi": "Hindi",
    "indic_kn": "Kannada",
    "indic_ml": "Malayalam",
    "indic_mr": "Marathi",
    "indic_or": "Odia",
    "indic_pa": "Punjabi",
    "indic_ta": "Tamil",
    "indic_te": "Telugu",
}


# GPT-2 byte maps for byte-fragment detection
def build_gpt2_maps():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def count_byte_fragments(tokens):
    """Count single-byte GPT-2 tokens (byte fragments)."""
    return sum(1 for t in tokens if len(t) == 1 and ord(t[0]) > 127)


def analyze_file(tokenizer, filepath, max_lines=None):
    """Tokenize a file and compute per-language stats."""
    total_chars = 0
    total_tokens = 0
    total_words = 0
    total_byte_frags = 0
    total_token_pieces = 0
    lines_processed = 0

    with open(filepath, "r", encoding="utf-8") as f:
        batch = []
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            batch.append(line)

            if len(batch) >= 512:
                enc = tokenizer(batch, add_special_tokens=False)
                for j, ids in enumerate(enc["input_ids"]):
                    text = batch[j]
                    total_chars += len(text)
                    total_tokens += len(ids)
                    total_words += len(text.split())
                    # Get token strings for fragment detection
                    tok_strs = tokenizer.convert_ids_to_tokens(ids)
                    total_byte_frags += count_byte_fragments(tok_strs)
                    total_token_pieces += len(tok_strs)
                    lines_processed += 1
                batch = []

        # Process remaining
        if batch:
            enc = tokenizer(batch, add_special_tokens=False)
            for j, ids in enumerate(enc["input_ids"]):
                text = batch[j]
                total_chars += len(text)
                total_tokens += len(ids)
                total_words += len(text.split())
                tok_strs = tokenizer.convert_ids_to_tokens(ids)
                total_byte_frags += count_byte_fragments(tok_strs)
                total_token_pieces += len(tok_strs)
                lines_processed += 1

    return {
        "lines": lines_processed,
        "chars": total_chars,
        "tokens": total_tokens,
        "words": total_words,
        "byte_frags": total_byte_frags,
        "token_pieces": total_token_pieces,
        "chars_per_token": total_chars / total_tokens if total_tokens else 0,
        "tokens_per_word": total_tokens / total_words if total_words else 0,
        "byte_frag_pct": (
            total_byte_frags / total_token_pieces * 100 if total_token_pieces else 0
        ),
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("  FERTILITY & COMPRESSION ANALYSIS: OLD vs HYBRID")
    print("=" * 80)

    # Load tokenizers
    print("\n  Loading OLD tokenizer...", end=" ", flush=True)
    old_tok = PreTrainedTokenizerFast.from_pretrained(str(OLD_DIR))
    print(f"vocab={old_tok.vocab_size:,}")

    print("  Loading HYBRID tokenizer...", end=" ", flush=True)
    hyb_tok = PreTrainedTokenizerFast.from_pretrained(str(HYBRID_DIR))
    print(f"vocab={hyb_tok.vocab_size:,}")

    # Find SFT files
    sft_files = {}
    for fname in sorted(os.listdir(SFT_DIR)):
        if fname.endswith(".txt"):
            key = fname.replace(".txt", "")
            if key in LANG_MAP:
                sft_files[key] = SFT_DIR / fname

    # Also add English from golden samples or a small English excerpt
    # We'll use the group2 file which has English content
    group_files = (
        {"english_sft": SFT_DIR / "group2.txt"}
        if (SFT_DIR / "group2.txt").exists()
        else {}
    )

    print(f"\n  Found {len(sft_files)} Indic SFT files + {len(group_files)} English")

    # Analyze each language
    results = {}
    print(f"\n{'Language':12s} | {'':^30s} OLD {'':^30s} | {'':^30s} HYBRID")
    print(
        f"{'':12s} | {'c/t':>6s} {'t/w':>6s} {'frag%':>7s} {'tokens':>12s} | {'c/t':>6s} {'t/w':>6s} {'frag%':>7s} {'tokens':>12s} | {'Δtok%':>7s}"
    )
    print("-" * 110)

    all_files = {**sft_files, **group_files}

    for key in sorted(all_files):
        filepath = all_files[key]
        lang_name = LANG_MAP.get(key, key.replace("_sft", "").title())

        # Analyze with both tokenizers (use max 50K lines for speed)
        old_stats = analyze_file(old_tok, filepath, max_lines=50000)
        hyb_stats = analyze_file(hyb_tok, filepath, max_lines=50000)

        delta_tok = (
            (hyb_stats["tokens"] - old_stats["tokens"]) / old_stats["tokens"] * 100
            if old_stats["tokens"]
            else 0
        )

        print(
            f"{lang_name:12s} | {old_stats['chars_per_token']:6.2f} {old_stats['tokens_per_word']:6.2f} {old_stats['byte_frag_pct']:6.1f}% {old_stats['tokens']:>12,} | "
            f"{hyb_stats['chars_per_token']:6.2f} {hyb_stats['tokens_per_word']:6.2f} {hyb_stats['byte_frag_pct']:6.1f}% {hyb_stats['tokens']:>12,} | {delta_tok:>+6.1f}%"
        )

        results[key] = {
            "language": lang_name,
            "old": old_stats,
            "hybrid": hyb_stats,
            "delta_tokens_pct": delta_tok,
        }

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    old_total_tok = sum(r["old"]["tokens"] for r in results.values())
    hyb_total_tok = sum(r["hybrid"]["tokens"] for r in results.values())
    old_total_chr = sum(r["old"]["chars"] for r in results.values())
    hyb_total_chr = sum(r["hybrid"]["chars"] for r in results.values())

    print("\n  Total tokens across all SFT files:")
    print(f"    OLD:    {old_total_tok:>15,}")
    print(f"    HYBRID: {hyb_total_tok:>15,}")
    print(
        f"    Savings: {old_total_tok - hyb_total_tok:>15,} tokens ({(1 - hyb_total_tok/old_total_tok)*100:.1f}% fewer)"
    )

    print("\n  Overall chars/token:")
    print(f"    OLD:    {old_total_chr / old_total_tok:.3f}")
    print(f"    HYBRID: {hyb_total_chr / hyb_total_tok:.3f}")

    # Languages with biggest improvements
    print("\n  Biggest improvements (by token reduction):")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["delta_tokens_pct"])
    for key, r in sorted_results[:5]:
        print(f"    {r['language']:12s}: {r['delta_tokens_pct']:+.1f}% tokens")

    # Write JSON results
    out_path = BASE / "output_hybrid" / "fertility_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {out_path}")

    print(f"\n  Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
