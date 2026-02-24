from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from transformers import PreTrainedTokenizerFast


def check_prompt_injection(tokenizer: PreTrainedTokenizerFast):
    print("=== Prompt Injection Leakage ===")
    test_str = "<|assistant|>"
    enc = tokenizer.encode(test_str, add_special_tokens=False)
    decoded = [tokenizer.decode([i]) for i in enc]
    print(f"Input: {test_str!r}")
    print(f"Encoded IDs: {enc}")
    print(f"Decoded chunks: {decoded}")
    
    special_id = tokenizer.convert_tokens_to_ids(test_str)
    if len(enc) == 1 and enc[0] == special_id:
        print("WARNING: Tokenizer allows raw text to encode directly into special tokens!")
    else:
        print("SAFE: Special tokens are split or escaped when passed as raw text.")
    print()


def check_numeric_tokenization(tokenizer: PreTrainedTokenizerFast):
    print("=== Numeric Tokenization ===")
    test_nums = ["1", "12", "123", "1234", "12345", "123456"]
    for num in test_nums:
        enc = tokenizer.encode(num, add_special_tokens=False)
        decoded = [tokenizer.decode([i]) for i in enc]
        print(f"{num:>8} -> IDs: {str(enc):<15} Chunks: {decoded}")
    print("Observation: Watch for greedy merging (e.g. up to 3 digits) vs single-digit splitting.")
    print()


def check_whitespace_prefix(tokenizer: PreTrainedTokenizerFast):
    print("=== Whitespace Prefix Invariance ===")
    word1, word2 = "Hello", " Hello"
    enc1 = tokenizer.encode(word1, add_special_tokens=False)
    enc2 = tokenizer.encode(word2, add_special_tokens=False)
    print(f"Input: {word1!r} -> IDs: {enc1}")
    print(f"Input: {word2!r} -> IDs: {enc2}")
    if enc1 != enc2:
        print("Observation: Tokenizer treats prefixed-whitespace as a distinct token (common for BPE).")
    print()


def check_compression_by_language(tokenizer: PreTrainedTokenizerFast, parquet_path: Path, sample_size: int = 50000):
    print(f"=== Compression Ratio by Language (sample size: {sample_size}) ===")
    if not parquet_path.exists():
        print(f"Dataset {parquet_path} not found. Skipping compression test.")
        return

    try:
        df = pd.read_parquet(parquet_path, columns=["text", "language"])
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    results = []
    for lang, group in df.groupby("language"):
        texts = group["text"].tolist()
        if not texts:
            continue
        
        total_chars = sum(len(t) for t in texts)
        total_tokens = sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)
        
        if total_tokens > 0:
            results.append({
                "language": lang,
                "chars_per_token": total_chars / total_tokens,
                "samples": len(texts)
            })

    results.sort(key=lambda x: x["chars_per_token"], reverse=True)
    
    print(f"{'Language':<10} | {'Chars/Token':<15} | {'Samples'}")
    print("-" * 40)
    for r in results:
        print(f"{r['language']:<10} | {r['chars_per_token']:<15.2f} | {r['samples']}")
    print("Observation: Higher is better (more efficient). Ratios near 1.0 indicate heavy byte-fallback.")
    print()


def main():
    ap = argparse.ArgumentParser(description="Run qualitative checks on tokenizer behavior.")
    ap.add_argument("--tokenizer-dir", type=Path, default=Path("experiments/6_tokenizer_design_lab/tsai_131k_tokenizer"))
    ap.add_argument("--parquet-path", type=Path, default=Path("/home/ubuntu/raw_shard.parquet"))
    ap.add_argument("--sample-size", type=int, default=50000)
    args = ap.parse_args()

    if not args.tokenizer_dir.exists():
        print(f"Error: Tokenizer directory {args.tokenizer_dir} not found.")
        sys.exit(1)

    print(f"Loading tokenizer from {args.tokenizer_dir}...\n")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(args.tokenizer_dir))

    check_prompt_injection(tokenizer)
    check_numeric_tokenization(tokenizer)
    check_whitespace_prefix(tokenizer)
    check_compression_by_language(tokenizer, args.parquet_path, args.sample_size)


if __name__ == "__main__":
    main()
