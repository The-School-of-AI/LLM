#!/usr/bin/env python3
"""
Compare multiple tokenizers on a dataset.
Calculates:
- Tokens per character
- Byte fallback rate
- Vocabulary coverage
- Token distribution
- Compression efficiency
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install transformers numpy")
    sys.exit(1)


def calculate_byte_fallback_rate(tokenizer, tokens):
    """
    Calculate the percentage of tokens that are byte-level fallbacks.
    Byte fallback tokens are typically represented as <0xXX> or single high bytes.
    """
    byte_fallback_count = 0

    for token_id in tokens:
        token_str = tokenizer.decode([token_id])
        # Check if it's a single byte or looks like a fallback
        if len(token_str) == 1 and ord(token_str) > 127:
            byte_fallback_count += 1
        # Also check for explicit byte tokens
        elif token_str.startswith("<0x") and token_str.endswith(">"):
            byte_fallback_count += 1

    return (byte_fallback_count / len(tokens) * 100) if tokens else 0


def load_jsonl_dataset(data_dir, text_field="text", max_files=None):
    """Load JSONL dataset and extract text from specified field."""
    data_dir = Path(data_dir)
    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    if max_files:
        jsonl_files = jsonl_files[:max_files]

    print(f"  Found {len(jsonl_files)} JSONL files")

    texts = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Try multiple possible text fields
                    text = None
                    for field in [
                        text_field,
                        "text",
                        "content",
                        "Question",
                        "Answer",
                        "Explanation",
                    ]:
                        if field in data and data[field]:
                            # Concatenate all text fields for NCERT format
                            if isinstance(data.get("Question"), str) and isinstance(
                                data.get("Answer"), str
                            ):
                                text = f"{data.get('Question', '')} {data.get('Answer', '')} {data.get('Explanation', '')}"
                                break
                            else:
                                text = data[field]
                                break

                    if text and isinstance(text, str) and len(text.strip()) > 0:
                        texts.append(text.strip())
                except json.JSONDecodeError:
                    continue

    return texts


def evaluate_tokenizer(tokenizer_path, texts, tokenizer_name):
    """Evaluate a single tokenizer on the dataset."""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {tokenizer_name}")
    print(f"{'='*80}")

    # Load tokenizer
    print(f"📂 Loading tokenizer from: {tokenizer_path}")
    try:
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        load_time = time.time() - start_time
        print(
            f"✓ Tokenizer loaded in {load_time:.2f}s (vocab size: {len(tokenizer):,})"
        )
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return None

    # Process texts
    print(f"\n🔄 Processing {len(texts):,} samples...")

    total_chars = 0
    total_tokens = 0
    tokens_per_char_list = []
    byte_fallback_rates = []
    all_tokens = []
    decode_mismatches = 0

    start_time = time.time()

    for i, text in enumerate(texts):
        if not text or len(text) == 0:
            continue

        # Tokenize
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            print(f"  Warning: Error tokenizing sample {i}: {e}")
            continue

        # Calculate metrics
        num_chars = len(text)
        num_tokens = len(tokens)

        if num_chars > 0 and num_tokens > 0:
            tokens_per_char = num_tokens / num_chars
            tokens_per_char_list.append(tokens_per_char)

            # Byte fallback rate
            byte_fallback = calculate_byte_fallback_rate(tokenizer, tokens)
            byte_fallback_rates.append(byte_fallback)

            # Update totals
            total_chars += num_chars
            total_tokens += num_tokens
            all_tokens.extend(tokens)

            # Verify decode
            try:
                decoded = tokenizer.decode(tokens)
                if decoded != text:
                    decode_mismatches += 1
            except Exception:
                decode_mismatches += 1

        # Progress
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1:,}/{len(texts):,} samples...")

    process_time = time.time() - start_time

    # Calculate metrics
    results = {
        "tokenizer_name": tokenizer_name,
        "vocab_size": len(tokenizer),
        "load_time": load_time,
        "process_time": process_time,
        "total_samples": len(texts),
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "avg_tokens_per_char": total_tokens / total_chars if total_chars > 0 else 0,
        "median_tokens_per_char": (
            np.median(tokens_per_char_list) if tokens_per_char_list else 0
        ),
        "std_tokens_per_char": (
            np.std(tokens_per_char_list) if tokens_per_char_list else 0
        ),
        "avg_byte_fallback": np.mean(byte_fallback_rates) if byte_fallback_rates else 0,
        "median_byte_fallback": (
            np.median(byte_fallback_rates) if byte_fallback_rates else 0
        ),
        "unique_tokens": len(set(all_tokens)),
        "vocab_coverage": (
            (len(set(all_tokens)) / len(tokenizer) * 100) if len(tokenizer) > 0 else 0
        ),
        "decode_mismatches": decode_mismatches,
        "compression_ratio": total_chars / total_tokens if total_tokens > 0 else 0,
    }

    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS: {tokenizer_name}")
    print(f"{'='*80}")

    print("\n⏱️  Performance:")
    print(f"  • Load time: {results['load_time']:.2f}s")
    print(
        f"  • Process time: {results['process_time']:.2f}s ({results['total_samples']/results['process_time']:.1f} samples/sec)"
    )

    print("\n📊 Tokenization Metrics:")
    print(f"  • Total characters: {results['total_chars']:,}")
    print(f"  • Total tokens: {results['total_tokens']:,}")
    print(f"  • Avg tokens/char: {results['avg_tokens_per_char']:.4f}", end="")

    # Quality indicator
    tpc = results["avg_tokens_per_char"]
    if tpc < 1.5:
        print(" ✅ (Excellent)")
    elif tpc < 2.0:
        print(" ✅ (Good)")
    elif tpc < 2.5:
        print(" ⚠️  (Fair)")
    else:
        print(" ❌ (Poor)")

    print(f"  • Median tokens/char: {results['median_tokens_per_char']:.4f}")
    print(f"  • Std tokens/char: {results['std_tokens_per_char']:.4f}")
    print(f"  • Compression ratio: {results['compression_ratio']:.4f} chars/token")
    print(
        f"  • Efficiency: {(1/results['avg_tokens_per_char'])*100:.1f}% (higher is better)"
    )

    print("\n🔢 Byte Fallback:")
    print(f"  • Avg byte fallback: {results['avg_byte_fallback']:.2f}%", end="")

    # Quality indicator
    bf = results["avg_byte_fallback"]
    if bf < 10:
        print(" ✅ (Excellent)")
    elif bf < 20:
        print(" ✅ (Good)")
    elif bf < 30:
        print(" ⚠️  (Fair)")
    else:
        print(" ❌ (Poor)")

    print(f"  • Median byte fallback: {results['median_byte_fallback']:.2f}%")

    print("\n📚 Vocabulary:")
    print(f"  • Vocab size: {results['vocab_size']:,}")
    print(f"  • Unique tokens used: {results['unique_tokens']:,}")
    print(f"  • Vocab coverage: {results['vocab_coverage']:.2f}%")

    print("\n✓ Decode Accuracy:")
    print(
        f"  • Mismatches: {results['decode_mismatches']:,} / {results['total_samples']:,}"
    )
    print(
        f"  • Accuracy: {(1 - results['decode_mismatches']/results['total_samples'])*100:.2f}%"
    )

    # Percentile analysis
    if tokens_per_char_list:
        print("\n📈 Tokens/char Distribution:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(tokens_per_char_list, p)
            print(f"  {p:2d}th percentile: {val:.4f}")

    # Top tokens
    token_counter = Counter(all_tokens)
    print("\n🔝 Top 15 Most Frequent Tokens:")
    for i, (token_id, count) in enumerate(token_counter.most_common(15), 1):
        token_str = tokenizer.decode([token_id])
        token_repr = (
            repr(token_str) if len(token_str) <= 15 else repr(token_str[:15]) + "..."
        )
        percentage = (count / results["total_tokens"]) * 100
        print(f"  {i:2d}. {token_repr:20s} → {count:8,} ({percentage:5.2f}%)")

    return results


def compare_tokenizers(tokenizer_configs, data_dir, max_files=None):
    """Compare multiple tokenizers on the same dataset."""
    print("=" * 80)
    print("TOKENIZER COMPARISON ON NCERT DATASET")
    print("=" * 80)

    # Load dataset
    print(f"\n📥 Loading dataset from: {data_dir}")
    texts = load_jsonl_dataset(data_dir, max_files=max_files)
    # texts=texts[:10000]
    print(f"✓ Loaded {len(texts):,} samples")

    # Calculate dataset stats
    total_chars = sum(len(t) for t in texts)
    avg_chars = total_chars / len(texts) if texts else 0
    print(f"  • Total characters: {total_chars:,}")
    print(f"  • Avg characters/sample: {avg_chars:.1f}")

    # Evaluate each tokenizer
    all_results = []
    for name, path in tokenizer_configs.items():
        try:
            results = evaluate_tokenizer(path, texts, name)
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"\n❌ Error evaluating {name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Comparison summary
    if len(all_results) >= 2:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        # Sort by tokens per char (lower is better)
        all_results.sort(key=lambda x: x["avg_tokens_per_char"])

        print("\n🏆 Ranking by Tokens/Char (lower is better):")
        print(
            f"{'Rank':<6} {'Tokenizer':<30} {'Tokens/Char':<15} {'Byte Fallback':<15} {'Vocab Coverage':<15}"
        )
        print("-" * 80)

        for i, r in enumerate(all_results, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(
                f"{medal:<6} {r['tokenizer_name']:<30} {r['avg_tokens_per_char']:<15.4f} {r['avg_byte_fallback']:<15.2f} {r['vocab_coverage']:<15.2f}"
            )

        # Detailed comparison
        print("\n📊 Detailed Comparison:")

        # Best tokenizer
        best = all_results[0]
        print(f"\n✅ BEST OVERALL: {best['tokenizer_name']}")
        print(f"  • Tokens/char: {best['avg_tokens_per_char']:.4f}")
        print(f"  • Compression: {best['compression_ratio']:.4f} chars/token")
        print(f"  • Byte fallback: {best['avg_byte_fallback']:.2f}%")

        # Comparison with others
        print(f"\n📈 Efficiency Comparison (vs {best['tokenizer_name']}):")
        for r in all_results[1:]:
            diff_tpc = (
                (r["avg_tokens_per_char"] / best["avg_tokens_per_char"]) - 1
            ) * 100
            diff_bf = r["avg_byte_fallback"] - best["avg_byte_fallback"]
            print(f"  • {r['tokenizer_name']}:")
            print(f"    - Tokens/char: {diff_tpc:+.2f}% worse")
            print(f"    - Byte fallback: {diff_bf:+.2f}% difference")
            print(
                f"    - Token overhead: {(r['total_tokens'] - best['total_tokens']):+,} tokens"
            )

        # Vocab size comparison
        print("\n📚 Vocabulary Size:")
        for r in sorted(all_results, key=lambda x: x["vocab_size"]):
            print(f"  • {r['tokenizer_name']:<30} {r['vocab_size']:>10,} tokens")

    print("\n" + "=" * 80)
    print("✅ COMPARISON COMPLETE")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare tokenizers on NCERT dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/chethan/Downloads/ncert",
        help="Path to NCERT dataset directory",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of JSONL files to process (default: all)",
    )

    args = parser.parse_args()

    # Tokenizers to compare
    tokenizer_configs = {
        "merged_stable": "/Users/chethan/Documents/era/rrf/merged_tokenizer_gptoss_primary_stable",
        "gptoss": "/Users/chethan/Documents/era/rrf/gptoss",
        "gemma": "/Users/chethan/Documents/era/rrf/gemma",
        "deepseek_llm": "/Users/chethan/Documents/era/rrf/deepseek_llm",
    }

    # Verify paths exist
    for name, path in list(tokenizer_configs.items()):
        if not Path(path).exists():
            print(f"⚠️  Warning: {name} not found at {path}, skipping...")
            del tokenizer_configs[name]

    if not tokenizer_configs:
        print("❌ Error: No valid tokenizer paths found")
        sys.exit(1)

    # Run comparison
    results = compare_tokenizers(tokenizer_configs, args.data_dir, args.max_files)
