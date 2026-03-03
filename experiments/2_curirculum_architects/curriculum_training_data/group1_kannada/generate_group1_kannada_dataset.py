#!/usr/bin/env python3
# isort: skip_file
"""
Generate Group 1 Kannada Language Dataset (200,000 question-answer pairs)
Combines all 10 statement types and creates final dataset with minimum 512 tokens per data point.
Mirrors the Hindi implementation; uses same format (Q? A।) and combine logic.
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import combine_qa_pairs_to_reach_min_tokens_kannada, count_tokens

# Statement files and their target counts
STATEMENT_FILES = [
    ("group1_s1.txt", 28600, "S1: Spelling + Letter Listing"),
    ("group1_s2.txt", 25800, "S2: Letter Position"),
    ("group1_s3.txt", 20000, "S3: Sound Matching"),
    ("group1_s4.txt", 25800, "S4: Letter Count"),
    ("group1_s5.txt", 20000, "S5: Rhyming"),
    ("group1_s6.txt", 10000, "S6: Classification"),
    ("group1_s7.txt", 21200, "S7: Position of Letter"),
    ("group1_s8.txt", 12000, "S8: Number Spelling"),
    ("group1_s9.txt", 19200, "S9: Last Letter"),
    ("group1_s10.txt", 13000, "S10: Word Comparison"),
    ("group1_s11.txt", 10000, "S11: Ottakshara & Kagunita"),
]

TOTAL_TARGET = 210000
MIN_TOKENS_PER_DATAPOINT = 512


def load_qa_pairs_from_file(filepath: str) -> list[tuple[str, str]]:
    """Load Q&A pairs from a statement file (format: Q? A। per line)"""
    qa_pairs = []
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found, skipping...")
        return qa_pairs

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "?" in line:
                parts = line.split("?", 1)
                if len(parts) == 2:
                    query = parts[0].strip() + "?"
                    answer_part = parts[1].strip()
                    # Strip both . and । (Kannada uses . not ।)
                    answer = answer_part.rstrip(".").rstrip("।").strip()
                    qa_pairs.append((query, answer))

    return qa_pairs


def main():
    """Main function to generate the combined Kannada dataset"""
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(os.path.dirname(script_dir), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "group1_kannada.txt")

    print("=" * 80)
    print("Generating Group 1 Kannada Dataset")
    print("=" * 80)

    all_qa_pairs = []
    total_loaded = 0

    for filename, target_count, description in STATEMENT_FILES:
        filepath = os.path.join(script_dir, filename)
        qa_pairs = load_qa_pairs_from_file(filepath)
        all_qa_pairs.extend(qa_pairs)
        total_loaded += len(qa_pairs)
        print(f"{description}: Loaded {len(qa_pairs)} pairs (target: {target_count})")

    print(f"\nTotal Q&A pairs loaded: {total_loaded}")

    # Curated set: deduplicate by (query, answer), no duplicates
    seen = set()
    unique_pairs = []
    for q, a in all_qa_pairs:
        key = (q, a)
        if key not in seen:
            seen.add(key)
            unique_pairs.append((q, a))
    if len(unique_pairs) < total_loaded:
        print(f"Deduplicated: {total_loaded} -> {len(unique_pairs)} unique pairs")
    all_qa_pairs = unique_pairs
    total_loaded = len(all_qa_pairs)

    print(f"Target: {TOTAL_TARGET}")
    if total_loaded < TOTAL_TARGET:
        print(
            f"Warning: Only {total_loaded} pairs loaded, less than target {TOTAL_TARGET}"
        )

    random.shuffle(all_qa_pairs)

    print(
        f"\nCombining pairs into data points (min {MIN_TOKENS_PER_DATAPOINT} tokens each)..."
    )
    combined_samples = combine_qa_pairs_to_reach_min_tokens_kannada(
        all_qa_pairs, min_tokens=MIN_TOKENS_PER_DATAPOINT
    )

    print(f"Generated {len(combined_samples)} data points")

    print("\nVerifying token counts...")
    token_counts = [count_tokens(sample) for sample in combined_samples]
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    avg_tokens = sum(token_counts) / len(token_counts)

    print(f"Token count - Min: {min_tokens}, Max: {max_tokens}, Avg: {avg_tokens:.1f}")

    if min_tokens < MIN_TOKENS_PER_DATAPOINT:
        print(
            f"Warning: Some samples have fewer than {MIN_TOKENS_PER_DATAPOINT} tokens!"
        )

    print(f"\nWriting to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in combined_samples:
            f.write(sample)

    print("\n✓ Dataset generation complete!")
    print(f"  Output file: {output_file}")
    print(f"  Total data points: {len(combined_samples)}")
    print(f"  Total Q&A pairs: {total_loaded}")
    print("=" * 80)


if __name__ == "__main__":
    main()
