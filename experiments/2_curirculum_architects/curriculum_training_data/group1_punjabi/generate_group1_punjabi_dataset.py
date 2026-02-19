#!/usr/bin/env python3
# isort: skip_file
"""
Generate Group 1 Punjabi Language Dataset (200,000 question-answer pairs)
Combines all 10 statement types and creates final dataset with minimum 512 tokens per data point.
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import (  # noqa: E402
    combine_qa_pairs_to_reach_min_tokens_hindi,
    count_tokens,
)

# Statement files and their target counts
# We expect S1-S9 to generate ~30k-40k unique pairs.
# S10 will fill the rest to reach 200,000 unique.
STATEMENT_FILES = [
    ("group1_s1.txt", 10000, "S1: Spelling"),
    ("group1_s2.txt", 10000, "S2: Letter Position"),
    ("group1_s3.txt", 5000, "S3: Sound Matching"),
    ("group1_s4.txt", 5000, "S4: Letter Count"),
    ("group1_s5.txt", 2000, "S5: Rhyming"),
    ("group1_s6.txt", 5000, "S6: Classification"),
    ("group1_s7.txt", 10000, "S7: Position of Letter"),
    ("group1_s8.txt", 600, "S8: Number Spelling"),
    ("group1_s9.txt", 5000, "S9: Last Letter"),
    ("group1_s10.txt", 150000, "S10: Word Comparison"),
]

TOTAL_TARGET = 200000
MIN_TOKENS_PER_DATAPOINT = 512


def load_qa_pairs_from_file(filepath: str) -> list[tuple[str, str]]:
    """Load Q&A pairs from a statement file"""
    qa_pairs = []
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found, skipping...")
        return qa_pairs

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse Q? A। format
            if "?" in line:
                # Split by "?" first
                parts = line.split("?", 1)
                if len(parts) == 2:
                    query = parts[0].strip() + "?"
                    answer_part = parts[1].strip()
                    # Split by "।" to get individual answers
                    if "।" in answer_part:
                        # Take first answer (before first "।")
                        answer = answer_part.split("।", 1)[0].strip()
                    else:
                        answer = answer_part.strip()
                    if not answer:
                        print(f"Warning: Skipping empty answer for query: {query}")
                        continue
                    qa_pairs.append((query, answer))

    return qa_pairs


def main():
    """Main function to generate the combined Punjabi dataset"""
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(os.path.dirname(script_dir), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "group1_punjabi.txt")

    print("=" * 80)
    print("Generating Group 1 Punjabi Dataset")
    print("=" * 80)

    # Load all Q&A pairs from statement files
    all_qa_pairs = []
    total_loaded = 0

    for filename, target_count, description in STATEMENT_FILES:
        filepath = os.path.join(script_dir, filename)
        qa_pairs = load_qa_pairs_from_file(filepath)
        all_qa_pairs.extend(qa_pairs)
        total_loaded += len(qa_pairs)
        print(f"{description}: Loaded {len(qa_pairs)} pairs (target: {target_count})")

    print(f"\nTotal Q&A pairs loaded: {total_loaded}")
    print(f"Target: {TOTAL_TARGET}")

    if total_loaded < TOTAL_TARGET:
        print(
            f"Warning: Only {total_loaded} pairs loaded, less than target {TOTAL_TARGET}"
        )

    # Shuffle all pairs
    random.shuffle(all_qa_pairs)

    # Combine pairs into data points with minimum token count
    print(
        f"\nCombining pairs into data points (min {MIN_TOKENS_PER_DATAPOINT} tokens each)..."
    )
    combined_samples = combine_qa_pairs_to_reach_min_tokens_hindi(
        all_qa_pairs, min_tokens=MIN_TOKENS_PER_DATAPOINT
    )

    print(f"Generated {len(combined_samples)} data points")

    # Verify token counts
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

    # Write to output file
    print(f"\nWriting to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in combined_samples:
            f.write(sample + "\n")

    print("\n✓ Dataset generation complete!")
    print(f"  Output file: {output_file}")
    print(f"  Total data points: {len(combined_samples)}")
    print(f"  Total Q&A pairs: {total_loaded}")
    print("=" * 80)


if __name__ == "__main__":
    main()
