import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt_utils import count_tokens


def verify_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_pairs = 0
    all_pairs = set()
    low_token_counts = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Count tokens
        tokens = count_tokens(line)
        if tokens < 512:
            low_token_counts.append((i + 1, tokens))

        # Split into pairs
        pairs = line.split("।")
        pairs = [p.strip() + "।" for p in pairs if p.strip()]

        total_pairs += len(pairs)
        for p in pairs:
            all_pairs.add(p)

    print(f"Total lines (data points): {len(lines)}")
    print(f"Total Q&A pairs: {total_pairs}")
    print(f"Unique Q&A pairs: {len(all_pairs)}")
    print(f"Duplicates: {total_pairs - len(all_pairs)}")

    if low_token_counts:
        print(f"Lines with less than 512 tokens: {len(low_token_counts)}")
        for line_num, count in low_token_counts[:5]:
            print(f"  Line {line_num}: {count} tokens")
    else:
        print("All lines have at least 512 tokens.")


if __name__ == "__main__":
    verify_dataset(
        r"z:\era-v4\capstone\lightning-language-models\LLM2\experiments\2_curirculum_architects\curriculum_training_data\output\group1_punjabi.txt"
    )
