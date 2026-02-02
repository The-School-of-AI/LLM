import json
import os
import random

results_dir = "/home/ubuntu/results"


def load_samples(filename):
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def print_params(name, samples):
    print(f"\n{'=' * 20} {name} (Total: {len(samples)}) {'=' * 20}")


def analyze_b0(samples):
    print_params("B0 (Nursery/Short)", samples)
    if not samples:
        return

    # Random samples
    print("\n--- Random Samples ---")
    for s in random.sample(samples, min(3, len(samples))):
        print(
            f"ID: {s['id']} | Len: {len(s['full_text'].split())} words | Rationale: {s['rationale']}"
        )
        print(f"Snippet: {s['text_snippet']}")

    # Outlier check: Longest B0 text
    longest = max(samples, key=lambda x: len(x["full_text"].split()))
    print("\n--- Outlier: Longest B0 Sample ---")
    print(
        f"ID: {longest['id']} | Len: {len(longest['full_text'].split())} words | Rationale: {longest['rationale']}"
    )
    print(f"Full Text: {longest['full_text'][:500]}...")


def analyze_b1(samples):
    print_params("B1 (Primary)", samples)
    if not samples:
        return

    # Random samples
    print("\n--- Random Samples ---")
    for s in random.sample(samples, min(3, len(samples))):
        print(
            f"ID: {s['id']} | Len: {len(s['full_text'].split())} words | Rationale: {s['rationale']}"
        )
        print(f"Snippet: {s['text_snippet']}")

    # Outlier check: Longest B1 text (might belong higher?)
    longest = max(samples, key=lambda x: len(x["full_text"].split()))
    print("\n--- Outlier: Longest B1 Sample ---")
    print(
        f"ID: {longest['id']} | Len: {len(longest['full_text'].split())} words | Rationale: {longest['rationale']}"
    )
    print(f"Full Text: {longest['full_text'][:500]}...")


def analyze_b2(samples):
    print_params("B2 (High School)", samples)
    if not samples:
        return

    # Random samples
    print("\n--- Random Samples ---")
    for s in random.sample(samples, min(3, len(samples))):
        print(
            f"ID: {s['id']} | Len: {len(s['full_text'].split())} words | Rationale: {s['rationale']}"
        )
        print(f"Snippet: {s['text_snippet']}")


def analyze_b3(samples):
    print_params("B3 (Undergraduate)", samples)
    if not samples:
        return

    # Show all since there are only 9
    print("\n--- All B3 Samples ---")
    for s in samples:
        print(
            f"ID: {s['id']} | Len: {len(s['full_text'].split())} words | Rationale: {s['rationale']}"
        )
        print(f"Snippet: {s['text_snippet']}")


def main():
    b0 = load_samples("B0.jsonl")
    b1 = load_samples("B1.jsonl")
    b2 = load_samples("B2.jsonl")
    b3 = load_samples("B3.jsonl")
    b4 = load_samples("B4.jsonl")

    analyze_b0(b0)
    analyze_b1(b1)

    print("\n" + "=" * 50)
    print("--- NEW LOGIC CHECKS ---")

    print_params("B2 (High School) - Should be substantial paragraphs", b2)
    if b2:
        print("\n--- B2 Samples (Check for listicles) ---")
        for s in random.sample(b2, min(5, len(b2))):
            print(f"ID: {s['id']} | Rationale: {s['rationale']}")
            print(f"Snippet: {s['text_snippet'][:100]}...")

    print_params("B3 (Undergraduate) - Should be clean code/reasoning", b3)
    if b3:
        print("\n--- B3 Samples (Check for false code positives) ---")
        for s in b3:
            print(f"ID: {s['id']} | Rationale: {s['rationale']}")
            print(f"Snippet: {s['text_snippet'][:100]}...")

    if b4:
        print_params("B4 (Graduate)", b4)


if __name__ == "__main__":
    main()
