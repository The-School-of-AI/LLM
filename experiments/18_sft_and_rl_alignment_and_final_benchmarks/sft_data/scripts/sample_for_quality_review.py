#!/usr/bin/env python3
"""
Sample N examples for manual SFT data quality review (e.g. 100+).
Team 18 — SFT Data 7.1 checklist item 5.
"""
import json
import argparse
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Sample SFT examples for manual quality review.")
    ap.add_argument("input", type=Path, help="Input JSONL (standardized conversations)")
    ap.add_argument("output", type=Path, help="Output JSONL with sampled examples")
    ap.add_argument("--n", type=int, default=100, help="Number of examples to sample (default 100)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    lines = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    random.seed(args.seed)
    n = min(args.n, len(lines))
    chosen = random.sample(lines, n)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for line in chosen:
            f.write(line + "\n")

    print(f"Sampled {n} examples for review -> {args.output}")
    print("Review for: formatting, empty assistant replies, benchmark-like prompts, offensive content.")


if __name__ == "__main__":
    main()
