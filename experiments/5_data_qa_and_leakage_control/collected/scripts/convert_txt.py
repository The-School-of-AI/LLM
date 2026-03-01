#!/usr/bin/env python3
"""
Convert a plain-text Q&A file to JSONL for the contamination scanner.

Input format (multiple Q&A pairs packed on each line):
    What is the capital of France? Paris. What color is the sky? Blue.

Each Q&A pair becomes its own JSONL record.

Usage:
    python scripts/convert_txt.py <input.txt> <output.jsonl>
"""

import json
import re
import sys
from pathlib import Path


def extract_qa_pairs(line: str) -> list[str]:
    """Split a line of packed Q&A pairs into individual 'question answer' strings."""
    # Match: anything ending with '?' followed by the answer ending with '.'
    pairs = re.findall(r"([^?.]+\?)\s+([^?.]+\.)", line)
    return [f"{q.strip()} {a.strip()}" for q, a in pairs]


def convert(input_path: str, output_path: str):
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    records = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            pairs = extract_qa_pairs(text)
            for pair in pairs:
                records.append({"id": f"qa_{len(records) + 1}", "text": pair})

    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Converted {len(records)} Q&A pairs -> {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/convert_txt.py <input.txt> <output.jsonl>")
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2])
