#!/usr/bin/env python3
"""
Build a file of hashes (one per line) from a benchmark test set JSONL.
Use this to create inputs for decontaminate_against_benchmarks.py.
Same hash function (normalize + SHA256) so SFT prompt hashes can be compared.
Team 18 — benchmark decontamination pipeline.
"""
import json
import hashlib
import argparse
from pathlib import Path


def normalize_for_hash(text: str) -> str:
    """Same as dedup_against_pretrain and decontaminate_against_benchmarks."""
    return " ".join((text or "").strip().split())


def main():
    ap = argparse.ArgumentParser(description="Build benchmark test-set hash file from JSONL.")
    ap.add_argument("input", type=Path, help="Benchmark test set JSONL (e.g. MATH test, HumanEval prompts)")
    ap.add_argument("output", type=Path, help="Output file: one hash per line")
    ap.add_argument(
        "--text-field",
        type=str,
        default="question",
        help="Field to extract for hashing (default: question). Tried: question, input, prompt, problem.",
    )
    ap.add_argument(
        "--concat-fields",
        type=str,
        default=None,
        metavar="FIELDS",
        help="Comma-separated fields to concatenate (e.g. question,choices). Overrides --text-field.",
    )
    args = ap.parse_args()

    hashes = set()
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if args.concat_fields:
                parts = []
                for field in args.concat_fields.split(","):
                    v = obj.get(field.strip())
                    if v is not None:
                        parts.append(str(v) if not isinstance(v, list) else " ".join(str(x) for x in v))
                text = " ".join(parts)
            else:
                text = (
                    obj.get(args.text_field)
                    or obj.get("question")
                    or obj.get("input")
                    or obj.get("prompt")
                    or obj.get("problem")
                    or ""
                )
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)
            text = normalize_for_hash(str(text))
            if text:
                hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for h in sorted(hashes):
            f.write(h + "\n")

    print(f"Wrote {len(hashes)} hashes to {args.output} (from {args.input})")


if __name__ == "__main__":
    main()
