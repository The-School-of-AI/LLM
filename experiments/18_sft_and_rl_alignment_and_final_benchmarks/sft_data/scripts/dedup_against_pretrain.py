#!/usr/bin/env python3
"""
Deduplicate SFT data against pre-training data (hash-based).
Team 18 — SFT Data 7.1 checklist item 6.
Requires a file of pre-training hashes (one per line) from Team 5 or your pipeline.
"""
import json
import hashlib
import argparse
from pathlib import Path


def normalize_for_hash(text: str) -> str:
    """Normalize text before hashing: strip, collapse whitespace."""
    return " ".join((text or "").strip().split())


def conversation_to_hash(conversation: dict) -> str:
    """Single hash for the whole conversation (order-dependent)."""
    turns = conversation.get("conversations", [])
    parts = [f"{t['role']}:{normalize_for_hash(t.get('content') or '')}" for t in turns]
    blob = "\n".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_pretrain_hashes(path: Path) -> set[str]:
    """Load set of pre-training hashes (one hash per line)."""
    out = set()
    with open(path) as f:
        for line in f:
            h = line.strip()
            if h:
                out.add(h)
    return out


def main():
    ap = argparse.ArgumentParser(description="Dedup SFT JSONL against pre-training hashes.")
    ap.add_argument("input", type=Path, help="Input SFT JSONL (standardized conversations)")
    ap.add_argument("output", type=Path, help="Output JSONL (duplicates removed)")
    ap.add_argument("--pretrain-hashes", type=Path, required=True,
                    help="File with one pre-training hash per line (from Team 5 or pretrain pipeline)")
    ap.add_argument("--dedup-within-sft", action="store_true",
                    help="Also remove duplicates within SFT (same hash appearing twice)")
    args = ap.parse_args()

    pretrain = load_pretrain_hashes(args.pretrain_hashes)
    print(f"Loaded {len(pretrain)} pre-training hashes from {args.pretrain_hashes}")

    seen_sft = set()
    kept = 0
    dropped_pretrain = 0
    dropped_internal = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line)
            h = conversation_to_hash(conv)
            if h in pretrain:
                dropped_pretrain += 1
                continue
            if args.dedup_within_sft and h in seen_sft:
                dropped_internal += 1
                continue
            seen_sft.add(h)
            fout.write(line + "\n")
            kept += 1

    print(f"Kept: {kept}, dropped (vs pretrain): {dropped_pretrain}, dropped (internal dup): {dropped_internal}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
