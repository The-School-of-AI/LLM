#!/usr/bin/env python3
"""
Create train/val split from SFT JSONL with fixed seed for reproducibility.
Team 18 — SFT Data 7.1 checklist item 7.
"""
import json
import argparse
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Train/val split for SFT JSONL.")
    ap.add_argument("input", type=Path, help="Input JSONL")
    ap.add_argument("--train-out", type=Path, required=True, help="Output train JSONL")
    ap.add_argument("--val-out", type=Path, required=True, help="Output val JSONL")
    ap.add_argument("--val-ratio", type=float, default=0.05, help="Validation fraction (default 0.05)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    lines = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    random.seed(args.seed)
    random.shuffle(lines)
    n = len(lines)
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    train_lines = lines[:n_train]
    val_lines = lines[n_train:]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.train_out, "w") as f:
        for line in train_lines:
            f.write(line + "\n")
    with open(args.val_out, "w") as f:
        for line in val_lines:
            f.write(line + "\n")

    print(f"Seed: {args.seed}, val_ratio: {args.val_ratio}")
    print(f"Train: {n_train} -> {args.train_out}")
    print(f"Val:   {n_val}   -> {args.val_out}")


if __name__ == "__main__":
    main()
