#!/usr/bin/env python3
"""
Decontaminate SFT data against benchmark test sets (hash-based).
Team 18 — Dataset → Benchmark Coverage Matrix; ensures no benchmark contamination.
Removes SFT examples whose prompt (user content) or full conversation hashes
match any benchmark test-set hash. Use same hash function as dedup_against_pretrain.py.
"""
import json
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict


def normalize_for_hash(text: str) -> str:
    """Normalize text before hashing: strip, collapse whitespace. Same as dedup_against_pretrain."""
    return " ".join((text or "").strip().split())


def conversation_to_hash_full(conversation: dict) -> str:
    """Hash whole conversation (order-dependent). Same as dedup_against_pretrain."""
    turns = conversation.get("conversations", [])
    parts = [f"{t['role']}:{normalize_for_hash(t.get('content') or '')}" for t in turns]
    blob = "\n".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def conversation_to_hash_prompt_only(conversation: dict) -> str:
    """Hash only user-side content (prompt) for benchmark overlap check."""
    turns = conversation.get("conversations", [])
    user_parts = [normalize_for_hash(t.get("content") or "") for t in turns if t.get("role") == "user"]
    blob = "\n".join(user_parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_benchmark_hashes_from_file(path: Path) -> set[str]:
    """Load set of hashes from a file (one hash per line)."""
    out = set()
    with open(path) as f:
        for line in f:
            h = line.strip()
            if h:
                out.add(h)
    return out


def load_benchmark_hashes_from_jsonl(path: Path, text_field: str) -> set[str]:
    """Load benchmark items from JSONL, extract text_field, normalize and hash. Returns set of hashes."""
    hashes = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_field) or obj.get("question") or obj.get("input") or obj.get("prompt") or ""
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)
            text = normalize_for_hash(str(text))
            if text:
                hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
    return hashes


def main():
    ap = argparse.ArgumentParser(
        description="Remove SFT examples that match benchmark test-set hashes (no benchmark contamination)."
    )
    ap.add_argument("input", type=Path, help="Input SFT JSONL (standardized conversations)")
    ap.add_argument("output", type=Path, help="Output JSONL (contaminated examples removed)")
    ap.add_argument(
        "--benchmark-hashes",
        type=Path,
        action="append",
        default=[],
        metavar="FILE",
        help="File with one benchmark test-set hash per line. Can be repeated for multiple benchmarks.",
    )
    ap.add_argument(
        "--benchmark-hashes-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory of hash files (e.g. math_test.txt, gsm8k_test.txt). Each file = one benchmark.",
    )
    ap.add_argument(
        "--benchmark-jsonl",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help="Benchmark test set as JSONL; extract text and hash. Format 'name:path'. Uses --text-field.",
    )
    ap.add_argument(
        "--text-field",
        type=str,
        default="question",
        help="Field to extract from --benchmark-jsonl for hashing (default: question).",
    )
    ap.add_argument(
        "--hash-mode",
        choices=["prompt", "full"],
        default="prompt",
        help="Hash SFT by 'prompt' (user content only) or 'full' (entire conversation). Default: prompt.",
    )
    ap.add_argument(
        "--removed-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional: write removed (contaminated) examples to this JSONL for audit.",
    )
    args = ap.parse_args()

    # Build union of all benchmark hashes (with optional names for reporting)
    all_bench_hashes: set[str] = set()
    benchmark_sources: list[tuple[str, Path]] = []  # (name, path) for reporting which benchmark caused drop

    for p in args.benchmark_hashes:
        h = load_benchmark_hashes_from_file(p)
        all_bench_hashes |= h
        benchmark_sources.append((p.name, p))

    if args.benchmark_hashes_dir and args.benchmark_hashes_dir.is_dir():
        for f in sorted(args.benchmark_hashes_dir.iterdir()):
            if f.is_file():
                h = load_benchmark_hashes_from_file(f)
                all_bench_hashes |= h
                benchmark_sources.append((f.name, f))

    for spec in args.benchmark_jsonl:
        if ":" not in spec:
            raise SystemExit(f"--benchmark-jsonl must be 'name:path', got: {spec}")
        name, path = spec.split(":", 1)
        path = Path(path.strip())
        if not path.exists():
            raise SystemExit(f"Benchmark JSONL not found: {path}")
        h = load_benchmark_hashes_from_jsonl(path, args.text_field)
        all_bench_hashes |= h
        benchmark_sources.append((name.strip(), path))

    if not all_bench_hashes:
        raise SystemExit(
            "No benchmark hashes loaded. Use --benchmark-hashes, --benchmark-hashes-dir, or --benchmark-jsonl."
        )

    print(f"Loaded {len(all_bench_hashes)} benchmark test-set hashes from {len(benchmark_sources)} source(s)")

    hash_fn = conversation_to_hash_prompt_only if args.hash_mode == "prompt" else conversation_to_hash_full
    print(f"Hash mode: {args.hash_mode} (SFT example)")

    kept = 0
    removed = 0
    removed_lines: list[str] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line)
            h = hash_fn(conv)
            if h in all_bench_hashes:
                removed += 1
                if args.removed_out is not None:
                    removed_lines.append(line)
                continue
            fout.write(line + "\n")
            kept += 1

    print(f"Kept: {kept}, removed (benchmark overlap): {removed}")
    print(f"Output: {args.output}")

    if args.removed_out is not None and removed_lines:
        args.removed_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.removed_out, "w") as f:
            for ln in removed_lines:
                f.write(ln + "\n")
        print(f"Removed examples written to: {args.removed_out}")

    if removed > 0:
        print("(Decontamination complete; no benchmark test-set overlap in output.)")


if __name__ == "__main__":
    main()
