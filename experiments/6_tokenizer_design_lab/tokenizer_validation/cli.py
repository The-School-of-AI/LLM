from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

if __package__:
    from .datasets import SampledText, sample_golden, sample_pretraining, sample_sft
    from .metrics import analyze_byte_fallback, length_stats
    from .sft_template import render_chat, try_split_prompt_answer
else:
    # Allows running this file directly:
    #   python3 experiments/6_tokenizer_design_lab/tokenizer_validation/cli.py
    _THIS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(_THIS_DIR))
    from datasets import SampledText, sample_golden, sample_pretraining, sample_sft
    from metrics import analyze_byte_fallback, length_stats
    from sft_template import render_chat, try_split_prompt_answer


def _load_tokenizer(tokenizer_dir: Path):
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))


def _normalize_ws(s: str) -> str:
    return " ".join(s.split())


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_sft_chats(samples: list[SampledText], *, system_prompt: str | None) -> tuple[list[str], dict[str, int]]:
    chats: list[str] = []
    split_stats = {"paired": 0, "unpaired": 0}

    for s in samples:
        maybe = try_split_prompt_answer(s.text)
        if maybe is None:
            split_stats["unpaired"] += 1
            chat = render_chat(system=system_prompt, user=s.text, assistant="")
            chats.append(chat)
            continue

        split_stats["paired"] += 1
        prompt, answer = maybe
        chats.append(render_chat(system=system_prompt, user=prompt, assistant=answer))

    return chats, split_stats


def _unk_rate(input_ids: list[int], *, unk_id: int | None) -> float:
    if not input_ids:
        return 0.0
    if unk_id is None:
        return 0.0
    return sum(1 for x in input_ids if x == unk_id) / len(input_ids)


def validate_dataset(
    *,
    name: str,
    texts: list[str],
    tokenizer,
    out_dir: Path,
    roundtrip_exact_on_ascii: bool,
) -> dict[str, Any]:
    unk_id = getattr(tokenizer, "unk_token_id", None)

    lengths: list[int] = []
    unk_rates: list[float] = []
    byte_rates: list[float] = []
    mismatches: list[dict[str, Any]] = []

    # Track special token leakage in raw text (especially for pretraining)
    special_ids = set(tokenizer.all_special_ids)
    special_leakage_count = 0

    # Pre-compute byte token IDs for this vocab (assume len(token)==1 means fallback byte token in BPE)
    vocab = tokenizer.get_vocab()
    byte_ids = {v for k, v in vocab.items() if len(k) == 1 or k.startswith("<0x")}

    for i, t in enumerate(tqdm(texts, desc=f"tokenize:{name}", leave=False)):
        enc = tokenizer(t, add_special_tokens=False, truncation=False)
        input_ids = enc["input_ids"]
        lengths.append(len(input_ids))
        unk_rates.append(_unk_rate(input_ids, unk_id=unk_id))
        byte_rates.append(analyze_byte_fallback(input_ids, byte_ids))

        # Check if any special token IDs were produced by tokenizing this raw text
        # (For SFT, we expect special tokens because we injected them via template, so we skip leakage counting for SFT)
        if name != "sft":
            if any(tid in special_ids for tid in input_ids):
                special_leakage_count += 1

        dec = tokenizer.decode(input_ids, skip_special_tokens=False)
        is_ascii = all(ord(ch) < 128 for ch in t)

        ok = False
        if roundtrip_exact_on_ascii and is_ascii:
            ok = dec == t
        else:
            ok = _normalize_ws(dec) == _normalize_ws(t)

        if not ok and len(mismatches) < 200:
            mismatches.append(
                {
                    "idx": i,
                    "orig": t[:2000],
                    "decoded": dec[:2000],
                }
            )

    stats = length_stats(lengths)
    avg_unk = sum(unk_rates) / len(unk_rates) if unk_rates else 0.0
    avg_byte = sum(byte_rates) / len(byte_rates) if byte_rates else 0.0
    mismatch_rate = (len(mismatches) / len(texts)) if texts else 0.0

    # write artifacts
    _ensure_dir(out_dir)
    with (out_dir / f"{name}_mismatches.json").open("w", encoding="utf-8") as f:
        json.dump(mismatches, f, ensure_ascii=False, indent=2)

    return {
        "dataset": name,
        "count": stats.count,
        "len_mean": stats.mean,
        "len_p50": stats.p50,
        "len_p90": stats.p90,
        "len_p95": stats.p95,
        "len_p99": stats.p99,
        "len_max": stats.max_len,
        "unk_rate_mean": avg_unk,
        "byte_fallback_rate_mean": avg_byte,
        "roundtrip_mismatch_rate_top200": mismatch_rate,
        "special_token_leakage_count": special_leakage_count if name != "sft" else 0,
    }


def validate_sft_loss_masking(*, chats: list[str], tokenizer, out_dir: Path) -> dict[str, Any]:
    # Minimal feasibility check: ensure that role markers appear in the tokenized stream.
    role_tokens = ["<|system|>", "<|user|>", "<|assistant|>"]

    role_token_ids: dict[str, int | None] = {}
    for rt in role_tokens:
        tid = tokenizer.convert_tokens_to_ids(rt)
        role_token_ids[rt] = None if tid is None or tid == tokenizer.unk_token_id else int(tid)

    missing = [rt for rt, tid in role_token_ids.items() if tid is None]

    found_counts = Counter()
    for t in tqdm(chats, desc="sft:role_tokens", leave=False):
        ids = tokenizer(t, add_special_tokens=False, truncation=False)["input_ids"]
        for rt, tid in role_token_ids.items():
            if tid is None:
                continue
            found_counts[rt] += sum(1 for x in ids if x == tid)

    report = {
        "missing_role_tokens": missing,
        "role_token_ids": role_token_ids,
        "role_token_occurrences": dict(found_counts),
    }
    _ensure_dir(out_dir)
    with (out_dir / "sft_masking_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", type=Path, default=Path("experiments/6_tokenizer_design_lab/tsai_131k_tokenizer"))
    ap.add_argument("--pretraining-parquet", type=Path, default=Path("/home/ubuntu/raw_shard.parquet"))
    ap.add_argument("--golden-jsonl", type=Path, default=Path("/home/ubuntu/golden_samples_cleaned_v3.jsonl"))
    ap.add_argument("--sft-dir", type=Path, default=Path("/home/ubuntu/SFT"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/tokenizer_validation"))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-pretraining", type=int, default=2000)
    ap.add_argument("--n-golden", type=int, default=2000)
    ap.add_argument("--n-sft", type=int, default=2000)
    ap.add_argument("--system-prompt", type=str, default="")

    args = ap.parse_args()

    tokenizer = _load_tokenizer(args.tokenizer_dir)

    system_prompt = args.system_prompt if args.system_prompt.strip() else None

    pre = sample_pretraining(args.pretraining_parquet, n=args.n_pretraining, seed=args.seed)
    gold = sample_golden(args.golden_jsonl, n=args.n_golden, seed=args.seed)

    sft_paths = sorted([p for p in args.sft_dir.glob("*.txt") if p.is_file()])
    sft = sample_sft(sft_paths, n=args.n_sft, seed=args.seed)

    sft_chats, sft_split_stats = _iter_sft_chats(sft, system_prompt=system_prompt)

    texts_by_dataset: dict[str, list[str]] = {
        "pretraining": [s.text for s in pre],
        "golden": [s.text for s in gold],
        "sft": sft_chats,
    }

    dataset_summaries: list[dict[str, Any]] = []
    for name, texts in texts_by_dataset.items():
        summary = validate_dataset(
            name=name,
            texts=texts,
            tokenizer=tokenizer,
            out_dir=args.out_dir,
            roundtrip_exact_on_ascii=True,
        )
        dataset_summaries.append(summary)

    sft_mask_report = validate_sft_loss_masking(chats=sft_chats, tokenizer=tokenizer, out_dir=args.out_dir)

    final = {
        "tokenizer_dir": str(args.tokenizer_dir),
        "seed": args.seed,
        "sizes": {k: len(v) for k, v in texts_by_dataset.items()},
        "sft_split_stats": sft_split_stats,
        "dataset_summaries": dataset_summaries,
        "sft_masking": sft_mask_report,
    }

    _ensure_dir(args.out_dir)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # Flat CSV for quick viewing
    with (args.out_dir / "length_stats.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(dataset_summaries[0].keys()))
        w.writeheader()
        for row in dataset_summaries:
            w.writerow(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
