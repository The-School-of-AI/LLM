#!/usr/bin/env python3
"""
Top 16K Token Frequency Analysis for Hybrid Tokenizer
======================================================
Tokenizes raw_shard.parquet (sampled) + SFT text files using the hybrid
tokenizer and produces:
  1. token_frequency.csv   — full vocab sorted by frequency
  2. top_16k_token_stats.md — detailed stats report

Usage:
  pip install transformers pyarrow pandas tqdm
  python top16k_analysis.py [--shard-rows 50000]
"""

import argparse, csv, json, sys, time
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from tqdm import tqdm

# ── CLI ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Top 16K Token Analysis")
parser.add_argument("--shard-rows", type=int, default=50_000,
                    help="Max rows from raw_shard.parquet (default 50000, 0=all)")
parser.add_argument("--tokenizer", default=None,
                    help="Path to tokenizer dir (default: same dir as script)")
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).parent.resolve()
TOKENIZER_DIR = Path(args.tokenizer) if args.tokenizer else SCRIPT_DIR
DOWNLOADS = Path("/Users/jayantgurushrivastava/Downloads")
SFT_DIR = Path("/Users/jayantgurushrivastava/Documents/LLM_1.0/LLM/tests/6_tokenizer_design_lab/tokenizer_validation/sft_data")

RAW_SHARD = DOWNLOADS / "raw_shard.parquet"
OUTPUT_DIR = SCRIPT_DIR / "report"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Load tokenizer ───────────────────────────────────────────────
print("=" * 60)
print("  Loading tokenizer...")
print("=" * 60)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=True)
VOCAB_SIZE = len(tokenizer)
print(f"  Vocab size: {VOCAB_SIZE:,}")
print(f"  BOS: {tokenizer.bos_token}  EOS: {tokenizer.eos_token}")
print()

# ── Tokenize & count ────────────────────────────────────────────
freq = Counter()
total_docs = 0
t0 = time.time()

# 1) raw_shard.parquet
if RAW_SHARD.exists():
    pf = pq.ParquetFile(str(RAW_SHARD))
    total_rows = pf.metadata.num_rows
    rows_to_tok = total_rows if args.shard_rows == 0 else min(args.shard_rows, total_rows)
    print(f"  raw_shard.parquet: {total_rows:,} total rows, tokenizing {rows_to_tok:,}")

    scanned = 0
    for batch in tqdm(pf.iter_batches(batch_size=2000, columns=["text"]),
                      desc="  Tokenizing shard", unit="batch"):
        remaining = rows_to_tok - scanned
        if remaining <= 0:
            break
        texts = batch.column("text").to_pylist()[:remaining]
        for text in texts:
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            freq.update(ids)
            total_docs += 1
        scanned += len(texts)
        if scanned >= rows_to_tok:
            break

    print(f"  Tokenized {total_docs:,} docs from shard")
else:
    print(f"  ⚠️  raw_shard.parquet not found at {RAW_SHARD}")

# 2) SFT text files
sft_files = sorted(SFT_DIR.glob("*.txt")) if SFT_DIR.exists() else []
for txt_file in sft_files:
    with open(txt_file, encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip("\r\n") for l in f if l.strip()]
    for line in tqdm(lines, desc=f"  {txt_file.name}", unit="line", leave=False):
        ids = tokenizer.encode(line, add_special_tokens=False)
        freq.update(ids)
        total_docs += 1
    print(f"  ✅ {txt_file.name}: {len(lines):,} lines")

elapsed = time.time() - t0
total_tokens = sum(freq.values())
print(f"\n  Done! {total_docs:,} docs, {total_tokens:,} tokens in {elapsed:.1f}s")
print()

# ── Build sorted frequency list ─────────────────────────────────
print("  Building frequency table...")
all_entries = []
for i in range(VOCAB_SIZE):
    raw = tokenizer.convert_ids_to_tokens(i) if i < VOCAB_SIZE else ""
    decoded = tokenizer.decode([i], skip_special_tokens=False)
    all_entries.append({
        "token_id": i,
        "token_raw": raw or "",
        "token_decoded": decoded,
        "count": freq.get(i, 0),
    })

all_entries.sort(key=lambda x: -x["count"])

# ── Write token_frequency.csv ───────────────────────────────────
csv_path = OUTPUT_DIR / "token_frequency.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["token_id", "token_raw", "token_decoded", "count"])
    w.writeheader()
    w.writerows(all_entries)
print(f"  ✅ Wrote {csv_path}")

# ── Compute stats ───────────────────────────────────────────────
non_zero = [e for e in all_entries if e["count"] > 0]
zero_count = VOCAB_SIZE - len(non_zero)

top_16k = all_entries[:16000]
top_16k_total = sum(e["count"] for e in top_16k)
coverage_16k = 100.0 * top_16k_total / total_tokens

# Coverage milestones
milestones_table = []
for k in [100, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, len(non_zero)]:
    if k > len(all_entries):
        continue
    s = sum(e["count"] for e in all_entries[:k])
    pct = 100.0 * s / total_tokens
    milestones_table.append((k, pct))

# Cumulative coverage milestones
cum = 0
pct_milestones = [50, 75, 80, 85, 90, 95, 97, 99, 99.5, 99.9]
pct_results = []
mi = 0
for i, e in enumerate(all_entries):
    cum += e["count"]
    pct = 100.0 * cum / total_tokens
    while mi < len(pct_milestones) and pct >= pct_milestones[mi]:
        pct_results.append((pct_milestones[mi], i + 1))
        mi += 1
    if mi >= len(pct_milestones):
        break

# Categorize top 16K
categories = Counter()
indic_list = []
for e in top_16k:
    decoded = e["token_decoded"]
    raw = e["token_raw"]
    if "�" in decoded:
        categories["byte_fragments"] += 1; continue
    has_indic = False
    for ch in decoded.strip():
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F:
            has_indic = True; break
    if has_indic:
        categories["indic_tokens"] += 1
        indic_list.append(e); continue
    stripped = decoded.strip()
    if not stripped and len(decoded) > 0:
        categories["whitespace"] += 1; continue
    if len(stripped) == 1:
        if stripped.isdigit(): categories["numbers"] += 1
        elif stripped.isalpha(): categories["single_chars"] += 1
        else: categories["punctuation"] += 1
        continue
    if stripped.isdigit():
        categories["numbers"] += 1; continue
    if raw.startswith("Ġ") and stripped.isalpha():
        categories["english_words"] += 1; continue
    if stripped.isalpha() and not raw.startswith("Ġ"):
        categories["subwords"] += 1; continue
    categories["other"] += 1

# Frequency buckets
freq_buckets = {"10M+": 0, "1M–10M": 0, "100K–1M": 0, "10K–100K": 0, "1K–10K": 0, "<1K": 0}
for e in top_16k:
    c = e["count"]
    if c >= 10_000_000: freq_buckets["10M+"] += 1
    elif c >= 1_000_000: freq_buckets["1M–10M"] += 1
    elif c >= 100_000: freq_buckets["100K–1M"] += 1
    elif c >= 10_000: freq_buckets["10K–100K"] += 1
    elif c >= 1_000: freq_buckets["1K–10K"] += 1
    else: freq_buckets["<1K"] += 1

# ── Write markdown report ───────────────────────────────────────
md_path = OUTPUT_DIR / "top_16k_token_stats.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Top 16K Token Statistics — Hybrid Tokenizer\n\n")
    f.write(f"**Tokenizer:** `TSAI-131k Hybrid`  \n")
    f.write(f"**Corpus:** {total_docs:,} documents ({total_tokens:,} tokens)  \n")
    f.write(f"**Shard rows tokenized:** {args.shard_rows if args.shard_rows else 'ALL'}  \n")
    f.write(f"**SFT files:** {len(sft_files)}  \n")
    f.write(f"**Analysis time:** {elapsed:.1f}s  \n\n")
    f.write("---\n\n")

    f.write("## Executive Summary\n\n")
    f.write("| Metric | Value |\n|--------|-------|\n")
    f.write(f"| Total vocabulary | {VOCAB_SIZE:,} |\n")
    f.write(f"| Tokens with ≥1 occurrence | {len(non_zero):,} ({100*len(non_zero)/VOCAB_SIZE:.1f}%) |\n")
    f.write(f"| Zero-frequency tokens | {zero_count:,} ({100*zero_count/VOCAB_SIZE:.1f}%) |\n")
    f.write(f"| **Top 16K coverage** | **{coverage_16k:.2f}%** of corpus |\n")
    f.write(f"| Total corpus tokens | {total_tokens:,} |\n")
    f.write(f"| Top 16K token count | {top_16k_total:,} |\n")
    f.write(f"| Min frequency in top 16K | {top_16k[-1]['count']:,} |\n\n")
    f.write("---\n\n")

    f.write("## Cumulative Coverage by Vocab Size\n\n")
    f.write("| Top-K Tokens | Corpus Coverage |\n|:---:|:---:|\n")
    for k, pct in milestones_table:
        bold = "**" if k == 16000 else ""
        f.write(f"| {bold}{k:,}{bold} | {bold}{pct:.2f}%{bold} |\n")
    f.write("\n")

    f.write("## Coverage Milestones\n\n")
    f.write("| Coverage % | Tokens Needed |\n|:---:|:---:|\n")
    for pct, cnt in pct_results:
        f.write(f"| {pct:.1f}% | {cnt:,} |\n")
    f.write("\n---\n\n")

    f.write("## Token Category Breakdown (Top 16K)\n\n")
    f.write("| Category | Count | % of 16K |\n|----------|------:|:---:|\n")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        f.write(f"| {cat} | {count:,} | {100*count/16000:.1f}% |\n")
    f.write("\n")

    f.write("## Frequency Buckets (Top 16K)\n\n")
    f.write("| Range | Count |\n|:---:|:---:|\n")
    for bucket, count in freq_buckets.items():
        f.write(f"| {bucket} | {count:,} |\n")
    f.write("\n---\n\n")

    f.write("## Top 50 Most Frequent Tokens\n\n")
    f.write("| Rank | ID | Token | Count | Cum. % |\n|---:|---:|-------|------:|:---:|\n")
    cum = 0
    for i, e in enumerate(all_entries[:50]):
        cum += e["count"]
        cpct = 100 * cum / total_tokens
        tok_display = repr(e["token_decoded"])
        if len(tok_display) > 30:
            tok_display = tok_display[:27] + "..."
        f.write(f"| {i+1} | {e['token_id']} | {tok_display} | {e['count']:,} | {cpct:.2f}% |\n")
    f.write("\n---\n\n")

    f.write("## Boundary of Top 16K (Ranks 15,991–16,000)\n\n")
    f.write("| Rank | ID | Token | Count |\n|---:|---:|-------|------:|\n")
    for i, e in enumerate(all_entries[15990:16000]):
        f.write(f"| {15991+i} | {e['token_id']} | {repr(e['token_decoded'])} | {e['count']:,} |\n")
    f.write("\n---\n\n")

    f.write(f"## Indic Tokens in Top 16K ({len(indic_list)} total)\n\n")
    f.write("| Rank | ID | Token | Count |\n|---:|---:|-------|------:|\n")
    for e in indic_list[:30]:
        rank = top_16k.index(e) + 1
        f.write(f"| {rank} | {e['token_id']} | {repr(e['token_decoded'])} | {e['count']:,} |\n")
    if len(indic_list) > 30:
        f.write(f"\n*... and {len(indic_list)-30} more Indic tokens in top 16K*\n")
    f.write("\n")

print(f"  ✅ Wrote {md_path}")
print()
print(f"  === SUMMARY ===")
print(f"  Top 16K coverage: {coverage_16k:.2f}%")
print(f"  Non-zero tokens:  {len(non_zero):,} / {VOCAB_SIZE:,}")
print(f"  Indic in top 16K: {len(indic_list)}")
print(f"  Cutoff count:     {top_16k[-1]['count']:,}")
