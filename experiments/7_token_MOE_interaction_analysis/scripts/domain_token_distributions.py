#!/usr/bin/env python3

import argparse
import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer

# -----------------------------
# CONFIG
# -----------------------------

# DOMAINS = [
#     "general_web_clean",
#     "encyclopedic",
#     "news_nonpolitical",
#     "technical_docs",
#     "math_science",
#     "code_repos",
#     "dialogue_chat",
#     "planning_reasoning_curated",
# ]

# Note: These are actually modalities from curriculum metadata, not domains
DOMAINS = [
    "general_text",
    "agentic_traces",
    "research_papers",
    "technical_text",
    "code",
    "math",
    "reasoning",
]

DEFAULT_EPS = 1e-8
JOIN_KEY = "id"  # change if needed

# -----------------------------
# Helpers
# -----------------------------


def drop_null_type_columns(table: pa.Table) -> pa.Table:
    keep = []
    for field in table.schema:
        if not pa.types.is_null(field.type):
            keep.append(field.name)
    return table.select(keep)


def drop_unsupported_columns(table: pa.Table) -> pa.Table:
    keep = []
    for field in table.schema:
        if pa.types.is_struct(field.type):
            continue
        if pa.types.is_list(field.type):
            continue
        if pa.types.is_map(field.type):
            continue
        keep.append(field.name)
    return table.select(keep)


# -----------------------------
# Domain mapping logic
# -----------------------------


def map_to_domain(record: dict) -> str:
    if record.get("has_code", False):
        return "code_repos"

    if record.get("has_math", False):
        return "math_science"

    if record.get("has_reason", False) or record.get("has_agentic", False):
        return "planning_reasoning_curated"

    dataset_domain = (record.get("dataset_domain", "") or "").lower()

    if "wiki" in dataset_domain:
        return "encyclopedic"

    if "news" in dataset_domain:
        return "news_nonpolitical"

    if "dialogue" in dataset_domain or "chat" in dataset_domain:
        return "dialogue_chat"

    if "technical" in dataset_domain or "docs" in dataset_domain:
        return "technical_docs"

    return "general_web_clean"


# def map_to_domain(record):
#     if record.get("has_code"):
#         return "code_repos"
#
#     if record.get("has_math"):
#         return "math_science"
#
#     if record.get("dataset_domain"):
#         d = record["dataset_domain"].lower()
#         if "science" in d or "physics" in d or "math" in d:
#             return "math_science"
#         if "technical" in d:
#             return "technical_docs"
#
#     # Heuristic fallback using entropy + sentence length
#     if record.get("avg_sentence_length", 0) > 20:
#         return "technical_docs"
#
#     return "general_web_clean"

# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_dir", type=str, required=True, help="dataset/parquet/")
    parser.add_argument("--meta_dir", type=str, required=True, help="dataset/meta/")
    parser.add_argument("--tokenizer", type=str, default="gpt-oss")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens_per_doc", type=int, default=8192)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--output", type=str, default="domain_token_distributions.npz")
    args = parser.parse_args()

    text_files = sorted(glob.glob(os.path.join(args.text_dir, "*.parquet")))
    if not text_files:
        raise RuntimeError("No text parquet files found")

    print(f"Found {len(text_files)} text shards")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    vocab_size = tokenizer.vocab_size
    print("Vocab size:", vocab_size)

    # Allocate counters. Use numpy, not python dicts
    counts = {d: np.zeros(vocab_size, dtype=np.int64) for d in DOMAINS}
    totals = {d: 0 for d in DOMAINS}

    batch_texts = []
    batch_domains = []

    def flush_batch():
        if not batch_texts:
            return

        enc = tokenizer(batch_texts, add_special_tokens=False, truncation=False)

        for ids, dom in zip(enc["input_ids"], batch_domains):
            if dom not in counts:
                continue

            ids = ids[: args.max_tokens_per_doc]

            if not ids:
                continue

            ids = np.asarray(ids, dtype=np.int64)
            counts[dom] += np.bincount(ids, minlength=vocab_size)
            totals[dom] += len(ids)

        batch_texts.clear()
        batch_domains.clear()

    # Iterate shards
    for tf in tqdm(text_files, desc="Shards"):
        base = os.path.basename(tf).replace(".parquet", "")
        meta_file = os.path.join(args.meta_dir, f"{base}_metadata.parquet")

        if not os.path.exists(meta_file):
            print("WARNING: metadata missing for", base)
            continue

        # text_table = drop_null_type_columns(pq.read_table(tf))
        # meta_table = drop_null_type_columns(pq.read_table(meta_file))
        text_table = pq.read_table(tf)
        meta_table = pq.read_table(meta_file)
        text_table = drop_unsupported_columns(drop_null_type_columns(text_table))
        meta_table = drop_unsupported_columns(drop_null_type_columns(meta_table))

        # Logging some schema info
        print(text_table.schema.field(JOIN_KEY))
        print(meta_table.schema.field(JOIN_KEY))
        for f in meta_table.schema:
            if pa.types.is_null(f.type):
                print("Dropping null column:", f.name)

        # Normalize join key types (string vs large_string)
        text_table = text_table.set_column(
            text_table.schema.get_field_index(JOIN_KEY),
            JOIN_KEY,
            text_table[JOIN_KEY].cast(pa.string()),
        )
        meta_table = meta_table.set_column(
            meta_table.schema.get_field_index(JOIN_KEY),
            JOIN_KEY,
            meta_table[JOIN_KEY].cast(pa.string()),
        )

        # Inner join on JOIN_KEY
        joined = text_table.join(meta_table, keys=JOIN_KEY, join_type="inner")

        for row in joined.to_pylist():
            text = row.get(args.text_field, None)
            if not text:
                continue

            # dom = map_to_domain(row)
            dom = row.get("modality_primary_modality", "general_text")

            batch_texts.append(text)
            batch_domains.append(dom)

            if len(batch_texts) >= args.batch_size:
                flush_batch()

        flush_batch()

    # Normalize
    P = {}
    for d in DOMAINS:
        if totals[d] == 0:
            P[d] = np.ones(vocab_size) / vocab_size
            continue

        Pd = counts[d].astype(np.float64) / totals[d]
        Pd += args.eps
        Pd /= Pd.sum()
        P[d] = Pd

    np.savez(args.output, **P)
    print("Saved:", args.output)

    # Sanity
    print("\nTop tokens per domain:")
    for d in DOMAINS:
        top = np.argsort(-P[d])[:20]
        print("\n", d)
        print(tokenizer.convert_ids_to_tokens(top.tolist()))

    print("\nTotals:")
    for d in DOMAINS:
        print(f"{d:28s} {totals[d]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
