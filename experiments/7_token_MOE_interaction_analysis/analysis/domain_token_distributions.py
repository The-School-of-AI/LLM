#!/usr/bin/env python3

import argparse
import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer

# For each modality, calculate what tokens appear most often

# -----------------------------
# CONFIG
# -----------------------------

DOMAINS = [
    "general_text",
    "indic_text",
    "structured_knowledge",
    # "research_papers",
    # "technical_text",
    "code",
    "math",
    "cot_reasoning",
    "agentic_traces",
]

SOURCE_DEFAULT_MODALITY = {
    # Web
    "C4": "general_text",
    "refinedweb": "general_text",
    "cc_head": "general_text",
    "cc_middle": "general_text",
    "cc_tail": "general_text",
    "cc_news": "structured_knowledge",
    "reddit": "general_text",
    "books": "structured_knowledge",
    "megawika": "structured_knowledge",

    # Code
    "Starcoder": "code",
    "stackexchange": "structured_knowledge",

    # Math / formal
    "proof_pile_2-algebraic_stack": "math",
    "proof_pile_2-open_web_math": "math",
    "redpajama-arxiv": "structured_knowledge",
    "ncert": "structured_knowledge",
    "pes2o": "structured_knowledge",

    # Instruction
    "flan": "cot_reasoning",

    # Indic
    "sangraha_as": "general_text",
    "sangraha_bn": "general_text",
    "sangraha_gu": "general_text",
    "sangraha_hi": "general_text",
    "sangraha_kn": "general_text",
    "sangraha_ml": "general_text",
    "sangraha_mr": "general_text",
    "sangraha_or": "general_text",
    "sangraha_pa": "general_text",
    "sangraha_ta": "general_text",
    "sangraha_te": "general_text",
}

INDIC_SOURCES = {
    "sangraha_as",
    "sangraha_bn",
    "sangraha_gu",
    "sangraha_hi",
    "sangraha_kn",
    "sangraha_ml",
    "sangraha_mr",
    "sangraha_or",
    "sangraha_pa",
    "sangraha_ta",
    "sangraha_te",
}

DEFAULT_EPS = 1e-8

# -----------------------------
# Modality mapping logic
# -----------------------------


def map_to_modality(row):
    src = row.get("source")

    # ---- LANGUAGE SPLIT FIRST ----
    if src in INDIC_SOURCES:
        return "indic_text"

    # Baseline from source
    base = SOURCE_DEFAULT_MODALITY.get(src, "general_text")

    # ---- THEN DOMAIN / CODE / MATH / REASONING ----
    # Strong overrides only
    agentic = row.get("agentic_score", 0) or 0
    cot = row.get("cot_score", 0) or 0
    code = row.get("code_score", 0) or 0
    math = row.get("math_score", 0) or 0

    # Override only if signal is strong enough
    if agentic >= 12:
        return "agentic_traces"

    if cot >= 15:
        return "cot_reasoning"

    if code >= 30:
        return "code"

    if math >= 20:
        return "math"

    return base

# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to combined raw_shard.parquet")
    parser.add_argument("--tokenizer", type=str, default="../tsai_131k_tokenizer/")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_tokens_per_doc", type=int, default=8192)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--output", type=str, default="artifacts/domain_token_distributions.npz")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    vocab_size = len(tokenizer)
    print("Vocab size:", vocab_size)

    # Allocate counters.
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

    # Process the large parquet file in batches to save memory
    pf = pq.ParquetFile(args.input_file)
    print(f"Processing {args.input_file} in batches...")
    
    # We need text + metadata flags
    columns = ["text", "domain", "source", "has_code", "has_cot", "has_reasoning", "has_agentic", 
                "math_score", "cot_score", "code_score", "reasoning_score", "agentic_score",
                ]
    
    # Filter columns based on availability in the file to avoid error if some are missing
    available_cols = [c for c in columns if c in pf.schema.names]
    row_counts = {d: 0 for d in DOMAINS}
    
    for batch in tqdm(pf.iter_batches(batch_size=args.batch_size, columns=available_cols), total=pf.num_row_groups):
        df_batch = batch.to_pandas()
        
        for _, row in df_batch.iterrows():
            text = row.get(args.text_field)
            if not text or not isinstance(text, str):
                continue

            dom = map_to_modality(row)
            row_counts[dom] += 1
            
            batch_texts.append(text)
            batch_domains.append(dom)

        flush_batch()

    # Normalize
    P = {}
    for d in DOMAINS:
        if totals[d] == 0:
            print(f"WARNING: No data for modality '{d}' - using uniform prior")
            P[d] = np.ones(vocab_size) / vocab_size
            continue

        Pd = counts[d].astype(np.float64) / totals[d]
        Pd += args.eps
        Pd /= Pd.sum()
        P[d] = Pd

    save_dict = {}
    for d in DOMAINS:
        save_dict[d] = P[d]
        save_dict[d + "__total"] = np.array([totals[d]], dtype=np.int64)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez(args.output, **save_dict)
    print("Saved:", args.output)

    # Sanity
    print("\nTop tokens per modality:")
    for d in DOMAINS:
        if totals[d] == 0: continue
        top = np.argsort(-P[d])[:20]
        print("\n", d)
        print(tokenizer.convert_ids_to_tokens(top.tolist()))

    print("\nTotals:")
    for d in DOMAINS:
        print(f"{d:28s} {totals[d]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
