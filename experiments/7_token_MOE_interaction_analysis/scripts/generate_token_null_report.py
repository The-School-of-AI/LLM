#!/usr/bin/env python3

import argparse
from datetime import datetime

import numpy as np
from transformers import AutoTokenizer


def token_str(tokenizer, tid):
    return tokenizer.convert_ids_to_tokens([int(tid)])[0]


def generate_report(
    output_path,
    tokenizer,
    Pg,
    max_aff,
    null_score,
    argmax_aff,
    modalities,
    min_prob,
    low_aff_thresh,
    junk_prob_thresh,
    topk=50,
):

    # ----------------------------
    # Null-attracting tokens
    # ----------------------------
    valid = Pg > min_prob
    null_rank = np.argsort(-null_score * valid)[:topk]

    # ----------------------------
    # Junk tokens
    # ----------------------------
    junk_mask = (Pg < junk_prob_thresh) & (np.abs(max_aff) < low_aff_thresh)
    junk_ids = np.where(junk_mask)[0][:topk]

    # ----------------------------
    # Clusters (simple heuristics)
    # ----------------------------
    morph_suffixes = []
    glue_words = []
    separators = []

    for tid in null_rank:
        tok = token_str(tokenizer, tid)
        if tok.endswith(("er", "ed", "ing", "s")):
            morph_suffixes.append(tid)
        elif tok.strip() in {",", ".", "(", ")", '"', "'"}:
            separators.append(tid)
        elif tok.startswith("Ġ"):
            glue_words.append(tid)

    # ----------------------------
    # Write markdown
    # ----------------------------
    with open(output_path, "w") as f:
        f.write("# Token Null Routing Map\n\n")
        f.write(f"_Auto-generated on {datetime.utcnow().isoformat()} UTC_\n\n")

        # ---- Section 1
        f.write("## 1. Top Token IDs by Null-Routing Affinity\n\n")
        f.write("| Rank | Token | Pg | Max Affinity | Null Score |\n")
        f.write("|------|-------|----|--------------|------------|\n")

        for i, tid in enumerate(null_rank):
            tok = token_str(tokenizer, tid)
            f.write(
                f"| {i+1} | `{tok}` | {Pg[tid]:.2e} | "
                f"{max_aff[tid]:.2e} | {null_score[tid]:.2e} |\n"
            )

        # ---- Section 2
        f.write("\n## 2. Junk Token Signatures (Filtered)\n\n")
        f.write("| Token | Pg | Notes |\n")
        f.write("|-------|----|-------|\n")

        for tid in junk_ids:
            tok = token_str(tokenizer, tid)
            f.write(f"| `{tok}` | {Pg[tid]:.2e} | rare + no affinity |\n")

        # ---- Section 3
        f.write("\n## 3. Null-Attracting Token Clusters\n\n")

        def write_cluster(title, tids):
            f.write(f"### {title}\n\n")
            for tid in tids[:15]:
                tok = token_str(tokenizer, tid)
                f.write(f"- `{tok}` (Pg={Pg[tid]:.1e})\n")
            f.write("\n")

        write_cluster("Morphological fragments", morph_suffixes)
        write_cluster("Lexical glue tokens", glue_words)
        write_cluster("Structural separators", separators)

        # ---- Section 4 (placeholder)
        f.write(
            "## 4. Local Token Patterns (Bigrams / Trigrams)\n\n"
            "_Not yet computed. Planned extension._\n\n"
        )

        # ---- Notes
        f.write(
            "## 5. Notes\n\n"
            "- Null-attracting tokens are frequent and domain-agnostic.\n"
            "- Junk tokens are rare and excluded via frequency gating.\n"
            "- Token behavior is dataset-dependent; rerun as data improves.\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--affinity_npz", required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--output", default="token_null_map.md")

    parser.add_argument("--min_prob", type=float, default=1e-5)
    parser.add_argument("--junk_prob", type=float, default=1e-8)
    parser.add_argument("--low_aff", type=float, default=1e-3)
    parser.add_argument("--topk", type=int, default=30)

    args = parser.parse_args()

    data = np.load(args.affinity_npz, allow_pickle=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    generate_report(
        output_path=args.output,
        tokenizer=tokenizer,
        Pg=data["Pg"],
        max_aff=data["max_affinity"],
        null_score=data["null_score"],
        argmax_aff=data["argmax_affinity"],
        modalities=data["modalities"],
        min_prob=args.min_prob,
        low_aff_thresh=args.low_aff,
        junk_prob_thresh=args.junk_prob,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
