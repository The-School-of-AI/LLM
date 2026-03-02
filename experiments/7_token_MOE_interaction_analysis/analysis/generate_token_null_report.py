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
    max_affinity,
    null_routing_score,
    argmax_affinity,
    modalities,
    min_prob,
    low_aff_thresh,
    junk_prob_thresh,
    topk=20,
):
    # ----------------------------
    # Frequency-gated null ranking
    # ----------------------------
    valid_mask = Pg > min_prob
    gated_score = null_routing_score.copy()
    gated_score[~valid_mask] = -np.inf

    null_rank = np.argsort(-gated_score)[:topk]

    # ----------------------------
    # Junk tokens (rare + no affinity)
    # ----------------------------
    junk_mask = (Pg < junk_prob_thresh) & (np.abs(max_affinity) < low_aff_thresh)
    junk_ids = np.where(junk_mask)[0][:topk]

    # ----------------------------
    # Simple clustering heuristics
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
        f.write("# Token Null Routing Report\n\n")
        f.write(f"_Auto-generated on {datetime.utcnow().isoformat()} UTC_\n\n")

        # ---- Section 1
        f.write("## 1. Top Tokens by Null Routing Score (Frequency-Gated)\n\n")
        f.write("| Rank | Token | Pg | Max Affinity | Null Routing Score |\n")
        f.write("|------|-------|----|--------------|--------------------|\n")

        for i, tid in enumerate(null_rank):
            tok = token_str(tokenizer, tid)
            f.write(
                f"| {i+1} | `{tok}` | {Pg[tid]:.2e} | "
                f"{max_affinity[tid]:.2e} | {null_routing_score[tid]:.2e} |\n"
            )

        # ---- Section 2
        f.write("\n## 2. Junk Token Candidates (Rare + No Affinity)\n\n")
        f.write("| Token | Pg | Max Affinity |\n")
        f.write("|-------|----|--------------|\n")

        for tid in junk_ids:
            tok = token_str(tokenizer, tid)
            f.write(f"| `{tok}` | {Pg[tid]:.2e} | {max_affinity[tid]:.2e} |\n")

        # ---- Section 3
        f.write("\n## 3. Null-Attracting Token Clusters\n\n")

        def write_cluster(title, tids):
            f.write(f"### {title}\n\n")
            for tid in tids[:15]:
                tok = token_str(tokenizer, tid)
                f.write(
                    f"- `{tok}` (Pg={Pg[tid]:.1e}, "
                    f"null_score={null_routing_score[tid]:.1e})\n"
                )
            f.write("\n")

        write_cluster("Morphological fragments", morph_suffixes)
        write_cluster("Lexical glue tokens", glue_words)
        write_cluster("Structural separators", separators)

        # ---- Section 4
        f.write(
            "## 4. Notes\n\n"
            "- Null routing score already includes frequency gating.\n"
            "- Junk tokens are intentionally excluded from rankings.\n"
            "- High scores indicate load on the null / shared pathway.\n"
            "- Results stabilize as dataset size increases.\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--affinity_npz", required=True)
    parser.add_argument("--tokenizer", default="../tsai_131k_tokenizer/")
    parser.add_argument("--output", default="reports/token_null_map.md")

    parser.add_argument("--min_prob", type=float, default=1e-5)
    parser.add_argument("--junk_prob", type=float, default=1e-8)
    parser.add_argument("--low_aff", type=float, default=3e-3)
    parser.add_argument("--topk", type=int, default=20)

    args = parser.parse_args()

    data = np.load(args.affinity_npz, allow_pickle=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    generate_report(
        output_path=args.output,
        tokenizer=tokenizer,
        Pg=data["Pg"],
        max_affinity=data["max_affinity"],
        null_routing_score=data["null_routing_score"],
        argmax_affinity=data["argmax_affinity"],
        modalities=data["modalities"],
        min_prob=args.min_prob,
        low_aff_thresh=args.low_aff,
        junk_prob_thresh=args.junk_prob,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
