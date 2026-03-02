import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from token_affinity_analysis import (
    classify_tokens,
    compute_affinities,
    compute_null_routing_score,
)
from transformers import AutoTokenizer

# -------------------------
# Config
# -------------------------


class AnalysisConfig:
    min_global_prob = 1e-5
    junk_prob_thresh = 1e-8
    low_affinity_thresh = 3e-3
    null_topk = 20
    eps: float = 1e-12


# -------------------------
# Helpers
# -------------------------


def print_token_table(title, token_ids, tokenizer, Pg, max_aff, null_score, topk=20):
    print("\n==============================")
    print(title)
    for tid in token_ids[:topk]:
        tok = tokenizer.convert_ids_to_tokens([tid])[0]
        print(
            f"{tok:>15s} | "
            f"Pg={Pg[tid]:.2e} | "
            f"max_aff={max_aff[tid]:.2e} | "
            f"null_score={null_score[tid]:.2e}"
        )


def draw_plots(Pg, max_affinity, null_routing_score):
    plt.figure(figsize=(6, 5))
    plt.scatter(Pg, np.abs(max_affinity), s=5, alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Token Probability Pg")
    plt.ylabel("|Max Expert Affinity|")
    plt.title("Token Frequency vs Expert Affinity")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(Pg, null_routing_score, s=5, alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Token Probability Pg")
    plt.ylabel("Null Routing Score")
    plt.title("Null Routing vs Frequency")
    plt.grid(True)
    plt.show()


# -------------------------
# Main
# -------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="artifacts/domain_token_distributions.npz")
    parser.add_argument("--tokenizer", type=str, default="../tsai_131k_tokenizer/")
    parser.add_argument("--min_tokens", type=int, default=5000)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    ARTIFACTS_DIR = "artifacts"
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR, exists_ok=True)

    cfg = AnalysisConfig()
    TOPK = getattr(args, "topk", cfg.null_topk)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.model_max_length = 10**9

    data = np.load(args.npz)

    # -------------------------
    # Load modalities
    # -------------------------

    modalities = [k for k in data.files if not k.endswith("__total")]

    print("Modalities:")
    for m in modalities:
        print(f"  {m}: {data[m + '__total'][0]}")

    modalities = [m for m in modalities if data[m + "__total"][0] > args.min_tokens]

    if not modalities:
        raise ValueError("No modalities passed min_tokens threshold")

    print("Keeping modalities:", modalities)

    weights = np.array(
        [data[m + "__total"][0] for m in modalities],
        dtype=np.float64,
    )
    weights /= weights.sum()

    Ps = np.stack([data[m] for m in modalities], axis=0)

    Pg = np.average(Ps, axis=0, weights=weights)
    Pg /= Pg.sum()

    # -------------------------
    # Entropy diagnostics
    # -------------------------

    print("\n==============================")
    print("ENTROPY PER MODALITY")
    #Lower entropy ⇒ more concentrated distribution ⇒ stronger specialization
    for i, m in enumerate(modalities):
        Pm = Ps[i]
        entropy = -np.sum(Pm * np.log(Pm + cfg.eps))
        print(f"{m:25s} entropy = {entropy:.4f}")

    # Global entropy
    entropy_global = -np.sum(Pg * np.log(Pg + cfg.eps))
    print(f"\nGlobal Pg entropy = {entropy_global:.4f}")

    # Higher perplexity → more spread tokens. Lower perplexity → sharper modality
    perplexity = np.exp(entropy)
    print(f"{m:25s} entropy={entropy:.4f} | perplexity={perplexity:.2f}")

    print("\n==============================")
    print("KL Divergence Matrix")

    for i, mi in enumerate(modalities):
        for j, mj in enumerate(modalities):
            if i == j:
                continue
            KL = np.sum(Ps[i] * (np.log(Ps[i] + cfg.eps) - np.log(Ps[j] + cfg.eps)))
            print(f"KL({mi:20s} || {mj:20s}) = {KL:.4f}")

    """Notes: 
    KL < 0.3 → almost same distribution
    KL ~ 0.5–1 → moderately different
    KL > 2 → strongly different
    KL > 5 → essentially different language space
    """

    # -------------------------
    # Core analysis
    # -------------------------

    aff = compute_affinities(Ps, Pg, weights, cfg.eps)

    null_routing_score = compute_null_routing_score(
        Pg,
        aff["max_affinity"],
        aff["domain_agnostic_score"],
    )

    buckets = classify_tokens(
        Pg,
        aff["max_affinity"],
        null_routing_score,
        cfg,
    )

    # ---------------------------------------
    # TOP TOKENS BY EXPERT AFFINITY
    # ---------------------------------------

    print("\n==============================")
    print("TOP TOKENS BY EXPERT AFFINITY")

    top_affinity_ids = np.argsort(-aff["max_affinity"])[:TOPK]

    for tid in top_affinity_ids:
        tok = tokenizer.convert_ids_to_tokens([tid])[0]
        expert_id = aff["argmax_affinity"][tid]
        expert_name = modalities[expert_id]

        print(
            f"{tok:>15s} | "
            f"max_aff={aff['max_affinity'][tid]:.3f} | "
            f"Pg={Pg[tid]:.2e} | "
            f"expert={expert_name}"
        )

    print("\n==============================")
    print("TOP TOKENS PER EXPERT")

    for i, m in enumerate(modalities):
        scores = aff["log_odds"][i]
        top_ids = np.argsort(-scores)[:TOPK]

        print(f"\n--- {m} ---")
        for tid in top_ids:
            tok = tokenizer.convert_ids_to_tokens([tid])[0]
            print(f"{tok:>15s} | " f"log_odds={scores[tid]:.3f} | " f"Pg={Pg[tid]:.2e}")

    # -------------------------
    # Console diagnostics
    # -------------------------

    print("\n==============================")
    print("TOP TOKENS BY NULL ROUTING SCORE (frequency-gated)")
    for tid in buckets["top_null_ids"][:20]:
        tok = tokenizer.convert_ids_to_tokens([tid])[0]
        print(
            f"{tok:>15s} | "
            f"Pg={Pg[tid]:.2e} | "
            f"max_aff={aff['max_affinity'][tid]:.2e} | "
            f"null_score={null_routing_score[tid]:.2e}"
        )

    print_token_table(
        "NULL-LIKE TOKENS (frequent + low affinity)",
        buckets["null_like_ids"],
        tokenizer,
        Pg,
        aff["max_affinity"],
        null_routing_score,
    )

    print_token_table(
        "JUNK TOKEN CANDIDATES (rare + low affinity)",
        buckets["junk_ids"],
        tokenizer,
        Pg,
        aff["max_affinity"],
        null_routing_score,
    )

    # -------------------------
    # Save artifacts
    # -------------------------

    np.savez(
        os.path.join(ARTIFACTS_DIR, "token_affinity_stats.npz"),
        Pg=Pg,
        null_routing_score=null_routing_score,
        max_affinity=aff["max_affinity"],
        argmax_affinity=aff["argmax_affinity"],
        domain_agnostic_score=aff["domain_agnostic_score"],
        modalities=np.array(modalities),
        log_odds=aff["log_odds"],
    )

    print("\nSaved token_affinity_stats.npz")

    # if not args.no_plots:
    #     draw_plots(Pg, aff["max_affinity"], null_routing_score)


if __name__ == "__main__":
    main()
