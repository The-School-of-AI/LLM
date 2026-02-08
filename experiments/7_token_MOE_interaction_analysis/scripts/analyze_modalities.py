import argparse

import numpy as np
from transformers import AutoTokenizer


def main():
    # Computes global token distribution using saved token counts and computes log-odds per modality to get:
    # 1) distinctive tokens per modality
    # 2) null-token candidates
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="domain_token_distributions.npz")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--null_topk", type=int, default=50)
    parser.add_argument("--min_tokens", type=int, default=5000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.model_max_length = 10**9
    MIN_TOKENS = args.min_tokens

    # Higher value for more syntactic nulls and fewer semantic words. Eg more towards "and", "in", "was" and
    # less towards "important", "developments" etc
    MIN_GLOBAL_PROB = 1e-5  # 5e-6 #3e-6 #1e-6

    data = np.load(args.npz)

    # Filter small modalities
    modalities = [k for k in data.files if not k.endswith("__total")]
    print("Modalities:", modalities)
    for m in modalities:
        print(m, data[m + "__total"][0])

    valid_modalities = [m for m in modalities if data[m + "__total"][0] > MIN_TOKENS]

    print("Keeping modalities:", valid_modalities)
    modalities = valid_modalities

    if len(modalities) == 0:
        raise ValueError("No modalities passed min_tokens threshold")

    # modality weights from true totals
    weights = np.array([data[m + "__total"][0] for m in modalities], dtype=np.float64)
    weights /= weights.sum()

    # Stack distributions
    Ps = np.stack([data[m] for m in modalities], axis=0)

    # Global distribution
    # Pg = Ps.sum(axis=0)
    Pg = np.average(
        Ps, axis=0, weights=weights
    )  # Re-normalize Pg with modality weights
    Pg /= Pg.sum()

    eps = 1e-12

    # Log-odds per modality
    log_odds = np.log((Ps + eps) / (Pg + eps))

    # ---------------------------------------
    # Expert affinity calculations
    # ---------------------------------------

    # log_odds shape: [num_modalities, vocab_size]

    # Max affinity: strongest expert preference per token
    max_affinity = log_odds.max(axis=0)

    # Also useful: which expert it prefers
    argmax_affinity = log_odds.argmax(axis=0)

    # Optional: mean absolute affinity (secondary signal)
    mean_abs_affinity = np.mean(np.abs(log_odds), axis=0)

    # Inspect low-affinity tokens (null candidates), ie tokens that do not strongly prefer any expert
    # Only consider tokens with higher frequency. Low freq tokens are junk
    mask = Pg > MIN_GLOBAL_PROB
    masked_aff = np.abs(max_affinity).copy()
    masked_aff[~mask] = np.inf
    low_affinity_ids = np.argsort(masked_aff)[: args.topk]
    low_affinity_tokens = tokenizer.convert_ids_to_tokens(low_affinity_ids.tolist())

    # ---------------------------------------
    # Distinctive tokens per modality
    # ---------------------------------------

    for i, m in enumerate(modalities):
        scores = log_odds[i]
        top = np.argsort(-scores)[: args.topk]
        toks = tokenizer.convert_ids_to_tokens(top.tolist())

        print("\n==============================")
        print(m)
        print(toks)

    # ---------------------------------------
    # Null token candidates
    # Tokens with near-zero variance of log-odds
    # ---------------------------------------

    # var = log_odds.var(axis=0) # This is not accurate when there is very high imbalance between modalities

    # Variance across modalities - use weighted variance
    mean = np.average(log_odds, axis=0, weights=weights)
    var = np.average((log_odds - mean) ** 2, axis=0, weights=weights)
    print("Weights:", dict(zip(modalities, weights)))

    # Only consider tokens with enough total probability
    # null_ids = np.argsort(var)[: args.null_topk]
    mask = Pg > MIN_GLOBAL_PROB
    filtered_var = var.copy()
    filtered_var[~mask] = np.inf
    null_ids = np.argsort(filtered_var)[: args.null_topk]
    null_tokens = tokenizer.convert_ids_to_tokens(null_ids.tolist())

    # ---------------------------------------
    # Distinctive tokens per modality
    # ---------------------------------------

    # Identify junk tokens
    LOW_AFF_THRESH = 1e-3  # near-zero affinity
    LOW_PROB_THRESH = 1e-8  # very rare tokens

    junk_mask = (np.abs(max_affinity) < LOW_AFF_THRESH) & (Pg < LOW_PROB_THRESH)

    junk_ids = np.where(junk_mask)[0][: args.topk]
    junk_tokens = tokenizer.convert_ids_to_tokens(junk_ids.tolist())

    print("\n==============================")
    print("NULL TOKEN CANDIDATES (lowest variance across modalities):")
    print(null_tokens)

    # Also show most globally frequent
    print("\nMost globally frequent tokens:")
    top_global = np.argsort(-Pg)[: args.null_topk]
    print(tokenizer.convert_ids_to_tokens(top_global.tolist()))

    # Top tokens by affinity
    null_score = var  # weighted variance of log_odds
    top_affinity_ids = np.argsort(-max_affinity)[: args.topk]
    top_affinity_tokens = tokenizer.convert_ids_to_tokens(top_affinity_ids.tolist())

    print("\n==============================")
    print("TOP TOKENS BY EXPERT AFFINITY")
    for tid, tok in zip(top_affinity_ids, top_affinity_tokens):
        print(
            f"{tok:>15s} | max_aff={max_affinity[tid]:.3f} | "
            f"null_score={null_score[tid]:.2e} | "
            f"expert={modalities[argmax_affinity[tid]]}"
        )

    print("\n==============================")
    print("LOW EXPERT AFFINITY TOKENS (NULL-LIKE)")
    for tid, tok in zip(low_affinity_ids, low_affinity_tokens):
        print(
            f"{tok:>15s} | max_aff={max_affinity[tid]:.3e} | "
            f"null_score={null_score[tid]:.2e}"
        )

    print("\n==============================")
    print("JUNK TOKEN CANDIDATES (rare + low affinity):")
    for tid, tok in zip(junk_ids, junk_tokens):
        print(f"{tok:>15s} | Pg={Pg[tid]:.2e} | " f"max_aff={max_affinity[tid]:.2e}")

    # Save affinity stats
    np.savez(
        "token_affinity_stats.npz",
        max_affinity=max_affinity,
        mean_abs_affinity=mean_abs_affinity,
        null_score=null_score,
        Pg=Pg,
        argmax_affinity=argmax_affinity,
        modalities=np.array(modalities),
    )


if __name__ == "__main__":
    main()
