import numpy as np

# ---------------------------------------------------------
# Core affinity computation
# ---------------------------------------------------------


def compute_affinities(Ps, Pg, weights, eps=1e-12):
    """
    Computes expert affinities and domain-agnostic (nullness) signal.
    """
    log_odds = np.log((Ps + eps) / (Pg + eps))

    max_affinity = log_odds.max(axis=0)
    argmax_affinity = log_odds.argmax(axis=0)

    # Weighted variance across modalities
    mean = np.average(log_odds, axis=0, weights=weights)
    var = np.average((log_odds - mean) ** 2, axis=0, weights=weights)

    # High when token is domain-agnostic
    domain_agnostic_score = 1.0 / (var + eps)

    return {
        "log_odds": log_odds,
        "max_affinity": max_affinity,
        "argmax_affinity": argmax_affinity,
        "domain_agnostic_score": domain_agnostic_score,
    }


# ---------------------------------------------------------
# Null routing score (single source of truth)
# ---------------------------------------------------------


def compute_null_routing_score(Pg, max_affinity, domain_agnostic_score, eps=1e-12):
    """
    Expected null expert load contributed by each token.
    """
    affinity_suppression = 1.0 / (np.abs(max_affinity) + eps)

    return Pg * affinity_suppression * domain_agnostic_score


# ---------------------------------------------------------
# Token classification (ALL gating happens here)
# ---------------------------------------------------------


def classify_tokens(Pg, max_affinity, null_routing_score, cfg):
    """
    Returns token buckets with proper frequency gating.
    """

    valid_freq = Pg > cfg.min_global_prob

    junk_mask = (Pg < cfg.junk_prob_thresh) & (
        np.abs(max_affinity) < cfg.low_affinity_thresh
    )

    null_like_mask = valid_freq & (np.abs(max_affinity) < cfg.low_affinity_thresh)

    # Rank ONLY valid-frequency tokens
    gated_score = null_routing_score.copy()
    gated_score[~valid_freq] = -np.inf

    top_null_ids = np.argsort(-gated_score)[: cfg.null_topk]

    return {
        "junk_ids": np.where(junk_mask)[0],
        "null_like_ids": np.where(null_like_mask)[0],
        "top_null_ids": top_null_ids,
    }
