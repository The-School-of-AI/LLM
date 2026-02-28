"""
Retrieval metrics: Hit@k, MRR, MRR@K (paper standard), Recall@k, NDCG@k.
MRR@10 is the official metric for MS MARCO / IndicMSMARCO (only rank <= 10 counts).
"""

from __future__ import annotations

import numpy as np


def compute_retrieval_metrics(
    similarity_matrix: np.ndarray,
    gold_indices: np.ndarray | None = None,
    mrr_at_k: int = 10,
    recall_at_k_list: tuple[int, ...] = (1, 5, 10),
    ndcg_at_k: int = 10,
) -> dict[str, float]:
    """
    Compute retrieval metrics from (n_queries, n_passages) similarity matrix.

    - similarity_matrix: higher = more similar.
    - gold_indices: (n_queries,) index of relevant passage per query. If None, diagonal.
    - mrr_at_k: only count reciprocal rank if rank <= mrr_at_k (0 otherwise). Paper standard = 10.
    - recall_at_k_list: k values for Recall@k.
    - ndcg_at_k: NDCG computed at this k.

    Returns: hit_at_1, mrr, mrr_at_k (e.g. mrr_at_10), recall_at_1, recall_at_5, recall_at_10, ndcg_at_10, n.
    """
    n_q, n_p = similarity_matrix.shape
    if gold_indices is None:
        gold_indices = np.arange(n_q)
    else:
        gold_indices = np.asarray(gold_indices).ravel()
    assert len(gold_indices) == n_q

    hit_at_1 = 0.0
    mrr_sum = 0.0
    mrr_at_k_sum = 0.0
    recall_sums: dict[int, float] = {k: 0.0 for k in recall_at_k_list}
    ndcg_sum = 0.0

    for i in range(n_q):
        ranks = np.argsort(similarity_matrix[i])[::-1]
        gold_pos = np.where(ranks == gold_indices[i])[0][0]
        rank_1based = gold_pos + 1

        if rank_1based == 1:
            hit_at_1 += 1
        mrr_sum += 1.0 / rank_1based
        if rank_1based <= mrr_at_k:
            mrr_at_k_sum += 1.0 / rank_1based
        # else: contribution 0 for MRR@K

        for k in recall_at_k_list:
            if rank_1based <= k:
                recall_sums[k] += 1.0

        # NDCG@k: single relevant doc, so DCG = 1/log2(rank+1), IDCG = 1, NDCG = DCG
        if rank_1based <= ndcg_at_k:
            ndcg_sum += 1.0 / np.log2(rank_1based + 1)
        # else NDCG contribution 0 (or could use 0)

    n = float(n_q)
    result: dict[str, float] = {
        "hit_at_1": hit_at_1 / n if n else 0.0,
        "mrr": mrr_sum / n if n else 0.0,
        f"mrr_at_{mrr_at_k}": mrr_at_k_sum / n if n else 0.0,
        "n": int(n),
    }
    for k in recall_at_k_list:
        result[f"recall_at_{k}"] = recall_sums[k] / n if n else 0.0
    result[f"ndcg_at_{ndcg_at_k}"] = ndcg_sum / n if n else 0.0
    return result
