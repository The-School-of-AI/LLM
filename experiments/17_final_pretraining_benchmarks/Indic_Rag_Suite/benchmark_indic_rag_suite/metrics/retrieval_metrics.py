"""
Retrieval metrics: Hit@k, MRR, MRR@K (paper standard), Recall@k, Precision@k, NDCG@k.
MRR@10 is the official metric for MS MARCO / IndicMSMARCO (only rank <= 10 counts).
"""

from __future__ import annotations

import numpy as np


def compute_retrieval_metrics(
    similarity_matrix: np.ndarray,
    gold_indices: np.ndarray | None = None,
    mrr_at_k: int = 10,
    recall_at_k_list: tuple[int, ...] | list[int] = (1, 5, 10, 20),
    ndcg_at_k_list: tuple[int, ...] | list[int] = (5, 10),
) -> dict[str, float]:
    """
    Compute retrieval metrics from (n_queries, n_passages) similarity matrix.

    - similarity_matrix: higher = more similar.
    - gold_indices: (n_queries,) index of relevant passage per query. If None, diagonal.
    - mrr_at_k: only count reciprocal rank if rank <= mrr_at_k (0 otherwise). Paper standard = 10.
    - recall_at_k_list: k values for Recall@k and Precision@k (Precision@k = 1/k if hit in top-k else 0).
    - ndcg_at_k_list: k values for NDCG@k.

    Returns: hit_at_1, hit_at_5, mrr, mrr_at_k, recall_at_k, precision_at_k, ndcg_at_k, n.
    """
    recall_at_k_list = tuple(recall_at_k_list)
    ndcg_at_k_list = tuple(ndcg_at_k_list)
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
    precision_sums: dict[int, float] = {k: 0.0 for k in recall_at_k_list}
    ndcg_sums: dict[int, float] = {k: 0.0 for k in ndcg_at_k_list}

    for i in range(n_q):
        ranks = np.argsort(similarity_matrix[i])[::-1]
        gold_pos = np.where(ranks == gold_indices[i])[0][0]
        rank_1based = gold_pos + 1

        if rank_1based == 1:
            hit_at_1 += 1
        mrr_sum += 1.0 / rank_1based
        if rank_1based <= mrr_at_k:
            mrr_at_k_sum += 1.0 / rank_1based

        for k in recall_at_k_list:
            if rank_1based <= k:
                recall_sums[k] += 1.0
                precision_sums[k] += 1.0 / k  # one relevant doc in top-k -> precision = 1/k

        for k in ndcg_at_k_list:
            if rank_1based <= k:
                ndcg_sums[k] += 1.0 / np.log2(rank_1based + 1)

    n = float(n_q)
    result: dict[str, float] = {
        "hit_at_1": hit_at_1 / n if n else 0.0,
        "mrr": mrr_sum / n if n else 0.0,
        f"mrr_at_{mrr_at_k}": mrr_at_k_sum / n if n else 0.0,
        "n": int(n),
    }
    for k in recall_at_k_list:
        result[f"recall_at_{k}"] = recall_sums[k] / n if n else 0.0
        result[f"precision_at_{k}"] = precision_sums[k] / n if n else 0.0
        result[f"hit_at_{k}"] = recall_sums[k] / n if n else 0.0
    for k in ndcg_at_k_list:
        result[f"ndcg_at_{k}"] = ndcg_sums[k] / n if n else 0.0
    return result
