"""
Retrieval evaluation: encode queries and passages, compute similarity, then metrics.
Default is monolingual (paper protocol): pool per language = that language's passages only.
For IndicMSMARCO the official metric is MRR@10; we also report Hit@1, Recall@k, NDCG@10.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from benchmark_indic_rag_suite.metrics.retrieval_metrics import compute_retrieval_metrics
from benchmark_indic_rag_suite.models.base import RetrievalModelBase

logger = logging.getLogger(__name__)


def run_retrieval_eval(
    data_by_lang: dict[str, list[dict]],
    model: RetrievalModelBase,
    batch_size: int = 16,
    add_cross_lang_negatives: bool = False,
    mrr_at_k: int = 10,
    recall_at_k_list: tuple[int, ...] | list[int] = (1, 5, 10, 20),
    ndcg_at_k_list: tuple[int, ...] | list[int] = (5, 10),
    **encode_kwargs: Any,
) -> dict[str, dict[str, float]]:
    """
    Run retrieval evaluation per language. Pool is monolingual per language (paper protocol)
    unless add_cross_lang_negatives=True (adds other languages as distractors; not in paper).
    """
    results: dict[str, dict[str, float]] = {}
    lang_order = list(data_by_lang.keys())

    if add_cross_lang_negatives and len(lang_order) < 2:
        logger.warning("add_cross_lang_negatives=True but only one language; ignoring")
        add_cross_lang_negatives = False

    all_p_embs: dict[str, np.ndarray] = {}
    for lang in lang_order:
        rows = data_by_lang[lang]
        if not rows:
            continue
        passages = [r["passage"] for r in rows]
        all_p_embs[lang] = model.encode_passages(passages, batch_size=batch_size, **encode_kwargs)

    recall_at_k_list = tuple(recall_at_k_list)
    ndcg_at_k_list = tuple(ndcg_at_k_list)
    for lang, rows in data_by_lang.items():
        if not rows:
            results[lang] = {"hit_at_1": 0.0, "mrr": 0.0, f"mrr_at_{mrr_at_k}": 0.0, "n": 0}
            for k in recall_at_k_list:
                results[lang][f"recall_at_{k}"] = 0.0
                results[lang][f"precision_at_{k}"] = 0.0
                results[lang][f"hit_at_{k}"] = 0.0
            for k in ndcg_at_k_list:
                results[lang][f"ndcg_at_{k}"] = 0.0
            continue
        queries = [r["query"] for r in rows]
        n = len(rows)
        q_embs = model.encode_queries(queries, batch_size=batch_size, **encode_kwargs)

        if add_cross_lang_negatives:
            other_langs = [l for l in lang_order if l != lang and l in all_p_embs]
            pool_parts = [all_p_embs[lang]]
            for o in other_langs:
                pool_parts.append(all_p_embs[o])
            p_embs = np.vstack(pool_parts)
            gold_indices = np.arange(n)
        else:
            p_embs = all_p_embs[lang]
            gold_indices = None

        sim = cosine_similarity(q_embs, p_embs)
        metrics = compute_retrieval_metrics(
            sim,
            gold_indices=gold_indices,
            mrr_at_k=mrr_at_k,
            recall_at_k_list=recall_at_k_list,
            ndcg_at_k_list=ndcg_at_k_list,
        )
        results[lang] = metrics
        recall_10 = metrics.get("recall_at_10", 0.0)
        ndcg_10 = metrics.get("ndcg_at_10", 0.0)
        logger.info(
            "%s: Hit@1=%.4f MRR=%.4f MRR@%d=%.4f Recall@10=%.4f NDCG@10=%.4f n=%d pool=%d",
            lang,
            metrics["hit_at_1"],
            metrics["mrr"],
            mrr_at_k,
            metrics[f"mrr_at_{mrr_at_k}"],
            recall_10,
            ndcg_10,
            metrics["n"],
            p_embs.shape[0],
        )
    return results
