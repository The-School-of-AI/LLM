"""
Optional generation evaluator: RAGAS.
Not required for Indic-RAG-Suite; the required evaluation is retrieval + EM/F1 (custom/small models).
RAGAS adds extra metrics (e.g. faithfulness, answer relevancy) and can run with local small models.
Install ragas separately to use; missing deps are handled gracefully.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Env vars for Azure OpenAI (used when set so RAGAS uses Azure instead of OpenAI)
AZURE_OPENAI_API_KEY = "AZURE_OPENAI_API_KEY"
AZURE_OPENAI_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_CHAT_DEPLOYMENT = "AZURE_OPENAI_CHAT_DEPLOYMENT"  # e.g. gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"  # e.g. text-embedding-ada-002
AZURE_OPENAI_API_VERSION = "AZURE_OPENAI_API_VERSION"  # e.g. 2024-02-15-preview

# Env vars for RAGAS with local (small) models when no OpenAI/Azure keys are set
RAGAS_LOCAL_LLM = "RAGAS_LOCAL_LLM"  # default: google/flan-t5-small
RAGAS_LOCAL_EMBEDDING = "RAGAS_LOCAL_EMBEDDING"  # default: sentence-transformers/paraphrase-MiniLM-L3-v2


def _get_ragas_local_llm_and_embeddings():  # -> tuple[Any, Any] | tuple[None, None]
    """
    When no OpenAI/Azure keys are set, return (llm, embeddings) for RAGAS using local
    HuggingFace models (small models, same idea as custom evaluator). Requires
    langchain_community and transformers. Optional env: RAGAS_LOCAL_LLM, RAGAS_LOCAL_EMBEDDING.
    """
    llm_model = os.environ.get(RAGAS_LOCAL_LLM, "google/flan-t5-small")
    emb_model = os.environ.get(RAGAS_LOCAL_EMBEDDING, "sentence-transformers/paraphrase-MiniLM-L3-v2")
    try:
        from langchain_community.llms import HuggingFacePipeline
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from transformers import pipeline as hf_pipeline
    except ImportError as e:
        logger.warning(
            "RAGAS local models requested but langchain_community or transformers not available: %s. "
            "Install with: pip install langchain_community transformers. Skipping RAGAS.",
            e,
        )
        return None, None
    try:
        device = 0 if _device_has_cuda() else -1
        pipe = hf_pipeline(
            "text2text-generation",
            model=llm_model,
            max_new_tokens=256,
            device=device,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        embeddings = HuggingFaceEmbeddings(model_name=emb_model)
        logger.info(
            "Using local HuggingFace models for RAGAS (llm=%s, embedding=%s).",
            llm_model,
            emb_model,
        )
        return llm, embeddings
    except Exception as e:
        logger.warning("Failed to create local RAGAS LLM/embeddings: %s. Skipping RAGAS.", e)
        return None, None


def _device_has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _get_ragas_azure_llm_and_embeddings():  # -> tuple[Any, Any] | tuple[None, None]
    """
    If Azure OpenAI env vars are set, return (llm, embeddings) for RAGAS (LangChain models).
    RAGAS will wrap them internally. Otherwise return (None, None).
    """
    if not os.environ.get(AZURE_OPENAI_API_KEY) or not os.environ.get(AZURE_OPENAI_ENDPOINT):
        return None, None
    try:
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    except ImportError as e:
        logger.warning(
            "Azure OpenAI env vars are set but langchain-openai is not installed. "
            "Install with: pip install langchain-openai. RAGAS will use default (OpenAI) if available. %s",
            e,
        )
        return None, None
    endpoint = os.environ[AZURE_OPENAI_ENDPOINT].rstrip("/")
    api_version = os.environ.get(AZURE_OPENAI_API_VERSION, "2024-02-15-preview")
    chat_deployment = os.environ.get(AZURE_OPENAI_CHAT_DEPLOYMENT, "gpt-4")
    embedding_deployment = os.environ.get(AZURE_OPENAI_EMBEDDING_DEPLOYMENT, "text-embedding-ada-002")
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=os.environ[AZURE_OPENAI_API_KEY],
            api_version=api_version,
            azure_deployment=chat_deployment,
            model=chat_deployment,
            validate_base_url=False,
        )
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            api_key=os.environ[AZURE_OPENAI_API_KEY],
            api_version=api_version,
            azure_deployment=embedding_deployment,
            model=embedding_deployment,
            validate_base_url=False,
        )
        logger.info("Using Azure OpenAI for RAGAS (chat=%s, embedding=%s).", chat_deployment, embedding_deployment)
        return llm, embeddings
    except Exception as e:
        logger.warning("Failed to create Azure OpenAI LLM/embeddings: %s. RAGAS may use default.", e)
        return None, None


def _ragas_result_to_dict(result: Any) -> dict[str, Any]:
    """Convert RAGAS EvaluationResult to a JSON-serializable dict of metrics."""
    # RAGAS EvaluationResult has _repr_dict: {metric_name: mean_score} (see ragas/dataset_schema.py)
    if hasattr(result, "_repr_dict") and result._repr_dict is not None:
        return {k: float(v) if v is not None else None for k, v in result._repr_dict.items()}
    # Fallback: aggregate from result.scores (list of dicts, one per sample)
    if hasattr(result, "scores") and result.scores and isinstance(result.scores, list):
        import math
        out: dict[str, Any] = {}
        for k in result.scores[0].keys():
            vals = [
                d[k] for d in result.scores
                if d.get(k) is not None and not (isinstance(d.get(k), (int, float)) and math.isnan(d.get(k)))
            ]
            if vals:
                out[k] = sum(vals) / len(vals)
            else:
                out[k] = None
        return out
    if hasattr(result, "items"):
        return dict(result)
    return {}


def run_ragas(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Run RAGAS evaluate on samples. Each sample must have query, prediction, contexts (list), answer.
    Builds HF Dataset with question, answer (model), contexts, ground_truth. Returns a dict of
    RAGAS metrics (e.g. faithfulness, answer_relevancy, context_precision) or None if RAGAS
    is unavailable or fails.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
    except ImportError as e:
        logger.warning(
            "RAGAS not installed (pip install ragas). Skipping RAGAS evaluation. ImportError: %s",
            e,
        )
        return None
    ragas_data = []
    for s in samples:
        ragas_data.append({
            "question": s.get("query", ""),
            "answer": s.get("prediction", ""),
            "contexts": s.get("contexts") or [s.get("context", "")],
            "ground_truth": s.get("answer", ""),
        })
    try:
        hf_ds = Dataset.from_list(ragas_data)
        llm, embeddings = _get_ragas_azure_llm_and_embeddings()
        if llm is None or embeddings is None:
            # No Azure: use local (small) models when no OpenAI key, so we can run without any API
            if not os.environ.get("OPENAI_API_KEY"):
                llm, embeddings = _get_ragas_local_llm_and_embeddings()
        if llm is not None and embeddings is not None:
            result = ragas_evaluate(hf_ds, llm=llm, embeddings=embeddings)
        else:
            result = ragas_evaluate(hf_ds)
        metrics_dict = _ragas_result_to_dict(result)
        if metrics_dict:
            return metrics_dict
        logger.warning("RAGAS returned no metrics. Skipping RAGAS output.")
        return None
    except Exception as e:
        logger.warning(
            "RAGAS evaluation failed: %s. Skipping RAGAS metrics. "
            "Common causes: missing OPENAI_API_KEY (or other LLM API key), network errors, or dataset validation. "
            "See https://docs.ragas.io/ for setup.",
            e,
            exc_info=True,
        )
        return None
