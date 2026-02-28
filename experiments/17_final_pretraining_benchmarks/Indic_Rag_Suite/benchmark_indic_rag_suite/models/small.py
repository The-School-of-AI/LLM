"""
Small models for verification: sentence-transformers (retrieval) and FLAN-T5-small (generation).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmark_indic_rag_suite.models.base import GenerationModelBase, RetrievalModelBase
from benchmark_indic_rag_suite.models.registry import register_generation_model, register_retrieval_model


@register_retrieval_model("small")
def _create_small_retrieval(
    device: str = "cpu",
    model_name_or_path: str | None = None,
    **kwargs: Any,
) -> RetrievalModelBase:
    return SmallRetrievalModel(device=device, model_name_or_path=model_name_or_path)


class SmallRetrievalModel(RetrievalModelBase):
    """Small sentence-transformers model for retrieval verification."""

    def __init__(self, device: str = "cpu", model_name_or_path: str | None = None):
        from sentence_transformers import SentenceTransformer

        name = model_name_or_path or "sentence-transformers/paraphrase-MiniLM-L3-v2"
        self._model = SentenceTransformer(name)
        self._device = device
        self._model = self._model.to(device)

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs: Any) -> np.ndarray:
        return self._model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=kwargs.get("show_progress_bar", True),
            convert_to_numpy=True,
        )

    def encode_passages(self, passages: list[str], batch_size: int = 16, **kwargs: Any) -> np.ndarray:
        return self._model.encode(
            passages,
            batch_size=batch_size,
            show_progress_bar=kwargs.get("show_progress_bar", True),
            convert_to_numpy=True,
        )

    def to(self, device: str) -> RetrievalModelBase:
        self._device = device
        self._model = self._model.to(device)
        return self


@register_generation_model("small")
def _create_small_generation(
    device: str = "cpu",
    model_name_or_path: str | None = None,
    **kwargs: Any,
) -> GenerationModelBase:
    return SmallGenerationModel(device=device, model_name_or_path=model_name_or_path)


class SmallGenerationModel(GenerationModelBase):
    """Small FLAN-T5 for generation verification."""

    def __init__(self, device: str = "cpu", model_name_or_path: str | None = None):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        name = model_name_or_path or "google/flan-t5-small"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(name)
        self._device = device
        self._model = self._model.to(device)

    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 64,
        **kwargs: Any,
    ) -> str:
        prompt = f"Question: {query}\nContext: {context[:500]}\nAnswer:"
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)
        out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def to(self, device: str) -> GenerationModelBase:
        self._device = device
        self._model = self._model.to(device)
        return self
