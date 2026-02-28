"""
HuggingFace backends for evaluating any foundation model (or encoder) by path.
Use --retrieval-backend hf --retrieval-model <path> or --generation-backend hf --generation-model <path>.
Supports Gemma-1B via: --generation-backend hf --generation-model google/gemma-2-1b
"""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmark_indic_rag_suite.models.base import GenerationModelBase, RetrievalModelBase
from benchmark_indic_rag_suite.models.registry import register_generation_model, register_retrieval_model


@register_retrieval_model("hf")
def _create_hf_retrieval(
    device: str = "cpu",
    model_name_or_path: str | None = None,
    **kwargs: Any,
) -> RetrievalModelBase:
    if not model_name_or_path:
        raise ValueError("retrieval_backend=hf requires retrieval_model_name_or_path (e.g. --retrieval-model <path>)")
    return HfRetrievalModel(device=device, model_name_or_path=model_name_or_path)


class HfRetrievalModel(RetrievalModelBase):
    """Any SentenceTransformer-compatible or HF encoder from model_name_or_path."""

    def __init__(self, device: str = "cpu", model_name_or_path: str | None = None):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name_or_path or "")
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


@register_generation_model("hf")
def _create_hf_generation(
    device: str = "cpu",
    model_name_or_path: str | None = None,
    **kwargs: Any,
) -> GenerationModelBase:
    if not model_name_or_path:
        raise ValueError("generation_backend=hf requires generation_model_name_or_path (e.g. --generation-model <path>)")
    return HfGenerationModel(device=device, model_name_or_path=model_name_or_path)


class HfGenerationModel(GenerationModelBase):
    """Any HuggingFace causal LM or seq2seq from model_name_or_path for RAG answer generation (e.g. Gemma-1B)."""

    def __init__(self, device: str = "cpu", model_name_or_path: str | None = None):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        name = model_name_or_path or ""
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        config = AutoConfig.from_pretrained(name)
        if config.is_encoder_decoder:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype="auto")
            self._is_seq2seq = True
        else:
            self._model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto")
            self._is_seq2seq = False
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
        pad_token_id = self._tokenizer.eos_token_id or self._tokenizer.pad_token_id
        if self._is_seq2seq:
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        full = self._tokenizer.decode(out[0], skip_special_tokens=True)
        if "Answer:" in full:
            answer = full.split("Answer:")[-1].strip()
        else:
            answer = full.strip()
        if "\n" in answer:
            answer = answer.split("\n")[0].strip()
        return answer[:500].strip()

    def to(self, device: str) -> GenerationModelBase:
        self._device = device
        self._model = self._model.to(device)
        return self
