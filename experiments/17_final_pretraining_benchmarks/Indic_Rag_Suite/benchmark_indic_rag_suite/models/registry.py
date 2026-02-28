"""Registry for retrieval and generation backends."""

from __future__ import annotations

from typing import Callable, TypeVar

from benchmark_indic_rag_suite.models.base import GenerationModelBase, RetrievalModelBase

T = TypeVar("T", RetrievalModelBase, GenerationModelBase)
_retrieval_factories: dict[str, Callable[..., RetrievalModelBase]] = {}
_generation_factories: dict[str, Callable[..., GenerationModelBase]] = {}


def register_retrieval_model(name: str):
    def decorator(fn: Callable[..., RetrievalModelBase]):
        _retrieval_factories[name] = fn
        return fn
    return decorator


def register_generation_model(name: str):
    def decorator(fn: Callable[..., GenerationModelBase]):
        _generation_factories[name] = fn
        return fn
    return decorator


def get_retrieval_model(
    backend: str,
    device: str = "cpu",
    model_name_or_path: str | None = None,
    **kwargs,
) -> RetrievalModelBase:
    if backend not in _retrieval_factories:
        raise ValueError(f"Unknown retrieval backend: {backend}. Available: {list(_retrieval_factories.keys())}")
    return _retrieval_factories[backend](device=device, model_name_or_path=model_name_or_path, **kwargs).to(device)


def get_generation_model(
    backend: str,
    device: str = "cpu",
    model_name_or_path: str | None = None,
    **kwargs,
) -> GenerationModelBase:
    if backend not in _generation_factories:
        raise ValueError(f"Unknown generation backend: {backend}. Available: {list(_generation_factories.keys())}")
    return _generation_factories[backend](device=device, model_name_or_path=model_name_or_path, **kwargs).to(device)
