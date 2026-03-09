"""Model factory with decorator-based registration."""

from __future__ import annotations

from typing import Any, Callable

from benchmark_indicgenbench.models.base import GenerationModelBase

_REGISTRY: dict[str, Callable[..., GenerationModelBase]] = {}


def register_model(name: str) -> Callable:
    def decorator(fn: Callable[..., GenerationModelBase]) -> Callable[..., GenerationModelBase]:
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_model(backend: str, **kwargs: Any) -> GenerationModelBase:
    if backend not in _REGISTRY:
        # Trigger auto-registration by importing backend modules
        if backend == "hf":
            import benchmark_indicgenbench.models.hf_backend  # noqa: F401
        elif backend == "small":
            import benchmark_indicgenbench.models.small  # noqa: F401
        else:
            raise ValueError(f"Unknown backend: {backend}. Available: {list(_REGISTRY.keys())}")

    if backend not in _REGISTRY:
        raise ValueError(f"Backend '{backend}' not found after import. Available: {list(_REGISTRY.keys())}")

    return _REGISTRY[backend](**kwargs)
