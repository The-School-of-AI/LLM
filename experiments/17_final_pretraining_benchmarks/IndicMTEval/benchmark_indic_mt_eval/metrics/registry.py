"""Metric factory registry."""

from __future__ import annotations

from typing import Callable

MetricFn = Callable[..., float]

_REGISTRY: dict[str, MetricFn] = {}


def register_metric(name: str):
    def decorator(fn: MetricFn) -> MetricFn:
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_metric(name: str) -> MetricFn:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown metric '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_metrics() -> list[str]:
    return list(_REGISTRY.keys())
