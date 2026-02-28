"""Abstract interfaces for retrieval and generation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class RetrievalModelBase(ABC):
    @abstractmethod
    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs: Any) -> np.ndarray:
        ...

    @abstractmethod
    def encode_passages(self, passages: list[str], batch_size: int = 16, **kwargs: Any) -> np.ndarray:
        ...

    def to(self, device: str) -> RetrievalModelBase:
        return self


class GenerationModelBase(ABC):
    @abstractmethod
    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 64,
        **kwargs: Any,
    ) -> str:
        ...

    def to(self, device: str) -> GenerationModelBase:
        return self
