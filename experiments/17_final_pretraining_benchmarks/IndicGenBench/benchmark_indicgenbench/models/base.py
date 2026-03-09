"""Abstract interface for generation models."""

from __future__ import annotations

from abc import ABC, abstractmethod


class GenerationModelBase(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        ...
