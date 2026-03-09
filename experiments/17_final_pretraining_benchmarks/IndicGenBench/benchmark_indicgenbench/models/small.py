"""Small/dummy model for smoke testing without GPU or large downloads."""

from __future__ import annotations

import random
from typing import Any

from benchmark_indicgenbench.models.base import GenerationModelBase
from benchmark_indicgenbench.models.registry import register_model


@register_model("small")
def _create_small_model(**kwargs: Any) -> GenerationModelBase:
    return SmallTestModel()


class SmallTestModel(GenerationModelBase):
    """Returns a deterministic dummy response for pipeline testing."""

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        # Return a short dummy response to exercise the full pipeline
        random.seed(hash(prompt) % 2**32)
        words = ["the", "a", "is", "was", "in", "of", "to", "and", "for", "that"]
        length = random.randint(3, 10)
        return " ".join(random.choices(words, k=length))
