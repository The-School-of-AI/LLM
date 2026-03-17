"""
TemplateBase ABC — every chat template renderer implements this interface.

Contract:
    apply(conversations) -> (formatted_text, role_spans)

    role_spans is a list of dicts:
        {"role": str, "start": int, "end": int}

    where start/end are character-level offsets into formatted_text.

    Semantics:
    - For assistant turns: span starts AFTER the role header
      (e.g. after "<|im_start|>assistant\\n") and ends INCLUDING the EOS token.
      This means loss is computed on the assistant's actual reply AND the EOS token,
      teaching the model to terminate correctly.
    - For all other roles: spans cover the full segment including header and suffix,
      but these positions are masked to -100 in Stage 5.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TemplateBase(ABC):

    @abstractmethod
    def apply(
        self,
        conversations: list[dict],
        system_prompt: str | None = None,
    ) -> tuple[str, list[dict]]:
        """
        Render a conversation into a single string and compute role spans.

        Args:
            conversations: List of {"role": str, "content": str} dicts.
            system_prompt:  Optional system prompt to prepend. If None, no system
                            turn is injected (unless conversations already contains one).

        Returns:
            formatted_text: The full rendered conversation string.
            role_spans:     List of {"role": str, "start": int, "end": int}.
                            One entry per turn in the rendered output.
        """
