"""
Tokenizer-native template renderer.
Delegates to tokenizer.apply_chat_template() for formatting,
then reconstructs role_spans by rendering each prefix incrementally
and measuring character offsets.

This is the correct approach when the tokenizer already has a built-in
chat template (e.g. Llama 3.1, Mistral, Qwen2) — it avoids drift between
training and inference formatting.
"""
from __future__ import annotations

import logging

from pipeline.templates.base import TemplateBase

logger = logging.getLogger(__name__)


class TokenizerNativeTemplate(TemplateBase):
    """
    Wraps tokenizer.apply_chat_template (HuggingFace).

    Span reconstruction algorithm:
        For each turn i, render the conversation up to turn i and up to turn i+1.
        The span for turn i is [len(prefix_i), len(prefix_{i+1})].
        For assistant turns, we additionally try to strip the role header from
        the start of the span by comparing prefix_i with and without add_generation_prompt.
    """

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def apply(
        self,
        conversations: list[dict],
        system_prompt: str | None = None,
    ) -> tuple[str, list[dict]]:
        turns = list(conversations)
        if system_prompt and (not turns or turns[0].get("role") != "system"):
            turns = [{"role": "system", "content": system_prompt}] + turns

        messages = [{"role": t.get("role", ""), "content": t.get("content") or ""} for t in turns]

        # Full formatted text
        formatted_text: str = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Reconstruct spans by rendering prefix up to each turn
        spans: list[dict] = []
        for i, turn in enumerate(messages):
            prefix_before = self._tokenizer.apply_chat_template(
                messages[:i],
                tokenize=False,
                add_generation_prompt=False,
            ) if i > 0 else ""

            prefix_after = self._tokenizer.apply_chat_template(
                messages[:i + 1],
                tokenize=False,
                add_generation_prompt=False,
            )

            seg_start = len(prefix_before)
            seg_end   = len(prefix_after)

            if turn["role"] == "assistant":
                # Try to identify where the header ends and content begins.
                # Strategy: render the header alone as a generation prompt from the prior context,
                # then compare lengths.
                try:
                    header_end_text = self._tokenizer.apply_chat_template(
                        messages[:i],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    content_start = len(header_end_text)
                    if seg_start <= content_start <= seg_end:
                        span_start = content_start
                    else:
                        span_start = seg_start
                except Exception:
                    span_start = seg_start
            else:
                span_start = seg_start

            spans.append({"role": turn["role"], "start": span_start, "end": seg_end})

        return formatted_text, spans
