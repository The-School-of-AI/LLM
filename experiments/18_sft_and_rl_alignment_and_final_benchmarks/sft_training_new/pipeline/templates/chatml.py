"""
ChatML template renderer.

Format:
    <|im_start|>system
    {system}<|im_end|>
    <|im_start|>user
    {user}<|im_end|>
    <|im_start|>assistant
    {assistant}<|im_end|>

Loss masking semantics for assistant turns:
    - span.start = character position immediately AFTER "<|im_start|>assistant\\n"
      (header is excluded from loss — model does not learn to reproduce the role marker)
    - span.end   = character position immediately AFTER the trailing "<|im_end|>\\n"
      (EOS token IS included — model learns to terminate)

For non-assistant turns (system, user) spans cover the full segment so Stage 5
can mask them completely to -100.
"""
from __future__ import annotations

from pipeline.templates.base import TemplateBase

_IMSTART = "<|im_start|>"
_IMEND   = "<|im_end|>"


class ChatMLTemplate(TemplateBase):

    def apply(
        self,
        conversations: list[dict],
        system_prompt: str | None = None,
    ) -> tuple[str, list[dict]]:
        parts: list[str] = []
        spans: list[dict] = []

        # Prepend system prompt if provided and not already first turn
        turns = list(conversations)
        if system_prompt and (not turns or turns[0].get("role") != "system"):
            turns = [{"role": "system", "content": system_prompt}] + turns

        cursor = 0
        for turn in turns:
            role    = turn.get("role", "")
            content = (turn.get("content") or "").strip()

            header  = f"{_IMSTART}{role}\n"
            body    = content
            footer  = f"{_IMEND}\n"
            segment = header + body + footer

            seg_start = cursor
            seg_end   = cursor + len(segment)

            if role == "assistant":
                # Loss span: after header, including footer (EOS)
                span_start = seg_start + len(header)
                span_end   = seg_end
            else:
                # Non-assistant: full segment (will be masked to -100)
                span_start = seg_start
                span_end   = seg_end

            spans.append({"role": role, "start": span_start, "end": span_end})
            parts.append(segment)
            cursor = seg_end

        return "".join(parts), spans
