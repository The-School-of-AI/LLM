"""
Llama 3 / Llama 3.1 template renderer.

Format:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {assistant}<|eot_id|>

Loss masking semantics for assistant turns:
    - span.start = character position immediately AFTER "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
    - span.end   = character position immediately AFTER "<|eot_id|>"
"""
from __future__ import annotations

from pipeline.templates.base import TemplateBase

_BOS           = "<|begin_of_text|>"
_START_HEADER  = "<|start_header_id|>"
_END_HEADER    = "<|end_header_id|>"
_EOT           = "<|eot_id|>"
_HEADER_SEP    = "\n\n"   # two newlines between header and content per Llama 3 format


class Llama3Template(TemplateBase):

    def apply(
        self,
        conversations: list[dict],
        system_prompt: str | None = None,
    ) -> tuple[str, list[dict]]:
        parts: list[str] = []
        spans: list[dict] = []

        turns = list(conversations)
        if system_prompt and (not turns or turns[0].get("role") != "system"):
            turns = [{"role": "system", "content": system_prompt}] + turns

        cursor = 0
        first = True
        for turn in turns:
            role    = turn.get("role", "")
            content = (turn.get("content") or "").strip()

            bos_tok = _BOS if first else ""
            first = False

            header  = f"{bos_tok}{_START_HEADER}{role}{_END_HEADER}{_HEADER_SEP}"
            body    = content
            footer  = _EOT
            segment = header + body + footer

            seg_start = cursor
            seg_end   = cursor + len(segment)

            if role == "assistant":
                span_start = seg_start + len(header)   # after header
                span_end   = seg_end                   # including EOT
            else:
                span_start = seg_start
                span_end   = seg_end

            spans.append({"role": role, "start": span_start, "end": span_end})
            parts.append(segment)
            cursor = seg_end

        return "".join(parts), spans
