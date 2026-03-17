"""
Tests for Stage 3 — Chat Template Application.
Focus: role_spans character offsets are correct and semantically meaningful.
"""
import pytest
from pipeline.templates.chatml import ChatMLTemplate
from pipeline.templates.llama3 import Llama3Template


SIMPLE_CONV = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there"},
]

MULTI_TURN_CONV = [
    {"role": "user", "content": "Question one"},
    {"role": "assistant", "content": "Answer one"},
    {"role": "user", "content": "Question two"},
    {"role": "assistant", "content": "Answer two"},
]


class TestChatMLTemplate:

    def test_returns_text_and_spans(self):
        t = ChatMLTemplate()
        text, spans = t.apply(SIMPLE_CONV)
        assert isinstance(text, str)
        assert isinstance(spans, list)
        assert len(spans) == 2

    def test_spans_cover_correct_roles(self):
        t = ChatMLTemplate()
        text, spans = t.apply(SIMPLE_CONV)
        assert spans[0]["role"] == "user"
        assert spans[1]["role"] == "assistant"

    def test_spans_are_valid_offsets(self):
        t = ChatMLTemplate()
        text, spans = t.apply(SIMPLE_CONV)
        for span in spans:
            assert 0 <= span["start"] < span["end"] <= len(text)

    def test_assistant_content_in_span(self):
        t = ChatMLTemplate()
        text, spans = t.apply(SIMPLE_CONV)
        assistant_span = next(s for s in spans if s["role"] == "assistant")
        segment = text[assistant_span["start"]:assistant_span["end"]]
        assert "Hi there" in segment

    def test_assistant_span_excludes_header(self):
        """Assistant span must NOT start with the role header token."""
        t = ChatMLTemplate()
        text, spans = t.apply(SIMPLE_CONV)
        assistant_span = next(s for s in spans if s["role"] == "assistant")
        segment = text[assistant_span["start"]:assistant_span["end"]]
        assert not segment.startswith("<|im_start|>")

    def test_assistant_span_includes_eos(self):
        """Assistant span must include the EOS token (<|im_end|>)."""
        t = ChatMLTemplate()
        text, spans = t.apply(SIMPLE_CONV)
        assistant_span = next(s for s in spans if s["role"] == "assistant")
        segment = text[assistant_span["start"]:assistant_span["end"]]
        assert "<|im_end|>" in segment

    def test_header_immediately_before_assistant_span(self):
        """The character immediately before the assistant span start is the end of the header."""
        t = ChatMLTemplate()
        text, spans = t.apply(SIMPLE_CONV)
        assistant_span = next(s for s in spans if s["role"] == "assistant")
        header = "<|im_start|>assistant\n"
        start = assistant_span["start"]
        assert text[start - len(header):start] == header

    def test_system_prompt_injected(self):
        t = ChatMLTemplate()
        text, spans = t.apply(SIMPLE_CONV, system_prompt="Be helpful.")
        assert len(spans) == 3  # system, user, assistant
        assert spans[0]["role"] == "system"
        assert "Be helpful." in text

    def test_system_prompt_not_duplicated(self):
        """If conversation already starts with system, don't add another one."""
        conv_with_sys = [{"role": "system", "content": "Existing system"}, *SIMPLE_CONV]
        t = ChatMLTemplate()
        _, spans = t.apply(conv_with_sys, system_prompt="Another system")
        system_spans = [s for s in spans if s["role"] == "system"]
        assert len(system_spans) == 1

    def test_multi_turn_all_spans_present(self):
        t = ChatMLTemplate()
        text, spans = t.apply(MULTI_TURN_CONV)
        assert len(spans) == 4
        roles = [s["role"] for s in spans]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_multi_turn_all_assistant_spans_include_eos(self):
        t = ChatMLTemplate()
        text, spans = t.apply(MULTI_TURN_CONV)
        for span in spans:
            if span["role"] == "assistant":
                segment = text[span["start"]:span["end"]]
                assert "<|im_end|>" in segment

    def test_span_slicing_recovers_content(self):
        """text[start:end] for each span must contain that turn's content."""
        t = ChatMLTemplate()
        text, spans = t.apply(SIMPLE_CONV)
        for turn, span in zip(SIMPLE_CONV, spans):
            segment = text[span["start"]:span["end"]]
            assert turn["content"] in segment

    def test_spans_are_non_overlapping(self):
        t = ChatMLTemplate()
        text, spans = t.apply(MULTI_TURN_CONV)
        for i in range(len(spans) - 1):
            assert spans[i]["end"] <= spans[i + 1]["start"]


class TestLlama3Template:

    def test_returns_text_and_spans(self):
        t = Llama3Template()
        text, spans = t.apply(SIMPLE_CONV)
        assert isinstance(text, str)
        assert len(spans) == 2

    def test_assistant_content_in_span(self):
        t = Llama3Template()
        text, spans = t.apply(SIMPLE_CONV)
        assistant_span = next(s for s in spans if s["role"] == "assistant")
        segment = text[assistant_span["start"]:assistant_span["end"]]
        assert "Hi there" in segment

    def test_assistant_span_excludes_header(self):
        t = Llama3Template()
        text, spans = t.apply(SIMPLE_CONV)
        assistant_span = next(s for s in spans if s["role"] == "assistant")
        segment = text[assistant_span["start"]:assistant_span["end"]]
        assert "<|start_header_id|>" not in segment

    def test_assistant_span_includes_eot(self):
        t = Llama3Template()
        text, spans = t.apply(SIMPLE_CONV)
        assistant_span = next(s for s in spans if s["role"] == "assistant")
        segment = text[assistant_span["start"]:assistant_span["end"]]
        assert "<|eot_id|>" in segment

    def test_bos_at_start(self):
        t = Llama3Template()
        text, spans = t.apply(SIMPLE_CONV)
        assert text.startswith("<|begin_of_text|>")
