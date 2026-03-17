"""
Tests for Stage 4 — Tokenization utilities.
Tests char_spans_to_token_spans conversion logic (no tokenizer required).
"""
import pytest
from pipeline.stage4_tokenize import char_spans_to_token_spans, _clip_token_spans, _make_chunks


class TestCharSpansToTokenSpans:

    def _make_offset_mapping(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return spans

    def test_basic_conversion(self):
        # Text: "Hello World"
        # Tokens: [He|llo| |Wor|ld]
        offset_mapping = [(0, 2), (2, 5), (5, 6), (6, 9), (9, 11)]
        char_spans = [{"role": "user", "start": 0, "end": 6}]
        token_spans = char_spans_to_token_spans(char_spans, offset_mapping)
        assert len(token_spans) == 1
        ts = token_spans[0]
        assert ts["role"] == "user"
        assert ts["token_start"] == 0
        assert ts["token_end"] > 0

    def test_empty_spans(self):
        offset_mapping = [(0, 5), (5, 10)]
        result = char_spans_to_token_spans([], offset_mapping)
        assert result == []

    def test_special_tokens_skipped(self):
        # (0,0) = special token (BOS)
        offset_mapping = [(0, 0), (0, 5), (5, 10)]
        char_spans = [{"role": "assistant", "start": 5, "end": 10}]
        token_spans = char_spans_to_token_spans(char_spans, offset_mapping)
        assert len(token_spans) == 1
        # token_start should skip the (0,0) special token
        assert token_spans[0]["token_start"] == 2

    def test_role_preserved(self):
        offset_mapping = [(0, 3), (3, 6), (6, 9)]
        char_spans = [{"role": "assistant", "start": 3, "end": 9}]
        token_spans = char_spans_to_token_spans(char_spans, offset_mapping)
        assert token_spans[0]["role"] == "assistant"

    def test_multiple_spans(self):
        # 10 tokens, each covering 1 char
        offset_mapping = [(i, i+1) for i in range(10)]
        char_spans = [
            {"role": "user",      "start": 0, "end": 5},
            {"role": "assistant", "start": 5, "end": 10},
        ]
        token_spans = char_spans_to_token_spans(char_spans, offset_mapping)
        assert len(token_spans) == 2
        assert token_spans[0]["token_start"] == 0
        assert token_spans[0]["token_end"] == 5
        assert token_spans[1]["token_start"] == 5
        assert token_spans[1]["token_end"] == 10


class TestClipTokenSpans:

    def test_spans_within_limit_unchanged(self):
        spans = [{"role": "user", "token_start": 0, "token_end": 5}]
        result = _clip_token_spans(spans, max_len=10)
        assert result[0]["token_end"] == 5

    def test_spans_clipped_at_max_len(self):
        spans = [{"role": "assistant", "token_start": 5, "token_end": 15}]
        result = _clip_token_spans(spans, max_len=10)
        assert result[0]["token_end"] == 10

    def test_spans_beyond_max_len_removed(self):
        spans = [{"role": "user", "token_start": 10, "token_end": 15}]
        result = _clip_token_spans(spans, max_len=10)
        assert result == []

    def test_empty_spans(self):
        result = _clip_token_spans([], max_len=512)
        assert result == []


class TestMakeChunks:

    def test_no_split_needed(self):
        ids = list(range(10))
        mask = [1] * 10
        spans = [{"role": "user", "token_start": 0, "token_end": 5},
                 {"role": "assistant", "token_start": 5, "token_end": 10}]
        chunks = _make_chunks(ids, mask, spans, max_len=512, overlap=0)
        assert len(chunks) == 1
        assert chunks[0][0] == ids

    def test_split_produces_multiple_chunks(self):
        ids = list(range(20))
        mask = [1] * 20
        spans = []
        chunks = _make_chunks(ids, mask, spans, max_len=10, overlap=0)
        assert len(chunks) == 2
        assert len(chunks[0][0]) == 10
        assert len(chunks[1][0]) == 10

    def test_overlap_creates_extra_chunk(self):
        ids = list(range(15))
        mask = [1] * 15
        spans = []
        # max_len=10, overlap=5, stride=5 → chunks at 0-10, 5-15
        chunks = _make_chunks(ids, mask, spans, max_len=10, overlap=5)
        assert len(chunks) == 2
        assert chunks[0][0][0] == 0
        assert chunks[1][0][0] == 5
