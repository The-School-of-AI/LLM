"""
Tests for Stage 1 — Schema Validation.
"""
import pytest
from pipeline.config import Stage1Config
from pipeline.stage1_validate import ConversationValidator


def _make_cfg(**kwargs) -> Stage1Config:
    cfg = Stage1Config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _record(turns: list) -> dict:
    return {"conversations": turns}


class TestConversationValidator:

    def test_valid_single_turn(self):
        v = ConversationValidator(_make_cfg())
        rec = _record([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ])
        ok, reason = v.validate(rec)
        assert ok is True
        assert reason == ""

    def test_missing_conversations_field(self):
        v = ConversationValidator(_make_cfg())
        ok, reason = v.validate({"instruction": "hi"})
        assert ok is False
        assert "missing" in reason.lower() or "invalid" in reason.lower()

    def test_too_few_turns(self):
        v = ConversationValidator(_make_cfg(min_turns=2))
        rec = _record([{"role": "user", "content": "Hi"}])
        ok, reason = v.validate(rec)
        assert ok is False
        assert "too_few_turns" in reason

    def test_too_many_turns(self):
        v = ConversationValidator(_make_cfg(max_turns=3))
        turns = [
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
        ]
        ok, reason = v.validate(_record(turns))
        assert ok is False
        assert "too_many_turns" in reason

    def test_missing_required_role(self):
        v = ConversationValidator(_make_cfg(required_roles=["user", "assistant"]))
        rec = _record([{"role": "user", "content": "Hi"}, {"role": "user", "content": "Hello"}])
        ok, reason = v.validate(rec)
        # Either fails on missing assistant OR fails because last is not assistant
        assert ok is False

    def test_last_turn_not_assistant(self):
        v = ConversationValidator(_make_cfg(require_last_turn_is_assistant=True))
        rec = _record([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ])
        ok, reason = v.validate(rec)
        assert ok is False
        assert "last_turn_not_assistant" in reason

    def test_system_turn_allowed(self):
        v = ConversationValidator(_make_cfg(allow_system_turn=True))
        rec = _record([
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ])
        ok, _ = v.validate(rec)
        assert ok is True

    def test_system_turn_disallowed(self):
        v = ConversationValidator(_make_cfg(allow_system_turn=False))
        rec = _record([
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ])
        ok, reason = v.validate(rec)
        assert ok is False
        assert "system_turn_not_allowed" in reason

    def test_empty_content_rejected(self):
        v = ConversationValidator(_make_cfg(min_content_chars=1))
        rec = _record([
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Hello"},
        ])
        ok, reason = v.validate(rec)
        assert ok is False
        assert "content_too_short" in reason

    def test_unknown_role_rejected(self):
        v = ConversationValidator(_make_cfg())
        rec = _record([
            {"role": "human", "content": "Hi"},   # not user/assistant/system
            {"role": "assistant", "content": "Hi"},
        ])
        ok, reason = v.validate(rec)
        assert ok is False
        assert "unknown_role" in reason

    def test_multi_turn_allowed(self):
        v = ConversationValidator(_make_cfg(allow_multi_turn=True))
        rec = _record([
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
        ])
        ok, _ = v.validate(rec)
        assert ok is True

    def test_multi_turn_disallowed(self):
        v = ConversationValidator(_make_cfg(allow_multi_turn=False))
        rec = _record([
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
        ])
        ok, reason = v.validate(rec)
        assert ok is False
        assert "multi_turn_not_allowed" in reason
