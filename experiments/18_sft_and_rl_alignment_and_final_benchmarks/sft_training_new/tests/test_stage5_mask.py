"""
Tests for Stage 5 — Loss Masking.
Focus: correctness of the labels array produced by apply_loss_mask.
"""
import pytest
from pipeline.stage5_mask import apply_loss_mask

IGNORE = -100


class TestApplyLossMask:

    def test_assistant_tokens_get_input_id_value(self):
        input_ids = [10, 20, 30, 40, 50]
        token_role_spans = [
            {"role": "user",      "token_start": 0, "token_end": 2},
            {"role": "assistant", "token_start": 2, "token_end": 5},
        ]
        labels = apply_loss_mask(input_ids, token_role_spans, {"assistant"}, IGNORE)
        assert labels[0] == IGNORE   # user
        assert labels[1] == IGNORE   # user
        assert labels[2] == 30       # assistant
        assert labels[3] == 40       # assistant
        assert labels[4] == 50       # assistant

    def test_user_tokens_are_masked(self):
        input_ids = [10, 20, 30, 40]
        token_role_spans = [
            {"role": "user",      "token_start": 0, "token_end": 2},
            {"role": "assistant", "token_start": 2, "token_end": 4},
        ]
        labels = apply_loss_mask(input_ids, token_role_spans, {"assistant"}, IGNORE)
        assert all(l == IGNORE for l in labels[:2])

    def test_system_tokens_are_masked(self):
        input_ids = [1, 2, 3, 4, 5, 6]
        token_role_spans = [
            {"role": "system",    "token_start": 0, "token_end": 2},
            {"role": "user",      "token_start": 2, "token_end": 4},
            {"role": "assistant", "token_start": 4, "token_end": 6},
        ]
        labels = apply_loss_mask(input_ids, token_role_spans, {"assistant"}, IGNORE)
        assert labels[0] == IGNORE
        assert labels[1] == IGNORE
        assert labels[2] == IGNORE
        assert labels[3] == IGNORE
        assert labels[4] == 5
        assert labels[5] == 6

    def test_all_masked_when_no_assistant_span(self):
        input_ids = [10, 20, 30]
        token_role_spans = [{"role": "user", "token_start": 0, "token_end": 3}]
        labels = apply_loss_mask(input_ids, token_role_spans, {"assistant"}, IGNORE)
        assert all(l == IGNORE for l in labels)

    def test_empty_input_ids(self):
        labels = apply_loss_mask([], [], {"assistant"}, IGNORE)
        assert labels == []

    def test_labels_same_length_as_input_ids(self):
        input_ids = list(range(20))
        token_role_spans = [
            {"role": "user",      "token_start": 0,  "token_end": 10},
            {"role": "assistant", "token_start": 10, "token_end": 20},
        ]
        labels = apply_loss_mask(input_ids, token_role_spans, {"assistant"}, IGNORE)
        assert len(labels) == len(input_ids)

    def test_multi_turn_only_assistant_spans_unmasked(self):
        """Multi-turn: user1 / assistant1 / user2 / assistant2 — all user masked."""
        input_ids = list(range(20))
        token_role_spans = [
            {"role": "user",      "token_start": 0,  "token_end": 5},
            {"role": "assistant", "token_start": 5,  "token_end": 10},
            {"role": "user",      "token_start": 10, "token_end": 15},
            {"role": "assistant", "token_start": 15, "token_end": 20},
        ]
        labels = apply_loss_mask(input_ids, token_role_spans, {"assistant"}, IGNORE)
        # user1
        assert all(l == IGNORE for l in labels[0:5])
        # assistant1
        assert all(l != IGNORE for l in labels[5:10])
        # user2
        assert all(l == IGNORE for l in labels[10:15])
        # assistant2
        assert all(l != IGNORE for l in labels[15:20])

    def test_span_clamped_to_sequence_length(self):
        """Spans with token_end beyond seq length should not raise."""
        input_ids = [1, 2, 3]
        token_role_spans = [
            {"role": "assistant", "token_start": 0, "token_end": 100},  # beyond seq
        ]
        labels = apply_loss_mask(input_ids, token_role_spans, {"assistant"}, IGNORE)
        assert labels == [1, 2, 3]

    def test_custom_train_on_roles(self):
        """Configuring train_on_roles={"user"} should mask assistant instead."""
        input_ids = [10, 20, 30, 40]
        token_role_spans = [
            {"role": "user",      "token_start": 0, "token_end": 2},
            {"role": "assistant", "token_start": 2, "token_end": 4},
        ]
        labels = apply_loss_mask(input_ids, token_role_spans, {"user"}, IGNORE)
        assert labels[0] == 10
        assert labels[1] == 20
        assert labels[2] == IGNORE
        assert labels[3] == IGNORE
