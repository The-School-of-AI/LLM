"""
Tests for training batch assembly logic.

Covers:
  - Proxy + selected concatenation when include_proxy_in_training=True
  - Selected-only batch when include_proxy_in_training=False
  - Bypass mode (candidate_multiplier=1) produces empty proxy tensor
  - Sequence length alignment between proxy and selected
  - Training batch size = n_proxy + n_selected when proxy included
  - Training batch size = n_candidates when bypassed
  - Token count computation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

EXP_ROOT = str(Path(__file__).resolve().parent.parent)
if EXP_ROOT not in sys.path:
    sys.path.insert(0, EXP_ROOT)


class TestBatchAssemblyWithProxy:
    """include_proxy_in_training=True — proxy samples join the training batch."""

    def test_cat_proxy_and_selected(self):
        """training_ids = cat([proxy, selected[:, :target_len]])."""
        proxy_ids_train = torch.randint(0, 100, (2, 32))  # 2 proxy, seq_len=32
        candidate_ids = torch.randint(0, 100, (8, 128))  # 8 candidates, seq_len=128
        local_indices = torch.tensor([1, 3, 5])  # 3 selected

        selected_ids = candidate_ids[local_indices]
        target_len = proxy_ids_train.size(1)
        training_ids = torch.cat(
            [proxy_ids_train, selected_ids[:, :target_len]], dim=0
        )

        assert training_ids.shape == (5, 32)  # 2 proxy + 3 selected
        # First 2 rows are proxy
        assert torch.equal(training_ids[:2], proxy_ids_train)
        # Last 3 rows are truncated selected
        assert torch.equal(training_ids[2:], selected_ids[:, :32])

    def test_training_batch_size(self):
        """Training batch = proxy_count + selected_count."""
        n_proxy = 2
        n_selected = 2
        proxy_ids_train = torch.randint(0, 100, (n_proxy, 64))
        selected_ids = torch.randint(0, 100, (n_selected, 64))

        training_ids = torch.cat([proxy_ids_train, selected_ids], dim=0)
        assert training_ids.size(0) == n_proxy + n_selected

    def test_token_count(self):
        """Total train tokens = batch_size * seq_len."""
        proxy_ids_train = torch.randint(0, 100, (1, 128))
        selected_ids = torch.randint(0, 100, (1, 128))
        training_ids = torch.cat([proxy_ids_train, selected_ids], dim=0)
        assert training_ids.numel() == 2 * 128

    def test_seq_len_mismatch_resolved(self):
        """When proxy seq_len < candidate seq_len, selected is truncated."""
        proxy_ids_train = torch.randint(0, 100, (1, 32))
        candidate_ids = torch.randint(0, 100, (4, 128))
        local_indices = torch.tensor([0, 2])

        selected_ids = candidate_ids[local_indices]
        target_len = proxy_ids_train.size(1)
        training_ids = torch.cat(
            [proxy_ids_train, selected_ids[:, :target_len]], dim=0
        )

        assert training_ids.shape == (3, 32)

    def test_equal_seq_lens(self):
        """When proxy and candidate have same seq_len, no truncation needed."""
        seq_len = 128
        proxy_ids_train = torch.randint(0, 100, (1, seq_len))
        selected_ids = torch.randint(0, 100, (3, seq_len))

        training_ids = torch.cat([proxy_ids_train, selected_ids], dim=0)
        assert training_ids.shape == (4, seq_len)


class TestBatchAssemblyWithoutProxy:
    """include_proxy_in_training=False — only OPUS-selected candidates train."""

    def test_selected_only(self):
        """When proxy excluded, training_ids = selected_ids (no proxy)."""
        candidate_ids = torch.randint(0, 100, (8, 128))
        local_indices = torch.tensor([0, 2, 4, 6])
        selected_ids = candidate_ids[local_indices]

        # Simulates the else branch in the training loop
        proxy_ids_train = torch.empty(0, 128, dtype=candidate_ids.dtype)
        include_proxy = False

        if include_proxy and proxy_ids_train.size(0) > 0:
            target_len = proxy_ids_train.size(1)
            training_ids = torch.cat(
                [proxy_ids_train, selected_ids[:, :target_len]], dim=0
            )
        else:
            training_ids = selected_ids

        assert torch.equal(training_ids, selected_ids)
        assert training_ids.size(0) == 4

    def test_empty_proxy_triggers_selected_only(self):
        """Even with include_proxy=True, empty proxy → selected only."""
        selected_ids = torch.randint(0, 100, (4, 128))
        proxy_ids_train = torch.empty(0, 128, dtype=selected_ids.dtype)
        include_proxy = True

        if include_proxy and proxy_ids_train.size(0) > 0:
            target_len = proxy_ids_train.size(1)
            training_ids = torch.cat(
                [proxy_ids_train, selected_ids[:, :target_len]], dim=0
            )
        else:
            training_ids = selected_ids

        assert torch.equal(training_ids, selected_ids)


class TestBypassMode:
    """candidate_multiplier=1 — OPUS disabled, all candidates used directly."""

    def test_bypass_uses_all_candidates(self):
        """In bypass mode, local_indices = arange(n_candidates)."""
        n_candidates = 4
        candidate_ids = torch.randint(0, 100, (n_candidates, 128))
        local_indices = torch.arange(n_candidates)

        selected_ids = candidate_ids[local_indices]
        assert torch.equal(selected_ids, candidate_ids)

    def test_bypass_proxy_is_empty(self):
        """In bypass mode, proxy_ids_train is empty tensor."""
        candidate_ids = torch.randint(0, 100, (4, 128))
        proxy_ids_train = torch.empty(
            0, candidate_ids.size(1),
            dtype=candidate_ids.dtype,
        )

        assert proxy_ids_train.size(0) == 0
        assert proxy_ids_train.size(1) == candidate_ids.size(1)

    def test_bypass_training_equals_candidates(self):
        """Bypass mode: training_ids == candidate_ids."""
        n_candidates = 4
        candidate_ids = torch.randint(0, 100, (n_candidates, 128))
        local_indices = torch.arange(n_candidates)

        selected_ids = candidate_ids[local_indices]
        proxy_ids_train = torch.empty(0, 128, dtype=candidate_ids.dtype)
        include_proxy = True

        if include_proxy and proxy_ids_train.size(0) > 0:
            target_len = proxy_ids_train.size(1)
            training_ids = torch.cat(
                [proxy_ids_train, selected_ids[:, :target_len]], dim=0
            )
        else:
            training_ids = selected_ids

        assert torch.equal(training_ids, candidate_ids)
        assert training_ids.numel() == n_candidates * 128


class TestBatchInvariants:
    """Cross-cutting invariants that must hold regardless of mode."""

    @pytest.mark.parametrize("n_proxy,n_selected", [(0, 4), (1, 3), (2, 2), (4, 0)])
    def test_total_batch_size(self, n_proxy, n_selected):
        """Total training batch = proxy + selected for all combinations."""
        seq_len = 64
        proxy = torch.randint(0, 100, (n_proxy, seq_len))
        selected = torch.randint(0, 100, (n_selected, seq_len))

        if n_proxy > 0:
            training_ids = torch.cat([proxy, selected], dim=0)
        else:
            training_ids = selected

        assert training_ids.size(0) == n_proxy + n_selected

    def test_training_ids_dtype_preserved(self):
        """training_ids should have same dtype as input candidates."""
        proxy = torch.randint(0, 100, (1, 64), dtype=torch.long)
        selected = torch.randint(0, 100, (3, 64), dtype=torch.long)
        training_ids = torch.cat([proxy, selected], dim=0)
        assert training_ids.dtype == torch.long

    def test_training_ids_2d(self):
        """training_ids is always 2D: (batch, seq_len)."""
        proxy = torch.randint(0, 100, (2, 64))
        selected = torch.randint(0, 100, (2, 64))
        training_ids = torch.cat([proxy, selected], dim=0)
        assert training_ids.dim() == 2

    def test_no_data_leakage_between_proxy_and_selected(self):
        """Proxy and selected portions of training_ids don't overlap."""
        proxy = torch.ones(2, 64, dtype=torch.long) * 999
        selected = torch.zeros(2, 64, dtype=torch.long)
        training_ids = torch.cat([proxy, selected], dim=0)

        # First 2 rows should be all 999s (proxy)
        assert (training_ids[:2] == 999).all()
        # Last 2 rows should be all 0s (selected)
        assert (training_ids[2:] == 0).all()
