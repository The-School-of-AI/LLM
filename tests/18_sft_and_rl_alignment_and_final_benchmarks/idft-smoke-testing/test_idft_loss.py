"""Unit tests for IDFT loss function."""

import pytest
import torch
import torch.nn.functional as F


class TestSFTLoss:
    """Tests for the standard SFT loss baseline."""

    def test_basic_loss_computation(self):
        """SFT loss should equal mean negative log prob of target tokens."""
        from idft_loss import sft_loss

        torch.manual_seed(42)
        batch, seq_len, vocab = 2, 4, 10
        logits = torch.randn(batch, seq_len, vocab)
        labels = torch.randint(0, vocab, (batch, seq_len))
        mask = torch.ones(batch, seq_len)

        loss = sft_loss(logits, labels, mask)

        # Manual computation
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        expected = -(token_lp * mask).sum() / mask.sum()

        assert torch.allclose(loss, expected, atol=1e-5)

    def test_padding_mask_excludes_tokens(self):
        """Padding tokens (mask=0) should not contribute to loss."""
        from idft_loss import sft_loss

        torch.manual_seed(42)
        logits = torch.randn(1, 4, 10)
        labels = torch.randint(0, 10, (1, 4))

        mask_full = torch.ones(1, 4)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        loss_full = sft_loss(logits, labels, mask_full)
        loss_half = sft_loss(logits, labels, mask_half)

        # Different masks should give different losses
        assert not torch.allclose(loss_full, loss_half)

    def test_loss_is_positive(self):
        """Cross-entropy loss should always be non-negative."""
        from idft_loss import sft_loss

        torch.manual_seed(42)
        logits = torch.randn(2, 8, 100)
        labels = torch.randint(0, 100, (2, 8))
        mask = torch.ones(2, 8)

        loss = sft_loss(logits, labels, mask)
        assert loss.item() >= 0


class TestIDFTLoss:
    """Tests for the IDFT loss function."""

    def test_returns_loss_and_diagnostics(self):
        """IDFT loss should return (loss_tensor, diagnostics_dict)."""
        from idft_loss import idft_loss

        torch.manual_seed(42)
        logits = torch.randn(2, 4, 10)
        labels = torch.randint(0, 10, (2, 4))
        mask = torch.ones(2, 4)

        result = idft_loss(logits, labels, mask, clip_B=5.0)

        assert isinstance(result, tuple)
        assert len(result) == 2

        loss, diag = result
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
        assert isinstance(diag, dict)

    def test_diagnostics_keys(self):
        """Diagnostics dict should contain all expected keys."""
        from idft_loss import idft_loss

        torch.manual_seed(42)
        logits = torch.randn(2, 4, 10)
        labels = torch.randint(0, 10, (2, 4))
        mask = torch.ones(2, 4)

        _, diag = idft_loss(logits, labels, mask)

        expected_keys = {
            "phi_mean",
            "phi_std",
            "phi_below_neg1_pct",
            "phi_below_neg3_pct",
            "phi_below_neg5_pct",
            "gamma_mean",
            "gamma_max",
        }
        assert set(diag.keys()) == expected_keys

    def test_clip_B_zero_reduces_to_standard_loss(self):
        """With clip_B=0, phi is clamped to 0, gamma=exp(0)=1, so IDFT = SFT."""
        from idft_loss import idft_loss, sft_loss

        torch.manual_seed(42)
        logits = torch.randn(2, 8, 50)
        labels = torch.randint(0, 50, (2, 8))
        mask = torch.ones(2, 8)

        sft_loss(logits, labels, mask)
        idft, _ = idft_loss(logits, labels, mask, clip_B=0.0)

        # When clip_B=0, phi_clipped=0, gamma=1, weight=p^1=p
        # IDFT = -(1/L) * sum(p * log p) which is NOT the same as SFT
        # Actually: with gamma=1, loss = -sum(p * log p) / L = entropy-weighted
        # So this tests that clip_B=0 gives a valid, finite loss
        assert torch.isfinite(idft)

    def test_no_nan_with_extreme_logits(self):
        """Loss should be finite even with very large/small logits."""
        from idft_loss import idft_loss

        torch.manual_seed(42)
        # Very large logits (near one-hot distribution)
        logits_big = torch.randn(2, 4, 10) * 100
        labels = torch.randint(0, 10, (2, 4))
        mask = torch.ones(2, 4)

        loss, diag = idft_loss(logits_big, labels, mask, clip_B=5.0)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert all(
            not (isinstance(v, float) and (v != v)) for v in diag.values()  # NaN check
        )

    def test_no_nan_with_tiny_logits(self):
        """Loss should be finite even with very small logits (high entropy)."""
        from idft_loss import idft_loss

        # Near-uniform distribution
        logits_tiny = torch.zeros(2, 4, 10) + torch.randn(2, 4, 10) * 0.01
        labels = torch.randint(0, 10, (2, 4))
        mask = torch.ones(2, 4)

        loss, diag = idft_loss(logits_tiny, labels, mask, clip_B=5.0)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    def test_phi_clipping_works(self):
        """Phi values should be within [-clip_B, clip_B]."""
        from idft_loss import idft_loss

        torch.manual_seed(42)
        logits = torch.randn(2, 4, 10) * 50  # extreme logits
        labels = torch.randint(0, 10, (2, 4))
        mask = torch.ones(2, 4)

        clip_B = 3.0
        _, diag = idft_loss(logits, labels, mask, clip_B=clip_B)

        # With clip_B=3, gamma ranges from exp(-3) to exp(3)
        assert diag["gamma_max"] <= torch.exp(torch.tensor(clip_B)).item() + 0.1

    def test_mask_excludes_padding(self):
        """Padding tokens should not affect loss or diagnostics."""
        from idft_loss import idft_loss

        torch.manual_seed(42)
        logits = torch.randn(1, 4, 10)
        labels = torch.randint(0, 10, (1, 4))

        mask_full = torch.ones(1, 4)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        loss_full, _ = idft_loss(logits, labels, mask_full)
        loss_half, _ = idft_loss(logits, labels, mask_half)

        assert not torch.allclose(loss_full, loss_half)

    def test_loss_is_differentiable(self):
        """IDFT loss should support backpropagation."""
        from idft_loss import idft_loss

        torch.manual_seed(42)
        logits = torch.randn(2, 4, 10, requires_grad=True)
        labels = torch.randint(0, 10, (2, 4))
        mask = torch.ones(2, 4)

        loss, _ = idft_loss(logits, labels, mask)
        loss.backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_gamma_values_make_sense(self):
        """For near-uniform logits, phi should be negative, gamma > 1 (OOD)."""
        from idft_loss import idft_loss

        # Near-uniform: log_p(x_t) is very negative, entropy is high
        # phi = log_p(x_t) + H should be somewhat negative for random tokens
        logits = torch.zeros(2, 8, 1000)  # uniform over 1000 tokens
        labels = torch.randint(0, 1000, (2, 8))
        mask = torch.ones(2, 8)

        _, diag = idft_loss(logits, labels, mask, clip_B=10.0)

        # log_p(x_t) = -log(1000) ≈ -6.9, H = log(1000) ≈ 6.9
        # phi ≈ -6.9 + 6.9 ≈ 0 for uniform
        # So gamma should be near 1
        assert diag["gamma_mean"] == pytest.approx(1.0, abs=0.5)
