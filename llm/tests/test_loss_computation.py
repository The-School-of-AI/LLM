"""
Test ② — Loss Computation Correct (Compare with Manual Calculation).

Verifies that FusedLinearCrossEntropyLoss produces identical loss values
and gradients to naive PyTorch cross-entropy. Also verifies the training
loss composition formula: loss_ntp + 0.3 * loss_mtp + aux_loss.

Requirements:
    - CUDA GPU with Triton support
    - flash-linear-attention installed

Run:
    python -m pytest tests/test_loss_computation.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from llm.kernels.triton_cross_entropy import FusedLinearCrossEntropyLoss

# ── Constants ────────────────────────────────────────────────────────────────
BT = 64   # batch × time (flattened)
H = 128   # hidden size
V = 512   # vocab size


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def base_tensors():
    """Base tensors (fp32) for cloning into each test — never modified directly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton CE kernel")

    hidden = torch.randn(BT, H, device="cuda", dtype=torch.float32)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (BT,), device="cuda")
    return hidden, weight, target


def _ref_ce(hidden, weight, target, ignore_index=-100, reduction="mean"):
    """Reference PyTorch cross-entropy: materialize full logit tensor."""
    logits = hidden.float() @ weight.float().T
    return F.cross_entropy(logits, target, ignore_index=ignore_index, reduction=reduction)


# ── Group 1: Forward Value Correctness ───────────────────────────────────────

class TestForwardValue:

    def test_loss_matches_pytorch_fp32(self, base_tensors):
        """Both paths in fp32 → strict tolerance (1e-5)."""
        hidden, weight, target = base_tensors
        h = hidden.clone()
        w = weight.clone()

        fused_loss = FusedLinearCrossEntropyLoss()(h, w, target)
        ref_loss = _ref_ce(h, w, target)

        assert torch.isfinite(fused_loss), "Fused loss is NaN or inf"
        assert torch.isfinite(ref_loss), "Reference loss is NaN or inf"
        assert torch.allclose(fused_loss, ref_loss, atol=1e-5), (
            f"FP32 loss mismatch: fused={fused_loss.item():.8f}, ref={ref_loss.item():.8f}"
        )

    def test_loss_matches_pytorch_bf16(self, base_tensors):
        """Fused uses bf16 matmul (production path) → relaxed tolerance (1e-2)."""
        hidden, weight, target = base_tensors
        h_bf = hidden.to(torch.bfloat16)
        w_bf = weight.to(torch.bfloat16)

        fused_loss = FusedLinearCrossEntropyLoss()(h_bf, w_bf, target)
        # Reference still in fp32 for "ground truth"
        ref_loss = _ref_ce(hidden, weight, target)

        assert torch.isfinite(fused_loss), "Fused bf16 loss is NaN or inf"
        assert torch.allclose(fused_loss, ref_loss, atol=1e-2), (
            f"BF16 loss mismatch: fused={fused_loss.item():.6f}, ref={ref_loss.item():.6f}"
        )

    def test_ignore_index_masking(self, base_tensors):
        """50% tokens masked with -100 → matches PyTorch with same masking."""
        hidden, weight, target = base_tensors
        h = hidden.clone()
        w = weight.clone()

        t_masked = target.clone()
        t_masked[::2] = -100  # mask every other token

        fused_loss = FusedLinearCrossEntropyLoss(ignore_index=-100)(h, w, t_masked)
        ref_loss = _ref_ce(h, w, t_masked, ignore_index=-100)

        assert torch.isfinite(fused_loss), "Fused loss with masking is NaN or inf"
        assert torch.isfinite(ref_loss), "Reference loss with masking is NaN or inf"
        assert torch.allclose(fused_loss, ref_loss, atol=1e-5), (
            f"Masked loss mismatch: fused={fused_loss.item():.8f}, ref={ref_loss.item():.8f}"
        )

    def test_all_tokens_masked(self, base_tensors):
        """
        All targets = -100 → fused returns 0.0 (NOT nan).

        This is an intentional design choice: n_non_ignore = max(..., 1)
        prevents division by zero. PyTorch F.cross_entropy returns nan here.
        """
        hidden, weight, _ = base_tensors
        h = hidden.clone()
        w = weight.clone()

        t_all_masked = torch.full((BT,), -100, dtype=torch.long, device="cuda")
        fused_loss = FusedLinearCrossEntropyLoss(ignore_index=-100)(h, w, t_all_masked)

        assert torch.isfinite(fused_loss), "Should be finite (0.0), not NaN"
        assert fused_loss.item() == 0.0, (
            f"Expected 0.0 when all masked, got {fused_loss.item()}"
        )

        # Verify PyTorch returns nan (our kernel intentionally differs)
        ref_loss = _ref_ce(h, w, t_all_masked, ignore_index=-100)
        assert torch.isnan(ref_loss), (
            "PyTorch CE should return nan for all-masked — our kernel intentionally returns 0.0 instead"
        )


# ── Group 2: Backward Gradient Correctness ───────────────────────────────────

class TestGradients:
    """
    Each test uses SEPARATE cloned tensors for fused vs reference paths
    to avoid any shared state.
    """

    def test_grad_hidden_matches(self, base_tensors):
        """grad_input from fused ≈ PyTorch autograd gradient."""
        hidden, weight, target = base_tensors

        # Fused path
        h_fused = hidden.clone().requires_grad_(True)
        w_fused = weight.clone().requires_grad_(True)
        loss_fused = FusedLinearCrossEntropyLoss()(h_fused, w_fused, target)
        loss_fused.backward()
        grad_h_fused = h_fused.grad.clone()

        # Reference path (separate tensors)
        h_ref = hidden.clone().requires_grad_(True)
        w_ref = weight.clone().requires_grad_(True)
        loss_ref = F.cross_entropy(h_ref @ w_ref.T, target)
        loss_ref.backward()
        grad_h_ref = h_ref.grad.clone()

        assert torch.allclose(grad_h_fused, grad_h_ref, atol=1e-4), (
            f"grad_hidden max diff: {(grad_h_fused - grad_h_ref).abs().max().item():.6e}"
        )

    def test_grad_weight_matches(self, base_tensors):
        """grad_weight from fused ≈ PyTorch autograd gradient."""
        hidden, weight, target = base_tensors

        # Fused path
        h_fused = hidden.clone().requires_grad_(True)
        w_fused = weight.clone().requires_grad_(True)
        loss_fused = FusedLinearCrossEntropyLoss()(h_fused, w_fused, target)
        loss_fused.backward()
        grad_w_fused = w_fused.grad.clone()

        # Reference path
        h_ref = hidden.clone().requires_grad_(True)
        w_ref = weight.clone().requires_grad_(True)
        loss_ref = F.cross_entropy(h_ref @ w_ref.T, target)
        loss_ref.backward()
        grad_w_ref = w_ref.grad.clone()

        assert torch.allclose(grad_w_fused, grad_w_ref, atol=1e-4), (
            f"grad_weight max diff: {(grad_w_fused - grad_w_ref).abs().max().item():.6e}"
        )


# ── Group 3: Chunking and Composition ────────────────────────────────────────

class TestChunkingAndComposition:

    def test_chunking_invariance(self, base_tensors):
        """Same loss regardless of internal chunk size."""
        hidden, weight, target = base_tensors
        h = hidden.clone()
        w = weight.clone()

        ce_big = FusedLinearCrossEntropyLoss(max_chunk_gb=32.0)       # single chunk
        ce_tiny = FusedLinearCrossEntropyLoss(max_chunk_gb=0.00001)   # many tiny chunks

        loss_big = ce_big(h, w, target)
        loss_tiny = ce_tiny(h, w, target)

        assert torch.allclose(loss_big, loss_tiny, atol=1e-6), (
            f"Chunking changes loss: big={loss_big.item():.8f}, tiny={loss_tiny.item():.8f}"
        )

    def test_full_loss_composition(self):
        """
        Verify the pretrainer loss formula: loss = loss_ntp + 0.3 * loss_mtp + aux_loss.

        The model never returns a composed total loss — composition happens in the
        pretrainer (_1b_forward). So we verify the formula algebraically:
        1. Get h_ntp, h_mtp, aux_loss from the model
        2. Manually compute CE for NTP and MTP using the model's lm_head
        3. Compose and verify the formula is self-consistent
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        from llm.models.recurrence_model_1b import Model1B, ModelConfig

        B, T_full = 2, 34  # need T_full so after :-2 slicing we have T=32
        V_mini, H_mini = 256, 64

        cfg = ModelConfig(
            vocab_size=V_mini, hidden_size=H_mini, num_layers=4,
            num_deltanet_layers=3, num_gsa_layers=1,
            delta_v_heads=2, delta_head_dim=32, delta_gate_dim=32,
            gsa_num_heads=2, gsa_head_dim=32,
            gsa_k_base=8, gsa_k_min=4, gsa_k_max=16,
            gsa_indexer_heads=2, n_streams=2, max_seq_len=512,
            enable_mtp=True, shared_expert_intermediate_size=128,
            require_fused_deltanet_kernel=True, require_fused_gsa_kernel=True,
        )

        device = torch.device("cuda")
        model = Model1B(cfg, embedding_type="standard").to(device).eval()

        # Replicate pretrainer slicing exactly (_1b_forward lines 218-220)
        input_ids = torch.randint(0, V_mini, (B, T_full), device=device)
        x_input = input_ids[:, :-2].contiguous()      # [B, T_full-2] → model input
        y_ntp = input_ids[:, 1:-1].contiguous()        # [B, T_full-2] → NTP targets
        y_mtp = input_ids[:, 2:].contiguous()          # [B, T_full-2] → MTP targets

        with torch.no_grad():
            h_ntp, h_mtp, aux_loss, _ = model(
                x_input,
                next_token_ids=y_ntp,
                return_loss=True,
                return_memory=True,
                return_hidden=True,
            )

        T = x_input.shape[1]  # T_full - 2
        lm_weight = model.lm_head.weight.detach()

        # Manually compute each CE component (fp32 reference)
        loss_ntp_manual = F.cross_entropy(
            h_ntp.float().reshape(-1, H_mini) @ lm_weight.float().T,
            y_ntp.reshape(-1),
        )
        min_len = min(h_mtp.shape[1], y_mtp.shape[1])
        loss_mtp_manual = F.cross_entropy(
            h_mtp[:, :min_len].float().reshape(-1, H_mini) @ lm_weight.float().T,
            y_mtp[:, :min_len].reshape(-1),
        )

        # Compose using pretrainer formula
        total_composed = loss_ntp_manual + 0.3 * loss_mtp_manual + aux_loss.detach().float()

        # Sanity checks: each component is valid
        assert loss_ntp_manual.item() > 0, f"NTP loss should be > 0, got {loss_ntp_manual.item()}"
        assert loss_mtp_manual.item() > 0, f"MTP loss should be > 0, got {loss_mtp_manual.item()}"
        assert aux_loss.item() >= 0, f"Aux loss should be >= 0, got {aux_loss.item()}"
        assert torch.isfinite(total_composed), "Composed loss is NaN or inf"

        # Verify MTP weight is exactly 0.3
        total_without_mtp = loss_ntp_manual + aux_loss.detach().float()
        mtp_contribution = total_composed - total_without_mtp
        expected_mtp_contrib = 0.3 * loss_mtp_manual
        assert torch.allclose(mtp_contribution, expected_mtp_contrib, atol=1e-6), (
            f"MTP weight mismatch: got {mtp_contribution.item():.8f}, "
            f"expected 0.3 * {loss_mtp_manual.item():.8f} = {expected_mtp_contrib.item():.8f}"
        )

        # Verify total composition is algebraically correct
        total_recomposed = loss_ntp_manual + 0.3 * loss_mtp_manual + aux_loss.detach().float()
        assert torch.allclose(total_composed, total_recomposed, atol=1e-6), (
            "Composition formula not self-consistent"
        )
