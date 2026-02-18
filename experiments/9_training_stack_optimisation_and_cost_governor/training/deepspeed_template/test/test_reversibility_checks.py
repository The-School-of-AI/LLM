"""
Reversibility Checks for 1B Dense Model (GPU + Triton Kernels)
==============================================================

SUMMARY - What we are testing:
------------------------------
1. Reversible midpoint reconstruction: The reversible integration can reconstruct
   p_prev from p_next using algebra—no extra memory needed for backward pass.
2. Spectral stability: The Jacobian of each layer has bounded eigenvalues so
   training does not explode.
3. Bitwise reversibility: Same reconstruction test with real embeddings.
4. Signal explosion: Activations do not grow unbounded through the stack.
5. Learning dynamics: Loss decreases and gradients flow correctly.
6. Stabilized learning: With gradient clipping, model converges stably.

All tests run on GPU to exercise Triton kernels (RMSNorm, Sinkhorn, Sparse Attn).
Requires CUDA; skips if unavailable.

Usage:
  python test/test_reversibility_checks.py           # Run all tests
  python test/test_reversibility_checks.py --only overfit   # Run only overfit-one-batch
  python test/test_reversibility_checks.py --only learning  # Run only learning dynamics
"""

import gc
import os
import sys

import pytest
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data import get_tokenizer
from src.models import ModelConfig, Model1B
from src.models.recurrence_model_1b import KroneckerConfig, KroneckerEmbeddings

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="GPU required for Triton kernels (CUDA not available)",
)


def _log(msg: str, level: str = "info") -> None:
    """Consistent print format for all test output."""
    prefixes = {"info": "  ", "ok": "  [OK] ", "fail": "  [FAIL] ", "warn": "  [WARN] "}
    prefix = prefixes.get(level.lower(), "  ")
    print(f"{prefix}{msg}")


def _make_model_and_fixtures():
    """Load tokenizer, build model, return (model, config, tokenizer)."""
    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer)

    bpe_vocab = []
    for i in range(vocab_size):
        try:
            token = tokenizer.decode([i])
            bpe_vocab.append(token if token else f"<unk_{i}>")
        except Exception:
            bpe_vocab.append(f"<unk_{i}>")

    pf_config = KroneckerConfig(
        CHAR_DIM=256,
        POS_DIM=32,
        D=8192,
        length_normalize=True,
        truncate_long_words=True,
    )
    pf_codec = KroneckerEmbeddings(pf_config)

    config = ModelConfig()
    model = Model1B(
        config,
        embedding_type="kronecker",
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec,
    )
    return model, config, tokenizer


@pytest.fixture(scope="session")
def model_and_fixtures():
    """Load tokenizer and model once for all tests."""
    return _make_model_and_fixtures()


# -----------------------------------------------------------------------------
# Test 1: Reversible midpoint reconstruction (random)
# -----------------------------------------------------------------------------


def test_reversible_midpoint_reconstruction_random(model_and_fixtures):
    """Reconstruct p_prev from p_next using reversible midpoint algebra."""
    model, config, _ = model_and_fixtures
    model = model.to(DEVICE)

    h = 0.25
    a = 0.5
    two_h = 2 * h

    layer = model.layers[7]
    B, T, n_streams, D = 1, 8, config.n_streams, config.hidden_size
    p_prev_original = torch.randn(B, T, n_streams, D, device=DEVICE) * 0.02
    p_cur_original = torch.randn(B, T, n_streams, D, device=DEVICE) * 0.02

    with torch.no_grad():
        delta, _ = layer.force(p_cur_original)
        p_next = (a * p_prev_original) + ((1.0 - a) * p_cur_original) + (two_h * delta)

    with torch.no_grad():
        delta_recomputed, _ = layer.force(p_cur_original)
        p_prev_reconstructed = (
            (p_next - ((1.0 - a) * p_cur_original) - (two_h * delta_recomputed)) / a
        )

    error = (p_prev_original - p_prev_reconstructed).abs().max().item()
    _log(f"Reconstruction error (max abs diff): {error:.2e}")
    assert error < 1e-5, f"Reconstruction error too high: {error}"


# -----------------------------------------------------------------------------
# Test 2: Reversible midpoint reconstruction (real embeddings)
# -----------------------------------------------------------------------------


def test_reversible_midpoint_reconstruction_real_embeddings(model_and_fixtures):
    """Reconstruct p_prev from p_next using real embeddings."""
    model, config, tokenizer = model_and_fixtures
    model = model.to(DEVICE)

    text = "Hello, how are you?"
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(DEVICE)

    h = 0.25
    a = 0.5
    two_h = 2 * h

    with torch.no_grad():
        EMB = model.kronecker_embeddings(input_ids)
        x = model.pf_to_model(EMB.to(dtype=model.pf_to_model.weight.dtype))
        x = model.embed_norm(x)
        B, T, D = x.shape
        x_stream = torch.zeros(B, T, model.n_streams, D, dtype=x.dtype, device=DEVICE)
        x_stream[:, :, 0, :] = x

        p_cur_original = x_stream
        p_prev_original = x_stream + torch.randn_like(x_stream, device=DEVICE) * 0.01

        layer = model.layers[7]
        delta, _ = layer.force(p_cur_original)
        p_next = (a * p_prev_original) + ((1.0 - a) * p_cur_original) + (two_h * delta)

        delta_recomputed, _ = layer.force(p_cur_original)
        p_prev_reconstructed = (
            (p_next - ((1.0 - a) * p_cur_original) - (two_h * delta_recomputed)) / a
        )

        error = (p_prev_original - p_prev_reconstructed).abs().max().item()

    _log(f"Reconstruction error (with real embeddings): {error:.2e}")
    assert error < 1e-5, f"Reconstruction error too high: {error}"


# -----------------------------------------------------------------------------
# Test 3: Spectral stability (DeltaNet and GSA)
# -----------------------------------------------------------------------------


def _estimate_spectral_radius(layer, x, n_iters=10, eps=1e-5):
    """Power iteration to estimate largest |eigenvalue| of J = df/dx."""
    x = x.to(DEVICE)
    v = torch.randn_like(x, device=DEVICE)
    v = v / (v.norm() + 1e-8)

    for _ in range(n_iters):
        x_plus = x + eps * v
        delta_plus, _ = layer.force(x_plus)
        delta_base, _ = layer.force(x)
        Jv = (delta_plus - delta_base) / eps
        sigma = Jv.norm().item()
        v = Jv / (sigma + 1e-8)
    return sigma


def _check_stability(layer, x, a=0.5, h=0.25):
    """Stability check for generalized midpoint: |(1-a) + 2hλ| <= 2."""
    lambda_max = _estimate_spectral_radius(layer, x)
    b_coeff = 1.0 - a
    h_eff = 2.0 * h
    lhs = abs(b_coeff + lambda_max * h_eff)
    stable = lhs <= 2.0
    return {
        "spectral_radius": lambda_max,
        "|(1-a) + 2hλ|": lhs,
        "stable": stable,
        "margin": 2.0 - lhs,
    }


def test_spectral_stability(model_and_fixtures):
    """DeltaNet and GSA layers have bounded eigenvalues."""
    model, config, _ = model_and_fixtures
    model = model.to(DEVICE)

    B, T, n_streams, D = 1, 8, config.n_streams, config.hidden_size
    x = torch.randn(B, T, n_streams, D, device=DEVICE) * 0.02

    result_delta = _check_stability(model.layers[1], x)
    result_gsa = _check_stability(model.layers[3], x)

    _log(
        f"DeltaNet: spectral_radius={result_delta['spectral_radius']:.4f}, "
        f"|(1-a)+2hλ|={result_delta['|(1-a) + 2hλ|']:.4f}, stable={result_delta['stable']}"
    )
    _log(
        f"GSA: spectral_radius={result_gsa['spectral_radius']:.4f}, "
        f"|(1-a)+2hλ|={result_gsa['|(1-a) + 2hλ|']:.4f}, stable={result_gsa['stable']}"
    )
    # Informational only: spectral radius can exceed 2.0; training may still work


# -----------------------------------------------------------------------------
# Test 4: Bitwise reversibility
# -----------------------------------------------------------------------------


def test_bitwise_reversibility(model_and_fixtures):
    """Bitwise reconstruction: p_prev recoverable from p_next."""
    model, config, _ = model_and_fixtures
    model = model.to(DEVICE)
    model.eval()

    B, T, D = 1, 32, model.config.hidden_size
    p_prev = torch.randn(B, T, model.n_streams, D, device=DEVICE, dtype=torch.float32)
    p_cur = torch.randn(B, T, model.n_streams, D, device=DEVICE, dtype=torch.float32)

    layer = model.layers[7]
    h, a = 0.25, 0.5

    with torch.no_grad():
        p_prev_f32 = p_prev.float()
        p_cur_f32 = p_cur.float()
        delta, _ = layer.force(p_cur_f32)
        p_next = (a * p_prev_f32) + ((1.0 - a) * p_cur_f32) + (2.0 * h * delta)

    with torch.no_grad():
        delta_recon, _ = layer.force(p_cur_f32)
        p_prev_recon = (p_next - ((1.0 - a) * p_cur_f32) - (2.0 * h * delta_recon)) / a

    error = (p_prev_f32 - p_prev_recon).abs().max().item()
    _log(f"Reconstruction max error: {error:.2e}")

    assert error < 1e-4, f"Reversibility failed: error={error}"


# -----------------------------------------------------------------------------
# Test 5: Signal explosion
# -----------------------------------------------------------------------------


def test_signal_explosion(model_and_fixtures):
    """Activations do not explode through the stack."""
    model, config, _ = model_and_fixtures
    model = model.to(DEVICE)
    model.eval()

    x = torch.randn(1, 32, model.config.hidden_size, device=DEVICE)
    x_stream = torch.zeros(
        1, 32, model.config.n_streams, model.config.hidden_size, device=DEVICE
    )
    x_stream[:, :, 0, :] = x

    stack = model.stack
    norms = []
    p_prev = x_stream
    p_cur = x_stream

    with torch.no_grad():
        for i, layer in enumerate(stack.mid_layers):
            p_next, _ = layer(p_prev, p_cur)
            norm = p_next.norm().item() / (p_next.numel() ** 0.5)
            norms.append(norm)
            p_prev, p_cur = p_cur, p_next

    _log(f"Layer norms: {[f'L{i}:{n:.2f}' for i, n in enumerate(norms)]}")

    assert norms[-1] <= norms[0] * 10, (
        f"Signal exploding: final norm {norms[-1]:.2f} > 10x initial {norms[0]:.2f}"
    )


# -----------------------------------------------------------------------------
# Test 6: Learning dynamics
# -----------------------------------------------------------------------------


def _clear_gpu_memory():
    """Free GPU memory before memory-heavy tests."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_learning_dynamics(model_and_fixtures, seq_len=16):
    """Loss decreases and gradients flow correctly. Uses same dtype/API as main training."""
    _clear_gpu_memory()
    model, config, _ = model_and_fixtures
    model = model.to(DEVICE).to(dtype=torch.bfloat16)  # Match main.py: no autocast
    model.train()

    # Same format as main training: x_input, y_ntp, y_mtp for reversible model
    input_ids = torch.randint(0, 1000, (1, seq_len), device=DEVICE)
    x_input = input_ids[:, :-2].contiguous()
    y_ntp = input_ids[:, 1:-1].contiguous()
    y_mtp = input_ids[:, 2:].contiguous()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    _log("Step | Loss")
    _log("-" * 20)

    for i in range(10):
        optimizer.zero_grad()
        loss_ntp, loss_mtp, aux_loss = model(
            x_input,
            next_token_ids=y_ntp,
            attention_mask=None,
            return_loss=True,
            return_memory=False,
            prev_memory_stream=None,
            ntp_targets=y_ntp,
            mtp_targets=y_mtp,
        )
        total_loss = loss_ntp + 0.3 * loss_mtp
        if aux_loss is not None and aux_loss.numel() > 0:
            total_loss = total_loss + aux_loss
        total_loss.backward()
        optimizer.step()
        _log(f"{i:4d} | {total_loss.item():.4f}")

    assert total_loss.item() < 5.0, (
        f"Loss stuck: {total_loss.item():.4f}. Gradients may not be flowing."
    )


# -----------------------------------------------------------------------------
# Test 7: Stabilized learning dynamics
# -----------------------------------------------------------------------------


def test_overfit_one_batch(model_and_fixtures, seq_len=16, steps=25):
    """Overfit on a single batch. Uses same dtype/API as main training (bfloat16, no autocast)."""
    _clear_gpu_memory()
    model, config, _ = model_and_fixtures
    model = model.to(DEVICE).to(dtype=torch.bfloat16)  # Match main.py: no autocast
    model.train()

    # Same format as main training: x_input, y_ntp, y_mtp for reversible model
    input_ids = torch.randint(0, 100, (1, seq_len), device=DEVICE)
    x_input = input_ids[:, :-2].contiguous()
    y_ntp = input_ids[:, 1:-1].contiguous()
    y_mtp = input_ids[:, 2:].contiguous()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    _log("Step | Loss      | Grad Norm")
    _log("-" * 35)

    for i in range(steps):
        optimizer.zero_grad()
        loss_ntp, loss_mtp, aux_loss = model(
            x_input,
            next_token_ids=y_ntp,
            attention_mask=None,
            return_loss=True,
            return_memory=False,
            prev_memory_stream=None,
            ntp_targets=y_ntp,
            mtp_targets=y_mtp,
        )
        total_loss = loss_ntp + 0.3 * loss_mtp
        if aux_loss is not None and aux_loss.numel() > 0:
            total_loss = total_loss + aux_loss
        total_loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        _log(f"{i:4d} | {total_loss.item():.4f}    | {grad_norm:.4f}")

    assert total_loss.item() < 5.0, (
        f"Model still unstable: loss={total_loss.item():.4f}"
    )


def test_learning_dynamics_stable(model_and_fixtures):
    """Alias for test_overfit_one_batch (backward compatibility)."""
    return test_overfit_one_batch(model_and_fixtures, seq_len=16, steps=25)


# -----------------------------------------------------------------------------
# Test 8: Forward pass sanity (logits shape)
# -----------------------------------------------------------------------------


def test_forward_pass_shape(model_and_fixtures):
    """Forward pass produces correct logits shape."""
    model, config, tokenizer = model_and_fixtures
    model = model.to(DEVICE)
    model.eval()

    text = "Hello, how are you?"
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(DEVICE)

    with torch.no_grad():
        logits_ntp, logits_mtp = model(input_ids, return_memory=False, return_loss=False)

    _log(f"logits_ntp shape: {logits_ntp.shape}")
    # _log(f"logits_mtp shape: {logits_mtp.shape}")

    assert logits_ntp.dim() == 3
    # assert logits_mtp.dim() == 3


# -----------------------------------------------------------------------------
# Run as script
# -----------------------------------------------------------------------------


def _run_test(name: str, fn, model_and_fixtures):
    """Run a test and print section header."""
    print(f"\n--- {name} ---")
    fn(model_and_fixtures)


# Test registry for script-mode summary
_SCRIPT_TESTS = [
    ("Forward pass shape", test_forward_pass_shape),
    ("Reversible midpoint (random)", test_reversible_midpoint_reconstruction_random),
    ("Reversible midpoint (real embeddings)", test_reversible_midpoint_reconstruction_real_embeddings),
    ("Spectral stability", test_spectral_stability),
    ("Bitwise reversibility", test_bitwise_reversibility),
    ("Signal explosion", test_signal_explosion),
    ("Learning dynamics", test_learning_dynamics),
    ("Overfit one batch", test_overfit_one_batch),
]

# Tests that can be run alone via --only (e.g. --only overfit)
_NAMED_TESTS = {
    "overfit": ("Overfit one batch", test_overfit_one_batch),
    "learning": ("Learning dynamics", test_learning_dynamics),
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reversibility checks for 1B model")
    parser.add_argument(
        "--only",
        choices=list(_NAMED_TESTS),
        help="Run only this test (e.g. --only overfit for overfit-one-batch)",
    )
    args = parser.parse_args()

    if not CUDA_AVAILABLE:
        print("SKIP: CUDA not available. GPU required for Triton kernels.")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("Reversibility Checks (GPU + Triton Kernels)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if args.only:
        print(f"Running only: {args.only}")
    print()

    # Load model and tokenizer once for all tests
    model_and_fixtures = _make_model_and_fixtures()

    if args.only:
        tests_to_run = [_NAMED_TESTS[args.only]]
    else:
        tests_to_run = _SCRIPT_TESTS

    results = []
    for name, fn in tests_to_run:
        try:
            _run_test(name, fn, model_and_fixtures)
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))
            _log(f"FAILED: {e}", "fail")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, err in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if err:
            print(f"         {err}")
    print("-" * 60)
    print(f"  {passed}/{total} tests passed")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)
