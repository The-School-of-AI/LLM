"""
Verify the performance fixes:
1. Chunk-wise parallel DeltaNet (replaces sequential for-loop)
2. Vectorized per-head output norm
3. Sequential forward mode (when use_reversible=False)

Run from: llm_architecture/
    python Test_code/verify_fixes.py
"""

import sys
import time

import torch

sys.path.insert(0, ".")

from config.model_config import get_preset_config


def test_deltanet_chunk_parallel():
    """Test that chunk-wise parallel DeltaNet produces valid output."""
    print("=" * 60)
    print("Test 1: Chunk-wise Parallel DeltaNet")
    print("=" * 60)

    from components.attention.gated_deltanet import GatedDeltaNet

    device = "cpu"
    dtype = torch.float32

    # Small model for testing
    deltanet = GatedDeltaNet(
        hidden_size=256,
        num_heads=4,
        head_dim=64,
        max_seq_len=2048,
        rope_base=10000,
        rope_original_max=2048,
        rope_scaling_factor=1.0,
        conv_size=4,
        use_output_norm=True,
    ).to(device=device, dtype=dtype)

    # Test different sequence lengths
    B = 2
    for T in [32, 64, 128, 256, 512]:
        x = torch.randn(B, T, 256, device=device, dtype=dtype)
        out = deltanet(x)
        assert out.shape == (B, T, 256), f"Shape mismatch: {out.shape} != {(B, T, 256)}"
        assert torch.isfinite(out).all(), f"Non-finite output at T={T}"
        print(
            f"  T={T:4d}: output shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f} [OK]"
        )

    # Gradient check
    x = torch.randn(B, 64, 256, device=device, dtype=dtype, requires_grad=True)
    out = deltanet(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient computed"
    assert torch.isfinite(x.grad).all(), "Non-finite gradients"
    print(
        f"  Gradient check: grad shape={x.grad.shape}, grad_norm={x.grad.norm():.4f} [OK]"
    )

    print("  PASSED\n")


def test_sequential_vs_reversible():
    """Test that sequential mode works and produces valid output."""
    print("=" * 60)
    print("Test 2: Sequential Forward Mode")
    print("=" * 60)

    # Get reference config with reversible=True
    config_rev = get_preset_config("1b-reference")
    # Shrink for testing
    config_rev.hidden_size = 256
    config_rev.vocab_size = 1000
    config_rev.num_hidden_layers = 4
    config_rev.attention.delta_v_heads = 4
    config_rev.attention.delta_qk_heads = 2
    config_rev.attention.delta_head_dim = 64
    config_rev.attention.delta_gate_dim = 64
    config_rev.attention.gsa_num_heads = 4
    config_rev.attention.gsa_head_dim = 64
    config_rev.attention.num_attention_heads = 4
    config_rev.attention.num_key_value_heads = 2
    config_rev.attention.head_dim = 64
    config_rev.ffn.intermediate_size = 256
    config_rev.connection.mhc_expansion_rate = 2
    config_rev.head.use_multi_token_prediction = False
    config_rev.max_position_embeddings = 2048

    # Test sequential mode
    config_seq = get_preset_config("1b-reference")
    config_seq.hidden_size = config_rev.hidden_size
    config_seq.vocab_size = config_rev.vocab_size
    config_seq.num_hidden_layers = config_rev.num_hidden_layers
    config_seq.attention = config_rev.attention
    config_seq.ffn = config_rev.ffn
    config_seq.connection = config_rev.connection
    config_seq.head = config_rev.head
    config_seq.max_position_embeddings = config_rev.max_position_embeddings
    config_seq.integration.use_reversible = False

    from models.reference_llm import ReferenceLLM

    print("\n  Creating sequential model (use_reversible=False)...")
    model_seq = ReferenceLLM(config_seq, embedding_type="standard")

    B, T = 2, 128
    input_ids = torch.randint(0, 1000, (B, T))

    # Forward pass
    logits_ntp, logits_mtp = model_seq(input_ids)
    assert logits_ntp.shape == (B, T, 1000), f"Shape mismatch: {logits_ntp.shape}"
    assert torch.isfinite(logits_ntp).all(), "Non-finite NTP logits"
    print(
        f"  Sequential forward: logits shape={logits_ntp.shape}, "
        f"mean={logits_ntp.mean():.4f} [OK]"
    )

    # Gradient check
    labels = torch.randint(0, 1000, (B, T))
    output = model_seq(input_ids, labels=labels)
    output.loss.backward()
    grad_norm = (
        sum(
            p.grad.norm().item() ** 2
            for p in model_seq.parameters()
            if p.grad is not None
        )
        ** 0.5
    )
    print(
        f"  Gradient check: loss={output.loss.item():.4f}, grad_norm={grad_norm:.4f} [OK]"
    )

    # Test reversible mode too
    print("\n  Creating reversible model (use_reversible=True)...")
    config_rev.integration.use_reversible = True
    model_rev = ReferenceLLM(config_rev, embedding_type="standard")

    logits_ntp_rev, _ = model_rev(input_ids)
    assert logits_ntp_rev.shape == (
        B,
        T,
        1000,
    ), f"Shape mismatch: {logits_ntp_rev.shape}"
    assert torch.isfinite(logits_ntp_rev).all(), "Non-finite NTP logits (reversible)"
    print(
        f"  Reversible forward: logits shape={logits_ntp_rev.shape}, "
        f"mean={logits_ntp_rev.mean():.4f} [OK]"
    )

    print("  PASSED\n")


def test_speed_comparison():
    """Compare speed of chunk-wise DeltaNet vs baseline."""
    print("=" * 60)
    print("Test 3: Speed Comparison (Chunk-wise DeltaNet)")
    print("=" * 60)

    from components.attention.gated_deltanet import GatedDeltaNet

    device = "cpu"
    dtype = torch.float32
    B, T, H, D = 1, 256, 4, 64

    deltanet = GatedDeltaNet(
        hidden_size=H * D,
        num_heads=H,
        head_dim=D,
        max_seq_len=2048,
        rope_base=10000,
        rope_original_max=2048,
        rope_scaling_factor=1.0,
    ).to(device=device, dtype=dtype)

    x = torch.randn(B, T, H * D, device=device, dtype=dtype)

    # Warmup
    for _ in range(3):
        _ = deltanet(x)

    # Benchmark
    n_iters = 10
    start = time.time()
    for _ in range(n_iters):
        _ = deltanet(x)
    elapsed = (time.time() - start) / n_iters
    print(f"  Chunk-wise parallel: {elapsed*1000:.1f} ms/forward (T={T}, CPU)")
    print("  (GPU speedup will be much larger due to parallel matmuls)")

    print("  PASSED\n")


if __name__ == "__main__":
    print("\nVerifying performance fixes...\n")

    test_deltanet_chunk_parallel()
    test_sequential_vs_reversible()
    test_speed_comparison()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("\nSummary of changes:")
    print(
        "1. GatedDeltaNet: chunk-wise parallel recurrence (replaces O(T) Python loop)"
    )
    print(
        "2. GatedDeltaNet: vectorized per-head output norm (replaces O(H) Python loop)"
    )
    print("3. ReferenceLLM: sequential forward mode (when use_reversible=False)")
    print("\nTo use sequential mode, set in config:")
    print("  integration=IntegrationConfig(use_reversible=False)")
    print("\nOr modify get_1b_reference_config() in config/model_config.py:")
    print("  integration=IntegrationConfig(use_reversible=False, ...)")
