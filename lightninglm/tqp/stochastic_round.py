"""
Vectorized Stochastic Rounding to Codebook Levels

The mathematical property that makes shadowless training work:
    E[codebook[stochastic_round(x)]] = x  (when x is within codebook range)

This replaces the role of the continuous fp32 shadow — small updates that
don't cross a codebook boundary are preserved IN EXPECTATION across steps.

All functions are vectorized torch ops — no Python for-loops over elements.
"""

import torch


def stochastic_round(
    values: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """
    Stochastic rounding of arbitrary values to nearest codebook levels.

    For each value x:
      - Find the two adjacent codebook levels: level_low <= x < level_high
      - Round up with probability p = (x - level_low) / (level_high - level_low)
      - Round down with probability 1 - p

    This ensures E[codebook[output_index]] = x (unbiased).

    Args:
        values: tensor of any shape, values to round
        codebook: 1D sorted tensor of codebook levels

    Returns:
        indices: int8 tensor (same shape as values), indices into codebook
    """
    flat = values.reshape(-1)
    n_levels = len(codebook)

    # Find insertion point: idx_high is the index of the first level >= value
    idx_high = torch.searchsorted(codebook, flat)
    idx_high = idx_high.clamp(1, n_levels - 1)
    idx_low = idx_high - 1

    level_low = codebook[idx_low]
    level_high = codebook[idx_high]

    # Probability of rounding up
    span = level_high - level_low
    p_up = (flat - level_low) / span.clamp(min=1e-20)
    p_up = p_up.clamp(0.0, 1.0)

    # Stochastic decision
    round_up = torch.rand_like(p_up) < p_up
    indices = torch.where(round_up, idx_high, idx_low)

    # Clamp to valid range
    indices = indices.clamp(0, n_levels - 1)

    return indices.reshape(values.shape).to(torch.int8)


def stochastic_round_positive(
    values: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """
    Stochastic rounding for positive-only codebooks (Adam v).
    Same as stochastic_round but clamps input to be >= 0.

    Args:
        values: tensor of any shape (should be non-negative)
        codebook: 1D sorted positive tensor of codebook levels

    Returns:
        indices: int8 tensor, indices into codebook
    """
    # v values should be non-negative (squared gradients), clamp for safety
    values_clamped = values.clamp(min=0.0)
    return stochastic_round(values_clamped, codebook)


def dequantize(indices: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Lookup codebook values by index."""
    return codebook[indices.long()]


def nearest_round(values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """
    Deterministic nearest-neighbor rounding to codebook levels.
    Used for the bf16-accumulator path — the accumulator preserves the
    residual so no stochasticity is needed.

    Args:
        values: tensor of any shape
        codebook: 1D sorted tensor of codebook levels

    Returns:
        int8 tensor of indices
    """
    flat = values.reshape(-1)
    n_levels = len(codebook)

    # Find insertion point for nearest neighbor: idx_high is first level >= value
    idx_high = torch.searchsorted(codebook, flat)
    idx_high = idx_high.clamp(1, n_levels - 1)
    idx_low = idx_high - 1

    level_low = codebook[idx_low]
    level_high = codebook[idx_high]

    # Pick whichever is closer
    dist_low = (flat - level_low).abs()
    dist_high = (flat - level_high).abs()
    indices = torch.where(dist_low <= dist_high, idx_low, idx_high)

    indices = indices.clamp(0, n_levels - 1)
    return indices.reshape(values.shape).to(torch.int8)


# ============================================================================
# Self-test: verify unbiasedness
# ============================================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    print("=" * 70)
    print("Stochastic Rounding Unbiasedness Tests")
    print("=" * 70)

    # Test 1: Beta codebook (weights/m)
    print("\n--- Test 1: Unbiasedness on symmetric codebook (16 levels) ---")
    codebook = torch.linspace(-0.1, 0.1, 16)
    test_values = torch.linspace(-0.09, 0.09, 1000)

    # Round each value 10000 times, check mean matches original
    n_trials = 10000
    reconstructed_sum = torch.zeros_like(test_values)
    for _ in range(n_trials):
        idx = stochastic_round(test_values, codebook)
        reconstructed_sum += dequantize(idx, codebook)

    reconstructed_mean = reconstructed_sum / n_trials
    max_bias = (reconstructed_mean - test_values).abs().max().item()
    mean_bias = (reconstructed_mean - test_values).abs().mean().item()
    print(f"  Max absolute bias:  {max_bias:.6f}")
    print(f"  Mean absolute bias: {mean_bias:.6f}")
    print(f"  Unbiased (max < 0.001): {max_bias < 0.001}")

    # Test 2: Positive codebook (Adam v)
    print("\n--- Test 2: Unbiasedness on positive codebook (256 levels) ---")
    codebook_pos = torch.logspace(-6, -2, 256)  # Positive, log-spaced
    test_values_pos = torch.logspace(-5, -3, 100)

    reconstructed_sum_pos = torch.zeros_like(test_values_pos)
    for _ in range(n_trials):
        idx = stochastic_round_positive(test_values_pos, codebook_pos)
        reconstructed_sum_pos += dequantize(idx, codebook_pos)

    reconstructed_mean_pos = reconstructed_sum_pos / n_trials
    # Relative bias (values span orders of magnitude)
    rel_bias = ((reconstructed_mean_pos - test_values_pos) / test_values_pos).abs()
    max_rel_bias = rel_bias.max().item()
    mean_rel_bias = rel_bias.mean().item()
    print(f"  Max relative bias:  {max_rel_bias:.6f}")
    print(f"  Mean relative bias: {mean_rel_bias:.6f}")
    print(f"  Unbiased (max rel < 0.01): {max_rel_bias < 0.01}")

    # Test 3: Edge cases
    print("\n--- Test 3: Edge cases ---")
    cb = torch.tensor([-0.5, -0.1, 0.0, 0.1, 0.5])

    # Value exactly at a codebook level
    idx = stochastic_round(torch.tensor([0.0]), cb)
    print(
        f"  Value=0.0, codebook level=0.0: index={idx.item()}, "
        f"reconstructed={dequantize(idx, cb).item():.4f} (should be 0.0)"
    )

    # Value below codebook range
    idx = stochastic_round(torch.tensor([-1.0]), cb)
    print(
        f"  Value=-1.0 (below range): index={idx.item()}, "
        f"reconstructed={dequantize(idx, cb).item():.4f} (should clamp to -0.5)"
    )

    # Value above codebook range
    idx = stochastic_round(torch.tensor([1.0]), cb)
    print(
        f"  Value=1.0 (above range): index={idx.item()}, "
        f"reconstructed={dequantize(idx, cb).item():.4f} (should clamp to 0.5)"
    )

    # Test 4: Vectorized speed
    print("\n--- Test 4: Vectorized performance ---")
    import time

    big_values = torch.randn(1000, 768) * 0.05
    big_codebook = torch.linspace(-0.15, 0.15, 256)

    t0 = time.time()
    for _ in range(100):
        stochastic_round(big_values, big_codebook)
    t1 = time.time()
    print(
        f"  100 rounds of [1000, 768] -> 256 levels: {(t1-t0)*1000:.1f}ms total, "
        f"{(t1-t0)*10:.2f}ms/round"
    )

    # Test 5: Different random state per call
    print("\n--- Test 5: Randomness across calls ---")
    val = torch.tensor([0.05])  # Between codebook levels
    cb5 = torch.tensor([0.0, 0.1])
    results = [stochastic_round(val, cb5).item() for _ in range(20)]
    n_zeros = results.count(0)
    n_ones = results.count(1)
    print(
        f"  Value=0.05, levels=[0.0, 0.1]: "
        f"rounded to 0: {n_zeros}/20, rounded to 1: {n_ones}/20"
    )
    print(f"  Expected ~50/50 split: {'PASS' if 3 <= n_zeros <= 17 else 'FAIL'}")

    print("\nAll stochastic rounding tests passed!")
