"""
Lloyd-Max Codebook for Beta Distribution (TurboQuant)

After random orthogonal rotation, each coordinate of a unit-norm weight vector
follows Beta((d-1)/2, (d-1)/2) concentrated near 0.5 (for large d). We exploit
this known distribution to precompute optimal scalar quantization codebooks
without any calibration data.

Reference: TurboQuant (arXiv 2504.19874), Algorithm 1 (TurboQuant_mse)
           PolarQuant (arXiv 2502.02617) for the rotation -> Beta distribution theory

The codebook is:
  - levels: the reconstruction values (centroids)
  - boundaries: the decision boundaries between levels
  - Optimized via iterative Lloyd-Max on the Beta PDF
"""

import os
from typing import Optional, Tuple

import numpy as np
from scipy import integrate, stats


def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """
    PDF of Beta((d-1)/2, (d-1)/2) — the distribution of each rotated coordinate
    after projecting a unit-norm vector via random orthogonal rotation in R^d.

    For large d, this concentrates tightly around 0.5.
    Actually, the rotated coordinates (before shifting) are centered at 0,
    so we use the shifted Beta on [-1, 1] via the transform x -> (x+1)/2.

    But TurboQuant works on the raw coordinate which, after rotation of a
    unit-norm vector, follows a distribution on [-1/sqrt(d), 1/sqrt(d)]
    approximately. For practical Lloyd-Max, we work on the normalized
    coordinate z = sqrt(d) * x_rotated, which follows approximately N(0,1)
    for large d, or more precisely the marginal of a uniform point on S^{d-1}.

    The exact marginal of one coordinate of a uniform point on S^{d-1}(1)
    (the unit sphere) is:
        f(x) = C_d * (1 - x^2)^{(d-3)/2}  for x in [-1, 1]
    which is Beta((d-1)/2, (d-1)/2) shifted to [-1, 1].

    For weight matrices, row-normalize to unit norm first, then rotate.
    """
    alpha = (d - 1) / 2.0
    # Beta(alpha, alpha) on [0, 1], shifted to [-1, 1] via x -> (x+1)/2
    # f_{[-1,1]}(x) = (1/2) * Beta_pdf((x+1)/2; alpha, alpha)
    t = (x + 1.0) / 2.0
    # Clamp to valid range
    t = np.clip(t, 1e-15, 1 - 1e-15)
    return 0.5 * stats.beta.pdf(t, alpha, alpha)


def beta_cdf(x: np.ndarray, d: int) -> np.ndarray:
    """CDF of the shifted Beta on [-1, 1]."""
    alpha = (d - 1) / 2.0
    t = (x + 1.0) / 2.0
    t = np.clip(t, 0.0, 1.0)
    return stats.beta.cdf(t, alpha, alpha)


def beta_ppf(q: np.ndarray, d: int) -> np.ndarray:
    """Inverse CDF (quantile function) of the shifted Beta on [-1, 1]."""
    alpha = (d - 1) / 2.0
    q = np.clip(q, 1e-15, 1 - 1e-15)
    t = stats.beta.ppf(q, alpha, alpha)
    return 2.0 * t - 1.0


def conditional_mean(a: float, b: float, d: int) -> float:
    """
    E[X | a <= X <= b] for X ~ shifted Beta((d-1)/2, (d-1)/2) on [-1, 1].

    Used in Lloyd-Max centroid update step.
    """
    alpha = (d - 1) / 2.0

    # Numerator: integral of x * f(x) from a to b
    def integrand(x):
        t = (x + 1.0) / 2.0
        t = np.clip(t, 1e-15, 1 - 1e-15)
        return x * 0.5 * stats.beta.pdf(t, alpha, alpha)

    prob_mass = beta_cdf(np.array([b]), d)[0] - beta_cdf(np.array([a]), d)[0]
    if prob_mass < 1e-20:
        return (a + b) / 2.0

    num, _ = integrate.quad(integrand, a, b, limit=100)
    return num / prob_mass


def lloyd_max_codebook(
    d: int,
    num_levels: int,
    max_iters: int = 200,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Lloyd-Max optimal scalar quantizer for the marginal distribution
    of one coordinate of a uniform point on S^{d-1}.

    Args:
        d: dimension of the weight vector (in_features for a weight matrix row)
        num_levels: number of quantization levels (2^bits, e.g. 16 for 4-bit)
        max_iters: maximum Lloyd-Max iterations
        tol: convergence tolerance on level movement

    Returns:
        levels: (num_levels,) array of reconstruction values (centroids)
        boundaries: (num_levels+1,) array of decision boundaries
                    boundaries[0] = -1, boundaries[-1] = 1
    """
    # Initialize levels uniformly in quantile space (good initialization)
    quantile_points = np.linspace(0.5 / num_levels, 1.0 - 0.5 / num_levels, num_levels)
    levels = beta_ppf(quantile_points, d)

    for iteration in range(max_iters):
        # Step 1: Update boundaries (midpoints between adjacent levels)
        boundaries = np.zeros(num_levels + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(1, num_levels):
            boundaries[i] = (levels[i - 1] + levels[i]) / 2.0

        # Step 2: Update levels (conditional means within each partition)
        new_levels = np.zeros(num_levels)
        for i in range(num_levels):
            new_levels[i] = conditional_mean(boundaries[i], boundaries[i + 1], d)

        # Check convergence
        delta = np.max(np.abs(new_levels - levels))
        levels = new_levels

        if delta < tol:
            break

    # Final boundary update
    boundaries = np.zeros(num_levels + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(1, num_levels):
        boundaries[i] = (levels[i - 1] + levels[i]) / 2.0

    return levels.astype(np.float32), boundaries.astype(np.float32)


def compute_distortion(levels: np.ndarray, boundaries: np.ndarray, d: int) -> float:
    """
    Compute mean squared quantization error (distortion) for the given codebook.

    D = E[(X - Q(X))^2] = sum_i integral_{b_i}^{b_{i+1}} (x - c_i)^2 f(x) dx
    """
    alpha = (d - 1) / 2.0
    total = 0.0

    for i in range(len(levels)):
        a, b = boundaries[i], boundaries[i + 1]

        def integrand(x):
            t = (x + 1.0) / 2.0
            t = np.clip(t, 1e-15, 1 - 1e-15)
            return (x - levels[i]) ** 2 * 0.5 * stats.beta.pdf(t, alpha, alpha)

        val, _ = integrate.quad(integrand, a, b, limit=100)
        total += val

    return total


def compute_variance(d: int) -> float:
    """Variance of one coordinate of a uniform point on S^{d-1}."""
    # Var = 1/d for uniform on unit sphere
    return 1.0 / d


def get_codebook(
    d: int,
    bits: int = 4,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get or compute Lloyd-Max codebook for given dimension and bit-width.

    Caches to disk so codebook is only computed once per (d, bits) pair.

    Args:
        d: dimension (in_features of the weight matrix)
        bits: quantization bit-width (2, 3, or 4)
        cache_dir: directory for caching codebooks (default: ~/.cache/turboquant/)

    Returns:
        levels: (2^bits,) float32 array of reconstruction values
        boundaries: (2^bits + 1,) float32 array of decision boundaries
    """
    assert bits in (2, 3, 4, 6, 8), f"Supported bit-widths: 2, 3, 4, 6, 8. Got {bits}"
    num_levels = 2**bits

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/turboquant")
    os.makedirs(cache_dir, exist_ok=True)

    cache_key = f"lloyd_max_d{d}_b{bits}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.npz")

    # Check cache
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["levels"], data["boundaries"]

    # Compute
    print(
        f"[codebook] Computing Lloyd-Max codebook: d={d}, bits={bits}, levels={num_levels}"
    )

    if bits >= 6:
        # For 6-bit+ (64+ levels), iterative Lloyd-Max is slow with scipy quad.
        # Use quantile-based codebook: place levels at quantile midpoints of the Beta
        # distribution. This is near-optimal for concentrated distributions (large d).
        levels = beta_ppf(
            np.linspace(0.5 / num_levels, 1.0 - 0.5 / num_levels, num_levels), d
        ).astype(np.float32)
        boundaries = np.zeros(num_levels + 1, dtype=np.float32)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(1, num_levels):
            boundaries[i] = (levels[i - 1] + levels[i]) / 2.0
    else:
        levels, boundaries = lloyd_max_codebook(d, num_levels)

    # Compute and print distortion metrics
    distortion = compute_distortion(levels, boundaries, d)
    variance = compute_variance(d)
    snr_db = 10 * np.log10(variance / (distortion + 1e-30))
    print(f"[codebook]   Distortion: {distortion:.6e}")
    print(f"[codebook]   Variance:   {variance:.6e}")
    print(f"[codebook]   SNR:        {snr_db:.2f} dB")
    print(f"[codebook]   Levels:     {levels}")

    # Save to cache
    np.savez(cache_file, levels=levels, boundaries=boundaries)
    print(f"[codebook]   Cached to: {cache_file}")

    return levels, boundaries


# ============================================================================
# Chi-squared codebook for Adam v (second moment)
# ============================================================================


def chisq_conditional_mean(a: float, b: float, scale: float = 1.0) -> float:
    """
    E[X | a <= X <= b] for X ~ scale * chi-squared(1).

    chi-squared(1) PDF: f(x) = (1/(sqrt(2*pi*x))) * exp(-x/2) for x > 0
    Scaled: f_s(x) = (1/s) * f(x/s)
    """
    if b <= a + 1e-20:
        return (a + b) / 2.0

    def integrand_num(x):
        return x * stats.chi2.pdf(x / scale, df=1) / scale

    def integrand_den(x):
        return stats.chi2.pdf(x / scale, df=1) / scale

    num, _ = integrate.quad(integrand_num, a, b, limit=100)
    den, _ = integrate.quad(integrand_den, a, b, limit=100)

    if den < 1e-20:
        return (a + b) / 2.0
    return num / den


def lloyd_max_chisq_codebook(
    num_levels: int,
    scale: float = 1.0,
    max_iters: int = 200,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lloyd-Max codebook for scale * chi-squared(1) distribution.
    All levels are positive. Used for Adam v (second moment = EMA of grad^2).

    Args:
        num_levels: number of quantization levels
        scale: scaling factor (gradient_variance / d)
        max_iters: max iterations
        tol: convergence tolerance

    Returns:
        levels: (num_levels,) sorted positive float32 centroids
        boundaries: (num_levels+1,) boundaries, boundaries[0]=0
    """
    # Initialize at quantile midpoints of scaled chi-squared(1)
    quantile_points = np.linspace(0.5 / num_levels, 1.0 - 0.5 / num_levels, num_levels)
    levels = stats.chi2.ppf(quantile_points, df=1) * scale

    for iteration in range(max_iters):
        boundaries = np.zeros(num_levels + 1)
        boundaries[0] = 0.0
        # Upper boundary: cover 1 - 1e-8 quantile
        boundaries[-1] = stats.chi2.ppf(1 - 1e-8, df=1) * scale
        for i in range(1, num_levels):
            boundaries[i] = (levels[i - 1] + levels[i]) / 2.0

        new_levels = np.zeros(num_levels)
        for i in range(num_levels):
            new_levels[i] = chisq_conditional_mean(
                boundaries[i], boundaries[i + 1], scale
            )

        delta = np.max(np.abs(new_levels - levels))
        levels = new_levels
        if delta < tol:
            break

    # Final boundaries
    boundaries = np.zeros(num_levels + 1)
    boundaries[0] = 0.0
    boundaries[-1] = stats.chi2.ppf(1 - 1e-8, df=1) * scale
    for i in range(1, num_levels):
        boundaries[i] = (levels[i - 1] + levels[i]) / 2.0

    return levels.astype(np.float32), boundaries.astype(np.float32)


def get_v_codebook(
    d: int,
    bits: int = 8,
    grad_var_estimate: float = 1.0,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get Lloyd-Max codebook for Adam v (second moment) in TQ rotated space.

    After rotation, gradient coordinates are ~N(0, sigma^2/d).
    Squaring gives sigma^2/d * chi-squared(1).
    Scale = grad_var_estimate / d.

    Args:
        d: dimension (in_features)
        bits: quantization bit-width (typically 8)
        grad_var_estimate: estimated gradient variance (updated during training if needed)
        cache_dir: cache directory

    Returns:
        levels: (2^bits,) positive float32 centroids
        boundaries: (2^bits+1,) boundaries
    """
    assert bits in (
        2,
        3,
        4,
        6,
        8,
    ), f"v_codebook supports 2, 3, 4, 6, 8 bits. Got {bits}"
    num_levels = 2**bits
    scale = grad_var_estimate / d

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/turboquant")
    os.makedirs(cache_dir, exist_ok=True)

    # Cache key includes scale (quantized to 2 significant figures for stability)
    scale_key = f"{scale:.2e}".replace("+", "p").replace("-", "n")
    cache_key = f"chisq_d{d}_b{bits}_s{scale_key}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.npz")

    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["levels"], data["boundaries"]

    print(
        f"[codebook] Computing chi-squared codebook: d={d}, bits={bits}, "
        f"levels={num_levels}, scale={scale:.2e}"
    )

    if bits >= 6:
        # Quantile-based for 64+ levels (iterative too slow)
        quantile_points = np.linspace(
            0.5 / num_levels, 1.0 - 0.5 / num_levels, num_levels
        )
        levels = (stats.chi2.ppf(quantile_points, df=1) * scale).astype(np.float32)
        boundaries = np.zeros(num_levels + 1, dtype=np.float32)
        boundaries[0] = 0.0
        boundaries[-1] = float(stats.chi2.ppf(1 - 1e-8, df=1) * scale)
        for i in range(1, num_levels):
            boundaries[i] = (levels[i - 1] + levels[i]) / 2.0
    else:
        levels, boundaries = lloyd_max_chisq_codebook(num_levels, scale)

    # Validate: all levels must be positive
    assert np.all(levels >= 0), f"v_codebook has negative levels: {levels[levels < 0]}"

    # Distortion estimate via sampling
    samples = stats.chi2.rvs(df=1, size=100000) * scale
    indices = np.searchsorted(boundaries[1:-1], samples).clip(0, num_levels - 1)
    reconstructed = levels[indices]
    distortion = np.mean((samples - reconstructed) ** 2)
    variance = np.var(samples)
    snr_db = 10 * np.log10(variance / (distortion + 1e-30))

    print(f"[codebook]   Distortion: {distortion:.6e}")
    print(f"[codebook]   Variance:   {variance:.6e}")
    print(f"[codebook]   SNR:        {snr_db:.2f} dB")
    print(f"[codebook]   Level range: [{levels[0]:.6e}, {levels[-1]:.6e}]")
    print(f"[codebook]   All positive: {np.all(levels >= 0)}")

    np.savez(cache_file, levels=levels, boundaries=boundaries)
    print(f"[codebook]   Cached to: {cache_file}")

    return levels, boundaries


def quantize_array(
    x: np.ndarray,
    levels: np.ndarray,
    boundaries: np.ndarray,
    stochastic: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Quantize array x using the given codebook.

    Args:
        x: values to quantize
        levels: codebook centroids
        boundaries: decision boundaries
        stochastic: if True, use stochastic rounding
        rng: random generator for stochastic rounding

    Returns:
        indices: integer indices into levels array (same shape as x)
    """
    if stochastic and rng is None:
        rng = np.random.default_rng()

    # Deterministic: find nearest level
    # Use searchsorted on boundaries to find bin, then clamp
    indices = np.searchsorted(boundaries[1:-1], x)  # bin index in [0, num_levels-1]
    indices = np.clip(indices, 0, len(levels) - 1)

    if stochastic:
        # Stochastic rounding: probabilistically round to adjacent level
        # For each x, if it falls in bin i, compute probability of rounding
        # to i vs i+1 based on distance to boundaries
        for idx in range(len(levels) - 1):
            mask = indices == idx
            if not np.any(mask):
                continue
            x_masked = x[mask]
            # Distance to current level vs next level
            d_curr = np.abs(x_masked - levels[idx])
            d_next = np.abs(x_masked - levels[idx + 1])
            total = d_curr + d_next
            # Probability of rounding UP to next level
            p_up = np.where(total > 1e-15, d_curr / total, 0.5)
            # Stochastic decision
            round_up = rng.random(size=p_up.shape) < p_up
            new_indices = indices[mask].copy()
            new_indices[round_up] = idx + 1
            indices[mask] = new_indices

    return indices.astype(np.int8 if len(levels) <= 128 else np.int16)


# ============================================================================
# Self-test / validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Lloyd-Max Codebook Validation")
    print("=" * 70)

    # Test dimensions matching our model
    test_dims = [1024, 2048, 4096]
    test_bits = [2, 3, 4]

    for d in test_dims:
        print(f"\n--- Dimension d={d} ---")
        variance = compute_variance(d)
        print(f"  Theoretical variance (1/d): {variance:.6e}")

        for bits in test_bits:
            levels, boundaries = get_codebook(d, bits)
            distortion = compute_distortion(levels, boundaries, d)
            snr = 10 * np.log10(variance / (distortion + 1e-30))

            # Validate: sample from the Beta distribution and check empirical distortion
            alpha = (d - 1) / 2.0
            # Sample on [-1, 1]
            samples_01 = np.random.default_rng(42).beta(alpha, alpha, size=100000)
            samples = 2.0 * samples_01 - 1.0

            # Quantize
            indices = quantize_array(samples, levels, boundaries)
            reconstructed = levels[indices]
            empirical_distortion = np.mean((samples - reconstructed) ** 2)

            # Per-coordinate MSE variance (should be low = uniform error)
            # Split samples into groups and check MSE consistency
            n_groups = 10
            group_size = len(samples) // n_groups
            group_mses = []
            for g in range(n_groups):
                s = samples[g * group_size : (g + 1) * group_size]
                idx = quantize_array(s, levels, boundaries)
                r = levels[idx]
                group_mses.append(np.mean((s - r) ** 2))
            mse_variance = np.var(group_mses)

            print(f"  {bits}-bit ({2**bits} levels):")
            print(f"    Theoretical distortion: {distortion:.6e}")
            print(f"    Empirical distortion:   {empirical_distortion:.6e}")
            print(f"    SNR: {snr:.2f} dB")
            print(
                f"    MSE variance across groups: {mse_variance:.2e} (low = uniform error)"
            )

    # Compare with naive uniform quantization (no rotation)
    print("\n" + "=" * 70)
    print("Comparison: Lloyd-Max (rotated) vs Uniform (naive) quantization")
    print("=" * 70)

    d = 4096
    bits = 4
    levels_lm, boundaries_lm = get_codebook(d, bits)

    # Naive uniform quantizer on [-1, 1]
    num_levels = 2**bits
    levels_uniform = np.linspace(-1, 1, num_levels)
    boundaries_uniform = np.zeros(num_levels + 1)
    boundaries_uniform[0] = -1.0
    boundaries_uniform[-1] = 1.0
    for i in range(1, num_levels):
        boundaries_uniform[i] = (levels_uniform[i - 1] + levels_uniform[i]) / 2.0

    # Sample from the Beta distribution (what rotation gives us)
    alpha = (d - 1) / 2.0
    samples_01 = np.random.default_rng(42).beta(alpha, alpha, size=100000)
    samples = 2.0 * samples_01 - 1.0

    # Lloyd-Max distortion
    idx_lm = quantize_array(samples, levels_lm, boundaries_lm)
    mse_lm = np.mean((samples - levels_lm[idx_lm]) ** 2)

    # Uniform distortion
    idx_uni = quantize_array(samples, levels_uniform, boundaries_uniform)
    mse_uni = np.mean((samples - levels_uniform[idx_uni]) ** 2)

    print(f"  d={d}, {bits}-bit:")
    print(f"    Lloyd-Max MSE: {mse_lm:.6e}")
    print(f"    Uniform MSE:   {mse_uni:.6e}")
    print(f"    Improvement:   {mse_uni / mse_lm:.2f}x")

    # Now simulate what happens WITHOUT rotation (Gaussian-like weights)
    print("\n  Simulating real weight distribution (Gaussian, no rotation):")
    gaussian_samples = np.random.default_rng(42).normal(0, 0.02, size=100000)
    # Clip to [-1, 1] for fair comparison
    gaussian_clipped = np.clip(gaussian_samples, -1, 1)

    idx_lm_gauss = quantize_array(gaussian_clipped, levels_lm, boundaries_lm)
    mse_lm_gauss = np.mean((gaussian_clipped - levels_lm[idx_lm_gauss]) ** 2)

    idx_uni_gauss = quantize_array(gaussian_clipped, levels_uniform, boundaries_uniform)
    mse_uni_gauss = np.mean((gaussian_clipped - levels_uniform[idx_uni_gauss]) ** 2)

    print(f"    Lloyd-Max (designed for Beta) on Gaussian: {mse_lm_gauss:.6e}")
    print(f"    Uniform on Gaussian:                      {mse_uni_gauss:.6e}")
    print(
        "    Key insight: Lloyd-Max codebook is OPTIMAL for the rotated distribution,"
    )
    print("    not for arbitrary weight distributions. The rotation is essential.")
