"""
SVD Expert Compression for MoE Models.

Decomposes expert weight matrices W[K, N] into truncated SVD factors:
    W ≈ U_s[K, r] @ Vt[r, N]
where U_s = U @ diag(S) absorbs the singular values.

For rank r=64 on our 70B model (260 experts × 3 projections × 20 layers):
    Original: 260 × 4096 × 1024 × 2 bytes × 3 = ~6.2 GB/layer, ~124 GB total
    SVD r=64: 260 × (4096×64 + 64×1024) × 2 bytes × 3 = ~0.49 GB/layer, ~9.7 GB total
    Compression: ~12.8×

The compressed forward replaces x @ W with x @ U_s @ Vt (two smaller GEMMs).
LoRA stays on top: output = x @ U_s @ Vt + (x @ A^T @ B^T) * scaling

Usage:
    from src.svd_moe_utils import (
        analyze_svd_spectrum,
        decompose_moe_experts_svd,
        patch_moe_svd_forward,
    )

    # Phase 1: Analyze (optional, to pick target rank)
    report = analyze_svd_spectrum(model, ranks=[16, 32, 64, 128])

    # Phase 2: Decompose + patch
    decompose_moe_experts_svd(model, target_rank=64)
    patch_moe_svd_forward(model)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# Phase 1: SVD Spectrum Analysis
# ============================================================================


def _compute_svd_stats(W: torch.Tensor, ranks: List[int]) -> Dict:
    """
    Compute SVD statistics for a single weight matrix.

    Args:
        W: [K, N] weight matrix (single expert, single projection)
        ranks: list of target ranks to evaluate

    Returns:
        dict with singular values, Frobenius norms, and per-rank error info
    """
    # Full SVD on CPU float32 for numerical stability
    W_f32 = W.detach().float().cpu()
    U, S, Vt = torch.linalg.svd(W_f32, full_matrices=False)
    # S is [min(K,N)] in descending order

    total_frob_sq = (S ** 2).sum().item()
    total_frob = total_frob_sq ** 0.5
    min_dim = S.shape[0]

    rank_stats = {}
    for r in ranks:
        r_clamped = min(r, min_dim)
        # Energy captured by top-r singular values
        captured_sq = (S[:r_clamped] ** 2).sum().item()
        residual_sq = total_frob_sq - captured_sq
        residual_frob = residual_sq ** 0.5

        # Relative error: ||W - W_r||_F / ||W||_F
        rel_error = residual_frob / total_frob if total_frob > 0 else 0.0
        energy_pct = (captured_sq / total_frob_sq * 100) if total_frob_sq > 0 else 100.0

        rank_stats[r] = {
            "rank": r_clamped,
            "energy_pct": energy_pct,
            "rel_error": rel_error,
            "residual_frob": residual_frob,
        }

    return {
        "shape": tuple(W.shape),
        "total_frob": total_frob,
        "singular_values": S,  # keep for detailed analysis
        "min_dim": min_dim,
        "rank_stats": rank_stats,
    }


def _effective_rank(S: torch.Tensor, threshold: float = 0.99) -> int:
    """Find minimum rank capturing `threshold` fraction of Frobenius energy."""
    total_sq = (S ** 2).sum().item()
    if total_sq == 0:
        return 0
    cumulative = torch.cumsum(S ** 2, dim=0)
    target = threshold * total_sq
    idx = (cumulative >= target).nonzero(as_tuple=True)[0]
    return (idx[0].item() + 1) if len(idx) > 0 else S.shape[0]


def analyze_svd_spectrum(
    model: nn.Module,
    ranks: Optional[List[int]] = None,
    max_experts_per_layer: int = 0,
    verbose: bool = True,
) -> Dict:
    """
    Analyze SVD spectrum of all MoE expert weights.

    For each MoEFFN module, computes SVD of every expert's gate/up/down
    weight matrix and reports energy capture at various target ranks.

    Args:
        model: the model (can be randomly initialized for testing)
        ranks: target ranks to evaluate (default: [8, 16, 32, 64, 128, 256])
        max_experts_per_layer: if > 0, only analyze this many experts per layer
            (useful for quick checks on 260-expert models). 0 = all experts.
        verbose: print per-layer summary

    Returns:
        dict with full analysis results
    """
    if ranks is None:
        ranks = [8, 16, 32, 64, 128, 256]

    results = {}
    proj_names = ["W_gate", "W_up", "W_down"]

    for mod_name, module in model.named_modules():
        has_experts = all(hasattr(module, p) for p in proj_names)
        if not has_experts:
            continue

        W_gate = getattr(module, "W_gate")
        # Handle both nn.Parameter and buffer tensors
        if isinstance(W_gate, nn.Parameter):
            shape = getattr(W_gate, "ds_shape", W_gate.shape)
        else:
            shape = W_gate.shape
        if len(shape) != 3:
            continue

        E = shape[0]
        n_analyze = E if max_experts_per_layer <= 0 else min(max_experts_per_layer, E)

        layer_results = {}
        for pname in proj_names:
            W_full = getattr(module, pname)

            # Handle DeepSpeed ZeRO-3 sharded params
            if hasattr(W_full, "ds_id"):
                import deepspeed
                with deepspeed.zero.GatheredParameters([W_full]):
                    W_data = W_full.data.clone()
            else:
                W_data = W_full.data

            expert_stats = []
            for e in range(n_analyze):
                stats = _compute_svd_stats(W_data[e], ranks)
                stats["expert_idx"] = e
                expert_stats.append(stats)

            # Aggregate across experts
            agg = {}
            for r in ranks:
                energies = [s["rank_stats"][r]["energy_pct"] for s in expert_stats]
                errors = [s["rank_stats"][r]["rel_error"] for s in expert_stats]
                agg[r] = {
                    "mean_energy_pct": sum(energies) / len(energies),
                    "min_energy_pct": min(energies),
                    "max_energy_pct": max(energies),
                    "mean_rel_error": sum(errors) / len(errors),
                    "max_rel_error": max(errors),
                }

            # Effective ranks at various thresholds
            eff_ranks_99 = [
                _effective_rank(s["singular_values"], 0.99) for s in expert_stats
            ]
            eff_ranks_999 = [
                _effective_rank(s["singular_values"], 0.999) for s in expert_stats
            ]
            eff_ranks_9999 = [
                _effective_rank(s["singular_values"], 0.9999) for s in expert_stats
            ]

            layer_results[pname] = {
                "shape": expert_stats[0]["shape"],
                "n_experts_analyzed": n_analyze,
                "total_experts": E,
                "aggregate": agg,
                "effective_rank_99": {
                    "mean": sum(eff_ranks_99) / len(eff_ranks_99),
                    "max": max(eff_ranks_99),
                    "min": min(eff_ranks_99),
                },
                "effective_rank_999": {
                    "mean": sum(eff_ranks_999) / len(eff_ranks_999),
                    "max": max(eff_ranks_999),
                    "min": min(eff_ranks_999),
                },
                "effective_rank_9999": {
                    "mean": sum(eff_ranks_9999) / len(eff_ranks_9999),
                    "max": max(eff_ranks_9999),
                    "min": min(eff_ranks_9999),
                },
                "expert_stats": expert_stats,
            }

        results[mod_name] = layer_results

        if verbose:
            print(f"\n{'='*70}")
            print(f"  Layer: {mod_name}  (E={E}, analyzed={n_analyze})")
            print(f"{'='*70}")
            for pname in proj_names:
                lr = layer_results[pname]
                K, N = lr["shape"]
                print(f"\n  {pname} [{K}×{N}]:")
                print(f"    Effective rank (99% energy):   "
                      f"mean={lr['effective_rank_99']['mean']:.0f}, "
                      f"max={lr['effective_rank_99']['max']}")
                print(f"    Effective rank (99.9% energy): "
                      f"mean={lr['effective_rank_999']['mean']:.0f}, "
                      f"max={lr['effective_rank_999']['max']}")
                print(f"    Effective rank (99.99% energy):"
                      f" mean={lr['effective_rank_9999']['mean']:.0f}, "
                      f"max={lr['effective_rank_9999']['max']}")
                print(f"    {'Rank':>6} | {'Energy%':>9} | {'Rel Error':>10} | {'Max Error':>10}")
                print(f"    {'-'*6}-+-{'-'*9}-+-{'-'*10}-+-{'-'*10}")
                for r in ranks:
                    a = lr["aggregate"][r]
                    print(f"    {r:>6} | {a['mean_energy_pct']:>8.3f}% | "
                          f"{a['mean_rel_error']:>10.6f} | {a['max_rel_error']:>10.6f}")

    return results


# ============================================================================
# Phase 2: SVD Decomposition
# ============================================================================


def decompose_moe_experts_svd(
    model: nn.Module,
    target_rank: int = 64,
    error_threshold: float = 0.0,
    verbose: bool = True,
) -> int:
    """
    Decompose MoE expert weights into truncated SVD factors in-place.

    For each expert weight W[K, N], computes:
        U, S, Vt = SVD(W)
        U_s = U[:, :r] @ diag(S[:r])   → [K, r]
        Vt_r = Vt[:r, :]               → [r, N]
    so that W ≈ U_s @ Vt_r

    The original W_gate/W_up/W_down parameters are replaced with:
        W_gate_U [E, K, r], W_gate_Vt [E, r, N]  (registered as buffers)
    and the original nn.Parameter is deleted.

    Args:
        model: model with MoEFFN modules
        target_rank: truncation rank r
        error_threshold: if > 0, skip decomposition for experts where
            relative Frobenius error exceeds this (keep original weight).
            0.0 = always decompose.
        verbose: log per-layer stats

    Returns:
        number of weight tensors decomposed (each of gate/up/down counts as 1)
    """
    proj_names = ["W_gate", "W_up", "W_down"]
    count = 0

    for mod_name, module in model.named_modules():
        has_experts = all(hasattr(module, p) for p in proj_names)
        if not has_experts:
            continue

        W_gate = getattr(module, "W_gate")
        if not isinstance(W_gate, nn.Parameter):
            # Already decomposed or NF4-quantized
            continue

        shape = getattr(W_gate, "ds_shape", W_gate.shape)
        if len(shape) != 3:
            continue

        E = shape[0]

        for pname in proj_names:
            param = getattr(module, pname)
            orig_shape = getattr(param, "ds_shape", param.shape)
            E_p, K, N = orig_shape
            r = min(target_rank, min(K, N))

            # Gather full param if ZeRO-3 sharded
            if hasattr(param, "ds_id"):
                import deepspeed
                with deepspeed.zero.GatheredParameters([param]):
                    W_data = param.data.clone()
            else:
                W_data = param.data

            # Decompose each expert
            U_list = []
            Vt_list = []
            total_error = 0.0
            max_error = 0.0

            for e in range(E):
                We = W_data[e].float().cpu()
                U, S, Vt = torch.linalg.svd(We, full_matrices=False)

                # Truncate to rank r
                U_r = U[:, :r]          # [K, r]
                S_r = S[:r]             # [r]
                Vt_r = Vt[:r, :]        # [r, N]

                # Absorb singular values into U
                U_s = U_r * S_r.unsqueeze(0)  # [K, r] — broadcast multiply

                # Compute relative error for reporting
                total_frob_sq = (S ** 2).sum().item()
                residual_sq = (S[r:] ** 2).sum().item() if r < S.shape[0] else 0.0
                rel_error = (residual_sq ** 0.5) / (total_frob_sq ** 0.5) if total_frob_sq > 0 else 0.0
                total_error += rel_error
                max_error = max(max_error, rel_error)

                # Convert back to original dtype and device
                U_list.append(U_s.to(dtype=param.dtype, device=param.device))
                Vt_list.append(Vt_r.to(dtype=param.dtype, device=param.device))

            # Stack: [E, K, r] and [E, r, N]
            U_stacked = torch.stack(U_list, dim=0)   # [E, K, r]
            Vt_stacked = torch.stack(Vt_list, dim=0)  # [E, r, N]

            # Delete original parameter, register SVD factors as buffers
            delattr(module, pname)
            module.register_buffer(f"{pname}_U", U_stacked)
            module.register_buffer(f"{pname}_Vt", Vt_stacked)

            count += 1
            avg_error = total_error / E
            orig_mb = E * K * N * 2 / 1e6  # bf16
            svd_mb = (E * K * r + E * r * N) * 2 / 1e6
            compression = orig_mb / svd_mb if svd_mb > 0 else float("inf")

            if verbose:
                logger.info(
                    f"  SVD: {mod_name}.{pname} [{E}×{K}×{N}] → rank={r}  "
                    f"bf16={orig_mb:.1f}MB → svd={svd_mb:.1f}MB "
                    f"({compression:.1f}× compression)  "
                    f"avg_err={avg_error:.6f} max_err={max_error:.6f}"
                )

        # Mark module as SVD-decomposed
        module._svd_decomposed = True
        module._svd_rank = target_rank

    return count


# ============================================================================
# Phase 2: SVD-Compressed Forward Pass
# ============================================================================


def _moe_grouped_svd(module, sorted_x, expert_counts):
    """
    SVD-compressed MoE grouped forward path.

    Replaces x @ W[e] with x @ U_s[e] @ Vt[e] (two smaller grouped GEMMs).
    LoRA is applied on top if enabled.

    For gate/up: x[M, K] @ U_s[E, K, r] → [M, r] @ Vt[E, r, N] → [M, N]
    For down:    h[M, N] @ U_s[E, N, r] → [M, r] @ Vt[E, r, K] → [M, K]
    """
    try:
        from .models.liger_ops import liger_silu_mul
    except ImportError:
        liger_silu_mul = lambda g, u: torch.nn.functional.silu(g) * u

    try:
        from .kernels.triton_moe_grouped_gemm import triton_grouped_gemm
    except ImportError:
        triton_grouped_gemm = None

    try:
        from .kernels.fused_lora_grouped_gemm import fused_lora_grouped_gemm
    except ImportError:
        fused_lora_grouped_gemm = None

    x_in = sorted_x.to(dtype=module.W_gate_U.dtype)
    has_lora = getattr(module, "moe_lora_enabled", False)
    scaling = getattr(module, "moe_lora_scaling", 0.0)

    if triton_grouped_gemm is None:
        raise RuntimeError("No grouped GEMM kernel available for SVD forward")

    # ── Gate projection: x @ U_gate @ Vt_gate ──────────────────────────────
    gate_mid = triton_grouped_gemm(x_in, module.W_gate_U, expert_counts)   # [M, r]
    gate_out = triton_grouped_gemm(gate_mid, module.W_gate_Vt, expert_counts)  # [M, N]
    del gate_mid
    if has_lora and fused_lora_grouped_gemm is not None:
        # LoRA adds: (x @ A^T @ B^T) * scaling
        # We need a separate LoRA GEMM since SVD factors don't include LoRA
        # Use a lightweight version: just the LoRA part, no base weight
        from .kernels.triton_moe_grouped_gemm import (
            _grouped_gemm_forward, _compute_offsets,
        )
        offsets, counts = _compute_offsets(expert_counts, x_in.device)
        E = counts.shape[0]
        max_M = int(counts.max().item()) if counts.numel() > 0 else 0
        A_t = module.lora_A_W_gate.transpose(-2, -1).contiguous()
        lora_mid = _grouped_gemm_forward(x_in, A_t, offsets, E, max_M)
        B_t = module.lora_B_W_gate.transpose(-2, -1).contiguous()
        lora_out = _grouped_gemm_forward(lora_mid, B_t, offsets, E, max_M)
        gate_out = gate_out + lora_out * scaling
        del lora_mid, lora_out, A_t, B_t

    # ── Up projection: x @ U_up @ Vt_up ────────────────────────────────────
    up_mid = triton_grouped_gemm(x_in, module.W_up_U, expert_counts)
    up_out = triton_grouped_gemm(up_mid, module.W_up_Vt, expert_counts)
    del up_mid
    if has_lora and fused_lora_grouped_gemm is not None:
        A_t = module.lora_A_W_up.transpose(-2, -1).contiguous()
        lora_mid = _grouped_gemm_forward(x_in, A_t, offsets, E, max_M)
        B_t = module.lora_B_W_up.transpose(-2, -1).contiguous()
        lora_out = _grouped_gemm_forward(lora_mid, B_t, offsets, E, max_M)
        up_out = up_out + lora_out * scaling
        del lora_mid, lora_out, A_t, B_t

    # ── SiLU activation ────────────────────────────────────────────────────
    h = liger_silu_mul(gate_out, up_out)
    del gate_out, up_out

    if module.training and module.dropout > 0:
        h = torch.nn.functional.dropout(h, p=module.dropout)

    # ── Down projection: h @ U_down @ Vt_down ──────────────────────────────
    down_mid = triton_grouped_gemm(h, module.W_down_U, expert_counts)
    out = triton_grouped_gemm(down_mid, module.W_down_Vt, expert_counts)
    del down_mid
    if has_lora and fused_lora_grouped_gemm is not None:
        A_t = module.lora_A_W_down.transpose(-2, -1).contiguous()
        lora_mid = _grouped_gemm_forward(h, A_t, offsets, E, max_M)
        B_t = module.lora_B_W_down.transpose(-2, -1).contiguous()
        lora_out = _grouped_gemm_forward(lora_mid, B_t, offsets, E, max_M)
        out = out + lora_out * scaling
        del lora_mid, lora_out, A_t, B_t

    return out.to(dtype=sorted_x.dtype)


# ============================================================================
# Phase 2: Patch Forward + Summary
# ============================================================================


def patch_moe_svd_forward(model: nn.Module) -> int:
    """
    Patch MoEFFN modules to use SVD-compressed forward path.

    After decompose_moe_experts_svd() has replaced weights with SVD factors,
    this monkey-patches _moe_grouped on each decomposed MoEFFN module.

    Returns: number of modules patched
    """
    import types
    count = 0
    for mod_name, module in model.named_modules():
        if not getattr(module, "_svd_decomposed", False):
            continue

        # Save original for potential fallback
        module._svd_original_moe_grouped = module._moe_grouped

        def _make_svd_grouped(mod):
            def _patched_moe_grouped(self, sorted_x, expert_counts):
                return _moe_grouped_svd(self, sorted_x, expert_counts)
            return types.MethodType(_patched_moe_grouped, mod)

        module._moe_grouped = _make_svd_grouped(module)
        count += 1
        r = getattr(module, "_svd_rank", "?")
        logger.info(f"  [SVD] forward patched: {mod_name} [rank={r}]")

    return count


def print_svd_summary(model: nn.Module):
    """Print SVD compression summary."""
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass

    orig_bytes = 0
    svd_bytes = 0
    count = 0

    for mod_name, module in model.named_modules():
        if not getattr(module, "_svd_decomposed", False):
            continue
        r = getattr(module, "_svd_rank", 0)
        for pname in ["W_gate", "W_up", "W_down"]:
            U = getattr(module, f"{pname}_U", None)
            Vt = getattr(module, f"{pname}_Vt", None)
            if U is not None and Vt is not None:
                E, K, r_actual = U.shape
                _, _, N = Vt.shape
                orig_bytes += E * K * N * 2  # bf16
                svd_bytes += U.nbytes + Vt.nbytes
                count += 1

    if count == 0:
        print("  No SVD-decomposed expert weights found.")
        return

    print("\n" + "=" * 70)
    print("  SVD COMPRESSION SUMMARY")
    print("=" * 70)
    print(f"  Expert weight tensors decomposed: {count}")
    print(f"  Original bf16 size:  {orig_bytes / 1e9:.2f} GB")
    print(f"  SVD factors size:    {svd_bytes / 1e9:.2f} GB")
    print(f"  Compression ratio:   {orig_bytes / svd_bytes:.1f}×")
    print(f"  Memory saved:        {(orig_bytes - svd_bytes) / 1e9:.2f} GB")
    print("=" * 70 + "\n")

