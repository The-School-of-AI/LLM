"""
Benchmark suite for 70B A100 training speedup experiments.

Covers all 6 experiments from the Speedup 70B A100 Training plan:
  1. Fused Add+RMSNorm (in-kernel residual add vs separate PyTorch add)
  2. torch.compile effect on elementwise chains
  3. Fused delta entrance profiling
  4. ZeRO-3 bucket size tuning (requires multi-GPU, prints guidance)
  5. Fused CE chunk size / BLOCK_SIZE tuning
  6. gc.collect / empty_cache overhead measurement

Usage:
    python bench_speedup_experiments.py                    # Run all experiments
    python bench_speedup_experiments.py --exp 1            # Run experiment 1 only
    python bench_speedup_experiments.py --exp 1 2 5        # Run experiments 1, 2, 5
"""

import argparse
import time
import sys
import os
import gc

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _timer(fn, warmup=10, iters=100, sync=True):
    """Time a function with warmup and return average ms."""
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Fused Add+RMSNorm
# ══════════════════════════════════════════════════════════════════════════════

def bench_exp1_fused_add_rmsnorm():
    """
    Compare:
      A) Separate: x = x + residual; y = rmsnorm(x)     (2 kernels, 1 intermediate)
      B) Fused:    y = rmsnorm(x, residual=residual)     (1 kernel, no intermediate)
    """
    from src.kernels.triton_rmsnorm import triton_rmsnorm, LigerRMSNormFunction

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Fused Add + RMSNorm")
    print("=" * 70)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = 1e-6

    for label, (B, T, D) in [
        ("1B  (B=1, T=4096, D=2048)", (1, 4096, 2048)),
        ("3B  (B=1, T=4096, D=3072)", (1, 4096, 3072)),
        ("70B (B=1, T=4096, D=4096)", (1, 4096, 4096)),
        ("70B (B=2, T=4096, D=4096)", (2, 4096, 4096)),
    ]:
        N = B * T
        x = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
        residual = torch.randn(N, D, device=device, dtype=dtype)
        weight = torch.ones(D, device=device, dtype=dtype)

        # A) Separate: add then norm
        def separate():
            x_data = x.data
            combined = x_data + residual
            return LigerRMSNormFunction.apply(combined, weight, eps)

        # B) Fused: add inside kernel
        def fused():
            return LigerRMSNormFunction.apply(x.data, weight, eps, residual)

        t_sep = _timer(separate)
        t_fused = _timer(fused)
        speedup = t_sep / t_fused

        print(f"  {label}: separate={t_sep:.3f}ms  fused={t_fused:.3f}ms  speedup={speedup:.2f}x")

        # Memory comparison
        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            _ = separate()
        torch.cuda.synchronize()
        mem_sep = torch.cuda.max_memory_allocated() / 1e6

        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            _ = fused()
        torch.cuda.synchronize()
        mem_fused = torch.cuda.max_memory_allocated() / 1e6

        print(f"           peak_mem: separate={mem_sep:.1f}MB  fused={mem_fused:.1f}MB  saved={mem_sep - mem_fused:.1f}MB")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2: torch.compile effect on elementwise chains
# ══════════════════════════════════════════════════════════════════════════════

def bench_exp2_torch_compile():
    """
    Measure speedup from torch.compile on representative elementwise chains
    that occur between matmuls in the model (sigmoid, gating, residual adds).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: torch.compile on Elementwise Chains")
    print("=" * 70)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    D = 4096
    for label, N in [("N=4096", 4096), ("N=16384", 16384), ("N=32768", 32768)]:
        x = torch.randn(N, D, device=device, dtype=dtype)
        gate = torch.randn(N, 1, device=device, dtype=dtype)
        residual = torch.randn(N, D, device=device, dtype=dtype)
        weight = torch.ones(D, device=device, dtype=dtype)

        # Typical model elementwise chain: sigmoid gating + residual + rmsnorm-like
        def chain_eager(x, gate, residual, weight):
            g = torch.sigmoid(gate)
            x = x * g + residual
            x_f = x.float()
            var = x_f.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var.to(x.dtype) + 1e-6)
            return weight * x

        try:
            chain_compiled = torch.compile(chain_eager, mode="reduce-overhead")

            t_eager = _timer(lambda: chain_eager(x, gate, residual, weight))

            # Warmup compile (extra warmup needed for compilation)
            for _ in range(5):
                chain_compiled(x, gate, residual, weight)
            torch.cuda.synchronize()

            t_compiled = _timer(lambda: chain_compiled(x, gate, residual, weight))
            speedup = t_eager / t_compiled

            print(f"  {label} D={D}: eager={t_eager:.3f}ms  compiled={t_compiled:.3f}ms  speedup={speedup:.2f}x")
        except Exception as e:
            print(f"  {label} D={D}: torch.compile failed: {e}")

    # Test with a mini model-like module
    print("\n  --- Module-level compile (simulates decoder sublayer elementwise ops) ---")

    class ElemChain(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.norm_w = torch.nn.Parameter(torch.ones(d))
            self.gate_proj = torch.nn.Linear(d, 1, bias=False)

        def forward(self, x, residual):
            g = torch.sigmoid(self.gate_proj(x))
            x = x * g + residual
            x_f = x.float()
            var = x_f.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var.to(x.dtype) + 1e-6)
            return self.norm_w * x

    for label, N in [("N=4096", 4096), ("N=32768", 32768)]:
        mod = ElemChain(D).to(device=device, dtype=dtype)
        x = torch.randn(N, D, device=device, dtype=dtype)
        residual = torch.randn(N, D, device=device, dtype=dtype)

        t_eager = _timer(lambda: mod(x, residual))

        try:
            mod_c = torch.compile(mod, mode="reduce-overhead")
            for _ in range(5):
                mod_c(x, residual)
            torch.cuda.synchronize()

            t_compiled = _timer(lambda: mod_c(x, residual))
            speedup = t_eager / t_compiled
            print(f"  {label} D={D}: eager={t_eager:.3f}ms  compiled={t_compiled:.3f}ms  speedup={speedup:.2f}x")
        except Exception as e:
            print(f"  {label} D={D}: torch.compile failed: {e}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 3: Fused Delta Entrance profiling
# ══════════════════════════════════════════════════════════════════════════════

def bench_exp3_delta_entrance():
    """
    Compare fused delta entrance (conv+SiLU+L2Norm+RoPE) vs unfused PyTorch ops.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Fused Delta Entrance (Conv + SiLU + L2Norm + RoPE)")
    print("=" * 70)

    try:
        from src.kernels.triton_delta_entrance import fused_delta_entrance, pytorch_unfused_exact
    except ImportError as e:
        print(f"  SKIPPED: Cannot import delta entrance kernels: {e}")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16

    for label, (B, T, H, D) in [
        ("70B (B=1, T=4096, H=32, D=128)", (1, 4096, 32, 128)),
        ("70B (B=1, T=4096, H=64, D=64)",  (1, 4096, 64, 64)),
    ]:
        K = 4  # conv kernel size
        key_dim = H * D

        q_in = torch.randn(B, T, key_dim, device=device, dtype=dtype)
        k_in = torch.randn(B, T, key_dim, device=device, dtype=dtype)
        v_in = torch.randn(B, T, key_dim, device=device, dtype=dtype)

        wq = torch.randn(1, 1, K, device=device, dtype=dtype)
        wk = torch.randn(1, 1, K, device=device, dtype=dtype)
        wv = torch.randn(1, 1, K, device=device, dtype=dtype)
        bq = torch.zeros(key_dim, device=device, dtype=dtype)
        bk = torch.zeros(key_dim, device=device, dtype=dtype)
        bv = torch.zeros(key_dim, device=device, dtype=dtype)

        cos = torch.randn(T, D // 2, device=device, dtype=dtype)
        sin = torch.randn(T, D // 2, device=device, dtype=dtype)

        def run_unfused():
            return pytorch_unfused_exact(q_in, k_in, v_in, wq, wk, wv, bq, bk, bv, cos, sin, H, D)

        def run_fused():
            return fused_delta_entrance(q_in, k_in, v_in, wq, wk, wv, bq, bk, bv, cos, sin, None)

        try:
            t_unfused = _timer(run_unfused)
            t_fused = _timer(run_fused)
            speedup = t_unfused / t_fused
            print(f"  {label}:")
            print(f"    unfused={t_unfused:.3f}ms  fused={t_fused:.3f}ms  speedup={speedup:.2f}x")
        except Exception as e:
            print(f"  {label}: FAILED: {e}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 4: ZeRO-3 bucket size tuning guidance
# ══════════════════════════════════════════════════════════════════════════════

def bench_exp4_zero3_guidance():
    """
    Print ZeRO-3 bucket size tuning guidance and generate variant configs.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: ZeRO-3 Communication Tuning")
    print("=" * 70)

    base_config = {
        "reduce_bucket_size": 50_000_000,
        "allgather_bucket_size": 50_000_000,
        "stage3_prefetch_bucket_size": 50_000_000,
        "stage3_param_persistence_threshold": 1_000_000,
    }

    variants = {
        "baseline (50M buckets)": base_config,
        "large buckets (200M)": {
            **base_config,
            "reduce_bucket_size": 200_000_000,
            "allgather_bucket_size": 200_000_000,
            "stage3_prefetch_bucket_size": 200_000_000,
        },
        "small buckets (10M)": {
            **base_config,
            "reduce_bucket_size": 10_000_000,
            "allgather_bucket_size": 10_000_000,
            "stage3_prefetch_bucket_size": 10_000_000,
        },
        "large persistence threshold (10M)": {
            **base_config,
            "stage3_param_persistence_threshold": 10_000_000,
        },
    }

    import json

    ds_base_path = os.path.join(os.path.dirname(__file__), "..", "deepspeed", "zero-3-70b-moe-lora-bs8.json")
    if os.path.exists(ds_base_path):
        with open(ds_base_path) as f:
            ds_base = json.load(f)
    else:
        print(f"  WARNING: Base config not found at {ds_base_path}")
        ds_base = None

    print("\n  Generated ZeRO-3 variant configs for benchmarking:")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "deepspeed")

    for name, overrides in variants.items():
        print(f"\n  --- {name} ---")
        for k, v in overrides.items():
            print(f"    {k}: {v:,}")

        if ds_base is not None:
            variant_cfg = json.loads(json.dumps(ds_base))
            variant_cfg["zero_optimization"].update(overrides)
            fname = f"zero-3-70b-exp4-{name.replace(' ', '-').replace('(', '').replace(')', '')}.json"
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "w") as f:
                json.dump(variant_cfg, f, indent=2)
            print(f"    Saved: {fpath}")

    print("\n  To benchmark, run training with each config and compare tok/s:")
    print("    deepspeed main.py --config <config.yaml with deepspeed config_path pointing to each variant>")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 5: Fused CE chunk size and BLOCK_SIZE tuning
# ══════════════════════════════════════════════════════════════════════════════

def bench_exp5_fused_ce_tuning():
    """
    Benchmark fused linear + cross entropy with different chunk sizes and BLOCK_SIZE.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Fused Linear + Cross Entropy Tuning")
    print("=" * 70)

    from src.kernels.triton_cross_entropy import FusedLinearCrossEntropyLoss
    import src.kernels.triton_cross_entropy as ce_module

    device = torch.device("cuda")
    dtype = torch.bfloat16

    V = 131072  # 2^17 vocab
    D = 4096    # hidden dim for 70B

    for label, (B, T) in [
        ("B=1 T=4096", (1, 4096)),
        ("B=1 T=2048", (1, 2048)),
    ]:
        N = B * T
        hidden = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
        lm_head_weight = torch.randn(V, D, device=device, dtype=dtype, requires_grad=True)
        targets = torch.randint(0, V, (N,), device=device)

        # Baseline: unfused (matmul + F.cross_entropy)
        def unfused():
            with torch.no_grad():
                logits = hidden.detach() @ lm_head_weight.detach().T
                return F.cross_entropy(logits.float(), targets)

        t_unfused = _timer(unfused, warmup=3, iters=10)
        print(f"\n  {label}, V={V}, D={D}:")
        print(f"    Unfused (matmul+CE): {t_unfused:.1f}ms")

        # Test different chunk sizes
        for chunk_gb in [0.5, 2.0, 4.0, 8.0]:
            fused_ce = FusedLinearCrossEntropyLoss(max_chunk_gb=chunk_gb)

            def fused():
                h = hidden.detach().requires_grad_(True)
                w = lm_head_weight.detach().requires_grad_(True)
                return fused_ce(h, w, targets)

            try:
                t_fused = _timer(fused, warmup=3, iters=10)
                speedup = t_unfused / t_fused if t_fused > 0 else float("inf")
                print(f"    Fused (chunk={chunk_gb}GB): {t_fused:.1f}ms  speedup={speedup:.2f}x")
            except Exception as e:
                print(f"    Fused (chunk={chunk_gb}GB): FAILED: {e}")

        # Test different BLOCK_SIZE values
        orig_block_size = ce_module._MAX_FUSED_SIZE
        for block_size in [4096, 8192, 16384, 32768]:
            ce_module._MAX_FUSED_SIZE = block_size
            fused_ce = FusedLinearCrossEntropyLoss(max_chunk_gb=8.0)

            def fused():
                h = hidden.detach().requires_grad_(True)
                w = lm_head_weight.detach().requires_grad_(True)
                return fused_ce(h, w, targets)

            try:
                t_fused = _timer(fused, warmup=3, iters=10)
                speedup = t_unfused / t_fused if t_fused > 0 else float("inf")
                print(f"    Fused (BLOCK_SIZE={block_size}): {t_fused:.1f}ms  speedup={speedup:.2f}x")
            except Exception as e:
                print(f"    Fused (BLOCK_SIZE={block_size}): FAILED: {e}")

        ce_module._MAX_FUSED_SIZE = orig_block_size

    print()


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 6: gc.collect / empty_cache overhead
# ══════════════════════════════════════════════════════════════════════════════

def bench_exp6_cleanup_overhead():
    """
    Measure the overhead of gc.collect() and torch.cuda.empty_cache() per step.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Per-Step Cleanup Overhead")
    print("=" * 70)

    device = torch.device("cuda")

    # Simulate allocations to give gc and allocator something to work with
    tensors = [torch.randn(1024, 1024, device=device) for _ in range(20)]
    del tensors
    torch.cuda.synchronize()

    # Measure gc.collect alone
    iters = 100
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        gc.collect()
    t_gc = (time.perf_counter() - t0) / iters * 1000.0

    # Measure empty_cache alone
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t_cache = (time.perf_counter() - t0) / iters * 1000.0

    # Measure cuda.synchronize alone
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.cuda.synchronize()
    t_sync = (time.perf_counter() - t0) / iters * 1000.0

    # All three together (what happens every step currently)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    t_all = (time.perf_counter() - t0) / iters * 1000.0

    print(f"\n  Per-call overhead (avg of {iters} calls):")
    print(f"    cuda.synchronize():   {t_sync:.3f}ms")
    print(f"    gc.collect():         {t_gc:.3f}ms")
    print(f"    cuda.empty_cache():   {t_cache:.3f}ms")
    print(f"    All three combined:   {t_all:.3f}ms")
    print()

    # With actual tensor churn (more realistic)
    print("  With tensor allocation churn (simulating training step):")
    D = 4096

    def simulate_step_with_cleanup():
        ts = [torch.randn(4096, D, device=device, dtype=torch.bfloat16) for _ in range(5)]
        result = ts[0] + ts[1] * ts[2]
        del ts
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def simulate_step_no_cleanup():
        ts = [torch.randn(4096, D, device=device, dtype=torch.bfloat16) for _ in range(5)]
        result = ts[0] + ts[1] * ts[2]
        del ts
        return result

    t_with = _timer(simulate_step_with_cleanup, warmup=5, iters=50)
    t_without = _timer(simulate_step_no_cleanup, warmup=5, iters=50)
    overhead = t_with - t_without

    print(f"    With cleanup:     {t_with:.3f}ms/step")
    print(f"    Without cleanup:  {t_without:.3f}ms/step")
    print(f"    Cleanup overhead: {overhead:.3f}ms/step")

    if overhead > 0:
        # Extrapolate to training
        step_time_s = 11.6  # baseline step time
        overhead_s = overhead / 1000
        pct = overhead_s / step_time_s * 100
        print(f"\n    At {step_time_s}s/step baseline, cleanup overhead = {overhead_s*1000:.1f}ms = {pct:.2f}% of step time")
        print(f"    Reducing to every 10 steps saves ~{pct * 0.9:.2f}% of total training time")

    print("\n  Recommendation:")
    print("    Set env vars to reduce frequency:")
    print("      T19_STEP_GC_COLLECT=0       # disable per-step gc.collect")
    print("      T19_STEP_EMPTY_CACHE=0      # disable per-step empty_cache")
    print("      T19_STEP_CUDA_SYNC=0        # disable per-step sync (if stable)")
    print("    Or modify train.py to run cleanup every N steps instead.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Speedup experiments benchmark")
    parser.add_argument("--exp", type=int, nargs="*", default=None,
                        help="Experiment numbers to run (1-6). Default: all")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. All experiments require GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu_name} (sm_{cap[0]}{cap[1]})")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    experiments = {
        1: ("Fused Add+RMSNorm", bench_exp1_fused_add_rmsnorm),
        2: ("torch.compile", bench_exp2_torch_compile),
        3: ("Fused Delta Entrance", bench_exp3_delta_entrance),
        4: ("ZeRO-3 Tuning", bench_exp4_zero3_guidance),
        5: ("Fused CE Tuning", bench_exp5_fused_ce_tuning),
        6: ("Cleanup Overhead", bench_exp6_cleanup_overhead),
    }

    to_run = args.exp if args.exp else list(experiments.keys())

    for exp_num in sorted(to_run):
        if exp_num not in experiments:
            print(f"\nWARNING: Unknown experiment {exp_num}, skipping")
            continue
        name, fn = experiments[exp_num]
        try:
            fn()
        except Exception as e:
            print(f"\nEXPERIMENT {exp_num} ({name}) FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All experiments complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
