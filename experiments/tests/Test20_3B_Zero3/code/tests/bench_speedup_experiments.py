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

    Falls back to backend="eager" or "aot_eager" if inductor (Triton) is unavailable
    due to version mismatch.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: torch.compile on Elementwise Chains")
    print("=" * 70)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Detect which backends are available
    available_backends = []
    _test_fn = lambda x: x + 1
    _test_t = torch.tensor(1.0, device=device)
    for backend in ["inductor", "aot_eager", "eager"]:
        try:
            _c = torch.compile(_test_fn, backend=backend)
            _c(_test_t)
            available_backends.append(backend)
        except Exception:
            pass

    if not available_backends:
        print("  ERROR: No torch.compile backends available. Skipping.")
        print("  FIX: Install triton-nightly to match PyTorch 2.7:")
        print("    pip install -U triton-nightly")
        return

    backend = available_backends[0]
    mode = "reduce-overhead" if backend == "inductor" else None
    print(f"  Using backend: {backend} (available: {available_backends})")
    if backend != "inductor":
        print(f"  NOTE: 'inductor' failed (likely Triton version mismatch).")
        print(f"  FIX: pip install -U triton-nightly   (to match PyTorch {torch.__version__})")
        print(f"  Using '{backend}' for now — inductor would give better speedups.")

    D = 4096
    for label, N in [("N=4096", 4096), ("N=16384", 16384), ("N=32768", 32768)]:
        x = torch.randn(N, D, device=device, dtype=dtype)
        gate = torch.randn(N, 1, device=device, dtype=dtype)
        residual = torch.randn(N, D, device=device, dtype=dtype)
        weight = torch.ones(D, device=device, dtype=dtype)

        def chain_eager(x, gate, residual, weight):
            g = torch.sigmoid(gate)
            x = x * g + residual
            x_f = x.float()
            var = x_f.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var.to(x.dtype) + 1e-6)
            return weight * x

        try:
            compile_kwargs = {"backend": backend}
            if mode:
                compile_kwargs["mode"] = mode
            chain_compiled = torch.compile(chain_eager, **compile_kwargs)

            t_eager = _timer(lambda: chain_eager(x, gate, residual, weight))

            for _ in range(5):
                chain_compiled(x, gate, residual, weight)
            torch.cuda.synchronize()

            t_compiled = _timer(lambda: chain_compiled(x, gate, residual, weight))
            speedup = t_eager / t_compiled

            print(f"  {label} D={D}: eager={t_eager:.3f}ms  compiled={t_compiled:.3f}ms  speedup={speedup:.2f}x")
        except Exception as e:
            print(f"  {label} D={D}: torch.compile failed: {e}")

    print("\n  --- Module-level compile ---")

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
            compile_kwargs = {"backend": backend}
            if mode:
                compile_kwargs["mode"] = mode
            mod_c = torch.compile(mod, **compile_kwargs)
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
    Conv weights are depthwise: shape (C, 1, K) where C = H * D, K = 4.
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

        # Depthwise conv weights: (C, 1, K) for groups=C convolution
        wq = torch.randn(key_dim, 1, K, device=device, dtype=dtype)
        wk = torch.randn(key_dim, 1, K, device=device, dtype=dtype)
        wv = torch.randn(key_dim, 1, K, device=device, dtype=dtype)
        bq = torch.zeros(key_dim, device=device, dtype=dtype)
        bk = torch.zeros(key_dim, device=device, dtype=dtype)
        bv = torch.zeros(key_dim, device=device, dtype=dtype)

        cos = torch.randn(T, D // 2, device=device, dtype=dtype)
        sin = torch.randn(T, D // 2, device=device, dtype=dtype)

        def run_unfused():
            return pytorch_unfused_exact(q_in, k_in, v_in, wq, wk, wv, bq, bk, bv, cos, sin, None)

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

    # Try multiple paths for the base deepspeed config
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.join(_script_dir, "..", "deepspeed", "zero-3-70b-moe-lora-bs8.json"),
        os.path.join(_script_dir, "..", "..", "deepspeed", "zero-3-70b-moe-lora-bs8.json"),
    ]
    ds_base_path = None
    for _p in _candidates:
        if os.path.exists(_p):
            ds_base_path = _p
            break
    if ds_base_path:
        with open(ds_base_path) as f:
            ds_base = json.load(f)
    else:
        print(f"  WARNING: Base config not found. Searched: {_candidates}")
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
    Benchmark fused linear + cross entropy: both speed and PEAK MEMORY.

    The main value of fused CE is MEMORY savings (avoids materializing the
    [B*T, vocab] logits tensor). For V=131072, D=4096, N=4096:
      - Unfused logits: 4096 * 131072 * 4 bytes = 2.0 GB
      - Fused: processes in chunks, peak is just one chunk

    Speed-wise fused CE is slower (chunked matmul + Triton CE vs single cuBLAS + F.cross_entropy),
    but the memory savings enable training that would otherwise OOM.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Fused Linear + Cross Entropy (Speed + Memory)")
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

        logits_size_gb = N * V * 4 / 1e9
        print(f"\n  {label}, V={V}, D={D}:")
        print(f"    Unfused logits tensor: {N}x{V} x 4 bytes = {logits_size_gb:.1f} GB")

        hidden = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
        lm_head_weight = torch.randn(V, D, device=device, dtype=dtype, requires_grad=True)
        targets = torch.randint(0, V, (N,), device=device)

        # --- Speed comparison ---
        def unfused():
            with torch.no_grad():
                logits = hidden.detach() @ lm_head_weight.detach().T
                return F.cross_entropy(logits.float(), targets)

        t_unfused = _timer(unfused, warmup=3, iters=10)
        print(f"    Unfused speed: {t_unfused:.1f}ms")

        for chunk_gb in [0.5, 2.0, 8.0]:
            fused_ce = FusedLinearCrossEntropyLoss(max_chunk_gb=chunk_gb)
            def fused(ce=fused_ce):
                h = hidden.detach().requires_grad_(True)
                w = lm_head_weight.detach().requires_grad_(True)
                return ce(h, w, targets)
            try:
                t_fused = _timer(fused, warmup=3, iters=10)
                print(f"    Fused (chunk={chunk_gb}GB): {t_fused:.1f}ms  ({t_unfused/t_fused:.2f}x)")
            except Exception as e:
                print(f"    Fused (chunk={chunk_gb}GB): FAILED: {e}")

        # --- Memory comparison (the real benefit) ---
        print(f"\n    Peak memory comparison:")

        # Unfused memory
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        _base_mem = torch.cuda.memory_allocated() / 1e9
        for _ in range(3):
            _ = unfused()
        torch.cuda.synchronize()
        mem_unfused = (torch.cuda.max_memory_allocated() / 1e9) - _base_mem

        # Fused memory (chunk=0.5GB)
        fused_ce_small = FusedLinearCrossEntropyLoss(max_chunk_gb=0.5)
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        _base_mem = torch.cuda.memory_allocated() / 1e9
        for _ in range(3):
            h = hidden.detach().requires_grad_(True)
            w = lm_head_weight.detach().requires_grad_(True)
            _ = fused_ce_small(h, w, targets)
        torch.cuda.synchronize()
        mem_fused = (torch.cuda.max_memory_allocated() / 1e9) - _base_mem

        saved_gb = mem_unfused - mem_fused
        print(f"      Unfused peak: {mem_unfused:.2f} GB")
        print(f"      Fused peak:   {mem_fused:.2f} GB")
        print(f"      SAVED:        {saved_gb:.2f} GB  ({saved_gb/mem_unfused*100:.0f}% reduction)")
        print(f"      (Fused CE is slower but saves ~{logits_size_gb:.1f}GB by not materializing logits)")

        # BLOCK_SIZE tuning (quick)
        print(f"\n    BLOCK_SIZE tuning (best of chunk sizes):")
        orig_block_size = ce_module._MAX_FUSED_SIZE
        for block_size in [4096, 8192, 16384, 32768]:
            ce_module._MAX_FUSED_SIZE = block_size
            fused_ce_bs = FusedLinearCrossEntropyLoss(max_chunk_gb=8.0)
            def fused_bs(ce=fused_ce_bs):
                h = hidden.detach().requires_grad_(True)
                w = lm_head_weight.detach().requires_grad_(True)
                return ce(h, w, targets)
            try:
                t_fused = _timer(fused_bs, warmup=3, iters=10)
                print(f"      BLOCK_SIZE={block_size:>5}: {t_fused:.1f}ms")
            except Exception as e:
                print(f"      BLOCK_SIZE={block_size:>5}: FAILED: {e}")
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
