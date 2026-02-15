"""
Kaggle Benchmark: Baseline vs Fused vs Batched vs Persistent
=============================================================

Upload to Kaggle along with:
- moe_fused_kernel.py      (MoEFFN_Fused + MoEFFN_Batched)
- moe_persistent_kernel.py (MoEFFN_Persistent)
- moe_standalone_kaggle.py (MoEFFN baseline)

Run:  !python /kaggle/input/<dataset>/kaggle_benchmark_phase1.py
"""

import torch
import torch.nn.functional as F
import time
import sys
import gc

print("=" * 70)
print("MoE Optimization Benchmark — All Variants")
print("=" * 70)

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ CUDA not available")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
cc       = torch.cuda.get_device_capability(0)
num_sms  = torch.cuda.get_device_properties(0).multi_processor_count
print(f"GPU: {gpu_name}")
print(f"Compute capability: {cc}")
print(f"GPU memory: {gpu_mem:.1f} GB")
print(f"SMs: {num_sms}")

# ── imports ──────────────────────────────────────────────────────────────────

try:
    from moe_standalone_kaggle import MoEFFN as MoEFFN_Baseline
    print("✅ Imported MoEFFN (baseline)")
except ImportError as e:
    print(f"❌ {e}"); sys.exit(1)

try:
    from moe_fused_kernel import MoEFFN_Batched
    print("✅ Imported MoEFFN_Batched")
except ImportError as e:
    print(f"⚠️  MoEFFN_Batched not available: {e}")
    MoEFFN_Batched = None

try:
    from moe_persistent_kernel import MoEFFN_Persistent, set_block_config, get_block_config
    print("✅ Imported MoEFFN_Persistent")
except ImportError as e:
    print(f"⚠️  MoEFFN_Persistent not available: {e}")
    MoEFFN_Persistent = None
    set_block_config = None
    get_block_config = None


def _rel_diff(a, b):
    return (a - b).abs().max().item() / (a.abs().max().item() + 1e-8)


def _free():
    gc.collect()
    torch.cuda.empty_cache()


# On H100, TF32 is enabled by default for float32 matmuls.
# Triton tl.dot and cuBLAS accumulate in different orders, causing ~1e-3 diffs.
# Disable TF32 during correctness tests for exact comparison;
# re-enable for benchmarks to get real-world performance.

def _disable_tf32():
    """Disable TF32 for precise correctness testing."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _enable_tf32():
    """Re-enable TF32 for real-world benchmark performance."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ── 1. Persistent kernel gradient correctness ───────────────────────────────

def test_persistent_gradients():
    """Verify MoEFFN_Persistent matches baseline (forward + backward)."""
    if MoEFFN_Persistent is None:
        print("\n⚠️  Skipping persistent test (not imported)")
        return True

    print("\n" + "=" * 70)
    print("Test 1: Persistent Kernel — Gradient Equivalence")
    print("=" * 70)

    _disable_tf32()   # exact comparison (both cuBLAS and Triton)

    device = "cuda"
    B, T, D, E, H = 2, 128, 576, 8, 1536
    print(f"Config: B={B}, T={T}, D={D}, experts={E}, hidden={H}")
    if cc[0] >= 8:
        print("  (TF32 disabled for exact comparison)")

    torch.manual_seed(42)
    baseline = MoEFFN_Baseline(D, H, E, top_k=2, data_sparsity=0.5).to(device)

    torch.manual_seed(42)
    persistent = MoEFFN_Persistent(D, H, E, top_k=2, data_sparsity=0.5).to(device)

    with torch.no_grad():
        persistent.W_gate.copy_(baseline.W_gate)
        persistent.W_up.copy_(baseline.W_up)
        persistent.W_down.copy_(baseline.W_down)
        persistent.shared_gate.weight.copy_(baseline.shared_gate.weight)
        persistent.shared_up.weight.copy_(baseline.shared_up.weight)
        persistent.shared_down.weight.copy_(baseline.shared_down.weight)
        persistent.gate.gate.weight.copy_(baseline.gate.gate.weight)
        persistent.gate.logit_bias.copy_(baseline.gate.logit_bias)
        persistent.gate.null_logit.copy_(baseline.gate.null_logit)

    x_base = torch.randn(B, T, D, device=device, requires_grad=True)
    x_pers = x_base.clone().detach().requires_grad_(True)

    out_b, aux_b = baseline(x_base)
    (out_b.sum() + aux_b).backward()

    out_p, aux_p = persistent(x_pers)
    (out_p.sum() + aux_p).backward()

    out_rdiff  = _rel_diff(out_b, out_p)
    grad_rdiff = _rel_diff(x_base.grad, x_pers.grad)

    out_abs  = (out_b - out_p).abs().max().item()
    grad_abs = (x_base.grad - x_pers.grad).abs().max().item()

    print(f"\n  Output  abs diff : {out_abs:.2e}  rel diff : {out_rdiff:.2e}")
    print(f"  Grad    abs diff : {grad_abs:.2e}  rel diff : {grad_rdiff:.2e}")

    ok = out_rdiff < 1e-3 and grad_rdiff < 1e-3
    print(f"\n  {'✅ PASS' if ok else '❌ FAIL'}")

    del baseline, persistent, x_base, x_pers
    _free()
    _enable_tf32()    # restore for benchmarks
    return ok


# ── 2. Batched kernel gradient correctness ──────────────────────────────────

def test_batched_gradients():
    """Verify MoEFFN_Batched matches baseline."""
    if MoEFFN_Batched is None:
        print("\n⚠️  Skipping batched test (not imported)")
        return True

    print("\n" + "=" * 70)
    print("Test 2: Batched GEMM — Gradient Equivalence")
    print("=" * 70)

    _disable_tf32()   # exact comparison (both cuBLAS and Triton)

    device = "cuda"
    B, T, D, E, H = 2, 128, 576, 8, 1536
    print(f"Config: B={B}, T={T}, D={D}, experts={E}, hidden={H}")
    if cc[0] >= 8:
        print("  (TF32 disabled for exact comparison)")

    torch.manual_seed(42)
    baseline = MoEFFN_Baseline(D, H, E, top_k=2, data_sparsity=0.5).to(device)

    torch.manual_seed(42)
    batched = MoEFFN_Batched(D, H, E, top_k=2, data_sparsity=0.5).to(device)

    with torch.no_grad():
        batched.W_gate.copy_(baseline.W_gate)
        batched.W_up.copy_(baseline.W_up)
        batched.W_down.copy_(baseline.W_down)
        batched.shared_gate.weight.copy_(baseline.shared_gate.weight)
        batched.shared_up.weight.copy_(baseline.shared_up.weight)
        batched.shared_down.weight.copy_(baseline.shared_down.weight)
        batched.gate.gate.weight.copy_(baseline.gate.gate.weight)
        batched.gate.logit_bias.copy_(baseline.gate.logit_bias)
        batched.gate.null_logit.copy_(baseline.gate.null_logit)

    x_base = torch.randn(B, T, D, device=device, requires_grad=True)
    x_bat  = x_base.clone().detach().requires_grad_(True)

    out_b, aux_b = baseline(x_base)
    (out_b.sum() + aux_b).backward()

    out_t, aux_t = batched(x_bat)
    (out_t.sum() + aux_t).backward()

    out_rdiff  = _rel_diff(out_b, out_t)
    grad_rdiff = _rel_diff(x_base.grad, x_bat.grad)
    out_abs    = (out_b - out_t).abs().max().item()
    grad_abs   = (x_base.grad - x_bat.grad).abs().max().item()

    print(f"\n  Output  abs diff : {out_abs:.2e}  rel diff : {out_rdiff:.2e}")
    print(f"  Grad    abs diff : {grad_abs:.2e}  rel diff : {grad_rdiff:.2e}")

    ok = out_rdiff < 1e-3 and grad_rdiff < 1e-3
    print(f"\n  {'✅ PASS' if ok else '❌ FAIL'}")

    del baseline, batched, x_base, x_bat
    _free()
    _enable_tf32()    # restore for benchmarks
    return ok


# ── 3. Performance benchmark ────────────────────────────────────────────────

def _bench_one(model, x, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def benchmark():
    print("\n" + "=" * 70)
    print("Test 3: Performance Benchmark")
    print("=" * 70)

    _enable_tf32()   # TF32 ON for real-world benchmark performance
    if cc[0] >= 8:
        print("  TF32 enabled for benchmark (real-world H100 performance)")

    device = "cuda"

    if gpu_mem < 20:
        configs = [
            ("Small (8 experts)",    4, 512,  576,   8, 1536),
            ("Medium (32 experts)",  2, 256, 1024,  32, 1024),
        ]
        print(f"  ⚠️  Smaller configs for {gpu_mem:.0f}GB GPU")
    else:
        configs = [
            ("Small (8 experts)",    4, 512,  576,   8, 1536),
            ("Large (254 experts)",  2, 512, 4096, 254, 1024),
        ]

    warmup, iters = 5, 50
    results = []

    # which variants are available
    variants = [("Baseline", MoEFFN_Baseline)]
    if MoEFFN_Batched is not None:
        variants.append(("Batched", MoEFFN_Batched))
    if MoEFFN_Persistent is not None:
        variants.append(("Persistent", MoEFFN_Persistent))

    for name, B, T, D, E, H in configs:
        print(f"\n{name}:")
        print(f"  Shape: B={B}, T={T}, D={D}, Experts={E}, Hidden={H}")

        x = torch.randn(B, T, D, device=device)
        times = {}

        for vname, vclass in variants:
            try:
                _free()
                model = vclass(D, H, E, top_k=2, data_sparsity=0.5).to(device).eval()
                t = _bench_one(model, x, warmup, iters)
                times[vname] = t
                del model
            except torch.cuda.OutOfMemoryError:
                print(f"  ⚠️  OOM for {vname} — skipping")
                _free()
                times[vname] = None
            except Exception as e:
                print(f"  ⚠️  {vname} error: {e}")
                times[vname] = None

        del x; _free()

        base_t = times.get("Baseline")
        if base_t is None:
            print("  ⚠️  Baseline failed — skipping config")
            continue

        row = [name, base_t / iters * 1e3]
        print(f"  {'Baseline':12s}: {base_t:.4f}s  ({base_t/iters*1e3:.2f} ms/iter)")

        for vname, _ in variants[1:]:
            t = times.get(vname)
            if t is not None:
                spd = base_t / t
                tag = "✅" if spd >= 1.1 else ("⚠️" if spd >= 0.95 else "❌")
                print(f"  {vname:12s}: {t:.4f}s  ({t/iters*1e3:.2f} ms/iter)  {spd:.2f}x {tag}")
                row.append(t / iters * 1e3)
                row.append(spd)
            else:
                row.extend([None, None])

        results.append(row)

    return results, [v[0] for v in variants]


# ── 4. Block-size tuning (persistent kernel only) ────────────────────────────

def tune_block_sizes():
    """
    Sweep block sizes, num_warps, num_stages on the large-expert config.
    Finds the optimal configuration for the persistent kernel on this GPU.
    """
    if MoEFFN_Persistent is None or set_block_config is None:
        print("\n⚠️  Persistent kernel not available — skipping tuning")
        return None

    print("\n" + "=" * 70)
    print("Test 4: Block-Size Tuning (Persistent Kernel)")
    print("=" * 70)

    _enable_tf32()
    device = "cuda"

    # Use the large config where persistent kernel shines
    if gpu_mem >= 40:
        B, T, D, E, H = 2, 512, 4096, 254, 1024
        label = "Large (254 experts)"
    else:
        B, T, D, E, H = 2, 256, 1024, 32, 1024
        label = "Medium (32 experts)"

    print(f"  Tuning on: {label}")
    print(f"  Shape: B={B}, T={T}, D={D}, Experts={E}, Hidden={H}\n")

    # ── get baseline time first ──
    _free()
    baseline = MoEFFN_Baseline(D, H, E, top_k=2, data_sparsity=0.5).to(device).eval()
    x = torch.randn(B, T, D, device=device)
    base_t = _bench_one(baseline, x, 5, 30)
    base_ms = base_t / 30 * 1e3
    print(f"  Baseline: {base_ms:.2f} ms/iter\n")
    del baseline; _free()

    # ── tuning grid ──
    # H100 tensor cores work with 16×16 tiles; blocks should be multiples of 16
    # Larger blocks = more compute/tile but fewer tiles = worse load balance
    configs = [
        # (BLOCK_M, BLOCK_K, BLOCK_N, num_warps, num_stages)
        # ── vary BLOCK_M (rows per tile) ──
        (32,  64,  64, 4, 3),
        (64,  64,  64, 4, 3),   # current default
        (128, 64,  64, 4, 3),
        # ── vary BLOCK_N (columns per tile) ──
        (64,  64,  32, 4, 3),
        (64,  64, 128, 4, 3),
        (128, 64, 128, 4, 3),
        # ── vary BLOCK_K (inner loop chunk) ──
        (64,  32,  64, 4, 3),
        (64, 128,  64, 4, 3),
        # ── vary num_warps ──
        (64,  64,  64, 2, 3),
        (64,  64,  64, 8, 3),
        (128, 64,  64, 8, 3),
        (128, 64, 128, 8, 3),
        # ── vary num_stages (software pipelining) ──
        (64,  64,  64, 4, 2),
        (64,  64,  64, 4, 4),
        (128, 64,  64, 4, 2),
        (128, 64, 128, 4, 2),
        # ── promising combos ──
        (128, 128, 64, 8, 3),
        (64,  128, 128, 8, 3),
        (128, 64, 128, 8, 2),
    ]

    results = []
    x = torch.randn(B, T, D, device=device)
    warmup, iters = 3, 20

    for i, (bm, bk, bn, nw, ns) in enumerate(configs):
        tag = f"  [{i+1:2d}/{len(configs)}]  M={bm:3d} K={bk:3d} N={bn:3d} warps={nw} stages={ns}"
        try:
            _free()
            set_block_config(BLOCK_M=bm, BLOCK_K=bk, BLOCK_N=bn,
                             num_warps=nw, num_stages=ns)
            model = MoEFFN_Persistent(D, H, E, top_k=2, data_sparsity=0.5).to(device).eval()
            t = _bench_one(model, x, warmup, iters)
            ms = t / iters * 1e3
            spd = base_ms / ms
            star = " ⭐" if spd >= 1.5 else (" ✅" if spd >= 1.1 else "")
            print(f"{tag}  →  {ms:7.2f} ms  {spd:.2f}x{star}")
            results.append((bm, bk, bn, nw, ns, ms, spd))
            del model
        except Exception as e:
            err = str(e)[:60]
            print(f"{tag}  →  ❌ {err}")
            results.append((bm, bk, bn, nw, ns, float('inf'), 0.0))

    del x; _free()

    # ── find best ──
    if results:
        results.sort(key=lambda r: r[5])  # sort by ms
        best = results[0]
        print(f"\n  {'─' * 60}")
        print(f"  🏆 BEST:  M={best[0]} K={best[1]} N={best[2]} "
              f"warps={best[3]} stages={best[4]}")
        print(f"           {best[5]:.2f} ms/iter  →  {best[6]:.2f}x vs baseline")
        print(f"  {'─' * 60}")

        # Set the best config for subsequent runs
        set_block_config(BLOCK_M=best[0], BLOCK_K=best[1], BLOCK_N=best[2],
                         num_warps=best[3], num_stages=best[4])

        # Top-5 table
        print(f"\n  Top 5:")
        print(f"  {'Rank':>4s}  {'M':>3s} {'K':>3s} {'N':>3s} {'W':>2s} {'S':>2s}  {'ms/iter':>8s}  {'speedup':>7s}")
        for rank, r in enumerate(results[:5], 1):
            print(f"  {rank:4d}  {r[0]:3d} {r[1]:3d} {r[2]:3d} {r[3]:2d} {r[4]:2d}  {r[5]:7.2f}ms  {r[6]:6.2f}x")

    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p_ok = test_persistent_gradients()
    if not p_ok:
        print("\n❌ Persistent gradient test failed — stopping.")
        return

    b_ok = test_batched_gradients()
    if not b_ok:
        print("\n❌ Batched gradient test failed — stopping.")
        return

    _free()
    results, variant_names = benchmark()

    if results:
        print("\n" + "=" * 70)
        print("Summary (default block sizes)")
        print("=" * 70)

        header = f"  {'Config':25s}  {'Baseline':>9s}"
        for v in variant_names[1:]:
            header += f"  {v:>9s}  {'speed':>6s}"
        print(header)

        for row in results:
            name = row[0]
            base_ms = row[1]
            line = f"  {name:25s}  {base_ms:7.2f}ms"
            idx = 2
            for v in variant_names[1:]:
                if idx + 1 < len(row) and row[idx] is not None:
                    line += f"  {row[idx]:7.2f}ms  {row[idx+1]:5.2f}x"
                else:
                    line += f"  {'N/A':>9s}  {'N/A':>6s}"
                idx += 2
            print(line)

        print("=" * 70)

    # ── Block-size tuning sweep ──
    _free()
    tune_results = tune_block_sizes()

    if tune_results:
        best = tune_results[0]
        print(f"\n{'=' * 70}")
        print(f"FINAL: Use set_block_config(BLOCK_M={best[0]}, BLOCK_K={best[1]}, "
              f"BLOCK_N={best[2]}, num_warps={best[3]}, num_stages={best[4]})")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ {e}")
        import traceback; traceback.print_exc()
