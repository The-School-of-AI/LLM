"""
Throughput benchmark: fused kernels vs unfused baselines.

Uses realistic model dimensions from recurrence_model_70b_moe.py:
  - hidden_size = 4096
  - GSA QKV:    D=4096 → D=4096  (W_q, W_k, W_v each [4096, 4096])
  - DeltaNet QKVG: D_in=4096 → D_out=4096  (W_q, W_k, W_v, W_g each [4096, 4096])
  - O+gate:    D=4096
  - MoE:       E=260, D=4096, H=1024  (routed experts)

Run from code/:
  python tests/bench_fused_throughput.py

Interpretation (A100, sm_80):
  - QKV / QKVG / O+gate: Fused kernels use small blocks (num_stages=1) to fit 163 KB
    shared memory, so cuBLAS (F.linear) is usually faster. Fused path is for correctness
    and for small-GPU (e.g. T4) where it avoids OOM.
  - MoE: Triton fused can beat 3x grouped_gemm at small/medium N; at large N
    grouped_gemm may win. Prefer Triton for typical batch sizes if both are correct.
"""

import sys
from pathlib import Path

code_root = Path(__file__).resolve().parent.parent
src = code_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

import torch
import torch.nn.functional as F
import time

assert torch.cuda.is_available(), "CUDA required for benchmarking"

WARMUP = 10
ITERS = 100
DTYPE = torch.bfloat16


def _sync():
    torch.cuda.synchronize()


def _bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms / iters


# ── Fused QKV vs 3x F.linear (GSA: D=4096) ──────────────────────────────────

def bench_fused_qkv():
    from kernels.fused_qkv_proj import fused_qkv_proj_forward

    D = 4096
    for N in [256, 1024, 4096]:
        x = torch.randn(N, D, device="cuda", dtype=DTYPE) * 0.02
        W_q = torch.randn(D, D, device="cuda", dtype=DTYPE) * 0.02
        W_k = torch.randn(D, D, device="cuda", dtype=DTYPE) * 0.02
        W_v = torch.randn(D, D, device="cuda", dtype=DTYPE) * 0.02

        def unfused():
            F.linear(x, W_q)
            F.linear(x, W_k)
            F.linear(x, W_v)

        def fused():
            fused_qkv_proj_forward(x, W_q, W_k, W_v)

        t_unfused = _bench(unfused)
        t_fused = _bench(fused)
        speedup = t_unfused / t_fused if t_fused > 0 else float("inf")
        print(f"  QKV  N={N:5d}  D={D}  unfused={t_unfused:.3f}ms  fused={t_fused:.3f}ms  speedup={speedup:.2f}x")


# ── Fused QKVG vs 4x F.linear (DeltaNet: D_in=4096, D_out=4096) ────────────

def bench_fused_qkvg():
    from kernels.fused_qkv_proj import fused_qkvg_proj_forward

    D_in, D_out = 4096, 4096
    for N in [256, 1024, 4096]:
        x = torch.randn(N, D_in, device="cuda", dtype=DTYPE) * 0.02
        W_q = torch.randn(D_out, D_in, device="cuda", dtype=DTYPE) * 0.02
        W_k = torch.randn(D_out, D_in, device="cuda", dtype=DTYPE) * 0.02
        W_v = torch.randn(D_out, D_in, device="cuda", dtype=DTYPE) * 0.02
        W_g = torch.randn(D_out, D_in, device="cuda", dtype=DTYPE) * 0.02

        def unfused():
            F.linear(x, W_q)
            F.linear(x, W_k)
            F.linear(x, W_v)
            F.linear(x, W_g)

        def fused():
            fused_qkvg_proj_forward(x, W_q, W_k, W_v, W_g)

        t_unfused = _bench(unfused)
        t_fused = _bench(fused)
        speedup = t_unfused / t_fused if t_fused > 0 else float("inf")
        print(f"  QKVG N={N:5d}  D_in={D_in} D_out={D_out}  unfused={t_unfused:.3f}ms  fused={t_fused:.3f}ms  speedup={speedup:.2f}x")


# ── Fused O+gate vs unfused (GSA: D=4096) ───────────────────────────────────

def bench_fused_o_gate():
    from kernels.fused_qkv_proj import fused_o_gate_proj_forward

    D = 4096
    for N in [256, 1024, 4096]:
        x = torch.randn(N, D, device="cuda", dtype=DTYPE) * 0.02
        o_sparse = torch.randn(N, D, device="cuda", dtype=DTYPE) * 0.02
        W_go = torch.randn(D, D, device="cuda", dtype=DTYPE) * 0.02
        W_o = torch.randn(D, D, device="cuda", dtype=DTYPE) * 0.02

        def unfused():
            g_o = torch.sigmoid(F.linear(x, W_go))
            F.linear(o_sparse * g_o, W_o)

        def fused():
            fused_o_gate_proj_forward(x, o_sparse, W_go, W_o)

        t_unfused = _bench(unfused)
        t_fused = _bench(fused)
        speedup = t_unfused / t_fused if t_fused > 0 else float("inf")
        print(f"  Ogate N={N:5d}  D={D}  unfused={t_unfused:.3f}ms  fused={t_fused:.3f}ms  speedup={speedup:.2f}x")


# ── Fused MoE expert: Triton vs grouped_gemm fallback (E=260, D=4096, H=1024) ─

def bench_fused_moe():
    from kernels.fused_moe_expert import fused_moe_expert_forward, has_fused_moe_expert_triton
    try:
        from kernels.moe_grouped_gemm import moe_grouped_gemm
        has_gg = True
    except ImportError:
        has_gg = False

    E, D, H = 260, 4096, 1024
    top_k = 8
    for total_tokens in [256, 1024, 4096]:
        N = total_tokens * top_k
        counts = torch.full((E,), N // E, device="cuda", dtype=torch.int64)
        remainder = N - counts.sum().item()
        if remainder > 0:
            counts[:int(remainder)] += 1
        N_actual = int(counts.sum().item())
        x = torch.randn(N_actual, D, device="cuda", dtype=DTYPE) * 0.02
        W_gate = torch.randn(E, D, H, device="cuda", dtype=DTYPE) * 0.02
        W_up = torch.randn(E, D, H, device="cuda", dtype=DTYPE) * 0.02
        W_down = torch.randn(E, H, D, device="cuda", dtype=DTYPE) * 0.02

        if has_gg:
            def grouped_gemm_path():
                fused_moe_expert_forward(x, W_gate, W_up, W_down, counts, use_triton=False)
            t_gg = _bench(grouped_gemm_path)
        else:
            t_gg = float("nan")

        if has_fused_moe_expert_triton():
            def triton_path():
                fused_moe_expert_forward(x, W_gate, W_up, W_down, counts, use_triton=True)
            t_triton = _bench(triton_path)
        else:
            t_triton = float("nan")

        # Reference: 3x grouped_gemm + silu_mul (only if available)
        if has_gg:
            def ref_path():
                gate_out = moe_grouped_gemm(x, W_gate, counts)
                up_out = moe_grouped_gemm(x, W_up, counts)
                h = F.silu(gate_out) * up_out
                moe_grouped_gemm(h, W_down, counts)
            t_ref = _bench(ref_path)
        else:
            t_ref = float("nan")

        print(f"  MoE  tokens={total_tokens:5d} (N={N_actual})  E={E}  D={D}  H={H}")
        print(f"       3xGG+silu={t_ref:.3f}ms  GG-fallback={t_gg:.3f}ms  Triton={t_triton:.3f}ms")


if __name__ == "__main__":
    gpu = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu} (sm_{major}{minor}), dtype={DTYPE}, warmup={WARMUP}, iters={ITERS}")
    print()

    print("=== Fused QKV (GSA) ===")
    bench_fused_qkv()
    print()

    print("=== Fused QKVG (DeltaNet) ===")
    bench_fused_qkvg()
    print()

    print("=== Fused O+gate (GSA) ===")
    bench_fused_o_gate()
    print()

    print("=== Fused MoE expert ===")
    bench_fused_moe()
    print()

    print("Done.")
