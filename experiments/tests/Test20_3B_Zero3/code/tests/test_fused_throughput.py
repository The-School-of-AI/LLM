"""
Tests for throughput optimization kernels: fused MoE expert and fused QKV.

- Correctness: compare fused path vs reference (grouped_gemm + silu_mul for MoE;
  three linears for QKV) on same inputs.
- Optional: report relative timing when CUDA is available.

Run from code/:
  python -m pytest tests/test_fused_throughput.py -v
  or: python tests/test_fused_throughput.py
"""

import sys
from pathlib import Path

# Add src so we can import kernels
code_root = Path(__file__).resolve().parent.parent
src = code_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

import torch


def test_fused_moe_expert_correctness():
    """Fused MoE expert output matches 3x grouped_gemm + silu_mul (when grouped_gemm available)."""
    try:
        from kernels.fused_moe_expert import fused_moe_expert_forward
    except ImportError:
        raise RuntimeError("kernels not importable (missing deps or path)")
    try:
        from kernels.moe_grouped_gemm import moe_grouped_gemm
    except ImportError:
        print("test_fused_moe_expert_correctness skipped (grouped_gemm not available)")
        return

    if not torch.cuda.is_available():
        return  # skip on CPU

    # grouped_gemm backend only supports bfloat16
    E, D, H = 4, 128, 64
    m_sizes = torch.tensor([16, 24, 8, 32], device="cuda")
    expert_counts = m_sizes
    N = int(m_sizes.sum().item())
    dtype = torch.bfloat16
    x = torch.randn(N, D, device="cuda", dtype=dtype) * 0.02
    W_gate = torch.randn(E, D, H, device="cuda", dtype=dtype) * 0.02
    W_up = torch.randn(E, D, H, device="cuda", dtype=dtype) * 0.02
    W_down = torch.randn(E, H, D, device="cuda", dtype=dtype) * 0.02

    out_fused = fused_moe_expert_forward(
        x, W_gate, W_up, W_down, expert_counts, use_triton=False
    )
    gate_ref = moe_grouped_gemm(x, W_gate, expert_counts)
    up_ref = moe_grouped_gemm(x, W_up, expert_counts)
    h_ref = torch.nn.functional.silu(gate_ref) * up_ref
    out_ref = moe_grouped_gemm(h_ref, W_down, expert_counts)

    torch.testing.assert_close(out_fused.float(), out_ref.float(), atol=1e-2, rtol=1e-2)
    print("test_fused_moe_expert_correctness passed (use_triton=False)")


def test_fused_moe_expert_triton():
    """Fused MoE expert Triton kernel matches grouped_gemm reference."""
    try:
        from kernels.fused_moe_expert import fused_moe_expert_forward, has_fused_moe_expert_triton
    except ImportError:
        raise RuntimeError("kernels not importable (missing deps or path)")
    if not has_fused_moe_expert_triton():
        print("test_fused_moe_expert_triton skipped (Triton not available)")
        return
    try:
        from kernels.moe_grouped_gemm import moe_grouped_gemm
    except ImportError:
        print("test_fused_moe_expert_triton skipped (grouped_gemm not available)")
        return
    if not torch.cuda.is_available():
        return

    E, D, H = 4, 128, 64
    m_sizes = torch.tensor([16, 24, 8, 32], device="cuda")
    expert_counts = m_sizes
    N = int(m_sizes.sum().item())
    dtype = torch.bfloat16
    x = torch.randn(N, D, device="cuda", dtype=dtype) * 0.02
    W_gate = torch.randn(E, D, H, device="cuda", dtype=dtype) * 0.02
    W_up = torch.randn(E, D, H, device="cuda", dtype=dtype) * 0.02
    W_down = torch.randn(E, H, D, device="cuda", dtype=dtype) * 0.02

    out_triton = fused_moe_expert_forward(x, W_gate, W_up, W_down, expert_counts, use_triton=True)
    gate_ref = moe_grouped_gemm(x, W_gate, expert_counts)
    up_ref = moe_grouped_gemm(x, W_up, expert_counts)
    h_ref = torch.nn.functional.silu(gate_ref) * up_ref
    out_ref = moe_grouped_gemm(h_ref, W_down, expert_counts)

    torch.testing.assert_close(out_triton.float(), out_ref.float(), atol=1e-2, rtol=1e-2)
    print("test_fused_moe_expert_triton passed (use_triton=True)")


def test_fused_qkv_correctness():
    """Fused QKV output matches three separate F.linear(x, W_*)."""
    try:
        from kernels.fused_qkv_proj import fused_qkv_proj_forward
    except ImportError:
        raise RuntimeError("kernels not importable")

    if not torch.cuda.is_available():
        return

    N, D = 64, 128
    x = torch.randn(N, D, device="cuda", dtype=torch.float32) * 0.02
    W_q = torch.randn(D, D, device="cuda", dtype=torch.float32) * 0.02
    W_k = torch.randn(D, D, device="cuda", dtype=torch.float32) * 0.02
    W_v = torch.randn(D, D, device="cuda", dtype=torch.float32) * 0.02

    q_f, k_f, v_f = fused_qkv_proj_forward(x, W_q, W_k, W_v)
    q_ref = torch.nn.functional.linear(x, W_q)
    k_ref = torch.nn.functional.linear(x, W_k)
    v_ref = torch.nn.functional.linear(x, W_v)

    torch.testing.assert_close(q_f.float(), q_ref.float(), atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(k_f.float(), k_ref.float(), atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(v_f.float(), v_ref.float(), atol=1e-4, rtol=1e-3)
    print("test_fused_qkv_correctness passed")


def test_fused_qkvg_correctness():
    """Fused QKVG output matches four separate F.linear(x, W_*)."""
    try:
        from kernels.fused_qkv_proj import fused_qkvg_proj_forward
    except ImportError:
        raise RuntimeError("kernels not importable")

    if not torch.cuda.is_available():
        return

    N, D_in, D_out = 64, 256, 128
    x = torch.randn(N, D_in, device="cuda", dtype=torch.float32) * 0.02
    W_q = torch.randn(D_out, D_in, device="cuda", dtype=torch.float32) * 0.02
    W_k = torch.randn(D_out, D_in, device="cuda", dtype=torch.float32) * 0.02
    W_v = torch.randn(D_out, D_in, device="cuda", dtype=torch.float32) * 0.02
    W_g = torch.randn(D_out, D_in, device="cuda", dtype=torch.float32) * 0.02

    q_f, k_f, v_f, g_f = fused_qkvg_proj_forward(x, W_q, W_k, W_v, W_g)
    q_ref = torch.nn.functional.linear(x, W_q)
    k_ref = torch.nn.functional.linear(x, W_k)
    v_ref = torch.nn.functional.linear(x, W_v)
    g_ref = torch.nn.functional.linear(x, W_g)

    torch.testing.assert_close(q_f.float(), q_ref.float(), atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(k_f.float(), k_ref.float(), atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(v_f.float(), v_ref.float(), atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(g_f.float(), g_ref.float(), atol=1e-4, rtol=1e-3)
    print("test_fused_qkvg_correctness passed")


def test_fused_o_gate_correctness():
    """Fused O+gate output matches o_proj(o_sparse * sigmoid(W_go(x)))."""
    try:
        from kernels.fused_qkv_proj import fused_o_gate_proj_forward
    except ImportError:
        raise RuntimeError("kernels not importable")

    if not torch.cuda.is_available():
        return

    N, D = 64, 128
    x = torch.randn(N, D, device="cuda", dtype=torch.float32) * 0.02
    o_sparse = torch.randn(N, D, device="cuda", dtype=torch.float32) * 0.02
    W_go = torch.randn(D, D, device="cuda", dtype=torch.float32) * 0.02
    W_o = torch.randn(D, D, device="cuda", dtype=torch.float32) * 0.02

    out_f = fused_o_gate_proj_forward(x, o_sparse, W_go, W_o)
    g_o = torch.sigmoid(torch.nn.functional.linear(x, W_go))
    out_ref = torch.nn.functional.linear(o_sparse * g_o, W_o)

    torch.testing.assert_close(out_f.float(), out_ref.float(), atol=1e-4, rtol=1e-3)
    print("test_fused_o_gate_correctness passed")


if __name__ == "__main__":
    test_fused_moe_expert_correctness()
    test_fused_moe_expert_triton()
    test_fused_qkv_correctness()
    test_fused_qkvg_correctness()
    test_fused_o_gate_correctness()
    print("All tests passed.")
