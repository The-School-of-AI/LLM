import os
import sys

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kernels"))
import time

from triton_sparse_attn import pytorch_sparse_attention, triton_sparse_attention


def run_benchmark_for_config(B, H, T_q, T_kv, D, k_sel, correlated=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print(
            "CUDA not available. This script is intended to benchmark on a GPU (like T4)."
        )
        return

    print(
        f"\n[{torch.cuda.get_device_name(0)}] Benchmarking: B={B}, H={H}, T={T_q}, D={D}, K_sel={k_sel} (Correlated={correlated})"
    )
    print("-" * 110)
    print(
        f"{'Method':<22} | {'FWD Time':<10} | {'BWD Time':<10} | {'Total Time':<10} | {'FWD Mem (MB)':<14} | {'BWD Mem (MB)':<14}"
    )
    print("-" * 110)

    # 1. Initialize Tensors
    q = torch.randn(
        B, T_q, H, D, device=device, dtype=torch.float16, requires_grad=True
    )
    k = torch.randn(
        B, T_kv, H, D, device=device, dtype=torch.float16, requires_grad=True
    )
    v = torch.randn(
        B, T_kv, H, D, device=device, dtype=torch.float16, requires_grad=True
    )

    if correlated:
        BLOCK_Q = 64
        num_q_blocks = max(1, T_q // BLOCK_Q)
        base_indices = torch.randint(
            0, T_kv, (B, H, num_q_blocks, k_sel), device=device, dtype=torch.int64
        )
        indices = base_indices.repeat_interleave(BLOCK_Q, dim=2)
        indices = indices[:, :, :T_q, :]
    else:
        indices = torch.randint(
            0, T_kv, (B, H, T_q, k_sel), device=device, dtype=torch.int64
        )

    mask = (torch.rand(B, H, T_q, k_sel, device=device) < 0.9).float()
    scale = 1.0 / (D**0.5)

    grad_out = torch.randn_like(q)

    def profile_func(name, fwd_func):
        q.grad, k.grad, v.grad = None, None, None

        # Warmup
        for _ in range(3):
            out = fwd_func()
            out.backward(grad_out, retain_graph=True)

        # Time and Mem FWD Only
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_fwd = time.time()
        for _ in range(10):
            out = fwd_func()
        torch.cuda.synchronize()
        fwd_time = (time.time() - start_fwd) / 10.0 * 1000
        fwd_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

        # Time and Mem BWD Only
        out = fwd_func()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_bwd = time.time()
        for _ in range(10):
            out.backward(grad_out, retain_graph=True)
        torch.cuda.synchronize()
        bwd_time = (time.time() - start_bwd) / 10.0 * 1000
        bwd_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

        # Time Total
        torch.cuda.synchronize()
        start_tot = time.time()
        for _ in range(10):
            out = fwd_func()
            out.backward(grad_out, retain_graph=True)
        torch.cuda.synchronize()
        tot_time = (time.time() - start_tot) / 10.0 * 1000

        print(
            f"{name:<22} | {fwd_time:>7.2f} ms | {bwd_time:>7.2f} ms | {tot_time:>7.2f} ms | {fwd_mem_mb:>12.2f} | {bwd_mem_mb:>12.2f}"
        )

    # ----------------------------------------------------
    # Baseline: Dense PyTorch Attention (FlashAttention compatible)
    # ----------------------------------------------------
    try:
        q_dense = q.permute(0, 2, 1, 3).clone().detach().requires_grad_(True)
        k_dense = k.permute(0, 2, 1, 3).clone().detach().requires_grad_(True)
        v_dense = v.permute(0, 2, 1, 3).clone().detach().requires_grad_(True)
        grad_dense = grad_out.permute(0, 2, 1, 3)

        def run_dense_fwd():
            with torch.nn.attention.sdpa_kernel(
                [
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.MATH,
                ]
            ):
                return torch.nn.functional.scaled_dot_product_attention(
                    q_dense, k_dense, v_dense
                )

        # Need custom BWD profiling due to different tensor structure for Dense
        q_dense.grad, k_dense.grad, v_dense.grad = None, None, None
        for _ in range(3):
            out = run_dense_fwd()
            out.backward(grad_dense, retain_graph=True)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_fwd = time.time()
        for _ in range(10):
            run_dense_fwd()
        torch.cuda.synchronize()
        dense_fwd_time = (time.time() - start_fwd) / 10.0 * 1000
        dense_fwd_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

        out = run_dense_fwd()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_bwd = time.time()
        for _ in range(10):
            out.backward(grad_dense, retain_graph=True)
        torch.cuda.synchronize()
        dense_bwd_time = (time.time() - start_bwd) / 10.0 * 1000
        dense_bwd_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

        torch.cuda.synchronize()
        start_tot = time.time()
        for _ in range(10):
            out = run_dense_fwd()
            out.backward(grad_dense, retain_graph=True)
        torch.cuda.synchronize()
        dense_tot_time = (time.time() - start_tot) / 10.0 * 1000

        print(
            f"{'Dense PyTorch':<22} | {dense_fwd_time:>7.2f} ms | {dense_bwd_time:>7.2f} ms | {dense_tot_time:>7.2f} ms | {dense_fwd_mem_mb:>12.2f} | {dense_bwd_mem_mb:>12.2f}"
        )
    except torch.cuda.OutOfMemoryError:
        print(
            f"{'Dense PyTorch':<22} | {'OOM':>10} | {'OOM':>10} | {'OOM':>10} | {'OOM':>14} | {'OOM':>14}"
        )
        torch.cuda.empty_cache()

    # ----------------------------------------------------
    # Baseline: PyTorch Sparse Gather
    # ----------------------------------------------------
    try:

        def run_py_sparse_fwd():
            return pytorch_sparse_attention(
                q, k, v, indices, mask, scale, chunk_size=32
            )

        profile_func("PyTorch Sparse", run_py_sparse_fwd)
    except torch.cuda.OutOfMemoryError:
        print(f"{'PyTorch Sparse':<25} | {'OOM':>10} | {'OOM':>10} | {'OOM':>10}")
        torch.cuda.empty_cache()

    # ----------------------------------------------------
    # V1: Triton Sparse
    # ----------------------------------------------------
    def run_triton_v1_fwd():
        return triton_sparse_attention(
            q, k, v, indices, mask, scale, use_triton_backward=True
        )

    profile_func("Triton Sparse V1", run_triton_v1_fwd)


def benchmark_attention():
    # Test 1: Standard Config (Random Indices)
    run_benchmark_for_config(
        B=2, H=8, T_q=4096, T_kv=4096, D=128, k_sel=128, correlated=False
    )

    # Test 2: Standard Config (Correlated Indices mimicking real Attention)
    run_benchmark_for_config(
        B=2, H=8, T_q=4096, T_kv=4096, D=128, k_sel=128, correlated=True
    )

    # Test 3: High Batch Config (Simulating production inference/training depth)
    run_benchmark_for_config(
        B=8, H=8, T_q=4096, T_kv=4096, D=128, k_sel=128, correlated=False
    )


if __name__ == "__main__":
    benchmark_attention()
