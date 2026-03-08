#!/usr/bin/env python3
"""
Isolation benchmark: Expert Parallelism via all-to-all dispatch.

Compares two MoE strategies on 8x A100 with NVLink:
  A) Replicated experts (current) — all 8 GPUs have all 20 experts
  B) Expert-parallel             — each GPU owns 2-3 experts, tokens routed via all-to-all

Usage:
  torchrun --nproc_per_node=8 scripts/benchmark_expert_parallel.py

This is a standalone microbenchmark (no DeepSpeed, no model, no data).
"""

import os
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Replicated experts (current approach) — no communication, all experts local
# ---------------------------------------------------------------------------
def moe_replicated(
    sorted_x: torch.Tensor,          # [M, D]
    expert_counts: torch.Tensor,      # [E] on CPU
    W_gate: torch.Tensor,             # [E, D, H]
    W_up: torch.Tensor,               # [E, D, H]
    W_down: torch.Tensor,             # [E, H, D]
    gmm_fn,
):
    """Each GPU computes all experts on its local tokens."""
    x_in = sorted_x.to(dtype=W_gate.dtype)
    gate_out = gmm_fn(x_in, W_gate, expert_counts)
    up_out = gmm_fn(x_in, W_up, expert_counts)
    h = F.silu(gate_out) * up_out
    out = gmm_fn(h, W_down, expert_counts)
    return out.to(dtype=sorted_x.dtype)


# ---------------------------------------------------------------------------
# Expert-parallel — all-to-all dispatch, local compute, all-to-all combine
# ---------------------------------------------------------------------------
def moe_expert_parallel(
    sorted_x: torch.Tensor,          # [M, D]   — sorted by expert_id
    sorted_expert_ids: torch.Tensor,  # [M]      — expert ids (sorted)
    expert_counts: torch.Tensor,      # [E] on GPU
    W_gate_local: torch.Tensor,       # [E_local, D, H]
    W_up_local: torch.Tensor,         # [E_local, D, H]
    W_down_local: torch.Tensor,       # [E_local, H, D]
    expert_to_rank: torch.Tensor,     # [E] — maps expert_id -> owning rank
    rank: int,
    world_size: int,
    gmm_fn,
):
    """
    Expert-parallel MoE dispatch using all-to-all.

    1. Each GPU has tokens sorted by expert. Some experts are remote.
    2. Re-sort tokens by destination rank (grouping for all-to-all).
    3. all_to_all: send tokens to owning GPU.
    4. Local grouped_gemm on received tokens.
    5. all_to_all: send results back.
    6. Un-sort to original expert order.
    """
    M, D = sorted_x.shape
    E = expert_counts.shape[0]
    dtype = sorted_x.dtype

    # --- Step 1: Compute send counts per rank ---
    # expert_counts[e] tells how many tokens go to expert e.
    # expert_to_rank[e] tells which rank owns expert e.
    dest_rank = expert_to_rank[sorted_expert_ids]  # [M] — destination rank per token

    # Count tokens per destination rank
    send_counts = torch.zeros(world_size, dtype=torch.long, device=sorted_x.device)
    for r in range(world_size):
        send_counts[r] = (dest_rank == r).sum()

    # --- Step 2: Sort tokens by destination rank (stable, preserves expert order within rank) ---
    rank_sort_idx = dest_rank.argsort(stable=True)
    tokens_to_send = sorted_x[rank_sort_idx]        # [M, D] re-ordered by dest rank
    experts_to_send = sorted_expert_ids[rank_sort_idx]  # [M] expert ids in rank order

    # --- Step 3: Exchange send_counts so each rank knows recv_counts ---
    send_counts_list = send_counts.tolist()
    recv_counts_tensor = torch.zeros(world_size, dtype=torch.long, device=sorted_x.device)
    dist.all_to_all_single(recv_counts_tensor, send_counts)
    recv_counts_list = recv_counts_tensor.tolist()

    total_recv = sum(recv_counts_list)

    # --- Step 4: all_to_all for tokens ---
    send_splits = [int(c) for c in send_counts_list]
    recv_splits = [int(c) for c in recv_counts_list]

    send_chunks = list(tokens_to_send.split(send_splits, dim=0))
    recv_chunks = [torch.empty(rc, D, dtype=dtype, device=sorted_x.device) for rc in recv_splits]
    dist.all_to_all(recv_chunks, send_chunks)
    received_tokens = torch.cat(recv_chunks, dim=0)  # [total_recv, D]

    # Also send expert_ids so we know which local expert to use
    send_id_chunks = list(experts_to_send.split(send_splits, dim=0))
    recv_id_chunks = [torch.empty(rc, dtype=torch.long, device=sorted_x.device) for rc in recv_splits]
    dist.all_to_all(recv_id_chunks, send_id_chunks)
    received_expert_ids = torch.cat(recv_id_chunks, dim=0)  # [total_recv]

    # --- Step 5: Local expert compute via grouped_gemm ---
    if total_recv > 0:
        # Map global expert_id to local expert index
        local_expert_start = 0
        for r in range(rank):
            local_expert_start += (E + world_size - 1 - r) // world_size  # not quite right, use simple mapping
        # Simple mapping: expert e is on rank (e % world_size), local index = e // world_size
        local_ids = received_expert_ids // world_size  # local expert index

        # Re-sort received tokens by local expert id for grouped_gemm
        local_sort_idx = local_ids.argsort(stable=True)
        sorted_recv = received_tokens[local_sort_idx]
        sorted_local_ids = local_ids[local_sort_idx]

        E_local = W_gate_local.shape[0]
        local_counts = torch.bincount(sorted_local_ids, minlength=E_local).cpu().to(torch.int64).contiguous()

        x_in = sorted_recv.to(dtype=W_gate_local.dtype)
        gate_out = gmm_fn(x_in, W_gate_local, local_counts)
        up_out = gmm_fn(x_in, W_up_local, local_counts)
        h = F.silu(gate_out) * up_out
        local_out = gmm_fn(h, W_down_local, local_counts)
        local_out = local_out.to(dtype=dtype)

        # Un-sort back to received order
        unsort_idx = torch.empty_like(local_sort_idx)
        unsort_idx[local_sort_idx] = torch.arange(total_recv, device=sorted_x.device)
        result_to_send = local_out[unsort_idx]
    else:
        result_to_send = torch.empty(0, D, dtype=dtype, device=sorted_x.device)

    # --- Step 6: all_to_all results back ---
    # recv_splits becomes send_splits (we're sending back to where tokens came from)
    send_back_chunks = list(result_to_send.split(recv_splits, dim=0))
    recv_back_chunks = [torch.empty(sc, D, dtype=dtype, device=sorted_x.device) for sc in send_splits]
    dist.all_to_all(recv_back_chunks, send_back_chunks)
    results_back = torch.cat(recv_back_chunks, dim=0)  # [M, D]

    # --- Step 7: Un-sort from rank order back to original expert order ---
    unsort_rank = torch.empty_like(rank_sort_idx)
    unsort_rank[rank_sort_idx] = torch.arange(M, device=sorted_x.device)
    final_out = results_back[unsort_rank]

    return final_out


def benchmark(fn, warmup=5, iters=20, label=""):
    """Run fn for warmup+iters, return median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    return median


def main():
    rank, world_size = setup()
    device = torch.device(f"cuda:{rank}")

    # --- Model dimensions (3B MoE) ---
    E = 20        # total experts
    D = 1536      # d_model
    H = 4096      # d_hidden
    B = 2         # batch size per GPU
    T = 4096      # sequence length
    K = 2         # top_k
    real_frac = 0.8  # fraction of non-null assignments

    N = B * T
    num_real = int(N * K * real_frac)

    # --- Setup grouped_gemm ---
    import grouped_gemm
    ops = getattr(grouped_gemm, "ops", grouped_gemm)

    def gmm_fn(a, b, counts):
        if isinstance(counts, torch.Tensor) and counts.device.type != "cpu":
            counts = counts.detach().cpu().to(torch.int64).contiguous()
        elif isinstance(counts, torch.Tensor):
            counts = counts.to(torch.int64).contiguous()
        return ops.gmm(a, b, counts, trans_b=False)

    # --- Create weights ---
    # Full weights for replicated mode
    W_gate = torch.randn(E, D, H, device=device, dtype=torch.bfloat16)
    W_up = torch.randn(E, D, H, device=device, dtype=torch.bfloat16)
    W_down = torch.randn(E, H, D, device=device, dtype=torch.bfloat16)

    # Expert-parallel: each rank owns experts where (expert_id % world_size == rank)
    local_expert_ids = [e for e in range(E) if e % world_size == rank]
    E_local = len(local_expert_ids)
    W_gate_local = W_gate[local_expert_ids]  # [E_local, D, H]
    W_up_local = W_up[local_expert_ids]
    W_down_local = W_down[local_expert_ids]

    expert_to_rank = torch.tensor([e % world_size for e in range(E)], device=device, dtype=torch.long)

    # --- Simulate sorted tokens (different per rank for realism) ---
    torch.manual_seed(42 + rank)
    sorted_expert_ids = torch.sort(torch.randint(0, E, (num_real,), device=device))[0]
    expert_counts = torch.bincount(sorted_expert_ids, minlength=E)
    expert_counts_cpu = expert_counts.cpu().to(torch.int64).contiguous()
    sorted_x = torch.randn(num_real, D, device=device, dtype=torch.bfloat16)

    # --- Warmup + correctness check ---
    out_rep = moe_replicated(sorted_x, expert_counts_cpu, W_gate, W_up, W_down, gmm_fn)
    out_ep = moe_expert_parallel(
        sorted_x, sorted_expert_ids, expert_counts,
        W_gate_local, W_up_local, W_down_local,
        expert_to_rank, rank, world_size, gmm_fn,
    )

    # Check outputs match (they should, modulo floating point)
    max_diff = (out_rep.float() - out_ep.float()).abs().max().item()
    if rank == 0:
        print(f"Correctness check: max_diff = {max_diff:.6f}")

    # --- Benchmark replicated ---
    ms_rep = benchmark(
        lambda: moe_replicated(sorted_x, expert_counts_cpu, W_gate, W_up, W_down, gmm_fn),
        warmup=5, iters=30, label="replicated",
    )

    # --- Benchmark expert-parallel ---
    ms_ep = benchmark(
        lambda: moe_expert_parallel(
            sorted_x, sorted_expert_ids, expert_counts,
            W_gate_local, W_up_local, W_down_local,
            expert_to_rank, rank, world_size, gmm_fn,
        ),
        warmup=5, iters=30, label="expert-parallel",
    )

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"MoE Expert Parallelism Benchmark (E={E}, D={D}, H={H})")
        print(f"Tokens per GPU: {num_real} (B={B}, T={T}, K={K})")
        print(f"World size: {world_size} GPUs")
        print(f"Local experts: {E_local} per GPU")
        print(f"{'='*60}")
        print(f"Replicated (current):   {ms_rep:.2f} ms/layer  (all {E} experts on every GPU)")
        print(f"Expert-parallel:        {ms_ep:.2f} ms/layer  ({E_local} experts per GPU + all-to-all)")
        speedup = ms_rep / ms_ep if ms_ep > 0 else float("inf")
        print(f"Speedup:                {speedup:.2f}x")
        if speedup > 1:
            print(f"Expert-parallel is {(speedup - 1) * 100:.1f}% FASTER")
        else:
            print(f"Expert-parallel is {(1 - speedup) * 100:.1f}% SLOWER (all-to-all overhead > compute savings)")

        # Memory comparison
        full_expert_mem_gb = E * D * H * 2 * 3 / 1e9  # 3 weight matrices, bf16
        local_expert_mem_gb = E_local * D * H * 2 * 3 / 1e9
        print(f"\nMemory per GPU:")
        print(f"  Replicated:      {full_expert_mem_gb:.2f} GB (all {E} experts)")
        print(f"  Expert-parallel:  {local_expert_mem_gb:.2f} GB ({E_local} experts)")
        print(f"  Savings:          {full_expert_mem_gb - local_expert_mem_gb:.2f} GB ({(1 - local_expert_mem_gb/full_expert_mem_gb)*100:.0f}%)")

    # --- Also benchmark 8B dimensions ---
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Now benchmarking 8B MoE dimensions...")
        print(f"{'='*60}")

    # 8B has 20 layers, each with MoE. Scale token count for 8B.
    E_8b = 20
    D_8b = 2048
    H_8b = 5632
    num_real_8b = int(N * K * real_frac)

    W_gate_8b = torch.randn(E_8b, D_8b, H_8b, device=device, dtype=torch.bfloat16)
    W_up_8b = torch.randn(E_8b, D_8b, H_8b, device=device, dtype=torch.bfloat16)
    W_down_8b = torch.randn(E_8b, H_8b, D_8b, device=device, dtype=torch.bfloat16)

    local_ids_8b = [e for e in range(E_8b) if e % world_size == rank]
    E_local_8b = len(local_ids_8b)
    W_gate_local_8b = W_gate_8b[local_ids_8b]
    W_up_local_8b = W_up_8b[local_ids_8b]
    W_down_local_8b = W_down_8b[local_ids_8b]

    expert_to_rank_8b = torch.tensor([e % world_size for e in range(E_8b)], device=device, dtype=torch.long)

    torch.manual_seed(42 + rank)
    sorted_expert_ids_8b = torch.sort(torch.randint(0, E_8b, (num_real_8b,), device=device))[0]
    expert_counts_8b = torch.bincount(sorted_expert_ids_8b, minlength=E_8b)
    expert_counts_cpu_8b = expert_counts_8b.cpu().to(torch.int64).contiguous()
    sorted_x_8b = torch.randn(num_real_8b, D_8b, device=device, dtype=torch.bfloat16)

    ms_rep_8b = benchmark(
        lambda: moe_replicated(sorted_x_8b, expert_counts_cpu_8b, W_gate_8b, W_up_8b, W_down_8b, gmm_fn),
        warmup=5, iters=30, label="replicated-8b",
    )

    ms_ep_8b = benchmark(
        lambda: moe_expert_parallel(
            sorted_x_8b, sorted_expert_ids_8b, expert_counts_8b,
            W_gate_local_8b, W_up_local_8b, W_down_local_8b,
            expert_to_rank_8b, rank, world_size, gmm_fn,
        ),
        warmup=5, iters=30, label="expert-parallel-8b",
    )

    if rank == 0:
        speedup_8b = ms_rep_8b / ms_ep_8b if ms_ep_8b > 0 else float("inf")
        print(f"8B: Replicated:       {ms_rep_8b:.2f} ms/layer")
        print(f"8B: Expert-parallel:  {ms_ep_8b:.2f} ms/layer")
        print(f"8B: Speedup:          {speedup_8b:.2f}x")

        full_8b = E_8b * D_8b * H_8b * 2 * 3 / 1e9
        local_8b = E_local_8b * D_8b * H_8b * 2 * 3 / 1e9
        print(f"8B: VRAM savings:     {full_8b - local_8b:.2f} GB per GPU ({(1 - local_8b/full_8b)*100:.0f}%)")

    cleanup()


if __name__ == "__main__":
    main()
