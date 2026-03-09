#!/usr/bin/env python3
"""
Profile a single 8B MoE training step with kernel-level + MoE sub-component timing.

Usage:
  cd /mnt/local-nvme/LLM/experiments/tests/Test19/code
  deepspeed --num_gpus=8 ../scripts/profile_8b_step.py

Produces:
  1. torch.profiler table (top CUDA kernels by total GPU time)
  2. MoE sub-component breakdown (router, sort, grouped_gemm, scatter)
  3. Step-level breakdown (forward, backward, optimizer)
"""

import os
import sys
import time
import gc
import warnings

# Suppress pynvml warnings
warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning)

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import deepspeed
import torch
import yaml

# --- Paths (adjust if needed) ---
CODE_DIR = "/mnt/local-nvme/LLM/experiments/tests/Test19/code"
CONFIG_PATH = "/mnt/local-nvme/LLM/experiments/tests/Test19/benchmark/configs_phase2/p2_baseline_8b.yaml"
DS_CONFIG_PATH = "/mnt/local-nvme/LLM/experiments/tests/Test19/benchmark/deepspeed/8bmoe_baseline.json"

sys.path.insert(0, CODE_DIR)

from src.models.recurrence_model_8b_moe import (
    Model3B as Model8B_Rev,
    ModelConfig as ModelConfig_8B_Rev,
)
from src.utils import set_seed
from src.bin_idx_dataloader import build_bin_idx_dataloader


def print_rank_0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg, flush=True)


def patch_moe_timing(model):
    """
    Monkey-patch MoE forward to record sub-component CUDA event timers.
    Returns a dict that accumulates timing results.
    """
    timings = {
        "router": [], "sort": [], "grouped_gemm_fwd": [],
        "scatter": [], "shared_expert": [], "moe_total": [],
    }

    # Find all MoEFFN modules
    moe_modules = []
    for name, module in model.module.named_modules():
        if type(module).__name__ == "MoEFFN":
            moe_modules.append((name, module))

    if not moe_modules:
        print_rank_0("WARNING: No MoEFFN modules found!")
        return timings

    print_rank_0(f"Patching {len(moe_modules)} MoEFFN modules for timing...")

    def make_timed_forward(original_forward, mod_name):
        def timed_forward(self_ref, x):
            B, T, D = x.shape
            N = B * T
            K = self_ref.top_k
            E = self_ref.num_experts
            device, dtype = x.device, x.dtype

            # Events
            ev = lambda: torch.cuda.Event(enable_timing=True)
            e_start, e_router, e_sort, e_gemm, e_scatter, e_end = (
                ev(), ev(), ev(), ev(), ev(), ev()
            )

            e_start.record()

            # --- Shared expert ---
            x_cast = x.to(dtype=self_ref.shared_gate.weight.dtype)
            from src.models.liger_ops import liger_silu_mul
            shared_h = liger_silu_mul(self_ref.shared_gate(x_cast), self_ref.shared_up(x_cast))
            if self_ref.training and self_ref.dropout > 0:
                import torch.nn.functional as F
                shared_h = F.dropout(shared_h, p=self_ref.dropout)
            shared_out = self_ref.shared_down(shared_h)

            e_router.record()

            # --- Router ---
            topk_idx, topk_weight, is_null, aux_loss = self_ref.gate(x_cast)
            if self_ref.track_last_indices:
                self_ref.last_indices = topk_idx.detach()
            else:
                self_ref.last_indices = None

            flat_x = x_cast.reshape(N, D)
            if self_ref.permute_fusion_enabled:
                flat_idx_k = topk_idx.reshape(-1)
                flat_weight_k = topk_weight.reshape(-1)
                flat_is_null_k = is_null.reshape(-1)
                token_idx_k = torch.arange(N, device=device, dtype=torch.long).repeat_interleave(K)
                real_mask_k = ~flat_is_null_k
                real_token_indices = token_idx_k[real_mask_k]
                real_expert_indices = flat_idx_k[real_mask_k]
                real_weights = flat_weight_k[real_mask_k]
            else:
                flat_idx = topk_idx.view(N, K)
                flat_weight = topk_weight.view(N, K)
                flat_is_null = is_null.view(N, K)
                real_mask = ~flat_is_null
                token_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K)
                real_token_indices = token_indices[real_mask]
                real_expert_indices = flat_idx[real_mask]
                real_weights = flat_weight[real_mask]

            e_sort.record()

            # --- Sort by expert ---
            sort_idx = real_expert_indices.argsort()
            sorted_token_indices = real_token_indices[sort_idx]
            sorted_expert_indices = real_expert_indices[sort_idx]
            sorted_weights = real_weights[sort_idx]
            sorted_x = flat_x[sorted_token_indices]
            expert_counts = torch.bincount(sorted_expert_indices, minlength=E)

            num_real_assignments = sorted_token_indices.size(0)

            # --- Grouped GEMM ---
            if num_real_assignments > 0:
                if self_ref.active_moe_backend == "grouped_gemm":
                    try:
                        sorted_out = self_ref._moe_grouped(sorted_x, expert_counts)
                    except Exception:
                        sorted_out = self_ref._moe_vectorized(sorted_x, sorted_expert_indices)
                else:
                    sorted_out = self_ref._moe_vectorized(sorted_x, sorted_expert_indices)

                e_gemm.record()

                # --- Scatter/combine ---
                if self_ref.fast_scatter_enabled:
                    sorted_out = sorted_out.to(dtype=dtype)
                    sorted_out.mul_(sorted_weights.unsqueeze(-1).to(dtype=sorted_out.dtype))
                    routed_out = torch.zeros(N, D, device=device, dtype=dtype)
                    routed_out.index_add_(0, sorted_token_indices, sorted_out)
                else:
                    weighted_out = sorted_out * sorted_weights.unsqueeze(-1)
                    routed_out = torch.zeros(N, D, device=device, dtype=dtype)
                    routed_out.scatter_add_(
                        0, sorted_token_indices.unsqueeze(-1).expand(-1, D), weighted_out
                    )
            else:
                e_gemm.record()
                routed_out = torch.zeros(N, D, device=device, dtype=dtype)

            e_scatter.record()

            y = shared_out + routed_out.view(B, T, D)
            if self_ref.t4_enabled:
                _ = self_ref.t4_dispatcher

            e_end.record()

            # Store events for later timing extraction
            if not hasattr(self_ref, '_profile_events'):
                self_ref._profile_events = []
            self_ref._profile_events.append(
                (e_start, e_router, e_sort, e_gemm, e_scatter, e_end)
            )

            return y, aux_loss

        import types
        return timed_forward

    for mod_name, mod in moe_modules:
        mod._orig_forward = mod.forward
        mod.forward = make_timed_forward(mod.forward, mod_name).__get__(mod)
        mod._profile_events = []

    return moe_modules


def extract_moe_timings(moe_modules):
    """Extract timing from stored CUDA events after synchronization."""
    torch.cuda.synchronize()

    all_timings = []
    for mod_name, mod in moe_modules:
        for events in getattr(mod, '_profile_events', []):
            e_start, e_router, e_sort, e_gemm, e_scatter, e_end = events
            all_timings.append({
                "module": mod_name,
                "shared_expert_ms": e_start.elapsed_time(e_router),
                "router_ms": e_router.elapsed_time(e_sort),
                "sort_gemm_ms": e_sort.elapsed_time(e_gemm),
                "scatter_ms": e_gemm.elapsed_time(e_scatter),
                "total_ms": e_start.elapsed_time(e_end),
            })
        mod._profile_events = []  # Reset

    return all_timings


def main():
    # --- Parse args for DeepSpeed ---
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # --- Load config ---
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    set_seed(42)

    # --- Init model ---
    model_config = ModelConfig_8B_Rev()
    model = Model8B_Rev(model_config)

    total_params = sum(p.numel() for p in model.parameters())
    print_rank_0(f"\n8B Model: {total_params / 1e9:.2f}B parameters")
    print_rank_0(f"Architecture: {model_config.num_layers} layers, "
                 f"{model_config.num_real_experts} experts, top_k={model_config.top_k}")
    print_rank_0(f"Hidden: {model_config.hidden_size}, Expert FFN: {model_config.expert_intermediate_size}, "
                 f"Shared FFN: {model_config.shared_expert_intermediate_size}")

    # Override MoE backend to auto
    for name, module in model.named_modules():
        if hasattr(module, 'active_moe_backend'):
            module.active_moe_backend = "grouped_gemm"
            module.allow_vectorized_fallback = True

    # --- DeepSpeed init ---
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=DS_CONFIG_PATH,
    )

    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank}")

    # --- Build dataloader ---
    from src.data import build_dataloader_from_config
    train_loader = build_dataloader_from_config(cfg, split="train")

    # --- Patch MoE for timing ---
    moe_modules = patch_moe_timing(model_engine)

    # --- Warmup steps (3 steps, no profiling) ---
    print_rank_0("\n=== Warmup (3 steps) ===")
    model_engine.train()

    from src.models.liger_ops import LigerFusedLinearCrossEntropyLoss
    ce_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='mean')

    data_iter = iter(train_loader)
    for step in range(3):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model_engine(input_ids)
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            logits, aux_loss = outputs[0], outputs[1]
        else:
            logits, aux_loss = outputs, torch.tensor(0.0, device=device)

        # Simple loss
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        if isinstance(aux_loss, torch.Tensor):
            loss = loss + aux_loss

        model_engine.backward(loss)
        model_engine.step()

        # Clear MoE events from warmup
        for _, mod in moe_modules:
            mod._profile_events = []

    torch.cuda.synchronize()
    torch.distributed.barrier()
    gc.collect()
    torch.cuda.empty_cache()

    print_rank_0("\n=== Profiling Step (with torch.profiler) ===")

    # --- Profile 1 step with torch.profiler for kernel breakdown ---
    batch = next(data_iter)
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        # Step-level timing
        torch.cuda.synchronize()
        t_step_start = time.perf_counter()

        # Forward
        torch.cuda.synchronize()
        t_fwd_start = time.perf_counter()
        outputs = model_engine(input_ids)
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            logits, aux_loss = outputs[0], outputs[1]
        else:
            logits, aux_loss = outputs, torch.tensor(0.0, device=device)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        if isinstance(aux_loss, torch.Tensor):
            loss = loss + aux_loss
        torch.cuda.synchronize()
        t_fwd_end = time.perf_counter()

        # Backward
        t_bwd_start = time.perf_counter()
        model_engine.backward(loss)
        torch.cuda.synchronize()
        t_bwd_end = time.perf_counter()

        # Optimizer step
        t_opt_start = time.perf_counter()
        model_engine.step()
        torch.cuda.synchronize()
        t_opt_end = time.perf_counter()

        t_step_end = time.perf_counter()

    # --- Extract MoE sub-component timings ---
    moe_timings = extract_moe_timings(moe_modules)

    if rank == 0:
        fwd_ms = (t_fwd_end - t_fwd_start) * 1000
        bwd_ms = (t_bwd_end - t_bwd_start) * 1000
        opt_ms = (t_opt_end - t_opt_start) * 1000
        step_ms = (t_step_end - t_step_start) * 1000

        print(f"\n{'='*80}")
        print(f"STEP-LEVEL BREAKDOWN (wall-clock, rank 0)")
        print(f"{'='*80}")
        print(f"  Forward:    {fwd_ms:8.1f} ms  ({fwd_ms/step_ms*100:5.1f}%)")
        print(f"  Backward:   {bwd_ms:8.1f} ms  ({bwd_ms/step_ms*100:5.1f}%)")
        print(f"  Optimizer:  {opt_ms:8.1f} ms  ({opt_ms/step_ms*100:5.1f}%)")
        print(f"  Total step: {step_ms:8.1f} ms")
        tokens = input_ids.numel() * torch.distributed.get_world_size()
        print(f"  Throughput: {tokens / (step_ms / 1000):.0f} tok/s")

        print(f"\n{'='*80}")
        print(f"MoE SUB-COMPONENT BREAKDOWN (CUDA events, per-layer average)")
        print(f"{'='*80}")

        if moe_timings:
            # Aggregate per-layer
            from collections import defaultdict
            agg = defaultdict(lambda: {"shared_expert": 0, "router": 0, "sort_gemm": 0, "scatter": 0, "total": 0, "count": 0})
            for t in moe_timings:
                key = t["module"]
                agg[key]["shared_expert"] += t["shared_expert_ms"]
                agg[key]["router"] += t["router_ms"]
                agg[key]["sort_gemm"] += t["sort_gemm_ms"]
                agg[key]["scatter"] += t["scatter_ms"]
                agg[key]["total"] += t["total_ms"]
                agg[key]["count"] += 1

            total_moe_ms = 0
            total_shared = 0
            total_router = 0
            total_gemm = 0
            total_scatter = 0

            print(f"{'Layer':<45} {'Shared':>8} {'Router':>8} {'SortGMM':>8} {'Scatter':>8} {'Total':>8}")
            print("-" * 93)
            for key in sorted(agg.keys()):
                v = agg[key]
                n = v["count"]
                se = v["shared_expert"] / n
                ro = v["router"] / n
                sg = v["sort_gemm"] / n
                sc = v["scatter"] / n
                tot = v["total"] / n
                total_moe_ms += v["total"]
                total_shared += v["shared_expert"]
                total_router += v["router"]
                total_gemm += v["sort_gemm"]
                total_scatter += v["scatter"]
                print(f"{key:<45} {se:7.2f}  {ro:7.2f}  {sg:7.2f}  {sc:7.2f}  {tot:7.2f}")

            num_layers = len(agg)
            print("-" * 93)
            print(f"{'ALL MoE LAYERS TOTAL':<45} {total_shared:7.1f}  {total_router:7.1f}  {total_gemm:7.1f}  {total_scatter:7.1f}  {total_moe_ms:7.1f}")
            print(f"{'ALL MoE LAYERS (% of fwd)':<45} {total_shared/fwd_ms*100:6.1f}%  {total_router/fwd_ms*100:6.1f}%  {total_gemm/fwd_ms*100:6.1f}%  {total_scatter/fwd_ms*100:6.1f}%  {total_moe_ms/fwd_ms*100:6.1f}%")

        # --- Kernel-level breakdown from torch.profiler ---
        print(f"\n{'='*80}")
        print(f"TOP 40 CUDA KERNELS (by total GPU time)")
        print(f"{'='*80}")
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=40,
            top_level_events_only=False,
        ))

        # --- Categorized kernel summary ---
        print(f"\n{'='*80}")
        print(f"CATEGORIZED KERNEL SUMMARY")
        print(f"{'='*80}")

        events = prof.key_averages()
        categories = {
            "NCCL (allgather/reduce)": [],
            "GEMM/MatMul": [],
            "Elementwise (add/mul/silu/...)": [],
            "Memory (copy/fill/index)": [],
            "Triton kernels": [],
            "Softmax/Norm": [],
            "Other": [],
        }

        for evt in events:
            name = evt.key.lower()
            cuda_us = evt.cuda_time_total
            if cuda_us == 0:
                continue
            if "nccl" in name:
                categories["NCCL (allgather/reduce)"].append((evt.key, cuda_us))
            elif any(k in name for k in ["gemm", "gemv", "matmul", "cublas", "gmm"]):
                categories["GEMM/MatMul"].append((evt.key, cuda_us))
            elif any(k in name for k in ["add", "mul", "silu", "gelu", "relu", "sigmoid", "tanh", "elementwise", "fused"]):
                categories["Elementwise (add/mul/silu/...)"].append((evt.key, cuda_us))
            elif any(k in name for k in ["copy", "fill", "index", "scatter", "gather", "cat", "memcpy", "memset"]):
                categories["Memory (copy/fill/index)"].append((evt.key, cuda_us))
            elif "triton" in name or "kernel" in name:
                categories["Triton kernels"].append((evt.key, cuda_us))
            elif any(k in name for k in ["softmax", "norm", "layer_norm", "rms"]):
                categories["Softmax/Norm"].append((evt.key, cuda_us))
            else:
                categories["Other"].append((evt.key, cuda_us))

        total_cuda_us = sum(
            cuda_us for cat_list in categories.values() for _, cuda_us in cat_list
        )
        for cat_name, items in sorted(categories.items(), key=lambda x: -sum(v for _, v in x[1])):
            cat_total = sum(v for _, v in items)
            if cat_total == 0:
                continue
            pct = cat_total / total_cuda_us * 100 if total_cuda_us > 0 else 0
            print(f"\n  {cat_name}: {cat_total/1000:.1f} ms ({pct:.1f}%)")
            for kname, ktime in sorted(items, key=lambda x: -x[1])[:5]:
                print(f"    {ktime/1000:8.1f} ms  {kname}")

    # --- Run 2 more profiled steps for MoE-only timing (no torch.profiler overhead) ---
    print_rank_0(f"\n{'='*80}")
    print_rank_0("MoE TIMING OVER 3 ADDITIONAL STEPS (no profiler overhead)")
    print_rank_0(f"{'='*80}")

    for _, mod in moe_modules:
        mod._profile_events = []

    step_times = []
    for step in range(3):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model_engine(input_ids)
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            logits, aux_loss = outputs[0], outputs[1]
        else:
            logits, aux_loss = outputs, torch.tensor(0.0, device=device)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        if isinstance(aux_loss, torch.Tensor):
            loss = loss + aux_loss

        model_engine.backward(loss)
        model_engine.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append((t1 - t0) * 1000)

        gc.collect()
        torch.cuda.empty_cache()

    moe_timings2 = extract_moe_timings(moe_modules)

    if rank == 0 and moe_timings2:
        from collections import defaultdict
        agg2 = defaultdict(lambda: {"shared_expert": 0, "router": 0, "sort_gemm": 0, "scatter": 0, "total": 0, "count": 0})
        for t in moe_timings2:
            key = t["module"]
            agg2[key]["shared_expert"] += t["shared_expert_ms"]
            agg2[key]["router"] += t["router_ms"]
            agg2[key]["sort_gemm"] += t["sort_gemm_ms"]
            agg2[key]["scatter"] += t["scatter_ms"]
            agg2[key]["total"] += t["total_ms"]
            agg2[key]["count"] += 1

        avg_step_ms = sum(step_times) / len(step_times)
        total_moe_ms = sum(v["total"] for v in agg2.values()) / 3  # per step avg

        # Per-step averages
        total_se = sum(v["shared_expert"] for v in agg2.values()) / 3
        total_ro = sum(v["router"] for v in agg2.values()) / 3
        total_sg = sum(v["sort_gemm"] for v in agg2.values()) / 3
        total_sc = sum(v["scatter"] for v in agg2.values()) / 3

        print(f"\nAverage step time: {avg_step_ms:.1f} ms")
        tokens = input_ids.numel() * torch.distributed.get_world_size()
        print(f"Throughput: {tokens / (avg_step_ms / 1000):.0f} tok/s")
        print(f"\nMoE forward per step (all {len(agg2)} layers):")
        print(f"  Shared expert:     {total_se:7.1f} ms  ({total_se/avg_step_ms*100:5.1f}% of step)")
        print(f"  Router:            {total_ro:7.1f} ms  ({total_ro/avg_step_ms*100:5.1f}% of step)")
        print(f"  Sort + GroupedGEMM:{total_sg:7.1f} ms  ({total_sg/avg_step_ms*100:5.1f}% of step)")
        print(f"  Scatter/combine:   {total_sc:7.1f} ms  ({total_sc/avg_step_ms*100:5.1f}% of step)")
        print(f"  MoE total (fwd):   {total_moe_ms:7.1f} ms  ({total_moe_ms/avg_step_ms*100:5.1f}% of step)")
        non_moe = avg_step_ms - total_moe_ms
        print(f"  Non-MoE (attn+bwd+opt+comm): {non_moe:.1f} ms  ({non_moe/avg_step_ms*100:.1f}% of step)")

    print_rank_0("\nDone.")


if __name__ == "__main__":
    main()
