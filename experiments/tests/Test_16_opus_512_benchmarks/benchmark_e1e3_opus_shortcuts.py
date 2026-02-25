"""
Benchmark E1-E3: OPUS Scoring Shortcuts
========================================

OPUS scoring only needs hidden-state gradients, NOT logits or CE loss.
This benchmark measures how much time we save by skipping components
that are unnecessary during OPUS scoring:

  E1: Skip MTP block entirely
  E2: Skip CE computation (not needed — OPUS uses gradient sketches, not loss)
  E3: Skip lm_head (4096→131072 matmul — saves the biggest single operation)

Each test measures a backbone-only forward+backward to quantify savings.
Since the model requires Kronecker embeddings and complex setup, we
benchmark using isolated components that replicate the model's compute.

Usage:
    python benchmark_e1e3_opus_shortcuts.py --dtype bf16     # AWS
    python benchmark_e1e3_opus_shortcuts.py --dtype fp16     # Colab
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Resolve imports ──────────────────────────────────────────────────────────

def _setup_imports():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    candidates = [
        os.path.join(repo_root, "experiments", "tests",
                     "Test_14_gsa_only_liger_kernels_1000steps-28k", "code"),
        os.path.join(repo_root, "experiments", "tests",
                     "Test_14_gsa_only_liger_kernels_1000steps-OngoingRun3", "code"),
    ]
    for code_dir in candidates:
        if os.path.isdir(os.path.join(code_dir, "src", "kernels")):
            if code_dir not in sys.path:
                sys.path.insert(0, code_dir)
            print(f"  Using codebase: {os.path.basename(os.path.dirname(code_dir))}")
            return code_dir
    return None

CODE_DIR = _setup_imports()


@dataclass
class BenchConfig:
    batch_size: int = 4
    hidden_size: int = 4096
    vocab_size: int = 131072    # 2^17
    seq_len: int = 512          # OPUS scoring length
    warmup_iters: int = 10
    bench_iters: int = 50
    dtype_str: str = "bf16"

    @property
    def dtype(self):
        return torch.bfloat16 if self.dtype_str == "bf16" else torch.float16


# ── Benchmarking Utility ─────────────────────────────────────────────────────

def bench_fn(fn, warmup, iters, label):
    """Benchmark a callable fn() that returns a tensor."""
    for _ in range(warmup):
        fn()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    drop = max(1, iters // 10)
    times = times[drop:]
    return {
        "label": label,
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_mem_gb": peak_mem,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  E3: lm_head COST — the single biggest skippable operation
# ══════════════════════════════════════════════════════════════════════════════

def run_e3_lm_head(cfg):
    """Measure forward+backward cost of lm_head (4096 → 131072)."""
    device = torch.device("cuda")
    dtype = cfg.dtype
    B, T, D, V = cfg.batch_size, cfg.seq_len, cfg.hidden_size, cfg.vocab_size

    print(f"\n{'━'*80}")
    print(f"  E3: lm_head Cost (Linear {D} → {V})")
    print(f"  This is the single biggest operation that OPUS scoring can skip.")
    print(f"  Hidden: [B={B}, T={T}, D={D}] → Logits: [B={B}, T={T}, V={V}]")
    print(f"{'━'*80}")

    lm_head = nn.Linear(D, V, bias=False, device=device, dtype=dtype)
    h = torch.randn(B, T, D, device=device, dtype=dtype)

    # Forward only
    def fwd_only():
        h_in = h.clone().requires_grad_(True)
        logits = lm_head(h_in)
        return logits

    # Forward + backward
    def fwd_bwd():
        h_in = h.clone().requires_grad_(True)
        logits = lm_head(h_in)
        logits.sum().backward()

    fwd_result = bench_fn(fwd_only, cfg.warmup_iters, cfg.bench_iters, "lm_head_fwd")
    fwd_bwd_result = bench_fn(fwd_bwd, cfg.warmup_iters, cfg.bench_iters, "lm_head_fwd+bwd")

    print(f"\n  lm_head forward only:    {fwd_result['avg_ms']:>8.2f} ms  |  Mem: {fwd_result['peak_mem_gb']:.2f} GB")
    print(f"  lm_head forward+backward: {fwd_bwd_result['avg_ms']:>8.2f} ms  |  Mem: {fwd_bwd_result['peak_mem_gb']:.2f} GB")
    print(f"\n  → Skipping lm_head saves {fwd_bwd_result['avg_ms']:.2f} ms per scoring pass")
    print(f"  → Memory saved: {fwd_bwd_result['peak_mem_gb']:.2f} GB (B*T*V = {B*T*V/1e9:.2f}B floats)")

    del lm_head
    torch.cuda.empty_cache()
    return {"fwd_ms": fwd_result['avg_ms'], "fwd_bwd_ms": fwd_bwd_result['avg_ms'],
            "peak_mem_gb": fwd_bwd_result['peak_mem_gb']}


# ══════════════════════════════════════════════════════════════════════════════
#  E2: CE LOSS COST
# ══════════════════════════════════════════════════════════════════════════════

def run_e2_ce_loss(cfg):
    """Measure cross-entropy cost on logits [B*T, V]."""
    device = torch.device("cuda")
    dtype = cfg.dtype
    B, T, V = cfg.batch_size, cfg.seq_len, cfg.vocab_size

    print(f"\n{'━'*80}")
    print(f"  E2: Cross-Entropy Cost on Logits [{B*T}, {V}]")
    print(f"  OPUS scoring doesn't need CE at all — it uses gradient sketches.")
    print(f"{'━'*80}")

    logits = torch.randn(B * T, V, device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, V, (B * T,), device=device)

    # Standard CE
    def ce_fwd_bwd():
        l = logits.clone().requires_grad_(True)
        loss = F.cross_entropy(l, targets)
        loss.backward()

    ce_result = bench_fn(ce_fwd_bwd, cfg.warmup_iters, cfg.bench_iters, "ce_fwd+bwd")

    # Fused CE (if available)
    fused_result = None
    try:
        from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
        lm_head_weight = torch.randn(V, cfg.hidden_size, device=device, dtype=dtype)
        h = torch.randn(B * T, cfg.hidden_size, device=device, dtype=dtype)

        fused_ce = LigerFusedLinearCrossEntropyLoss()

        def fused_ce_fwd_bwd():
            h_in = h.clone().requires_grad_(True)
            loss = fused_ce(h_in, lm_head_weight, targets)
            loss.backward()

        fused_result = bench_fn(fused_ce_fwd_bwd, cfg.warmup_iters, cfg.bench_iters, "fused_ce_fwd+bwd")
    except Exception as e:
        print(f"  (Fused CE unavailable: {e})")

    print(f"\n  Standard CE (fwd+bwd):   {ce_result['avg_ms']:>8.2f} ms  |  Mem: {ce_result['peak_mem_gb']:.2f} GB")
    if fused_result:
        print(f"  Fused CE (fwd+bwd):      {fused_result['avg_ms']:>8.2f} ms  |  Mem: {fused_result['peak_mem_gb']:.2f} GB")
    print(f"\n  → Skipping CE saves {ce_result['avg_ms']:.2f} ms per scoring pass")

    torch.cuda.empty_cache()
    result = {"ce_ms": ce_result['avg_ms'], "ce_mem_gb": ce_result['peak_mem_gb']}
    if fused_result:
        result["fused_ce_ms"] = fused_result['avg_ms']
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  E1: MTP BLOCK COST
# ══════════════════════════════════════════════════════════════════════════════

def run_e1_mtp(cfg):
    """Measure cost of MTP block (GSA + MLP + fusion + lm_head for t+2)."""
    device = torch.device("cuda")
    dtype = cfg.dtype
    B, T, D, V = cfg.batch_size, cfg.seq_len, cfg.hidden_size, cfg.vocab_size

    print(f"\n{'━'*80}")
    print(f"  E1: MTP Block Cost (fusion + GSA + MLP + lm_head)")
    print(f"  MTP predicts t+2 tokens — completely unnecessary for OPUS scoring.")
    print(f"{'━'*80}")

    # Try to import real MTP block
    mtp = None
    mtp_label = "MTP-Sim"
    try:
        from src.models.recurrence_model_1b import MTPTransformerBlock, ModelConfig
        config = ModelConfig()
        mtp = MTPTransformerBlock(config).to(device=device, dtype=dtype)
        mtp_label = "MTP-Real"
        print("  ✅ Using REAL MTPTransformerBlock")
    except Exception as e:
        print(f"  ⚠️  Real MTP unavailable: {e}")
        print("  Using simulated MTP cost (fusion + attn + MLP)")

        # Simulate MTP cost: fusion_proj + attention layer + MLP + lm_head
        class MTPSimulated(nn.Module):
            def __init__(self):
                super().__init__()
                self.fusion = nn.Linear(D * 2, D, bias=False, device=device, dtype=dtype)
                self.attn_q = nn.Linear(D, D, bias=False, device=device, dtype=dtype)
                self.attn_k = nn.Linear(D, D, bias=False, device=device, dtype=dtype)
                self.attn_v = nn.Linear(D, D, bias=False, device=device, dtype=dtype)
                self.attn_o = nn.Linear(D, D, bias=False, device=device, dtype=dtype)
                self.mlp_gate = nn.Linear(D, D * 2, bias=False, device=device, dtype=dtype)
                self.mlp_up = nn.Linear(D, D * 2, bias=False, device=device, dtype=dtype)
                self.mlp_down = nn.Linear(D * 2, D, bias=False, device=device, dtype=dtype)
            def forward(self, h_t, next_emb):
                x = self.fusion(torch.cat([h_t, next_emb], dim=-1))
                H = 16
                q = self.attn_q(x).view(B, T, H, D // H).transpose(1, 2)
                k = self.attn_k(x).view(B, T, H, D // H).transpose(1, 2)
                v = self.attn_v(x).view(B, T, H, D // H).transpose(1, 2)
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                x = x + self.attn_o(o.transpose(1, 2).reshape(B, T, D))
                gate = F.silu(self.mlp_gate(x))
                up = self.mlp_up(x)
                x = x + self.mlp_down(gate * up)
                return x

        mtp = MTPSimulated()

    # Also need lm_head for MTP (it projects to vocab)
    lm_head = nn.Linear(D, V, bias=False, device=device, dtype=dtype)

    h_t = torch.randn(B, T, D, device=device, dtype=dtype)
    next_emb = torch.randn(B, T, D, device=device, dtype=dtype)

    def mtp_fwd_bwd():
        h_in = h_t.clone().requires_grad_(True)
        next_in = next_emb.clone().requires_grad_(True)
        h_mtp = mtp(h_in, next_in)
        logits_mtp = lm_head(h_mtp)
        logits_mtp.sum().backward()

    def mtp_no_lm_head():
        h_in = h_t.clone().requires_grad_(True)
        next_in = next_emb.clone().requires_grad_(True)
        h_mtp = mtp(h_in, next_in)
        h_mtp.sum().backward()

    full_result = bench_fn(mtp_fwd_bwd, cfg.warmup_iters, cfg.bench_iters, "mtp_full")
    no_head_result = bench_fn(mtp_no_lm_head, cfg.warmup_iters, cfg.bench_iters, "mtp_no_head")

    print(f"\n  MTP block + lm_head (fwd+bwd): {full_result['avg_ms']:>8.2f} ms  ({mtp_label})")
    print(f"  MTP block only (fwd+bwd):      {no_head_result['avg_ms']:>8.2f} ms")
    print(f"  lm_head portion:               {full_result['avg_ms'] - no_head_result['avg_ms']:>8.2f} ms")
    print(f"\n  → Skipping entire MTP saves {full_result['avg_ms']:.2f} ms per scoring pass")

    del mtp, lm_head
    torch.cuda.empty_cache()
    return {"mtp_full_ms": full_result['avg_ms'], "mtp_block_ms": no_head_result['avg_ms'],
            "peak_mem_gb": full_result['peak_mem_gb']}


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(cfg):
    device = torch.device("cuda")
    print(f"\n{'='*80}")
    print(f"  BENCHMARK E1-E3: OPUS Scoring Shortcuts")
    print(f"  What can we skip during OPUS scoring at T={cfg.seq_len}?")
    print(f"  GPU: {torch.cuda.get_device_name()} | B={cfg.batch_size} | dtype={cfg.dtype_str}")
    print(f"{'='*80}")

    e3_result = run_e3_lm_head(cfg)
    e2_result = run_e2_ce_loss(cfg)
    e1_result = run_e1_mtp(cfg)

    # Backbone cost (8 decoder layers at T=512)
    # We estimate this from the total scoring time minus the skippable parts
    total_skippable = (
        e3_result['fwd_bwd_ms'] +      # E3: lm_head (NTP)
        e2_result['ce_ms'] +            # E2: CE loss (NTP)
        e1_result['mtp_full_ms']        # E1: MTP block + its lm_head + its CE
    )

    # Also account for NTP CE separately from MTP CE
    total_skippable_with_mtp_ce = total_skippable + e2_result['ce_ms']  # MTP also has its own CE

    print(f"\n{'='*80}")
    print(f"  SUMMARY: Total Savings from OPUS Scoring Shortcuts")
    print(f"{'='*80}")
    print(f"")
    print(f"  Component               │  Time (ms)  │  Action for OPUS Scoring")
    print(f"  ────────────────────────┼─────────────┼──────────────────────────────")
    print(f"  E3: lm_head (NTP)       │  {e3_result['fwd_bwd_ms']:>9.2f}  │  SKIP — use return_hidden=True ✂️")
    print(f"  E2: CE loss (NTP)       │  {e2_result['ce_ms']:>9.2f}  │  SKIP — OPUS uses grad sketches ✂️")
    print(f"  E1: MTP block + head    │  {e1_result['mtp_full_ms']:>9.2f}  │  SKIP — not needed for scoring ✂️")
    print(f"  E2: CE loss (MTP)       │  {e2_result['ce_ms']:>9.2f}  │  SKIP — comes with skipping MTP ✂️")
    print(f"  ────────────────────────┼─────────────┼──────────────────────────────")
    print(f"  TOTAL SAVED             │  {total_skippable_with_mtp_ce:>9.2f}  │  Per scoring forward+backward")
    print(f"")

    # Context: from profiler, total step at 4096 = 5463ms
    # At 512, backbone should be ~8× faster ≈ 680ms
    # So if we save ~300ms from shortcuts, that's a huge fraction
    print(f"  For reference (rough estimate):")
    print(f"    Backbone at 512 (8 layers) ≈ est. 600-800ms")
    print(f"    Saved from shortcuts:        ≈ {total_skippable_with_mtp_ce:.0f}ms")
    print(f"    OPUS scoring pass:          ≈ {total_skippable_with_mtp_ce + 700:.0f}ms → ~700ms with shortcuts")
    print(f"")
    print(f"  These shortcuts don't change the math — OPUS scoring only needs")
    print(f"  dL/d(activations), not logits or CE loss. The model already has")
    print(f"  return_hidden=True for this purpose.")

    # Save
    os.makedirs("results", exist_ok=True)
    out_path = "results/e1e3_opus_shortcuts.json"
    with open(out_path, "w") as f:
        json.dump({
            "benchmark": "E1E3_opus_shortcuts",
            "config": {
                "batch_size": cfg.batch_size, "seq_len": cfg.seq_len,
                "hidden_size": cfg.hidden_size, "vocab_size": cfg.vocab_size,
                "dtype": cfg.dtype_str, "gpu": torch.cuda.get_device_name(),
            },
            "e3_lm_head": e3_result,
            "e2_ce_loss": e2_result,
            "e1_mtp": e1_result,
            "total_skippable_ms": total_skippable_with_mtp_ce,
        }, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E1-E3: OPUS Scoring Shortcuts")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    cfg = BenchConfig(
        batch_size=args.batch_size, seq_len=args.seq_len,
        warmup_iters=args.warmup, bench_iters=args.iters,
        dtype_str=args.dtype,
    )

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        exit(1)

    run_benchmark(cfg)
