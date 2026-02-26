#!/usr/bin/env python3
"""
AWS Compile Test — Verify torch.compile works with Model1B at full scale
========================================================================

Run on AWS (single GPU, no DeepSpeed):
    python test_compile.py

This tests:
1. Model creates correctly (standard embeddings, no Kronecker)
2. enable_torch_compile() applies without error
3. Forward pass compiles at FULL training scale (BS=4, seq_len=4096)
4. Backward pass compiles (MidpointFunction.backward recomputes force())
5. Reports graph break count and compilation status

Uses BS=4, seq_len=4096 to match real training dimensions.
No DeepSpeed, no data loading, no checkpoints — pure compile test.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch._dynamo

# ── Check environment ──────────────────────────────────────────────────────
print("=" * 70)
print("AWS torch.compile Test for Model1B (full scale)")
print("=" * 70)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory: {mem_gb:.1f} GB")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {cap[0]}.{cap[1]}")
else:
    print("ERROR: No GPU found.")
    sys.exit(1)

try:
    import triton
    print(f"Triton: {triton.__version__}")
except ImportError:
    print("Triton: NOT FOUND (will use PyTorch fallbacks)")

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    print("FLA: OK")
except ImportError:
    print("ERROR: FLA not found. Run: pip install fla==0.4.7")
    sys.exit(1)

print("=" * 70)

# ── Import model ───────────────────────────────────────────────────────────
print("\n[1/6] Importing model...")
from src.models.recurrence_model_1b import (
    Model1B,
    ModelConfig,
    enable_torch_compile,
)
print("  OK")

# ── Create model ───────────────────────────────────────────────────────────
print("\n[2/6] Creating model (standard embeddings, no Kronecker)...")
config = ModelConfig()
config.vocab_size = 1024  # Small vocab (no tokenizer needed)

model = Model1B(config=config, embedding_type="standard")
model = model.to(dtype=torch.bfloat16, device="cuda")

total_params = sum(p.numel() for p in model.parameters())
print(f"  Model: {total_params / 1e9:.3f}B parameters")
print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ── Apply torch.compile ───────────────────────────────────────────────────
print("\n[3/6] Applying enable_torch_compile(mode='default')...")
torch._dynamo.reset()
model = enable_torch_compile(model, compile_mode="default")
print("  OK")

# ── Forward pass at FULL training scale ───────────────────────────────────
BS = 4
SEQ_LEN = 4096  # Full training sequence length
VOCAB = config.vocab_size

print(f"\n[4/6] Forward pass (BS={BS}, seq_len={SEQ_LEN}) — triggers JIT compilation...")
print(f"  Input tokens: {BS * (SEQ_LEN - 2):,} (matches training warmup)")

dummy_ids = torch.randint(0, VOCAB, (BS, SEQ_LEN), device="cuda")
dummy_mask = torch.ones(BS, SEQ_LEN, dtype=torch.long, device="cuda")

x_input = dummy_ids[:, :-2].contiguous()
y_ntp = dummy_ids[:, 1:-1].contiguous()
attn_mask = dummy_mask[:, :-2].contiguous()

model.train()
torch.cuda.synchronize()
t0 = time.time()

try:
    h_ntp, h_mtp, aux_loss = model(
        x_input,
        next_token_ids=y_ntp,
        attention_mask=attn_mask,
        return_loss=True,
        return_memory=False,
        prev_memory_stream=None,
        return_hidden=True,
    )
    torch.cuda.synchronize()
    fwd_time = time.time() - t0
    print(f"  FORWARD OK ({fwd_time:.1f}s)")
    print(f"  h_ntp shape: {h_ntp.shape}")
    print(f"  h_mtp shape: {h_mtp.shape if h_mtp is not None else 'None'}")
    if aux_loss is not None:
        print(f"  aux_loss: {aux_loss.item():.6f}")
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  GPU peak:   {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
except Exception as e:
    print(f"  FORWARD FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Backward pass ─────────────────────────────────────────────────────────
print("\n[5/6] Backward pass — triggers backward JIT compilation...")
loss = h_ntp.sum()
if h_mtp is not None:
    loss = loss + h_mtp.sum()
if aux_loss is not None and aux_loss.numel() > 0:
    loss = loss + aux_loss.mean()

torch.cuda.synchronize()
t0 = time.time()
try:
    loss.backward()
    torch.cuda.synchronize()
    bwd_time = time.time() - t0
    print(f"  BACKWARD OK ({bwd_time:.1f}s)")
except Exception as e:
    print(f"  BACKWARD FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Second forward+backward (compiled, no JIT overhead) ──────────────────
print("\n[6/6] Second pass (should be fast — already compiled)...")
model.zero_grad()
del h_ntp, h_mtp, aux_loss, loss
torch.cuda.empty_cache()

dummy_ids2 = torch.randint(0, VOCAB, (BS, SEQ_LEN), device="cuda")
dummy_mask2 = torch.ones(BS, SEQ_LEN, dtype=torch.long, device="cuda")
x2 = dummy_ids2[:, :-2].contiguous()
y2 = dummy_ids2[:, 1:-1].contiguous()
m2 = dummy_mask2[:, :-2].contiguous()

torch.cuda.synchronize()
t0 = time.time()
h_ntp2, h_mtp2, aux2 = model(
    x2, next_token_ids=y2, attention_mask=m2,
    return_loss=True, return_memory=False, prev_memory_stream=None, return_hidden=True,
)
loss2 = h_ntp2.sum()
if h_mtp2 is not None:
    loss2 = loss2 + h_mtp2.sum()
if aux2 is not None and aux2.numel() > 0:
    loss2 = loss2 + aux2.mean()
loss2.backward()
torch.cuda.synchronize()
step2_time = time.time() - t0
print(f"  Second fwd+bwd: {step2_time:.1f}s")

# ── Gradient check ────────────────────────────────────────────────────────
has_grad = sum(1 for p in model.parameters() if p.grad is not None)
total_p = sum(1 for p in model.parameters())
print(f"  Parameters with gradients: {has_grad}/{total_p}")

# ── Graph break report ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPILATION REPORT")
print("=" * 70)

counters = dict(torch._dynamo.utils.counters)
graph_breaks = counters.get("graph_break", {})
if graph_breaks:
    print(f"\nGraph breaks ({sum(graph_breaks.values())} total):")
    for reason, count in sorted(graph_breaks.items(), key=lambda x: -x[1]):
        print(f"  [{count}x] {reason[:120]}")
else:
    print("\nNo graph breaks detected")

compile_stats = counters.get("stats", {})
if compile_stats:
    print(f"\nCompilation stats:")
    for key, val in sorted(compile_stats.items()):
        print(f"  {key}: {val}")

print(f"\nGPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU peak:   {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print("\n" + "=" * 70)
print("RESULT: torch.compile WORKS at full training scale")
print("=" * 70)
print(f"\nCompilation time (first pass): fwd={fwd_time:.1f}s + bwd={bwd_time:.1f}s")
print(f"Compiled step time (second pass): {step2_time:.1f}s")
print()

# ── Cleanup ───────────────────────────────────────────────────────────────
del model
torch.cuda.empty_cache()
