"""
Synthetic leak test using the REAL Test19 3B MoE model code.
Bypasses data loading, tokenizer, checkpointing entirely.
Sends random token IDs through the model under ZeRO-3.

Config: 4 DeltaNet layers, 4 real MoE experts, d_model=512, seq=1024, BS=1
Run on GCP L4 (1 GPU):
  cd ~/test19/src
  export PATH=$PATH:~/.local/bin
  deepspeed --num_gpus 1 run_synth.py
"""

import os
import sys
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

# Pre-load kernels package so relative imports inside models/ don't fail
try:
    import kernels as _km
    sys.modules.setdefault("src.kernels", _km)
    print(f"[init] kernels loaded")
except Exception as _e:
    print(f"[init] kernels not found: {_e}")

# Ensure user site-packages are on path (grouped_gemm, fla)
import site
for _sp in (site.getusersitepackages() if hasattr(site, "getusersitepackages") else []):
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

# Use fp32 cast path into FLA to avoid bf16 Triton compilation issues on this VM
os.environ.setdefault("T17_DN_FLA_NATIVE_DTYPE", "0")

from models.recurrence_model_3b_moe import Model3B, ModelConfig


def build_config():
    """
    Test19 model scaled to fit a single L4 (24GB VRAM, 32GB RAM):
      - 4 DeltaNet layers (no GSA)
      - 4 real MoE experts + 4 null = 8 total slots
      - d_model=512, seq_len=1024, batch=1
    """
    cfg = ModelConfig()

    # Core dims
    cfg.hidden_size              = 512
    cfg.num_layers               = 4
    cfg.vocab_size               = 4096

    # 4 DeltaNet layers, 0 GSA
    cfg.num_deltanet_layers      = 4
    cfg.num_gsa_layers           = 0
    cfg.delta_v_heads            = 4        # 512 / 128
    cfg.delta_head_dim           = 128
    cfg.delta_gate_dim           = 64

    # GSA not used — zero it out to avoid init errors
    cfg.gsa_num_heads            = 2
    cfg.gsa_head_dim             = 64
    cfg.gsa_k_base               = 16
    cfg.gsa_k_min                = 4
    cfg.gsa_k_max                = 32
    cfg.gsa_indexer_heads        = 1

    # MoE: 4 real + 4 null, top-k=2
    cfg.num_real_experts         = 4
    cfg.num_null_experts         = 4
    cfg.total_expert_slots       = 8
    cfg.top_k                    = 2
    cfg.expert_intermediate_size         = 128
    cfg.shared_expert_intermediate_size  = 256
    cfg.data_sparsity            = 0.5
    cfg.moe_backend              = "auto"   # grouped_gemm if available, else vectorized
    cfg.require_fused_moe_kernel         = False
    cfg.allow_moe_vectorized_fallback    = True

    # mHC stream routing
    cfg.n_streams                = 4
    cfg.sinkhorn_iters           = 3

    # Disable MTP, keep reversible stack
    cfg.enable_mtp               = False
    cfg.mtp_reversible           = False

    # RoPE / context
    cfg.max_seq_len              = 8192
    cfg.rope_base                = 10000
    cfg.rope_original_max_position = 4096
    cfg.rope_scaling_factor      = 1.0
    cfg.dropout                  = 0.0

    # Allow kernel fallbacks for this test environment
    cfg.require_fused_deltanet_kernel = False
    cfg.require_fused_gsa_kernel      = False

    return cfg


def main():
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    is_rank0   = (local_rank == 0)

    SEQ_LEN    = 1024
    BATCH      = 1
    NUM_STEPS  = 10

    ds_config = {
        "train_batch_size":                BATCH,
        "train_micro_batch_size_per_gpu":  BATCH,
        "gradient_accumulation_steps":     1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_param":     {"device": "cpu", "pin_memory": True},
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "overlap_comm":      True,
            "contiguous_gradients": True,
            "reduce_bucket_size":          5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e4,
        },
    }

    if is_rank0:
        print("=" * 60)
        print("Test19 3B MoE — Synthetic Leak Test")
        print("Fix: removed manual _zero3_gather_ctx from backward")
        print("=" * 60)

    cfg = build_config()

    if is_rank0:
        print(f"Model config: hidden={cfg.hidden_size}, layers={cfg.num_layers}, "
              f"experts={cfg.num_real_experts}+{cfg.num_null_experts}={cfg.total_expert_slots}")
        print(f"Run: seq={SEQ_LEN}, batch={BATCH}, steps={NUM_STEPS}")
        print()

    model = Model3B(cfg, embedding_type="bpe")
    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-4)

    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )

    if is_rank0:
        import subprocess
        ds_ver = deepspeed.__version__
        gg_ver = subprocess.run(
            [sys.executable, "-m", "pip", "show", "grouped-gemm"],
            capture_output=True, text=True
        ).stdout
        for ln in gg_ver.splitlines():
            if "Version" in ln:
                print(f"grouped-gemm {ln.strip()}")
        print(f"DeepSpeed: {ds_ver}")
        print(f"FLA_NATIVE_DTYPE={os.environ.get('T17_DN_FLA_NATIVE_DTYPE','1')}")
        print()
        print(f"{'Step':>4}  {'VRAM GB':>8}  {'Δ VRAM GB':>10}  {'Status':>8}  Loss")
        print("-" * 55)

    prev_alloc = None
    for step in range(NUM_STEPS):
        # Synthetic token IDs
        input_ids = torch.randint(0, cfg.vocab_size, (BATCH, SEQ_LEN),
                                  device=f"cuda:{local_rank}")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = engine(input_ids)

        # Handle (logits, aux_loss) or plain logits
        if isinstance(out, tuple):
            logits, *rest = out
            loss = logits.float().mean()
            if rest and isinstance(rest[0], torch.Tensor) and rest[0].numel() == 1:
                loss = loss + 0.01 * rest[0].float()
        else:
            loss = out.float().mean()

        engine.backward(loss)
        engine.step()

        alloc_gb = torch.cuda.memory_allocated(local_rank) / 1e9
        if is_rank0:
            if prev_alloc is not None:
                delta = alloc_gb - prev_alloc
                if   delta >  0.5:  tag = "🔴 LEAK"
                elif delta < -0.5:  tag = "🟡 FREED"
                else:               tag = "✅ STABLE"
                print(f"{step+1:>4}  {alloc_gb:>8.3f}  {delta:>+10.3f}  {tag:>8}  {loss.item():.5f}")
            else:
                print(f"{step+1:>4}  {alloc_gb:>8.3f}  {'(warmup)':>10}  {'':>8}  {loss.item():.5f}")
            prev_alloc = alloc_gb

    if is_rank0:
        print("-" * 55)
        print("Done.\n")
        if prev_alloc is not None and NUM_STEPS > 1:
            print("✅ No leak detected." if abs(prev_alloc - alloc_gb) < 0.5
                  else "🔴 Memory grew — leak may still exist.")


if __name__ == "__main__":
    main()
