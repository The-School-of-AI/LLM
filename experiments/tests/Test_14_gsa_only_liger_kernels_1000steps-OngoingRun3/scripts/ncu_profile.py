#!/usr/bin/env python3
"""
Generate Nsight Compute (ncu) profiles for baseline vs optimized model.

Produces two .ncu-rep files with per-kernel metrics: occupancy, memory
throughput, compute throughput, warp stalls, register usage, etc.

Unlike nsys (timeline-level), ncu replays each kernel multiple times to
collect hardware counters — so it profiles fewer steps but gives deeper
per-kernel insight.

Usage:

    cd experiments/tests/Test_14_gsa_only_liger_kernels_1000steps-OngoingRun3

    # Run both profiles automatically (recommended)
    PYTHONPATH="/workspace/LLM:${PYTHONPATH:-}" \
      .venv/bin/python scripts/ncu_profile.py \
      --config configs/test14_gsa_only_liger_kernels_1000steps.yaml \
      --mode both

    # Or profile individually under ncu:
    ncu --set full --replay-mode kernel \
        --target-processes all \
        --launch-skip-before-match 0 \
        --force-overwrite \
        -o results/ncu_baseline \
        .venv/bin/python scripts/ncu_profile.py \
        --config configs/test14_gsa_only_liger_kernels_1000steps.yaml \
        --mode baseline

Files generated:
    results/ncu_baseline.ncu-rep
    results/ncu_optimized.ncu-rep
"""

import argparse
import gc
import os
import subprocess
import sys

# Reduce CUDA allocator fragmentation
if "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.cuda.nvtx as nvtx
import yaml

# Add code/ and scripts/ to path
CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code")
sys.path.insert(0, CODE_DIR)
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from src.models.recurrence_model_1b import (
    Model1B, ModelConfig, KroneckerConfig, KroneckerEmbeddings,
)


# ════════════════════════════════════════════════════════════════════════
# MODEL BUILDER
# ════════════════════════════════════════════════════════════════════════

def build_model(cfg, config_path, device):
    """Build Model1B from config."""
    model_config = ModelConfig()
    embedding_type = cfg["model"].get("embedding_type", "standard")
    bpe_vocab = None
    pf_codec = None
    vocab_size = 131075

    if embedding_type == "kronecker":
        pf_config = KroneckerConfig(
            CHAR_DIM=256, POS_DIM=32, D=8192,
            length_normalize=True, truncate_long_words=True,
        )
        pf_codec = KroneckerEmbeddings(pf_config)
        model_config.vocab_size = vocab_size
        bpe_vocab = [f"tok_{i}" for i in range(vocab_size)]

    model = Model1B(
        config=model_config,
        embedding_type=embedding_type,
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec,
    )
    model = model.to(dtype=torch.bfloat16, device=device)

    init_path = os.path.normpath(
        os.path.join(os.path.dirname(config_path), cfg["training"]["init_model_path"])
    )
    if os.path.exists(init_path):
        state = torch.load(init_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)

    model.train()
    return model, vocab_size


# ════════════════════════════════════════════════════════════════════════
# PROFILED FORWARD + BACKWARD
# ════════════════════════════════════════════════════════════════════════

def run_fwd_bwd(model, input_ids, mask, vocab_size, step_idx=0):
    """Run a single forward+backward step with NVTX annotations."""
    nvtx.range_push(f"step_{step_idx}")

    nvtx.range_push("forward")
    results = model(input_ids, attention_mask=mask, return_hidden=True)
    h_ntp = results[0]
    loss = h_ntp.sum()
    nvtx.range_pop()  # forward

    nvtx.range_push("backward")
    loss.backward()
    nvtx.range_pop()  # backward

    nvtx.range_push("zero_grad")
    model.zero_grad(set_to_none=True)
    nvtx.range_pop()  # zero_grad

    nvtx.range_pop()  # step_N
    return float(loss.detach())


def profile_model(model, batch_size, seq_len, vocab_size, device,
                  warmup=3, profiled_steps=1, label=""):
    """
    Warmup then run profiled steps.

    ncu handles profiling externally — this script just runs the workload.
    We use cudaProfilerApi so ncu only captures the profiled steps.
    Default: 1 profiled step (ncu replays kernels internally, so 1 step
    already gives full coverage).
    """
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # ── Warmup (outside profiling range) ──
    print(f"  [{label}] Warmup ({warmup} steps)...")
    for i in range(warmup):
        _ = run_fwd_bwd(model, input_ids, mask, vocab_size, step_idx=i)
    torch.cuda.synchronize()

    # ── Profiled steps ──
    print(f"  [{label}] Profiling ({profiled_steps} step(s))...")
    print(f"  [{label}] ncu will replay each kernel — this may take several minutes.")
    torch.cuda.cudart().cudaProfilerStart()
    nvtx.range_push(label)

    for i in range(profiled_steps):
        loss = run_fwd_bwd(model, input_ids, mask, vocab_size, step_idx=i)
        print(f"    Step {i} loss: {loss:.4f}")

    nvtx.range_pop()  # label
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"  [{label}] Done.")


# ════════════════════════════════════════════════════════════════════════
# SINGLE-MODE RUNNER (called by ncu)
# ════════════════════════════════════════════════════════════════════════

def run_single_mode(args, cfg, mode):
    """Run baseline or optimized model workload for ncu to profile."""
    device = torch.device("cuda:0")
    seq_len = cfg["data"]["max_length"]

    # Determine batch size (use smaller default for ncu — replay is slow)
    if args.batch_size:
        batch_size = args.batch_size
    else:
        ds_cfg_path = os.path.normpath(
            os.path.join(os.path.dirname(args.config), cfg["deepspeed"]["config_path"])
        )
        with open(ds_cfg_path) as f:
            ds_cfg = yaml.safe_load(f)
        batch_size = ds_cfg.get("train_micro_batch_size_per_gpu", 64)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Mode: {mode}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"Warmup: {args.warmup}, Profiled steps: {args.steps}")

    # Build model
    print("\nBuilding model...")
    model, vocab_size = build_model(cfg, args.config, device)
    alloc = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded: {alloc:.1f} GB")

    if mode == "optimized":
        from triton_optimizations import apply_all_optimizations
        import src.models.liger_ops as liger_ops_mod
        patch_info = apply_all_optimizations(model, liger_ops_mod)
        print(f"Optimizations applied: {patch_info}")

    # Profile
    profile_model(
        model, batch_size, seq_len, vocab_size, device,
        warmup=args.warmup,
        profiled_steps=args.steps,
        label=mode,
    )

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nPeak GPU memory: {peak:.1f} GB")


# ════════════════════════════════════════════════════════════════════════
# BOTH MODE: launches ncu subprocesses
# ════════════════════════════════════════════════════════════════════════

def run_both(args, cfg):
    """Launch ncu for both baseline and optimized as subprocesses."""
    script_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(script_path))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    python_bin = os.path.join(base_dir, ".venv", "bin", "python")
    if not os.path.exists(python_bin):
        python_bin = sys.executable

    env = os.environ.copy()
    env["PYTHONPATH"] = f"/workspace/LLM:{env.get('PYTHONPATH', '')}"

    common_args = [
        "--config", args.config,
        "--warmup", str(args.warmup),
        "--steps", str(args.steps),
    ]
    if args.batch_size:
        common_args += ["--batch-size", str(args.batch_size)]

    # Map metric set names
    set_name = args.metrics

    for mode in ["baseline", "optimized"]:
        output_path = os.path.join(results_dir, f"ncu_{mode}")

        ncu_cmd = [
            "ncu",
            "--set", set_name,
            "--replay-mode", "kernel",
            "--target-processes", "all",
            "--launch-skip-before-match", "0",
            "--nvtx", "--nvtx-include", f"{mode}/",
            "--force-overwrite",
            "-o", output_path,
            python_bin, script_path,
            "--mode", mode,
        ] + common_args

        # Add kernel filter if specified
        if args.kernel_filter:
            ncu_cmd.insert(ncu_cmd.index("--force-overwrite"), "--kernel-name")
            ncu_cmd.insert(ncu_cmd.index("--force-overwrite"), args.kernel_filter)

        print("=" * 72)
        print(f"  NCU PROFILING: {mode.upper()}")
        print("=" * 72)
        print(f"  Metric set: {set_name}")
        print(f"  Command: {' '.join(ncu_cmd)}")
        print()
        print(f"  NOTE: ncu replays each kernel multiple times to collect")
        print(f"        hardware counters. This will take significantly")
        print(f"        longer than a normal run.")
        print()

        result = subprocess.run(ncu_cmd, env=env)

        if result.returncode != 0:
            print(f"\n  ERROR: ncu profile for {mode} failed (exit code {result.returncode})")
            if result.returncode == 127:
                print(f"  ncu not found. Install NVIDIA Nsight Compute or use:")
                print(f"    sudo apt install nsight-compute")
            continue

        rep_file = output_path + ".ncu-rep"
        if os.path.exists(rep_file):
            size_mb = os.path.getsize(rep_file) / 1e6
            print(f"\n  Generated: {rep_file} ({size_mb:.1f} MB)")
        print()

    # Final summary
    print("=" * 72)
    print("  NCU PROFILE COMPLETE")
    print("=" * 72)
    print(f"  Baseline:  results/ncu_baseline.ncu-rep")
    print(f"  Optimized: results/ncu_optimized.ncu-rep")
    print()
    print("  Open in Nsight Compute GUI:")
    print("    ncu-ui results/ncu_baseline.ncu-rep")
    print("    ncu-ui results/ncu_optimized.ncu-rep")
    print()
    print("  Compare side-by-side:")
    print("    ncu-ui --page details results/ncu_baseline.ncu-rep results/ncu_optimized.ncu-rep")
    print()
    print("  Key metrics to compare:")
    print("    - SM [%]: compute utilization")
    print("    - Memory [%]: memory bandwidth utilization")
    print("    - Achieved Occupancy: warp scheduling efficiency")
    print("    - L1/L2 Hit Rate: cache effectiveness")
    print("    - DRAM Throughput: global memory bandwidth")
    print("    - Registers/Thread: register pressure")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Nsight Compute (ncu) profiling for baseline vs optimized model"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", required=True,
                        choices=["baseline", "optimized", "both"],
                        help="Profile mode: baseline, optimized, or both")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: from DeepSpeed config)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup steps before profiling")
    parser.add_argument("--steps", type=int, default=1,
                        help="Profiled fwd+bwd steps (default: 1, ncu replays internally)")
    parser.add_argument("--metrics", default="full",
                        choices=["basic", "full", "roofline", "source"],
                        help="ncu metric set (default: full)")
    parser.add_argument("--kernel-filter", default=None,
                        help="Regex to filter kernel names (e.g. 'fused_rope|rmsnorm')")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.mode == "both":
        run_both(args, cfg)
    else:
        run_single_mode(args, cfg, args.mode)


if __name__ == "__main__":
    main()
