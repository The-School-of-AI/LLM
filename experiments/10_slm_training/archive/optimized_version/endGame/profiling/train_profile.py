#!/usr/bin/env python3
"""
Profiling-instrumented training script for recurrence_model_1b.py

Mirrors the original train_recurrence_1b.py with added profiling support for:
  - Nsight Systems (nsys):     CUDA Profiler API hooks + NVTX markers
  - Nsight Compute (ncu):      CUDA Profiler API hooks + reduced step count
  - PyTorch Profiler (pytorch): torch.profiler with Chrome trace + TensorBoard

Usage:
    # No profiling (identical to original training script)
    uv run python profiling/train_profile.py --batch-size 8 --seq-length 2048

    # PyTorch Profiler (standalone — no external tool needed)
    uv run python profiling/train_profile.py --batch-size 8 --seq-length 2048 \
        --profile-mode pytorch --warmup-steps 10 --num-profile-steps 5

    # Nsight Systems (launched via nsys from run_profiling.sh)
    uv run python profiling/train_profile.py --batch-size 8 --seq-length 2048 \
        --profile-mode nsys --warmup-steps 10 --num-profile-steps 5

    # Nsight Compute (launched via ncu from run_profiling.sh)
    uv run python profiling/train_profile.py --batch-size 8 --seq-length 2048 \
        --profile-mode ncu --warmup-steps 10 --num-profile-steps 5
"""

import argparse
import gc
import os
import sys
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
from data_utils import SYNTHStream
from recurrence_model_1b import KroneckerConfig, KroneckerEmbeddings, create_model_1b
from torch.utils.data import DataLoader

# Add parent directory to path so we can import project modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# Profiling Utilities
# =============================================================================


def compute_profile_range(warmup_steps: int, num_profile_steps: int):
    """Compute (profile_start, profile_end) from warmup + count."""
    profile_start = warmup_steps
    profile_end = warmup_steps + num_profile_steps - 1
    return profile_start, profile_end


@contextmanager
def nvtx_range(name: str, profile_mode: str):
    """NVTX range context manager — active only for nsys/ncu modes."""
    if profile_mode in ("nsys", "ncu") and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if profile_mode in ("nsys", "ncu") and torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def cuda_profiler_start():
    """Signal Nsight Systems / Compute to start capturing."""
    if torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop():
    """Signal Nsight Systems / Compute to stop capturing."""
    if torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStop()


def _write_profiler_summaries(prof, output_dir: str):
    """Write detailed key_averages reports for bottleneck identification."""
    os.makedirs(output_dir, exist_ok=True)

    separator = "=" * 120

    # 1. Top operators by CUDA time (self)
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("TOP OPERATORS BY CUDA TIME (self)\n")
        f.write(separator + "\n")
        f.write("Use this to find which low-level ops dominate GPU time.\n")
        f.write("High 'Self CUDA %' = prime candidate for Triton kernel or fusion.\n\n")
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
        f.write(table)
        f.write("\n\n")

        f.write("\nTOP OPERATORS BY TOTAL CUDA TIME (including children)\n")
        f.write(separator + "\n")
        table2 = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
        f.write(table2)
    print(f"    → {summary_path}")

    # 2. Grouped by nn.Module — shows which layers are slowest
    module_path = os.path.join(output_dir, "summary_by_module.txt")
    with open(module_path, "w") as f:
        f.write("OPERATORS GROUPED BY nn.Module\n")
        f.write(separator + "\n")
        f.write(
            "Shows which model components (attention, MoE, RMSNorm, etc.) are slowest.\n"
        )
        f.write("Look for high 'Self CUDA %' to find fusion/kernel targets.\n\n")
        table = prof.key_averages(group_by_input_shape=False).table(
            sort_by="self_cuda_time_total", row_limit=80
        )
        f.write(table)
    print(f"    → {module_path}")

    # 3. Grouped by input shape — reveals which tensor sizes dominate
    shape_path = os.path.join(output_dir, "summary_by_shape.txt")
    with open(shape_path, "w") as f:
        f.write("OPERATORS GROUPED BY INPUT SHAPE\n")
        f.write(separator + "\n")
        f.write("Shows operator performance at each tensor size.\n")
        f.write("Useful for identifying shape-specific bottlenecks.\n\n")
        table = prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=80
        )
        f.write(table)
    print(f"    → {shape_path}")

    # 4. Export stacks for flame graph generation
    stacks_path = os.path.join(output_dir, "stacks_cuda.txt")
    try:
        prof.export_stacks(stacks_path, metric="self_cuda_time_total")
        print(
            f"    → {stacks_path}  (flame graph: flamegraph.pl < stacks_cuda.txt > flame.svg)"
        )
    except Exception as e:
        print(f"    ⚠ Stacks export failed: {e}")

    stacks_cpu_path = os.path.join(output_dir, "stacks_cpu.txt")
    try:
        prof.export_stacks(stacks_cpu_path, metric="self_cpu_time_total")
        print(f"    → {stacks_cpu_path}")
    except Exception:
        pass

    # 5. Chrome trace JSON (separate from TensorBoard — may already be saved by tb_handler)
    chrome_path = os.path.join(output_dir, "chrome_trace.json")
    try:
        prof.export_chrome_trace(chrome_path)
        print(f"    → {chrome_path}  (open in chrome://tracing)")
    except RuntimeError:
        # TensorBoard handler already saved the trace
        print("    → Chrome trace already saved by TensorBoard handler")


def setup_pytorch_profiler(warmup_steps: int, num_profile_steps: int, output_dir: str):
    """Create a torch.profiler.profile instance with proper schedule.

    PyTorch Profiler schedule phases (each phase = 1 training step call to profiler.step()):
      wait   = warmup_steps - 1   (skip, no overhead)
      warmup = 1                  (profiler warms up its internals)
      active = num_profile_steps  (actual data collection)
    Total profiler.step() calls needed: wait + warmup + active
    """
    os.makedirs(output_dir, exist_ok=True)

    wait_phase = max(0, warmup_steps - 1)
    warmup_phase = 1
    active_phase = num_profile_steps

    print(
        f"  [pytorch] Schedule: wait={wait_phase}, warmup={warmup_phase}, "
        f"active={active_phase}  (total steps needed: {wait_phase + warmup_phase + active_phase})"
    )

    schedule = torch.profiler.schedule(
        wait=wait_phase,
        warmup=warmup_phase,
        active=active_phase,
        repeat=1,
    )

    def trace_handler(prof):
        """Custom handler: write TensorBoard trace + detailed summaries."""
        # TensorBoard trace
        tb_handler = torch.profiler.tensorboard_trace_handler(output_dir)
        tb_handler(prof)
        # Detailed text summaries
        _write_profiler_summaries(prof, output_dir)

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )
    return profiler


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train recurrence_model_1b with profiling support"
    )

    # Training arguments (same as original)
    parser.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="Sequence length (default: 512, try 2048/4096 for A100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (auto-selected per device if not set)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bf16 mixed precision (on by default for CUDA)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="../synth_local_en",
        help="Path to local SYNTH dataset",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer.json (auto-resolved if not set)",
    )
    parser.add_argument("--log-interval", type=int, default=1, help="Log every N steps")
    parser.add_argument(
        "--grad-accum", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (auto-selected per device)",
    )

    # Profiling arguments
    parser.add_argument(
        "--profile-mode",
        type=str,
        default="none",
        choices=["none", "nsys", "ncu", "pytorch"],
        help="Profiling mode: nsys, ncu, pytorch, or none (default: none)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of warmup steps before profiling begins (default: 10)",
    )
    parser.add_argument(
        "--num-profile-steps",
        type=int,
        default=5,
        help="Number of steps to profile after warmup (default: 5)",
    )
    parser.add_argument(
        "--profile-output-dir",
        type=str,
        default=None,
        help="Output directory for profiling data (auto-generated if not set)",
    )

    return parser.parse_args()


# =============================================================================
# Device Detection (same as original)
# =============================================================================


def detect_device_and_config(args):
    """Detect device and set optimal defaults."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: CUDA — {gpu_name} ({gpu_mem_gb:.1f} GB)")

        if args.batch_size is None:
            if gpu_mem_gb >= 70:
                args.batch_size = 8
            elif gpu_mem_gb >= 35:
                args.batch_size = 4
            elif gpu_mem_gb >= 14:
                args.batch_size = 2
            else:
                args.batch_size = 1

        if args.num_workers is None:
            args.num_workers = 4

        args.use_bf16 = not args.no_bf16 and torch.cuda.is_bf16_supported()
        if args.use_bf16:
            print("  bf16 autocast: ENABLED")
        else:
            print("  bf16 autocast: DISABLED")

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS (Apple Silicon)")
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1.0"
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.9"
        if args.batch_size is None:
            args.batch_size = 2
        if args.num_workers is None:
            args.num_workers = 0
        args.use_bf16 = False
    else:
        device = torch.device("cpu")
        print("Device: CPU")
        if args.batch_size is None:
            args.batch_size = 1
        if args.num_workers is None:
            args.num_workers = 0
        args.use_bf16 = False

    print(
        f"  batch_size={args.batch_size}, seq_length={args.seq_length}, "
        f"grad_accum={args.grad_accum}"
    )
    print(f"  effective_batch = {args.batch_size * args.grad_accum}")

    return device


# =============================================================================
# Kernel Status (same as original)
# =============================================================================


def print_kernel_status():
    """Print which kernels are available and will be used."""
    try:
        from recurrence_model_1b import (
            HAS_FLA,
            HAS_TRITON,
            fla_gated_delta_rule,
            triton_rmsnorm,
            triton_sinkhorn_knopp,
            triton_sparse_attention,
        )
    except ImportError:
        HAS_TRITON = False
        HAS_FLA = False
        triton_rmsnorm = None
        triton_sinkhorn_knopp = None
        triton_sparse_attention = None
        fla_gated_delta_rule = None

    cuda = torch.cuda.is_available()
    print("\nKernel Status:")
    print(f"  Triton available:     {HAS_TRITON}")
    print(f"  fla available:        {HAS_FLA}")
    print(f"  CUDA available:       {cuda}")

    if cuda and HAS_TRITON:
        print(f"  Triton RMSNorm:       {'ACTIVE' if triton_rmsnorm else 'missing'}")
        print(
            f"  Triton Sinkhorn:      {'ACTIVE' if triton_sinkhorn_knopp else 'missing'}"
        )
        print(
            f"  Triton Sparse Attn:   {'ACTIVE' if triton_sparse_attention else 'missing'}"
        )
    elif cuda and not HAS_TRITON:
        print("  NOTE: Install triton for GPU acceleration: pip install triton")

    if cuda and HAS_FLA:
        print(
            f"  fla GatedDeltaRule:   {'ACTIVE' if fla_gated_delta_rule else 'missing'}"
        )
    elif cuda and not HAS_FLA:
        print("  NOTE: Install fla for fused DeltaNet: pip install fla")

    if not cuda:
        print("  All kernels: FALLBACK (PyTorch) — no CUDA GPU detected")

    print()


# =============================================================================
# Tokenizer (same as original)
# =============================================================================


def load_tokenizer(tokenizer_path):
    """Load tokenizer from tokenizer.json"""
    from transformers import PreTrainedTokenizerFast

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"  Loaded tokenizer with {vocab_size:,} tokens")

    bpe_vocab = []
    for token_id in range(vocab_size):
        try:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            bpe_vocab.append(token_text)
        except Exception:
            bpe_vocab.append(f"<TOKEN_{token_id}>")

    return tokenizer, bpe_vocab


# =============================================================================
# Model Creation (same as original)
# =============================================================================


def create_model(vocab_size, bpe_vocab, pf_codec, device, use_bf16):
    """Create model and move to device with optional bf16."""
    print("Creating Model1B...")

    model = create_model_1b(
        embedding_type="kronecker", bpe_vocab=bpe_vocab, pf_codec=pf_codec
    )

    if use_bf16:
        model = model.to(dtype=torch.bfloat16, device=device)
        print(f"  Model on {device} (bfloat16)")
    else:
        model = model.to(device)
        print(f"  Model on {device} (float32)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  Parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable"
    )

    return model


# =============================================================================
# Profiled Training Loop
# =============================================================================


def training_loop(model, train_loader, device, args):
    """Main training loop with profiling instrumentation.

    Profiling timeline:
      Steps 0 .. warmup_steps-1           → warmup (no profiling)
      Steps warmup_steps .. profile_end   → PROFILED
      Steps profile_end+1 .. max_steps-1  → post-profile (runs normally)
    """
    profile_mode = args.profile_mode
    warmup_steps = args.warmup_steps
    num_profile_steps = args.num_profile_steps
    profile_start, profile_end = compute_profile_range(warmup_steps, num_profile_steps)

    # Ensure max_steps covers warmup + profiling
    min_steps_needed = profile_end + 1
    if profile_mode != "none" and args.max_steps < min_steps_needed:
        print(
            f"  Adjusting max_steps from {args.max_steps} to {min_steps_needed} "
            f"(need {warmup_steps} warmup + {num_profile_steps} profile steps)"
        )
        args.max_steps = min_steps_needed

    # Resolve output directory
    if args.profile_output_dir:
        profile_output_dir = args.profile_output_dir
    else:
        profile_output_dir = os.path.join(SCRIPT_DIR, f"output_{profile_mode}")

    # For ncu mode, cap max_steps to avoid very long replayed runs
    if profile_mode == "ncu":
        capped_steps = min(args.max_steps, profile_end + 2)
        if capped_steps != args.max_steps:
            print(f"  [ncu] Capping max_steps from {args.max_steps} to {capped_steps}")
            args.max_steps = capped_steps

    print(f"\nStarting training ({args.max_steps} steps)...")
    if profile_mode != "none":
        print(f"  Profiling mode:    {profile_mode}")
        print(f"  Warmup steps:      {warmup_steps} (steps 0–{warmup_steps - 1})")
        print(
            f"  Profile steps:     {num_profile_steps} (steps {profile_start}–{profile_end})"
        )
        print(f"  Output dir:        {profile_output_dir}")

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, fused=(device.type == "cuda")
    )
    criterion = nn.CrossEntropyLoss()

    data_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    # CUDA warmup
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(
            f"  CUDA memory before training: {torch.cuda.memory_allocated()/1e9:.2f} GB"
        )

    total_tokens = 0
    t_start = time.time()

    # ---- Setup PyTorch Profiler ----
    pytorch_profiler = None
    if profile_mode == "pytorch":
        pytorch_profiler = setup_pytorch_profiler(
            warmup_steps, num_profile_steps, profile_output_dir
        )
        pytorch_profiler.__enter__()

    # ---- CUDA Profiler API control for nsys / ncu ----
    profiler_api_active = False

    try:
        for step in range(args.max_steps):
            # -- Start CUDA profiler capture at profile_start --
            if (
                profile_mode in ("nsys", "ncu")
                and step == profile_start
                and not profiler_api_active
            ):
                print(f"  [{profile_mode}] === Profiling START (step {step}) ===")
                if device.type == "cuda":
                    torch.cuda.synchronize()
                cuda_profiler_start()
                profiler_api_active = True

            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)

            # Prepare multi-token prediction inputs
            with nvtx_range("data_prep", profile_mode):
                x_input = input_ids[:, :-2].contiguous()
                y_ntp = input_ids[:, 1:-1].contiguous()
                y_mtp = input_ids[:, 2:].contiguous()
                del input_ids

            t0 = time.time()

            # Forward pass
            with nvtx_range("forward", profile_mode):
                logits_ntp, logits_mtp, aux_loss = model(
                    x_input,
                    next_token_ids=y_ntp,
                    return_loss=True,
                    return_memory=False,
                    prev_memory_stream=None,
                )

            # Compute losses
            with nvtx_range("loss", profile_mode):
                vocab_size = logits_ntp.size(-1)
                loss_ntp = criterion(logits_ntp.view(-1, vocab_size), y_ntp.view(-1))
                loss_mtp = criterion(logits_mtp.view(-1, vocab_size), y_mtp.view(-1))
                loss = (loss_ntp + 0.3 * loss_mtp + aux_loss) / args.grad_accum

            # Backward
            with nvtx_range("backward", profile_mode):
                loss.backward()

            # Gradient accumulation step
            with nvtx_range("optimizer", profile_mode):
                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            dt = (time.time() - t0) * 1000.0
            total_tokens += x_input.numel()

            # Logging
            if step % args.log_interval == 0:
                tok_sec = x_input.numel() / max(dt / 1000.0, 1e-9)
                avg_tok_sec = total_tokens / max(time.time() - t_start, 1e-9)

                mem_str = ""
                if device.type == "cuda":
                    mem_cur = torch.cuda.memory_allocated() / 1e9
                    mem_peak = torch.cuda.max_memory_allocated() / 1e9
                    mem_str = f" | mem: {mem_cur:.1f}/{mem_peak:.1f} GB"

                profile_marker = ""
                if profile_mode != "none" and profile_start <= step <= profile_end:
                    profile_marker = " [PROFILING]"

                print(
                    f"step {step:4d} | loss_ntp: {loss_ntp.item():.4f} | "
                    f"loss_mtp: {loss_mtp.item():.4f} | aux: {aux_loss.item():.4f} | "
                    f"dt: {dt:6.1f}ms | tok/s: {tok_sec:,.0f} (avg: {avg_tok_sec:,.0f})"
                    f"{mem_str}{profile_marker}"
                )

                if step == 0:
                    print(
                        f"  shapes: x={x_input.shape}, logits_ntp={logits_ntp.shape}, "
                        f"logits_mtp={logits_mtp.shape if logits_mtp is not None else 'None'}"
                    )
                    if device.type == "cuda":
                        print(
                            f"  CUDA peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
                        )

            # Cleanup
            del logits_ntp, logits_mtp, x_input, y_ntp, y_mtp, loss, aux_loss

            # PyTorch profiler step
            if pytorch_profiler is not None:
                pytorch_profiler.step()

            # -- Stop CUDA profiler capture after profile_end --
            if (
                profile_mode in ("nsys", "ncu")
                and step == profile_end
                and profiler_api_active
            ):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                cuda_profiler_stop()
                profiler_api_active = False
                print(f"  [{profile_mode}] === Profiling STOP (step {step}) ===")

            # Periodic memory cleanup
            if step % 50 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass

    finally:
        # Ensure profiler is stopped even on error
        if profiler_api_active:
            cuda_profiler_stop()

        # Finalize PyTorch profiler
        if pytorch_profiler is not None:
            pytorch_profiler.__exit__(None, None, None)
            print(
                f"\n  [pytorch] Profiling complete. Artifacts saved to: {profile_output_dir}"
            )
            print(
                f"  [pytorch] View summaries:    cat {profile_output_dir}/summary.txt"
            )
            print(
                f"  [pytorch] Module breakdown:  cat {profile_output_dir}/summary_by_module.txt"
            )
            print(
                f"  [pytorch] TensorBoard:       tensorboard --logdir {profile_output_dir}"
            )
            print(
                f"  [pytorch] Chrome trace:      chrome://tracing → {profile_output_dir}/chrome_trace.json"
            )

    # Final stats
    elapsed = time.time() - t_start
    print(f"\nTraining complete: {args.max_steps} steps in {elapsed:.1f}s")
    print(f"  Average throughput: {total_tokens / elapsed:,.0f} tok/s")
    if device.type == "cuda":
        print(f"  Peak CUDA memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    print("=" * 70)
    print("RECURRENCE MODEL 1B — PROFILING TRAINING")
    print("=" * 70)

    # Resolve tokenizer path (look in project root if not absolute)
    if args.tokenizer is None:
        args.tokenizer = os.path.join(PROJECT_ROOT, "tokenizer.json")
    elif not os.path.isabs(args.tokenizer):
        # Try relative to project root
        candidate = os.path.join(PROJECT_ROOT, args.tokenizer)
        if os.path.isfile(candidate):
            args.tokenizer = candidate

    # Device setup
    device = detect_device_and_config(args)

    # Kernel report
    print_kernel_status()

    # Tokenizer
    tokenizer, bpe_vocab = load_tokenizer(args.tokenizer)

    # Kronecker embeddings
    print("Setting up Kronecker embeddings...")
    pf_cfg = KroneckerConfig(
        CHAR_DIM=256,
        POS_DIM=32,
        D=8192,
        length_normalize=True,
        truncate_long_words=True,
    )
    pf_codec = KroneckerEmbeddings(pf_cfg)

    # Model
    model = create_model(
        tokenizer.vocab_size, bpe_vocab, pf_codec, device, args.use_bf16
    )

    # Dataset
    print(f"Loading SYNTH dataset (seq_len={args.seq_length})...")
    dataset = SYNTHStream(
        tokenizer=tokenizer,
        dataset_name="PleIAs/SYNTH",
        local_path=args.dataset_path,
        seq_len=args.seq_length,
        batch_size=args.batch_size,
        shuffle_buffer=1000,
        seed=42,
        include_query=True,
        include_reasoning=True,
        include_answer=True,
        combine_separator="\n\n",
        filter_language="en",
        start_step=0,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(
        f"  DataLoader: batch_size={args.batch_size}, "
        f"num_workers={args.num_workers}, pin_memory={device.type == 'cuda'}"
    )

    # Train
    print("\n" + "=" * 70)
    training_loop(model, train_loader, device, args)
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
