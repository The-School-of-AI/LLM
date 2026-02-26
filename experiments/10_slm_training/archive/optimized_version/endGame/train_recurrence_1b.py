#!/usr/bin/env python3
"""
Training script for recurrence_model_1b.py with Kronecker embeddings

Supports CUDA (A100/H100/T4) with Triton kernel acceleration,
MPS (Apple Silicon) fallback, and CPU fallback.

Usage:
    python train_recurrence_1b.py                        # Auto-detect device
    python train_recurrence_1b.py --seq-length 4096      # Longer sequences
    python train_recurrence_1b.py --batch-size 8         # Larger batches
    python train_recurrence_1b.py --max-steps 1000       # More steps
    python train_recurrence_1b.py --no-bf16              # Disable bf16
"""

import argparse
import gc
import os
import time

import torch
import torch.nn as nn

# Import existing data utilities
from data_utils import SYNTHStream

# Import the 1B recurrence model
from recurrence_model_1b import KroneckerConfig, KroneckerEmbeddings, create_model_1b
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser(description="Train recurrence_model_1b")
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
        "--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer.json"
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
    return parser.parse_args()


def detect_device_and_config(args):
    """Detect device and set optimal defaults."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: CUDA — {gpu_name} ({gpu_mem_gb:.1f} GB)")

        # Auto batch size based on GPU memory and seq length
        if args.batch_size is None:
            if gpu_mem_gb >= 70:  # A100-80GB, H100
                args.batch_size = 8
            elif gpu_mem_gb >= 35:  # A100-40GB
                args.batch_size = 4
            elif gpu_mem_gb >= 14:  # T4, V100
                args.batch_size = 2
            else:
                args.batch_size = 1

        if args.num_workers is None:
            args.num_workers = 4

        # Enable bf16 by default on CUDA
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
        # Kernels not imported into model — fallback
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


def load_tokenizer(tokenizer_path):
    """Load tokenizer from tokenizer.json"""
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

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  Parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable"
    )

    return model


def training_loop(model, train_loader, device, args):
    """Main training loop with CUDA optimizations."""
    print(f"\nStarting training ({args.max_steps} steps)...")

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, fused=(device.type == "cuda")
    )
    criterion = nn.CrossEntropyLoss()

    # Note: We do NOT use torch.autocast here. The reversible midpoint stack
    # recomputes the forward pass during backward under torch.enable_grad()
    # (outside any autocast context). If the model weights are bf16 but the
    # reconstructed activations are float32, F.linear will raise a dtype
    # mismatch. Instead, the model is cast to bf16 at creation time, so all
    # ops naturally run in bf16 without needing autocast.

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

    for step in range(args.max_steps):
        t0 = time.time()

        # Accumulate gradients over micro-batches
        accum_loss_ntp = 0.0
        accum_loss_mtp = 0.0
        accum_aux = 0.0
        step_tokens = 0

        for micro in range(args.grad_accum):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)

            # Prepare multi-token prediction inputs
            x_input = input_ids[:, :-2].contiguous()
            y_ntp = input_ids[:, 1:-1].contiguous()
            y_mtp = input_ids[:, 2:].contiguous()
            del input_ids

            # Forward pass (no autocast — model already in bf16, reversible stack
            # recomputes forward during backward outside any autocast context)
            logits_ntp, logits_mtp, aux_loss = model(
                x_input,
                next_token_ids=y_ntp,
                return_loss=True,
                return_memory=False,
                prev_memory_stream=None,
            )

            # Compute losses (upcast logits to float32 for stable cross-entropy
            # over 131k vocab — bf16 log_softmax accumulation can lose precision)
            vocab_size = logits_ntp.size(-1)
            loss_ntp = criterion(
                logits_ntp.float().view(-1, vocab_size), y_ntp.view(-1)
            )
            loss_mtp = criterion(
                logits_mtp.float().view(-1, vocab_size), y_mtp.view(-1)
            )

            # NaN watchdog — detect which component produced NaN
            if torch.isnan(loss_ntp) or torch.isnan(loss_mtp) or torch.isnan(aux_loss):
                with torch.no_grad():
                    print(f"\n⚠️  NaN detected at step {step} micro {micro}!")
                    print(
                        f"  loss_ntp={loss_ntp.item()}, loss_mtp={loss_mtp.item()}, aux={aux_loss.item()}"
                    )

            loss = (loss_ntp + 0.3 * loss_mtp + aux_loss) / args.grad_accum

            # Backward — accumulates gradients
            loss.backward()

            # Track losses for logging (detached scalars, no graph references)
            accum_loss_ntp += loss_ntp.item()
            accum_loss_mtp += loss_mtp.item()
            accum_aux += aux_loss.item()
            step_tokens += x_input.numel()

            # CRITICAL: Free computation graph immediately after backward.
            # Without this, the next micro-batch's forward allocates memory
            # while the previous graph is still alive, doubling peak VRAM.
            del (
                logits_ntp,
                logits_mtp,
                x_input,
                y_ntp,
                y_mtp,
                loss,
                loss_ntp,
                loss_mtp,
                aux_loss,
            )

        # Optimizer step (after all micro-batches accumulated)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        dt = (time.time() - t0) * 1000.0
        total_tokens += step_tokens

        # Logging (averaged over micro-batches)
        if step % args.log_interval == 0:
            avg_ntp = accum_loss_ntp / args.grad_accum
            avg_mtp = accum_loss_mtp / args.grad_accum
            avg_aux = accum_aux / args.grad_accum

            tok_sec = step_tokens / max(dt / 1000.0, 1e-9)
            avg_tok_sec = total_tokens / max(time.time() - t_start, 1e-9)

            mem_str = ""
            if device.type == "cuda":
                mem_cur = torch.cuda.memory_allocated() / 1e9
                mem_peak = torch.cuda.max_memory_allocated() / 1e9
                mem_str = f" | mem: {mem_cur:.1f}/{mem_peak:.1f} GB"

            print(
                f"step {step:4d} | loss_ntp: {avg_ntp:.4f} | "
                f"loss_mtp: {avg_mtp:.4f} | aux: {avg_aux:.4f} | "
                f"dt: {dt:6.1f}ms | tok/s: {tok_sec:,.0f} (avg: {avg_tok_sec:,.0f})"
                f"{mem_str}"
            )

            # Step 0: detailed debug
            if step == 0:
                if device.type == "cuda":
                    print(
                        f"  CUDA peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
                    )

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

    # Final stats
    elapsed = time.time() - t_start
    print(f"\nTraining complete: {args.max_steps} steps in {elapsed:.1f}s")
    print(f"  Average throughput: {total_tokens / elapsed:,.0f} tok/s")
    if device.type == "cuda":
        print(f"  Peak CUDA memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


def main():
    args = parse_args()

    print("=" * 70)
    print("RECURRENCE MODEL 1B — TRAINING")
    print("=" * 70)

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
