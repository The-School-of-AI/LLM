#!/usr/bin/env python3
"""
Probe maximum micro_batch_size that fits in GPU memory for the 1B reversible model.

Runs a forward + backward pass at increasing batch sizes until OOM, then reports
the last successful size. Uses the same model init path and config as run.sh.

Usage:
    cd experiments/tests/Test_14_gsa_only_liger_kernels_1000steps-OngoingRun3
    python scripts/find_max_batch.py --config configs/test14_gsa_only_liger_kernels_1000steps.yaml
"""

import argparse
import gc
import os
import sys

# Match main.py: reduce CUDA allocator fragmentation
if "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import yaml

# Add code/ to path so imports resolve
CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code")
sys.path.insert(0, CODE_DIR)

from src.models.recurrence_model_1b import (
    Model1B,
    ModelConfig,
    KroneckerConfig,
    KroneckerEmbeddings,
)


def get_gpu_mem():
    """Return (allocated_gb, reserved_gb, total_gb)."""
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    return a, r, t


def try_batch(model, batch_size, seq_len, vocab_size, device):
    """Run fwd + bwd at given batch_size. Returns (success, peak_gb)."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    try:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # Model1B.forward(input_ids, attention_mask=..., return_hidden=True)
        # returns (h_ntp, h_mtp_or_None, aux_loss)
        results = model(input_ids, attention_mask=mask, return_hidden=True)
        h_ntp = results[0]  # [B, T, H]
        # Simulate loss: just sum hidden states (same memory profile as real CE)
        loss = h_ntp.sum()
        loss.backward()

        peak = torch.cuda.max_memory_allocated() / 1e9
        # Cleanup
        model.zero_grad(set_to_none=True)
        del input_ids, mask, results, h_ntp, loss
        torch.cuda.empty_cache()
        gc.collect()
        return True, peak
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        gc.collect()
        return False, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--start", type=int, default=64, help="Starting batch size")
    parser.add_argument("--step", type=int, default=16, help="Batch size increment")
    parser.add_argument("--max-try", type=int, default=256, help="Max batch size to try")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seq_len = cfg["data"]["max_length"]
    init_path = os.path.normpath(
        os.path.join(os.path.dirname(args.config), cfg["training"]["init_model_path"])
    )

    device = torch.device("cuda:0")

    # Build model (same as main.py)
    print("Building model...")
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

    # Load init weights
    if os.path.exists(init_path):
        print(f"Loading weights from {init_path}")
        state = torch.load(init_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)

    model.train()

    alloc, reserved, total = get_gpu_mem()
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total:.1f} GB")
    print(f"Model loaded: {alloc:.1f} GB allocated, {reserved:.1f} GB reserved")
    print(f"Seq length: {seq_len}")
    print(f"Vocab size: {vocab_size}")
    print(f"\n{'batch':>6} {'status':>8} {'peak_gb':>10} {'util%':>8}")
    print("-" * 38)

    last_good = 0
    last_peak = 0.0
    bs = args.start

    while bs <= args.max_try:
        ok, peak = try_batch(model, bs, seq_len, vocab_size, device)
        if ok:
            pct = peak / total * 100
            print(f"{bs:>6} {'OK':>8} {peak:>9.1f}G {pct:>7.1f}%")
            last_good = bs
            last_peak = peak
            bs += args.step
        else:
            print(f"{bs:>6} {'OOM':>8} {'--':>10} {'--':>8}")
            break

    print(f"\n{'='*50}")
    print(f"MAX MICRO_BATCH_SIZE: {last_good}")
    print(f"Peak memory at max: {last_peak:.1f} GB / {total:.1f} GB ({last_peak/total*100:.0f}%)")
    print(f"\nRecommended DeepSpeed config (train_batch = micro_batch, grad_accum=1):")
    print(f'  "train_batch_size": {last_good},')
    print(f'  "train_micro_batch_size_per_gpu": {last_good},')
    print(f'  "gradient_accumulation_steps": 1,')

    # Also suggest keeping effective batch=64 if possible
    if last_good >= 64:
        accum = max(1, 64 // last_good) if last_good <= 64 else 1
        effective = last_good * accum
        print(f"\nTo keep effective batch=64: micro_batch={min(last_good,64)}, grad_accum={64//min(last_good,64)}")


if __name__ == "__main__":
    main()
