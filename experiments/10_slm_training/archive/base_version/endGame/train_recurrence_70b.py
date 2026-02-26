#!/usr/bin/env python3
"""
Training script for recurrence_model_70b.py with Kronecker embeddings

This adapts the training pipeline to work with the 70B recurrence model
(user-modified to 10 experts for Mac testing).

Usage:
    python train_recurrence_70b.py
"""

import os

# MPS memory management - critical for Mac M1
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1.0"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.9"
os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"

import gc
import time

import torch
import torch.nn as nn

# Import existing data utilities
from data_utils import SYNTHStream

# Import the 70B recurrence model (local file in endGame directory)
from recurrence_model_70b import KroneckerConfig, KroneckerEmbeddings, create_model_70b
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast


def load_tokenizer_from_json(tokenizer_path):
    """Load tokenizer from tokenizer.json"""
    print(f"📚 Loading tokenizer from {tokenizer_path}...")

    # Load using PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    # Extract vocab
    vocab_size = tokenizer.vocab_size
    print(f"   ✓ Loaded tokenizer with {vocab_size:,} tokens")

    # Create BPE vocab list (token strings for each ID)
    bpe_vocab = []
    for token_id in range(vocab_size):
        try:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            bpe_vocab.append(token_text)
        except Exception:
            bpe_vocab.append(f"<TOKEN_{token_id}>")

    print(f"   ✓ Created vocabulary with {len(bpe_vocab)} tokens")
    print(f"   Sample tokens: {bpe_vocab[:10]}")

    return tokenizer, bpe_vocab


def setup_kronecker_embeddings(vocab_size):
    """Setup Kronecker embeddings for the given vocabulary size"""
    print("🔧 Setting up Kronecker embeddings...")

    # Create Kronecker config
    # D = CHAR_DIM × POS_DIM = 256 × 32 = 8192
    pf_cfg = KroneckerConfig(
        CHAR_DIM=256,  # Byte vocabulary (0-255)
        POS_DIM=32,  # Max 32 bytes per token
        D=8192,  # Total embedding dimension
        length_normalize=True,
        truncate_long_words=True,
    )

    # Create codec
    pf_codec = KroneckerEmbeddings(pf_cfg)

    print(
        f"   ✓ Kronecker config: {pf_cfg.CHAR_DIM}×{pf_cfg.POS_DIM} = {pf_cfg.D} dims"
    )

    return pf_codec


def create_model(vocab_size, bpe_vocab, pf_codec, device):
    """Create 70B recurrence model with Kronecker embeddings"""
    print("🤖 Creating Model70B with Kronecker embeddings...")

    model = create_model_70b(
        embedding_type="kronecker", bpe_vocab=bpe_vocab, pf_codec=pf_codec
    )

    # Move to device
    model = model.to(device)

    print(f"   ✓ Model created and moved to {device}")

    return model


def simple_training_loop(model, train_loader, device, num_steps=100):
    """Simplified training loop just to test if everything runs"""
    print("🔥 Starting training loop...")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    data_iter = iter(train_loader)

    for step in range(num_steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)

        # Prepare multi-token prediction inputs
        # x_input: [B, T-2], y_ntp: [B, T-2], y_mtp: [B, T-2]
        x_input = input_ids[:, :-2].contiguous()
        y_ntp = input_ids[:, 1:-1].contiguous()
        y_mtp = input_ids[:, 2:].contiguous()

        del input_ids

        # Forward pass
        t0 = time.time()

        # Model forward (returns logits_ntp, logits_mtp, aux_loss)
        logits_ntp, logits_mtp, aux_loss = model(
            x_input,
            next_token_ids=y_ntp,
            return_loss=True,
            return_memory=False,  # Don't need memory for now
            prev_memory_stream=None,
        )

        # DEBUG: Check if MTP is actually being computed (step 0 only)
        if step == 0:
            print("\nDEBUG - Model output:")
            print(f"  logits_mtp is None: {logits_mtp is None}")
            if logits_mtp is not None:
                print(
                    f"  logits_mtp contains NaN: {torch.isnan(logits_mtp).any().item()}"
                )
                print(f"  logits_mtp mean: {logits_mtp.mean().item():.4f}")
                print(f"  logits_mtp std: {logits_mtp.std().item():.4f}")

        # Compute losses
        vocab_size = logits_ntp.size(-1)
        loss_ntp = criterion(logits_ntp.view(-1, vocab_size), y_ntp.view(-1))
        loss_mtp = criterion(logits_mtp.view(-1, vocab_size), y_mtp.view(-1))

        # DEBUG: Check shapes and perplexities (step 0 only)
        if step == 0:
            print(f"\nDEBUG - Step {step}:")
            print(
                f"  x_input shape: {x_input.shape}, y_ntp shape: {y_ntp.shape}, y_mtp shape: {y_mtp.shape}"
            )
            print(
                f"  logits_ntp shape: {logits_ntp.shape}, logits_mtp shape: {logits_mtp.shape}"
            )
            print(f"  NTP perplexity: {torch.exp(loss_ntp).item():.2f}")
            print(f"  MTP perplexity: {torch.exp(loss_mtp).item():.2f}")
            print(f"  aux_loss: {aux_loss.item():.6f}")

            # Check MoE routing stats
            ntp_preds = logits_ntp.argmax(dim=-1)
            mtp_preds = logits_mtp.argmax(dim=-1)
            ntp_acc = (ntp_preds == y_ntp).float().mean().item()
            mtp_acc = (mtp_preds == y_mtp).float().mean().item()
            print(f"  NTP accuracy: {ntp_acc:.4f}")
            print(f"  MTP accuracy: {mtp_acc:.4f}")

        # Combined loss (NTP + 0.3*MTP + aux)
        loss = loss_ntp + 0.3 * loss_mtp + aux_loss

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        dt = (time.time() - t0) * 1000.0

        # Log every step (since we're just testing)
        if step % 1 == 0:
            tok_sec = x_input.numel() / max(dt / 1000.0, 1e-9)
            print(
                f"step {step:3d} | loss_ntp: {loss_ntp.item():.4f} | "
                f"loss_mtp: {loss_mtp.item():.4f} | aux: {aux_loss.item():.4f} | "
                f"dt: {dt:6.1f}ms | tok/sec: {tok_sec:8.1f}"
            )

        # Cleanup
        del logits_ntp, logits_mtp, x_input, y_ntp, y_mtp, loss

        # Memory cleanup every 10 steps
        if step % 10 == 0:
            gc.collect()
            if device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    print("🏁 Training test complete!")


def main():
    """Main function"""
    print("=" * 80)
    print("🚀 RECURRENCE MODEL 70B (10 EXPERTS) - TRAINING SETUP TEST")
    print("=" * 80)

    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")

    # Load tokenizer
    tokenizer_path = "/Users/rohanshravan/TSAI/ValidationCheck/endGame/tokenizer.json"
    tokenizer, bpe_vocab = load_tokenizer_from_json(tokenizer_path)

    # Setup Kronecker embeddings
    pf_codec = setup_kronecker_embeddings(tokenizer.vocab_size)

    # Create model
    model = create_model(tokenizer.vocab_size, bpe_vocab, pf_codec, device)

    # Load dataset
    print("📊 Loading SYNTH dataset...")
    dataset = SYNTHStream(
        tokenizer=tokenizer,
        dataset_name="PleIAs/SYNTH",
        local_path="../synth_local_en",  # Your local dataset
        seq_len=64,  # Small sequence for Mac
        batch_size=2,  # Small batch for Mac
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
        batch_size=2,  # Very small batch for 70B model on Mac
        num_workers=0,  # Single worker for Mac
        pin_memory=False,
        drop_last=True,
    )

    print("✓ Dataset loaded")

    # Run simple training loop
    print("\n" + "=" * 80)
    print("Testing training loop (100 steps)...")
    print("=" * 80)

    simple_training_loop(model, train_loader, device, num_steps=100)

    print("\n" + "=" * 80)
    print("✅ SUCCESS! Model loads and trains correctly.")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Monitor MoE routing statistics (aux_loss should be > 0)")
    print("2. Check expert utilization")
    print("3. Compare convergence with 1B dense model")
    print("4. Scale up for GPU training")


if __name__ == "__main__":
    main()
