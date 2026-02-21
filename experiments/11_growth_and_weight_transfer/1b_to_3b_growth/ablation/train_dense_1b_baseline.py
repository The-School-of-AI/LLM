#!/usr/bin/env python3
"""
Phase 1: Clean 1B Dense Baseline Training (Stabilized)

Trains a fresh 1B dense model with gradient accumulation and learning rate warmup.
This establishes a STABLE ground truth for MoE experiments.

Usage:
    cd endGame && python -m ablation.train_dense_1b_baseline
"""

import os
import sys
import time
import gc
import math
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation.common import (
    detect_device, load_tokenizer, create_kronecker_codec,
    create_1b_model, create_data_loader, setup_logging,
    log_header, log_step, prepare_inputs, compute_losses,
    ENDGAME_DIR,
)
from training import save_checkpoint


# ============================================================================
# Config (Stabilized)
# ============================================================================
# Goal: ~32-64 effective batch size for stability.
# If MPS has memory limits, we use small physical batch and high accumulation.
NUM_UPDATES = 1000        # Total optimizer updates
BATCH_SIZE = 4            # Physical batch size (increased from 1, fits in 16GB MPS)
ACCUM_STEPS = 8           # Accumulate gradients (4 * 8 = 32 effective batch size)
SEQ_LEN = 64
LR_MAX = 3e-4             # Standard init for 1B (was 1e-4)
WARMUP_UPDATES = 100      # Soft start to prevent divergance
GRAD_CLIP = 1.0
CHECKPOINT_INTERVAL = 100
SAVE_DIR = os.path.join(ENDGAME_DIR, "checkpoints", "dense_1b_baseline")
LOG_PATH = os.path.join(ENDGAME_DIR, "logs", "dense_1b_baseline.log")


def train_baseline():
    print("=" * 80)
    print("PHASE 1: 1B Dense Baseline Training (Stabilized)")
    print(f"Updates: {NUM_UPDATES} | Eff Batch: {BATCH_SIZE*ACCUM_STEPS} | LR: {LR_MAX}")
    print(f"Physical Batch: {BATCH_SIZE} | Accum: {ACCUM_STEPS}")
    print("=" * 80)

    # Setup
    device = detect_device()
    tokenizer, bpe_vocab = load_tokenizer()
    pf_codec = create_kronecker_codec(tokenizer.vocab_size)
    model = create_1b_model(device, bpe_vocab, pf_codec)
    logger = setup_logging(LOG_PATH)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=WARMUP_UPDATES, 
        num_training_steps=NUM_UPDATES
    )

    # Data
    train_loader = create_data_loader(tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    data_iter = iter(train_loader)

    # Training
    model.train()
    log_header(logger)
    logger.info(f"# Config: updates={NUM_UPDATES}, batch={BATCH_SIZE}x{ACCUM_STEPS}, lr={LR_MAX}")
    logger.info(f"# Device: {device}")
    logger.info(f"# Params: {sum(p.numel() for p in model.parameters()):,}")
    logger.info("#")

    all_losses = []
    
    # Init stats
    t0 = time.time()
    
    for update_step in range(1, NUM_UPDATES + 1):
        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        # Accumulation Loop
        for micro_step in range(ACCUM_STEPS):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            x_input, y_ntp, y_mtp = prepare_inputs(input_ids)
            del input_ids

            # Forward
            logits_ntp, logits_mtp, aux_loss = model(
                x_input, next_token_ids=y_ntp, return_loss=True, return_memory=False,
            )

            # Loss
            losses = compute_losses(logits_ntp, logits_mtp, y_ntp, y_mtp, aux_loss)
            loss_step = losses["total"]
            
            # Scale loss for accumulation
            loss_scaled = loss_step / ACCUM_STEPS
            loss_scaled.backward()
            
            accum_loss += loss_step.item()
            
            # Keep last values for logging
            last_loss_ntp = losses["loss_ntp"].item()
            last_loss_mtp = losses["loss_mtp"].item()
            last_aux = losses["aux"]
            
            del logits_ntp, logits_mtp, x_input, y_ntp, y_mtp, loss_step, loss_scaled, losses

        # Optimizer Step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        
        # Stats
        avg_loss = accum_loss / ACCUM_STEPS
        all_losses.append(avg_loss)
        
        dt_ms = (time.time() - t0) * 1000.0
        # Tokens processed: Batch * Seq * Accum
        tokens_per_update = BATCH_SIZE * SEQ_LEN * ACCUM_STEPS
        tok_sec = tokens_per_update / max(dt_ms / 1000.0, 1e-9)
        current_lr = scheduler.get_last_lr()[0]

        log_step(
            logger, update_step, last_loss_ntp, last_loss_mtp, avg_loss,
            last_aux, current_lr, grad_norm.item(), tok_sec, dt_ms,
        )
        
        # Reset timer
        t0 = time.time()

        # Checkpoint
        if update_step % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                model=model, optimizer=optimizer, lr_scheduler=scheduler,
                step=update_step, loss=avg_loss, embedding_type="kronecker",
                save_dir=SAVE_DIR, keep_step_checkpoint=True,
            )

        # Memory cleanup
        if update_step % 10 == 0:
            gc.collect()
            if device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    # Final checkpoint
    save_checkpoint(
        model=model, optimizer=optimizer, lr_scheduler=scheduler,
        step=NUM_UPDATES, loss=all_losses[-1], embedding_type="kronecker",
        save_dir=SAVE_DIR, keep_step_checkpoint=True,
    )

    # Summary
    logger.info("#")
    logger.info("# ====== TRAINING SUMMARY ======")
    logger.info(f"# Final loss:      {all_losses[-1]:.4f}")
    logger.info(f"# Min loss:        {min(all_losses):.4f} (step {all_losses.index(min(all_losses)) + 1})")
    logger.info(f"# Mean loss:       {sum(all_losses) / len(all_losses):.4f}")
    logger.info(f"# Last 100 avg:    {sum(all_losses[-100:]) / 100:.4f}")
    logger.info(f"# Checkpoint dir:  {SAVE_DIR}")
    logger.info(f"# Log file:        {LOG_PATH}")

    print("\n" + "=" * 80)
    print("BASELINE COMPLETE")
    print(f"  Final loss:     {all_losses[-1]:.4f}")
    print(f"  Checkpoint:     {SAVE_DIR}")
    print(f"  Log:            {LOG_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    train_baseline()