#!/usr/bin/env python3
"""
Experiment 4 — Training: SVD + Active Routing (Stabilized)

Trains the full dense-to-MoE transition with:
  - SVD compressed experts (1024 -> 512)
  - Active routing bias (-2.65/+2.65)
  - Warmup freeze for first 100 updates
  - Gradient Accumulation (Eff Batch 32)
  - Cosine Scheduler

Usage:
    cd endGame && python -m ablation.train_exp4_svd_moe
"""

import os
import sys
import time
import gc
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation.common import (
    detect_device, load_tokenizer, create_kronecker_codec,
    create_1b_model, create_3b_model, create_data_loader,
    get_reference_batch, setup_logging, log_header, log_step_moe,
    prepare_inputs, compute_losses, compute_moe_metrics,
    ENDGAME_DIR,
)
from ablation.moe_diagnostics import (
    run_all_diagnostics, log_diagnostics, log_compact_diagnostics,
    save_detailed_diagnostics,
)

LOGS_DIR = os.path.join(ENDGAME_DIR, "logs")
from training import save_checkpoint, set_moe_freeze_state

BASELINE_CHECKPOINT = os.path.join(ENDGAME_DIR, "checkpoints", "dense_1b_baseline", "kronecker_latest.pt")
EXP4_CHECKPOINT = os.path.join(ENDGAME_DIR, "checkpoints", "exp4_svd_moe", "init.pt")
SAVE_DIR = os.path.join(ENDGAME_DIR, "checkpoints", "exp4_svd_moe")
LOG_PATH = os.path.join(ENDGAME_DIR, "logs", "exp4_svd_moe.log")

# Stabilized Config
NUM_UPDATES = 500       # Total optimizer steps
WARMUP_UPDATES = 50   # Steps to freeze experts / warm up LR 
LR_MAX = 3e-4          # Matches baseline
BATCH_SIZE = 4          # Physical batch
ACCUM_STEPS = 8         # Effective batch = 32
SEQ_LEN = 64
GRAD_CLIP = 1.0
CHECKPOINT_INTERVAL = 500


def main():
    print("=" * 80)
    print(f"EXP 4 TRAIN: SVD + Active Routing (Stabilized)")
    print(f"Updates: {NUM_UPDATES} | Warmup: {WARMUP_UPDATES} | Eff Batch: {BATCH_SIZE*ACCUM_STEPS}")
    print("=" * 80)

    device = detect_device()
    logger = setup_logging(LOG_PATH)
    tokenizer, bpe_vocab = load_tokenizer()
    pf_codec = create_kronecker_codec(tokenizer.vocab_size)

    # Check prerequisites
    for path, name in [(BASELINE_CHECKPOINT, "1B baseline"), (EXP4_CHECKPOINT, "Exp4 init")]:
        if not os.path.exists(path):
            print(f"  ERROR: {name} not found: {path}")
            return

    # Load 3B model (default config)
    print("\nLoading 3B MoE model (SVD init, active routing)...")
    model_3b, config_3b = create_3b_model(device, bpe_vocab, pf_codec)
    model_3b.load_state_dict(
        torch.load(EXP4_CHECKPOINT, map_location=device)["model_state_dict"]
    )

    # Initial eval comparison
    print("\nInitial eval...")
    x_ref, y_ntp_ref, y_mtp_ref = get_reference_batch(tokenizer, device)
    criterion = nn.CrossEntropyLoss()

    model_1b = create_1b_model(device, bpe_vocab, pf_codec)
    model_1b.load_state_dict(
        torch.load(BASELINE_CHECKPOINT, map_location=device)["model_state_dict"]
    )

    model_1b.eval()
    model_3b.eval()
    with torch.no_grad():
        logits_1b, _, _ = model_1b(x_ref, next_token_ids=y_ntp_ref, return_loss=True, return_memory=False)
        logits_3b, _, aux_3b = model_3b(x_ref, next_token_ids=y_ntp_ref, return_loss=True, return_memory=False)
        V = logits_1b.size(-1)
        loss_1b = criterion(logits_1b.view(-1, V), y_ntp_ref.view(-1)).item()
        loss_3b = criterion(logits_3b.view(-1, V), y_ntp_ref.view(-1)).item()

    logger.info("=" * 60)
    logger.info("EXP 4: SVD + Active Routing (Stabilized)")
    logger.info("=" * 60)
    logger.info(f"Batch Size: {BATCH_SIZE} x {ACCUM_STEPS} = {BATCH_SIZE*ACCUM_STEPS}")
    logger.info(f"Router bias: -1.0/+1.0")
    logger.info(f"Expert output scale: learnable (init=0.01)")
    logger.info(f"Warmup freeze: {WARMUP_UPDATES} updates")
    logger.info(f"Initial comparison:")
    logger.info(f"  1B baseline loss: {loss_1b:.6f}")
    logger.info(f"  3B MoE loss:      {loss_3b:.6f}")
    logger.info(f"")

    # Initial MoE diagnostics (after eval forward pass)
    ref_input_ids = torch.cat([x_ref, y_ntp_ref[:, -1:], y_mtp_ref[:, -1:]], dim=1)
    init_report = run_all_diagnostics(model_3b, input_ids=ref_input_ids, tokenizer=tokenizer)
    log_diagnostics(init_report, logger, step="init", verbose=True)
    saved_path = save_detailed_diagnostics(init_report, LOGS_DIR, step="init")
    logger.info(f"  Full token map saved: {saved_path}")

    del model_1b, logits_1b, logits_3b, ref_input_ids
    gc.collect()

    # Training
    logger.info(f"Training for {NUM_UPDATES} steps (warmup freeze first {WARMUP_UPDATES})")
    log_header(logger)

    model_3b.train()
    optimizer = torch.optim.AdamW(model_3b.parameters(), lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Cosine scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=WARMUP_UPDATES, 
        num_training_steps=NUM_UPDATES
    )
    
    train_loader = create_data_loader(tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    data_iter = iter(train_loader)

    log_header(logger)
    all_losses = []
    
    t0 = time.time()

    for update_step in range(1, NUM_UPDATES + 1):
        # Warmup freeze logic (based on update step)
        set_moe_freeze_state(model_3b, update_step, warmup_steps=WARMUP_UPDATES)

        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        # Accumulation Loop
        for micro_step in range(ACCUM_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            x_input, y_ntp, y_mtp = prepare_inputs(input_ids)

            logits_ntp, logits_mtp, aux_loss = model_3b(
                x_input, next_token_ids=y_ntp, return_loss=True, return_memory=False,
            )

            losses = compute_losses(logits_ntp, logits_mtp, y_ntp, y_mtp, aux_loss)
            loss_step = losses["total"]
            
            # Scale and backward
            loss_scaled = loss_step / ACCUM_STEPS
            loss_scaled.backward()
            
            accum_loss += loss_step.item()
            
            # Keep metrics for logging
            last_loss_ntp = losses["loss_ntp"].item()
            last_loss_mtp = losses["loss_mtp"].item()
            last_aux = losses["aux"]
            
            # Only compute costly null rate on last micro-step
            if micro_step == ACCUM_STEPS - 1:
                moe_metrics = compute_moe_metrics(model_3b)
            
            del logits_ntp, logits_mtp, x_input, y_ntp, y_mtp, input_ids, loss_step, loss_scaled, losses

        # Optimize
        grad_norm = torch.nn.utils.clip_grad_norm_(model_3b.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        dt_ms = (time.time() - t0) * 1000.0
        tokens_per_update = BATCH_SIZE * SEQ_LEN * ACCUM_STEPS
        tok_sec = tokens_per_update / max(dt_ms / 1000.0, 1e-9)
        current_lr = scheduler.get_last_lr()[0]
        
        avg_loss = accum_loss / ACCUM_STEPS
        all_losses.append(avg_loss)

        log_step_moe(
            logger, update_step, last_loss_ntp, last_loss_mtp,
            avg_loss, last_aux, current_lr, grad_norm.item(),
            tok_sec, dt_ms, moe_metrics["null_rate"],
        )
        t0 = time.time()

        # Diagnostics
        if update_step % 50 == 0:
            log_compact_diagnostics(model_3b, logger, update_step)

        if update_step == WARMUP_UPDATES:
            logger.info(f"# >>> WARMUP COMPLETE at step {update_step}. Routed experts unfrozen. <<<")

        # Checkpoint
        if update_step % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                model=model_3b, optimizer=optimizer, lr_scheduler=scheduler,
                step=update_step, loss=avg_loss, embedding_type="kronecker",
                save_dir=SAVE_DIR, keep_step_checkpoint=True,
            )

        if update_step % 10 == 0:
            gc.collect()
            if device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    # Final checkpoint
    save_checkpoint(
        model=model_3b, optimizer=optimizer, lr_scheduler=scheduler,
        step=NUM_UPDATES, loss=all_losses[-1], embedding_type="kronecker",
        save_dir=SAVE_DIR, keep_step_checkpoint=True,
    )

    # Final diagnostics (eval pass on reference batch)
    model_3b.eval()
    x_ref, y_ntp_ref, y_mtp_ref = get_reference_batch(tokenizer, device)
    with torch.no_grad():
        model_3b(x_ref, next_token_ids=y_ntp_ref, return_loss=True, return_memory=False)
    ref_input_ids = torch.cat([x_ref, y_ntp_ref[:, -1:], y_mtp_ref[:, -1:]], dim=1)
    final_report = run_all_diagnostics(model_3b, input_ids=ref_input_ids, tokenizer=tokenizer)
    log_diagnostics(final_report, logger, step="final", verbose=True)
    saved_path = save_detailed_diagnostics(final_report, LOGS_DIR, step="final")
    logger.info(f"  Full token map saved: {saved_path}")
    del x_ref, y_ntp_ref, y_mtp_ref, ref_input_ids

    # Summary
    logger.info(f"")
    logger.info(f"# ====== EXP 4 TRAINING SUMMARY ======")
    logger.info(f"# Initial loss_ntp:      {all_losses[0]:.4f}")
    logger.info(f"# Loss at warmup end:    {all_losses[min(WARMUP_UPDATES, len(all_losses)-1)]:.4f}")
    logger.info(f"# Final loss_ntp:        {all_losses[-1]:.4f}")
    logger.info(f"# Min loss_ntp:          {min(all_losses):.4f}")
    logger.info(f"# Baseline loss_ntp:     {loss_1b:.4f}")
    logger.info(f"# Delta (final-base):    {all_losses[-1] - loss_1b:.4f}")
    if WARMUP_UPDATES > 0:
        logger.info(f"# Pre-warmup avg (0-{WARMUP_UPDATES-1}):  {sum(all_losses[:WARMUP_UPDATES]) / WARMUP_UPDATES:.4f}")
    else:
        logger.info("# Pre-warmup avg: N/A (Warmup Skipped)")
    logger.info(f"# Post-warmup avg ({WARMUP_UPDATES}+):  {sum(all_losses[WARMUP_UPDATES:]) / max(len(all_losses) - WARMUP_UPDATES, 1):.4f}")

    print(f"\nExp 4 training complete.")
    print(f"  Initial loss: {all_losses[0]:.4f}")
    print(f"  Final loss:   {all_losses[-1]:.4f}")
    print(f"  Baseline:     {loss_1b:.4f}")


if __name__ == "__main__":
    main()