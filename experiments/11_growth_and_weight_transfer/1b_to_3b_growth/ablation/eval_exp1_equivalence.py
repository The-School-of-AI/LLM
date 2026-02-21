#!/usr/bin/env python3
"""
Experiment 1 — Evaluation: Architecture Equivalence (Null Routing)

Compares forward pass output of:
  - 1B dense model (baseline)
  - 3B MoE model with null routing forced

Expected: loss difference < 1e-4

Usage:
    cd endGame && python -m ablation.eval_exp1_equivalence
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation.common import (
    detect_device, load_tokenizer, create_kronecker_codec,
    create_1b_model, create_3b_model, get_reference_batch,
    setup_logging, ENDGAME_DIR,
)

BASELINE_CHECKPOINT = os.path.join(ENDGAME_DIR, "checkpoints", "dense_1b_baseline", "kronecker_latest.pt")
EXP1_CHECKPOINT = os.path.join(ENDGAME_DIR, "checkpoints", "exp1_null_routing", "init.pt")
LOG_PATH = os.path.join(ENDGAME_DIR, "logs", "exp1_null_routing.log")
THRESHOLD = 1e-4


def main():
    print("=" * 80)
    print("EXP 1 EVAL: Architecture Equivalence Test")
    print("=" * 80)

    device = torch.device("cpu")  # CPU for deterministic comparison
    logger = setup_logging(LOG_PATH)
    tokenizer, bpe_vocab = load_tokenizer()
    pf_codec = create_kronecker_codec(tokenizer.vocab_size)

    # Check prerequisites
    for path, name in [(BASELINE_CHECKPOINT, "1B baseline"), (EXP1_CHECKPOINT, "Exp1 init")]:
        if not os.path.exists(path):
            print(f"  ERROR: {name} checkpoint not found: {path}")
            return

    # Load 1B model
    print("\nLoading 1B dense model...")
    model_1b = create_1b_model(device, bpe_vocab, pf_codec)
    ckpt_1b = torch.load(BASELINE_CHECKPOINT, map_location=device)
    model_1b.load_state_dict(ckpt_1b["model_state_dict"])

    # Load 3B model
    print("\nLoading 3B MoE model (null routing)...")
    model_3b, _ = create_3b_model(device, bpe_vocab, pf_codec)
    ckpt_3b = torch.load(EXP1_CHECKPOINT, map_location=device)
    model_3b.load_state_dict(ckpt_3b["model_state_dict"])

    # Get reference batch
    print("\nPreparing reference batch...")
    x_input, y_ntp, y_mtp = get_reference_batch(tokenizer, device)
    print(f"  x_input: {x_input.shape}, y_ntp: {y_ntp.shape}, y_mtp: {y_mtp.shape}")

    # Forward pass comparison
    print("\nRunning forward passes...")
    criterion = nn.CrossEntropyLoss()

    model_1b.eval()
    model_3b.eval()

    with torch.no_grad():
        # 1B forward
        logits_1b, logits_mtp_1b, aux_1b = model_1b(
            x_input, next_token_ids=y_ntp, return_loss=True, return_memory=False,
        )

        # 3B forward
        logits_3b, logits_mtp_3b, aux_3b = model_3b(
            x_input, next_token_ids=y_ntp, return_loss=True, return_memory=False,
        )

        # Compute losses
        V = logits_1b.size(-1)
        loss_ntp_1b = criterion(logits_1b.view(-1, V), y_ntp.view(-1)).item()
        loss_ntp_3b = criterion(logits_3b.view(-1, V), y_ntp.view(-1)).item()

        loss_mtp_1b = criterion(logits_mtp_1b.view(-1, V), y_mtp.view(-1)).item() if logits_mtp_1b is not None else 0.0
        loss_mtp_3b = criterion(logits_mtp_3b.view(-1, V), y_mtp.view(-1)).item() if logits_mtp_3b is not None else 0.0

        # Logit differences
        logit_diff_ntp = (logits_1b - logits_3b).abs().max().item()
        logit_diff_mtp = 0.0
        if logits_mtp_1b is not None and logits_mtp_3b is not None:
            logit_diff_mtp = (logits_mtp_1b - logits_mtp_3b).abs().max().item()

    # Results
    loss_diff_ntp = abs(loss_ntp_1b - loss_ntp_3b)
    loss_diff_mtp = abs(loss_mtp_1b - loss_mtp_3b)

    pass_ntp = loss_diff_ntp < THRESHOLD
    pass_mtp = loss_diff_mtp < THRESHOLD
    pass_logit = logit_diff_ntp < THRESHOLD

    logger.info("=" * 60)
    logger.info("EXP 1: Architecture Equivalence Results")
    logger.info("=" * 60)
    logger.info(f"")
    logger.info(f"1B Dense:")
    logger.info(f"  loss_ntp = {loss_ntp_1b:.6f}")
    logger.info(f"  loss_mtp = {loss_mtp_1b:.6f}")
    logger.info(f"  aux_loss = {aux_1b.item():.6f}")
    logger.info(f"")
    logger.info(f"3B MoE (null routing):")
    logger.info(f"  loss_ntp = {loss_ntp_3b:.6f}")
    logger.info(f"  loss_mtp = {loss_mtp_3b:.6f}")
    logger.info(f"  aux_loss = {aux_3b.item():.6f} (expected to differ - gate computes L_bal/L_z)")
    logger.info(f"")
    logger.info(f"Differences:")
    logger.info(f"  NTP loss diff:   {loss_diff_ntp:.6e}  {'PASS' if pass_ntp else 'FAIL'} (threshold={THRESHOLD})")
    logger.info(f"  MTP loss diff:   {loss_diff_mtp:.6e}  {'PASS' if pass_mtp else 'FAIL'} (threshold={THRESHOLD})")
    logger.info(f"  NTP logit diff:  {logit_diff_ntp:.6e}  {'PASS' if pass_logit else 'FAIL'}")
    logger.info(f"  MTP logit diff:  {logit_diff_mtp:.6e}")
    logger.info(f"")

    all_pass = pass_ntp and pass_mtp
    if all_pass:
        logger.info("RESULT: PASS - 3B MoE with null routing == 1B Dense")
    else:
        logger.info("RESULT: FAIL - Architecture mismatch detected!")
        logger.info("  Investigate weight copy path (embeddings, attention, shared expert)")

    logger.info("=" * 60)

    print(f"\n{'PASS' if all_pass else 'FAIL'} | NTP diff={loss_diff_ntp:.6e} | MTP diff={loss_diff_mtp:.6e}")


if __name__ == "__main__":
    main()