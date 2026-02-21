#!/usr/bin/env python3
"""
Experiment 2 — Evaluation: SVD Init + Null Routing Equivalence

Same test as Experiment 1 but with SVD-initialized routed experts.
Since routing is forced null, output should still match 1B dense.
This verifies the SVD initialization code doesn't corrupt the shared expert.

Usage:
    cd endGame && python -m ablation.eval_exp2_svd_equivalence
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation.common import (
    load_tokenizer, create_kronecker_codec,
    create_1b_model, create_3b_model, get_reference_batch,
    setup_logging, ENDGAME_DIR,
)

BASELINE_CHECKPOINT = os.path.join(ENDGAME_DIR, "checkpoints", "dense_1b_baseline", "kronecker_latest.pt")
EXP2_CHECKPOINT = os.path.join(ENDGAME_DIR, "checkpoints", "exp2_svd_null", "init.pt")
LOG_PATH = os.path.join(ENDGAME_DIR, "logs", "exp2_svd_null.log")
THRESHOLD = 1e-4


def main():
    print("=" * 80)
    print("EXP 2 EVAL: SVD Init + Null Routing Equivalence")
    print("=" * 80)

    device = torch.device("cpu")
    logger = setup_logging(LOG_PATH)
    tokenizer, bpe_vocab = load_tokenizer()
    pf_codec = create_kronecker_codec(tokenizer.vocab_size)

    for path, name in [(BASELINE_CHECKPOINT, "1B baseline"), (EXP2_CHECKPOINT, "Exp2 init")]:
        if not os.path.exists(path):
            print(f"  ERROR: {name} not found: {path}")
            return

    # Load models
    print("\nLoading 1B dense model...")
    model_1b = create_1b_model(device, bpe_vocab, pf_codec)
    model_1b.load_state_dict(torch.load(BASELINE_CHECKPOINT, map_location=device)["model_state_dict"])

    print("\nLoading 3B MoE model (SVD init, null routing)...")
    model_3b, _ = create_3b_model(device, bpe_vocab, pf_codec)
    model_3b.load_state_dict(torch.load(EXP2_CHECKPOINT, map_location=device)["model_state_dict"])

    # Reference batch
    x_input, y_ntp, y_mtp = get_reference_batch(tokenizer, device)

    # Compare
    criterion = nn.CrossEntropyLoss()
    model_1b.eval()
    model_3b.eval()

    with torch.no_grad():
        logits_1b, logits_mtp_1b, aux_1b = model_1b(
            x_input, next_token_ids=y_ntp, return_loss=True, return_memory=False,
        )
        logits_3b, logits_mtp_3b, aux_3b = model_3b(
            x_input, next_token_ids=y_ntp, return_loss=True, return_memory=False,
        )

        V = logits_1b.size(-1)
        loss_ntp_1b = criterion(logits_1b.view(-1, V), y_ntp.view(-1)).item()
        loss_ntp_3b = criterion(logits_3b.view(-1, V), y_ntp.view(-1)).item()
        loss_mtp_1b = criterion(logits_mtp_1b.view(-1, V), y_mtp.view(-1)).item() if logits_mtp_1b is not None else 0.0
        loss_mtp_3b = criterion(logits_mtp_3b.view(-1, V), y_mtp.view(-1)).item() if logits_mtp_3b is not None else 0.0

        logit_diff_ntp = (logits_1b - logits_3b).abs().max().item()

    loss_diff_ntp = abs(loss_ntp_1b - loss_ntp_3b)
    loss_diff_mtp = abs(loss_mtp_1b - loss_mtp_3b)
    pass_ntp = loss_diff_ntp < THRESHOLD
    pass_mtp = loss_diff_mtp < THRESHOLD

    logger.info("=" * 60)
    logger.info("EXP 2: SVD Init + Null Routing Equivalence Results")
    logger.info("=" * 60)
    logger.info(f"")
    logger.info(f"1B Dense:      loss_ntp={loss_ntp_1b:.6f}  loss_mtp={loss_mtp_1b:.6f}")
    logger.info(f"3B SVD+Null:   loss_ntp={loss_ntp_3b:.6f}  loss_mtp={loss_mtp_3b:.6f}")
    logger.info(f"")
    logger.info(f"NTP loss diff: {loss_diff_ntp:.6e}  {'PASS' if pass_ntp else 'FAIL'}")
    logger.info(f"MTP loss diff: {loss_diff_mtp:.6e}  {'PASS' if pass_mtp else 'FAIL'}")
    logger.info(f"NTP logit max diff: {logit_diff_ntp:.6e}")
    logger.info(f"")

    all_pass = pass_ntp and pass_mtp
    if all_pass:
        logger.info("RESULT: PASS - SVD init does not corrupt shared expert path")
    else:
        logger.info("RESULT: FAIL - SVD init corrupted shared expert weights!")
        if not pass_ntp and pass_mtp:
            logger.info("  NTP failed but MTP passed: check backbone layer copy")
        elif pass_ntp and not pass_mtp:
            logger.info("  MTP failed but NTP passed: check MTP block copy")

    logger.info("=" * 60)
    print(f"\n{'PASS' if all_pass else 'FAIL'} | NTP diff={loss_diff_ntp:.6e} | MTP diff={loss_diff_mtp:.6e}")


if __name__ == "__main__":
    main()