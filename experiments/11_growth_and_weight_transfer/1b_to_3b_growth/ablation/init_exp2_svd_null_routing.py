#!/usr/bin/env python3
"""
Experiment 2 — Initializer: SVD Compression + Null Routing

Full spectral initialization (SVD compression + rotation for routed experts),
but routing is forced to null. This verifies SVD init doesn't corrupt the
shared expert path.

Usage:
    cd endGame && python -m ablation.init_exp2_svd_null_routing
"""

import os
import sys
import gc
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation.common import (
    detect_device, load_tokenizer, create_kronecker_codec,
    create_1b_model, create_3b_model, force_null_routing,
    ENDGAME_DIR,
)
from spectral_moe_initializer import SpectralMoEInitializer
from recurrence_model_1b import ModelConfig as Config1B
from recurrence_model_3b import ModelConfig as Config3B

BASELINE_CHECKPOINT = os.path.join(ENDGAME_DIR, "checkpoints", "dense_1b_baseline", "kronecker_latest.pt")
SAVE_PATH = os.path.join(ENDGAME_DIR, "checkpoints", "exp2_svd_null", "init.pt")


def main():
    print("=" * 80)
    print("EXP 2 INIT: SVD Compression + Null Routing")
    print("=" * 80)

    device = torch.device("cpu")
    tokenizer, bpe_vocab = load_tokenizer()
    pf_codec = create_kronecker_codec(tokenizer.vocab_size)

    # 1. Load 1B baseline
    print("\nLoading 1B dense baseline...")
    model_1b = create_1b_model(device, bpe_vocab, pf_codec)
    if not os.path.exists(BASELINE_CHECKPOINT):
        print(f"  ERROR: Baseline checkpoint not found: {BASELINE_CHECKPOINT}")
        return
    checkpoint = torch.load(BASELINE_CHECKPOINT, map_location=device)
    model_1b.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded from step {checkpoint.get('step', '?')}")

    # 2. Create 3B MoE model
    print("\nCreating 3B MoE model...")
    model_3b, config_3b = create_3b_model(device, bpe_vocab, pf_codec)

    # 3. Full spectral initialization (SVD + rotation + bias)
    # This is the production init pipeline
    print("\nRunning full SpectralMoEInitializer...")
    initializer = SpectralMoEInitializer(
        dense_model=model_1b,
        moe_model=model_3b,
        num_routed_experts=config_3b.num_real_experts,
        intermediate_dense=Config1B.shared_expert_intermediate_size,
        intermediate_moe=config_3b.expert_intermediate_size,
        rotation_eps=0.005,
        device=device,
    )
    initializer.initialize()

    # 4. OVERRIDE: Force null routing (after full SVD init)
    # This means routed experts ARE properly initialized via SVD,
    # but they will never fire. Only shared expert contributes.
    print("\nOverriding: forcing null routing...")
    force_null_routing(model_3b, logit_bias=-100.0, null_logit=100.0)

    # 5. Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save({
        "model_state_dict": model_3b.state_dict(),
        "experiment": "exp2_svd_null_routing",
        "source_checkpoint": BASELINE_CHECKPOINT,
        "source_step": checkpoint.get("step", 0),
        "source_loss": checkpoint.get("loss", 0),
        "svd_compression": f"{Config1B.shared_expert_intermediate_size} -> {config_3b.expert_intermediate_size}",
        "rotation_eps": 0.005,
    }, SAVE_PATH)
    print(f"\nSaved: {SAVE_PATH}")

    del model_1b, model_3b
    gc.collect()
    print("\nExp 2 init complete. Run eval_exp2_svd_equivalence.py next.")


if __name__ == "__main__":
    main()