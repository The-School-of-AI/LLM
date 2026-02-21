#!/usr/bin/env python3
"""
Experiment 4 — Initializer: SVD Compression + Active Routing (Full MoE)

Production-grade spectral initialization:
  - SVD compression 1024 -> 512
  - Structured rotation (eps=0.005)
  - Active routing bias (-2.65/+2.65)

This is the real dense-to-MoE transition test.

Usage:
    cd endGame && python -m ablation.init_exp4_svd_active
"""

import os
import sys
import gc
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation.common import (
    load_tokenizer, create_kronecker_codec,
    create_1b_model, create_3b_model, ENDGAME_DIR,
    set_active_routing_bias,
)
from spectral_moe_initializer import SpectralMoEInitializer
from recurrence_model_1b import ModelConfig as Config1B

BASELINE_CHECKPOINT = os.path.join(ENDGAME_DIR, "checkpoints", "dense_1b_baseline", "kronecker_latest.pt")
SAVE_PATH = os.path.join(ENDGAME_DIR, "checkpoints", "exp4_svd_moe", "init.pt")


def main():
    print("=" * 80)
    print("EXP 4 INIT: SVD Compression + Active Routing (Full MoE)")
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
    print(f"  Loaded from step {checkpoint.get('step', '?')}, loss={checkpoint.get('loss', '?')}")

    # 2. Create 3B MoE model (default config: expert_intermediate_size=512)
    print("\nCreating 3B MoE model (default config, SVD compression)...")
    model_3b, config_3b = create_3b_model(device, bpe_vocab, pf_codec)

    # 3. Full spectral initialization (production pipeline)
    print("\nRunning SpectralMoEInitializer (SVD + rotation)...")
    print(f"  Compression: {Config1B.shared_expert_intermediate_size} -> {config_3b.expert_intermediate_size}")
    print(f"  Rotation eps: 0.005")

    initializer = SpectralMoEInitializer(
        dense_model=model_1b,
        moe_model=model_3b,
        num_routed_experts=config_3b.num_real_experts,
        intermediate_dense=Config1B.shared_expert_intermediate_size,
        intermediate_moe=config_3b.expert_intermediate_size,
        rotation_eps=0.005,
        device=device,
        svd_mode="independent", # joint for joint Stacked Wg, Wu, Wd compression (Better)
    )
    initializer.initialize()

    # 4. Set Active Routing Bias
    # With expert_output_scale=0.01, routed experts contribute <1% at init,
    # so the loss spike is prevented regardless of routing. Moderate null bias
    # (-1.0/+1.0, gap=2.0) gives gentle guidance without suppression.
    print("\nSetting moderate routing bias (-1.0/+1.0)...")
    set_active_routing_bias(model_3b, logit_bias=0.0, null_logit=-1.0)
    print("✅ Active Routing Bias: logit_bias=-1.0, null_logit=+1.0 (moderate, expert_output_scale handles safety)")
    # 5. Validate expert diversity
    print("\nExpert diversity report:")
    for layer_idx, layer in enumerate(model_3b.layers):
        moe_layer = layer.mlp_block.sublayer.moe
        if hasattr(moe_layer, 'W_gate') and isinstance(moe_layer.W_gate, nn.Parameter):
            gate_weights = moe_layer.W_gate.data.view(config_3b.num_real_experts, -1)
            norm = gate_weights.norm(p=2, dim=1, keepdim=True)
            normalized = gate_weights / (norm + 1e-8)
            sim_matrix = torch.mm(normalized, normalized.t())
            mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
            mean_sim = sim_matrix[mask].mean().item()
            min_sim = sim_matrix[mask].min().item()
            max_sim = sim_matrix[mask].max().item()
            status = "OK" if 0.90 <= mean_sim <= 0.99 else "CHECK"
            print(f"  Layer {layer_idx}: mean={mean_sim:.4f} [{min_sim:.3f}, {max_sim:.3f}] {status}")

    # 6. Verify router biases
    print("\nRouter bias check:")
    for layer_idx, layer in enumerate(model_3b.layers):
        moe = layer.mlp_block.sublayer.moe
        if hasattr(moe, 'gate') and moe.gate is not None:
            lb = moe.gate.logit_bias.data.mean().item()
            nl = moe.gate.null_logit.data.item()
            print(f"  Layer {layer_idx}: logit_bias={lb:.2f}, null_logit={nl:.2f}, gap={nl-lb:.2f}")

    # 7. Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save({
        "model_state_dict": model_3b.state_dict(),
        "experiment": "exp4_svd_active",
        "source_checkpoint": BASELINE_CHECKPOINT,
        "source_step": checkpoint.get("step", 0),
        "source_loss": checkpoint.get("loss", 0),
        "svd_compression": f"{Config1B.shared_expert_intermediate_size} -> {config_3b.expert_intermediate_size}",
        "rotation_eps": 0.005,
        "routing_bias": "-1.0/+1.0",
    }, SAVE_PATH)
    print(f"\nSaved: {SAVE_PATH}")

    del model_1b, model_3b
    gc.collect()
    print("\nExp 4 init complete. Run train_exp4_svd_moe.py next.")


if __name__ == "__main__":
    main()