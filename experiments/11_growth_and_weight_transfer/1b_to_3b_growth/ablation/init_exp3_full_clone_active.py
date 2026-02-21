"""
Experiment 3 — Initializer: Full Clone + Active Routing (No SVD Compression)

Clones dense FFN weights to routed experts at FULL dimension (1024, no SVD),
with small rotation for diversity and active routing bias (-2.65/+2.65).

This isolates routing effects from compression effects.

Key difference from Exp1/2:
  - expert_intermediate_size = 1024 (same as shared, no compression)
  - Routing is active (not forced to null)
  - Custom init (no SpectralMoEInitializer, which always compresses)

Usage:
    cd endGame && python -m ablation.init_exp3_full_clone_active
"""

import os
import sys
import gc
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation.common import (
    load_tokenizer, create_kronecker_codec,
    create_1b_model, create_3b_model, set_active_routing_bias,
    random_small_rotation, ENDGAME_DIR,
)
from recurrence_model_1b import ModelConfig as Config1B

BASELINE_CHECKPOINT = os.path.join(ENDGAME_DIR, "checkpoints", "dense_1b_baseline", "kronecker_latest.pt")
SAVE_PATH = os.path.join(ENDGAME_DIR, "checkpoints", "exp3_full_clone", "init.pt")
ROTATION_EPS = 0.005


def copy_non_moe_components(model_1b, model_3b):
    """Copy all non-MoE components from 1B to 3B (identical to SpectralMoEInitializer)."""
    # Embeddings
    if hasattr(model_1b, 'kronecker_embeddings') and hasattr(model_3b, 'kronecker_embeddings'):
        model_3b.kronecker_embeddings.load_state_dict(model_1b.kronecker_embeddings.state_dict())
    elif hasattr(model_1b, 'token_embed') and hasattr(model_3b, 'token_embed'):
        model_3b.token_embed.load_state_dict(model_1b.token_embed.state_dict())

    # Embedding norm
    if hasattr(model_1b, 'embed_norm') and hasattr(model_3b, 'embed_norm'):
        if model_1b.embed_norm is not None and model_3b.embed_norm is not None:
            model_3b.embed_norm.load_state_dict(model_1b.embed_norm.state_dict())

    # Final norm and head
    model_3b.norm.load_state_dict(model_1b.norm.state_dict())
    model_3b.lm_head.load_state_dict(model_1b.lm_head.state_dict())

    # Kronecker projection
    if hasattr(model_1b, 'pf_to_model') and hasattr(model_3b, 'pf_to_model'):
        if model_1b.pf_to_model is not None and model_3b.pf_to_model is not None:
            model_3b.pf_to_model.load_state_dict(model_1b.pf_to_model.state_dict())

    # Memory gate
    if hasattr(model_1b, 'memory_gate_proj') and hasattr(model_3b, 'memory_gate_proj'):
        model_3b.memory_gate_proj.load_state_dict(model_1b.memory_gate_proj.state_dict())


def clone_ffn_to_experts(dense_mlp, moe_layer, num_experts, device):
    """
    Clone dense FFN weights to routed experts at full dimension.
    Each expert gets the dense weights + small rotation for diversity.
    NO SVD compression.
    """
    # Get dense weights
    # nn.Linear convention: weight is (out_features, in_features)
    # gate_proj: (intermediate, hidden) = (1024, 512)
    # up_proj:   (intermediate, hidden) = (1024, 512)
    # down_proj: (hidden, intermediate) = (512, 1024)
    Wg = dense_mlp.shared_gate.weight.data.to(device)  # (1024, 512)
    Wu = dense_mlp.shared_up.weight.data.to(device)     # (1024, 512)
    Wd = dense_mlp.shared_down.weight.data.to(device)   # (512, 1024)

    # Copy shared expert
    moe_layer.shared_gate.weight.data.copy_(Wg)
    moe_layer.shared_up.weight.data.copy_(Wu)
    moe_layer.shared_down.weight.data.copy_(Wd)

    intermediate_size = Wg.shape[0]  # 1024

    # Clone to each routed expert with small rotation
    for e in range(num_experts):
        R = random_small_rotation(intermediate_size, eps=ROTATION_EPS, device=device)

        # Rotate in intermediate space
        Wg_e = R @ Wg         # (1024, 512) — rotated gate
        Wu_e = R @ Wu         # (1024, 512) — rotated up
        Wd_e = Wd @ R.T       # (512, 1024) — rotated down

        # MoEFFN batched format: W_gate[e] has shape (d_model, d_hidden)
        # chunk_x @ W_gate[e]: (tokens, 512) @ (512, 1024) -> (tokens, 1024)
        # So W_gate[e] = Wg_e.T = (512, 1024)
        moe_layer.W_gate.data[e] = Wg_e.T  # (512, 1024)
        moe_layer.W_up.data[e] = Wu_e.T    # (512, 1024)
        # W_down[e]: h @ W_down[e]: (tokens, 1024) @ (1024, 512) -> (tokens, 512)
        # So W_down[e] = Wd_e.T = (1024, 512)
        moe_layer.W_down.data[e] = Wd_e.T  # (1024, 512)


def main():
    print("=" * 80)
    print("EXP 3 INIT: Full Clone + Active Routing (No Compression)")
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

    # 2. Create 3B model with FULL intermediate size (no compression)
    print("\nCreating 3B MoE model with expert_intermediate_size=1024 (no compression)...")
    model_3b, config_3b = create_3b_model(
        device, bpe_vocab, pf_codec,
        config_overrides={"expert_intermediate_size": Config1B.shared_expert_intermediate_size},
    )

    # Verify shapes
    sample_moe = model_3b.layers[0].mlp_block.sublayer.moe
    print(f"  W_gate shape: {sample_moe.W_gate.shape}")  # Expect (8, 512, 1024)
    print(f"  W_up shape:   {sample_moe.W_up.shape}")
    print(f"  W_down shape: {sample_moe.W_down.shape}")

    # 3. Copy non-MoE components
    print("\nCopying non-MoE components...")
    copy_non_moe_components(model_1b, model_3b)

    # 4. Copy attention + MLP wrapper (coeffs + norm) per layer
    print("\nCopying layer weights...")
    for layer_idx in range(len(model_1b.layers)):
        dense_block = model_1b.layers[layer_idx]
        moe_block = model_3b.layers[layer_idx]

        # Attention (full sublayer)
        moe_block.attn_block.load_state_dict(dense_block.attn_block.state_dict())

        # MLP wrapper (coeffs + norm only, not sublayer itself)
        moe_block.mlp_block.coeffs.load_state_dict(dense_block.mlp_block.coeffs.state_dict())
        moe_block.mlp_block.norm.load_state_dict(dense_block.mlp_block.norm.state_dict())

        # Clone FFN to experts (full dimension, no SVD)
        dense_mlp = dense_block.mlp_block.sublayer.moe
        moe_layer = moe_block.mlp_block.sublayer.moe
        clone_ffn_to_experts(dense_mlp, moe_layer, config_3b.num_real_experts, device)

    # 5. Initialize MTP block
    if hasattr(model_1b, 'mtp_block') and hasattr(model_3b, 'mtp_block'):
        if model_1b.mtp_block is not None and model_3b.mtp_block is not None:
            print("\nInitializing MTP block...")
            dense_mtp = model_1b.mtp_block
            moe_mtp = model_3b.mtp_block

            # Copy standard components
            moe_mtp.fusion_proj.load_state_dict(dense_mtp.fusion_proj.state_dict())
            moe_mtp.attn.load_state_dict(dense_mtp.attn.state_dict())
            moe_mtp.attn_block.load_state_dict(dense_mtp.attn_block.state_dict())
            moe_mtp.mlp_block.coeffs.load_state_dict(dense_mtp.mlp_block.coeffs.state_dict())
            moe_mtp.mlp_block.norm.load_state_dict(dense_mtp.mlp_block.norm.state_dict())

            # Clone MTP FFN to experts
            dense_mtp_mlp = dense_mtp.mlp.moe
            moe_mtp_layer = moe_mtp.mlp.moe
            clone_ffn_to_experts(dense_mtp_mlp, moe_mtp_layer, config_3b.num_real_experts, device)

    # 6. Set active routing bias
    print("\nSetting active routing bias...")
    set_active_routing_bias(model_3b, logit_bias=0.0, null_logit=0.0)

    # 7. Validate expert diversity
    print("\nValidating expert diversity...")
    for layer_idx, layer in enumerate(model_3b.layers):
        moe_layer = layer.mlp_block.sublayer.moe
        gate_weights = moe_layer.W_gate.data.view(config_3b.num_real_experts, -1)
        norm = gate_weights.norm(p=2, dim=1, keepdim=True)
        normalized = gate_weights / (norm + 1e-8)
        sim_matrix = torch.mm(normalized, normalized.t())
        mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
        mean_sim = sim_matrix[mask].mean().item()
        print(f"  Layer {layer_idx}: mean cosine sim = {mean_sim:.4f}")

    # 8. Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save({
        "model_state_dict": model_3b.state_dict(),
        "experiment": "exp3_full_clone_active",
        "source_checkpoint": BASELINE_CHECKPOINT,
        "source_step": checkpoint.get("step", 0),
        "source_loss": checkpoint.get("loss", 0),
        "expert_intermediate_size": Config1B.shared_expert_intermediate_size,
        "compression": "none",
        "rotation_eps": ROTATION_EPS,
        "routing_bias": "0.0/0.0",
        "config_overrides": {"expert_intermediate_size": Config1B.shared_expert_intermediate_size},
    }, SAVE_PATH)
    print(f"\nSaved: {SAVE_PATH}")

    del model_1b, model_3b
    gc.collect()
    print("\nExp 3 init complete. Run train_exp3_full_clone.py next.")


if __name__ == "__main__":
    main()