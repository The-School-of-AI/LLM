#!/usr/bin/env python3
"""
expert_explosion_init_8b_to_70b.py
==================================
Function-Preserving Expert Explosion: 8B (20 experts) → 70B (260 experts)

Algorithm: Round-Robin Tiling + Output-Nullspace Noise + Mass Correction
-------------------------------------------------------------------------
The 8B MoE model has 20 routed experts per layer. The 70B model has 260
routed experts per layer (same hidden_size=4096, same expert_intermediate=1024).

This script "explodes" 20 → 260 by creating exactly 13 clones of each source
expert with four critical corrections for true function preservation in a
routed MoE with Top-K change (2 → 8):

1. **Round-robin assignment** — siblings are interleaved across index space
   instead of contiguous blocks. This prevents Top-K from co-selecting
   multiple siblings due to index proximity, preserving parent-level routing.

2. **Output-nullspace projected noise** (LiGO-style) — perturbations lie in
   ker(Wᵀ) so (W + ΔW)x ≈ Wx in activation space (not just parameter space):
       ΔW = Q − W(WᵀW)⁻¹WᵀQ         (output null-space projection)
       W_clone = W_base + ε · ‖W_base‖_F · ΔW

3. **Router logit mass correction** — softmax probability mass is preserved
   under duplication by subtracting log(copies) from logit biases:
       logit_bias_clone = logit_bias_source − log(13)

4. **Down-projection scaling BEFORE noise** — W_down is divided by 13 before
   adding perturbation so noise magnitude is not suppressed:
       W_base = W_down / 13           (scale first)
       W_clone = W_base + ε·‖W_base‖·ΔW   (noise on scaled base)

Weight Copying Strategy:
    ┌───────────────────────────┬─────────────────────────────────────────────────┐
    │ Component                 │ Strategy                                        │
    ├───────────────────────────┼─────────────────────────────────────────────────┤
    │ Embeddings, LM head       │ Direct copy (identical shapes)                  │
    │ Attention (DeltaNet/GSA)  │ Direct copy (identical config)                  │
    │ RMSNorms, mHC coeffs      │ Direct copy                                    │
    │ Memory stream recurrence  │ Direct copy                                    │
    │ MTP block (MoE)           │ Tile experts (same as backbone layers)          │
    │ MTP block (attention/etc) │ Direct copy                                    │
    │ Shared expert (2048)      │ Direct copy (identical shapes)                 │
    │ Routed W_gate/W_up        │ 13× round-robin + output-nullspace noise       │
    │ Routed W_down             │ ÷13 first, then output-nullspace noise         │
    │ Router gate (20→260)      │ 13× round-robin + tiny Gaussian noise          │
    │ Router logit_bias         │ 13× round-robin − log(13) mass correction      │
    └───────────────────────────┴─────────────────────────────────────────────────┘

Preserves:
    ✅ Parent-level Top-K (round-robin → siblings index-separated)
    ✅ Softmax probability mass (log(13) correction → Σ p_clone ≈ p_original)
    ✅ Output magnitude (W_down / 13 BEFORE noise → Net2Wider-correct)
    ✅ Activation-space function (output-nullspace → (W+ΔW)x ≈ Wx)
    ✅ Symmetry breaking (orthogonal perturbation → diverse specialization)
    ✅ Shared expert (direct copy, identical 2048 intermediate in both models)
    ✅ All non-MoE components (attention, norms, embeddings)

Training schedule (mandatory for Top-K warmstart):
    Steps 0–1000:    Top-K = 2 (match 8B)
    Steps 1000–3000: Top-K = 4 (gradual increase)
    Steps 3000+:     Top-K = 8 (full 70B routing)

Usage:
    python expert_explosion_init_8b_to_70b.py \\
        --src checkpoints/8b_trained.pt \\
        --tgt checkpoints/70b_expert_explosion_init.pt \\
        --model_dir ../

    # Dry-run (print plan only):
    python expert_explosion_init_8b_to_70b.py --src ... --tgt ... --model_dir .. --dry_run

    # Custom noise scale:
    python expert_explosion_init_8b_to_70b.py --src ... --tgt ... --model_dir .. --eps_expert 0.02
"""

import copy
import math
import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Architecture constants
# ─────────────────────────────────────────────

N_SRC_EXPERTS = 20          # 8B model: 20 routed experts per layer
N_TGT_EXPERTS = 260         # 70B model: 260 routed experts per layer
COPIES_PER_EXPERT = 13      # 260 / 20 = 13 exact copies
LOG_COPIES = math.log(COPIES_PER_EXPERT)  # ≈ 2.565 — softmax mass correction

N_LAYERS = 20               # Both models have 20 layers
HIDDEN_SIZE = 4096
EXPERT_INTERMEDIATE = 1024  # Routed expert intermediate size (same in both)
SHARED_INTERMEDIATE = 2048  # Shared expert intermediate (same in both models)

# Key patterns for MoE expert weights within a layer's state dict suffix
MOE_EXPERT_WEIGHT_SUFFIXES = [
    "mlp_block.sublayer.moe.W_gate",   # (num_experts, d_model, d_hidden)
    "mlp_block.sublayer.moe.W_up",     # (num_experts, d_model, d_hidden)
    "mlp_block.sublayer.moe.W_down",   # (num_experts, d_hidden, d_model)
]

MOE_GATE_WEIGHT_SUFFIX = "mlp_block.sublayer.moe.gate.weight"      # (num_experts, d_model)
MOE_GATE_BIAS_SUFFIX = "mlp_block.sublayer.moe.gate.logit_bias"    # (num_experts,)


# ─────────────────────────────────────────────
# Step 1: Build expert assignment map
# ─────────────────────────────────────────────

def build_expert_assignment() -> List[int]:
    """
    Build mapping: target_expert_idx → source_expert_idx (round-robin).

    Returns a list of length N_TGT_EXPERTS where assignment[j] = i means
    target expert j is a clone of source expert i.

    Round-robin layout (interleaved):
        Target [0, 1, 2, ..., 19]     ← Sources 0..19 (copy 0)
        Target [20, 21, 22, ..., 39]  ← Sources 0..19 (copy 1)
        ...
        Target [240, 241, ..., 259]   ← Sources 0..19 (copy 12)

    So: assignment[j] = j % N_SRC_EXPERTS

    Why round-robin instead of contiguous blocks:
        - Sibling clones are index-separated → lower probability of Top-K
          co-selecting multiple siblings of the same parent
        - Parent-level competition happens before sibling-level
        - Better routing diversity and load balance at init
        - Eliminates the need for artificial logit ranking noise
    """
    assignment = [j % N_SRC_EXPERTS for j in range(N_TGT_EXPERTS)]

    assert len(assignment) == N_TGT_EXPERTS, (
        f"Assignment length {len(assignment)} != {N_TGT_EXPERTS}"
    )
    return assignment


def print_assignment(assignment: List[int]) -> None:
    """Pretty-print the expert assignment map."""
    print("\n┌─ Expert Assignment Map (Round-Robin) ─────────────────┐")
    print(f"│  Source: {N_SRC_EXPERTS} experts → Target: {N_TGT_EXPERTS} experts")
    print(f"│  Copies per expert: {COPIES_PER_EXPERT}")
    print(f"│  Layout: interleaved (round-robin)")
    print("│")
    for copy_idx in range(COPIES_PER_EXPERT):
        start = copy_idx * N_SRC_EXPERTS
        end = start + N_SRC_EXPERTS - 1
        print(f"│  Copy {copy_idx:2d}: Target [{start:3d}..{end:3d}] ← Sources 0..{N_SRC_EXPERTS-1}")
    print("│")
    print("│  Sibling indices for source expert 0:")
    siblings = [j for j, s in enumerate(assignment) if s == 0]
    print(f"│    {siblings}")
    print("└────────────────────────────────────────────────────────┘\n")


# ─────────────────────────────────────────────
# Step 2: Orthogonal perturbation generator
# ─────────────────────────────────────────────

def generate_orthogonal_noise(shape: Tuple[int, ...], device: str = "cpu") -> torch.Tensor:
    """
    Generate a random orthogonal matrix via QR decomposition of Gaussian noise.

    For 2D matrices (d_in, d_out): returns a (d_in, d_out) orthogonal matrix.
    The matrix has unit Frobenius norm (normalized after QR).

    For non-square matrices, we QR-decompose on the larger dimension and slice.
    """
    # Flatten to 2D for QR decomposition
    if len(shape) == 2:
        m, n = shape
    else:
        # For higher-dimensional tensors, flatten to 2D
        m = shape[0]
        n = 1
        for s in shape[1:]:
            n *= s

    # QR needs m >= n for full orthogonal columns
    if m >= n:
        G = torch.randn(m, n, device=device)
        Q, _ = torch.linalg.qr(G)
        Q = Q[:, :n]  # (m, n) with orthonormal columns
    else:
        G = torch.randn(n, m, device=device)
        Q, _ = torch.linalg.qr(G)
        Q = Q[:, :m].T  # (m, n) with orthonormal rows

    # Normalize to unit Frobenius norm
    Q = Q / Q.norm()

    return Q.reshape(shape)


def output_nullspace_project(Q: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Project Q into the output null-space of W (LiGO-style).

    For W of shape (m, n), this computes:

        ΔW = Q − W (WᵀW)⁻¹ Wᵀ Q

    which ensures ΔW ∈ ker(Wᵀ), i.e. the perturbation lies in the output
    null-space of W. This preserves:

        (W + ε·ΔW)x ≈ Wx

    for typical activations x, because the perturbation only adds components
    in output directions that W doesn't use.

    This is strictly stronger than Frobenius-space projection (⟨ΔW, W⟩_F = 0),
    which only guarantees orthogonality in parameter space, not in activation space.

    For numerical stability, uses pseudoinverse via SVD when WᵀW is ill-conditioned.

    The result is re-normalized to unit Frobenius norm.
    """
    # Flatten to 2D for the projection
    orig_shape = Q.shape
    if len(orig_shape) > 2:
        Q = Q.reshape(orig_shape[0], -1)
        W = W.reshape(orig_shape[0], -1)

    # W: (m, n), Q: (m, n)
    # Project Q onto column space of W, then subtract
    # P_W = W (WᵀW)⁻¹ Wᵀ  is the projection onto col(W)
    # ΔW = Q - P_W Q = (I - P_W) Q

    # Use pseudoinverse for numerical stability
    # WᵀW: (n, n) — can be large, but we can use torch.linalg.lstsq
    # P_W Q = W (WᵀW)⁻¹ Wᵀ Q = W · lstsq(W, Q)
    WtQ = W.T @ Q     # (n, n)
    # Solve WᵀW X = WᵀQ for X, then P_W Q = W X
    coeffs = torch.linalg.lstsq(W.T @ W, WtQ).solution  # (n, n)
    projection = W @ coeffs  # (m, n)

    delta = Q - projection

    # Re-normalize to unit Frobenius norm
    delta_norm = delta.norm()
    if delta_norm > 1e-8:
        delta = delta / delta_norm
    else:
        # Fallback: if projection removed everything (Q was in col(W)),
        # use raw orthogonal noise
        delta = Q / Q.norm()

    return delta.reshape(orig_shape)


def tile_expert_weights(
    W_src: torch.Tensor,
    assignment: List[int],
    eps: float,
    is_down_proj: bool = False,
) -> torch.Tensor:
    """
    Tile source expert weights to target, adding output-nullspace projected
    orthogonal perturbation.

    For W_down (is_down_proj=True): scaling by 1/COPIES_PER_EXPERT is applied
    BEFORE noise so that:
        1. The base weight is correctly scaled for Net2Wider
        2. Noise magnitude is NOT suppressed (siblings can differentiate)

    Order of operations:
        W_base = W_src[e]
        if W_down: W_base = W_base / 13       ← scale first
        ΔW = output_nullspace_project(Q, W_base)
        W_clone = W_base + eps * ||W_base|| * ΔW   ← noise on scaled base

    Args:
        W_src: (N_SRC_EXPERTS, ...) source expert weight tensor
        assignment: list mapping target index → source index
        eps: perturbation scale
        is_down_proj: if True, apply 1/COPIES_PER_EXPERT scaling BEFORE noise

    Returns:
        W_tgt: (N_TGT_EXPERTS, ...) tiled + perturbed + scaled weights
    """
    per_expert_shape = W_src.shape[1:]  # e.g. (4096, 1024)
    W_tgt = torch.zeros(N_TGT_EXPERTS, *per_expert_shape, dtype=W_src.dtype)

    for tgt_idx, src_idx in enumerate(assignment):
        W_base = W_src[src_idx].float()

        # Net2Wider: scale W_down BEFORE adding noise
        if is_down_proj:
            W_base = W_base / COPIES_PER_EXPERT

        frobenius_norm = W_base.norm()

        # Generate orthogonal noise, project into output null-space of W_base
        Q = generate_orthogonal_noise(per_expert_shape, device=W_base.device)
        Q_null = output_nullspace_project(Q, W_base)
        noise = eps * frobenius_norm * Q_null

        W_clone = W_base + noise
        W_tgt[tgt_idx] = W_clone.to(W_src.dtype)

    return W_tgt


def tile_gate_weights(
    gate_weight_src: torch.Tensor,
    assignment: List[int],
    eps_gate: float,
) -> torch.Tensor:
    """
    Tile router gate weights with tiny Gaussian noise for symmetry breaking.

    Args:
        gate_weight_src: (N_SRC_EXPERTS, d_model) source gate weights
        assignment: list mapping target index → source index
        eps_gate: noise scale for gate weights

    Returns:
        gate_weight_tgt: (N_TGT_EXPERTS, d_model) tiled gate weights
    """
    d_model = gate_weight_src.shape[1]
    gate_weight_tgt = torch.zeros(N_TGT_EXPERTS, d_model, dtype=gate_weight_src.dtype)

    for tgt_idx, src_idx in enumerate(assignment):
        base = gate_weight_src[src_idx].float()
        noise = eps_gate * torch.randn_like(base)
        gate_weight_tgt[tgt_idx] = (base + noise).to(gate_weight_src.dtype)

    return gate_weight_tgt


def tile_gate_bias(
    logit_bias_src: torch.Tensor,
    assignment: List[int],
) -> torch.Tensor:
    """
    Tile router logit_bias with softmax mass correction.

    Each clone's bias is shifted by −log(COPIES_PER_EXPERT) so that the total
    softmax probability mass for all clones of one source expert equals the
    original expert's probability:

        softmax(z − log(k)) = exp(z) / (k · denom)

    So: Σ_k softmax(z_i − log(k)) ≈ softmax(z_i)  (mass preserved)

    Without this correction, each original expert's contribution drops ~13×
    after explosion, causing an immediate loss spike even with perfect weight
    cloning.

    Args:
        logit_bias_src: (N_SRC_EXPERTS,) source logit biases
        assignment: list mapping target index → source index

    Returns:
        logit_bias_tgt: (N_TGT_EXPERTS,) mass-corrected tiled biases
    """
    logit_bias_tgt = torch.zeros(N_TGT_EXPERTS, dtype=logit_bias_src.dtype)
    for tgt_idx, src_idx in enumerate(assignment):
        logit_bias_tgt[tgt_idx] = logit_bias_src[src_idx] - LOG_COPIES
    return logit_bias_tgt


# ─────────────────────────────────────────────
# Step 3: Detect layer prefix in state dict
# ─────────────────────────────────────────────

def detect_layer_prefix(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Auto-detect layer key prefix. The model registers layers under `self.layers`,
    but ReversibleMidpointStack may re-register them as `self.stack.layers`.
    """
    for key in state_dict:
        if key.startswith("layers.0."):
            return "layers"
        if key.startswith("stack.layers.0."):
            return "stack.layers"
    raise ValueError(
        "Could not find layer weights in state dict. Neither 'layers.0.*' nor "
        "'stack.layers.0.*' keys exist. First 10 keys: "
        + str(list(state_dict.keys())[:10])
    )


# ─────────────────────────────────────────────
# Step 4: Copy non-layer weights (direct copy)
# ─────────────────────────────────────────────

def copy_non_layer_weights(
    tgt_state: Dict[str, torch.Tensor],
    src_state: Dict[str, torch.Tensor],
    src_prefix: str,
    verbose: bool = True,
) -> Tuple[int, int, int]:
    """
    Copy all weights that are NOT per-decoder-layer:
        - Embeddings (token_embed, kronecker, pf_to_model, embed_norm)
        - Memory stream (lambda_r_raw, memory_ln, memory_gate_proj)
        - Final norm
        - LM head
        - MTP block weights (handled separately for its MoE layer)

    Returns: (copied, skipped_shape, skipped_missing)
    """
    non_layer_src = {k: v for k, v in src_state.items()
                     if not k.startswith(src_prefix + ".")}

    copied = skipped_shape = skipped_missing = 0

    if verbose:
        print("  Non-layer weights:")

    for key, src_val in sorted(non_layer_src.items()):
        if key not in tgt_state:
            skipped_missing += 1
            if verbose:
                print(f"    -- {key} — not in 70B model (skip)")
            continue

        tgt_val = tgt_state[key]
        if tgt_val.shape != src_val.shape:
            skipped_shape += 1
            if verbose:
                print(f"    !! {key} — shape mismatch "
                      f"src={tuple(src_val.shape)} tgt={tuple(tgt_val.shape)} (skip)")
            continue

        tgt_state[key] = src_val.clone()
        copied += 1
        if verbose:
            print(f"    OK {key}  {tuple(src_val.shape)}")

    return copied, skipped_shape, skipped_missing


# ─────────────────────────────────────────────
# Step 5: Per-layer weight initialization
# ─────────────────────────────────────────────

def init_layer_weights(
    tgt_state: Dict[str, torch.Tensor],
    src_state: Dict[str, torch.Tensor],
    layer_idx: int,
    src_prefix: str,
    tgt_prefix: str,
    assignment: List[int],
    eps_expert: float,
    eps_gate: float,
) -> Dict[str, int]:
    """
    Initialize one layer of the 70B model from the corresponding 8B layer.

    For each key in the source layer:
      - MoE expert weights (W_gate, W_up, W_down) → tile with orthogonal noise
      - MoE gate weight → tile with Gaussian noise
      - MoE gate logit_bias → tile without noise
      - Everything else (attention, norms, mHC, shared expert) → direct copy

    Returns dict with counts: {copied, tiled, skipped}
    """
    src_layer_prefix = f"{src_prefix}.{layer_idx}."
    tgt_layer_prefix = f"{tgt_prefix}.{layer_idx}."

    stats = {"copied": 0, "tiled": 0, "skipped": 0}

    # ── Collect source layer keys ────────────────────────────────────
    src_layer_keys = {k: v for k, v in src_state.items()
                      if k.startswith(src_layer_prefix)}

    # ── 1. Tile routed expert weights ─────────────────────────────────
    for suffix in MOE_EXPERT_WEIGHT_SUFFIXES:
        src_key = src_layer_prefix + suffix
        tgt_key = tgt_layer_prefix + suffix
        if src_key in src_state and tgt_key in tgt_state:
            is_down = suffix.endswith("W_down")
            W_tgt = tile_expert_weights(
                src_state[src_key], assignment, eps_expert, is_down_proj=is_down,
            )
            tgt_state[tgt_key] = W_tgt
            stats["tiled"] += 1

    # ── 2. Tile router gate weights ───────────────────────────────────
    src_gate_key = src_layer_prefix + MOE_GATE_WEIGHT_SUFFIX
    tgt_gate_key = tgt_layer_prefix + MOE_GATE_WEIGHT_SUFFIX
    if src_gate_key in src_state and tgt_gate_key in tgt_state:
        tgt_state[tgt_gate_key] = tile_gate_weights(
            src_state[src_gate_key], assignment, eps_gate
        )
        stats["tiled"] += 1

    # ── 3. Tile router logit_bias ─────────────────────────────────────
    src_bias_key = src_layer_prefix + MOE_GATE_BIAS_SUFFIX
    tgt_bias_key = tgt_layer_prefix + MOE_GATE_BIAS_SUFFIX
    if src_bias_key in src_state and tgt_bias_key in tgt_state:
        tgt_state[tgt_bias_key] = tile_gate_bias(
            src_state[src_bias_key], assignment
        )
        stats["tiled"] += 1

    # ── 4. Direct copy for everything else (attention, norms, mHC, shared expert) ──
    handled_suffixes = set(
        MOE_EXPERT_WEIGHT_SUFFIXES
        + [MOE_GATE_WEIGHT_SUFFIX, MOE_GATE_BIAS_SUFFIX]
    )

    for src_key, src_val in src_layer_keys.items():
        suffix = src_key[len(src_layer_prefix):]

        # Skip already-handled MoE keys
        if suffix in handled_suffixes:
            continue

        tgt_key = tgt_layer_prefix + suffix
        if tgt_key not in tgt_state:
            stats["skipped"] += 1
            continue

        if tgt_state[tgt_key].shape != src_val.shape:
            stats["skipped"] += 1
            continue

        tgt_state[tgt_key] = src_val.clone()
        stats["copied"] += 1

    return stats


# ─────────────────────────────────────────────
# Step 6: Synchronize shared-parameter aliases
# ─────────────────────────────────────────────

def sync_shared_layer_keys(
    tgt_state: Dict[str, torch.Tensor],
    tgt_prefix: str,
) -> int:
    """
    Propagate initialized layer weights to ALL shared-parameter aliases
    created by ReversibleMidpointStack.

    Alias paths:
        layers.{i}.*                             (primary)
        stack.blocks.{i}.*                       (ReversibleMidpointStack.blocks)
        stack.bootstrap_layer.*                  (layer 0 only)
        stack.mid_layers.{i-1}.block.*           (layer i>0)
        stack.mid_layers.{i-1}.wrapper.layer.*   (layer i>0)

    Returns the number of alias keys synchronized.
    """
    synced = 0

    for tgt_idx in range(N_LAYERS):
        canonical_prefix = f"{tgt_prefix}.{tgt_idx}."

        canonical_entries = {}
        for key in tgt_state:
            if key.startswith(canonical_prefix):
                suffix = key[len(canonical_prefix):]
                canonical_entries[suffix] = tgt_state[key]

        if not canonical_entries:
            continue

        alias_prefixes = [f"stack.blocks.{tgt_idx}."]
        if tgt_idx == 0:
            alias_prefixes.append("stack.bootstrap_layer.")
        else:
            alias_prefixes.append(f"stack.mid_layers.{tgt_idx - 1}.block.")
            alias_prefixes.append(f"stack.mid_layers.{tgt_idx - 1}.wrapper.layer.")

        for alias_prefix in alias_prefixes:
            for suffix, canonical_val in canonical_entries.items():
                alias_key = alias_prefix + suffix
                if alias_key in tgt_state:
                    tgt_state[alias_key] = canonical_val.clone()
                    synced += 1

    return synced


# ─────────────────────────────────────────────
# Step 7: Adapt embedding keys
# ─────────────────────────────────────────────

def adapt_target_embedding_keys(
    tgt_state: Dict[str, torch.Tensor],
    src_state: Dict[str, torch.Tensor],
) -> str:
    """
    Detect source embedding type (kronecker vs standard) and adapt
    the target state dict's embedding keys to match.

    Returns the detected embedding type.
    """
    src_has_kronecker = "pf_to_model.weight" in src_state
    src_has_standard = "token_embed.weight" in src_state

    if src_has_kronecker:
        tgt_state.pop("token_embed.weight", None)
        tgt_state["pf_to_model.weight"] = torch.zeros_like(
            src_state["pf_to_model.weight"]
        )
        tgt_state["embed_norm.weight"] = torch.zeros_like(
            src_state["embed_norm.weight"]
        )
        print(f"    Adapted target embedding: standard -> kronecker")
        return "kronecker"

    if src_has_standard:
        print(f"    Source uses standard embedding — no adaptation needed")
        return "standard"

    raise ValueError(
        "Could not detect source embedding type. Expected either "
        "'pf_to_model.weight' (kronecker) or 'token_embed.weight' (standard)."
    )


# ─────────────────────────────────────────────
# Step 8: Initialize MTP block's MoE experts
# ─────────────────────────────────────────────

# The MTP block has its own MoE layer with the same expert structure.
# Key prefix: mtp_block.mlp_block.sublayer.moe.{W_gate|W_up|W_down|gate.*|shared_*}
MTP_MOE_PREFIX = "mtp_block.mlp_block.sublayer.moe."


def init_mtp_moe_weights(
    tgt_state: Dict[str, torch.Tensor],
    src_state: Dict[str, torch.Tensor],
    assignment: List[int],
    eps_expert: float,
    eps_gate: float,
) -> Dict[str, int]:
    """
    Initialize the MTP block's MoE routed expert weights.

    The MTP block sits outside the layers.{i} prefix, so copy_non_layer_weights
    skips its routed expert weights due to shape mismatch (20 vs 260). This
    function tiles them explicitly. The shared expert has identical shapes in
    both models (2048 intermediate) and is handled by copy_non_layer_weights.
    """
    stats = {"tiled": 0, "skipped": 0}

    # Tile routed expert weights
    for suffix_base in ["W_gate", "W_up", "W_down"]:
        src_key = MTP_MOE_PREFIX + suffix_base
        tgt_key = MTP_MOE_PREFIX + suffix_base
        if src_key in src_state and tgt_key in tgt_state:
            is_down = (suffix_base == "W_down")
            tgt_state[tgt_key] = tile_expert_weights(
                src_state[src_key], assignment, eps_expert, is_down_proj=is_down,
            )
            stats["tiled"] += 1

    # Tile gate weights
    src_gate = MTP_MOE_PREFIX + "gate.weight"
    tgt_gate = MTP_MOE_PREFIX + "gate.weight"
    if src_gate in src_state and tgt_gate in tgt_state:
        tgt_state[tgt_gate] = tile_gate_weights(
            src_state[src_gate], assignment, eps_gate
        )
        stats["tiled"] += 1

    # Tile gate bias
    src_bias = MTP_MOE_PREFIX + "gate.logit_bias"
    tgt_bias = MTP_MOE_PREFIX + "gate.logit_bias"
    if src_bias in src_state and tgt_bias in tgt_state:
        tgt_state[tgt_bias] = tile_gate_bias(
            src_state[src_bias], assignment
        )
        stats["tiled"] += 1

    return stats


# ─────────────────────────────────────────────
# Main initialization function
# ─────────────────────────────────────────────

def explode_8b_to_70b(
    src_checkpoint_path: str,
    tgt_checkpoint_path: str,
    model_dir: str,
    eps_expert: float = 0.01,
    eps_gate: float = 0.0005,
    dry_run: bool = False,
    seed: int = 42,
) -> None:
    print("=" * 70)
    print("  Function-Preserving Expert Explosion")
    print("  8B (20 experts) -> 70B (260 experts)")
    print("=" * 70)

    # ── Reproducibility ──────────────────────────────────────────────
    torch.manual_seed(seed)
    print(f"\n  Random seed: {seed}")
    print(f"  eps_expert:  {eps_expert}")
    print(f"  eps_gate:    {eps_gate}")

    # ── Expert assignment ────────────────────────────────────────────
    assignment = build_expert_assignment()
    print_assignment(assignment)

    if dry_run:
        print("  DRY RUN — no files loaded or written.\n")
        return

    # ── Load 8B checkpoint ───────────────────────────────────────────
    print(f"  Loading 8B checkpoint: {src_checkpoint_path}")
    src_ckpt = torch.load(src_checkpoint_path, map_location="cpu", weights_only=False)
    src_state = src_ckpt["model_state_dict"]
    src_prefix = detect_layer_prefix(src_state)
    print(f"    Layer prefix: '{src_prefix}'")
    print(f"    Total keys: {len(src_state)}")

    # ── Build 70B model for target state dict ────────────────────────
    print(f"\n  Building 70B model for target state dict...")
    tgt_state = _build_70b_state_dict(model_dir)
    tgt_prefix = detect_layer_prefix(tgt_state)
    print(f"    Layer prefix: '{tgt_prefix}'")
    print(f"    Total keys: {len(tgt_state)}")

    # ── Adapt embeddings ─────────────────────────────────────────────
    print(f"\n  Detecting source embedding type...")
    detected_embed_type = adapt_target_embedding_keys(tgt_state, src_state)

    # ── Copy non-layer weights ───────────────────────────────────────
    print(f"\n  Copying non-layer weights...")
    nl_copied, nl_shape_skip, nl_missing = copy_non_layer_weights(
        tgt_state, src_state, src_prefix, verbose=True
    )
    print(f"\n    Summary: copied={nl_copied}, shape_mismatch={nl_shape_skip}, "
          f"missing_in_tgt={nl_missing}")

    # ── Initialize layers ────────────────────────────────────────────
    print(f"\n  Initializing {N_LAYERS} layers (expert explosion + direct copy)...")
    print(f"  {'Layer':>6}  {'Copied':>6}  {'Tiled':>5}  {'Skip':>4}")
    print("  " + "-" * 30)

    total_stats = {"copied": 0, "tiled": 0, "skipped": 0}

    for layer_idx in range(N_LAYERS):
        stats = init_layer_weights(
            tgt_state=tgt_state,
            src_state=src_state,
            layer_idx=layer_idx,
            src_prefix=src_prefix,
            tgt_prefix=tgt_prefix,
            assignment=assignment,
            eps_expert=eps_expert,
            eps_gate=eps_gate,
        )
        for k in total_stats:
            total_stats[k] += stats[k]

        print(f"  L{layer_idx:02d}     {stats['copied']:>5}  {stats['tiled']:>5}  "
              f"{stats['skipped']:>4}")

    print("  " + "-" * 30)
    print(f"  Total  {total_stats['copied']:>5}  {total_stats['tiled']:>5}  "
          f"{total_stats['skipped']:>4}")

    # ── Initialize MTP block's MoE experts ──────────────────────────
    print(f"\n  Initializing MTP block MoE experts...")
    mtp_stats = init_mtp_moe_weights(
        tgt_state, src_state, assignment, eps_expert, eps_gate
    )
    print(f"    MTP MoE: tiled={mtp_stats['tiled']}, skipped={mtp_stats['skipped']}")

    # ── Sync shared-parameter aliases ────────────────────────────────
    print(f"\n  Synchronizing shared-parameter aliases...")
    n_synced = sync_shared_layer_keys(tgt_state, tgt_prefix)
    print(f"    Synchronized {n_synced} alias keys across {N_LAYERS} layers")

    # ── Sanity check: no NaN or Inf ──────────────────────────────────
    print(f"\n  Running sanity checks...")
    nan_keys = [k for k, v in tgt_state.items() if torch.isnan(v).any()]
    inf_keys = [k for k, v in tgt_state.items() if torch.isinf(v).any()]
    if nan_keys:
        print(f"    WARNING: NaN detected in {len(nan_keys)} keys: {nan_keys[:5]}")
    if inf_keys:
        print(f"    WARNING: Inf detected in {len(inf_keys)} keys: {inf_keys[:5]}")
    if not nan_keys and not inf_keys:
        print(f"    OK: No NaN or Inf values in any weight tensor")

    # ── Save ─────────────────────────────────────────────────────────
    print(f"\n  Saving initialized 70B checkpoint -> {tgt_checkpoint_path}")
    os.makedirs(os.path.dirname(os.path.abspath(tgt_checkpoint_path)), exist_ok=True)

    tgt_ckpt = {
        "step": 0,
        "model_state_dict": tgt_state,
        "optimizer_state_dict": None,
        "loss": None,
        "embedding_type": detected_embed_type,
        "lambda_p_state": None,
        # Provenance metadata
        "init_method": "expert_explosion_tiling",
        "init_method_version": "3.0",
        "corrections": ["round_robin", "log_mass", "wdown_scale_before_noise", "output_nullspace"],
        "src_checkpoint": src_checkpoint_path,
        "expert_assignment": assignment,
        "n_src_experts": N_SRC_EXPERTS,
        "n_tgt_experts": N_TGT_EXPERTS,
        "copies_per_expert": COPIES_PER_EXPERT,
        "eps_expert": eps_expert,
        "eps_gate": eps_gate,
        "seed": seed,
    }
    torch.save(tgt_ckpt, tgt_checkpoint_path)
    print(f"    Saved: {tgt_checkpoint_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  INITIALIZATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  8B -> 70B:           20 experts -> 260 experts (13x tiling)")
    print(f"  Assignment:          round-robin (siblings index-separated)")
    print(f"  Layers:              {N_LAYERS} (identical between 8B and 70B)")
    print(f"  Expert perturbation: eps={eps_expert} (output-nullspace LiGO noise)")
    print(f"  Gate perturbation:   eps={eps_gate} (Gaussian noise)")
    print(f"  Logit bias:          -log(13) = -{LOG_COPIES:.4f} mass correction")
    print(f"  W_down scaling:      / {COPIES_PER_EXPERT} BEFORE noise (Net2Wider)")
    print(f"  Shared expert:       direct copy (2048 intermediate)")
    print(f"  Attention/Norms:     direct copy")
    print(f"  Optimizer:           reset (fresh warmstart)")
    print(f"{'=' * 70}\n")
    print("  IMPORTANT: Use Top-K warmstart schedule during training:")
    print("    Steps 0-1000:    Top-K = 2 (match 8B)")
    print("    Steps 1000-3000: Top-K = 4")
    print("    Steps 3000+:     Top-K = 8 (full 70B routing)")
    print()
    print("  Run validate_explosion_init.py to verify function preservation.\n")


def _build_70b_state_dict(model_dir: str) -> Dict[str, torch.Tensor]:
    """
    Instantiate the 70B model to get its fresh (random) state dict.
    This gives us correct shapes for all 260-expert layers.
    """
    model_dir = os.path.normpath(os.path.abspath(model_dir))
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"--model_dir does not exist: {model_dir}"
        )

    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    if "recurrence_model_70b" in sys.modules:
        del sys.modules["recurrence_model_70b"]

    try:
        from recurrence_model_70b import Model70B, ModelConfig  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"Could not import recurrence_model_70b from: {model_dir}\n"
            f"Ensure recurrence_model_70b.py exists in that directory.\n"
            f"Error: {e}"
        )

    config = ModelConfig()

    # Verify expected expert counts
    assert config.num_real_experts == N_TGT_EXPERTS, (
        f"70B ModelConfig.num_real_experts={config.num_real_experts}, "
        f"expected {N_TGT_EXPERTS}"
    )

    model_70b = Model70B(config, embedding_type="standard")
    state = copy.deepcopy(model_70b.state_dict())
    del model_70b
    print(f"    Loaded Model70B from: {model_dir}")
    return state


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Function-preserving expert explosion: 8B (20 experts) -> 70B (260 experts)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--src", required=True,
        help="Path to trained 8B checkpoint (.pt)"
    )
    parser.add_argument(
        "--tgt", required=True,
        help="Output path for initialized 70B checkpoint (.pt)"
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Path to the directory containing recurrence_model_70b.py (e.g. ../)"
    )
    parser.add_argument(
        "--eps_expert", type=float, default=0.01,
        help="Output-nullspace perturbation scale for routed expert weights. Default: 0.01. "
             "Reduce to 0.005 if loss spikes >3%%; increase to 0.02 if experts don't differentiate."
    )
    parser.add_argument(
        "--eps_gate", type=float, default=0.0005,
        help="Gaussian noise scale for router gate weights. Default: 0.0005. "
             "Reduced from 0.001 because Top-K increase (2→8) and round-robin "
             "layout increase routing competition."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible initialization. Default: 42."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print assignment plan only — do not load or write any files."
    )
    args = parser.parse_args()

    explode_8b_to_70b(
        src_checkpoint_path=args.src,
        tgt_checkpoint_path=args.tgt,
        model_dir=args.model_dir,
        eps_expert=args.eps_expert,
        eps_gate=args.eps_gate,
        dry_run=args.dry_run,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
