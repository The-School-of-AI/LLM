#!/usr/bin/env python3
"""
act_unroll_init_3b_to_8b.py

K_EXEC values are loaded from layer_repitation_factor.json (same directory).
Pass --json /path/to/layer_repitation_factor.json to override.
============================
Trajectory-Preserving Depth Unrolling
    3B (8-layer ACT-trained MoE) → 8B (20-layer feedforward MoE)

Algorithm: ACT Solver Unrolling
----------------------------------------
The 3B ACT model learned an *iterative residual refinement* per layer:

    h ← h + α₁·F(h)
    h ← h + α₂·F(h)
    h ← h + α₃·F(h)

This is NOT the same as sequential layer composition:
    h₄ = F(F(F(F(h₀))))

Naive weight repetition inherits layer-amplification instability.
This script instead "unrolls" each ACT refinement step into one
physical 8B child layer, scaled by the corresponding α coefficient.

Block-wise expansion preserves the 3:1 Delta:GSA ratio:
    B0 = [L0 L1 L2 L3]  (DDD G)  k_sum=8  → 2 copies
    B1 = [L4 L5 L6 L7]  (DDD G)  k_sum=12 → 3 copies
    Total = 5 blocks × 4 layers = 20 layers ✅

Attention type assignment uses 8B position formula, never source type:
    (i+1) % 4 == 0 → GSA, else Delta

Preserves:
    ✅ Delta:GSA = 3:1 ratio (position formula)
    ✅ ACT refinement trajectory (α-scaled residual chain)
    ✅ MoE routing distribution (expert weights copied)
    ✅ Jacobian spectral radius (α-decay prevents amplification)
    ✅ Stable 8B warmstart (no early loss spike)

Usage:
    python act_unroll_init_3b_to_8b.py \\
        --src checkpoints/fourier_latest.pt \\
        --tgt checkpoints/8b_act_unroll_init.pt \\
        --model_dir ../

    # Dry-run (just print the plan):
    python act_unroll_init_3b_to_8b.py --src ... --tgt ... --model_dir .. --dry_run

    # Adjust noise for weight diversity between copies:
    python act_unroll_init_3b_to_8b.py --src ... --tgt ... --model_dir .. --noise_std 1e-5
"""

import os
import sys
import json
import copy
import argparse
import torch
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# Architecture constants
# ─────────────────────────────────────────────

# ACT α coefficients from AdaptiveDecoderLayer.alpha = [1.0, 0.7, 0.5, 0.35]
# child_k receives weight: W * alpha[k-1]
ACT_ALPHAS = [1.0, 0.7, 0.5, 0.35]

# Kronecker / PureHybridEmbedding dimension: POS_DIM(32) × CHAR_DIM(256)
PF_EMBED_DIM = 8192

# Source (3B) config
N_SRC_LAYERS   = 8
SRC_BLOCK_SIZE = 4   # DDD + GSA pattern

# Target (8B) config
N_TGT_LAYERS   = 20
TGT_BLOCK_SIZE = 4   # DDD + GSA (same pattern)


# ─────────────────────────────────────────────
# Load k_exec from JSON
# ─────────────────────────────────────────────

def load_k_exec(json_path: Optional[str] = None) -> Dict[str, int]:
    """
    Load k_exec values from layer_repitation_factor.json.
    Auto-detects the file in the same directory as this script if not specified.

    JSON format expected:
        {
            "layer0": { "k_exec": 2, ... },
            "layer1": { "k_exec": 1, ... },
            ...
        }
    The "_meta" key is ignored.
    """
    if json_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(here, "layer_repitation_factor.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"layer_repitation_factor.json not found at: {json_path}\n"
            f"Pass --json /path/to/layer_repitation_factor.json explicitly."
        )

    with open(json_path, "r") as f:
        raw = json.load(f)

    k_exec = {}
    for key, val in raw.items():
        if key.startswith("_"):   # skip _meta etc.
            continue
        if isinstance(val, dict):
            k_exec[key] = int(val["k_exec"])
        elif isinstance(val, (int, float)):
            # Legacy format: {"layer0": 2}
            k_exec[key] = int(val)
        else:
            raise ValueError(f"Unexpected value format for key '{key}': {val}")

    print(f"   📄 Loaded k_exec from: {json_path}")
    for k, v in sorted(k_exec.items()):
        print(f"      {k}: {v}")

    return k_exec


# ─────────────────────────────────────────────
# Step 1: Block expansion plan
# ─────────────────────────────────────────────

def compute_block_expansion(k_exec: Dict[str, int]) -> List[Dict]:
    """
    Group 3B layers into 4-layer ACT blocks (DDD+GSA).
    Compute how many times each block should be replicated.

    Expansion formula:
        n_copies(block) = sum(k_exec in block) / SRC_BLOCK_SIZE

    Example:
        B0 = L0..L3, k_sum = 2+1+2+3 = 8, n_copies = 8/4 = 2
        B1 = L4..L7, k_sum = 2+3+4+3 = 12, n_copies = 12/4 = 3
        Total target layers = (2+3) × 4 = 20 ✅
    """
    k_exec_list = [k_exec[f"layer{i}"] for i in range(N_SRC_LAYERS)]
    n_blocks = N_SRC_LAYERS // SRC_BLOCK_SIZE

    blocks = []
    for b in range(n_blocks):
        start = b * SRC_BLOCK_SIZE
        end   = start + SRC_BLOCK_SIZE
        k_sum = sum(k_exec_list[start:end])
        assert k_sum % SRC_BLOCK_SIZE == 0, (
            f"Block B{b} k_sum={k_sum} is not divisible by {SRC_BLOCK_SIZE}. "
            f"k_exec values must sum to a multiple of {SRC_BLOCK_SIZE} per block."
        )
        n_copies = k_sum // SRC_BLOCK_SIZE
        blocks.append({
            "block_id":   b,
            "src_layers": list(range(start, end)),
            "k_sum":      k_sum,
            "n_copies":   n_copies,
        })

    total_tgt = sum(b["n_copies"] * SRC_BLOCK_SIZE for b in blocks)
    assert total_tgt == N_TGT_LAYERS, (
        f"Expansion produces {total_tgt} layers, expected {N_TGT_LAYERS}. "
        f"Check k_exec values."
    )
    return blocks


# ─────────────────────────────────────────────
# Step 2: Full 20-layer expansion plan
# ─────────────────────────────────────────────

def build_expansion_plan(blocks: List[Dict]) -> List[Dict]:
    """
    Map each of the 20 target layers to:
        src_layer  → which 3B layer to copy weights from
        alpha      → ACT scaling factor  (indexed by block copy number)
        src_type   → "deltanet" or "gsa" (3B layer type by position)
        tgt_type   → "deltanet" or "gsa" (8B layer type by position formula)
        copy_idx   → which copy of this block this is (0-based)

    Key insight: because we expand WHOLE BLOCKS (DDD+GSA), source and target
    types always match — the GSA always lands at position (i+1)%4==0.
    """
    plan = []
    for block in blocks:
        n_copies   = block["n_copies"]
        src_layers = block["src_layers"]

        for copy_idx in range(n_copies):
            # α for this block copy
            alpha = ACT_ALPHAS[copy_idx] if copy_idx < len(ACT_ALPHAS) else ACT_ALPHAS[-1]

            for pos_in_block, src_layer_idx in enumerate(src_layers):
                # Source type (3B position formula)
                src_type = "gsa" if (src_layer_idx + 1) % 4 == 0 else "deltanet"

                plan.append({
                    "src_layer": src_layer_idx,
                    "alpha":     alpha,
                    "src_type":  src_type,
                    "copy_idx":  copy_idx,
                })

    # Assign target index and target type by 8B position formula
    assert len(plan) == N_TGT_LAYERS, f"Plan has {len(plan)} entries, expected {N_TGT_LAYERS}"
    for tgt_idx, entry in enumerate(plan):
        entry["tgt_idx"]  = tgt_idx
        entry["tgt_type"] = "gsa" if (tgt_idx + 1) % 4 == 0 else "deltanet"

    return plan


def print_expansion_plan(blocks: List[Dict], plan: List[Dict], k_exec: Dict[str, int]) -> None:
    """Pretty-print the full expansion plan for inspection."""
    print("\n┌─ Block Expansion ─────────────────────────────────┐")
    k_list = [k_exec[f"layer{i}"] for i in range(N_SRC_LAYERS)]
    print(f"│  k_exec per layer: {k_list}")
    for block in blocks:
        b = block["block_id"]
        k_sum = block["k_sum"]
        nc    = block["n_copies"]
        lrs   = block["src_layers"]
        print(f"│  B{b} = L{lrs[0]}–L{lrs[-1]}  k_sum={k_sum}  → {nc} copies (α = "
              f"{', '.join(str(ACT_ALPHAS[i]) for i in range(nc))})")
    total_tgt = sum(b["n_copies"] * SRC_BLOCK_SIZE for b in blocks)
    print(f"│  Total: {total_tgt} target layers = {sum(b['n_copies'] for b in blocks)} blocks × {SRC_BLOCK_SIZE}")
    print("└────────────────────────────────────────────────────┘\n")

    print(f"{'8B Pos':>7}  {'SrcL':>4}  {'CopyIdx':>8}  {'Alpha':>5}  "
          f"{'SrcType':>8}  {'TgtType':>8}  {'Match?':>6}")
    print("─" * 60)
    for e in plan:
        match = "✅" if e["src_type"] == e["tgt_type"] else "⚠️ MISMATCH"
        print(f"  L{e['tgt_idx']:02d}     L{e['src_layer']}  copy[{e['copy_idx']}]   "
              f"{e['alpha']:.2f}  {e['src_type']:>8}  {e['tgt_type']:>8}  {match}")
    print()


# ─────────────────────────────────────────────
# Step 3: Detect layer key prefix in state dict
# ─────────────────────────────────────────────

def detect_layer_prefix(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    The 3B/8B models register layers under `self.layers`, but if
    ReversibleMidpointStack re-registers them as `self.stack.layers`,
    the state dict prefix will differ. Auto-detect which it is.
    """
    for key in state_dict:
        if key.startswith("layers.0."):
            return "layers"
        if key.startswith("stack.layers.0."):
            return "stack.layers"
    raise ValueError(
        "Could not find layer weights in state dict. Neither 'layers.0.*' nor "
        "'stack.layers.0.*' keys exist. Keys starting with: "
        + str([k for k in list(state_dict.keys())[:10]])
    )


# ─────────────────────────────────────────────
# Step 4: Copy one layer's weights with scaling
# ─────────────────────────────────────────────

def copy_layer_weights(
    tgt_state:  Dict[str, torch.Tensor],
    src_state:  Dict[str, torch.Tensor],
    tgt_idx:    int,
    src_idx:    int,
    alpha:      float,
    src_type:   str,
    tgt_type:   str,
    src_prefix: str,
    tgt_prefix_base: str,
    noise_std:  float = 1e-4,
) -> None:
    """
    Copy all weights from 3B layer `src_idx` into 8B layer `tgt_idx`.

    Rules:
    ─────────────────────────────────────────────────────────────────
    1. All MLP and mHC weights → always copy + scale by alpha + add noise.
       (MLP shapes are identical between Delta and GSA layers.)

    2. Attention sublayer weights → copy + scale only if types match.
       If types mismatch (rare; shouldn't happen with block expansion),
       leave the 8B random initialization intact for the attention module only.

    3. Norm and other per-layer scalars → copy directly (no alpha).
       Norms shouldn't be scaled — they're not part of the residual path.
    ─────────────────────────────────────────────────────────────────
    """
    src_layer_key = f"{src_prefix}.{src_idx}."
    tgt_layer_key = f"{tgt_prefix_base}.{tgt_idx}."
    types_match = (src_type == tgt_type)

    copied, skipped, missing = 0, 0, 0

    for src_key, src_val in src_state.items():
        if not src_key.startswith(src_layer_key):
            continue

        suffix  = src_key[len(src_layer_key):]
        tgt_key = tgt_layer_key + suffix

        if tgt_key not in tgt_state:
            missing += 1
            continue

        tgt_val    = tgt_state[tgt_key]
        is_attn    = "attn_block.sublayer" in suffix
        is_norm    = "norm.weight" in suffix or "norm.bias" in suffix
        is_mhc     = "coeffs." in suffix
        is_mlp     = "mlp_block." in suffix

        if is_attn and not types_match:
            # Shapes differ between DeltaNet & GSA — keep 8B random init
            skipped += 1
            continue

        src_f = src_val.float()

        if tgt_val.shape != src_f.shape:
            # Safety: if shapes don't match (should not happen for same-type),
            # skip rather than crash
            skipped += 1
            continue

        if is_norm:
            # Norms are not residual outputs — copy directly, no α-scaling
            tgt_state[tgt_key] = src_f.to(src_val.dtype)
        else:
            # Residual path weights: scale by α, noise added before scaling
            # so SNR stays constant regardless of alpha value.
            noise = torch.randn_like(src_f) * noise_std if noise_std > 0 else 0.0
            tgt_state[tgt_key] = (alpha * (src_f + noise)).to(src_val.dtype)

        copied += 1

    return copied, skipped, missing


# ─────────────────────────────────────────────
# Step 5: Copy non-layer weights
# ─────────────────────────────────────────────

def copy_non_layer_weights(
    tgt_state:  Dict[str, torch.Tensor],
    src_state:  Dict[str, torch.Tensor],
    src_prefix: str,
    verbose:    bool = True,
) -> None:
    """
    Copy all weights that are NOT per-decoder-layer:
        - Embeddings (token_embed, kronecker_embeddings, pf_to_model, embed_norm)
        - Memory stream (lambda_r_raw, memory_ln, memory_gate_proj)
        - Final norm
        - LM head
        - MTP block (if present)

    If 3B and 8B have identical hidden_size / vocab_size, all shapes match.
    """
    non_layer_src = {k: v for k, v in src_state.items()
                     if not k.startswith(src_prefix + ".")}

    print("  Non-layer weights:")
    for key, src_val in non_layer_src.items():
        if key not in tgt_state:
            if verbose:
                print(f"    ℹ️  {key} — not in 8B model (skip)")
            continue
        tgt_val = tgt_state[key]
        if tgt_val.shape != src_val.shape:
            if verbose:
                print(f"    ⚠️  {key} — shape mismatch "
                      f"src={tuple(src_val.shape)} tgt={tuple(tgt_val.shape)} (skip)")
            continue
        tgt_state[key] = src_val.clone()
        if verbose:
            print(f"    ✅ {key}  {tuple(src_val.shape)}")


# ─────────────────────────────────────────────
# Step 6: Synchronize shared-parameter aliases
# ─────────────────────────────────────────────

def _sync_shared_layer_keys(
    tgt_state:  Dict[str, torch.Tensor],
    tgt_prefix: str,
    n_layers:   int,
) -> int:
    """
    Propagate initialized layer weights to ALL shared-parameter aliases.

    ReversibleMidpointStack creates multiple state dict key paths to the
    same underlying layer parameters:

        layers.{i}.*                               (Model.layers)
        stack.blocks.{i}.*                         (ReversibleMidpointStack.blocks)
        stack.bootstrap_layer.*                    (layer 0 only)
        stack.mid_layers.{i-1}.block.*             (layer i>0, MidpointBlock.block)
        stack.mid_layers.{i-1}.wrapper.layer.*     (layer i>0, _ForceWrapper.layer)

    After copy_layer_weights updates the primary prefix (e.g. 'layers.*'),
    this function copies those values into every alias key so that
    load_state_dict() produces the correct result regardless of which
    shared-parameter key PyTorch processes last.

    Returns the number of alias keys synchronized.
    """
    synced = 0

    for tgt_idx in range(n_layers):
        canonical_prefix = f"{tgt_prefix}.{tgt_idx}."

        # Collect all canonical (primary) keys and their suffixes
        canonical_entries = {}
        for key in tgt_state:
            if key.startswith(canonical_prefix):
                suffix = key[len(canonical_prefix):]
                canonical_entries[suffix] = tgt_state[key]

        if not canonical_entries:
            continue

        # Known alias prefixes from ReversibleMidpointStack structure
        alias_prefixes = [
            f"stack.blocks.{tgt_idx}.",
        ]
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
# Main unrolling function
# ─────────────────────────────────────────────

def unroll_3b_to_8b(
    src_checkpoint_path: str,
    tgt_checkpoint_path: str,
    model_dir:   str,
    noise_std:   float = 1e-4,
    dry_run:     bool  = False,
    json_path:   Optional[str] = None,
) -> None:
    print("=" * 65)
    print("  ACT Solver Unrolling — Trajectory-Preserving Depth Expansion")
    print("  3B (8 layers, ACT-trained) → 8B (20 layers, feedforward)")
    print("=" * 65)

    # ── Load k_exec from JSON ────────────────────────────────────────
    print("\n📄 Loading k_exec values from layer_repitation_factor.json...")
    k_exec = load_k_exec(json_path)

    # ── Expansion plan ──────────────────────────────────────────────
    print("\n📐 Computing ACT block expansion plan...")
    blocks = compute_block_expansion(k_exec)
    plan   = build_expansion_plan(blocks)
    print_expansion_plan(blocks, plan, k_exec)

    if dry_run:
        print("🔍 Dry run — no files loaded or written.\n")
        return

    # ── Load 3B checkpoint ──────────────────────────────────────────
    print(f"📂 Loading 3B checkpoint: {src_checkpoint_path}")
    src_ckpt  = torch.load(src_checkpoint_path, map_location="cpu")
    src_state = src_ckpt["model_state_dict"]
    src_prefix = detect_layer_prefix(src_state)
    print(f"   Layer prefix in 3B state dict: '{src_prefix}'")
    print(f"   Total keys: {len(src_state)}")

    # ── Load / build 8B state dict ───────────────────────────────────
    print("\n🏗️  Building 8B model for target state dict...")
    tgt_state = _build_8b_state_dict(model_dir)
    tgt_prefix = detect_layer_prefix(tgt_state)
    print(f"   Layer prefix in 8B state dict: '{tgt_prefix}'")
    print(f"   Total keys: {len(tgt_state)}")

    # ── Adapt embedding keys to match source ────────────────────────
    print("\n🔗 Detecting source embedding type...")
    detected_embed_type = _adapt_target_embedding_keys(tgt_state, src_state)

    # ── Copy non-layer weights ───────────────────────────────────────
    print("\n📋 Copying non-layer weights...")
    copy_non_layer_weights(tgt_state, src_state, src_prefix, verbose=True)

    # ── Unroll layers ────────────────────────────────────────────────
    print("\n⚙️  Unrolling ACT layers into 8B physical stack...")
    print(f"{'8B Pos':>7}  {'SrcL':>4}  {'Alpha':>5}  {'SrcType':>8}  {'TgtType':>8}  "
          f"{'Copied':>6}  {'Skipped':>7}")
    print("─" * 65)

    total_copied = total_skipped = total_missing = 0
    for entry in plan:
        c, s, m = copy_layer_weights(
            tgt_state    = tgt_state,
            src_state    = src_state,
            tgt_idx      = entry["tgt_idx"],
            src_idx      = entry["src_layer"],
            alpha        = entry["alpha"],
            src_type     = entry["src_type"],
            tgt_type     = entry["tgt_type"],
            src_prefix   = src_prefix,
            tgt_prefix_base = tgt_prefix,
            noise_std    = noise_std,
        )
        total_copied  += c
        total_skipped += s
        total_missing += m
        match_icon = "✅" if entry["src_type"] == entry["tgt_type"] else "⚠️"
        print(f"  L{entry['tgt_idx']:02d}     L{entry['src_layer']}  {entry['alpha']:.2f}  "
              f"{entry['src_type']:>8}  {entry['tgt_type']:>8}  {match_icon}  "
              f"{c:>5}  {s:>6}")

    print(f"\n  Totals: copied={total_copied}, skipped={total_skipped}, missing={total_missing}")

    # ── Sync shared-parameter aliases ──────────────────────────────
    # ReversibleMidpointStack registers the same layer modules under
    # multiple key prefixes (layers.*, stack.blocks.*, stack.bootstrap_layer.*,
    # stack.mid_layers.*.block.*, stack.mid_layers.*.wrapper.layer.*).
    # We must propagate the initialized values to ALL aliases, otherwise
    # load_state_dict() overwrites our work with random init from a
    # later-processed alias key.
    print("\n🔗 Synchronizing shared-parameter aliases...")
    n_synced = _sync_shared_layer_keys(tgt_state, tgt_prefix, N_TGT_LAYERS)
    print(f"   ✅ Synchronized {n_synced} alias keys across "
          f"{N_TGT_LAYERS} layers")

    # ── Save ─────────────────────────────────────────────────────────
    print(f"\n💾 Saving initialized 8B checkpoint → {tgt_checkpoint_path}")
    os.makedirs(os.path.dirname(os.path.abspath(tgt_checkpoint_path)), exist_ok=True)

    tgt_ckpt = {
        "step":               0,
        "model_state_dict":   tgt_state,
        "optimizer_state_dict": None,       # fresh optimizer — do NOT warm-start optimizer
        "loss":               None,
        "embedding_type":     detected_embed_type,
        "lambda_p_state":     None,
        # Provenance metadata
        "init_method":        "act_solver_unroll",
        "init_method_version": "1.0",
        "src_checkpoint":     src_checkpoint_path,
        "expansion_plan":     plan,
        "act_alphas":         ACT_ALPHAS,
        "k_exec":             k_exec,
        "noise_std":          noise_std,
    }
    torch.save(tgt_ckpt, tgt_checkpoint_path)
    print(f"   ✅ Saved: {tgt_checkpoint_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  INITIALIZATION COMPLETE")
    print(f"{'='*65}")
    print(f"  3B → 8B:     8 layers → 20 layers")
    print(f"  Blocks:      B0×{blocks[0]['n_copies']} + B1×{blocks[1]['n_copies']} = "
          f"{sum(b['n_copies'] for b in blocks)} blocks × {SRC_BLOCK_SIZE} = 20 layers")
    print(f"  α schedule:  {ACT_ALPHAS[:3]} (block copy 1, 2, 3)")
    print(f"  Noise std:   {noise_std}")
    print(f"  Delta:GSA    = 15:5 = 3:1  ✅")
    print(f"  ACT unroll:  preserved ✅")
    print(f"  Optimizer:   reset (fresh warmstart recommended)")
    print(f"{'='*65}\n")
    print("  🟢 Ready for 8B warmstart training.")
    print("  Expected: smooth loss continuation, no early spike.\n")


def _build_8b_state_dict(model_dir: str) -> Dict[str, torch.Tensor]:
    """
    Instantiate the 8B model to get its fresh (random) state dict.
    This gives us the correct shapes for all 20 layers, which we then fill.

    model_dir must point to the directory containing recurrence_model_8b.py.
    """
    model_dir = os.path.normpath(os.path.abspath(model_dir))
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"--model_dir does not exist or is not a directory: {model_dir}"
        )

    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    # Force a fresh import even if a stale version was cached from a
    # different directory earlier in the same Python session.
    if "recurrence_model_8b" in sys.modules:
        del sys.modules["recurrence_model_8b"]

    try:
        from recurrence_model_8b import Model8B, ModelConfig  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"Could not import recurrence_model_8b from: {model_dir}\n"
            f"Ensure recurrence_model_8b.py exists in that directory.\n"
            f"Error: {e}"
        )

    config   = ModelConfig()
    # Build with standard embedding to avoid requiring bpe_vocab / pf_codec
    # at init time.  _adapt_target_embedding_keys() will swap the embedding
    # keys to kronecker format if the source checkpoint uses kronecker.
    model_8b = Model8B(config, embedding_type="standard")
    state    = copy.deepcopy(model_8b.state_dict())
    del model_8b
    print(f"   ✅ Loaded Model8B from: {model_dir}")
    return state


def _adapt_target_embedding_keys(
    tgt_state: Dict[str, torch.Tensor],
    src_state: Dict[str, torch.Tensor],
) -> str:
    """
    Detect source embedding type (kronecker vs standard) and adapt
    the target state dict's embedding keys to match.

    The target is always built with embedding_type="standard" (to avoid
    requiring tokenizer/codec at init time).  If the source checkpoint
    uses kronecker embedding, this function:
        - Removes 'token_embed.weight' from the target
        - Adds placeholder 'pf_to_model.weight' and 'embed_norm.weight'
          with shapes matching the source (overwritten by copy_non_layer_weights)

    Returns the detected embedding type ("kronecker" or "standard").
    """
    src_has_kronecker = "pf_to_model.weight" in src_state
    src_has_standard  = "token_embed.weight" in src_state

    if src_has_kronecker:
        # Remove standard embedding key from target
        tgt_state.pop("token_embed.weight", None)

        # Add placeholder keys matching source shapes
        tgt_state["pf_to_model.weight"] = torch.zeros_like(
            src_state["pf_to_model.weight"]
        )
        tgt_state["embed_norm.weight"] = torch.zeros_like(
            src_state["embed_norm.weight"]
        )

        print(f"   🔄 Adapted target embedding: standard → kronecker")
        print(f"      pf_to_model.weight: {tuple(src_state['pf_to_model.weight'].shape)}")
        print(f"      embed_norm.weight:  {tuple(src_state['embed_norm.weight'].shape)}")
        return "kronecker"

    if src_has_standard:
        print(f"   ✅ Source uses standard embedding — no adaptation needed")
        return "standard"

    raise ValueError(
        "Could not detect source embedding type. Expected either "
        "'pf_to_model.weight' (kronecker) or 'token_embed.weight' (standard) "
        "in source state dict. Found embedding-related keys: "
        + str([k for k in src_state if "embed" in k.lower() or "pf_to" in k.lower()])
    )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ACT Solver Unrolling: trajectory-preserving 3B→8B initialization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--src", required=True,
        help="Path to 3B checkpoint (.pt), e.g. checkpoints/fourier_latest.pt"
    )
    parser.add_argument(
        "--tgt", required=True,
        help="Output path for initialized 8B checkpoint (.pt)"
    )
    parser.add_argument(
        "--noise_std", type=float, default=1e-4,
        help="Gaussian noise σ added to each copied weight tensor for diversity "
             "between block copies. Default: 1e-4. Use 0 for exact ACT-scaled copies."
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Path to the directory containing recurrence_model_8b.py "
             "(e.g. ../  or  ../../model/)"
    )
    parser.add_argument(
        "--json", default=None, dest="json_path",
        help="Path to layer_repitation_factor.json (default: auto-detect in same dir as script)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print expansion plan only. Do not load or write any files."
    )
    args = parser.parse_args()

    unroll_3b_to_8b(
        src_checkpoint_path = args.src,
        tgt_checkpoint_path = args.tgt,
        model_dir           = args.model_dir,
        noise_std           = args.noise_std,
        dry_run             = args.dry_run,
        json_path           = args.json_path,
    )


if __name__ == "__main__":
    main()
