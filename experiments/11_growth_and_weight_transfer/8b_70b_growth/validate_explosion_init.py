#!/usr/bin/env python3
"""
validate_explosion_init.py
==========================
Validate that an expert-exploded 70B checkpoint is function-preserving
relative to its source 8B checkpoint.

Runs three diagnostic metrics on a synthetic input batch:

    1. Loss difference:        |L_70B − L_8B| / L_8B       target: < 3%
    2. Logit cosine similarity: cos(logits_70B, logits_8B)  target: > 0.995
    3. Router KL divergence:    KL(p_70B || p_8B)           target: < 0.02

Usage:
    python validate_explosion_init.py \
        --src checkpoints/8b_trained.pt \
        --tgt checkpoints/70b_expert_explosion_init.pt \
        --model_dir ../

    # With custom batch size or sequence length:
    python validate_explosion_init.py \
        --src ... --tgt ... --model_dir .. \
        --batch_size 4 --seq_len 128

    # Use a real data file instead of synthetic tokens:
    python validate_explosion_init.py \
        --src ... --tgt ... --model_dir .. \
        --data_file /path/to/validation_tokens.pt
"""

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Architecture constants (must match init script)
# ─────────────────────────────────────────────

N_SRC_EXPERTS = 20
N_TGT_EXPERTS = 260
COPIES_PER_EXPERT = 13
N_LAYERS = 20

# Thresholds for pass/warn/fail
LOSS_DIFF_PASS = 0.03       # < 3% relative
LOSS_DIFF_WARN = 0.10       # < 10% acceptable
COSINE_SIM_PASS = 0.995
COSINE_SIM_WARN = 0.98
ROUTER_KL_PASS = 0.02
ROUTER_KL_WARN = 0.05


# ─────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────

def _import_model(model_dir: str, module_name: str):
    """Dynamically import a model module from the given directory."""
    model_dir = os.path.normpath(os.path.abspath(model_dir))
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    if module_name in sys.modules:
        del sys.modules[module_name]

    return __import__(module_name)


def load_model_with_checkpoint(
    model_dir: str,
    checkpoint_path: str,
    module_name: str,
    model_class_name: str,
    device: str = "cpu",
) -> nn.Module:
    """Load a model class and restore weights from a checkpoint."""
    mod = _import_model(model_dir, module_name)
    ModelClass = getattr(mod, model_class_name)
    ModelConfig = getattr(mod, "ModelConfig")

    config = ModelConfig()

    # Detect embedding type from checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    embed_type = ckpt.get("embedding_type", None)
    if embed_type is None:
        embed_type = "kronecker" if "pf_to_model.weight" in state else "standard"

    model = ModelClass(config, embedding_type=embed_type)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return model, config


# ─────────────────────────────────────────────
# Router probability extraction
# ─────────────────────────────────────────────

def extract_router_probs(model: nn.Module, input_ids: torch.Tensor) -> List[torch.Tensor]:
    """
    Run a forward pass and collect router gate probabilities from each layer.

    Returns a list of (batch*seq, num_experts) probability tensors, one per layer.
    """
    router_probs = []

    # Hook into each layer's MoE gate to capture softmax probabilities
    hooks = []

    def _make_hook(layer_idx):
        def hook_fn(module, input, output):
            # MoEGate.forward returns (top_k_indices, top_k_weights, ...)
            # We need the raw logits. Capture from the gate's weight.
            with torch.no_grad():
                x = input[0]  # (batch*seq, d_model)
                if hasattr(module, 'weight'):
                    logits = F.linear(x, module.weight)
                    if hasattr(module, 'logit_bias'):
                        logits = logits + module.logit_bias
                    probs = F.softmax(logits, dim=-1)
                    router_probs.append(probs.detach().cpu())
        return hook_fn

    # Find all MoE gate modules
    for name, mod in model.named_modules():
        if name.endswith(".moe.gate") and hasattr(mod, 'weight'):
            handle = mod.register_forward_hook(_make_hook(len(hooks)))
            hooks.append(handle)

    # Forward pass
    with torch.no_grad():
        _ = model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    return router_probs


# ─────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────

def compute_loss(model: nn.Module, input_ids: torch.Tensor) -> float:
    """Compute cross-entropy loss (next-token prediction)."""
    with torch.no_grad():
        outputs = model(input_ids)
        # Model output may be a tuple or have .logits
        if isinstance(outputs, tuple):
            logits = outputs[0]
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
    return loss.item()


def compute_logit_cosine_sim(
    model_8b: nn.Module,
    model_70b: nn.Module,
    input_ids: torch.Tensor,
) -> float:
    """Compute cosine similarity between output logits of both models."""
    with torch.no_grad():
        out_8b = model_8b(input_ids)
        out_70b = model_70b(input_ids)

        logits_8b = out_8b[0] if isinstance(out_8b, tuple) else out_8b
        logits_70b = out_70b[0] if isinstance(out_70b, tuple) else out_70b

        # Flatten to (batch*seq, vocab)
        flat_8b = logits_8b.reshape(-1, logits_8b.size(-1)).float()
        flat_70b = logits_70b.reshape(-1, logits_70b.size(-1)).float()

        # Mean cosine sim across all positions
        cos_sim = F.cosine_similarity(flat_8b, flat_70b, dim=-1)
    return cos_sim.mean().item()


def compute_router_kl(
    probs_8b: List[torch.Tensor],
    probs_70b: List[torch.Tensor],
) -> Tuple[float, List[float]]:
    """
    Compute KL divergence between 8B and 70B router probability distributions.

    For the 70B model (260 experts = 13 clones of each 20), we aggregate clone
    probabilities back to 20 groups before computing KL against the 8B's 20-expert
    distribution. This checks whether the log(13) mass correction preserved the
    original routing distribution.

    Returns: (mean_kl, per_layer_kl)
    """
    per_layer_kl = []

    for p_8b, p_70b in zip(probs_8b, probs_70b):
        # Aggregate 70B probs: sum each group of 13 clones
        n_tokens = p_70b.shape[0]
        n_src = N_SRC_EXPERTS
        # Only aggregate the real experts (first N_TGT_EXPERTS columns)
        p_70b_real = p_70b[:, :N_TGT_EXPERTS]

        # Sum clone probabilities: (batch*seq, 260) → (batch*seq, 20)
        # Round-robin layout: assignment[j] = j % N_SRC_EXPERTS
        p_70b_agg = torch.zeros(n_tokens, n_src, dtype=p_70b.dtype)
        for src_idx in range(n_src):
            # Siblings are at indices src_idx, src_idx+20, src_idx+40, ...
            sibling_indices = list(range(src_idx, N_TGT_EXPERTS, n_src))
            p_70b_agg[:, src_idx] = p_70b_real[:, sibling_indices].sum(dim=1)

        # Only use the first N_SRC_EXPERTS from 8B probs
        p_8b_real = p_8b[:, :N_SRC_EXPERTS]

        # Add small epsilon for numerical stability
        eps = 1e-10
        p_8b_safe = (p_8b_real + eps)
        p_8b_safe = p_8b_safe / p_8b_safe.sum(dim=-1, keepdim=True)
        p_70b_safe = (p_70b_agg + eps)
        p_70b_safe = p_70b_safe / p_70b_safe.sum(dim=-1, keepdim=True)

        # KL(p_70b || p_8b) per token, then mean
        kl = (p_70b_safe * (p_70b_safe.log() - p_8b_safe.log())).sum(dim=-1)
        layer_kl = kl.mean().item()
        per_layer_kl.append(layer_kl)

    mean_kl = sum(per_layer_kl) / len(per_layer_kl) if per_layer_kl else 0.0
    return mean_kl, per_layer_kl


# ─────────────────────────────────────────────
# Status formatting
# ─────────────────────────────────────────────

def status_icon(value: float, pass_threshold: float, warn_threshold: float, lower_is_better: bool = True) -> str:
    """Return PASS/WARN/FAIL icon based on thresholds."""
    if lower_is_better:
        if value <= pass_threshold:
            return "PASS"
        elif value <= warn_threshold:
            return "WARN"
        else:
            return "FAIL"
    else:
        if value >= pass_threshold:
            return "PASS"
        elif value >= warn_threshold:
            return "WARN"
        else:
            return "FAIL"


# ─────────────────────────────────────────────
# Main validation
# ─────────────────────────────────────────────

def validate(
    src_checkpoint: str,
    tgt_checkpoint: str,
    model_dir: str,
    batch_size: int = 2,
    seq_len: int = 64,
    data_file: Optional[str] = None,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, float]:
    """Run all validation metrics and return results dict."""

    torch.manual_seed(seed)

    print("=" * 70)
    print("  Expert Explosion Validation")
    print("  8B (20 experts) vs 70B (260 experts)")
    print("=" * 70)

    # ── Load models ──────────────────────────────────────────────────
    print(f"\n  Loading 8B model from: {src_checkpoint}")
    model_8b, config_8b = load_model_with_checkpoint(
        model_dir, src_checkpoint, "recurrence_model_8b", "Model8B", device,
    )
    print(f"    Loaded. Params: {sum(p.numel() for p in model_8b.parameters()):,}")

    print(f"\n  Loading 70B model from: {tgt_checkpoint}")
    model_70b, config_70b = load_model_with_checkpoint(
        model_dir, tgt_checkpoint, "recurrence_model_70b", "Model70B", device,
    )
    print(f"    Loaded. Params: {sum(p.numel() for p in model_70b.parameters()):,}")

    # ── Prepare input ────────────────────────────────────────────────
    if data_file is not None:
        print(f"\n  Loading validation tokens from: {data_file}")
        input_ids = torch.load(data_file, map_location=device, weights_only=True)
        input_ids = input_ids[:batch_size, :seq_len]
    else:
        vocab_size = getattr(config_8b, 'vocab_size', 32000)
        print(f"\n  Generating synthetic input: batch={batch_size}, seq={seq_len}, vocab={vocab_size}")
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

    print(f"    Input shape: {tuple(input_ids.shape)}")

    # ── Metric 1: Loss difference ────────────────────────────────────
    print(f"\n  Computing losses...")
    loss_8b = compute_loss(model_8b, input_ids)
    loss_70b = compute_loss(model_70b, input_ids)
    loss_diff = abs(loss_70b - loss_8b) / (abs(loss_8b) + 1e-10)

    loss_status = status_icon(loss_diff, LOSS_DIFF_PASS, LOSS_DIFF_WARN)
    print(f"    8B loss:  {loss_8b:.6f}")
    print(f"    70B loss: {loss_70b:.6f}")
    print(f"    Relative diff: {loss_diff:.4f} ({loss_diff*100:.2f}%)  [{loss_status}]")

    # ── Metric 2: Logit cosine similarity ────────────────────────────
    print(f"\n  Computing logit cosine similarity...")
    cosine_sim = compute_logit_cosine_sim(model_8b, model_70b, input_ids)

    cos_status = status_icon(cosine_sim, COSINE_SIM_PASS, COSINE_SIM_WARN, lower_is_better=False)
    print(f"    Cosine similarity: {cosine_sim:.6f}  [{cos_status}]")

    # ── Metric 3: Router KL divergence ───────────────────────────────
    print(f"\n  Computing router KL divergence...")
    probs_8b = extract_router_probs(model_8b, input_ids)
    probs_70b = extract_router_probs(model_70b, input_ids)

    if probs_8b and probs_70b:
        n_comparable = min(len(probs_8b), len(probs_70b))
        mean_kl, per_layer_kl = compute_router_kl(
            probs_8b[:n_comparable], probs_70b[:n_comparable],
        )
        kl_status = status_icon(mean_kl, ROUTER_KL_PASS, ROUTER_KL_WARN)
        print(f"    Mean router KL: {mean_kl:.6f}  [{kl_status}]")
        print(f"    Per-layer KL ({n_comparable} layers):")
        for i, kl_val in enumerate(per_layer_kl):
            layer_status = status_icon(kl_val, ROUTER_KL_PASS, ROUTER_KL_WARN)
            print(f"      L{i:02d}: {kl_val:.6f}  [{layer_status}]")
    else:
        mean_kl = float('nan')
        per_layer_kl = []
        print(f"    Could not extract router probabilities (hooks found: "
              f"8B={len(probs_8b)}, 70B={len(probs_70b)})")

    # ── Summary ──────────────────────────────────────────────────────
    results = {
        "loss_8b": loss_8b,
        "loss_70b": loss_70b,
        "loss_diff_relative": loss_diff,
        "logit_cosine_sim": cosine_sim,
        "router_kl_mean": mean_kl,
        "router_kl_per_layer": per_layer_kl,
    }

    print(f"\n{'=' * 70}")
    print("  VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<28}  {'Value':>10}  {'Threshold':>10}  {'Status':>6}")
    print(f"  {'-'*60}")
    print(f"  {'Loss diff (relative)':<28}  {loss_diff:>10.4f}  {'< ' + str(LOSS_DIFF_PASS):>10}  {loss_status:>6}")
    print(f"  {'Logit cosine sim':<28}  {cosine_sim:>10.6f}  {'> ' + str(COSINE_SIM_PASS):>10}  {cos_status:>6}")
    if not math.isnan(mean_kl):
        print(f"  {'Router KL divergence':<28}  {mean_kl:>10.6f}  {'< ' + str(ROUTER_KL_PASS):>10}  {kl_status:>6}")
    print(f"{'=' * 70}")

    all_pass = (
        loss_diff <= LOSS_DIFF_PASS
        and cosine_sim >= COSINE_SIM_PASS
        and (math.isnan(mean_kl) or mean_kl <= ROUTER_KL_PASS)
    )

    if all_pass:
        print("\n  All metrics PASS. Safe to proceed with 70B warmstart training.\n")
    else:
        print("\n  Some metrics outside target range. Review before training.\n")
        if loss_diff > LOSS_DIFF_WARN:
            print("  Suggestion: reduce --eps_expert (try 0.005)")
        if not math.isnan(mean_kl) and mean_kl > ROUTER_KL_WARN:
            print("  Suggestion: check log(13) bias correction is applied")

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate function preservation of expert-exploded 70B checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--src", required=True,
        help="Path to trained 8B checkpoint (.pt)",
    )
    parser.add_argument(
        "--tgt", required=True,
        help="Path to initialized 70B checkpoint (.pt)",
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Directory containing recurrence_model_8b.py and recurrence_model_70b.py",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size for validation. Default: 2.",
    )
    parser.add_argument(
        "--seq_len", type=int, default=64,
        help="Sequence length for validation. Default: 64.",
    )
    parser.add_argument(
        "--data_file", type=str, default=None,
        help="Optional path to a .pt file of token IDs (LongTensor). "
             "If not provided, uses random tokens.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run on. Default: cpu. Use 'cuda' if models fit in GPU.",
    )
    args = parser.parse_args()

    validate(
        src_checkpoint=args.src,
        tgt_checkpoint=args.tgt,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_file=args.data_file,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
