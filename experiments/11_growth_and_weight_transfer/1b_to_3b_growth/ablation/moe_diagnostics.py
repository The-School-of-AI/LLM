#!/usr/bin/env python3
"""
MoE Diagnostics — Comprehensive routing analysis for Dense-to-MoE experiments.

Provides:
  1. Per-expert token count distribution (including null)
  2. Token-to-expert mapping (which distinct tokens go to which expert)
  3. Router entropy and load balance metrics
  4. Expert contribution ratio (routed vs shared signal magnitude)
  5. Routing weight distribution stats
  6. Expert specialization analysis (what tokens each expert prefers)

Usage:
    from ablation.moe_diagnostics import MoEDiagnostics

    diag = MoEDiagnostics(model, tokenizer, num_experts=8)
    report = diag.run_diagnostics(input_ids)    # after forward pass
    diag.log_report(report, logger, step=100)
"""

import os
import torch
import torch.nn.functional as F
from collections import defaultdict


def _get_moe_layers(model):
    """Extract all MoE layers from the model (backbone + MTP)."""
    layers = []

    if hasattr(model, 'layers'):
        for idx, layer in enumerate(model.layers):
            if (hasattr(layer, 'mlp_block')
                    and hasattr(layer.mlp_block, 'sublayer')
                    and hasattr(layer.mlp_block.sublayer, 'moe')):
                moe = layer.mlp_block.sublayer.moe
                layers.append((f"layer_{idx}", moe))

    if hasattr(model, 'mtp_block') and model.mtp_block is not None:
        if hasattr(model.mtp_block, 'mlp') and hasattr(model.mtp_block.mlp, 'moe'):
            moe = model.mtp_block.mlp.moe
            layers.append(("mtp", moe))

    return layers


def expert_token_distribution(model):
    """
    Per-expert token count distribution across all layers.

    Returns:
        dict: {
            "per_layer": {layer_name: {"expert_counts": [n_e0, ..., n_e7], "null_count": int, "total": int}},
            "aggregate": {"expert_counts": [...], "null_count": int, "total": int}
        }
    """
    moe_layers = _get_moe_layers(model)
    result = {"per_layer": {}, "aggregate": None}

    agg_expert_counts = None
    agg_null = 0
    agg_total = 0

    for name, moe in moe_layers:
        if moe.last_indices is None:
            continue

        indices = moe.last_indices  # (B, T, K)
        num_experts = moe.num_experts
        is_null = (indices >= num_experts)

        # Count per real expert
        flat_idx = indices.view(-1)
        flat_null = is_null.view(-1)
        real_idx = flat_idx[~flat_null]
        counts = torch.bincount(real_idx, minlength=num_experts)[:num_experts]
        null_count = flat_null.sum().item()
        total = flat_idx.numel()

        counts_list = counts.tolist()
        result["per_layer"][name] = {
            "expert_counts": counts_list,
            "null_count": int(null_count),
            "total": total,
        }

        if agg_expert_counts is None:
            agg_expert_counts = counts.clone()
        else:
            agg_expert_counts += counts
        agg_null += null_count
        agg_total += total

    if agg_expert_counts is not None:
        result["aggregate"] = {
            "expert_counts": agg_expert_counts.tolist(),
            "null_count": int(agg_null),
            "total": agg_total,
        }

    return result


def token_expert_mapping(model, input_ids, tokenizer):
    """
    Map which distinct tokens (decoded text) go to which expert.
    Collects ALL distinct tokens — no truncation.

    Args:
        model: The 3B MoE model (after forward pass)
        input_ids: (B, T) input token IDs used in the forward pass
        tokenizer: For decoding token IDs to text

    Returns:
        dict: {
            "per_layer": {
                layer_name: {
                    "expert_0": [(token_text, count), ...],  # ALL tokens, sorted by count desc
                    ...
                    "null": [(token_text, count), ...]
                }
            }
        }
    """
    moe_layers = _get_moe_layers(model)
    result = {"per_layer": {}}

    # input_ids is (B, T_full), but model uses x=input[:-2], so adjust
    # The MoE sees tokens at positions [0..T-1] where T = T_full - 2
    x_ids = input_ids[:, :-2]  # (B, T) matching model's x_input

    for name, moe in moe_layers:
        if moe.last_indices is None:
            continue

        indices = moe.last_indices  # (B, T, K)
        num_experts = moe.num_experts
        B, T, K = indices.shape

        # Expand token IDs to match (B, T, K)
        ids_expanded = x_ids[:, :T].unsqueeze(-1).expand(B, T, K)  # (B, T, K)

        layer_map = {}
        for e in range(num_experts):
            mask = (indices == e)
            if mask.any():
                expert_token_ids = ids_expanded[mask].tolist()
                token_counts = defaultdict(int)
                for tid in expert_token_ids:
                    try:
                        text = tokenizer.decode([tid]).replace('\n', '\\n').replace('\t', '\\t')
                    except Exception:
                        text = f"<id={tid}>"
                    token_counts[text] += 1
                # Sort by count descending — ALL tokens kept
                sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])
                layer_map[f"expert_{e}"] = sorted_tokens
            else:
                layer_map[f"expert_{e}"] = []

        # Null expert
        null_mask = (indices >= num_experts)
        if null_mask.any():
            null_token_ids = ids_expanded[null_mask].tolist()
            null_counts = defaultdict(int)
            for tid in null_token_ids:
                try:
                    text = tokenizer.decode([tid]).replace('\n', '\\n').replace('\t', '\\t')
                except Exception:
                    text = f"<id={tid}>"
                null_counts[text] += 1
            sorted_null = sorted(null_counts.items(), key=lambda x: -x[1])
            layer_map["null"] = sorted_null
        else:
            layer_map["null"] = []

        result["per_layer"][name] = layer_map

    return result


def router_entropy(model):
    """
    Compute router entropy per layer (measures routing decision uncertainty).

    High entropy = uniform routing (experts equally used)
    Low entropy = concentrated routing (few experts dominate)
    Max entropy for 8+null experts with top_k=2 depends on distribution.

    Returns:
        dict: {layer_name: {"entropy": float, "max_entropy": float, "normalized_entropy": float}}
    """
    moe_layers = _get_moe_layers(model)
    result = {}

    for name, moe in moe_layers:
        gate = moe.gate
        if not hasattr(gate, 'last_probs') or gate.last_probs is None:
            continue

        probs = gate.last_probs  # (B, T, num_experts + num_null_copies)
        num_slots = probs.shape[-1]

        # Average probability distribution across all tokens
        avg_probs = probs.mean(dim=(0, 1))  # (num_slots,)
        avg_probs = avg_probs.clamp(min=1e-10)

        entropy = -(avg_probs * avg_probs.log()).sum().item()
        max_entropy = torch.tensor(num_slots, dtype=torch.float).log().item()
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        result[name] = {
            "entropy": entropy,
            "max_entropy": max_entropy,
            "normalized_entropy": normalized,
        }

    return result


def routing_weight_stats(model):
    """
    Statistics on the actual routing weights assigned to selected experts.

    Returns per-layer:
        - mean/std/min/max of real expert weights
        - fraction of tokens with zero real weight (100% null)
    """
    moe_layers = _get_moe_layers(model)
    result = {}

    for name, moe in moe_layers:
        if moe.last_weights is None or moe.last_indices is None:
            continue

        weights = moe.last_weights  # (B, T, K) — real expert weights (null slots get 0)
        indices = moe.last_indices  # (B, T, K)
        num_experts = moe.num_experts

        is_null = (indices >= num_experts)
        real_weights = weights[~is_null]

        if real_weights.numel() > 0:
            stats = {
                "mean": real_weights.mean().item(),
                "std": real_weights.std().item(),
                "min": real_weights.min().item(),
                "max": real_weights.max().item(),
                "num_real_assignments": int(real_weights.numel()),
            }
        else:
            stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "num_real_assignments": 0}

        # Tokens with zero real experts (all K selections are null)
        B, T, K = is_null.shape
        all_null_tokens = is_null.all(dim=-1)  # (B, T)
        stats["zero_real_frac"] = all_null_tokens.float().mean().item()

        result[name] = stats

    return result


def expert_load_balance(model):
    """
    Load balance coefficient (how evenly tokens are distributed among active experts).

    CV (coefficient of variation) of expert counts:
      - CV = 0: perfectly balanced
      - CV > 1: highly imbalanced

    Also computes the max/min expert ratio.
    """
    dist = expert_token_distribution(model)
    result = {}

    for layer_name, layer_data in dist["per_layer"].items():
        counts = torch.tensor(layer_data["expert_counts"], dtype=torch.float)
        total_real = counts.sum()

        if total_real > 0:
            mean_count = counts.mean()
            std_count = counts.std()
            cv = (std_count / mean_count).item() if mean_count > 0 else 0.0
            max_min_ratio = (counts.max() / counts.min()).item() if counts.min() > 0 else float('inf')
        else:
            cv = 0.0
            max_min_ratio = 0.0

        result[layer_name] = {
            "cv": cv,
            "max_min_ratio": max_min_ratio,
            "total_real": int(total_real.item()),
            "null_count": layer_data["null_count"],
            "null_pct": layer_data["null_count"] / layer_data["total"] * 100 if layer_data["total"] > 0 else 0,
        }

    return result


def run_all_diagnostics(model, input_ids=None, tokenizer=None):
    """
    Run all diagnostic analyses and return a consolidated report.

    Args:
        model: 3B MoE model (after forward pass)
        input_ids: Optional (B, T) for token mapping
        tokenizer: Optional, needed for token mapping

    Returns:
        dict with all diagnostic results
    """
    report = {
        "token_distribution": expert_token_distribution(model),
        "load_balance": expert_load_balance(model),
        "router_entropy": router_entropy(model),
        "weight_stats": routing_weight_stats(model),
    }

    if input_ids is not None and tokenizer is not None:
        report["token_expert_map"] = token_expert_mapping(model, input_ids, tokenizer)

    return report


def log_diagnostics(report, logger, step=None, verbose=True):
    """
    Log diagnostic report to logger in a readable format.

    Args:
        report: Output from run_all_diagnostics()
        logger: logging.Logger instance
        step: Optional training step number
        verbose: If True, log token-expert mapping details
    """
    prefix = f"[step={step}] " if step is not None else ""
    logger.info("")
    logger.info(f"{'=' * 70}")
    logger.info(f"{prefix}MOE DIAGNOSTICS REPORT")
    logger.info(f"{'=' * 70}")

    # 1. Token Distribution
    dist = report.get("token_distribution", {})
    agg = dist.get("aggregate")
    if agg:
        total = agg["total"]
        null_pct = agg["null_count"] / total * 100 if total > 0 else 0
        logger.info(f"")
        logger.info(f"--- TOKEN DISTRIBUTION (aggregate) ---")
        logger.info(f"  Total routing slots: {total}")
        logger.info(f"  Null selections:     {agg['null_count']} ({null_pct:.1f}%)")
        logger.info(f"  Expert counts:       {agg['expert_counts']}")

    for layer_name, layer_data in dist.get("per_layer", {}).items():
        total = layer_data["total"]
        null_pct = layer_data["null_count"] / total * 100 if total > 0 else 0
        real_total = sum(layer_data["expert_counts"])
        logger.info(f"  {layer_name}: real={real_total} null={layer_data['null_count']}({null_pct:.1f}%) counts={layer_data['expert_counts']}")

    # 2. Load Balance
    lb = report.get("load_balance", {})
    if lb:
        logger.info(f"")
        logger.info(f"--- LOAD BALANCE ---")
        for layer_name, stats in lb.items():
            logger.info(
                f"  {layer_name}: CV={stats['cv']:.3f} max/min={stats['max_min_ratio']:.2f} "
                f"real={stats['total_real']} null={stats['null_pct']:.1f}%"
            )

    # 3. Router Entropy
    ent = report.get("router_entropy", {})
    if ent:
        logger.info(f"")
        logger.info(f"--- ROUTER ENTROPY ---")
        for layer_name, stats in ent.items():
            logger.info(
                f"  {layer_name}: entropy={stats['entropy']:.4f} "
                f"(max={stats['max_entropy']:.4f}, normalized={stats['normalized_entropy']:.4f})"
            )

    # 4. Weight Stats
    ws = report.get("weight_stats", {})
    if ws:
        logger.info(f"")
        logger.info(f"--- ROUTING WEIGHT STATS ---")
        for layer_name, stats in ws.items():
            logger.info(
                f"  {layer_name}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
                f"min={stats['min']:.4f} max={stats['max']:.4f} "
                f"zero_real_frac={stats['zero_real_frac']:.4f} "
                f"n_real={stats['num_real_assignments']}"
            )

    # 5. Token-Expert Mapping (verbose — top 5 in log, full detail in saved file)
    if verbose:
        token_map = report.get("token_expert_map", {})
        if token_map:
            logger.info(f"")
            logger.info(f"--- TOKEN-EXPERT MAPPING (top 5 shown, full detail in saved file) ---")
            for layer_name, expert_map in token_map.get("per_layer", {}).items():
                logger.info(f"  [{layer_name}]")
                for expert_name, tokens in expert_map.items():
                    if tokens:
                        top5 = tokens[:5]
                        token_str = ", ".join(f"'{t}'x{c}" for t, c in top5)
                        remaining = len(tokens) - 5
                        suffix = f" (+{remaining} more)" if remaining > 0 else ""
                        logger.info(f"    {expert_name} [{len(tokens)} distinct]: {token_str}{suffix}")
                    else:
                        logger.info(f"    {expert_name}: (no tokens)")

    logger.info(f"{'=' * 70}")
    logger.info("")


def save_detailed_diagnostics(report, save_dir, step=None):
    """
    Save the COMPLETE token-expert mapping to a detailed text file.
    Every distinct token per expert, per layer — nothing truncated.

    Output format:
        logs/diagnostics/
            token_map_step_100.txt    (or token_map_init.txt / token_map_final.txt)

    File contains a formatted table per layer showing all tokens.
    """
    diag_dir = os.path.join(save_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    step_label = f"step_{step}" if step is not None else "unknown"
    filepath = os.path.join(diag_dir, f"token_map_{step_label}.txt")

    token_map = report.get("token_expert_map", {})
    dist = report.get("token_distribution", {})
    ws = report.get("weight_stats", {})
    lb = report.get("load_balance", {})
    ent = report.get("router_entropy", {})

    lines = []
    lines.append(f"{'=' * 90}")
    lines.append(f"MOE DETAILED DIAGNOSTICS — Step: {step_label}")
    lines.append(f"{'=' * 90}")
    lines.append("")

    # ── Summary table ──
    lines.append(f"{'─' * 90}")
    lines.append(f"{'LAYER':<12} {'REAL':>6} {'NULL':>6} {'NULL%':>7} {'CV':>7} {'ENTROPY':>9} {'AVG_W':>8} {'ZERO%':>7}")
    lines.append(f"{'─' * 90}")

    for layer_name in dist.get("per_layer", {}):
        ld = dist["per_layer"][layer_name]
        real = sum(ld["expert_counts"])
        null = ld["null_count"]
        null_pct = null / ld["total"] * 100 if ld["total"] > 0 else 0

        cv_val = lb.get(layer_name, {}).get("cv", 0)
        ent_val = ent.get(layer_name, {}).get("normalized_entropy", 0)
        w_val = ws.get(layer_name, {}).get("mean", 0)
        z_val = ws.get(layer_name, {}).get("zero_real_frac", 0) * 100

        lines.append(
            f"{layer_name:<12} {real:>6} {null:>6} {null_pct:>6.1f}% {cv_val:>7.3f} {ent_val:>9.4f} {w_val:>8.4f} {z_val:>6.1f}%"
        )

    lines.append(f"{'─' * 90}")
    lines.append("")

    # ── Per-layer detailed token mapping ──
    if token_map:
        for layer_name, expert_map in token_map.get("per_layer", {}).items():
            lines.append(f"{'═' * 90}")
            lines.append(f"  {layer_name}")
            lines.append(f"{'═' * 90}")

            for expert_name, tokens in expert_map.items():
                total_assignments = sum(c for _, c in tokens)
                distinct_count = len(tokens)
                lines.append(f"")
                lines.append(f"  {expert_name}  ({total_assignments} assignments, {distinct_count} distinct tokens)")
                lines.append(f"  {'─' * 60}")

                if tokens:
                    # Table header
                    lines.append(f"  {'RANK':>4}  {'COUNT':>5}  {'%':>6}  TOKEN")
                    lines.append(f"  {'─' * 60}")
                    for rank, (text, count) in enumerate(tokens, 1):
                        pct = count / total_assignments * 100 if total_assignments > 0 else 0
                        # Escape for readability
                        display_text = repr(text) if text.strip() == '' or len(text) == 0 else f"'{text}'"
                        lines.append(f"  {rank:>4}  {count:>5}  {pct:>5.1f}%  {display_text}")
                else:
                    lines.append(f"  (no tokens routed to this expert)")

            lines.append("")

    # ── Per-layer expert count distribution ──
    lines.append(f"{'═' * 90}")
    lines.append(f"  EXPERT COUNT DISTRIBUTION (per layer)")
    lines.append(f"{'═' * 90}")
    for layer_name, ld in dist.get("per_layer", {}).items():
        counts = ld["expert_counts"]
        total = sum(counts) + ld["null_count"]
        lines.append(f"")
        lines.append(f"  {layer_name}:")
        for i, c in enumerate(counts):
            bar = '█' * int(c / max(max(counts), 1) * 40)
            pct = c / total * 100 if total > 0 else 0
            lines.append(f"    expert_{i}: {c:>5} ({pct:>5.1f}%) {bar}")
        null_pct = ld["null_count"] / total * 100 if total > 0 else 0
        bar = '█' * int(ld["null_count"] / max(max(counts + [ld["null_count"]]), 1) * 40)
        lines.append(f"    null:     {ld['null_count']:>5} ({null_pct:>5.1f}%) {bar}")

    lines.append("")
    lines.append(f"{'=' * 90}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return filepath


def expert_output_scales(model):
    """Extract expert_output_scale values from all MoE layers."""
    moe_layers = _get_moe_layers(model)
    scales = {}
    for name, moe in moe_layers:
        if hasattr(moe, 'expert_output_scale'):
            scales[name] = moe.expert_output_scale.item()
    return scales


def log_compact_diagnostics(model, logger, step):
    """
    Log a single compact line of MoE diagnostics per step.
    Useful for periodic monitoring without verbose output.
    """
    dist = expert_token_distribution(model)
    agg = dist.get("aggregate")
    if agg is None:
        return

    total = agg["total"]
    null_pct = agg["null_count"] / total * 100 if total > 0 else 0
    counts = agg["expert_counts"]
    counts_t = torch.tensor(counts, dtype=torch.float)
    cv = (counts_t.std() / counts_t.mean()).item() if counts_t.mean() > 0 else 0.0

    ws = routing_weight_stats(model)
    avg_weight = 0.0
    zero_frac = 1.0
    for stats in ws.values():
        avg_weight = max(avg_weight, stats["mean"])
        zero_frac = min(zero_frac, stats["zero_real_frac"])

    # Expert output scale tracking
    scales = expert_output_scales(model)
    scale_str = ""
    if scales:
        scale_vals = list(scales.values())
        scale_str = f" | exp_scale={sum(scale_vals)/len(scale_vals):.4f}"

    logger.info(
        f"  [diag] step={step} | null%={null_pct:.1f} | "
        f"expert_counts={counts} | CV={cv:.3f} | "
        f"avg_w={avg_weight:.4f} | zero_real%={zero_frac*100:.1f}{scale_str}"
    )