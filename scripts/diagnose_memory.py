#!/usr/bin/env python3
"""
Memory OOM diagnostic for Model1B (DeltaNet + GSA, reversible stack).

Scans for the #1 OOM pattern: Python for-loops over sequence length in forward/force
that retain full autograd graphs (~160 MB per step × T = huge).

Usage:
  python scripts/diagnose_memory.py --batch 16 --seq_len 512 --device_gb 48
  python scripts/diagnose_memory.py --batch 16 --seq_len 2048 --device_gb 40

See also: docs/MEMORY_OOM_REPORT.md (bootstrap layer checkpointing fix).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory OOM diagnostic (Model1B)")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    p.add_argument("--device_gb", type=float, default=48.0, help="Device memory in GB")
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--head_dim", type=int, default=128)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Section 1: Params, buffers, optimizer state (rough)
# -----------------------------------------------------------------------------
def section1_params_optimizer(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("Section 1: Parameters and optimizer state (estimate)")
    print("=" * 60)
    try:
        from src.models.recurrence_model_1b import ModelConfig, Model1B
        cfg = ModelConfig()
        n_params = sum(p.numel() for p in Model1B(cfg).parameters())
        n_b = n_params * 4  # fp32
        n_grad = n_params * 2  # bf16
        # AdamW: 2 states per param (fp32)
        opt_state = n_params * 2 * 4
        total_est = (n_b + n_grad + opt_state) / (1024**3)
        print(f"  Parameters:     {n_params:,}  (~{n_params*2/1e9:.2f} GB bf16)")
        print(f"  Gradients:     ~{n_grad/(1024**3):.2f} GB")
        print(f"  Optimizer:     ~{opt_state/(1024**3):.2f} GB (AdamW fp32)")
        print(f"  Total (model+grad+opt): ~{total_est:.2f} GB")
    except Exception as e:
        print(f"  (Could not load model: {e})")


# -----------------------------------------------------------------------------
# Section 2: Activation sizes (simplified — formula only)
# -----------------------------------------------------------------------------
def section2_activation_formula(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("Section 2: Per-step activation formula (DeltaNet-style loop)")
    print("=" * 60)
    B, T, H, D = args.batch, args.seq_len, args.num_heads, args.head_dim
    # One (B, H, D, D) tensor in fp32 = B*H*D*D*4
    per_tensor = B * H * D * D * 4
    num_tensors = 5  # v_outer, k_outer, S, orthogonal_proj, einsum intermediate
    per_step_mb = (per_tensor * num_tensors) / (1024 * 1024)
    total_retained_mb = per_step_mb * T
    print(f"  Batch={B}, Heads={H}, head_dim={D}, Seq={T}")
    print(f"  Per-step tensors (fp32): ~{num_tensors} x ({B},{H},{D},{D}) = {per_step_mb:.1f} MB/step")
    print(f"  If loop retains autograd: {per_step_mb:.1f} x {T} = {total_retained_mb:.1f} MB ({total_retained_mb/1024:.2f} GB)")


# -----------------------------------------------------------------------------
# Section 3: SCAN for Python for-loops in forward/force (CRITICAL)
# -----------------------------------------------------------------------------
def _find_loops_in_file(path: Path) -> List[Tuple[int, str, str]]:
    """Return list of (line_no, func_name, snippet) for 'for ... in range' in any method."""
    text = path.read_text()
    findings = []
    lines = text.splitlines()
    current_method = None
    indent_method = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def "):
            current_indent = len(line) - len(line.lstrip())
            if indent_method is not None and current_indent <= indent_method:
                current_method = None
            name_match = re.match(r"def\s+(\w+)\s*\(", stripped)
            if name_match:
                current_method = name_match.group(1)
                indent_method = current_indent
            continue
        if current_method is not None:
            current_indent = len(line) - len(line.lstrip())
            if stripped.startswith("def ") and current_indent <= indent_method:
                current_method = None
                continue
            if "for " in stripped and " in " in stripped and "range(" in stripped:
                findings.append((i + 1, current_method, stripped[:80]))
    return findings


def section3_scan_loops(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("Section 3: SCAN for Python for-loops in forward/force (CRITICAL)")
    print("=" * 60)
    model_dir = ROOT / "src" / "models"
    files = list(model_dir.glob("*.py"))
    all_findings: List[Tuple[Path, int, str, str]] = []
    for path in sorted(files):
        for line_no, func, snippet in _find_loops_in_file(path):
            all_findings.append((path, line_no, func, snippet))
    if not all_findings:
        print("  No 'for ... in range(...)' loops found in any method.")
        return
    # Highlight the dangerous one: loop over T (sequence) in DeltaNet/attention path
    dangerous = [
        (p, ln, f, s) for p, ln, f, s in all_findings
        if "_delta_rule_python" in f or ("forward" in f and ("range(T)" in s or "range(t" in s.lower()))
    ]
    rest = [(p, ln, f, s) for (p, ln, f, s) in all_findings if (p, ln, f, s) not in dangerous]
    if dangerous:
        print("  CRITICAL (sequence-length loop in training path):")
        for path, line_no, func, snippet in dangerous:
            short = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            print(f"    {short}:{line_no}  def {func}  ->  {snippet}")
    if rest:
        print("  Other methods with 'for ... in range(...)' (lower OOM risk):")
        for path, line_no, func, snippet in rest:
            short = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            print(f"    {short}:{line_no}  def {func}  ->  {snippet}")
    print("\n  If these run WITH autograd (no checkpoint), peak memory = per_step_MB x seq_len.")
    print("  Fix: wrap the layer with gradient checkpointing (e.g. bootstrap_layer.force in")
    print("  reversible_ops_midpoint.py). DeltaNet loop lives in _delta_rule_python, called from forward.")


# -----------------------------------------------------------------------------
# Section 4: Gradient checkpointing usage
# -----------------------------------------------------------------------------
def section4_checkpoint_usage(_args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("Section 4: Gradient checkpointing usage")
    print("=" * 60)
    patterns = [
        ("grad_checkpoint", "reversible_ops_midpoint.py"),
        ("checkpoint(", "torch.utils.checkpoint / grad_checkpoint"),
        ("configure_activation_checkpointing", "DeepSpeed"),
    ]
    for pattern, desc in patterns:
        count = 0
        for path in [ROOT / "src" / "models" / "reversible_ops_midpoint.py", ROOT / "src" / "models" / "recurrence_model_1b.py"]:
            if path.exists():
                count += len(re.findall(re.escape(pattern), path.read_text()))
        print(f"  {desc}: {count} occurrence(s)")


# -----------------------------------------------------------------------------
# Section 5: Peak memory estimate vs device
# -----------------------------------------------------------------------------
def section5_peak_estimate(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("Section 5: Peak memory estimate vs device budget")
    print("=" * 60)
    B, T, H, D = args.batch, args.seq_len, args.num_heads, args.head_dim
    per_step = B * H * D * D * 4 * 5  # 5 fp32 tensors per step
    peak_loop_gb = (per_step * T) / (1024**3)
    device_gb = args.device_gb
    print(f"  DeltaNet loop (if uncheckpointed): ~{peak_loop_gb:.2f} GB (batch={B}, seq={T})")
    print(f"  Device budget: {device_gb:.2f} GB")
    if peak_loop_gb > device_gb * 0.9:
        print("  -> RISK: Loop memory alone is close to or exceeds device. Ensure bootstrap uses checkpoint.")
    else:
        print("  -> Loop memory is within device if checkpointing is applied.")


# -----------------------------------------------------------------------------
# Section 6: GSA sparse attention (where OOM often happens on 40GB GPUs)
# -----------------------------------------------------------------------------
def section6_gsa_sparse_attn(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("Section 6: GSA sparse attention path (triton_sparse_attn.py)")
    print("=" * 60)
    device_gb = args.device_gb
    # Typical observed: PyTorch reports ~37.8 GB allocated when OOM at k_gathered
    baseline_observed_gb = 37.8
    headroom_gb = device_gb - baseline_observed_gb
    print("  Training uses PyTorch sparse-attention fallback (Triton only at inference).")
    print("  Fallback allocates k_gathered/v_gathered per chunk (chunk_size=2 -> ~16 MB each).")
    print(f"  On ~{device_gb:.0f} GB GPUs, forward often reaches ~{baseline_observed_gb:.1f} GB before GSA.")
    print(f"  Headroom before GSA: ~{headroom_gb:.1f} GB (fragmentation can leave only 20–100 MB free).")
    if headroom_gb < 0.5:
        print("  -> OOM likely in GSA: not enough headroom for k_gathered. Reduce max_length or enable")
        print("     activation checkpointing / reduce gsa_k_max in model config.")


def main() -> int:
    args = parse_args()
    print("Memory OOM Diagnostic (Model1B)")
    print(f"  batch={args.batch}, seq_len={args.seq_len}, device_gb={args.device_gb}")
    section1_params_optimizer(args)
    section2_activation_formula(args)
    section3_scan_loops(args)
    section4_checkpoint_usage(args)
    section5_peak_estimate(args)
    section6_gsa_sparse_attn(args)
    print("\n  Summary: If OOM is in triton_sparse_attn.py (k_gathered), total activation memory")
    print("  before GSA is using almost all GPU memory. Apply: lower data.max_length, or")
    print("  enable activation checkpointing in DeepSpeed config, or reduce gsa_k_max.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
