#!/usr/bin/env python3
"""
Generate reversibility analysis by directly importing compute.py.
Sweeps batch sizes 1→2048 and creates reversibility_analysis.md.

Usage:
    python3 generate_reversibility_report.py
"""
import json
import os
import sys

# Import from compute.py in same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute import (
    TrainingStage,
    bytes_for_precision,
    load_config,
    normalize_deepspeed_config,
)

CONFIG = "configs/moe_team8/stage4_70b_moe.json"
OUTPUT = "reversibility_analysis.md"
GPU_MEM_GIB = 141.0  # H200
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def get_memory(config_path, batch_size, reversible, quantization="bf16"):
    """Compute memory per GPU for given settings — matches compute.py main exactly."""
    config = load_config(config_path)
    config = normalize_deepspeed_config(config)

    hw = config["hardware"]
    stage_conf = config["stages"][0]

    # Override batch size and reversibility
    arch = dict(stage_conf["architecture"])
    arch["reversible"] = reversible
    hw["_micro_batch_size"] = batch_size

    stage = TrainingStage(
        name=stage_conf["name"],
        total_tokens=float(stage_conf["total_tokens"]),
        architecture=arch,
        description="",
    )

    # Call exactly like main does (positional args in same order)
    mem = stage.calculate_memory_per_gpu(
        None,  # params
        hw["num_gpus"],  # num_gpus
        hw.get("zero_stage", 2),  # zero_stage
        quantization,  # quantization
        hw.get("cpu_offload", False),  # cpu_offload
        hw.get("cpu_offload_config"),  # cpu_offload_config
        int(hw.get("expert_parallel_size", hw.get("ep", 1))),  # expert_parallel_size
        hw.get("_checkpoint_factor"),  # checkpoint_factor
        hw.get("_include_activation_memory"),  # include_activation_memory
        batch_size,  # micro_batch_size
        hw.get("_partition_activations", False),  # partition_activations
    )
    return mem["memory_per_gpu_gb"]


def main():
    print("=" * 70)
    print("Reversibility Analysis — 70B MoE")
    print("=" * 70)

    results = []
    for bs in BATCH_SIZES:
        print(f"  batch={bs:>5}  ", end="", flush=True)
        std_bf16 = get_memory(CONFIG, bs, False, "bf16")
        rev_bf16 = get_memory(CONFIG, bs, True, "bf16")
        std_fp8 = get_memory(CONFIG, bs, False, "fp8")
        rev_fp8 = get_memory(CONFIG, bs, True, "fp8")
        results.append((bs, std_bf16, rev_bf16, std_fp8, rev_fp8))
        print(
            f"std_bf16={std_bf16:.1f}  rev_bf16={rev_bf16:.1f}  "
            f"std_fp8={std_fp8:.1f}  rev_fp8={rev_fp8:.1f}"
        )

    # --- Build Markdown ---
    lines = []
    lines.append("# Reversible Training: Memory Impact Analysis")
    lines.append("")
    lines.append("## 70B MoE (DeltaNet+GSA) — 8× H200 GPUs (141 GiB/GPU)")
    lines.append("")
    lines.append(
        "Compares GPU memory at different micro batch sizes, with and without reversible training."
    )
    lines.append("")
    lines.append(
        '> **Paper**: Gal et al., *"Reversing Large Language Models for Efficient Training'
    )
    lines.append(
        '> and Fine-Tuning"* ([arXiv:2512.02056v2](https://arxiv.org/abs/2512.02056))'
    )
    lines.append("")

    # BF16
    lines.append("### BF16 Precision")
    lines.append("")
    lines.append(
        "| Micro Batch | Standard (GiB) | Reversible (GiB) | Δ Savings (GiB) | Std Fits? | Rev Fits? |"
    )
    lines.append("|:---:|---:|---:|---:|:---:|:---:|")
    for bs, sb, rb, sf8, rf8 in results:
        d = sb - rb
        s_ok = "✅" if sb <= GPU_MEM_GIB else "❌ OOM"
        r_ok = "✅" if rb <= GPU_MEM_GIB else "❌ OOM"
        lines.append(f"| {bs} | {sb:.1f} | {rb:.1f} | {d:.1f} | {s_ok} | {r_ok} |")

    lines.append("")

    # FP8
    lines.append("### FP8 Precision")
    lines.append("")
    lines.append(
        "| Micro Batch | Standard (GiB) | Reversible (GiB) | Δ Savings (GiB) | Std Fits? | Rev Fits? |"
    )
    lines.append("|:---:|---:|---:|---:|:---:|:---:|")
    for bs, sb, rb, sf8, rf8 in results:
        d = sf8 - rf8
        s_ok = "✅" if sf8 <= GPU_MEM_GIB else "❌ OOM"
        r_ok = "✅" if rf8 <= GPU_MEM_GIB else "❌ OOM"
        lines.append(f"| {bs} | {sf8:.1f} | {rf8:.1f} | {d:.1f} | {s_ok} | {r_ok} |")

    lines.append("")

    # Key findings
    std_max_bf16 = max(
        [bs for bs, sb, rb, _, _ in results if sb <= GPU_MEM_GIB], default=0
    )
    rev_max_bf16 = max(
        [bs for bs, sb, rb, _, _ in results if rb <= GPU_MEM_GIB], default=0
    )
    std_max_fp8 = max(
        [bs for bs, _, _, sf8, rf8 in results if sf8 <= GPU_MEM_GIB], default=0
    )
    rev_max_fp8 = max(
        [bs for bs, _, _, sf8, rf8 in results if rf8 <= GPU_MEM_GIB], default=0
    )

    lines.append("### Key Findings")
    lines.append("")
    lines.append("| Metric | BF16 | FP8 |")
    lines.append("|--------|------|-----|")
    lines.append(f"| Max batch (Standard) | **{std_max_bf16}** | **{std_max_fp8}** |")
    lines.append(f"| Max batch (Reversible) | **{rev_max_bf16}** | **{rev_max_fp8}** |")
    if std_max_bf16 > 0:
        lines.append(
            f"| Batch increase | **{rev_max_bf16 / std_max_bf16:.0f}×** | **{rev_max_fp8 / max(std_max_fp8, 1):.0f}×** |"
        )
    lines.append("")

    lines.append("### How Reversibility Works")
    lines.append("")
    lines.append("| Component | Standard | Reversible |")
    lines.append("|-----------|----------|------------|")
    lines.append(
        "| Activation Memory | `B × S × H × L × 10 × bytes` (O(layers)) | `2 × B × S × H × bytes` (O(1)) |"
    )
    lines.append("| FLOPs Overhead | 1.0× | 1.33× (recompute fwd during bwd) |")
    lines.append("| Memory Bottleneck | Activations + Weights | Weights only |")
    lines.append("")
    lines.append(
        "Standard training stores activations for all layers — memory grows with batch size AND depth."
    )
    lines.append(
        "Reversible training reconstructs activations during backward using invertible dynamics"
    )
    lines.append(
        "(midpoint/leapfrog), storing only 2 hidden-state tensors regardless of depth."
    )
    lines.append(
        "The activation cost becomes negligible; the bottleneck shifts to model weights + optimizer states."
    )
    lines.append("")

    text = "\n".join(lines)
    with open(OUTPUT, "w") as f:
        f.write(text)

    print(f"\n{'=' * 70}")
    print(f"Report written to: {OUTPUT}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
