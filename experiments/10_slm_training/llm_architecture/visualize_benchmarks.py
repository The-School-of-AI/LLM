"""
Benchmark Visualization Script
===============================

Generates visual charts from benchmark JSON results for easy comparison.

Usage:
    python visualize_benchmarks.py
    python visualize_benchmarks.py --output plots/
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

# Style configuration
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "Base (GQA)": "#2ecc71",
    "DeepSeek (MLA)": "#3498db",
    "GSA": "#e74c3c",
    "Full": "#f39c12",
}


def load_benchmark_data(results_dir: Path) -> Dict:
    """Load all benchmark JSON files."""
    data = {"micro": {}, "tiny": {}, "full": {}, "4k8k": {}, "4k_train": {}}

    def classify_variant(filename: str):
        f = filename.lower()
        if "1b_base" in f:
            return "Base (GQA)"
        if "1b_gsa" in f:
            return "GSA"
        if "1b_deepseek" in f:
            return "DeepSeek (MLA)"
        if "1b_full" in f:
            return "Full"
        return None

    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)

            variant = classify_variant(json_file.name)
            if not variant:
                continue

            # Classify by benchmark type
            fname = json_file.name.lower()
            if "4k8k" in fname:
                data["4k8k"][variant] = content
            elif "4k_train" in fname:
                data["4k_train"][variant] = content
            elif "fp16" in fname and variant == "GSA":
                continue  # Skip old fp16 file
            else:
                profile = content.get("profile", "full")
                data[profile][variant] = content

        except Exception as e:
            print(f"Warning: Could not load {json_file.name}: {e}")

    return data


def extract_metrics(benchmark_data: Dict, seq_len: int = 256) -> Dict:
    """Extract key metrics for visualization."""
    metrics = {}

    for variant, data in benchmark_data.items():
        inference = next(
            (x for x in data.get("inference", []) if x["seq_len"] == seq_len), None
        )
        training = next(
            (x for x in data.get("training", []) if x["seq_len"] == seq_len), None
        )

        # If no exact match, take first available
        if not inference and data.get("inference"):
            inference = data["inference"][0]
        if not training and data.get("training"):
            training = data["training"][0]

        metrics[variant] = {
            "params_b": data.get("parameters_billions", 0),
            "inference_tps": inference["tokens_per_sec"] if inference else 0,
            "training_tps": training["tokens_per_sec"] if training else 0,
            "inference_mem": inference["memory_gb"] if inference else 0,
            "training_mem": training["memory_gb"] if training else 0,
            "seq_len": inference["seq_len"] if inference else 0,
        }

    return metrics


def extract_long_context_metrics(benchmark_data: Dict) -> Dict:
    """Extract metrics for 4k and 8k sequences."""
    metrics = {}

    for variant, data in benchmark_data.items():
        inf_4k = next(
            (x for x in data.get("inference", []) if x["seq_len"] == 4096), None
        )
        inf_8k = next(
            (x for x in data.get("inference", []) if x["seq_len"] == 8192), None
        )

        metrics[variant] = {
            "params_b": data.get("parameters_billions", 0),
            "inf_4k_tps": inf_4k["tokens_per_sec"] if inf_4k else 0,
            "inf_8k_tps": inf_8k["tokens_per_sec"] if inf_8k else 0,
            "inf_4k_mem": inf_4k["memory_gb"] if inf_4k else 0,
            "inf_8k_mem": inf_8k["memory_gb"] if inf_8k else 0,
        }

    return metrics


def plot_throughput_comparison(data: Dict, output_dir: Path, profile: str = "full"):
    """Plot inference and training throughput comparison."""
    if not data:
        return

    metrics = extract_metrics(data)
    variants = list(metrics.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Inference throughput
    inference_tps = [metrics[v]["inference_tps"] for v in variants]
    colors = [COLORS.get(v, "#95a5a6") for v in variants]

    bars1 = ax1.bar(
        range(len(variants)), inference_tps, color=colors, alpha=0.8, edgecolor="black"
    )
    ax1.set_xlabel("Architecture", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Tokens/Second", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Inference Throughput ({profile.title()} Profile)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(range(len(variants)))
    ax1.set_xticklabels(variants, rotation=15, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Training throughput
    training_tps = [metrics[v]["training_tps"] for v in variants]
    bars2 = ax2.bar(
        range(len(variants)), training_tps, color=colors, alpha=0.8, edgecolor="black"
    )
    ax2.set_xlabel("Architecture", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Tokens/Second", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"Training Throughput ({profile.title()} Profile)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(range(len(variants)))
    ax2.set_xticklabels(variants, rotation=15, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / f"throughput_comparison_{profile}.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"✓ Saved: throughput_comparison_{profile}.png")
    plt.close()


def plot_memory_usage(data: Dict, output_dir: Path, profile: str = "full"):
    """Plot memory usage comparison."""
    if not data:
        return

    metrics = extract_metrics(data)
    variants = list(metrics.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(variants))
    width = 0.35

    inference_mem = [metrics[v]["inference_mem"] for v in variants]
    training_mem = [metrics[v]["training_mem"] for v in variants]

    bars1 = ax.bar(
        x - width / 2,
        inference_mem,
        width,
        label="Inference",
        color="#3498db",
        alpha=0.8,
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        training_mem,
        width,
        label="Training",
        color="#e74c3c",
        alpha=0.8,
        edgecolor="black",
    )

    ax.set_xlabel("Architecture", fontsize=12, fontweight="bold")
    ax.set_ylabel("Peak Memory (GB)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Memory Usage Comparison ({profile.title()} Profile)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()
    plt.savefig(
        output_dir / f"memory_usage_{profile}.png", dpi=300, bbox_inches="tight"
    )
    print(f"✓ Saved: memory_usage_{profile}.png")
    plt.close()


def plot_efficiency_metrics(data: Dict, output_dir: Path, profile: str = "full"):
    """Plot efficiency metrics (throughput per parameter)."""
    if not data:
        return

    metrics = extract_metrics(data)
    variants = list(metrics.keys())

    # Calculate efficiency: throughput per billion parameters
    efficiency = []
    for v in variants:
        params = metrics[v]["params_b"]
        if params > 0:
            avg_tps = (metrics[v]["inference_tps"] + metrics[v]["training_tps"]) / 2
            efficiency.append(avg_tps / params)
        else:
            efficiency.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS.get(v, "#95a5a6") for v in variants]
    bars = ax.bar(
        range(len(variants)), efficiency, color=colors, alpha=0.8, edgecolor="black"
    )

    ax.set_xlabel("Architecture", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "Avg Tokens/Sec per Billion Parameters", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        f"Parameter Efficiency ({profile.title()} Profile)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / f"efficiency_{profile}.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: efficiency_{profile}.png")
    plt.close()


def plot_summary_dashboard(data_full: Dict, data_tiny: Dict, output_dir: Path):
    """Create a comprehensive dashboard view."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Full-size throughput
    if data_full:
        metrics_full = extract_metrics(data_full)
        variants_full = list(metrics_full.keys())

        ax1 = fig.add_subplot(gs[0, 0])
        inference_tps = [metrics_full[v]["inference_tps"] for v in variants_full]
        colors = [COLORS.get(v, "#95a5a6") for v in variants_full]
        ax1.bar(
            range(len(variants_full)),
            inference_tps,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax1.set_title("Full-Size: Inference Throughput", fontweight="bold", fontsize=11)
        ax1.set_ylabel("Tokens/s", fontsize=10)
        ax1.set_xticks(range(len(variants_full)))
        ax1.set_xticklabels(variants_full, rotation=15, ha="right", fontsize=9)
        ax1.grid(axis="y", alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        training_tps = [metrics_full[v]["training_tps"] for v in variants_full]
        ax2.bar(
            range(len(variants_full)),
            training_tps,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax2.set_title("Full-Size: Training Throughput", fontweight="bold", fontsize=11)
        ax2.set_ylabel("Tokens/s", fontsize=10)
        ax2.set_xticks(range(len(variants_full)))
        ax2.set_xticklabels(variants_full, rotation=15, ha="right", fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

    # Tiny profile throughput
    if data_tiny:
        metrics_tiny = extract_metrics(data_tiny)
        variants_tiny = list(metrics_tiny.keys())

        ax3 = fig.add_subplot(gs[1, 0])
        inference_tps = [metrics_tiny[v]["inference_tps"] for v in variants_tiny]
        colors = [COLORS.get(v, "#95a5a6") for v in variants_tiny]
        ax3.bar(
            range(len(variants_tiny)),
            inference_tps,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax3.set_title(
            "Tiny Profile: Inference Throughput", fontweight="bold", fontsize=11
        )
        ax3.set_ylabel("Tokens/s", fontsize=10)
        ax3.set_xticks(range(len(variants_tiny)))
        ax3.set_xticklabels(variants_tiny, rotation=15, ha="right", fontsize=9)
        ax3.grid(axis="y", alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        training_tps = [metrics_tiny[v]["training_tps"] for v in variants_tiny]
        ax4.bar(
            range(len(variants_tiny)),
            training_tps,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax4.set_title(
            "Tiny Profile: Training Throughput", fontweight="bold", fontsize=11
        )
        ax4.set_ylabel("Tokens/s", fontsize=10)
        ax4.set_xticks(range(len(variants_tiny)))
        ax4.set_xticklabels(variants_tiny, rotation=15, ha="right", fontsize=9)
        ax4.grid(axis="y", alpha=0.3)

    # Memory comparison (full-size)
    if data_full:
        ax5 = fig.add_subplot(gs[2, 0])
        x = np.arange(len(variants_full))
        width = 0.35
        inference_mem = [metrics_full[v]["inference_mem"] for v in variants_full]
        training_mem = [metrics_full[v]["training_mem"] for v in variants_full]
        ax5.bar(
            x - width / 2,
            inference_mem,
            width,
            label="Inference",
            color="#3498db",
            alpha=0.8,
            edgecolor="black",
        )
        ax5.bar(
            x + width / 2,
            training_mem,
            width,
            label="Training",
            color="#e74c3c",
            alpha=0.8,
            edgecolor="black",
        )
        ax5.set_title("Full-Size: Memory Usage", fontweight="bold", fontsize=11)
        ax5.set_ylabel("Peak Memory (GB)", fontsize=10)
        ax5.set_xticks(x)
        ax5.set_xticklabels(variants_full, rotation=15, ha="right", fontsize=9)
        ax5.legend(fontsize=9)
        ax5.grid(axis="y", alpha=0.3)

    # Parameter counts
    ax6 = fig.add_subplot(gs[2, 1])
    if data_full:
        params = [metrics_full[v]["params_b"] for v in variants_full]
        colors = [COLORS.get(v, "#95a5a6") for v in variants_full]
        ax6.bar(
            range(len(variants_full)),
            params,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax6.set_title("Model Size (Full-Size)", fontweight="bold", fontsize=11)
        ax6.set_ylabel("Parameters (Billions)", fontsize=10)
        ax6.set_xticks(range(len(variants_full)))
        ax6.set_xticklabels(variants_full, rotation=15, ha="right", fontsize=9)
        ax6.grid(axis="y", alpha=0.3)
        for i, v in enumerate(params):
            ax6.text(i, v, f"{v:.2f}B", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "LLM Architecture Benchmark Dashboard", fontsize=16, fontweight="bold", y=0.995
    )

    plt.savefig(output_dir / "benchmark_dashboard.png", dpi=300, bbox_inches="tight")
    print("[OK] Saved: benchmark_dashboard.png")
    plt.close()


def plot_long_context_comparison(
    data_4k8k: Dict, data_4k_train: Dict, output_dir: Path
):
    """Plot 4K-8K context comparison charts - the main visualization."""
    if not data_4k8k:
        return

    metrics = extract_long_context_metrics(data_4k8k)
    variants = list(metrics.keys())

    # Get training metrics if available
    train_4k = {}
    for variant, data in data_4k_train.items():
        train_result = next(
            (x for x in data.get("training", []) if x["seq_len"] == 4096), None
        )
        if train_result:
            train_4k[variant] = train_result["tokens_per_sec"]

    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    colors = [COLORS.get(v, "#95a5a6") for v in variants]

    # 1. Inference Throughput at 4K
    ax1 = fig.add_subplot(gs[0, 0])
    inf_4k = [metrics[v]["inf_4k_tps"] for v in variants]
    bars1 = ax1.bar(
        range(len(variants)), inf_4k, color=colors, alpha=0.85, edgecolor="black"
    )
    ax1.set_title("Inference @ 4K Tokens", fontweight="bold", fontsize=13)
    ax1.set_ylabel("Tokens/Second", fontsize=11)
    ax1.set_xticks(range(len(variants)))
    ax1.set_xticklabels(variants, rotation=20, ha="right", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h,
            f"{int(h):,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 2. Inference Throughput at 8K
    ax2 = fig.add_subplot(gs[0, 1])
    inf_8k = [metrics[v]["inf_8k_tps"] for v in variants]
    bars2 = ax2.bar(
        range(len(variants)), inf_8k, color=colors, alpha=0.85, edgecolor="black"
    )
    ax2.set_title("Inference @ 8K Tokens", fontweight="bold", fontsize=13)
    ax2.set_ylabel("Tokens/Second", fontsize=11)
    ax2.set_xticks(range(len(variants)))
    ax2.set_xticklabels(variants, rotation=20, ha="right", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h,
            f"{int(h):,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 3. Training Throughput at 4K
    ax3 = fig.add_subplot(gs[0, 2])
    train_vals = [train_4k.get(v, 0) for v in variants]
    bars3 = ax3.bar(
        range(len(variants)), train_vals, color=colors, alpha=0.85, edgecolor="black"
    )
    ax3.set_title("Training @ 4K Tokens", fontweight="bold", fontsize=13)
    ax3.set_ylabel("Tokens/Second", fontsize=11)
    ax3.set_xticks(range(len(variants)))
    ax3.set_xticklabels(variants, rotation=20, ha="right", fontsize=10)
    ax3.grid(axis="y", alpha=0.3)
    for bar in bars3:
        h = bar.get_height()
        if h > 0:
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                h,
                f"{int(h):,}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    # 4. Memory at 4K vs 8K
    ax4 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(variants))
    width = 0.35
    mem_4k = [metrics[v]["inf_4k_mem"] for v in variants]
    mem_8k = [metrics[v]["inf_8k_mem"] for v in variants]
    ax4.bar(
        x - width / 2,
        mem_4k,
        width,
        label="4K Context",
        color="#3498db",
        alpha=0.8,
        edgecolor="black",
    )
    ax4.bar(
        x + width / 2,
        mem_8k,
        width,
        label="8K Context",
        color="#e74c3c",
        alpha=0.8,
        edgecolor="black",
    )
    ax4.set_title("Memory Usage: 4K vs 8K", fontweight="bold", fontsize=13)
    ax4.set_ylabel("Peak Memory (GB)", fontsize=11)
    ax4.set_xticks(x)
    ax4.set_xticklabels(variants, rotation=20, ha="right", fontsize=10)
    ax4.legend(fontsize=10)
    ax4.grid(axis="y", alpha=0.3)

    # 5. Throughput Scaling (4K to 8K)
    ax5 = fig.add_subplot(gs[1, 1])
    scaling = []
    for v in variants:
        if metrics[v]["inf_4k_tps"] > 0:
            scaling.append(metrics[v]["inf_8k_tps"] / metrics[v]["inf_4k_tps"] * 100)
        else:
            scaling.append(0)
    bars5 = ax5.bar(
        range(len(variants)), scaling, color=colors, alpha=0.85, edgecolor="black"
    )
    ax5.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Ideal (50%)")
    ax5.set_title("Throughput Retention: 4K→8K", fontweight="bold", fontsize=13)
    ax5.set_ylabel("% of 4K Throughput", fontsize=11)
    ax5.set_xticks(range(len(variants)))
    ax5.set_xticklabels(variants, rotation=20, ha="right", fontsize=10)
    ax5.legend(fontsize=10)
    ax5.grid(axis="y", alpha=0.3)
    for bar in bars5:
        h = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            h,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 6. Summary Table as plot
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    table_data = []
    for v in variants:
        table_data.append(
            [
                v,
                f"{metrics[v]['params_b']:.3f}B",
                f"{int(metrics[v]['inf_4k_tps']):,}",
                f"{int(metrics[v]['inf_8k_tps']):,}",
                f"{int(train_4k.get(v, 0)):,}",
            ]
        )

    table = ax6.table(
        cellText=table_data,
        colLabels=["Variant", "Params", "Inf@4K", "Inf@8K", "Train@4K"],
        loc="center",
        cellLoc="center",
        colWidths=[0.25, 0.15, 0.18, 0.18, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color header row
    for j in range(5):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Color variant cells
    for i, v in enumerate(variants):
        table[(i + 1, 0)].set_facecolor(COLORS.get(v, "#95a5a6"))
        table[(i + 1, 0)].set_text_props(fontweight="bold")

    ax6.set_title("Summary Table (Tokens/Sec)", fontweight="bold", fontsize=13, pad=20)

    fig.suptitle(
        "LLM Architecture Benchmark: 4K-8K Context Length Comparison\n(Micro Profile: 256 hidden, 4 layers)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(
        output_dir / "long_context_comparison.png", dpi=300, bbox_inches="tight"
    )
    print("[OK] Saved: long_context_comparison.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument(
        "--results",
        type=str,
        default="results",
        help="Directory containing benchmark JSON files (default: results)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("BENCHMARK VISUALIZATION")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load data
    print("Loading benchmark data...")
    data = load_benchmark_data(results_dir)

    print(f"Found {len(data.get('micro', {}))} micro profile variants")
    print(f"Found {len(data.get('tiny', {}))} tiny profile variants")
    print(f"Found {len(data.get('full', {}))} full-size variants")
    print(f"Found {len(data.get('4k8k', {}))} 4k-8k context variants")
    print(f"Found {len(data.get('4k_train', {}))} 4k training variants")
    print()

    # Generate plots
    print("Generating visualizations...")

    # Long context comparison (4K-8K) - Main visualization
    if data.get("4k8k"):
        plot_long_context_comparison(data["4k8k"], data.get("4k_train", {}), output_dir)

    if data.get("full"):
        plot_throughput_comparison(data["full"], output_dir, "full")
        plot_memory_usage(data["full"], output_dir, "full")
        plot_efficiency_metrics(data["full"], output_dir, "full")

    if data.get("tiny"):
        plot_throughput_comparison(data["tiny"], output_dir, "tiny")
        plot_memory_usage(data["tiny"], output_dir, "tiny")
        plot_efficiency_metrics(data["tiny"], output_dir, "tiny")

    # Dashboard
    plot_summary_dashboard(data.get("full", {}), data.get("tiny", {}), output_dir)

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"All plots saved to: {output_dir.absolute()}")
    print()
    print("Generated plots:")
    for plot_file in sorted(output_dir.glob("*.png")):
        print(f"  - {plot_file.name}")


if __name__ == "__main__":
    main()
