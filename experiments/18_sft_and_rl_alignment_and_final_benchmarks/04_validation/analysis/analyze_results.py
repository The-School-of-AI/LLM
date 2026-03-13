"""
Results Analysis & Report Generator
======================================
Takes evaluation result JSON files (output of run_evaluation.py) and
produces:
  1. A rich console report with metrics tables
  2. A Markdown report saved to ../results/report_{timestamp}.md
  3. A per-prompt CSV for human review / annotation workflow
  4. Matplotlib charts (optional, if matplotlib is installed)

Usage:
    python analyze_results.py --results ../results/evaluation_results_*.json
    python analyze_results.py --results ../results/evaluation_results_20260305_120000.json --charts
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_results(result_paths: list[str]) -> list[dict]:
    """Load and merge multiple result JSON files."""
    all_results = []
    for path in result_paths:
        p = Path(path)
        if not p.exists():
            print(f"[WARN] File not found: {path}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_results.extend(data)
        elif isinstance(data, dict) and "results" in data:
            all_results.extend(data["results"])
        else:
            print(f"[WARN] Unexpected format in {path}")
    print(f"Loaded {len(all_results)} results total.")
    return all_results


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_markdown_report(summary: dict, results: list[dict], output_path: str) -> None:
    """Write a structured Markdown validation report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict = "**PASS ✅**" if summary["validation_pass"] else "**FAIL ❌**"
    if_ok = "✅" if summary["validation_criteria"]["if_improved"] else "❌"
    hall_ok = "✅" if summary["validation_criteria"]["no_new_hallucinations"] else "❌"

    lines = [
        "# SFT Validation Report",
        f"*Generated: {timestamp}*",
        "",
        "---",
        "",
        f"## Overall Verdict: {verdict}",
        "",
        f"| Criterion | Status |",
        f"|-----------|--------|",
        f"| Instruction-following rate improved (SFT ≥ Base) | {if_ok} |",
        f"| No new hallucination patterns introduced | {hall_ok} |",
        "",
        "---",
        "",
        "## 1. Evaluation Overview",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total prompts evaluated | {summary['total_prompts']} |",
        f"| Evaluation date | {timestamp.split()[0]} |",
        "",
        "---",
        "",
        "## 2. Instruction-Following Metrics",
        "",
        "| Metric | Base Model | SFT Model | Delta |",
        "|--------|-----------|-----------|-------|",
        f"| IF Rate (% prompts followed) | {summary['base_if_rate']:.1%} | {summary['sft_if_rate']:.1%} | {summary['if_improvement_pp']:+.2f} pp |",
        f"| Average IF Score | {summary['if_score_avg_base']:.4f} | {summary['if_score_avg_sft']:.4f} | {summary['if_score_delta']:+.4f} |",
        "",
        "### Instruction-Following by Category (SFT)",
        "",
        "| Category | SFT IF Rate | n |",
        "|----------|------------|---|",
    ]

    for cat, data in summary.get("per_category_if_rate", {}).items():
        lines.append(f"| {cat} | {data['sft_if_rate']:.1%} | {data['n']} |")

    lines += [
        "",
        "### Instruction-Following by Difficulty (SFT)",
        "",
        "| Difficulty | SFT IF Rate | n |",
        "|------------|------------|---|",
    ]
    for diff, data in summary.get("per_difficulty_if_rate", {}).items():
        lines.append(f"| {diff} | {data['sft_if_rate']:.1%} | {data['n']} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Hallucination Metrics",
        "",
        "| Metric | Base Model | SFT Model | Delta |",
        "|--------|-----------|-----------|-------|",
        f"| Hallucination Detection Rate | {summary['base_hallucination_rate']:.1%} | {summary['sft_hallucination_rate']:.1%} | {summary['sft_hallucination_rate'] - summary['base_hallucination_rate']:+.1%} |",
        f"| Avg Risk Score Delta (SFT - Base) | — | — | {summary['hallucination_delta']:+.4f} |",
        f"| New hallucination prompts (SFT introduced, base clean) | — | — | {summary['new_hallucination_count']} |",
        "",
    ]

    if summary.get("new_hallucination_prompts"):
        lines.append("#### Prompts with New Hallucinations (SFT introduced)")
        lines.append("")
        for pid in summary["new_hallucination_prompts"]:
            lines.append(f"- `{pid}`")
        lines.append("")

    lines += [
        "---",
        "",
        "## 4. Regression Analysis",
        "",
        "Prompts where SFT performed **worse** than base by > 10 pp:",
        "",
    ]

    regressions = summary.get("regression_prompts", [])
    if regressions:
        lines.append("| Prompt ID | IF Score Delta |")
        lines.append("|-----------|---------------|")
        for r in regressions:
            lines.append(f"| `{r['prompt_id']}` | {r['if_delta']:+.4f} |")
    else:
        lines.append("*No significant regressions detected.* ✅")

    lines += [
        "",
        "---",
        "",
        "## 5. Most Improved Prompts (SFT vs Base)",
        "",
        "| Prompt ID | IF Score Delta |",
        "|-----------|---------------|",
    ]
    for r in summary.get("most_improved_prompts", [])[:10]:
        lines.append(f"| `{r['prompt_id']}` | {r['if_delta']:+.4f} |")

    lines += [
        "",
        "---",
        "",
        "## 6. Worst Performing Prompts (SFT)",
        "",
        "| Prompt ID | SFT IF Score |",
        "|-----------|-------------|",
    ]
    for r in summary.get("worst_sft_prompts", []):
        lines.append(f"| `{r['prompt_id']}` | {r['sft_if_score']:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## 7. Sample Prompt-Level Results",
        "",
        "| Prompt ID | Category | Base IF | SFT IF | IF Δ | Base Hall | SFT Hall |",
        "|-----------|----------|---------|--------|------|-----------|----------|",
    ]
    for r in results[:20]:  # show first 20
        base_if = r["base"]["instruction_following"]["score"]
        sft_if = r["sft"]["instruction_following"]["score"]
        delta = r["delta"]["if_score_change"]
        base_h = "🔴" if r["base"]["hallucination"]["detected"] else "🟢"
        sft_h = "🔴" if r["sft"]["hallucination"]["detected"] else "🟢"
        lines.append(
            f"| `{r['prompt_id']}` | {r.get('category','')} | {base_if:.2f} | {sft_if:.2f} | {delta:+.2f} | {base_h} | {sft_h} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 8. Recommendations",
        "",
        "Based on this evaluation:",
        "",
    ]

    # Auto-generate recommendations
    recs = []
    if summary["if_improvement"] < 0.05:
        recs.append("⚠️  IF improvement is marginal (<5 pp). Consider targeted SFT on instruction-constraint prompt types.")
    if summary["new_hallucination_count"] > 0:
        recs.append(f"🚨  {summary['new_hallucination_count']} prompts showed **new hallucinations** in SFT. Investigate these prompts and consider DPO/RLHF for factuality.")
    if summary.get("per_difficulty_if_rate", {}).get("hard", {}).get("sft_if_rate", 1.0) < 0.5:
        recs.append("⚠️  Hard-difficulty prompts still have low IF rate. Increase hard-prompt proportion in SFT data.")
    if regressions:
        recs.append(f"⚠️  {len(regressions)} prompts regressed. Review these in the CSV report and add them to the SFT training set.")
    if not recs:
        recs.append("✅ No critical issues detected. SFT validation criteria met.")

    for rec in recs:
        lines.append(f"- {rec}")

    lines += ["", "---", "", "*End of SFT Validation Report*"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown report saved to {output_path}")


# ---------------------------------------------------------------------------
# Optional charts
# ---------------------------------------------------------------------------

def generate_charts(summary: dict, output_dir: str) -> None:
    """Generate comparison bar charts using matplotlib."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        print("[WARN] matplotlib/numpy not installed. Skipping charts. Install with: pip install matplotlib numpy")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Chart 1: Base vs SFT IF rates by category
    cats = list(summary["per_category_if_rate"].keys())
    sft_rates = [summary["per_category_if_rate"][c]["sft_if_rate"] for c in cats]

    # We don't store per-category base in summary — use overall for comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("SFT Validation Results", fontsize=14, fontweight="bold")

    # Bar 1: IF rate by category
    x = np.arange(len(cats))
    axes[0].bar(x, sft_rates, color="steelblue", alpha=0.8, label="SFT IF Rate")
    axes[0].axhline(summary["base_if_rate"], color="orange", linestyle="--", label=f"Base overall ({summary['base_if_rate']:.1%})")
    axes[0].axhline(summary["sft_if_rate"], color="green", linestyle="--", label=f"SFT overall ({summary['sft_if_rate']:.1%})")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cats, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Instruction-Following Rate")
    axes[0].set_title("IF Rate by Category (SFT)")
    axes[0].legend(fontsize=8)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Bar 2: Overall Base vs SFT comparison
    metrics_labels = ["IF Rate", "Avg IF Score"]
    base_vals = [summary["base_if_rate"], summary["if_score_avg_base"]]
    sft_vals = [summary["sft_if_rate"], summary["if_score_avg_sft"]]
    x2 = np.arange(len(metrics_labels))
    w = 0.35
    axes[1].bar(x2 - w / 2, base_vals, w, label="Base", color="salmon", alpha=0.85)
    axes[1].bar(x2 + w / 2, sft_vals, w, label="SFT", color="steelblue", alpha=0.85)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(metrics_labels)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Base vs SFT — Key Metrics")
    axes[1].legend()
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    chart_path = str(out / "comparison_charts.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Charts saved to {chart_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SFT Validation Results Analyser")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Path(s) to evaluation_results_*.json file(s).",
    )
    parser.add_argument(
        "--output-dir",
        default="../results",
        help="Directory for report and CSV output.",
    )
    parser.add_argument(
        "--charts",
        action="store_true",
        help="Generate matplotlib comparison charts.",
    )
    args = parser.parse_args()

    # --- Load data ---
    results = load_results(args.results)
    if not results:
        print("No results loaded. Exiting.")
        sys.exit(1)

    # --- Compute metrics ---
    collector = MetricsCollector()
    collector.add_results(results)
    summary = collector.summary()

    # --- Console report ---
    collector.print_summary_table()

    # --- Markdown report ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    md_path = str(output_dir / f"report_{timestamp}.md")
    generate_markdown_report(summary, results, md_path)

    # --- Per-prompt CSV ---
    csv_path = str(output_dir / f"per_prompt_review_{timestamp}.csv")
    collector.to_csv(csv_path)
    print(f"Per-prompt CSV saved to {csv_path}")

    # --- Summary JSON ---
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to {summary_path}")

    # --- Charts (optional) ---
    if args.charts:
        generate_charts(summary, str(output_dir / "charts"))


if __name__ == "__main__":
    main()
