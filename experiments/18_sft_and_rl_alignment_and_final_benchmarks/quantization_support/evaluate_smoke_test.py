#!/usr/bin/env python3
"""
Smoke Test Evaluation — Phase 3
Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

Evaluates trained checkpoints on benchmarks using lm-evaluation-harness.
Compares Standard SFT vs IDFT vs base model.

Usage:
    python evaluate_smoke_test.py \
        --checkpoint_dir ./outputs/idft_smoke_test/sft_best \
        --output_json results_sft.json

    python evaluate_smoke_test.py \
        --checkpoint_dir ./outputs/idft_smoke_test/idft_best \
        --output_json results_idft.json
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Benchmark configuration matching the smoke test plan
BENCHMARKS = {
    "gsm8k": {
        "task": "gsm8k",
        "category": "math",
        "num_fewshot": 5,
        "description": "Math reasoning (full)",
    },
    "math_500": {
        "task": "minerva_math",
        "category": "math",
        "num_fewshot": 4,
        "description": "Hard math (MATH-500)",
    },
    "humaneval": {
        "task": "humaneval",
        "category": "code",
        "num_fewshot": 0,
        "description": "Code generation",
    },
    "mmlu_stem": {
        "task": "mmlu_stem",
        "category": "general",
        "num_fewshot": 5,
        "description": "General knowledge (STEM)",
    },
    "truthfulqa": {
        "task": "truthfulqa_mc2",
        "category": "safety",
        "num_fewshot": 0,
        "description": "Safety/factuality",
    },
}

# Eval settings from the plan
EVAL_SETTINGS = {
    "temperature": 0.3,
    "num_runs_small": 8,  # benchmarks < 1000 samples
    "num_runs_large": 2,  # benchmarks > 1000 samples
    "max_gen_tokens": 2048,
}


def run_lm_eval(
    model_path: str,
    task: str,
    num_fewshot: int = 0,
    batch_size: str = "auto",
    output_path: Optional[str] = None,
    use_peft: bool = False,
    base_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run lm-evaluation-harness on a model checkpoint.

    Args:
        model_path: Path to model or HF model name.
        task: Benchmark task name.
        num_fewshot: Number of few-shot examples.
        batch_size: Batch size for evaluation.
        output_path: Path to save results JSON.
        use_peft: Whether model_path is a PEFT adapter.
        base_model: Base model name (required if use_peft=True).

    Returns:
        Dict with evaluation results.
    """
    cmd = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_path}",
        "--tasks",
        task,
        "--num_fewshot",
        str(num_fewshot),
        "--batch_size",
        batch_size,
    ]

    if use_peft and base_model:
        cmd[4] = f"pretrained={base_model},peft={model_path}"

    if output_path:
        cmd.extend(["--output_path", output_path])

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error(f"lm_eval failed: {result.stderr}")
            return {"error": result.stderr, "task": task}

        # Parse output
        if output_path and Path(output_path).exists():
            with open(output_path) as f:
                return json.load(f)
        return {"stdout": result.stdout, "task": task}

    except subprocess.TimeoutExpired:
        logger.error(f"Evaluation timed out for task {task}")
        return {"error": "timeout", "task": task}
    except FileNotFoundError:
        logger.error("lm_eval not found. Install with: pip install lm-eval")
        return {"error": "lm_eval not installed", "task": task}


def evaluate_checkpoint(
    checkpoint_path: str,
    label: str,
    benchmarks: Optional[List[str]] = None,
    use_peft: bool = False,
    base_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a checkpoint on all smoke test benchmarks.

    Args:
        checkpoint_path: Path to model checkpoint.
        label: Label for this condition (e.g., "sft", "idft", "base").
        benchmarks: List of benchmark names to run. None = all.
        use_peft: Whether checkpoint is a PEFT adapter.
        base_model: Base model name (required if use_peft=True).

    Returns:
        Dict with all benchmark results.
    """
    if benchmarks is None:
        benchmarks = list(BENCHMARKS.keys())

    results = {"label": label, "checkpoint": checkpoint_path, "benchmarks": {}}

    for name in benchmarks:
        if name not in BENCHMARKS:
            logger.warning(f"Unknown benchmark: {name}, skipping")
            continue

        bench = BENCHMARKS[name]
        logger.info(f"Evaluating {label} on {name} ({bench['description']})...")

        output_path = f"eval_results_{label}_{name}.json"
        result = run_lm_eval(
            model_path=checkpoint_path,
            task=bench["task"],
            num_fewshot=bench["num_fewshot"],
            output_path=output_path,
            use_peft=use_peft,
            base_model=base_model,
        )

        results["benchmarks"][name] = {
            "category": bench["category"],
            "description": bench["description"],
            "raw_results": result,
        }

    return results


def compute_aggregate_scores(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute Math-Avg and General-Avg from benchmark results.

    Returns:
        Dict with aggregate scores.
    """
    math_scores = []
    general_scores = []

    for name, bench_result in results.get("benchmarks", {}).items():
        raw = bench_result.get("raw_results", {})
        # Try to extract accuracy from lm-eval output format
        score = None
        if "results" in raw:
            task_results = raw["results"]
            for task_name, task_data in task_results.items():
                if "acc" in task_data:
                    score = task_data["acc"] * 100
                elif "acc_norm" in task_data:
                    score = task_data["acc_norm"] * 100

        if score is not None:
            if bench_result["category"] == "math":
                math_scores.append(score)
            else:
                general_scores.append(score)

    aggregates = {}
    if math_scores:
        aggregates["math_avg"] = sum(math_scores) / len(math_scores)
    if general_scores:
        aggregates["general_avg"] = sum(general_scores) / len(general_scores)

    return aggregates


def compare_conditions(
    sft_results: Dict[str, Any],
    idft_results: Dict[str, Any],
    base_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Apply the decision framework from the smoke test plan.

    Returns:
        Dict with comparison, deltas, and recommendation.
    """
    sft_agg = compute_aggregate_scores(sft_results)
    idft_agg = compute_aggregate_scores(idft_results)

    comparison = {
        "sft_scores": sft_agg,
        "idft_scores": idft_agg,
        "deltas": {},
        "recommendation": "",
        "outcome": "",
    }

    if base_results:
        comparison["base_scores"] = compute_aggregate_scores(base_results)

    # Compute deltas (IDFT - SFT)
    math_delta = idft_agg.get("math_avg", 0) - sft_agg.get("math_avg", 0)
    general_delta = idft_agg.get("general_avg", 0) - sft_agg.get("general_avg", 0)
    comparison["deltas"] = {
        "math_avg_delta": math_delta,
        "general_avg_delta": general_delta,
    }

    # Decision framework
    if math_delta >= 2.0 and general_delta >= -0.5:
        comparison["outcome"] = "strong_positive"
        comparison["recommendation"] = (
            "ADOPT: IDFT beats SFT by >= 2% on Math-Avg with <= 0.5% "
            "General-Avg regression. Integrate into full SFT pipeline."
        )
    elif math_delta >= 1.0 and general_delta >= 0.0:
        comparison["outcome"] = "moderate_positive"
        comparison["recommendation"] = (
            "ADOPT: IDFT beats SFT by >= 1% on Math-Avg with no "
            "General-Avg regression. Integrate into full SFT pipeline."
        )
    elif math_delta > 0 and general_delta < 0:
        comparison["outcome"] = "mixed"
        comparison["recommendation"] = (
            "INVESTIGATE: IDFT shows mixed results. Consider per-dataset "
            "IDFT or adjusting clip_B."
        )
    else:
        comparison["outcome"] = "negative"
        comparison["recommendation"] = (
            "DO NOT ADOPT: IDFT does not outperform standard SFT. "
            "Stick with standard SFT loss."
        )

    return comparison


def print_results_table(
    sft_results: Dict[str, Any],
    idft_results: Dict[str, Any],
    base_results: Optional[Dict[str, Any]] = None,
):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("IDFT SMOKE TEST RESULTS")
    print("=" * 80)

    header = f"{'Benchmark':<20} {'Category':<12} {'Base':>8} {'SFT':>8} {'IDFT':>8} {'Delta':>8}"
    print(header)
    print("-" * 80)

    all_benchmarks = set(
        list(sft_results.get("benchmarks", {}).keys())
        + list(idft_results.get("benchmarks", {}).keys())
    )

    for name in sorted(all_benchmarks):
        category = ""
        base_score = "-"
        sft_score = "-"
        idft_score = "-"
        delta = "-"

        if name in sft_results.get("benchmarks", {}):
            category = sft_results["benchmarks"][name].get("category", "")
        if name in idft_results.get("benchmarks", {}):
            category = idft_results["benchmarks"][name].get("category", "")

        # Extract scores (placeholder - actual extraction depends on lm-eval output)
        print(
            f"{name:<20} {category:<12} {base_score:>8} {sft_score:>8} {idft_score:>8} {delta:>8}"
        )

    print("=" * 80)

    comparison = compare_conditions(sft_results, idft_results, base_results)
    print(f"\nOutcome: {comparison['outcome'].upper()}")
    print(f"Recommendation: {comparison['recommendation']}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="IDFT Smoke Test Evaluation")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--label", type=str, required=True, help="Condition label (sft, idft, base)"
    )
    parser.add_argument(
        "--output_json", type=str, required=True, help="Path to save results JSON"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="Benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--use_peft",
        action="store_true",
        help="Checkpoint is a PEFT adapter (requires --base_model)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name for PEFT adapter loading",
    )
    args = parser.parse_args()

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint_dir,
        label=args.label,
        benchmarks=args.benchmarks,
        use_peft=args.use_peft,
        base_model=args.base_model,
    )

    # Save results
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
