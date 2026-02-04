"""
MMLU (Massive Multitask Language Understanding) Benchmark Analysis

Analyzes all 57 subjects in the MMLU benchmark.
Computes token counts and identifies metrics for this multi-task benchmark.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from metric_finder import MetricFinder
from token_counter import TokenCounter

# All MMLU subjects (57 total across STEM, humanities, social sciences, and other)
MMLU_SUBJECTS = [
    # STEM
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
    # Humanities
    "formal_logic",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "prehistory",
    "professional_law",
    "world_religions",
    # Social Sciences
    "econometrics",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_sexuality",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    # Other
    "business_ethics",
    "clinical_knowledge",
    "college_medicine",
    "global_facts",
    "human_aging",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "nutrition",
    "professional_accounting",
    "professional_medicine",
    "virology",
]


def analyze_mmlu(
    tokenizer_name: str = "Xenova/gpt-4",
    split: str = "test",
    output_dir: Path = None,
    subjects_to_analyze: list = None,
):
    """
    Analyze all MMLU subjects.

    Args:
        tokenizer_name: Tokenizer to use for counting (default: Xenova/gpt-4 using cl100k_base)
        split: Dataset split to analyze - test/validation/dev (default: test)
        output_dir: Directory to save results
        subjects_to_analyze: List of subjects to analyze (defaults to all 57)
    """
    if subjects_to_analyze is None:
        subjects_to_analyze = MMLU_SUBJECTS

    print("\n" + "=" * 80)
    print("MMLU (Massive Multitask Language Understanding) Benchmark Analysis")
    print("=" * 80)
    print(f"\nTotal subjects to analyze: {len(subjects_to_analyze)}")
    print(f"Split: {split}")
    print(f"Tokenizer: {tokenizer_name}\n")

    # Initialize utilities
    counter = TokenCounter(tokenizer_name=tokenizer_name)
    metric_finder = MetricFinder()

    # Get metric information
    metric_info = metric_finder.find_metric("MMLU")
    print(f"Benchmark Metric: {metric_info.get('metric', 'Unknown')}")
    print(f"Description: {metric_info.get('description', 'N/A')}\n")

    results = []
    successful = 0
    failed = 0

    # Analyze each subject
    for i, subject in enumerate(subjects_to_analyze, 1):
        print(f"[{i}/{len(subjects_to_analyze)}] Analyzing: {subject}")

        try:
            stats = counter.count_dataset("cais/mmlu", split=split, name=subject)

            results.append(
                {
                    "subject": subject,
                    "status": "success",
                    "dataset_name": "cais/mmlu",
                    "split": split,
                    "num_samples": stats["num_samples"],
                    "total_tokens": stats["total_tokens"],
                    "mean_tokens": stats["mean_tokens"],
                    "min_tokens": stats["min_tokens"],
                    "max_tokens": stats["max_tokens"],
                    "tokenizer": tokenizer_name,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            successful += 1
            print(
                f"  ✓ {stats['num_samples']:,} samples | {stats['total_tokens']:,} tokens | "
                f"avg: {stats['mean_tokens']:.1f}\n"
            )

        except Exception as e:
            results.append(
                {
                    "subject": subject,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            failed += 1
            print(f"  ✗ Error: {e}\n")

    # Calculate aggregates
    total_samples = sum(
        r.get("num_samples", 0) for r in results if r["status"] == "success"
    )
    total_tokens = sum(
        r.get("total_tokens", 0) for r in results if r["status"] == "success"
    )
    avg_tokens = total_tokens / total_samples if total_samples > 0 else 0

    # Create summary
    summary = {
        "benchmark": "MMLU",
        "dataset_name": "cais/mmlu",
        "split": split,
        "total_subjects": len(subjects_to_analyze),
        "successful_subjects": successful,
        "failed_subjects": failed,
        "aggregate_stats": {
            "total_samples": total_samples,
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": round(avg_tokens, 2),
        },
        "metric": metric_info.get("metric", "Unknown"),
        "metric_description": metric_info.get("description", "N/A"),
        "paper": metric_info.get("paper", "N/A"),
        "tokenizer": tokenizer_name,
        "timestamp": datetime.now().isoformat(),
        "per_subject_results": results,
    }

    # Save results
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "mmlu_analysis.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Benchmark: MMLU (Massive Multitask Language Understanding)")
    print(f"Split: {split}")
    print(f"Total subjects: {len(subjects_to_analyze)}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print("\nAggregate Statistics:")
    print(f"  Total samples ({split}): {total_samples:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens/sample: {avg_tokens:.2f}")
    print(f"\nEvaluation Metric: {metric_info.get('metric', 'Unknown')}")
    print(f"\nResults saved to: {output_file}")
    print("=" * 80 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MMLU benchmark - all 57 subjects across STEM, humanities, social sciences"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Xenova/gpt-4",
        help="Tokenizer to use (default: Xenova/gpt-4 using cl100k_base for FLOPS)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation", "dev"],
        help="Dataset split to analyze (default: test). Useful for data/training teams.",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument(
        "--subject",
        type=str,
        help="Analyze only a specific subject (e.g., 'abstract_algebra')",
    )
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="List all available MMLU subjects and exit",
    )

    args = parser.parse_args()

    # List subjects if requested
    if args.list_subjects:
        print("\nAvailable MMLU Subjects (57 total):\n")
        print("STEM:")
        for subject in MMLU_SUBJECTS[:19]:
            print(f"  - {subject}")
        print("\nHumanities:")
        for subject in MMLU_SUBJECTS[19:32]:
            print(f"  - {subject}")
        print("\nSocial Sciences:")
        for subject in MMLU_SUBJECTS[32:44]:
            print(f"  - {subject}")
        print("\nOther:")
        for subject in MMLU_SUBJECTS[44:]:
            print(f"  - {subject}")
        print()
        return

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.subject:
        # Analyze single subject
        if args.subject not in MMLU_SUBJECTS:
            print(f"Error: Subject '{args.subject}' not found.")
            print("Use --list-subjects to see all available subjects.")
            print(f"Example subjects: {', '.join(MMLU_SUBJECTS[:5])}...")
            return

        # Pass only the selected subject
        analyze_mmlu(
            tokenizer_name=args.tokenizer,
            split=args.split,
            output_dir=output_dir,
            subjects_to_analyze=[args.subject],
        )
    else:
        # Analyze all subjects
        analyze_mmlu(
            tokenizer_name=args.tokenizer, split=args.split, output_dir=output_dir
        )


if __name__ == "__main__":
    main()
