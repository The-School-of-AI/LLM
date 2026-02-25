"""Download and cache benchmark datasets from the Hugging Face Hub.

Usage:
    python scripts/download_benchmarks.py
    python scripts/download_benchmarks.py --output-dir /data/benchmarks
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

# config can be: None | str | list[str] (list = multiple configs concatenated into one file)
BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

BENCHMARKS = {
    # Original
    "mmlu": ("cais/mmlu", "all", "test"),
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "winogrande": ("winogrande", "winogrande_xl", "validation"),
    "gsm8k": ("gsm8k", "main", "test"),
    "humaneval": ("openai_humaneval", None, "test"),
    "boolq": ("google/boolq", None, "validation"),
    "piqa": (
        "ybisk/piqa",
        None,
        "validation",
    ),  # ybisk/piqa = updated, script-free version
    # Previously added
    "triviaqa": ("trivia_qa", "rc.nocontext", "validation"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, "test"),
    "math": (
        "lighteval/MATH",
        MATH_SUBJECTS,
        "test",
    ),  # hendrycks/competition_math no longer on Hub
    "truthfulqa": ("truthful_qa", "multiple_choice", "validation"),
    "ifeval": ("google/IFEval", None, "train"),
    # New
    "bbh": ("lukaemon/bbh", BBH_TASKS, "test"),
    # apps / indicqa / indicglue removed: use legacy dataset scripts no longer supported by HF
}

# Priority order for extracting the question text from a sample
TEXT_FIELDS = [
    "question",
    "problem",
    "prompt",
    "input",
    "goal",
    "sentence",
    "premise",
    "ctx",
]


def load_configs(dataset_id: str, configs: list[str], split: str):
    """Load multiple dataset configs and yield all items."""
    for cfg in configs:
        ds = load_dataset(dataset_id, cfg, split=split)
        yield from ds


def download_all(output_dir: Path) -> None:
    """Download all benchmarks and write them as JSONL files.

    Args:
        output_dir: Directory where ``*_test.jsonl`` files are written.
            Created automatically if it does not exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    failed: list[tuple[str, str]] = []

    for name, (dataset_id, config, split) in BENCHMARKS.items():
        print(f"\n📥 Downloading {name}...")

        try:
            output_file = output_dir / f"{name}_test.jsonl"
            count = 0

            with open(output_file, "w", encoding="utf-8") as f:
                if isinstance(config, list):
                    items = load_configs(dataset_id, config, split)
                elif config:
                    items = load_dataset(dataset_id, config, split=split)
                else:
                    items = load_dataset(dataset_id, split=split)

                for item in items:
                    text = next(
                        (item[field] for field in TEXT_FIELDS if field in item),
                        str(item),
                    )
                    f.write(json.dumps({"question": text, "source": name}) + "\n")
                    count += 1

            print(f"✅ {name}: {count} samples → {output_file}")

        except Exception as e:
            print(f"❌ {name}: {e}")
            failed.append((name, str(e)))

    print("\n" + "=" * 50)
    if failed:
        print(f"⚠️  {len(failed)} failed:")
        for name, err in failed:
            print(f"   {name}: {err}")
    else:
        print("✅ All benchmarks downloaded successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks"),
        help="Directory to write benchmark JSONL files (default: benchmarks/)",
    )
    args = parser.parse_args()
    download_all(args.output_dir)


if __name__ == "__main__":
    main()
