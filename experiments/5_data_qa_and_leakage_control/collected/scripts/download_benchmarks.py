"""Download and cache benchmark datasets from the Hugging Face Hub.

Downloads TEST splits for contamination checking and DEV/FEW-SHOT splits
where benchmarks use them during evaluation (e.g. MMLU 5-shot dev, BBH 3-shot).

Usage:
    python download_benchmarks.py
    python download_benchmarks.py --output-dir /data/benchmarks

Fixes applied vs previous version:
  - TriviaQA / large datasets: use streaming=True to avoid disk exhaustion
  - leval: use trust_remote_code=True (legacy dataset scripts)
  - piqa: removed — HF has fully deprecated dataset scripts; trust_remote_code no longer works
  - apps: parquet bypass via load_dataset("parquet", data_files=hf://...) skips legacy apps.py
  - mmlu_indic: explicit config list required (sarvamai/mmlu-indic has 21 language/script configs)
  - MATH: switched to EleutherAI/hendrycks_math (lighteval/MATH was removed)
  - ToolBench: switched to tuandunghcmut/toolbench-v1 (ToolBench/ToolBench returns 404)
  - aime_2026_II: added to registry; fails gracefully until MathArena publishes it
  - sarvamai/ARC-Challenge-IN: not on Hub or GitHub, skipped with a clear note
  - IndicMTEval: fetched directly from GitHub raw JSONL files (not on HF Hub)
  - HF_TOKEN: set env var HF_TOKEN to avoid rate limits and access gated datasets

Special requirements:
  - GPQA Diamond: accept gated terms at https://huggingface.co/datasets/Idavidrein/gpqa
    then: export HF_TOKEN=<your_token>
  - RULER: synthetic generator, no static dataset — must generate locally:
    git clone https://github.com/NVIDIA/RULER && cd RULER/scripts/data/synthetic/json
    python download_paulgraham_essay.py && bash download_qa_dataset.sh
    cd ../.. && python prepare.py
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Sub-task / config lists
# ---------------------------------------------------------------------------

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

LEVAL_SUBTASKS = [
    "coursera",
    "gsm100",
    "quality",
    "topic_retrieval_longchat",
    "tpo",
    "financial_qa",
    "gov_report_summ",
    "legal_contract_qa",
    "meeting_summ",
    "multidoc_qa",
    "narrative_qa",
    "natural_question",
    "news_summ",
    "paper_assistant",
    "patent_summ",
    "review_summ",
    "scientific_qa",
    "tv_show_summ",
]

INDICQA_LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

MMLU_INDIC_CONFIGS = [
    "bn",
    "bn_roman",
    "en",
    "gu",
    "gu_roman",
    "hi",
    "hi_roman",
    "kn",
    "kn_roman",
    "ml",
    "ml_roman",
    "mr",
    "mr_roman",
    "or",
    "or_roman",
    "pa",
    "pa_roman",
    "ta",
    "ta_roman",
    "te",
    "te_roman",
]

INDICGLUE_TASKS = [
    # Verified against ai4bharat/indic_glue available configs (Feb 2026)
    # Configs not present (removed from dataset): wnli.{or,pa,ta,te,as,bn,kn,ml},
    #   copa.{or,pa,ta,te}, sna.{hi,mr,as}, inltkh.{hi,or,pa,as,bn,kn}
    "wnli.hi",
    "wnli.mr",
    "wnli.gu",
    "copa.hi",
    "copa.mr",
    "copa.gu",
    "sna.bn",
    "csqa.as",
    "csqa.bn",
    "csqa.gu",
    "csqa.hi",
    "csqa.kn",
    "csqa.ml",
    "csqa.mr",
    "csqa.or",
    "csqa.pa",
    "csqa.ta",
    "csqa.te",
    "wstp.as",
    "wstp.bn",
    "wstp.gu",
    "wstp.hi",
    "wstp.kn",
    "wstp.ml",
    "wstp.mr",
    "wstp.or",
    "wstp.pa",
    "wstp.ta",
    "wstp.te",
    "inltkh.gu",
    "inltkh.ml",
    "inltkh.mr",
    "inltkh.ta",
    "inltkh.te",
    "bbca.hi",
    "iitp-mr.hi",
    "iitp-pr.hi",
    "actsa-sc.te",
    "md.hi",
    "wiki-ner.as",
    "wiki-ner.bn",
    "wiki-ner.gu",
    "wiki-ner.hi",
    "wiki-ner.kn",
    "wiki-ner.ml",
    "wiki-ner.mr",
    "wiki-ner.or",
    "wiki-ner.pa",
    "wiki-ner.ta",
    "wiki-ner.te",
]

INDICGENBENCH_REPOS = [
    # Each entry: (repo, indic_field, english_field)
    # flores_in / xorqa_in / xquad_in: examples = {source: Indic, target: English}
    # crosssum_in:                      examples = {summary: Indic, text: English}
    # xorqa_in / xquad_in: broken on HF Hub (generator error) — skipped gracefully
    ("google/IndicGenBench_flores_in", "source", "target"),
    ("google/IndicGenBench_xorqa_in", "source", "target"),
    ("google/IndicGenBench_xquad_in", "source", "target"),
    ("google/IndicGenBench_crosssum_in", "summary", "text"),
]

TOOLBENCH_CONFIGS = [
    "g1_instruction",
    "g1_category",
    "g1_tool",
    "g2_instruction",
    "g2_category",
    "g3_instruction",
]

# ---------------------------------------------------------------------------
# Benchmark registry
#
# Each entry: (dataset_id, config, split, output_tag, streaming, trust_remote_code)
#
# streaming=True        → avoids downloading entire corpus to HF disk cache
# trust_remote_code=True → required for datasets with a legacy dataset.py script
# ---------------------------------------------------------------------------

BENCHMARKS: dict[str, list[tuple]] = {
    # ── Legacy benchmarks kept from original script ────────────────────────
    "hellaswag": [("Rowan/hellaswag", None, "validation", "test", False, False)],
    "winogrande": [("winogrande", "winogrande_xl", "validation", "test", False, False)],
    "boolq": [("google/boolq", None, "validation", "test", False, False)],
    # piqa → ybisk/piqa contains piqa.py, a legacy dataset script; HF no longer supports
    #         these even with trust_remote_code=True. Skipped — see NOT downloadable note.
    # ── Core English benchmarks ────────────────────────────────────────────
    "mmlu": [
        ("cais/mmlu", "all", "test", "test", False, False),  # 14,042
        ("cais/mmlu", "all", "dev", "dev", False, False),  # 285  ← 5-shot prompts
        ("cais/mmlu", "all", "validation", "val", False, False),  # ~1,531
    ],
    "mmlu_pro": [
        ("TIGER-Lab/MMLU-Pro", None, "test", "test", False, False),  # 12,032
        ("TIGER-Lab/MMLU-Pro", None, "validation", "dev", False, False),  # 70
    ],
    "triviaqa": [
        # streaming=True: rc.nocontext avoids downloading 8 GB of full passage corpus
        (
            "mandarjoshi/trivia_qa",
            "rc.nocontext",
            "test",
            "test",
            True,
            False,
        ),  # 17,210
        (
            "mandarjoshi/trivia_qa",
            "rc.nocontext",
            "validation",
            "dev",
            True,
            False,
        ),  # 18,669
    ],
    "gpqa_diamond": [
        # GATED — export HF_TOKEN=<token> after accepting terms at:
        # https://huggingface.co/datasets/Idavidrein/gpqa
        ("Idavidrein/gpqa", "gpqa_diamond", "train", "test", False, False),  # 198
    ],
    "gsm8k": [
        ("openai/gsm8k", "main", "test", "test", False, False),  # 1,319
        ("openai/gsm8k", "main", "train", "dev", False, False),  # 7,473
    ],
    "bbh": [
        ("lukaemon/bbh", BBH_TASKS, "test", "test", False, False),  # 6,511 total
    ],
    "arc_challenge": [
        ("allenai/ai2_arc", "ARC-Challenge", "test", "test", False, False),  # 1,172
        ("allenai/ai2_arc", "ARC-Challenge", "validation", "dev", False, False),  # 299
    ],
    "math": [
        # lighteval/MATH was removed from HF Hub — correct path is EleutherAI/hendrycks_math
        # trust_remote_code=True: hendrycks_math.py legacy script
        (
            "EleutherAI/hendrycks_math",
            MATH_SUBJECTS,
            "test",
            "test",
            False,
            True,
        ),  # 5,000
        (
            "EleutherAI/hendrycks_math",
            MATH_SUBJECTS,
            "train",
            "dev",
            False,
            True,
        ),  # 7,500
    ],
    "ifeval": [
        ("google/IFEval", None, "train", "test", False, False),  # 541 (only one split)
    ],
    "simpleqa_verified": [
        (
            "google/simpleqa-verified",
            "simpleqa_verified",
            "eval",
            "test",
            False,
            False,
        ),  # 1,000
    ],
    "humaneval": [
        ("openai/openai_humaneval", None, "test", "test", False, False),  # 164
    ],
    # apps → codeparrot/apps contains apps.py, a legacy dataset script; HF no longer supports
    #        these even with trust_remote_code=True. Skipped — see NOT downloadable note.
    "aime_2025": [
        ("MathArena/aime_2025", None, "train", "test", False, False),  # 30 problems
    ],
    "aime_2026_I": [
        ("MathArena/aime_2026_I", None, "train", "test", False, False),  # 15 problems
    ],
    "aime_2026_II": [
        # Not yet on HF Hub as of Feb 2026 — attempt anyway, fails gracefully if unpublished
        # Monitor: https://huggingface.co/MathArena
        (
            "MathArena/aime_2026_II",
            None,
            "train",
            "test",
            False,
            False,
        ),  # 15 problems (when published)
    ],
    "truthfulqa": [
        ("truthful_qa", "multiple_choice", "validation", "test", False, False),  # 817
    ],
    "swe_bench_verified": [
        ("princeton-nlp/SWE-bench_Verified", None, "test", "test", False, False),  # 500
    ],
    "toolbench": [
        # ToolBench/ToolBench returns 404 — correct mirror is tuandunghcmut/toolbench-v1
        # 6 benchmark eval configs loaded as separate splits
        (
            "tuandunghcmut/toolbench-v1",
            TOOLBENCH_CONFIGS,
            "train",
            "test",
            False,
            False,
        ),
    ],
    # ── Long-context ───────────────────────────────────────────────────────
    "leval": [
        # trust_remote_code=True: LEval.py legacy script still present
        ("L4NLP/LEval", LEVAL_SUBTASKS, "test", "test", False, True),  # 2,043 total
    ],
    # ── Indic benchmarks ───────────────────────────────────────────────────
    "mmlu_indic": [
        # streaming=True: large translated corpus
        # Config must be explicit — dataset has 21 language/script configs
        (
            "sarvamai/mmlu-indic",
            MMLU_INDIC_CONFIGS,
            "test",
            "test",
            True,
            False,
        ),  # ~14,042
    ],
    # arc_c_in → sarvamai/ARC-Challenge-IN not found on HF Hub as of Feb 2026
    # Monitor: https://huggingface.co/sarvamai
    "indicglue": [
        # streaming=True: 50+ task-language configs, large total
        ("ai4bharat/indic_glue", INDICGLUE_TASKS, "test", "test", True, False),
    ],
    "indicqa": [
        # streaming=True: 11 languages
        (
            "ai4bharat/IndicQA",
            [f"indicqa.{lang}" for lang in INDICQA_LANGS],
            "test",
            "test",
            True,
            False,
        ),  # ~9,571
    ],
    # indicgenbench handled separately below (each subtask is its own HF repo path)
}

# Priority order for extracting question text from a sample
TEXT_FIELDS = [
    "question",
    "problem",
    "prompt",
    "input",
    "goal",
    "sentence",
    "premise",
    "ctx",
    "text",  # IndicGLUE (bbca, sna, inltkh tasks)
    "sectionText",  # IndicGLUE WSTP (word/title selection) tasks
    "tokens",  # IndicGLUE wiki-ner (list of word strings → joined)
    "instruction",  # ToolBench
    "conversations",  # ToolBench alternate
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_text(item: dict) -> str:
    """Return the best available text field from a dataset item."""
    for f in TEXT_FIELDS:
        val = item.get(f)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, list) and val:
            first = val[0]
            # ToolBench conversations: list of {from, value} dicts
            if isinstance(first, dict) and "value" in first:
                return str(first["value"])
            # wiki-ner tokens: list of word strings → join into sentence
            if isinstance(first, str):
                return " ".join(val)
    return str(item)


def _load(
    dataset_id: str, config, split: str, streaming: bool, trust_remote_code: bool
):
    """Load a single dataset config, return an iterable."""
    kwargs: dict = {"split": split, "streaming": streaming}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if config:
        return load_dataset(dataset_id, config, **kwargs)
    return load_dataset(dataset_id, **kwargs)


def load_multi_config(
    dataset_id: str,
    configs: list[str],
    split: str,
    streaming: bool,
    trust_remote_code: bool,
):
    """Load multiple dataset configs and yield all items."""
    for cfg in configs:
        try:
            ds = _load(dataset_id, cfg, split, streaming, trust_remote_code)
            yield from ds
        except Exception as e:
            print(f"   ⚠️  Skipping config '{cfg}': {e}")


def write_jsonl(items, output_file: Path, source: str) -> int:
    """Write items to a JSONL file, return count written."""
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in items:
            text = extract_text(item)
            f.write(
                json.dumps({"question": text, "source": source}, ensure_ascii=False)
                + "\n"
            )
            count += 1
    return count


# ---------------------------------------------------------------------------
# Supplemental downloaders (bypass HF registry for problematic datasets)
# ---------------------------------------------------------------------------


def download_apps_parquet(output_dir: Path, failed: list) -> None:
    """Load APPS directly from parquet files, bypassing the legacy apps.py script.

    codeparrot/apps has apps.py which HF no longer allows running. Loading as
    'parquet' with explicit hf:// file globs skips the script entirely.
    """
    for tag, hf_glob in [
        ("test", "hf://datasets/codeparrot/apps/data/test-*.parquet"),
        ("dev", "hf://datasets/codeparrot/apps/data/train-*.parquet"),
    ]:
        label = f"apps [{tag}]"
        print(f"\n📥 Downloading {label} (parquet bypass)...")
        output_file = output_dir / f"apps_{tag}.jsonl"
        try:
            ds = load_dataset("parquet", data_files={tag: hf_glob}, split=tag)
            count = write_jsonl(ds, output_file, source=f"apps_{tag}")
            print(f"   ✅ {count:,} samples → {output_file.name}")
        except Exception as e:
            print(f"   ❌ {label}: {e}")
            failed.append((label, str(e)))


# IndicMTEval language-code → filename in GitHub repo
_INDICMTEVAL_FILES = {
    "hi": "Hin_test.jsonl",
    "ta": "Tam_test.jsonl",
    "mr": "Mar_test.jsonl",
    "ml": "Mal_test.jsonl",
    "gu": "Guj_test.jsonl",
}
_INDICMTEVAL_BASE = (
    "https://raw.githubusercontent.com/AI4Bharat/IndicMT-Eval"
    "/master/Dataset/Indic%20MT%20Eval/"
)


def download_indicmteval(output_dir: Path, failed: list) -> None:
    """Fetch IndicMTEval test JSONL files directly from GitHub.

    The dataset is not on HuggingFace Hub. Each item has 'source' (English)
    and 'reference' (Indic translation); both are written for contamination
    coverage of both sides of the translation pair.
    """
    print("\n📥 Downloading indicmteval [test] (GitHub)...")
    output_file = output_dir / "indicmteval_test.jsonl"
    count = 0
    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            for lang, filename in _INDICMTEVAL_FILES.items():
                url = _INDICMTEVAL_BASE + filename
                try:
                    with urllib.request.urlopen(url) as resp:
                        for raw_line in resp:
                            line = raw_line.decode("utf-8").strip()
                            if not line:
                                continue
                            item = json.loads(line)
                            for field in ("source", "reference"):
                                text = item.get(field, "").strip()
                                if text:
                                    out_f.write(
                                        json.dumps(
                                            {
                                                "question": text,
                                                "source": f"indicmteval_{lang}_{field}",
                                            },
                                            ensure_ascii=False,
                                        )
                                        + "\n"
                                    )
                                    count += 1
                    print(f"   ✅ {lang} ({filename})")
                except Exception as e:
                    print(f"   ⚠️  Skipping {filename}: {e}")
        print(f"   ✅ indicmteval total: {count:,} samples → {output_file.name}")
    except Exception as e:
        print(f"   ❌ indicmteval: {e}")
        failed.append(("indicmteval [test]", str(e)))


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------


def download_all(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    failed: list[tuple[str, str]] = []

    # ── Standard benchmarks from registry ──────────────────────────────────
    for name, splits in BENCHMARKS.items():
        for dataset_id, config, split, tag, streaming, trc in splits:
            label = f"{name} [{tag}]"
            print(f"\n📥 Downloading {label}...")
            output_file = output_dir / f"{name}_{tag}.jsonl"

            try:
                if isinstance(config, list):
                    items = load_multi_config(dataset_id, config, split, streaming, trc)
                else:
                    items = _load(dataset_id, config, split, streaming, trc)

                count = write_jsonl(items, output_file, source=f"{name}_{tag}")
                print(f"   ✅ {count:,} samples → {output_file.name}")

            except Exception as e:
                print(f"   ❌ {label}: {e}")
                failed.append((label, str(e)))

    # ── IndicGenBench (each subtask is a separate HF repo path) ────────────
    print("\n📥 Downloading indicgenbench [test]...")
    igb_file = output_dir / "indicgenbench_test.jsonl"
    igb_count = 0
    try:
        with open(igb_file, "w", encoding="utf-8") as f:
            for repo, indic_field, english_field in INDICGENBENCH_REPOS:
                try:
                    ds = load_dataset(repo, split="test")
                    subtask = repo.split("/")[-1]
                    for item in ds:
                        examples = item.get("examples") or {}
                        for field in (indic_field, english_field):
                            text = (examples.get(field) or "").strip()
                            if text:
                                f.write(
                                    json.dumps(
                                        {
                                            "question": text,
                                            "source": f"indicgenbench_{subtask}",
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                                igb_count += 1
                    print(f"   ✅ {subtask}")
                except Exception as e:
                    print(f"   ⚠️  Skipping {repo}: {e}")
        print(f"   ✅ indicgenbench total: {igb_count:,} samples → {igb_file.name}")
    except Exception as e:
        print(f"   ❌ indicgenbench: {e}")
        failed.append(("indicgenbench", str(e)))

    # ── APPS (parquet bypass — codeparrot/apps has a deprecated dataset script) ──
    download_apps_parquet(output_dir, failed)

    # ── IndicMTEval (GitHub-only — not on HF Hub) ──────────────────────────
    download_indicmteval(output_dir, failed)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failed:
        print(f"⚠️  {len(failed)} split(s) failed:")
        for lbl, err in failed:
            print(f"   {lbl}: {err}")
        print()
        print("Common fixes:")
        print("  GPQA Diamond      → export HF_TOKEN=<token>  (gated dataset)")
        print("  Rate limit 429    → export HF_TOKEN=<token>  (unauthenticated cap)")
        print("  Disk space [28]   → free space or redirect cache:")
        print("                      export HF_HOME=/path/to/bigger/disk")
    else:
        print("✅ All benchmarks downloaded successfully!")

    print()
    print("📝 Datasets NOT downloadable via this script:")
    print(
        "   piqa          → ybisk/piqa uses legacy dataset script; HF no longer supports these"
    )
    print("   arc_c_in      → sarvamai/ARC-Challenge-IN not found on HF Hub or GitHub")
    print("   RULER         → synthetic generator; must generate locally:")
    print("                     git clone https://github.com/NVIDIA/RULER")
    print("                     cd RULER/scripts/data/synthetic/json")
    print(
        "                     python download_paulgraham_essay.py && bash download_qa_dataset.sh"
    )
    print("                     cd ../.. && python prepare.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download benchmark test (and dev) splits for contamination checking.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks"),
        help="Directory to write JSONL files (default: benchmarks/)",
    )
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("⚠️  HF_TOKEN not set — some datasets (GPQA) require authentication.")
        print("   Run: export HF_TOKEN=<your_huggingface_token>\n")

    download_all(args.output_dir)


if __name__ == "__main__":
    main()
