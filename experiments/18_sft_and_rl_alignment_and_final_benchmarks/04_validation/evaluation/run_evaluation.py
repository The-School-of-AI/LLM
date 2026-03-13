"""
SFT Validation — Evaluation Runner
====================================
Orchestrates loading of prompts, calling base and SFT models (or reading
pre-collected outputs from CSV), running scoring, and saving results.

Supports two modes:
  1. API mode   — calls OpenAI-compatible endpoints for base and SFT models.
  2. CSV mode   — reads human-collected or pre-generated outputs from a CSV file.

Usage (API mode):
    python run_evaluation.py --mode api --config ../config/config.yaml

Usage (CSV mode):
    python run_evaluation.py --mode csv --input ../data/model_outputs.csv

Results are saved to ../results/evaluation_results_{timestamp}.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Ensure local modules are importable when running from any directory
sys.path.insert(0, str(Path(__file__).parent))

from instruction_following import InstructionFollowingScorer
from hallucination_detector import HallucinationDetector
from metrics import MetricsCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def load_prompts(prompts_path: str) -> list[dict]:
    """Load and return the evaluation prompt dataset."""
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", [])
    logger.info("Loaded %d prompts from %s", len(prompts), prompts_path)
    return prompts


# ---------------------------------------------------------------------------
# API-based model caller
# ---------------------------------------------------------------------------

def call_model_api(
    prompt_text: str,
    model_endpoint: str,
    model_name: str,
    api_key: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """
    Call an OpenAI-compatible chat completions API.
    Returns the model's response text, or an error string on failure.
    """
    try:
        import openai  # type: ignore
    except ImportError:
        raise ImportError(
            "openai package not installed. Install it with: pip install openai"
        )

    client = openai.OpenAI(api_key=api_key, base_url=model_endpoint)

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(
                "API call attempt %d/%d failed: %s", attempt, retries, exc
            )
            if attempt < retries:
                time.sleep(backoff * attempt)
            else:
                return f"[ERROR] API call failed after {retries} attempts: {exc}"


# ---------------------------------------------------------------------------
# CSV-based output loader
# ---------------------------------------------------------------------------

def load_csv_outputs(csv_path: str) -> dict[str, dict]:
    """
    Load pre-generated model outputs from a CSV file.

    Expected CSV columns:
        prompt_id, base_output, sft_output

    Returns:
        dict mapping prompt_id -> {"base_output": str, "sft_output": str}
    """
    import csv

    outputs: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("prompt_id", "").strip()
            if pid:
                outputs[pid] = {
                    "base_output": row.get("base_output", "").strip(),
                    "sft_output": row.get("sft_output", "").strip(),
                }
    logger.info("Loaded outputs for %d prompts from %s", len(outputs), csv_path)
    return outputs


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_prompt(
    prompt: dict,
    base_output: str,
    sft_output: str,
    if_scorer: "InstructionFollowingScorer",
    hallucination_detector: "HallucinationDetector",
    annotator_note: str = "",
) -> dict:
    """
    Score a single prompt for both base and SFT outputs.
    Returns a result dict for this prompt.
    """
    prompt_id = prompt["id"]
    rubric = prompt.get("scoring_rubric", {})
    constraints = prompt.get("instruction_constraints", {})
    anchors = prompt.get("hallucination_anchors", [])
    ground_truth = prompt.get("ground_truth_facts", {})

    # --- Instruction-following scores ---
    base_if = if_scorer.score(
        output=base_output,
        rubric=rubric,
        constraints=constraints,
    )
    sft_if = if_scorer.score(
        output=sft_output,
        rubric=rubric,
        constraints=constraints,
    )

    # --- Hallucination detection ---
    base_hall = hallucination_detector.detect(
        output=base_output,
        anchors=anchors,
        ground_truth=ground_truth,
    )
    sft_hall = hallucination_detector.detect(
        output=sft_output,
        anchors=anchors,
        ground_truth=ground_truth,
    )

    result = {
        "prompt_id": prompt_id,
        "category": prompt.get("category"),
        "sub_category": prompt.get("sub_category"),
        "difficulty": prompt.get("difficulty"),
        "prompt_text": prompt.get("prompt"),
        "base_output": base_output,
        "sft_output": sft_output,
        "annotator_note": annotator_note,
        "base": {
            "instruction_following": base_if,
            "hallucination": base_hall,
        },
        "sft": {
            "instruction_following": sft_if,
            "hallucination": sft_hall,
        },
        "delta": {
            "if_score_change": round(sft_if["score"] - base_if["score"], 4),
            "hallucination_risk_change": round(
                sft_hall["risk_score"] - base_hall["risk_score"], 4
            ),
        },
    }
    return result


def run_api_evaluation(
    prompts: list[dict],
    cfg: dict,
    if_scorer: "InstructionFollowingScorer",
    hallucination_detector: "HallucinationDetector",
    metrics_collector: "MetricsCollector",
    output_path: str,
) -> list[dict]:
    """Run evaluation by calling both models via API for each prompt."""
    base_cfg = cfg["models"]["base"]
    sft_cfg = cfg["models"]["sft"]
    results = []

    for i, prompt in enumerate(prompts, 1):
        pid = prompt["id"]
        logger.info("[%d/%d] Evaluating prompt %s ...", i, len(prompts), pid)

        base_out = call_model_api(
            prompt_text=prompt["prompt"],
            model_endpoint=base_cfg.get("endpoint", ""),
            model_name=base_cfg["model_name"],
            api_key=base_cfg.get("api_key", os.getenv("BASE_MODEL_API_KEY", "")),
            max_tokens=cfg.get("generation", {}).get("max_tokens", 1024),
            temperature=cfg.get("generation", {}).get("temperature", 0.0),
        )

        sft_out = call_model_api(
            prompt_text=prompt["prompt"],
            model_endpoint=sft_cfg.get("endpoint", ""),
            model_name=sft_cfg["model_name"],
            api_key=sft_cfg.get("api_key", os.getenv("SFT_MODEL_API_KEY", "")),
            max_tokens=cfg.get("generation", {}).get("max_tokens", 1024),
            temperature=cfg.get("generation", {}).get("temperature", 0.0),
        )

        result = evaluate_prompt(
            prompt=prompt,
            base_output=base_out,
            sft_output=sft_out,
            if_scorer=if_scorer,
            hallucination_detector=hallucination_detector,
        )
        results.append(result)
        metrics_collector.add_result(result)

        # Checkpoint save every 10 prompts
        if i % 10 == 0:
            _save_results(results, output_path + ".checkpoint")
            logger.info("Checkpoint saved after %d prompts.", i)

    return results


def run_csv_evaluation(
    prompts: list[dict],
    csv_outputs: dict[str, dict],
    if_scorer: "InstructionFollowingScorer",
    hallucination_detector: "HallucinationDetector",
    metrics_collector: "MetricsCollector",
) -> list[dict]:
    """Run evaluation using pre-loaded outputs from a CSV file."""
    results = []
    matched = 0
    skipped = 0

    for i, prompt in enumerate(prompts, 1):
        pid = prompt["id"]
        if pid not in csv_outputs:
            logger.warning("No CSV output found for prompt %s — skipping.", pid)
            skipped += 1
            continue

        matched += 1
        logger.info("[%d/%d] Evaluating prompt %s ...", i, len(prompts), pid)

        outputs = csv_outputs[pid]
        result = evaluate_prompt(
            prompt=prompt,
            base_output=outputs["base_output"],
            sft_output=outputs["sft_output"],
            if_scorer=if_scorer,
            hallucination_detector=hallucination_detector,
        )
        results.append(result)
        metrics_collector.add_result(result)

    logger.info("CSV evaluation complete: %d evaluated, %d skipped.", matched, skipped)
    return results


def _save_results(results: list[dict], path: str) -> None:
    """Persist results as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFT Validation Evaluation Runner"
    )
    parser.add_argument(
        "--mode",
        choices=["api", "csv"],
        default="csv",
        help="Evaluation mode: 'api' to call models live, 'csv' to use pre-collected outputs.",
    )
    parser.add_argument(
        "--config",
        default="../config/config.yaml",
        help="Path to config.yaml (required for API mode).",
    )
    parser.add_argument(
        "--input",
        default="../data/model_outputs.csv",
        help="Path to CSV file with pre-generated outputs (CSV mode).",
    )
    parser.add_argument(
        "--prompts",
        default="../prompts/evaluation_prompts.json",
        help="Path to the evaluation prompts JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="../results",
        help="Directory where result files will be saved.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Optional: filter to specific categories (e.g. instruction_following factual_qa).",
    )
    parser.add_argument(
        "--difficulties",
        nargs="*",
        choices=["easy", "medium", "hard"],
        help="Optional: filter to specific difficulty levels.",
    )
    args = parser.parse_args()

    # --- Load config ---
    cfg: dict[str, Any] = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", config_path)
    else:
        logger.warning("Config file not found at %s — using defaults.", config_path)

    # --- Load prompts ---
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        # Try relative to this file's parent
        prompts_path = Path(__file__).parent.parent / "prompts" / "evaluation_prompts.json"
    prompts = load_prompts(str(prompts_path))

    # --- Optional filtering ---
    if args.categories:
        prompts = [p for p in prompts if p.get("category") in args.categories]
        logger.info("Filtered to %d prompts in categories: %s", len(prompts), args.categories)
    if args.difficulties:
        prompts = [p for p in prompts if p.get("difficulty") in args.difficulties]
        logger.info("Filtered to %d prompts at difficulties: %s", len(prompts), args.difficulties)

    if not prompts:
        logger.error("No prompts to evaluate after filtering. Exiting.")
        sys.exit(1)

    # --- Initialise scorers ---
    if_scorer = InstructionFollowingScorer(cfg.get("scoring", {}))
    hall_detector = HallucinationDetector(cfg.get("hallucination", {}))
    metrics_collector = MetricsCollector()

    # --- Run evaluation ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"evaluation_results_{timestamp}.json")

    if args.mode == "api":
        logger.info("Running in API mode ...")
        results = run_api_evaluation(
            prompts=prompts,
            cfg=cfg,
            if_scorer=if_scorer,
            hallucination_detector=hall_detector,
            metrics_collector=metrics_collector,
            output_path=output_path,
        )
    else:
        logger.info("Running in CSV mode from: %s", args.input)
        csv_outputs = load_csv_outputs(args.input)
        results = run_csv_evaluation(
            prompts=prompts,
            csv_outputs=csv_outputs,
            if_scorer=if_scorer,
            hallucination_detector=hall_detector,
            metrics_collector=metrics_collector,
        )

    # --- Save final results ---
    _save_results(results, output_path)

    # --- Print summary ---
    summary = metrics_collector.summary()
    print("\n" + "=" * 60)
    print("  SFT VALIDATION — EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Prompts evaluated    : {summary['total_prompts']}")
    print(f"  Base IF rate         : {summary['base_if_rate']:.1%}")
    print(f"  SFT  IF rate         : {summary['sft_if_rate']:.1%}")
    print(f"  IF improvement       : {summary['if_improvement']:+.1%}")
    print(f"  Base hallucination % : {summary['base_hallucination_rate']:.1%}")
    print(f"  SFT  hallucination % : {summary['sft_hallucination_rate']:.1%}")
    print(f"  Hall. risk change    : {summary['hallucination_delta']:+.4f}")
    print(f"  Results file         : {output_path}")
    print("=" * 60 + "\n")

    # Save summary separately
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
