"""
MMLU Evaluation on the frozen reduced subset.

Each question is a 4-choice multiple-choice problem. We use log-likelihood
scoring (which choice has the highest probability) for robustness with
quantized models that may not follow instructions.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)

CHOICE_LABELS = ["A", "B", "C", "D"]


def _format_question(question: dict) -> str:
    """Format a question into a zero-shot prompt."""
    choices_text = "\n".join(
        f"{CHOICE_LABELS[i]}. {c}" for i, c in enumerate(question["choices"])
    )
    return (
        f"Question: {question['question']}\n"
        f"{choices_text}\n"
        f"Answer:"
    )


def _score_by_likelihood(model_wrapper: Any, prompt: str, choices: list[str]) -> int:
    """Return index of choice with lowest NLL (highest probability)."""
    best_idx = 0
    best_nll = float("inf")
    for i, choice in enumerate(choices):
        full_text = prompt + " " + CHOICE_LABELS[i]
        nll = model_wrapper.log_likelihood(full_text)
        if nll < best_nll:
            best_nll = nll
            best_idx = i
    return best_idx


def _score_by_generation(model_wrapper: Any, prompt: str) -> int:
    """Parse generated text for A/B/C/D."""
    generated = model_wrapper.generate(prompt).strip()
    for i, label in enumerate(CHOICE_LABELS):
        if generated.upper().startswith(label):
            return i
    # Fallback: first character match
    first_char = generated[0].upper() if generated else "A"
    if first_char in CHOICE_LABELS:
        return CHOICE_LABELS.index(first_char)
    return 0


def run_mmlu_eval(
    model_wrapper: Any,
    subset_path: Path,
    use_likelihood: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run MMLU evaluation on the frozen subset.

    Returns a result dict with overall accuracy, per-category accuracy,
    per-question results, and timing information.
    """
    with open(subset_path) as f:
        data = json.load(f)

    questions = data["questions"]
    metadata = data["metadata"]

    results_per_question: list[dict] = []
    category_stats: dict[str, dict] = {}
    domain_stats: dict[str, dict] = {}
    lang_stats: dict[str, dict] = {}

    start_time = time.time()

    for q in tqdm(questions, desc="MMLU", disable=not verbose):
        prompt = _format_question(q)
        t0 = time.time()

        if use_likelihood:
            predicted = _score_by_likelihood(model_wrapper, prompt, q["choices"])
        else:
            predicted = _score_by_generation(model_wrapper, prompt)

        elapsed = time.time() - t0
        correct = predicted == q["answer"]

        qr = {
            "id": q.get("id", f"{q['category']}_{len(results_per_question)}"),
            "category": q["category"],
            "domain": q.get("domain", "general_knowledge"),
            "language": q.get("language", "en"),
            "correct": correct,
            "predicted": CHOICE_LABELS[predicted],
            "gold": CHOICE_LABELS[q["answer"]],
            "elapsed_s": round(elapsed, 3),
        }
        results_per_question.append(qr)

        cat = q["category"]
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
        category_stats[cat]["total"] += 1
        if correct:
            category_stats[cat]["correct"] += 1

        dom = q.get("domain", "general_knowledge")
        if dom not in domain_stats:
            domain_stats[dom] = {"correct": 0, "total": 0}
        domain_stats[dom]["total"] += 1
        if correct:
            domain_stats[dom]["correct"] += 1

        lang = q.get("language", "en")
        if lang not in lang_stats:
            lang_stats[lang] = {"correct": 0, "total": 0}
        lang_stats[lang]["total"] += 1
        if correct:
            lang_stats[lang]["correct"] += 1

    total_time = time.time() - start_time
    total_q = len(questions)
    total_correct = sum(1 for r in results_per_question if r["correct"])

    # Compute per-category accuracy
    for cat, stats in category_stats.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0

    # Compute per-domain accuracy
    DOMAIN_ORDER = ["math", "reasoning", "science", "coding", "general_knowledge"]
    for dom, stats in domain_stats.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0

    # Compute per-language accuracy (English vs each Indic language)
    for lang, stats in lang_stats.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0

    overall_accuracy = total_correct / total_q if total_q else 0.0

    # Build ordered domain summary
    domain_accuracies = {}
    for dom in DOMAIN_ORDER:
        if dom in domain_stats:
            s = domain_stats[dom]
            domain_accuracies[dom] = {
                "accuracy": round(s["accuracy"], 4),
                "correct": s["correct"],
                "total": s["total"],
            }
    # Any unexpected domains appended after
    for dom, s in domain_stats.items():
        if dom not in domain_accuracies:
            domain_accuracies[dom] = {
                "accuracy": round(s["accuracy"], 4),
                "correct": s["correct"],
                "total": s["total"],
            }

    return {
        "eval_type": "mmlu_subset",
        "subset_version": metadata.get("version", 1),
        "subset_seed": metadata.get("seed"),
        "total_questions": total_q,
        "total_correct": total_correct,
        "overall_accuracy": round(overall_accuracy, 4),
        "domain_accuracies": domain_accuracies,
        "language_accuracies": {
            lang: {
                "accuracy": round(s["accuracy"], 4),
                "correct": s["correct"],
                "total": s["total"],
            }
            for lang, s in lang_stats.items()
        },
        "category_accuracies": {
            cat: round(s["accuracy"], 4) for cat, s in category_stats.items()
        },
        "total_time_s": round(total_time, 2),
        "questions_per_second": round(total_q / total_time, 3) if total_time > 0 else 0,
        "per_question_results": results_per_question,
    }
