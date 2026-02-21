"""Math prose probe: evaluates natural-language math reasoning."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _check_answer(generated: str, probe: dict) -> bool:
    """Return True if any expected keyword appears in the generated text."""
    gen_lower = generated.lower()
    for kw in probe.get("answer_keywords", [probe.get("answer", "")]):
        if str(kw).lower() in gen_lower:
            return True
    return False


def run_math_probe(
    model_wrapper: Any,
    probes_path: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run math/reasoning probes.

    Returns accuracy and per-probe details.
    """
    with open(probes_path) as f:
        data = json.load(f)

    probes = data["probes"]
    results: list[dict] = []
    category_stats: dict[str, dict] = {}
    difficulty_stats: dict[str, dict] = {}

    start_time = time.time()

    for probe in tqdm(probes, desc="Math Probes", disable=not verbose):
        t0 = time.time()
        generated = model_wrapper.generate(probe["prompt"])
        elapsed = time.time() - t0

        correct = _check_answer(generated, probe)

        result = {
            "id": probe["id"],
            "category": probe.get("category", "unknown"),
            "difficulty": probe.get("difficulty", "medium"),
            "correct": correct,
            "gold_answer": probe.get("answer", ""),
            "generated_preview": generated[:200].strip(),
            "elapsed_s": round(elapsed, 3),
        }
        results.append(result)

        cat = probe.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
        category_stats[cat]["total"] += 1
        if correct:
            category_stats[cat]["correct"] += 1

        diff = probe.get("difficulty", "medium")
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"correct": 0, "total": 0}
        difficulty_stats[diff]["total"] += 1
        if correct:
            difficulty_stats[diff]["correct"] += 1

    total_time = time.time() - start_time
    total_correct = sum(1 for r in results if r["correct"])
    accuracy = total_correct / len(probes) if probes else 0.0

    return {
        "eval_type": "math_prose_probe",
        "num_probes": len(probes),
        "total_correct": total_correct,
        "accuracy": round(accuracy, 4),
        "total_time_s": round(total_time, 2),
        "per_category": {
            cat: {
                "accuracy": round(s["correct"] / s["total"], 4) if s["total"] else 0.0,
                "correct": s["correct"],
                "total": s["total"],
            }
            for cat, s in category_stats.items()
        },
        "by_difficulty": {
            diff: {
                "accuracy": round(s["correct"] / s["total"], 4) if s["total"] else 0.0,
                "correct": s["correct"],
                "total": s["total"],
            }
            for diff, s in difficulty_stats.items()
        },
        "per_probe_results": results,
    }
