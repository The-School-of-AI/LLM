"""Consistency probe: checks that rephrased questions get the same answer."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _contains_keyword(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def run_consistency_probe(
    model_wrapper: Any,
    probes_path: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run consistency probes.

    For each group of rephrased questions:
    - Generate answers for all variants
    - Check if each answer contains the expected keyword(s)
    - Compute agreement_rate = fraction of variants that give the correct answer
    - Fully consistent group = all variants agree AND are correct

    Returns aggregate and per-group results.
    """
    with open(probes_path) as f:
        data = json.load(f)

    groups = data["probe_groups"]
    group_results: list[dict] = []

    start_time = time.time()

    for group in tqdm(groups, desc="Consistency Probes", disable=not verbose):
        expected_kws = group["expected_answer_keywords"]
        variant_results: list[dict] = []

        for variant in group["variants"]:
            t0 = time.time()
            generated = model_wrapper.generate(variant["prompt"])
            elapsed = time.time() - t0

            hit = _contains_keyword(generated, expected_kws)
            variant_results.append({
                "id": variant["id"],
                "prompt": variant["prompt"],
                "generated_preview": generated[:200].strip(),
                "correct": hit,
                "elapsed_s": round(elapsed, 3),
            })

        num_correct = sum(1 for v in variant_results if v["correct"])
        total_variants = len(variant_results)
        agreement_rate = num_correct / total_variants if total_variants else 0.0
        fully_consistent = agreement_rate == 1.0

        group_results.append({
            "group_id": group["group_id"],
            "topic": group["topic"],
            "num_variants": total_variants,
            "num_correct": num_correct,
            "agreement_rate": round(agreement_rate, 4),
            "fully_consistent": fully_consistent,
            "variants": variant_results,
        })

    total_time = time.time() - start_time

    # Aggregate
    total_groups = len(group_results)
    fully_consistent_count = sum(1 for g in group_results if g["fully_consistent"])
    all_agreement_rates = [g["agreement_rate"] for g in group_results]
    mean_agreement = sum(all_agreement_rates) / len(all_agreement_rates) if all_agreement_rates else 0.0

    return {
        "eval_type": "consistency_probe",
        "num_groups": total_groups,
        "fully_consistent_groups": fully_consistent_count,
        "consistency_rate": round(fully_consistent_count / total_groups, 4) if total_groups else 0.0,
        "mean_agreement_rate": round(mean_agreement, 4),
        "total_time_s": round(total_time, 2),
        "per_group_results": group_results,
    }
