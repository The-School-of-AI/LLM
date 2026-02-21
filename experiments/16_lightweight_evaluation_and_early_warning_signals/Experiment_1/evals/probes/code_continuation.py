"""Code continuation probe: tests whether model can sensibly continue code snippets."""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _check_continuation(continuation: str, probe: dict) -> dict[str, Any]:
    """
    Heuristic evaluation of a code continuation.

    Checks:
    1. keyword_match  — any expected keyword present?
    2. pattern_match  — any regex pattern matches?
    3. no_crash       — continuation is non-empty and not pure whitespace
    4. not_repetitive — no large repeated block (simple repetition detection)
    """
    cont_lower = continuation.lower()
    text = continuation.strip()

    # 1. Keyword check (case-insensitive)
    keyword_hit = any(kw.lower() in cont_lower for kw in probe.get("expected_keywords", []))

    # 2. Pattern check
    pattern_hit = False
    for pat in probe.get("expected_patterns", []):
        if re.search(pat, continuation, re.IGNORECASE):
            pattern_hit = True
            break

    # 3. Non-empty
    no_crash = len(text) >= 1

    # 4. Not purely repetitive (detect if >50% of tokens are the same)
    tokens = text.split()
    if len(tokens) > 5:
        most_common_count = max(tokens.count(t) for t in set(tokens))
        not_repetitive = most_common_count / len(tokens) < 0.5
    else:
        not_repetitive = True

    # Pass if keyword OR pattern hit, AND not empty, AND not repetitive
    passed = (keyword_hit or pattern_hit) and no_crash and not_repetitive

    return {
        "passed": passed,
        "keyword_hit": keyword_hit,
        "pattern_hit": pattern_hit,
        "no_crash": no_crash,
        "not_repetitive": not_repetitive,
    }


def run_code_probe(
    model_wrapper: Any,
    probes_path: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run code continuation probes.

    Returns pass rate and per-probe details.
    """
    with open(probes_path) as f:
        data = json.load(f)

    probes = data["probes"]
    results: list[dict] = []
    category_stats: dict[str, dict] = {}

    start_time = time.time()

    for probe in tqdm(probes, desc="Code Probes", disable=not verbose):
        t0 = time.time()
        continuation = model_wrapper.generate(probe["prompt"])
        elapsed = time.time() - t0

        check = _check_continuation(continuation, probe)

        result = {
            "id": probe["id"],
            "language": probe.get("language", "python"),
            "category": probe.get("category", "unknown"),
            "description": probe.get("description", ""),
            "passed": check["passed"],
            "keyword_hit": check["keyword_hit"],
            "pattern_hit": check["pattern_hit"],
            "continuation_preview": continuation[:200].strip(),
            "elapsed_s": round(elapsed, 3),
        }
        results.append(result)

        cat = probe.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"passed": 0, "total": 0}
        category_stats[cat]["total"] += 1
        if check["passed"]:
            category_stats[cat]["passed"] += 1

    total_time = time.time() - start_time
    total_passed = sum(1 for r in results if r["passed"])
    pass_rate = total_passed / len(probes) if probes else 0.0

    return {
        "eval_type": "code_continuation_probe",
        "num_probes": len(probes),
        "total_passed": total_passed,
        "pass_rate": round(pass_rate, 4),
        "total_time_s": round(total_time, 2),
        "per_category": {
            cat: {
                "pass_rate": round(s["passed"] / s["total"], 4) if s["total"] else 0.0,
                "passed": s["passed"],
                "total": s["total"],
            }
            for cat, s in category_stats.items()
        },
        "per_probe_results": results,
    }
