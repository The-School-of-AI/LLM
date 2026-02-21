"""Language modeling probe: measures perplexity on curated passages."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_lm_probe(
    model_wrapper: Any,
    probes_path: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Compute perplexity on each curated passage.

    Perplexity = exp(avg NLL per token). Lower is better.

    Returns per-passage and aggregate results.
    """
    with open(probes_path) as f:
        data = json.load(f)

    probes = data["probes"]
    results: list[dict] = []
    category_nlls: dict[str, list[float]] = {}

    start_time = time.time()

    for probe in tqdm(probes, desc="LM Probes", disable=not verbose):
        t0 = time.time()
        nll = model_wrapper.log_likelihood(probe["text"])
        elapsed = time.time() - t0

        ppl = math.exp(nll) if not math.isnan(nll) and nll < 100 else float("inf")

        result = {
            "id": probe["id"],
            "category": probe["category"],
            "nll": round(nll, 4),
            "perplexity": round(ppl, 2),
            "elapsed_s": round(elapsed, 3),
        }
        results.append(result)

        cat = probe["category"]
        if cat not in category_nlls:
            category_nlls[cat] = []
        if not math.isnan(nll):
            category_nlls[cat].append(nll)

    total_time = time.time() - start_time
    valid_nlls = [r["nll"] for r in results if not math.isnan(r["nll"])]
    mean_nll = sum(valid_nlls) / len(valid_nlls) if valid_nlls else float("nan")
    mean_ppl = math.exp(mean_nll) if not math.isnan(mean_nll) and mean_nll < 100 else float("inf")

    return {
        "eval_type": "language_modeling_probe",
        "num_probes": len(probes),
        "mean_nll": round(mean_nll, 4),
        "mean_perplexity": round(mean_ppl, 2),
        "total_time_s": round(total_time, 2),
        "per_category": {
            cat: {
                "mean_nll": round(sum(nlls) / len(nlls), 4),
                "mean_perplexity": round(math.exp(sum(nlls) / len(nlls)), 2),
            }
            for cat, nlls in category_nlls.items() if nlls
        },
        "per_probe_results": results,
    }
