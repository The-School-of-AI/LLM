"""
Metrics Collector
==================
Accumulates per-prompt evaluation results and computes aggregate metrics
for the Base vs SFT comparison report.

Key metrics produced
---------------------
Instruction-Following:
  - base_if_rate          : % of prompts where base model followed instructions
  - sft_if_rate           : % of prompts where SFT model followed instructions
  - if_improvement        : percentage-point gain (sft - base)
  - if_score_avg_base     : mean IF score (base)
  - if_score_avg_sft      : mean IF score (SFT)
  - per_category_if_rate  : IF rate breakdown by prompt category
  - per_difficulty_if_rate: IF rate breakdown by difficulty level

Hallucination:
  - base_hallucination_rate  : % prompts where hallucination detected (base)
  - sft_hallucination_rate   : % prompts where hallucination detected (SFT)
  - hallucination_delta      : mean risk score change (sft - base)
  - new_hallucination_prompts: prompts where SFT introduced hallucination not in base

Summary:
  - validation_pass          : True if IF improved AND no new hall. patterns
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class MetricsCollector:
    """
    Accumulates results and computes aggregate SFT validation metrics.
    """

    def __init__(self) -> None:
        self._results: list[dict] = []

    def add_result(self, result: dict) -> None:
        """Add a single prompt evaluation result."""
        self._results.append(result)

    def add_results(self, results: list[dict]) -> None:
        """Add multiple results at once."""
        self._results.extend(results)

    # ------------------------------------------------------------------
    # Summary computation
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Compute and return the full metrics summary dict.
        Call after all results have been added.
        """
        results = self._results
        if not results:
            return {"error": "No results to summarise."}

        n = len(results)

        # --- Base IF ---
        base_if_followed = [
            r["base"]["instruction_following"]["followed"] for r in results
        ]
        base_if_scores = [
            r["base"]["instruction_following"]["score"] for r in results
        ]

        # --- SFT IF ---
        sft_if_followed = [
            r["sft"]["instruction_following"]["followed"] for r in results
        ]
        sft_if_scores = [
            r["sft"]["instruction_following"]["score"] for r in results
        ]

        base_if_rate = sum(base_if_followed) / n
        sft_if_rate = sum(sft_if_followed) / n
        if_improvement = sft_if_rate - base_if_rate
        if_score_avg_base = sum(base_if_scores) / n
        if_score_avg_sft = sum(sft_if_scores) / n

        # --- Base hallucination ---
        base_hall_detected = [
            r["base"]["hallucination"]["detected"] for r in results
        ]
        base_hall_scores = [
            r["base"]["hallucination"]["risk_score"] for r in results
        ]

        # --- SFT hallucination ---
        sft_hall_detected = [
            r["sft"]["hallucination"]["detected"] for r in results
        ]
        sft_hall_scores = [
            r["sft"]["hallucination"]["risk_score"] for r in results
        ]

        base_hall_rate = sum(base_hall_detected) / n
        sft_hall_rate = sum(sft_hall_detected) / n
        hall_delta_scores = [
            sft - base for sft, base in zip(sft_hall_scores, base_hall_scores)
        ]
        hallucination_delta = sum(hall_delta_scores) / n

        # New hallucinations: SFT flagged but base was NOT flagged
        new_hallucination_prompts = [
            r["prompt_id"]
            for r, base_d, sft_d in zip(results, base_hall_detected, sft_hall_detected)
            if sft_d and not base_d
        ]

        # --- Per-category breakdown ---
        per_category_if = self._per_group_if_rate(results, "category")
        per_difficulty_if = self._per_group_if_rate(results, "difficulty")
        per_category_hall = self._per_group_hall_rate(results, "category")

        # --- Worst performing prompts (SFT IF score < 0.5) ---
        worst_prompts = sorted(
            [
                {"prompt_id": r["prompt_id"], "sft_if_score": r["sft"]["instruction_following"]["score"]}
                for r in results
                if r["sft"]["instruction_following"]["score"] < 0.5
            ],
            key=lambda x: x["sft_if_score"],
        )[:10]

        # --- Best improvements (largest positive delta) ---
        most_improved = sorted(
            [
                {"prompt_id": r["prompt_id"], "if_delta": r["delta"]["if_score_change"]}
                for r in results
            ],
            key=lambda x: x["if_delta"],
            reverse=True,
        )[:10]

        # --- Regression prompts (SFT worse than base) ---
        regressions = [
            {"prompt_id": r["prompt_id"], "if_delta": r["delta"]["if_score_change"]}
            for r in results
            if r["delta"]["if_score_change"] < -0.1
        ]

        # --- Validation pass criteria ---
        # PASS if: SFT IF rate >= base IF rate AND no new hallucination patterns
        validation_pass = (sft_if_rate >= base_if_rate) and (len(new_hallucination_prompts) == 0)

        return {
            "total_prompts": n,
            # Instruction following
            "base_if_rate": round(base_if_rate, 4),
            "sft_if_rate": round(sft_if_rate, 4),
            "if_improvement": round(if_improvement, 4),
            "if_improvement_pp": round(if_improvement * 100, 2),  # percentage points
            "if_score_avg_base": round(if_score_avg_base, 4),
            "if_score_avg_sft": round(if_score_avg_sft, 4),
            "if_score_delta": round(if_score_avg_sft - if_score_avg_base, 4),
            "per_category_if_rate": per_category_if,
            "per_difficulty_if_rate": per_difficulty_if,
            # Hallucination
            "base_hallucination_rate": round(base_hall_rate, 4),
            "sft_hallucination_rate": round(sft_hall_rate, 4),
            "hallucination_delta": round(hallucination_delta, 4),
            "new_hallucination_count": len(new_hallucination_prompts),
            "new_hallucination_prompts": new_hallucination_prompts,
            "per_category_hall_rate": per_category_hall,
            # Diagnostics
            "worst_sft_prompts": worst_prompts,
            "most_improved_prompts": most_improved,
            "regression_prompts": regressions,
            # Verdict
            "validation_pass": validation_pass,
            "validation_criteria": {
                "if_improved": sft_if_rate >= base_if_rate,
                "no_new_hallucinations": len(new_hallucination_prompts) == 0,
            },
        }

    # ------------------------------------------------------------------
    # Disaggregated metrics
    # ------------------------------------------------------------------

    def _per_group_if_rate(self, results: list[dict], group_key: str) -> dict:
        """Compute IF rate grouped by a result attribute (category, difficulty, etc.)."""
        groups: dict[str, list[bool]] = defaultdict(list)
        for r in results:
            key = r.get(group_key, "unknown")
            groups[key].append(r["sft"]["instruction_following"]["followed"])

        return {
            k: {
                "sft_if_rate": round(sum(v) / len(v), 4),
                "n": len(v),
            }
            for k, v in sorted(groups.items())
        }

    def _per_group_hall_rate(self, results: list[dict], group_key: str) -> dict:
        """Compute hallucination detection rate grouped by a result attribute."""
        groups: dict[str, list[bool]] = defaultdict(list)
        for r in results:
            key = r.get(group_key, "unknown")
            groups[key].append(r["sft"]["hallucination"]["detected"])

        return {
            k: {
                "sft_hall_rate": round(sum(v) / len(v), 4),
                "n": len(v),
            }
            for k, v in sorted(groups.items())
        }

    # ------------------------------------------------------------------
    # Report generation helpers
    # ------------------------------------------------------------------

    def print_summary_table(self) -> None:
        """Print a human-readable text summary table."""
        s = self.summary()
        if "error" in s:
            print(s["error"])
            return

        print("\n" + "=" * 65)
        print("  SFT VALIDATION METRICS REPORT")
        print("=" * 65)
        print(f"  Total prompts evaluated    : {s['total_prompts']}")
        print()
        print("  INSTRUCTION FOLLOWING")
        print(f"  ├─ Base IF rate            : {s['base_if_rate']:.1%}")
        print(f"  ├─ SFT  IF rate            : {s['sft_if_rate']:.1%}")
        print(f"  ├─ Improvement             : {s['if_improvement_pp']:+.2f} pp")
        print(f"  ├─ Base avg IF score       : {s['if_score_avg_base']:.4f}")
        print(f"  └─ SFT  avg IF score       : {s['if_score_avg_sft']:.4f}")
        print()
        print("  HALLUCINATION")
        print(f"  ├─ Base hallucination rate : {s['base_hallucination_rate']:.1%}")
        print(f"  ├─ SFT  hallucination rate : {s['sft_hallucination_rate']:.1%}")
        print(f"  ├─ Avg risk score delta    : {s['hallucination_delta']:+.4f}")
        print(f"  └─ New hallucination cases : {s['new_hallucination_count']}")
        if s["new_hallucination_prompts"]:
            print(f"       → {s['new_hallucination_prompts']}")
        print()
        print("  PER-CATEGORY IF RATE (SFT)")
        for cat, data in s["per_category_if_rate"].items():
            bar = "█" * int(data["sft_if_rate"] * 20)
            print(f"  ├─ {cat:<30} {data['sft_if_rate']:.1%}  {bar}  (n={data['n']})")
        print()
        print("  PER-DIFFICULTY IF RATE (SFT)")
        for diff, data in s["per_difficulty_if_rate"].items():
            bar = "█" * int(data["sft_if_rate"] * 20)
            print(f"  ├─ {diff:<30} {data['sft_if_rate']:.1%}  {bar}  (n={data['n']})")
        print()
        print("  REGRESSION PROMPTS (SFT worse by > 10pp)")
        if s["regression_prompts"]:
            for r in s["regression_prompts"]:
                print(f"  ├─ {r['prompt_id']}: delta = {r['if_delta']:+.4f}")
        else:
            print("  ├─ None detected ✓")
        print()
        verdict = "✅ PASS" if s["validation_pass"] else "❌ FAIL"
        print(f"  VALIDATION VERDICT: {verdict}")
        if_ok = "✅" if s["validation_criteria"]["if_improved"] else "❌"
        hall_ok = "✅" if s["validation_criteria"]["no_new_hallucinations"] else "❌"
        print(f"    {if_ok} IF rate improved (or maintained)")
        print(f"    {hall_ok} No new hallucination patterns introduced")
        print("=" * 65 + "\n")

    def to_csv(self, output_path: str) -> None:
        """Export per-prompt summary to CSV for spreadsheet review."""
        import csv
        from pathlib import Path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "prompt_id", "category", "difficulty",
            "base_if_score", "sft_if_score", "if_delta",
            "base_if_followed", "sft_if_followed",
            "base_hall_risk", "sft_hall_risk", "hall_risk_delta",
            "base_hall_detected", "sft_hall_detected",
            "new_hallucination",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._results:
                base_hall_det = r["base"]["hallucination"]["detected"]
                sft_hall_det = r["sft"]["hallucination"]["detected"]
                writer.writerow({
                    "prompt_id": r["prompt_id"],
                    "category": r.get("category", ""),
                    "difficulty": r.get("difficulty", ""),
                    "base_if_score": r["base"]["instruction_following"]["score"],
                    "sft_if_score": r["sft"]["instruction_following"]["score"],
                    "if_delta": r["delta"]["if_score_change"],
                    "base_if_followed": r["base"]["instruction_following"]["followed"],
                    "sft_if_followed": r["sft"]["instruction_following"]["followed"],
                    "base_hall_risk": r["base"]["hallucination"]["risk_score"],
                    "sft_hall_risk": r["sft"]["hallucination"]["risk_score"],
                    "hall_risk_delta": r["delta"]["hallucination_risk_change"],
                    "base_hall_detected": base_hall_det,
                    "sft_hall_detected": sft_hall_det,
                    "new_hallucination": (sft_hall_det and not base_hall_det),
                })
