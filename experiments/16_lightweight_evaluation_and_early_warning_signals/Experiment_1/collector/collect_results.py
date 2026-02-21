"""
Result Collector — aggregates individual JSON result files from all team members
into a single structured database (JSON) and prints a summary table.

Usage:
    # Pull new results from a shared folder and aggregate:
    python collector/collect_results.py --results-dir results/raw --out results/aggregated_results.json

    # Pretty-print the current trend summary:
    python collector/collect_results.py --summary
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


METRIC_KEYS = {
    "mmlu": ("overall_accuracy", "↑"),
    "language_modeling": ("mean_perplexity", "↓"),
    "code_continuation": ("pass_rate", "↑"),
    "math_prose": ("accuracy", "↑"),
    "consistency": ("mean_agreement_rate", "↑"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate eval results")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results" / "raw")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "aggregated_results.json")
    p.add_argument("--summary", action="store_true", help="Print summary table and exit")
    return p.parse_args()


def extract_summary(result: dict) -> dict:
    """Extract scalar metrics from a raw result file."""
    summary: dict = {
        "run_id": result.get("run_id", "unknown"),
        "checkpoint_name": result.get("checkpoint_name", "unknown"),
        "quant_mode": result.get("quant_mode", "unknown"),
        "backend": result.get("backend", "unknown"),
        "timestamp_utc": result.get("timestamp_utc", ""),
        "hostname": result.get("system_info", {}).get("hostname", "unknown"),
        "metrics": {},
    }
    eval_results = result.get("eval_results", {})
    for eval_name, (metric_key, direction) in METRIC_KEYS.items():
        if eval_name in eval_results:
            value = eval_results[eval_name].get(metric_key)
            if value is not None:
                summary["metrics"][eval_name] = {
                    "value": round(float(value), 5),
                    "metric": metric_key,
                    "direction": direction,
                }
    return summary


def load_all_results(results_dir: Path) -> list[dict]:
    """Load all JSON result files from the directory."""
    records = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            records.append(data)
        except Exception as e:
            print(f"[WARN] Could not parse {f.name}: {e}")
    return records


def aggregate(records: list[dict]) -> dict:
    """Build the aggregated database from raw result files."""
    summaries = [extract_summary(r) for r in records]
    # Sort by timestamp
    summaries.sort(key=lambda s: s.get("timestamp_utc", ""))
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_runs": len(summaries),
        "runs": summaries,
    }


def print_summary_table(agg: dict) -> None:
    runs = agg["runs"]
    if not runs:
        print("No results yet.")
        return

    headers = ["Checkpoint", "Quant", "MMLU↑", "PPL↓", "Code↑", "Math↑", "Consist↑", "Host", "Time"]
    rows = []
    for run in runs:
        m = run["metrics"]
        rows.append([
            run["checkpoint_name"][:20],
            run["quant_mode"],
            f"{m.get('mmlu', {}).get('value', 'N/A'):.4f}" if "mmlu" in m else "N/A",
            f"{m.get('language_modeling', {}).get('value', 'N/A'):.2f}" if "language_modeling" in m else "N/A",
            f"{m.get('code_continuation', {}).get('value', 'N/A'):.4f}" if "code_continuation" in m else "N/A",
            f"{m.get('math_prose', {}).get('value', 'N/A'):.4f}" if "math_prose" in m else "N/A",
            f"{m.get('consistency', {}).get('value', 'N/A'):.4f}" if "consistency" in m else "N/A",
            run["hostname"][:12],
            run["timestamp_utc"][:16],
        ])

    col_widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)

    print("\n" + "=" * (sum(col_widths) + 2 * (len(headers) - 1)))
    print("  EARLY WARNING — Eval Results Summary")
    print("=" * (sum(col_widths) + 2 * (len(headers) - 1)))
    print(fmt.format(*headers))
    print("-" * (sum(col_widths) + 2 * (len(headers) - 1)))
    for row in rows:
        print(fmt.format(*row))
    print("=" * (sum(col_widths) + 2 * (len(headers) - 1)))
    print(f"  Total runs: {agg['total_runs']}  |  Generated: {agg['generated_at']}")
    print()


def main() -> None:
    args = parse_args()

    if args.summary:
        if args.out.exists():
            with open(args.out) as f:
                agg = json.load(f)
            print_summary_table(agg)
        else:
            print(f"No aggregated file found at {args.out}. Run without --summary first.")
        return

    print(f"Scanning: {args.results_dir}")
    records = load_all_results(args.results_dir)
    print(f"Found {len(records)} result files.")

    agg = aggregate(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregated results saved to: {args.out}")
    print_summary_table(agg)


if __name__ == "__main__":
    main()
