import argparse
import datetime as _dt
import json
from pathlib import Path


METRICS = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]


def _coerce_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return None
    return None


def _paper_to_fraction(v):
    """Normalize paper values to 0-1 fractions.

    The IndicIFEval paper tables report accuracies as percentages (0-100).
    This helper accepts either representation:
      - 0-1 (already a fraction)
      - 0-100 (percentage)
    """

    f = _coerce_float(v)
    if f is None:
        return None
    if f > 1.0:
        # Treat as percent points.
        return f / 100.0
    return f


def _read_json(path: Path):
    # lm-eval sometimes writes UTF-8 with BOM on Windows.
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)


def _extract_task_metrics(results_json: dict) -> tuple[str, dict[str, float | str]]:
    results = results_json.get("results") or {}
    if not results:
        raise ValueError("results.json missing 'results' or it is empty")

    # Expect exactly one task per run directory.
    if len(results) != 1:
        task_names = ", ".join(sorted(results.keys()))
        raise ValueError(f"Expected 1 task in results.json, found {len(results)}: {task_names}")

    task_name = next(iter(results.keys()))
    task_obj = results[task_name] or {}

    extracted: dict[str, float | str] = {}
    for metric in METRICS:
        # lm-eval stores metric keys like: "prompt_level_strict_acc,none"
        key = f"{metric},none"
        val = task_obj.get(key, None)
        extracted[metric] = val

    return task_name, extracted


def _fmt(v: float | str | None) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a simple Hindi IndicIFEval comparison report (Trans vs Ground).")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--trans_dir", required=True, help="Run output directory for indicifeval_trans_hi")
    ap.add_argument("--ground_dir", required=True, help="Run output directory for indicifeval_ground_hi")
    ap.add_argument("--out", required=True, help="Markdown output file path")
    ap.add_argument(
        "--paper_json",
        default="",
        help=(
            "Optional JSON file containing paper baseline metrics to compare against. "
            "Expected shape: {\"indicifeval_trans_hi\": {metric: value, ...}, \"indicifeval_ground_hi\": {...}} "
            "where metric is one of: " + ", ".join(METRICS)
        ),
    )

    args = ap.parse_args()

    trans_dir = Path(args.trans_dir)
    ground_dir = Path(args.ground_dir)
    out_path = Path(args.out)

    trans_results_path = trans_dir / "results.json"
    ground_results_path = ground_dir / "results.json"

    if not trans_results_path.exists():
        raise FileNotFoundError(f"Missing {trans_results_path}")
    if not ground_results_path.exists():
        raise FileNotFoundError(f"Missing {ground_results_path}")

    trans_json = _read_json(trans_results_path)
    ground_json = _read_json(ground_results_path)

    trans_task, trans_metrics = _extract_task_metrics(trans_json)
    ground_task, ground_metrics = _extract_task_metrics(ground_json)

    now_utc = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    paper: dict = {}
    if args.paper_json:
        paper_path = Path(args.paper_json)
        if not paper_path.exists():
            raise FileNotFoundError(f"paper_json not found: {paper_path}")
        paper = _read_json(paper_path) or {}

    def paper_metrics_for(task_name: str) -> dict[str, float | str | None]:
        # Allow either exact task-name keys or shorthand keys.
        obj = paper.get(task_name)
        if obj is None:
            if "trans" in task_name:
                obj = paper.get("trans") or paper.get("indicifeval_trans_hi")
            elif "ground" in task_name:
                obj = paper.get("ground") or paper.get("indicifeval_ground_hi")
        if not isinstance(obj, dict):
            obj = {}
        out: dict[str, float | str | None] = {}
        for m in METRICS:
            out[m] = _paper_to_fraction(obj.get(m))
        return out

    trans_paper = paper_metrics_for(trans_task)
    ground_paper = paper_metrics_for(ground_task)

    lines: list[str] = []
    lines.append("# IndicIFEval Hindi (hi) — Full Report")
    lines.append("")
    lines.append(f"- Model: {args.model}")
    lines.append(f"- Generated: {now_utc}")
    lines.append(f"- Trans run dir: {trans_dir.as_posix()}")
    lines.append(f"- Ground run dir: {ground_dir.as_posix()}")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| Task | Split | prompt_level_strict_acc | inst_level_strict_acc | prompt_level_loose_acc | inst_level_loose_acc |")
    lines.append("|---|---|---:|---:|---:|---:|")

    lines.append(
        "| "
        + trans_task
        + " | model | "
        + " | ".join(_fmt(trans_metrics[m]) for m in METRICS)
        + " |"
    )
    if args.paper_json:
        lines.append(
            "| "
            + trans_task
            + " | paper | "
            + " | ".join(_fmt(trans_paper[m]) for m in METRICS)
            + " |"
        )
    lines.append(
        "| "
        + ground_task
        + " | model | "
        + " | ".join(_fmt(ground_metrics[m]) for m in METRICS)
        + " |"
    )
    if args.paper_json:
        lines.append(
            "| "
            + ground_task
            + " | paper | "
            + " | ".join(_fmt(ground_paper[m]) for m in METRICS)
            + " |"
        )

    # Simple mean across the two runs (only if values are numeric).
    avg_vals: list[str] = []
    for m in METRICS:
        a = trans_metrics.get(m)
        b = ground_metrics.get(m)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            avg_vals.append(f"{((float(a) + float(b)) / 2.0):.4f}")
        else:
            avg_vals.append("N/A")

    lines.append("| **avg(trans, ground)** | model | " + " | ".join(avg_vals) + " |")
    lines.append("")

    lines.append("## Paper comparison")
    lines.append("")

    if not args.paper_json:
        lines.append(
            "This report can compute paper-vs-model deltas if you provide `--paper_json`. "
            "Values may be specified either as fractions (0-1) or percentages (0-100). Example:\n"
        )
        lines.append("```json")
        lines.append("{")
        lines.append('  "indicifeval_trans_hi": {')
        lines.append('    "prompt_level_loose_acc": 30.5')
        lines.append("  },")
        lines.append('  "indicifeval_ground_hi": {')
        lines.append('    "prompt_level_loose_acc": 56.9')
        lines.append("  }")
        lines.append("}")
        lines.append("```")
        lines.append("")
    else:
        # Focus on the metric that the paper tables report: prompt-level loose accuracy.
        def pct(v):
            if isinstance(v, (int, float)):
                return float(v) * 100.0
            return None

        def fmt_pct(v):
            return "N/A" if v is None else f"{v:.1f}"

        trans_model_pct = pct(trans_metrics.get("prompt_level_loose_acc"))
        trans_paper_pct = pct(trans_paper.get("prompt_level_loose_acc"))
        ground_model_pct = pct(ground_metrics.get("prompt_level_loose_acc"))
        ground_paper_pct = pct(ground_paper.get("prompt_level_loose_acc"))

        def delta_pp(model_pct, paper_pct):
            if model_pct is None or paper_pct is None:
                return None
            return model_pct - paper_pct

        lines.append(
            "Paper reference: https://arxiv.org/html/2602.22125v1 (Table 1 = Trans, Table 2 = Ground)."
        )
        lines.append("")
        lines.append("| Task | Metric | Model (%) | Paper (%) | Δ (pp) |")
        lines.append("|---|---|---:|---:|---:|")
        lines.append(
            "| "
            + trans_task
            + " | prompt_level_loose_acc | "
            + fmt_pct(trans_model_pct)
            + " | "
            + fmt_pct(trans_paper_pct)
            + " | "
            + ("N/A" if delta_pp(trans_model_pct, trans_paper_pct) is None else f"{delta_pp(trans_model_pct, trans_paper_pct):+.1f}")
            + " |"
        )
        lines.append(
            "| "
            + ground_task
            + " | prompt_level_loose_acc | "
            + fmt_pct(ground_model_pct)
            + " | "
            + fmt_pct(ground_paper_pct)
            + " | "
            + ("N/A" if delta_pp(ground_model_pct, ground_paper_pct) is None else f"{delta_pp(ground_model_pct, ground_paper_pct):+.1f}")
            + " |"
        )
        lines.append("")
        lines.append("Notes:")
        lines.append("- The paper reports prompt-level *loose* accuracy as percentages; this report compares against `prompt_level_loose_acc * 100`." )
        lines.append("- Table 1 (Trans) uses a curated common subset of 321 prompts per language; your local run may use a different set/config.")
        lines.append("- Table 2 (Ground) is not directly comparable to Trans (different prompts, no English baseline).")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
