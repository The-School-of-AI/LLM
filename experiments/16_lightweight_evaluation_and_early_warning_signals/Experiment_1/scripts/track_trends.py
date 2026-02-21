#!/usr/bin/env python3
"""
Phase 2 / 3 — Trend Tracking System.

Reads the aggregated results file (or raw results directory) and:
  1. Builds a trend database (results/trend_db.json)
  2. Detects regressions, plateaus, and instability
  3. Generates checkpoint-vs-checkpoint trend plots (matplotlib + plotly)

Usage:
    python scripts/track_trends.py [--config configs/eval_config.yaml] [--plot-format html|png|both]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import load_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


METRICS = {
    "mmlu_accuracy":        ("mmlu",               "overall_accuracy",     "↑"),
    "lm_perplexity":        ("language_modeling",   "mean_perplexity",      "↓"),
    "code_pass_rate":       ("code_continuation",   "pass_rate",            "↑"),
    "math_accuracy":        ("math_prose",          "accuracy",             "↑"),
    "consistency_rate":     ("consistency",         "mean_agreement_rate",  "↑"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trend tracking and plotting")
    p.add_argument("--config", default=None)
    p.add_argument("--agg-file", type=Path, default=ROOT / "results" / "aggregated_results.json")
    p.add_argument("--out-plots", type=Path, default=ROOT / "results" / "plots")
    p.add_argument("--trend-db", type=Path, default=ROOT / "results" / "trend_db.json")
    p.add_argument("--plot-format", choices=["html", "png", "both"], default="both")
    return p.parse_args()


def load_aggregated(agg_file: Path) -> list[dict]:
    """Load the aggregated results and return sorted list of run summaries."""
    with open(agg_file) as f:
        agg = json.load(f)
    runs = agg.get("runs", [])
    runs.sort(key=lambda r: r.get("timestamp_utc", ""))
    return runs


def extract_metric_series(runs: list[dict]) -> dict[str, list[tuple[str, float]]]:
    """
    Returns {metric_name: [(checkpoint_name, value), ...]} sorted by time.
    """
    series: dict[str, list] = {m: [] for m in METRICS}
    for run in runs:
        ckpt = run.get("checkpoint_name", "?")
        m = run.get("metrics", {})
        for metric_name, (eval_key, val_key, _) in METRICS.items():
            if eval_key in m:
                val = m[eval_key].get("value")
                if val is not None:
                    series[metric_name].append((ckpt, float(val)))
    return series


def detect_anomalies(
    series: dict[str, list[tuple[str, float]]],
    cfg: dict,
) -> dict[str, list[dict]]:
    """
    Detect regressions, plateaus, and instability for each metric.

    Returns {metric_name: [{"checkpoint": ..., "type": ..., "detail": ...}]}
    """
    trend_cfg = cfg["trend_tracking"]
    reg_thresh = trend_cfg["regression_threshold"]        # e.g. -0.02
    plateau_win = trend_cfg["plateau_window"]             # e.g. 3
    plateau_thresh = trend_cfg["plateau_threshold"]       # e.g. 0.005
    instab_std = trend_cfg["instability_std_threshold"]   # e.g. 0.05

    anomalies: dict[str, list] = {m: [] for m in METRICS}

    for metric_name, points in series.items():
        _, direction = METRICS[metric_name][2], METRICS[metric_name][2]
        is_lower_better = METRICS[metric_name][2] == "↓"

        for i in range(1, len(points)):
            prev_ckpt, prev_val = points[i - 1]
            curr_ckpt, curr_val = points[i]

            # Delta: positive = improvement for ↑ metrics, negative for ↓
            if is_lower_better:
                delta = prev_val - curr_val   # lower is better → improvement = drop
            else:
                delta = curr_val - prev_val   # higher is better → improvement = rise

            # Regression: meaningful drop in performance
            if delta < reg_thresh:
                anomalies[metric_name].append({
                    "checkpoint": curr_ckpt,
                    "type": "REGRESSION",
                    "delta": round(delta, 5),
                    "prev_val": prev_val,
                    "curr_val": curr_val,
                    "detail": f"{metric_name} dropped by {abs(delta):.4f} from {prev_ckpt} to {curr_ckpt}",
                })

        # Plateau detection over rolling window
        if len(points) >= plateau_win:
            window_vals = [v for _, v in points[-plateau_win:]]
            window_range = max(window_vals) - min(window_vals)
            if window_range < plateau_thresh:
                anomalies[metric_name].append({
                    "checkpoint": points[-1][0],
                    "type": "PLATEAU",
                    "window": plateau_win,
                    "range": round(window_range, 6),
                    "detail": f"{metric_name} plateaued over last {plateau_win} checkpoints (range={window_range:.5f})",
                })

        # Instability: high std dev over the full series
        if len(points) >= 3:
            vals = [v for _, v in points]
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            if std > instab_std:
                anomalies[metric_name].append({
                    "checkpoint": points[-1][0],
                    "type": "INSTABILITY",
                    "std": round(std, 5),
                    "detail": f"{metric_name} shows high variance (std={std:.4f}) across checkpoints",
                })

    return anomalies


def compute_deltas(series: dict[str, list[tuple[str, float]]]) -> dict[str, list[dict]]:
    """Compute checkpoint-over-checkpoint deltas for each metric."""
    deltas: dict[str, list] = {}
    for metric_name, points in series.items():
        is_lower_better = METRICS[metric_name][2] == "↓"
        delta_list = []
        for i in range(1, len(points)):
            prev_ckpt, prev_val = points[i - 1]
            curr_ckpt, curr_val = points[i]
            raw_delta = curr_val - prev_val
            if is_lower_better:
                improvement = -raw_delta   # negative change = improvement
            else:
                improvement = raw_delta

            delta_list.append({
                "from_checkpoint": prev_ckpt,
                "to_checkpoint": curr_ckpt,
                "raw_delta": round(raw_delta, 6),
                "improvement": round(improvement, 6),
            })
        deltas[metric_name] = delta_list
    return deltas


def build_trend_db(runs: list[dict], series: dict, deltas: dict, anomalies: dict) -> dict:
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "checkpoints": [r["checkpoint_name"] for r in runs],
        "metric_series": {m: [{"checkpoint": c, "value": v} for c, v in pts] for m, pts in series.items()},
        "deltas": deltas,
        "anomalies": anomalies,
        "anomaly_summary": {
            m: [a["type"] for a in alist] for m, alist in anomalies.items() if alist
        },
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_matplotlib(
    series: dict[str, list[tuple[str, float]]],
    anomalies: dict[str, list[dict]],
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not installed; skipping PNG plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"REGRESSION": "red", "PLATEAU": "orange", "INSTABILITY": "purple"}

    for metric_name, points in series.items():
        if len(points) < 2:
            continue

        checkpoints = [p[0] for p in points]
        values = [p[1] for p in points]
        x = list(range(len(checkpoints)))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, values, marker="o", linewidth=2, color="#4c9be8", label=metric_name)
        ax.fill_between(x, values, alpha=0.15, color="#4c9be8")

        # Mark anomalies
        ann_handles = []
        for anom in anomalies.get(metric_name, []):
            ckpt = anom["checkpoint"]
            if ckpt in checkpoints:
                xi = checkpoints.index(ckpt)
                color = colors.get(anom["type"], "gray")
                ax.axvline(xi, color=color, linestyle="--", alpha=0.7)
                ax.annotate(
                    anom["type"],
                    xy=(xi, values[xi]),
                    xytext=(xi + 0.15, values[xi]),
                    fontsize=7,
                    color=color,
                )
                ann_handles.append(mpatches.Patch(color=color, label=anom["type"]))

        ax.set_xticks(x)
        ax.set_xticklabels(checkpoints, rotation=45, ha="right", fontsize=8)
        direction = METRICS[metric_name][2]
        ax.set_ylabel(f"{metric_name} ({direction})")
        ax.set_title(f"Trend: {metric_name}")
        ax.grid(True, alpha=0.3)
        if ann_handles:
            ax.legend(handles=ann_handles, fontsize=8, loc="best")

        plt.tight_layout()
        fig.savefig(out_dir / f"trend_{metric_name}.png", dpi=150)
        plt.close(fig)
        logger.info(f"  Saved PNG: trend_{metric_name}.png")

    # Combined overview plot
    n = len([m for m, pts in series.items() if len(pts) >= 2])
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]
    ax_idx = 0
    for metric_name, points in series.items():
        if len(points) < 2:
            continue
        ax = axes[ax_idx]
        checkpoints = [p[0] for p in points]
        values = [p[1] for p in points]
        x = list(range(len(checkpoints)))
        ax.plot(x, values, marker="o", linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(checkpoints, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(metric_name, fontsize=8)
        ax.grid(True, alpha=0.3)
        for anom in anomalies.get(metric_name, []):
            ckpt = anom["checkpoint"]
            if ckpt in checkpoints:
                xi = checkpoints.index(ckpt)
                color = colors.get(anom["type"], "gray")
                ax.axvline(xi, color=color, linestyle="--", alpha=0.6)
        ax_idx += 1

    fig.suptitle("Team 16 Early Warning — All Metric Trends", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "trend_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved PNG: trend_overview.png")


def plot_plotly(
    series: dict[str, list[tuple[str, float]]],
    anomalies: dict[str, list[dict]],
    out_dir: Path,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        logger.warning("plotly not installed; skipping HTML plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    anom_color = {"REGRESSION": "red", "PLATEAU": "orange", "INSTABILITY": "purple"}

    valid_metrics = [(m, pts) for m, pts in series.items() if len(pts) >= 2]
    if not valid_metrics:
        return

    n = len(valid_metrics)
    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[m for m, _ in valid_metrics],
        shared_xaxes=False,
        vertical_spacing=0.08,
    )

    for row, (metric_name, points) in enumerate(valid_metrics, 1):
        checkpoints = [p[0] for p in points]
        values = [p[1] for p in points]
        direction = METRICS[metric_name][2]

        fig.add_trace(
            go.Scatter(
                x=checkpoints,
                y=values,
                mode="lines+markers",
                name=f"{metric_name} ({direction})",
                hovertemplate="%{x}<br>%{y:.5f}<extra></extra>",
            ),
            row=row, col=1,
        )

        # Anomaly vertical lines
        for anom in anomalies.get(metric_name, []):
            ckpt = anom["checkpoint"]
            if ckpt in checkpoints:
                color = anom_color.get(anom["type"], "gray")
                fig.add_vline(
                    x=ckpt,
                    line_color=color,
                    line_dash="dash",
                    opacity=0.6,
                    annotation_text=anom["type"],
                    annotation_font_size=10,
                    row=row, col=1,
                )

    fig.update_layout(
        height=350 * n,
        title_text="Team 16 Early Warning — Checkpoint Trend Dashboard",
        showlegend=True,
    )
    out_path = out_dir / "trend_dashboard.html"
    fig.write_html(str(out_path))
    logger.info(f"  Saved HTML: trend_dashboard.html")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if not args.agg_file.exists():
        print(f"ERROR: Aggregated results file not found: {args.agg_file}")
        print("Run: python collector/collect_results.py first.")
        sys.exit(1)

    runs = load_aggregated(args.agg_file)
    logger.info(f"Loaded {len(runs)} runs from {args.agg_file}")

    series = extract_metric_series(runs)
    deltas = compute_deltas(series)
    anomalies = detect_anomalies(series, cfg)
    trend_db = build_trend_db(runs, series, deltas, anomalies)

    # Save trend DB
    args.trend_db.parent.mkdir(parents=True, exist_ok=True)
    with open(args.trend_db, "w") as f:
        json.dump(trend_db, f, indent=2)
    logger.info(f"Trend DB saved: {args.trend_db}")

    # Print anomaly summary
    print("\n=== Anomaly Summary ===")
    any_anomaly = False
    for metric, anoms in anomalies.items():
        if anoms:
            any_anomaly = True
            for a in anoms:
                print(f"  [{a['type']}] {a['detail']}")
    if not any_anomaly:
        print("  No anomalies detected.")

    # Plots
    args.out_plots.mkdir(parents=True, exist_ok=True)
    if args.plot_format in ("png", "both"):
        logger.info("Generating matplotlib PNG plots ...")
        plot_matplotlib(series, anomalies, args.out_plots)
    if args.plot_format in ("html", "both"):
        logger.info("Generating plotly HTML dashboard ...")
        plot_plotly(series, anomalies, args.out_plots)

    print(f"\nPlots saved to: {args.out_plots}")


if __name__ == "__main__":
    main()
