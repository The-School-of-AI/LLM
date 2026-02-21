#!/usr/bin/env python3
"""
Phase 2/3 — Early Warning Report Generator.

Reads the trend database and produces:
  1. A structured JSON early warning report
  2. A human-readable Markdown report
  3. An HTML report (with embedded trend plots, if available)

Reports are saved to results/reports/.

Usage:
    python scripts/generate_report.py [--trend-db results/trend_db.json] [--out-dir results/reports]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import load_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

METRIC_LABELS = {
    "mmlu_accuracy":    ("MMLU Accuracy",        "↑"),
    "lm_perplexity":    ("LM Perplexity",         "↓"),
    "code_pass_rate":   ("Code Pass Rate",        "↑"),
    "math_accuracy":    ("Math Accuracy",         "↑"),
    "consistency_rate": ("Consistency Rate",      "↑"),
}

SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate early warning report")
    p.add_argument("--trend-db", type=Path, default=ROOT / "results" / "trend_db.json")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "reports")
    p.add_argument("--config", default=None)
    return p.parse_args()


def classify_severity(anomaly_type: str, delta: float | None = None) -> str:
    if anomaly_type == "REGRESSION":
        if delta is not None and abs(delta) > 0.05:
            return "HIGH"
        return "MEDIUM"
    if anomaly_type == "INSTABILITY":
        return "MEDIUM"
    return "LOW"   # PLATEAU


def confidence_level(num_anomalies: int, num_checkpoints: int, cfg: dict) -> tuple[str, float]:
    """Estimate confidence of early warning based on data volume."""
    thresholds = cfg["early_warning"]["confidence_levels"]
    if num_checkpoints < 3:
        score = thresholds["low"]
        label = "LOW"
    elif num_checkpoints < 6:
        score = thresholds["medium"]
        label = "MEDIUM"
    else:
        score = thresholds["high"]
        label = "HIGH"
    # Adjust down if very few anomalies
    if num_anomalies == 0:
        score *= 0.5
    return label, round(score, 2)


def build_warning_items(anomalies: dict, checkpoints: list) -> list[dict]:
    items = []
    for metric, anom_list in anomalies.items():
        label, direction = METRIC_LABELS.get(metric, (metric, "?"))
        for anom in anom_list:
            severity = classify_severity(anom["type"], anom.get("delta"))
            items.append({
                "metric": metric,
                "metric_label": label,
                "type": anom["type"],
                "severity": severity,
                "checkpoint": anom.get("checkpoint", "?"),
                "detail": anom.get("detail", ""),
                "delta": anom.get("delta"),
                "raw": anom,
            })
    items.sort(key=lambda x: (SEVERITY_ORDER.get(x["severity"], 9), x["metric"]))
    return items


def generate_markdown_report(
    trend_db: dict,
    warning_items: list[dict],
    conf_label: str,
    conf_score: float,
    cfg: dict,
) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    checkpoints = trend_db.get("checkpoints", [])
    teams = cfg["early_warning"]["notify_teams"]

    lines = [
        f"# Early Warning Report",
        f"",
        f"**Generated:** {ts}  ",
        f"**Notifying:** {', '.join(teams)}  ",
        f"**Checkpoints evaluated:** {len(checkpoints)}  ",
        f"**Confidence:** {conf_label} ({conf_score:.0%})  ",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
    ]

    if not warning_items:
        lines += [
            f"✅ **No regressions, plateaus, or instability detected.**",
            f"",
            f"All metrics are trending normally across {len(checkpoints)} checkpoints.",
            f"",
        ]
    else:
        high = [w for w in warning_items if w["severity"] == "HIGH"]
        med  = [w for w in warning_items if w["severity"] == "MEDIUM"]
        low  = [w for w in warning_items if w["severity"] == "LOW"]

        if high:
            lines.append(f"🔴 **{len(high)} HIGH severity warning(s)** — immediate attention required.")
        if med:
            lines.append(f"🟠 **{len(med)} MEDIUM severity warning(s)**.")
        if low:
            lines.append(f"🟡 **{len(low)} LOW severity warning(s)** (plateaus / minor issues).")
        lines.append("")

    # Warnings table
    if warning_items:
        lines += [
            f"## Warning Details",
            f"",
            f"| Severity | Type | Metric | Checkpoint | Detail |",
            f"|----------|------|--------|------------|--------|",
        ]
        for w in warning_items:
            sev_icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}.get(w["severity"], "⚪")
            delta_str = f" (Δ={w['delta']:+.4f})" if w.get("delta") is not None else ""
            lines.append(
                f"| {sev_icon} {w['severity']} | {w['type']} | {w['metric_label']} | "
                f"`{w['checkpoint']}` | {w['detail']}{delta_str} |"
            )
        lines.append("")

    # Metric trends
    lines += ["## Metric Trends", ""]
    series = trend_db.get("metric_series", {})
    deltas = trend_db.get("deltas", {})

    for metric, pts in series.items():
        label, direction = METRIC_LABELS.get(metric, (metric, "?"))
        if not pts:
            continue
        lines.append(f"### {label} ({direction})")
        lines.append("")
        lines.append(f"| Checkpoint | Value | Δ vs prev |")
        lines.append(f"|------------|-------|-----------|")
        delta_map = {d["to_checkpoint"]: d["improvement"] for d in deltas.get(metric, [])}
        for pt in pts:
            ckpt = pt["checkpoint"]
            val  = pt["value"]
            imp  = delta_map.get(ckpt)
            imp_str = f"{imp:+.5f}" if imp is not None else "—"
            lines.append(f"| `{ckpt}` | {val:.5f} | {imp_str} |")
        lines.append("")

    # NLL deltas section
    if "lm_perplexity" in series and series["lm_perplexity"]:
        lines += ["## NLL / Perplexity Deltas", ""]
        lm_deltas = deltas.get("lm_perplexity", [])
        if lm_deltas:
            lines.append("| From | To | Δ Perplexity | Interpretation |")
            lines.append("|------|----|-------------|----------------|")
            for d in lm_deltas:
                interp = "Improved ✓" if d["improvement"] > 0 else ("Worsened ✗" if d["improvement"] < -0.01 else "Stable ~")
                lines.append(f"| `{d['from_checkpoint']}` | `{d['to_checkpoint']}` | {d['raw_delta']:+.4f} | {interp} |")
            lines.append("")

    lines += [
        "---",
        "",
        f"*Report generated by Team 16 Early Warning System — Experiment 1*",
        f"*Confidence: {conf_label} ({conf_score:.0%}) based on {len(checkpoints)} checkpoints*",
    ]

    return "\n".join(lines)


def generate_html_report(
    md_content: str,
    warning_items: list[dict],
    plots_dir: Path,
    out_dir: Path,
) -> str:
    # Embed plot images if available
    plot_html = ""
    png_plots = sorted(plots_dir.glob("trend_*.png")) if plots_dir.exists() else []
    if png_plots:
        plot_html = "<h2>Trend Plots</h2>\n"
        for p in png_plots:
            # Use relative path
            rel = p.relative_to(out_dir.parent) if out_dir.parent in p.parents else p
            plot_html += f'<img src="../plots/{p.name}" style="max-width:100%;margin:1em 0;border:1px solid #444;" />\n'

    html_dashboard = ""
    dashboard_html = plots_dir / "trend_dashboard.html"
    if dashboard_html.exists():
        html_dashboard = f'<p><a href="../plots/trend_dashboard.html" target="_blank">🔗 Open Interactive Dashboard</a></p>\n'

    severity_colors = {"HIGH": "#f38ba8", "MEDIUM": "#fab387", "LOW": "#f9e2af"}
    warn_rows = ""
    for w in warning_items:
        color = severity_colors.get(w["severity"], "#cdd6f4")
        warn_rows += (
            f"<tr style='border-left:4px solid {color}'>"
            f"<td><b>{w['severity']}</b></td>"
            f"<td>{w['type']}</td>"
            f"<td>{w['metric_label']}</td>"
            f"<td><code>{w['checkpoint']}</code></td>"
            f"<td>{w['detail']}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Early Warning Report</title>
  <style>
    body {{ font-family: 'Segoe UI', monospace; background: #1e1e2e; color: #cdd6f4; padding: 2em; max-width: 1200px; margin: auto; }}
    h1,h2,h3 {{ color: #89b4fa; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
    th {{ background: #313244; padding: 8px 12px; text-align: left; color: #89dceb; }}
    td {{ padding: 6px 12px; border-bottom: 1px solid #313244; }}
    code {{ background: #313244; padding: 2px 6px; border-radius: 4px; }}
    .high {{ color: #f38ba8; }} .med {{ color: #fab387; }} .low {{ color: #f9e2af; }}
    pre {{ background: #313244; padding: 1em; overflow-x: auto; border-radius: 6px; }}
    a {{ color: #89b4fa; }}
  </style>
</head>
<body>
  <h1>🔍 Early Warning Report</h1>
  {html_dashboard}
  {'<h2>Warnings</h2><table><tr><th>Severity</th><th>Type</th><th>Metric</th><th>Checkpoint</th><th>Detail</th></tr>' + warn_rows + '</table>' if warning_items else '<p style="color:#a6e3a1">✅ No anomalies detected.</p>'}
  {plot_html}
  <h2>Full Report (Markdown)</h2>
  <pre>{md_content}</pre>
</body>
</html>"""
    return html


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if not args.trend_db.exists():
        print(f"ERROR: Trend DB not found: {args.trend_db}")
        print("Run: python scripts/track_trends.py first.")
        sys.exit(1)

    with open(args.trend_db) as f:
        trend_db = json.load(f)

    checkpoints = trend_db.get("checkpoints", [])
    anomalies = trend_db.get("anomalies", {})

    warning_items = build_warning_items(anomalies, checkpoints)
    conf_label, conf_score = confidence_level(len(warning_items), len(checkpoints), cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # JSON report
    json_report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "confidence": {"label": conf_label, "score": conf_score},
        "num_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
        "num_warnings": len(warning_items),
        "warnings": warning_items,
        "notify_teams": cfg["early_warning"]["notify_teams"],
    }
    json_path = args.out_dir / f"early_warning_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)
    logger.info(f"JSON report: {json_path}")

    # Markdown report
    md_content = generate_markdown_report(trend_db, warning_items, conf_label, conf_score, cfg)
    md_path = args.out_dir / f"early_warning_{ts}.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info(f"Markdown report: {md_path}")

    # Also write a stable "latest" symlink-equivalent (overwrite)
    for suffix in (".json", ".md"):
        latest = args.out_dir / f"latest_early_warning{suffix}"
        src = json_path if suffix == ".json" else md_path
        with open(src) as fr, open(latest, "w") as fw:
            fw.write(fr.read())

    # HTML report
    plots_dir = ROOT / "results" / "plots"
    html_content = generate_html_report(md_content, warning_items, plots_dir, args.out_dir)
    html_path = args.out_dir / f"early_warning_{ts}.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    latest_html = args.out_dir / "latest_early_warning.html"
    with open(latest_html, "w") as f:
        f.write(html_content)
    logger.info(f"HTML report: {html_path}")

    # Print to console
    print("\n" + "=" * 70)
    print(md_content)
    print("=" * 70)
    print(f"\nReports saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
