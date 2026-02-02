#!/usr/bin/env python3
"""
Update validation_report.md with metrics JSON from train_tiny_dashboard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BEGIN = "<!-- BEGIN AUTO:tiny_validation -->"
END = "<!-- END AUTO:tiny_validation -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON")
    parser.add_argument(
        "--report",
        default="moe_deliverables/validation_report.md",
        help="Path to validation_report.md",
    )
    return parser.parse_args()


def format_block(metrics: dict) -> str:
    gates = metrics.get("routing_health_gates", {})
    alerts = metrics.get("alerts", [])

    lines = [
        "## Tiny Validation Run (Auto-Generated)",
        "",
        f"- Config: `{metrics.get('config')}`",
        f"- Steps: {metrics.get('steps')}",
        f"- Batch × Seq: {metrics.get('batch_size')} × {metrics.get('seq_len')}",
        f"- Avg Loss: {metrics.get('avg_loss')}",
        f"- Final Loss: {metrics.get('final_loss')}",
        f"- Avg Tokens/sec: {metrics.get('avg_tokens_per_sec')}",
        f"- Final Tokens/sec: {metrics.get('final_tokens_per_sec')}",
        "",
        "### Routing Health Gates",
    ]
    if gates:
        for k, v in gates.items():
            lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    else:
        lines.append("- No gate data available")

    lines.extend(
        [
            "",
            "### Null-on-Junk Stats",
            f"- Junk→Null (%): {metrics.get('null_on_junk_pct')}",
            f"- Signal→Null (%): {metrics.get('null_on_signal_pct')}",
            "",
            "### Instability Signatures",
        ]
    )

    if alerts:
        for alert in alerts:
            lines.append(f"- {alert.get('severity')}: {alert.get('message')}")
    else:
        lines.append("- None detected")

    lines.extend(
        [
            "",
            "### Mitigation Decisions",
            "- None (tiny sanity run)",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    report_path = Path(args.report)
    report_text = report_path.read_text(encoding="utf-8")

    block = format_block(metrics)
    if BEGIN in report_text and END in report_text:
        before, rest = report_text.split(BEGIN, 1)
        _, after = rest.split(END, 1)
        new_text = before + BEGIN + "\n" + block + "\n" + END + after
    else:
        new_text = report_text.rstrip() + "\n\n" + BEGIN + "\n" + block + "\n" + END + "\n"

    report_path.write_text(new_text, encoding="utf-8")


if __name__ == "__main__":
    main()
