"""
FunnelTracker — shared drop reporter passed across all pipeline stages.
Records how many examples each filter/stage drops and why.
Produces a structured JSON report and a rich console table at pipeline end.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class StageDrops:
    stage_name: str
    filter_name: str
    count: int = 0
    example_reasons: list[str] = field(default_factory=list)
    _max_reasons: int = 50

    def add(self, reason: str = "") -> None:
        self.count += 1
        if len(self.example_reasons) < self._max_reasons:
            self.example_reasons.append(reason)


class FunnelTracker:
    """
    Thread-safe funnel reporter.

    Usage inside a stage/filter:
        tracker.record_drop("stage2", "near_dedup", reason="Jaccard=0.92 vs key='abc'")
        tracker.record_stage_output("stage2", 95000)
    """

    def __init__(self, total_input: int, output_path: Optional[Path] = None) -> None:
        self._lock = threading.Lock()
        self.total_input = total_input
        self.output_path = output_path
        # Ordered insertion → stage order preserved
        self._drops: dict[str, StageDrops] = {}
        self._stage_output_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record_drop(self, stage: str, filter_name: str, reason: str = "") -> None:
        key = f"{stage}.{filter_name}"
        with self._lock:
            if key not in self._drops:
                self._drops[key] = StageDrops(stage, filter_name)
            self._drops[key].add(reason)

    def record_stage_output(self, stage: str, count: int) -> None:
        with self._lock:
            self._stage_output_counts[stage] = count

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> dict:
        """Return a structured report dict (JSON-serialisable)."""
        rows = []
        running = self.total_input
        for drops in self._drops.values():
            rows.append({
                "stage_filter": f"{drops.stage_name}.{drops.filter_name}",
                "dropped": drops.count,
                "remaining_after": running - drops.count,
                "drop_rate_pct": round(100.0 * drops.count / max(1, running), 2),
                "sample_reasons": drops.example_reasons[:5],
            })
            running -= drops.count

        return {
            "total_input": self.total_input,
            "total_output": running,
            "overall_retention_pct": round(100.0 * running / max(1, self.total_input), 2),
            "stage_output_counts": dict(self._stage_output_counts),
            "stages": rows,
        }

    def save(self) -> None:
        """Write JSON report and print a rich table to the console."""
        report = self.report()
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "w") as f:
                json.dump(report, f, indent=2)
        self._print_rich_table(report)

    def _print_rich_table(self, report: dict) -> None:
        try:
            from rich.table import Table
            from rich.console import Console

            table = Table(title="Pipeline Funnel Report", show_lines=True)
            table.add_column("Stage.Filter", style="cyan", no_wrap=True)
            table.add_column("Dropped", justify="right", style="red")
            table.add_column("Remaining", justify="right", style="green")
            table.add_column("Drop %", justify="right")

            for row in report["stages"]:
                table.add_row(
                    row["stage_filter"],
                    str(row["dropped"]),
                    str(row["remaining_after"]),
                    f"{row['drop_rate_pct']:.1f}%",
                )

            table.add_section()
            table.add_row(
                "[bold]TOTAL[/bold]",
                str(report["total_input"] - report["total_output"]),
                str(report["total_output"]),
                f"{100 - report['overall_retention_pct']:.1f}% lost",
            )
            Console().print(table)
        except ImportError:
            # Fallback: plain text
            print("\n=== Pipeline Funnel Report ===")
            for row in report["stages"]:
                print(f"  {row['stage_filter']:40s}  dropped={row['dropped']:6d}  "
                      f"remaining={row['remaining_after']:6d}  ({row['drop_rate_pct']:.1f}%)")
            print(f"  TOTAL: {report['total_input']} → {report['total_output']} "
                  f"({report['overall_retention_pct']:.1f}% retained)")
            print()
