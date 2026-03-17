"""
Stage 2 — Cleaning & Filtering.

Runs a configurable chain of filters on the validated dataset.
Each filter is independently enabled/disabled via config.
Removed examples are written to the rejected file with a reason tag.
A funnel report is accumulated via FunnelTracker.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from pipeline.config import PipelineConfig, Stage2Config
from pipeline.funnel_tracker import FunnelTracker
from pipeline.filters.base import BaseFilter
from pipeline.io.readers import iter_records

logger = logging.getLogger(__name__)


def _build_filter_chain(cfg: Stage2Config, work_dir: Path) -> list[BaseFilter]:
    """Instantiate enabled filters in execution order."""
    filters: list[BaseFilter] = []

    if cfg.length_filter.enabled:
        from pipeline.filters.length_filter import LengthFilter
        filters.append(LengthFilter(cfg.length_filter))

    if cfg.lang_filter.enabled:
        from pipeline.filters.lang_filter import LangFilter
        filters.append(LangFilter(cfg.lang_filter))

    if cfg.exact_dedup.enabled:
        from pipeline.filters.exact_dedup import ExactDedup
        filters.append(ExactDedup(cfg.exact_dedup))

    if cfg.near_dedup.enabled:
        from pipeline.filters.near_dedup import NearDedup
        filters.append(NearDedup(cfg.near_dedup))

    if cfg.toxicity_filter.enabled:
        from pipeline.filters.toxicity_filter import ToxicityFilter
        filters.append(ToxicityFilter(cfg.toxicity_filter))

    if cfg.pii_filter.enabled:
        from pipeline.filters.pii_filter import PIIFilter
        filters.append(PIIFilter(cfg.pii_filter))

    if cfg.repetition_filter.enabled:
        from pipeline.filters.repetition_filter import RepetitionFilter
        filters.append(RepetitionFilter(cfg.repetition_filter))

    if cfg.slop_filter.enabled:
        from pipeline.filters.slop_filter import SlopFilter
        filters.append(SlopFilter(cfg.slop_filter))

    if cfg.benchmark_decontam.enabled:
        from pipeline.filters.benchmark_decontam import BenchmarkDecontam
        # Resolve relative paths for the benchmark hashes dir
        bc = cfg.benchmark_decontam
        filters.append(BenchmarkDecontam(bc))

    logger.info("Stage 2: filter chain = [%s]", ", ".join(f.name for f in filters))
    return filters


def run(cfg: PipelineConfig, tracker: FunnelTracker | None = None) -> None:
    stage_cfg = cfg.stage2
    if not stage_cfg.enabled:
        logger.info("Stage 2 (clean) disabled — skipping")
        return

    input_path  = cfg.work_path(stage_cfg.input_file)
    output_path = cfg.work_path(stage_cfg.output_file)
    reject_path = cfg.work_path(stage_cfg.rejected_file)
    decontam_removed_path = cfg.work_path(stage_cfg.benchmark_decontam.removed_out)

    if cfg.globals.save_intermediates and output_path.exists() and cfg.globals.resume_from_stage > 2:
        logger.info("Stage 2: output %s exists, resuming from next stage", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    filter_chain = _build_filter_chain(stage_cfg, Path(cfg.globals.work_dir))

    total = kept = 0
    filter_drop_counts: dict[str, int] = {}

    with (
        open(output_path, "w", encoding="utf-8") as fout,
        open(reject_path, "w", encoding="utf-8") as frej,
        open(decontam_removed_path, "w", encoding="utf-8") as fdecontam,
    ):
        for record in iter_records(input_path):
            total += 1
            dropped = False

            for flt in filter_chain:
                keep, reason = flt.filter(record)
                if not keep:
                    # Route benchmark decontam removals to a separate file
                    if flt.name == "benchmark_decontam":
                        record["_reject_reason"] = reason
                        fdecontam.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        record["_reject_reason"] = reason
                        frej.write(json.dumps(record, ensure_ascii=False) + "\n")

                    if tracker:
                        tracker.record_drop("stage2", flt.name, reason)
                    filter_drop_counts[flt.name] = filter_drop_counts.get(flt.name, 0) + 1
                    dropped = True
                    break

            if not dropped:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    if tracker:
        tracker.record_stage_output("stage2", kept)

    logger.info("Stage 2 complete: total=%d, kept=%d", total, kept)
    for fname, count in filter_drop_counts.items():
        logger.info("  %s dropped: %d", fname, count)
    logger.info("Output: %s", output_path)
