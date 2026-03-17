"""
Stage 6 — Quality Validation.

Reads the final masked Arrow file, computes statistics, writes a review
sample, runs gate checks, optionally shards into train/val Arrow files,
and optionally uploads to S3.

If any gate check fails, the pipeline exits with a non-zero status and
does NOT upload to S3.
"""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

from pipeline.config import PipelineConfig
from pipeline.funnel_tracker import FunnelTracker
from pipeline.io.arrow_writer import iter_arrow, ShardedArrowWriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = (p / 100.0) * (len(arr) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(arr) - 1)
    frac = idx - lo
    return arr[lo] * (1 - frac) + arr[hi] * frac


def compute_stats(records: list[dict], report_cfg, tracker: FunnelTracker | None) -> dict:
    seq_lens:         list[int]   = []
    unmasked_counts:  list[int]   = []
    source_counts:    dict[str, int] = {}
    lang_counts:      dict[str, int] = {}
    zero_unmasked     = 0

    for rec in records:
        seq_len  = rec.get("_seq_len", len(rec.get("input_ids", [])))
        unmasked = rec.get("_unmasked_tokens", 0)
        source   = rec.get("_source", "unknown")
        lang     = rec.get("_lang", "unknown")

        if isinstance(seq_len, (int, float)):
            seq_lens.append(int(seq_len))
        if isinstance(unmasked, (int, float)):
            unmasked_counts.append(int(unmasked))
            if int(unmasked) == 0:
                zero_unmasked += 1

        source_counts[source] = source_counts.get(source, 0) + 1
        if report_cfg.compute_per_language_counts:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

    n = len(records)
    total_tokens  = sum(seq_lens)
    total_unmasked = sum(unmasked_counts)

    stats: dict[str, Any] = {
        "total_examples":         n,
        "total_tokens":           total_tokens,
        "zero_unmasked_count":    zero_unmasked,
        "zero_unmasked_fraction": zero_unmasked / max(1, n),
    }

    if report_cfg.compute_length_stats and seq_lens:
        stats["seq_len"] = {
            "min":    min(seq_lens),
            "max":    max(seq_lens),
            "mean":   sum(seq_lens) / len(seq_lens),
            "p5":     _percentile(seq_lens,  5),
            "p50":    _percentile(seq_lens, 50),
            "p95":    _percentile(seq_lens, 95),
            "p99":    _percentile(seq_lens, 99),
        }

    if report_cfg.compute_unmasked_ratio_stats and unmasked_counts and seq_lens:
        ratios = [u / max(1, s) for u, s in zip(unmasked_counts, seq_lens)]
        stats["unmasked_ratio"] = {
            "mean":  sum(ratios) / len(ratios),
            "min":   min(ratios),
            "max":   max(ratios),
            "total": total_unmasked / max(1, total_tokens),
        }

    if report_cfg.compute_per_source_counts:
        stats["per_source_counts"] = source_counts

    if report_cfg.compute_per_language_counts:
        stats["per_language_counts"] = lang_counts

    if tracker:
        stats["funnel"] = tracker.report()

    return stats


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def run_gate(stats: dict, gate_cfg) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failures)."""
    failures = []
    n = stats.get("total_examples", 0)

    if n < gate_cfg.min_total_examples:
        failures.append(f"total_examples={n} < min={gate_cfg.min_total_examples}")

    zuf = stats.get("zero_unmasked_fraction", 0.0)
    if zuf > gate_cfg.max_zero_unmasked_fraction:
        failures.append(
            f"zero_unmasked_fraction={zuf:.4f} > max={gate_cfg.max_zero_unmasked_fraction}"
        )

    if "unmasked_ratio" in stats:
        ur = stats["unmasked_ratio"].get("total", 1.0)
        if ur < gate_cfg.min_unmasked_ratio:
            failures.append(
                f"unmasked_ratio={ur:.4f} < min={gate_cfg.min_unmasked_ratio}"
            )

    if "seq_len" in stats:
        p5  = stats["seq_len"].get("p5",  0)
        p95 = stats["seq_len"].get("p95", 0)
        if p5 < gate_cfg.min_p5_length:
            failures.append(f"p5_seq_len={p5} < min={gate_cfg.min_p5_length}")
        if p95 > gate_cfg.max_p95_length:
            failures.append(f"p95_seq_len={p95} > max={gate_cfg.max_p95_length}")

    return (len(failures) == 0), failures


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run(cfg: PipelineConfig, tracker: FunnelTracker | None = None) -> None:
    stage_cfg = cfg.stage6
    if not stage_cfg.enabled:
        logger.info("Stage 6 (quality) disabled — skipping")
        return

    input_path = cfg.work_path(stage_cfg.input_file)

    # -----------------------------------------------------------------------
    # Load all records into memory (for stats + sharding)
    # -----------------------------------------------------------------------
    logger.info("Stage 6: loading records from %s", input_path)
    records: list[dict] = list(iter_arrow(input_path))
    logger.info("Stage 6: %d records loaded", len(records))

    # -----------------------------------------------------------------------
    # Compute statistics
    # -----------------------------------------------------------------------
    stats = compute_stats(records, stage_cfg.report, tracker)

    # -----------------------------------------------------------------------
    # Review sample
    # -----------------------------------------------------------------------
    output_dir = Path(cfg.globals.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    review_path = output_dir / stage_cfg.review_output
    rng = random.Random(stage_cfg.review_sample_seed)
    sample_size = min(stage_cfg.review_sample_size, len(records))
    sample = rng.sample(records, sample_size)

    with open(review_path, "w", encoding="utf-8") as f:
        for rec in sample:
            # Write a human-readable subset for review
            review_rec = {
                "formatted_text": rec.get("formatted_text", "")[:500] + "...",
                "_source":        rec.get("_source", ""),
                "_lang":          rec.get("_lang", ""),
                "_seq_len":       rec.get("_seq_len", 0),
                "_unmasked_tokens": rec.get("_unmasked_tokens", 0),
            }
            f.write(json.dumps(review_rec, ensure_ascii=False) + "\n")
    logger.info("Stage 6: review sample written → %s", review_path)

    # -----------------------------------------------------------------------
    # Write quality report
    # -----------------------------------------------------------------------
    report_path = output_dir / stage_cfg.report.output_json
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stage 6: quality report → %s", report_path)

    # -----------------------------------------------------------------------
    # Gate check
    # -----------------------------------------------------------------------
    gate_passed = True
    if stage_cfg.gate.enabled:
        gate_passed, failures = run_gate(stats, stage_cfg.gate)
        stats["gate_passed"] = gate_passed
        stats["gate_failures"] = failures

        # Re-write report with gate result
        with open(report_path, "w") as f:
            json.dump(stats, f, indent=2)

        if not gate_passed:
            logger.error("Stage 6: GATE FAILED — %d check(s) failed:", len(failures))
            for msg in failures:
                logger.error("  - %s", msg)
            logger.error("Aborting — output will NOT be uploaded to S3.")
            sys.exit(1)
        else:
            logger.info("Stage 6: gate PASSED ✓")

    # -----------------------------------------------------------------------
    # Train / val split and sharding
    # -----------------------------------------------------------------------
    n = len(records)
    n_val   = max(1, int(n * stage_cfg.train_val_split_ratio))
    n_train = n - n_val
    rng2 = random.Random(cfg.globals.seed)
    shuffled = list(records)
    rng2.shuffle(shuffled)
    train_records = shuffled[:n_train]
    val_records   = shuffled[n_train:]

    shard_dir = output_dir / stage_cfg.output_dir_sharded
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_size_mb = cfg.globals.shard_size_mb
    compression   = cfg.globals.arrow_compression

    with ShardedArrowWriter(shard_dir, prefix="sft_train", split="train",
                            shard_size_mb=shard_size_mb, compression=compression) as w:
        for rec in train_records:
            w.write(rec)
    logger.info("Stage 6: %d train shards written", len(w.shard_paths))

    with ShardedArrowWriter(shard_dir, prefix="sft_train", split="val",
                            shard_size_mb=shard_size_mb, compression=compression) as w:
        for rec in val_records:
            w.write(rec)
    logger.info("Stage 6: %d val shards written", len(w.shard_paths))

    if tracker:
        tracker.record_stage_output("stage6", n)

    # -----------------------------------------------------------------------
    # S3 upload (only if gate passed)
    # -----------------------------------------------------------------------
    if cfg.s3.enabled and gate_passed:
        from pipeline.io.s3_uploader import upload_directory
        logger.info("Stage 6: uploading shards to s3://%s/%s", cfg.s3.bucket, cfg.s3.prefix)
        upload_directory(
            local_dir=shard_dir,
            bucket=cfg.s3.bucket,
            prefix=cfg.s3.prefix,
            aws_profile=cfg.s3.aws_profile,
            checksum_verify=cfg.s3.checksum_verify,
            multipart_threshold_mb=cfg.s3.multipart_threshold_mb,
            glob="*.arrow",
        )
        logger.info("Stage 6: S3 upload complete")
    elif cfg.s3.enabled and not gate_passed:
        logger.warning("Stage 6: S3 upload SKIPPED — gate did not pass")

    logger.info(
        "Stage 6 complete: %d train / %d val examples → %s",
        n_train, n_val, shard_dir,
    )
