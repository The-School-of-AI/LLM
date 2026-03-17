"""
Stage 0 — Data Mixing.

Loads multiple source datasets (HuggingFace Hub or local files),
normalises each to the canonical conversation format, applies per-source
caps and weighted proportional sampling, shuffles, and writes a single
merged JSONL.

Normalisation is delegated to ``normalize_to_conversation`` from the
existing standardize_conversation_format.py (imported as a library).
"""
from __future__ import annotations

import json
import logging
import math
import random
import sys
from pathlib import Path

from pipeline.config import PipelineConfig, SourceConfig
from pipeline.funnel_tracker import FunnelTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import the shared normalisation helper
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "01_sft_data" / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))
try:
    from standardize_conversation_format import normalize_to_conversation  # type: ignore
except ImportError:
    # Fallback: define a minimal version if the import fails (e.g. running tests)
    def normalize_to_conversation(record: dict, format_: str) -> dict | None:  # type: ignore
        if format_ == "already_conversation":
            conv = record.get("conversations") or record.get("messages") or []
            if not conv:
                return None
            normalized = []
            for t in conv:
                r = (t.get("role") or t.get("from", "")).lower()
                c = (t.get("content") or t.get("value", "")).strip()
                if r in ("human", "user"):
                    normalized.append({"role": "user", "content": c})
                elif r in ("gpt", "assistant", "bot"):
                    normalized.append({"role": "assistant", "content": c})
                elif r == "system":
                    normalized.append({"role": "system", "content": c})
            return {"conversations": normalized} if normalized else None
        if format_ == "alpaca":
            inst = record.get("instruction") or ""
            inp  = record.get("input") or ""
            out  = record.get("output") or record.get("response", "")
            user_content = inst.strip()
            if inp.strip():
                user_content += "\n\n" + inp.strip()
            return {"conversations": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": out.strip()},
            ]}
        if format_ == "sharegpt":
            conv = record.get("conversations") or record.get("conversation", [])
            turns = []
            for t in conv:
                r = (t.get("from") or t.get("role", "")).lower()
                c = (t.get("value") or t.get("content", "")).strip()
                if r in ("human", "user"):
                    turns.append({"role": "user", "content": c})
                elif r in ("gpt", "assistant", "bot"):
                    turns.append({"role": "assistant", "content": c})
                elif r == "system":
                    turns.append({"role": "system", "content": c})
            return {"conversations": turns} if turns else None
        return None


def _load_source(src: SourceConfig, seed: int) -> list[dict]:
    """Load and normalise one source. Returns a list of conversation dicts."""
    records: list[dict] = []

    if src.path:
        from pipeline.io.readers import iter_records
        raw_iter = iter_records(src.path)
    elif src.hf_id:
        try:
            from datasets import load_dataset
            ds = load_dataset(src.hf_id, split=src.hf_split, trust_remote_code=False)
            raw_iter = (dict(row) for row in ds)
        except Exception as exc:
            logger.error("Failed to load HuggingFace dataset %s: %s", src.hf_id, exc)
            return []
    else:
        logger.error("Source '%s': neither path nor hf_id is set — skipping", src.name)
        return []

    skipped = 0
    for raw in raw_iter:
        norm = normalize_to_conversation(raw, src.format)
        if not norm:
            skipped += 1
            continue
        norm["_source"] = src.name
        records.append(norm)
        if src.max_samples and len(records) >= src.max_samples * 10:
            # Early stopping to avoid loading huge datasets into RAM unnecessarily
            break

    if skipped:
        logger.info("Source '%s': skipped %d records that failed normalisation", src.name, skipped)

    # Apply max_samples cap with deterministic shuffle
    if src.max_samples and len(records) > src.max_samples:
        rng = random.Random(src.subsample_seed)
        rng.shuffle(records)
        records = records[: src.max_samples]

    logger.info("Source '%s': %d records loaded", src.name, len(records))
    return records


def run(cfg: PipelineConfig, tracker: FunnelTracker | None = None) -> None:
    stage_cfg = cfg.stage0
    if not stage_cfg.enabled:
        logger.info("Stage 0 (mix) disabled — skipping")
        return

    output_path = cfg.work_path(stage_cfg.output_file)

    # Resume: skip if output already exists and save_intermediates is on
    if cfg.globals.save_intermediates and output_path.exists() and cfg.globals.resume_from_stage > 0:
        logger.info("Stage 0: output %s exists, resuming from next stage", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load all sources
    # -----------------------------------------------------------------------
    all_records: list[list[dict]] = []
    for src in stage_cfg.sources:
        records = _load_source(src, cfg.globals.seed)
        all_records.append(records)

    # -----------------------------------------------------------------------
    # Compute target counts (weighted proportional allocation)
    # -----------------------------------------------------------------------
    total_available = sum(len(r) for r in all_records)
    total_target = stage_cfg.total_target or total_available

    weights = [src.weight for src in stage_cfg.sources]
    weight_sum = sum(weights) or 1.0
    targets = [
        min(len(all_records[i]), round(total_target * (w / weight_sum)))
        for i, w in enumerate(weights)
    ]
    # Adjust last source to hit exact total_target
    actual_total = sum(targets)
    if actual_total != total_target and targets:
        diff = total_target - actual_total
        targets[-1] = max(0, targets[-1] + diff)

    # -----------------------------------------------------------------------
    # Sample from each source
    # -----------------------------------------------------------------------
    sampled: list[dict] = []
    rng = random.Random(cfg.globals.seed)
    for i, (records, target, src) in enumerate(zip(all_records, targets, stage_cfg.sources)):
        if target <= 0:
            continue
        src_rng = random.Random(src.subsample_seed + i)
        src_rng.shuffle(records)
        sampled.extend(records[:target])
        logger.info(
            "Stage 0: source '%s' — target=%d, actual=%d",
            src.name, target, min(len(records), target)
        )

    # Final global shuffle
    rng.shuffle(sampled)

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in sampled:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if tracker:
        tracker.record_stage_output("stage0", len(sampled))

    logger.info("Stage 0 complete: %d examples → %s", len(sampled), output_path)
