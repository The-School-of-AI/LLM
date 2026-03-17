"""
Stage 5 — Loss Masking.

Reads the tokenized Arrow file from Stage 4. For each record:
  1. Initialises ``labels`` = [-100] * len(input_ids)
  2. For every token_role_span where role is in ``train_on_roles``,
     copies input_ids[token_start:token_end] into labels at those positions.
  3. Validates that at least min_unmasked_tokens positions are unmasked.
     Records failing this gate are dropped.
  4. Writes the updated Arrow file with a ``labels`` column added.

This approach is provably correct because spans are computed at render time
(Stage 3) — it does NOT scan input_ids for special token IDs, which is the
fragile approach used in DataCollatorForCompletionOnlyLM.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.funnel_tracker import FunnelTracker
from pipeline.io.arrow_writer import iter_arrow, _records_to_table

logger = logging.getLogger(__name__)


def apply_loss_mask(
    input_ids: list[int],
    token_role_spans: list[dict],
    train_on_roles: set[str],
    ignore_index: int,
) -> list[int]:
    """
    Build labels array.
    Positions in train_on_roles spans: copy input_ids value.
    All other positions: ignore_index (-100).
    """
    labels = [ignore_index] * len(input_ids)
    for span in token_role_spans:
        if span.get("role") in train_on_roles:
            ts = span.get("token_start", 0)
            te = span.get("token_end", 0)
            for i in range(ts, min(te, len(input_ids))):
                labels[i] = input_ids[i]
    return labels


def run(cfg: PipelineConfig, tracker: FunnelTracker | None = None) -> None:
    stage_cfg = cfg.stage5
    if not stage_cfg.enabled:
        logger.info("Stage 5 (mask) disabled — skipping")
        return

    input_path  = cfg.work_path(stage_cfg.input_file)
    output_path = cfg.work_path(stage_cfg.output_file)

    if cfg.globals.save_intermediates and output_path.exists() and cfg.globals.resume_from_stage > 5:
        logger.info("Stage 5: output %s exists, resuming from next stage", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_on_roles = set(stage_cfg.train_on_roles)
    ignore_index   = stage_cfg.ignore_index
    min_unmasked   = stage_cfg.min_unmasked_tokens

    total = kept = dropped = 0
    records_out: list[dict] = []

    for record in iter_arrow(input_path):
        total += 1
        input_ids = record.get("input_ids", [])
        if not isinstance(input_ids, list):
            input_ids = list(input_ids)

        tspans_raw = record.get("token_role_spans", "[]")
        if isinstance(tspans_raw, str):
            try:
                token_role_spans = json.loads(tspans_raw)
            except json.JSONDecodeError:
                token_role_spans = []
        else:
            token_role_spans = tspans_raw or []

        labels = apply_loss_mask(input_ids, token_role_spans, train_on_roles, ignore_index)
        unmasked_count = sum(1 for l in labels if l != ignore_index)

        if unmasked_count < min_unmasked:
            if tracker:
                tracker.record_drop(
                    "stage5", "zero_unmasked",
                    f"unmasked={unmasked_count}<{min_unmasked}"
                )
            dropped += 1
            continue

        record["labels"]           = labels
        record["_unmasked_tokens"] = unmasked_count
        records_out.append(record)
        kept += 1

    # Write Arrow
    import pyarrow.ipc as ipc
    from pipeline.io.arrow_writer import _records_to_table

    table = _records_to_table(records_out)
    opts = ipc.IpcWriteOptions(compression=cfg.globals.arrow_compression)
    with ipc.new_file(str(output_path), table.schema, options=opts) as writer:
        writer.write_table(table)

    if tracker:
        tracker.record_stage_output("stage5", kept)

    logger.info(
        "Stage 5 complete: total=%d, kept=%d, dropped(zero_unmasked)=%d → %s",
        total, kept, dropped, output_path,
    )
