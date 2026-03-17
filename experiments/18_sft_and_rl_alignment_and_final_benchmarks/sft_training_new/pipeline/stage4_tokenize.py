"""
Stage 4 — Tokenization.

Tokenizes ``formatted_text`` from Stage 3 and converts character-level
``role_spans`` to token-level ``token_role_spans`` using the tokenizer's
``offset_mapping``.

Outputs Arrow IPC format (not JSONL) because input_ids / attention_mask are
integer arrays that are much more compact in columnar binary format.

Overflow handling (examples longer than max_length):
  - drop:      discard the example entirely.
  - truncate:  clip to max_length (right or left, per config).
  - split:     create overlapping chunks of max_length tokens.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.funnel_tracker import FunnelTracker
from pipeline.io.readers import iter_records
from pipeline.io.arrow_writer import ShardedArrowWriter

logger = logging.getLogger(__name__)


def char_spans_to_token_spans(
    char_spans: list[dict],
    offset_mapping: list[tuple[int, int]],
) -> list[dict]:
    """
    Convert character-level role_spans to token-level token_role_spans.

    offset_mapping[i] = (char_start, char_end) of token i.
    Tokens with offset (0, 0) are special tokens added by the tokenizer
    (BOS, EOS, padding) — they are skipped during span conversion.
    """
    token_spans = []
    for span in char_spans:
        c_start = span["start"]
        c_end   = span["end"]
        tok_start: int | None = None
        tok_end: int | None = None

        for i, (t_start, t_end) in enumerate(offset_mapping):
            if t_start == 0 and t_end == 0:
                continue  # special token
            if tok_start is None and t_end > c_start:
                tok_start = i
            if t_start < c_end:
                tok_end = i + 1  # exclusive upper bound

        token_spans.append({
            "role":        span["role"],
            "token_start": tok_start if tok_start is not None else 0,
            "token_end":   tok_end   if tok_end   is not None else 0,
        })
    return token_spans


def _clip_token_spans(
    token_spans: list[dict],
    max_len: int,
) -> list[dict]:
    """Clip token spans to [0, max_len] after truncation."""
    result = []
    for s in token_spans:
        ts = min(s["token_start"], max_len)
        te = min(s["token_end"],   max_len)
        if ts < te:
            result.append({"role": s["role"], "token_start": ts, "token_end": te})
    return result


def _make_chunks(
    input_ids: list[int],
    attention_mask: list[int],
    token_spans: list[dict],
    max_len: int,
    overlap: int,
) -> list[tuple[list[int], list[int], list[dict]]]:
    """Split a long sequence into overlapping chunks."""
    stride = max(1, max_len - overlap)
    chunks = []
    start = 0
    while start < len(input_ids):
        end = start + max_len
        ids_chunk  = input_ids[start:end]
        mask_chunk = attention_mask[start:end]
        # Adjust token spans to chunk offset
        spans_chunk = []
        for s in token_spans:
            ts = max(0, s["token_start"] - start)
            te = min(len(ids_chunk), s["token_end"] - start)
            if ts < te:
                spans_chunk.append({"role": s["role"], "token_start": ts, "token_end": te})
        chunks.append((ids_chunk, mask_chunk, spans_chunk))
        if end >= len(input_ids):
            break
        start += stride
    return chunks


def run(cfg: PipelineConfig, tracker: FunnelTracker | None = None) -> None:
    stage_cfg = cfg.stage4
    if not stage_cfg.enabled:
        logger.info("Stage 4 (tokenize) disabled — skipping")
        return

    input_path  = cfg.work_path(stage_cfg.input_file)
    output_path = cfg.work_path(stage_cfg.output_file)

    if cfg.globals.save_intermediates and output_path.exists() and cfg.globals.resume_from_stage > 4:
        logger.info("Stage 4: output %s exists, resuming from next stage", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        stage_cfg.tokenizer_name_or_path,
        trust_remote_code=stage_cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len  = stage_cfg.max_length
    strategy = stage_cfg.overflow_strategy
    overlap  = stage_cfg.split_overlap

    total = kept = dropped = 0

    # Write to a single Arrow shard at the output path (not sharded yet — sharding happens in Stage 6)
    records_out: list[dict] = []

    for record in iter_records(input_path):
        total += 1
        formatted_text = record.get("formatted_text", "")
        role_spans_raw = record.get("role_spans", [])
        if isinstance(role_spans_raw, str):
            try:
                role_spans_raw = json.loads(role_spans_raw)
            except json.JSONDecodeError:
                role_spans_raw = []

        enc = tokenizer(
            formatted_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        input_ids     = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offset_mapping = enc.get("offset_mapping", [(0, 0)] * len(input_ids))

        token_spans = char_spans_to_token_spans(role_spans_raw, offset_mapping)
        seq_len = len(input_ids)

        if seq_len <= max_len:
            out_records = [(input_ids, attention_mask, token_spans)]
        elif strategy == "drop":
            if tracker:
                tracker.record_drop("stage4", "overflow_drop", f"seq_len={seq_len}>{max_len}")
            dropped += 1
            continue
        elif strategy == "truncate":
            if stage_cfg.truncate_side == "right":
                input_ids      = input_ids[:max_len]
                attention_mask = attention_mask[:max_len]
            else:  # left
                input_ids      = input_ids[-max_len:]
                attention_mask = attention_mask[-max_len:]
            token_spans = _clip_token_spans(token_spans, max_len)
            out_records = [(input_ids, attention_mask, token_spans)]
        elif strategy == "split":
            chunks = _make_chunks(input_ids, attention_mask, token_spans, max_len, overlap)
            out_records = chunks
        else:
            out_records = [(input_ids[:max_len], attention_mask[:max_len],
                            _clip_token_spans(token_spans, max_len))]

        for chunk_idx, (ids, mask, tspans) in enumerate(out_records):
            out_rec = {k: v for k, v in record.items()
                       if k not in ("formatted_text", "role_spans")}
            out_rec["formatted_text"]    = formatted_text
            out_rec["role_spans"]        = json.dumps(role_spans_raw)
            out_rec["input_ids"]         = ids
            out_rec["attention_mask"]    = mask
            out_rec["token_role_spans"]  = json.dumps(tspans)
            out_rec["_seq_len"]          = len(ids)
            if len(out_records) > 1:
                out_rec["_chunk_idx"] = chunk_idx
            records_out.append(out_rec)
            kept += 1

    # Write all records to a single Arrow file
    import pyarrow as pa
    import pyarrow.ipc as ipc
    from pipeline.io.arrow_writer import _records_to_table

    table = _records_to_table(records_out)
    opts = ipc.IpcWriteOptions(compression=cfg.globals.arrow_compression)
    with ipc.new_file(str(output_path), table.schema, options=opts) as writer:
        writer.write_table(table)

    if tracker:
        tracker.record_stage_output("stage4", kept)

    logger.info(
        "Stage 4 complete: total=%d, kept=%d, dropped=%d → %s",
        total, kept, dropped, output_path,
    )
