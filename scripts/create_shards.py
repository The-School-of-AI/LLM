"""
create_shards.py — Tokenize a HuggingFace dataset and write directory-per-shard output.

Produces the directory layout expected by bin_idx_dataloader.py:

    <output_dir>/
      shard_00000/
        tokens.bin      — uint32 token IDs, 4096-token fixed blocks
        tokens.idx      — uint64 byte offsets (one per block + final sentinel)
        metadata.json   — shard identity, curriculum tags, audit counts
      shard_00001/
        ...

Tokenization approach (matches tokenizer team spec):
  1. Load dataset text column row by row
  2. Tokenize each row (no truncation)
  3. Append EOS token after each row
  4. Concatenate all rows into one flat token stream
  5. Cut stream into fixed 4096-token blocks
  6. Tail shorter than 4096 tokens is dropped and logged

Usage — quick test with wikitext:
    python scripts/create_shards.py \\
        --dataset wikitext \\
        --dataset-config wikitext-2-raw-v1 \\
        --split train \\
        --output-dir /tmp/test_shards \\
        --tokenizer gpt2 \\
        --band B0 \\
        --domain general \\
        --stage 1

Usage — with the TSAI tokenizer:
    python scripts/create_shards.py \\
        --dataset wikitext \\
        --dataset-config wikitext-103-raw-v1 \\
        --split train \\
        --output-dir data/wikitext_shards \\
        --tokenizer src/tokenizer \\
        --tokens-per-shard 4096000 \
        --band B1 \\
        --domain general \\
        --stage 1


Note on --tokenizer:
    Pass either a HuggingFace model name (e.g. "gpt2") or a local directory path
    containing tokenizer.json + special_tokens_map.json + tokenizer_config.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List

import numpy as np

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("create_shards")

BLOCK_SIZE = 4096  # tokens per block — project constant
IDX_HEADER = b"\x00" * 8  # 8-byte header written at start of every .idx file
DTYPE = np.uint32  # token ID storage dtype
BYTES_PER_TOKEN = 4  # sizeof(uint32)


# ---------------------------------------------------------------------------
# Tokenizer hash (mirrors bin_idx_dataloader.compute_tokenizer_hash)
# ---------------------------------------------------------------------------


def _compute_tokenizer_hash(tokenizer_dir: str) -> str:
    files = ["tokenizer.json", "special_tokens_map.json"]
    h = hashlib.sha256()
    for fname in sorted(files):
        fpath = os.path.join(tokenizer_dir, fname)
        if not os.path.exists(fpath):
            logger.warning("Tokenizer file missing for hash: %s", fpath)
            continue
        with open(fpath, "rb") as f:
            h.update(fname.encode())
            h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Token stream generator
# ---------------------------------------------------------------------------


def _token_stream(dataset, tokenizer, text_column: str) -> Iterator[List[int]]:
    """
    Yield one list of token IDs per dataset row (with EOS appended).
    Rows with empty text are skipped and counted.
    """
    skipped = 0
    for row in dataset:
        text = row.get(text_column, "")
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            ids.append(tokenizer.eos_token_id)
            yield ids
    if skipped:
        logger.info("Skipped %d empty rows.", skipped)


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


def _write_shard(
    shard_dir: Path,
    blocks: List[List[int]],
    tokenizer,
    tokenizer_dir: str,
    rows_input: int,
    rows_with_eos: int,
    rows_dropped: int,
    tokens_dropped: int,
    band: str,
    domain: str,
    stage: int,
    source_name: str,
) -> None:
    """Write tokens.bin, tokens.idx, metadata.json into shard_dir."""
    shard_dir.mkdir(parents=True, exist_ok=True)

    bin_path = shard_dir / "tokens.bin"
    idx_path = shard_dir / "tokens.idx"
    meta_path = shard_dir / "metadata.json"

    num_blocks = len(blocks)
    total_tokens = num_blocks * BLOCK_SIZE

    # --- tokens.bin ---
    flat = np.array([tok for block in blocks for tok in block], dtype=DTYPE)
    with open(bin_path, "wb") as f:
        f.write(flat.tobytes())

    # --- tokens.idx ---
    # One uint64 offset per block start + one final sentinel (= file size in bytes)
    offsets = np.arange(num_blocks + 1, dtype=np.uint64) * (
        BLOCK_SIZE * BYTES_PER_TOKEN
    )
    with open(idx_path, "wb") as f:
        f.write(IDX_HEADER)
        f.write(offsets.tobytes())

    # --- metadata.json ---
    tok_hash = (
        _compute_tokenizer_hash(tokenizer_dir)
        if os.path.isdir(tokenizer_dir)
        else "unavailable"
    )
    metadata = {
        "tokenizer_hash": tok_hash,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ),
        "vocab_size": len(tokenizer),
        "block_size": BLOCK_SIZE,
        "num_blocks": num_blocks,
        "total_tokens": total_tokens,
        "rows_input": rows_input,
        "rows_with_eos": rows_with_eos,
        "rows_dropped": rows_dropped,
        "tokens_dropped": tokens_dropped,
        "drop_reason": "tail_truncation_at_block_boundary",
        "band": band,
        "domain": domain,
        "stage": stage,
        "source_file": source_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Wrote shard %s | blocks=%d | tokens=%d | rows_input=%d | rows_dropped=%d",
        shard_dir.name,
        num_blocks,
        total_tokens,
        rows_input,
        rows_dropped,
    )


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def create_shards(
    dataset_name: str,
    dataset_config: str,
    split: str,
    output_dir: str,
    tokenizer_name_or_path: str,
    tokens_per_shard: int,
    text_column: str,
    band: str,
    domain: str,
    stage: int,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install the 'datasets' package: pip install datasets")
        sys.exit(1)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("Install the 'transformers' package: pip install transformers")
        sys.exit(1)

    # Load tokenizer
    logger.info("Loading tokenizer from: %s", tokenizer_name_or_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    except Exception as e:
        logger.error("Failed to load tokenizer: %s", e)
        logger.error(
            "If using code/src/tokenizer/, it is currently missing tokenizer.json. "
            "Use a HuggingFace name (e.g. --tokenizer gpt2) for testing."
        )
        sys.exit(1)

    if tokenizer.eos_token_id is None:
        logger.error("Tokenizer has no eos_token_id. Cannot append EOS per row.")
        sys.exit(1)

    logger.info(
        "Tokenizer: vocab_size=%d, eos=%d (%s), pad=%s",
        len(tokenizer),
        tokenizer.eos_token_id,
        tokenizer.eos_token,
        tokenizer.pad_token_id,
    )

    # Load dataset
    logger.info(
        "Loading dataset: %s / %s / split=%s",
        dataset_name,
        dataset_config or "default",
        split,
    )
    try:
        if dataset_config:
            ds = load_dataset(
                dataset_name, dataset_config, split=split, trust_remote_code=False
            )
        else:
            ds = load_dataset(dataset_name, split=split, trust_remote_code=False)
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        sys.exit(1)

    if text_column not in ds.column_names:
        logger.error(
            "Column '%s' not found. Available columns: %s",
            text_column,
            ds.column_names,
        )
        sys.exit(1)

    logger.info("Dataset loaded: %d rows, columns: %s", len(ds), ds.column_names)

    # Determine tokenizer directory for hash (only valid for local paths)
    tok_dir = (
        tokenizer_name_or_path
        if os.path.isdir(tokenizer_name_or_path)
        else tokenizer_name_or_path
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    blocks_per_shard = tokens_per_shard // BLOCK_SIZE
    source_name = f"{dataset_name}/{dataset_config or 'default'}/{split}"

    # Streaming tokenization — build shards incrementally
    token_buffer: List[int] = []
    current_blocks: List[List[int]] = []
    shard_index = 0

    rows_input_total = 0
    rows_with_eos_total = 0
    total_tokens_written = 0

    # Per-shard counters
    shard_rows_input = 0
    shard_rows_with_eos = 0

    for row_tokens in _token_stream(ds, tokenizer, text_column):
        rows_input_total += 1
        rows_with_eos_total += 1
        shard_rows_input += 1
        shard_rows_with_eos += 1

        token_buffer.extend(row_tokens)

        # Carve complete blocks out of buffer
        while len(token_buffer) >= BLOCK_SIZE:
            current_blocks.append(token_buffer[:BLOCK_SIZE])
            token_buffer = token_buffer[BLOCK_SIZE:]

            # Flush shard when we have enough blocks
            if len(current_blocks) >= blocks_per_shard:
                shard_dir = output_path / f"shard_{shard_index:05d}"
                _write_shard(
                    shard_dir=shard_dir,
                    blocks=current_blocks,
                    tokenizer=tokenizer,
                    tokenizer_dir=tok_dir,
                    rows_input=shard_rows_input,
                    rows_with_eos=shard_rows_with_eos,
                    rows_dropped=0,  # no tail drop mid-stream
                    tokens_dropped=0,
                    band=band,
                    domain=domain,
                    stage=stage,
                    source_name=source_name,
                )
                total_tokens_written += len(current_blocks) * BLOCK_SIZE
                shard_index += 1
                current_blocks = []
                shard_rows_input = 0
                shard_rows_with_eos = 0

    # Final shard: write remaining complete blocks, drop partial tail
    tokens_dropped = len(token_buffer)
    rows_dropped = 1 if tokens_dropped > 0 else 0  # at most one partial row at tail

    if current_blocks:
        shard_dir = output_path / f"shard_{shard_index:05d}"
        _write_shard(
            shard_dir=shard_dir,
            blocks=current_blocks,
            tokenizer=tokenizer,
            tokenizer_dir=tok_dir,
            rows_input=shard_rows_input,
            rows_with_eos=shard_rows_with_eos - rows_dropped,
            rows_dropped=rows_dropped,
            tokens_dropped=tokens_dropped,
            band=band,
            domain=domain,
            stage=stage,
            source_name=source_name,
        )
        total_tokens_written += len(current_blocks) * BLOCK_SIZE
        shard_index += 1

    if tokens_dropped > 0:
        logger.info(
            "Tail drop: %d tokens discarded (< one full block). "
            "Logged in final shard metadata.json.",
            tokens_dropped,
        )

    logger.info(
        "Done. %d shards written to %s | total_tokens=%d | rows_input=%d | rows_with_eos=%d",
        shard_index,
        output_dir,
        total_tokens_written,
        rows_input_total,
        rows_with_eos_total,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tokenize a HuggingFace dataset and write directory-per-shard output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", required=True, help="HuggingFace dataset name, e.g. 'wikitext'"
    )
    p.add_argument(
        "--dataset-config",
        default="",
        help="Dataset config/subset, e.g. 'wikitext-2-raw-v1'. Leave empty if none.",
    )
    p.add_argument(
        "--split", default="train", help="Dataset split: train / validation / test"
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write shard subdirectories into",
    )
    p.add_argument(
        "--tokenizer",
        required=True,
        help=(
            "HuggingFace model name (e.g. 'gpt2') or local tokenizer directory. "
            "Use a HuggingFace name for testing — code/src/tokenizer/ is currently "
            "missing tokenizer.json."
        ),
    )
    p.add_argument(
        "--tokens-per-shard",
        type=int,
        default=24_000_000,
        help="Target token count per shard (rounded down to nearest block). "
        "Default 24M ≈ 5859 blocks of 4096.",
    )
    p.add_argument(
        "--text-column", default="text", help="Name of the text column in the dataset"
    )
    p.add_argument(
        "--band",
        default="B0",
        help="Curriculum band for metadata, e.g. B0 / B1 / B2 / B3 / B4 / B5",
    )
    p.add_argument(
        "--domain",
        default="general",
        help="Domain tag for metadata, e.g. general / math / code / reasoning",
    )
    p.add_argument(
        "--stage",
        type=int,
        default=1,
        help="Training stage for metadata (1 / 2 / 3 / 4)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.tokens_per_shard < BLOCK_SIZE:
        logger.error(
            "--tokens-per-shard must be >= %d (one block). Got %d.",
            BLOCK_SIZE,
            args.tokens_per_shard,
        )
        sys.exit(1)

    create_shards(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        output_dir=args.output_dir,
        tokenizer_name_or_path=args.tokenizer,
        tokens_per_shard=args.tokens_per_shard,
        text_column=args.text_column,
        band=args.band,
        domain=args.domain,
        stage=args.stage,
    )
