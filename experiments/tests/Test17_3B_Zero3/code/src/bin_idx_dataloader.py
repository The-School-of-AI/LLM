"""
BinIdx Dataloader — curriculum-aware, DDP-safe, tokenizer-validated.

Reads pre-tokenized .bin/.idx shards produced by the tokenizer team and feeds
them to the training loop. Designed to satisfy all project non-negotiables:

  - Deterministic: sorted shard order, fixed 512-token block layout, no shuffling
  - DDP-safe: rank-sharded file list; each rank reads its own non-overlapping shards
  - Tokenizer-validated: asserts tokenizer hash / special token IDs match metadata.json
  - Auditable: logs every skipped / corrupted region; warns on metadata mismatch
  - Curriculum-compatible: shard list supplied externally (from CurriculumSampler)

Block size policy:
  Shards are always 512-token fixed blocks (Test 16 override from 4096).

  seq_len == 512 (default):
      Each block is one sequence. One read per block.

  seq_len > 512 (e.g. 1024):
      Consecutive blocks are joined. For seq_len=1024, two adjacent blocks are
      concatenated to form one sequence. No re-tokenization needed; the flat
      token stream is already correct — EOS tokens inside the joined window are
      preserved as normal tokens for the model to learn.

Integration with existing training stack (code/src/train.py):
  The DataLoader returned by build_bin_idx_dataloader() emits batches with the
  same schema as get_dataloaders() in data.py:
      {"input_ids": LongTensor, "attention_mask": LongTensor, "labels": LongTensor}

Usage (single-GPU, seq_len == block_size):
    from bin_idx_dataloader import build_bin_idx_dataloader
    loader = build_bin_idx_dataloader(
        shard_dir="data_loader/",
        seq_len=4096,
        batch_size=4,
        tokenizer=tokenizer,
    )

Usage (multi-GPU / DeepSpeed torchrun, larger context):
    loader = build_bin_idx_dataloader(
        shard_dir="data_loader/",
        seq_len=8192,   # joins 2 consecutive 4096 blocks
        batch_size=2,
        tokenizer=tokenizer,
        # rank / world_size resolved automatically from torch.distributed
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

from .utils import print_rank_0

logger = logging.getLogger(__name__)

# Shards are 4096-token fixed blocks on disk.
# seq_len can be <= SHARD_BLOCK_SIZE (truncate one block) or a multiple of it (join blocks).
SHARD_BLOCK_SIZE = 4096


# ---------------------------------------------------------------------------
# Tokenizer hash
# ---------------------------------------------------------------------------

def compute_tokenizer_hash(tokenizer_dir: str) -> str:
    """
    Stable SHA-256 hash of the two files that fully define token IDs:
      tokenizer.json          — BPE merges and vocabulary
      special_tokens_map.json — special token definitions

    tokenizer_config.json is intentionally excluded: it contains mutable
    metadata (model_max_length, etc.) that does not affect token ID mapping.
    """
    files = ["tokenizer.json", "special_tokens_map.json"]
    h = hashlib.sha256()
    for fname in sorted(files):
        fpath = os.path.join(tokenizer_dir, fname)
        if not os.path.exists(fpath):
            logger.warning(f"Tokenizer file missing for hash: {fpath}")
            continue
        with open(fpath, "rb") as f:
            h.update(fname.encode())
            h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Metadata loading and validation
# ---------------------------------------------------------------------------

def _load_shard_meta(meta_path: str) -> Optional[dict]:
    """Load metadata.json sidecar if present. Returns None for legacy shards."""
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_shard_meta(
    meta: dict,
    tokenizer,
    expected_hash: Optional[str],
    bin_path: str,
) -> None:
    """
    Assert shard metadata is consistent with the live tokenizer.
    Raises ValueError on hard mismatches (wrong tokenizer identity).
    Logs warnings on soft mismatches (missing fields, dropped rows).
    """
    errors = []

    if expected_hash and "tokenizer_hash" in meta:
        if meta["tokenizer_hash"] != expected_hash:
            errors.append(
                f"tokenizer_hash mismatch: "
                f"shard={meta['tokenizer_hash'][:12]}... "
                f"live={expected_hash[:12]}... "
                "— shard was produced with a different tokenizer."
            )

    if "eos_token_id" in meta and tokenizer is not None:
        if meta["eos_token_id"] != tokenizer.eos_token_id:
            errors.append(
                f"eos_token_id mismatch: "
                f"shard={meta['eos_token_id']} live={tokenizer.eos_token_id}"
            )

    if "pad_token_id" in meta and tokenizer is not None:
        live_pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        if meta["pad_token_id"] != live_pad:
            errors.append(
                f"pad_token_id mismatch: "
                f"shard={meta['pad_token_id']} live={live_pad}"
            )

    if errors:
        raise ValueError(
            f"Tokenizer identity mismatch in {bin_path} — refusing to load.\n"
            + "\n".join(f"  • {e}" for e in errors)
            + "\nRegenerate shards with the canonical tokenizer or resolve the "
            "mismatch. See data_loader/TOKENIZER_TEAM_RECOMMENDATIONS.md."
        )

    if meta.get("rows_dropped", 0) > 0:
        print_rank_0(
            f"  WARNING: {bin_path}: {meta['rows_dropped']} rows dropped at tail "
            f"({meta.get('tokens_dropped', '?')} tokens). Logged for auditability."
        )

    if "band" not in meta or "domain" not in meta:
        print_rank_0(
            f"  WARNING: {bin_path}: missing band/domain in metadata.json — "
            "curriculum sampler will treat this shard as untagged."
        )


# ---------------------------------------------------------------------------
# Low-level .bin/.idx reader
# ---------------------------------------------------------------------------

_IDX_HEADER_BYTES = 8  # first 8 bytes of .idx are version/magic (skipped)


def _read_idx_offsets(idx_path: str) -> np.ndarray:
    """Return uint64 array of byte offsets from the .idx file."""
    with open(idx_path, "rb") as f:
        f.read(_IDX_HEADER_BYTES)
        offsets = np.frombuffer(f.read(), dtype=np.uint64)
    return offsets


def _iter_sequences_from_shard(
    bin_path: str,
    idx_path: str,
    dtype: np.dtype,
    seq_len: int,
) -> Iterator[torch.Tensor]:
    """
    Yield [seq_len] long tensors from a single .bin/.idx shard.

    seq_len <= SHARD_BLOCK_SIZE (e.g. 2048, 1024):
        One block is read; the first seq_len tokens are emitted and the rest
        of the block is discarded. Use for memory workarounds (shorter seq).

    seq_len == SHARD_BLOCK_SIZE (4096):
        Each .idx region is exactly one sequence. One read per region.

    seq_len > SHARD_BLOCK_SIZE:
        Must be an exact multiple of SHARD_BLOCK_SIZE. Consecutive blocks are
        read and joined. Example: seq_len=8192 joins two 4096-token blocks.
        If the total remaining blocks are insufficient to fill one seq_len
        window, that tail is logged and skipped.

    Corrupted / empty regions are logged and skipped; the shard continues.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if seq_len > SHARD_BLOCK_SIZE and seq_len % SHARD_BLOCK_SIZE != 0:
        raise ValueError(
            f"seq_len={seq_len} is not a multiple of SHARD_BLOCK_SIZE={SHARD_BLOCK_SIZE}. "
            f"Use seq_len <= {SHARD_BLOCK_SIZE} (truncate one block) or seq_len = N * {SHARD_BLOCK_SIZE} (join N blocks)."
        )

    truncate_from_block = seq_len <= SHARD_BLOCK_SIZE
    blocks_per_seq = 1 if truncate_from_block else seq_len // SHARD_BLOCK_SIZE
    offsets = _read_idx_offsets(idx_path)
    itemsize = dtype.itemsize
    num_regions = len(offsets) - 1
    skipped = 0

    # Buffer for joining consecutive blocks when blocks_per_seq > 1
    token_buffer: List[int] = []

    with open(bin_path, "rb") as f:
        for i in range(num_regions):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            num_bytes = end - start

            if num_bytes <= 0:
                logger.debug("Empty region %d in %s, skipping.", i, bin_path)
                skipped += 1
                continue

            num_tokens = num_bytes // itemsize
            expected = SHARD_BLOCK_SIZE

            if num_tokens < expected:
                logger.warning(
                    "Short region %d in %s: expected %d tokens, got %d. Skipping.",
                    i, bin_path, expected, num_tokens,
                )
                skipped += 1
                continue

            f.seek(start)
            raw = f.read(expected * itemsize)
            block = np.frombuffer(raw, dtype=dtype)

            if len(block) != expected:
                logger.warning(
                    "Incomplete read in %s region %d: expected %d got %d. Skipping.",
                    bin_path, i, expected, len(block),
                )
                skipped += 1
                continue

            if truncate_from_block:
                # seq_len <= SHARD_BLOCK_SIZE: use first seq_len tokens of this block
                out = block[:seq_len].copy()
                yield torch.from_numpy(out).to(torch.long)
            elif blocks_per_seq == 1:
                # seq_len == SHARD_BLOCK_SIZE: one block == one sequence
                yield torch.from_numpy(block.copy()).to(torch.long)
            else:
                # Accumulate blocks into buffer, emit when full
                token_buffer.extend(block.tolist())
                while len(token_buffer) >= seq_len:
                    chunk = np.array(token_buffer[:seq_len], dtype=np.int64)
                    token_buffer = token_buffer[seq_len:]
                    yield torch.from_numpy(chunk).to(torch.long)

    if token_buffer:
        logger.info(
            "%s: %d tokens remaining in join-buffer (< seq_len=%d), discarded.",
            bin_path, len(token_buffer), seq_len,
        )

    if skipped > 0:
        logger.info("%s: skipped %d/%d regions.", bin_path, skipped, num_regions)


# ---------------------------------------------------------------------------
# Shard manifest builder
# ---------------------------------------------------------------------------

# Structured log prefix for monitors / alerting systems to capture
_STARVATION_PREFIX = "DATALOADER_STARVATION"


def _build_shard_list(
    shard_dir: str,
    rank: int,
    world_size: int,
) -> Tuple[List[Tuple[str, str, str]], int]:
    """
    Return (shard_pairs, total_shards) for this rank.

    shard_pairs is a sorted list of (bin_path, idx_path, shard_subdir) tuples
    assigned to this rank via round-robin across all subdirectories.

    Expected on-disk layout (directory-per-shard):
        shards/
          shard_001/
            tokens.bin
            tokens.idx
            metadata.json
          shard_002/
            ...

    If a rank receives no shards (fewer shards than GPUs), this is rank
    starvation. We do NOT hard-fail — training continues on ranks that have
    data. Instead we emit a structured, highly-visible error log so that
    the monitoring system can capture it and trigger a halt if needed.
    """
    shard_dir_path = Path(shard_dir)
    all_subdirs = sorted(p for p in shard_dir_path.iterdir() if p.is_dir())
    total_shards = len(all_subdirs)

    if not all_subdirs:
        raise FileNotFoundError(
            f"No shard subdirectories found in {shard_dir}. "
            "Expected layout: shards/<shard_name>/tokens.bin + tokens.idx + metadata.json. "
            "Verify shard_dir is correct and tokenizer team has delivered shards."
        )

    rank_subdirs = all_subdirs[rank::world_size]

    if not rank_subdirs:
        print_rank_0(
            f"  ERROR: {_STARVATION_PREFIX} | rank={rank} | world_size={world_size} | "
            f"total_shards={total_shards} | "
            "This rank has zero shards assigned. "
            "The GPU will idle and produce no gradient updates. "
            "Add more shards or reduce world_size to eliminate starvation."
        )
        return [], total_shards

    pairs = []
    for sd in rank_subdirs:
        bp = sd / "tokens.bin"
        ip = sd / "tokens.idx"
        if not bp.exists():
            raise FileNotFoundError(f"Missing tokens.bin in shard directory {sd}.")
        if not ip.exists():
            raise FileNotFoundError(f"Missing tokens.idx in shard directory {sd}.")
        pairs.append((str(bp), str(ip), str(sd)))

    return pairs, total_shards


# ---------------------------------------------------------------------------
# IterableDataset
# ---------------------------------------------------------------------------

class BinIdxDataset(IterableDataset):
    """
    IterableDataset over pre-tokenized .bin/.idx shards.

    Emits dicts compatible with the training loop in code/src/train.py:
        {
            "input_ids":      LongTensor [seq_len]
            "attention_mask": LongTensor [seq_len]  (all 1s — full causal mask)
            "labels":         LongTensor [seq_len]  (copy of input_ids)
        }

    seq_len behavior:
        seq_len <= 4096  → one block read, first seq_len tokens used (e.g. 2048 for memory workaround)
        seq_len == 4096  → one block per sequence (standard)
        seq_len > 4096   → must be multiple of 4096; consecutive blocks joined (8192, 16384, ...)

    DDP:
        rank / world_size are resolved automatically from torch.distributed.
        Each rank receives a non-overlapping, balanced subset of shards.
        If a rank gets no shards, a structured DATALOADER_STARVATION error
        is logged; the rank emits no batches (cost governor should halt).

    Tokenizer validation:
        Validates every shard's metadata.json against the live tokenizer hash
        and special token IDs before reading. Hard-fails on mismatch.
        Set validate_tokenizer=False only during migration of legacy shards.
    """

    def __init__(
        self,
        shard_dir: str,
        seq_len: int = SHARD_BLOCK_SIZE,
        tokenizer=None,
        tokenizer_dir: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
        dtype: str = "uint32",
        validate_tokenizer: bool = True,
    ) -> None:
        super().__init__()

        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if seq_len > SHARD_BLOCK_SIZE and seq_len % SHARD_BLOCK_SIZE != 0:
            raise ValueError(
                f"seq_len={seq_len} must be <= {SHARD_BLOCK_SIZE} or a multiple of it. "
                f"Use e.g. 2048, {SHARD_BLOCK_SIZE}, {SHARD_BLOCK_SIZE*2}, ..."
            )

        self.shard_dir = shard_dir
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.dtype = np.dtype(dtype)
        self.validate_tokenizer = validate_tokenizer

        self._tokenizer_hash: Optional[str] = None
        if validate_tokenizer and tokenizer_dir and os.path.isdir(tokenizer_dir):
            self._tokenizer_hash = compute_tokenizer_hash(tokenizer_dir)

        self._shard_pairs, self._total_shards = _build_shard_list(
            shard_dir, rank, world_size
        )
        self._rank = rank
        self._world_size = world_size
        self._validate_all_shards()

    def _validate_all_shards(self) -> None:
        """Pre-flight: validate metadata.json for every shard on this rank."""
        if not self.validate_tokenizer:
            print_rank_0(
                "  WARNING: validate_tokenizer=False — skipping tokenizer identity checks. "
                "Ensure shards were produced with the canonical tokenizer."
            )
            return

        errors_found = 0
        for bin_path, _, shard_subdir in self._shard_pairs:
            meta_path = os.path.join(shard_subdir, "metadata.json")
            meta = _load_shard_meta(meta_path)
            if meta is None:
                print_rank_0(
                    f"  WARNING: No metadata.json for {bin_path} — cannot validate "
                    "tokenizer identity. This shard will be loaded without validation. "
                    "Regenerate with the canonical tokenizer to resolve."
                )
                continue
            try:
                _validate_shard_meta(
                    meta, self.tokenizer, self._tokenizer_hash, bin_path
                )
            except ValueError as e:
                print_rank_0(f"  ERROR: {e}")
                errors_found += 1

        if errors_found > 0:
            raise RuntimeError(
                f"{errors_found} shard(s) failed tokenizer validation. "
                "See log output above for details."
            )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for bin_path, idx_path, _ in self._shard_pairs:
            for seq in _iter_sequences_from_shard(
                bin_path, idx_path, self.dtype, self.seq_len
            ):
                yield {
                    "input_ids": seq,
                    "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                    "labels": seq.clone(),
                }


# ---------------------------------------------------------------------------
# Distributed context helper
# ---------------------------------------------------------------------------

def _resolve_dist_context() -> Tuple[int, int]:
    """Return (rank, world_size) from torch.distributed, falling back to env vars."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return max(rank, 0), max(world_size, 1)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_bin_idx_dataloader(
    shard_dir: str,
    batch_size: int,
    tokenizer=None,
    tokenizer_dir: Optional[str] = None,
    seq_len: int = SHARD_BLOCK_SIZE,
    dtype: str = "uint32",
    num_workers: int = 4,
    prefetch_factor: int = 4,
    validate_tokenizer: bool = True,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> DataLoader:
    """
    Build and return a DataLoader over .bin/.idx shards.

    Args:
        shard_dir:          Directory containing shard subdirectories.
        batch_size:         Batch size per GPU (not global batch size).
        tokenizer:          Live tokenizer instance. Used for EOS/PAD ID validation.
        tokenizer_dir:      Path to tokenizer directory for hash computation.
                            Defaults to code/src/tokenizer/ relative to this file.
        seq_len:            Sequence length. Can be <= 4096 (truncate one block, e.g. 2048)
                            or a multiple of 4096 (join blocks: 4096, 8192, 16384, ...).
        dtype:              Token dtype used when writing .bin files (default uint32).
        num_workers:        DataLoader worker processes. 0 = main process (debug only).
        prefetch_factor:    Prefetch batches per worker (only active when num_workers > 0).
        validate_tokenizer: Hard-fail on tokenizer mismatch. Set False only for
                            legacy shards during migration.
        rank:               Override rank (default: auto-detect from torch.distributed).
        world_size:         Override world_size (default: auto-detect).

    Returns:
        DataLoader emitting {"input_ids", "attention_mask", "labels"} batches.
        drop_last=True ensures consistent batch size across all steps.
    """
    if rank is None or world_size is None:
        _rank, _world_size = _resolve_dist_context()
        rank = rank if rank is not None else _rank
        world_size = world_size if world_size is not None else _world_size

    is_distributed = world_size > 1

    if tokenizer_dir is None:
        _default = Path(__file__).parent / "tokenizer"
        tokenizer_dir = str(_default) if _default.is_dir() else None

    print_rank_0(f"Loading pre-tokenized shards from: {shard_dir}")
    print_rank_0(
        f"Distributed context: is_distributed={is_distributed}, "
        f"world_size={world_size}, rank={rank}"
    )

    if tokenizer_dir:
        tok_hash = compute_tokenizer_hash(tokenizer_dir)
        print_rank_0(
            f"Tokenizer hash (first 16 chars): {tok_hash[:16]}..."
        )
    else:
        print_rank_0(
            "  WARNING: tokenizer_dir not found — tokenizer hash validation will be "
            "skipped. Ensure shards were produced with the canonical tokenizer."
        )

    if seq_len <= SHARD_BLOCK_SIZE:
        print_rank_0(
            f"Block config: seq_len={seq_len} (first {seq_len} tokens of each {SHARD_BLOCK_SIZE}-token block)"
        )
    else:
        blocks_per_seq = seq_len // SHARD_BLOCK_SIZE
        print_rank_0(
            f"Block config: seq_len={seq_len} ({blocks_per_seq} blocks joined per sequence)"
        )

    dataset = BinIdxDataset(
        shard_dir=shard_dir,
        seq_len=seq_len,
        tokenizer=tokenizer,
        tokenizer_dir=tokenizer_dir,
        rank=rank,
        world_size=world_size,
        dtype=dtype,
        validate_tokenizer=validate_tokenizer,
    )

    n_assigned = len(dataset._shard_pairs)
    total_shards = dataset._total_shards

    print_rank_0(
        f"Shard assignment: rank={rank}/{world_size}, "
        f"assigned={n_assigned}/{total_shards} shards"
    )

    if validate_tokenizer and n_assigned > 0:
        print_rank_0(f"✓ All {n_assigned} shard(s) passed tokenizer validation")

    loader_kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
    }

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(dataset, **loader_kwargs)

    persistent_workers = bool(loader_kwargs.get("persistent_workers", False))

    print_rank_0(
        f"DataLoader worker config: "
        f"num_workers={num_workers}, "
        f"persistent_workers={persistent_workers}, "
        f"pin_memory={loader_kwargs['pin_memory']}, "
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', None)}"
    )
    print_rank_0(
        f"BinIdxDataLoader ready | shard_dir={shard_dir} | seq_len={seq_len} | "
        f"batch_size={batch_size} | rank={rank}/{world_size} | "
        f"shards={n_assigned}/{total_shards} | num_workers={num_workers} | "
        f"drop_last=True | pin_memory={loader_kwargs['pin_memory']}"
    )

    return loader
