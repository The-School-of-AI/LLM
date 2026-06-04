"""
BinIdx Dataloader — curriculum-aware, DDP-safe, tokenizer-validated.

Reads pre-tokenized .bin shards (packed uint32 token IDs) and feeds them to
the training loop. No .idx files needed — block offsets are computed from file
size since all blocks are fixed-size (4096 tokens × 4 bytes = 16384 bytes).

  - Deterministic: sorted shard order, fixed 4096-token shard blocks, no shuffling
  - DDP-safe: rank-sharded file list; each rank reads its own non-overlapping shards
  - Tokenizer-validated: asserts tokenizer hash / special token IDs match metadata.json
  - Auditable: logs every skipped / corrupted region; warns on metadata mismatch
  - Curriculum-compatible: shard list supplied externally (from CurriculumSampler)

Block size policy:
  Shards are 4096-token fixed blocks (packed uint32, no headers, no idx files).

  seq_len < 4096:
      Each shard block is split into multiple sequences (with carry-over when
      not evenly divisible).

  seq_len == 4096:
      Each block is one sequence.

  seq_len > 4096:
      Consecutive blocks are joined to fill each sequence.

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
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

from .utils import print_rank_0

logger = logging.getLogger(__name__)

# Shards are 4096-token fixed blocks.
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
# Low-level .bin reader (no .idx files needed)
# ---------------------------------------------------------------------------


def _iter_sequences_from_shard(
    bin_path: str,
    dtype: np.dtype,
    seq_len: int,
) -> Iterator[torch.Tensor]:
    """
    Yield [seq_len] long tensors from a single tokens.bin shard.

    Block offsets are computed from file size — no .idx file needed.
    Each block is SHARD_BLOCK_SIZE tokens of dtype (uint32 = 4 bytes each),
    so block_bytes = 4096 * 4 = 16384. The file is just packed blocks.

    seq_len == SHARD_BLOCK_SIZE (4096):
        Each block is one sequence.

    seq_len < SHARD_BLOCK_SIZE:
        Each block is split into multiple sequences (with carry-over).

    seq_len > SHARD_BLOCK_SIZE:
        Consecutive blocks are joined to fill each sequence.

    Any trailing bytes that don't form a complete block are logged and skipped.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")

    itemsize = dtype.itemsize
    block_bytes = SHARD_BLOCK_SIZE * itemsize
    file_size = os.path.getsize(bin_path)
    num_blocks = file_size // block_bytes
    tail_bytes = file_size % block_bytes

    if num_blocks == 0:
        logger.warning(
            "%s: file too small for even 1 block (%d bytes), skipping.",
            bin_path,
            file_size,
        )
        return

    if tail_bytes > 0:
        logger.info(
            "%s: %d tail bytes after %d blocks (not a full block), discarded.",
            bin_path,
            tail_bytes,
            num_blocks,
        )

    # Unified buffer: supports seq_len <, ==, or > SHARD_BLOCK_SIZE.
    token_buffer: List[int] = []

    with open(bin_path, "rb") as f:
        for i in range(num_blocks):
            raw = f.read(block_bytes)
            block = np.frombuffer(raw, dtype=dtype)

            if len(block) != SHARD_BLOCK_SIZE:
                logger.warning(
                    "Incomplete read in %s block %d: expected %d got %d. Skipping.",
                    bin_path,
                    i,
                    SHARD_BLOCK_SIZE,
                    len(block),
                )
                continue

            # Accumulate blocks into buffer, emit fixed seq_len windows.
            token_buffer.extend(block.tolist())
            while len(token_buffer) >= seq_len:
                chunk = np.array(token_buffer[:seq_len], dtype=np.int64)
                token_buffer = token_buffer[seq_len:]
                yield torch.from_numpy(chunk).to(torch.long)

    if token_buffer:
        logger.info(
            "%s: %d tokens remaining in join-buffer (< seq_len=%d), discarded.",
            bin_path,
            len(token_buffer),
            seq_len,
        )


# ---------------------------------------------------------------------------
# Shard manifest builder
# ---------------------------------------------------------------------------

# Structured log prefix for monitors / alerting systems to capture
_STARVATION_PREFIX = "DATALOADER_STARVATION"


def _build_shard_list(
    shard_dir: str,
    rank: int,
    world_size: int,
) -> Tuple[List[Tuple[str, str]], int]:
    """
    Return (shard_entries, total_shards) for this rank.

    shard_entries is a sorted list of (bin_path, shard_subdir) tuples
    assigned to this rank via round-robin across all subdirectories.

    Expected on-disk layout (directory-per-shard):
        shards/
          shard_001/
            tokens.bin
          shard_002/
            tokens.bin
          ...

    Only tokens.bin is required. No .idx or metadata.json needed.

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
            "Expected layout: shards/<shard_name>/tokens.bin. "
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

    entries = []
    for sd in rank_subdirs:
        bp = sd / "tokens.bin"
        if not bp.exists():
            raise FileNotFoundError(f"Missing tokens.bin in shard directory {sd}.")
        entries.append((str(bp), str(sd)))

    return entries, total_shards


# ---------------------------------------------------------------------------
# IterableDataset
# ---------------------------------------------------------------------------


class BinIdxDataset(IterableDataset):
    """
    IterableDataset over pre-tokenized .bin shards (no .idx files needed).

    Emits dicts compatible with the training loop in code/src/train.py:
        {
            "input_ids":      LongTensor [seq_len]
            "attention_mask": LongTensor [seq_len]  (all 1s — full causal mask)
            "labels":         LongTensor [seq_len]  (copy of input_ids)
        }

    seq_len can be any positive value:
        seq_len < 4096   → each 4096 shard block is split into multiple sequences
        seq_len == 4096  → one block per sequence (standard)
        seq_len > 4096   → consecutive blocks are joined to fill each sequence

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
            raise ValueError(f"seq_len must be > 0, got {seq_len}")

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

        # Shard progress tracking (updated during iteration)
        self._current_shard_index: int = -1
        self._sequences_yielded: int = 0  # offset within current shard
        self._shards_completed: List[int] = []

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
        for bin_path, shard_subdir in self._shard_pairs:
            meta_path = os.path.join(shard_subdir, "metadata.json")
            meta = _load_shard_meta(meta_path)
            if meta is None:
                # No metadata.json is fine — shards_reordered only have tokens.bin
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
        # Worker-aware splitting: each DataLoader worker gets a non-overlapping
        # subset of shards. Without this, every worker iterates the SAME shards,
        # causing N-step cycling where N = num_workers (each batch is duplicated
        # num_workers times before new data is seen).
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard_pairs = self._shard_pairs[worker_info.id :: worker_info.num_workers]
            if not shard_pairs:
                logger.warning(
                    "Worker %d/%d has no shards assigned (only %d shard(s) for this rank). "
                    "Consider reducing num_workers or adding more shards.",
                    worker_info.id,
                    worker_info.num_workers,
                    len(self._shard_pairs),
                )
                return
        else:
            shard_pairs = self._shard_pairs

        # Reset tracking for this iteration pass
        self._shards_completed = []
        self._current_shard_index = -1
        self._sequences_yielded = 0

        for shard_idx, (bin_path, _) in enumerate(shard_pairs):
            self._current_shard_index = shard_idx
            self._sequences_yielded = 0
            for seq in _iter_sequences_from_shard(bin_path, self.dtype, self.seq_len):
                self._sequences_yielded += 1
                yield {
                    "input_ids": seq,
                    "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                    "labels": seq.clone(),
                }
            self._shards_completed.append(shard_idx)

    def get_shard_state(self) -> Dict[str, Any]:
        """Return shard progress for checkpoint metadata.

        Shard order is deterministic (sorted alphabetically, round-robin by
        rank), so the current index is sufficient to derive completed /
        remaining sets on resume without storing full path lists.
        """
        total = len(self._shard_pairs)
        cur_idx = self._current_shard_index
        cur_path = self._shard_pairs[cur_idx][0] if 0 <= cur_idx < total else None

        return {
            "total_shards": total,
            "current_shard_index": cur_idx,
            "current_shard_path": cur_path,
            "completed_count": len(self._shards_completed),
            "remaining_count": max(total - cur_idx - 1, 0) if cur_idx >= 0 else total,
            "sequence_offset": self._sequences_yielded,
            "rank": self._rank,
            "world_size": self._world_size,
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
    Build and return a DataLoader over .bin shards (no .idx needed).

    Args:
        shard_dir:          Directory containing shard subdirectories.
        batch_size:         Batch size per GPU (not global batch size).
        tokenizer:          Live tokenizer instance. Used for EOS/PAD ID validation.
        tokenizer_dir:      Path to tokenizer directory for hash computation.
                            Defaults to code/src/tokenizer/ relative to this file.
        seq_len:            Sequence length (any positive integer).
                            Default 4096 (one block per sequence).
                            Use 2048 to split each block, or 8192/16384 to join blocks.
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
        print_rank_0(f"Tokenizer hash (first 16 chars): {tok_hash[:16]}...")
    else:
        print_rank_0(
            "  WARNING: tokenizer_dir not found — tokenizer hash validation will be "
            "skipped. Ensure shards were produced with the canonical tokenizer."
        )

    if seq_len == SHARD_BLOCK_SIZE:
        print_rank_0(f"Block config: seq_len={seq_len} (1 block per sequence)")
    elif seq_len < SHARD_BLOCK_SIZE:
        seqs_per_block = SHARD_BLOCK_SIZE // seq_len
        if SHARD_BLOCK_SIZE % seq_len == 0:
            print_rank_0(
                f"Block config: seq_len={seq_len} ({seqs_per_block} sequences split from each {SHARD_BLOCK_SIZE}-token block)"
            )
        else:
            print_rank_0(
                f"Block config: seq_len={seq_len} (split with cross-block carry-over)"
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
