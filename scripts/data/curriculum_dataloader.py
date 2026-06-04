#!/usr/bin/env python3
"""
Curriculum-aware dataloader for band-separated training shards.

Reads shards from band-separated directories (band_B0/, band_B1/, etc.)
and samples blocks according to curriculum weights defined in curriculum.yaml.

Emits {"input_ids", "attention_mask", "labels"} tensors for the training loop.
Blocks are fully packed (no padding), so attention_mask is all 1s.

Usage:
    from curriculum_dataloader import build_curriculum_dataloader

    loader = build_curriculum_dataloader(
        shard_dir="/path/to/shards",
        curriculum_path="curriculum.yaml",
        stage="1B",
        batch_size=4,
    )
    for batch in loader:
        input_ids = batch["input_ids"]       # [batch_size, 4096]
        attention_mask = batch["attention_mask"]  # [batch_size, 4096] (all 1s)
        labels = batch["labels"]             # [batch_size, 4096]
"""

from __future__ import annotations

import mmap
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset

SHARD_BLOCK_SIZE = 4096
IDX_HEADER_BYTES = 8
BYTES_PER_TOKEN = 4  # uint32


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# CURRICULUM CONFIG
# ═══════════════════════════════════════════════════════════════════════════


class CurriculumConfig:
    """Parsed curriculum configuration for one stage."""

    def __init__(self, yaml_path: str, stage: str):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Curriculum config not found: {yaml_path}")

        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        if stage not in raw.get("stages", {}):
            available = list(raw.get("stages", {}).keys())
            raise KeyError(
                f"Stage '{stage}' not found in {yaml_path}. "
                f"Available stages: {available}"
            )

        self.stage = stage
        self.version = raw.get("version", "unknown")
        self.block_size = raw.get("block_size", SHARD_BLOCK_SIZE)

        stage_cfg = raw["stages"][stage]
        self._band_weights = dict(stage_cfg.get("band_weights", {}))
        self._modality_weights = dict(stage_cfg.get("modality_weights", {}))
        self._guardrails = raw.get("guardrails", {})

        # Validate weights sum to 1.0
        bw_sum = sum(self._band_weights.values())
        if abs(bw_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Stage '{stage}' band_weights sum to {bw_sum}, expected 1.0"
            )

    @property
    def band_weights(self) -> Dict[str, float]:
        return dict(self._band_weights)

    @property
    def modality_weights(self) -> Dict[str, float]:
        return dict(self._modality_weights)

    @property
    def guardrails(self) -> Dict[str, Any]:
        return dict(self._guardrails)

    def effective_weights(self, available_bands: List[str]) -> Dict[str, float]:
        """
        Redistribute weights for missing bands proportionally to available bands.
        Logs a WARNING for each missing band with non-zero weight.
        """
        available_set = set(available_bands)
        missing_weight = 0.0
        available_weight = 0.0

        for band, w in self._band_weights.items():
            if band in available_set:
                available_weight += w
            else:
                if w > 0:
                    _log(
                        f"  WARNING: Band {band} has weight {w:.3f} "
                        f"but no shards available — redistributing"
                    )
                missing_weight += w

        if available_weight == 0:
            raise ValueError("No bands with non-zero weight have shards available")

        # Redistribute proportionally
        scale = 1.0 / available_weight
        effective = {}
        for band in available_bands:
            orig_w = self._band_weights.get(band, 0.0)
            effective[band] = orig_w * scale

        return effective

    def __repr__(self) -> str:
        bands = ", ".join(f"{b}={w:.2f}" for b, w in self._band_weights.items())
        return f"CurriculumConfig(stage={self.stage}, bands=[{bands}])"


# ═══════════════════════════════════════════════════════════════════════════
# SHARD READER (mmap-based)
# ═══════════════════════════════════════════════════════════════════════════


class _ShardReader:
    """mmap-based reader for a single shard directory."""

    def __init__(self, shard_dir: str):
        self.shard_dir = shard_dir
        bin_path = os.path.join(shard_dir, "tokens.bin")
        idx_path = os.path.join(shard_dir, "tokens.idx")

        if not os.path.exists(bin_path) or not os.path.exists(idx_path):
            raise FileNotFoundError(f"Missing tokens.bin or tokens.idx in {shard_dir}")

        # Read .idx offsets
        with open(idx_path, "rb") as f:
            f.read(IDX_HEADER_BYTES)
            self._offsets = np.frombuffer(f.read(), dtype=np.uint64)

        self._num_blocks = len(self._offsets) - 1
        if self._num_blocks <= 0:
            raise ValueError(f"Shard {shard_dir} has {self._num_blocks} blocks")

        # mmap tokens.bin
        self._file = open(bin_path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def read_block(self, block_idx: int) -> np.ndarray:
        """Read block as uint32 numpy array. .copy() for mmap safety."""
        start = int(self._offsets[block_idx])
        end = int(self._offsets[block_idx + 1])
        return np.frombuffer(self._mm[start:end], dtype=np.uint32).copy()

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()


# ═══════════════════════════════════════════════════════════════════════════
# BAND SHARD POOL (per-band cycling iterator)
# ═══════════════════════════════════════════════════════════════════════════


class _BandShardPool:
    """
    Manages all shards for one band. Infinite cycling through blocks.
    Shards are shuffled between epochs.
    """

    def __init__(self, band: str, shard_dirs: List[str], seed: int = 42):
        self.band = band
        self._shard_dirs = list(shard_dirs)
        self._rng = random.Random(seed)
        self._epoch = 0

        # Open all shard readers
        self._readers: List[_ShardReader] = []
        total_blocks = 0
        for sd in self._shard_dirs:
            reader = _ShardReader(sd)
            self._readers.append(reader)
            total_blocks += reader.num_blocks

        self._total_blocks = total_blocks
        self._blocks_served = 0

        # Build block index: list of (reader_idx, block_idx) pairs
        self._block_index: List[Tuple[int, int]] = []
        self._rebuild_index()

        # Position in current epoch
        self._pos = 0

    def _rebuild_index(self) -> None:
        """Rebuild and shuffle the block index for a new epoch."""
        self._block_index = []
        reader_order = list(range(len(self._readers)))
        self._rng.shuffle(reader_order)
        for ri in reader_order:
            for bi in range(self._readers[ri].num_blocks):
                self._block_index.append((ri, bi))
        self._rng.shuffle(self._block_index)
        self._pos = 0

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    @property
    def num_shards(self) -> int:
        return len(self._shard_dirs)

    @property
    def blocks_served(self) -> int:
        return self._blocks_served

    def next_block(self) -> np.ndarray:
        """Return the next block. Cycles infinitely with reshuffling."""
        if self._pos >= len(self._block_index):
            self._epoch += 1
            _log(
                f"  Band {self.band}: epoch {self._epoch} complete, "
                f"reshuffling {len(self._readers)} shards "
                f"({self._total_blocks} blocks)"
            )
            self._rebuild_index()

        ri, bi = self._block_index[self._pos]
        self._pos += 1
        self._blocks_served += 1
        return self._readers[ri].read_block(bi)

    def close(self) -> None:
        for reader in self._readers:
            reader.close()
        self._readers.clear()


# ═══════════════════════════════════════════════════════════════════════════
# CURRICULUM STATISTICS TRACKER
# ═══════════════════════════════════════════════════════════════════════════


class _CurriculumStats:
    """Tracks actual band sampling proportions for compliance monitoring."""

    def __init__(self, target_weights: Dict[str, float]):
        self._targets = dict(target_weights)
        self._counts: Dict[str, int] = defaultdict(int)
        self._total = 0

    def record(self, band: str) -> None:
        self._counts[band] += 1
        self._total += 1

    @property
    def total_blocks(self) -> int:
        return self._total

    def summary(self) -> str:
        if self._total == 0:
            return "  No blocks sampled yet"
        lines = [f"  Band stats ({self._total:,} blocks):"]
        for band in sorted(self._targets.keys()):
            target = self._targets[band]
            actual_count = self._counts.get(band, 0)
            actual_pct = actual_count / self._total if self._total > 0 else 0
            delta = actual_pct - target
            marker = "" if abs(delta) < 0.05 else " <<<" if abs(delta) >= 0.10 else " <"
            lines.append(
                f"    {band}: target={target:.3f} actual={actual_pct:.3f} "
                f"({actual_count:,} blocks){marker}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CURRICULUM ITERABLE DATASET
# ═══════════════════════════════════════════════════════════════════════════


class CurriculumDataset(IterableDataset):
    """
    IterableDataset that samples blocks from band-separated shards
    according to curriculum weights.

    Emits:
        {
            "input_ids":      LongTensor [seq_len],
            "attention_mask": LongTensor [seq_len],  (all 1s, fully packed)
            "labels":         LongTensor [seq_len],  (same as input_ids)
        }
    """

    def __init__(
        self,
        shard_dir: str,
        curriculum_config: CurriculumConfig,
        seq_len: int = SHARD_BLOCK_SIZE,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        log_interval: int = 500,
    ) -> None:
        super().__init__()

        self._shard_dir = shard_dir
        self._config = curriculum_config
        self._seq_len = seq_len
        self._rank = rank
        self._world_size = world_size
        self._seed = seed
        self._log_interval = log_interval
        self._blocks_per_seq = max(1, seq_len // SHARD_BLOCK_SIZE)

        _log("CurriculumDataset initializing...")
        _log(f"  shard_dir:    {shard_dir}")
        _log(f"  stage:        {curriculum_config.stage}")
        _log(f"  seq_len:      {seq_len} ({self._blocks_per_seq} block(s)/seq)")
        _log(f"  rank:         {rank}/{world_size}")

        # Discover band directories
        shard_root = Path(shard_dir)
        band_shards: Dict[str, List[str]] = {}

        for entry in sorted(shard_root.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("band_"):
                continue
            band_name = entry.name[len("band_") :]
            # Find shard subdirs in this band
            shard_subdirs = sorted(
                [
                    str(sd)
                    for sd in entry.iterdir()
                    if sd.is_dir() and (sd / "tokens.bin").exists()
                ]
            )
            if shard_subdirs:
                band_shards[band_name] = shard_subdirs

        if not band_shards:
            raise FileNotFoundError(
                f"No band directories with shards found in {shard_dir}. "
                f"Expected layout: band_B0/shard_000000/tokens.bin"
            )

        # Stripe shards per rank
        self._band_pools: Dict[str, _BandShardPool] = {}
        total_blocks_this_rank = 0

        for band, shards in sorted(band_shards.items()):
            rank_shards = shards[rank::world_size]
            if rank_shards:
                pool = _BandShardPool(band, rank_shards, seed=seed + hash(band))
                self._band_pools[band] = pool
                total_blocks_this_rank += pool.total_blocks
                _log(
                    f"  Band {band}: {len(rank_shards)}/{len(shards)} shards "
                    f"({pool.total_blocks:,} blocks) for rank {rank}"
                )
            else:
                _log(
                    f"  Band {band}: 0/{len(shards)} shards for rank {rank} "
                    f"(assigned to other ranks)"
                )

        if not self._band_pools:
            raise ValueError(
                f"Rank {rank} has no shards assigned. "
                f"Need at least {world_size} shards per band."
            )

        _log(f"  Total blocks for rank {rank}: {total_blocks_this_rank:,}")

        # Compute effective weights (redistribute missing bands)
        available_bands = list(self._band_pools.keys())
        self._effective_weights = curriculum_config.effective_weights(available_bands)

        _log("  Effective weights (after redistribution):")
        for band, w in sorted(self._effective_weights.items()):
            orig = curriculum_config.band_weights.get(band, 0)
            marker = "" if abs(w - orig) < 0.001 else f" (was {orig:.3f})"
            _log(f"    {band}: {w:.3f}{marker}")

        # Stats tracker
        self._stats = _CurriculumStats(self._effective_weights)
        _log("CurriculumDataset ready.")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self._seed + self._rank)
        bands = list(self._effective_weights.keys())
        weights = [self._effective_weights[b] for b in bands]

        seqs_emitted = 0
        t_start = time.time()
        ones = torch.ones(self._seq_len, dtype=torch.long)

        while True:
            # Accumulate blocks for one sequence
            token_chunks: List[np.ndarray] = []
            for _ in range(self._blocks_per_seq):
                chosen_band = rng.choices(bands, weights=weights, k=1)[0]
                self._stats.record(chosen_band)
                block = self._band_pools[chosen_band].next_block()
                token_chunks.append(block)

            # Concatenate and convert
            tokens = np.concatenate(token_chunks).astype(np.int64)
            # Trim to exact seq_len (handles seq_len not divisible by block_size)
            tokens = tokens[: self._seq_len]

            seq = torch.from_numpy(tokens)
            yield {
                "input_ids": seq,
                "attention_mask": ones,
                "labels": seq.clone(),
            }

            seqs_emitted += 1

            # Periodic logging
            if seqs_emitted % self._log_interval == 0:
                elapsed = time.time() - t_start
                total_tokens = seqs_emitted * self._seq_len
                tps = total_tokens / elapsed if elapsed > 0 else 0
                _log(
                    f"[Curriculum rank={self._rank}] "
                    f"{seqs_emitted:,} seqs, "
                    f"{total_tokens:,} tokens, "
                    f"{tps / 1e6:.2f}M tok/s"
                )
                _log(self._stats.summary())

    def close(self) -> None:
        for pool in self._band_pools.values():
            pool.close()
        self._band_pools.clear()

    @property
    def stats(self) -> _CurriculumStats:
        return self._stats

    @property
    def effective_weights(self) -> Dict[str, float]:
        return dict(self._effective_weights)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC FACTORY
# ═══════════════════════════════════════════════════════════════════════════


def build_curriculum_dataloader(
    shard_dir: str,
    curriculum_path: str,
    stage: str,
    batch_size: int,
    seq_len: int = SHARD_BLOCK_SIZE,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
    seed: int = 42,
    log_interval: int = 500,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> DataLoader:
    """
    Build a curriculum-aware DataLoader.

    Drop-in replacement for build_bin_idx_dataloader().

    Args:
        shard_dir:        Root directory containing band_B0/, band_B1/, etc.
        curriculum_path:  Path to curriculum.yaml
        stage:            Training stage ("1B", "3B", "8B", "70B")
        batch_size:       Micro batch size per GPU
        seq_len:          Sequence length (default 4096)
        num_workers:      DataLoader workers (0 = main process, recommended)
        prefetch_factor:  Prefetch batches per worker (None for default)
        seed:             Random seed for reproducibility
        log_interval:     Print curriculum stats every N sequences
        rank:             GPU rank (auto-detected if None)
        world_size:       Total GPUs (auto-detected if None)

    Returns:
        DataLoader yielding {"input_ids", "attention_mask", "labels"}
    """
    # Auto-detect distributed context
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            if rank is None:
                rank = dist.get_rank()
            if world_size is None:
                world_size = dist.get_world_size()
    except (ImportError, RuntimeError):
        pass

    if rank is None:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    _log(
        f"Building curriculum dataloader: stage={stage}, bs={batch_size}, "
        f"seq_len={seq_len}, rank={rank}/{world_size}"
    )

    config = CurriculumConfig(curriculum_path, stage)

    dataset = CurriculumDataset(
        shard_dir=shard_dir,
        curriculum_config=config,
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
        seed=seed,
        log_interval=log_interval,
    )

    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(dataset, **loader_kwargs)

    _log(
        f"CurriculumDataLoader ready | stage={stage} | "
        f"batch_size={batch_size} | seq_len={seq_len} | "
        f"rank={rank}/{world_size}"
    )

    return loader
