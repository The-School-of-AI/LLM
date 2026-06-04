#!/usr/bin/env python3
"""
Curriculum-aware dataloader v2 for D1-D4 + AON band architecture.

Key changes from v1:
  - Reads curriculum_v2.yaml with D1/D2/D3/D4/AON pools
  - AON (Always-ON) pool is injected directly, bypassing OPUS scoring
  - AON has internal 50/50 split between bench_train and indic_guaranteed
  - Manifest-driven: reads pre-computed shard lists from manifests/ directory
  - Supports warmup stages (WU_3B, WU_8B, WU_70B)
  - Fully deterministic given seed + rank

Usage:
    from curriculum_dataloader_v2 import build_curriculum_v2_dataloader

    # For OPUS candidate batches (D1-D4 only, to be scored by OPUS):
    candidate_loader = build_curriculum_v2_dataloader(
        shard_dir="/path/to/shards_reordered",
        manifest_dir="/path/to/manifests",
        curriculum_path="curriculum_v2.yaml",
        stage="1B",
        batch_size=8,
        mode="opus_candidates",    # Only D1-D4 pools
    )

    # For AON batches (injected directly, bypasses OPUS):
    aon_loader = build_curriculum_v2_dataloader(
        shard_dir="/path/to/shards_reordered",
        manifest_dir="/path/to/manifests",
        curriculum_path="curriculum_v2.yaml",
        stage="1B",
        batch_size=1,
        mode="always_on",          # Only AON pool
    )

    # For combined (non-OPUS training, curriculum-weighted including AON):
    combined_loader = build_curriculum_v2_dataloader(
        shard_dir="/path/to/shards_reordered",
        manifest_dir="/path/to/manifests",
        curriculum_path="curriculum_v2.yaml",
        stage="1B",
        batch_size=4,
        mode="combined",           # D1-D4 + AON at 8%
    )
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

# Pool names recognized by v2
OPUS_POOLS = ["D1", "D2", "D3", "D4"]
AON_POOL = "AON"
ALL_POOLS = OPUS_POOLS + [AON_POOL]


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# CURRICULUM CONFIG v2
# ═══════════════════════════════════════════════════════════════════════════


class CurriculumConfigV2:
    """Parsed curriculum v2 configuration for one stage."""

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

        # AON config
        aon_cfg = raw.get("aon_config", {})
        self._aon_injection_rate = aon_cfg.get("injection_rate", 0.08)
        self._aon_internal_split = aon_cfg.get(
            "internal_split", {"bench_train": 0.50, "indic_guaranteed": 0.50}
        )

        # Pool definitions (for reference)
        self._pools = raw.get("pools", {})

        # Validate weights sum to 1.0
        bw_sum = sum(self._band_weights.values())
        if abs(bw_sum - 1.0) > 1e-4:
            raise ValueError(
                f"Stage '{stage}' band_weights sum to {bw_sum:.6f}, expected 1.0"
            )

    @property
    def band_weights(self) -> Dict[str, float]:
        return dict(self._band_weights)

    @property
    def opus_weights(self) -> Dict[str, float]:
        """Weights for OPUS-eligible pools only (D1-D4), renormalized."""
        opus_w = {k: v for k, v in self._band_weights.items() if k in OPUS_POOLS}
        total = sum(opus_w.values())
        if total == 0:
            raise ValueError("No OPUS-eligible pools have weight > 0")
        return {k: v / total for k, v in opus_w.items()}

    @property
    def aon_injection_rate(self) -> float:
        return self._aon_injection_rate

    @property
    def aon_internal_split(self) -> Dict[str, float]:
        return dict(self._aon_internal_split)

    def effective_weights(
        self, available_pools: List[str], mode: str
    ) -> Dict[str, float]:
        """
        Compute effective weights based on available pools and mode.

        mode="opus_candidates": only D1-D4, renormalized
        mode="always_on": only AON sub-pools
        mode="combined": all pools at original weights
        """
        if mode == "opus_candidates":
            target = {k: v for k, v in self._band_weights.items() if k in OPUS_POOLS}
        elif mode == "always_on":
            # AON split into bench_train and indic_guaranteed
            aon_w = self._band_weights.get(AON_POOL, 0.08)
            return {
                "AON_bench": aon_w * self._aon_internal_split.get("bench_train", 0.5),
                "AON_indic": aon_w
                * self._aon_internal_split.get("indic_guaranteed", 0.5),
            }
        elif mode == "combined":
            target = dict(self._band_weights)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Redistribute missing pools
        available_set = set(available_pools)
        missing_w = 0.0
        available_w = 0.0
        for pool, w in target.items():
            if pool in available_set:
                available_w += w
            else:
                if w > 0:
                    _log(
                        f"  WARNING: Pool {pool} has weight {w:.3f} "
                        f"but no shards available — redistributing"
                    )
                missing_w += w

        if available_w == 0:
            raise ValueError("No pools with non-zero weight have shards available")

        scale = (available_w + missing_w) / available_w
        return {
            p: target.get(p, 0) * scale for p in available_pools if target.get(p, 0) > 0
        }

    def __repr__(self) -> str:
        pools = ", ".join(f"{p}={w:.2f}" for p, w in self._band_weights.items())
        return f"CurriculumConfigV2(stage={self.stage}, [{pools}])"


# ═══════════════════════════════════════════════════════════════════════════
# SHARD READER (mmap-based, unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════


class _ShardReader:
    """mmap-based reader for a single shard directory.

    Supports two shard formats:
      1. Original shards (s3://t1-dataacquisition-dataset-shards/shards/):
         tokens.bin + tokens.idx + metadata.json — .idx has 8-byte header
         followed by uint64 byte offsets for each block boundary.

      2. Reordered shards (s3://t1-dataacquisition-datasets-2/shards_reordered/):
         tokens.bin ONLY — token IDs frequency-reordered, no .idx or metadata.
         Offsets are computed from file size assuming fixed SHARD_BLOCK_SIZE
         (4096 tokens × 4 bytes/token = 16384 bytes per block).
    """

    def __init__(self, shard_dir: str):
        self.shard_dir = shard_dir
        bin_path = os.path.join(shard_dir, "tokens.bin")
        idx_path = os.path.join(shard_dir, "tokens.idx")

        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Missing tokens.bin in {shard_dir}")

        if os.path.exists(idx_path):
            # Format 1: read offsets from .idx file
            with open(idx_path, "rb") as f:
                f.read(IDX_HEADER_BYTES)
                self._offsets = np.frombuffer(f.read(), dtype=np.uint64)
        else:
            # Format 2: no .idx — compute offsets from .bin file size.
            # Each block = SHARD_BLOCK_SIZE tokens × 4 bytes (uint32).
            bytes_per_block = SHARD_BLOCK_SIZE * 4
            file_size = os.path.getsize(bin_path)
            num_blocks = file_size // bytes_per_block
            if num_blocks <= 0:
                raise ValueError(
                    f"Shard {shard_dir}: tokens.bin too small "
                    f"({file_size} bytes, need >= {bytes_per_block})"
                )
            # Build offset array: [0, bytes_per_block, 2*bytes_per_block, ...]
            self._offsets = np.arange(num_blocks + 1, dtype=np.uint64) * bytes_per_block

        self._num_blocks = len(self._offsets) - 1
        if self._num_blocks <= 0:
            raise ValueError(f"Shard {shard_dir} has {self._num_blocks} blocks")

        self._file = open(bin_path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def read_block(self, block_idx: int) -> np.ndarray:
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
# POOL SHARD MANAGER (manifest-driven)
# ═══════════════════════════════════════════════════════════════════════════


class _PoolShardManager:
    """
    Manages shards for one pool. Reads shard list from manifest file.
    Infinite cycling through blocks with deterministic reshuffling.
    """

    def __init__(
        self,
        pool_name: str,
        shard_root: str,
        shard_paths: List[str],
        rank: int,
        world_size: int,
        seed: int = 42,
    ):
        self.pool_name = pool_name

        # Stripe shards by rank
        rank_shards = shard_paths[rank::world_size]
        if not rank_shards:
            _log(f"  WARNING: Pool {pool_name}: 0 shards for rank {rank}/{world_size}")
            self._readers = []
            self._total_blocks = 0
            self._block_index = []
            self._rng = random.Random(seed)
            self._pos = 0
            self._epoch = 0
            self._blocks_served = 0
            return

        # Open shard readers
        self._readers: List[_ShardReader] = []
        total_blocks = 0
        failed = 0
        for sp in rank_shards:
            full_path = os.path.join(shard_root, sp)
            try:
                reader = _ShardReader(full_path)
                self._readers.append(reader)
                total_blocks += reader.num_blocks
            except (FileNotFoundError, ValueError) as e:
                failed += 1
                if failed <= 3:
                    _log(f"  WARNING: {pool_name}/{sp}: {e}")

        if failed > 3:
            _log(
                f"  WARNING: {pool_name}: {failed} total failed shards "
                f"(showing first 3)"
            )

        self._total_blocks = total_blocks
        self._rng = random.Random(seed + hash(pool_name))
        self._epoch = 0
        self._blocks_served = 0

        self._block_index: List[Tuple[int, int]] = []
        self._rebuild_index()
        self._pos = 0

        _log(
            f"  Pool {pool_name}: {len(self._readers)}/{len(rank_shards)} "
            f"shards ({total_blocks:,} blocks) for rank {rank}"
        )

    def _rebuild_index(self) -> None:
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
        return len(self._readers)

    @property
    def blocks_served(self) -> int:
        return self._blocks_served

    @property
    def is_empty(self) -> bool:
        return len(self._readers) == 0

    def next_block(self) -> np.ndarray:
        if self.is_empty:
            raise RuntimeError(f"Pool {self.pool_name} has no shards")

        if self._pos >= len(self._block_index):
            self._epoch += 1
            _log(
                f"  Pool {self.pool_name}: epoch {self._epoch}, "
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
# STATS TRACKER
# ═══════════════════════════════════════════════════════════════════════════


class _CurriculumStatsV2:
    """Tracks actual pool sampling proportions for compliance monitoring."""

    def __init__(self, target_weights: Dict[str, float]):
        self._targets = dict(target_weights)
        self._counts: Dict[str, int] = defaultdict(int)
        self._total = 0

    def record(self, pool: str) -> None:
        self._counts[pool] += 1
        self._total += 1

    @property
    def total_blocks(self) -> int:
        return self._total

    def summary(self) -> str:
        if self._total == 0:
            return "  No blocks sampled yet"
        lines = [f"  Pool stats ({self._total:,} blocks):"]
        for pool in sorted(self._targets.keys()):
            target = self._targets[pool]
            actual_count = self._counts.get(pool, 0)
            actual_pct = actual_count / self._total
            delta = actual_pct - target
            marker = ""
            if abs(delta) >= 0.10:
                marker = " <<<"
            elif abs(delta) >= 0.05:
                marker = " <"
            lines.append(
                f"    {pool:>12s}: target={target:.3f} actual={actual_pct:.3f} "
                f"({actual_count:,} blocks){marker}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CURRICULUM V2 DATASET
# ═══════════════════════════════════════════════════════════════════════════


class CurriculumDatasetV2(IterableDataset):
    """
    IterableDataset that samples blocks from pool-separated shards
    according to curriculum v2 weights.

    Modes:
        "opus_candidates" — yields from D1-D4 only (for OPUS scoring)
        "always_on"       — yields from AON only (bypass OPUS)
        "combined"        — yields from all pools with AON injected at 8%
    """

    def __init__(
        self,
        shard_dir: str,
        manifest_dir: str,
        curriculum_config: CurriculumConfigV2,
        mode: str = "combined",
        seq_len: int = SHARD_BLOCK_SIZE,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        log_interval: int = 500,
    ) -> None:
        super().__init__()

        self._shard_dir = shard_dir
        self._config = curriculum_config
        self._mode = mode
        self._seq_len = seq_len
        self._rank = rank
        self._world_size = world_size
        self._seed = seed
        self._log_interval = log_interval
        self._blocks_per_seq = max(1, seq_len // SHARD_BLOCK_SIZE)

        _log("CurriculumDatasetV2 initializing...")
        _log(f"  shard_dir:    {shard_dir}")
        _log(f"  manifest_dir: {manifest_dir}")
        _log(f"  stage:        {curriculum_config.stage}")
        _log(f"  mode:         {mode}")
        _log(f"  seq_len:      {seq_len} ({self._blocks_per_seq} block(s)/seq)")
        _log(f"  rank:         {rank}/{world_size}")

        # Load shard lists from manifest
        manifest_path = Path(manifest_dir)
        self._pool_managers: Dict[str, _PoolShardManager] = {}

        pool_files = self._get_pool_files(mode)

        for pool_name, shard_file in pool_files.items():
            fpath = manifest_path / shard_file
            if not fpath.exists():
                _log(
                    f"  WARNING: Shard list {fpath} not found, skipping pool {pool_name}"
                )
                continue
            with open(fpath) as f:
                shard_paths = [line.strip() for line in f if line.strip()]

            mgr = _PoolShardManager(
                pool_name=pool_name,
                shard_root=shard_dir,
                shard_paths=shard_paths,
                rank=rank,
                world_size=world_size,
                seed=seed,
            )
            if not mgr.is_empty:
                self._pool_managers[pool_name] = mgr

        if not self._pool_managers:
            raise ValueError(f"No pools have shards for rank {rank} in mode {mode}")

        # Compute effective weights
        available_pools = list(self._pool_managers.keys())
        self._effective_weights = curriculum_config.effective_weights(
            available_pools, mode
        )

        _log("  Effective weights:")
        for pool, w in sorted(self._effective_weights.items()):
            _log(f"    {pool:>12s}: {w:.3f}")

        self._stats = _CurriculumStatsV2(self._effective_weights)
        _log(f"CurriculumDatasetV2 ready ({mode} mode).")

    def _get_pool_files(self, mode: str) -> Dict[str, str]:
        """Map pool names to manifest shard list files based on mode."""
        if mode == "opus_candidates":
            return {
                "D1": "D1_shards.txt",
                "D2": "D2_shards.txt",
                "D3": "D3_shards.txt",
                "D4": "D4_shards.txt",
            }
        elif mode == "always_on":
            return {
                "AON_bench": "AON_bench_train_shards.txt",
                "AON_indic": "AON_indic_shards.txt",
            }
        elif mode == "combined":
            return {
                "D1": "D1_shards.txt",
                "D2": "D2_shards.txt",
                "D3": "D3_shards.txt",
                "D4": "D4_shards.txt",
                "AON_bench": "AON_bench_train_shards.txt",
                "AON_indic": "AON_indic_shards.txt",
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self._seed + self._rank)
        pools = list(self._effective_weights.keys())
        weights = [self._effective_weights[p] for p in pools]

        seqs_emitted = 0
        t_start = time.time()
        ones = torch.ones(self._seq_len, dtype=torch.long)

        while True:
            token_chunks: List[np.ndarray] = []
            chosen_pools_this_seq = []

            for _ in range(self._blocks_per_seq):
                chosen_pool = rng.choices(pools, weights=weights, k=1)[0]
                self._stats.record(chosen_pool)
                chosen_pools_this_seq.append(chosen_pool)
                block = self._pool_managers[chosen_pool].next_block()
                token_chunks.append(block)

            tokens = np.concatenate(token_chunks).astype(np.int64)
            tokens = tokens[: self._seq_len]

            seq = torch.from_numpy(tokens)
            yield {
                "input_ids": seq,
                "attention_mask": ones,
                "labels": seq.clone(),
                "_pool": chosen_pools_this_seq[0],  # Primary pool for this seq
            }

            seqs_emitted += 1

            if seqs_emitted % self._log_interval == 0:
                elapsed = time.time() - t_start
                total_tokens = seqs_emitted * self._seq_len
                tps = total_tokens / elapsed if elapsed > 0 else 0
                _log(
                    f"[CurriculumV2 rank={self._rank} {self._mode}] "
                    f"{seqs_emitted:,} seqs, "
                    f"{total_tokens / 1e9:.2f}B tokens, "
                    f"{tps / 1e6:.2f}M tok/s"
                )
                _log(self._stats.summary())

    def close(self) -> None:
        for pool in self._pool_managers.values():
            pool.close()
        self._pool_managers.clear()

    @property
    def stats(self) -> _CurriculumStatsV2:
        return self._stats

    @property
    def effective_weights(self) -> Dict[str, float]:
        return dict(self._effective_weights)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC FACTORY
# ═══════════════════════════════════════════════════════════════════════════


def build_curriculum_v2_dataloader(
    shard_dir: str,
    manifest_dir: str,
    curriculum_path: str,
    stage: str,
    batch_size: int,
    mode: str = "combined",
    seq_len: int = SHARD_BLOCK_SIZE,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
    seed: int = 42,
    log_interval: int = 500,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> DataLoader:
    """
    Build a curriculum v2 DataLoader.

    Args:
        shard_dir:        Root directory containing band subdirectories
        manifest_dir:     Directory containing *_shards.txt manifest files
        curriculum_path:  Path to curriculum_v2.yaml
        stage:            Training stage ("1B", "WU_3B", "3B", etc.)
        batch_size:       Micro batch size per GPU
        mode:             "opus_candidates" | "always_on" | "combined"
        seq_len:          Sequence length (default 4096)
        num_workers:      DataLoader workers (0 = main process)
        prefetch_factor:  Prefetch batches per worker
        seed:             Random seed for reproducibility
        log_interval:     Print stats every N sequences
        rank:             GPU rank (auto-detected if None)
        world_size:       Total GPUs (auto-detected if None)

    Returns:
        DataLoader yielding {"input_ids", "attention_mask", "labels", "_pool"}
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
        f"Building curriculum v2 dataloader: stage={stage}, mode={mode}, "
        f"bs={batch_size}, seq_len={seq_len}, rank={rank}/{world_size}"
    )

    config = CurriculumConfigV2(curriculum_path, stage)

    dataset = CurriculumDatasetV2(
        shard_dir=shard_dir,
        manifest_dir=manifest_dir,
        curriculum_config=config,
        mode=mode,
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
        f"CurriculumV2DataLoader ready | stage={stage} | mode={mode} | "
        f"batch_size={batch_size} | seq_len={seq_len}"
    )

    return loader
