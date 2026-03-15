"""
Curriculum Dataloader v2 — manifest-driven, multi-pool, weighted sampling.

Three operating modes:
    opus_candidates : yields batches from D1-D4 only (for OPUS scoring)
    always_on       : yields batches from AON only (bench_train + indic)
    combined        : D1-D4 at curriculum weights + AON at 8% injection

Shard order is deterministic (seed-shuffled manifests, rank-striped).
Each pool tracks its own shard index for checkpoint/resume.

Usage:
    loader = build_curriculum_v2_dataloader(
        shard_root="/mnt/local-nvme/data/shards_reordered",
        manifest_dir="manifests",
        curriculum_path="configs/curriculum_v2.yaml",
        stage="1B",
        seq_len=4096,
        rank=0, world_size=8,
        mode="combined",
    )
    for batch in loader:
        # batch["input_ids"], batch["labels"], batch["_pool"]
        ...
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import IterableDataset

from src.bin_idx_dataloader import _iter_sequences_from_shard, SHARD_BLOCK_SIZE

logger = logging.getLogger(__name__)


def _print_rank_0(msg: str) -> None:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Pool — holds shard list + iteration state for one data pool
# ---------------------------------------------------------------------------

class _Pool:
    """One logical data pool (D1, D2, ..., AON_bench, AON_indic)."""

    def __init__(
        self,
        name: str,
        shard_paths: List[str],
        shard_root: str,
        seq_len: int,
        dtype: np.dtype = np.dtype("uint32"),
    ) -> None:
        self.name = name
        self.shard_root = shard_root
        self.seq_len = seq_len
        self.dtype = dtype

        # Full manifest-ordered shard list for this rank
        self.shard_paths = shard_paths
        self.total_shards = len(shard_paths)

        # Iteration state (for checkpoint tracking)
        self.current_shard_index: int = -1
        self.completed_count: int = 0
        self._iter: Optional[Iterator[torch.Tensor]] = None
        self._exhausted = False

    def _resolve(self, rel_path: str) -> str:
        return os.path.join(self.shard_root, rel_path, "tokens.bin")

    def next_sequence(self) -> Optional[torch.Tensor]:
        """Return next [seq_len] tensor, or None if pool exhausted."""
        while not self._exhausted:
            # Try current shard iterator
            if self._iter is not None:
                try:
                    return next(self._iter)
                except StopIteration:
                    self.completed_count += 1
                    self._iter = None

            # Advance to next shard
            next_idx = self.current_shard_index + 1
            if next_idx >= self.total_shards:
                self._exhausted = True
                return None

            self.current_shard_index = next_idx
            bin_path = self._resolve(self.shard_paths[next_idx])
            if not os.path.exists(bin_path):
                logger.warning("Pool %s: missing %s, skipping.", self.name, bin_path)
                self.completed_count += 1
                continue
            self._iter = _iter_sequences_from_shard(bin_path, self.dtype, self.seq_len)

        return None

    def get_state(self) -> Dict[str, Any]:
        return {
            "pool": self.name,
            "total_shards": self.total_shards,
            "current_shard_index": self.current_shard_index,
            "completed_count": self.completed_count,
            "remaining_count": max(self.total_shards - self.current_shard_index - 1, 0)
                if self.current_shard_index >= 0 else self.total_shards,
            "exhausted": self._exhausted,
        }

    def resume_from(self, shard_index: int) -> None:
        """Skip ahead to shard_index (for checkpoint resume)."""
        # Mark all shards before shard_index as completed
        skip_to = max(shard_index - 1, -1)
        self.current_shard_index = skip_to
        self.completed_count = max(shard_index, 0)
        self._iter = None
        self._exhausted = False


# ---------------------------------------------------------------------------
# CurriculumDatasetV2 — the IterableDataset
# ---------------------------------------------------------------------------

class CurriculumDatasetV2(IterableDataset):
    """
    Multi-pool curriculum dataset with weighted sampling.

    Reads pre-shuffled manifest files, stripes shards by rank, and samples
    pools according to stage weights. AON is injected at a fixed rate.
    """

    OPUS_POOLS = ("D1", "D2", "D3", "D4")
    AON_SUB_POOLS = ("AON_bench", "AON_indic")

    def __init__(
        self,
        shard_root: str,
        manifest_dir: str,
        curriculum_path: str,
        stage: str,
        seq_len: int = SHARD_BLOCK_SIZE,
        rank: int = 0,
        world_size: int = 1,
        mode: str = "combined",
        seed: int = 42,
        dtype: str = "uint32",
    ) -> None:
        super().__init__()
        self.shard_root = shard_root
        self.manifest_dir = manifest_dir
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.mode = mode
        self.seed = seed
        self._dtype = np.dtype(dtype)

        # Load curriculum config
        with open(curriculum_path, "r") as f:
            self._curriculum = yaml.safe_load(f)

        stage_cfg = self._curriculum["stages"].get(stage)
        if stage_cfg is None:
            raise ValueError(
                f"Unknown stage '{stage}'. Available: {list(self._curriculum['stages'].keys())}"
            )
        self._stage = stage
        self._weights = stage_cfg["band_weights"]

        # Load manifest JSON for pool→file mapping
        manifest_json_path = os.path.join(manifest_dir, "curriculum_v2_manifest.json")
        with open(manifest_json_path, "r") as f:
            self._manifest = json.load(f)

        # Build pools
        self.pools: Dict[str, _Pool] = {}
        self._build_pools()

        # Compute sampling weights for active pools
        self._pool_names: List[str] = []
        self._pool_weights: List[float] = []
        self._setup_sampling()

    def _load_manifest_shards(self, filename: str) -> List[str]:
        """Load shard paths from a manifest .txt file, stripe by rank."""
        path = os.path.join(self.manifest_dir, filename)
        with open(path, "r") as f:
            all_shards = [line.strip() for line in f if line.strip()]
        # Deterministic rank striping
        return all_shards[self.rank :: self.world_size]

    def _build_pools(self) -> None:
        manifest_pools = self._manifest["pools"]

        # D1-D4
        for pool_name in self.OPUS_POOLS:
            pool_def = manifest_pools[pool_name]
            shards = self._load_manifest_shards(pool_def["shard_list_file"])
            self.pools[pool_name] = _Pool(
                pool_name, shards, self.shard_root, self.seq_len, self._dtype,
            )

        # AON sub-pools
        aon_def = manifest_pools["AON"]["sub_pools"]
        bench_shards = self._load_manifest_shards(aon_def["bench_train"]["shard_list_file"])
        self.pools["AON_bench"] = _Pool(
            "AON_bench", bench_shards, self.shard_root, self.seq_len, self._dtype,
        )
        indic_shards = self._load_manifest_shards(aon_def["indic_guaranteed"]["shard_list_file"])
        self.pools["AON_indic"] = _Pool(
            "AON_indic", indic_shards, self.shard_root, self.seq_len, self._dtype,
        )

    def _setup_sampling(self) -> None:
        """Configure pool names and weights based on mode."""
        if self.mode == "opus_candidates":
            for name in self.OPUS_POOLS:
                self._pool_names.append(name)
                self._pool_weights.append(self._weights[name])
        elif self.mode == "always_on":
            aon_split = self._curriculum["aon_config"]["internal_split"]
            self._pool_names.append("AON_bench")
            self._pool_weights.append(aon_split["bench_train"])
            self._pool_names.append("AON_indic")
            self._pool_weights.append(aon_split["indic_guaranteed"])
        elif self.mode == "combined":
            # D1-D4 weights are scaled to (1 - AON_rate)
            aon_rate = self._weights.get("AON", 0.08)
            opus_scale = 1.0 - aon_rate
            for name in self.OPUS_POOLS:
                self._pool_names.append(name)
                self._pool_weights.append(self._weights[name] / opus_scale * opus_scale)
            # AON sub-pools split the AON allocation
            aon_split = self._curriculum["aon_config"]["internal_split"]
            self._pool_names.append("AON_bench")
            self._pool_weights.append(aon_rate * aon_split["bench_train"])
            self._pool_names.append("AON_indic")
            self._pool_weights.append(aon_rate * aon_split["indic_guaranteed"])
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Use: opus_candidates, always_on, combined")

        # Normalize
        total = sum(self._pool_weights)
        self._pool_weights = [w / total for w in self._pool_weights]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed + self.rank)
        seq_count = 0

        while True:
            # Weighted pool selection
            chosen_name = rng.choices(self._pool_names, weights=self._pool_weights, k=1)[0]
            pool = self.pools[chosen_name]
            seq = pool.next_sequence()

            if seq is None:
                # Pool exhausted — remove from sampling and re-normalize
                if chosen_name in self._pool_names:
                    idx = self._pool_names.index(chosen_name)
                    self._pool_names.pop(idx)
                    self._pool_weights.pop(idx)
                if not self._pool_names:
                    break  # All pools exhausted
                total = sum(self._pool_weights)
                self._pool_weights = [w / total for w in self._pool_weights]
                continue

            seq_count += 1
            yield {
                "input_ids": seq,
                "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                "labels": seq.clone(),
                "_pool": chosen_name,
            }

    def get_shard_state(self) -> Dict[str, Any]:
        """Per-pool shard indices for checkpoint metadata."""
        return {
            "stage": self._stage,
            "mode": self.mode,
            "rank": self.rank,
            "world_size": self.world_size,
            "pools": {name: pool.get_state() for name, pool in self.pools.items()},
        }

    def resume_from_state(self, state: Dict[str, Any]) -> None:
        """Resume iteration from a checkpoint's shard state."""
        pool_states = state.get("pools", {})
        for name, pstate in pool_states.items():
            if name in self.pools:
                idx = pstate.get("current_shard_index", -1)
                self.pools[name].resume_from(idx)
                _print_rank_0(
                    f"  Pool {name}: resuming from shard {idx}/{self.pools[name].total_shards}"
                )


# ---------------------------------------------------------------------------
# Builder function
# ---------------------------------------------------------------------------

def build_curriculum_v2_dataloader(
    shard_root: str,
    manifest_dir: str,
    curriculum_path: str,
    stage: str,
    batch_size: int = 1,
    seq_len: int = SHARD_BLOCK_SIZE,
    rank: int = 0,
    world_size: int = 1,
    mode: str = "combined",
    seed: int = 42,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader wrapping CurriculumDatasetV2."""
    dataset = CurriculumDatasetV2(
        shard_root=shard_root,
        manifest_dir=manifest_dir,
        curriculum_path=curriculum_path,
        stage=stage,
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
        mode=mode,
        seed=seed,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
