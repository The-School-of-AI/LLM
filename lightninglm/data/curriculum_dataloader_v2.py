"""
Curriculum Dataloader v2 — manifest-driven, multi-pool, weighted sampling.

Three operating modes:
    opus_candidates : yields batches from D1-D4 only (for OPUS scoring)
    always_on       : yields batches from AON only (bench_train + indic)
    combined        : D1-D4 at curriculum weights + AON at 25% (8 of 32)

Shard order is deterministic (seed-shuffled manifests, rank-striped).
Each pool tracks its own shard index for checkpoint/resume.

Usage:
    loader = build_curriculum_v2_dataloader(
        shard_root="data/training_shards_8k",
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

import ctypes
import json
import logging
import multiprocessing as mp
import os
import random
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import IterableDataset

from lightninglm.data.bin_idx_dataloader import (
    SHARD_BLOCK_SIZE,
    _iter_sequences_from_shard,
)

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
        self._sequences_yielded: int = 0  # offset within current shard
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
                    seq = next(self._iter)
                    self._sequences_yielded += 1
                    return seq
                except StopIteration:
                    self.completed_count += 1
                    self._iter = None

            # Advance to next shard
            next_idx = self.current_shard_index + 1
            if next_idx >= self.total_shards:
                self._exhausted = True
                return None

            self.current_shard_index = next_idx
            self._sequences_yielded = 0  # reset offset for new shard
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
            "remaining_count": (
                max(self.total_shards - self.current_shard_index - 1, 0)
                if self.current_shard_index >= 0
                else self.total_shards
            ),
            "sequence_offset": self._sequences_yielded,
            "exhausted": self._exhausted,
        }

    def resume_from(self, shard_index: int, sequence_offset: int = 0) -> None:
        """Skip ahead to shard_index and fast-forward within the shard.

        Args:
            shard_index: Resume from this shard (shards before it are marked completed).
            sequence_offset: Number of sequences to skip within the shard.
                Each shard is ~8192 sequences at seq_len=4096. Without this,
                resuming mid-shard loses up to ~6 hours of progress.
        """
        # Mark all shards before shard_index as completed
        skip_to = max(shard_index - 1, -1)
        self.current_shard_index = skip_to
        self.completed_count = max(shard_index, 0)
        self._sequences_yielded = 0
        self._iter = None
        self._exhausted = False

        # If we have an offset, open the shard and skip sequences
        if sequence_offset > 0 and 0 <= shard_index < self.total_shards:
            bin_path = self._resolve(self.shard_paths[shard_index])
            if os.path.exists(bin_path):
                self.current_shard_index = shard_index
                self._iter = _iter_sequences_from_shard(
                    bin_path, self.dtype, self.seq_len
                )
                skipped = 0
                for _ in range(sequence_offset):
                    try:
                        next(self._iter)
                        skipped += 1
                    except StopIteration:
                        # Shard exhausted during skip — mark completed, move on
                        self.completed_count += 1
                        self._iter = None
                        break
                self._sequences_yielded = skipped


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
        num_workers: int = 0,
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
        self._num_workers = max(num_workers, 1)  # treat 0 as 1 for indexing

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

        # Pool-selection RNG (saved/restored for bit-exact resume)
        self._rng: Optional[random.Random] = None
        self._rng_state_to_restore = None

        # Compute sampling weights for active pools
        self._pool_names: List[str] = []
        self._pool_weights: List[float] = []
        self._setup_sampling()

        # ── Shared-memory progress tracking for multi-worker checkpointing ──
        # Layout per pool: [worker_0_shard_idx, worker_0_seq_offset,
        #                   worker_1_shard_idx, worker_1_seq_offset, ...]
        # Total ints: num_pools * num_workers * 2
        # Workers write their local (shard_index, seq_offset) here so the
        # main process can read an accurate aggregate in get_shard_state().
        self._pool_order: List[str] = sorted(self.pools.keys())
        n_slots = len(self._pool_order) * self._num_workers * 2
        self._progress = mp.Array(ctypes.c_long, n_slots, lock=False)
        # Initialize to -1 (no progress yet)
        for i in range(n_slots):
            self._progress[i] = -1

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
                pool_name,
                shards,
                self.shard_root,
                self.seq_len,
                self._dtype,
            )

        # AON sub-pools
        aon_def = manifest_pools["AON"]["sub_pools"]
        bench_shards = self._load_manifest_shards(
            aon_def["bench_train"]["shard_list_file"]
        )
        self.pools["AON_bench"] = _Pool(
            "AON_bench",
            bench_shards,
            self.shard_root,
            self.seq_len,
            self._dtype,
        )
        indic_shards = self._load_manifest_shards(
            aon_def["indic_guaranteed"]["shard_list_file"]
        )
        self.pools["AON_indic"] = _Pool(
            "AON_indic",
            indic_shards,
            self.shard_root,
            self.seq_len,
            self._dtype,
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
            aon_rate = self._weights.get("AON", 0.25)
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
            raise ValueError(
                f"Unknown mode '{self.mode}'. Use: opus_candidates, always_on, combined"
            )

        # Normalize
        total = sum(self._pool_weights)
        self._pool_weights = [w / total for w in self._pool_weights]

    def _progress_offset(self, pool_name: str, worker_id: int) -> int:
        """Return the index into self._progress for (pool, worker)."""
        pool_idx = self._pool_order.index(pool_name)
        return (pool_idx * self._num_workers + worker_id) * 2

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # ── Multi-worker support ─────────────────────────────────────────
        # When num_workers > 0, PyTorch spawns N worker processes, each
        # getting a *copy* of this IterableDataset.  Without splitting,
        # every worker iterates the same shards → duplicate data → the
        # loss sawtooth pattern (resets every ~num_workers steps).
        #
        # Fix: each worker builds its OWN set of pools containing only
        # its slice of the shard lists.  Worker i takes shards
        # i, i+nw, i+2*nw, ... from each pool's manifest.
        # ─────────────────────────────────────────────────────────────────
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # Build per-worker pools with striped shard lists
        worker_pools: Dict[str, _Pool] = {}
        for name, pool in self.pools.items():
            worker_shards = pool.shard_paths[worker_id::num_workers]
            wp = _Pool(name, worker_shards, self.shard_root, self.seq_len, self._dtype)
            # If we have a resume state (sequence_offset, shard_index), apply it.
            # The checkpoint stores the *minimum* global shard index across all
            # workers (conservative resume point).  All global shards < g are
            # guaranteed consumed.  Shard g itself is partially consumed.
            #
            # Worker striping: worker k owns global shards k, k+nw, k+2*nw, ...
            # So worker k's local index for global shard g_k is g_k // nw,
            # and g_k = local * nw + k.
            #
            # For each worker, we need to skip all its local shards whose
            # global index is < g (fully consumed), and resume mid-shard if
            # the worker owns shard g exactly.
            if pool.current_shard_index >= 0:
                g = pool.current_shard_index  # min global shard from checkpoint
                owner_worker = g % num_workers
                local_idx = g // num_workers

                if worker_id == owner_worker:
                    # This worker owns the in-progress shard
                    wp.resume_from(local_idx, sequence_offset=pool._sequences_yielded)
                elif worker_id < owner_worker:
                    # This worker's global shard at local_idx is
                    # (local_idx * nw + worker_id) < g, so it's consumed.
                    # Start at local_idx + 1.
                    if local_idx + 1 < len(worker_shards):
                        wp.resume_from(local_idx + 1, sequence_offset=0)
                    else:
                        wp._exhausted = True
                else:
                    # worker_id > owner_worker: this worker's global shard at
                    # local_idx is (local_idx * nw + worker_id) > g, so it
                    # has NOT been consumed yet.  Start from local_idx.
                    if local_idx < len(worker_shards):
                        wp.resume_from(local_idx, sequence_offset=0)
                    # else: no shards left for this worker
            worker_pools[name] = wp

        # Each worker gets a unique RNG so pool selection doesn't repeat
        rng_seed = self.seed + self.rank * 10000 + worker_id
        rng = random.Random(rng_seed)

        # Restore RNG state only for worker 0 (checkpoint state)
        if worker_id == 0 and self._rng_state_to_restore is not None:
            rng.setstate(self._rng_state_to_restore)
            self._rng_state_to_restore = None

        # If this is the main-process iterator (num_workers=0), keep self._rng
        # in sync for get_shard_state() / checkpoint.
        if num_workers == 1:
            self._rng = rng

        # Local copies of pool names/weights so exhaustion doesn't mutate self
        active_names = list(self._pool_names)
        active_weights = list(self._pool_weights)

        # Reference to shared progress array (survives fork)
        progress = self._progress

        while True:
            chosen_name = rng.choices(active_names, weights=active_weights, k=1)[0]
            pool = worker_pools[chosen_name]
            seq = pool.next_sequence()

            if seq is None:
                # Pool exhausted — remove from active set
                if chosen_name in active_names:
                    idx = active_names.index(chosen_name)
                    active_names.pop(idx)
                    active_weights.pop(idx)
                if not active_names:
                    break
                total = sum(active_weights)
                active_weights = [w / total for w in active_weights]
                continue

            # Write progress to shared memory (lock-free; main process reads
            # atomically enough for checkpoint — off-by-one is fine).
            # Convert local shard index back to global index for the pool.
            local_si = pool.current_shard_index
            global_si = local_si * num_workers + worker_id
            off = self._progress_offset(chosen_name, worker_id)
            progress[off] = global_si
            progress[off + 1] = pool._sequences_yielded

            yield {
                "input_ids": seq,
                "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                "labels": seq.clone(),
                "_pool": chosen_name,
            }

    def get_shard_state(self) -> Dict[str, Any]:
        """Per-pool shard progress for checkpoint metadata.

        With num_workers > 0, iteration happens in forked workers.  Each
        worker writes its (global_shard_index, sequence_offset) into a
        shared-memory array.  We aggregate here: the *minimum* global
        shard index across workers is the safe resume point (everything
        before it has been consumed by all workers).
        """
        pools_state: Dict[str, Dict[str, Any]] = {}
        nw = self._num_workers

        for name in self._pool_order:
            pool = self.pools[name]
            min_shard = -1
            min_offset = 0

            # Read per-worker progress from shared memory
            for wid in range(nw):
                off = self._progress_offset(name, wid)
                w_shard = self._progress[off]
                w_seq = self._progress[off + 1]

                if w_shard < 0:
                    # Worker hasn't started this pool yet — skip
                    continue

                if min_shard < 0 or w_shard < min_shard:
                    min_shard = w_shard
                    min_offset = w_seq
                elif w_shard == min_shard:
                    # Same shard — take the smaller offset (conservative)
                    min_offset = min(min_offset, w_seq)

            pools_state[name] = {
                "pool": name,
                "total_shards": pool.total_shards,
                "current_shard_index": min_shard,
                "completed_count": max(min_shard, 0),
                "remaining_count": (
                    max(pool.total_shards - min_shard - 1, 0)
                    if min_shard >= 0
                    else pool.total_shards
                ),
                "sequence_offset": max(min_offset, 0),
                "exhausted": min_shard >= pool.total_shards - 1 and min_offset <= 0,
            }

        state = {
            "stage": self._stage,
            "mode": self.mode,
            "rank": self.rank,
            "world_size": self.world_size,
            "num_workers": nw,
            "pools": pools_state,
        }
        # Save pool-selection RNG for bit-exact reproducibility on resume
        if self._rng is not None:
            state["pool_rng_state"] = self._rng.getstate()
        return state

    def resume_from_state(self, state: Dict[str, Any]) -> None:
        """Resume iteration from a checkpoint's shard state."""
        pool_states = state.get("pools", {})
        for name, pstate in pool_states.items():
            if name in self.pools:
                idx = pstate.get("current_shard_index", -1)
                seq_off = pstate.get("sequence_offset", 0)
                self.pools[name].resume_from(idx, sequence_offset=seq_off)
                _print_rank_0(
                    f"  Pool {name}: resuming from shard {idx}/{self.pools[name].total_shards}"
                    f", sequence_offset={seq_off}"
                )
        # Restore pool-selection RNG for bit-exact reproducibility
        rng_state = state.get("pool_rng_state")
        if rng_state is not None:
            self._rng_state_to_restore = rng_state
            _print_rank_0("  Pool-selection RNG state restored from checkpoint")


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
        num_workers=num_workers,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
