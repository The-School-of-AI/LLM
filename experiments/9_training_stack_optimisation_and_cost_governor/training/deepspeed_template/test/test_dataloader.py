"""
Benchmark: SPDL pipeline vs. standard PyTorch DataLoader
=========================================================

Simulates the full training-day data path:
  Phase 1 - mock "S3 -> NVMe staging"  (file copies)
  Phase 2 - benchmark STANDARD PyTorch DataLoader  (bin_idx_source -> DataLoader)
  Phase 3 - benchmark SPDL pipeline               (SPDLIterableDataset -> DataLoader)
  Phase 4 - print side-by-side comparison table

Each benchmark phase forks one process per simulated GPU rank so that the
file-level sharding logic in bin_idx_source (rank::world_size) is exercised.

Windows notes
-------------
* deepspeed is mocked to avoid import-time errors.
* worker processes call os._exit(0) to bypass torch / multiprocessing teardown
  errors that are benign but verbose on Windows.
"""

import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
import types
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ── Mock deepspeed so it can be imported on any machine ──────────────────────
class _DummyFlopsProfiler:
    pass


_ds = types.ModuleType("deepspeed")
_ds_prof = types.ModuleType("deepspeed.profiling")
_ds_acc = types.ModuleType("deepspeed.accelerator")
_ds_flops = types.ModuleType("deepspeed.profiling.flops_profiler")
_ds_flops.FlopsProfiler = _DummyFlopsProfiler
sys.modules.update(
    {
        "deepspeed": _ds,
        "deepspeed.profiling": _ds_prof,
        "deepspeed.accelerator": _ds_acc,
        "deepspeed.profiling.flops_profiler": _ds_flops,
    }
)

# ── Add project root so src.* sub-imports resolve ────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

# ── Stub the `src` package and its sub-modules to avoid triggering
# src/__init__.py (which imports train.py -> psutil, unavailable in spawned
# worker processes that use system Python rather than the venv).
import importlib
import importlib.util

# 1. Stub src package itself
_src_stub = types.ModuleType("src")
_src_stub.__path__ = [os.path.join(_PROJECT_ROOT, "src")]
_src_stub.__package__ = "src"
sys.modules["src"] = _src_stub  # unconditional: always use our clean stub

# 2. Stub src.utils so `from .utils import print_rank_0` inside data.py works.
#    data.py calls print_rank_0 for informational messages; route to print().
_utils_stub = types.ModuleType("src.utils")
_utils_stub.print_rank_0 = lambda msg, *a, **kw: print(msg)
_utils_stub.set_seed = lambda seed: None  # safe no-op for any callers
sys.modules["src.utils"] = _utils_stub
setattr(_src_stub, "utils", _utils_stub)

# 3. Load src.data directly (bypasses src/__init__.py entirely)
_data_spec = importlib.util.spec_from_file_location(
    "src.data",
    os.path.join(_PROJECT_ROOT, "src", "data.py"),
    submodule_search_locations=[],
)
data_module = importlib.util.module_from_spec(_data_spec)
data_module.__package__ = "src"  # needed for relative imports inside data.py
sys.modules["src.data"] = data_module
_data_spec.loader.exec_module(data_module)

from src.data import SPDLIterableDataset, bin_idx_source, get_dataloaders

# Try to detect whether SPDL is actually importable at runtime
try:
    from spdl.pipeline import PipelineBuilder  # noqa: F401

    SPDL_AVAILABLE = True
except Exception:
    SPDL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Shared result container written to a multiprocessing Queue
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RankResult:
    rank: int
    num_batches: int
    num_tokens: int
    elapsed: float
    fetch_time: float
    gpu_time: float

    @property
    def throughput(self):
        return self.num_tokens / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def wait_ratio(self):
        return self.fetch_time / self.elapsed if self.elapsed > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-rank worker functions
# ─────────────────────────────────────────────────────────────────────────────


def _run_standard_rank(
    rank: int,
    world_size: int,
    tokens_dir: str,
    result_queue: mp.Queue,
    max_length: int = 4096,
    batch_size: int = 8,
):
    """
    Benchmark one rank using a plain PyTorch DataLoader backed by bin_idx_source.
    No SPDL -- this is the baseline.
    """
    from torch.utils.data import DataLoader, IterableDataset

    # bin_idx_source prints "SPDL source: ..." via print_rank_0 -- suppress those
    # log lines in the standard baseline so output is not misleading.
    data_module.print_rank_0 = lambda *a, **kw: None

    class _BinIdxDataset(IterableDataset):
        def __init__(self, shard_dir, seq_len, rank, world_size):
            self.shard_dir = shard_dir
            self.seq_len = seq_len
            self.rank = rank
            self.world_size = world_size

        def __iter__(self):
            # Report progress under the [Standard] label, not "SPDL source:"
            bin_files = sorted(
                f for f in os.listdir(self.shard_dir) if f.endswith(".bin")
            )
            my_files = bin_files[self.rank :: self.world_size]
            print(
                f"    [Standard] Rank {self.rank}/{self.world_size} "
                f"scanning {len(my_files)} shards"
            )
            total = 0
            for seq in bin_idx_source(
                self.shard_dir,
                seq_len=self.seq_len,
                rank=self.rank,
                world_size=self.world_size,
            ):
                total += 1
                input_ids = seq  # shape [seq_len]
                yield {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": input_ids.clone(),
                }
            print(f"    [Standard] Rank {self.rank} yielded {total:,} sequences")

    dataset = _BinIdxDataset(tokens_dir, max_length, rank, world_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    t0 = time.perf_counter()
    num_tokens = 0
    num_batches = 0
    fetch_time = 0.0
    gpu_time = 0.0

    t_fetch_start = time.perf_counter()
    for batch in loader:
        t_fetch_end = time.perf_counter()
        fetch_time += t_fetch_end - t_fetch_start

        num_batches += 1
        num_tokens += batch["input_ids"].numel()

        t_gpu_start = time.perf_counter()
        time.sleep(0.005)  # simulate forward/backward
        t_gpu_end = time.perf_counter()
        gpu_time += t_gpu_end - t_gpu_start

        t_fetch_start = time.perf_counter()

    elapsed = time.perf_counter() - t0
    result = RankResult(rank, num_batches, num_tokens, elapsed, fetch_time, gpu_time)
    result_queue.put(result)
    sys.stdout.flush()
    os._exit(0)


def _run_spdl_rank(
    rank: int,
    world_size: int,
    tokens_dir: str,
    result_queue: mp.Queue,
    max_length: int = 4096,
    batch_size: int = 8,
):
    """
    Benchmark one rank using the SPDL pipeline (SPDLIterableDataset).
    """
    # Override the distributed context so SPDLIterableDataset uses correct rank
    data_module._resolve_distributed_context = lambda: (True, world_size, rank)

    # Prefix all internal data_module log lines with [SPDL] for clarity
    data_module.print_rank_0 = lambda msg, *a, **kw: print(f"    [SPDL] {msg}")

    from torch.utils.data import DataLoader

    dataset = SPDLIterableDataset(
        shard_dir=tokens_dir,
        seq_len=max_length,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        num_threads=4,
        prefetch_buffer=16,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)

    t0 = time.perf_counter()
    num_tokens = 0
    num_batches = 0
    fetch_time = 0.0
    gpu_time = 0.0

    t_fetch_start = time.perf_counter()
    for batch in loader:
        t_fetch_end = time.perf_counter()
        fetch_time += t_fetch_end - t_fetch_start

        num_batches += 1
        num_tokens += batch["input_ids"].numel()

        t_gpu_start = time.perf_counter()
        time.sleep(0.005)  # simulate forward/backward
        t_gpu_end = time.perf_counter()
        gpu_time += t_gpu_end - t_gpu_start

        t_fetch_start = time.perf_counter()

    elapsed = time.perf_counter() - t0
    result = RankResult(rank, num_batches, num_tokens, elapsed, fetch_time, gpu_time)
    result_queue.put(result)
    sys.stdout.flush()
    os._exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-process runner
# ─────────────────────────────────────────────────────────────────────────────


def _run_distributed(
    target_fn,
    world_size: int,
    tokens_dir: str,
    max_length: int,
    batch_size: int,
    label: str,
) -> list[RankResult]:
    """Fork *world_size* processes running *target_fn* and collect RankResults."""
    print(f"\n  Launching {world_size} parallel processes for [{label}]...")

    manager = mp.Manager()
    result_queue = manager.Queue()

    processes = []
    t0 = time.perf_counter()
    for rank in range(world_size):
        p = mp.Process(
            target=target_fn,
            args=(rank, world_size, tokens_dir, result_queue, max_length, batch_size),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    wall_clock = time.perf_counter() - t0

    results: list[RankResult] = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())
    results.sort(key=lambda r: r.rank)

    print(f"  Wall-clock time: {wall_clock:.2f}s")
    for r in results:
        print(
            f"    Rank {r.rank}: {r.num_batches} batches | "
            f"{r.num_tokens:,} tokens | "
            f"{r.throughput:,.0f} tok/s | "
            f"DataLoader wait={r.fetch_time:.2f}s "
            f"({r.wait_ratio:.1%} of total)"
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table printer
# ─────────────────────────────────────────────────────────────────────────────


def _print_comparison(std_results: list[RankResult], spdl_results: list[RankResult]):
    """Print a side-by-side per-rank comparison table and aggregate summary."""

    print("\n" + "=" * 72)
    print("BENCHMARK COMPARISON: Standard DataLoader vs SPDL Pipeline")
    print("=" * 72)

    hdr = (
        f"{'Rank':>4}  "
        f"{'Batches':>9}  "
        f"{'Tokens':>12}  "
        f"{'Thruput (tok/s)':>16}  "
        f"{'FetchWait (s)':>14}  "
        f"{'WaitRatio':>10}"
    )
    sep = "-" * 72

    for label, results in (
        ("Standard PyTorch DataLoader", std_results),
        ("SPDL Pipeline", spdl_results),
    ):
        print(f"\n  {label}")
        print("  " + hdr)
        print("  " + sep)
        for r in results:
            print(
                f"  {r.rank:>4}  "
                f"{r.num_batches:>9,}  "
                f"{r.num_tokens:>12,}  "
                f"{r.throughput:>16,.0f}  "
                f"{r.fetch_time:>14.2f}  "
                f"{r.wait_ratio:>9.1%}"
            )

    # Aggregate speedup
    def _agg(results):
        total_tok = sum(r.num_tokens for r in results)
        total_sec = max(r.elapsed for r in results)  # wall clock ≈ slowest rank
        total_wait = sum(r.fetch_time for r in results)
        return total_tok, total_sec, total_wait

    print(f"\n  {'Metric':<28} {'Standard':>16} {'SPDL':>14} {'SPDL Speedup':>12}")
    print("  " + sep)

    std_tok, std_sec, std_wait = _agg(std_results)
    spdl_tok, spdl_sec, spdl_wait = _agg(spdl_results)

    std_thru = std_tok / std_sec if std_sec > 0 else 0
    spdl_thru = spdl_tok / spdl_sec if spdl_sec > 0 else 0
    thru_ratio = spdl_thru / std_thru if std_thru > 0 else float("nan")

    std_wait_r = std_wait / (std_sec * len(std_results)) if std_sec > 0 else 0
    spdl_wait_r = spdl_wait / (spdl_sec * len(spdl_results)) if spdl_sec > 0 else 0

    def _pct_improvement(new, old):
        if old == 0:
            return "N/A"
        return (
            f"{(old - new) / old:.1%} (lower)"
            if new < old
            else f"{(new - old) / old:.1%} (higher)"
        )

    print(
        f"  {'Aggregate throughput (tok/s)':<28} {std_thru:>16,.0f} {spdl_thru:>14,.0f} {thru_ratio:>11.2f}x"
    )
    print(
        f"  {'Avg DataLoader wait ratio':<28} {std_wait_r:>15.1%} {spdl_wait_r:>13.1%} {_pct_improvement(spdl_wait_r, std_wait_r):>12}"
    )
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── Locate source tokens directory ───────────────────────────────────────
    original_tokens_dir = os.path.abspath(
        os.path.join(_PROJECT_ROOT, "..", "..", "tokens")
    )
    if not os.path.exists(original_tokens_dir):
        print(f"ERROR: Could not find sample tokens directory at {original_tokens_dir}")
        sys.exit(1)

    # Benchmark parameters
    MAX_LENGTH = 4096
    BATCH_SIZE = 8
    WORLD_SIZE = 2

    print(
        f"Benchmark config: block_size={MAX_LENGTH}, batch_size={BATCH_SIZE}, "
        f"world_size={WORLD_SIZE}"
    )

    # =========================================================================
    # Phase 1 - Simulate S3 -> NVMe staging
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 1: Simulating S3 -> local NVMe staging")

    nvme_dir = tempfile.mkdtemp(prefix="mock_nvme_")
    print(f"  Created mock NVMe staging directory: {nvme_dir}")

    # Locate a single .bin/.idx pair in the tokens directory
    bin_file = idx_file = None
    for fname in os.listdir(original_tokens_dir):
        if fname.endswith(".bin"):
            bin_file = os.path.join(original_tokens_dir, fname)
        elif fname.endswith(".idx"):
            idx_file = os.path.join(original_tokens_dir, fname)

    if bin_file is None or idx_file is None:
        print("ERROR: No .bin/.idx file pair found in tokens directory.")
        shutil.rmtree(nvme_dir, ignore_errors=True)
        sys.exit(1)

    NUM_SHARDS = 8  # Distribute across WORLD_SIZE GPUs
    t0_stage = time.perf_counter()
    total_bytes = 0
    for i in range(NUM_SHARDS):
        new_bin = os.path.join(nvme_dir, f"shard_{i:03d}.bin")
        new_idx = os.path.join(nvme_dir, f"shard_{i:03d}.idx")
        shutil.copy2(bin_file, new_bin)
        shutil.copy2(idx_file, new_idx)
        total_bytes += os.path.getsize(new_bin) + os.path.getsize(new_idx)

    stage_elapsed = time.perf_counter() - t0_stage
    print(
        f"  Staged {NUM_SHARDS} shards ({total_bytes / 1024**2:.1f} MB) "
        f"in {stage_elapsed:.2f}s "
        f"({total_bytes / 1024**2 / stage_elapsed:.1f} MB/s)"
    )

    # =========================================================================
    # Phase 2 - Standard PyTorch DataLoader (baseline, no SPDL)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Standard PyTorch DataLoader (baseline)")

    std_results = _run_distributed(
        target_fn=_run_standard_rank,
        world_size=WORLD_SIZE,
        tokens_dir=nvme_dir,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        label="Standard",
    )

    # =========================================================================
    # Phase 3 - SPDL pipeline
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 3: SPDL pipeline")

    if not SPDL_AVAILABLE:
        print("  [SKIP] SPDL is not available in the current environment.")
        print("         Install with: pip install spdl")
        shutil.rmtree(nvme_dir, ignore_errors=True)
        sys.exit(0)

    spdl_results = _run_distributed(
        target_fn=_run_spdl_rank,
        world_size=WORLD_SIZE,
        tokens_dir=nvme_dir,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        label="SPDL",
    )

    # =========================================================================
    # Phase 4 - Side-by-side comparison
    # =========================================================================
    if std_results and spdl_results:
        _print_comparison(std_results, spdl_results)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Cleaning up mock NVMe directory...")
    time.sleep(0.5)
    shutil.rmtree(nvme_dir, ignore_errors=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
