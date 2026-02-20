"""
Data loading utilities for DeepSpeed training.

This module provides functions for loading tokenizers and creating dataloaders
for training language models.

Supports three modes:
1. Offline: load_from_disk for pre-tokenized datasets (preferred for production)
2. Online: download + tokenize on-the-fly (fallback for dev/testing)
3. Streaming: HuggingFace streaming for very large datasets like FineWeb
   (no disk download, tokenizes on-the-fly)

Uses the standard LLM pre-training approach: concatenate all text, tokenize,
then chunk into fixed-length sequences. Every token is a real token — no
padding waste.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from spdl.pipeline import PipelineBuilder
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from .shard_tracker import ShardTracker
from .utils import print_rank_0


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset that tokenizes + chunks text on-the-fly.

    For use with very large datasets (e.g., FineWeb) where downloading
    would be impractical. Tokenizes each example, maintains a token buffer,
    and yields fixed-length chunks.

    Multi-GPU: uses worker_info + distributed rank to shard the stream.
    """

    def __init__(self, hf_dataset_iter, tokenizer, max_length, split_name="train"):
        self.hf_dataset_iter = hf_dataset_iter
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split_name = split_name

    def __iter__(self):
        buffer = []
        for example in self.hf_dataset_iter:
            text = example.get("text", "")
            if not text or not text.strip():
                continue
            tokens = self.tokenizer(text, return_attention_mask=False)["input_ids"]
            buffer.extend(tokens)

            # Yield full chunks from the buffer
            while len(buffer) >= self.max_length:
                chunk = buffer[: self.max_length]
                buffer = buffer[self.max_length :]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                attention_mask = torch.ones(self.max_length, dtype=torch.long)
                labels = input_ids.clone()
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }


# ============================================================================
# SPDL pipeline
# ============================================================================


def read_idx(idx_path: str) -> np.ndarray:
    """Read a Megatron .idx file and return an array of byte offsets."""
    with open(idx_path, "rb") as f:
        _ = f.read(8)  # Skip 8-byte header
        offsets = np.frombuffer(f.read(), dtype=np.uint64)
    return offsets


def _should_skip_region(
    bin_path: str,
    i: int,
    start: int,
    end: int,
    itemsize: int,
    seq_len: int,
) -> bool:
    """Check if a .bin region should be skipped (corrupt/too small)."""
    num_bytes = end - start
    if num_bytes <= 0:
        print_rank_0(f"WARNING: Corrupt/empty region: {bin_path} [{i}] {start}-{end}")
        return True
    num_tokens = num_bytes // itemsize
    if num_tokens == 0 or num_tokens // seq_len == 0:
        return True
    return False


def bin_idx_source(
    shard_dir: str,
    seq_len: int = 4096,
    dtype: np.dtype = np.uint32,
    rank: int = 0,
    world_size: int = 1,
    exclude_files: Optional[Set[str]] = None,
    on_shard_complete: Optional[Callable[[str], None]] = None,
):
    """
    Generator yielding torch tensors of shape [seq_len] from .bin/.idx shards.

    Handles distributed file-level sharding: files are split across ranks
    using round-robin assignment.

    Args:
        shard_dir: Directory containing .bin/.idx shard files.
        seq_len: Sequence length for each yielded tensor.
        dtype: NumPy dtype of the token IDs stored in .bin files.
        rank: Current distributed rank.
        world_size: Total number of distributed ranks.
        exclude_files: Set of shard basenames to skip (already processed).
        on_shard_complete: Callback invoked with the shard basename after
            all sequences from that shard have been yielded.

    Yields:
        torch.Tensor of shape (seq_len,) with dtype torch.long
    """
    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize
    bin_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".bin"))
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {shard_dir}")

    # Distributed file-level sharding
    # CRITICAL: Sharding must happen BEFORE exclusion to maintain consistent
    # rank assignments across resumes.
    if world_size > 1:
        bin_files = bin_files[rank::world_size]

    # Exclude already-processed shards
    if exclude_files:
        pre_count = len(bin_files)
        bin_files = [f for f in bin_files if f not in exclude_files]
        skipped = pre_count - len(bin_files)
        if skipped > 0:
            print_rank_0(
                f"SPDL source: Rank {rank} excluding {skipped} already-processed "
                f"shards ({pre_count} total, {len(bin_files)} remaining)"
            )

    if not bin_files:
        print_rank_0(
            f"SPDL source: Rank {rank} has no remaining shards after exclusion"
        )
        return

    print_rank_0(
        f"SPDL source: Rank {rank}/{world_size} scanning "
        f"{len(bin_files)} shards from {shard_dir}"
    )

    total_yielded = 0
    for bin_file in bin_files:
        bin_path = os.path.join(shard_dir, bin_file)
        idx_path = bin_path.replace(".bin", ".idx")
        if not os.path.exists(idx_path):
            print_rank_0(f"WARNING: Missing .idx for {bin_file}, skipping")
            continue

        offsets = read_idx(idx_path)
        with open(bin_path, "rb") as f:
            for i in range(len(offsets) - 1):
                start = int(offsets[i])
                end = int(offsets[i + 1])
                if _should_skip_region(bin_path, i, start, end, itemsize, seq_len):
                    continue
                num_tokens = (end - start) // itemsize
                num_full = num_tokens // seq_len
                read_tokens = num_full * seq_len

                f.seek(start)
                raw = f.read(read_tokens * itemsize)
                tokens = np.frombuffer(raw, dtype=dtype)
                if tokens.size != read_tokens:
                    continue

                for j in range(0, len(tokens), seq_len):
                    total_yielded += 1
                    yield torch.from_numpy(tokens[j : j + seq_len].astype(np.int64))

        # Notify caller that this shard has been fully consumed
        if on_shard_complete is not None:
            on_shard_complete(bin_file)

    print_rank_0(f"SPDL source: Rank {rank} yielded {total_yielded:,} sequences")


def build_spdl_pipeline(
    shard_dir: str,
    seq_len: int = 4096,
    batch_size: int = 8,
    dtype: np.dtype = np.uint32,
    rank: int = 0,
    world_size: int = 1,
    num_threads: int = 4,
    prefetch_buffer: int = 16,
    exclude_files: Optional[Set[str]] = None,
    on_shard_complete: Optional[Callable[[str], None]] = None,
):
    """
    Build an SPDL pipeline for streaming .bin/.idx shards.

    Pipeline stages:
      1. bin_idx_source: yields individual sequences as torch tensors
      2. aggregate: collects batch_size sequences
      3. sink: prefetch buffer for async consumption

    Args:
        exclude_files: Set of shard basenames to skip (already processed).
        on_shard_complete: Callback invoked after each shard is fully consumed.

    Returns:
        SPDL Pipeline object (context manager, iterable)
    """

    source = bin_idx_source(
        shard_dir,
        seq_len=seq_len,
        dtype=dtype,
        rank=rank,
        world_size=world_size,
        exclude_files=exclude_files,
        on_shard_complete=on_shard_complete,
    )

    return (
        PipelineBuilder()
        .add_source(source)
        .aggregate(batch_size)
        .add_sink(prefetch_buffer)
        .build(num_threads=num_threads)
    )


class SPDLIterableDataset(IterableDataset):
    """
    Wraps an SPDL pipeline as a PyTorch IterableDataset.

    This allows using the SPDL pipeline with a standard PyTorch DataLoader
    while preserving SPDL's async prefetching and thread pool.

    Each iteration yields a dict with 'input_ids', 'attention_mask', 'labels'.

    When a ``shard_tracker`` is provided, already-processed shards are excluded
    at file-discovery time and newly consumed shards are marked in the tracker
    as they complete.
    """

    def __init__(
        self,
        shard_dir: str,
        seq_len: int = 4096,
        batch_size: int = 8,
        dtype: np.dtype = np.uint32,
        rank: int = 0,
        world_size: int = 1,
        num_threads: int = 4,
        prefetch_buffer: int = 16,
        shard_tracker: Optional[ShardTracker] = None,
    ):
        self.shard_dir = shard_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.rank = rank
        self.world_size = world_size
        self.num_threads = num_threads
        self.prefetch_buffer = prefetch_buffer
        self.shard_tracker = shard_tracker

    def __iter__(self):
        # Resolve exclude set and completion callback from the tracker
        exclude_files: Optional[Set[str]] = None
        on_shard_complete: Optional[Callable[[str], None]] = None
        if self.shard_tracker is not None:
            exclude_files = self.shard_tracker.get_processed_files()
            on_shard_complete = lambda name: self.shard_tracker.mark_processed(
                name, rank=self.rank, source_dir=self.shard_dir
            )

        pipeline = build_spdl_pipeline(
            self.shard_dir,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            dtype=self.dtype,
            rank=self.rank,
            world_size=self.world_size,
            num_threads=self.num_threads,
            prefetch_buffer=self.prefetch_buffer,
            exclude_files=exclude_files,
            on_shard_complete=on_shard_complete,
        )
        pipeline.start()
        try:
            for batch_list in pipeline:
                # batch_list is a list of batch_size tensors, each [seq_len]
                input_ids = torch.stack(batch_list)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": input_ids.clone(),
                }
        finally:
            pipeline.stop()


# ============================================================================
# Dataloader
# ============================================================================


def get_tokenizer(tokenizer_path: str = None):
    """
    Load and configure the TSAI 131K tokenizer.

    Args:
        tokenizer_path: Path to the tokenizer directory (default: src/tokenizer/)

    Returns:
        Configured tokenizer instance (TSAI 131K - 2^17 vocab size)
    """
    import os

    if tokenizer_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_path = os.path.join(current_dir, "tokenizer")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"TSAI 131K tokenizer not found at: {tokenizer_path}\n"
            "Expected directory structure: src/tokenizer/ with tokenizer.json, "
            "tokenizer_config.json, and special_tokens_map.json"
        )

    print_rank_0(f"  Loading TSAI 131K tokenizer from: {tokenizer_path}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print_rank_0(f"  Tokenizer loaded:")
    print_rank_0(f"    - Vocab size: {tokenizer.vocab_size:,}")
    print_rank_0(f"    - Total tokens (with special): {len(tokenizer):,}")
    print_rank_0(
        f"    - BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})"
    )
    print_rank_0(
        f"    - EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})"
    )
    print_rank_0(
        f"    - PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})"
    )

    return tokenizer


def tokenize_function(
    examples: Dict[str, List[str]],
    tokenizer,
    max_length: Optional[int] = None,
    pad_to_max_length: bool = False,
    truncation: bool = True,
) -> Dict[str, List[List[int]]]:
    """
    Tokenize text examples for language modeling.

    Args:
        examples: Dictionary with 'text' key containing text examples
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length (used only when truncation=True)
        pad_to_max_length: If True, pad every sample to max_length
        truncation: If True, truncate to max_length

    Returns:
        Dictionary with tokenized inputs
    """
    if truncation and max_length is None:
        raise ValueError("max_length must be provided when truncation=True")

    tokenized = tokenizer(
        examples["text"],
        truncation=truncation,
        padding="max_length" if pad_to_max_length else False,
        max_length=max_length if truncation else None,
        return_tensors=None,
    )

    tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
    return tokenized


def _is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _stage_s3_dataset_to_local_nvme(s3_uri: str, local_nvme_cache_dir: str) -> str:
    """
    Download a dataset directory from S3 to local NVMe and return local path.
    """
    try:
        import boto3
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "boto3 is required for S3 dataset staging but is not available."
        ) from exc

    bucket, prefix = _parse_s3_uri(s3_uri)
    if not prefix:
        raise ValueError(
            "S3 dataset URI must include a prefix, e.g. s3://bucket/path/to/dataset"
        )

    nvme_root = Path(local_nvme_cache_dir).expanduser().resolve()
    local_dir = nvme_root / bucket / prefix
    marker = local_dir / "_STAGED_COMPLETE"

    if marker.exists():
        print_rank_0(f"Using existing staged dataset on NVMe: {local_dir}")
        return str(local_dir)

    print_rank_0(f"Staging tokenized dataset from {s3_uri} to NVMe: {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")

    file_count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix) :].lstrip("/")
            out_path = local_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(bucket, key, str(out_path))
            file_count += 1

    if file_count == 0:
        raise FileNotFoundError(f"No dataset files found under {s3_uri}")

    marker.write_text("ok\n", encoding="utf-8")
    print_rank_0(f"Staged {file_count} files to local NVMe")
    return str(local_dir)


def _resolve_distributed_context() -> Tuple[bool, int, int]:
    """
    Resolve distributed world-size/rank even before torch.distributed is initialized.
    """
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))

    if world_size < 1:
        world_size = 1
    if rank < 0:
        rank = 0

    return world_size > 1, world_size, rank


def _normalize_block_sizes(
    block_sizes: Optional[List[int]], max_length: int
) -> List[int]:
    raw = block_sizes or [max_length]
    sizes = sorted({int(v) for v in raw if int(v) > 0})
    if not sizes:
        raise ValueError("At least one positive block size is required")
    return sizes


def _parse_block_size_counts(
    block_size_counts: Optional[Dict[Any, Any]]
) -> Optional[Dict[int, int]]:
    if not block_size_counts:
        return None
    parsed: Dict[int, int] = {}
    for key, value in block_size_counts.items():
        size = int(key)
        count = int(value)
        if size <= 0 or count < 0:
            raise ValueError(f"Invalid block_size_counts entry: {key} -> {value}")
        parsed[size] = count
    return parsed


def _next_target_size(
    ordered_sizes: List[int],
    buffer_len: int,
    remaining_counts: Optional[Dict[int, int]],
) -> Optional[int]:
    if remaining_counts is None:
        # Single-size training is strongly preferred for throughput.
        size = ordered_sizes[0]
        return size if buffer_len >= size else None

    # Try larger sizes first to minimize fragmentation.
    for size in sorted(ordered_sizes, reverse=True):
        if remaining_counts.get(size, 0) > 0 and buffer_len >= size:
            return size
    return None


def _all_requested_blocks_generated(remaining_counts: Optional[Dict[int, int]]) -> bool:
    if remaining_counts is None:
        return False
    return all(v == 0 for v in remaining_counts.values())


def _emit_block(
    out_input_ids: List[List[int]],
    out_attention_mask: List[List[int]],
    out_labels: List[List[int]],
    out_sequence_length: List[int],
    block: List[int],
    target_size: int,
    pad_token_id: int,
    drop_remainder: bool,
) -> None:
    if len(block) < target_size:
        if drop_remainder:
            return
        pad_len = target_size - len(block)
        padded_ids = block + [pad_token_id] * pad_len
        attention_mask = [1] * len(block) + [0] * pad_len
        labels = block + [-100] * pad_len
        out_input_ids.append(padded_ids)
        out_attention_mask.append(attention_mask)
        out_labels.append(labels)
        out_sequence_length.append(target_size)
        return

    ids = block[:target_size]
    out_input_ids.append(ids)
    out_attention_mask.append([1] * target_size)
    out_labels.append(ids.copy())
    out_sequence_length.append(target_size)


def _pack_split_to_fixed_blocks(
    split_dataset: Dataset,
    block_sizes: List[int],
    block_size_counts: Optional[Dict[int, int]],
    eos_token_id: Optional[int],
    pad_token_id: int,
    domain_column: Optional[str],
    concat_across_domains: bool,
    drop_remainder: bool,
) -> Dataset:
    """
    Concatenate tokenized docs and cut fixed-size training blocks.

    If block_size_counts is provided, that defines exactly how many blocks
    of each size to emit (best for explicit 4k/8k/16k budgeting).
    """
    if "input_ids" not in split_dataset.column_names:
        raise ValueError("Packing requires 'input_ids' column")

    ordered_sizes = sorted(block_sizes)
    smallest_size = ordered_sizes[0]

    remaining_counts = None
    if block_size_counts is not None:
        remaining_counts = {
            size: block_size_counts.get(size, 0) for size in ordered_sizes
        }

    buffers: Dict[str, List[int]] = defaultdict(list)
    out_input_ids: List[List[int]] = []
    out_attention_mask: List[List[int]] = []
    out_labels: List[List[int]] = []
    out_sequence_length: List[int] = []

    def flush_buffer(domain_key: str) -> None:
        buf = buffers[domain_key]
        while len(buf) >= smallest_size:
            if _all_requested_blocks_generated(remaining_counts):
                return
            target = _next_target_size(ordered_sizes, len(buf), remaining_counts)
            if target is None:
                return
            block = buf[:target]
            del buf[:target]
            _emit_block(
                out_input_ids,
                out_attention_mask,
                out_labels,
                out_sequence_length,
                block,
                target,
                pad_token_id,
                drop_remainder,
            )
            if remaining_counts is not None:
                remaining_counts[target] -= 1

    for example in split_dataset:
        if _all_requested_blocks_generated(remaining_counts):
            break

        ids = example.get("input_ids")
        if not ids:
            continue

        if (
            domain_column
            and not concat_across_domains
            and domain_column in split_dataset.column_names
        ):
            domain_key = str(example.get(domain_column, "__unknown_domain__"))
        else:
            domain_key = "__all__"

        buffers[domain_key].extend(ids)
        if eos_token_id is not None:
            buffers[domain_key].append(eos_token_id)

        flush_buffer(domain_key)

    # Optionally keep tails by padding up to the smallest configured size.
    if not drop_remainder:
        for domain_key, buf in buffers.items():
            if not buf:
                continue
            if _all_requested_blocks_generated(remaining_counts):
                break
            target = _next_target_size(ordered_sizes, len(buf), remaining_counts)
            if target is None:
                target = smallest_size
            _emit_block(
                out_input_ids,
                out_attention_mask,
                out_labels,
                out_sequence_length,
                buf,
                target,
                pad_token_id,
                drop_remainder,
            )
            buffers[domain_key] = []
            if remaining_counts is not None:
                remaining_counts[target] -= 1

    if remaining_counts is not None:
        missing = {k: v for k, v in remaining_counts.items() if v > 0}
        if missing:
            print_rank_0(
                "WARNING: requested block counts were not fully met "
                f"(insufficient tokens): {missing}"
            )

    if not out_input_ids:
        raise ValueError(
            "Packing produced zero blocks. Lower block size, disable strict counts, "
            "or provide more source tokens."
        )

    return Dataset.from_dict(
        {
            "input_ids": out_input_ids,
            "attention_mask": out_attention_mask,
            "labels": out_labels,
            "sequence_length": out_sequence_length,
        }
    )


def _pack_dataset_dict(
    tokenized_dataset: DatasetDict,
    block_sizes: List[int],
    block_size_counts: Optional[Dict[int, int]],
    tokenizer,
    domain_column: Optional[str],
    concat_across_domains: bool,
    drop_remainder: bool,
) -> DatasetDict:
    packed = {}
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for split_name in tokenized_dataset.keys():
        split = tokenized_dataset[split_name]
        packed[split_name] = _pack_split_to_fixed_blocks(
            split_dataset=split,
            block_sizes=block_sizes,
            block_size_counts=block_size_counts,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            domain_column=domain_column,
            concat_across_domains=concat_across_domains,
            drop_remainder=drop_remainder,
        )

    return DatasetDict(packed)


def _build_causal_lm_collate_fn(pad_token_id: int):
    def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(int(item["input_ids"].shape[0]) for item in batch)
        bsz = len(batch)

        input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = int(item["input_ids"].shape[0])
            input_ids[i, :seq_len] = item["input_ids"]
            attention_mask[i, :seq_len] = item["attention_mask"]
            labels[i, :seq_len] = item["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return _collate


def get_dataloaders(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    tokenizer=None,
    batch_size: int = 8,
    max_length: int = 128,
    num_workers: int = 12,
    tokenized_dataset_path: Optional[str] = None,
    dataset_cache_dir: Optional[str] = None,
    local_nvme_cache_dir: Optional[str] = None,
    require_local_nvme: bool = False,
    pack_into_blocks: bool = False,
    block_sizes: Optional[List[int]] = None,
    block_size_counts: Optional[Dict[Any, Any]] = None,
    domain_column: Optional[str] = None,
    concat_across_domains: bool = False,
    drop_remainder: bool = True,
    streaming: bool = False,
    use_spdl: bool = False,
    shard_manifest_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Load dataset and create dataloaders for training, validation, and testing.

    Args:
        shard_manifest_path: Path to ``consumed_shards.json``. When provided,
            a :class:`ShardTracker` is created to exclude already-processed
            shards and to track newly consumed ones.  The tracker is returned
            in ``dataset_info["shard_tracker"]`` so the training loop can
            persist it alongside checkpoints.
    """
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    if streaming:
        # Streaming mode : to test until the tokenization is complete
        print_rank_0(
            f"Loading dataset in STREAMING mode: {dataset_name} ({dataset_config})"
        )
        ds = load_dataset(dataset_name, dataset_config, streaming=True, split="train")

        train_dataset = StreamingTextDataset(
            ds, tokenizer, max_length, split_name="train"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=min(
                num_workers, 2
            ),  # Streaming doesn't benefit from many workers
            pin_memory=True,
        )

        print_rank_0("  Streaming mode: eval/test will use first 100 train examples")
        eval_loader = train_loader
        test_loader = train_loader

        dataset_info = {
            "train_size": -1,  # Unknown for streaming
            "eval_size": -1,
            "test_size": -1,
            "vocab_size": tokenizer.vocab_size if tokenizer else 0,
            "streaming": True,
        }
        return train_loader, eval_loader, test_loader, dataset_info

    # Use the pre tokenized dataset from S3
    resolved_tokenized_path = tokenized_dataset_path
    if tokenized_dataset_path:
        if _is_s3_uri(tokenized_dataset_path):
            if not local_nvme_cache_dir:
                raise ValueError(
                    "tokenized_dataset_path is S3 but local_nvme_cache_dir is not set. "
                    "Training must stage shards to local NVMe first."
                )
            resolved_tokenized_path = _stage_s3_dataset_to_local_nvme(
                tokenized_dataset_path, local_nvme_cache_dir
            )

        if require_local_nvme and local_nvme_cache_dir:
            nvme_root = Path(local_nvme_cache_dir).expanduser().resolve()
            resolved = Path(resolved_tokenized_path).expanduser().resolve()
            if nvme_root not in resolved.parents and resolved != nvme_root:
                raise RuntimeError(
                    "require_local_nvme=True but tokenized dataset path is outside local NVMe root."
                )

        if use_spdl:
            print_rank_0(
                f"Loading dataset in SPDL mode from: {resolved_tokenized_path}"
            )
            is_distributed, world_size, rank = _resolve_distributed_context()

            # Initialise shard tracker for excluding already-processed files
            shard_tracker: Optional[ShardTracker] = None
            if shard_manifest_path:
                # Disable auto_save to prevent race conditions and desync from checkpoints
                shard_tracker = ShardTracker(shard_manifest_path, auto_save=False)
                print_rank_0(
                    f"  ShardTracker loaded: {shard_tracker.num_processed} "
                    f"shards already processed (manifest: {shard_manifest_path})"
                )

            # Helper to locate split subdirectories if they exist
            def get_shard_dir(split_name: str) -> str:
                split_dir = os.path.join(resolved_tokenized_path, split_name)
                return (
                    split_dir if os.path.exists(split_dir) else resolved_tokenized_path
                )

            train_dataset = SPDLIterableDataset(
                shard_dir=get_shard_dir("train"),
                seq_len=max_length,
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
                num_threads=max(1, num_workers),
                shard_tracker=shard_tracker,
            )

            eval_dataset = SPDLIterableDataset(
                shard_dir=get_shard_dir("validation"),
                seq_len=max_length,
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
                num_threads=1,  # Keep threads light for eval
            )

            test_dataset = SPDLIterableDataset(
                shard_dir=get_shard_dir("test"),
                seq_len=max_length,
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
                num_threads=1,
            )

            # batch_size=None and num_workers=0 because SPDL internally batches and handles threads
            train_loader = DataLoader(
                train_dataset, batch_size=None, num_workers=0, pin_memory=True
            )
            eval_loader = DataLoader(
                eval_dataset, batch_size=None, num_workers=0, pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=None, num_workers=0, pin_memory=True
            )

            dataset_info = {
                "train_size": -1,
                "eval_size": -1,
                "test_size": -1,
                "vocab_size": tokenizer.vocab_size if tokenizer else 0,
                "resolved_tokenized_dataset_path": resolved_tokenized_path,
                "spdl_mode": True,
                "shard_tracker": shard_tracker,
            }
            return train_loader, eval_loader, test_loader, dataset_info
        else:
            print_rank_0(
                f"Loading pre-tokenized dataset from disk: {resolved_tokenized_path}"
            )
            tokenized_dataset = load_from_disk(resolved_tokenized_path)
    else:
        if require_local_nvme:
            raise RuntimeError(
                "require_local_nvme=True requires tokenized_dataset_path (local or s3://). "
                "Online tokenization path is disabled."
            )

        print_rank_0(f"Loading dataset: {dataset_name} ({dataset_config})")
        dataset = load_dataset(
            dataset_name, dataset_config, cache_dir=dataset_cache_dir
        )

        def filter_empty(example):
            return len(example["text"].strip()) > 0

        dataset = dataset.filter(filter_empty)

        if pack_into_blocks:
            print_rank_0("Tokenizing dataset for block packing...")
            tokenized_dataset = dataset.map(
                lambda examples: tokenize_function(
                    examples,
                    tokenizer,
                    max_length=None,
                    pad_to_max_length=False,
                    truncation=False,
                ),
                batched=True,
                remove_columns=dataset["train"].column_names,
            )

            normalized_sizes = _normalize_block_sizes(block_sizes, max_length)
            parsed_counts = _parse_block_size_counts(block_size_counts)
            if parsed_counts:
                unknown_sizes = sorted(set(parsed_counts) - set(normalized_sizes))
                if unknown_sizes:
                    raise ValueError(
                        "block_size_counts has sizes not present in block_sizes: "
                        f"{unknown_sizes}"
                    )

            print_rank_0(
                "Packing token stream into fixed training blocks: "
                f"sizes={normalized_sizes}, counts={parsed_counts}"
            )
            if domain_column and not concat_across_domains:
                print_rank_0(
                    f"Domain-aware packing enabled. Will not mix domains using column '{domain_column}'."
                )

            tokenized_dataset = _pack_dataset_dict(
                tokenized_dataset=tokenized_dataset,
                block_sizes=normalized_sizes,
                block_size_counts=parsed_counts,
                tokenizer=tokenizer,
                domain_column=domain_column,
                concat_across_domains=concat_across_domains,
                drop_remainder=drop_remainder,
            )
        else:
            print_rank_0(
                "Tokenizing dataset with fixed truncation/padding (legacy path)..."
            )
            tokenized_dataset = dataset.map(
                lambda examples: tokenize_function(
                    examples,
                    tokenizer,
                    max_length=max_length,
                    pad_to_max_length=True,
                    truncation=True,
                ),
                batched=True,
                remove_columns=dataset["train"].column_names,
            )

    if not isinstance(tokenized_dataset, DatasetDict):
        raise ValueError(
            "Tokenized dataset must be a DatasetDict with train/validation/test splits."
        )

    for split_name in tokenized_dataset.keys():
        split_columns = [
            col
            for col in ("input_ids", "attention_mask", "labels", "sequence_length")
            if col in tokenized_dataset[split_name].column_names
        ]
        tokenized_dataset[split_name].set_format(type="torch", columns=split_columns)

    effective_workers = max(int(num_workers), 0)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": effective_workers,
    }
    if effective_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["persistent_workers"] = True

    is_distributed, world_size, rank = _resolve_distributed_context()
    print_rank_0(
        f"Distributed context: is_distributed={is_distributed}, world_size={world_size}, rank={rank}"
    )

    train_sampler = None
    eval_sampler = None
    test_sampler = None

    if is_distributed:
        train_sampler = DistributedSampler(
            tokenized_dataset["train"],
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        eval_sampler = DistributedSampler(
            tokenized_dataset["validation"],
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        test_sampler = DistributedSampler(
            tokenized_dataset["test"],
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    if world_size > 1 and train_sampler is None:
        raise RuntimeError(
            "Distributed training detected but DistributedSampler is not configured. "
            "Refusing to run due to potential duplicated data across GPUs."
        )

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate_fn = _build_causal_lm_collate_fn(pad_token_id)

    train_loader = DataLoader(
        tokenized_dataset["train"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=is_distributed,
        collate_fn=collate_fn,
        pin_memory=True,
        **loader_kwargs,
    )

    eval_loader = DataLoader(
        tokenized_dataset["validation"],
        shuffle=False,
        sampler=eval_sampler,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=True,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        tokenized_dataset["test"],
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=True,
        **loader_kwargs,
    )

    persistent_workers = bool(getattr(train_loader, "persistent_workers", False))
    prefetch_factor = getattr(train_loader, "prefetch_factor", None)

    print_rank_0(
        "DataLoader worker config: "
        f"num_workers={train_loader.num_workers}, "
        f"persistent_workers={persistent_workers}, "
        f"pin_memory={train_loader.pin_memory}, "
        f"prefetch_factor={prefetch_factor}"
    )

    dataset_info = {
        "train_size": len(tokenized_dataset["train"]),
        "eval_size": len(tokenized_dataset["validation"]),
        "test_size": len(tokenized_dataset["test"]),
        "vocab_size": tokenizer.vocab_size,
        "train_sampler": train_sampler,
        "eval_sampler": eval_sampler,
        "test_sampler": test_sampler,
        "resolved_tokenized_dataset_path": resolved_tokenized_path,
        "worker_config": {
            "num_workers": train_loader.num_workers,
            "persistent_workers": persistent_workers,
            "pin_memory": train_loader.pin_memory,
            "prefetch_factor": prefetch_factor,
        },
    }

    return train_loader, eval_loader, test_loader, dataset_info
