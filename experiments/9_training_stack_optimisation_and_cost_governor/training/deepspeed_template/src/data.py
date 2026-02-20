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

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import logging
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
                chunk = buffer[:self.max_length]
                buffer = buffer[self.max_length:]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                attention_mask = torch.ones(self.max_length, dtype=torch.long)
                labels = input_ids.clone()
                yield {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}



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
    print_rank_0(f"    - BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print_rank_0(f"    - EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print_rank_0(f"    - PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    return tokenizer

# =================================================================
# SPDL Helper Functions
# =================================================================
def read_idx(idx_path):
    """Read the .idx file and return an array of offsets."""
    import numpy as np
    with open(idx_path, "rb") as f:
        _ = f.read(8)  # Skip header
        offsets = np.frombuffer(f.read(), dtype=np.uint64)
    return offsets

def _should_skip_region(bin_path, i, start, end, itemsize, seq_len):
    """Helper to check if a region in the bin/idx file should be skipped."""
    num_bytes = end - start
    if num_bytes <= 0:
        return True
    num_tokens = num_bytes // itemsize
    if num_tokens == 0:
        return True
    num_full = num_tokens // seq_len
    if num_full == 0:
        return True
    return False

def load_tokens_from_bin_idx(bin_path, idx_path, dtype, seq_len):
    """Yield token sequences from a .bin file using .idx offsets."""
    itemsize = dtype.itemsize
    offsets = read_idx(idx_path)
    
    with open(bin_path, "rb") as f:
        for i in range(len(offsets) - 1):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            if _should_skip_region(bin_path, i, start, end, itemsize, seq_len):
                continue
            
            num_bytes = end - start
            num_tokens = num_bytes // itemsize
            num_full = num_tokens // seq_len
            read_tokens = num_full * seq_len
            
            f.seek(start)
            tokens = np.frombuffer(f.read(read_tokens * itemsize), dtype=dtype)
            
            if tokens.size != read_tokens:
                continue
                
            for j in range(0, len(tokens), seq_len):
                yield torch.from_numpy(tokens[j:j+seq_len].astype(np.int64))

def bin_idx_source(shard_dir, seq_len, dtype, rank=0, world_size=1):
    """
    Generator yielding token sequences from .bin/.idx shards in directory.
    Handles Distributed Sharding: splits files across ranks.
    """
    files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".bin"))
    
    # Shard files across ranks
    if world_size > 1:
        files = files[rank::world_size]
        print_rank_0(f"SPDL: Rank {rank} processing {len(files)} shards (shard stride {world_size})")
    
    for bin_file in files:
        bin_path = os.path.join(shard_dir, bin_file)
        idx_path = bin_path.replace(".bin", ".idx")
        yield from load_tokens_from_bin_idx(bin_path, idx_path, dtype, seq_len)

def build_pipeline(shard_dir, seq_len, dtype, batch_size, rank=0, world_size=1):
    """Build SPDL pipeline yielding batches of {input_ids, attention_mask, labels}."""
    from spdl.pipeline import PipelineBuilder
    
    # Define source generator
    source = bin_idx_source(shard_dir, seq_len, dtype, rank, world_size)
    
    # Transform function to match training batch format
    # SPDL aggregation yields a list/tensor of token sequences [B, L]
    # We need to wrap it into the dict expected by train_epoch
    
    def transform_to_batch(batch_tokens):
        # batch_tokens is a list of tensors [L], or valid tensor [B, L] depending on SPDL version
        # Assuming aggregation yields list of tensors
        if isinstance(batch_tokens, list):
            input_ids = torch.stack(batch_tokens)
        else:
            input_ids = batch_tokens
            
        # Create masks/labels
        B, L = input_ids.shape
        attention_mask = torch.ones((B, L), dtype=torch.long)
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids.long(), # Ensure long for embeddings
            "attention_mask": attention_mask,
            "labels": labels
        }

    # Pipeline
    # Note: SPDL aggregation happens BEFORE transform usually, or we use map. 
    # Let's use Source -> Aggregate -> Map
    
    pipeline = (
        PipelineBuilder()
        .add_source(source)
        .aggregate(batch_size)
        .transform(transform_to_batch)
        .add_sink(4) # buffer size
        .build(num_threads=2) # hardcoded small threads per GPU to avoid contention
    )
    return pipeline


def get_dataloaders(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    tokenizer=None,
    batch_size: int = 8,
    max_length: int = 128,
    num_workers: int = 12,
    tokenized_dataset_path: Optional[str] = None,
    streaming: bool = False,
    # --- New Arguments ---
    use_dataloader: bool = False,
    shard_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Load dataset and create dataloaders for training, validation, and testing.

    Supports three modes:
    1. SPDL Mode (New): Reads .bin/.idx shards from shard_dir (use_dataloader=True)
    2. Offline (preferred): Pass tokenized_dataset_path
    3. Online (fallback): HuggingFace load_dataset

    Args:
        dataset_name: HuggingFace dataset name (online mode)
        dataset_config: HuggingFace dataset config (online mode)
        tokenizer: Tokenizer instance (required for online/streaming mode)
        batch_size: Micro-batch size per GPU
        max_length: Maximum sequence length
        num_workers: DataLoader workers per GPU
        tokenized_dataset_path: Path to pre-tokenized dataset on disk (offline mode)
        streaming: If True, use HF streaming mode
        use_dataloader: If True, use SPDL pipeline (reads .bin/.idx shards)
        shard_dir: Path to directory containing .bin/.idx files (required if use_dataloader=True)

    Returns:
        Tuple of (train_loader, eval_loader, test_loader, dataset_info)
    """
    if tokenizer is None and tokenized_dataset_path is None and not use_dataloader:
        raise ValueError("tokenizer must be provided for online tokenisation")

    # =================================================================
    # SPDL Pipeline Mode (New)
    # =================================================================
    if use_dataloader:
        if not shard_dir:
            raise ValueError("shard_dir must be provided when use_dataloader=True")

        print_rank_0(f"Loading dataset via SPDL pipeline from: {shard_dir}")
        try:
            import spdl.pipeline
        except ImportError:
            raise ImportError("spdl not installed. Please install spdl to use use_dataloader=True")

        # Config constants for SPDL
        # DTYPE is usually uint16 or uint32 depending on vocab size. 
        # We'll default to uint32 to be safe, or user can change code if needed.
        # Ideally this should be in config, but for now we hardcode or infer.
        dtype = np.uint32 
        
        # Distributed info
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
            
        print_rank_0(f"  SPDL: Rank {rank}/{world_size}, Batch Size {batch_size}, Seq Len {max_length}")

        # Build pipelines
        # Train
        train_loader = build_pipeline(
            shard_dir, 
            seq_len=max_length, 
            dtype=dtype, 
            batch_size=batch_size, 
            rank=rank, 
            world_size=world_size
        )
        
        eval_loader = train_loader
        test_loader = train_loader

        dataset_info = {
            "train_size": -1, # Unknown/Huge
            "eval_size": -1,
            "test_size": -1,
            "vocab_size": tokenizer.vocab_size if tokenizer else 0,
            "streaming": True, # Behaves like streaming
        }
        return train_loader, eval_loader, test_loader, dataset_info

    # =================================================================
    # Streaming mode — for very large datasets (FineWeb, etc.)
    # =================================================================
    if streaming:
        print_rank_0(f"Loading dataset in STREAMING mode: {dataset_name} ({dataset_config})")
        ds = load_dataset(dataset_name, dataset_config, streaming=True, split="train")

        train_dataset = StreamingTextDataset(ds, tokenizer, max_length, split_name="train")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=min(num_workers, 2),  # Streaming doesn't benefit from many workers
            pin_memory=True,
        )

        # For streaming, eval/test use a tiny in-memory sample
        # (FineWeb doesn't have validation/test splits)
        print_rank_0("  Streaming mode: eval/test will use first 100 train examples")
        eval_loader = train_loader  # Placeholder — override in main if needed
        test_loader = train_loader

        dataset_info = {
            "train_size": -1,  # Unknown for streaming
            "eval_size": -1,
            "test_size": -1,
            "vocab_size": tokenizer.vocab_size if tokenizer else 0,
            "streaming": True,
        }
        return train_loader, eval_loader, test_loader, dataset_info

    # -----------------------------------------------------------------
    # Load dataset (offline or online)
    # -----------------------------------------------------------------
    if tokenized_dataset_path is not None:
        print_rank_0(f"Loading pre-tokenized dataset from disk: {tokenized_dataset_path}")
        dataset = load_from_disk(tokenized_dataset_path)
    else:
        print_rank_0(f"Loading dataset: {dataset_name} ({dataset_config})")
        dataset = load_dataset(dataset_name, dataset_config)

    # -----------------------------------------------------------------
    # Tokenize & pack (only needed when running online)
    # -----------------------------------------------------------------
    eos_token = (tokenizer.eos_token if tokenizer and tokenizer.eos_token else "")

    def tokenize_and_concat(split_dataset):
        """Concatenate all texts, tokenize, and chunk into fixed-length sequences."""
        all_text = eos_token.join(
            text for text in split_dataset["text"] if text.strip()
        )
        all_ids = tokenizer(all_text, return_attention_mask=False)["input_ids"]

        total_tokens = len(all_ids)
        n_chunks = total_tokens // max_length
        print_rank_0(
            f"    Total tokens: {total_tokens:,} -> {n_chunks:,} chunks of {max_length:,}"
        )

        chunks = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(n_chunks):
            start = i * max_length
            end = start + max_length
            ids = all_ids[start:end]
            chunks["input_ids"].append(ids)
            chunks["attention_mask"].append([1] * max_length)
            chunks["labels"].append(ids.copy())
        return chunks

    def make_tensor_dataset(split_dataset, split_name):
        print_rank_0(f"  Processing {split_name} split...")
        chunks = tokenize_and_concat(split_dataset)
        n = len(chunks["input_ids"])
        if n == 0:
            print_rank_0(f"    WARNING: {split_name} has 0 packed sequences!")
            return TensorDataset(
                torch.zeros(1, max_length, dtype=torch.long),
                torch.ones(1, max_length, dtype=torch.long),
                torch.zeros(1, max_length, dtype=torch.long),
            )
        input_ids = torch.tensor(chunks["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(chunks["attention_mask"], dtype=torch.long)
        labels = torch.tensor(chunks["labels"], dtype=torch.long)
        return TensorDataset(input_ids, attention_mask, labels)

    print_rank_0("Tokenizing and packing dataset...")
    train_dataset = make_tensor_dataset(dataset["train"], "train")
    eval_dataset = make_tensor_dataset(dataset["validation"], "validation")
    test_dataset = make_tensor_dataset(dataset["test"], "test")

    # -----------------------------------------------------------------
    # Distributed samplers (required for multi-GPU data sharding)
    # -----------------------------------------------------------------
    distributed = dist.is_available() and dist.is_initialized()

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None

    # -----------------------------------------------------------------
    # DataLoader construction
    # -----------------------------------------------------------------
    effective_workers = num_workers if num_workers > 0 else 0
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=effective_workers,
        pin_memory=True,
    )
    if effective_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["persistent_workers"] = True

    def collate_fn(batch):
        input_ids = torch.stack([b[0] for b in batch])
        attention_mask = torch.stack([b[1] for b in batch])
        labels = torch.stack([b[2] for b in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_loader = DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    eval_loader = DataLoader(
        eval_dataset,
        shuffle=False,
        sampler=eval_sampler,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    dataset_info = {
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "test_size": len(test_dataset),
        "vocab_size": tokenizer.vocab_size if tokenizer else 0,
    }

    return train_loader, eval_loader, test_loader, dataset_info
