"""
Data loading utilities for DeepSpeed training.

This module provides functions for loading tokenizers and creating dataloaders
for training language models.

Supports two modes:
1. Offline: load_from_disk for pre-tokenized datasets (preferred for production)
2. Online: download + tokenize on-the-fly (fallback for dev/testing)

Uses the standard LLM pre-training approach: concatenate all text, tokenize,
then chunk into fixed-length sequences. Every token is a real token — no
padding waste.
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from .utils import print_rank_0


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


def get_dataloaders(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    tokenizer=None,
    batch_size: int = 8,
    max_length: int = 128,
    num_workers: int = 12,
    tokenized_dataset_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Load dataset and create dataloaders for training, validation, and testing.

    Supports two modes:
    - Offline (preferred): Pass ``tokenized_dataset_path`` pointing to a
      datasets.save_to_disk() directory.  Skips tokenisation entirely.
    - Online (fallback): Downloads from HuggingFace Hub and tokenises on the fly.

    Uses the standard LLM pre-training approach:
    1. Concatenate all text with EOS separators
    2. Tokenize the entire concatenation at once
    3. Chunk into fixed-length sequences of max_length

    Every token in every batch is a real token.  No padding.

    Args:
        dataset_name: HuggingFace dataset name (online mode)
        dataset_config: HuggingFace dataset config (online mode)
        tokenizer: Tokenizer instance (required for online mode)
        batch_size: Micro-batch size per GPU
        max_length: Maximum sequence length
        num_workers: DataLoader workers per GPU
        tokenized_dataset_path: Path to pre-tokenized dataset on disk (offline mode)

    Returns:
        Tuple of (train_loader, eval_loader, test_loader, dataset_info)
    """
    if tokenizer is None and tokenized_dataset_path is None:
        raise ValueError("tokenizer must be provided for online tokenisation")

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
