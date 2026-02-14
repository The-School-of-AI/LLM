"""
Data loading utilities for DeepSpeed training.

This module provides functions for loading tokenizers and creating dataloaders
for training language models.
"""

from typing import Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

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
    
    # Default to the TSAI 131K tokenizer in src/tokenizer/
    if tokenizer_path is None:
        # Get the directory of this file (src/data.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_path = os.path.join(current_dir, "tokenizer")
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"TSAI 131K tokenizer not found at: {tokenizer_path}\n"
            "Expected directory structure: src/tokenizer/ with tokenizer.json, "
            "tokenizer_config.json, and special_tokens_map.json"
        )
    
    print_rank_0(f"  Loading TSAI 131K tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    print_rank_0(f"  Tokenizer loaded:")
    print_rank_0(f"    - Vocab size: {tokenizer.vocab_size:,}")
    print_rank_0(f"    - Total tokens (with special): {len(tokenizer):,}")
    print_rank_0(f"    - BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print_rank_0(f"    - EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print_rank_0(f"    - PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    return tokenizer


def tokenize_function(examples, tokenizer, max_length=128):
    """
    Tokenize text examples for language modeling.

    Args:
        examples: Dictionary with 'text' key containing text examples
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        Dictionary with tokenized inputs
    """
    # Tokenize the texts
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )

    # For causal language modeling, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def get_dataloaders(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    tokenizer=None,
    batch_size: int = 8,
    max_length: int = 128,
    num_workers: int = 12,  # 12 workers per GPU = 96 total workers on p4d.24xlarge (96 vCPUs, 8 GPUs)
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Load dataset and create dataloaders for training, validation, and testing.

    Args:
        dataset_name: Name of the dataset from HuggingFace datasets
        dataset_config: Configuration name for the dataset
        tokenizer: Tokenizer instance (required)
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, eval_loader, test_loader, dataset_info)
    """
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    # Load dataset
    print_rank_0(f"Loading dataset: {dataset_name} ({dataset_config})")
    dataset = load_dataset(dataset_name, dataset_config)

    # Filter out empty examples
    def filter_empty(example):
        return len(example["text"].strip()) > 0

    dataset = dataset.filter(filter_empty)

    # Tokenize dataset
    print_rank_0("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Set format for PyTorch
    tokenized_dataset.set_format(type="torch")

    # Create dataloaders with AGGRESSIVE optimizations for p4d.24xlarge (96 vCPUs, 8 GPUs)
    # With DeepSpeed, each GPU process creates its own dataloader
    # num_workers=12 per GPU × 8 GPUs = 96 total worker processes (uses ALL vCPUs)
    # prefetch_factor=4 keeps 4 batches ready per worker (384 batches total pre-loaded!)
    # This ensures GPU NEVER waits for data
    # persistent_workers=True keeps workers alive between epochs (eliminates startup overhead)
    effective_workers = num_workers if num_workers > 0 else 12
    
    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=True,
        prefetch_factor=4,  # Increased from 2 to 4 for maximum throughput
        persistent_workers=True if effective_workers > 0 else False,
        # NOTE: Don't use multiprocessing_context='fork' with DeepSpeed - breaks NCCL!
    )

    eval_loader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True if effective_workers > 0 else False,
    )

    test_loader = DataLoader(
        tokenized_dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True if effective_workers > 0 else False,
    )

    # Dataset info
    dataset_info = {
        "train_size": len(tokenized_dataset["train"]),
        "eval_size": len(tokenized_dataset["validation"]),
        "test_size": len(tokenized_dataset["test"]),
        "vocab_size": tokenizer.vocab_size,
    }

    return train_loader, eval_loader, test_loader, dataset_info
