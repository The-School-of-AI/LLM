"""
Data loading utilities for DeepSpeed training.

This module provides functions for loading tokenizers and creating dataloaders
for training language models.
"""

from typing import Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def get_tokenizer(model_name: str):
    """
    Load and configure tokenizer for the specified model.

    Args:
        model_name: Name of the pretrained model from HuggingFace

    Returns:
        Configured tokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    num_workers: int = 0,
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
    print(f"Loading dataset: {dataset_name} ({dataset_config})")
    dataset = load_dataset(dataset_name, dataset_config)

    # Filter out empty examples
    def filter_empty(example):
        return len(example["text"].strip()) > 0

    dataset = dataset.filter(filter_empty)

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Set format for PyTorch
    tokenized_dataset.set_format(type="torch")

    # Create dataloaders
    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        tokenized_dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Dataset info
    dataset_info = {
        "train_size": len(tokenized_dataset["train"]),
        "eval_size": len(tokenized_dataset["validation"]),
        "test_size": len(tokenized_dataset["test"]),
        "vocab_size": tokenizer.vocab_size,
    }

    return train_loader, eval_loader, test_loader, dataset_info
