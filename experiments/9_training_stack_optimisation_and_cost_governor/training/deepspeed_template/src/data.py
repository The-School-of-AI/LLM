# src/data.py

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def get_tokenizer(model_name: str):
    """
    Load tokenizer strictly from local cache.
    Will FAIL immediately if not downloaded already.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
        use_fast=True,
    )

    # GPT-style models need a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_dataloaders(
    dataset_name,
    dataset_config,
    tokenizer,
    batch_size,
    max_length,
):
    """
    Load dataset (cached locally by HF datasets),
    tokenize, and return PyTorch DataLoaders.
    """

    dataset = load_dataset(
        dataset_name,
        dataset_config,
    )

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # 🔴 CRITICAL FIX
    # Convert HF dataset outputs from Python lists → torch.Tensor
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
    )

    eval_loader = DataLoader(
        tokenized["validation"],
        batch_size=batch_size,
    )

    test_loader = DataLoader(
        tokenized["test"],
        batch_size=batch_size,
    )

    return train_loader, eval_loader, test_loader, tokenizer