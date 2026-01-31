"""
Quick Wikitext-2 training using GPT-2 tokenizer.

This script is a lightweight alternative to training/train.py so you can
smoke-test the model while a custom tokenizer is being built.
"""

import argparse
import random
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

# Add repo root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import get_preset_config, PRESET_CONFIGS
from models.llm import LLM
from training.train import TrainingConfig, Trainer

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "Missing dependencies. Install with: pip install datasets transformers"
    ) from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TokenBlockDataset(Dataset):
    """Simple fixed-length token block dataset."""

    def __init__(self, token_ids: List[int], seq_length: int, stride: int):
        self.token_ids = token_ids
        self.seq_length = seq_length
        self.stride = stride
        if len(token_ids) < seq_length:
            self.num_blocks = 0
        else:
            self.num_blocks = (len(token_ids) - seq_length) // stride + 1

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.seq_length
        block = self.token_ids[start:end]
        input_ids = torch.tensor(block, dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def build_token_ids(
    split: str,
    tokenizer,
    add_eos: bool = True,
    max_tokens: Optional[int] = None
) -> List[int]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    token_ids: List[int] = []
    eos_id = tokenizer.eos_token_id

    for text in dataset["text"]:
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        token_ids.extend(ids)
        if add_eos and eos_id is not None:
            token_ids.append(eos_id)
        if max_tokens is not None and len(token_ids) >= max_tokens:
            token_ids = token_ids[:max_tokens]
            break

    return token_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 1B LLM on WikiText-2 with GPT-2 tokenizer"
    )

    # Model
    parser.add_argument(
        "--preset",
        type=str,
        default="1b-base",
        choices=list(PRESET_CONFIGS.keys()),
        help="Model preset"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Hugging Face tokenizer name (default: gpt2)"
    )

    # Dataset
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="WikiText-2 split"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=256,
        help="Sequence length"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride between blocks (default: seq-length)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Cap total tokens for a tiny smoke test"
    )

    # Training
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--no-amp", action="store_true")

    # Experiment
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="wikitext2_gpt2"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/wikitext2_gpt2"
    )
    parser.add_argument("--seed", type=int, default=42)

    # Logging / DataLoader
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_ids = build_token_ids(
        split=args.dataset_split,
        tokenizer=tokenizer,
        add_eos=True,
        max_tokens=args.max_tokens
    )

    stride = args.seq_length if args.stride is None else args.stride
    dataset = TokenBlockDataset(token_ids, seq_length=args.seq_length, stride=stride)
    if len(dataset) == 0:
        raise ValueError(
            "Not enough tokens for the chosen seq-length. "
            "Reduce --seq-length or increase --max-tokens."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Model config
    model_config = get_preset_config(args.preset)
    model_config.vocab_size = len(tokenizer)

    # Training config
    training_config = TrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        seq_length=args.seq_length,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        experiment_name=args.experiment_name,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        use_amp=not args.no_amp
    )

    model = LLM(model_config)
    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        training_config=training_config,
        model_config=model_config
    )
    trainer.train()


if __name__ == "__main__":
    main()
