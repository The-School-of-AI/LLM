"""
Quick Wikitext-2 training using GPT-2 tokenizer.

This script is a lightweight alternative to training/train.py so you can
smoke-test the model while a custom tokenizer is being built.

Supports two configuration modes:
1. Preset mode: --preset 1b-base (uses Python preset configs)
2. YAML mode: --config configs/1b_base.yaml (uses YAML config files)

CLI arguments always override config file values.
"""

import argparse
import random
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import yaml

# Add repo root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig, get_preset_config, PRESET_CONFIGS
from models.llm import LLM
from training.train import TrainingConfig, Trainer

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "Missing dependencies. Install with: pip install datasets transformers"
    ) from exc


def load_config_from_yaml(config_path: str) -> Tuple[ModelConfig, Dict[str, Any]]:
    """
    Load model and training config from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Tuple of (ModelConfig, training_config_dict)
    """
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Extract training config (if present)
    training_data = config_data.pop('training', {})
    
    # Load model config
    model_config = ModelConfig.from_dict(config_data)
    
    return model_config, training_data


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
        description="Train 1B LLM on WikiText-2 with GPT-2 tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using YAML config file (recommended)
  python train_wikitext2_gpt2.py --config ../configs/1b_deepseek_gsa.yaml
  python train_wikitext2_gpt2.py --config ../configs/1b_base.yaml --batch-size 4
  
  # Using preset (legacy mode)
  python train_wikitext2_gpt2.py --preset 1b-base --device cuda
  
  # YAML config with CLI overrides
  python train_wikitext2_gpt2.py --config ../configs/1b_gsa.yaml --seq-length 512 --max-steps 500

Note: CLI arguments always override config file values.
        """
    )

    # Configuration source
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/1b_base.yaml). Takes precedence over --preset."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="1b-base",
        choices=list(PRESET_CONFIGS.keys()),
        help="Model preset (used if --config not provided)"
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
        default=None,
        help="Sequence length (overrides config)"
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

    # Training (can override config file)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")

    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use: auto (best available), cuda, mps (Apple Silicon), or cpu"
    )

    # Experiment
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Logging / DataLoader
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    set_seed(args.seed)

    # Load configuration
    if args.config:
        # YAML config mode
        print(f"Loading configuration from: {args.config}")
        model_config, training_dict = load_config_from_yaml(args.config)
        
        # Build training config from YAML
        training_config = TrainingConfig(
            max_steps=training_dict.get('max_steps', 200),
            batch_size=training_dict.get('batch_size', 2),
            gradient_accumulation_steps=training_dict.get('gradient_accumulation_steps', 1),
            seq_length=training_dict.get('seq_length', 256),
            learning_rate=training_dict.get('learning_rate', 3e-4),
            warmup_steps=training_dict.get('warmup_steps', 20),
            device=training_dict.get('device', 'auto'),
            experiment_name=training_dict.get('experiment_name', 'wikitext2_gpt2'),
            checkpoint_dir=training_dict.get('checkpoint_dir', './checkpoints/wikitext2_gpt2'),
            seed=args.seed,
            log_interval=training_dict.get('log_interval', 10),
            save_interval=training_dict.get('save_interval', 200),
            use_amp=training_dict.get('use_amp', True),
        )
    else:
        # Preset mode (legacy)
        print(f"Using preset: {args.preset}")
        model_config = get_preset_config(args.preset)
        training_config = TrainingConfig(
            max_steps=200,
            batch_size=2,
            gradient_accumulation_steps=1,
            seq_length=256,
            learning_rate=3e-4,
            warmup_steps=20,
            device='auto',
            experiment_name='wikitext2_gpt2',
            checkpoint_dir='./checkpoints/wikitext2_gpt2',
            seed=args.seed,
            log_interval=10,
            save_interval=200,
        )

    # CLI overrides (only if explicitly provided)
    if args.max_steps is not None:
        training_config.max_steps = args.max_steps
    if args.batch_size is not None:
        training_config.batch_size = args.batch_size
    if args.gradient_accumulation is not None:
        training_config.gradient_accumulation_steps = args.gradient_accumulation
    if args.seq_length is not None:
        training_config.seq_length = args.seq_length
    if args.learning_rate is not None:
        training_config.learning_rate = args.learning_rate
    if args.warmup_steps is not None:
        training_config.warmup_steps = args.warmup_steps
    if args.device is not None:
        training_config.device = args.device
    if args.experiment_name is not None:
        training_config.experiment_name = args.experiment_name
    if args.checkpoint_dir is not None:
        training_config.checkpoint_dir = args.checkpoint_dir
    if args.log_interval is not None:
        training_config.log_interval = args.log_interval
    if args.save_interval is not None:
        training_config.save_interval = args.save_interval
    if args.no_amp:
        training_config.use_amp = False

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_ids = build_token_ids(
        split=args.dataset_split,
        tokenizer=tokenizer,
        add_eos=True,
        max_tokens=args.max_tokens
    )

    stride = training_config.seq_length if args.stride is None else args.stride
    dataset = TokenBlockDataset(token_ids, seq_length=training_config.seq_length, stride=stride)
    if len(dataset) == 0:
        raise ValueError(
            "Not enough tokens for the chosen seq-length. "
            "Reduce --seq-length or increase --max-tokens."
        )

    # Determine if we should pin memory (only for CUDA)
    pin_memory = training_config.device == "cuda" or (training_config.device == "auto" and torch.cuda.is_available())

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    # Update vocab size from tokenizer
    model_config.vocab_size = len(tokenizer)

    model = LLM(model_config)
    model.gradient_checkpointing_enable()
    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        training_config=training_config,
        model_config=model_config
    )
    trainer.train()


if __name__ == "__main__":
    main()