"""
Simple Dataset Loader for Training

Uses TinyStories or similar small datasets for quick iteration.
Supports:
- Streaming from HuggingFace datasets
- Simple tokenization via HF tokenizer
- Stateful data sampling (resume training)
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, Iterator, Dict, Any
import random


class TextDataset(Dataset):
    """Simple text dataset that tokenizes on-the-fly."""
    
    def __init__(
        self,
        texts: list,
        tokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for large corpora."""
    
    def __init__(
        self,
        hf_dataset_name: str = "roneneldan/TinyStories",
        split: str = "train",
        tokenizer = None,
        max_length: int = 512,
        seed: int = 42,
    ):
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self._dataset = None
    
    def _load_dataset(self):
        """Lazy load the HF dataset."""
        if self._dataset is None:
            try:
                from datasets import load_dataset
                self._dataset = load_dataset(
                    self.hf_dataset_name,
                    split=self.split,
                    streaming=True,
                )
            except ImportError:
                raise ImportError("Please install datasets: pip install datasets")
        return self._dataset
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        dataset = self._load_dataset()
        
        for example in dataset:
            # Get text field (varies by dataset)
            if "text" in example:
                text = example["text"]
            elif "story" in example:
                text = example["story"]
            elif "content" in example:
                text = example["content"]
            else:
                text = str(list(example.values())[0])
            
            if not text or len(text.strip()) < 10:
                continue
            
            # Tokenize
            if self.tokenizer is not None:
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].squeeze(0)
            else:
                # Fallback: simple character-level encoding
                input_ids = torch.tensor([ord(c) % 49152 for c in text[:self.max_length]])
                if len(input_ids) < self.max_length:
                    input_ids = torch.cat([
                        input_ids,
                        torch.zeros(self.max_length - len(input_ids), dtype=torch.long)
                    ])
            
            yield {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
            }


class DummyDataset(Dataset):
    """
    Dummy dataset for quick testing without downloading data.
    Generates random token sequences.
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        max_length: int = 512,
        vocab_size: int = 49152,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.seed = seed
        
        # Pre-generate for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Seeded random for reproducibility
        torch.manual_seed(self.seed + idx)
        input_ids = torch.randint(0, self.vocab_size, (self.max_length,))
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def get_dataloader(
    dataset_name: str = "dummy",
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 0,
    tokenizer = None,
    **kwargs,
) -> DataLoader:
    """
    Factory function to create a dataloader.
    
    Args:
        dataset_name: "dummy", "tinystories", or a HuggingFace dataset name
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: DataLoader workers
        tokenizer: HuggingFace tokenizer (optional)
    
    Returns:
        DataLoader instance
    """
    if dataset_name == "dummy":
        dataset = DummyDataset(
            max_length=max_length,
            vocab_size=kwargs.get("vocab_size", 49152),
            num_samples=kwargs.get("num_samples", 10000),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
    
    elif dataset_name == "tinystories":
        dataset = StreamingTextDataset(
            hf_dataset_name="roneneldan/TinyStories",
            tokenizer=tokenizer,
            max_length=max_length,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    
    else:
        # Assume HuggingFace dataset name
        dataset = StreamingTextDataset(
            hf_dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )


if __name__ == "__main__":
    # Test dummy dataset
    print("Testing DummyDataset...")
    dataset = DummyDataset(num_samples=100, max_length=64)
    dataloader = DataLoader(dataset, batch_size=4)
    
    batch = next(iter(dataloader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    
    print("\nDataset ready for training!")
