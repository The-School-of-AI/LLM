"""
Simple Dataset Loader for Training

Uses TinyStories or similar small datasets for quick iteration.
Supports:
- Streaming from HuggingFace datasets
- Simple tokenization via HF tokenizer
- Stateful data sampling (resume training)
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, Sampler
from typing import Optional, Iterator, Dict, Any, List
import random


class StatefulSampler(Sampler):
    """
    A sampler that maintains state for resumable training.
    Tracks which samples have been seen and can be saved/restored.
    """
    
    def __init__(self, data_source: Dataset, shuffle: bool = True, seed: int = 42):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.current_index = 0
        self._indices: Optional[List[int]] = None
        self._generate_indices()
    
    def _generate_indices(self):
        """Generate shuffled indices for current epoch."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        
        if self.shuffle:
            self._indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        else:
            self._indices = list(range(len(self.data_source)))
    
    def __iter__(self) -> Iterator[int]:
        """Iterate from current position."""
        while self.current_index < len(self._indices):
            yield self._indices[self.current_index]
            self.current_index += 1
        
        # Reset for next epoch
        self.epoch += 1
        self.current_index = 0
        self._generate_indices()
    
    def __len__(self) -> int:
        return len(self.data_source)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get sampler state for checkpointing."""
        return {
            "epoch": self.epoch,
            "current_index": self.current_index,
            "seed": self.seed,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore sampler state from checkpoint."""
        self.epoch = state_dict["epoch"]
        self.seed = state_dict.get("seed", self.seed)
        self._generate_indices()
        self.current_index = state_dict["current_index"]
        print(f"  🔄 Restored sampler: epoch={self.epoch}, position={self.current_index}/{len(self)}")


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
    """Streaming dataset for large corpora with resume support."""
    
    def __init__(
        self,
        hf_dataset_name: str = "roneneldan/TinyStories",
        split: str = "train",
        tokenizer = None,
        max_length: int = 512,
        seed: int = 42,
        skip_samples: int = 0,  # Number of samples to skip on resume
    ):
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.skip_samples = skip_samples
        self.samples_yielded = 0  # Track samples yielded in current session
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
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            "samples_seen": self.samples_yielded + self.skip_samples,
            "skip_samples": self.skip_samples,
        }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        dataset = self._load_dataset()
        skipped = 0
        
        # Log skip progress for large skips
        if self.skip_samples > 0:
            print(f"  ⏩ Skipping {self.skip_samples} previously seen samples...")
        
        for example in dataset:
            # Skip previously seen samples
            if skipped < self.skip_samples:
                skipped += 1
                if skipped % 10000 == 0:
                    print(f"     Skipped {skipped}/{self.skip_samples}...")
                continue
            
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
            
            self.samples_yielded += 1
            yield {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
            }


class TinyShakespeareDataset(Dataset):
    """
    TinyShakespeare dataset - downloads ~1MB of Shakespeare text.
    Uses simple character-level encoding (no tokenizer needed).
    """
    
    URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    def __init__(
        self,
        max_length: int = 512,
        vocab_size: int = 49152,
        cache_dir: str = "./data",
    ):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.cache_dir = cache_dir
        
        # Download and load text
        self.text = self._load_text()
        
        # Create character-to-index mapping
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        
        # Create samples (sliding window)
        self.samples = []
        stride = max_length // 2  # 50% overlap
        for i in range(0, len(self.text) - max_length, stride):
            self.samples.append(self.text[i:i + max_length])
        
        print(f"  📚 Loaded TinyShakespeare: {len(self.text):,} chars, {len(self.samples):,} samples")
    
    def _load_text(self) -> str:
        """Download or load cached text."""
        import os
        import urllib.request
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, "tiny_shakespeare.txt")
        
        if not os.path.exists(cache_path):
            print("  ⬇️  Downloading TinyShakespeare...")
            urllib.request.urlretrieve(self.URL, cache_path)
        
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.samples[idx]
        
        # Simple character-level encoding: use ASCII value mod vocab_size
        # This ensures we stay within vocab bounds
        input_ids = torch.tensor(
            [ord(c) % self.vocab_size for c in text],
            dtype=torch.long
        )
        
        return {
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
    seed: int = 42,
    return_sampler: bool = False,
    skip_samples: int = 0,  # NEW: For resume support
    return_dataset: bool = False,  # NEW: Return dataset for state access
    **kwargs,
):
    """
    Factory function to create a dataloader with optional stateful sampler.
    
    Args:
        dataset_name: "dummy", "tinystories", or a HuggingFace dataset name
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: DataLoader workers
        tokenizer: HuggingFace tokenizer (optional)
        seed: Random seed for reproducibility
        return_sampler: If True, return (dataloader, sampler) for state management
        skip_samples: Number of samples to skip (for resume)
        return_dataset: If True, return (dataloader, dataset) for state access
    
    Returns:
        DataLoader instance, or tuple with sampler/dataset if requested
    """
    sampler = None
    
    if dataset_name == "dummy":
        dataset = DummyDataset(
            max_length=max_length,
            vocab_size=kwargs.get("vocab_size", 49152),
            num_samples=kwargs.get("num_samples", 10000),
        )
        sampler = StatefulSampler(dataset, shuffle=True, seed=seed)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )
    
    elif dataset_name == "tiny_shakespeare" or dataset_name == "shakespeare":
        dataset = TinyShakespeareDataset(
            max_length=max_length,
            vocab_size=kwargs.get("vocab_size", 49152),
        )
        sampler = StatefulSampler(dataset, shuffle=True, seed=seed)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )
    
    elif dataset_name == "tinystories":
        # Streaming dataset with resume support
        dataset = StreamingTextDataset(
            hf_dataset_name="roneneldan/TinyStories",
            tokenizer=tokenizer,
            max_length=max_length,
            skip_samples=skip_samples,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    
    else:
        # Assume HuggingFace dataset name (streaming)
        dataset = StreamingTextDataset(
            hf_dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_length=max_length,
            skip_samples=skip_samples,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    
    if return_dataset:
        return dataloader, dataset
    if return_sampler:
        return dataloader, sampler
    return dataloader


if __name__ == "__main__":
    # Test dummy dataset
    print("Testing DummyDataset...")
    dataset = DummyDataset(num_samples=100, max_length=64)
    dataloader = DataLoader(dataset, batch_size=4)
    
    batch = next(iter(dataloader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    
    print("\nDataset ready for training!")
