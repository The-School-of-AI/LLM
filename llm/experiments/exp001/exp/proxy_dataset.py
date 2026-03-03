from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers.tokenization_utils_tokenizers import TokenizersBackend


@dataclass
class ProxyDatasetConfig:
    local_path: str
    seq_len: int
    batch_size: int
    shuffle_buffer: int = 10000
    start_step: int = 0
    num_workers: int = 2


class SYNTHStream(IterableDataset):
    """
    STRICT Loader: Instant resume using Arrow slicing.
    Supports deterministic ordering for reproducible training.
    """

    def __init__(
        self,
        tokenizer,
        dataset_name="PleIAs/SYNTH",
        local_path="../synth_local_en",
        seq_len=512,
        batch_size=16,
        shuffle_buffer=10000,
        seed=42,
        include_query=True,
        include_reasoning=True,
        include_answer=True,
        combine_separator="\n\n",
        filter_language="en",
        start_step=0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = start_step
        self.combine_separator = combine_separator
        self.include_query = include_query
        self.include_reasoning = include_reasoning
        self.include_answer = include_answer
        self.filter_language = filter_language
        self.full_path = local_path

    def _construct_text(self, ex: Dict[str, Any]) -> Optional[str]:
        """Construct training text from dataset example"""
        # Fast language filter
        if self.filter_language:
            lang = ex.get("language")
            if not lang or (
                isinstance(lang, str) and lang.lower() != self.filter_language.lower()
            ):
                return None

        parts = []
        query = ex.get("query", "").strip()
        if self.include_query and query:
            parts.append(f"<|im_start|>user\n{query}<|im_end|>")

        reasoning = ex.get("synthetic_reasoning", "").strip()
        answer = ex.get("synthetic_answer", "").strip()

        assistant_parts = []
        if self.include_reasoning and reasoning:
            assistant_parts.append(f"`<think>`\n{reasoning}\n`</think>`")
        if self.include_answer and answer:
            assistant_parts.append(answer)

        if assistant_parts:
            assistant_text = self.combine_separator.join(assistant_parts)
            parts.append(f"<|im_start|>assistant\n{assistant_text}")

        if not parts:
            return None
        return "\n".join(parts)

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text and return input_ids as a plain list."""
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors=None,
            max_length=self.seq_len * 2,
            truncation=True,
            padding=False,
        )
        return encoded["input_ids"]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset with deterministic resume support"""
        # Load and shuffle full dataset (global shuffle = deterministic)
        full_ds = load_from_disk(self.full_path)
        full_ds = full_ds.shuffle(seed=self.seed)

        print(f"📊 Dataset loaded: {len(full_ds)} rows")
        print(f"📊 Deterministic Resume: Fast-forwarding {self.start_step} steps...")

        it = iter(full_ds)

        buf: List[int] = []
        samples_to_skip = self.start_step * self.batch_size
        samples_skipped = 0

        # ----------------------------------------------------------------
        # PHASE 1: Fast-Forward (Burn tokens to restore exact state)
        # ----------------------------------------------------------------
        if samples_to_skip > 0:
            pbar = tqdm(total=samples_to_skip, desc="⏩ Fast-Forwarding", unit="seq")

            while samples_skipped < samples_to_skip:
                # Fill buffer
                while len(buf) < self.seq_len:
                    ex: Dict[str, Any]
                    try:
                        ex = next(it)  # type: ignore
                    except StopIteration:
                        it = iter(full_ds)  # Restart same permutation
                        ex = next(it)  # type: ignore

                    text = self._construct_text(ex)
                    if not text:
                        continue

                    ids = self._tokenize(text)
                    if not ids:
                        continue

                    buf.extend(ids)
                    # Keep buffer reasonable size
                    if len(buf) > 4 * self.seq_len:
                        buf[:] = buf[-(4 * self.seq_len) :]

                # Consume from buffer (discard)
                while len(buf) >= self.seq_len and samples_skipped < samples_to_skip:
                    buf = buf[self.seq_len :]
                    samples_skipped += 1
                    pbar.update(1)

            pbar.close()
            print(
                f"✅ Fast-forward complete. Resuming exactly at step {self.start_step}."
            )

        # ----------------------------------------------------------------
        # PHASE 2: Yield Training Data
        # ----------------------------------------------------------------
        while True:
            while len(buf) < self.seq_len:
                try:
                    ex = next(it)  # type: ignore
                except StopIteration:
                    print("🔄 Dataset finished, restarting...")
                    it = iter(full_ds)
                    ex = next(it)  # type: ignore

                text = self._construct_text(ex)
                if not text:
                    continue

                ids = self._tokenize(text)
                if not ids:
                    continue

                buf.extend(ids)
                if len(buf) > 4 * self.seq_len:
                    buf[:] = buf[-(4 * self.seq_len) :]

            block = buf[: self.seq_len]
            buf = buf[self.seq_len :]
            yield {
                "input_ids": torch.tensor(block, dtype=torch.long),
                "labels": torch.tensor(block, dtype=torch.long),
            }


def get_proxy_dataloader(
    tokenizer: TokenizersBackend, config: ProxyDatasetConfig, seed: int
) -> DataLoader:
    ds = SYNTHStream(
        tokenizer,
        seed=seed,
        local_path=config.local_path,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        shuffle_buffer=config.shuffle_buffer,
        start_step=config.start_step,
    )
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
