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
    dataset_format: str = "auto"  # auto | token_ids | text | synth_qa
    token_ids_column: str = "input_ids"
    text_column: str = "text"
    language_column: str = "language"
    filter_language: str | None = "en"
    shuffle_buffer: int = 10000
    start_step: int = 0
    num_workers: int = 2


class ProxyStream(IterableDataset):
    """
    STRICT Loader: Instant resume using Arrow slicing.
    Supports deterministic ordering for reproducible training.
    """

    def __init__(
        self,
        tokenizer,
        local_path="./_data/proxy_local",
        seq_len=512,
        batch_size=16,
        dataset_format="auto",
        token_ids_column="input_ids",
        text_column="text",
        language_column="language",
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
        self.dataset_format = dataset_format
        self.token_ids_column = token_ids_column
        self.text_column = text_column
        self.language_column = language_column
        self.seed = seed
        self.start_step = start_step
        self.combine_separator = combine_separator
        self.include_query = include_query
        self.include_reasoning = include_reasoning
        self.include_answer = include_answer
        self.filter_language = filter_language
        self.full_path = local_path

    def _construct_synth_text(self, ex: Dict[str, Any]) -> Optional[str]:
        """Construct text from SYNTH-style QA examples."""
        # Fast language filter
        if self.filter_language:
            lang = ex.get(self.language_column)
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

    def _detect_mode(self, full_ds) -> str:
        cols = set(full_ds.column_names)
        mode = self.dataset_format
        if mode != "auto":
            return mode

        if self.token_ids_column in cols:
            return "token_ids"
        if self.text_column in cols:
            return "text"
        if {
            "query",
            "synthetic_reasoning",
            "synthetic_answer",
        }.intersection(cols):
            return "synth_qa"
        raise ValueError(
            "Could not auto-detect proxy dataset format. "
            f"Columns={sorted(cols)}. "
            "Set proxy.dataset_format to one of: token_ids, text, synth_qa."
        )

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

    def _example_to_ids(self, ex: Dict[str, Any], mode: str) -> Optional[List[int]]:
        if mode == "token_ids":
            ids = ex.get(self.token_ids_column)
            if ids is None:
                return None
            if torch.is_tensor(ids):
                ids = ids.tolist()
            if not isinstance(ids, list) or len(ids) == 0:
                return None
            return [int(x) for x in ids]

        if mode == "text":
            text = ex.get(self.text_column, "")
            if not isinstance(text, str) or not text.strip():
                return None
            return self._tokenize(text)

        if mode == "synth_qa":
            text = self._construct_synth_text(ex)
            if not text:
                return None
            return self._tokenize(text)

        raise ValueError(f"Unsupported proxy dataset_format: {mode}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset with deterministic resume support"""
        # Load and shuffle full dataset (global shuffle = deterministic)
        full_ds = load_from_disk(self.full_path)
        full_ds = full_ds.shuffle(seed=self.seed)
        mode = self._detect_mode(full_ds)

        print(f"📊 Dataset loaded: {len(full_ds)} rows")
        print(f"📊 Proxy dataset mode: {mode}")
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

                    ids = self._example_to_ids(ex, mode)
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

                ids = self._example_to_ids(ex, mode)
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
    ds = ProxyStream(
        tokenizer,
        seed=seed,
        local_path=config.local_path,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        dataset_format=config.dataset_format,
        token_ids_column=config.token_ids_column,
        text_column=config.text_column,
        language_column=config.language_column,
        filter_language=config.filter_language,
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


# Backward-compatible alias for older references.
SYNTHStream = ProxyStream
