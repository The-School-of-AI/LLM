"""
Data Loading and Processing for SmolLM Training

Includes:
- SYNTH dataset streaming with deterministic resume
- Prompt sampler for generation
- BPE token utilities for Fourier embeddings
- Character vocabulary utilities
"""

import os
import random
import sys
import tempfile
import uuid
from typing import Any, Dict, Iterator, List, Optional

import torch
from datasets import load_from_disk
from torch.utils.data import IterableDataset
from tqdm import tqdm

# ============================================================================
# BPE TOKEN UTILITIES (For Fourier Embeddings)
# ============================================================================


def discover_chars_from_bpe_tokenizer(tokenizer, vocab_size=50272):
    """
    Extract all unique characters from the BPE tokenizer's vocabulary.
    This ensures we can handle any BPE token that appears in the dataset.
    """
    print("🔍 Discovering characters from BPE tokenizer...")
    all_chars = set()

    for token_id in tqdm(
        range(min(vocab_size, len(tokenizer))), desc="Extracting chars"
    ):
        try:
            token_text = tokenizer.decode([token_id])
            all_chars.update(token_text)
        except Exception:
            continue

    chars_list = sorted(list(all_chars))
    char_to_id = {ch: i for i, ch in enumerate(chars_list)}

    print(f"📝 Found {len(chars_list)} unique characters in BPE vocabulary")
    print(f"📝 Sample characters: {chars_list[:20]}...")

    return chars_list, char_to_id


def pad_char_vocab_128(chars):
    """Pad character vocabulary to exactly 128 chars"""
    base = [chr(i) for i in range(32, 127)]
    for ch in base:
        if len(chars) >= 128:
            break
        if ch not in chars:
            chars.append(ch)

    chars = chars[:128]

    seen = set()
    uniq = []
    for ch in chars:
        if ch not in seen:
            uniq.append(ch)
            seen.add(ch)

    i = 0
    while len(uniq) < 128:
        placeholder = f"¤{i}"
        if placeholder not in seen:
            uniq.append(placeholder)
            seen.add(placeholder)
        i += 1

    char_to_id = {ch: i for i, ch in enumerate(uniq)}
    return uniq, char_to_id


def create_bpe_token_strings(tokenizer, vocab_size=50272):
    """
    Convert BPE token IDs to strings for the Fourier embeddings.
    """
    print("🔄 Converting BPE tokens to strings for Fourier processing...")
    bpe_vocab = []

    for token_id in tqdm(
        range(min(vocab_size, len(tokenizer))), desc="Converting BPE tokens"
    ):
        try:
            token_text = tokenizer.decode([token_id])
            bpe_vocab.append(token_text)
        except Exception:
            bpe_vocab.append(f"<TOKEN_{token_id}>")

    print(f"📝 Created {len(bpe_vocab)} BPE token strings")
    print(f"📝 Sample tokens: {bpe_vocab[:10]}")
    return bpe_vocab


# ============================================================================
# SYNTH DATASET
# ============================================================================


def _is_hf_saved_dataset_dir(path: str) -> bool:
    """
    Return True if `path` looks like a HuggingFace load_from_disk directory.
    """
    if not os.path.isdir(path):
        return False

    # Dataset roots typically have dataset_info.json.
    # DatasetDict roots typically have dataset_dict.json.
    dataset_info = os.path.join(path, "dataset_info.json")
    dataset_dict = os.path.join(path, "dataset_dict.json")
    return os.path.isfile(dataset_info) or os.path.isfile(dataset_dict)


def _dataset_path_candidates(local_path: str) -> List[str]:
    """
    Build candidate absolute paths for a local dataset directory.
    """
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    leaf_name = os.path.basename(os.path.normpath(local_path))

    raw_candidates = []

    if os.path.isabs(local_path):
        raw_candidates.append(local_path)
    else:
        raw_candidates.append(os.path.join(cwd, local_path))
        raw_candidates.append(os.path.join(script_dir, local_path))

        # Also try parent dirs with only the leaf name.
        if leaf_name:
            raw_candidates.append(os.path.join(os.path.dirname(cwd), leaf_name))
            raw_candidates.append(os.path.join(os.path.dirname(script_dir), leaf_name))

    # De-duplicate while preserving order.
    seen = set()
    unique_candidates = []
    for path in raw_candidates:
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        unique_candidates.append(abs_path)

    return unique_candidates


def _resolve_local_dataset_path(local_path: str) -> Optional[str]:
    """
    Resolve the best local dataset path by checking common locations.
    Also supports one level of nesting (e.g. synth_local_en/synth_local_en).
    """
    for candidate in _dataset_path_candidates(local_path):
        if _is_hf_saved_dataset_dir(candidate):
            return candidate

        if not os.path.isdir(candidate):
            continue

        # Some manual copies create an extra wrapper directory.
        for child_name in sorted(os.listdir(candidate)):
            child = os.path.join(candidate, child_name)
            if _is_hf_saved_dataset_dir(child):
                return child

    return None


def _shuffle_with_temp_cache(dataset, seed: int):
    """
    Shuffle deterministically while forcing indices cache into temp dir.
    This avoids write failures when dataset directories are read-only.
    """
    cache_file = os.path.join(
        tempfile.gettempdir(),
        f"synth_shuffle_indices_{os.getpid()}_{uuid.uuid4().hex}.arrow",
    )
    return dataset.shuffle(seed=seed, indices_cache_file_name=cache_file)


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

        # --- SMART PATH FINDER ---
        resolved_path = _resolve_local_dataset_path(local_path)
        if resolved_path is None:
            checked_paths = _dataset_path_candidates(local_path)
            pretty_paths = "\n".join(f"   - {p}" for p in checked_paths)
            raise FileNotFoundError(
                "Could not locate a HuggingFace dataset directory for "
                f"local_path='{local_path}'.\n"
                "Checked:\n"
                f"{pretty_paths}\n"
                "Expected files: dataset_info.json or dataset_dict.json.\n"
                "If your folder exists but is empty, place the saved dataset "
                "contents there first.\n"
                "Tip: run `python download_mini_synth.py --output-dir ../synth_local_en` "
                "from endGame/ to create it."
            )

        self.full_path = resolved_path

        print(f"📂 SYNTHStream loading from: {self.full_path}")

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

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset with deterministic resume support"""
        try:
            # Load and shuffle full dataset (global shuffle = deterministic)
            full_ds = load_from_disk(self.full_path)
            full_ds = _shuffle_with_temp_cache(full_ds, self.seed)

            print(f"📊 Dataset loaded: {len(full_ds)} rows")
            print(
                f"📊 Deterministic Resume: Fast-forwarding {self.start_step} steps..."
            )

            it = iter(full_ds)

        except Exception as e:
            print(f"❌ Critical Error loading dataset: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

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
                    try:
                        ex = next(it)
                    except StopIteration:
                        it = iter(full_ds)  # Restart same permutation
                        ex = next(it)

                    text = self._construct_text(ex)
                    if not text:
                        continue

                    encoded = self.tokenizer(
                        text,
                        add_special_tokens=False,
                        return_tensors=None,
                        max_length=self.seq_len * 2,
                        truncation=True,
                        padding=False,
                    )
                    ids = encoded["input_ids"]
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
                    ex = next(it)
                except StopIteration:
                    print("🔄 Dataset finished, restarting...")
                    it = iter(full_ds)
                    ex = next(it)

                text = self._construct_text(ex)
                if not text:
                    continue

                encoded = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_tensors=None,
                    max_length=self.seq_len * 2,
                    truncation=True,
                    padding=False,
                )
                ids = encoded["input_ids"]
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


# ============================================================================
# SYNTH PROMPT SAMPLER (For Generation)
# ============================================================================


class SYNTHPromptSampler:
    """
    STRICT Sampler: Checks multiple locations for synth_local.
    Provides deterministic prompt sampling for evaluation.
    """

    def __init__(
        self,
        dataset_name="PleIAs/SYNTH",
        local_path="../synth_local_en",
        tokenizer=None,
        seed=42,
    ):
        self.tokenizer = tokenizer
        self.seed = seed

        # --- SMART PATH FINDER ---
        resolved_path = _resolve_local_dataset_path(local_path)
        if resolved_path is None:
            print(
                "[PROMPTS] ⚠️ No valid HuggingFace local dataset found "
                f"for local_path='{local_path}'."
            )
            self.dataset = None
            return

        self.full_path = resolved_path

        print(f"[PROMPTS] Initializing sampler from: {self.full_path}")

        try:
            self.dataset = load_from_disk(self.full_path)
            print(f"[PROMPTS] ✅ Loaded {len(self.dataset)} examples locally")
        except Exception as e:
            print(f"[PROMPTS] ❌ Failed to load local: {e}")
            self.dataset = None

    def sample_token_ids(self, n: int = 5, step: int = 0) -> List[torch.Tensor]:
        """
        Sample n prompts for generation evaluation.
        Uses step as seed modifier for deterministic sampling.
        """
        if self.dataset is None:
            return []

        prompts_t = []
        rng = random.Random(self.seed + step)
        total_rows = len(self.dataset)
        attempts = 0

        while len(prompts_t) < n and attempts < n * 10:
            idx = rng.randint(0, total_rows - 1)
            ex = self.dataset[idx]
            attempts += 1

            lang = ex.get("language", "")
            if lang is None or str(lang).lower() != "en":
                continue

            query = ex.get("query", "").strip()
            if not query:
                continue

            formatted = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
            encoded = self.tokenizer(
                formatted,
                add_special_tokens=False,
                return_tensors=None,
                max_length=512,
                truncation=True,
                padding=False,
            )
            prompts_t.append(torch.tensor(encoded["input_ids"], dtype=torch.long))

        return prompts_t
