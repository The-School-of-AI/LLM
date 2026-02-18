"""
Data Loading and Processing for SmolLM Training

Includes:
- SYNTH dataset streaming with deterministic resume
- Prompt sampler for generation
- BPE token utilities for Fourier embeddings
- Character vocabulary utilities
"""

import os
import sys
import random
from typing import Optional, List, Dict, Any, Iterator
import torch
from torch.utils.data import IterableDataset
from datasets import load_from_disk
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

    for token_id in tqdm(range(min(vocab_size, len(tokenizer))), desc="Extracting chars"):
        try:
            token_text = tokenizer.decode([token_id])
            all_chars.update(token_text)
        except Exception as e:
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
        placeholder = f'¤{i}'
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

    for token_id in tqdm(range(min(vocab_size, len(tokenizer))), desc="Converting BPE tokens"):
        try:
            token_text = tokenizer.decode([token_id])
            bpe_vocab.append(token_text)
        except Exception as e:
            bpe_vocab.append(f"<TOKEN_{token_id}>")

    print(f"📝 Created {len(bpe_vocab)} BPE token strings")
    print(f"📝 Sample tokens: {bpe_vocab[:10]}")
    return bpe_vocab


# ============================================================================
# SYNTH DATASET
# ============================================================================

class SYNTHStream(IterableDataset):
    """
    STRICT Loader: Instant resume using Arrow slicing.
    Supports deterministic ordering for reproducible training.
    """
    def __init__(self, tokenizer, dataset_name="PleIAs/SYNTH", local_path="../synth_local_en",
                 seq_len=512, batch_size=16, shuffle_buffer=10000, seed=42,
                 include_query=True, include_reasoning=True, include_answer=True,
                 combine_separator="\n\n", filter_language="en", start_step=0):
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
        cwd_path = os.path.join(os.getcwd(), local_path)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, local_path)
        parent_path = os.path.join(os.path.dirname(script_dir), local_path)

        if os.path.exists(cwd_path):
            self.full_path = cwd_path
        elif os.path.exists(script_path):
            self.full_path = script_path
        elif os.path.exists(parent_path):
            self.full_path = parent_path
        else:
            print(f"\n❌ ERROR: Could not find '{local_path}'. Run download_mini_synth.py first.")
            sys.exit(1)

        print(f"📂 SYNTHStream loading from: {self.full_path}")

    def _construct_text(self, ex: Dict[str, Any]) -> Optional[str]:
        """Construct training text from dataset example"""
        # Fast language filter
        if self.filter_language:
            lang = ex.get("language")
            if not lang or (isinstance(lang, str) and lang.lower() != self.filter_language.lower()):
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
            full_ds = full_ds.shuffle(seed=self.seed)

            print(f"📊 Dataset loaded: {len(full_ds)} rows")
            print(f"📊 Deterministic Resume: Fast-forwarding {self.start_step} steps...")

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

                    encoded = self.tokenizer.encode_plus(
                        text, add_special_tokens=False, return_tensors=None,
                        max_length=self.seq_len * 2, truncation=True, padding=False,
                    )
                    ids = encoded["input_ids"]
                    if not ids:
                        continue

                    buf.extend(ids)
                    # Keep buffer reasonable size
                    if len(buf) > 4 * self.seq_len:
                        buf[:] = buf[-(4 * self.seq_len):]

                # Consume from buffer (discard)
                while len(buf) >= self.seq_len and samples_skipped < samples_to_skip:
                    buf = buf[self.seq_len:]
                    samples_skipped += 1
                    pbar.update(1)

            pbar.close()
            print(f"✅ Fast-forward complete. Resuming exactly at step {self.start_step}.")

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

                encoded = self.tokenizer.encode_plus(
                    text, add_special_tokens=False, return_tensors=None,
                    max_length=self.seq_len * 2, truncation=True, padding=False,
                )
                ids = encoded["input_ids"]
                if not ids:
                    continue

                buf.extend(ids)
                if len(buf) > 4 * self.seq_len:
                    buf[:] = buf[-(4 * self.seq_len):]

            block = buf[:self.seq_len]
            buf = buf[self.seq_len:]
            yield {
                "input_ids": torch.tensor(block, dtype=torch.long),
                "labels": torch.tensor(block, dtype=torch.long)
            }


# ============================================================================
# SYNTH PROMPT SAMPLER (For Generation)
# ============================================================================

class SYNTHPromptSampler:
    """
    STRICT Sampler: Checks multiple locations for synth_local.
    Provides deterministic prompt sampling for evaluation.
    """
    def __init__(self, dataset_name="PleIAs/SYNTH", local_path="../synth_local_en",
                 tokenizer=None, seed=42):
        self.tokenizer = tokenizer
        self.seed = seed

        # --- SMART PATH FINDER ---
        cwd_path = os.path.join(os.getcwd(), local_path)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, local_path)
        parent_path = os.path.join(os.path.dirname(script_dir), local_path)

        if os.path.exists(cwd_path):
            self.full_path = cwd_path
        elif os.path.exists(script_path):
            self.full_path = script_path
        elif os.path.exists(parent_path):
            self.full_path = parent_path
        else:
            print(f"[PROMPTS] ⚠️ Local dataset not found in CWD, Script Dir, or Parent Dir.")
            self.dataset = None
            return

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
            encoded = self.tokenizer.encode_plus(
                formatted, add_special_tokens=False, return_tensors=None,
                max_length=512, truncation=True, padding=False
            )
            prompts_t.append(torch.tensor(encoded["input_ids"], dtype=torch.long))

        return prompts_t
