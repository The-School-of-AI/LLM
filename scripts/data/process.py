#!/usr/bin/env python3
"""
Data processing pipeline: parquet → clean → tokenize → band → pack → shard.

Produces shards in bin_idx format compatible with bin_idx_dataloader.py:
  shard_dir/
    band_<B>/
      shard_000000/
        tokens.bin    — uint32 token IDs, packed 4096-token blocks
        tokens.idx    — uint64 byte offsets for each block
        metadata.json — tokenizer_hash, eos/pad IDs, band, domain, stats

Performance notes:
  - multiprocessing.Pool for parallel parquet file processing
  - tokenizers library (Rust) for fast tokenization (~3M chars/s English)
  - BandPacker: stateful packing with carry-over — zero tail loss between batches
  - ShardWriter: incremental flush to disk — O(shard_max_blocks) peak memory per band
  - _write_shard: arithmetic .idx offsets (no f.tell() syscalls), single np.concatenate write
  - Resume support: existing shards (with metadata.json) are skipped

Usage (local test):
    python process.py \\
        --input-dir Tokenizer/indic_tokenizer_samples_by_size \\
        --output-dir /tmp/test_shards \\
        --tokenizer-dir Tokenizer/output_hybrid \\
        --sources erav4_lang_hi,sangraha_hi \\
        --workers 4

Usage (production on c6id.metal):
    python process.py \\
        --input-dir /mnt/nvme/normalized_data \\
        --output-dir /mnt/nvme/shards/candidates \\
        --tokenizer-dir /mnt/nvme/tokenizer \\
        --workers 90 \\
        --shard-max-blocks 8192
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import resource
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from clean import CleaningStats, clean_text, is_valid_document
from tokenizers import Tokenizer

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _ts() -> str:
    """Wall-clock timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S")


def _rss_mb() -> float:
    """Current process RSS in MB (main process only, excludes workers)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in bytes on Linux, KB on macOS
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def _fmt_eta(elapsed: float, done: int, total: int) -> str:
    """Format ETA string from progress."""
    if done == 0 or done >= total:
        return ""
    remaining = elapsed / done * (total - done)
    if remaining < 60:
        return f"ETA {remaining:.0f}s"
    elif remaining < 3600:
        return f"ETA {remaining / 60:.1f}m"
    else:
        return f"ETA {remaining / 3600:.1f}h"


def log(msg: str) -> None:
    """Print with timestamp and flush."""
    print(f"[{_ts()}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

BLOCK_SIZE = 4096  # tokens per block — matches bin_idx_dataloader
BLOCK_BYTES = BLOCK_SIZE * 4  # uint32 = 4 bytes per token
IDX_HEADER = b"\x00" * 8  # 8-byte header reserved for version/magic

DEFAULT_SHARD_MAX_BLOCKS = 8192  # 8192 × 4096 × 4 ≈ 128 MB per shard


# ═══════════════════════════════════════════════════════════════════════════
# BAND / DOMAIN MAP
# ═══════════════════════════════════════════════════════════════════════════
# Authoritative mapping from DATA_STRATEGY.md + T2Expectation.md.
# Loaded from --band-map JSON at startup if provided; this dict is fallback.

_BUILTIN_BAND_MAP: Dict[str, Tuple[str, str, str]] = {
    # (band, domain_group, modality)
    # ── English Web (B0-B2) ──────────────────────────────────────────────
    "refinedweb": ("B1", "general_web_clean", "general_text"),
    "C4": ("B1", "general_web_clean", "general_text"),
    "cc_head": ("B0", "general_web_clean", "general_text"),
    "cc_middle": ("B1", "general_web_clean", "general_text"),
    "cc_tail": ("B1", "general_web_clean", "general_text"),
    "cc_news": ("B1", "news_nonpolitical", "general_text"),
    "reddit": ("B0", "dialogue_chat", "general_text"),
    "stackexchange": ("B2", "technical_docs", "general_text"),
    "megawika": ("B1", "encyclopedic", "general_text"),
    "books": ("B2", "general_web_clean", "general_text"),
    # ── English Academic/STEM (B2-B5) ────────────────────────────────────
    "pes2o": ("B3", "math_science", "general_text"),
    "redpajama-arxiv": ("B4", "math_science", "general_text"),
    "proof_pile_2-algebraic_stack": ("B5", "math_science", "general_text"),
    "proof_pile_2-open_web_math": ("B4", "math_science", "general_text"),
    "flan": ("B2", "technical_docs", "general_text"),
    # ── Code (B3-B5) ─────────────────────────────────────────────────────
    "Starcoder": ("B3", "code_repos", "code"),
    # ── Indic: Sangraha (B0-B2) ──────────────────────────────────────────
    "sangraha_hi": ("B1", "general_web_clean", "general_text"),
    "sangraha_bn": ("B1", "general_web_clean", "general_text"),
    "sangraha_ta": ("B1", "general_web_clean", "general_text"),
    "sangraha_te": ("B1", "general_web_clean", "general_text"),
    "sangraha_kn": ("B1", "general_web_clean", "general_text"),
    "sangraha_ml": ("B1", "general_web_clean", "general_text"),
    "sangraha_gu": ("B1", "general_web_clean", "general_text"),
    "sangraha_mr": ("B1", "general_web_clean", "general_text"),
    "sangraha_pa": ("B1", "general_web_clean", "general_text"),
    "sangraha_or": ("B1", "general_web_clean", "general_text"),
    "sangraha_as": ("B1", "general_web_clean", "general_text"),
    # ── Indic: AI4Bharat (B0-B2) ─────────────────────────────────────────
    "ai-bharath-BPCC_seed": ("B1", "general_web_clean", "general_text"),
    "ai-bharath-comparable": ("B1", "general_web_clean", "general_text"),
    "ai-bharath-daily": ("B0", "news_nonpolitical", "general_text"),
    "ai-bharath-ilci": ("B1", "general_web_clean", "general_text"),
    "ai-bharath-massive": ("B0", "dialogue_chat", "general_text"),
    "ai-bharath-nllb_filtered": ("B1", "general_web_clean", "general_text"),
    "ai-bharath-samanantar": ("B1", "general_web_clean", "general_text"),
    "ai-bharath-wiki": ("B1", "encyclopedic", "general_text"),
    # ── Indic: ERAV4 (B0-B1) ─────────────────────────────────────────────
    "erav4_lang_hi": ("B0", "dialogue_chat", "general_text"),
    "erav4_lang_as": ("B0", "dialogue_chat", "general_text"),
    "erav4_lang_kn": ("B0", "dialogue_chat", "general_text"),
    "erav4_lang_mr": ("B0", "dialogue_chat", "general_text"),
    "erav4_lang_pa": ("B0", "dialogue_chat", "general_text"),
    "erav4_lang_te": ("B0", "dialogue_chat", "general_text"),
    "erav4_math": ("B1", "math_science", "general_text"),
    "erav4_pattern": ("B1", "technical_docs", "general_text"),
    # ── Indic: Other (B0-B2) ─────────────────────────────────────────────
    "samvaad_hi": ("B0", "dialogue_chat", "general_text"),
    "sarvamai_mmlu": ("B2", "technical_docs", "general_text"),
    "ncert": ("B2", "technical_docs", "general_text"),
}

_DEFAULT_BAND = ("B1", "general_web_clean", "general_text")
SOURCE_BAND_MAP: Dict[str, Tuple[str, str, str]] = {}
_ALLOW_UNKNOWN_BAND: bool = False

# Band-domain policy from T2Expectation.md — which domains are allowed in which bands.
_BAND_DOMAIN_POLICY: Dict[str, List[str]] = {
    "B0": ["general_web_clean", "dialogue_chat"],
    "B1": ["general_web_clean", "encyclopedic", "dialogue_chat"],
    "B2": ["encyclopedic", "news_nonpolitical", "technical_docs"],
    "B3": ["technical_docs", "math_science", "code_repos"],
    "B4": ["math_science", "code_repos", "planning_reasoning_curated"],
    "B5": ["planning_reasoning_curated", "math_science", "code_repos"],
}


def load_band_map(
    band_map_path: Optional[str] = None, allow_unknown: bool = False
) -> None:
    """Load SOURCE_BAND_MAP from builtin dict + optional JSON overrides."""
    global SOURCE_BAND_MAP, _ALLOW_UNKNOWN_BAND
    SOURCE_BAND_MAP = dict(_BUILTIN_BAND_MAP)
    _ALLOW_UNKNOWN_BAND = allow_unknown

    if band_map_path and os.path.exists(band_map_path):
        with open(band_map_path) as f:
            overrides = json.load(f)
        for source, entry in overrides.items():
            SOURCE_BAND_MAP[source] = (
                entry["band"],
                entry["domain"],
                entry["modality"],
            )
        print(f"  Loaded {len(overrides)} band overrides from {band_map_path}")


def get_band_info(source: str) -> Tuple[str, str, str]:
    """
    Return (band, domain_group, modality).

    Raises ValueError for unknown sources unless --allow-unknown-band is set,
    in which case falls back to B1/general_web_clean with a loud WARNING.
    """
    result = SOURCE_BAND_MAP.get(source)
    if result is None:
        if _ALLOW_UNKNOWN_BAND:
            log(
                f"  WARNING: Source '{source}' not in SOURCE_BAND_MAP — "
                f"defaulting to {_DEFAULT_BAND}. Add it to band_map.json."
            )
            return _DEFAULT_BAND
        raise ValueError(
            f"Source '{source}' not in SOURCE_BAND_MAP and --allow-unknown-band "
            f"is not set. Add it to _BUILTIN_BAND_MAP or band_map.json, or use "
            f"--allow-unknown-band to default to {_DEFAULT_BAND}."
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TOKENIZER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


def compute_tokenizer_hash(tokenizer_dir: str) -> str:
    """
    SHA-256 of tokenizer.json + special_tokens_map.json.
    Must match bin_idx_dataloader.compute_tokenizer_hash exactly.
    """
    files = ["tokenizer.json", "special_tokens_map.json"]
    h = hashlib.sha256()
    for fname in sorted(files):
        fpath = os.path.join(tokenizer_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "rb") as f:
            h.update(fname.encode())
            h.update(f.read())
    return h.hexdigest()


def load_tokenizer(tokenizer_dir: str) -> Tuple[Tokenizer, int, int]:
    """
    Load tokenizer, return (tokenizer, eos_token_id, pad_token_id).
    EOS/PAD resolved from special_tokens_map.json, with fallback to known names.
    """
    tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))

    stm_path = os.path.join(tokenizer_dir, "special_tokens_map.json")
    eos_id = None
    pad_id = None

    if os.path.exists(stm_path):
        with open(stm_path) as f:
            stm = json.load(f)
        eos_content = stm.get("eos_token", {})
        pad_content = stm.get("pad_token", {})
        if isinstance(eos_content, dict):
            eos_content = eos_content.get("content", "")
        if isinstance(pad_content, dict):
            pad_content = pad_content.get("content", "")
        if eos_content:
            eos_id = tok.token_to_id(eos_content)
        if pad_content:
            pad_id = tok.token_to_id(pad_content)

    if eos_id is None:
        eos_id = tok.token_to_id("<|end_of_text|>")
    if pad_id is None:
        pad_id = tok.token_to_id("<|pad|>")
    if eos_id is None:
        raise ValueError("Could not determine EOS token ID from tokenizer")
    if pad_id is None:
        pad_id = 0

    return tok, eos_id, pad_id


# ═══════════════════════════════════════════════════════════════════════════
# PACKING — stateful per-band with carry-over
# ═══════════════════════════════════════════════════════════════════════════


class BandPacker:
    """
    Stateful packing buffer for one band.

    Maintains a carry-over buffer between add_documents() calls so no tokens
    are lost at batch boundaries. The carry is always < BLOCK_SIZE tokens.

    Accepts pre-packed numpy arrays from workers (token_array + doc_lengths)
    for fast operation. Internally uses a pre-allocated numpy buffer with
    write pointer to avoid Python list extend/slice overhead.
    """

    def __init__(self, eos_id: int, block_size: int = BLOCK_SIZE):
        self.eos_id = eos_id
        self.block_size = block_size
        # Pre-allocated buffer: 2x block_size is enough since we emit
        # whenever we hit block_size and carry < block_size
        self._buf = np.empty(block_size * 2, dtype=np.uint32)
        self._pos = 0  # write pointer into _buf

    def add_documents(
        self,
        token_array: np.ndarray,
        doc_lengths: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Add documents from a flat token array + length array.
        Returns any complete blocks.
        """
        if len(token_array) == 0:
            return []

        blocks: List[np.ndarray] = []
        buf = self._buf
        pos = self._pos
        eos = self.eos_id
        bs = self.block_size

        offset = 0
        for dlen in doc_lengths:
            doc_end = offset + dlen

            # Grow buffer if needed (rare — only if a single doc > block_size)
            needed = pos + dlen + 1  # +1 for EOS
            if needed > len(buf):
                new_buf = np.empty(max(needed * 2, bs * 4), dtype=np.uint32)
                new_buf[:pos] = buf[:pos]
                buf = new_buf

            # Copy doc tokens + EOS into buffer
            buf[pos : pos + dlen] = token_array[offset:doc_end]
            pos += dlen
            buf[pos] = eos
            pos += 1

            # Emit complete blocks
            while pos >= bs:
                blocks.append(buf[:bs].copy())
                # Shift remainder to front
                remain = pos - bs
                if remain > 0:
                    buf[:remain] = buf[bs:pos]
                pos = remain

            offset = doc_end

        self._buf = buf
        self._pos = pos
        return blocks

    def flush(self) -> Tuple[List[np.ndarray], int]:
        """
        Flush any remaining carry-over.
        Returns (final_blocks, tail_tokens_discarded).
        """
        blocks: List[np.ndarray] = []
        buf = self._buf
        pos = self._pos
        bs = self.block_size

        while pos >= bs:
            blocks.append(buf[:bs].copy())
            remain = pos - bs
            if remain > 0:
                buf[:remain] = buf[bs:pos]
            pos = remain

        tail = pos
        self._pos = 0
        return blocks, tail


# ═══════════════════════════════════════════════════════════════════════════
# SHARD WRITER — incremental flush with resume
# ═══════════════════════════════════════════════════════════════════════════


class ShardWriter:
    """
    Accumulates blocks for one band and flushes to disk when shard_max_blocks
    is reached. Supports resume: skips shards whose metadata.json exists.

    Metadata is passed at flush time from the caller — the writer never stores
    or mutates metadata dicts, avoiding json.dump crashes on internal state.
    """

    def __init__(
        self,
        output_dir: str,
        band: str,
        shard_max_blocks: int,
        shard_counter: List[int],
    ):
        self.output_dir = output_dir
        self.band = band
        self.shard_max_blocks = shard_max_blocks
        self.shard_counter = shard_counter  # shared mutable [int] for unique IDs
        self._pending: List[np.ndarray] = []
        self.shards_written = 0
        self.shards_skipped = 0
        self.blocks_written = 0

    def add_blocks(self, blocks: List[np.ndarray], metadata: Dict[str, Any]) -> None:
        """Add blocks. Flushes to disk when pending >= shard_max_blocks."""
        self._pending.extend(blocks)
        while len(self._pending) >= self.shard_max_blocks:
            chunk = self._pending[: self.shard_max_blocks]
            self._pending = self._pending[self.shard_max_blocks :]
            self._write_chunk(chunk, metadata)

    def flush_remaining(self, metadata: Dict[str, Any]) -> None:
        """Flush any remaining blocks (< shard_max_blocks) as a final shard."""
        if self._pending:
            self._write_chunk(self._pending, metadata)
            self._pending = []

    def _write_chunk(self, blocks: List[np.ndarray], metadata: Dict[str, Any]) -> None:
        shard_idx = self.shard_counter[0]
        self.shard_counter[0] += 1

        shard_name = f"shard_{shard_idx:06d}"
        shard_path = os.path.join(self.output_dir, f"band_{self.band}", shard_name)
        meta_path = os.path.join(shard_path, "metadata.json")

        if os.path.exists(meta_path):
            self.shards_skipped += 1
            log(f"  SHARD {shard_name} ({self.band}) — skipped (resume)")
        else:
            _write_shard(shard_path, blocks, metadata)
            self.shards_written += 1
            size_mb = len(blocks) * BLOCK_BYTES / (1024 * 1024)
            log(
                f"  SHARD {shard_name} ({self.band}) — {len(blocks):,} blocks, {size_mb:.0f} MB written"
            )
        self.blocks_written += len(blocks)


def _write_shard(
    shard_dir: str,
    blocks: List[np.ndarray],
    metadata: Dict[str, Any],
) -> None:
    """
    Write one shard to disk in bin_idx format.

    .idx offsets computed arithmetically (blocks are fixed-size).
    tokens.bin written in one np.concatenate call.
    """
    os.makedirs(shard_dir, exist_ok=True)

    n = len(blocks)

    # tokens.bin — single contiguous write
    all_tokens = np.concatenate(blocks)
    with open(os.path.join(shard_dir, "tokens.bin"), "wb") as f:
        f.write(all_tokens.tobytes())

    # tokens.idx — arithmetic offsets (no syscalls)
    offsets = np.arange(n + 1, dtype=np.uint64) * BLOCK_BYTES
    with open(os.path.join(shard_dir, "tokens.idx"), "wb") as f:
        f.write(IDX_HEADER)
        f.write(offsets.tobytes())

    # metadata.json
    full_meta = {
        **metadata,
        "num_blocks": n,
        "block_size": BLOCK_SIZE,
        "total_tokens": n * BLOCK_SIZE,
        "dtype": "uint32",
    }
    with open(os.path.join(shard_dir, "metadata.json"), "w") as f:
        json.dump(full_meta, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# WORKER: Process one parquet file
# ═══════════════════════════════════════════════════════════════════════════


def _init_worker(tokenizer_dir: str) -> None:
    """Initialize per-worker tokenizer (called once per process)."""
    global _worker_tok, _worker_eos_id
    _worker_tok, _worker_eos_id, _ = load_tokenizer(tokenizer_dir)


# Regex for code density measurement — structural markers, not keywords
_CODE_MARKERS_RE = re.compile(
    r"^\s*(?:def |class |import |from .+ import |function |const |var |let |"
    r"#include|package |public |private |protected |module |use |fn )",
    re.MULTILINE,
)
# Regex for math density measurement — LaTeX patterns and equations
_MATH_MARKERS_RE = re.compile(
    r"\\(?:frac|sum|int|prod|lim|begin\{equation\}|begin\{align\}|"
    r"mathbb|mathcal|partial|nabla|infty|sqrt|over)"
    r"|\$[^$]+\$"
)


def _compute_bigram_entropy(text: str) -> float:
    """
    Character bigram entropy — language-agnostic complexity proxy.

    Simple repetitive text → low entropy, complex diverse text → high entropy.
    Works identically across all scripts (Devanagari, Latin, Tamil, etc.).

    Uses Counter(zip()) which runs in C — ~10-15x faster than the pure Python
    loop version for a 10KB document.
    """
    if len(text) < 2:
        return 0.0
    counts = Counter(zip(text, text[1:]))  # C-level counting, tuple keys not strings
    total = len(text) - 1
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _process_parquet_file(args: Tuple[str, str, str, str, str]) -> Dict[str, Any]:
    """
    Process a single parquet file: read → clean → tokenize.
    Band/domain/modality are resolved by the main process and passed in
    (workers run in separate processes and don't have SOURCE_BAND_MAP).

    Also collects per-document complexity features for band_audit.json:
      - doc_chars: document length in characters
      - char_bigram_entropy: language-agnostic complexity signal
      - had_instruction_format: whether ghost tags were stripped
      - code_density: fraction of lines with structural code markers
      - math_density: LaTeX pattern hits per token
    """
    parquet_path, source_name, band, domain, modality = args
    global _worker_tok, _worker_eos_id
    cleaning = CleaningStats()

    stats: Dict[str, Any] = {
        "file": os.path.basename(parquet_path),
        "rows_read": 0,
        "rows_cleaned": 0,
        "rows_dropped_empty": 0,
        "rows_dropped_short": 0,
        "total_chars": 0,
        "total_tokens": 0,
    }

    # Complexity stats accumulators (running sums, not per-doc lists)
    cx_doc_count = 0
    # Reservoir sampling for median doc chars — O(reservoir_size) memory
    # regardless of doc count, vs O(N) for the full list
    _CX_RESERVOIR_SIZE = 10000
    cx_char_reservoir: List[int] = []
    cx_reservoir_count = 0
    cx_entropy_sum = 0.0
    cx_instruction_count = 0
    cx_code_density_sum = 0.0
    cx_math_density_sum = 0.0

    # Pack tokens as flat arrays for fast pickling (10x faster than nested lists)
    token_chunks: List[np.ndarray] = []
    doc_lengths: List[int] = []
    languages: List[str] = []

    try:
        pf = pq.ParquetFile(parquet_path)
    except Exception as e:
        stats["error"] = str(e)
        return {
            "source": source_name,
            "band": band,
            "domain": domain,
            "modality": modality,
            "language": "unknown",
            "token_array": np.array([], dtype=np.uint32),
            "doc_lengths": np.array([], dtype=np.int32),
            "stats": stats,
            "cleaning_stats": cleaning,
            "complexity": None,
        }

    for batch in pf.iter_batches(batch_size=50000):
        rows = batch.to_pydict()
        texts = rows.get("text", [])
        langs = rows.get("language", ["unknown"] * len(texts))

        stats["rows_read"] += len(texts)

        for text, lang in zip(texts, langs):
            if not text:
                stats["rows_dropped_empty"] += 1
                continue

            raw_len = len(text)
            cleaned = clean_text(text, stats=cleaning)
            if not cleaned:
                stats["rows_dropped_empty"] += 1
                continue

            if not is_valid_document(cleaned):
                stats["rows_dropped_short"] += 1
                continue

            stats["rows_cleaned"] += 1
            stats["total_chars"] += len(cleaned)

            encoded = _worker_tok.encode(cleaned)
            token_ids = encoded.ids
            if not token_ids:
                stats["rows_dropped_short"] += 1
                stats["rows_cleaned"] -= 1
                continue

            stats["total_tokens"] += len(token_ids)
            token_chunks.append(np.array(token_ids, dtype=np.uint32))
            doc_lengths.append(len(token_ids))
            if lang:
                languages.append(str(lang))

            # ── Complexity features (all O(n) or cheaper) ──
            cx_doc_count += 1
            doc_chars = len(cleaned)
            # Reservoir sampling for median — O(reservoir_size) memory
            cx_reservoir_count += 1
            if len(cx_char_reservoir) < _CX_RESERVOIR_SIZE:
                cx_char_reservoir.append(doc_chars)
            else:
                j = random.randint(0, cx_reservoir_count - 1)
                if j < _CX_RESERVOIR_SIZE:
                    cx_char_reservoir[j] = doc_chars

            # Character bigram entropy
            cx_entropy_sum += _compute_bigram_entropy(cleaned)

            # Instruction format detection: >50 chars stripped = likely had markers
            cx_instruction_count += int((raw_len - len(cleaned)) > 50)

            # Code density: structural code markers / total lines
            num_lines = cleaned.count("\n") + 1
            code_hits = len(_CODE_MARKERS_RE.findall(cleaned))
            cx_code_density_sum += code_hits / max(1, num_lines)

            # Math density: LaTeX pattern hits / total tokens
            math_hits = len(_MATH_MARKERS_RE.findall(cleaned))
            cx_math_density_sum += math_hits / max(1, len(token_ids))

    most_common_lang = (
        Counter(languages).most_common(1)[0][0] if languages else "unknown"
    )

    # Build per-file complexity summary
    complexity = None
    if cx_doc_count > 0:
        cx_char_reservoir.sort()
        median_idx = len(cx_char_reservoir) // 2
        complexity = {
            "doc_count": cx_doc_count,
            "median_doc_chars": (
                cx_char_reservoir[median_idx] if cx_char_reservoir else 0
            ),
            "mean_char_bigram_entropy": round(cx_entropy_sum / cx_doc_count, 3),
            "pct_instruction_stripped": round(cx_instruction_count / cx_doc_count, 4),
            "mean_code_density": round(cx_code_density_sum / cx_doc_count, 4),
            "mean_math_density": round(cx_math_density_sum / cx_doc_count, 6),
        }

    # Pack all token chunks into one contiguous array for fast pickling
    # (~10x faster than pickle of List[List[int]] through multiprocessing pipe)
    if token_chunks:
        token_array = np.concatenate(token_chunks)
    else:
        token_array = np.array([], dtype=np.uint32)
    doc_lengths_array = np.array(doc_lengths, dtype=np.int32)

    return {
        "source": source_name,
        "band": band,
        "domain": domain,
        "modality": modality,
        "language": most_common_lang,
        "token_array": token_array,
        "doc_lengths": doc_lengths_array,
        "stats": stats,
        "cleaning_stats": cleaning,
        "complexity": complexity,
    }


# ═══════════════════════════════════════════════════════════════════════════
# FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════


def discover_parquet_files(
    input_dir: str,
    sources: Optional[List[str]] = None,
) -> List[Tuple[str, str, str, str, str]]:
    """
    Discover parquet files under input_dir/source=<NAME>/*.parquet.
    Resolves band/domain/modality in the main process (workers don't have
    SOURCE_BAND_MAP due to spawn-based multiprocessing).
    """
    results = []
    found_sources: set = set()
    input_path = Path(input_dir)

    for source_dir in sorted(input_path.iterdir()):
        if not source_dir.is_dir():
            continue
        dirname = source_dir.name
        source_name = (
            dirname[len("source=") :] if dirname.startswith("source=") else dirname
        )

        if sources and source_name not in sources:
            continue

        parquet_files = sorted(glob.glob(str(source_dir / "*.parquet")))
        if parquet_files:
            found_sources.add(source_name)
        # Resolve band in main process where SOURCE_BAND_MAP is populated
        band, domain, modality = get_band_info(source_name)
        for pf in parquet_files:
            results.append((pf, source_name, band, domain, modality))

    if sources:
        for src in sorted(set(sources) - found_sources):
            warnings.warn(
                f"Requested source '{src}' not found under {input_dir} — "
                f"no files will be processed for it.",
                stacklevel=2,
            )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


def run_pipeline(
    input_dir: str,
    output_dir: str,
    tokenizer_dir: str,
    sources: Optional[List[str]] = None,
    workers: int = 4,
    shard_max_blocks: int = DEFAULT_SHARD_MAX_BLOCKS,
    band_map_path: Optional[str] = None,
    allow_unknown_band: bool = False,
    dry_run: bool = False,
    verify_after: bool = False,
) -> None:
    t_start = time.time()

    load_band_map(band_map_path, allow_unknown=allow_unknown_band)

    log("=" * 70)
    log("DATA PROCESSING PIPELINE")
    log("=" * 70)
    log(f"  Input:            {input_dir}")
    log(f"  Output:           {output_dir}")
    log(f"  Tokenizer:        {tokenizer_dir}")
    log(f"  Workers:          {workers}")
    log(f"  Block size:       {BLOCK_SIZE} tokens")
    log(f"  Max blocks/shard: {shard_max_blocks}")
    log(f"  RSS at start:     {_rss_mb():.0f} MB")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    log("Loading tokenizer...")
    tok, eos_id, pad_id = load_tokenizer(tokenizer_dir)
    tok_hash = compute_tokenizer_hash(tokenizer_dir)
    log(f"  Vocab size: {tok.get_vocab_size():,}")
    log(f"  EOS ID:     {eos_id}   PAD ID: {pad_id}")
    log(f"  Hash:       {tok_hash[:16]}...")

    # ── Discover files ─────────────────────────────────────────────────────
    log("Discovering parquet files...")
    file_list = discover_parquet_files(input_dir, sources)
    if not file_list:
        log("ERROR: No parquet files found. Check --input-dir and --sources.")
        sys.exit(1)

    source_counts: Dict[str, int] = defaultdict(int)
    source_band_info: Dict[str, Tuple[str, str, str]] = {}
    for _, src, band, domain, modality in file_list:
        source_counts[src] += 1
        source_band_info[src] = (band, domain, modality)
    log(f"  Found {len(file_list)} files across {len(source_counts)} sources:")
    for src, count in sorted(source_counts.items()):
        band, domain, modality = source_band_info[src]
        log(f"    {src}: {count} files → {band} ({domain})")

    # ── Band-domain policy check ──────────────────────────────────────────
    domain_violations: List[Dict[str, str]] = []
    for src, (band, domain, modality) in sorted(source_band_info.items()):
        allowed = _BAND_DOMAIN_POLICY.get(band, [])
        if allowed and domain not in allowed:
            violation = {
                "source": src,
                "band": band,
                "domain": domain,
                "violation": f"{domain} not in {band} allowed domains {allowed}",
            }
            domain_violations.append(violation)
            log(
                f"  WARNING: band_domain_policy violation: {src} → "
                f"{band}/{domain} — {domain} not in {allowed}"
            )
    if domain_violations:
        log(f"  {len(domain_violations)} band_domain_policy violation(s) found")
    else:
        log(f"  Band-domain policy: all {len(source_band_info)} sources OK")

    # ── Dry run: print table and exit ─────────────────────────────────────
    if dry_run:
        log("DRY RUN — printing source→band table and exiting.")
        print(f"\n{'Source':<35} {'Band':>4}  {'Domain':<28} {'Files':>5}")
        print("-" * 78)
        for src, count in sorted(source_counts.items()):
            band, domain, _ = source_band_info[src]
            marker = (
                " !!!" if any(v["source"] == src for v in domain_violations) else ""
            )
            print(f"  {src:<33} {band:>4}  {domain:<28} {count:>5}{marker}")
        print(f"\n  Total: {len(file_list)} files, {len(source_counts)} sources")
        if domain_violations:
            print(
                f"  !!! = band_domain_policy violation ({len(domain_violations)} total)"
            )
        return

    # ── Process + pack + write (streaming, interleaved) ───────────────────
    log(f"Spawning {workers} worker processes (each loads tokenizer)...")
    t_pool_start = time.time()

    # Shared shard counter for globally unique shard IDs across all bands
    shard_counter: List[int] = [0]

    # Per-band state — tracked SEPARATELY from shard metadata to avoid pollution
    band_packers: Dict[str, BandPacker] = {}
    band_writers: Dict[str, ShardWriter] = {}
    band_sources: Dict[str, set] = defaultdict(set)
    band_doc_counts: Dict[str, int] = defaultdict(int)
    # Per-band language and domain tracking — Counter for accurate aggregation
    band_languages: Dict[str, Counter] = defaultdict(Counter)
    band_domains: Dict[str, Counter] = defaultdict(Counter)
    band_modalities: Dict[str, Counter] = defaultdict(Counter)

    # Per-source complexity stats for band_audit.json
    source_complexity: Dict[str, Dict[str, Any]] = {}

    total_stats: Dict[str, int] = defaultdict(int)
    total_cleaning = CleaningStats()

    # chunksize=1 for better load balancing with skewed file sizes
    # (a single large StarCoder parquet vs many tiny ERAV4 parquets)
    chunksize = 1

    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(tokenizer_dir,),
    ) as pool:
        t_pool_ready = time.time()
        log(
            f"Worker pool ready in {t_pool_ready - t_pool_start:.1f}s — "
            f"processing {len(file_list)} files..."
        )

        results_iter = pool.imap_unordered(
            _process_parquet_file,
            file_list,
            chunksize=chunksize,
        )

        t_process = time.time()
        last_progress_time = t_process  # track wall-clock for time-based logging

        for i, result in enumerate(results_iter, 1):
            stats = result["stats"]
            band = result["band"]
            source = result["source"]

            # Accumulate global stats
            for key in (
                "rows_read",
                "rows_cleaned",
                "rows_dropped_empty",
                "rows_dropped_short",
                "total_chars",
                "total_tokens",
            ):
                total_stats[key] += stats.get(key, 0)
            total_cleaning += result["cleaning_stats"]

            if "error" in stats:
                total_stats["files_errored"] += 1
                log(f"  [{i}/{len(file_list)}] ERROR {stats['file']}: {stats['error']}")
                continue

            total_stats["files_processed"] += 1

            # Track per-band metadata via Counters for accurate aggregation
            band_sources[band].add(source)
            n_cleaned = stats.get("rows_cleaned", 0)
            band_doc_counts[band] += n_cleaned
            band_languages[band][result["language"]] += n_cleaned
            band_domains[band][result["domain"]] += n_cleaned
            band_modalities[band][result["modality"]] += n_cleaned

            # Accumulate per-source complexity stats
            cx = result.get("complexity")
            if cx:
                if source not in source_complexity:
                    source_complexity[source] = {
                        "band": band,
                        "domain": result["domain"],
                        "doc_count": 0,
                        "char_lengths": [],
                        "entropy_sum": 0.0,
                        "instruction_count": 0,
                        "code_density_sum": 0.0,
                        "math_density_sum": 0.0,
                    }
                sc = source_complexity[source]
                sc["doc_count"] += cx["doc_count"]
                sc["char_lengths"].append(cx["median_doc_chars"])
                sc["entropy_sum"] += cx["mean_char_bigram_entropy"] * cx["doc_count"]
                sc["instruction_count"] += round(
                    cx["pct_instruction_stripped"] * cx["doc_count"]
                )
                sc["code_density_sum"] += cx["mean_code_density"] * cx["doc_count"]
                sc["math_density_sum"] += cx["mean_math_density"] * cx["doc_count"]

            # Initialize per-band packer and writer on first encounter
            if band not in band_packers:
                band_packers[band] = BandPacker(eos_id)
                band_writers[band] = ShardWriter(
                    output_dir,
                    band,
                    shard_max_blocks,
                    shard_counter,
                )

            # Pack tokens → blocks → shard writer (streaming: memory bounded)
            new_blocks = band_packers[band].add_documents(
                result["token_array"],
                result["doc_lengths"],
            )
            if new_blocks:
                # Use most common language/domain/modality for this band
                top_lang = band_languages[band].most_common(1)[0][0]
                top_domain = band_domains[band].most_common(1)[0][0]
                top_modality = band_modalities[band].most_common(1)[0][0]
                shard_meta = {
                    "tokenizer_hash": tok_hash,
                    "eos_token_id": eos_id,
                    "pad_token_id": pad_id,
                    "band": band,
                    "domain": top_domain,
                    "modality": top_modality,
                    "language": top_lang,
                    "languages": [l for l, _ in band_languages[band].most_common(3)],
                    "sources": sorted(band_sources[band]),
                }
                band_writers[band].add_blocks(new_blocks, shard_meta)

            # Progress logging — print on EITHER:
            #   a) every 5% of files, OR
            #   b) every 30 seconds (whichever comes first)
            #   c) always on last file
            now = time.time()
            elapsed = now - t_process
            pct_interval = max(1, len(file_list) // 20)
            time_triggered = (now - last_progress_time) >= 30.0
            pct_triggered = i % pct_interval == 0
            is_last = i == len(file_list)

            if pct_triggered or time_triggered or is_last:
                last_progress_time = now
                chars_s = total_stats["total_chars"] / elapsed if elapsed > 0 else 0
                toks_s = total_stats["total_tokens"] / elapsed if elapsed > 0 else 0
                eta = _fmt_eta(elapsed, i, len(file_list))
                total_shards_so_far = sum(
                    w.shards_written for w in band_writers.values()
                )
                bands_str = " ".join(
                    f"{b}:{band_doc_counts[b]:,}d"
                    for b in sorted(band_doc_counts.keys())
                )
                log(
                    f"  [{i}/{len(file_list)}] "
                    f"{total_stats['rows_cleaned']:,} docs, "
                    f"{total_stats['total_tokens']:,} tok, "
                    f"{chars_s / 1e6:.1f}M ch/s, {toks_s / 1e6:.2f}M tok/s | "
                    f"shards={total_shards_so_far} RSS={_rss_mb():.0f}MB "
                    f"{eta}"
                )
                if bands_str:
                    log(f"    bands: {bands_str}")

    t_process_done = time.time()
    log(f"Processing complete in {t_process_done - t_process:.1f}s")

    # ── Final flush ────────────────────────────────────────────────────────
    log("Flushing final shards...")
    total_tail_discarded = 0

    for band in sorted(band_packers.keys()):
        # Flush carry-over from packer
        final_blocks, tail = band_packers[band].flush()
        total_tail_discarded += tail

        top_lang = (
            band_languages[band].most_common(1)[0][0]
            if band_languages[band]
            else "unknown"
        )
        top_domain = (
            band_domains[band].most_common(1)[0][0] if band_domains[band] else "unknown"
        )
        top_modality = (
            band_modalities[band].most_common(1)[0][0]
            if band_modalities[band]
            else "unknown"
        )
        shard_meta = {
            "tokenizer_hash": tok_hash,
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
            "band": band,
            "domain": top_domain,
            "modality": top_modality,
            "language": top_lang,
            "languages": [l for l, _ in band_languages[band].most_common(3)],
            "sources": sorted(band_sources[band]),
        }

        if final_blocks:
            band_writers[band].add_blocks(final_blocks, shard_meta)

        # Flush any remaining pending blocks in the writer
        band_writers[band].flush_remaining(shard_meta)

        w = band_writers[band]
        log(
            f"  {band}: {band_doc_counts[band]:,} docs → "
            f"{w.blocks_written:,} blocks → {w.shards_written} shards"
            + (f" ({w.shards_skipped} skipped)" if w.shards_skipped else "")
        )

    # ── Summary ────────────────────────────────────────────────────────────
    total_shards = sum(w.shards_written for w in band_writers.values())
    total_skipped = sum(w.shards_skipped for w in band_writers.values())
    total_blocks = sum(w.blocks_written for w in band_writers.values())
    t_total = time.time() - t_start

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Time:              {t_total:.1f}s")
    print(f"  Files processed:   {total_stats['files_processed']:,}")
    print(f"  Files errored:     {total_stats['files_errored']:,}")
    print(f"  Rows read:         {total_stats['rows_read']:,}")
    print(f"  Rows cleaned:      {total_stats['rows_cleaned']:,}")
    dropped = total_stats["rows_dropped_empty"] + total_stats["rows_dropped_short"]
    print(
        f"  Rows dropped:      {dropped:,} "
        f"(empty={total_stats['rows_dropped_empty']:,}, "
        f"short={total_stats['rows_dropped_short']:,})"
    )
    print(f"  Total chars:       {total_stats['total_chars']:,}")
    print(f"  Total tokens:      {total_stats['total_tokens']:,}")
    print(
        f"  Blocks written:    {total_blocks:,} ({total_blocks * BLOCK_SIZE:,} tokens)"
    )
    print(f"  Shards written:    {total_shards:,}")
    if total_skipped:
        print(f"  Shards skipped:    {total_skipped:,} (resume)")
    print(f"  Tail discarded:    {total_tail_discarded:,} tokens")
    print(f"  Output dir:        {output_dir}")
    if t_total > 0:
        print(
            f"  Throughput:        "
            f"{total_stats['total_chars'] / t_total / 1e6:.1f}M chars/s, "
            f"{total_stats['total_tokens'] / t_total / 1e6:.2f}M tok/s"
        )

    # Cleaning stats
    print("\n  Cleaning stats:")
    print(total_cleaning.summary())

    print("=" * 70)

    # ── Manifest ───────────────────────────────────────────────────────────
    manifest = {
        "pipeline_version": "2.0",
        "tokenizer_hash": tok_hash,
        "tokenizer_dir": tokenizer_dir,
        "block_size": BLOCK_SIZE,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "total_blocks": total_blocks,
        "total_shards": total_shards,
        "total_tokens": total_blocks * BLOCK_SIZE,
        "total_docs": total_stats["rows_cleaned"],
        "bands": {
            band: {
                "docs": band_doc_counts[band],
                "sources": sorted(band_sources.get(band, set())),
                "blocks": band_writers[band].blocks_written,
                "languages": [l for l, _ in band_languages[band].most_common(5)],
                "domains": [d for d, _ in band_domains[band].most_common(5)],
            }
            for band in sorted(band_writers.keys())
        },
        "stats": dict(total_stats),
        "elapsed_seconds": round(t_total, 1),
    }
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    # ── Band audit (Phase 1 — stats only, no thresholds) ──────────────────
    per_source_stats: Dict[str, Dict[str, Any]] = {}
    for src, sc in sorted(source_complexity.items()):
        n = sc["doc_count"]
        if n == 0:
            continue
        # Approximate median from per-file medians
        file_medians = sorted(sc["char_lengths"])
        approx_median = file_medians[len(file_medians) // 2] if file_medians else 0
        per_source_stats[src] = {
            "band": sc["band"],
            "domain": sc["domain"],
            "stats": {
                "doc_count": n,
                "median_doc_chars": approx_median,
                "mean_char_bigram_entropy": round(sc["entropy_sum"] / n, 3),
                "pct_instruction_stripped": round(sc["instruction_count"] / n, 4),
                "mean_code_density": round(sc["code_density_sum"] / n, 4),
                "mean_math_density": round(sc["math_density_sum"] / n, 6),
            },
        }

    band_audit = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pipeline_version": "2.0",
        "total_sources": len(per_source_stats),
        "per_source_stats": per_source_stats,
        "domain_policy_violations": domain_violations,
    }
    audit_path = os.path.join(output_dir, "band_audit.json")
    with open(audit_path, "w") as f:
        json.dump(band_audit, f, indent=2)
    print(f"Band audit written to {audit_path}")

    if per_source_stats:
        print(f"\n  Complexity stats summary ({len(per_source_stats)} sources):")
        print(
            f"  {'Source':<30} {'Band':>4}  {'Docs':>8}  {'MedCh':>6}  "
            f"{'Entropy':>7}  {'Inst%':>5}  {'Code':>5}  {'Math':>6}"
        )
        print("  " + "-" * 82)
        for src, info in sorted(per_source_stats.items()):
            s = info["stats"]
            print(
                f"  {src:<30} {info['band']:>4}  {s['doc_count']:>8,}  "
                f"{s['median_doc_chars']:>6,}  {s['mean_char_bigram_entropy']:>7.2f}  "
                f"{s['pct_instruction_stripped']:>5.1%}  "
                f"{s['mean_code_density']:>5.3f}  {s['mean_math_density']:>6.4f}"
            )

    if domain_violations:
        print(f"\n  {len(domain_violations)} band_domain_policy violation(s):")
        for v in domain_violations:
            print(f"    {v['source']}: {v['violation']}")

    # ── Optional post-processing verification ─────────────────────────────
    if verify_after:
        print("\n" + "=" * 70)
        print("RUNNING POST-PROCESSING VERIFICATION")
        print("=" * 70)
        from verify import main as verify_main

        sys.argv = [
            "verify.py",
            "--shard-dir",
            output_dir,
            "--tokenizer-dir",
            tokenizer_dir,
        ]
        try:
            verify_main()
        except SystemExit:
            pass  # verify_main calls sys.exit


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process parquet data into bin_idx training shards"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing source=<NAME>/*.parquet files",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for shards"
    )
    parser.add_argument(
        "--tokenizer-dir",
        required=True,
        help="Directory containing tokenizer.json + special_tokens_map.json",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated source names to process (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--shard-max-blocks",
        type=int,
        default=DEFAULT_SHARD_MAX_BLOCKS,
        help=f"Max blocks per shard (default: {DEFAULT_SHARD_MAX_BLOCKS})",
    )
    parser.add_argument(
        "--band-map", type=str, default=None, help="Path to band_map.json override file"
    )
    parser.add_argument(
        "--allow-unknown-band",
        action="store_true",
        help="Default unknown sources to B1 instead of raising ValueError",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover files + print source→band table + exit (no processing)",
    )
    parser.add_argument(
        "--verify-after",
        action="store_true",
        help="Run verify.py on output after processing completes",
    )
    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_dir=args.tokenizer_dir,
        sources=[s.strip() for s in args.sources.split(",")] if args.sources else None,
        workers=args.workers,
        shard_max_blocks=args.shard_max_blocks,
        band_map_path=args.band_map,
        allow_unknown_band=args.allow_unknown_band,
        dry_run=args.dry_run,
        verify_after=args.verify_after,
    )


if __name__ == "__main__":
    main()
