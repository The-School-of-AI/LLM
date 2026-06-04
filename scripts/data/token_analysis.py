#!/usr/bin/env python3
"""
Token Frequency & Band Analysis — Exhaustive Edition v2
=======================================================
30 metrics (18 original + 8 from review + 4 free additions).

Speed optimizations:
  - ord()-based Unicode script classification (no unicodedata.name() lookups)
  - Vectorized n-gram hashing via matrix multiply
  - Pre-computed coefficients

Metrics 1-18: freq, coverage, entropy, curves, buckets, top/bottom, doc lengths,
  richness, cross-band KL/JS, differential, unseen, bigrams, script breakdown,
  special tokens, fertility, Zipf, fragmentation, repetition rate
Metrics 19-26: position bias, sequence length entropy, bigram PMI, merge depth,
  fertility by script, cross-doc leakage, sentence boundaries, garbage rate
Metrics 27-30: numeric token dist, whitespace token dist, char-length histogram,
  Jaccard overlap (cross-band)

Usage:
  python3 token_analysis.py \\
    --source s3://t1-dataacquisition-datasets-2/shards \\
    --workers 16 \\
    --tokenizer-dir /mnt/nvme0/FINAL_TOKENIZER \\
    --output /mnt/nvme0/token_analysis_full.json \\
    --save-freq-npy

  # Quick test
  python3 token_analysis.py --source s3://... --max-shards-per-band 3 --workers 4 \\
    --tokenizer-dir /mnt/nvme0/FINAL_TOKENIZER
"""

import json
import os
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from numpy.lib.stride_tricks import sliding_window_view as _swv

    _HAS_SWV = True
except ImportError:
    _HAS_SWV = False

# ─── Config ───────────────────────────────────────────────────────────────────
VOCAB_SIZE = 131_072
EOS_ID = 130_717
PAD_ID = 130_718
BOS_ID = 130_716
TOOL_TOKEN_IDS = [130_815, 130_816, 130_817, 130_818]
SPECIAL_IDS = {
    "bos": BOS_ID,
    "eos": EOS_ID,
    "pad": PAD_ID,
    "tool_call_open": 130_815,
    "tool_call_close": 130_816,
    "tool_response_open": 130_817,
    "tool_response_close": 130_818,
}

DOC_LEN_BINS = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 1_000_000]
DOC_LEN_LABELS = [
    "0-32",
    "32-64",
    "64-128",
    "128-256",
    "256-512",
    "512-1K",
    "1K-2K",
    "2K-4K",
    "4K-8K",
    "8K+",
]

RICHNESS_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001]
RICHNESS_LABELS = [
    "0-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
]

SCRIPT_CATEGORIES = [
    "latin",
    "devanagari",
    "bengali",
    "tamil",
    "telugu",
    "gujarati",
    "kannada",
    "malayalam",
    "oriya",
    "gurmukhi",
    "sinhala",
    "cjk",
    "cyrillic",
    "arabic",
    "hebrew",
    "greek",
    "thai",
    "digits",
    "punctuation",
    "whitespace",
    "other",
]
SCRIPT_CAT_TO_IDX = {s: i for i, s in enumerate(SCRIPT_CATEGORIES)}
N_SCRIPTS = len(SCRIPT_CATEGORIES)

BIGRAM_TOP_K = 2000
REPETITION_NGRAM = 4
POSITION_WINDOW = 32  # tokens at doc start/end for position bias
SENTENCE_END_CHARS = frozenset(".!?।。！？؟᠃")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Bigram key shift — vocab is 131072 = 2^17, use 18-bit shift for safety
BIGRAM_SHIFT = 18
BIGRAM_MASK = (1 << BIGRAM_SHIFT) - 1  # 0x3FFFF

# Pre-computed n-gram hash coefficients
_NGRAM_COEFFS = np.array([131071**k for k in range(REPETITION_NGRAM)], dtype=np.int64)
assert (
    131071 ** (REPETITION_NGRAM - 1) < np.iinfo(np.int64).max
), f"REPETITION_NGRAM={REPETITION_NGRAM} causes int64 overflow in hash coefficients"

# ─── Module-level lookup tables (set before forking workers) ──────────────────
_TOKEN_TEXTS = None
_TOKEN_CHAR_LENS = None
_TOKEN_SCRIPT = None
_TOKEN_IS_SPACE_PREFIX = None
_TOKEN_IS_SPECIAL = None
_TOKEN_MERGE_DEPTH = None
_TOKEN_IS_GARBAGE = None
_TOKEN_IS_SENT_END = None
_TOKEN_IS_NUMERIC = None  # token text is purely digits
_TOKEN_IS_WHITESPACE_ONLY = None  # token text is purely whitespace (\n, \t, spaces)
_TOKEN_IS_ALPHA_CONT = None  # non-space-prefix AND starts with alpha (mid-word piece)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── Script Classification (fast, ord()-based) ───────────────────────────────


def _classify_char_script(ch):
    """Classify a character into a script category using ord() ranges."""
    cat = unicodedata.category(ch)
    if cat[0] == "Z":
        return "whitespace"
    if cat[0] in ("P", "S"):
        return "punctuation"
    if cat[0] == "N":
        return "digits"

    cp = ord(ch)
    if (
        (0x0041 <= cp <= 0x024F)
        or (0x1E00 <= cp <= 0x1EFF)
        or (0x2C60 <= cp <= 0x2C7F)
        or (0xA720 <= cp <= 0xA7FF)
    ):
        return "latin"
    if 0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF:
        return "devanagari"
    if 0x0980 <= cp <= 0x09FF:
        return "bengali"
    if 0x0A00 <= cp <= 0x0A7F:
        return "gurmukhi"
    if 0x0A80 <= cp <= 0x0AFF:
        return "gujarati"
    if 0x0B00 <= cp <= 0x0B7F:
        return "oriya"
    if 0x0B80 <= cp <= 0x0BFF:
        return "tamil"
    if 0x0C00 <= cp <= 0x0C7F:
        return "telugu"
    if 0x0C80 <= cp <= 0x0CFF:
        return "kannada"
    if 0x0D00 <= cp <= 0x0D7F:
        return "malayalam"
    if 0x0D80 <= cp <= 0x0DFF:
        return "sinhala"
    if 0x0E00 <= cp <= 0x0E7F:
        return "thai"
    if 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF:
        return "greek"
    if 0x0400 <= cp <= 0x052F:
        return "cyrillic"
    if (
        0x0600 <= cp <= 0x06FF
        or 0x0750 <= cp <= 0x077F
        or 0xFB50 <= cp <= 0xFDFF
        or 0xFE70 <= cp <= 0xFEFF
    ):
        return "arabic"
    if 0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F:
        return "hebrew"
    if (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x3040 <= cp <= 0x309F)
        or (0x30A0 <= cp <= 0x30FF)
        or (0xAC00 <= cp <= 0xD7AF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x20000 <= cp <= 0x2A6DF)
    ):
        return "cjk"
    return "other"


def _dominant_script(text):
    """Get the dominant script category of a text string."""
    if not text:
        return "other"
    counts = Counter()
    for ch in text:
        sc = _classify_char_script(ch)
        if sc not in ("whitespace", "punctuation", "digits"):
            counts[sc] += 1
    if not counts:
        for ch in text:
            sc = _classify_char_script(ch)
            if sc != "whitespace":
                return sc
        return "whitespace"
    return counts.most_common(1)[0][0]


# ─── Tokenizer Lookup Tables ─────────────────────────────────────────────────


def build_lookup_tables(tokenizer_dir):
    """Build per-token lookup tables from the tokenizer. Must be called before forking."""
    global _TOKEN_TEXTS, _TOKEN_CHAR_LENS, _TOKEN_SCRIPT
    global _TOKEN_IS_SPACE_PREFIX, _TOKEN_IS_SPECIAL
    global _TOKEN_MERGE_DEPTH, _TOKEN_IS_GARBAGE, _TOKEN_IS_SENT_END
    global _TOKEN_IS_NUMERIC, _TOKEN_IS_WHITESPACE_ONLY, _TOKEN_IS_ALPHA_CONT

    log(f"Building token lookup tables from {tokenizer_dir}...")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(f"{tokenizer_dir}/tokenizer.json")

    texts = []
    char_lens = np.zeros(VOCAB_SIZE, dtype=np.int32)
    scripts = np.zeros(VOCAB_SIZE, dtype=np.int8)
    is_space = np.zeros(VOCAB_SIZE, dtype=bool)
    is_special = np.zeros(VOCAB_SIZE, dtype=bool)
    is_garbage = np.zeros(VOCAB_SIZE, dtype=bool)
    is_sent_end = np.zeros(VOCAB_SIZE, dtype=bool)
    is_numeric = np.zeros(VOCAB_SIZE, dtype=bool)
    is_ws_only = np.zeros(VOCAB_SIZE, dtype=bool)
    is_alpha_cont = np.zeros(
        VOCAB_SIZE, dtype=bool
    )  # non-space-prefix AND starts with letter

    for sid in SPECIAL_IDS.values():
        if sid < VOCAB_SIZE:
            is_special[sid] = True
    for tid in TOOL_TOKEN_IDS:
        if tid < VOCAB_SIZE:
            is_special[tid] = True

    for tid in range(VOCAB_SIZE):
        try:
            text = tok.decode([tid])
        except Exception:
            text = ""
        texts.append(text)
        char_lens[tid] = len(text)

        if text and (text[0] == " " or text[0] == "\u0120"):
            is_space[tid] = True

        if is_special[tid]:
            scripts[tid] = SCRIPT_CAT_TO_IDX.get("other", N_SCRIPTS - 1)
        else:
            sc = _dominant_script(text)
            scripts[tid] = SCRIPT_CAT_TO_IDX.get(sc, N_SCRIPTS - 1)

        # 26. Garbage detection
        if not is_special[tid]:
            if not text or not text.strip():
                is_garbage[tid] = True
            else:
                stripped = "".join(c for c in text if c not in (" ", "\t", "\n"))
                if stripped and all(
                    unicodedata.category(c)[0] == "C"
                    or c in ("\ufffd", "\ufffe", "\uffff")
                    for c in stripped
                ):
                    is_garbage[tid] = True

        # 25. Sentence boundary detection
        # Only count tokens that end with sentence-ending punctuation AND are
        # either pure punctuation (len 1-3) or end with punct after a letter.
        # Avoids false positives on code tokens like ".end", "U.S.", etc.
        stripped_r = text.rstrip()
        if stripped_r and stripped_r[-1] in SENTENCE_END_CHARS:
            clean = stripped_r.lstrip()
            # Pure punctuation tokens (e.g. ".", "?", "!", "。")
            if len(clean) <= 3 and all(
                unicodedata.category(c)[0] in ("P", "S", "Z") or c in SENTENCE_END_CHARS
                for c in clean
            ):
                is_sent_end[tid] = True
            # Tokens ending in sentence punct after a letter (e.g. "word.")
            elif len(clean) > 1 and clean[-2].isalpha():
                is_sent_end[tid] = True

        # 27. Numeric token detection
        if text and text.strip() and text.strip().isdigit():
            is_numeric[tid] = True

        # 28. Whitespace-only token detection
        if text and not text.strip():
            is_ws_only[tid] = True

        # Alpha continuation: non-space-prefix token whose first non-whitespace
        # char is alphabetic.  These are mid-word subword pieces like "tion",
        # "ing", "er".  Punctuation / digits / code operators are NOT counted.
        if not is_space[tid] and not is_special[tid] and text:
            first_alpha = text.lstrip()
            if first_alpha and first_alpha[0].isalpha():
                is_alpha_cont[tid] = True

        if tid % 20000 == 0 and tid > 0:
            log(f"  ... {tid:,}/{VOCAB_SIZE:,} tokens processed")

    _TOKEN_TEXTS = texts
    _TOKEN_CHAR_LENS = char_lens
    _TOKEN_SCRIPT = scripts
    _TOKEN_IS_SPACE_PREFIX = is_space
    _TOKEN_IS_SPECIAL = is_special
    _TOKEN_IS_GARBAGE = is_garbage
    _TOKEN_IS_SENT_END = is_sent_end
    _TOKEN_IS_NUMERIC = is_numeric
    _TOKEN_IS_WHITESPACE_ONLY = is_ws_only
    _TOKEN_IS_ALPHA_CONT = is_alpha_cont

    # ── 22. BPE merge depth ──
    merge_depth = np.zeros(VOCAB_SIZE, dtype=np.int16)
    try:
        with open(f"{tokenizer_dir}/tokenizer.json") as f:
            tok_data = json.load(f)
        vocab_map = tok_data.get("model", {}).get("vocab", {})
        merges = tok_data.get("model", {}).get("merges", [])

        if merges:
            depths = {}
            for token_str in vocab_map:
                depths[token_str] = 0
            for merge_str in merges:
                if isinstance(merge_str, list):
                    if len(merge_str) != 2:
                        continue
                    left, right = merge_str
                else:
                    parts = merge_str.split(" ", 1)
                    if len(parts) != 2:
                        continue
                    left, right = parts
                merged_str = left + right
                d_left = depths.get(left, 0)
                d_right = depths.get(right, 0)
                depths[merged_str] = max(d_left, d_right) + 1

            for token_str, tid in vocab_map.items():
                if isinstance(tid, int) and tid < VOCAB_SIZE:
                    merge_depth[tid] = depths.get(token_str, 0)

            nonzero_depths = merge_depth[merge_depth > 0]
            log(
                f"  Merge depths: max={int(merge_depth.max())}, "
                f"mean={float(nonzero_depths.mean()):.1f}, "
                f"tokens_with_merges={len(nonzero_depths):,}"
            )
        else:
            log("  No merges found in tokenizer.json — merge depth disabled")
    except Exception as e:
        log(f"  Warning: Could not compute merge depths: {e}")

    _TOKEN_MERGE_DEPTH = merge_depth

    log(f"  Lookup tables built for {VOCAB_SIZE:,} tokens")
    log(f"  Space-prefixed: {int(is_space.sum()):,}")
    log(f"  Special: {int(is_special.sum()):,}")
    log(f"  Garbage: {int(is_garbage.sum()):,}")
    log(f"  Sentence-ending: {int(is_sent_end.sum()):,}")
    log(f"  Numeric: {int(is_numeric.sum()):,}")
    log(f"  Whitespace-only: {int(is_ws_only.sum()):,}")
    n_by_script = Counter()
    for i in range(VOCAB_SIZE):
        n_by_script[SCRIPT_CATEGORIES[scripts[i]]] += 1
    for sc in sorted(n_by_script, key=n_by_script.get, reverse=True)[:10]:
        log(f"    {sc}: {n_by_script[sc]:,} tokens")


# ─── Shard Discovery ─────────────────────────────────────────────────────────


def list_shards_s3(bucket_url):
    """List shards from S3, grouped by band."""
    log(f"Listing shards from {bucket_url} ...")
    cmd = ["aws", "s3", "ls", f"{bucket_url}/", "--recursive"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    url_parts = bucket_url.replace("s3://", "").split("/")
    bucket_root = f"s3://{url_parts[0]}"

    bands = defaultdict(list)
    for line in result.stdout.strip().split("\n"):
        if not line or "tokens.bin" not in line:
            continue
        path = line.strip().split()[-1]
        parts = path.split("/")
        band = None
        for p in parts:
            if p.startswith("band_"):
                band = p
                break
        if not band:
            continue
        shard_prefix = "/".join(parts[:-1])
        bands[band].append(f"{bucket_root}/{shard_prefix}")

    for b in sorted(bands):
        log(f"  {b}: {len(bands[b])} shards")
    return dict(bands)


def list_shards_local(base_dir):
    """List shards from local directory, grouped by band."""
    log(f"Listing shards from {base_dir} ...")
    bands = defaultdict(list)
    base = Path(base_dir)

    for band_dir in sorted(base.iterdir()):
        if not band_dir.is_dir() or not band_dir.name.startswith("band_"):
            continue
        for shard_dir in sorted(band_dir.iterdir()):
            if (shard_dir / "tokens.bin").exists():
                bands[band_dir.name].append(str(shard_dir))

    for b in sorted(bands):
        log(f"  {b}: {len(bands[b])} shards")
    return dict(bands)


# ─── Shard Processing ────────────────────────────────────────────────────────


def analyze_shard(args):
    """Analyze a single shard: all 26 metrics."""
    shard_path, is_s3, band = args

    try:
        # ── Read tokens ──
        if is_s3:
            cmd = ["aws", "s3", "cp", f"{shard_path}/tokens.bin", "-"]
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                return {
                    "band": band,
                    "ok": False,
                    "error": f"S3 download failed: {result.stderr[:200]}",
                    "shard": shard_path,
                }
            tokens = np.frombuffer(result.stdout, dtype=np.uint32)
        else:
            tokens = np.fromfile(f"{shard_path}/tokens.bin", dtype=np.uint32)

        n_total = len(tokens)

        # ── 1. Token frequency ──
        freq = np.bincount(tokens, minlength=VOCAB_SIZE).astype(np.int64)
        n_pad = int(freq[PAD_ID])
        n_eos = int(freq[EOS_ID])
        n_content = n_total - n_pad - n_eos

        # ── Content tokens (exclude PAD and EOS) ──
        content_tokens = tokens[(tokens != PAD_ID) & (tokens != EOS_ID)]

        # ── 12. Bigram analysis ──
        content_no_pad = tokens[tokens != PAD_ID]
        n_total_bigrams = 0
        if len(content_no_pad) > 1:
            t1 = content_no_pad[:-1].astype(np.int64)
            t2 = content_no_pad[1:].astype(np.int64)
            eos_mask = (t1 != EOS_ID) & (t2 != EOS_ID)
            t1 = t1[eos_mask]
            t2 = t2[eos_mask]
            bigram_keys = (t1 << BIGRAM_SHIFT) | t2
            unique_bigrams, bigram_counts = np.unique(bigram_keys, return_counts=True)
            n_total_bigrams = int(bigram_counts.sum())
            n_unique_bigrams = len(unique_bigrams)  # save before pruning
            if len(unique_bigrams) > BIGRAM_TOP_K:
                top_idx = np.argpartition(-bigram_counts, BIGRAM_TOP_K)[:BIGRAM_TOP_K]
                unique_bigrams = unique_bigrams[top_idx]
                bigram_counts = bigram_counts[top_idx]
            top_bigrams = list(zip(unique_bigrams.tolist(), bigram_counts.tolist()))
        else:
            top_bigrams = []
            n_unique_bigrams = 0

        # ── 13. Script/language breakdown ──
        if _TOKEN_SCRIPT is not None and len(content_tokens) > 0:
            script_ids = _TOKEN_SCRIPT[content_tokens]
            script_counts = np.bincount(script_ids, minlength=N_SCRIPTS).tolist()
        else:
            script_counts = [0] * N_SCRIPTS

        # ── 14. Special token counts ──
        special_counts = {}
        for name, sid in SPECIAL_IDS.items():
            special_counts[name] = int(freq[sid]) if sid < VOCAB_SIZE else 0

        # ── 15. Token fertility ──
        total_chars = 0
        n_content_tokens_for_fertility = len(content_tokens)
        if _TOKEN_CHAR_LENS is not None and len(content_tokens) > 0:
            total_chars = int(_TOKEN_CHAR_LENS[content_tokens].sum())

        # ── 17. Subword fragmentation ──
        n_words = 0
        frag_hist = [0, 0, 0, 0, 0, 0]
        avg_fragments = 0.0
        if _TOKEN_IS_SPACE_PREFIX is not None and len(content_tokens) > 0:
            is_word_start = _TOKEN_IS_SPACE_PREFIX[content_tokens]
            n_words = int(is_word_start.sum())
            word_starts = np.where(is_word_start)[0]
            if len(word_starts) > 1:
                word_lens = np.diff(word_starts)
                word_lens = np.append(word_lens, len(content_tokens) - word_starts[-1])
                frag_bins = [1, 2, 3, 4, 5, 6, 1000]
                frag_hist, _ = np.histogram(word_lens, bins=frag_bins)
                frag_hist = frag_hist.tolist()
                avg_fragments = float(np.mean(word_lens))

        # ── 23. Fertility by script ──
        fertility_by_script = {}
        if (
            _TOKEN_SCRIPT is not None
            and _TOKEN_CHAR_LENS is not None
            and len(content_tokens) > 0
        ):
            ct_scripts = _TOKEN_SCRIPT[content_tokens]
            ct_charlens = _TOKEN_CHAR_LENS[content_tokens]
            for sc_idx, sc_name in enumerate(SCRIPT_CATEGORIES):
                if sc_name in ("whitespace", "punctuation", "digits", "other"):
                    continue
                mask = ct_scripts == sc_idx
                n_sc = int(mask.sum())
                if n_sc > 100:
                    chars_sc = int(ct_charlens[mask].sum())
                    fertility_by_script[sc_name] = {"tokens": n_sc, "chars": chars_sc}

        # ── 25. Sentence boundary tokens ──
        n_sent_end_tokens = 0
        if _TOKEN_IS_SENT_END is not None and len(content_tokens) > 0:
            n_sent_end_tokens = int(_TOKEN_IS_SENT_END[content_tokens].sum())

        # ── 26. Garbage token count ──
        n_garbage_tokens = 0
        if _TOKEN_IS_GARBAGE is not None and len(content_tokens) > 0:
            n_garbage_tokens = int(_TOKEN_IS_GARBAGE[content_tokens].sum())

        # ── 27. Numeric token count ──
        n_numeric_tokens = 0
        if _TOKEN_IS_NUMERIC is not None and len(content_tokens) > 0:
            n_numeric_tokens = int(_TOKEN_IS_NUMERIC[content_tokens].sum())

        # ── 28. Whitespace token count ──
        n_whitespace_tokens = 0
        if _TOKEN_IS_WHITESPACE_ONLY is not None and len(content_tokens) > 0:
            n_whitespace_tokens = int(_TOKEN_IS_WHITESPACE_ONLY[content_tokens].sum())

        # ── 29. Token char-length histogram ──
        char_len_hist = []
        if _TOKEN_CHAR_LENS is not None and len(content_tokens) > 0:
            cl = _TOKEN_CHAR_LENS[content_tokens]
            # Bins: 1, 2, 3, 4, 5, 6-10, 11-20, 21+
            char_len_bins = [1, 2, 3, 4, 5, 6, 11, 21, 1001]
            char_len_hist, _ = np.histogram(cl, bins=char_len_bins)
            char_len_hist = char_len_hist.tolist()

        # ── Document-level analysis ──
        eos_positions = np.where(tokens == EOS_ID)[0]

        doc_lengths = []
        doc_unique_counts = []
        total_repeated_ngrams = 0
        total_ngrams_checked = 0

        # 19. Position bias accumulators
        intro_freq = np.zeros(VOCAB_SIZE, dtype=np.int64)
        outro_freq = np.zeros(VOCAB_SIZE, dtype=np.int64)

        # 24. Cross-doc leakage counters
        n_clean_boundaries = 0
        n_leaky_boundaries = 0
        prev_doc_last_token = None  # track last token of previous doc

        prev = 0
        for doc_idx, eos_pos in enumerate(eos_positions):
            doc = tokens[prev:eos_pos]
            doc = doc[doc != PAD_ID]
            doc_len = len(doc)

            if doc_len > 0:
                doc_lengths.append(doc_len)
                doc_unique_counts.append(len(np.unique(doc)))

                # ── 18. Repetition rate (vectorized n-gram hashing) ──
                if doc_len >= REPETITION_NGRAM:
                    ngram_count = doc_len - REPETITION_NGRAM + 1
                    total_ngrams_checked += ngram_count
                    d64 = doc.astype(np.int64)
                    if _HAS_SWV:
                        windows = _swv(d64, REPETITION_NGRAM)
                        hashes = windows @ _NGRAM_COEFFS
                    else:
                        hashes = np.zeros(ngram_count, dtype=np.int64)
                        for k in range(REPETITION_NGRAM):
                            hashes += d64[k : k + ngram_count] * _NGRAM_COEFFS[k]
                    n_unique_ng = len(np.unique(hashes))
                    total_repeated_ngrams += ngram_count - n_unique_ng

                # ── 19. Position bias ──
                intro_end = min(doc_len, POSITION_WINDOW)
                outro_start = max(0, doc_len - POSITION_WINDOW)
                intro_freq += np.bincount(
                    doc[:intro_end].astype(np.int64), minlength=VOCAB_SIZE
                )
                outro_freq += np.bincount(
                    doc[outro_start:].astype(np.int64), minlength=VOCAB_SIZE
                )

                # ── 24. Cross-doc leakage ──
                # Leakage = prev doc ends with an alpha continuation piece AND
                # current doc starts with an alpha continuation piece, meaning a
                # word was split across the EOS boundary (e.g. "comput" EOS "er").
                # Punctuation/digits/operators ending one doc and starting the
                # next is normal and NOT leakage.
                if _TOKEN_IS_ALPHA_CONT is not None and prev_doc_last_token is not None:
                    if (
                        _TOKEN_IS_ALPHA_CONT[prev_doc_last_token]
                        and _TOKEN_IS_ALPHA_CONT[doc[0]]
                    ):
                        n_leaky_boundaries += 1
                    else:
                        n_clean_boundaries += 1

                prev_doc_last_token = doc[-1]

            prev = int(eos_pos) + 1

        # Document length histogram
        len_hist, _ = np.histogram(doc_lengths, bins=DOC_LEN_BINS)

        # Vocabulary richness histogram
        if doc_lengths:
            richness = [u / l for u, l in zip(doc_unique_counts, doc_lengths)]
            richness_hist, _ = np.histogram(richness, bins=RICHNESS_BINS)
        else:
            richness_hist = np.zeros(len(RICHNESS_BINS) - 1, dtype=np.int64)

        return {
            "band": band,
            "freq": freq,
            "n_total": n_total,
            "n_content": n_content,
            "n_pad": n_pad,
            "n_eos": n_eos,
            "n_docs": len(doc_lengths),
            "doc_len_hist": len_hist.tolist(),
            "richness_hist": richness_hist.tolist(),
            "doc_len_sum": int(sum(doc_lengths)),
            "doc_unique_sum": int(sum(doc_unique_counts)),
            "top_bigrams": top_bigrams,
            "n_unique_bigrams": n_unique_bigrams,
            "n_total_bigrams": n_total_bigrams,
            "script_counts": script_counts,
            "special_counts": special_counts,
            "total_chars": total_chars,
            "n_content_tokens_fertility": n_content_tokens_for_fertility,
            "n_words": n_words,
            "frag_hist": frag_hist,
            "avg_fragments_per_word": avg_fragments,
            "total_repeated_ngrams": total_repeated_ngrams,
            "total_ngrams_checked": total_ngrams_checked,
            # New v2 metrics
            "intro_freq": intro_freq,
            "outro_freq": outro_freq,
            "fertility_by_script": fertility_by_script,
            "n_sent_end_tokens": n_sent_end_tokens,
            "n_garbage_tokens": n_garbage_tokens,
            "n_numeric_tokens": n_numeric_tokens,
            "n_whitespace_tokens": n_whitespace_tokens,
            "char_len_hist": char_len_hist,
            "n_clean_boundaries": n_clean_boundaries,
            "n_leaky_boundaries": n_leaky_boundaries,
            "ok": True,
        }
    except Exception:
        import traceback

        return {
            "band": band,
            "ok": False,
            "error": traceback.format_exc()[-500:],
            "shard": shard_path,
        }


# ─── Per-Band Report ─────────────────────────────────────────────────────────


def compute_band_report(band, freq, stats):
    """Compute comprehensive stats for a single band (all 26 metrics)."""
    total_tokens = int(freq.sum())
    n_pad = int(freq[PAD_ID])
    n_eos = int(freq[EOS_ID])

    content_freq = freq.copy()
    content_freq[PAD_ID] = 0
    content_freq[EOS_ID] = 0
    n_content = int(content_freq.sum())

    # ── 2. Vocab coverage ──
    n_seen = int(np.count_nonzero(freq))
    n_content_seen = int(np.count_nonzero(content_freq))
    n_unseen = VOCAB_SIZE - n_seen

    # ── 3. Token entropy ──
    total_c = content_freq.sum()
    if total_c > 0:
        probs = content_freq[content_freq > 0].astype(np.float64) / total_c
        entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        entropy = 0.0

    # ── 6. Top/bottom 100 tokens ──
    sorted_ids = np.argsort(-content_freq)
    top100 = [(int(i), int(content_freq[i])) for i in sorted_ids[:100]]
    seen_ids = np.where(content_freq > 0)[0]
    if len(seen_ids) > 0:
        rare_order = np.argsort(content_freq[seen_ids])
        bottom100_ids = seen_ids[rare_order[:100]]
        bottom100 = [(int(i), int(content_freq[i])) for i in bottom100_ids]
    else:
        bottom100 = []

    # ── 5. Frequency buckets ──
    nonzero_freqs = content_freq[content_freq > 0]
    freq_percentiles = {}
    if len(nonzero_freqs) > 0:
        freq_percentiles = {
            f"p{p}": int(np.percentile(nonzero_freqs, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

    # ── 4. Coverage curves ──
    sorted_freqs = np.sort(content_freq)[::-1]
    cumsum = np.cumsum(sorted_freqs)
    total_c_val = cumsum[-1] if len(cumsum) > 0 else 1
    coverage = {}
    for pct in [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]:
        idx = int(np.searchsorted(cumsum, total_c_val * pct)) + 1
        coverage[f"{pct*100:.1f}%"] = idx

    freq_buckets = {
        "1-9": int(((content_freq >= 1) & (content_freq < 10)).sum()),
        "10-99": int(((content_freq >= 10) & (content_freq < 100)).sum()),
        "100-999": int(((content_freq >= 100) & (content_freq < 1000)).sum()),
        "1K-9K": int(((content_freq >= 1000) & (content_freq < 10000)).sum()),
        "10K-99K": int(((content_freq >= 10000) & (content_freq < 100000)).sum()),
        "100K-999K": int(((content_freq >= 100000) & (content_freq < 1000000)).sum()),
        "1M-9M": int(((content_freq >= 1000000) & (content_freq < 10000000)).sum()),
        "10M+": int((content_freq >= 10000000).sum()),
    }

    n_rare_lt10 = int(((content_freq > 0) & (content_freq < 10)).sum())
    n_rare_lt100 = int(((content_freq > 0) & (content_freq < 100)).sum())
    n_rare_lt1000 = int(((content_freq > 0) & (content_freq < 1000)).sum())

    # ── 16. Zipf's law fit ──
    zipf_result = {}
    if len(nonzero_freqs) > 100:
        sorted_nz = np.sort(nonzero_freqs)[::-1]
        ranks = np.arange(1, len(sorted_nz) + 1, dtype=np.float64)
        try:
            log_ranks = np.log(ranks)
            log_freqs = np.log(sorted_nz.astype(np.float64))
            coeffs = np.polyfit(log_ranks, log_freqs, 1)
            alpha = -coeffs[0]
            C = np.exp(coeffs[1])
            predicted = coeffs[0] * log_ranks + coeffs[1]
            ss_res = np.sum((log_freqs - predicted) ** 2)
            ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            zipf_result = {
                "alpha": round(float(alpha), 4),
                "C": round(float(C), 2),
                "r_squared": round(float(r_squared), 6),
                "interpretation": (
                    "natural_text"
                    if 0.9 < alpha < 1.3
                    else (
                        "curated_concentrated"
                        if alpha > 1.3
                        else "diverse_flat" if alpha < 0.9 else "unknown"
                    )
                ),
            }
        except Exception:
            zipf_result = {"error": "fit_failed"}

    # ── 14. Special token usage ──
    special_usage = {}
    for name, sid in SPECIAL_IDS.items():
        if sid < VOCAB_SIZE:
            special_usage[name] = int(freq[sid])

    # ── 13. Script breakdown ──
    script_breakdown = {}
    script_total = sum(stats.get("script_counts", []))
    if script_total > 0:
        for i, sc in enumerate(SCRIPT_CATEGORIES):
            cnt = (
                stats["script_counts"][i]
                if i < len(stats.get("script_counts", []))
                else 0
            )
            if cnt > 0:
                script_breakdown[sc] = {
                    "tokens": cnt,
                    "pct": round(100 * cnt / script_total, 2),
                }

    # ── 15. Fertility ──
    fertility = {}
    if stats.get("n_content_tokens_fertility", 0) > 0:
        chars_per_token = stats["total_chars"] / stats["n_content_tokens_fertility"]
        fertility = {
            "total_chars": stats["total_chars"],
            "total_content_tokens": stats["n_content_tokens_fertility"],
            "chars_per_token": round(chars_per_token, 4),
        }

    # ── 12. Top bigrams ──
    top_bigrams_report = []
    if stats.get("top_bigrams"):
        sorted_bg = sorted(stats["top_bigrams"], key=lambda x: x[1], reverse=True)[:100]
        for key, count in sorted_bg:
            t1 = int(key >> BIGRAM_SHIFT)
            t2 = int(key & BIGRAM_MASK)
            top_bigrams_report.append(
                {
                    "token1_id": t1,
                    "token2_id": t2,
                    "count": count,
                }
            )

    # ── 17. Fragmentation ──
    frag_labels = ["1-piece", "2-piece", "3-piece", "4-piece", "5-piece", "6+piece"]
    fragmentation = {}
    if stats.get("n_words", 0) > 0:
        fragmentation = {
            "total_words": stats["n_words"],
            "avg_subwords_per_word": round(stats["avg_fragments_per_word"], 4),
            "distribution": dict(zip(frag_labels, stats.get("frag_hist", []))),
        }

    # ── 18. Repetition ──
    repetition = {}
    if stats.get("total_ngrams_checked", 0) > 0:
        rep_rate = stats["total_repeated_ngrams"] / stats["total_ngrams_checked"]
        repetition = {
            "ngram_size": REPETITION_NGRAM,
            "total_ngrams": stats["total_ngrams_checked"],
            "repeated_ngrams": stats["total_repeated_ngrams"],
            "repetition_rate": round(rep_rate, 6),
        }

    # ── 19. Position bias ──
    position_bias = {}
    intro_f = stats.get("intro_freq")
    outro_f = stats.get("outro_freq")
    if intro_f is not None and outro_f is not None:
        overall_p = content_freq.astype(np.float64)
        overall_sum = overall_p.sum()
        if overall_sum > 0:
            overall_p = overall_p / overall_sum

        intro_sum = float(intro_f.sum())
        outro_sum = float(outro_f.sum())
        eps = 1e-12

        if intro_sum > 0:
            intro_p = intro_f.astype(np.float64) / intro_sum
            intro_ratio = np.log2((intro_p + eps) / (overall_p + eps))
            intro_top_ids = np.argsort(-intro_ratio)[:30]
            intro_top = []
            for tid in intro_top_ids:
                if intro_ratio[tid] > 1.0 and intro_f[tid] > 100:
                    intro_top.append(
                        {
                            "id": int(tid),
                            "log2_ratio": round(float(intro_ratio[tid]), 3),
                            "intro_count": int(intro_f[tid]),
                            "total_count": int(freq[tid]),
                        }
                    )
            position_bias["intro_overrepresented"] = intro_top[:10]

        if outro_sum > 0:
            outro_p = outro_f.astype(np.float64) / outro_sum
            outro_ratio = np.log2((outro_p + eps) / (overall_p + eps))
            outro_top_ids = np.argsort(-outro_ratio)[:30]
            outro_top = []
            for tid in outro_top_ids:
                if outro_ratio[tid] > 1.0 and outro_f[tid] > 100:
                    outro_top.append(
                        {
                            "id": int(tid),
                            "log2_ratio": round(float(outro_ratio[tid]), 3),
                            "outro_count": int(outro_f[tid]),
                            "total_count": int(freq[tid]),
                        }
                    )
            position_bias["outro_overrepresented"] = outro_top[:10]

        position_bias["intro_window"] = POSITION_WINDOW

    # ── 20. Sequence length entropy ──
    seq_len_entropy = 0.0
    doc_len_hist_arr = np.array(stats.get("doc_len_hist", []))
    total_docs_for_entropy = doc_len_hist_arr.sum()
    if total_docs_for_entropy > 0:
        p = (
            doc_len_hist_arr[doc_len_hist_arr > 0].astype(np.float64)
            / total_docs_for_entropy
        )
        seq_len_entropy = float(-np.sum(p * np.log2(p)))

    # ── 21. Bigram PMI ──
    bigram_pmi = []
    if stats.get("top_bigrams") and stats.get("n_total_bigrams", 0) > 0:
        total_uni = int(content_freq.sum())
        total_bi = stats["n_total_bigrams"]
        sorted_bg = sorted(stats["top_bigrams"], key=lambda x: x[1], reverse=True)[:200]
        for key, count in sorted_bg:
            t1 = int(key >> BIGRAM_SHIFT)
            t2 = int(key & BIGRAM_MASK)
            if t1 < VOCAB_SIZE and t2 < VOCAB_SIZE and total_uni > 0:
                p_xy = count / total_bi
                p_x = int(content_freq[t1]) / total_uni
                p_y = int(content_freq[t2]) / total_uni
                if p_x > 0 and p_y > 0:
                    pmi_val = float(np.log2(p_xy / (p_x * p_y)))
                    bigram_pmi.append(
                        {
                            "token1_id": t1,
                            "token2_id": t2,
                            "count": count,
                            "pmi": round(pmi_val, 4),
                        }
                    )
        bigram_pmi.sort(key=lambda x: x["pmi"], reverse=True)
        bigram_pmi = bigram_pmi[:50]

    # ── 22. Merge depth ──
    merge_depth_stats = {}
    if _TOKEN_MERGE_DEPTH is not None and n_content > 0:
        weighted_depth = float(
            np.sum(content_freq.astype(np.float64) * _TOKEN_MERGE_DEPTH)
        )
        avg_depth = weighted_depth / n_content
        depth_dist = {}
        max_d = int(_TOKEN_MERGE_DEPTH.max())
        for d in range(max_d + 1):
            mask = _TOKEN_MERGE_DEPTH == d
            count = int(content_freq[mask].sum())
            if count > 0:
                depth_dist[str(d)] = count
        merge_depth_stats = {
            "avg_weighted_merge_depth": round(avg_depth, 4),
            "max_merge_depth": max_d,
            "depth_distribution": depth_dist,
        }

    # ── 23. Fertility by script ──
    fertility_by_script_report = {}
    fbs = stats.get("fertility_by_script", {})
    for sc_name, sc_data in fbs.items():
        if sc_data["tokens"] > 0:
            fertility_by_script_report[sc_name] = {
                "tokens": sc_data["tokens"],
                "chars": sc_data["chars"],
                "chars_per_token": round(sc_data["chars"] / sc_data["tokens"], 4),
            }

    # ── 24. Cross-doc leakage ──
    leakage = {}
    n_clean = stats.get("n_clean_boundaries", 0)
    n_leaky = stats.get("n_leaky_boundaries", 0)
    if n_clean + n_leaky > 0:
        leakage = {
            "clean_boundaries": n_clean,
            "leaky_boundaries": n_leaky,
            "leakage_rate": round(n_leaky / (n_clean + n_leaky), 6),
        }

    # ── 25. Sentence boundary ──
    sentence_stats = {}
    n_sent = stats.get("n_sent_end_tokens", 0)
    n_docs_total = stats.get("n_docs", 0)
    if n_docs_total > 0:
        sentence_stats = {
            "total_sentence_end_tokens": n_sent,
            "avg_sentences_per_doc": round(n_sent / n_docs_total, 2),
            "avg_tokens_per_sentence": (
                round(n_content / n_sent, 1) if n_sent > 0 else 0
            ),
        }

    # ── 26. Garbage rate ──
    garbage_stats = {}
    n_garbage = stats.get("n_garbage_tokens", 0)
    if n_content > 0:
        garbage_stats = {
            "garbage_token_count": n_garbage,
            "garbage_rate": round(n_garbage / n_content, 8),
        }

    # ── 27. Numeric token distribution ──
    numeric_stats = {}
    n_numeric = stats.get("n_numeric_tokens", 0)
    if n_content > 0:
        numeric_stats = {
            "numeric_token_count": n_numeric,
            "numeric_rate": round(n_numeric / n_content, 6),
        }

    # ── 28. Whitespace token distribution ──
    whitespace_stats = {}
    n_ws = stats.get("n_whitespace_tokens", 0)
    if n_content > 0:
        whitespace_stats = {
            "whitespace_token_count": n_ws,
            "whitespace_rate": round(n_ws / n_content, 6),
        }

    # ── 29. Token char-length histogram ──
    char_len_labels = ["1", "2", "3", "4", "5", "6-10", "11-20", "21+"]
    char_len_dist = dict(zip(char_len_labels, stats.get("char_len_hist", [0] * 8)))

    return {
        "band": band,
        "total_tokens": total_tokens,
        "content_tokens": n_content,
        "pad_tokens": n_pad,
        "eos_tokens": n_eos,
        "pad_ratio": round(n_pad / total_tokens, 4) if total_tokens > 0 else 0,
        "vocab_total": VOCAB_SIZE,
        "vocab_seen": n_seen,
        "vocab_unseen": n_unseen,
        "vocab_content_seen": n_content_seen,
        "vocab_coverage_pct": round(100 * n_content_seen / VOCAB_SIZE, 2),
        "token_entropy_bits": round(entropy, 4),
        "max_possible_entropy_bits": round(float(np.log2(VOCAB_SIZE)), 4),
        "n_rare_lt10": n_rare_lt10,
        "n_rare_lt100": n_rare_lt100,
        "n_rare_lt1000": n_rare_lt1000,
        "freq_percentiles": freq_percentiles,
        "freq_buckets": freq_buckets,
        "coverage_tokens_needed": coverage,
        "top_100_tokens": top100,
        "bottom_100_tokens": bottom100,
        "n_docs": stats["n_docs"],
        "doc_len_hist": dict(zip(DOC_LEN_LABELS, stats["doc_len_hist"])),
        "richness_hist": dict(zip(RICHNESS_LABELS, stats["richness_hist"])),
        "avg_doc_len_tokens": (
            round(stats["doc_len_sum"] / stats["n_docs"], 1)
            if stats["n_docs"] > 0
            else 0
        ),
        "avg_doc_unique_tokens": (
            round(stats["doc_unique_sum"] / stats["n_docs"], 1)
            if stats["n_docs"] > 0
            else 0
        ),
        "avg_type_token_ratio": (
            round(stats["doc_unique_sum"] / stats["doc_len_sum"], 4)
            if stats["doc_len_sum"] > 0
            else 0
        ),
        "zipf_fit": zipf_result,
        "special_token_usage": special_usage,
        "script_breakdown": script_breakdown,
        "fertility": fertility,
        "top_100_bigrams": top_bigrams_report,
        "n_unique_bigrams_total": stats.get("n_unique_bigrams", 0),
        "fragmentation": fragmentation,
        "repetition": repetition,
        # New v2 metrics (19-26)
        "position_bias": position_bias,
        "sequence_length_entropy_bits": round(seq_len_entropy, 4),
        "top_50_bigram_pmi": bigram_pmi,
        "merge_depth": merge_depth_stats,
        "fertility_by_script": fertility_by_script_report,
        "cross_doc_leakage": leakage,
        "sentence_boundary": sentence_stats,
        "garbage_token_rate": garbage_stats,
        "numeric_token_dist": numeric_stats,
        "whitespace_token_dist": whitespace_stats,
        "char_length_histogram": char_len_dist,
    }


# ─── Cross-Band Comparison ───────────────────────────────────────────────────


def compute_cross_band(band_freqs):
    """KL divergence, JS divergence, unique/shared tokens between bands."""
    bands = sorted(band_freqs.keys())
    if len(bands) < 2:
        return {}

    dists = {}
    for b in bands:
        f = band_freqs[b].copy().astype(np.float64)
        f[PAD_ID] = 0
        f[EOS_ID] = 0
        total = f.sum()
        dists[b] = f / total if total > 0 else f

    comparisons = {}
    for i, b1 in enumerate(bands):
        for b2 in bands[i + 1 :]:
            p, q = dists[b1], dists[b2]
            eps = 1e-12
            p_s = p + eps
            q_s = q + eps
            p_s /= p_s.sum()
            q_s /= q_s.sum()

            kl_pq = float(np.sum(p_s * np.log2(p_s / q_s)))
            kl_qp = float(np.sum(q_s * np.log2(q_s / p_s)))
            m = 0.5 * (p_s + q_s)
            js = float(
                0.5 * np.sum(p_s * np.log2(p_s / m))
                + 0.5 * np.sum(q_s * np.log2(q_s / m))
            )

            f1, f2 = band_freqs[b1].copy(), band_freqs[b2].copy()
            f1[PAD_ID] = f1[EOS_ID] = 0
            f2[PAD_ID] = f2[EOS_ID] = 0
            unique_b1 = int(((f1 > 0) & (f2 == 0)).sum())
            unique_b2 = int(((f2 > 0) & (f1 == 0)).sum())
            shared = int(((f1 > 0) & (f2 > 0)).sum())

            ratio = np.zeros(VOCAB_SIZE)
            both_seen = (p > 0) & (q > 0)
            ratio[both_seen] = np.log2((p[both_seen] + eps) / (q[both_seen] + eps))
            favor_b1_ids = np.argsort(-ratio)[:20]
            favor_b1 = [
                (int(i), round(float(ratio[i]), 3), int(f1[i]), int(f2[i]))
                for i in favor_b1_ids
                if ratio[i] > 0.5
            ]
            favor_b2_ids = np.argsort(ratio)[:20]
            favor_b2 = [
                (int(i), round(float(-ratio[i]), 3), int(f2[i]), int(f1[i]))
                for i in favor_b2_ids
                if ratio[i] < -0.5
            ]

            # 30. Jaccard overlap coefficient
            jaccard_denom = unique_b1 + unique_b2 + shared
            jaccard = round(shared / jaccard_denom, 6) if jaccard_denom > 0 else 0

            comparisons[f"{b1}_vs_{b2}"] = {
                "kl_divergence_b1_to_b2": round(kl_pq, 6),
                "kl_divergence_b2_to_b1": round(kl_qp, 6),
                "jensen_shannon_divergence": round(js, 6),
                "jaccard_overlap": jaccard,
                f"unique_to_{b1}": unique_b1,
                f"unique_to_{b2}": unique_b2,
                "shared_tokens": shared,
                f"differential_tokens_favor_{b1}": favor_b1[:10],
                f"differential_tokens_favor_{b2}": favor_b2[:10],
            }

    return comparisons


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Exhaustive token analysis (30 metrics)"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="S3 URL or local dir with band_*/shard_*/ structure",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", default="token_analysis_report.json")
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument(
        "--max-shards-per-band",
        type=int,
        default=0,
        help="Limit shards per band for testing (0=all)",
    )
    parser.add_argument("--save-freq-npy", action="store_true")
    args = parser.parse_args()

    is_s3 = args.source.startswith("s3://")

    if args.tokenizer_dir:
        build_lookup_tables(args.tokenizer_dir)
    else:
        log(
            "WARNING: No --tokenizer-dir. Script/fertility/fragmentation/merge-depth disabled."
        )

    if is_s3:
        band_shards = list_shards_s3(args.source)
    else:
        band_shards = list_shards_local(args.source)

    if not band_shards:
        log("ERROR: No shards found!")
        sys.exit(1)

    total_shards = sum(len(v) for v in band_shards.values())
    log(f"Total: {total_shards} shards across {len(band_shards)} bands")

    tasks = []
    for band, shards in sorted(band_shards.items()):
        n = len(shards)
        if args.max_shards_per_band > 0:
            shards = shards[: args.max_shards_per_band]
            log(f"  {band}: using {len(shards)}/{n} shards (limited)")
        for s in shards:
            tasks.append((s, is_s3, band))

    log(f"Processing {len(tasks)} shards with {args.workers} workers...")

    # ── Accumulators ──
    band_freqs = defaultdict(lambda: np.zeros(VOCAB_SIZE, dtype=np.int64))
    band_stats = defaultdict(
        lambda: {
            "n_docs": 0,
            "doc_len_hist": [0] * (len(DOC_LEN_BINS) - 1),
            "richness_hist": [0] * (len(RICHNESS_BINS) - 1),
            "doc_len_sum": 0,
            "doc_unique_sum": 0,
            "top_bigrams": [],
            "n_unique_bigrams": 0,
            "n_total_bigrams": 0,
            "script_counts": [0] * N_SCRIPTS,
            "special_counts": {k: 0 for k in SPECIAL_IDS},
            "total_chars": 0,
            "n_content_tokens_fertility": 0,
            "n_words": 0,
            "frag_hist": [0, 0, 0, 0, 0, 0],
            "avg_fragments_per_word": 0.0,
            "_frag_word_count": 0,
            "_frag_sum": 0.0,
            "total_repeated_ngrams": 0,
            "total_ngrams_checked": 0,
            # New v2
            "intro_freq": np.zeros(VOCAB_SIZE, dtype=np.int64),
            "outro_freq": np.zeros(VOCAB_SIZE, dtype=np.int64),
            "fertility_by_script": {},
            "n_sent_end_tokens": 0,
            "n_garbage_tokens": 0,
            "n_numeric_tokens": 0,
            "n_whitespace_tokens": 0,
            "char_len_hist": [0, 0, 0, 0, 0, 0, 0, 0],  # 8 bins
            "n_clean_boundaries": 0,
            "n_leaky_boundaries": 0,
        }
    )

    done = 0
    errors = 0
    error_details = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyze_shard, t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            done += 1

            if not result.get("ok", False):
                errors += 1
                err = f"{result.get('shard', '?')}: {result.get('error', '?')[:200]}"
                error_details.append(err)
                if errors <= 5:
                    log(f"  ERROR: {err}")
                continue

            band = result["band"]
            band_freqs[band] += result["freq"]

            bs = band_stats[band]
            bs["n_docs"] += result["n_docs"]
            bs["doc_len_sum"] += result["doc_len_sum"]
            bs["doc_unique_sum"] += result["doc_unique_sum"]
            for i in range(len(result["doc_len_hist"])):
                bs["doc_len_hist"][i] += result["doc_len_hist"][i]
            for i in range(len(result["richness_hist"])):
                bs["richness_hist"][i] += result["richness_hist"][i]

            # Bigrams
            bs["top_bigrams"].extend(result["top_bigrams"])
            bs["n_unique_bigrams"] += result["n_unique_bigrams"]
            bs["n_total_bigrams"] += result["n_total_bigrams"]
            if len(bs["top_bigrams"]) > BIGRAM_TOP_K * 10:
                merged = defaultdict(int)
                for k, c in bs["top_bigrams"]:
                    merged[k] += c
                top_items = sorted(merged.items(), key=lambda x: x[1], reverse=True)[
                    : BIGRAM_TOP_K * 2
                ]
                bs["top_bigrams"] = [(k, c) for k, c in top_items]

            # Script counts
            for i in range(min(N_SCRIPTS, len(result["script_counts"]))):
                bs["script_counts"][i] += result["script_counts"][i]

            # Special counts
            for k in result.get("special_counts", {}):
                bs["special_counts"][k] = (
                    bs["special_counts"].get(k, 0) + result["special_counts"][k]
                )

            # Fertility
            bs["total_chars"] += result["total_chars"]
            bs["n_content_tokens_fertility"] += result["n_content_tokens_fertility"]

            # Fragmentation
            bs["n_words"] += result["n_words"]
            for i in range(min(6, len(result["frag_hist"]))):
                bs["frag_hist"][i] += result["frag_hist"][i]
            if result["n_words"] > 0:
                bs["_frag_sum"] += result["avg_fragments_per_word"] * result["n_words"]
                bs["_frag_word_count"] += result["n_words"]

            # Repetition
            bs["total_repeated_ngrams"] += result["total_repeated_ngrams"]
            bs["total_ngrams_checked"] += result["total_ngrams_checked"]

            # ── New v2 merging ──
            bs["intro_freq"] += result["intro_freq"]
            bs["outro_freq"] += result["outro_freq"]
            bs["n_sent_end_tokens"] += result["n_sent_end_tokens"]
            bs["n_garbage_tokens"] += result["n_garbage_tokens"]
            bs["n_numeric_tokens"] += result["n_numeric_tokens"]
            bs["n_whitespace_tokens"] += result["n_whitespace_tokens"]
            for i in range(min(8, len(result.get("char_len_hist", [])))):
                bs["char_len_hist"][i] += result["char_len_hist"][i]
            bs["n_clean_boundaries"] += result["n_clean_boundaries"]
            bs["n_leaky_boundaries"] += result["n_leaky_boundaries"]

            # Fertility by script
            for sc_name, sc_data in result.get("fertility_by_script", {}).items():
                if sc_name not in bs["fertility_by_script"]:
                    bs["fertility_by_script"][sc_name] = {"tokens": 0, "chars": 0}
                bs["fertility_by_script"][sc_name]["tokens"] += sc_data["tokens"]
                bs["fertility_by_script"][sc_name]["chars"] += sc_data["chars"]

            if done % 200 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                log(
                    f"  [{done}/{len(tasks)}] {rate:.1f} shards/s, "
                    f"ETA {eta/60:.1f}m, errors={errors}"
                )

    elapsed = time.time() - t0
    log(
        f"Processing complete: {done} shards in {elapsed:.0f}s "
        f"({elapsed/60:.1f}m), {errors} errors"
    )

    # Finalize aggregations
    for band, bs in band_stats.items():
        merged = defaultdict(int)
        for k, c in bs["top_bigrams"]:
            merged[k] += c
        top_items = sorted(merged.items(), key=lambda x: x[1], reverse=True)[
            :BIGRAM_TOP_K
        ]
        bs["top_bigrams"] = [(k, c) for k, c in top_items]

        if bs["_frag_word_count"] > 0:
            bs["avg_fragments_per_word"] = bs["_frag_sum"] / bs["_frag_word_count"]

    # ── Load tokenizer decoder ──
    decoder = None
    if args.tokenizer_dir:
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(f"{args.tokenizer_dir}/tokenizer.json")
            decoder = lambda tid: tok.decode([tid])
            log("Tokenizer loaded for token decoding")
        except Exception as e:
            log(f"Warning: Could not load tokenizer for decoding: {e}")

    # ── Generate reports ──
    report = {
        "metadata": {
            "source": args.source,
            "total_shards_found": total_shards,
            "shards_processed": done - errors,
            "shards_errored": errors,
            "vocab_size": VOCAB_SIZE,
            "eos_id": EOS_ID,
            "pad_id": PAD_ID,
            "elapsed_seconds": round(elapsed, 1),
            "workers": args.workers,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metrics": [
                "1_token_frequency",
                "2_vocab_coverage",
                "3_token_entropy",
                "4_coverage_curves",
                "5_frequency_buckets",
                "6_top_bottom_tokens",
                "7_doc_length_dist",
                "8_vocab_richness",
                "9_cross_band_divergence",
                "10_differential_tokens",
                "11_unseen_tokens",
                "12_bigram_analysis",
                "13_script_breakdown",
                "14_special_token_usage",
                "15_token_fertility",
                "16_zipf_law_fit",
                "17_subword_fragmentation",
                "18_repetition_rate",
                "19_position_bias",
                "20_sequence_length_entropy",
                "21_bigram_pmi",
                "22_merge_depth",
                "23_fertility_by_script",
                "24_cross_doc_leakage",
                "25_sentence_boundary",
                "26_garbage_token_rate",
                "27_numeric_token_dist",
                "28_whitespace_token_dist",
                "29_char_length_histogram",
                "30_jaccard_overlap",
            ],
        },
        "per_band": {},
        "cross_band": {},
        "global": {},
    }

    for band in sorted(band_freqs):
        log(f"Computing report for {band}...")
        br = compute_band_report(band, band_freqs[band], band_stats[band])

        if decoder:
            for key in ["top_100_tokens", "bottom_100_tokens"]:
                decoded = []
                for tid, count in br[key]:
                    try:
                        text = decoder(tid)
                    except Exception:
                        text = f"<token_{tid}>"
                    decoded.append({"id": tid, "count": count, "text": repr(text)})
                br[f"{key}_decoded"] = decoded

            decoded_bigrams = []
            for bg in br.get("top_100_bigrams", []):
                try:
                    t1_text = decoder(bg["token1_id"])
                    t2_text = decoder(bg["token2_id"])
                except Exception:
                    t1_text = f"<{bg['token1_id']}>"
                    t2_text = f"<{bg['token2_id']}>"
                decoded_bigrams.append(
                    {
                        **bg,
                        "token1_text": repr(t1_text),
                        "token2_text": repr(t2_text),
                        "combined": repr(t1_text + t2_text),
                    }
                )
            br["top_100_bigrams_decoded"] = decoded_bigrams

            # Decode position bias tokens
            for pos_key in ["intro_overrepresented", "outro_overrepresented"]:
                items = br.get("position_bias", {}).get(pos_key, [])
                for item in items:
                    try:
                        item["text"] = repr(decoder(item["id"]))
                    except Exception:
                        item["text"] = f"<token_{item['id']}>"

            # Decode PMI bigrams
            for item in br.get("top_50_bigram_pmi", []):
                try:
                    item["token1_text"] = repr(decoder(item["token1_id"]))
                    item["token2_text"] = repr(decoder(item["token2_id"]))
                except Exception:
                    pass

        if args.save_freq_npy:
            freq_path = os.path.join(
                os.path.dirname(args.output) or ".", f"freq_{band}.npy"
            )
            np.save(freq_path, band_freqs[band])
            br["freq_file"] = freq_path
            log(f"  Saved {freq_path}")

        report["per_band"][band] = br

    # ── Cross-band ──
    if len(band_freqs) >= 2:
        log("Computing cross-band comparisons...")
        cross = compute_cross_band(band_freqs)
        if decoder:
            for pair_key, pair_data in cross.items():
                for k, v in list(pair_data.items()):
                    if k.startswith("differential_tokens_favor_"):
                        decoded = []
                        for item in v:
                            tid = item[0]
                            try:
                                text = decoder(tid)
                            except Exception:
                                text = f"<token_{tid}>"
                            decoded.append(
                                {
                                    "id": tid,
                                    "log2_ratio": item[1],
                                    "count_favored": item[2],
                                    "count_other": item[3],
                                    "text": repr(text),
                                }
                            )
                        pair_data[f"{k}_decoded"] = decoded
        report["cross_band"] = cross

    # ── Global ──
    log("Computing global analysis...")
    global_freq = np.zeros(VOCAB_SIZE, dtype=np.int64)
    for f in band_freqs.values():
        global_freq += f
    global_freq[PAD_ID] = 0
    global_freq[EOS_ID] = 0

    unseen_ids = np.where(global_freq == 0)[0]
    total_content = int(global_freq.sum())

    global_report = {
        "total_content_tokens": total_content,
        "vocab_seen": int(np.count_nonzero(global_freq)),
        "vocab_unseen": len(unseen_ids),
        "vocab_unseen_pct": round(100 * len(unseen_ids) / VOCAB_SIZE, 2),
        "unseen_token_ids_sample": unseen_ids[:1000].tolist(),
    }

    if decoder:
        unseen_decoded = []
        for tid in unseen_ids[:500]:
            try:
                text = decoder(int(tid))
            except Exception:
                text = f"<token_{tid}>"
            unseen_decoded.append({"id": int(tid), "text": repr(text)})
        global_report["unseen_tokens_decoded_sample"] = unseen_decoded

    report["global"] = global_report

    if args.save_freq_npy:
        gpath = os.path.join(os.path.dirname(args.output) or ".", "freq_global.npy")
        np.save(gpath, global_freq)
        global_report["freq_file"] = gpath

    # ── Write report ──
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    log(f"Report written to {args.output}")

    # ── Print summary ──
    _print_summary(report)


def _print_summary(report):
    """Print human-readable summary of all 26 metrics."""
    print("\n" + "=" * 80)
    print("TOKEN ANALYSIS SUMMARY — 30 METRICS")
    print("=" * 80)

    for band in sorted(report["per_band"]):
        br = report["per_band"][band]
        print(f"\n{'─' * 80}")
        print(f"  {band}")
        print(f"{'─' * 80}")
        print(f"  Content tokens:     {br['content_tokens']:>18,}")
        print(f"  PAD ratio:          {br['pad_ratio']:>17.2%}")
        print(f"  Vocab coverage:     {br['vocab_coverage_pct']:>17.1f}%")
        print(
            f"  Token entropy:      {br['token_entropy_bits']:>17.2f} bits "
            f"(max {br['max_possible_entropy_bits']:.2f})"
        )
        print(f"  Rare (<10):         {br['n_rare_lt10']:>18,}")
        print(f"  Rare (<100):        {br['n_rare_lt100']:>18,}")
        print(f"  Rare (<1000):       {br['n_rare_lt1000']:>18,}")
        print(f"  Documents:          {br['n_docs']:>18,}")
        print(f"  Avg doc length:     {br['avg_doc_len_tokens']:>18.1f} tokens")
        print(f"  Type/token ratio:   {br['avg_type_token_ratio']:>18.4f}")

        c = br["coverage_tokens_needed"]
        print(f"  Coverage 50%:       {c.get('50.0%', 'N/A'):>18,} unique tokens")
        print(f"  Coverage 90%:       {c.get('90.0%', 'N/A'):>18,} unique tokens")
        print(f"  Coverage 99%:       {c.get('99.0%', 'N/A'):>18,} unique tokens")

        if br.get("zipf_fit") and "alpha" in br["zipf_fit"]:
            z = br["zipf_fit"]
            print(
                f"  Zipf alpha:         {z['alpha']:>18.4f} (R2={z['r_squared']:.4f}, {z['interpretation']})"
            )

        if br.get("fertility") and "chars_per_token" in br["fertility"]:
            print(
                f"  Fertility:          {br['fertility']['chars_per_token']:>18.4f} chars/token"
            )

        if br.get("fragmentation") and "avg_subwords_per_word" in br["fragmentation"]:
            fg = br["fragmentation"]
            print(
                f"  Fragmentation:      {fg['avg_subwords_per_word']:>18.4f} subwords/word"
            )

        if br.get("repetition") and "repetition_rate" in br["repetition"]:
            print(f"  Repetition (4-gram):{br['repetition']['repetition_rate']:>18.6f}")

        # ── New v2 metrics ──
        print(
            f"  Seq len entropy:    {br.get('sequence_length_entropy_bits', 0):>18.4f} bits"
        )

        if br.get("merge_depth") and "avg_weighted_merge_depth" in br["merge_depth"]:
            print(
                f"  Avg merge depth:    {br['merge_depth']['avg_weighted_merge_depth']:>18.4f}"
            )

        if (
            br.get("sentence_boundary")
            and "avg_sentences_per_doc" in br["sentence_boundary"]
        ):
            sb = br["sentence_boundary"]
            print(f"  Avg sent/doc:       {sb['avg_sentences_per_doc']:>18.2f}")
            print(f"  Avg tok/sentence:   {sb['avg_tokens_per_sentence']:>18.1f}")

        if br.get("cross_doc_leakage") and "leakage_rate" in br["cross_doc_leakage"]:
            print(
                f"  Doc leakage rate:   {br['cross_doc_leakage']['leakage_rate']:>18.6f}"
            )

        if br.get("garbage_token_rate") and "garbage_rate" in br["garbage_token_rate"]:
            print(
                f"  Garbage rate:       {br['garbage_token_rate']['garbage_rate']:>18.8f}"
            )

        if br.get("numeric_token_dist") and "numeric_rate" in br["numeric_token_dist"]:
            nd = br["numeric_token_dist"]
            print(
                f"  Numeric rate:       {nd['numeric_rate']:>18.6f} ({nd['numeric_token_count']:,} tokens)"
            )

        if (
            br.get("whitespace_token_dist")
            and "whitespace_rate" in br["whitespace_token_dist"]
        ):
            wd = br["whitespace_token_dist"]
            print(
                f"  Whitespace rate:    {wd['whitespace_rate']:>18.6f} ({wd['whitespace_token_count']:,} tokens)"
            )

        if br.get("char_length_histogram"):
            print("  Char-length dist:")
            for label, cnt in br["char_length_histogram"].items():
                print(f"    {label:>8} chars: {cnt:>15,}")

        if br.get("script_breakdown"):
            print("  Script breakdown:")
            for sc, data in sorted(
                br["script_breakdown"].items(),
                key=lambda x: x[1]["tokens"],
                reverse=True,
            )[:8]:
                print(f"    {sc:>15}: {data['tokens']:>15,} ({data['pct']:5.1f}%)")

        if br.get("fertility_by_script"):
            print("  Fertility by script:")
            for sc, data in sorted(
                br["fertility_by_script"].items(),
                key=lambda x: x[1]["tokens"],
                reverse=True,
            )[:8]:
                print(
                    f"    {sc:>15}: {data['chars_per_token']:.4f} ch/tok ({data['tokens']:,} tokens)"
                )

        if br.get("position_bias"):
            pb = br["position_bias"]
            if pb.get("intro_overrepresented"):
                print("  Position bias (intro top-3):")
                for item in pb["intro_overrepresented"][:3]:
                    text = item.get("text", f"<{item['id']}>")
                    print(f"    {text:>30}: ratio={item['log2_ratio']:.1f}x")
            if pb.get("outro_overrepresented"):
                print("  Position bias (outro top-3):")
                for item in pb["outro_overrepresented"][:3]:
                    text = item.get("text", f"<{item['id']}>")
                    print(f"    {text:>30}: ratio={item['log2_ratio']:.1f}x")

        print("  Freq buckets:")
        for bkt, cnt in br["freq_buckets"].items():
            print(f"    {bkt:>12}: {cnt:>8,} tokens")

        print("  Doc length distribution:")
        for label, cnt in br["doc_len_hist"].items():
            pct = 100 * cnt / br["n_docs"] if br["n_docs"] > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"    {label:>8}: {cnt:>12,} ({pct:5.1f}%) {bar}")

    if report["cross_band"]:
        print(f"\n{'─' * 80}")
        print("  CROSS-BAND COMPARISONS")
        print(f"{'─' * 80}")
        for pair, stats in report["cross_band"].items():
            print(f"\n  {pair}:")
            print(f"    JS divergence: {stats['jensen_shannon_divergence']:.6f}")
            print(f"    Jaccard:       {stats.get('jaccard_overlap', 0):.6f}")
            for k, v in stats.items():
                if k.startswith("unique_to_"):
                    print(f"    {k}: {v:,}")
            print(f"    Shared tokens: {stats['shared_tokens']:,}")

    g = report["global"]
    print(f"\n{'─' * 80}")
    print("  GLOBAL")
    print(f"{'─' * 80}")
    print(f"  Total content tokens: {g['total_content_tokens']:>15,}")
    print(f"  Vocab seen:           {g['vocab_seen']:>15,} / {VOCAB_SIZE:,}")
    print(
        f"  Vocab unseen:         {g['vocab_unseen']:>15,} ({g['vocab_unseen_pct']:.1f}%)"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
