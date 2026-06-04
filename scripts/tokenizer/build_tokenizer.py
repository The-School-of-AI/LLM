#!/usr/bin/env python3
"""
Build a 131,072-vocab Kronecker-aware multilingual BPE tokenizer.

Phases:
  1. Extract text from parquet files (parallel, all cores)
  2. Train Indic BPE (55K vocab) from extracted text
  3. Download Tekken, extract top 75K English/EU/Code tokens
  4. Assemble final tokenizer.json
  5. Quality checks (fertility, byte usage, KE constraints)

Requirements:
  pip install tokenizers pyarrow transformers tqdm

Usage:
  python build_tokenizer.py --data-dir ./indic_tokenizer_samples_by_size --output-dir ./output
"""

import argparse
import json
import os
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pyarrow.parquet as pq
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════

TOTAL_VOCAB = 131_072
N_SPECIAL = 256
N_BYTE = 256
N_MATH = 512
N_RESERVED = 256
N_ENGLISH = 75_000
N_INDIC = TOTAL_VOCAB - N_SPECIAL - N_BYTE - N_MATH - N_RESERVED - N_ENGLISH  # 54,792

# KroneckerEmbeddings constraints
KE_POS_DIM = 32  # max bytes per token
KE_CHAR_DIM = 256  # byte vocabulary size

# Temperature for language sampling (lower = more balanced)
SAMPLING_TEMPERATURE = 0.3

# Target Indic scripts and their Unicode script names
INDIC_SCRIPTS = {
    "Devanagari",  # Hindi, Marathi
    "Bengali",  # Bengali, Assamese
    "Tamil",
    "Telugu",
    "Kannada",
    "Malayalam",
    "Gujarati",
    "Gurmukhi",  # Punjabi
    "Oriya",  # Odia
}

# Language code to script mapping (for the dataset sources)
LANG_TO_SCRIPT = {
    "hi": "Devanagari",
    "mr": "Devanagari",
    "bn": "Bengali",
    "as": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Gurmukhi",
    "or": "Oriya",
}

# Pre-tokenization regex (script-aware, prevents cross-script BPE merges)
# NOTE: Uses \p{Latin} not \p{Script=Latin} — Oniguruma (tokenizer JSON loader)
# supports \p{ScriptName} but NOT \p{Script=ScriptName}.
PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{Latin}+"
    r"|\p{Devanagari}[\p{Devanagari}\p{M}]*"
    r"|\p{Bengali}[\p{Bengali}\p{M}]*"
    r"|\p{Tamil}[\p{Tamil}\p{M}]*"
    r"|\p{Telugu}[\p{Telugu}\p{M}]*"
    r"|\p{Kannada}[\p{Kannada}\p{M}]*"
    r"|\p{Malayalam}[\p{Malayalam}\p{M}]*"
    r"|\p{Gujarati}[\p{Gujarati}\p{M}]*"
    r"|\p{Gurmukhi}[\p{Gurmukhi}\p{M}]*"
    r"|\p{Oriya}[\p{Oriya}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s+"
)

# ═══════════════════════════════════════════════════════════
# SPECIAL TOKENS (LLaMA-3 style)
# ═══════════════════════════════════════════════════════════

SPECIAL_TOKENS = [
    "<|begin_of_text|>",  # 0: BOS
    "<|end_of_text|>",  # 1: EOS
    "<|pad|>",  # 2: Padding
    "<|unk|>",  # 3: Unknown
    "<|start_header_id|>",  # 4
    "<|end_header_id|>",  # 5
    "<|eot_id|>",  # 6: End of turn
    "<|python_tag|>",  # 7
    "<|tool_call|>",  # 8
    "<|tool_result|>",  # 9
    "<|system|>",  # 10
    "<|user|>",  # 11
    "<|assistant|>",  # 12
    "<|ipython|>",  # 13
]
# Fill remaining special slots
for i in range(len(SPECIAL_TOKENS), N_SPECIAL):
    SPECIAL_TOKENS.append(f"<|reserved_special_{i}|>")

# ═══════════════════════════════════════════════════════════
# MATH / LATEX TOKENS (512 slots)
# ═══════════════════════════════════════════════════════════

MATH_TOKENS = []

# Individual digits (10)
MATH_TOKENS.extend(list("0123456789"))

# Arithmetic operators (30)
MATH_TOKENS.extend(
    [
        "+",
        "-",
        "*",
        "/",
        "=",
        "<",
        ">",
        "^",
        "%",
        "!",
        "==",
        "!=",
        "<=",
        ">=",
        "+=",
        "-=",
        "*=",
        "/=",
        "**",
        "//",
        "<<",
        ">>",
        "&&",
        "||",
        "+-",
        "-+",
        "\u00d7",
        "\u00f7",
        "\u00b1",
        "\u2213",  # ×, ÷, ±, ∓
    ]
)

# Mathematical symbols (50)
MATH_TOKENS.extend(
    [
        "\u2248",
        "\u2260",
        "\u2264",
        "\u2265",
        "\u221d",
        "\u221e",  # ≈≠≤≥∝∞
        "\u221a",
        "\u221b",
        "\u221c",  # √∛∜
        "\u2200",
        "\u2203",
        "\u2204",
        "\u2208",
        "\u2209",
        "\u220b",
        "\u220c",  # ∀∃∄∈∉∋∌
        "\u2282",
        "\u2283",
        "\u2286",
        "\u2287",  # ⊂⊃⊆⊇
        "\u222a",
        "\u2229",
        "\u2205",
        "\u2201",
        "\u2206",  # ∪∩∅∁∆
        "\u2192",
        "\u2190",
        "\u2194",
        "\u21d2",
        "\u21d0",
        "\u21d4",
        "\u21a6",  # →←↔⇒⇐⇔↦
        "\u2202",
        "\u2207",
        "\u222b",
        "\u222c",
        "\u222d",
        "\u222e",  # ∂∇∫∬∭∮
        "\u2211",
        "\u220f",  # ∑∏
        "\u2295",
        "\u2297",
        "\u2299",
        "\u22a5",
        "\u2225",
        "\u2220",
        "\u00b0",  # ⊕⊗⊙⊥∥∠°
    ]
)

# Greek letters (48)
MATH_TOKENS.extend(
    list(
        "\u03b1\u03b2\u03b3\u03b4\u03b5\u03b6\u03b7\u03b8"  # αβγδεζηθ
        "\u03b9\u03ba\u03bb\u03bc\u03bd\u03be\u03bf\u03c0"  # ικλμνξοπ
        "\u03c1\u03c3\u03c4\u03c5\u03c6\u03c7\u03c8\u03c9"  # ρστυφχψω
        "\u0391\u0392\u0393\u0394\u0395\u0396\u0397\u0398"  # ΑΒΓΔΕΖΗΘ
        "\u0399\u039a\u039b\u039c\u039d\u039e\u039f\u03a0"  # ΙΚΛΜΝΞΟΠ
        "\u03a1\u03a3\u03a4\u03a5\u03a6\u03a7\u03a8\u03a9"  # ΡΣΤΥΦΧΨΩ
    )
)

# LaTeX commands (200+)
LATEX_COMMANDS = [
    # Fractions and roots
    "\\frac",
    "\\dfrac",
    "\\tfrac",
    "\\sqrt",
    "\\cbrt",
    # Summation and products
    "\\sum",
    "\\prod",
    "\\coprod",
    "\\bigcup",
    "\\bigcap",
    "\\bigoplus",
    "\\bigotimes",
    # Calculus
    "\\int",
    "\\iint",
    "\\iiint",
    "\\oint",
    "\\lim",
    "\\limsup",
    "\\liminf",
    "\\partial",
    "\\nabla",
    # Trigonometry
    "\\sin",
    "\\cos",
    "\\tan",
    "\\sec",
    "\\csc",
    "\\cot",
    "\\arcsin",
    "\\arccos",
    "\\arctan",
    "\\sinh",
    "\\cosh",
    "\\tanh",
    # Logarithms
    "\\log",
    "\\ln",
    "\\lg",
    "\\exp",
    # Min/max
    "\\min",
    "\\max",
    "\\sup",
    "\\inf",
    "\\arg",
    "\\argmin",
    "\\argmax",
    # Relations
    "\\leq",
    "\\geq",
    "\\neq",
    "\\approx",
    "\\equiv",
    "\\sim",
    "\\simeq",
    "\\cong",
    "\\propto",
    "\\ll",
    "\\gg",
    # Arrows
    "\\rightarrow",
    "\\leftarrow",
    "\\Rightarrow",
    "\\Leftarrow",
    "\\Leftrightarrow",
    "\\mapsto",
    "\\longrightarrow",
    "\\longleftarrow",
    # Formatting
    "\\mathbb",
    "\\mathcal",
    "\\mathbf",
    "\\mathrm",
    "\\mathit",
    "\\mathsf",
    "\\text",
    "\\textbf",
    # Environments
    "\\begin",
    "\\end",
    "{equation}",
    "{align}",
    "{matrix}",
    "{pmatrix}",
    "{bmatrix}",
    "{cases}",
    "{array}",
    # Brackets
    "\\left",
    "\\right",
    "\\langle",
    "\\rangle",
    "\\lceil",
    "\\rceil",
    "\\lfloor",
    "\\rfloor",
    "\\{",
    "\\}",
    # Set theory
    "\\in",
    "\\notin",
    "\\subset",
    "\\supset",
    "\\subseteq",
    "\\supseteq",
    "\\cup",
    "\\cap",
    "\\setminus",
    "\\emptyset",
    # Logic
    "\\forall",
    "\\exists",
    "\\nexists",
    "\\neg",
    "\\land",
    "\\lor",
    "\\implies",
    "\\iff",
    "\\therefore",
    "\\because",
    # Accents and modifiers
    "\\hat",
    "\\bar",
    "\\dot",
    "\\ddot",
    "\\tilde",
    "\\vec",
    "\\overline",
    "\\underline",
    "\\overbrace",
    "\\underbrace",
    # Matrices
    "\\binom",
    "\\choose",
    "\\pmatrix",
    "\\bmatrix",
    "\\vmatrix",
    "\\Vmatrix",
    # Spaces
    "\\quad",
    "\\qquad",
    "\\,",
    "\\;",
    "\\:",
    "\\!",
    "\\hspace",
    "\\vspace",
    # Misc
    "\\cdot",
    "\\cdots",
    "\\ldots",
    "\\vdots",
    "\\ddots",
    "\\times",
    "\\div",
    "\\pm",
    "\\mp",
    "\\circ",
    "\\bullet",
    "\\infty",
    "\\aleph",
    "\\hbar",
    "\\ell",
    "\\wp",
    "\\Re",
    "\\Im",
]
MATH_TOKENS.extend(LATEX_COMMANDS)

# Scientific notation fragments (30)
MATH_TOKENS.extend(
    [
        "e+",
        "e-",
        "E+",
        "E-",
        "e0",
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "e6",
        "e7",
        "e8",
        "e9",
        ".0",
        ".1",
        ".2",
        ".3",
        ".4",
        ".5",
        ".6",
        ".7",
        ".8",
        ".9",
        "1e",
        "2e",
        "10^",
        "10e",
    ]
)

# Common number patterns (a few)
MATH_TOKENS.extend(["00", "000", "0000", ".00", "0.", "1.", "2.", "3.14", "\u03c0"])

# Deduplicate and pad to N_MATH
MATH_TOKENS = list(dict.fromkeys(MATH_TOKENS))  # preserve order, remove dupes
if len(MATH_TOKENS) > N_MATH:
    print(f"WARNING: {len(MATH_TOKENS)} math tokens > {N_MATH} budget, truncating")
    MATH_TOKENS = MATH_TOKENS[:N_MATH]


# ═══════════════════════════════════════════════════════════
# BYTE TOKENS (256)
# ═══════════════════════════════════════════════════════════

BYTE_TOKENS = [f"<0x{i:02X}>" for i in range(256)]


# ═══════════════════════════════════════════════════════════
# PHASE 1: EXTRACT TEXT FROM PARQUET FILES
# ═══════════════════════════════════════════════════════════

import re

# Regex to extract Indic text from translation-pair format:
# "[as] অসমীয়া text [en] English text" → "অসমীয়া text"
# Also handles: "[en] English [hi] हिंदी text"
_LANG_TAG_RE = re.compile(r"\[([a-z]{2,3})\]\s*")

# Indic language codes in the dataset
_INDIC_LANG_CODES = {
    "hi",
    "bn",
    "ta",
    "te",
    "kn",
    "ml",
    "gu",
    "mr",
    "pa",
    "or",
    "as",
    # Also match BCP47/Flores codes
    "hin",
    "ben",
    "tam",
    "tel",
    "kan",
    "mal",
    "guj",
    "mar",
    "pan",
    "ori",
    "asm",
}

# Chat/system prompt patterns to strip
_CHAT_TAGS_RE = re.compile(
    r"<\|(?:user|assistant|system)\|>|System:\s*You are.*?\n", re.DOTALL
)

# Flores-style language tags like "asm_Beng", "eng_Latn"
_FLORES_TAG_RE = re.compile(r"\[\]\s*[a-z]{3}_[A-Z][a-z]{3}")


def _clean_text(text: str, source: str) -> Optional[str]:
    """
    Clean text based on source type. Extracts only Indic portions.
    Returns cleaned text or None if nothing useful.
    """
    if not text or len(text) < 5:
        return None

    # ── Translation pair sources ──
    # Format: "[as] Indic text [en] English text" or "[en] English [hi] Hindi"
    if source.startswith("ai-bharath-"):
        # Skip rows that are just language tags with no content
        if _FLORES_TAG_RE.search(text):
            stripped = _FLORES_TAG_RE.sub("", text).strip()
            if len(stripped) < 10:
                return None

        # Split by language tags and extract Indic portions
        parts = _LANG_TAG_RE.split(text)
        # parts = ['', 'as', 'Indic text ', 'en', 'English text ']
        indic_parts = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i].lower() in _INDIC_LANG_CODES:
                indic_text = parts[i + 1].strip()
                if indic_text and len(indic_text) > 3:
                    indic_parts.append(indic_text)
                i += 2
            else:
                i += 1

        if indic_parts:
            return " ".join(indic_parts)
        return None

    # ── Chat format sources ──
    # samvaad_hi: has <|user|> <|assistant|> tags, mixed English+Hindi
    # sarvamai_mmlu: has System: prompts, MCQ in Indic
    if source in ("samvaad_hi",) or source.startswith("sarvamai_"):
        text = _CHAT_TAGS_RE.sub("", text)
        # Remove common English instruction prefixes
        text = re.sub(
            r"(?:Can you tell me|please answer|Also,?\s*please answer in \w+\.?|"
            r"in simple English please|System:|Question:|Choices?:|Answer:)[^\n]*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        # For samvaad: split into lines and keep only lines with Indic chars
        # (drops English-only lines)
        lines = text.split("\n")
        indic_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check if line has any Indic Unicode (U+0900-U+0D7F covers all Brahmic)
            has_indic = any("\u0900" <= ch <= "\u0D7F" for ch in line)
            if has_indic:
                indic_lines.append(line)
        text = "\n".join(indic_lines).strip()
        if len(text) < 10:
            return None
        return text

    # ── Clean monolingual sources ──
    # sangraha_*, erav4_lang_*: already clean, use as-is
    text = text.strip()
    if len(text) < 10:
        return None
    return text


def _read_single_parquet(args: tuple) -> List[str]:
    """Read and clean text from a single parquet file. Runs in subprocess."""
    path, source = args
    try:
        pf = pq.ParquetFile(path)
        texts = []
        for batch in pf.iter_batches(batch_size=10_000, columns=["text"]):
            col = batch.column("text")
            for val in col:
                s = val.as_py()
                cleaned = _clean_text(s, source)
                if cleaned:
                    texts.append(cleaned)
        return texts
    except Exception as e:
        print(f"  Error reading {path}: {e}", file=sys.stderr)
        return []


def extract_texts_parallel(
    data_dir: str, num_workers: int = 10
) -> Dict[str, List[str]]:
    """
    Extract text from all parquet files, grouped by source.
    Uses multiprocessing for parallel I/O.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: Extracting text from parquet files")
    print("=" * 60)

    source_dirs = sorted(Path(data_dir).iterdir())
    # Collect all (source, filepath) pairs
    jobs = []
    for sd in source_dirs:
        if not sd.is_dir():
            continue
        source = sd.name.replace("source=", "")
        parquet_files = sorted(sd.glob("*.parquet"))
        for pf in parquet_files:
            jobs.append((source, str(pf)))

    print(f"  Found {len(jobs)} parquet files across {len(source_dirs)} sources")
    print(f"  Using {num_workers} workers")

    # Submit all reads in parallel
    texts_by_source: Dict[str, List[str]] = defaultdict(list)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {}
        for source, path in jobs:
            fut = pool.submit(_read_single_parquet, (path, source))
            futures[fut] = source

        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="  Reading parquet"
        ):
            source = futures[fut]
            texts = fut.result()
            texts_by_source[source].extend(texts)

    elapsed = time.time() - t0
    total_texts = sum(len(v) for v in texts_by_source.values())
    total_chars = sum(sum(len(t) for t in v) for v in texts_by_source.values())

    print(
        f"\n  Extracted {total_texts:,} documents, {total_chars/1e9:.2f} GB of text in {elapsed:.1f}s"
    )
    print("\n  Per-source breakdown:")
    for src in sorted(texts_by_source.keys()):
        n = len(texts_by_source[src])
        chars = sum(len(t) for t in texts_by_source[src])
        print(f"    {src:30s}: {n:>8,} docs, {chars/1e6:>8.1f} MB")

    return dict(texts_by_source)


def prepare_training_text(
    texts_by_source: Dict[str, List[str]],
    output_dir: str,
    temperature: float = SAMPLING_TEMPERATURE,
    max_bytes_per_lang: int = 50
    * 1024
    * 1024,  # 50 MB cap per language (600MB total fits in RAM)
) -> List[str]:
    """
    Prepare balanced training text files for BPE trainer.
    Uses temperature sampling to balance low-resource languages.
    Returns list of output file paths.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1b: Preparing balanced training text")
    print("=" * 60)

    # Detect language from source name
    lang_texts: Dict[str, List[str]] = defaultdict(list)
    for source, texts in texts_by_source.items():
        # Infer language from source name
        lang = None
        for code in LANG_TO_SCRIPT:
            if f"_{code}" in source or source.endswith(f"_{code}"):
                lang = code
                break
        if lang is None:
            # Multi-language sources — just add to "mixed"
            lang = "mixed"
        lang_texts[lang].extend(texts)

    # Compute per-language byte counts
    lang_bytes = {}
    for lang, texts in lang_texts.items():
        total_b = sum(len(t.encode("utf-8", errors="ignore")) for t in texts)
        lang_bytes[lang] = total_b

    print("\n  Raw per-language sizes:")
    for lang in sorted(lang_bytes, key=lambda k: -lang_bytes[k]):
        print(
            f"    {lang:6s}: {lang_bytes[lang]/1e6:>8.1f} MB ({len(lang_texts[lang]):>8,} docs)"
        )

    # Temperature-based sampling
    # p_lang ∝ n_lang^(1/T) — lower T = more balanced
    indic_langs = [l for l in lang_bytes if l in LANG_TO_SCRIPT]
    if indic_langs:
        raw_sizes = {l: lang_bytes[l] for l in indic_langs}
        total_raw = sum(raw_sizes.values())
        # Apply temperature
        tempered = {l: raw_sizes[l] ** (1.0 / temperature) for l in indic_langs}
        total_tempered = sum(tempered.values())
        target_fracs = {l: tempered[l] / total_tempered for l in indic_langs}

        print(f"\n  Temperature-sampled fractions (T={temperature}):")
        for lang in sorted(target_fracs, key=lambda k: -target_fracs[k]):
            raw_frac = raw_sizes[lang] / total_raw
            print(
                f"    {lang:6s}: raw={raw_frac*100:5.1f}% → sampled={target_fracs[lang]*100:5.1f}%"
            )

    # Write per-language text files
    os.makedirs(output_dir, exist_ok=True)
    output_files = []
    total_written = 0

    print(f"\n  Writing text files (cap: {max_bytes_per_lang/1e6:.0f} MB/lang):")
    for lang in sorted(lang_texts.keys()):
        texts = lang_texts[lang]
        out_path = os.path.join(output_dir, f"{lang}.txt")
        written_bytes = 0
        written_lines = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for t in texts:
                if written_bytes >= max_bytes_per_lang:
                    break
                line = t.strip()
                if line:
                    f.write(line + "\n")
                    written_bytes += len(line.encode("utf-8", errors="ignore")) + 1
                    written_lines += 1
        output_files.append(out_path)
        total_written += written_bytes
        capped = " (CAPPED)" if written_bytes >= max_bytes_per_lang else ""
        print(
            f"    {lang:8s}: {written_bytes/1e6:>6.1f} MB, {written_lines:>8,} lines{capped}"
        )

    print(
        f"    {'TOTAL':8s}: {total_written/1e6:>6.1f} MB across {len(output_files)} files"
    )
    return output_files


# ═══════════════════════════════════════════════════════════
# PHASE 2: TRAIN INDIC BPE
# ═══════════════════════════════════════════════════════════


def _format_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n/1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n/1024**2:.1f} MB"
    return f"{n/1024:.0f} KB"


def _monitor_training(stop_event, interval=10):
    """Background thread that prints memory/CPU stats during BPE training."""
    import resource

    start = time.time()
    tick = 0
    while not stop_event.is_set():
        stop_event.wait(interval)
        if stop_event.is_set():
            break
        tick += 1
        elapsed = time.time() - start
        # Get RSS from resource module (bytes on macOS)
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss_bytes = ru.ru_maxrss  # macOS: bytes, Linux: KB
        if sys.platform == "darwin":
            rss_mb = rss_bytes / 1024 / 1024
        else:
            rss_mb = rss_bytes / 1024
        mins, secs = divmod(int(elapsed), 60)
        print(
            f"    [monitor] {mins:02d}:{secs:02d} elapsed | RSS: {rss_mb:.0f} MB | "
            f"(training in progress...)",
            flush=True,
        )


def train_indic_bpe(
    text_files: List[str],
    vocab_size: int = N_INDIC + N_BYTE,  # include byte tokens as base
    output_path: str = "indic_bpe_raw.json",
    num_threads: int = 10,
) -> str:
    """
    Train a BPE tokenizer on Indic text using HuggingFace tokenizers library.
    Returns path to the trained tokenizer JSON.
    """
    import threading

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    print("\n" + "=" * 60)
    print("  PHASE 2: Training Indic BPE tokenizer")
    print("=" * 60)
    print(f"  Target vocab: {vocab_size}")
    print(f"  Input files: {len(text_files)}")
    print(f"  Threads: {num_threads}")

    # Filter to only Indic language files
    indic_codes = set(LANG_TO_SCRIPT.keys()) | {"mixed"}
    indic_files = [
        f for f in text_files if any(Path(f).stem == code for code in indic_codes)
    ]
    if not indic_files:
        indic_files = text_files  # fallback to all if filtering removes everything

    # Log file details
    total_input_bytes = 0
    total_input_lines = 0
    print(f"\n  Input files ({len(indic_files)}):")
    for f in indic_files:
        fsize = os.path.getsize(f)
        with open(f, "r") as fh:
            nlines = sum(1 for _ in fh)
        total_input_bytes += fsize
        total_input_lines += nlines
        print(f"    {Path(f).stem:8s}: {_format_bytes(fsize):>10s}, {nlines:>9,} lines")
    print(
        f"    {'TOTAL':8s}: {_format_bytes(total_input_bytes):>10s}, {total_input_lines:>9,} lines"
    )

    # Estimate: warn if data is too large for RAM
    estimated_memory = total_input_bytes * 15  # rough 15x multiplier for BPE internals
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF)
    current_rss = (
        ru.ru_maxrss / 1024 / 1024 if sys.platform == "darwin" else ru.ru_maxrss / 1024
    )
    print(f"\n  Estimated peak memory: ~{_format_bytes(int(estimated_memory))}")
    print(f"  Current RSS: {current_rss:.0f} MB")
    if estimated_memory > 50 * 1024**3:
        print(
            f"  WARNING: Estimated memory ({_format_bytes(int(estimated_memory))}) may exceed RAM!"
        )
        print("  Consider reducing max_bytes_per_lang.")

    # Create BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Set pre-tokenizer to our script-aware regex
    print("\n  Pre-tokenizer: Script-aware regex Split")
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=PRETOKENIZE_REGEX,
        behavior="isolated",
        invert=False,
    )

    # BPE trainer
    # CRITICAL: max_token_length counts CHARACTERS, not bytes.
    # Indic chars = 3 bytes each. max_token_length=10 → max 30 bytes ≤ POS_DIM=32.
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[],  # we add these later in assembly
        initial_alphabet=[chr(i) for i in range(256)],  # byte-level base (must be str)
        show_progress=True,
        max_token_length=10,  # 10 chars × 3 bytes/char = 30 bytes max (fits KE POS_DIM=32)
    )

    # Start monitoring thread
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_training, args=(stop_monitor, 15), daemon=True
    )
    monitor_thread.start()

    t0 = time.time()
    print(f"\n  Training BPE on {_format_bytes(total_input_bytes)} of text...")
    print("  (monitor prints stats every 15s)", flush=True)

    tokenizer.train(indic_files, trainer)

    # Stop monitor
    stop_monitor.set()
    monitor_thread.join(timeout=2)

    elapsed = time.time() - t0
    actual_vocab = tokenizer.get_vocab_size()
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n  Training complete in {mins}m {secs}s")
    print(f"  Actual vocab size: {actual_vocab:,}")
    print(f"  Merges needed: {actual_vocab - 256:,} (vocab - initial alphabet)")

    # Save
    tokenizer.save(output_path)
    fsize = os.path.getsize(output_path)
    print(f"  Saved to: {output_path} ({_format_bytes(fsize)})")

    return output_path


# ═══════════════════════════════════════════════════════════
# PHASE 3: DOWNLOAD & PROCESS TEKKEN
# ═══════════════════════════════════════════════════════════


def get_token_primary_script(token: str) -> Optional[str]:
    """Get the primary Unicode script of a token (by majority of characters)."""
    script_counts: Counter = Counter()
    for ch in token:
        try:
            # unicodedata doesn't have script info, use category as proxy
            cat = unicodedata.category(ch)
            name = unicodedata.name(ch, "")

            if name.startswith("DEVANAGARI"):
                script_counts["Devanagari"] += 1
            elif name.startswith("BENGALI"):
                script_counts["Bengali"] += 1
            elif name.startswith("TAMIL"):
                script_counts["Tamil"] += 1
            elif name.startswith("TELUGU"):
                script_counts["Telugu"] += 1
            elif name.startswith("KANNADA"):
                script_counts["Kannada"] += 1
            elif name.startswith("MALAYALAM"):
                script_counts["Malayalam"] += 1
            elif name.startswith("GUJARATI"):
                script_counts["Gujarati"] += 1
            elif name.startswith("GURMUKHI"):
                script_counts["Gurmukhi"] += 1
            elif name.startswith("ORIYA"):
                script_counts["Oriya"] += 1
            elif name.startswith("LATIN") or cat.startswith("L") and ord(ch) < 0x0250:
                script_counts["Latin"] += 1
            elif (
                name.startswith("CJK")
                or name.startswith("HANGUL")
                or name.startswith("HIRAGANA")
                or name.startswith("KATAKANA")
            ):
                script_counts["CJK"] += 1
            elif name.startswith("ARABIC"):
                script_counts["Arabic"] += 1
            elif name.startswith("CYRILLIC"):
                script_counts["Cyrillic"] += 1
            elif name.startswith("THAI"):
                script_counts["Thai"] += 1
            elif cat in ("Nd", "No", "Nl"):
                script_counts["Common"] += 1
            elif cat.startswith("P") or cat.startswith("S") or cat.startswith("Z"):
                script_counts["Common"] += 1
            elif cat.startswith("M"):
                script_counts["Mark"] += 1  # combining marks
            else:
                script_counts["Other"] += 1
        except ValueError:
            script_counts["Other"] += 1

    if not script_counts:
        return "Common"

    # Marks inherit from the base character script
    primary = script_counts.most_common(1)[0][0]
    if primary == "Mark" and len(script_counts) > 1:
        primary = script_counts.most_common(2)[1][0]

    return primary


def download_and_process_tekken(
    n_tokens: int = N_ENGLISH,
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Download Tekken tokenizer and extract top N English/EU/Code tokens.
    Returns (vocab_dict, merge_rules).
    """

    print("\n" + "=" * 60)
    print("  PHASE 3: Processing Tekken tokenizer")
    print("=" * 60)

    # Load Tekken via transformers (tekken.json has a non-standard format,
    # but transformers converts it to a proper BPE tokenizer with merges)
    t0 = time.time()
    print("  Loading Tekken via transformers...", flush=True)
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Base-2407")
    backend = hf_tok.backend_tokenizer
    tekken_data = json.loads(backend.to_str())
    print("  Loaded Tekken (transformers backend → BPE JSON)")

    # Extract vocab and merges from the backend BPE model
    tekken_model = tekken_data.get("model", {})
    tekken_vocab = tekken_model.get("vocab", {})
    tekken_merges = tekken_model.get("merges", [])

    print(f"  Tekken vocab size: {len(tekken_vocab):,}")
    print(f"  Tekken merge rules: {len(tekken_merges):,}")

    # Classify tokens by script
    script_counts: Counter = Counter()
    keep_tokens = []  # (token, rank)
    discard_tokens = []

    # Keep tokens based on script
    KEEP_SCRIPTS = {"Latin", "Common", "Mark", "Other"}
    DISCARD_SCRIPTS = {"CJK", "Arabic", "Cyrillic", "Thai"}

    for token, rank in sorted(tekken_vocab.items(), key=lambda x: x[1]):
        # Skip byte tokens — we add our own
        if token.startswith("<0x") and token.endswith(">"):
            continue
        # Skip special tokens
        if token.startswith("<") and token.endswith(">") and "|" in token:
            continue

        script = get_token_primary_script(token)
        script_counts[script] += 1

        if script in DISCARD_SCRIPTS:
            discard_tokens.append(token)
            continue

        # Skip Indic tokens from Tekken (we have our own)
        if script in INDIC_SCRIPTS:
            discard_tokens.append(token)
            continue

        # Check KE byte length constraint
        byte_len = len(token.encode("utf-8", errors="replace"))
        if byte_len > KE_POS_DIM:
            discard_tokens.append(token)
            continue

        keep_tokens.append((token, rank))

    # Sort by rank (lower = more frequent)
    keep_tokens.sort(key=lambda x: x[1])

    # Take top N
    selected = keep_tokens[:n_tokens]
    selected_set = {t for t, _ in selected}

    print("\n  Script distribution in Tekken:")
    for script, count in script_counts.most_common():
        print(f"    {script:15s}: {count:>6,}")
    print(f"\n  Selected {len(selected):,} English/EU/Code tokens (top {n_tokens:,})")
    print(
        f"  Discarded {len(discard_tokens):,} tokens (CJK/Arabic/Cyrillic/Thai/Indic/too-long)"
    )

    # Filter merge rules to only include selected tokens
    filtered_merges = []
    for merge in tekken_merges:
        # Merges can be "a b" strings or [a, b] lists depending on source
        if isinstance(merge, list):
            if len(merge) == 2:
                a, b = merge
            else:
                continue
        else:
            parts = merge.split(" ", 1)
            if len(parts) == 2:
                a, b = parts
            else:
                continue
        # Keep merge if both parts exist in our selected vocab
        # (they could be byte tokens too)
        merged = a + b
        if a in selected_set or a.startswith("<0x"):
            if b in selected_set or b.startswith("<0x"):
                if merged in selected_set:
                    filtered_merges.append((a, b))

    print(
        f"  Filtered merge rules: {len(filtered_merges):,} (from {len(tekken_merges):,})"
    )

    elapsed = time.time() - t0
    print(f"  Phase 3 complete in {elapsed:.1f}s")

    vocab_dict = {t: r for t, r in selected}
    return vocab_dict, filtered_merges


# ═══════════════════════════════════════════════════════════
# PHASE 4: ASSEMBLE FINAL TOKENIZER
# ═══════════════════════════════════════════════════════════


def extract_indic_tokens_and_merges(
    indic_bpe_path: str,
    n_tokens: int = N_INDIC,
    existing_tokens: Set[str] = None,
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Extract Indic tokens and merges from the trained BPE.
    Deduplicates against existing_tokens.
    """
    print("\n  Extracting Indic tokens from trained BPE...")

    with open(indic_bpe_path, "r", encoding="utf-8") as f:
        indic_data = json.load(f)

    indic_model = indic_data.get("model", {})
    indic_vocab = indic_model.get("vocab", {})
    indic_merges = indic_model.get("merges", [])

    print(f"  Raw Indic vocab: {len(indic_vocab):,}")
    print(f"  Raw Indic merges: {len(indic_merges):,}")

    existing = existing_tokens or set()

    # Filter: keep only tokens with Indic script content
    # Also keep Common script tokens that aren't already in English set
    keep = []
    for token, rank in sorted(indic_vocab.items(), key=lambda x: x[1]):
        # Skip byte tokens
        if token.startswith("<0x") and token.endswith(">"):
            continue
        # Skip if already in English set
        if token in existing:
            continue
        # Check KE constraint
        byte_len = len(token.encode("utf-8", errors="replace"))
        if byte_len > KE_POS_DIM:
            continue

        keep.append((token, rank))

    # Sort by rank and take top N
    keep.sort(key=lambda x: x[1])
    selected = keep[:n_tokens]
    selected_set = {t for t, _ in selected}

    print(f"  Selected {len(selected):,} Indic tokens (after dedup)")

    # Script distribution of selected tokens
    script_dist: Counter = Counter()
    for token, _ in selected:
        script = get_token_primary_script(token)
        script_dist[script] += 1

    print("  Script distribution:")
    for script, count in script_dist.most_common():
        print(f"    {script:15s}: {count:>6,}")

    # Filter merges
    filtered_merges = []
    for merge in indic_merges:
        # Merges can be "a b" strings or [a, b] lists depending on source
        if isinstance(merge, list):
            if len(merge) == 2:
                a, b = merge
            else:
                continue
        else:
            parts = merge.split(" ", 1)
            if len(parts) == 2:
                a, b = parts
            else:
                continue
        merged = a + b
        if (
            (a in selected_set or a.startswith("<0x"))
            and (b in selected_set or b.startswith("<0x"))
            and merged in selected_set
        ):
            filtered_merges.append((a, b))

    print(f"  Filtered Indic merges: {len(filtered_merges):,}")

    vocab_dict = {t: r for t, r in selected}
    return vocab_dict, filtered_merges


def assemble_tokenizer(
    english_vocab: Dict[str, int],
    english_merges: List[Tuple[str, str]],
    indic_vocab: Dict[str, int],
    indic_merges: List[Tuple[str, str]],
    output_dir: str,
) -> str:
    """
    Assemble the final tokenizer.json with all components.
    Token ID layout:
      0-255:         Special tokens
      256-511:       Byte tokens
      512-1023:      Math/LaTeX tokens
      1024-1279:     Reserved
      1280-76279:    English/EU/Code
      76280-131071:  Indic
    """
    print("\n" + "=" * 60)
    print("  PHASE 4: Assembling final tokenizer")
    print("=" * 60)
    t0 = time.time()

    os.makedirs(output_dir, exist_ok=True)

    # Build complete vocab with assigned IDs
    vocab = {}
    added_tokens = []
    current_id = 0

    # 1. Special tokens (0-255)
    print(f"  Adding {N_SPECIAL} special tokens (IDs 0-{N_SPECIAL-1})")
    for i, token in enumerate(SPECIAL_TOKENS):
        vocab[token] = current_id
        added_tokens.append(
            {
                "id": current_id,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
        current_id += 1

    # 2. Byte tokens (256-511)
    print(f"  Adding {N_BYTE} byte tokens (IDs {current_id}-{current_id+N_BYTE-1})")
    for token in BYTE_TOKENS:
        vocab[token] = current_id
        current_id += 1

    # 3. Math/LaTeX tokens (512-1023)
    n_math_actual = min(len(MATH_TOKENS), N_MATH)
    print(
        f"  Adding {n_math_actual} math/LaTeX tokens (IDs {current_id}-{current_id+N_MATH-1})"
    )
    for token in MATH_TOKENS:
        if token not in vocab:
            vocab[token] = current_id
            # Add math tokens as added_tokens so they match before BPE
            added_tokens.append(
                {
                    "id": current_id,
                    "content": token,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": False,
                }
            )
        current_id += 1
    # Pad remaining math slots
    while current_id < N_SPECIAL + N_BYTE + N_MATH:
        placeholder = f"<|math_reserved_{current_id}|>"
        vocab[placeholder] = current_id
        current_id += 1

    # 4. Reserved tokens (1024-1279)
    print(
        f"  Adding {N_RESERVED} reserved tokens (IDs {current_id}-{current_id+N_RESERVED-1})"
    )
    for i in range(N_RESERVED):
        token = f"<|reserved_{i}|>"
        vocab[token] = current_id
        added_tokens.append(
            {
                "id": current_id,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
        current_id += 1

    # 5. English/EU/Code tokens (1280-76279)
    english_start = current_id
    english_sorted = sorted(english_vocab.items(), key=lambda x: x[1])  # by rank
    n_eng_added = 0
    for token, _ in english_sorted:
        if token in vocab:
            continue  # skip duplicates with math/byte tokens
        vocab[token] = current_id
        current_id += 1
        n_eng_added += 1
        if n_eng_added >= N_ENGLISH:
            break
    print(
        f"  Added {n_eng_added:,} English/EU/Code tokens (IDs {english_start}-{current_id-1})"
    )

    # 6. Indic tokens (76280-131071)
    indic_start = current_id
    indic_sorted = sorted(indic_vocab.items(), key=lambda x: x[1])  # by rank
    n_indic_added = 0
    for token, _ in indic_sorted:
        if token in vocab:
            continue
        vocab[token] = current_id
        current_id += 1
        n_indic_added += 1
        if current_id >= TOTAL_VOCAB:
            break
    print(f"  Added {n_indic_added:,} Indic tokens (IDs {indic_start}-{current_id-1})")

    # Pad if we're short
    while current_id < TOTAL_VOCAB:
        token = f"<|pad_{current_id}|>"
        vocab[token] = current_id
        current_id += 1

    print(f"\n  Final vocab size: {len(vocab):,} (target: {TOTAL_VOCAB:,})")

    # Combine merge rules: English first, then Indic
    # (they don't interfere because pre-tokenizer splits on script boundaries)
    # IMPORTANT: Filter out merges involving literal space — the BPE merge format
    # uses space as separator ("a b"), so space-tokens cause parsing ambiguity.
    # These merges are meaningless anyway since pre-tokenizer splits on whitespace.
    all_merges = []
    skipped_space_merges = 0
    for a, b in english_merges:
        if " " in a or " " in b:
            skipped_space_merges += 1
            continue
        all_merges.append(f"{a} {b}")
    n_english_final = len(all_merges)
    for a, b in indic_merges:
        if " " in a or " " in b:
            skipped_space_merges += 1
            continue
        all_merges.append(f"{a} {b}")
    n_indic_final = len(all_merges) - n_english_final

    print(
        f"  Total merge rules: {len(all_merges):,} ({n_english_final:,} English + {n_indic_final:,} Indic)"
    )
    if skipped_space_merges > 0:
        print(
            f"  Skipped {skipped_space_merges:,} merges involving literal space tokens"
        )

    # Build tokenizer.json
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {
                "Regex": PRETOKENIZE_REGEX,
            },
            "behavior": "Isolated",
            "invert": False,
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {
                "<|begin_of_text|>": {
                    "id": "<|begin_of_text|>",
                    "ids": [0],
                    "tokens": ["<|begin_of_text|>"],
                },
            },
        },
        "decoder": {
            "type": "ByteFallback",
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<|unk|>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": all_merges,
        },
    }

    # Write tokenizer.json
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)
    print(f"\n  Wrote: {tokenizer_path} ({os.path.getsize(tokenizer_path)/1e6:.1f} MB)")

    # Write tokenizer_config.json
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "model_max_length": 131072,
        "clean_up_tokenization_spaces": False,
    }
    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Write special_tokens_map.json
    special_map = {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
    }
    stm_path = os.path.join(output_dir, "special_tokens_map.json")
    with open(stm_path, "w", encoding="utf-8") as f:
        json.dump(special_map, f, ensure_ascii=False, indent=2)

    print(f"  Wrote: {config_path}")
    print(f"  Wrote: {stm_path}")

    elapsed = time.time() - t0
    print(f"  Phase 4 complete in {elapsed:.1f}s")

    return tokenizer_path


# ═══════════════════════════════════════════════════════════
# PHASE 5: QUALITY CHECKS
# ═══════════════════════════════════════════════════════════

# Sample texts for testing (representative sentences)
TEST_TEXTS = {
    "English": "The quick brown fox jumps over the lazy dog. Machine learning models require careful evaluation.",
    "Hindi": "भारत एक विविधताओं से भरा देश है। यहाँ की संस्कृति बहुत समृद्ध है।",
    "Bengali": "বাংলাদেশ একটি দক্ষিণ এশিয়ার দেশ। এটি বিশ্বের সবচেয়ে ঘনবসতিপূর্ণ দেশগুলির মধ্যে একটি।",
    "Tamil": "தமிழ் ஒரு பழமையான மொழி. இது இந்தியாவின் அதிகாரப்பூர்வ மொழிகளில் ஒன்று.",
    "Telugu": "తెలుగు భాష భారతదేశంలో విస్తృతంగా మాట్లాడే భాషలలో ఒకటి. ఇది ద్రావిడ భాషా కుటుంబానికి చెందినది.",
    "Kannada": "ಕನ್ನಡ ಭಾಷೆ ಭಾರತದ ಪ್ರಮುಖ ಭಾಷೆಗಳಲ್ಲಿ ಒಂದು. ಇದು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಅಧಿಕೃತ ಭಾಷೆ.",
    "Malayalam": "മലയാളം ഭാഷ കേരളത്തിന്റെ ഔദ്യോഗിക ഭാഷയാണ്. ഇത് ദ്രാവിഡ ഭാഷാ കുടുംബത്തിൽ പെടുന്നു.",
    "Gujarati": "ગુજરાતી ભાષા ભારતની પ્રમુખ ભાષાઓમાંની એક છે. ગુજરાત રાજ્યની અધિકૃત ભાષા છે.",
    "Marathi": "मराठी भाषा महाराष्ट्राची अधिकृत भाषा आहे. ती देवनागरी लिपीत लिहिली जाते.",
    "Punjabi": "ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਪੰਜਾਬ ਦੀ ਮੁੱਖ ਭਾਸ਼ਾ ਹੈ। ਇਹ ਗੁਰਮੁਖੀ ਲਿਪੀ ਵਿੱਚ ਲਿਖੀ ਜਾਂਦੀ ਹੈ।",
    "Odia": "ଓଡ଼ିଆ ଭାଷା ଓଡ଼ିଶା ରାଜ୍ୟର ସରକାରୀ ଭାଷା। ଏହା ଏକ ଶାସ୍ତ୍ରୀୟ ଭାଷା ମାନ୍ୟତା ପ୍ରାପ୍ତ।",
    "Assamese": "অসমীয়া ভাষা অসমৰ চৰকাৰী ভাষা। ই পূব ভাৰতৰ এটা প্ৰধান ভাষা।",
    "French": "La France est un pays d'Europe occidentale. Sa capitale est Paris, la ville lumière.",
    "Spanish": "España es un país situado en el sur de Europa. Madrid es su capital.",
    "German": "Deutschland ist ein Land in Mitteleuropa. Berlin ist die Hauptstadt des Landes.",
    "Python": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Math": "The equation \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} gives the roots of ax^2 + bx + c = 0",
    "LaTeX": "\\sum_{i=1}^{n} \\frac{1}{i^2} = \\frac{\\pi^2}{6}",
    "Code-switch": "This is बहुत अच्छा, really good! মনে হয় এটা ভালো হবে।",
}


def run_quality_checks(tokenizer_path: str) -> bool:
    """
    Run comprehensive quality checks on the assembled tokenizer.
    Returns True if all checks pass.
    """
    from transformers import PreTrainedTokenizerFast

    print("\n" + "=" * 60)
    print("  PHASE 5: Quality Checks")
    print("=" * 60)

    # Load via PreTrainedTokenizerFast (handles the regex properly)
    output_dir = os.path.dirname(tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"\n  Vocab size: {vocab_size:,} (target: {TOTAL_VOCAB:,})")

    all_pass = True

    # Check 1: Vocab size
    if vocab_size != TOTAL_VOCAB:
        print(f"  FAIL: Vocab size {vocab_size} != {TOTAL_VOCAB}")
        all_pass = False
    else:
        print("  PASS: Vocab size correct")

    # Check 2: Fertility test
    print("\n  Fertility test (tokens per word):")
    print(
        f"  {'Language':15s} {'Tokens':>7s} {'Words':>6s} {'Fertility':>10s} {'Byte%':>6s} {'Status':>8s}"
    )
    print(f"  {'-'*55}")

    byte_token_ids = set(range(N_SPECIAL, N_SPECIAL + N_BYTE))  # IDs 256-511

    for lang, text in TEST_TEXTS.items():
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(token_ids)
        words = text.split()
        n_words = len(words)
        fertility = n_tokens / n_words if n_words > 0 else 0

        # Count byte token usage
        n_byte_tokens = sum(1 for tid in token_ids if tid in byte_token_ids)
        byte_pct = (n_byte_tokens / n_tokens * 100) if n_tokens > 0 else 0

        status = "OK"
        if fertility > 5.0:
            status = "WARN"
        if byte_pct > 5.0:
            status = "BYTE!"
            all_pass = False

        print(
            f"  {lang:15s} {n_tokens:>7d} {n_words:>6d} {fertility:>10.2f} {byte_pct:>5.1f}% {status:>8s}"
        )

    # Check 3: KE byte length constraint
    print(f"\n  KE byte length check (all tokens must be <= {KE_POS_DIM} bytes):")
    vocab = tokenizer.get_vocab()  # works on PreTrainedTokenizerFast too
    long_tokens = []
    for token, tid in vocab.items():
        # Skip special/reserved tokens
        if token.startswith("<|") or token.startswith("<0x"):
            continue
        byte_len = len(token.encode("utf-8", errors="replace"))
        if byte_len > KE_POS_DIM:
            long_tokens.append((token, byte_len, tid))

    if long_tokens:
        print(f"  FAIL: {len(long_tokens)} tokens exceed {KE_POS_DIM} bytes:")
        for token, blen, tid in long_tokens[:10]:
            print(f"    ID {tid}: {blen} bytes: {repr(token[:50])}")
        all_pass = False
    else:
        print(f"  PASS: All tokens <= {KE_POS_DIM} bytes")

    # Check 4: Cross-script tokens
    print("\n  Cross-script token check:")
    cross_script = []
    for token, tid in vocab.items():
        if token.startswith("<|") or token.startswith("<0x"):
            continue
        scripts = set()
        for ch in token:
            name = unicodedata.name(ch, "")
            if name.startswith("LATIN"):
                scripts.add("Latin")
            elif any(name.startswith(s.upper()) for s in INDIC_SCRIPTS):
                scripts.add("Indic")
        if "Latin" in scripts and "Indic" in scripts:
            cross_script.append((token, tid))

    if cross_script:
        print(f"  WARN: {len(cross_script)} cross-script tokens found:")
        for token, tid in cross_script[:10]:
            print(f"    ID {tid}: {repr(token)}")
    else:
        print("  PASS: No cross-script tokens")

    # Check 5: Round-trip test
    print("\n  Round-trip test (encode → decode):")
    roundtrip_pass = True
    for lang, text in TEST_TEXTS.items():
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        # Normalize whitespace for comparison
        if decoded.strip() != text.strip():
            print(f"  FAIL: {lang}")
            print(f"    Original: {text[:80]}")
            print(f"    Decoded:  {decoded[:80]}")
            roundtrip_pass = False

    if roundtrip_pass:
        print("  PASS: All round-trips successful")

    # Summary
    print(f"\n  {'=' * 40}")
    if all_pass:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — review above")
    print(f"  {'=' * 40}")

    return all_pass


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Build 131K Kronecker-aware tokenizer")
    parser.add_argument(
        "--data-dir",
        default="./indic_tokenizer_samples_by_size",
        help="Directory with downloaded Indic parquet files",
    )
    parser.add_argument(
        "--output-dir", default="./output", help="Output directory for final tokenizer"
    )
    parser.add_argument(
        "--work-dir", default="./work", help="Working directory for intermediate files"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of parallel workers for data extraction",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip Phase 1 (text extraction) if already done",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip Phase 2 (BPE training) if already done",
    )
    args = parser.parse_args()

    # Force unbuffered stdout for real-time logging
    sys.stdout.reconfigure(line_buffering=True)

    t_start = time.time()
    print("=" * 60)
    print("  BUILD 131K KRONECKER-AWARE TOKENIZER")
    print(f"  Target: {TOTAL_VOCAB:,} tokens")
    print(
        f"  Layout: {N_SPECIAL} special + {N_BYTE} byte + {N_MATH} math + "
        f"{N_RESERVED} reserved + {N_ENGLISH:,} English + {N_INDIC:,} Indic"
    )
    print(f"  Time: {time.strftime('%H:%M:%S')}")
    print("=" * 60, flush=True)

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    text_dir = os.path.join(args.work_dir, "texts")
    indic_bpe_path = os.path.join(args.work_dir, "indic_bpe_raw.json")

    # ── Phase 1: Extract text ──
    if not args.skip_extract:
        texts_by_source = extract_texts_parallel(args.data_dir, args.num_workers)
        text_files = prepare_training_text(texts_by_source, text_dir)
    else:
        print("\n  Skipping Phase 1 (--skip-extract)")
        text_files = sorted(Path(text_dir).glob("*.txt"))
        text_files = [str(f) for f in text_files]
        print(f"  Found {len(text_files)} text files in {text_dir}")

    # ── Phase 2: Train Indic BPE ──
    if not args.skip_train:
        indic_bpe_path = train_indic_bpe(
            text_files,
            output_path=indic_bpe_path,
            num_threads=args.num_workers,
        )
    else:
        print("\n  Skipping Phase 2 (--skip-train)")
        print(f"  Using existing: {indic_bpe_path}")

    # ── Phase 3: Download & process Tekken ──
    english_vocab, english_merges = download_and_process_tekken()

    # ── Phase 4: Assemble ──
    english_token_set = set(english_vocab.keys())
    indic_vocab, indic_merges = extract_indic_tokens_and_merges(
        indic_bpe_path,
        existing_tokens=english_token_set | set(MATH_TOKENS) | set(BYTE_TOKENS),
    )

    tokenizer_path = assemble_tokenizer(
        english_vocab,
        english_merges,
        indic_vocab,
        indic_merges,
        args.output_dir,
    )

    # ── Phase 5: Quality checks ──
    run_quality_checks(tokenizer_path)

    elapsed = time.time() - t_start
    print(f"\n  Total build time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
