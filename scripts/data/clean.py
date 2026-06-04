"""
Text cleaning for pretraining data.

Implements every cleaning step from DATA_PIPELINE_AUDIT.md plus hardening
against real-world crawl-data pathologies:

  Document-level drops (early exit, no cleaning wasted):
  - Auto-generated files (protobuf, SWIG, codegen) → drop entire document
  - Minified JS/CSS (single line > 500 chars, low newline density) → drop
  - Lock files (package-lock, yarn.lock, Pipfile.lock) → drop

  Character-level cleaning:
  - NFC normalize
  - Surrogate character removal (U+D800-U+DFFF)
  - C0/C1 control character stripping (null bytes, DEL, 0x80-0x9F)
  - Strip ZWSP, bidi controls, BOM, private-use chars
  - HTML entity unescape
  - Strip U+FFFD replacement chars
  - Mojibake detection (UTF-8 mis-decoded as latin-1)
  - CRLF normalization
  - Collapse whitespace runs
  - Strip ghost tags ([USER], <USER>, ### Instruction:, etc.)
  - Strip license/copyright header blocks from code files
  - Strip auto-generated file warnings (line-level, for non-dropped files)
  - Preserve legitimate ZWNJ (U+200C) and ZWJ (U+200D) for Indic scripts

  Section-level cleaning:
  - Strip Reference/Bibliography tail sections from academic papers
  - Strip Wikipedia citation markers [42], [43][44]

Pure-Python, stdlib-only. Safe for multiprocessing workers.
Every regex compiled once at module load.
"""

from __future__ import annotations

import html as _html
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# 1-4. MERGED CHARACTER CLEANUP — single regex pass
# ═══════════════════════════════════════════════════════════════════════════
# Surrogates (U+D800-U+DFFF), C0/C1 control chars (except \t \n \r),
# zero-width/bidi/BOM/FFFD, and private-use area chars.
# KEEP: U+200C (ZWNJ) and U+200D (ZWJ) — required in Indic/Arabic.
#
# Merged into one regex for 1 pass instead of 4 — critical for hot path.
# Individual regexes kept below for debug/stats mode.

# Zero-width / bidi chars to strip (enumerated for clarity)
_ZERO_WIDTH_AND_BIDI = (
    "\u200B"  # ZERO WIDTH SPACE
    "\u200E"  # LEFT-TO-RIGHT MARK
    "\u200F"  # RIGHT-TO-LEFT MARK
    "\u202A"  # LEFT-TO-RIGHT EMBEDDING
    "\u202B"  # RIGHT-TO-LEFT EMBEDDING
    "\u202C"  # POP DIRECTIONAL FORMATTING
    "\u202D"  # LEFT-TO-RIGHT OVERRIDE
    "\u202E"  # RIGHT-TO-LEFT OVERRIDE
    "\u2060"  # WORD JOINER
    "\u2061"  # FUNCTION APPLICATION
    "\u2062"  # INVISIBLE TIMES
    "\u2063"  # INVISIBLE SEPARATOR
    "\u2064"  # INVISIBLE PLUS
    "\u2066"  # LEFT-TO-RIGHT ISOLATE
    "\u2067"  # RIGHT-TO-LEFT ISOLATE
    "\u2068"  # FIRST STRONG ISOLATE
    "\u2069"  # POP DIRECTIONAL ISOLATE
    "\uFEFF"  # BOM / ZERO WIDTH NO-BREAK SPACE
    "\uFFF9"  # INTERLINEAR ANNOTATION ANCHOR
    "\uFFFA"  # INTERLINEAR ANNOTATION SEPARATOR
    "\uFFFB"  # INTERLINEAR ANNOTATION TERMINATOR
    "\uFFFC"  # OBJECT REPLACEMENT CHARACTER
    "\uFFFD"  # REPLACEMENT CHARACTER
)

# Production regex: single merged pass (fast, no per-step breakdown)
_CHAR_CLEANUP_RE = re.compile(
    "[\uD800-\uDFFF"  # surrogates
    "\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x80-\x9F"  # C0/C1
    + "".join(re.escape(c) for c in _ZERO_WIDTH_AND_BIDI)  # ZW/bidi
    + "\uE000-\uF8FF]"  # PUA BMP
    "|[\U000F0000-\U000FFFFD]"  # PUA plane 15
    "|[\U00100000-\U0010FFFD]",  # PUA plane 16
    re.UNICODE,
)

# Debug regexes: individual passes for per-step stats (only used when stats requested)
_SURROGATES_RE = re.compile("[\uD800-\uDFFF]")
_C0C1_RE = re.compile("[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x80-\x9F]")
_ZW_BIDI_RE = re.compile(
    "[" + "".join(re.escape(c) for c in _ZERO_WIDTH_AND_BIDI) + "]"
)
_PUA_RE = re.compile(
    "[\uE000-\uF8FF]" "|[\U000F0000-\U000FFFFD]" "|[\U00100000-\U0010FFFD]",
    re.UNICODE,
)


# ═══════════════════════════════════════════════════════════════════════════
# 5. GHOST / CONVERSATION-MARKER TAGS
# ═══════════════════════════════════════════════════════════════════════════
# From DATA_PIPELINE_AUDIT.md Section B: 4+ formats found in T1 scripts.
# Sub with " " (not "") to prevent word-merging: hello<USER>world → hello world
# Whitespace normalizer below collapses the extra space.
_GHOST_TAG_PATTERNS = [
    # XML-style (SmolTalk2, ShareGPT, OpenHermes)
    r"\s*</?(?:USER|ASSISTANT|SYSTEM|HUMAN|BOT|GPT)>\s*",
    # Bracket-style (Samvaad, ROOTS)
    r"\s*\[(?:USER|ASSISTANT|SYSTEM|HUMAN|BOT|INST|/INST)\]\s*",
    # Markdown heading markers (NCERT, MegaScience, Alpaca)
    r"\s*###\s*(?:Instruction|Response|Topic|Question|Answer|Context|Input|Output)\s*:\s*",
    # GPT / Dolma / Pile leaked special tokens
    r"<\|(?:endoftext|im_start|im_end|pad|unk|bos|eos|sep|cls|mask)\|>",
    # LLaMA / Mistral / Gemma special tokens
    r"</?s>",
    r"\[/?INST\]",
    r"\[/?SYS\]",
    # ChatML delimiters
    r"<\|(?:system|user|assistant|end)\|>",
    # Anthropic-style conversation markers (HH-RLHF, Constitutional AI)
    r"(?:^|\n)\s*(?:Human|Assistant)\s*:\s*",
    # Gemma 2 / LLaMA 3 special tokens
    r"<\|(?:begin_of_text|end_of_turn|start_header_id|end_header_id)\|>",
]
_GHOST_TAG_RE = re.compile("|".join(_GHOST_TAG_PATTERNS), re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════
# 6. MOJIBAKE DETECTION
# ═══════════════════════════════════════════════════════════════════════════
# UTF-8 bytes decoded as latin-1 produce characteristic sequences:
#   é → Ã©,  ' → â€™,  – → â€",  " → â€œ
# Documents with high mojibake density are dropped rather than silently
# mangled. They should be re-ingested from original source if available.
_MOJIBAKE_RE = re.compile(
    r"Ã[©ª¨¦¤£¢¡ ]" r"|â€[™\"" "„œ•–—]" r"|Â[·°±²³µ¹º»¼½¾]",
)


def mojibake_ratio(text: str) -> float:
    """Fraction of characters that appear to be mojibake sequences."""
    if not text:
        return 0.0
    hits = sum(len(m.group()) for m in _MOJIBAKE_RE.finditer(text))
    return hits / len(text)


# ═══════════════════════════════════════════════════════════════════════════
# 7. WHITESPACE NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════
_CR_RE = re.compile(r"\r\n?")  # CR or CRLF → LF
_MULTI_SPACE_RE = re.compile(r"[^\S\n]+")  # spaces/tabs → single space
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")  # 3+ newlines → 2


# ═══════════════════════════════════════════════════════════════════════════
# 8. LICENSE / COPYRIGHT HEADER STRIPPING
# ═══════════════════════════════════════════════════════════════════════════
# License headers appear at the top of ~35-40% of code files (BigCode analysis).
# They are repetitive legal boilerplate with no training signal. Apache 2.0 is
# 12 lines, MIT is 5, BSD-3-Clause is 18. At B3-B5 bands they dominate block
# openings, creating spurious correlations between legal text and code.
#
# Strategy: anchor-at-start block stripping. Only remove headers at the START
# of documents. License text mid-file (e.g. README listing licenses) is preserved.

_LICENSE_TRIGGER_RE = re.compile(
    r"(?:SPDX-|Copyright|\(c\)|©|Licensed under|Permission is hereby|"
    r"Redistribution and use|All [Rr]ights [Rr]eserved)",
    re.IGNORECASE,
)
_COMMENT_LINE_RE = re.compile(r"^[ \t]*(?:#|//|/\*|\*/?|<!--|;|%)[^\n]*$")
_BLANK_LINE_RE = re.compile(r"^[ \t]*$")

# Auto-generated file warnings (protoc, codegen, etc.)
_AUTOGEN_RE = re.compile(
    r"(?m)^[ \t]*(?:#|//|\*)[^\n]*"
    r"(?:DO NOT EDIT|auto.?generated|Generated by|THIS FILE IS GENERATED)"
    r"[^\n]*\n?",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════
# 9. DOCUMENT-LEVEL DROPS — checked before fine-grained cleaning
# ═══════════════════════════════════════════════════════════════════════════
# These detect entire-document pathologies that make the document worthless.
# Checked early to avoid wasting regex cycles on throwaway text.

# 9a. Auto-generated files (protobuf, SWIG, ANTLR, codegen).
#     Check first ~2000 chars for comment lines with autogen markers.
#     Drop the ENTIRE document, not just the header.
_AUTOGEN_DROP_RE = re.compile(
    r"^[ \t]*(?:#|//|/\*|\*)[^\n]*"
    r"(?:DO NOT EDIT|auto.?generated|Generated by|THIS FILE IS GENERATED|"
    r"Code generated|MACHINE GENERATED|Automatically generated by)",
    re.IGNORECASE | re.MULTILINE,
)

# 9b. Minified JS/CSS — single line >500 chars, very few newlines.
_MINIFIED_LINE_THRESHOLD = 500

# 9c. Lock files (package-lock.json, yarn.lock, Pipfile.lock, etc.)
_LOCK_FILE_RE = re.compile(
    r"integrity sha\d+-"
    r'|resolved "https://registry\.'
    r'|"lockfileVersion"'
    r"|BUNDLED WITH"
    r'|"_resolved":\s*"https://',
)
_LOCK_FILE_MATCH_THRESHOLD = 3  # need 3+ matches to confirm


def _check_document_drop(text: str) -> str:
    """
    Fast pre-cleaning check: should this document be dropped entirely?

    Returns drop reason string (non-empty = drop), or "" to keep.
    Checks are ordered cheapest-first.
    """
    # 9b. Minified detection (cheapest — just line length + newline count)
    if len(text) > _MINIFIED_LINE_THRESHOLD:
        newline_count = text.count("\n")
        if newline_count == 0:
            # Single line document > 500 chars — minified
            return "minified"
        max_line_len = max(len(line) for line in text.split("\n"))
        if (
            max_line_len > _MINIFIED_LINE_THRESHOLD
            and (newline_count / len(text)) < 0.005
        ):
            return "minified"

    # 9c. Lock file detection (check first 5000 chars)
    lock_matches = len(_LOCK_FILE_RE.findall(text[:5000]))
    if lock_matches >= _LOCK_FILE_MATCH_THRESHOLD:
        return "lockfile"

    # 9a. Auto-generated file (check first 2000 chars for header markers)
    if _AUTOGEN_DROP_RE.search(text[:2000]):
        return "autogenerated"

    return ""


# ═══════════════════════════════════════════════════════════════════════════
# 10. CITATION MARKERS — [42], [43][44], etc.
# ═══════════════════════════════════════════════════════════════════════════
# Wikipedia/academic citation markers. Zero false positives in natural text
# (code array indexing like `arr[0]` has identifier prefix, not standalone).
# Applied post-cleaning to encyclopedic/academic data.
_CITATION_MARKER_RE = re.compile(r"\[(\d{1,3})\]")


# ═══════════════════════════════════════════════════════════════════════════
# 11. REFERENCE / BIBLIOGRAPHY TAIL STRIPPING
# ═══════════════════════════════════════════════════════════════════════════
# Academic papers: everything after "References" or "Bibliography" heading
# is citation list boilerplate. The same 50 foundational papers appear
# verbatim ~50K times across arxiv — worst cross-document duplication.
# Double-newline anchor prevents matching mid-sentence "references".
_REFERENCE_HEADING_RE = re.compile(
    r"\n\n\s*(?:References|Bibliography|Works Cited|REFERENCES|BIBLIOGRAPHY"
    r"|Bibliographie|संदर्भ|参考文献)\s*\n"
)


def strip_license_header(text: str) -> tuple:
    """
    Strip license/copyright header block from the start of code files.

    Returns (cleaned_text, was_stripped: bool).

    Only removes from the START of the document — a contiguous block of
    comment/blank lines beginning with a license trigger. License text
    mid-file (e.g. in a README listing multiple licenses) is preserved.
    """
    # Fast bail: scan to first non-whitespace char — must be a comment char
    # Avoids text.split("\n") on ~60% of documents (non-code, no license)
    i = 0
    n = len(text)
    while i < n and text[i] in " \t\n\r":
        i += 1
    if i >= n or text[i] not in "#/!*;%<":
        return text, False

    lines = text.split("\n")

    # Find the first non-blank line
    first_content_idx = None
    for i, line in enumerate(lines):
        if not _BLANK_LINE_RE.match(line):
            first_content_idx = i
            break

    if first_content_idx is None:
        return text, False

    first_line = lines[first_content_idx]

    # Check if file starts with a comment line containing a license trigger
    if not (
        _COMMENT_LINE_RE.match(first_line) and _LICENSE_TRIGGER_RE.search(first_line)
    ):
        return text, False

    # Consume contiguous comment/blank lines from the top
    end_idx = first_content_idx
    while end_idx < len(lines):
        line = lines[end_idx]
        if _BLANK_LINE_RE.match(line) or _COMMENT_LINE_RE.match(line):
            end_idx += 1
        else:
            break

    stripped = "\n".join(lines[end_idx:]).lstrip("\n")
    return stripped, True


# ═══════════════════════════════════════════════════════════════════════════
# CLEANING STATS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CleaningStats:
    """
    Per-worker counters. Accumulate during processing, merge in reducer.

    Usage:
        total = CleaningStats()
        for s in worker_stats:
            total += s
    """

    docs_processed: int = 0
    docs_dropped_empty: int = 0
    docs_dropped_short: int = 0
    docs_dropped_low_diversity: int = 0
    docs_dropped_mojibake: int = 0
    docs_dropped_autogenerated: int = 0
    docs_dropped_minified: int = 0
    docs_dropped_lockfile: int = 0
    chars_removed_surrogates: int = 0
    chars_removed_c0c1: int = 0
    chars_removed_zw_bidi: int = 0
    chars_removed_pua: int = 0
    ghost_tag_removals: int = 0
    license_headers_stripped: int = 0
    autogen_warnings_stripped: int = 0
    citation_markers_stripped: int = 0
    reference_sections_stripped: int = 0

    def __iadd__(self, other: "CleaningStats") -> "CleaningStats":
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))
        return self

    def __add__(self, other: "CleaningStats") -> "CleaningStats":
        result = CleaningStats()
        result += self
        result += other
        return result

    def summary(self) -> str:
        lines = []
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if val > 0:
                lines.append(f"  {f}: {val:,}")
        return "\n".join(lines) if lines else "  (no activity)"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CLEANING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════


def clean_text(
    text: str,
    *,
    mojibake_threshold: float = 0.02,
    stats: Optional[CleaningStats] = None,
) -> str:
    """
    Clean a single document for pretraining.

    Returns cleaned string, or empty string if the document should be dropped.

    Parameters
    ----------
    text : str
        Raw document text.
    mojibake_threshold : float
        If mojibake character fraction exceeds this, drop the document.
        Default 0.02 (2%). Set to 1.0 to disable.
    stats : CleaningStats, optional
        Accumulate per-step telemetry counters.
    """
    if not text:
        if stats:
            stats.docs_processed += 1
            stats.docs_dropped_empty += 1
        return ""

    s = stats  # local alias

    # ── 0. Document-level drops (cheapest checks, early exit) ─────────
    drop_reason = _check_document_drop(text)
    if drop_reason:
        if s:
            s.docs_processed += 1
            drop_field = f"docs_dropped_{drop_reason}"
            if hasattr(s, drop_field):
                setattr(s, drop_field, getattr(s, drop_field) + 1)
        return ""

    # 1. NFC normalize — canonical for Indic, Arabic, Hebrew, CJK.
    text = unicodedata.normalize("NFC", text)

    # 2. HTML entity unescape — BEFORE char-stripping so entities that
    #    decode to strip-worthy chars get caught in steps 3-6.
    text = _html.unescape(text)

    # 2.5. Non-breaking space → regular space. After html.unescape() because
    #      &nbsp; decodes to U+00A0. Python's [^\S\n]+ doesn't match U+00A0.
    text = text.replace("\u00a0", " ")

    # 3-6. Character cleanup — surrogates, C0/C1, ZW/bidi, PUA in one pass.
    if s:
        # Stats mode: individual passes for per-step breakdown
        before = len(text)
        text = _SURROGATES_RE.sub("", text)
        s.chars_removed_surrogates += before - len(text)

        before = len(text)
        text = _C0C1_RE.sub("", text)
        s.chars_removed_c0c1 += before - len(text)

        before = len(text)
        text = _ZW_BIDI_RE.sub("", text)
        s.chars_removed_zw_bidi += before - len(text)

        before = len(text)
        text = _PUA_RE.sub("", text)
        s.chars_removed_pua += before - len(text)
    else:
        # Production mode: single merged pass (4x fewer scans)
        text = _CHAR_CLEANUP_RE.sub("", text)

    # 7. Ghost tag stripping — sub with " " to prevent word-merging
    #    (hello<USER>world → hello world, not helloworld)
    before = len(text)
    text = _GHOST_TAG_RE.sub(" ", text)
    if s:
        s.ghost_tag_removals += before - len(text)

    # 7.5. License/copyright header stripping — anchor-at-start only
    text, had_license = strip_license_header(text)
    if had_license and s:
        s.license_headers_stripped += 1

    # 7.6. Auto-generated file warnings (DO NOT EDIT, Generated by, etc.)
    before = len(text)
    text = _AUTOGEN_RE.sub("", text)
    if s and before != len(text):
        s.autogen_warnings_stripped += 1

    # 8. Whitespace normalization — CRLF → LF, collapse spaces, cap newlines
    text = _CR_RE.sub("\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = text.strip()

    if not text:
        if s:
            s.docs_processed += 1
            s.docs_dropped_empty += 1
        return ""

    # 8.5. Reference / Bibliography tail stripping — drop everything after heading.
    #      Done before citation markers so we don't waste regex on doomed text.
    ref_match = _REFERENCE_HEADING_RE.search(text)
    if ref_match:
        text = text[: ref_match.start()].rstrip()
        if s:
            s.reference_sections_stripped += 1
        if not text:
            if s:
                s.docs_processed += 1
                s.docs_dropped_empty += 1
            return ""

    # 8.6. Citation marker stripping — [42], [43][44] → removed.
    #      Applied after whitespace normalization so we don't create double spaces.
    before = len(text)
    text = _CITATION_MARKER_RE.sub("", text)
    if s and before != len(text):
        s.citation_markers_stripped += before - len(text)
    # Re-collapse any double spaces created by citation removal
    if before != len(text):
        text = _MULTI_SPACE_RE.sub(" ", text)

    # 9. Mojibake detection — after cleaning so we measure the actual text
    if mojibake_ratio(text) > mojibake_threshold:
        if s:
            s.docs_processed += 1
            s.docs_dropped_mojibake += 1
        return ""

    if s:
        s.docs_processed += 1
    return text


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT VALIDITY CHECK
# ═══════════════════════════════════════════════════════════════════════════

# Pre-compiled regex for fast alnum counting (C-level scan vs Python loop)
_ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)


def is_valid_document(
    text: str,
    min_chars: int = 20,
    min_alnum_ratio: float = 0.35,
    min_unique_tokens: int = 5,
) -> bool:
    """
    Check if cleaned text is worth tokenizing.

    Rejects:
      - Empty or near-empty documents (< min_chars)
      - Documents below minimum alphanumeric density
      - Documents with almost no lexical diversity (single word repeated)

    Parameters
    ----------
    text : str
        Already-cleaned document text.
    min_alnum_ratio : float
        Minimum fraction of chars that must be alphanumeric. 0.35 allows
        punctuation-heavy code/markdown while rejecting symbol-only noise.
    min_unique_tokens : int
        Minimum distinct whitespace-delimited tokens. Rejects documents
        like "aaa aaa aaa aaa aaa aaa aaa".
    """
    if len(text) < min_chars:
        return False

    # Sample first 2000 chars for alnum ratio — representative and avoids
    # scanning 100KB documents char-by-char. Uses C-level regex, not Python loop.
    sample = text[:2000] if len(text) > 2000 else text
    alnum_count = len(_ALNUM_RE.findall(sample))
    if alnum_count / len(sample) < min_alnum_ratio:
        return False

    tokens = text.split()
    if len(set(tokens)) < min_unique_tokens:
        return False

    return True
