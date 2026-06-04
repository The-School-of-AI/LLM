#!/usr/bin/env python3
"""
T24 — Unified Kronecker-constraint verifier across pre-tokenizer types.

Handles tokenizers using any pre-tokenizer encoding (GPT-2 ByteLevel,
SentencePiece Metaspace, plain Split, etc.) by deferring to each
tokenizer's OWN decoder — no per-pre-tokenizer byte-mapping assumed.

Metrics per tokenizer:
  - vocab size, normal/special/bytefallback split
  - pre-tokenizer type (auto-detected from tokenizer.json or class name)
  - max UTF-8 byte length of any normal token (after decode → utf-8)
  - count of normal tokens with byte length > 32 (the KE POS_DIM ceiling)
  - count of normal tokens spanning ≥2 disjoint non-Common scripts

Supports two backends:
  - HF AutoTokenizer (tokenizer.decode([id], ...))
  - tiktoken (enc.decode_single_token_bytes(i))

Usage:
  python verify_kronecker_constraints_unified.py             # canonical run
  python verify_kronecker_constraints_unified.py PATH        # single tokenizer

Exit 0 on completion. Reports per-tokenizer; never fails on a single load
error — logs and skips that tokenizer.
"""

import json
import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


# --- Unicode script ranges (same as verify_no_cross_script_merges.py) ---
INDIC_RANGES = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Oriya": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Arabic": (0x0600, 0x06FF),
    "Sinhala": (0x0D80, 0x0DFF),
}


def script_of_char(ch):
    cp = ord(ch)
    for s, (lo, hi) in INDIC_RANGES.items():
        if lo <= cp <= hi:
            return s
    n = unicodedata.name(ch, "")
    if not n:
        return None
    if n.startswith("LATIN"):
        return "Latin"
    if (
        n.startswith("CJK")
        or n.startswith("HIRAGANA")
        or n.startswith("KATAKANA")
        or n.startswith("HANGUL")
    ):
        return "CJK"
    if n.startswith("CYRILLIC"):
        return "Cyrillic"
    if n.startswith("THAI"):
        return "Thai"
    if n.startswith("ARABIC"):
        return "Arabic"
    if n.startswith("HEBREW"):
        return "Hebrew"
    if n.startswith("GREEK"):
        return "Greek"
    return None


def text_scripts(s):
    return {sc for ch in s if (sc := script_of_char(ch)) is not None}


# --- Pre-tokenizer detection ---
def detect_pre_tokenizer(token_json_path_or_dir):
    """Inspect tokenizer.json to detect ByteLevel / Metaspace / Split / ..."""
    if token_json_path_or_dir is None:
        return "unknown"
    p = Path(token_json_path_or_dir)
    if p.is_dir():
        cand = p / "tokenizer.json"
        if not cand.exists():
            return "unknown"
        p = cand
    if not p.exists():
        return "unknown"
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return "unknown"
    pre = d.get("pre_tokenizer")
    if pre is None:
        return "None"
    if isinstance(pre, dict):
        t = pre.get("type")
        if t in ("ByteLevel", "Metaspace", "Split", "Whitespace"):
            return t
        if "pretokenizers" in pre:
            sub_types = [s.get("type") for s in pre.get("pretokenizers", [])]
            if "ByteLevel" in sub_types:
                return "ByteLevel"
            return "Sequence(" + "+".join(sub_types) + ")"
        return str(t)
    return str(pre)


# --- HF backend ---
def measure_hf(tok, max_ids=None):
    """Measure max byte / >32 / cross-script using tokenizer's own decoder."""
    vocab = tok.get_vocab()
    special_ids = set(getattr(tok, "all_special_ids", []) or [])
    n_normal = 0
    n_special = 0
    n_bytefallback = 0
    n_skipped = 0
    max_byte = 0
    max_byte_token = None
    max_byte_text = None
    n_over_32 = 0
    n_cross = 0
    cross_examples = []
    byte_fallback_re = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")

    ids = sorted(vocab.values())
    if max_ids is not None:
        ids = ids[:max_ids]
    for tid in ids:
        if tid in special_ids:
            n_special += 1
            continue
        # Some tokenizers have token strings; check byte-fallback by name
        try:
            tok_str = tok.convert_ids_to_tokens(tid)
        except Exception:
            tok_str = ""
        if tok_str and byte_fallback_re.match(tok_str):
            n_bytefallback += 1
            continue
        try:
            text = tok.decode(
                [tid], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        except Exception:
            n_skipped += 1
            continue
        if text is None:
            n_skipped += 1
            continue
        n_normal += 1
        bl = len(text.encode("utf-8"))
        if bl > max_byte:
            max_byte = bl
            max_byte_token = tok_str
            max_byte_text = text
        if bl > 32:
            n_over_32 += 1
        scripts = text_scripts(text)
        if len(scripts) >= 2:
            n_cross += 1
            if len(cross_examples) < 6:
                cross_examples.append((tok_str, text, sorted(scripts)))
    return {
        "vocab_size": len(vocab),
        "normal": n_normal,
        "special": n_special,
        "bytefallback": n_bytefallback,
        "skipped": n_skipped,
        "max_byte": max_byte,
        "max_byte_token": max_byte_token,
        "max_byte_text": max_byte_text,
        "n_over_32": n_over_32,
        "n_cross": n_cross,
        "cross_examples": cross_examples,
    }


# --- tiktoken backend (o200k_base) ---
def measure_tiktoken(enc_name):
    import tiktoken

    enc = tiktoken.get_encoding(enc_name)
    n_normal = 0
    n_skipped = 0
    max_byte = 0
    max_byte_text = None
    max_byte_id = None
    n_over_32 = 0
    n_cross = 0
    cross_examples = []
    for tid in range(enc.n_vocab):
        try:
            b = enc.decode_single_token_bytes(tid)
        except KeyError:
            n_skipped += 1
            continue
        n_normal += 1
        text = b.decode("utf-8", errors="replace")
        bl = len(b)
        if bl > max_byte:
            max_byte = bl
            max_byte_id = tid
            max_byte_text = text
        if bl > 32:
            n_over_32 += 1
        scripts = text_scripts(text)
        if len(scripts) >= 2:
            n_cross += 1
            if len(cross_examples) < 6:
                cross_examples.append((f"<id={tid}>", text, sorted(scripts)))
    return {
        "vocab_size": enc.n_vocab,
        "normal": n_normal,
        "special": 0,
        "bytefallback": 0,
        "skipped": n_skipped,
        "max_byte": max_byte,
        "max_byte_token": f"<id={max_byte_id}>",
        "max_byte_text": max_byte_text,
        "n_over_32": n_over_32,
        "n_cross": n_cross,
        "cross_examples": cross_examples,
    }


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TOKENIZER_DIR = REPO_ROOT / "tokenizer"

# --- Canonical run list ---
TOKENIZERS = [
    # display_label, kind, locator, trust_remote_code, pretok_hint
    ("BrahmicTokenizer-131K", "hf", str(LOCAL_TOKENIZER_DIR), False),
    ("o200k_base (tiktoken, 200K)", "tiktoken", "o200k_base", False),
    ("Tekken (Mistral-Nemo, 131K)", "hf", "mistralai/Mistral-Nemo-Base-2407", False),
    ("Sarvam-m (131K)", "hf", "sarvamai/sarvam-m", False),
    ("GPT-OSS-120B (200K)", "hf", "openai/gpt-oss-120b", False),
    ("Sarvam-1 (68K)", "hf", "sarvamai/sarvam-1", False),
    ("Sarvam-30B (262K)", "hf", "sarvamai/sarvam-30b", False),
    ("Gemma-3-1B (262K)", "hf", "google/gemma-3-1b-pt", False),
    ("Krutrim-1-instruct (70K)", "hf", "krutrim-ai-labs/Krutrim-1-instruct", False),
    ("Airavata (48K)", "hf", "ai4bharat/Airavata", False),
    ("Llama-3.1-8B (128K)", "hf", "meta-llama/Llama-3.1-8B", False),
    ("Qwen3-8B (152K)", "hf", "Qwen/Qwen3-8B", False),
    ("DeepSeek-R1 (129K)", "hf", "deepseek-ai/DeepSeek-R1", True),
]


def _resolve_pretok_path(label, kind, locator):
    """Find the local tokenizer.json so we can read pre-tokenizer type."""
    if kind == "tiktoken":
        return None
    if kind == "hf-local-json":
        return locator
    # Local path? Look for tokenizer.json inside
    p = Path(locator)
    if p.exists():
        if p.is_dir():
            cand = p / "tokenizer.json"
            if cand.exists():
                return str(cand)
        elif p.is_file():
            return str(p)
    # HF model id - try cache directory
    safe = "models--" + locator.replace("/", "--")
    cache = Path(os.path.expanduser(f"~/.cache/huggingface/hub/{safe}/snapshots"))
    if cache.exists():
        for snap in cache.iterdir():
            cand = snap / "tokenizer.json"
            if cand.exists():
                return str(cand)
    return None


def main(args):
    if args:
        # single tokenizer path mode — accepts tokenizer.json file, dir, or HF repo id
        path = args[0]
        from tokenizers import Tokenizer as FastTok
        from transformers import AutoTokenizer

        if os.path.isfile(path) and path.endswith(".json"):
            # Load directly as a fast tokenizer from the JSON file
            ft = FastTok.from_file(path)

            class W:
                def __init__(self, ft):
                    self.ft = ft
                    self.all_special_ids = []

                def get_vocab(self):
                    return self.ft.get_vocab()

                def convert_ids_to_tokens(self, i):
                    return self.ft.id_to_token(i)

                def decode(self, ids, **kw):
                    return self.ft.decode(ids)

            tok = W(ft)
            pre = detect_pre_tokenizer(path)
        elif os.path.isdir(path):
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            pre = detect_pre_tokenizer(_resolve_pretok_path(path, "hf", path)) or "?"
        else:
            # HF repo id
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            pre = detect_pre_tokenizer(_resolve_pretok_path(path, "hf", path)) or "?"
        stats = measure_hf(tok)

        # Per-tokenizer PASS/FAIL on the 2 KE constraints (max≤32, 0 cross-script).
        max_ok = stats["max_byte"] <= 32 and stats["n_over_32"] == 0
        cross_ok = stats["n_cross"] == 0
        ke_pass = max_ok and cross_ok

        print(f"tokenizer:              {path}")
        print(f"pre_tokenizer:          {pre}")
        print(f"vocab_size:             {stats['vocab_size']:,}")
        print(f"normal tokens:          {stats['normal']:,}")
        print(f"special tokens:         {stats['special']:,}")
        print(
            f"max byte length:        {stats['max_byte']}  (token={stats['max_byte_token']!r})"
        )
        print(f"tokens > 32 bytes:      {stats['n_over_32']}")
        print(f"cross-script tokens:    {stats['n_cross']}")
        print()
        print("KE constraint check (POS_DIM=32):")
        print(f"  max byte ≤ 32:        {'PASS' if max_ok else 'FAIL'}")
        print(f"  0 cross-script:       {'PASS' if cross_ok else 'FAIL'}")
        print()
        if ke_pass:
            print(
                "RESULT: PASS — tokenizer satisfies both Kronecker constraints at POS_DIM=32."
            )
            return 0
        else:
            print(
                f"RESULT: FAIL — tokenizer violates Kronecker constraints "
                f"(over-32-bytes={stats['n_over_32']}, cross-script={stats['n_cross']})."
            )
            return 1

    # Canonical multi-tokenizer run
    rows = []
    from transformers import AutoTokenizer

    for label, kind, loc, trc in TOKENIZERS:
        print(f"[{label}]  loading...", flush=True)
        pre = "?"
        try:
            if kind == "tiktoken":
                stats = measure_tiktoken(loc)
                pre = "tiktoken (BPE)"
            elif kind == "hf-local-json":
                tok = (
                    AutoTokenizer.from_pretrained(
                        os.path.dirname(loc), trust_remote_code=trc
                    )
                    if os.path.dirname(loc)
                    else AutoTokenizer.from_pretrained(loc, trust_remote_code=trc)
                )
                # Note: hf-local-json might not have config; use direct fast loader
                from tokenizers import Tokenizer as FastTok

                ft = FastTok.from_file(loc)

                # Build a minimal wrapper that supports .get_vocab / .decode / .convert_ids_to_tokens / .all_special_ids
                class W:
                    def __init__(self, ft):
                        self.ft = ft
                        self.all_special_ids = []

                    def get_vocab(self):
                        return self.ft.get_vocab()

                    def convert_ids_to_tokens(self, i):
                        return self.ft.id_to_token(i)

                    def decode(self, ids, **kw):
                        return self.ft.decode(ids)

                stats = measure_hf(W(ft))
                pre = detect_pre_tokenizer(loc)
            else:
                tok = AutoTokenizer.from_pretrained(loc, trust_remote_code=trc)
                pre = (
                    detect_pre_tokenizer(_resolve_pretok_path(label, kind, loc)) or "?"
                )
                stats = measure_hf(tok)
            rows.append((label, pre, stats, None))
            print(
                f"   pre={pre}  vocab={stats['vocab_size']:,}  max_byte={stats['max_byte']}  "
                f">32={stats['n_over_32']}  cross={stats['n_cross']}",
                flush=True,
            )
        except Exception as e:
            print(f"   FAIL: {type(e).__name__}: {str(e)[:200]}", flush=True)
            rows.append((label, pre, None, f"{type(e).__name__}: {str(e)[:200]}"))

    # Final table with PASS/FAIL per tokenizer
    print(
        "\n=== Kronecker constraint summary (decode-based byte length, POS_DIM=32) ==="
    )
    print(
        f'{"tokenizer":<32} {"pre":<14} {"vocab":>8} {"max":>5} {">32":>6} {"cross":>6}   KE'
    )
    print("-" * 90)
    n_pass = 0
    n_total_ok = 0
    for label, pre, stats, err in rows:
        if stats is None:
            print(f'{label:<32} {pre:<14} {"FAILED":>8}    -      -      -   ERROR')
            continue
        n_total_ok += 1
        ke_ok = (
            stats["max_byte"] <= 32
            and stats["n_over_32"] == 0
            and stats["n_cross"] == 0
        )
        marker = "PASS" if ke_ok else "FAIL"
        if ke_ok:
            n_pass += 1
        print(
            f'{label:<32} {pre:<14} {stats["vocab_size"]:>8,} {stats["max_byte"]:>5} '
            f'{stats["n_over_32"]:>6} {stats["n_cross"]:>6}   {marker}'
        )

    print()
    print(f"KE-compatible tokenizers: {n_pass} / {n_total_ok}")
    if n_pass > 0:
        passing = [
            r[0]
            for r in rows
            if r[2]
            and r[2]["max_byte"] <= 32
            and r[2]["n_over_32"] == 0
            and r[2]["n_cross"] == 0
        ]
        print(f"PASS: {', '.join(passing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
