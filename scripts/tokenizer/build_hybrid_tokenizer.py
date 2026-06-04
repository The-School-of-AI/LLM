#!/usr/bin/env python3
"""
Build hybrid tokenizer: OLD's English/EU + cherry-picked Indic from NEW.

Strategy:
  - Keep OLD tokenizer structure EXACTLY (GPT-2 ByteLevel pre-tokenizer, decoder, merges)
  - Drop ~2,400 dead tokens (unseen EU/Vietnamese/CJK/broken + Sinhala)
  - Add ~2,400 high-frequency Indic tokens from NEW (converted to GPT-2 byte form)
  - English compression stays at ~4.2 chars/tok (Ġ-prefix merges preserved)
  - Odia byte fragments should drop from 28.7% to <2%
"""

import csv
import json
import os
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════

BASE = Path(__file__).parent
OLD_TOKENIZER = (
    BASE.parent / "Test20_3B_Zero3" / "code" / "src" / "tokenizer" / "tokenizer.json"
)
NEW_TOKENIZER = BASE / "output" / "tokenizer.json"
UNUSED_CSV = BASE / "audit_old_combined" / "report" / "unused_tokens.csv"
NEW_FREQ_CSV = BASE / "audit_combined" / "report" / "token_frequency.csv"
OUTPUT_DIR = BASE / "output_hybrid"

# Try alternate paths
if not OLD_TOKENIZER.exists():
    OLD_TOKENIZER = (
        BASE / "Test20_3B_Zero3" / "code" / "src" / "tokenizer" / "tokenizer.json"
    )
if not NEW_TOKENIZER.exists():
    NEW_TOKENIZER = BASE / "Tokenizer" / "output" / "tokenizer.json"
if not UNUSED_CSV.exists():
    UNUSED_CSV = (
        BASE / "Tokenizer" / "audit_old_combined" / "report" / "unused_tokens.csv"
    )
if not NEW_FREQ_CSV.exists():
    NEW_FREQ_CSV = (
        BASE / "Tokenizer" / "audit_combined" / "report" / "token_frequency.csv"
    )

TOTAL_VOCAB = 131_072

# Target Indic scripts
INDIC_SCRIPTS = {
    "Devanagari",
    "Bengali",
    "Tamil",
    "Telugu",
    "Kannada",
    "Malayalam",
    "Gujarati",
    "Gurmukhi",
    "Oriya",
}

# Budget allocation per script (proportional to need)
INDIC_BUDGET = {
    "Oriya": 900,  # Odia: 38 tokens, 28.7% byte frag → CRITICAL
    "Gurmukhi": 450,  # Punjabi: 301 tokens, 4.6% → HIGH
    "Gujarati": 220,  # 1590 tokens, 2.2%
    "Malayalam": 180,  # 1646 tokens, 2.4%
    "Kannada": 180,  # 1292 tokens, 1.8%
    "Bengali": 130,  # 2099 tokens, 1.6%
    "Tamil": 130,  # 959 tokens, 1.7%
    "Telugu": 100,  # 1294 tokens
    "Devanagari": 100,  # 3957 tokens, 0.5% (lowest need)
}


# ═══════════════════════════════════════════════════════════
# GPT-2 BYTE ENCODING
# ═══════════════════════════════════════════════════════════


def build_gpt2_byte_maps():
    """Build GPT-2 byte ↔ unicode char bijection."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_to_gpt2 = {b: chr(c) for b, c in zip(bs, cs)}
    gpt2_to_byte = {chr(c): b for b, c in zip(bs, cs)}
    return byte_to_gpt2, gpt2_to_byte


def unicode_to_gpt2(text, byte_to_gpt2):
    """Convert Unicode text to GPT-2 byte-encoded form."""
    utf8_bytes = text.encode("utf-8")
    return "".join(byte_to_gpt2[b] for b in utf8_bytes)


def gpt2_to_unicode(gpt2_text, gpt2_to_byte):
    """Convert GPT-2 byte-encoded text to Unicode."""
    try:
        return bytes([gpt2_to_byte[c] for c in gpt2_text]).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None


def gpt2_to_unicode_lossy(gpt2_text, gpt2_to_byte):
    """Convert GPT-2 text to Unicode, replacing errors."""
    try:
        return bytes([gpt2_to_byte.get(c, 0) for c in gpt2_text]).decode(
            "utf-8", errors="replace"
        )
    except Exception:
        return gpt2_text


def get_token_script(text):
    """Get primary Unicode script of text."""
    scripts = Counter()
    for ch in text:
        if ch.strip() == "":
            continue
        name = unicodedata.name(ch, "")
        matched = False
        for script in INDIC_SCRIPTS | {"SINHALA"}:
            if script.upper() in name.upper():
                scripts[script] += 1
                matched = True
                break
        if not matched:
            if "LATIN" in name.upper():
                scripts["Latin"] += 1
            else:
                scripts["Other"] += 1
    return scripts.most_common(1)[0][0] if scripts else "Unknown"


# ═══════════════════════════════════════════════════════════
# STEP 1: IDENTIFY TOKENS TO DROP
# ═══════════════════════════════════════════════════════════


def identify_drop_ids(old_vocab, gpt2_to_byte):
    """Identify OLD token IDs to drop."""
    print("\n" + "=" * 60)
    print("  STEP 1: Identifying tokens to drop")
    print("=" * 60)

    # Load unused IDs from audit
    unused_ids = set()
    with open(UNUSED_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) >= 1:
                unused_ids.add(int(row[0]))

    print(f"  Unused tokens from audit: {len(unused_ids)}")

    drop_ids = set()
    keep_special = 0
    keep_indic = 0

    for token, tid in old_vocab.items():
        if tid not in unused_ids:
            continue

        # KEEP special <|...|> tokens
        if token.startswith("<|") and token.endswith("|>"):
            keep_special += 1
            continue

        # KEEP unused Indic tokens (they'll fire on broader data)
        if 117074 <= tid < 130716:
            decoded = gpt2_to_unicode_lossy(token, gpt2_to_byte)
            script = get_token_script(decoded)
            if script in INDIC_SCRIPTS:
                keep_indic += 1
                continue

        drop_ids.add(tid)

    print(f"  Unused → DROP: {len(drop_ids)}")
    print(f"  Unused → KEEP (special): {keep_special}")
    print(f"  Unused → KEEP (Indic): {keep_indic}")

    # Also drop Sinhala tokens (not a target language)
    sinhala_count = 0
    for token, tid in old_vocab.items():
        if tid in drop_ids or tid in unused_ids:
            continue
        decoded = gpt2_to_unicode_lossy(token, gpt2_to_byte)
        if "SINHALA" in "".join(
            unicodedata.name(ch, "") for ch in decoded if ch.strip()
        ):
            drop_ids.add(tid)
            sinhala_count += 1

    print(f"  Sinhala tokens dropped: {sinhala_count}")
    print(f"  TOTAL budget: {len(drop_ids)} slots")

    return drop_ids


# ═══════════════════════════════════════════════════════════
# STEP 2: SELECT INDIC TOKENS FROM NEW
# ═══════════════════════════════════════════════════════════


def find_missing_char_infrastructure(old_vocab, byte_to_gpt2):
    """Find ALL missing char-level infrastructure for Indic scripts.

    Returns three lists:
    1. missing_chars: bare character tokens (e.g., à¬Ń = ଭ)
    2. missing_space_chars: space-prefixed versions (e.g., Ġà¬Ń = ' ଭ')
    3. missing_space_intermediates: space + 2-byte prefix (e.g., ĠàŃ)

    All three are needed for BPE to correctly tokenize Indic text.
    In GPT-2 ByteLevel, `Ġ + à` fires at rank 98 (very early), consuming
    the first byte. The space-prefixed chain then needs:
      Ġà + SECOND → ĠàSECOND (intermediate)
      ĠàSECOND + THIRD → ĠàSECONDTHIRD (space+char)
    """
    scripts = {
        "Oriya": (0x0B00, 0x0B80),
        "Gurmukhi": (0x0A00, 0x0A80),
        "Gujarati": (0x0A80, 0x0B00),
        "Malayalam": (0x0D00, 0x0D80),
        "Kannada": (0x0C80, 0x0D00),
        "Bengali": (0x0980, 0x0A00),
        "Tamil": (0x0B80, 0x0C00),
        "Telugu": (0x0C00, 0x0C80),
        "Devanagari": (0x0900, 0x0980),
    }
    space_gpt2 = byte_to_gpt2[0x20]  # Ġ

    missing_chars = []  # (unicode_char, gpt2_form, script)
    missing_space_chars = []  # (unicode_text, gpt2_form, script)
    missing_space_intermediates = []  # (text_desc, gpt2_form, script)
    seen_intermediates = set()

    for script, (start, end) in scripts.items():
        for cp in range(start, end):
            ch = chr(cp)
            name = unicodedata.name(ch, "")
            if not name:
                continue
            utf8 = ch.encode("utf-8")
            gpt2 = "".join(byte_to_gpt2[b] for b in utf8)

            # Bare char token
            if gpt2 not in old_vocab:
                missing_chars.append((ch, gpt2, script))

            # Space-prefixed char token
            space_gpt2_char = space_gpt2 + gpt2
            if space_gpt2_char not in old_vocab:
                missing_space_chars.append((" " + ch, space_gpt2_char, script))

            # Space-prefixed intermediate (Ġà + second_byte → ĠàSECOND)
            if len(utf8) >= 3:
                first_gpt2 = byte_to_gpt2[utf8[0]]
                second_gpt2 = byte_to_gpt2[utf8[1]]
                space_inter = space_gpt2 + first_gpt2 + second_gpt2
                if (
                    space_inter not in old_vocab
                    and space_inter not in seen_intermediates
                ):
                    seen_intermediates.add(space_inter)
                    missing_space_intermediates.append(
                        (f"Ġ+0x{utf8[0]:02X}+0x{utf8[1]:02X}", space_inter, script)
                    )

    return missing_chars, missing_space_chars, missing_space_intermediates


def select_new_indic_tokens(
    new_data, old_decoded_indic_set, budget, byte_to_gpt2, old_vocab
):
    """Select Indic tokens from NEW tokenizer.

    Strategy:
    1. Phase A: Add missing char infrastructure (bare + space-prefixed + intermediates)
    2. Phase B: Fill remaining budget with word-level tokens by corpus frequency
    """
    print("\n" + "=" * 60)
    print(f"  STEP 2: Selecting up to {budget} Indic tokens from NEW")
    print("=" * 60)

    new_vocab = new_data["model"]["vocab"]
    new_merges = new_data["model"]["merges"]

    # ── Phase A: Character-level infrastructure ──
    missing_chars, missing_space_chars, missing_space_inters = (
        find_missing_char_infrastructure(old_vocab, byte_to_gpt2)
    )

    print("\n  Phase A: Character infrastructure")
    print(f"    Missing bare chars:           {len(missing_chars)}")
    print(f"    Missing space+chars:          {len(missing_space_chars)}")
    print(f"    Missing space intermediates:  {len(missing_space_inters)}")

    char_infra_total = (
        len(missing_chars) + len(missing_space_chars) + len(missing_space_inters)
    )
    print(f"    Total char infrastructure:    {char_infra_total}")

    if char_infra_total > budget:
        print(
            f"  WARNING: char infrastructure ({char_infra_total}) exceeds budget ({budget})!"
        )
        print("  Limiting to bare chars only + space intermediates")
        char_infra_total = len(missing_chars) + len(missing_space_inters)
        missing_space_chars = []  # skip these if budget too tight

    char_selected = []  # (text, gpt2_form, 0, script)

    # Add space intermediates first (only ~5 needed)
    for desc, gpt2, script in missing_space_inters:
        char_selected.append((desc, gpt2, 0, script))

    # Add bare char tokens
    for ch, gpt2, script in missing_chars:
        char_selected.append((ch, gpt2, 0, script))

    # Add space-prefixed char tokens
    for text, gpt2, script in missing_space_chars:
        if len(char_selected) >= budget:
            break
        char_selected.append((text, gpt2, 0, script))

    print(f"  Added {len(char_selected)} char-level tokens to vocab")
    by_script = Counter(s for _, _, _, s in char_selected)
    for s, c in by_script.most_common():
        print(f"    {s:15s}: {c}")

    remaining_budget = budget - len(char_selected)
    print(f"  Remaining budget for word-level: {remaining_budget}")

    # ── Phase B: Word-level tokens ──
    # Load corpus frequency for NEW tokenizer
    token_freq = {}
    try:
        with open(NEW_FREQ_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    try:
                        token_freq[int(row[0])] = int(row[-1])
                    except (ValueError, IndexError):
                        pass
        print(f"  Loaded frequency for {len(token_freq)} tokens")
    except FileNotFoundError:
        print("  WARNING: No frequency CSV, using rank proxy")
        for tok, tid in new_vocab.items():
            token_freq[tid] = max(0, 131072 - tid)

    # Classify NEW's Indic tokens by script
    indic_by_script = defaultdict(list)

    # Build set of GPT-2 forms we're already adding (char-level)
    char_gpt2_set = {gpt2 for _, gpt2, _, _ in char_selected}

    for token, tid in new_vocab.items():
        if token.startswith("<") and token.endswith(">"):
            continue
        if tid < 76217 or tid > 130264:
            continue

        script = get_token_script(token)
        if script not in INDIC_SCRIPTS:
            continue

        freq = token_freq.get(tid, 0)
        byte_len = len(token.encode("utf-8", errors="replace"))
        if byte_len > 32:
            continue

        # Skip if OLD already has this token (in Unicode form)
        if token in old_decoded_indic_set:
            continue

        # Skip single-char tokens we're already adding from Phase A
        gpt2_form = unicode_to_gpt2(token, byte_to_gpt2)
        if gpt2_form in char_gpt2_set:
            continue

        indic_by_script[script].append((token, tid, freq))

    # Sort by frequency
    for script in indic_by_script:
        indic_by_script[script].sort(key=lambda x: -x[2])

    print("\n  Available word-level Indic tokens (after dedup):")
    for script in sorted(indic_by_script):
        print(f"    {script:15s}: {len(indic_by_script[script]):>6,}")

    # Adjust budget per script proportionally
    scale = remaining_budget / sum(INDIC_BUDGET.values())
    adjusted_budget = {s: max(1, int(a * scale)) for s, a in INDIC_BUDGET.items()}

    word_selected = []  # (unicode_token, gpt2_token, freq, script)
    word_unicode_set = set()

    for script, alloc in sorted(adjusted_budget.items(), key=lambda x: -x[1]):
        available = indic_by_script.get(script, [])
        actual = min(alloc, len(available), remaining_budget - len(word_selected))
        if actual <= 0:
            continue

        for token, tid, freq in available[:actual]:
            if token not in word_unicode_set:
                gpt2_form = unicode_to_gpt2(token, byte_to_gpt2)
                word_selected.append((token, gpt2_form, freq, script))
                word_unicode_set.add(token)

        print(f"  Selected {actual:>4} word-level from {script:<15}")

    # Fill remaining budget from highest-frequency across all scripts
    remaining = remaining_budget - len(word_selected)
    if remaining > 0:
        all_remaining = []
        for script, tokens in indic_by_script.items():
            for token, tid, freq in tokens:
                if token not in word_unicode_set:
                    all_remaining.append((token, tid, freq, script))
        all_remaining.sort(key=lambda x: -x[2])
        for token, tid, freq, script in all_remaining[:remaining]:
            gpt2_form = unicode_to_gpt2(token, byte_to_gpt2)
            word_selected.append((token, gpt2_form, freq, script))
            word_unicode_set.add(token)
        print(f"  Filled {min(remaining, len(all_remaining))} remaining by frequency")

    # Combine: char-level + word-level
    selected = char_selected + word_selected
    print(
        f"\n  TOTAL SELECTED: {len(selected)} ({len(char_selected)} char + {len(word_selected)} word)"
    )

    # ── Collect merges ──
    # Set of ALL GPT-2 tokens being added (char + word)
    gpt2_selected_set = {gpt2 for _, gpt2, _, _ in selected}

    # Word-level merges from NEW tokenizer
    selected_merges = []
    for merge_str in new_merges:
        parts = merge_str.split(" ", 1)
        if len(parts) != 2:
            continue
        a, b = parts
        try:
            a_gpt2 = unicode_to_gpt2(a, byte_to_gpt2)
            b_gpt2 = unicode_to_gpt2(b, byte_to_gpt2)
            merged_gpt2 = a_gpt2 + b_gpt2
        except Exception:
            continue
        if merged_gpt2 in gpt2_selected_set:
            selected_merges.append([a_gpt2, b_gpt2])

    print(f"  Collected {len(selected_merges)} direct merge rules from NEW")

    # Byte→char reconstruction merges for ALL characters in selected tokens.
    # We need TWO chains per character:
    #   Bare:  à + ¬ → à¬,  à¬ + Ń → à¬Ń           (non-initial position)
    #   Space: Ġà + ¬ → Ġà¬, Ġà¬ + Ń → Ġà¬Ń       (word-initial position)
    # The space chain is needed because Ġ+à fires at rank 98, consuming à
    # before the bare chain's à+¬ can fire at rank 13K.
    space_gpt2 = byte_to_gpt2[0x20]  # Ġ
    first_gpt2 = byte_to_gpt2[0xE0]  # à (first byte of all 3-byte Indic chars)

    char_merges = []
    chars_covered = set()

    for unicode_tok, gpt2_tok, freq, script in selected:
        for ch in unicode_tok:
            if not isinstance(ch, str) or len(ch) != 1:
                continue
            if ch in chars_covered or ch == " ":
                continue
            chars_covered.add(ch)
            utf8 = ch.encode("utf-8")
            if len(utf8) <= 1:
                continue

            gpt2_chars = [byte_to_gpt2[b] for b in utf8]

            # ── Bare chain: byte1 + byte2 → inter, inter + byte3 → char ──
            accum = gpt2_chars[0]
            for i in range(1, len(gpt2_chars)):
                char_merges.append([accum, gpt2_chars[i]])
                accum = accum + gpt2_chars[i]

            # ── Space-prefixed chain: Ġbyte1 + byte2 → Ġinter, Ġinter + byte3 → Ġchar ──
            if len(utf8) >= 2:
                space_accum = space_gpt2 + gpt2_chars[0]  # Ġà (exists, rank 98)
                for i in range(1, len(gpt2_chars)):
                    space_merged = space_accum + gpt2_chars[i]
                    char_merges.append([space_accum, gpt2_chars[i]])
                    space_accum = space_merged

    # Deduplicate
    seen = set()
    unique_char_merges = []
    for m in char_merges:
        key = (m[0], m[1])
        if key not in seen:
            seen.add(key)
            unique_char_merges.append(m)

    print(f"  Generated {len(unique_char_merges)} byte→char + space→char merges")

    all_new_merges = unique_char_merges + selected_merges
    print(f"  Total NEW merges: {len(all_new_merges)}")

    return selected, all_new_merges


# ═══════════════════════════════════════════════════════════
# STEP 3-6: ASSEMBLE
# ═══════════════════════════════════════════════════════════


def assemble(old_data, drop_ids, new_indic, new_indic_merges, gpt2_to_byte):
    """Remove dead tokens, add new Indic tokens, write tokenizer.json."""
    print("\n" + "=" * 60)
    print("  STEP 3-6: Assembling hybrid tokenizer")
    print("=" * 60)

    old_vocab = old_data["model"]["vocab"]
    old_merges = old_data["model"]["merges"]
    old_added = old_data.get("added_tokens", [])

    # ── Build new vocab ──
    # Keep all non-dropped tokens with their original IDs
    kept_tokens = {}  # gpt2_token → old_id
    for token, tid in old_vocab.items():
        if tid not in drop_ids:
            kept_tokens[token] = tid

    print(f"  Kept {len(kept_tokens):,} tokens from OLD")

    # Find free IDs (from dropped tokens)
    all_old_ids = set(range(TOTAL_VOCAB))
    used_ids = set(kept_tokens.values())
    free_ids = sorted(all_old_ids - used_ids)
    print(f"  Free IDs: {len(free_ids)}")

    # Add new Indic tokens at free IDs
    new_token_map = {}  # gpt2_token → assigned_id
    added_count = 0
    for unicode_tok, gpt2_tok, freq, script in new_indic:
        if gpt2_tok in kept_tokens:
            continue  # already exists
        if not free_ids:
            print(f"  WARNING: Ran out of free IDs at {added_count}")
            break
        new_id = free_ids.pop(0)
        kept_tokens[gpt2_tok] = new_id
        new_token_map[gpt2_tok] = new_id
        added_count += 1

    print(f"  Added {added_count} new Indic tokens")

    # Pad remaining free IDs
    pad_count = 0
    for free_id in free_ids:
        pad_token = f"<|pad_{free_id}|>"
        kept_tokens[pad_token] = free_id
        pad_count += 1
    if pad_count:
        print(f"  Added {pad_count} padding tokens")

    print(f"  Final vocab: {len(kept_tokens):,}")

    # ── Filter merges ──
    kept_token_set = set(kept_tokens.keys())

    # Filter OLD merges: keep only if both parts exist in kept vocab
    valid_old_merges = []
    dropped_merge_count = 0
    for merge in old_merges:
        if isinstance(merge, list):
            a, b = merge[0], merge[1]
        else:
            parts = merge.split(" ", 1)
            if len(parts) != 2:
                dropped_merge_count += 1
                continue
            a, b = parts

        merged = a + b
        if a in kept_token_set and b in kept_token_set and merged in kept_token_set:
            valid_old_merges.append([a, b])
        else:
            dropped_merge_count += 1

    print(
        f"  Valid OLD merges: {len(valid_old_merges):,} (dropped {dropped_merge_count:,})"
    )

    # Add NEW Indic merges (already in GPT-2 form)
    existing_merge_set = {(m[0], m[1]) for m in valid_old_merges}
    valid_new_merges = []
    for merge in new_indic_merges:
        a, b = merge[0], merge[1]
        merged = a + b
        if a in kept_token_set and b in kept_token_set and merged in kept_token_set:
            if (a, b) not in existing_merge_set:
                valid_new_merges.append([a, b])
                existing_merge_set.add((a, b))

    print(f"  Valid NEW Indic merges: {len(valid_new_merges):,}")

    all_merges = valid_old_merges + valid_new_merges
    print(f"  Total merges: {len(all_merges):,}")

    # ── Update added_tokens list ──
    # Keep all original added_tokens that aren't dropped
    new_added_tokens = []
    for at in old_added:
        if at["id"] not in drop_ids:
            new_added_tokens.append(at)

    # ── Build tokenizer.json ──
    # Keep OLD's exact structure, just swap vocab + merges
    tokenizer_json = {
        "version": old_data.get("version", "1.0"),
        "truncation": old_data.get("truncation"),
        "padding": old_data.get("padding"),
        "added_tokens": new_added_tokens,
        "normalizer": old_data.get("normalizer"),
        "pre_tokenizer": old_data["pre_tokenizer"],  # KEEP OLD's exactly
        "post_processor": old_data["post_processor"],  # KEEP OLD's exactly
        "decoder": old_data["decoder"],  # KEEP OLD's exactly
        "model": {
            "type": "BPE",
            "dropout": old_data["model"].get("dropout"),
            "unk_token": old_data["model"].get("unk_token"),
            "continuing_subword_prefix": old_data["model"].get(
                "continuing_subword_prefix"
            ),
            "end_of_word_suffix": old_data["model"].get("end_of_word_suffix"),
            "fuse_unk": old_data["model"].get("fuse_unk", False),
            "byte_fallback": old_data["model"].get("byte_fallback", False),
            "ignore_merges": old_data["model"].get("ignore_merges", False),
            "vocab": kept_tokens,
            "merges": all_merges,
        },
    }

    # Write
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer_path = OUTPUT_DIR / "tokenizer.json"
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)
    size_mb = os.path.getsize(tokenizer_path) / 1e6
    print(f"\n  Wrote: {tokenizer_path} ({size_mb:.1f} MB)")

    # Copy config files
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "model_max_length": 131072,
        "clean_up_tokenization_spaces": False,
    }
    config_path = OUTPUT_DIR / "tokenizer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    special_map = {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
    }
    stm_path = OUTPUT_DIR / "special_tokens_map.json"
    with open(stm_path, "w", encoding="utf-8") as f:
        json.dump(special_map, f, ensure_ascii=False, indent=2)

    print(f"  Wrote: {config_path}")
    print(f"  Wrote: {stm_path}")

    return tokenizer_path


# ═══════════════════════════════════════════════════════════
# STEP 7: VALIDATE
# ═══════════════════════════════════════════════════════════


def validate(tokenizer_path, byte_to_gpt2, gpt2_to_byte):
    """Sanity-check the hybrid tokenizer."""
    print("\n" + "=" * 60)
    print("  STEP 7: Validation")
    print("=" * 60)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    print(f"  Vocab size: {tok.get_vocab_size():,}")
    assert tok.get_vocab_size() == TOTAL_VOCAB

    tests = {
        "English": "The quick brown fox jumps over the lazy dog. Machine learning requires evaluation.",
        "Code": "def hello():\n    print('Hello, World!')\n    return 42",
        "Hindi": "भारत एक विविधताओं से भरा देश है। यहाँ की संस्कृति बहुत समृद्ध है।",
        "Bengali": "বাংলাদেশ একটি দক্ষিণ এশিয়ার দেশ।",
        "Tamil": "தமிழ் ஒரு பழமையான மொழி ஆகும்.",
        "Telugu": "తెలుగు భాష చాలా అందమైన భాష.",
        "Kannada": "ಕನ್ನಡ ನಾಡಿನ ಅಧಿಕೃತ ಭಾಷೆ.",
        "Malayalam": "മലയാളം കേരളത്തിന്റെ ഭാഷയാണ്.",
        "Gujarati": "ગુજરાતી ભાષા ખૂબ જ સુંદર છે.",
        "Punjabi": "ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਬਹੁਤ ਸੁੰਦਰ ਹੈ।",
        "Odia": "ଓଡ଼ିଆ ଭାଷା ଭାରତର ଏକ ପ୍ରାଚୀନ ଭାଷା।",
        "Marathi": "मराठी भाषा महाराष्ट्राची राजभाषा आहे.",
        "Assamese": "অসমীয়া ভাষা অসমৰ চৰকাৰী ভাষা।",
    }

    print("\n  Round-trip + compression tests:")
    all_pass = True
    for lang, text in tests.items():
        enc = tok.encode(text)
        decoded = tok.decode(enc.ids)
        match = decoded.strip() == text.strip()
        cpt = len(text) / len(enc.ids) if enc.ids else 0

        # Count byte fragments (tokens that are single GPT-2 chars mapping to individual bytes)
        # In GPT-2 ByteLevel, byte fragments are single-char tokens
        n_frag = sum(1 for t in enc.tokens if len(t) == 1 and ord(t[0]) > 127)
        frag_pct = n_frag / len(enc.tokens) * 100 if enc.tokens else 0

        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(
            f"    {lang:12s}: {status} | {len(enc.ids):>4} tok | {cpt:.2f} c/t | {frag_pct:.1f}% frag"
        )
        if not match:
            print(f"      IN:  {repr(text[:60])}")
            print(f"      OUT: {repr(decoded[:60])}")

    # Extended Odia test
    odia_long = "ଓଡ଼ିଆ ଭାଷା ଭାରତର ଏକ ପ୍ରାଚୀନ ଭାଷା। ଏହା ଭାରତୀୟ ସମ୍ବିଧାନ ଦ୍ୱାରା ସ୍ୱୀକୃତ ୨୨ଟି ଅନୁସୂଚୀୟ ଭାଷା ମଧ୍ୟରୁ ଗୋଟିଏ।"
    enc_o = tok.encode(odia_long)
    n_frag_o = sum(1 for t in enc_o.tokens if len(t) == 1 and ord(t[0]) > 127)
    print(
        f"\n  Odia extended: {len(enc_o.ids)} tokens, {n_frag_o}/{len(enc_o.tokens)} byte frags ({n_frag_o/len(enc_o.tokens)*100:.1f}%)"
    )

    # Show Odia token breakdown
    print(f"  Odia tokens sample: {enc_o.tokens[:20]}")

    # Vocab stats
    vocab = tok.get_vocab()
    script_counts = Counter()
    for token in vocab:
        if token.startswith("<|") or token.startswith("<0x"):
            script_counts["Special/Byte"] += 1
        else:
            decoded = gpt2_to_unicode_lossy(token, gpt2_to_byte)
            script = get_token_script(decoded)
            script_counts[script] += 1

    print("\n  Vocab by script:")
    for script, count in script_counts.most_common(15):
        print(f"    {script:15s}: {count:>6,}")

    return all_pass


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    print("=" * 60)
    print("  HYBRID TOKENIZER BUILD")
    print("  OLD English (GPT-2 ByteLevel) + NEW Indic tokens")
    print("=" * 60)

    byte_to_gpt2, gpt2_to_byte = build_gpt2_byte_maps()

    # Load tokenizers
    print("\n  Loading OLD tokenizer...", end=" ", flush=True)
    with open(OLD_TOKENIZER, "r", encoding="utf-8") as f:
        old_data = json.load(f)
    old_vocab = old_data["model"]["vocab"]
    print(f"{len(old_vocab):,} tokens, {len(old_data['model']['merges']):,} merges")

    print("  Loading NEW tokenizer...", end=" ", flush=True)
    with open(NEW_TOKENIZER, "r", encoding="utf-8") as f:
        new_data = json.load(f)
    print(
        f"{len(new_data['model']['vocab']):,} tokens, {len(new_data['model']['merges']):,} merges"
    )

    # Step 1: Identify tokens to drop
    drop_ids = identify_drop_ids(old_vocab, gpt2_to_byte)

    # Decode OLD's existing Indic tokens to Unicode (for dedup in step 2)
    old_indic_unicode = set()
    for token, tid in old_vocab.items():
        if tid in drop_ids:
            continue
        if 117074 <= tid < 130716:
            decoded = gpt2_to_unicode(token, gpt2_to_byte)
            if decoded:
                old_indic_unicode.add(decoded)
    print(
        f"\n  OLD has {len(old_indic_unicode)} Indic tokens (decoded to Unicode for dedup)"
    )

    # Step 2: Select Indic tokens from NEW
    new_indic, new_indic_merges = select_new_indic_tokens(
        new_data, old_indic_unicode, len(drop_ids), byte_to_gpt2, old_vocab
    )

    # Steps 3-6: Assemble
    tokenizer_path = assemble(
        old_data, drop_ids, new_indic, new_indic_merges, gpt2_to_byte
    )

    # Step 7: Validate
    try:
        validate(tokenizer_path, byte_to_gpt2, gpt2_to_byte)
    except Exception as e:
        print(f"\n  Validation error: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  BUILD COMPLETE in {time.time() - t0:.1f}s")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
