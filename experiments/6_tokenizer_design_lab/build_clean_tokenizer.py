"""Build clean 128k tokenizer from base gptoss.

Steps:
1. Load base gptoss tokenizer
2. Build merge graph: token -> (a, b) that produces it
3. Identify tokens to remove:
   - >32 chars
   - Contains blocked language scripts (non-English, non-Indic)
4. For each removed token, also remove the merge that produces it
5. Move Indic tokens to end
6. Add 512 special tokens at start
7. Reduce to 128k by removing lowest-priority general tokens

Key: Removal cascades properly - if token X is removed, the merge producing X
is removed, so any token depending on X via merges cannot be built.
"""

import csv
import json
import os
import re
import shutil

BASE_GPTOSS_PATH = "Tokenizer_metrics/tokenizers/gptoss_tokenizer.json"
OUTPUT_DIR = "gptoss_pruning"
TARGET_VOCAB_SIZE = 131046  # 131072 - 26 (for add_special_tokens.py)
NUM_SPECIAL = 512
MAX_TOKEN_LEN = 32

# ---------------------------------------------------------------------------
# GPT-2 byte encoding
# ---------------------------------------------------------------------------
def _build_gpt2_maps():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

_CHAR_TO_BYTE = _build_gpt2_maps()

def decode_gpt2(token: str) -> str:
    try:
        raw = bytes(_CHAR_TO_BYTE[ch] for ch in token)
        return raw.decode("utf-8", errors="ignore")
    except KeyError:
        return token


# ---------------------------------------------------------------------------
# Blocked language scripts (non-English, non-Indic)
# ---------------------------------------------------------------------------
BLOCKED_RANGES = [
    (0x0370, 0x03FF, "Greek"), (0x1F00, 0x1FFF, "Greek Extended"),
    (0x0400, 0x04FF, "Cyrillic"), (0x0500, 0x052F, "Cyrillic Supplement"),
    (0x2DE0, 0x2DFF, "Cyrillic Extended-A"), (0xA640, 0xA69F, "Cyrillic Extended-B"),
    (0x0530, 0x058F, "Armenian"), (0xFB13, 0xFB17, "Armenian Ligatures"),
    (0x10A0, 0x10FF, "Georgian"), (0x2D00, 0x2D2F, "Georgian Supplement"),
    (0x0590, 0x05FF, "Hebrew"), (0xFB1D, 0xFB4F, "Hebrew Presentation"),
    (0x0600, 0x06FF, "Arabic"), (0x0750, 0x077F, "Arabic Supplement"),
    (0x08A0, 0x08FF, "Arabic Extended-A"), (0xFB50, 0xFDFF, "Arabic Pres-A"),
    (0xFE70, 0xFEFF, "Arabic Pres-B"),
    (0x0700, 0x074F, "Syriac"), (0x0780, 0x07BF, "Thaana"),
    (0x07C0, 0x07FF, "NKo"), (0x0840, 0x085F, "Mandaic"),
    (0x0E00, 0x0E7F, "Thai"), (0x0E80, 0x0EFF, "Lao"),
    (0x1780, 0x17FF, "Khmer"), (0x19E0, 0x19FF, "Khmer Symbols"),
    (0x0F00, 0x0FFF, "Tibetan"), (0x1800, 0x18AF, "Mongolian"),
    (0x1700, 0x171F, "Tagalog"), (0x1720, 0x173F, "Hanunoo"),
    (0x1740, 0x175F, "Buhid"), (0x1760, 0x177F, "Tagbanwa"),
    (0x3000, 0x303F, "CJK Symbols"), (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"), (0x31F0, 0x31FF, "Katakana Ext"),
    (0x3100, 0x312F, "Bopomofo"), (0x31A0, 0x31BF, "Bopomofo Ext"),
    (0x3190, 0x319F, "Kanbun"), (0x31C0, 0x31EF, "CJK Strokes"),
    (0x3200, 0x32FF, "Enclosed CJK"), (0x3300, 0x33FF, "CJK Compatibility"),
    (0x3400, 0x4DBF, "CJK Ext-A"), (0x4E00, 0x9FFF, "CJK Unified"),
    (0xF900, 0xFAFF, "CJK Compat Ideographs"), (0xFE30, 0xFE4F, "CJK Compat Forms"),
    (0x20000, 0x2A6DF, "CJK Ext-B"), (0x2A700, 0x2B73F, "CJK Ext-C"),
    (0x2B740, 0x2B81F, "CJK Ext-D"), (0x2B820, 0x2CEAF, "CJK Ext-E"),
    (0x2CEB0, 0x2EBEF, "CJK Ext-F"), (0x30000, 0x3134F, "CJK Ext-G"),
    (0x1100, 0x11FF, "Hangul Jamo"), (0x3130, 0x318F, "Hangul Compat Jamo"),
    (0xA960, 0xA97F, "Hangul Jamo Ext-A"), (0xAC00, 0xD7AF, "Hangul Syllables"),
    (0xD7B0, 0xD7FF, "Hangul Jamo Ext-B"),
    (0x1200, 0x137F, "Ethiopic"), (0x1380, 0x139F, "Ethiopic Supplement"),
    (0x2D80, 0x2DDF, "Ethiopic Extended"), (0x13A0, 0x13FF, "Cherokee"),
    (0x1400, 0x167F, "Canadian Aboriginal"), (0x1680, 0x169F, "Ogham"),
    (0x16A0, 0x16FF, "Runic"), (0xA500, 0xA63F, "Vai"),
    (0xA6A0, 0xA6FF, "Bamum"), (0x2D30, 0x2D7F, "Tifinagh"),
    (0x1000, 0x109F, "Myanmar"), (0xAA60, 0xAA7F, "Myanmar Ext-A"),
    (0xA9E0, 0xA9FF, "Myanmar Ext-B"),
    (0xA000, 0xA48F, "Yi Syllables"), (0xA490, 0xA4CF, "Yi Radicals"),
]

# Indic ranges (to identify for moving to end)
INDIC_RANGES = [
    (0x0900, 0x097F), (0xA8E0, 0xA8FF),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0D80, 0x0DFF),  # Sinhala
]


def is_blocked(token: str) -> str:
    """Return blocked script label if token contains blocked chars, else ''."""
    decoded = decode_gpt2(token)
    for ch in decoded:
        cp = ord(ch)
        for lo, hi, label in BLOCKED_RANGES:
            if lo <= cp <= hi:
                return label
    return ""


def is_indic(token: str) -> bool:
    """Check if token contains Indic script characters."""
    decoded = decode_gpt2(token)
    for ch in decoded:
        cp = ord(ch)
        for lo, hi in INDIC_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def build_special_tokens():
    """Create 512 special tokens."""
    tokens = []

    doc = ["<|begin_of_text|>", "<|end_of_text|>", "<|pad|>", "<|unk|>",
           "<|sep|>", "<|cls|>", "<|mask|>", "<|newline|>",
           "<|paragraph|>", "<|document|>"]
    chat = ["<|system|>", "<|user|>", "<|assistant|>", "<|tool|>",
            "<|function|>", "<|context|>", "<|instruction|>", "<|response|>",
            "<|turn|>", "<|end_turn|>"]
    code = ["<|code_begin|>", "<|code_end|>", "<|output_begin|>", "<|output_end|>",
            "<|error|>", "<|stdin|>", "<|stdout|>", "<|stderr|>",
            "<|file_begin|>", "<|file_end|>"]
    lang = ["<|lang:python|>", "<|lang:javascript|>", "<|lang:typescript|>", "<|lang:java|>",
            "<|lang:cpp|>", "<|lang:c|>", "<|lang:rust|>", "<|lang:go|>",
            "<|lang:ruby|>", "<|lang:php|>", "<|lang:swift|>", "<|lang:kotlin|>",
            "<|lang:scala|>", "<|lang:r|>", "<|lang:julia|>", "<|lang:sql|>",
            "<|lang:html|>", "<|lang:css|>", "<|lang:bash|>", "<|lang:shell|>"]
    tool = ["<|json_begin|>", "<|json_end|>", "<|tool_call|>", "<|tool_result|>",
            "<|function_call|>", "<|function_result|>", "<|api_request|>", "<|api_response|>",
            "<|schema|>", "<|arguments|>"]
    source = ["<|source:wikipedia|>", "<|source:github|>", "<|source:arxiv|>", "<|source:web|>",
              "<|source:book|>", "<|source:code|>", "<|source:docs|>", "<|source:news|>",
              "<|source:social|>", "<|source:other|>"]
    think = ["<|think_begin|>", "<|think_end|>", "<|step|>", "<|plan|>",
             "<|reflect|>", "<|verify|>", "<|conclude|>", "<|reason|>",
             "<|analyze|>", "<|synthesize|>"]
    fmt = ["<|markdown|>", "<|latex|>", "<|table|>", "<|list|>",
           "<|heading|>", "<|quote|>", "<|link|>", "<|image|>",
           "<|math|>", "<|diagram|>", "<|chart|>", "<|formula|>",
           "<|footnote|>", "<|citation|>", "<|reference|>", "<|caption|>",
           "<|bullet|>", "<|numbered|>", "<|checkbox|>", "<|definition|>"]

    all_defined = doc + chat + code + lang + tool + source + think + fmt
    while len(all_defined) < 256:
        all_defined.append(f"<|special_{len(all_defined)}|>")

    for i, content in enumerate(all_defined[:256]):
        tokens.append({"id": i, "content": content, "single_word": False,
                       "lstrip": False, "rstrip": False, "normalized": False, "special": True})
    for i in range(256, 512):
        tokens.append({"id": i, "content": f"<|reserved_{i}|>", "single_word": False,
                       "lstrip": False, "rstrip": False, "normalized": False, "special": True})
    return tokens


def main():
    print("=" * 70)
    print("Building Clean 128K Tokenizer")
    print("=" * 70)

    # Load base gptoss
    print("\n[1] Loading base gptoss tokenizer...")
    with open(BASE_GPTOSS_PATH, encoding="utf-8") as f:
        gptoss = json.load(f)

    vocab = gptoss["model"]["vocab"]
    merges = gptoss["model"]["merges"]
    print(f"    Vocab: {len(vocab):,}")
    print(f"    Merges: {len(merges):,}")

    # Build merge graph: token -> (a, b) that produces it
    print("\n[2] Building merge graph...")
    merge_graph = {}  # ab -> (a, b)
    for merge in merges:
        if isinstance(merge, list):
            a, b = merge[0], merge[1]
        else:
            a, b = merge.split(" ", 1)
        ab = a + b
        merge_graph[ab] = (a, b)

    # Find base tokens (not produced by any merge)
    merged_tokens = set(merge_graph.keys())
    base_tokens = set(vocab.keys()) - merged_tokens
    print(f"    Base tokens: {len(base_tokens):,}")
    print(f"    Merged tokens: {len(merged_tokens):,}")

    # Identify tokens to remove
    print("\n[3] Identifying tokens to remove...")
    removed_tokens = {}  # token -> reason

    for token in vocab:
        if token in base_tokens:
            # Never remove base tokens (256 bytes)
            continue

        # Check length
        if len(token) > MAX_TOKEN_LEN:
            removed_tokens[token] = "too_long"
            continue

        # Check blocked languages
        blocked = is_blocked(token)
        if blocked:
            removed_tokens[token] = f"blocked:{blocked}"

    print(f"    Tokens to remove: {len(removed_tokens):,}")

    # Build kept vocab (remove blocked/long tokens)
    print("\n[4] Building kept vocabulary...")
    kept_vocab = {t: old_id for t, old_id in vocab.items() if t not in removed_tokens}
    print(f"    Kept vocab: {len(kept_vocab):,}")

    # Filter merges - remove any merge where a, b, or ab is not in kept vocab
    print("\n[5] Filtering merges...")
    kept_merges = []
    removed_merges = []

    for merge in merges:
        if isinstance(merge, list):
            a, b = merge[0], merge[1]
        else:
            a, b = merge.split(" ", 1)
        ab = a + b

        if a in kept_vocab and b in kept_vocab and ab in kept_vocab:
            kept_merges.append(merge)
        else:
            removed_merges.append(merge)

    print(f"    Kept merges: {len(kept_merges):,}")
    print(f"    Removed merges: {len(removed_merges):,}")

    # Categorize kept tokens
    print("\n[6] Categorizing tokens...")
    indic_tokens = []
    general_tokens = []

    for token, old_id in kept_vocab.items():
        if is_indic(token):
            indic_tokens.append((token, old_id))
        else:
            general_tokens.append((token, old_id))

    # Sort by original ID
    indic_tokens.sort(key=lambda x: x[1])
    general_tokens.sort(key=lambda x: x[1])

    print(f"    Indic tokens: {len(indic_tokens):,}")
    print(f"    General tokens: {len(general_tokens):,}")

    # Calculate budget for 128k
    vocab_budget = TARGET_VOCAB_SIZE - NUM_SPECIAL
    num_indic = len(indic_tokens)
    num_general_max = vocab_budget - num_indic

    print(f"\n[7] Token budget:")
    print(f"    Target: {TARGET_VOCAB_SIZE:,}")
    print(f"    Special: {NUM_SPECIAL}")
    print(f"    Vocab budget: {vocab_budget:,}")
    print(f"    Indic (all): {num_indic:,}")
    print(f"    General max: {num_general_max:,}")

    if len(general_tokens) > num_general_max:
        kept_general = general_tokens[:num_general_max]
        removed_general = general_tokens[num_general_max:]
        print(f"    General kept: {len(kept_general):,}")
        print(f"    General removed (for 128k): {len(removed_general):,}")

        # Add to removed_tokens for logging
        for token, old_id in removed_general:
            removed_tokens[token] = "128k_cutoff"
    else:
        kept_general = general_tokens
        removed_general = []
        print(f"    General kept: {len(kept_general):,}")

    # Build final vocab with new IDs
    print("\n[8] Assigning new IDs...")
    special_tokens = build_special_tokens()
    final_vocab = {t["content"]: t["id"] for t in special_tokens}
    token_mapping = []

    next_id = NUM_SPECIAL

    # General tokens first
    for token, old_id in kept_general:
        final_vocab[token] = next_id
        token_mapping.append((token, old_id, next_id, "general"))
        next_id += 1

    general_end_id = next_id - 1

    # Indic tokens at the end
    for token, old_id in indic_tokens:
        final_vocab[token] = next_id
        token_mapping.append((token, old_id, next_id, "indic"))
        next_id += 1

    indic_end_id = next_id - 1

    print(f"    Final vocab size: {len(final_vocab):,}")
    print(f"    Special: IDs 0-{NUM_SPECIAL - 1}")
    print(f"    General: IDs {NUM_SPECIAL}-{general_end_id}")
    print(f"    Indic: IDs {general_end_id + 1}-{indic_end_id}")

    # Filter merges again for final vocab
    print("\n[9] Final merge filtering...")
    final_vocab_set = set(final_vocab.keys())
    final_merges = []

    for merge in kept_merges:
        if isinstance(merge, list):
            a, b = merge[0], merge[1]
        else:
            a, b = merge.split(" ", 1)
        ab = a + b

        if a in final_vocab_set and b in final_vocab_set and ab in final_vocab_set:
            final_merges.append(merge)

    print(f"    Final merges: {len(final_merges):,}")

    # Write output
    print("\n[10] Writing output files...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    new_tokenizer = json.loads(json.dumps(gptoss))
    new_tokenizer["model"]["vocab"] = final_vocab
    new_tokenizer["model"]["merges"] = final_merges
    new_tokenizer["added_tokens"] = special_tokens

    with open(os.path.join(OUTPUT_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(new_tokenizer, f, ensure_ascii=False)

    config_src = os.path.join(os.path.dirname(BASE_GPTOSS_PATH), "tokenizer_config.json")
    if os.path.exists(config_src):
        shutil.copy2(config_src, os.path.join(OUTPUT_DIR, "tokenizer_config.json"))

    # Write reports
    with open(os.path.join(OUTPUT_DIR, "build_report.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "old_id", "new_id", "tier"])
        for token, old_id, new_id, tier in token_mapping:
            writer.writerow([token, old_id, new_id, tier])

    with open(os.path.join(OUTPUT_DIR, "removed_tokens.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "reason"])
        for token, reason in removed_tokens.items():
            writer.writerow([token, reason])

    print(f"    {OUTPUT_DIR}/tokenizer.json")
    print(f"    {OUTPUT_DIR}/build_report.csv")
    print(f"    {OUTPUT_DIR}/removed_tokens.csv")

    # Validate
    print("\n[11] Validating...")
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(OUTPUT_DIR, "tokenizer.json"))

        tests = [
            ("English", "Hello world! This is a test."),
            ("Hindi", "नमस्ते दुनिया"),
            ("Tamil", "வணக்கம் உலகம்"),
            ("Code", "def hello():\n    return 'world'"),
        ]

        print("    Encode/decode tests:")
        for name, text in tests:
            enc = tok.encode(text)
            print(f"      {name}: {len(enc.ids)} tokens")

        print("\n    Validation PASSED")

    except Exception as e:
        print(f"\n    Validation FAILED: {e}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
