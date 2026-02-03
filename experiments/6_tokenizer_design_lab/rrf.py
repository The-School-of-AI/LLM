import csv
import heapq
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime

# ================= CONFIGURATION =================
# Directory containing your tokenizer folders
BASE_DIR = os.getcwd()

# THE PRIORITY ORDER
# 1. gptoss is PRIMARY (Source of Truth)
# 2. deepseek_llm is SECOND
# 3. deepseek_code is THIRD
# 4. Others follow
# 5. gemma is explicitly excluded or pushed to last as requested
FOLDERS_ORDER = [
    "gptoss",  # 1. Primary (TikToken based, likely needs tokenizer.json parsing)
    "deepseek_llm",  # 2. Secondary
    "deepseek_code",  # 3. Tertiary
    "mistral",
    "qwen",
    "qwen_code",
    # "olmo",
    # "olmocode",
    "bytedance_ouro",
]

OUTPUT_DIR = "./merged_tokenizer_gptoss_primary"
RRF_K = 60
MAX_VOCAB_SIZE = 128000  # Maximum vocabulary size (including special tokens)
NUM_SPECIAL_TOKENS = (
    512  # Reserve 512 slots for special tokens (256 current + 256 future)
)

# Setup logging
LOG_FILE = "rrf_merge.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(),  # Also print to console
    ],
)
logger = logging.getLogger(__name__)
# =================================================


def generate_special_tokens():
    """Generate special tokens for the tokenizer - fills ALL slots 0-511."""
    special_tokens = []

    # Document structure (0-9)
    special_tokens.extend(
        [
            ("<|begin_of_text|>", 0, "Begin of text marker"),
            ("<|end_of_text|>", 1, "End of text marker"),
            ("<|chunk_sep|>", 2, "Chunk separator"),
            ("<|pad|>", 3, "Padding token"),
            ("<|unk|>", 4, "Unknown token"),
            ("<|bos|>", 5, "Beginning of sequence"),
            ("<|eos|>", 6, "End of sequence"),
            ("<|reserved_7|>", 7, "Reserved"),
            ("<|reserved_8|>", 8, "Reserved"),
            ("<|reserved_9|>", 9, "Reserved"),
        ]
    )

    # Chat roles (10-19)
    special_tokens.extend(
        [
            ("<|system|>", 10, "System message"),
            ("<|user|>", 11, "User message"),
            ("<|assistant|>", 12, "Assistant message"),
            ("<|tool|>", 13, "Tool message"),
            ("<|reserved_14|>", 14, "Reserved"),
            ("<|reserved_15|>", 15, "Reserved"),
            ("<|reserved_16|>", 16, "Reserved"),
            ("<|reserved_17|>", 17, "Reserved"),
            ("<|reserved_18|>", 18, "Reserved"),
            ("<|reserved_19|>", 19, "Reserved"),
        ]
    )

    # Code blocks (20-29)
    special_tokens.extend(
        [
            ("<|code_begin|>", 20, "Code block begin"),
            ("<|code_end|>", 21, "Code block end"),
            ("<|reserved_22|>", 22, "Reserved"),
            ("<|reserved_23|>", 23, "Reserved"),
            ("<|reserved_24|>", 24, "Reserved"),
            ("<|reserved_25|>", 25, "Reserved"),
            ("<|reserved_26|>", 26, "Reserved"),
            ("<|reserved_27|>", 27, "Reserved"),
            ("<|reserved_28|>", 28, "Reserved"),
            ("<|reserved_29|>", 29, "Reserved"),
        ]
    )

    # Language tags (30-49)
    languages = [
        "python",
        "javascript",
        "typescript",
        "java",
        "c",
        "cpp",
        "csharp",
        "go",
        "rust",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "scala",
        "r",
        "sql",
        "html",
        "css",
        "bash",
        "shell",
    ]
    for i, lang in enumerate(languages):
        special_tokens.append((f"<|lang:{lang}|>", 30 + i, f"Language: {lang}"))

    # JSON and tool calling (50-59)
    special_tokens.extend(
        [
            ("<|json_begin|>", 50, "JSON begin"),
            ("<|json_end|>", 51, "JSON end"),
            ("<|tool_call|>", 52, "Tool call"),
            ("<|tool_result|>", 53, "Tool result"),
            ("<|reserved_54|>", 54, "Reserved"),
            ("<|reserved_55|>", 55, "Reserved"),
            ("<|reserved_56|>", 56, "Reserved"),
            ("<|reserved_57|>", 57, "Reserved"),
            ("<|reserved_58|>", 58, "Reserved"),
            ("<|reserved_59|>", 59, "Reserved"),
        ]
    )

    # Source metadata (60-69)
    sources = ["wikipedia", "github", "web", "books", "arxiv", "stackexchange"]
    for i, source in enumerate(sources):
        special_tokens.append((f"<|source:{source}|>", 60 + i, f"Source: {source}"))
    # Fill remaining 60-69
    for i in range(66, 70):
        special_tokens.append((f"<|reserved_{i}|>", i, "Reserved"))

    # Thinking/reasoning tokens (70-79)
    special_tokens.extend(
        [
            ("<|think_begin|>", 70, "Thinking begin"),
            ("<|think_end|>", 71, "Thinking end"),
            ("<|reason_begin|>", 72, "Reasoning begin"),
            ("<|reason_end|>", 73, "Reasoning end"),
            ("<|reserved_74|>", 74, "Reserved"),
            ("<|reserved_75|>", 75, "Reserved"),
            ("<|reserved_76|>", 76, "Reserved"),
            ("<|reserved_77|>", 77, "Reserved"),
            ("<|reserved_78|>", 78, "Reserved"),
            ("<|reserved_79|>", 79, "Reserved"),
        ]
    )

    # Format tokens (80-99)
    special_tokens.extend(
        [
            ("<|markdown|>", 80, "Markdown content"),
            ("<|latex|>", 81, "LaTeX content"),
            ("<|table|>", 82, "Table content"),
            ("<|list|>", 83, "List content"),
            ("<|reserved_84|>", 84, "Reserved"),
            ("<|reserved_85|>", 85, "Reserved"),
            ("<|reserved_86|>", 86, "Reserved"),
            ("<|reserved_87|>", 87, "Reserved"),
            ("<|reserved_88|>", 88, "Reserved"),
            ("<|reserved_89|>", 89, "Reserved"),
        ]
    )

    # Fill remaining slots up to 99
    for i in range(90, 100):
        special_tokens.append((f"<|reserved_{i}|>", i, "Reserved"))

    # Reserved for future use (100-511) - all placeholders
    for i in range(100, 512):
        special_tokens.append((f"<|reserved_{i}|>", i, f"Reserved token {i}"))

    return special_tokens


def should_keep_token(token):
    """
    Filter tokens based on:
    1. Remove special tokens (SPECIAL_*, <|...|>, etc.)
    2. Remove tokens with length > 32
    3. Keep Latin + Indic scripts, remove CJK (Chinese/Japanese/Korean)

    Byte-level BPE encoding:
    - Indic scripts (Devanagari, Tamil, etc.): Start with 0xE0 (à + second byte)
      à¤/à¥ = Devanagari, à¦/à§ = Bengali, à®/à¯ = Tamil, à°/à± = Telugu, etc.
    - CJK scripts: Start with 0xE3-ED
      ã = Japanese, ä-é = Chinese, ê-í = Korean
    """
    # Length check
    if len(token) > 32:
        return False

    # Special token patterns to remove
    special_patterns = [
        r"^SPECIAL_\d+$",  # SPECIAL_100, etc.
        r"^<\|.*\|>$",  # <|endoftext|>, etc.
        r"^\[.*\]$",  # [PAD], [CLS], etc.
        r"^<.*>$",  # <pad>, <eos>, etc.
        r"^▁SPECIAL",  # SentencePiece special tokens
    ]

    for pattern in special_patterns:
        if re.match(pattern, token):
            return False

    # CJK byte patterns to BLOCK (Chinese, Japanese, Korean)
    # These are the first bytes of multi-byte UTF-8 sequences for CJK
    cjk_chars = set("ãäåæçèéêëìí")  # E3, E4-E9, EA-ED ranges

    # Check if token contains CJK patterns
    for char in token:
        if char in cjk_chars:
            return False

    # Cyrillic patterns to BLOCK (Russian, Ukrainian, etc.)
    # Cyrillic uses: Ð (0xD0), Ñ (0xD1), Ò (0xD2), Ó (0xD3)
    cyrillic_chars = set("ÐÑÒÓ")
    for char in token:
        if char in cyrillic_chars:
            return False

    # Arabic patterns to BLOCK
    # Arabic uses: Ø (0xD8), Ù (0xD9), Ú (0xDA), Û (0xDB)
    arabic_chars = set("ØÙÚÛ")
    for char in token:
        if char in arabic_chars:
            return False

    # Everything else is kept (Latin, Indic, symbols, numbers, etc.)
    return True


class TokenizerData:
    def __init__(self, folder_name, priority):
        self.folder = folder_name
        self.path = os.path.join(BASE_DIR, folder_name)
        self.priority = priority
        self.vocab = {}  # token -> id
        self.merges = []  # List of (p1, p2) in order of rank
        self.token_to_parts = {}  # merged_token -> (p1, p2)

        self.load()

    def load(self):
        logger.info(f"[Priority {self.priority}] Loading {self.folder}...")

        json_path = os.path.join(self.path, "tokenizer.json")
        merges_txt_path = os.path.join(self.path, "merges.txt")
        vocab_json_path = os.path.join(self.path, "vocab.json")

        # 1. Try Loading from tokenizer.json (Critical for gptoss)
        if os.path.exists(json_path):
            logger.debug(f"  Found tokenizer.json for {self.folder}")
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract Vocab
                if "model" in data and "vocab" in data["model"]:
                    self.vocab = data["model"]["vocab"]
                    logger.debug(
                        f"  Loaded {len(self.vocab)} vocab entries from tokenizer.json"
                    )
                elif os.path.exists(vocab_json_path):
                    with open(vocab_json_path, "r", encoding="utf-8") as vf:
                        self.vocab = json.load(vf)
                    logger.debug(
                        f"  Loaded {len(self.vocab)} vocab entries from vocab.json"
                    )

                # Extract Merges
                # This is where we catch TikToken/HF Json formatted merges
                if "model" in data and "merges" in data["model"]:
                    raw_merges = data["model"]["merges"]
                    self._parse_merges_list(raw_merges)
                    logger.info(
                        f"  ✓ {self.folder}: Found {len(self.merges)} merges (JSON format), {len(self.vocab)} vocab entries"
                    )
                    return
            except Exception as e:
                logger.error(f"  Error reading JSON for {self.folder}: {e}")

        # 2. Fallback to merges.txt (Standard BPE)
        if os.path.exists(merges_txt_path):
            logger.debug(f"  Found merges.txt for {self.folder}")
            try:
                with open(merges_txt_path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()

                # Skip version header
                start_idx = 1 if lines and lines[0].startswith("#") else 0
                self._parse_merges_list(lines[start_idx:])

                # Load vocab if not already loaded from json
                if not self.vocab and os.path.exists(vocab_json_path):
                    with open(vocab_json_path, "r", encoding="utf-8") as vf:
                        self.vocab = json.load(vf)

                logger.info(
                    f"  ✓ {self.folder}: Found {len(self.merges)} merges (TXT format), {len(self.vocab)} vocab entries"
                )
                return
            except Exception as e:
                logger.error(f"  Error reading merges.txt for {self.folder}: {e}")

        logger.warning(f"  ⚠ {self.folder}: No merges found or file missing!")

    def _parse_merges_list(self, raw_list):
        seen_pairs = set()  # Track seen pairs for O(1) lookup

        for line in raw_list:
            # Handle different merge formats:
            # 1. List format: ['u', 'n'] (e.g., gptoss/TikToken)
            # 2. String format: "u n" (e.g., deepseek, mistral)
            if isinstance(line, list):
                # Already a list
                parts = line
            else:
                # String that needs splitting
                parts = line.split(" ")

            if len(parts) == 2:
                p1, p2 = parts[0], parts[1]
                pair = (p1, p2)

                # Only record if we haven't seen this specific pair yet (sanity check)
                if pair not in seen_pairs:
                    merged = p1 + p2
                    self.merges.append(pair)
                    self.token_to_parts[merged] = pair
                    seen_pairs.add(pair)


def perform_rrf_merge():
    logger.info("=" * 60)
    logger.info("RRF Tokenizer Merge - Starting Process")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"RRF_K parameter: {RRF_K}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 60)

    tokenizers = []

    # Load all tokenizers
    logger.info("\n📂 LOADING TOKENIZERS")
    logger.info("-" * 60)
    for i, folder in enumerate(FOLDERS_ORDER):
        if os.path.isdir(os.path.join(BASE_DIR, folder)):
            tokenizers.append(TokenizerData(folder, i))
        else:
            logger.warning(f"⚠ Skipping {folder} (Directory not found)")

    if not tokenizers:
        logger.error("❌ No tokenizers loaded. Exiting.")
        return

    logger.info(f"\n✓ Successfully loaded {len(tokenizers)} tokenizers")
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: CALCULATING GLOBAL RRF SCORES")
    logger.info("=" * 60)

    token_scores = defaultdict(float)
    token_decomposition = {}
    base_tokens = set()
    token_origins = {}  # Track which tokenizer contributed each token first

    for tok in tokenizers:
        logger.info(f"Processing {tok.folder} (Priority {tok.priority})...")
        # Create a lookup for Rank based on order in merges list
        # Rank 0 = First merge (Highest Priority)
        merge_rank_map = {
            m_tok: i for i, (p1, p2) in enumerate(tok.merges) for m_tok in [p1 + p2]
        }

        for token in tok.vocab:
            # Track origin - only record the FIRST tokenizer that has this token
            if token not in token_origins:
                token_origins[token] = tok.folder

            # RRF Calculation
            # If token is a merge, use its rank. If base char, rank 0.
            rank = merge_rank_map.get(token, 0)
            score = 1.0 / (RRF_K + rank + 1)
            token_scores[token] += score

            # Store Decomposition
            # CRITICAL: We only keep the decomposition from the FIRST tokenizer that defines it.
            # Since 'gptoss' is first in list (Priority 0), its definitions win.
            if token in tok.token_to_parts:
                if token not in token_decomposition:
                    token_decomposition[token] = tok.token_to_parts[token]
            else:
                # If it's not a merge in ANY tokenizer seen so far, it's a base candidate
                base_tokens.add(token)

        logger.info(
            f"  Contributed {len(tok.vocab)} vocab entries, {len(tok.merges)} merge rules"
        )

    # Clean base tokens: remove any that act as a merge in our finalized decomposition map
    final_base_tokens = {t for t in base_tokens if t not in token_decomposition}

    logger.info("\n📊 RRF Score Calculation Results:")
    logger.info(f"  • Unique Tokens: {len(token_scores):,}")
    logger.info(f"  • Merge Rules: {len(token_decomposition):,}")
    logger.info(f"  • Base Tokens: {len(final_base_tokens):,}")

    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: TOPOLOGICAL SORT & MERGE RECONSTRUCTION")
    logger.info("=" * 60)

    final_merges = []
    final_vocab_set = set(final_base_tokens)

    # Dependency Graph
    logger.info("Building dependency graph...")
    child_to_parents = defaultdict(list)
    missing_children_count = defaultdict(int)

    for parent, (p1, p2) in token_decomposition.items():
        child_to_parents[p1].append(parent)
        child_to_parents[p2].append(parent)

        count = 0
        if p1 not in final_vocab_set:
            count += 1
        if p2 not in final_vocab_set:
            count += 1
        missing_children_count[parent] = count

    logger.info(f"  Dependency graph built with {len(child_to_parents)} nodes")

    # Priority Queue: Stores (-score, token)
    logger.info("Initializing priority queue...")
    pq = []
    for token, count in missing_children_count.items():
        if count == 0:
            heapq.heappush(pq, (-token_scores[token], token))

    logger.info(f"  Priority queue initialized with {len(pq)} ready tokens")

    # Process Queue WITHOUT limit (we'll apply limit after filtering)
    logger.info("Processing topological sort...")
    logger.info(
        "  Building all possible merges (will limit to 128k after filtering)..."
    )

    processed_count = 0
    while pq:
        _, token = heapq.heappop(pq)

        if token in final_vocab_set:
            continue

        parts = token_decomposition[token]
        final_merges.append(parts)
        final_vocab_set.add(token)

        processed_count += 1
        if processed_count % 10000 == 0:
            logger.info(f"  Processed {processed_count:,} merges...")

        # Notify parents
        if token in child_to_parents:
            for parent in child_to_parents[token]:
                missing_children_count[parent] -= 1
                if missing_children_count[parent] == 0:
                    heapq.heappush(pq, (-token_scores[parent], parent))

    logger.info(f"  ✓ Topological sort complete: {len(final_merges):,} merges ordered")
    logger.info(
        f"  ✓ Total regular tokens (before filtering): {len(final_vocab_set):,}"
    )

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: FILTERING TOKENS")
    logger.info("=" * 60)
    logger.info("Applying filters:")
    logger.info("  • Removing special tokens (SPECIAL_*, <|...|>, etc.)")
    logger.info("  • Removing tokens with length > 32")
    logger.info("  • Removing CJK scripts (Chinese, Japanese, Korean)")
    logger.info("  • Removing Cyrillic (Russian, Ukrainian, etc.)")
    logger.info("  • Removing Arabic")
    logger.info(
        "  • Keeping Latin + Indic scripts (Devanagari, Tamil, Telugu, Bengali, etc.)"
    )

    # Filter base tokens
    logger.info("\nFiltering base tokens...")
    original_base_count = len(final_base_tokens)

    # Track some examples of removed tokens
    removed_examples = []
    for t in list(final_base_tokens)[:1000]:  # Sample first 1000
        if not should_keep_token(t):
            removed_examples.append(repr(t))
            if len(removed_examples) >= 10:
                break

    final_base_tokens = {t for t in final_base_tokens if should_keep_token(t)}
    removed_base = original_base_count - len(final_base_tokens)
    logger.info(
        f"  Base tokens: {original_base_count:,} → {len(final_base_tokens):,} (removed {removed_base:,})"
    )

    if removed_examples:
        logger.info(f"  Examples of removed tokens: {', '.join(removed_examples[:5])}")

    # Filter merges - only keep merges where both parts and result are in allowed set
    logger.info("Filtering merge rules...")
    original_merge_count = len(final_merges)
    filtered_merges = []
    final_vocab_set = set(final_base_tokens)
    skipped_merges = []  # Track skipped merges with reasons

    for p1, p2 in final_merges:
        merged = p1 + p2
        skip_reason = None

        # Check if the resulting token passes the filter
        if not should_keep_token(merged):
            skip_reason = "failed_filter"
        # Also check if both parts are available in our vocabulary
        elif p1 not in final_vocab_set and p2 not in final_vocab_set:
            skip_reason = "both_parts_missing"
        elif p1 not in final_vocab_set:
            skip_reason = "part1_missing"
        elif p2 not in final_vocab_set:
            skip_reason = "part2_missing"

        if skip_reason:
            # Track which tokenizer this merge came from
            merge_origin = token_origins.get(merged, "unknown")
            skipped_merges.append(
                {
                    "merged_token": merged,
                    "merged_token_repr": repr(merged),
                    "part1": p1,
                    "part1_repr": repr(p1),
                    "part2": p2,
                    "part2_repr": repr(p2),
                    "skip_reason": skip_reason,
                    "origin_tokenizer": merge_origin,
                    "rrf_score": token_scores.get(merged, 0),
                }
            )
        else:
            filtered_merges.append((p1, p2))
            final_vocab_set.add(merged)

    removed_merges = original_merge_count - len(filtered_merges)
    logger.info(
        f"  Merge rules: {original_merge_count:,} → {len(filtered_merges):,} (removed {removed_merges:,})"
    )
    logger.info(f"  Skipped merges tracked: {len(skipped_merges):,}")

    final_merges = filtered_merges

    logger.info("\n📊 After filtering:")
    logger.info(f"  • Total vocabulary size: {len(final_vocab_set):,}")
    logger.info(f"  • Base tokens: {len(final_base_tokens):,}")
    logger.info(f"  • Merge rules: {len(final_merges):,}")

    # Apply 128k limit AFTER filtering
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: APPLYING 128K VOCABULARY LIMIT")
    logger.info("=" * 60)
    logger.info(
        f"  Target vocab size: {MAX_VOCAB_SIZE:,} (including {NUM_SPECIAL_TOKENS} special tokens)"
    )
    max_regular_tokens = MAX_VOCAB_SIZE - NUM_SPECIAL_TOKENS
    logger.info(f"  Max regular tokens: {max_regular_tokens:,}")

    current_size = len(final_base_tokens) + len(final_merges)
    logger.info(f"  Current size after filtering: {current_size:,}")

    if current_size > max_regular_tokens:
        # Need to trim merges
        tokens_to_remove = current_size - max_regular_tokens
        logger.info(f"  Trimming {tokens_to_remove:,} merges to reach limit...")

        # Keep only the first (max_regular_tokens - base_tokens) merges
        max_merges = max_regular_tokens - len(final_base_tokens)
        final_merges = final_merges[:max_merges]

        logger.info(f"  ✓ Trimmed to {len(final_merges):,} merges")
    else:
        logger.info("  ✓ Under limit, no trimming needed")

    final_vocab_size = len(final_base_tokens) + len(final_merges) + NUM_SPECIAL_TOKENS
    logger.info(f"  Final vocabulary size: {final_vocab_size:,}")

    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: SAVING OUTPUT FILES")
    logger.info("=" * 60)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"  Created output directory: {OUTPUT_DIR}")

    # Write merges.txt
    logger.info("Writing merges.txt...")
    merges_path = os.path.join(OUTPUT_DIR, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for p1, p2 in final_merges:
            f.write(f"{p1} {p2}\n")
    logger.info(f"  ✓ Saved {len(final_merges):,} merge rules to {merges_path}")

    # Write vocab.json with special tokens
    logger.info("Building vocabulary with special tokens...")

    # Generate special tokens
    special_tokens = generate_special_tokens()
    logger.info(f"  Generated {len(special_tokens)} special tokens (IDs 0-511)")

    vocab_dict = {}

    # Add special tokens first (IDs 0-511)
    for token_str, token_id, description in special_tokens:
        vocab_dict[token_str] = token_id

    # Start regular tokens at ID 512
    idx = NUM_SPECIAL_TOKENS

    # Add base tokens (sorted)
    sorted_base = sorted(list(final_base_tokens))
    for t in sorted_base:
        vocab_dict[t] = idx
        idx += 1

    logger.info(
        f"  Added {len(final_base_tokens):,} base tokens (IDs {NUM_SPECIAL_TOKENS}-{idx-1})"
    )

    # Add merged tokens
    merge_start_id = idx
    for p1, p2 in final_merges:
        merged = p1 + p2
        vocab_dict[merged] = idx
        idx += 1

    logger.info(
        f"  Added {len(final_merges):,} merged tokens (IDs {merge_start_id}-{idx-1})"
    )

    logger.info("Writing vocab.json...")
    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✓ Saved {len(vocab_dict):,} vocabulary entries to {vocab_path}")

    # Export non-gptoss tokens to CSV
    logger.info("\nExporting non-gptoss tokens to CSV...")
    csv_path = os.path.join(OUTPUT_DIR, "non_gptoss_tokens.csv")

    non_gptoss_tokens = []
    for token, vocab_id in vocab_dict.items():
        origin = token_origins.get(token, "unknown")
        if origin != "gptoss":
            token_type = "merge" if token in token_decomposition else "base"
            merge_parts = ""
            if token_type == "merge":
                p1, p2 = token_decomposition[token]
                merge_parts = f"{repr(p1)} + {repr(p2)}"

            non_gptoss_tokens.append(
                {
                    "token": token,
                    "token_repr": repr(token),
                    "origin_tokenizer": origin,
                    "token_type": token_type,
                    "merge_parts": merge_parts,
                    "rrf_score": token_scores.get(token, 0),
                    "vocab_id": vocab_id,
                }
            )

    # Sort by vocab_id for easier analysis
    non_gptoss_tokens.sort(key=lambda x: x["vocab_id"])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if non_gptoss_tokens:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "vocab_id",
                    "token",
                    "token_repr",
                    "origin_tokenizer",
                    "token_type",
                    "merge_parts",
                    "rrf_score",
                ],
            )
            writer.writeheader()
            writer.writerows(non_gptoss_tokens)

    logger.info(
        f"  ✓ Exported {len(non_gptoss_tokens):,} non-gptoss tokens to {csv_path}"
    )

    # Summary by tokenizer
    origin_counts = defaultdict(int)
    for token_info in non_gptoss_tokens:
        origin_counts[token_info["origin_tokenizer"]] += 1

    logger.info("\n  Breakdown by source tokenizer:")
    for tokenizer, count in sorted(
        origin_counts.items(), key=lambda x: x[1], reverse=True
    ):
        logger.info(f"    • {tokenizer}: {count:,} tokens")

    # Export skipped merges to CSV
    logger.info("\nExporting skipped merges to CSV...")
    skipped_csv_path = os.path.join(OUTPUT_DIR, "skipped_merges.csv")

    # Sort by RRF score (highest first) to see what high-value merges we lost
    skipped_merges.sort(key=lambda x: x["rrf_score"], reverse=True)

    with open(skipped_csv_path, "w", newline="", encoding="utf-8") as f:
        if skipped_merges:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "merged_token",
                    "merged_token_repr",
                    "part1",
                    "part1_repr",
                    "part2",
                    "part2_repr",
                    "skip_reason",
                    "origin_tokenizer",
                    "rrf_score",
                ],
            )
            writer.writeheader()
            writer.writerows(skipped_merges)

    logger.info(
        f"  ✓ Exported {len(skipped_merges):,} skipped merges to {skipped_csv_path}"
    )

    # Summary by skip reason
    skip_reason_counts = defaultdict(int)
    for merge_info in skipped_merges:
        skip_reason_counts[merge_info["skip_reason"]] += 1

    logger.info("\n  Breakdown by skip reason:")
    for reason, count in sorted(
        skip_reason_counts.items(), key=lambda x: x[1], reverse=True
    ):
        logger.info(f"    • {reason}: {count:,} merges")

    # Create complete tokenizer.json file
    logger.info("\nCreating tokenizer.json...")
    tokenizer_json_path = os.path.join(OUTPUT_DIR, "tokenizer.json")

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": token_id,
                "content": token_str,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for token_str, token_id, _ in special_tokens
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<|unk|>",
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": True,
            "vocab": vocab_dict,
            "merges": [f"{p1} {p2}" for p1, p2 in final_merges],
        },
    }

    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✓ Saved tokenizer.json to {tokenizer_json_path}")

    # Create tokenizer_config.json
    logger.info("Creating tokenizer_config.json...")
    tokenizer_config_path = os.path.join(OUTPUT_DIR, "tokenizer_config.json")

    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            str(token_id): {
                "content": token_str,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
            for token_str, token_id, _ in special_tokens
        },
        "bos_token": "<|bos|>",
        "clean_up_tokenization_spaces": True,
        "eos_token": "<|eos|>",
        "legacy": False,
        "model_max_length": 8192,
        "pad_token": "<|pad|>",
        "padding_side": "right",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "truncation_side": "right",
        "unk_token": "<|unk|>",
        "use_default_system_prompt": False,
        "chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}<|system|>{{ message['content'] }}{% elif message['role'] == 'user' %}<|user|>{{ message['content'] }}{% elif message['role'] == 'assistant' %}<|assistant|>{{ message['content'] }}{% endif %}{% endfor %}",
        "vocabulary_size": len(vocab_dict),
        "special_tokens": {
            "bos_token": "<|bos|>",
            "eos_token": "<|eos|>",
            "unk_token": "<|unk|>",
            "pad_token": "<|pad|>",
            "additional_special_tokens": [
                token_str for token_str, _, _ in special_tokens
            ],
        },
    }

    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✓ Saved tokenizer_config.json to {tokenizer_config_path}")

    # Create special_tokens_map.json
    logger.info("Creating special_tokens_map.json...")
    special_tokens_map_path = os.path.join(OUTPUT_DIR, "special_tokens_map.json")

    special_tokens_map = {
        "bos_token": {
            "content": "<|bos|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
        },
        "eos_token": {
            "content": "<|eos|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
        },
        "pad_token": {
            "content": "<|pad|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
        },
        "unk_token": {
            "content": "<|unk|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
        },
    }

    with open(special_tokens_map_path, "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✓ Saved special_tokens_map.json to {special_tokens_map_path}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ SUCCESS! MERGED TOKENIZER CREATED")
    logger.info("=" * 60)
    logger.info(f"📁 Output Location: {OUTPUT_DIR}")
    logger.info("\n📊 Final Tokenizer Statistics:")
    logger.info(f"  • Total Vocabulary Size: {len(vocab_dict):,} tokens")
    logger.info(f"  • Special Tokens: {NUM_SPECIAL_TOKENS} (IDs 0-511)")
    logger.info(
        f"  • Base Tokens: {len(final_base_tokens):,} (IDs {NUM_SPECIAL_TOKENS}+)"
    )
    logger.info(f"  • Merge Rules: {len(final_merges):,}")
    logger.info(f"  • Non-gptoss tokens: {len(non_gptoss_tokens):,}")
    logger.info(f"  • Vocabulary limit: {MAX_VOCAB_SIZE:,} (enforced after filtering)")
    logger.info("\n🌍 Language Support:")
    logger.info("  • Latin scripts: English and other European languages")
    logger.info(
        "  • Indic scripts: Hindi (Devanagari), Tamil, Telugu, Bengali, Gujarati,"
    )
    logger.info("    Kannada, Malayalam, Odia, Punjabi (Gurmukhi)")
    logger.info("  • Byte-level BPE encoding for efficient compression")
    logger.info("  • Filtered out: Chinese, Japanese, Korean, Arabic, Cyrillic")
    logger.info("  • Max token length: 32 characters")
    logger.info("\n📄 Output Files:")
    logger.info("  Core tokenizer files:")
    logger.info("    • tokenizer.json - HuggingFace tokenizer definition")
    logger.info("    • tokenizer_config.json - Tokenizer configuration")
    logger.info("    • special_tokens_map.json - Special tokens mapping")
    logger.info("    • vocab.json - Complete vocabulary with IDs")
    logger.info("    • merges.txt - BPE merge rules")
    logger.info("  Analysis files:")
    logger.info("    • non_gptoss_tokens.csv - Contributions from other tokenizers")
    logger.info("    • skipped_merges.csv - Merges excluded by filters")
    logger.info("\n💡 Usage:")
    logger.info("  from transformers import AutoTokenizer")
    logger.info(f"  tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}')")
    logger.info(f"\n📝 Log file: {LOG_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        perform_rrf_merge()
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ ERROR: Process failed with exception")
        logger.error("=" * 60)
        logger.error(f"Exception: {str(e)}", exc_info=True)
        logger.error(f"📝 Check log file for details: {LOG_FILE}")
        raise
