# Tokenizer Quality Audit — Complete Reference Guide

A comprehensive, automated testing and reporting system that evaluates a HuggingFace-compatible
BPE tokenizer across **22 distinct quality dimensions** before model training begins.

---

## Table of Contents

1. [What This Tool Does](#1-what-this-tool-does)
2. [Project File Structure](#2-project-file-structure)
3. [Input Files & Required Format](#3-input-files--required-format)
4. [How to Install & Run](#4-how-to-install--run)
5. [All 22 Tests — What They Check and Why](#5-all-22-tests--what-they-check-and-why)
6. [Output Files — Column-by-Column Guide](#6-output-files--column-by-column-guide)
7. [Reading the Markdown Report](#7-reading-the-markdown-report)
8. [Understanding Key Findings](#8-understanding-key-findings)
9. [Adding New Data Sources](#9-adding-new-data-sources)
10. [Frequently Asked Questions](#10-frequently-asked-questions)

---

## 1. What This Tool Does

Before training a large language model (LLM), it is critical to verify that the tokenizer:

- **Covers all languages** in the training corpus without resorting to single-byte fallback tokens
- **Round-trips losslessly** — encoding then decoding produces the original text, byte for byte
- **Uses special tokens correctly** — chat control tokens (`<|user|>`, `<|assistant|>`) appear as
  single IDs and are not confused by near-lookalike strings
- **Marks the right spans for learning** — SFT (Supervised Fine-Tuning) loss masking identifies
  exactly the assistant turns that the model should learn
- **Has no contamination** — pretraining data is free from injected control tokens or reserved
  placeholder tokens that could corrupt model behaviour

This script runs all 22 checks automatically across every dataset, writes a human-readable
Markdown report, and emits a machine-readable JSON for CI integration.

---

## 2. Project File Structure

```
tokeniser_testing/
│
├── tokenizer_audit.py          ← Main audit script (run this)
├── README.md                   ← This document
│
├── tokeniser/                  ← Your tokenizer (INPUT — do not modify)
│   ├── tokenizer.json          ← BPE vocabulary + merge rules (7.7 MB)
│   ├── tokenizer_config.json   ← Special tokens, class, settings (65 KB)
│   └── special_tokens_map.json ← BOS / EOS / PAD / UNK mappings
│
├── data/                       ← Datasets (INPUT — do not modify)
│   ├── golden_samples_cleaned_v3.jsonl   ← 128 hand-curated QA samples
│   ├── raw_shard.parquet                 ← 630K documents with text (1.5 GB)
│   ├── raw_manifest.parquet             ← 629K rows, metadata only (40 MB)
│   └── manifest.parquet                 ← 3.3M rows, metadata only (165 MB)
│
├── sft_data/                   ← Supervised fine-tuning data (INPUT)
│   ├── group1_assamese.txt     ← Assamese QA (12,294 lines)
│   ├── group1_hindi.txt        ← Hindi QA (14,323 lines)
│   ├── group1_marathi.txt      ← Marathi QA (16,101 lines)
│   ├── group1_punjabi.txt      ← Punjabi QA (16,852 lines)
│   ├── group1_telugu.txt       ← Telugu QA (17,716 lines)
│   ├── group2.txt              ← Multilingual reasoning (9,710 lines)
│   └── group3.txt              ← Pattern/sequence tasks (2,512 lines)
│
└── report/                     ← Generated output (auto-created by script)
    ├── tokenizer_audit_report.md
    ├── tokenizer_audit_results.json
    ├── token_frequency.csv
    ├── freq_<dataset>.csv        (one per text dataset)
    └── golden_sample_token_counts.csv
```

> **Note:** The `report/` directory is created automatically on the first run.
> All files inside are **regenerated completely** on each run.

---

## 3. Input Files & Required Format

### Tokenizer (`tokeniser/`)

A standard HuggingFace tokenizer directory, loadable by `AutoTokenizer.from_pretrained()`.

| File | Purpose |
|------|---------|
| `tokenizer.json` | Core BPE model: vocabulary (131,072 entries) + merge rules |
| `tokenizer_config.json` | Class name, special token declarations, added_tokens_decoder |
| `special_tokens_map.json` | Maps role names (bos_token, eos_token, pad_token) to their strings |

### Data Files (`data/`)

| File | Format | Required Columns | What It Is |
|------|--------|------------------|------------|
| `golden_samples_cleaned_v3.jsonl` | JSON Lines | `id`, `tag`, `text` | 128 hand-crafted QA pairs covering math, code, and all Indic languages — the "truth set" |
| `raw_shard.parquet` | Parquet | `text`, `language`, `source` | 630K raw pre-training documents across languages |
| `raw_manifest.parquet` | Parquet | `language`, `domain`, `source`, `band`, `word_count`, `token_est` | Metadata for 629K docs — **no text column**, used for statistical analysis only |
| `manifest.parquet` | Parquet | same as raw_manifest | Metadata for the full 3.3M-document corpus |

### SFT Files (`sft_data/`)

Plain UTF-8 text files. Each **line** is treated as one document. Any `.txt` file you
drop into `sft_data/` is **automatically discovered** on the next run — no code changes needed.

---

## 4. How to Install & Run

### Python requirements

Python 3.9 or higher.

```bash
pip install transformers pyarrow pandas numpy tqdm
```

Or from the provided requirements file:

```
transformers>=4.40.0
pyarrow>=14.0.0
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
```

### Run modes

#### Standard run (recommended for most use cases)

Tokenizes all SFT files in full + samples 50,000 rows from `raw_shard`. The ghost-tag sweep
and special-token leakage scan always process **all 630K rows** regardless of mode.

```bash
python tokenizer_audit.py
```

#### Full shard run (most accurate vocab coverage)

Tokenizes all 630,140 rows of `raw_shard.parquet`. Takes ~30 minutes on modern hardware.
Use this before finalizing training runs to get the most accurate unused-token percentages.

```bash
python tokenizer_audit.py --full-shard
```

#### Debug / fast run

Tokenizes only 5,000 shard rows and 500 lines per SFT file.
Useful for rapid iteration while modifying the script.

```bash
python tokenizer_audit.py --shard-rows 5000 --sft-lines 500
```

#### Skip shard tokenization

Ghost-tag sweep, special-token leakage scan, and all other tests still run.
Token frequency analysis uses only SFT + golden data.

```bash
python tokenizer_audit.py --shard-rows 0
```

#### Custom paths

```bash
python tokenizer_audit.py \
  --tokenizer /path/to/my_tokenizer/ \
  --report    /path/to/output_dir/
```

### All CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tokenizer` | `tokeniser/` | Path to the HuggingFace tokenizer directory |
| `--report` | `report/` | Directory where all output files are written |
| `--shard-rows` | `50000` | Rows from `raw_shard.parquet` to tokenize. `0` = skip tokenization (ghost scan still runs). |
| `--sft-lines` | `0` | Lines per SFT file to tokenize. `0` = **all lines** (default). |
| `--full-shard` | off | Tokenize all 630K rows; overrides `--shard-rows`. |

---

## 5. All 22 Tests — What They Check and Why

### Test 1 — Special Token Integrity

**What it checks:** Verifies that every required control token is present in the vocabulary,
that no unwanted legacy tokens exist, and that no two special tokens share the same ID.

**Why it matters:** Missing a single special token (e.g. `<|pad|>`) crashes training or
produces silent data corruption in padding operations. Duplicate IDs cause two tokens to
compete for the same slot, breaking generation unpredictably.

**Expected result:** ✅ PASS for all sub-checks. Any FAIL is a blocker before training.

---

### Test 2 — Encode / Decode Round-trip

**What it checks:** Encodes 23 text samples spanning English, all major Indic scripts,
Arabic, Chinese, code, math expressions, JSON, CRLF line endings, and emoji, then
decodes each and confirms `decode(encode(text)) == text`.

**Why it matters:** A round-trip failure means the tokenizer permanently corrupts text.
A training sample that cannot be reconstructed from its tokens trains the model on
wrong data. Common causes are spurious leading spaces introduced by `add_prefix_space`.

**Expected result:** ✅ PASS (23/23). Any failure must be investigated immediately.

---

### Test 3 — Special Token Single-ID Check

**What it checks:** Each of the 356 added special tokens must encode as exactly
**one token ID**, not multiple IDs. Tested by calling `encode(token_string)` and
confirming the result has length 1.

**Why it matters:** If `<|assistant|>` encodes as `['<', '|', 'assistant', '|', '>']`
(5 IDs) instead of a single ID, then the SFT loss-masking logic which scans for that
specific ID will never find it — and the model will train on masked (zero-loss) tokens,
learning nothing from the training data.

**Expected result:** ✅ PASS (356/356). Failures require re-training the tokenizer with
correct `add_special_tokens` configuration.

---

### Test 4 — Ghost Tag / Format Drift

**What it checks:** Scans **every document in every dataset** (full 630K rows of
`raw_shard`, all SFT lines, all golden samples) for occurrences of old plain-text
chat formats: `[USER]`, `[ASSISTANT]`, `<AGENT>`, `[INST]`, `###Human:`,
`<|startoftext|>`, and similar.

**Why it matters:** These are legacy formats from earlier chat models. When a training
document contains `[ASSISTANT]`, the model sees it as 5 ordinary text tokens
(`[`, `ASS`, `IST`, `ANT`, `]`) — not as the structured control token `<|assistant|>`.
The model then learns that both formats are valid "assistant markers", producing
inconsistent output at inference time and breaking downstream applications that parse
structured responses.

**What to do if found:** Run a cleaning script that replaces all occurrences of the
old format tokens with their structured equivalents before training.

---

### Test 5 — Vocabulary Utilisation

**What it checks:** After tokenizing all datasets, measures how many of the 131,072
vocabulary entries were **never seen**. Reports unused%, rare-token count (seen fewer
than 5 times), and per-dataset breakdowns.

**Why it matters:** A vocabulary entry that never appears in training data becomes a
"dead token" — the model's embedding for it will remain at its random initialization
value and contribute noise to nearest-neighbor lookups. Very high unused percentages
(>20%) suggest the vocabulary was trained on more data than the current corpus covers,
or that certain language sections of the vocabulary have no matching training data.

**What to do:** Run `--full-shard` to get the most accurate number before drawing
conclusions. If still >20%, consider either pruning the vocabulary or adding more
diverse training data.

---

### Test 6 — Token Length Distribution

**What it checks:** For every document in every text dataset, counts the number of
tokens produced. Reports mean, median, standard deviation, and percentiles
(P25, P75, P90, P95, P99) per dataset and combined.

**Why it matters:** These statistics directly inform your training batch configuration.
If P99 is 4,800 tokens and you set `max_sequence_length=4096`, you are silently
truncating the top 1% of documents and potentially losing critical information from
long scientific or legal texts.

**Column explanations:**
- **N** — number of documents in this dataset
- **Mean** — average tokens per document
- **Median** — 50th percentile; less sensitive to outliers than mean
- **P90 / P95 / P99** — 90th / 95th / 99th percentile lengths; use these to choose your
  `max_sequence_length` without losing too much data
- **Max** — longest single document in tokens

---

### Test 7 — SFT Loss Masking Simulation

**What it checks:** Simulates the loss-masking step that SFT training frameworks apply.
For each sample format, encodes the full conversation, then runs the masking logic which
sets all tokens to `-100` (masked) except tokens inside
`<|assistant|>…<|end_turn|>` spans. Reports how many tokens are unmasked (eligible
for loss) and whether PAD tokens are correctly masked.

**Why it matters:** If the masking logic does not find any `<|assistant|>` span, every
token in that training sample is masked — the model computes zero loss and learns
nothing. This is a silent failure: training appears to proceed normally but the model
does not improve.

**Column explanations:**
- **Format** — the conversation template being tested (see format details below)
- **Tokens** — total token IDs in the encoded sequence
- **Unmasked** — tokens that will contribute to the training loss (the answer the model must learn to produce). Should be > 0 for any real response.
- **PAD OK** — ✅ means all `<|pad|>` tokens are correctly masked with `-100` so padding never contributes to loss
- **Assistant Detected** — ✅ means the masking logic found at least one `<|assistant|>` token and unmasked the content after it

---

#### How the masking logic works (step by step)

The `make_sft_label_mask` function scans token IDs left-to-right:

```
Every token starts as -100 (masked = does NOT train on it)

When it sees token ID for <|assistant|>:
  → All following tokens are UNMASKED (set to their real ID)
  → Until it hits <|end_turn|>, <|im_end|>, <|EOT|>, or EOS
  → That terminator token is also included in training (unmasked)
  → Then masking resumes for the next user/system turn

PAD tokens are always kept at -100 regardless of position.
```

The model only learns from the **assistant's words** — never from the system prompt,
user question, or tool results.

---

#### Why `fim` shows ❌ (Unmasked = 0)

**FIM** stands for **Fill-in-the-Middle** — a code completion training format.
Instead of a user/assistant conversation, FIM uses three special boundary tokens:

```
<|fim_prefix|>  def add(a, b):
<|fim_suffix|>      return result
<|fim_middle|>      result = a + b
```

The part the model must learn to generate is the **middle** section — everything
after `<|fim_middle|>`.

**The problem:** The masking logic only recognizes `<|assistant|>` as the "start
learning here" marker. It never encounters `<|assistant|>` in a FIM sequence, so
it masks every single token — Unmasked = 0.

**This is NOT a tokenizer bug.** The tokenizer correctly encodes all three FIM
tokens as single IDs. The issue is that FIM training requires its **own separate
masking rule**:

```python
# Correct FIM masking (not yet in the script):
if token == fim_middle_id:
    # unmask everything from here to EOS
```

**Action required:** If you plan to use FIM data during SFT, add a second masking
branch to your training data collator that handles `<|fim_middle|>` → EOS spans
in addition to the `<|assistant|>` → `<|end_turn|>` spans.

---

#### Why `golden_math_reasoning` shows ❌ (Unmasked = 0)

The three failing golden samples are real QA examples from your
`golden_samples_cleaned_v3.jsonl` file. They use the **old plain-bracket chat format**:

```
[USER] What is the integral of x²?
[ASSISTANT] The integral of x² is x³/3 + C ...
```

When the tokenizer encodes `[ASSISTANT]`, it does **not** produce the special token
`<|assistant|>` (token ID 130728). Instead it produces **5 ordinary text tokens**:

| Character | Token produced |
|-----------|----------------|
| `[` | regular text token for `[` |
| `ASS` | regular text subword |
| `IST` | regular text subword |
| `ANT` | regular text subword |
| `]` | regular text token for `]` |

The masking logic scans for token ID **130728** (`<|assistant|>`). It never finds it
in these samples because `[ASSISTANT]` encoded as 5 different IDs. So every token
stays at `-100`, Unmasked = 0, and the model would learn **nothing** from this sample.

**This confirms the golden samples cannot be used directly for SFT training.**
They are suitable for evaluation of general comprehension, but must be reformatted
before fine-tuning:

```
[USER] What is the integral of x²?          ← OLD (broken for SFT)
[ASSISTANT] The integral is x³/3 + C

<|user|>What is the integral of x²?<|end_turn|>   ← CORRECT
<|assistant|>The integral is x³/3 + C<|end_turn|>
```

After reformatting, the masking logic will correctly detect token ID 130728 and
unmask all the tokens between `<|assistant|>` and `<|end_turn|>`.

---

#### Summary of all Test 7 formats

| Format | Template Used | Why It Passes or Fails |
|--------|--------------|------------------------|
| `structured` | `<\|system\|>…<\|user\|>…<\|assistant\|>answer<\|end_turn\|>` | ✅ Uses real `<\|assistant\|>` token — masking finds it |
| `multi_turn` | Two full user/assistant rounds | ✅ Two separate assistant spans both unmasked |
| `code` | Assistant response wrapped in `<\|code_begin\|>…<\|code_end\|>` | ✅ Code block content is inside the assistant span |
| `tool_use` | Assistant calls a tool, then gives a final answer | ✅ Final `<\|assistant\|>` span correctly unmasked |
| `fim` | `<\|fim_prefix\|>…<\|fim_suffix\|>…<\|fim_middle\|>…` | ❌ No `<\|assistant\|>` token — needs separate FIM masking rule |
| `golden_math_reasoning` | `[USER]…[ASSISTANT]…` (plain brackets) | ❌ `[ASSISTANT]` → 5 text tokens, not token ID 130728 |

---

### Test 8 — Sequence Length Checklist (1K → 256K tokens)

**What it checks:** Sources real documents from `raw_shard` and SFT data, concatenates
them until a target token count is reached (1K, 2K, 4K, 8K, 16K, 32K, 64K, 131K, 256K),
then verifies:
1. Encode succeeds without error
2. Decode produces valid UTF-8 output
3. Re-encoding the decoded text produces the same token IDs (stability)

**Why it matters:** Tokenizers can have subtle length bugs that only manifest beyond
certain sizes (e.g. a batch dimension overflows at 65536 tokens). Confirming stability at
each checkpoint gives confidence before you invest weeks of GPU time in training at that
sequence length.

**Column explanations:**
- **Encode** — ✅ if tokenizer.encode() completed without error
- **Decode** — ✅ if tokenizer.decode() returned valid text
- **Re-encode Stable** — ✅ if `encode(decode(ids)) == ids`; any mismatch is a tokenizer bug
- **Status** — PASS, FAIL, or INSUFFICIENT_DATA (not enough real text was available to
  reach the target length — use `--full-shard` to pool more data)

---

### Test 9 — Multilingual Coverage

**What it checks:** Encodes a paragraph of native-script text for each of 21 languages
(English, Hindi, Telugu, Marathi, Punjabi, Assamese, Bengali, Tamil, Kannada, Gujarati,
Odia, Malayalam, Urdu, Arabic, Chinese, Japanese, Korean, Russian, French, German, Code),
checks for UNK tokens, and verifies round-trip fidelity.

**Why it matters:** A zero-UNK result with successful round-trip confirms the vocabulary
has full Unicode coverage for that script. Any UNK means some characters cannot be
represented and will be silently dropped during training and inference.

**Column explanations:**
- **Tokens** — how many tokens the paragraph produces (lower with more script coverage)
- **Round-trip** — ✅ / ❌ for lossless reconstruction
- **UNK Count** — number of `<unk>` tokens produced; should be 0 for all languages
- **UNK %** — UNK tokens as a percentage of total tokens for that language

---

### Test 10 — Semantic Duplicate Tokens

**What it checks:** Iterates the entire 131,072-token vocabulary and finds any two token
IDs that decode to the **same string**. Reports all duplicates found.

**Why it matters:** If token ID 42 and token ID 9876 both decode to `"hello"`, the model
has two embeddings for the same surface form. During training they receive separate
gradient updates, introducing noise. During inference, which ID gets generated is
unpredictable.

---

### Test 11 — Edge Cases / Byte Fallback

**What it checks:** A battery of 12 corner-case inputs: empty string, null byte, byte
order mark (BOM), repeated emoji, CRLF line endings, zero-width spaces, mixed-script
tokens, very long repeated characters, and raw control bytes.

**Why it matters:** These inputs appear in real web-crawled data. If any causes an
exception or produces unexpected output, it must be filtered from training data or the
tokenizer must be patched.

---

### Test 12 — Config Integrity

**What it checks:** Reads `tokenizer_config.json` and verifies key settings:
- `clean_up_tokenization_spaces` should be `False` for BPE (True would strip spaces from
  decoded output, corrupting indented code and Markdown)
- `padding_side` and `truncation_side` should match your training framework's expectation
- All special token IDs (BOS, EOS, PAD) are correctly declared
- `model_max_length` reflects the model's actual context window

---

### Test 13 — Byte-Fallback Rate & Tokens-per-Character Efficiency

**What it checks:** For each of 15 languages, encodes a representative paragraph and
measures:
1. **Chars-per-token** — how many Unicode characters each token represents on average
2. **Byte-fallback %** — what fraction of the encoded tokens are raw single-byte pieces

**Why it matters:** A BPE tokenizer should represent common Indic syllables as single
multi-character tokens. If the vocabulary lacks enough merges for a script, words get
fragmented into individual bytes (each byte = 1 token). This can triple or quadruple
the token count for that language, wasting context window and making it much harder for
the model to learn word-level meaning.

**Thresholds:**
- **Chars/token ≥ 2.0** is excellent for morphologically-rich scripts (Hindi, Telugu, etc.)
- **Byte-fallback < 5%** is ideal; 5–20% is a warning; >20% indicates poor script coverage

---

### Test 14 — Numeric Tokenization Analysis

**What it checks:** Encodes 30 numeric formats — single digits through 15-digit numbers,
scientific notation, financial values (₹, $), dates, version strings, phone numbers,
hex/binary/octal, and numbers embedded in sentences.

**Why it matters:** Numbers are encoded very differently across tokenizers. If the
tokenizer produces one token per digit (e.g. `1`, `2`, `3`, `4`, `5` for `12345`), the
model cannot learn multi-digit arithmetic without seeing enormous amounts of numerical data.
Tokenizers that group digits (e.g. `123`, `45`) are much more efficient for math tasks.

**What to look for:** Token counts of 1–2 for small integers are ideal. Token counts
equal to the number of digits (fully fragmented) is a warning for mathematical tasks.

---

### Test 15 — Reserved Token Utilization

**What it checks:** Reads all `<|reserved_N|>` placeholder tokens from
`tokenizer_config.json` and checks whether any of them appear in the training corpus.

**Why it matters:** Reserved tokens are intentionally left empty for future assignment.
If a reserved token accidentally appears in training data (e.g. injected by a data
processing bug), the model learns to generate it as regular text. When that reserved slot
is later activated (e.g. assigned to `<|tool_result|>`), the model will generate it
inappropriately.

**Expected result:** All reserved tokens should have **zero frequency** in the corpus.

---

### Test 16 — Special Token Leakage in Pretraining Data

**What it checks:** Scans **all 630,140 rows** of `raw_shard.parquet` for occurrences
of chat control tokens as plain text strings: `<|system|>`, `<|user|>`, `<|assistant|>`,
`<|end_turn|>`, `<|im_start|>`, `<|im_end|>`, `<|begin_of_text|>`, `<|end_of_text|>`,
and others.

**Why it matters:** Pretraining data is raw web crawl. Some web pages contain scraped
conversations from ChatGPT, Claude, or other chat systems, complete with their original
control tokens. If the model is pre-trained on text that contains `<|assistant|>` as
plain characters, it learns that this string can appear in any context — breaking the
strict role of `<|assistant|>` as a structural boundary token.

**Severity guide:**
- 🔵 LOW: < 10 documents — isolated occurrences, likely from copied model outputs
- 🟡 MEDIUM: 10–100 documents — partial contamination, consider filtering
- 🔴 HIGH: > 100 documents — systematic contamination, must filter before training

---

### Test 17 — Adversarial Token Injection Sweep

**What it checks:** 14 carefully crafted adversarial strings — Cyrillic lookalikes,
fullwidth Unicode brackets, partial special tokens, null bytes, RTL override characters,
and prompt-injection patterns — are encoded. The test checks whether any produce the
real `<|assistant|>` token ID (130728) in the output.

**Why it matters:** In a production chat application, user input is passed through the
tokenizer before reaching the model. If a user can craft a string that encodes as the
real assistant token, they can inject fake "assistant responses" into the conversation,
potentially extracting sensitive system prompt content or bypassing content filters.

**The adversarial strings tested:**
| Pattern | Technique |
|---------|-----------|
| `<\|assistant\| ` | Missing closing `>` |
| `<\|ASSISTANT\|>` | Uppercase variant |
| `＜\|assistant\|＞` | Fullwidth angle brackets (U+FF1C/FF1E) |
| `<｜assistant｜>` | Fullwidth pipe (U+FF5C) |
| `<\|аssistant\|>` | Cyrillic `а` instead of Latin `a` |
| `\u202e<\|assistant\|>` | RTL override character prefix |
| `[INST]…[/INST]` | Llama-style injection |
| `###Assistant:` | Alpaca-style injection |

---

### Test 18 — Cross-Dataset Vocabulary Drift

**What it checks:** For each dataset, identifies tokens that appear **exclusively** in
that dataset and not in any other. Also computes a pairwise overlap matrix showing what
percentage of one dataset's token vocabulary is also seen in every other dataset.

**Why it matters:** Low vocabulary overlap between a pretraining dataset and an SFT
dataset means the model will encounter unfamiliar token patterns during fine-tuning.
High exclusive-token counts in a single SFT file may indicate that language is
under-represented in pretraining, leading to poor fine-tuning stability.

---

### Test 19 — Token Frequency Long-Tail Analysis

**What it checks:** Bins the entire vocabulary into 8 frequency buckets (never seen,
seen once, 2–4 times, 5–9 times, 10–99, 100–999, 1K–9K, 10K+). Reports the count of
tokens and percentage of total token occurrences in each bucket. Computes the Zipf ratio
(top-10 average frequency / bottom-10 non-zero average frequency).

**Why it matters:** This reveals the shape of your vocabulary utilization:
- A large "never seen" bucket means the vocabulary is larger than necessary for this data
- A large "seen once" or "2–4 times" bucket means many tokens are statistically marginal
  — embeddings for these tokens will be under-trained
- The Zipf ratio quantifies how top-heavy usage is: normal natural language has a high
  ratio, but extremely high values (>100K) with a large zero-frequency bucket suggest
  serious vocabulary coverage imbalance

---

### Test 20 — Chat Template Robustness

**What it checks:** Runs the SFT loss-masking simulation on 7 structured conversation
layouts: single-turn, two-turn, three-turn, empty assistant response, system + user +
assistant, consecutive assistant turns, and missing `<|end_turn|>`. For each, counts
how many separate assistant spans are correctly detected and how many tokens are unmasked.

**Why it matters:** Training on multi-turn conversations is common in SFT. If the masking
logic fails on any conversation layout, entire training samples contribute zero loss. The
test proactively catches edge cases before they silently degrade training quality.

---

### Test 21 — Mixed-Language Within Same Document

**What it checks:** Encodes 8 documents that combine multiple scripts in a single string:
Hindi + English code-switching, Telugu + English, code with inline Indic comments,
5-script greeting, and math notation mixed with Hindi text. Checks round-trip fidelity,
UNK count, and chars-per-token for the blended document.

**Why it matters:** Many real-world training documents are not monolingual. A South Asian
user asking a Python coding question may mix English terms with Hindi explanation. The
tokenizer must handle these documents efficiently without fragmentation or data loss.

---

### Test 22 — EOS / BOS Termination Behaviour

**What it checks:** Tests 8 edge cases around document boundary tokens:
- EOS or BOS token encoded alone
- Text wrapped between BOS and EOS
- Double EOS tokens
- EOS appearing mid-document
- PAD tokens embedded in sequences

**Why it matters:** `<|begin_of_text|>` and `<|end_of_text|>` are the primary signals
the model uses to recognize document boundaries during pre-training and to stop
generation during inference. If either token does not encode as a single ID, or does not
survive round-trip, the model's sense of document boundaries is broken — leading to
runaway generation (never stopping) or premature stopping.

---

## 6. Output Files — Column-by-Column Guide

### `tokenizer_audit_report.md`

The main human-readable report. Opens in any Markdown viewer (GitHub, VS Code, Notion).

### `tokenizer_audit_results.json`

All test results in structured JSON. Useful for:
- CI/CD integration (diff results across tokenizer versions)
- Programmatic comparison of two tokenizer candidates
- Feeding into dashboards or automated alerting

### `token_frequency.csv`

Combined frequency across **all text datasets** for every vocabulary entry.

| Column | Type | Description |
|--------|------|-------------|
| `token_id` | integer | Token's position in the vocabulary (0 to 131,071) |
| `token` | string | The string the token decodes to (may be a subword, punctuation, or raw byte) |
| `count` | integer | Total number of times this token appeared across all datasets combined. 0 = never seen in this run. |

Sorted by `count` descending — the most common tokens appear first.

**How to use:** Sort by `count` ascending to find your rarest tokens. Tokens with count=0
are entirely unused in the current data sample.

### `freq_<dataset>.csv`

One file per text dataset (e.g. `freq_golden_samples.csv`, `freq_raw_shard.csv`,
`freq_sft_group1_hindi.csv`). Same three columns as `token_frequency.csv` but showing
counts for that specific dataset only.

**How to use:** Load two freq files side-by-side to compare which scripts or words are
unique to one dataset vs. shared. Useful for diagnosing language-specific vocabulary gaps.

### `golden_sample_token_counts.csv`

Per-sample token counts for the 128 golden evaluation samples.

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Sample identifier (e.g. `golden_001`) |
| `tag` | string | Content category (e.g. `math_reasoning`, `indic_hindi`, `code_generation`) |
| `n_tokens` | integer | Number of tokens this sample produces when encoded |

**How to use:** Sort by `n_tokens` descending to find the longest golden samples.
Samples above your model's `max_sequence_length` will be truncated during evaluation,
making those test results unreliable.

---

## 7. Reading the Markdown Report

### Summary Scorecard

The first table in the report gives a pass/fail/warn for each of the 22 tests. Use this
as your **go/no-go checklist** before training:

- ✅ PASS — no issues found
- ⚠️ WARN — issues found but not necessarily blockers (e.g. high unused vocab %)
- ❌ FAIL — critical issue that **should be fixed before training**

### Dataset Inventory Table

Shows how many documents were found, how many were tokenized (vs. just metadata-scanned),
and the estimated total token count.

| Column | Description |
|--------|-------------|
| Dataset | Internal dataset key used throughout the report |
| Type | `jsonl`, `parquet_text`, `parquet_meta`, or `txt` |
| Total Docs | Total rows/lines in the source file |
| Tokenized | Rows actually encoded (may be less than total if `--shard-rows` was used) |
| Est. Tokens | For text datasets: actual token count. For metadata-only: estimate from `token_est` column. |

### Individual Dataset Reports

Each dataset gets its own section showing:
- Token length statistics (see Test 6 columns above)
- Language distribution (for datasets with a `language` column)
- Domain / source distribution (for manifest datasets)
- Ghost tags found (or confirmation that none were found)
- UNK token count

### Recommendations & Action Items

The final section translates test results into prioritized actions:
- 🔴 **Critical** — must fix before training (data contamination, round-trip failures)
- 🟡 **Warning** — should investigate (high byte-fallback, semantic duplicates)
- 🔵 **Informational** — best-practice notes (run full shard, verify model config)

---

## 8. Understanding Key Findings

### "Ghost tags found in golden_samples (128 occurrences)"

All 128 golden samples use the old plain-text `[USER]` / `[ASSISTANT]` bracket format
instead of the structured `<|user|>` / `<|assistant|>` tokens. This means:

1. **For evaluation:** These samples can still be used to test general comprehension,
   but the SFT loss masking test will correctly show 0 unmasked assistant tokens —
   because the masking logic expects the real `<|assistant|>` token ID.
2. **For SFT training:** These samples **cannot** be used directly. They must be
   reformatted to use `<|user|>…<|end_turn|><|assistant|>…<|end_turn|>` structure.

### "43.2% vocab unused"

With `--shard-rows 50000` (the default), only 50K of 630K rows are tokenized.
This under-samples rare tokens. **Always run `--full-shard` before drawing conclusions
about vocabulary coverage.** After a full run, if unused % is still >20%, the vocabulary
was likely trained on a broader data distribution than what is available here.

### "SFT Loss Masking: 0 unmasked tokens for FIM samples"

FIM (Fill-in-the-Middle) format uses `<|fim_prefix|>` / `<|fim_suffix|>` / `<|fim_middle|>`
tokens, not `<|assistant|>`. The current masking simulation only recognizes
`<|assistant|>` spans. FIM training requires a **separate masking implementation** that
treats `<|fim_middle|>` as the "response start" token.

### "Sequence length INSUFFICIENT_DATA at 131K / 256K"

The default 50K-row sample does not contain enough long documents to fill a 131K-token
context window from the test pool. Use `--full-shard` to pool 630K rows — this should
provide sufficient long-form text to reach both checkpoints.

---

## 9. Adding New Data Sources

### New SFT text file

1. Copy the `.txt` file into `sft_data/`
2. Re-run the script — it is discovered automatically

### New JSONL file (structured samples)

Add a new dataset block in `tokenizer_audit.py` following the `DS-A` (golden samples)
pattern. The block must:
1. Assign a unique dataset key (e.g. `"my_dataset"`)
2. Call `empty_ds(name, source_type)` to initialize the dataset dict
3. Iterate each line with `json.loads()`, extract the `text` field
4. Call `tokenizer.encode(text)` and update `ds["freq"]`, `ds["token_counts"]`
5. Add the key to `TEXT_DS_KEYS`

### New Parquet file with text

Follow the `DS-B` (raw_shard) pattern. Use batch iteration via
`pq.ParquetFile(...).iter_batches()` for memory efficiency with large files.

### New Parquet file — metadata only (no text column)

Follow the `DS-C` / `DS-D` (manifest) pattern. Only read the metadata columns
(`language`, `domain`, `source`, `band`, `word_count`, `token_est`). Use
`source_type="parquet_meta"` to signal that no tokenization is performed.

---

## 10. Frequently Asked Questions

**Q: How long does a full run take?**

With `--shard-rows 50000` (default): ~5–10 minutes.
With `--full-shard` (630K rows): ~25–40 minutes depending on CPU speed.
The ghost-tag sweep and special-token leakage scan always process the full corpus
regardless of `--shard-rows`.

**Q: Can I run this without `raw_shard.parquet`?**

Yes. Run with `--shard-rows 0`. All tests except the ones that specifically require
the shard (ghost-tag sweep uses it, but fails gracefully if missing) will still execute
against golden samples and SFT data.

**Q: What does a "round-trip failure" mean in practice?**

It means `tokenizer.decode(tokenizer.encode(text)) != text`. The most common cause
with BPE tokenizers is a leading space being prepended by `add_prefix_space=True` in
the post-processor. This makes the decoded string `" hello"` instead of `"hello"`.
For training data this is usually harmless, but for evaluation samples it causes
string comparison failures.

**Q: Why does Test 15 (reserved tokens) matter?**

If a reserved placeholder like `<|reserved_42|>` appears in training data, the model
learns it can occur in any context. When you later activate it (e.g. as `<|tool_result|>`),
the model generates it freely and incorrectly, requiring expensive re-training or
post-training alignment to suppress.

**Q: How do I fix ghost tags in training data?**

Run a preprocessing script on all affected files:

```python
import re

replacements = {
    r'\[USER\]':      '<|user|>',
    r'\[ASSISTANT\]': '<|assistant|>',
    r'\[INST\]':      '<|user|>',
    r'\[/INST\]':     '<|end_turn|>',
}

def clean_ghost_tags(text):
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text
```

**Q: What is a good chars-per-token value for Indic scripts?**

| Script | Target (chars/token) | Concern threshold |
|--------|---------------------|-------------------|
| Hindi (Devanagari) | ≥ 2.0 | < 1.5 |
| Telugu | ≥ 2.0 | < 1.5 |
| Tamil | ≥ 1.8 | < 1.4 |
| Punjabi (Gurmukhi) | ≥ 1.8 | < 1.4 |
| Assamese | ≥ 1.8 | < 1.4 |
| Bengali | ≥ 2.0 | < 1.5 |
| English | 3.5–5.0 | < 2.5 |
| Code | 2.0–4.0 | < 1.5 |

Values near 1.0 indicate byte-level fragmentation — the tokenizer is treating each
UTF-8 byte as a separate token rather than combining them into syllable-level pieces.

---

*Generated by tokenizer_audit.py — a Tokenizer Quality Audit tool.*
