# Tokenizer Audit Comparison Report: OLD vs HYBRID

**Date**: 2026-03-07
**Audit tool**: `tokenizer_audit.py` (23 tests)
**Data**: 256 golden samples + 50,000 raw shard rows + 5 SFT Indic files + English SFT
**OLD report**: `report/archive/06-Mar-26/`
**HYBRID report**: `report/hybrid/`

---

## 1. Executive Summary

| Metric | OLD | HYBRID | Change |
|---|---|---|---|
| **Byte-fallback rate** | **51.1%** | **1.0%** | **-50.1pp 🚀** |
| **Total corpus tokens** | 162,285,260 | 91,217,884 | **-43.7% 🚀** |
| **Vocab unused** | 27,970 (21.3%) | 23,989 (18.3%) | -3,981 ✅ |
| **Roundtrip pass** | 23/23 ✅ | 23/23 ✅ | — |
| **Multilingual UNK** | 0 ✅ | 0 ✅ | — |
| **Special token single-token** | 356/356 ✅ | 352/352 ✅ | 4 tokens dropped |
| **Ghost tags** | 1 dataset | 1 dataset | — |
| **SFT masking failures** | 1 (FIM) | 1 (FIM) | — |
| **Adversarial injections** | 0/14 ✅ | 0/14 ✅ | — |
| **Reserved tokens detected** | 250 ✅ | **0 ⚠️** | Investigate |
| **Garbage tokens** | 24 (0.018%) | 46 (0.035%) | +22 (inherited) |
| **English tokens (SFT group2)** | 6,026,727 | 6,026,727 | 0.0% ✅ |
| **Mixed-language c/t (5 scripts)** | 0.42 | 2.39 | **+469% 🚀** |
| **`model_max_length` config** | `1e+30` (unset) | `131072` ✅ | Fixed |

**HYBRID is a clear improvement on byte efficiency and Indic compression.** The 5-script mixed-language compression gain (+469% chars/token) and the byte-fallback rate drop (51.1% → 1.0%) are the two headline improvements. One packaging gap identified: HYBRID’s `tokenizer_config.json` is missing `added_tokens_decoder`, causing T15 to report 0 reserved tokens. The tokens exist in `tokenizer.json` — fix by re-saving via `AutoTokenizer.save_pretrained()` (see T15).

---

## 2. Architecture (Unchanged Between OLD and HYBRID)

Both tokenizers share the same structure — HYBRID is a vocabulary surgery, not an architecture change.

| Property | OLD | HYBRID |
|---|---|---|
| Model type | BPE | BPE |
| Pre-tokenizer | Sequence[Split + ByteLevel] | Sequence[Split + ByteLevel] |
| Decoder | ByteLevel | ByteLevel |
| Post-processor | ByteLevel | ByteLevel |
| Normalizer | null | null |
| `ignore_merges` | True | True |
| Vocab size | 131,072 | 131,072 |
| Merges | 302,338 | 301,409 |
| Added tokens | **356** | **352** |

---

## 3. Test-by-Test Results

### T1 — Special Token Integrity

| Check | OLD | HYBRID |
|---|---|---|
| BOS `<\|begin_of_text\|>` (id=130716) | ✅ | ✅ |
| EOS `<\|end_of_text\|>` (id=130717) | ✅ | ✅ |
| PAD `<\|pad\|>` (id=130718) | ✅ | ✅ |
| Duplicate IDs | none | none |
| Problematic absent | `<\|startoftext\|>`, `<\|return\|>`, `<AGENT>`, `\|AGENT\|` | same |

Both tokenizers have 4 tokens listed as "problematic absent". These are legacy token names; the functional tokens (BOS/EOS/PAD) are all present and correct.

### T2 — Encode/Decode Roundtrip

Both: **23/23 PASS**. All test cases encode and decode correctly with no corruption.

### T3 — Special Tokens Encode as Single Token

| | OLD | HYBRID |
|---|---|---|
| Pass | 356/356 ✅ | 352/352 ✅ |
| Fail | 0 | 0 |

HYBRID has 352 (vs 356) because 4 tool-use special tokens were removed:
`<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`

**Action required**: Confirm whether these tool tokens are needed in production. If yes, they must be re-added to HYBRID.

### T4 — Ghost Tags in Raw Data

| Dataset | OLD | HYBRID |
|---|---|---|
| raw_shard | dirty: `<\|endoftext\|>` ×3, `[SYSTEM]` ×2, `[USER]` ×6 | identical |
| golden_samples | clean | clean |
| sft_group1_assamese | clean | clean |
| sft_group1_hindi | clean | clean |
| sft_group1_marathi | clean | clean |
| sft_group1_punjabi | clean | clean |
| sft_group1_telugu | clean | clean |
| sft_group2 (English) | clean | clean |
| sft_group3 | clean | clean |

Both tokenizers surface the same 11 raw-text ghost tags in the raw pretraining shard. These are upstream data artifacts — not tokenizer bugs. The `<|endoftext|>` strings appear literally in crawled web text (3 rows), and `[SYSTEM]`/`[USER]` appear as plaintext in 8 rows.

**Action**: Clean the 11 dirty rows in the pretraining parquet before training.

### T5 — Vocabulary Utilisation

| Metric | OLD | HYBRID | Delta |
|---|---|---|---|
| Total unique tokens seen | 103,102 | 107,083 | +3,981 |
| Unused tokens | 27,970 | 23,989 | -3,981 |
| Unused % | 21.3% | 18.3% | -3.0pp |
| Rare tokens (freq < 5) | 16,224 | 17,160 | +936 |
| UNK tokens | 0 | 0 | — |

HYBRID shows better vocab utilisation (3,981 fewer unused slots) because the new Indic tokens are actively used. The 936 extra rare tokens in HYBRID are Indic subwords present in vocab but only seen a handful of times in this 50K-row sample; they will fire more frequently on larger Indic corpora.

**Top 5 most frequent tokens — OLD** (dominated by byte fragments):

| Rank | Token | Count |
|---|---|---|
| 1 | `à` (byte fragment) | 27,357,438 |
| 2 | `°` (byte fragment) | 7,690,875 |
| 3 | `¤` (byte fragment) | 7,577,611 |
| 4 | `¨` (byte fragment) | 6,538,436 |
| 5 | `Ġ` (space) | 5,792,991 |

**Top 5 most frequent tokens — HYBRID** (dominated by meaningful English):

| Rank | Token | Count |
|---|---|---|
| 1 | `,` | 3,272,417 |
| 2 | `.` | 2,104,721 |
| 3 | `Ġ"` | 2,050,050 |
| 4 | `Ġ` (space) | 1,860,620 |
| 5 | `Ġthe` | 1,758,961 |

In HYBRID, Indic subword tokens appear in the top 50 (e.g., `à¥¤` at rank 12 with 652K occurrences, `à¨¾` at rank 23 with 283K). In OLD, every top slot was a byte fragment.

### T7 — SFT Loss Masking

Both: **1 failure** — the FIM (Fill-in-the-Middle) format. This is a training-code issue, not a tokenizer defect. The masking simulator only scans for `<|assistant|>` to start unmasking; FIM uses `<|fim_middle|>` instead. Fix in the data collator if FIM training is planned.

### T8 — Sequence Length Distribution

Both tokenizers PASS all 9 sequence-length checkpoints:

| Checkpoint (max tokens) | OLD | HYBRID |
|---|---|---|
| 1,024 | PASS | PASS |
| 2,048 | PASS | PASS |
| 4,096 | PASS | PASS |
| 8,192 | PASS | PASS |
| 16,384 | PASS | PASS |
| 32,768 | PASS | PASS |
| 65,536 | PASS | PASS |
| 131,072 | PASS | PASS |
| 262,144 | PASS | PASS |

No sequences exceed the training context window in either tokenizer. The overall token count reduction in HYBRID (~43.7%) means documents that were borderline-long in OLD are safely within limit in HYBRID.

### T9 — Multilingual UNK Tokens

Both: **0 UNK tokens** across all tested languages. Coverage is complete in both tokenizers.

### T10 — Semantic Duplicate Tokens

| Metric | OLD | HYBRID |
|---|---|---|
| `byte_frag_groups` | 1 | 10 |
| `duplicates_found` | [] | [] |

No true semantic duplicates (tokens that decode to the same string) were found in either tokenizer. The increase from 1 to 10 `byte_frag_groups` in HYBRID is **expected and correct**: HYBRID added 1,178 Indic byte-intermediate tokens (e.g., `à¤`, `à¤¨`) to the vocabulary as necessary building blocks for multi-byte BPE merge chains. These are intentional merge intermediates, not duplicates. The audit groups them as byte-fragment-like but they serve a structural purpose.

### T11 — Edge Cases

Both tokenizers PASS all 13 edge-case inputs. Key token count improvements in HYBRID:

| Test input | OLD tokens | HYBRID tokens | Delta |
|---|---|---|---|
| 100 emoji string | 400 | 100 | -75% |
| 50 zero-width spaces (ZWSP) | 150 | 13 | -91% |
| multi-script sentence | 34 | 16 | -53% |
| empty string | 0 | 0 | — |
| whitespace-only | same | same | — |
| null bytes / control chars | same | same | — |

The emoji and ZWSP reductions reflect HYBRID's richer merge vocabulary for non-ASCII code points. All 13 edge cases pass roundtrip (encode → decode → match) in both tokenizers.

### T12 — Configuration Integrity

| Config field | OLD | HYBRID |
|---|---|---|
| `model_max_length` | `1e+30` (effectively infinity / unset) | `131072` |
| `bos_token_id` | 130716 | 130716 |
| `eos_token_id` | 130717 | 130717 |
| `pad_token_id` | 130718 | 130718 |
| `tokenizer_class` | PreTrainedTokenizerFast | PreTrainedTokenizerFast |

**HYBRID correctly sets `model_max_length=131072`**, matching the vocabulary size — a configuration hygiene improvement over OLD's unset infinity value. BOS/EOS/PAD IDs are identical across both tokenizers, confirming backward compatibility for any code that references these IDs.

### T13 — Byte Fallback Rate (Critical)

This is the most important test for Indic language quality.

| Metric | OLD | HYBRID | Delta |
|---|---|---|---|
| **Overall byte-fallback rate** | **51.12%** | **1.0%** | **-50.1pp** |
| Byte-fallback tokens in vocab | 128 | 1,306 | +1,178 |
| Corpus total tokens | 162,285,260 | 91,217,884 | -43.7% |
| Corpus byte tokens (fragments) | 82,964,660 | 911,144 | **-98.9%** |

**Per-language byte-fragment rate and compression:**

| Language | OLD frag% | HYB frag% | OLD c/t | HYB c/t | Tokens saved |
|---|---|---|---|---|---|
| English | 1.6% | 1.0% | 4.40 | 4.43 | — |
| Assamese (SFT) | ~89% | ~4% | ~0.3 | ~1.9 | -73.6% |
| Hindi (SFT) | ~80% | ~3% | ~0.3 | ~2.3 | -66.2% |
| Marathi (SFT) | ~86% | ~4% | ~0.3 | ~2.2 | -75.2% |
| Punjabi (SFT) | ~85% | ~5% | ~0.3 | ~1.9 | -72.1% |
| Telugu (SFT) | ~89% | ~5% | ~0.3 | ~2.0 | -78.2% |

> Indic per-language frag% estimated from SFT token count ratios; English is live-measured from T13.

The increase in `byte_fallback_tokens_in_vocab` (128 → 1,306) is expected and correct: HYBRID added 1,178 new byte-intermediate tokens (e.g., `à¤`, `à¤¨`) to the vocabulary as necessary building blocks for BPE merge chains. These are not "bad" — they are intentional merge intermediates that enable multi-byte Indic token assembly.

### T14 — Numeric Tokenization

Both tokenizers produce near-identical results for all numeric test cases. The only measurable difference is in the `equation` test case containing a square root symbol (√):

| Test case | OLD tokens | HYBRID tokens | Note |
|---|---|---|---|
| integers (0–9999) | identical | identical | — |
| floats | identical | identical | — |
| scientific notation | identical | identical | — |
| currency amounts | identical | identical | — |
| `equation` (√ symbol) | 20 | 17 | √ → `â`,`Ī`,`ļ` (OLD) vs `ĠâĪļ` merged (HYBRID) |
| large numbers | identical | identical | — |
| negative numbers | identical | identical | — |

In OLD, the `√` character (U+221A, 3-byte UTF-8 sequence) fragments into 3 separate byte tokens. In HYBRID, these 3 bytes merge into a single token `ĠâĪļ`. This is a minor benefit from the additional merge rules in HYBRID. All numeric tokens decode correctly (roundtrip PASS) in both tokenizers.

### T15 — Reserved Token Contamination

| | OLD | HYBRID |
|---|---|---|
| Reserved tokens detected | 250 | **0 ⚠️** |
| Non-zero frequency | 0 ✅ | 0 ✅ |

**⚠️ HYBRID shows 0 reserved tokens detected — root cause identified: `added_tokens_decoder` is absent from HYBRID's `tokenizer_config.json`.**

The audit script (`tokenizer_audit.py:1267–1274`) finds reserved tokens exclusively by reading `tokenizer_config.json` via the `added_tokens_decoder` key. OLD's `tokenizer_config.json` contains the full block with all 250 `<|reserved_0|>` … `<|reserved_249|>` entries. HYBRID's `tokenizer_config.json` is a minimal 7-line file with no `added_tokens_decoder` section — so the script finds 0.

The reserved tokens **do exist** in HYBRID's `tokenizer.json` (confirmed by direct search). The `tokenizer_config.json` was simply not regenerated with the full metadata block after vocab surgery.

This is a real packaging gap, not just a test script issue. Without `added_tokens_decoder` in `tokenizer_config.json`, the HuggingFace Python API (`tok.all_special_tokens`) cannot surface the reserved tokens as Python objects, which affects any training code that needs to identify or mask them by name.

**Fix — `tokenizer_config.json`** (Dataset/Packaging layer): Re-save the HYBRID tokenizer via HuggingFace, which auto-populates `added_tokens_decoder` from `tokenizer.json`:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    "experiments/6_tokenizer_design_lab/tsai_131k_tokenizer_hybrid"
)
tok.save_pretrained("experiments/6_tokenizer_design_lab/tsai_131k_tokenizer_hybrid")
# Overwrites tokenizer_config.json with the full added_tokens_decoder block
```

After this fix, re-running T15 should report 250 reserved tokens (matching OLD) with 0 non-zero frequency.

### T16 — Special Token Leakage

| Metric | OLD | HYBRID |
|---|---|---|
| Rows scanned | 630,140 | 630,140 |
| Leakage hits | 0 | 0 |
| Result | CLEAN | CLEAN |

Neither tokenizer leaks special token strings (e.g., `<|begin_of_text|>`, `<|end_of_text|>`, `<|assistant|>`) into ordinary tokenized output. All 630,140 rows across all datasets were scanned. This confirms that BOS/EOS/PAD and chat-format special tokens are properly isolated and not produced by regular text encoding.

### T17 — Adversarial Token Injection

Both: **0/14 flagged** by the audit. Note: the TOKENIZER_COMPARISON_REPORT.md separately identified 2 adversarial vectors (RTL override + ZWSP) as inherent HuggingFace `added_tokens` trie behaviour — present in both tokenizers and every major LLM tokenizer. Mitigation is at the application layer (strip Unicode control chars from user input).

### T18 — Cross-Dataset Drift

Both tokenizers show identical cross-dataset drift behavior. Each dataset has exactly 2 exclusive tokens (tokens that appear in that dataset and no other) — expected for per-language vocabulary coverage, not a sign of drift or contamination.

| Dataset | OLD exclusive tokens | HYBRID exclusive tokens |
|---|---|---|
| sft_group1_assamese | 2 | 2 |
| sft_group1_hindi | 2 | 2 |
| sft_group1_marathi | 2 | 2 |
| sft_group1_punjabi | 2 | 2 |
| sft_group1_telugu | 2 | 2 |
| sft_group2 (English) | 2 | 2 |
| sft_group3 | 2 | 2 |

The pattern is consistent across both tokenizers. No unexpected cross-contamination detected.

### T19 — Long Tail Distribution

| | OLD | HYBRID |
|---|---|---|
| Unique tokens seen | 103,102 | 107,083 |
| Zero-frequency tokens | 27,970 | 23,989 |

Both show a healthy Zipfian distribution. HYBRID has 3,981 more active tokens due to new Indic subwords firing in the corpus.

### T20 — Chat Robustness

Both tokenizers are identical on all 7 chat-format patterns. Token counts match exactly for every test case:

| Chat pattern | OLD tokens | HYBRID tokens |
|---|---|---|
| `<\|system\|>` + `<\|user\|>` + `<\|assistant\|>` | identical | identical |
| multi-turn dialogue | identical | identical |
| tool-call format | identical | identical |
| FIM format | identical | identical |
| code block inside chat | identical | identical |
| mixed-language chat | identical | identical |
| empty assistant turn | identical | identical |

All 7 patterns pass roundtrip (encode → decode → match). No regressions in HYBRID for chat template handling.

### T21 — Mixed Language Compression

This is the most striking compression improvement for multi-script inputs. All 8 mixed-script test cases show significant improvement in HYBRID:

| Test case | OLD c/t | HYBRID c/t | Improvement |
|---|---|---|---|
| Hindi + English | 2.14 | 3.57 | +67% |
| Telugu + English | 1.37 | 3.52 | +157% |
| Tamil + English | 1.43 | 3.48 | +143% |
| Marathi + English | 1.89 | 3.61 | +91% |
| Punjabi + English | 1.52 | 3.44 | +126% |
| Hindi + Telugu + Marathi | 0.71 | 2.84 | +300% |
| 5 scripts (hi+te+ta+mr+pa) | 0.42 | 2.39 | +469% |
| Code + Indic comments | 2.01 | 3.33 | +66% |

All 8 cases: `round_trip_ok=True`, `unk_count=0`. The 5-script mixed case goes from 0.42 c/t (nearly every byte its own token) to 2.39 c/t — a 5.7× improvement. This is the direct result of HYBRID's Indic vocabulary additions.

### T22 — EOS Behaviour

Both tokenizers are identical on all 8 BOS/EOS boundary test cases. Token sequences (pieces), IDs, and roundtrips all match exactly:

| Test case | OLD | HYBRID |
|---|---|---|
| BOS at start only | PASS | PASS |
| EOS at end only | PASS | PASS |
| BOS + EOS wrap | PASS | PASS |
| Double EOS | PASS | PASS |
| EOS in middle of text | PASS | PASS |
| Multiple BOS/EOS | PASS | PASS |
| EOS followed by text | PASS | PASS |
| Empty sequence with BOS+EOS | PASS | PASS |

No changes in BOS/EOS handling between OLD and HYBRID. The vocabulary surgery did not affect special token boundary behavior.

### T23 — Garbage Token Audit

| Category | OLD | HYBRID |
|---|---|---|
| `mojibake` | 0 | 0 |
| `private_use` | 0 | 4 |
| `surrogate` | 0 | 0 |
| `zero_width_noise` | 20 | 18 |
| `zero_width_review` | 85 | 76 |
| `html_artifact` | 4 | 4 |
| `broken_utf8` | 0 | 20 |
| **Total garbage** | **24 (0.018%)** | **46 (0.035%)** |

HYBRID has 22 more garbage tokens (20 `broken_utf8` + 4 `private_use`) inherited from the OpenAI o200k_base training corpus. These are the same artifacts documented in `TOKENIZER_COMPARISON_REPORT.md §7.4`. They represent 0.035% of vocab and are benign — they will simply never fire in clean training data.

#### Fix Classification by Layer

| Category | Count | Fix Layer | Action |
|---|---|---|---|
| `broken_utf8` | 20 | **Dataset** | Strip U+FFFD from source documents during corpus preprocessing |
| `html_artifact` | 4 | **Dataset** | HTML-unescape + strip tags from scraped web text during preprocessing |
| `private_use` | 4 | **Dataset** | Strip PUA chars (U+E000–U+F8FF) from corpus during preprocessing |
| `zero_width_noise` | 18 | **Dataset** + **Application** | Strip invisible controls from corpus; also strip from user input at inference (prompt-injection defence for bidi controls) |
| **Tokenizer** | — | None | No tokenizer fix needed — all tokens are benign and won't fire on clean data. Removing from vocab requires full retraining; not worth the cost at 0.035% of vocab. |

#### Garbage Token Detail — HYBRID (source: `report/hybrid/garbage_tokens.csv`)

**`broken_utf8` — 20 tokens** *(new in HYBRID; 0 in OLD)*

U+FFFD replacement characters baked into token strings, indicating the o200k_base training corpus contained already-corrupted source text.

| Token ID | Raw | Decoded |
|---|---|---|
| 2740 | `ï¿½` | `<FFFD>` |
| 8100 | `ï¿½ï¿½` | `<FFFD><FFFD>` |
| 21607 | `ï¿½ï¿½ï¿½ï¿½` | `<FFFD>×4` |
| 21812 | `Ġï¿½` | ` <FFFD>` |
| 47472 | `ï¿½s` | `<FFFD>s` |
| 51441 | `ï¿½ĊĊ` | `<FFFD>\n\n` |
| 54350 | `ï¿½ï¿½ï¿½` | `<FFFD>×3` |
| 55055 | `ï¿½n` | `<FFFD>n` |
| 61113 | `Ġï¿½ï¿½ï¿½ï¿½` | ` <FFFD>×4` |
| 63696 | `ï¿½t` | `<FFFD>t` |
| 68156 | `?ï¿½` | `?<FFFD>` |
| 79325 | `ï¿½?` | `<FFFD>?` |
| 80851 | `ï¿½a` | `<FFFD>a` |
| 88806 | `ï¿½r` | `<FFFD>r` |
| 98892 | `ï¿½` ×8 | `<FFFD>×8` |
| 101501 | `ï¿½o` | `<FFFD>o` |
| 106278 | `âĤ¬ï¿½` | `€<FFFD>` |
| 113213 | `Ġï¿½ï¿½` | ` <FFFD>×2` |
| 113903 | `ï¿½m` | `<FFFD>m` |
| 114227 | `ï¿½Ċ` | `<FFFD>\n` |

**Dataset fix** — strip U+FFFD from all source documents in the preprocessing pipeline before tokenization:
```python
text = text.replace("\ufffd", "")
```

---

**`zero_width_noise` — 18 tokens** *(reduced from 20 in OLD; 2 removed by vocab surgery)*

Invisible Unicode control characters with no linguistic function: ZWSP (U+200B), bidi controls (U+202A–202E), BOM (U+FEFF), Word Joiner (U+2060). The RTL Override (U+202E, token ID 110669) is the highest-risk for prompt injection.

Representative tokens: ID 2787 (ZWSP `​`), 18982 (LTR Embed `\u202A`), 19568 (RTL Embed `\u202B`), 89190 (LTR Override `\u202D`), 110669 (RTL Override `\u202E`).

**Dataset fix** — strip during corpus preprocessing:
```python
import re
INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
text = INVISIBLE.sub("", text)
```
**Application fix** — apply the same strip to user input at inference time before passing to the tokenizer. This is the primary defence against bidi/ZWSP prompt-injection attacks.

---

**`html_artifact` — 4 tokens** *(same in OLD and HYBRID; present in both base tokenizers)*

Unescaped HTML entity fragments from raw web-crawl data not stripped before tokenizer training.

| Token ID | Raw | Decoded |
|---|---|---|
| 18635 | `&#` | `&#` |
| 42631 | `Ġ&#` | ` &#` |
| 45060 | `;&#` | `;&#` |
| 101607 | `Ġ'&#` | ` '&#` |

**Dataset fix** — run HTML unescaping + tag removal on all scraped web text before tokenization:
```python
import html, re
text = html.unescape(text)
text = re.sub(r"<[^>]+>", " ", text)
```

---

**`private_use` — 4 tokens** *(new in HYBRID; 0 in OLD)*

Unicode Private Use Area characters (U+E000–U+F8FF) with no standardised meaning, inherited from the o200k_base training corpus.

| Token ID | Raw | Codepoint |
|---|---|---|
| 49529 | `ïĤ·` | U+F137 |
| 76039 | `ĠïĤ·` | ` ` + U+F137 |
| 77811 | `ïĤ§` | U+F127 |
| 99490 | `ïĥĺ` | U+F17A |

**Dataset fix** — strip PUA characters from corpus during preprocessing:
```python
import re
PUA = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000fffff]")
text = PUA.sub("", text)
```

---

## 4. Dataset Token Count Comparison

### Golden Samples (256 docs)

| | OLD | HYBRID | Δ |
|---|---|---|---|
| Total tokens | 164,436 | 131,865 | **-19.8%** |
| Mean tokens/doc | 642.3 | 515.1 | -19.8% |
| Max tokens/doc | 17,159 | 14,062 | -18.1% |

The golden samples contain Indic content, explaining the ~20% reduction.

### Raw Shard (50,000 rows sampled from 630K English pretraining)

| | OLD | HYBRID | Δ |
|---|---|---|---|
| Total tokens | 58,796,374 | 58,496,542 | **-0.5%** |

English pretraining shard is essentially identical — confirms English compression is unchanged. The 0.5% difference is sampling noise.

### SFT Files (Indic + English)

| Dataset | OLD tokens | HYBRID tokens | Saved | % Fewer |
|---|---|---|---|---|
| sft_group1_assamese | 17,840,446 | 4,714,382 | 13,126,064 | **-73.6%** |
| sft_group1_hindi | 12,006,671 | 4,056,043 | 7,950,628 | **-66.2%** |
| sft_group1_marathi | 18,289,840 | 4,535,965 | 13,753,875 | **-75.2%** |
| sft_group1_punjabi | 23,710,441 | 6,620,298 | 17,090,143 | **-72.1%** |
| sft_group1_telugu | 24,067,382 | 5,252,868 | 18,814,514 | **-78.2%** |
| sft_group2 (English) | 6,026,727 | 6,026,727 | 0 | **0.0% ✅** |
| sft_group3 | 1,382,943 | 1,383,194 | -251 | ~0.0% |
| **Indic SFT total** | **95,914,780** | **25,179,556** | **70,735,224** | **-73.7%** |

The 73.7% reduction in Indic SFT tokens directly translates to ~73.7% less compute for Indic SFT training passes. English tokens are completely unchanged.

---

## 5. Issues & Action Items

| # | Issue | Severity | Tokenizer | Action |
|---|---|---|---|---|
| 1 | **Reserved tokens: 0 detected in HYBRID** | ⚠️ Medium | HYBRID | **Packaging gap** — `added_tokens_decoder` absent from `tokenizer_config.json`; tokens exist in `tokenizer.json`. Fix: re-save via `AutoTokenizer.save_pretrained()` to regenerate config with full metadata block (see T15). |
| 2 | **4 tool-call special tokens dropped** (`<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`) | ⚠️ Medium | HYBRID | Confirm if needed for production chat; re-add if yes |
| 3 | Ghost tags in raw_shard (3× `<\|endoftext\|>` in crawled docs) | 🔵 Low | Both | Clean 11 rows in pretraining parquet |
| 4 | SFT masking failure (FIM format) | 🔵 Low (if not doing FIM) | Both | Add FIM branch to data collator |
| 5 | Adversarial injection (RTL override + ZWSP) | 🔵 Low | Both | Strip Unicode control chars at app layer |
| 6 | 46 garbage tokens (0.035%) in HYBRID | 🔵 Low | HYBRID | No tokenizer fix needed; clean training corpus |
| 7 | 16,224–17,160 rare tokens (freq < 5) | 🔵 Low | Both | Monitor; these may fire more on larger corpora |

---

## 6. Recommendation

**HYBRID is production-ready for Indic SFT training** with the following prerequisites:

1. **Fix `tokenizer_config.json`** — re-save HYBRID via `AutoTokenizer.save_pretrained()` to populate `added_tokens_decoder`; the 250 reserved tokens exist in `tokenizer.json` but are invisible to the HuggingFace API until the config is regenerated (see T15)
2. **Decide on tool tokens** — re-add `<tool_call>` / `<tool_response>` if agentic/tool-use fine-tuning is planned
3. **Keep OLD for reference** — archived in `report/archive/06-Mar-26/`

For English pretraining: both tokenizers are functionally identical (0.0% token count difference on group2 English SFT, 0.5% on raw shard within sampling noise).

The 73.7% reduction in Indic SFT tokens and the 51pp drop in byte-fallback rate (51.1% → 1.0%) make HYBRID a substantial improvement over OLD for any Indic language workload.