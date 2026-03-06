# Top 16K Token Statistics Report

**Tokenizer:** `tsai_131k_tokenizer` (BPE, 131,072 vocab)  
**Corpus:** ~1.03 billion tokens across pretraining shard (630K docs) + SFT data  
**Generated:** 2026-03-06

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total vocabulary | 131,072 |
| Tokens with ≥1 occurrence | 121,046 (92.4%) |
| Zero-frequency tokens | 10,026 (7.6%) |
| **Top 16K coverage** | **91.21%** of corpus |
| Total corpus tokens | 1,033,175,543 |
| Top 16K token count | 942,394,588 |
| Min frequency in top 16K | 4,961 |

> [!IMPORTANT]
> **16K tokens capture 91.21% of all token usage** across the entire corpus. The remaining ~115K vocabulary entries account for only 8.79% of usage, serving the long tail (rare words, multilingual coverage, code tokens).

---

## Cumulative Coverage by Vocab Size

Shows how many tokens are needed to cover a given percentage of the corpus:

| Top-K Tokens | Corpus Coverage | Incremental Gain |
|:------------:|:---------------:|:-----------------:|
| 100 | 41.32% | — |
| 235 | 50.00% | +8.68% from 135 more tokens |
| 500 | 57.71% | +7.71% from 265 more tokens |
| 1,000 | 64.78% | +7.07% from 500 more tokens |
| 2,000 | 71.84% | +7.06% from 1,000 more tokens |
| 2,722 | 75.00% | +3.16% from 722 more tokens |
| 4,000 | 78.86% | +3.86% from 1,278 more tokens |
| 4,497 | 80.00% | +1.14% from 497 more tokens |
| 7,637 | 85.00% | +5.00% from 3,140 more tokens |
| 8,000 | 85.42% | +0.42% from 363 more tokens |
| 13,699 | 90.00% | +4.58% from 5,699 more tokens |
| **16,000** | **91.21%** | **+1.21% from 2,301 more tokens** |
| 27,282 | 95.00% | +3.79% from 11,282 more tokens |
| 37,881 | 97.00% | +2.00% from 10,599 more tokens |
| 57,612 | 99.00% | +2.00% from 19,731 more tokens |
| 67,625 | 99.50% | +0.50% from 10,013 more tokens |
| 85,894 | 99.90% | +0.40% from 18,269 more tokens |
| 121,046 | 100.00% | +0.10% from 35,152 more tokens |

---

## Token Category Breakdown (Top 16K)

| Category | Count | % of 16K | Description |
|----------|------:|:--------:|-------------|
| English words | 9,464 | 59.1% | Complete words with word boundary (e.g., ` the`, ` from`, ` company`) |
| Subwords | 3,684 | 23.0% | BPE fragments without boundary (e.g., `ing`, `tion`, `er`) |
| Other | 1,737 | 10.9% | Mixed tokens: word+punctuation, abbreviations, multi-token patterns |
| Numbers | 334 | 2.1% | Digit sequences (e.g., `0`–`9`, `10`, `100`, `2024`) |
| Punctuation | 202 | 1.3% | Single punctuation marks (e.g., `,`, `.`, `?`, `!`) |
| Code tokens | 159 | 1.0% | Programming constructs (e.g., `return`, `def `, `()`, `->`) |
| **Indic tokens** | **136** | **0.8%** | Devanagari, Bengali, Gurmukhi, Telugu, Tamil, etc. |
| Single chars | 119 | 0.7% | Individual letters (`a`–`z`, `A`–`Z`) |
| Byte fragments | 92 | 0.6% | Incomplete UTF-8 byte tokens (decode to `�`) |
| Whitespace/indent | 73 | 0.5% | Spaces, tabs, indentation sequences |

---

## Frequency Distribution (Top 16K)

| Frequency Range | Token Count | % of 16K |
|:---------------:|:-----------:|:--------:|
| 10M – 100M | 9 | 0.06% |
| 1M – 10M | 93 | 0.58% |
| 100K – 1M | 931 | 5.82% |
| 10K – 100K | 8,048 | 50.30% |
| 1K – 10K | 6,919 | 43.24% |

> [!NOTE]
> The bulk of the top 16K sit in the 10K–100K range (50.3%) and 1K–10K range (43.2%). Only 9 tokens exceed 10M occurrences — these are the fundamental building blocks (`,`, `.`, ` the`, ` `, ` of`, etc.).

---

## Top 50 Most Frequent Tokens

| Rank | ID | Token | Count | Cum. % |
|-----:|---:|-------|------:|:------:|
| 1 | 156 | `à` (byte fragment) | 35,194,438 | 3.41% |
| 2 | 11 | `,` | 29,419,874 | 6.25% |
| 3 | 289 | ` the` | 26,919,944 | 8.86% |
| 4 | 220 | ` ` (space) | 21,179,176 | 10.91% |
| 5 | 13 | `.` | 20,599,061 | 12.90% |
| 6 | 320 | ` of` | 14,058,049 | 14.26% |
| 7 | 312 | ` to` | 13,722,433 | 15.59% |
| 8 | 319 | ` and` | 13,514,188 | 16.90% |
| 9 | 261 | ` a` | 11,410,725 | 18.00% |
| 10 | 302 | ` in` | 9,316,682 | 18.91% |
| 11 | 97 | `¤` (byte fragment) | 8,789,658 | 19.76% |
| 12 | 108 | `°` (byte fragment) | 8,407,413 | 20.57% |
| 13 | 101 | `¨` (byte fragment) | 7,611,169 | 21.31% |
| 14 | 363 | ` is` | 7,100,441 | 21.99% |
| 15 | 508 | `.\n` | 6,628,295 | 22.64% |
| 16 | 99 | `¦` (byte fragment) | 6,126,368 | 23.23% |
| 17 | 373 | ` for` | 5,314,116 | 23.74% |
| 18 | 198 | `\n` | 5,167,699 | 24.24% |
| 19 | 447 | ` that` | 5,093,617 | 24.74% |
| 20 | 25 | `:` | 4,391,843 | 25.16% |
| 21 | 16 | `1` | 4,289,873 | 25.58% |
| 22 | 337 | ` (` | 4,012,713 | 25.97% |
| 23 | 446 | ` with` | 3,928,762 | 26.35% |
| 24 | 370 | ` "` | 3,854,616 | 26.72% |
| 25 | 379 | ` on` | 3,784,370 | 27.08% |
| 26 | 15 | `0` | 3,574,903 | 27.43% |
| 27 | 17 | `2` | 3,493,529 | 27.77% |
| 28 | 271 | `   ` (3-space indent) | 3,491,737 | 28.11% |
| 29 | 310 | ` =` | 3,419,501 | 28.44% |
| 30 | 122 | `¾` (byte fragment) | 3,339,100 | 28.76% |
| 31 | 344 | ` I` | 3,312,806 | 29.08% |
| 32 | 443 | ` it` | 3,231,320 | 29.39% |
| 33 | 444 | ` you` | 3,126,997 | 29.70% |
| 34 | 437 | ` as` | 3,080,338 | 30.00% |
| 35 | 59 | `\` | 3,071,739 | 30.29% |
| 36 | 8 | `)` | 2,978,742 | 30.58% |
| 37 | 390 | ` be` | 2,869,254 | 30.86% |
| 38 | 503 | ` are` | 2,868,739 | 31.14% |
| 39 | 90 | `{` | 2,780,016 | 31.41% |
| 40 | 595 | ` was` | 2,643,663 | 31.66% |
| 41 | 12 | `-` | 2,626,338 | 31.92% |
| 42 | 305 | `       ` (7-space indent) | 2,600,513 | 32.17% |
| 43 | 1 | `"` | 2,551,529 | 32.41% |
| 44 | 235 | `į` (byte fragment) | 2,521,573 | 32.66% |
| 45 | 457 | ` this` | 2,454,205 | 32.90% |
| 46 | 562 | ` The` | 2,308,822 | 33.12% |
| 47 | 585 | ` by` | 2,305,833 | 33.34% |
| 48 | 494 | ` at` | 2,295,375 | 33.56% |
| 49 | 100 | `§` (byte fragment) | 2,287,188 | 33.79% |
| 50 | 600 | ` have` | 2,285,980 | 34.01% |

> [!WARNING]
> **Byte fragments (`à`, `¤`, `°`, `¨`, `¦`, `¾`, etc.) appear in the top 50.** These are raw UTF-8 byte tokens from Indic scripts that lack dedicated BPE merges. Rank #1 (`à`, 35M occurrences) is the leading byte of Devanagari/Bengali/etc. characters — each Indic character splits into 2–3 of these byte tokens.

---

## Boundary of Top 16K (Ranks 15,991–16,000)

| Rank | ID | Token | Count |
|-----:|---:|-------|------:|
| 15,991 | 13,521 | `(float` | 4,964 |
| 15,992 | 73,537 | ` imaginary` | 4,964 |
| 15,993 | 48,380 | ` insects` | 4,963 |
| 15,994 | 128,626 | ` কৰিলে` (Assamese) | 4,963 |
| 15,995 | 29,729 | ` applicants` | 4,962 |
| 15,996 | 19,753 | `315` | 4,962 |
| 15,997 | 10,642 | `Driver` | 4,962 |
| 15,998 | 48,495 | ` inherent` | 4,961 |
| 15,999 | 25,950 | ` forums` | 4,961 |
| 16,000 | 42,184 | ` rigid` | 4,961 |

> The cutoff frequency is **~4,961 occurrences**. Rank 16,001 (` rigid` → 4,960) is just 1 count below.

---

## Indic Script Tokens in Top 16K

**136 Indic tokens** made it into the top 16K. These are primarily high-frequency function words and common verbs:

| Rank | Token | Script | Count |
|-----:|-------|--------|------:|
| 162 | `।` (purna viram) | Devanagari | 659,214 |
| 569 | ` ਹੈ` (is) | Gurmukhi | 187,885 |
| 701 | ` ਜਾਂ` (or) | Gurmukhi | 151,890 |
| 725 | ` है` (is) | Devanagari | 147,356 |
| 868 | ` आहे` (is) | Devanagari (Marathi) | 120,486 |
| 879 | ` शब्द` (word) | Devanagari | 119,290 |
| 1,158 | ` ਵਿੱਚ` (in) | Gurmukhi | 90,000 |
| 1,357 | ` में` (in) | Devanagari (Hindi) | 77,350 |
| 1,367 | ` या` (or) | Devanagari | 76,792 |
| 1,580 | ` কি` (what) | Bengali | 66,611 |
| 1,688 | ` कौन` (who) | Devanagari (Hindi) | 62,590 |
| 1,840 | ` किंवा` (or) | Devanagari (Marathi) | 57,800 |
| 1,872 | ` से` (from) | Devanagari (Hindi) | 56,728 |
| 2,007 | ` क्या` (what) | Devanagari (Hindi) | 52,660 |
| 2,045 | ` का` (of) | Devanagari (Hindi) | 51,747 |
| 2,136 | ` আছে` (is/exists) | Bengali | 49,930 |
| 2,389 | ` ਅਤੇ` (and) | Gurmukhi | 44,571 |
| 2,473 | ` ఏ` (which) | Telugu | 42,938 |
| 3,307 | ` లో` (in) | Telugu | 31,346 |
| 3,537 | ` की` (of) | Devanagari (Hindi) | 29,031 |

> [!NOTE]
> Only 136 out of 16,000 (0.85%) are Indic tokens that survived as whole BPE merges. The vast majority of Indic text is tokenized via byte-fallback (individual UTF-8 bytes), which explains why byte fragments like `à` (rank #1) dominate the frequency chart.

---

## Key Takeaways

1. **Extreme Zipf distribution**: Just 235 tokens cover 50% of the corpus; 16K tokens cover 91.2%.
2. **English-dominated vocabulary**: 59% of the top 16K are complete English words, reflecting the 99.2% English corpus composition.
3. **Byte fragments in top ranks**: 6 of the top 50 tokens are raw UTF-8 bytes from Indic scripts — these are the most frequently seen "tokens" because every Indic character decomposes into them.
4. **Indic script underrepresented**: Only 136 Indic tokens in the top 16K despite 11 Indic languages in the data, confirming the high byte-fragment rate (81–95%) noted in the audit report.
5. **Code tokens are sparse**: Only 159 code-specific tokens in the top 16K (1%), but code constructs like `return`, `def`, `import`, `()`, `->` are well-covered.
6. **Sharp diminishing returns past 16K**: Going from 16K→32K tokens only gains +4.8% coverage (91.2%→96.0%).
