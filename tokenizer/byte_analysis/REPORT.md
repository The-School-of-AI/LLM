# Tokenizer Byte-Length Analysis — FINAL_TOKENIZER

- **Tokenizer path:** `tokenizer/tokenizer.json` in this release tree.
- **Type:** GPT-2 byte-level BPE (HF `tokenizers` format)
- **Vocab size:** 131072
- **Total bytes (sum of surface-form lengths):** 863,105

## Category counts

| Category | Count |
|---|---:|
| normal | 130720 |
| bytefallback | 0 |
| special | 352 |
| other | 0 |

## Byte-length distribution

| byte_count_bin | num_tokens | pct |
|---|---:|---:|
| 1 byte | 243 | 0.19% |
| 2 bytes | 3959 | 3.02% |
| 3 bytes | 12857 | 9.81% |
| 4 bytes | 19622 | 14.97% |
| 5-8 bytes | 64893 | 49.51% |
| 9-12 bytes | 22927 | 17.49% |
| 13-16 bytes | 5039 | 3.84% |
| 17-24 bytes | 1244 | 0.95% |
| 25-32 bytes | 288 | 0.22% |
| 33-48 bytes | 0 | 0.00% |
| 49+ bytes | 0 | 0.00% |

## Coverage at common POS_DIM values

| POS_DIM | fit (normal+bytefallback) | pct | fit (all incl. special) | pct |
|---:|---:|---:|---:|---:|
| 1 | 243 | 0.19% | 243 | 0.19% |
| 2 | 4202 | 3.21% | 4202 | 3.21% |
| 4 | 36681 | 28.06% | 36681 | 27.99% |
| 8 | 101561 | 77.69% | 101574 | 77.49% |
| 12 | 124455 | 95.21% | 124501 | 94.99% |
| 16 | 129198 | 98.84% | 129540 | 98.83% |
| 20 | 130010 | 99.46% | 130362 | 99.46% |
| 24 | 130432 | 99.78% | 130784 | 99.78% |
| 32 | 130720 | 100.00% | 131072 | 100.00% |
| 48 | 130720 | 100.00% | 131072 | 100.00% |
| 64 | 130720 | 100.00% | 131072 | 100.00% |

- **POS_DIM=16** covers **98.84%** of normal+bytefallback tokens (98.83% of all).
- **POS_DIM=32** covers **100.00%** of normal+bytefallback tokens (100.00% of all).
- **POS_DIM=32 truncates 0 normal+bytefallback tokens** (and 0 tokens overall, including specials).

### Longest 20 normal+bytefallback tokens truncated at POS_DIM=32

| token_id | num_bytes | category | piece_repr |
|---:|---:|---|---|
| _(none)_ | | | |

## Top 20 longest tokens overall

| token_id | num_bytes | num_codepoints | category | piece_repr |
|---:|---:|---:|---|---|
| 1062 | 32 | 32 | normal | 'ĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠ… |
| 2323 | 32 | 32 | normal | '----------------------------… |
| 2834 | 32 | 32 | normal | '****************************… |
| 5124 | 32 | 32 | normal | '============================… |
| 7791 | 32 | 32 | normal | '////////////////////////////… |
| 8773 | 32 | 32 | normal | '############################… |
| 19551 | 32 | 32 | normal | '____________________________… |
| 21578 | 32 | 32 | normal | '............................… |
| 57525 | 32 | 32 | normal | '%%%%%%%%%%%%%%%%%%%%%%%%%%%%… |
| 83241 | 32 | 32 | normal | '~~~~~~~~~~~~~~~~~~~~~~~~~~~~… |
| 111350 | 32 | 32 | normal | '++++++++++++++++++++++++++++… |
| 2056 | 31 | 31 | normal | 'ĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠ… |
| 122076 | 31 | 11 | normal | 'Ġà¤®à¤¹à¤¤à¥įà¤µà¤ªà¥Ĥà¤°à¥į… |
| 122792 | 31 | 11 | normal | 'Ġà´¸àµĨà´ķàµįà´°à´Łàµįà´Łà´±… |
| 123024 | 31 | 11 | normal | 'Ġà´ªàµįà´°à´¸à´¿à´¡à´¨àµįà´±… |
| 125066 | 31 | 11 | normal | 'Ġà¤°à¤¾à¤·à¥įà¤Łà¥įà¤°à¤ªà¤¤… |
| 125082 | 31 | 11 | normal | 'Ġà¦¬à¦¾à¦Ĥà¦²à¦¾à¦¦à§ĩà¦¶à§ĩ… |
| 125845 | 31 | 11 | normal | 'Ġàªµàª¿àª¸à«įàª¤àª¾àª°àª®àª¾… |
| 126299 | 31 | 11 | normal | 'Ġà¤ħà¤§à¤¿à¤ķà¤¾à¤°à¤¿à¤¯à¥ĭ… |
| 126907 | 31 | 11 | normal | 'Ġà¤®à¤¹à¤¾à¤°à¤¾à¤·à¥įà¤Łà¥į… |

## Category breakdown

| category | count | mean num_bytes | max num_bytes | p99 num_bytes |
|---|---:|---:|---:|---:|
| normal | 130720 | 6.56 | 32 | 18 |
| bytefallback | 0 | — | — | — |
| special | 352 | 14.74 | 20 | 19 |
| other | 0 | — | — | — |

## Surface-form recovery sanity check (10 random normal tokens)

| token_id | piece_repr | re-encoded ids | pass |
|---:|---|---|:---:|
| 110692 | '<Sprite' | [110692] | ✓ |
| 50504 | 'Ġ.....' | [50504] | ✓ |
| 99358 | 'Ġgladi' | [99358] | ✓ |
| 116699 | 'ĠPassing' | [116699] | ✓ |
| 55135 | 'endeu' | [55135] | ✓ |
| 5311 | 'Ġdifficult' | [5311] | ✓ |
| 33946 | '@implementation' | [33946] | ✓ |
| 126568 | 'Ġà®ĩà®©' | [126568] | ✓ |
| 67024 | 'Ġ$"{' | [67024] | ✓ |
| 63702 | 'nw' | [63702] | ✓ |

**10/10 round-trip correctly.**

---
**One-liner:** POS_DIM=32 covers 100.00% of normal+bytefallback tokens, 100.00% of all tokens (including specials). No normal+bytefallback token exceeds 32 bytes — the longest such token in the vocabulary is exactly 32 bytes, so POS_DIM=32 truncates nothing.
