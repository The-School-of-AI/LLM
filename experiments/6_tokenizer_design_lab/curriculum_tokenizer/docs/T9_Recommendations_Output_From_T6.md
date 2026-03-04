# Tokenizer Team — Recommendations for Training-Compatible Output
**Issued by:** Training stack team
**Last updated:** 2026-02-22
**Status:** Action required before next tokenization run

---

## Context

The training team consumes pre-tokenized `.bin/.idx` files produced by the tokenizer team.
We are now moving toward curriculum-governed, multi-GPU, crash-recoverable training.
This document describes exactly what the training stack requires from the tokenizer output
format, and why each requirement exists.

Read this before producing the next tokenization run.

---

## CRITICAL: Tokenizer Identity Mismatch Found in Current Files

**This must be resolved before any serious training run.**

The existing `.bin` files in `data_loader/` were generated with a tokenizer whose identity is:

```
eos_token_id : 130717
pad_token_id : 130718
vocab_size   : 131072
```

The tokenizer currently used by the training code (`code/src/tokenizer/`) is the **TSAI 131K tokenizer**, which has:

```
eos_token  : "<|end_of_text|>"      (different token content)
pad_token  : "<|pad|>"   (different token content)
added special tokens start at: 130716
```

These are **not the same tokenizer**. If the `.bin` files produced by one tokenizer are loaded by a model configured with the other, every token ID in the training data maps to the wrong vocabulary entry. The model trains on a corrupted ID space and produces garbage — silently, with no error.

**Resolution required:**
1. Canonize which tokenizer is the final one (training team and tokenizer team must agree)
2. All `.bin` files must be regenerated with that canonical tokenizer
3. The tokenizer hash (see Section 2) must be embedded in all future metadata

---

## 1. Required Metadata Sidecar per Shard

Each shard is a **subdirectory**. Every shard directory contains exactly three files with fixed names — no per-shard prefix coordination needed:

```
shards/
  shard_001/
    tokens.bin
    tokens.idx
    metadata.json
  shard_002/
    tokens.bin
    tokens.idx
    metadata.json
```

The `metadata.json` file is always named `metadata.json` inside its shard directory. Required fields:

```json
{
  "tokenizer_hash": "<sha256 of tokenizer.json + special_tokens_map.json>",
  "eos_token_id": 130717,
  "pad_token_id": 130718,
  "vocab_size": 131072,
  "block_size": 4096,
  "num_blocks": 6021,
  "total_tokens": 24662016,
  "rows_input": 120406,
  "rows_with_eos": 120404,
  "rows_dropped": 2,
  "tokens_dropped": 8192,
  "drop_reason": "tail_truncation_at_block_boundary",
  "band": "B2",
  "domain": "reasoning",
  "stage": 1,
  "source_file": "curriculum/part-00000-xxxx.parquet",
  "created_at": "2026-02-22T10:00:00Z",
  "tokenizer_version": "v1"
}
```

### Why each field matters

| Field | Reason |
|-------|--------|
| `tokenizer_hash` | Training loader validates this against the loaded tokenizer before reading a single token. Mismatch = hard fail, not silent corruption. |
| `eos_token_id`, `pad_token_id`, `vocab_size` | Sanity-checked against the live tokenizer at load time. If these don't match, the `.bin` was made with a different tokenizer version. |
| `dtype` | Training loader picks the right `numpy.dtype`. Must match what was written. |
| `block_size` | Must be 4096. The loader asserts this at startup. Used when joining consecutive blocks for larger context windows (e.g., 2 × 4096 = 8192). |
| `rows_input`, `rows_with_eos`, `rows_dropped`, `tokens_dropped` | **Non-negotiable auditability.** Project rules forbid silent data loss. Every dropped row must be logged and surfaced. |
| `band`, `domain`, `stage` | The curriculum sampler receives an external shard list organised by band/domain. Without these fields the sampler cannot build that list — curriculum-governed training breaks entirely. |
| `source_file` | Full traceability from training step back to raw source. Required for contamination audits. |

---

## 2. How to Compute the Tokenizer Hash

The hash must be stable and deterministic. Recommended method:

```python
import hashlib, json, os

def compute_tokenizer_hash(tokenizer_dir: str) -> str:
    # Combine the two files that fully define the tokenizer
    files = ["tokenizer.json", "special_tokens_map.json"]
    h = hashlib.sha256()
    for fname in sorted(files):
        fpath = os.path.join(tokenizer_dir, fname)
        with open(fpath, "rb") as f:
            h.update(fname.encode())   # include filename in hash
            h.update(f.read())
    return h.hexdigest()
```

Embed this in every `metadata.json`. The training loader will recompute it from `code/src/tokenizer/` at startup and assert equality.

**Do not include `tokenizer_config.json` in the hash** — it contains mutable metadata like `model_max_length` that doesn't affect token IDs.

---

## 3. Required Changes to the Tokenization Pipeline

### 3a. Block Size is Fixed at 4096 — Keep the Current Format

**Decision (instructor):** The training block size is always 4096 tokens. This is a project constant and will not change.

**What this means for the tokenizer team:** Continue producing fixed 4096-token blocks. No format change is needed here.

**How larger context windows work (seq_len > 4096):**
The training loader handles this without re-tokenization. If `seq_len=8192` is needed, the loader reads two consecutive 4096-token blocks and joins them in memory. The flat token stream is already correct — EOS tokens inside the joined window are preserved as regular tokens for the model to learn from.

```
seq_len=4096  → 1 block per training sequence  (default)
seq_len=8192  → 2 consecutive blocks joined    (loader handles this)
seq_len=16384 → 4 consecutive blocks joined    (loader handles this)
```

**Your only obligation:** produce clean, correctly-labelled 4096-token blocks with the metadata sidecar described in Section 1. The loader takes care of the rest.

> **Note for future:** if document boundary masking becomes a requirement (so the model does not attend across document boundaries within a joined window), the `.idx` format would need to carry document-level offsets rather than block offsets. This is not a current requirement but worth keeping in mind when designing your indexing logic.

### 3b. Tail Handling — Choose One and Log It

The current files silently drop the partial tail block (2 rows missing EOS). Choose one strategy and log it explicitly:

| Strategy | When to use | How to implement |
|----------|------------|-----------------|
| **Drop remainder** | Default for large datasets | Drop tokens that don't fill a complete block; log exact `rows_dropped` and `tokens_dropped` in `metadata.json` |
| **Pad remainder** | When every row matters | Pad last block to `block_size` with `pad_token_id`; set `attention_mask=0` for pad positions |
| **Carry to next shard** | Streaming / multi-shard pipelines | Write incomplete tail to a `remainder.bin`; prepend to next shard at tokenization time |

For now: **Drop remainder, log it exactly.** The current behavior of dropping silently is the problem — not the dropping itself.

### 3c. EOS Token Placement

Current approach: one EOS appended per row before concatenation. This is correct.

**Verify:** `rows_with_eos == rows_input` (or `rows_input - 1` at most for the final tail drop). If the count differs by more, investigate — it means some rows were processed without getting an EOS appended.

The training loader will assert this from `metadata.json`.

---

## 4. Validation Script to Run Before Handing Off

Run this on every batch of shards before delivering to the training team:

```
For each shard:
  [ ] shard directory contains exactly: tokens.bin, tokens.idx, metadata.json
  [ ] tokenizer_hash in metadata.json matches current canonical tokenizer
  [ ] eos_token_id, pad_token_id, vocab_size in metadata.json match live tokenizer
  [ ] metadata.total_tokens == actual tokens.bin file size / bytes_per_token
  [ ] len(idx_offsets) - 1 == metadata.num_blocks
  [ ] rows_dropped + rows_with_eos == rows_input
  [ ] max(token_ids_in_bin) < vocab_size
  [ ] band, domain, stage fields are non-empty strings in metadata.json
```

If any check fails: do not deliver that shard. Fix and re-run.

---

## 5. Summary of Deliverables

| Item | Current state | Required state |
|------|--------------|----------------|
| Tokenizer identity match | MISMATCH with training code | Must be resolved first |
| Tokenizer hash in metadata | Missing | Required |
| Band/domain/stage in metadata | Missing | Required |
| Tail truncation logged | Silent | Must be explicit in `metadata.json` |
| Block format | Fixed 4096 blocks | Keep as-is (instructor decision). Loader handles larger context by joining blocks. |
| Validation script | None | Run before every handoff |

---

## Questions

Direct to training stack team via the capstone project channel.
Reference this document version (2026-02-22) in all questions.
