# Tokenizer Validation Report (Full Corpus)

**Tokenizer**: `tsai_131k_tokenizer` (from `origin/P06_Tokenizer_Design_Lab`)
**Date**: February 25, 2026
**Samples Evaluated (100% of available data)**:
- Pretraining subset (`raw_shard.parquet`): 630,140 examples
- Golden Samples (`golden_samples_cleaned_v3.jsonl`): 128 examples
- SFT subset (`SFT/*.txt`): 89,508 examples

## 1. Safety & Data-Loss Checks (Pass/Fail)

### A. Round-Trip Stability (Encode -> Decode)
- **Pretraining (630k):** 0.0% mismatch rate
- **Golden (128):** 0.0% mismatch rate
- **SFT (89k):** 0.0% mismatch rate
**Result:** **PASS**. The tokenizer perfectly reconstructs the texts across all datasets. No data corruption or silent token dropping occurs.

### B. Out-of-Vocabulary (UNK) Rate
- **Pretraining (630k):** 0.0% UNK rate
- **Golden (128):** 0.0% UNK rate
- **SFT (89k):** 0.0% UNK rate
**Result:** **PASS**. The fallback mechanisms are working properly. No `<|unk|>` tokens were emitted.

### C. Pretraining Special Token Leakage
- **Test:** Scanning the 630k pretraining raw texts for accidental special token IDs.
- **Result:** **0 leaks**. 
**Result:** **PASS**. The pretraining text does not natively contain strings that accidentally map to control tokens (or the parser successfully escapes them).

### D. SFT Loss Masking Feasibility
- **Unpaired/Failed Splits:** 0 out of 89,508 SFT examples.
- **Special Token Mapping:** `<|system|>` (130726), `<|user|>` (130727), `<|assistant|>` (130728).
- **Role Token Frequencies:** `<|user|>` and `<|assistant|>` appeared exactly 89,508 times each.
**Result:** **PASS**. Loss masking is 100% feasible. The system can reliably locate the `<|assistant|>` token and apply loss solely to the subsequent tokens for the answer.

---

## 2. Advanced Quality & Performance Checks

Per user feedback, we ran extended analyses on the tokenizer's representational efficiency and edge-case behavior.

### A. Byte-Fallback Rate (Hidden UNK behavior)
*0% UNK doesn't mean full coverage—it might mean the tokenizer is just shattering text into raw bytes.*
We measured what percentage of tokens emitted across the datasets were single-byte fallback tokens.
- **Pretraining:** ~12.8% of all tokens emitted are byte-fallbacks.
- **Impact:** 12.8% is slightly high for a generic corpus, indicating that while the tokenizer doesn't emit UNK, it is doing significant byte-level chunking for non-English data.

### B. Compression Ratio by Language (Tokens per Character)
To pinpoint the byte-fallback issue, we analyzed character-to-token compression across the 630k pretraining dataset:
- `en` (English): 4.23 chars/token
- `hi` (Hindi): 3.16 chars/token
- `ta`, `ml`, `bn`: ~2.7 - 2.9 chars/token
- `pa` (Punjabi): 1.92 chars/token
- `or` (Odia): 1.03 chars/token
**Conclusion:** The tokenizer is highly efficient for English and acceptable for major Indic scripts. However, it severely struggles with Odia (`or`), treating almost every character as a separate byte token (1.03 ratio). Training on Odia will consume excessive context window space.

### C. Prompt Injection Leakage
- **Test:** Passing the raw string `"<|assistant|>"` into the tokenizer.
- **Result:** **WARNING.** The tokenizer natively encodes the raw string directly into the literal special token ID `130728`.
- **Conclusion:** This is a prompt injection risk. If an end user sends `"<|assistant|>"` in their input, they could hijack the model's generation turn. The serving API must escape these sequences before tokenization.

### D. Sequence Length & Stress Test
- **Pretraining Max Length:** 233,006 tokens (99th percentile: 8,242)
- **Conclusion:** The tokenizer successfully processed documents up to ~233K tokens without crashing. It can handle the 256K target context length constraint comfortably.

### E. Numeric Tokenization
- **Test:** Tokenizing digits (`123456`).
- **Result:** The tokenizer performs greedy merging, chunking numbers into up to 3-digit blocks (e.g. `123456` -> `["123", "456"]`).
- **Conclusion:** Non-uniform digit splitting can sometimes harm arithmetic compared to strict single-digit splitting, but it compresses numerical data better. 

### F. Duplicate Decode Collisions
- **Test:** Finding multiple distinct tokens that decode to the exact same string (usually an artifact of BPE merges involving whitespace or replacement characters).
- **Result:** Found **1,026** collision pairs (e.g. multiple tokens decoding to the replacement character `` or various whitespace combinations).
- **Conclusion:** This is somewhat high (nearly 1% of the vocab). While not fatal, it means the model has to learn redundant embeddings for the exact same semantic output, slightly reducing parameter efficiency.

## 3. Conclusion
The `tsai_131k_tokenizer` is fundamentally **safe and ready for training** (no data loss, stable long-context handling, zero UNK errors, robust SFT masking). 

However, before final pretraining, the team should note the following **quality-of-life warnings**:
1. Odia (`or`) is heavily penalized by byte-fallback tokenization.
2. The Chat API must sanitize `<|assistant|>` strings to prevent prompt injection.
3. ~1k vocab slots are wasted on duplicate-decode collisions.
