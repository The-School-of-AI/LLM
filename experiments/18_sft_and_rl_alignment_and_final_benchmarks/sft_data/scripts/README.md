# SFT Data Scripts — Team 18

Scripts for **Section 7.1 SFT Data** checklist. Run from this directory or from `sft_data/`.

## Prerequisites

- Python 3.10+
- Optional: `transformers` for `apply_chat_template.py --tokenizer` and `verify_loss_masking.py --tokenizer`

## Pipeline order

1. **Standardize format** → `standardize_conversation_format.py`
2. **Decontaminate vs benchmarks** → `decontaminate_against_benchmarks.py` (optional but recommended; see [DATASET_BENCHMARK_COVERAGE_MATRIX.md](../DATASET_BENCHMARK_COVERAGE_MATRIX.md))
3. **Apply chat template** → `apply_chat_template.py`
4. **Sample for quality review** → `sample_for_quality_review.py` (manual review of 100+)
5. **Dedup vs pre-training** → `dedup_against_pretrain.py` (requires Team 5 hashes)
6. **Train/val split** → `train_val_split.py`
7. **Verify loss masking** → `verify_loss_masking.py` (and in your training code)

## Usage

### 1. Standardize to system/user/assistant

```bash
python standardize_conversation_format.py input.jsonl standardized.jsonl --format alpaca
# or --format sharegpt | already_conversation
```

### 2. Decontaminate against benchmark test sets

Ensures no benchmark contamination: removes SFT examples whose **prompt** (user content) hashes match any benchmark test-set hash. Use after standardize (conversation format).

**Step 2a — Build benchmark hash files** (one per benchmark test set):

```bash
# Example: build hashes from MATH test JSONL (field 'problem' or 'question')
python build_benchmark_hashes.py /path/to/math_test.jsonl benchmark_hashes/math_test.txt --text-field problem
python build_benchmark_hashes.py /path/to/gsm8k_test.jsonl benchmark_hashes/gsm8k_test.txt --text-field question
# Or use a directory of hash files from Team 5 / benchmark owners
```

**Step 2b — Run decontamination** (use standardized JSONL):

```bash
# Single hash file
python decontaminate_against_benchmarks.py standardized.jsonl decontaminated.jsonl \
  --benchmark-hashes benchmark_hashes/math_test.txt \
  --benchmark-hashes benchmark_hashes/gsm8k_test.txt

# Or a directory of hash files
python decontaminate_against_benchmarks.py standardized.jsonl decontaminated.jsonl \
  --benchmark-hashes-dir benchmark_hashes/

# Or benchmark test sets as JSONL (script hashes on the fly)
python decontaminate_against_benchmarks.py standardized.jsonl decontaminated.jsonl \
  --benchmark-jsonl math:/path/to/math_test.jsonl \
  --benchmark-jsonl gsm8k:/path/to/gsm8k_test.jsonl \
  --text-field question

# Optional: write removed examples for audit
python decontaminate_against_benchmarks.py standardized.jsonl decontaminated.jsonl \
  --benchmark-hashes-dir benchmark_hashes/ --removed-out removed_contaminated.jsonl
```

Hash mode: default is `--hash-mode prompt` (user content only). Use `--hash-mode full` to match on full conversation. Same hash function as `dedup_against_pretrain.py` (SHA256 of normalized text).

### 3. Apply chat template

```bash
python apply_chat_template.py standardized.jsonl templated.jsonl --template chatml
# optional: --tokenizer path/to/model --max-length 2048
```

### 4. Sample for manual review (100+)

```bash
python sample_for_quality_review.py standardized.jsonl review_sample.jsonl --n 100
```

### 5. Dedup against pre-training

```bash
python dedup_against_pretrain.py standardized.jsonl deduped.jsonl --pretrain-hashes /path/to/pretrain_hashes.txt
# optional: --dedup-within-sft
```

### 6. Train/val split

```bash
python train_val_split.py deduped.jsonl --train-out train.jsonl --val-out val.jsonl --val-ratio 0.05 --seed 42
```

### 7. Verify loss masking

```bash
python verify_loss_masking.py train.jsonl --tokenizer path/to/model --sample 5
```

## Training implementation (items 8 & 9)

In your SFT training code (LoRA/QLoRA):

- Build **labels** so that only **assistant** token positions have the true token id; all other positions (system, user, padding) use **-100**.
- Use **right padding**; set label to **-100** for every padding position.
- Use `torch.nn.CrossEntropyLoss(ignore_index=-100)`.

These scripts do not implement the full training collator; integrate the same logic into your data loader.
