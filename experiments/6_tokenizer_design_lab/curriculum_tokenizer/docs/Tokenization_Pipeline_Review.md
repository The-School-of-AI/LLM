# Tokenization Pipeline Review & AWS Strategy

**Document version:** 2026-03-04
**Pipeline:** `tokenize_curriculum.py` — S3 Curriculum Tokenization Pipeline
**Data scale:** ~20B tokens (production), 11.7B (current Stage 1B coreset)
**Source data region:** us-east-1 (already available)

---

## Table of Contents

1. [Pipeline Review — Pending Items Checklist](#1-pipeline-review--pending-items-checklist)
2. [Local Testing Strategy](#2-local-testing-strategy)
3. [Parallel Tokenization — Current State & Enhancement](#3-parallel-tokenization--current-state--enhancement)
4. [AWS Deployment Guide](#4-aws-deployment-guide)
   - [4.1 Execution Strategy](#41-execution-strategy)
   - [4.2 Cost and Duration Breakdown](#42-cost-and-duration-breakdown)
   - [4.3 S3 Setup and IAM](#43-s3-setup-and-iam)
   - [4.4 Spot Instance Interrupt Handling](#44-spot-instance-interrupt-handling)

---

## 1. Pipeline Review — Pending Items Checklist

### P0 — BLOCKERS (pipeline cannot run correctly without these)

- [x] **[ARCH-01] Script downloads T2 instead of T1 — 2-level migration required.**
  - **Supersedes**: BUG-01 (T2 path construction no longer needed; T2 is entirely bypassed).
  - **Location**: `process_coreset_file()` — the entire source-file groupby and download block.
  - **Problem**: The script groups T3 rows by `source_url + source_doc_id` (a T2 band directory path), downloads T2 band files, and filters T2 by `id == chunk_id`. T2 has no `text` column, so the script always hits the `SKIP` branch and produces zero tokens.
  - **Fix**: Group by `t1_file_path` column (hardcoded — fixed by T3 schema). Build T1 URI as `args.t1_base_uri.rstrip("/") + "/" + t1_file_path`. Download T1 directly. Filter `T1.id == chunk_id`. Extract `T1.text`.
  - **New arg**: `--t1-base-uri` (default: `s3://t1-dataacquisition-datasets/processed_dataset/normalized_data`; override to `dataset/final/t1` for local testing).
  - **Dead args to remove**: `--src-doc-col`, `--url-col`, `--src-id-col`, `--coreset-id-col`, `--text-col`, `--band-col`, `--domain-col`.
  - **Hardcoded constants**: `t1_file_path` (T3 column), `chunk_id` (T3 ID column), `id` (T1 ID column), `text` (T1 text column), `band`, `domain`.

- [x] **[BUG-02] `key_exists()` uses wrong exception class.**
  - **Location**: Line 105: `except s3.exceptions.ClientError`
  - **Problem**: `boto3.client("s3").exceptions.ClientError` does not exist; `ClientError` is a `botocore` base class, not a service-specific exception. This raises `AttributeError` at runtime, breaking the checkpoint-skip logic entirely.
  - **Fix applied**: Added `from botocore.exceptions import ClientError` at top; changed to `except ClientError as e: if e.response["Error"]["Code"] in ("404", "NoSuchKey"): return False; raise`.

### P1 — CRITICAL (required for training team delivery)

- [x] **[META-01] `tokenizer_hash` missing from `metadata.json`.**
  - Required by `TOKENIZER_TEAM_RECOMMENDATIONS.md` §2. Must be SHA256 of `tokenizer.json` + `special_tokens_map.json` (sorted filenames prepended to hash input).
  - **Fix applied**: Added `compute_tokenizer_hash(tokenizer_dir)` function; pass hash into `ShardWriter.__init__()`; include in every `metadata.json`.

- [x] **[META-02] `band` and `band_distribution` missing from `metadata.json`.**
  - Required for curriculum sampler. Coreset parquets have a `band` column (B0/B1/B2).
  - **Fix applied**: Column name `band` is hardcoded (fixed by T3 schema — not a CLI arg); compute `value_counts()` per batch group; pass distribution into `ShardWriter`; write dominant band + full distribution dict.

- [x] **[META-03] `domain` missing from `metadata.json`.**
  - Same pattern as band. Column name `domain` is hardcoded (fixed by T3 schema — not a CLI arg).

- [x] **[META-04] `stage` missing from `metadata.json`.**
  - **Fix applied**: Added `--stage` arg (integer, e.g., `1`); pass into `ShardWriter`.

- [x] **[META-05] `source_file` missing from `metadata.json`.**
  - Must be the coreset batch parquet URI for full traceability. **Fix applied** in `ShardWriter`.

- [x] **[META-06] `created_at` missing from `metadata.json`.**
  - **Fix applied**: `datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")` in `flush_shard()`.

- [x] **[META-07] `tokenizer_version` missing from `metadata.json`.**
  - **Fix applied**: Added `--tokenizer-version` arg (string, e.g., `"v1"`).

- [x] **[AUDIT-01] `rows_input` not tracked per shard.**
  - Added counter `shard_rows_input` in `ShardWriter`; incremented in the token-packing loop.

- [x] **[AUDIT-02] `rows_with_eos` not tracked per shard.**
  - EOS IS correctly appended per row at line 440 (not a logic bug). The count was not persisted to `metadata.json`.
  - **Fix applied**: Added `shard_rows_with_eos` counter; incremented alongside the `buffer.append(eos)` call.

- [x] **[AUDIT-03] `rows_dropped` computed but not written to `metadata.json`.**
  - Lines 469-476 calculated `dropped_rows` but only `print()` it. **Fix applied**: persisted to metadata.

- [x] **[AUDIT-04] `tokens_dropped` computed but not written to `metadata.json`.**
  - Same: `dropped_tokens = len(buffer)` was computed but not persisted. **Fix applied**.

- [x] **[AUDIT-05] `drop_reason` not written to `metadata.json`.**
  - Must be `"tail_truncation_at_block_boundary"` or `"padded"`. **Fix applied**.

### P2 — IMPORTANT (for production AWS run)

- [x] **[PERF-01] File-level parallelism missing.**
  - 133 coreset batch files were processed sequentially. All CPU cores sat idle while one file downloaded/tokenized.
  - **Fix applied**: Added `--file-parallelism N` arg; uses `multiprocessing.get_context("spawn").Pool(N)`; worker function creates its own `boto3.client` and `AutoTokenizer` (cannot share across `fork`); each worker gets isolated `worker_tmp` subdirectory.
  - **Recommended settings**: c5.9xlarge → `--file-parallelism 12 --num-proc 3` (36 vCPU fully used).

- [x] **[SPOT-01] No Spot interrupt handling.**
  - **Fix applied**: Added IMDS polling daemon thread (checks `http://169.254.169.254/latest/meta-data/spot/termination-time` every 5s); added `signal.signal(SIGTERM, handler)` handler; sets a `threading.Event` flag; checks flag at the start of each source-file loop iteration; on termination: discards partial `accumulated_blocks`, only completed shards are preserved.

- [x] **[SPOT-02] No cross-interrupt progress state file.**
  - **Fix applied**: Writes `progress_state.json` to `dst_uri` after each batch file completes. On startup, reads this file and skips already-completed batch URIs (faster than per-shard S3 `head_object` checks).

### P3 — QUALITY FIXES

- [x] **[CORR-01] `tmp_dir` cleanup uses `os.rmdir()` which fails silently on non-empty dirs.**
  - Line 616. **Fix applied**: Uses `shutil.rmtree(tmp_dir, ignore_errors=True)`.

- [x] **[CORR-02] Global `manifest.json` uses Unix epoch float for timestamp.**
  - Line 600: `"timestamp": time.time()`. **Fix applied**: Uses ISO 8601 string for consistency with shard metadata.

- [x] **[CORR-03] `src_id_col` default `"id"` may not match source parquet column name.**
  - Resolved — T1 schema confirmed: the ID column is `id` (matches default). Hardcoded as a constant in the 2-level migration (ARCH-01); `--src-id-col` arg removed.

### P4 — TOOLING

- [x] **[VALID-01] No standalone validation script.**
  - `TOKENIZER_TEAM_RECOMMENDATIONS.md` §4 defines an 8-point checklist. Created `validate_shards.py` that: reads each shard's `metadata.json`, checks `total_tokens == file_size/4`, checks `len(idx_offsets)-1 == num_blocks`, verifies `tokenizer_hash`, validates `max(token_ids) < vocab_size`, checks `rows_dropped + rows_with_eos == rows_input`, and reports pass/fail per shard.

### Files Modified / Created

| File | Action |
|------|--------|
| `tokenize_curriculum.py` | Modified — all P0/P1/P2/P3 fixes applied |
| `Tokenization-Strategy-AWS.md` | Created — full AWS deployment guide |
| `scripts/create_mock_sources.py` | Redesigned — reads real T2 `file_path` values from local band files in `datafiles/`; generates mock T3 (with `t1_file_path` column) and mock T1 files with full schema; no T2 mocks created |
| `validate_shards.py` | Created — post-run 8-point validation script |

---

## 2. Local Testing Strategy

### Step 1 — Create mock source parquets (no S3 needed)

The generator builds a 2-level mock dataset under `dataset/final/` using the real T1 files that
already exist locally at `dataset/source/t1_rawdata/normalized_data/source=C4/` (5 files available).

```python
# scripts/create_mock_sources.py — redesigned for 2-level architecture
#
# Source T1 files (read-only, never modified):
#   dataset/source/t1_rawdata/normalized_data/source=C4/part-0000{0..4}-8299c866-...parquet
#
# Generation steps:
# 1. Read T3 coresets from coresets/1B/; collect all unique source_doc_id groups
#    and the chunk_ids that belong to each group.
# 2. For each unique source_doc_id group, assign a real T1 filename from the pool above
#    (round-robin; if more groups than available files, create copies named part-00005-..., etc.)
#    t1_file_path = "source=C4/<assigned_t1_filename>"
# 3. Write mock T3 parquets to dataset/final/t3/ — same as real T3 but with
#    t1_file_path column added (value = t1_file_path assigned in step 2).
# 4. For each assigned T1 file:
#      - Copy it to dataset/final/t1/<t1_file_path>
#      - Replace the id column values with the chunk_ids from T3 that map to this file
#        (preserves all other columns — hash, text, domain, language, metadata, etc.)
#    This ensures T1.id == T3.chunk_id filter always succeeds.
#    One decoy row (original id not in T3) is kept as-is to verify filtering works.
```

```bash
python scripts/create_mock_sources.py \
  --profile        minimal|small|parallel \
  --t3-source-dir  dataset/source/t3_coresets \
  --t1-source-dir  dataset/source/t1_rawdata/normalized_data \
  --output-dir     dataset/final
```

Output structure:
```
dataset/final/
  t1/
    source=C4/
      part-00000-8299c866-c99b-45fc-92d0-4d8b5c1f7503-c000.zstd.parquet  ← ids replaced
      part-00001-8299c866-c99b-45fc-92d0-4d8b5c1f7503-c000.zstd.parquet  ← ids replaced
      ...  (one file per unique source_doc_id group in T3; copies created if needed)
  t3/
    selected_indices_part_shard000_batch000000.parquet  ← real T3 + t1_file_path column
    ...
```

### Step 2 — Minimal smoke test (single T3 batch file)

```bash
python tokenize_curriculum.py \
  --coreset-uri   dataset/final/t3/selected_indices_part_shard000_batch000000.parquet \
  --dst-uri       dataset/final/tok_out \
  --tokenizer-path ./tsai_131k_tokenizer \
  --t1-base-uri   dataset/final/t1 \
  --block-size    4096 \
  --shard-size-mb 512 \
  --num-proc      2 \
  --drop-remainder \
  --stage         1 \
  --tokenizer-version v1 \
  --tmp-dir       /tmp/tok_tmp
```

### Step 3 — Verify output

```bash
# Check structure
find dataset/final/tok_out -type f

# Verify token count math
python -c "
import numpy as np, json, pathlib
shard_dir = next(pathlib.Path('dataset/final/tok_out').rglob('shard_000'))
meta = json.load(open(shard_dir / 'metadata.json'))
tokens = np.fromfile(shard_dir / 'tokens.bin', dtype=np.uint32)
print('total_tokens match:', meta['total_tokens'] == len(tokens))
print('max token id valid:', tokens.max() < 131072)
print('EOS count:', (tokens == 130717).sum())
print('band:', meta.get('band'), 'domain:', meta.get('domain'))
print('tokenizer_hash present:', 'tokenizer_hash' in meta)
"
```

### Step 4 — Resume test

```bash
# Delete one shard to simulate interrupted run
python -c "
import shutil, pathlib
shard = next(pathlib.Path('dataset/final/tok_out').rglob('shard_000'))
shutil.rmtree(shard); print('Deleted:', shard)
"
# Re-run with same args — shard_000 regenerated, completed shards skipped
python tokenize_curriculum.py \
  --coreset-uri   dataset/final/t3/selected_indices_part_shard000_batch000000.parquet \
  --dst-uri       dataset/final/tok_out \
  --tokenizer-path ./tsai_131k_tokenizer \
  --t1-base-uri   dataset/final/t1 \
  --block-size    4096 \
  --shard-size-mb 512 \
  --num-proc      2 \
  --drop-remainder \
  --stage         1 \
  --tokenizer-version v1 \
  --tmp-dir       /tmp/tok_tmp
```

### Step 5 — Directory-level run with all mock T3 files

```bash
python tokenize_curriculum.py \
  --coreset-uri    dataset/final/t3 \
  --dst-uri        dataset/final/tok_out_full \
  --tokenizer-path ./tsai_131k_tokenizer \
  --t1-base-uri    dataset/final/t1 \
  --file-parallelism 4 \
  --num-proc       2 \
  --block-size     4096 \
  --shard-size-mb  512 \
  --drop-remainder \
  --stage          1 \
  --tokenizer-version v1 \
  --tmp-dir        /tmp/tok_tmp
```

### Verification Steps

1. **Local smoke test**: Run single-file tokenization against mock source data → verify `metadata.json` has all required fields → run `validate_shards.py`
2. **Resume test**: Delete one shard → re-run → verify shard regenerated, others skipped
3. **Parallel test**: Run `--file-parallelism 4` on 4 batch files → verify 4 separate output directories with correct per-shard metadata
4. **AWS dry run**: Launch c5.4xlarge spot → run 2-3 batch files only → validate output → terminate
5. **Full AWS run**: Launch c5.9xlarge spot → process all 133 batch files → run validation → terminate

---

## 3. Parallel Tokenization — Current State & Enhancement

### Current capability

- **Within each file**: `datasets.Dataset.map()` with `num_proc=min(args.num_proc, 4)` — parallel HF tokenization (up to 4 subprocess workers per batch).
- **Across files**: Sequential loop (`for idx, uri in enumerate(target_files)`) — **no parallelism**. 133 batch files processed one at a time.

### Enhancement: `--file-parallelism N`

Architecture change (added to `main()`):

```python
ctx = multiprocessing.get_context("spawn")  # safe with boto3 + HF tokenizers
with ctx.Pool(processes=file_parallelism) as pool:
    results = pool.map(worker_process_coreset, worker_inputs)
```

Worker function (`_worker_process_coreset`):

- Creates its own `boto3.client("s3")` after fork (not safe to share across processes)
- Loads its own `AutoTokenizer` instance
- Uses isolated `worker_tmp = os.path.join(tmp_dir, f"worker_{worker_id:03d}")`

### Memory per worker

- Source parquet in memory: ~50–200 MB
- `accumulated_blocks` (ShardWriter, pre-flush): up to 512 MB
- HF Dataset object: proportional to filtered subset
- **Peak per worker**: ~700–900 MB

### Recommended parallelism settings

| Instance | vCPU | RAM | `--file-parallelism` | `--num-proc` |
|----------|------|-----|---------------------|-------------|
| c5.4xlarge | 16 | 32 GB | 8 | 2 |
| **c5.9xlarge** | **36** | **72 GB** | **12** | **3** |
| r5.4xlarge | 16 | 128 GB | 8 | 2 |

---

## 4. AWS Deployment Guide

**Recommended instance:** c5.9xlarge Spot
**Estimated cost:** ~$1.80–$2.50 total compute

### 4.1 Execution Strategy

#### Architecture Overview

```
S3 us-east-1 (CORESET INDEX — T3, already available):
  s3://t2-datacurriculum-353/coreset_outputs/coresets/1B/
    selected_indices_part_shard000_batch000000.parquet
    selected_indices_part_shard000_batch000001.parquet
    ...  (multiple batch files; each contains chunk_id + t1_file_path + band + domain columns)

S3 us-east-1 (RAW TEXT — T1, already available):
  s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/
    source=C4/
      part-00759-8299c866-c99b-45fc-92d0-4d8b5c1f7503-c000.zstd.parquet
      ...  (thousands of parquet files; looked up via T3.t1_file_path)

Local → uploaded to S3 before run:
  tsai_131k_tokenizer/
    tokenizer.json, special_tokens_map.json, tokenizer_config.json
  tokenize_curriculum.py  (+ validate_shards.py, scripts/)

S3 us-east-1 (DESTINATION — created during run):
  s3://your-training-bucket/tokenized/run_YYYYMMDD/
    progress_state.json                          ← cross-interrupt resume state
    manifest.json                                ← global summary after completion
    selected_indices_part_shard000_batch000000/
      shard_000/
        tokens.bin        ← uint32 little-endian token IDs (512 MB)
        tokens.idx        ← spdl-compatible binary index
        metadata.json     ← rich metadata (tokenizer_hash, band, rows_input, ...)
      shard_001/ ...
    selected_indices_part_shard000_batch000001/
      ...
```

#### Why Single Large Instance (Not Distributed)

- S3 source reads from EC2 in the **same region** (us-east-1) are **free** and very fast
- The bottleneck is **CPU tokenization**, not I/O
- A single `c5.9xlarge` (36 vCPU) with 12-way file-level parallelism saturates all cores
- No distributed coordination overhead — the checkpoint system handles interrupts natively
- EMR/ECS setup would add hours of overhead for a job that runs in ~4 hours

#### Step 1 — Prerequisite: Verify AWS Credentials and Bucket Access

```bash
# From your local machine
aws sts get-caller-identity  # verify credentials are set

# Verify read access to T3 coreset bucket
aws s3 ls s3://t2-datacurriculum-353/coreset_outputs/coresets/1B/ --region us-east-1

# Verify read access to T1 raw text bucket
aws s3 ls s3://t1-dataacquisition-datasets/processed_dataset/normalized_data/ --region us-east-1

# Create destination bucket if not already existing (see Section 4.3)
aws s3 ls s3://your-training-bucket || aws s3 mb s3://your-training-bucket --region us-east-1
```

#### Step 2 — Upload Tokenizer and Code to S3

T3 coresets and T1 raw text are already on S3 — no upload needed for source data.
Only the tokenizer and pipeline code need to be uploaded from your local machine.

> **Note:** Alternatively, the tokenizer JSON files (`tokenizer.json`, `special_tokens_map.json`,
> `tokenizer_config.json`) can be synced directly from the project GitHub repository onto the EC2
> instance instead of uploading from local.

```bash
# Set your bucket name
BUCKET="your-training-bucket"

# Upload tokenizer (~5 MB)
aws s3 sync tsai_131k_tokenizer/ s3://${BUCKET}/tsai_131k_tokenizer/ \
  --region us-east-1

# Upload code
aws s3 sync . s3://${BUCKET}/tokenizer-code/ \
  --region us-east-1 \
  --exclude ".git/*" \
  --exclude "*.pyc" \
  --exclude "__pycache__/*" \
  --exclude ".DS_Store" \
  --exclude "datasets/*" \
  --exclude "coresets/*" \
  --exclude "tsai_131k_tokenizer/*"
```

#### Step 3 — Create IAM Role for EC2 (one-time setup)

See [Section 4.3](#43-s3-setup-and-iam) for the full IAM policy. Quick setup:

```bash
# Save the trust policy
cat > /tmp/ec2-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

# Create role and instance profile
aws iam create-role \
  --role-name TokenizationInstanceRole \
  --assume-role-policy-document file:///tmp/ec2-trust.json

aws iam create-instance-profile --instance-profile-name TokenizationRole
aws iam add-role-to-instance-profile \
  --instance-profile-name TokenizationRole \
  --role-name TokenizationInstanceRole

# Attach the inline policy (see Section 4.3 for full JSON)
aws iam put-role-policy \
  --role-name TokenizationInstanceRole \
  --policy-name TokenizationPolicy \
  --policy-document file:///tmp/tokenization-policy.json
```

#### Step 4 — Launch EC2 Spot Instance

```bash
BUCKET="your-training-bucket"
RUN_ID="run_$(date +%Y%m%d)"
KEY_NAME="your-key-pair"       # Replace with your EC2 key pair name
SUBNET_ID="subnet-xxxxxxxx"   # Replace with a us-east-1 subnet
SG_ID="sg-xxxxxxxx"           # Replace with your security group

aws ec2 run-instances \
  --region us-east-1 \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type c5.9xlarge \
  --key-name ${KEY_NAME} \
  --security-group-ids ${SG_ID} \
  --subnet-id ${SUBNET_ID} \
  --iam-instance-profile Name=TokenizationRole \
  --block-device-mappings '[{
    "DeviceName": "/dev/xvda",
    "Ebs": {
      "VolumeSize": 200,
      "VolumeType": "gp3",
      "Iops": 3000,
      "Throughput": 125,
      "DeleteOnTermination": true
    }
  }]' \
  --instance-market-options '{
    "MarketType": "spot",
    "SpotOptions": {
      "SpotInstanceType": "one-time",
      "InstanceInterruptionBehavior": "terminate"
    }
  }' \
  --tag-specifications \
    'ResourceType=instance,Tags=[{Key=Name,Value=tokenization-run},{Key=RunId,Value='${RUN_ID}'}]' \
  --query 'Instances[0].InstanceId' \
  --output text
# Save the instance ID printed above
```

> **Tip:** If `c5.9xlarge` spot capacity is unavailable, try `c5.18xlarge` or `c5.4xlarge` as alternatives.

#### Step 5 — SSH and Run Tokenization

```bash
# Wait ~60 seconds for instance to boot, then SSH
INSTANCE_IP=$(aws ec2 describe-instances \
  --instance-ids i-xxxxxxxxxxxxxxxxx \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

ssh -i ${KEY_NAME}.pem ec2-user@${INSTANCE_IP}
```

On the instance, run the following bootstrap once:

```bash
# ---- BOOTSTRAP (run once after SSH) ----
BUCKET="your-training-bucket"
RUN_ID="run_$(date +%Y%m%d)"

# Install Python deps
sudo yum install -y python3.11 python3.11-pip tmux htop 2>/dev/null || \
  sudo apt-get install -y python3 python3-pip tmux htop 2>/dev/null || true

pip3 install numpy pandas pyarrow transformers datasets boto3 botocore tokenizers

# Download code + data
aws s3 sync s3://${BUCKET}/tokenizer-code/ ~/tokenizer/ --region us-east-1
aws s3 sync s3://${BUCKET}/coresets/1B/ ~/tokenizer/coresets/1B/ --region us-east-1
aws s3 sync s3://${BUCKET}/tsai_131k_tokenizer/ ~/tokenizer/tsai_131k_tokenizer/ --region us-east-1

cd ~/tokenizer
```

Then run the tokenizer inside tmux (protects against SSH disconnects):

```bash
# Start a tmux session so the job survives SSH disconnect
tmux new -s tokenize

# Full production run — adjust BUCKET and RUN_ID
BUCKET="your-training-bucket"
RUN_ID="run_$(date +%Y%m%d)"

python tokenize_curriculum.py \
  --coreset-uri   s3://t2-datacurriculum-353/coreset_outputs/coresets/1B \
  --dst-uri       s3://${BUCKET}/tokenized/${RUN_ID} \
  --tokenizer-path ./tsai_131k_tokenizer \
  --t1-base-uri   s3://t1-dataacquisition-datasets/processed_dataset/normalized_data \
  --block-size    4096 \
  --shard-size-mb 512 \
  --num-proc      3 \
  --file-parallelism 12 \
  --drop-remainder \
  --stage         1 \
  --tokenizer-version v1 \
  --tmp-dir       /tmp/tok_tmp \
  2>&1 | tee ~/tokenize_$(date +%Y%m%d_%H%M%S).log

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t tokenize
```

#### Step 6 — Monitor Progress

```bash
# Watch log (from another terminal)
tail -f ~/tokenize_*.log

# Check how many shards have been uploaded
aws s3 ls s3://${BUCKET}/tokenized/${RUN_ID}/ --recursive | grep metadata.json | wc -l

# Check progress state
aws s3 cp s3://${BUCKET}/tokenized/${RUN_ID}/progress_state.json - | python3 -c \
  "import json,sys; s=json.load(sys.stdin); print(f'Completed: {len(s[\"completed\"])} files')"

# Monitor CPU usage on instance
htop
```

#### Step 7 — Validate Output

After the run completes (or any time):

```bash
# Run on instance or locally after syncing outputs
python validate_shards.py \
  --shards-dir /tmp/tok_out_synced \
  --tokenizer-path ./tsai_131k_tokenizer \
  --verbose \
  2>&1 | tee ~/validation_$(date +%Y%m%d).log

# Or sync metadata only for a quick check
aws s3 sync s3://${BUCKET}/tokenized/${RUN_ID}/ /tmp/tok_out_synced/ \
  --exclude "*.bin" --include "*/metadata.json"
python validate_shards.py \
  --shards-dir /tmp/tok_out_synced \
  --tokenizer-path ./tsai_131k_tokenizer
```

#### Step 8 — Terminate Instance

```bash
# From your laptop — only after validating output
aws ec2 terminate-instances \
  --instance-ids i-xxxxxxxxxxxxxxxxx \
  --region us-east-1
```

---

### 4.2 Cost and Duration Breakdown

#### Throughput Assumptions

- TSAI 131K BPE tokenization throughput: ~3–5 million tokens/minute per vCPU
- S3 same-region read latency (EC2 → S3, us-east-1): negligible for this workload
- Each source parquet: ~50–200 MB compressed; ~5,000 unique source files total
- With 12-way file-level parallelism, 12 source files download concurrently
- S3 sustained read throughput to one instance: 2–5 GB/s for concurrent requests

**Estimated wall time (20B tokens, c5.9xlarge, 12 workers × 3 HF procs = 36 cores):**
- Tokenization: 20B tokens / (4M tokens/min × 36 cores) ≈ 140 min ≈ **~3–5 hours**
- S3 download overhead: ~500 GB source data / 2 GB/s = ~4 minutes (negligible)
- **Total: ~4 hours on c5.9xlarge**

#### Instance Options

| Instance | vCPU | RAM | `--file-parallelism` | `--num-proc` | Est. Duration | On-Demand $/hr | Spot $/hr (est.) |
|----------|------|-----|---------------------|-------------|--------------|----------------|-----------------|
| c5.xlarge | 4 | 8 GB | 1 | 4 | ~40 hrs | $0.17 | $0.05–0.07 |
| c5.4xlarge | 16 | 32 GB | 8 | 2 | ~10 hrs | $0.68 | $0.15–0.22 |
| **c5.9xlarge** | **36** | **72 GB** | **12** | **3** | **~4 hrs** | **$1.53** | **$0.35–0.55** |
| c5.18xlarge | 72 | 144 GB | 20 | 3 | ~2.5 hrs | $3.06 | $0.65–1.00 |
| r5.4xlarge | 16 | 128 GB | 8 | 2 | ~10 hrs | $1.01 | $0.22–0.35 |

> Use `r5.4xlarge` only if large source parquets (>300 MB) cause memory pressure on `c5`.

#### Total Cost Estimates (20B tokens, us-east-1)

| Component | Cost |
|-----------|------|
| c5.9xlarge Spot (~4 hrs) | $1.40–$2.20 |
| EBS gp3 200 GB (4 hrs) | $0.09 |
| S3 data transfer (same-region) | **$0.00** |
| S3 PUT requests (~500) | $0.003 |
| S3 GET requests (~5,000) | $0.002 |
| **Total compute** | **~$1.50–$2.30** |

#### Output Storage Cost

- Output shards: 20B tokens × 4 bytes = 80 GB minimum; with index files and metadata: ~90–100 GB
- S3 Standard: $0.023/GB/month → ~**$2.07–$2.30/month**
- S3 Infrequent Access (for archival): $0.0125/GB/month → **~$1.13–$1.25/month**

#### Cost Comparison Summary

| Strategy | Duration | Cost |
|----------|----------|------|
| c5.xlarge Spot (cheap, slow) | ~40 hrs | ~$2.00–$2.80 |
| c5.4xlarge Spot | ~10 hrs | ~$1.50–$2.20 |
| **c5.9xlarge Spot (recommended)** | **~4 hrs** | **~$1.50–$2.30** |
| c5.18xlarge Spot | ~2.5 hrs | ~$1.65–$2.50 |
| On-Demand c5.9xlarge (no Spot) | ~4 hrs | ~$6.12 |

**The c5.9xlarge Spot is the sweet spot**: fastest per-dollar, fits comfortably in 72 GB RAM
(12 workers × ~900 MB peak each = ~10.8 GB, leaving 61 GB headroom), and has high Spot
availability in us-east-1 during off-peak hours (Spot interruption rate < 5% for 4-hour jobs).

---

### 4.3 S3 Setup and IAM

#### IAM Policy for EC2 Instance

Save as `tokenization-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadCoresetIndex",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:HeadObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::t2-datacurriculum-353",
        "arn:aws:s3:::t2-datacurriculum-353/coreset_outputs/*"
      ]
    },
    {
      "Sid": "ReadRawTextData",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:HeadObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::t1-dataacquisition-datasets",
        "arn:aws:s3:::t1-dataacquisition-datasets/processed_dataset/normalized_data/*"
      ]
    },
    {
      "Sid": "ReadWriteTrainingBucket",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:HeadObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::your-training-bucket",
        "arn:aws:s3:::your-training-bucket/*"
      ]
    }
  ]
}
```

#### Destination Bucket Setup

```bash
BUCKET="your-training-bucket"
REGION="us-east-1"

# Create bucket
aws s3api create-bucket \
  --bucket ${BUCKET} \
  --region ${REGION}

# Block all public access
aws s3api put-public-access-block \
  --bucket ${BUCKET} \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

# Enable server-side encryption (AES-256)
aws s3api put-bucket-encryption \
  --bucket ${BUCKET} \
  --server-side-encryption-configuration '{
    "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
  }'
```

#### S3 Prefix Structure

```
s3://t2-datacurriculum-353/            (T3 coreset index — read-only source, already exists)
  coreset_outputs/coresets/1B/
    selected_indices_part_shard000_batch000000.parquet
    ...

s3://t1-dataacquisition-datasets/     (T1 raw text — read-only source, already exists)
  processed_dataset/normalized_data/
    source=C4/
      part-00759-8299c866-....parquet
      ...

s3://your-training-bucket/            (training output bucket)
  tsai_131k_tokenizer/
    tokenizer.json
    special_tokens_map.json
    tokenizer_config.json
  tokenizer-code/
    tokenize_curriculum.py
    validate_shards.py
    scripts/
  tokenized/
    run_20260303/                    ← one directory per run (timestamped)
      progress_state.json            ← interrupt/resume state (updated after each batch file)
      manifest.json                  ← global summary (written on successful completion)
      selected_indices_part_shard000_batch000000/
        shard_000/
          tokens.bin                 ← 512 MB uint32 token IDs
          tokens.idx                 ← spdl-compatible offset index
          metadata.json              ← rich metadata (see below)
        shard_001/ ...
      selected_indices_part_shard000_batch000001/
        ...
```

#### metadata.json Schema (per shard)

```json
{
  "format": "megatron_bin_idx",
  "idx_format": "spdl_v1",
  "token_dtype": "uint32",
  "bytes_per_token": 4,
  "block_size": 4096,
  "vocab_size": 131072,
  "pad_token_id": 130718,
  "eos_token_id": 130717,
  "num_blocks": 131072,
  "total_tokens": 536870912,
  "file_size_bytes": 2147483648,
  "shard_name": "shard_000",
  "tokenizer_hash": "sha256-of-tokenizer.json+special_tokens_map.json",
  "tokenizer_version": "v1",
  "band": "B0",
  "band_distribution": {"B0": 0.574, "B1": 0.187, "B2": 0.239},
  "domain": "web",
  "domain_distribution": {"web": 1.0},
  "stage": 1,
  "source_file": "coresets/1B/selected_indices_part_shard000_batch000000.parquet",
  "rows_input": 245000,
  "rows_with_eos": 244997,
  "rows_dropped": 3,
  "tokens_dropped": 8192,
  "drop_reason": "tail_truncation_at_block_boundary",
  "created_at": "2026-03-03T14:30:00Z"
}
```

---

### 4.4 Spot Instance Interrupt Handling

#### How AWS Spot Interrupts Work

1. AWS decides to reclaim the instance (capacity needed elsewhere)
2. **2-minute warning**: posted to EC2 Instance Metadata Service (IMDS)
   - `GET http://169.254.169.254/latest/meta-data/spot/termination-time`
   - Returns HTTP 200 + timestamp when termination is scheduled; HTTP 404 otherwise
3. **OS signal**: SIGTERM sent ~30 seconds before hard shutdown
4. **Hard shutdown**: instance is terminated

#### What `tokenize_curriculum.py` Does on Interruption

The updated script handles this at three layers:

**Layer 1 — IMDS polling daemon thread**
- Background thread polls IMDS every 5 seconds
- On HTTP 200 response: sets `_TERMINATION_DETECTED` threading.Event
- Processing loop checks this flag before each source file download

**Layer 2 — SIGTERM signal handler**
- `signal.signal(SIGTERM, handler)` registered at startup
- Also captures SIGINT (Ctrl+C) for local testing
- Handler sets the same `_TERMINATION_DETECTED` event

**Layer 3 — Graceful shutdown logic**
- On termination detected: the processing loop breaks before the next source file
- Partial `accumulated_blocks` in `ShardWriter` are **discarded** (not flushed)
- Only **fully uploaded shards** (with `metadata.json` on S3) are counted as complete
- `progress_state.json` on S3 records which batch files completed fully

#### Checkpoint and Resume Mechanism

The script has two levels of checkpointing:

| Level | Granularity | How it works |
|-------|-------------|-------------|
| **Shard level** | Per 512 MB shard | `flush_shard()` checks if `metadata.json` exists at S3 key; if yes, skips |
| **Batch file level** | Per coreset batch file | `progress_state.json` lists completed batch file URIs |

On resume (instance relaunched with same arguments):
1. `progress_state.json` is read from S3
2. Already-completed batch files are skipped immediately (no S3 HEAD requests)
3. In-progress batch files from the previous run are re-processed with shard-level skipping
4. Fully fresh batch files are processed normally

#### Auto-Restart with Auto Scaling Group (Optional)

For fully automated restart without manual intervention:

```bash
# 1. Create launch template
aws ec2 create-launch-template \
  --launch-template-name tokenization-lt \
  --launch-template-data '{
    "InstanceType": "c5.9xlarge",
    "ImageId": "ami-0c02fb55956c7d316",
    "IamInstanceProfile": {"Name": "TokenizationRole"},
    "KeyName": "your-key-pair",
    "BlockDeviceMappings": [{
      "DeviceName": "/dev/xvda",
      "Ebs": {"VolumeSize": 200, "VolumeType": "gp3", "Iops": 3000}
    }],
    "UserData": "'$(base64 -w 0 userdata_auto_restart.sh)'"
  }'

# 2. Create ASG with Spot + capacity-optimized strategy
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name tokenization-asg \
  --launch-template LaunchTemplateName=tokenization-lt,Version='$Latest' \
  --min-size 1 --max-size 1 --desired-capacity 1 \
  --vpc-zone-identifier "subnet-xxxxxxxx" \
  --mixed-instances-policy '{
    "InstancesDistribution": {
      "OnDemandBaseCapacity": 0,
      "OnDemandPercentageAboveBaseCapacity": 0,
      "SpotAllocationStrategy": "capacity-optimized"
    },
    "LaunchTemplate": {
      "LaunchTemplateSpecification": {
        "LaunchTemplateName": "tokenization-lt",
        "Version": "$Latest"
      },
      "Overrides": [
        {"InstanceType": "c5.9xlarge"},
        {"InstanceType": "c5.18xlarge"},
        {"InstanceType": "c5.4xlarge"}
      ]
    }
  }'

# 3. When the job is fully done, scale down to 0
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name tokenization-asg \
  --desired-capacity 0
```

**`userdata_auto_restart.sh`** — the script that runs on every instance launch:

```bash
#!/bin/bash
set -e
BUCKET="your-training-bucket"
RUN_ID="run_20260303"   # Fixed: always resume the same run

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Bootstrap starting..."

# Install deps (idempotent)
yum install -y python3.11 python3.11-pip tmux 2>/dev/null || \
  apt-get install -y python3 python3-pip tmux 2>/dev/null || true

pip3 install numpy pandas pyarrow transformers datasets boto3 botocore tokenizers

# Sync code and tokenizer (T3 and T1 data are already on S3 — no sync needed)
aws s3 sync s3://${BUCKET}/tokenizer-code/ /home/ec2-user/tokenizer/ --region us-east-1
aws s3 sync s3://${BUCKET}/tsai_131k_tokenizer/ /home/ec2-user/tokenizer/tsai_131k_tokenizer/ --region us-east-1
# Alternative: clone tokenizer files from GitHub instead of S3 if preferred

cd /home/ec2-user/tokenizer

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting tokenization (will auto-resume via progress_state.json)..."

python tokenize_curriculum.py \
  --coreset-uri   s3://t2-datacurriculum-353/coreset_outputs/coresets/1B \
  --dst-uri       s3://${BUCKET}/tokenized/${RUN_ID} \
  --tokenizer-path ./tsai_131k_tokenizer \
  --t1-base-uri   s3://t1-dataacquisition-datasets/processed_dataset/normalized_data \
  --block-size    4096 \
  --shard-size-mb 512 \
  --num-proc      3 \
  --file-parallelism 12 \
  --drop-remainder \
  --stage         1 \
  --tokenizer-version v1 \
  --tmp-dir       /tmp/tok_tmp \
  2>&1 | tee /home/ec2-user/tokenize_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Tokenization COMPLETE. Self-terminating instance."
  # Upload logs to S3
  aws s3 cp /home/ec2-user/tokenize_*.log s3://${BUCKET}/tokenized/${RUN_ID}/logs/ --region us-east-1
  # Self-terminate to stop billing
  INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
  aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} --region us-east-1
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Tokenization ended with exit code ${EXIT_CODE}. Instance kept for investigation."
  # Do NOT self-terminate on failure — allows SSH investigation
fi
```

> **Important:** The script does NOT self-terminate on failure (non-zero exit). This allows you to
> SSH in and investigate. Failed instances will continue billing until you terminate them manually.

#### Manual Resume After Spot Interruption

If using a single instance (no ASG) and the instance was interrupted:

```bash
# 1. Launch a new instance (same configuration as Step 4)
# 2. SSH in and run the exact same tokenize_curriculum.py command
#    with the SAME --dst-uri arguments
# 3. The script will automatically:
#    a. Read progress_state.json from S3 → skip completed batch files
#    b. For in-progress batch files: check shard metadata.json → skip complete shards
#    c. Resume from the last checkpoint

# Check what was completed before the interruption
aws s3 cp s3://${BUCKET}/tokenized/${RUN_ID}/progress_state.json - | \
  python3 -c "import json,sys; s=json.load(sys.stdin); \
  print(f'Completed: {len(s[\"completed\"])} / total batch files')"
```

#### Estimated Cost Impact of Interruptions

A Spot interruption at the midpoint of a 4-hour run on c5.9xlarge:
- Work lost: at most 1 shard (512 MB) worth of tokens — the last partially-accumulated shard
- Resume overhead: ~2–3 minutes to relaunch and re-download code/data from S3
- Extra cost for re-run: one additional Spot-hour × $0.45 = ~$0.45

For a 4-hour job at <5% interruption probability, expected extra cost is:
- 5% × $0.45 = **~$0.02 expected extra cost per run**

---

## Quick Reference

### Recommended Command (Production Run)

```bash
python tokenize_curriculum.py \
  --coreset-uri   s3://t2-datacurriculum-353/coreset_outputs/coresets/1B \
  --dst-uri       s3://your-training-bucket/tokenized/run_20260304 \
  --tokenizer-path ./tsai_131k_tokenizer \
  --t1-base-uri   s3://t1-dataacquisition-datasets/processed_dataset/normalized_data \
  --block-size    4096 \
  --shard-size-mb 512 \
  --num-proc      3 \
  --file-parallelism 12 \
  --drop-remainder \
  --stage         1 \
  --tokenizer-version v1 \
  --tmp-dir       /tmp/tok_tmp
```

### Validation Command (After Run)

```bash
python validate_shards.py \
  --shards-dir /tmp/synced_output \
  --tokenizer-path ./tsai_131k_tokenizer \
  --verbose
```

### Cost Summary

| What | Value |
|------|-------|
| Recommended instance | c5.9xlarge Spot, us-east-1 |
| Estimated duration | ~4 hours (20B tokens, 12 workers) |
| Estimated compute cost | **~$1.50–$2.30** |
| S3 transfer cost | **$0.00** (same-region) |
| Output storage | ~$2.07–$2.30/month (S3 Standard) |
| Spot interruption risk | <5% for 4-hour job in us-east-1 |
| Cost of interruption | ~$0.02 expected extra |
