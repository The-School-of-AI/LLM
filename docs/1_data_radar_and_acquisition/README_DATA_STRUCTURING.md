**Dataset Structuring**

**Pretraining Data Layout on AWS (S3)**

**Purpose**

This document defines the **standardized dataset structure** for all pretraining, SFT, alignment, and post-training data stored on AWS.

The goals are to ensure:

- Deterministic, reproducible training runs
- Clear separation between raw, processed, and training-ready data
- License and provenance traceability
- Easy deduplication, filtering, and reweighting
- Minimal risk of benchmark contamination
- Compatibility with distributed training pipelines

Once data is ingested into this structure, it is considered **eligible for irreversible pretraining use**.

**High-Level Principles**

- **Immutability by stage**
  - Raw data is never modified
  - Each processing step writes to a new prefix
- **One document = one file (after sharding)**
  - Avoid multi-document blobs
  - Enables deduplication, filtering, and removal
- **Separation of concerns**
  - Raw ingestion ≠ normalized text ≠ training shards
- **Traceability**
  - Every file must be traceable to:
    - source dataset
    - license
    - processing version

**S3 Bucket Layout**

s3://&lt;org&gt;-pretraining-data/

│

├── 00_raw/

├── 01_normalized/

├── 02_filtered/

├── 03_deduplicated/

├── 04_training_ready/

├── 05_eval_excluded/

├── 06_sft/

├── 07_alignment/

└── metadata/

**Directory Semantics**

**00_raw/ - Source Preservation (Read-Only)**

**Purpose:**  
Store datasets exactly as obtained from the original source.

**Rules:**

- No edits, no reformatting
- Compressed if necessary
- One subfolder per dataset

00_raw/

└── pile_cc/

├── original_dump/

└── LICENSE.txt

**01_normalized/ - Canonical Text Format**

**Purpose:**  
Convert all sources into a unified document format.

**Normalization rules:**

- UTF-8 encoding
- Plain text or JSONL only
- One logical document per record
- Strip HTML / markup
- Preserve paragraph boundaries
- No deduplication yet

01_normalized/

└── pile_cc/

├── shard_00000.jsonl

├── shard_00001.jsonl

└── normalization_manifest.json

**JSONL Schema (Required)**

{

"doc_id": "uuid-v4",

"source": "pile_cc",

"subset": "commoncrawl",

"text": "...",

"language": "en",

"timestamp": "2019-06-01",

"license": "CC-BY-4.0",

"metadata": {

"url": "...",

"domain": "news"

}

}

**02_filtered/ - Quality & Safety Filtering**

**Purpose:**  
Remove low-signal, unsafe, or out-of-scope content.

**Filtering includes:**

- Language detection
- Minimum length thresholds
- Boilerplate removal
- Profanity / spam heuristics
- Known synthetic patterns

02_filtered/

└── pile_cc/

├── shard_00000.jsonl

├── shard_00001.jsonl

└── filter_report.json

**03_deduplicated/ - Global Deduplication**

**Purpose:**  
Prevent overfitting and memorization at scale.

**Deduplication scope:**

- Intra-dataset
- Cross-dataset
- Across all domains

**Methods:**

- MinHash / SimHash
- Exact hash for short docs
- Thresholds documented per run

03_deduplicated/

└── global/

├── shard_00000.jsonl

├── shard_00001.jsonl

└── dedup_stats.json

**04_training_ready/ - Final Shards**

**Purpose:**  
Training-consumable, weighted, shuffled data.

**Properties:**

- Sharded by token count (not document count)
- Mixed across datasets per curriculum
- Benchmark-contaminated sources excluded

04_training_ready/

└── pretrain_v1/

├── shard_00000.tar

├── shard_00001.tar

└── dataset_manifest.json

**Manifest Example**

{

"version": "pretrain_v1",

"total_tokens": 2.1e12,

"effective_tokens": 1.4e12,

"domain_mix": {

"web": 45,

"code": 20,

"math": 10,

"books": 15,

"multilingual": 10

}

}

**05_eval_excluded/ - Quarantined Sources**

**Purpose:**  
Store data explicitly excluded due to benchmark contamination risk.

**Examples:**

- Benchmark datasets
- Leaked evaluation sets
- Model answers
- Instruction datasets used for testing

05_eval_excluded/

└── benchmarks/

├── gsm8k/

├── mmlu/

└── human_eval/

**Never mixed back into pretraining.**

**06_sft/ - Supervised Fine-Tuning Data**

06_sft/

└── instruction_following/

├── train.jsonl

├── validation.jsonl

└── LICENSE.txt

Schema includes:

- prompt
- response
- source
- annotator or generation method

**07_alignment/ - RLHF / Preference Data**

07_alignment/

└── preferences/

├── comparisons.jsonl

├── reward_model/

└── README.md

Strict separation from pretraining data.

**File Naming Conventions**

**Shards**

shard_&lt;5-digit-zero-padded&gt;.&lt;ext&gt;

Example:

shard_00042.jsonl

**Versions**

- Semantic versions preferred: pretrain_v1, pretrain_v1.1
- Never overwrite a versioned directory

**Metadata Directory**

metadata/

├── dataset_registry.json

├── license_registry.json

├── processing_versions.json

└── benchmark_blocklist.json

This directory is **authoritative** for:

- Dataset inclusion decisions
- License compliance
- Audit trails

**Operational Rules (Hard Requirements)**

- ❌ No training job may read from 00_raw, 01_normalized, or 02_filtered
- ❌ No benchmark data may exist outside 05_eval_excluded
- ❌ No dataset without a verified license enters 03_deduplicated
- ✅ All training runs reference a **single immutable manifest**

**Common Failure Modes Prevented by This Structure**

| **Risk** | **Mitigation** |
| --- | --- |
| Benchmark leakage | Physical directory isolation |
| License violations | Explicit license metadata |
| Overfitting | Global deduplication stage |
| Irreproducible runs | Versioned manifests |
| Dataset confusion | Strict naming conventions |

**Ownership & Change Control**

- Dataset structure changes require **cross-team approval**
- Once data enters 04_training_ready, changes require escalation
- This structure defines the **irreversibility point** of the pipeline
