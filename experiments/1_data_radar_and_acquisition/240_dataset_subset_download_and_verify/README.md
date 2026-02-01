# Dataset subset download and verification (Task #240)

Lightweight, config-driven tooling for **preflight validation of large text datasets**
before full ingestion or upload to S3.

This module focuses on **downloading small subsets**, validating dataset configs/splits,
and verifying that outputs are correctly formatted and parseable.

> Note: This is **not a replacement** for the full dataset ingestion and S3 upload
pipeline implemented under **Task #160**. It is intended as an upstream sanity-check
step.

---

## Supported datasets

- **Sangraha (AI4Bharat)**
  - Config: `verified`
  - Language splits (e.g. `hin`, `eng`)
- **IndicCorpV2 (AI4Bharat)**
  - Language-script splits (e.g. `hin_Deva`, `ben_Beng`)
- **Dolma (AllenAI)**
  - Optional shard-based download (disabled by default due to large shard sizes)

---

## What this does

1. Downloads **small subsets** of datasets using streaming or shard URLs
2. Writes outputs as **JSONL** under a local `out/` directory
3. Generates a **manifest** with counts and schema hints
4. Verifies:
   - files exist
   - record counts match expectations
   - JSONL is parseable

This enables early detection of:
- incorrect dataset configs
- invalid splits
- schema or formatting issues
- broken download paths

---

## How to run

```bash
cd experiments/1_data_radar_and_acquisition/240_dataset_subset_download_and_verify
pip install -r requirements.txt
python src/sample_download.py
python src/sample_verify.py
