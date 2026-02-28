# Quick Start Guide

## 60-Second Setup
```bash
# 1. Navigate to scanner project
cd collected/

# 2. Install dependencies  (uv required — install: curl -LsSf https://astral.sh/uv/install.sh | sh)
uv sync

# 3. If benchmarks are missing, download once
uv run python scripts/download_benchmarks.py
# Optional: write to a custom location
# uv run python scripts/download_benchmarks.py --output-dir /data/benchmarks

# 4. Scan your data
uv run python scripts/scan.py your_data.jsonl "Team Name" "Batch ID"

# 5. Check output
# ✅ APPROVED = Safe to use in training
# ❌ REJECTED = Contains benchmark contamination
```

## S3 One-Command Flow (Recommended for `.txt` in S3)
```bash
# 1. Configure S3 path and run settings in collected/config.json

# 2. Run full 3-layer scan (N-gram + MinHash + Semantic)
uv run python scripts/run.py

# 3. (Optional) No-semantic mode for faster runs:
# In config.json set: "enable_semantic": false
# Then run the same command:
uv run python scripts/run.py
```

What `scripts/run.py` does automatically:
- Reads config from `config.json`
- Streams `.txt` from S3 and runs contamination scan
- Runs either 3 layers or 2 layers depending on `enable_semantic`
- Writes reports locally to `reports/`

## Required Input Format
```jsonl
{"text": "Your training sample here"}
{"text": "Another training sample"}
```

## What Happens Next

**If APPROVED:**
- Proceed to training pipeline
- Attach report to submission

**If REJECTED:**
- Check `reports/*_CONTAMINATED_*.jsonl`
- Remove flagged samples
- Re-scan until approved

## Need Help?

- **Setup issues:** Check `README.md` prerequisites
- **Format questions:** Each line must be `{"text": "your sample here"}`
- **S3 flow:** See `collected/README.md`
- **Team support:** Slack #team5-data-qa

---

**That's it! Scanner is ready to protect your training pipeline.**
