# Quick Start Guide

## 60-Second Setup
```bash
# 1. Navigate to scanner project
cd collected/

# 2. Install dependencies  (uv required — install: curl -LsSf https://astral.sh/uv/install.sh | sh)
uv sync --no-install-project

# 3. If benchmarks are missing, download once
.venv/bin/python scripts/download_benchmarks.py
# Optional: write to a custom location
# .venv/bin/python scripts/download_benchmarks.py --output-dir /data/benchmarks

# 4. Scan your data
.venv/bin/python scripts/scan.py your_data.jsonl "Team Name" "Batch ID"

# 5. Check output
# ✅ APPROVED = Safe to use in training
# ❌ REJECTED = Contains benchmark contamination
```

## S3 One-Command Flow (Recommended for `.txt` in S3)
```bash
# 1. Configure S3 path and run settings in collected/S3_scan.json

# 2. Run the S3 flow (auto-downloads benchmarks if missing)
.venv/bin/python scripts/S3_run.py

# Optional: force CPU mode
.venv/bin/python scripts/S3_run.py --cpu
# or
S3_FORCE_CPU=1 .venv/bin/python scripts/S3_run.py
```

What `scripts/S3_run.py` does automatically:
- Detects GPU availability
- Installs semantic dependencies on first run
- Uses FAISS GPU on CUDA hosts when available, else falls back to CPU FAISS
- Streams `.txt` from S3 and runs contamination scan
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
- **S3 flow:** See `collected/S3_run.md`
- **Team support:** Slack #team5-data-qa

---

**That's it! Scanner is ready to protect your training pipeline.**
