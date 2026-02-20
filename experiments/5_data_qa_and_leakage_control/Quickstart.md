# Quick Start Guide

## 60-Second Setup
```bash
# 1. Navigate to scanner project
cd /home/ubuntu/LLM/experiments/5_data_qa_and_leakage_control/collected

# 2. Install dependencies
pip install -r requirements.txt

# 3. If benchmarks are missing, download once
python scripts/download_benchmarks.py
cp benchmark_registry/*_test.jsonl benchmarks/

# 4. Scan your data
python scripts/scan.py your_data.jsonl "Team Name" "Batch ID"

# 5. Check output
# ✅ APPROVED = Safe to use in training
# ❌ REJECTED = Contains benchmark contamination
```

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
- **Format questions:** See `tests/realistic_10k.jsonl` for example
- **Team support:** Slack #team5-data-qa

---

**That's it! Scanner is ready to protect your training pipeline.**
