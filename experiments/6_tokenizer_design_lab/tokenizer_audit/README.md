# Tokenizer Audit

Audit scripts and reports for verifying `tsai_131k_tokenizer` against all supervisor requirements.

## Files

| File | Description |
|------|-------------|
| `audit_tokenizer.py` | Main audit — checks vocab size, special tokens, Indic coverage, blocked scripts, code optimization, garbage tokens, ID layout |
| `audit_deep.py` | Deep audit — cross-file consistency, ID integrity, foreign-language token scan, build script checks |
| `AUDIT_REPORT.txt` | Raw output from main audit |
| `TOKENIZER_AUDIT_WALKTHROUGH.md` | Full report with tables, changelog, and open items |

## Running

```bash
cd tokenizer_audit

# Main audit (87 checks)
python audit_tokenizer.py

# Deep audit (cross-file consistency + build script)
python audit_deep.py
```

> **Note:** Both scripts expect `../tsai_131k_tokenizer/` and `../build_clean_tokenizer.py` to exist relative to this folder.

## Latest Results

- **Main audit:** 87 PASS, 0 FAIL, 0 WARN
- **Deep audit:** 0 issues
