## 🎯 Mission

Prevent benchmark contamination in training data. Contaminated models appear artificially strong on evaluations but fail in real-world use, damaging credibility and invalidating research.

**Our Job:** Ensure zero benchmark leakage reaches the 70B parameter model training pipeline.

---

## ✅ Start Here (This Repo)

In this repository, the runnable scanner project is under:

`experiments/5_data_qa_and_leakage_control/collected`

Use this exact flow:

```bash
cd experiments/5_data_qa_and_leakage_control/collected
uv sync
uv run python scripts/scan.py group4.jsonl "Team 4" "group4_batch_01"
```

For S3 input, use:

```bash
cd experiments/5_data_qa_and_leakage_control/collected
uv run python scripts/run.py
```

And in `collected/config.json`:
- `"enable_semantic": true` for full 3-layer scan
- `"enable_semantic": false` for N-gram + MinHash only

If benchmarks are missing, run:

```bash
uv run python scripts/download_benchmarks.py
```

---

## 🏗️ System Architecture


## Workflow Overview

```
┌─────────────────────────────────────────┐
│         TEAM SUBMITS DATA               │
│    (Team 4, 17, 3, etc.)               │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│     SYSTEM 1: SCANNER                   │
│  • Load data                            │
│  • Run 13-gram check (exact)            │
│  • Run MinHash check (near-duplicate)   │
│  • Run Semantic check (paraphrase)      │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
    CLEAN         CONTAMINATED
        │             │
        ↓             ↓
   APPROVED      REJECTED
        │             │
        │             └─→ Report to Team → Fix → Resubmit
        │
        ↓
┌─────────────────────────────────────────┐
│     GOES TO TRAINING                    │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│     SYSTEM 2: MONITOR                   │
│  • Watch benchmark scores               │
│  • Every checkpoint                     │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
    NORMAL        SPIKE DETECTED
        │             │
        ↓             ↓
   Continue       🚨 ALERT
        │             │
        │             ↓
        │      ┌─────────────────────────────┐
        │      │  SYSTEM 3: INVESTIGATE      │
        │      │  • Find time window         │
        │      │  • Identify data batch      │
        │      │  • Re-scan batch            │
        │      │  • Find contamination       │
        │      └──────────┬──────────────────┘
        │                 │
        │                 ↓
        │      ┌─────────────────────────────┐
        │      │  REMEDIATE                  │
        │      │  • Remove bad data          │
        │      │  • Rollback checkpoint      │
        │      │  • Document incident        │
        │      │  • Resume training          │
        │      └──────────┬──────────────────┘
        │                 │
        └────────←────────┘
               │
               ↓
        Training Complete
               │
               ↓
    No contamination scandals! 🎉

### Three-System Strategy



┌──────────────────────────────────────────────────────────────┐
│  SYSTEM 1: PRE-FLIGHT SCANNER                                │
│  Status: ✅ PRODUCTION READY                                 │
│                                                               │
│  What: Scans all incoming data before it enters training     │
│  How:  3-layer detection (N-gram + MinHash + Semantic)       │
│  Output: APPROVED ✅ or REJECTED ❌                           │
│  Coverage: ~95% detection with low false positive rate       │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  SYSTEM 2: TRAINING MONITOR                                  │
│  Status: 🔨 PLANNED                                          │
│                                                               │
│  What: Watches for contamination during training             │
│  How:  Validates benchmarks at each checkpoint               │
│  Triggers: Anomaly detection (unusual score spikes)          │
│  Action: Pause training → Alert Team 5 → Investigate         │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  SYSTEM 3: FORENSIC INVESTIGATION                            │
│  Status: 📋 PLANNED                                          │
│                                                               │
│  What: Traces contamination to source when detected          │
│  How:  Audit trail analysis, batch tracking                  │
│  Output: Root cause, remediation plan, incident report       │
│  Action: Remove bad data, coordinate re-training             │
└──────────────────────────────────────────────────────────────┘
```

---

## 📊 Current Status

### ✅ Completed (System 1)

- **3-layer detection engine:**
  - Layer 1: N-gram (13-word exact matching)
  - Layer 2: MinHash with word bigrams + LSH false-positive filtering (real Jaccard scores, threshold 0.8)
  - Layer 3: Semantic similarity via MiniLM + FAISS (cosine similarity, threshold 0.9)
- **Benchmark Registry:** 26 benchmarks indexed (~850k+ questions)
  - English: MMLU, MMLU-Pro, TriviaQA, TruthfulQA, GPQA Diamond, ARC-Challenge, BoolQ, HellaSwag, Winogrande, GSM8K, MATH, HumanEval, APPS, AIME 2025/2026-I, IFEval, SimpleQA Verified, BBH, SWE-bench Verified, ToolBench, L-Eval
  - Indic: MMLU-Indic, IndicGLUE, IndicQA, IndicGenBench, IndicMTEval
- **Confidence scores:** Real computed values (Jaccard / cosine), not hardcoded labels
- **Production Pipeline:** Single-command scanning with detailed per-layer reports
- **Download script:** Handles multi-config benchmarks (BBH 27 tasks, MATH 7 subjects), clear failure summary

### 🔨 In Progress

- System 2 (Training Monitor) design & implementation

### 📋 Backlog

- System 3 (Forensic Investigation)
- S3 direct integration
- Parallel processing for 1M+ sample datasets
- Web dashboard for report visualization

---

## 📁 Repository Structure
```
collected/
│
├── core/                       # Detection engine
│   ├── __init__.py
│   ├── utils.py                # Shared text normalisation
│   ├── registry.py             # Benchmark loader
│   ├── detectors.py            # N-gram + MinHash + Semantic detectors
│   └── scanner.py              # Main scanning orchestrator
│
├── benchmarks/                 # Protected test sets (DO NOT MODIFY)
│   │
│   │  ── English ──
│   ├── mmlu_test.jsonl              # 14,042 questions
│   ├── mmlu_pro_test.jsonl          # 12,032 questions
│   ├── triviaqa_test.jsonl          # 17,944 questions
│   ├── truthfulqa_test.jsonl        # 817 questions
│   ├── gpqa_diamond_test.jsonl      # 198 questions (gated, needs HF_TOKEN)
│   ├── arc_challenge_test.jsonl     # 1,172 questions
│   ├── boolq_test.jsonl             # 3,270 questions
│   ├── hellaswag_test.jsonl         # 10,042 questions
│   ├── winogrande_test.jsonl        # 1,267 questions
│   ├── bbh_test.jsonl               # 6,511 questions (27 tasks)
│   ├── gsm8k_test.jsonl             # 1,319 questions
│   ├── math_test.jsonl              # 5,000 questions (7 subjects)
│   ├── humaneval_test.jsonl         # 164 coding problems
│   ├── apps_test.jsonl              # 5,000 programming problems
│   ├── aime_2025_test.jsonl         # 30 problems
│   ├── aime_2026_I_test.jsonl       # 15 problems
│   ├── simpleqa_verified_test.jsonl # 1,000 questions
│   ├── ifeval_test.jsonl            # 541 questions
│   ├── swe_bench_verified_test.jsonl# 500 tasks
│   ├── toolbench_test.jsonl         # ~5,000 tool-use instructions
│   ├── leval_test.jsonl             # 2,043 long-context questions (18 subtasks)
│   │
│   │  ── Indic ──
│   ├── mmlu_indic_test.jsonl        # ~293,000 questions (11 langs + romanised)
│   ├── indicglue_test.jsonl         # ~300,000 samples (NLI/QA/NER, 11 langs)
│   ├── indicqa_test.jsonl           # ~9,571 questions (11 languages)
│   ├── indicgenbench_test.jsonl     # ~142,000 translation pairs (29 languages)
│   └── indicmteval_test.jsonl       # ~14,000 MT annotations (hi/ta/mr/ml/gu)
│
├── scripts/                    # CLI tools
│   ├── run.py                  # S3 one-command runner (reads config.json)
│   ├── scan_from_s3.py         # Direct S3 scanner entry point
│   ├── scan.py                 # Local JSONL scanner entry point
│   ├── scan_no_semantic.py     # Local JSONL scan (N-gram + MinHash only)
│   ├── replay.py               # Replay metadata / rerun a past scan
│   ├── download_benchmarks.py
│   └── convert_txt.py
│
├── config.json                 # S3/team/batch settings for run.py
├── aws.json                    # Optional AWS creds (gitignored)
├── .cache/indexes/             # Auto-generated index caches (fingerprinted)
│
├── reports/                    # Scan outputs (auto-generated)
│   ├── *.json                  # Main reports
│   ├── *_CONTAMINATED_*.jsonl  # Lists of flagged samples
│   └── run_registry.jsonl      # Permanent run audit trail
│
├── pyproject.toml              # Project/dependency metadata
├── uv.lock                     # Locked dependency resolution
└── README.md                   # Local run instructions
```


---

## 🔬 How Detection Works

### Layer 1: N-Gram Exact Matching

**Method:** Extracts 13-word sequences, checks for exact matches against all benchmarks.

```
Benchmark: "What is the capital city of France?"
Training:  "What is the capital city of France?"
Result:    ❌ EXACT MATCH
Confidence: 100%
```

**Catches:** Verbatim copying, copy-paste errors.

### Layer 2: MinHash Near-Duplicate Detection

**Method:** Computes word bigram fingerprints via MinHash LSH. Only reports matches where the real Jaccard similarity (computed after LSH candidate retrieval) is ≥ 0.8.

```
Benchmark: "What is the capital of France?"
Training:  "What's France's capital city?"
Result:    ❌ NEAR-DUPLICATE (82% Jaccard)
Confidence: 82%
```

**Catches:** Light rewording, partial matches, near-identical phrasing.

> Note: LSH is approximate — it returns candidates, then exact Jaccard filters out false positives below threshold. This is why confidence values are real numbers, not the old hardcoded "60-80%" label.

### Layer 3: Semantic Similarity (MiniLM + FAISS)

**Method:** Embeds all benchmark questions with `all-MiniLM-L6-v2`, builds a FAISS cosine index, scans training data in batches of 512. Reports matches with cosine ≥ 0.9.

```
Benchmark: "At what temperature does water boil?"
Training:  "Water boils at 100 degrees Celsius."
Result:    ❌ SEMANTIC MATCH (91% cosine)
Confidence: 91%
```

**Catches:** Paraphrased questions, restructured sentences.

**Memory:** Processes in batches — safe for 400k+ sample datasets on 16GB RAM.

### Priority

Each layer only flags samples **not already caught** by a stricter layer above it. N-GRAM > MINHASH > SEMANTIC.

---

## 🚀 Quick Start

### Installation
```bash
cd experiments/5_data_qa_and_leakage_control/collected
uv sync
```

### Basic Usage
```bash
uv run python scripts/scan.py <input_file> <team_name> <batch_name>

# Example
uv run python scripts/scan.py group4.jsonl "Team 4" "Batch_001"
```

### Input Format

JSONL with a `text` field per row:
```jsonl
{"id": "001", "text": "Sample training text here..."}
{"id": "002", "text": "Another training sample..."}
```

### Output

**Terminal:**
```
============================================================
✅ APPROVED
Contamination: 0/10000 (0.00%)
============================================================
```

**Files generated:**
- `reports/Batch_001_<timestamp>.json` — full report with findings per layer
- `reports/Batch_001_CONTAMINATED_<timestamp>.jsonl` — one line per flagged sample (if any)

**Exit codes:**
- `0` = APPROVED (safe for training)
- `1` = REJECTED (contaminated, do not use)

---

## 📋 Team Workflows

### For Data Teams (1, 3, 4, 17)

**Before submitting data for training:**

1. Prepare JSONL file with your training data
2. Run scanner:
```bash
uv run python scripts/scan.py your_data.jsonl "Team X" "Description"
```
3. Check result:
   - ✅ APPROVED → Submit to training pipeline
   - ❌ REJECTED → Review `reports/*_CONTAMINATED_*.jsonl`, remove flagged samples, rescan
4. Include report with your data submission

### For Team 5 (Data QA)

**Daily:**
- Scan all incoming data batches
- Review rejected batches
- Coordinate with teams on remediation
- Maintain audit trail

**Weekly:**
- Validate scanner on new test cases
- Update benchmarks if new evaluations released
- Generate contamination statistics

---

## 📈 Scaling

### With Semantic Layer (16GB RAM, 400k samples)

| Phase | Time |
|---|---|
| Build benchmark index (~850k vectors across 26 benchmarks) | ~8-12 min |
| Embed + scan 400k samples | ~15-20 min |
| Total | ~25-30 min |

Memory usage peaks at ~2-3GB for the full benchmark index (FAISS flat index, batch processing).

> For English-only scanning (no Indic benchmarks), index build is ~2 min (~70k vectors).

### For Larger Datasets

```bash
# Split and scan in parallel
split -l 100000 large_file.jsonl chunk_
for chunk in chunk_*; do
    uv run python scripts/scan.py "$chunk" "Team X" "$(basename $chunk)"
done
```

---

## 🛡️ Protected Benchmarks

Currently scanning against **26 benchmarks** (~850k+ questions):

### English Benchmarks

| Benchmark | Domain | Samples | Notes |
|---|---|---|---|
| MMLU | General knowledge (57 subjects) | 14,042 | |
| MMLU-Pro | General knowledge (harder, 10-choice) | 12,032 | |
| TriviaQA | Factual / trivia | 17,944 | |
| TruthfulQA | Factual accuracy | 817 | |
| GPQA Diamond | Expert-level science (PhD) | 198 | Requires HF_TOKEN (gated) |
| ARC-Challenge | Science / school knowledge | 1,172 | |
| BoolQ | Yes/no factual | 3,270 | |
| HellaSwag | Commonsense completion | 10,042 | |
| Winogrande | Commonsense reasoning | 1,267 | |
| BBH | Mixed reasoning (27 tasks) | 6,511 | |
| GSM8K | Math word problems | 1,319 | |
| MATH | Advanced mathematics (7 subjects) | 5,000 | |
| HumanEval | Code generation | 164 | |
| APPS | Competitive programming | 5,000 | |
| AIME 2025 | Competition math | 30 | |
| AIME 2026-I | Competition math | 15 | |
| SimpleQA Verified | Factual short-answer | 1,000 | |
| IFEval | Instruction following | 541 | |
| SWE-bench Verified | Software engineering tasks | 500 | |
| ToolBench | Multi-step tool use | ~5,000 | 6 configs |
| L-Eval | Long-context (18 subtasks) | 2,043 | |

### Indic Benchmarks

| Benchmark | Domain | Samples | Languages |
|---|---|---|---|
| MMLU-Indic | General knowledge (translated) | ~293,000 | 11 languages + romanised |
| IndicGLUE | NLI, QA, NER, classification | ~300,000 | 11 languages |
| IndicQA | Reading comprehension | ~9,571 | 11 languages |
| IndicGenBench | Translation (flores + crosssum) | ~142,000 | 29 languages |
| IndicMTEval | MT quality / MQM annotations | ~14,000 | hi, ta, mr, ml, gu |

### Not Available

| Benchmark | Reason |
|---|---|
| PIQA | HF deprecated dataset loading scripts (piqa.py) |
| ARC-C-IN | Not found on HF Hub or GitHub |
| RULER | Synthetic generator — must run locally (github.com/NVIDIA/RULER) |
| AIME 2026-II | Not yet published on HF Hub (monitor huggingface.co/MathArena) |

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
uv sync
```

### "FileNotFoundError: benchmarks/mmlu_test.jsonl"
```bash
uv run python scripts/download_benchmarks.py
```

### Semantic detector disabled warning
```bash
uv sync
```
Scanner runs fine without these — falls back to N-gram + MinHash only.

### Out of Memory
```bash
split -l 50000 large_file.jsonl chunk_
for chunk in chunk_*; do
    uv run python scripts/scan.py "$chunk" "Team X" "$(basename $chunk)"
done
```

---

## 📞 Support

**Team 5 - Data QA & Leakage Prevention**

- **Slack:** #team5-data-qa
- **Issues:** GitHub Issues or project tracker
- **Urgent:** Page on-call for production contamination

---

## 🔮 Roadmap

### Phase 1: System 1 ✅ Complete
- [x] N-gram exact matching (13-word)
- [x] MinHash near-duplicate detection with word bigrams
- [x] LSH false-positive filtering with real Jaccard scores
- [x] Semantic layer (MiniLM + FAISS, batch processing)
- [x] 26 benchmarks indexed (~850k+ questions, English + Indic)
- [x] Confidence scores are real computed values
- [x] Full type hints and docstrings across all modules
- [x] Configurable reports path and sample limit
- [x] Benchmarks path validation with actionable error messages
- [x] `--output-dir` CLI flag for download script
- [x] Git commit hash embedded in every report
- [x] Unique `run_id` per scan, registered with config + input file before detection starts
- [x] Failure classification (`INVALID_INPUT` / `OUT_OF_MEMORY` / `UNEXPECTED_ERROR`)
- [x] Replay script — reconstruct exact command from any past `run_id`
- [x] GitHub Actions CI gate on scanner code changes

### Phase 2: System 2 (Training Monitor)
- [ ] Checkpoint evaluation framework
- [ ] Anomaly detection algorithms
- [ ] Automated pause triggers
- [ ] Alert notification system

### Phase 3: System 3 (Forensics)
- [ ] Batch tracking and audit trails
- [ ] Root cause analysis tools
- [ ] Automated remediation workflows
- [ ] Incident reporting templates

### Phase 4: Production Hardening
- [ ] Web dashboard
- [ ] API endpoints
- [ ] S3 direct integration
- [ ] Automated CI/CD integration

---

## 📚 Technical References

- **GPT-4 Technical Report:** N-gram contamination methodology
- **Llama-2 Paper:** Benchmark contamination analysis
- **BigCode/BigScience:** text-dedup library
- **EleutherAI:** Large-scale deduplication (The Pile)



## Results with group4.synth data
## check results in reports folder.

![alt text](image.png)

![alt text](image-1.png)

![alt text](image-2.png)

![alt text](image-3.png)
