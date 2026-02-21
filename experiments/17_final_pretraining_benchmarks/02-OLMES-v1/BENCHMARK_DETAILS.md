# OLMES Benchmark Pipeline Documentation

This document provides a comprehensive breakdown of all benchmarks and tasks executed across training phases for both the **Developer Pipeline** (`benchmark-config.yaml`) and the **Industry Pipeline** (`industry-benchmarks.yaml`).

---

## 📈 1. Industry Pipeline Progression (industry-benchmarks.yaml)

| Tier | Stage | Main Objective | Key Tasks |
| :--- | :--- | :--- | :--- |
| 🥉 **Tier 1** | `pretrain_small` | Fast signal for 1B–3B models | `mmlu`, `triviaqa`, `arc_challenge`, `blimp`, `indic_glue` |
| 🥈 **Tier 2** | `pretrain_8b` | Standard industry milestone | + `mmlu_pro`, `gsm8k`, `minerva_math`, `truthfulqa`, RULER |
| 🥇 **Tier 3** | `pretrain_70b` | Full frontier sweep | + `bbh:cot`, `gpqa_diamond`, LongBench, disabled: `apps`, `aime_2025`, `simpleqa`, `l_eval` |
| 🏅 **Tier 4** | `sft` | Alignment & instruction | `ifeval`, `olmo3:adapt`, `truthfulqa`, `indic_glue`, `indic_qa` |

## 📈 2. Developer Pipeline Progression (benchmark-config.yaml)

| Phase | Stage | Main Objective | Key Tasks |
| :--- | :--- | :--- | :--- |
| **CI** | `pretrain_1b` | Frequent checkpoint signal | `olmo3:base_easy` tasks, `arc_challenge`, `blimp`, `indic_glue`, `niah_4k` |
| **Nightly** | `pretrain_3b` | Enhanced breadth | + `mmlu`, `triviaqa`, `gsm8k`, `truthfulqa`, `indic_qa` |
| **Milestone** | `pretrain_8b` | Full base suite | + `mmlu_pro`, `minerva_math`, RULER probes |
| **Milestone** | `pretrain_70b` | Reasoning readiness | + `bbh:cot`, `gpqa_diamond`, RULER, disabled: `aime_2025`, `apps`, `msgs`, `l_eval` |
| **Release** | `sft` | Instruction + agentic | `ifeval`, `olmo3:adapt`, `truthfulqa`, regression guard |
| **CI** | `ci_breadth` | Fast breadth sample | Sampled 2 examples from every capability bucket |

---

## 🗂 3. Benchmark → OLMES Task Mapping

| Benchmark Name | OLMES Task Name | Engine | Notes |
| :--- | :--- | :--- | :--- |
| MMLU | `mmlu::olmes` | olmes | 57-subject MC |
| TriviaQA | `triviaqa::olmes` | olmes | Open-domain QA |
| MMLU-Pro (MC) | `mmlu_pro:mc::none` | olmes | 10-way MC, harder than MMLU |
| MMLU-Pro (CoT) | `mmlu_pro:cot::none` | olmes | Tier 3 only |
| GPQA Diamond | `gpqa_diamond::olmes` | olmes | Expert-level graduate QA |
| GSM8K | `gsm8k::olmes` | olmes | Grade-school math |
| BBH | `bbh:cot::olmes` | olmes | Big Bench Hard (22 CoT tasks) |
| ARC-Challenge | `arc_challenge::olmes` | olmes | Science MC |
| MATH | `minerva_math::olmes` | olmes | Competition-level math |
| IFEval | `ifeval` | olmes | Instruction following |
| TruthfulQA | `truthfulqa:::olmo1` | olmes | Factuality / hallucination |
| BLiMP | `blimp` | harness | 67 linguistic sub-tasks |
| IndicGLUE | `indic_glue` | custom | 16 Indic NLU subsets |
| IndicQA | `indic_qa` | custom | QA in 10 Indic languages |
| Indic-Bias | `indic_bias` | custom | Gated – FairITales dataset |
| RULER | `niah_multikey_1`, `ruler_vt`, `ruler_cwe` | custom | Long-context robustness |
| LongBench | `longbench_narrativeqa`, `longbench_qasper`, `longbench_hotpotqa` | custom | Tier 3 only |
| SimpleQA_Verified | `simpleqa` | — | **disabled** – needs registration |
| APPS | `apps` | — | **disabled** – needs custom script |
| AIME 2025 | `aime_2025` | — | **disabled** – needs registration |
| L-Eval | `l_eval` | — | **disabled** – needs task suite setup |

---

## 🧠 4. Key Benchmark Details

### MMLU-Pro
`mmlu_pro:mc::none` (Tier 2+) and `mmlu_pro:cot::none` (Tier 3)
- 10-way multiple choice (vs MMLU's 4-way)
- ~12K expert-level questions across 14 subjects

### GPQA Diamond
`gpqa_diamond::olmes`
- Graduate-level questions in Biology, Physics, Chemistry
- Calibrated so PhD experts score ~65%; random baseline ~25%

### BBH (Big Bench Hard)
`bbh:cot::olmes` — 22 tasks requiring Chain-of-Thought:
Boolean Expressions, Causal Judgement, Date Understanding, Disambiguation QA, Dyck Languages, Formal Fallacies, Geometric Shapes, Hyperbaton, Logical Deduction (3/5/7 objects), Movie Recommendation, Multistep Arithmetic, Navigate, Object Counting, Penguins in a Table, Reasoning about Colored Objects, Ruin Names, Snarks, Sports Understanding, Temporal Sequences, Tracking Shuffled Objects, Web of Lies, Word Sorting.

### BLiMP
`blimp` — 67 sub-tasks evaluating grammatical acceptability:
Runs via lm-eval harness (not OLMES registry). Aggregated score across paradigms including anaphor agreement, subject-verb agreement, NPI licensing, island constraints.

---

## 🇮🇳 5. Indic Multilingual Coverage

### IndicGLUE (NLU Tasks)
Evaluating **16 distinct subsets** across major Indic scripts:
- **Logical Inference**: `wnli.hi`, `wnli.mr`, `wnli.gu`, `wnli.pa`, `wnli.bn`
- **Causality**: `copa.hi`, `copa.mr`, `copa.gu`
- **Commonsense QA**: `csqa.hi`, `csqa.te`, `csqa.ta`, `csqa.kn`, `csqa.as`, `csqa.ml`, `csqa.or`
- **Sentiment Analysis**: `actsa-sc.te`

### IndicQA (Generative Reading Comprehension)
Evaluating **10 languages**: Hindi (`hi`), Bengali (`bn`), Tamil (`ta`), Telugu (`te`), Malayalam (`ml`), Marathi (`mr`), Gujarati (`gu`), Kannada (`kn`), Punjabi (`pa`), Assamese (`as`).

---

## 📝 6. Glossary of Evaluation Metrics

| Term | Meaning |
| :--- | :--- |
| **MC** | Multiple Choice – log-likelihood over answer options. |
| **RC** | Reading Comprehension – generative, parsed against reference. |
| **BPB** | Bits Per Byte – perplexity proxy, fast and no generation needed. |
| **CoT** | Chain of Thought – prompted step-by-step reasoning before answering. |
| **`::olmes`** | Task uses OLMES-standardized prompt format and scoring. |
| **`:::olmo1`** | Task uses OLMo-1 prompt format variant. |
| **`::none`** | No special prompt wrapping; raw task format. |
