# OLMES Benchmark Pipeline Documentation

This document provides a comprehensive breakdown of all benchmarks, suites, and sub-tasks executed across the various training phases (1B Pretraining to SFT/Instruct).

---

## 📈 1. Pipeline Progression Overview
The evaluation strategy transitions from **"Easy/Fast"** tasks in early pretraining to **"Full/Complex"** tasks as the model scales.

| Phase | Main Objective | Key Suites Running | Output Directory |
| :--- | :--- | :--- | :--- |
| **CI / Breadth** | Fast Smoke Test | `ci_breadth` (Sampled from all buckets) | `./benchmark-results/ci_breadth/` |
| **Pretrain 1B** | Frequent Monitoring | `base_easy`, `indic_nlu` | `./benchmark-results/pretrain_1b/` |
| **Pretrain 3B** | Enhanced Signal | `base_easy`, `indic_standard`, `core_qa` | `./benchmark-results/pretrain_3b/` |
| **Pretrain 8B** | Full Base Milestone | `olmo3_base`, `mmlu`, `indic_standard` | `./benchmark-results/pretrain_8b/` |
| **Pretrain 70B** | Reasoning Readiness | `olmo3_base`, `bbh`, `indic_standard`, `coding_exec` | `./benchmark-results/pretrain_70b/` |
| **SFT / Instruct** | Instruction & Agency | `olmo3_adapt`, `indic_instruct`, `regression_guard` | `./benchmark-results/sft/` |

---

## 📈 2. Key Suite & Sub-task Mapping
This table explicitly maps the high-level "Key Suites" to their individual component tasks and datasets.

| Key Suite Name | Individual Sub-tasks / Components | Primary Focus |
| :--- | :--- | :--- |
| **`base_easy`** | ARC-Easy, ARC-Challenge, MMLU, CommonsenseQA, HellaSwag, Winogrande, SocialIQA, PIQA, CoQA, DROP, Jeopardy, NaturalQS, SQuAD, SciQ, QASPER, Basic Skills, Lab Bench, Lambada, MedMCQA, MedQA, SciRIFF. | Foundation / RC |
| **`olmo3_base`** | MMLU (STEM, Humanities, Social Sciences, Other), ARC-MC, MedMCQA, MedQA, SciQ, HellaSwag, Winogrande, Lambada, Basic Skills, DROP, Jeopardy, NaturalQS, SQuAD, CoQA, GSM8K, GSM-Symbolic, Minerva Math, BigCodeBench, HumanEval, LeetCode, DS1000, MBPP, MultiPL-E, HumanEval-FIM. | Scale Milestone |
| **`indic_standard`** | **IndicGLUE**: WNLI (hi, mr, gu, pa, bn), COPA (hi, mr, gu), CSQA (hi, te, ta, kn, as, ml, or), ACTSA (te). <br> **IndicQA**: (hi, bn, ta, te, ml, mr, gu, kn, pa, as). | Multilingual |
| **`bbh::olmes`** | Boolean Expressions, Causal Judgement, Date Understanding, Disambiguation QA, Dyck Languages, Formal Fallacies, Geometric Shapes, Hyperbaton, Logical Deduction, Movie Recommendation, Multistep Arithmetic, Navigate, Object Counting, Penguins in a Table, Reasoning about Colored Objects, Ruin Names, Snarks, Sports Understanding, Temporal Sequences, Tracking Shuffled Objects, Web of Lies, Word Sorting. | Logic / CoT |
| **`olmo3:adapt`** | IFEval, AlpacaEval, SimpleQA, PopQA, ZebraLogic, AGIEval (English), GPQA, Minerva Math, GSM8K, Omega, AIME (2024/2025), HumanEval+, MBPP+, LiveCodeBench. | Agency / Chat |

---

### 3. Multi-Dimensional Breakdown

#### 🇮🇳 Indic Language Coverage
All phases now run these expanded sweeps in varying sample limits:
*   **IndicGLUE (NLU):** WNLI (Logical), COPA (Causal), CSQA (Commonsense), ACTSA (Sentiment).
*   **IndicQA (Gen):** 10 Languages (hi, bn, ta, te, ml, mr, gu, kn, pa, as).

#### 💡 Glossary of Terms in Task Names:
*   **`:rc`**: Generative evaluation where the model's text output is parsed for the answer.
*   **`:mc`**: Multiple Choice evaluation based on token probabilities (log-likelihood).
*   **`:bpb`**: "Bits Per Byte" (Perplexity). Used for fast evaluations where we measure how "surprised" the model is by the correct answer without generating text.

---

## 🧠 4. English Capability Suites (Detailed Catalog)

### A. OLMES Core (QA & Common Sense)
These tasks form the "Signal of Life" for the model.
*   **ARC (AI2 Reasoning Challenge)**: Easy & Challenge sets.
*   **HellaSwag**: Common sense NLI.
*   **Winogrande**: Pronoun resolution/Common sense.
*   **PIQA / SIQA**: Physical and Social interaction QA.
*   **BoolQ**: Yes/No reading comprehension.
*   **CoQA / SQuAD**: Conversational and extractive question answering.
*   **DROP**: Reasoning over paragraphs (math/discrete).
*   **SciQ**: Science exam questions.

### B. MMLU (Massive Multitask Language Understanding)
*Suite Name:* `mmlu::olmes`
Includes **57 subjects** across STEM, Humanities, Social Sciences, and more.
*   **STEM**: Algebra, Astronomy, Biology (College/HS), Chemistry, Computer Science, Engineering, Mathematics, Physics, Statistics, Machine Learning.
*   **Humanities**: Art History, Ethnic Studies, History (European/US/World), Law, Philosophy, Prehistory, World Religions.
*   **Social Sciences**: Geography, Macro/Microeconomics, Psychology, Sociology, US Foreign Policy.
*   **Other**: Business Ethics, Clinical Knowledge, Human Aging, Management, Marketing, Nutrition, Virology.

### C. BBH (Big-Bench Hard)
*Suite Name:* `bbh::olmes`
Tasks where LLMs traditionally struggled, requiring Chain-of-Thought (CoT).
1.  Boolean Expressions
2.  Causal Judgement
3.  Date Understanding
4.  Disambiguation QA
5.  Dyck Languages
6.  Formal Fallacies
7.  Geometric Shapes
8.  Hyperbaton
9.  Logical Deduction (3, 5, 7 objects)
10. Movie Recommendation
11. Multistep Arithmetic
12. Navigate
13. Object Counting
14. Penguins in a Table
15. Reasoning about Colored Objects
16. Snarks (Sarcasm detection)
17. Sports Understanding
18. Temporal Sequences
19. Tracking Shuffled Objects
20. Web of Lies / Word Sorting

---

## 🇮🇳 5. Indic Multilingual Suites (Expanded)

These benchmarks use custom scripts to evaluate 11 major Indian languages.

### IndicGLUE (NLU Tasks)
Evaluating **16 distinct subsets** across all major scripts. In **Pretrain 1B**, we run this as `indic_nlu` without generative QA to maintain speed.
*   **Logical Inference**: `wnli.hi`, `wnli.mr`, `wnli.gu`, `wnli.pa`, `wnli.bn`
*   **Causality**: `copa.hi`, `copa.mr`, `copa.gu`
*   **Commonsense QA**: `csqa.hi`, `csqa.te`, `csqa.ta`, `csqa.kn`, `csqa.as`, `csqa.ml`, `csqa.or`
*   **Sentiment Analysis**: `actsa-sc.te`

### IndicQA (Generative Reading Comprehension)
Evaluating **10 languages** with a generative (RC) paradigm:
*   Hindi (`hi`), Bengali (`bn`), Tamil (`ta`), Telugu (`te`), Malayalam (`ml`), Marathi (`mr`), Gujarati (`gu`), Kannada (`kn`), Punjabi (`pa`), Assamese (`as`).

---

## 💻 6. Coding & Mathematics

| Category | Benchmark | Format | Description |
| :--- | :--- | :--- | :--- |
| **Math** | **GSM8K** | CoT | Grade School Math word problems. |
| **Math** | **Minerva** | CoT | High school/College competition math (Algebra to Calculus). |
| **Math** | **GSM-Symbolic** | CoT | Robustness test for GSM8K with variable numbers. |
| **Code** | **HumanEval** | Pass@k | Python function completion (standard). |
| **Code** | **MBPP** | Pass@k | Diverse basic programming problems. |
| **Code** | **BigCodeBench** | Exec | Instruction-to-code at scale. |
| **Code** | **LeetCode** | Exec | Competitive programming challenges. |

---

## 🤖 7. Agentic & Instruction (SFT Phase)

The `olmo3:adapt` suite focuses on how the model follows human intent.
*   **IFEval**: Instruction following (formatting, constraints).
*   **AlpacaEval**: Conversational helpfulness.
*   **PopQA / SimpleQA**: Factuality and hallucination checks.
*   **ZebraLogic**: Complex constraint-satisfaction reasoning.
*   **AGIEval**: College entrance and professional licensing exams.

---

## 📝 8. Glossary of Evaluation Metrics

*   **MC (Multiple Choice)**: Measured via log-likelihood of options (A, B, C, D). Robust but less representative of chat use.
*   **RC (Reading Comprehension)**: Generative evaluation where the model's text output is parsed and compared to references.
*   **BPB (Bits Per Byte)**: Perplexity-based measure. Very fast, used for frequent CI checks to track loss/surprisal trends.
*   **CoT (Chain of Thought)**: Model is prompted to "think step by step" before answering. Higher token cost but more accurate for reasoning.
