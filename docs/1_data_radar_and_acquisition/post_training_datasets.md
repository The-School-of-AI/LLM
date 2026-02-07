# 70B LLM Post-Training Data Strategy

This document defines the **post-training data strategy** for a 70B-parameter Large Language Model, covering **Supervised Fine-Tuning (SFT)**, **Alignment / Preference Optimization**, and **Optional Specialization**.

Post-training operates on top of a ~2T-token pretraining corpus and is explicitly designed to:
- improve instruction following and usability,
- align model behavior with human preferences and safety norms,
- optionally enable tool-use or agentic capabilities.

Post-training data is **high-signal, low-volume**, and deliberately scoped to avoid contaminating evaluation benchmarks.

---

## 🚀 Strategy Overview

Post-training follows a **three-stage curriculum**, with each stage targeting a distinct behavioral objective.

### 1. Supervised Fine-Tuning (SFT)
Establishes instruction compliance, conversational structure, and response formatting.

### 2. Alignment & Preference Optimization
Shapes helpfulness, harmlessness, verbosity, and refusal behavior using preference data (DPO / RM-style).

### 3. Optional Specialization
Introduces tool-use or agentic behavior only if required by downstream applications.

Indic datasets are included **only to preserve chat robustness**, not to claim reasoning, fluency, or multilingual parity.

---

## 📊 Dataset Evaluation & Manifest

| Dataset | Domain / Type | Primary Use | Post-Training Stage | Approx Share | License | Benchmark Risk |
|-------|---------------|-------------|---------------------|-------------|---------|---------------|
| **UltraChat (H4)** | General instruction (synthetic) | Main SFT backbone | SFT | 40–50% SFT | MIT | Medium |
| **HelpSteer v1** | Helpfulness & style (human-annotated) | Response quality shaping | SFT | 10–15% SFT | CC-BY-4.0 | Low |
| **IndicAlign-Instruct** | Indic instruction pairs | Chat robustness (Indic) | SFT | 1–3% SFT | CC-BY-4.0 | Medium |
| **HelpSteer3** | Preference / RM-style | Alignment (DPO/RM) | Alignment | 15–25% Align | CC-BY-4.0 | Low–Medium |
| **HH-RLHF** | Helpful/Harmless prefs | Safety & refusal baseline | Alignment | 30–40% Align | MIT | Low |
| **UltraFeedback** | Fine-grained preferences (GPT-4 annotated) | High-resolution alignment | Alignment (Optional) | 10–20% Align | MIT | High |
| **API-Bank** | Tool / API calling | Tool-use specialization | Specialization | 5–10% (if used) | MIT | Medium |
| **ToolBench** | Multi-step tool use | Agentic behavior | Specialization | 5–10% (if used) | Apache-2.0 | Medium–High |

---

## 🧠 Dataset-wise Use Cases & Scope

### UltraChat (H4-filtered)
**Use**
- Core supervised fine-tuning
- Instruction following and conversational structure
- General assistant behavior

**Out-of-Scope**
- Factual accuracy guarantees
- Reasoning or math benchmark performance
- Code generation claims

---

### HelpSteer v1
**Use**
- Style shaping (helpfulness, clarity, verbosity)
- Response quality calibration

**Out-of-Scope**
- Reasoning curriculum
- Knowledge injection
- Multilingual expansion

---

### IndicAlign-Instruct
**Use**
- Prevent chat regression on Indic prompts
- Maintain instruction compliance in Indic scripts

**Out-of-Scope**
- Indic reasoning or fluency claims
- Code or math capability improvements
- Multilingual parity assertions

---

### HelpSteer3
**Use**
- Preference learning (DPO / RM)
- Reinforcing desirable response patterns

**Out-of-Scope**
- Instruction learning from scratch
- Domain specialization

---

### HH-RLHF
**Use**
- Safety alignment baseline
- Helpful vs harmless calibration

**Out-of-Scope**
- Tool use
- Multilingual alignment
- Instruction diversity expansion

---

### UltraFeedback
**Use**
- High-resolution alignment signals
- Nuanced preference optimization

**Out-of-Scope**
- Open-only data purism
- Benchmark score claims without decontamination

---

### API-Bank
**Use**
- API calling and structured tool invocation
- Agent-style workflows

**Out-of-Scope**
- General chat improvement
- Pure language modeling

---

### ToolBench
**Use**
- Multi-step tool planning
- Complex agentic behaviors

**Out-of-Scope**
- Base SFT
- Reasoning benchmark improvements without tools

---

## 🧮 Post-Training Split Summary (70B Model)

### Supervised Fine-Tuning (SFT)
- UltraChat: **40–50%**
- HelpSteer: **10–15%**
- IndicAlign-Instruct: **1–3%**
- Remaining SFT: task-specific or filtered mixes

### Alignment / Preference Learning
- HH-RLHF: **30–40%**
- HelpSteer3: **15–25%**
- UltraFeedback (optional): **10–20%**

### Optional Specialization
- API-Bank + ToolBench: **5–10%**, only if tool-use is a product requirement

Total post-training volume remains **<0.3% of pretraining tokens**, consistent with modern 70B-class models.

---

## ⚙️ Engineering & Preprocessing Notes

- **SFT datasets:** schema normalization, length caps, light deduplication
- **Alignment datasets:** strict benchmark decontamination (keyword + hash based)
- **Indic datasets:** language-ID and script sanity checks only
- **Tool-use datasets:** structural validation of API calls and traces

Aggressive web-style filtering (entropy, perplexity, spam heuristics) is **not required** for post-training data.

---

## 🧾 Data Governance & Evaluation Hygiene

- All datasets are selected for **license clarity** and **redistribution safety**
- Benchmark contamination is explicitly addressed at preprocessing time
- Indic data is scoped to **presence and stability**, not capability claims
- Specialization data is isolated from baseline evaluations

---

## Summary

This post-training strategy provides a **balanced, modular, and defensible pipeline** for a 70B model:
- strong instruction following,
- robust alignment,
- optional specialization,
- minimal benchmark risk,
- and tightly scoped Indic support.

The design prioritizes **behavioral shaping over knowledge injection**, ensuring post-training enhances usability without distorting core evaluation signals.
