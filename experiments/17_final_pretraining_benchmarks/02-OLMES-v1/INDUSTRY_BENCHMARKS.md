# Industry Benchmark Standards (1B - 70B scale)

This document summarizes the state-of-the-art benchmark results for leading open-weight models across different parameter scales. These results serve as the "Gold Standard" targets for the OLMES pipeline.

## 🚀 Cross-Size Summary Table
| Benchmark | 1B-1.5B (SOTA) | 3B (SOTA) | 7B-9B (SOTA) | 70B+ (Frontier) |
| :--- | :--- | :--- | :--- | :--- |
| **MMLU** | 60.0% (Qwen 2.5) | 58.0% (Llama 3.2) | 73.0% (Llama 3.1) | 86.0% (Llama 3.1) |
| **GSM8K** | 69.0% (Qwen 2.5) | 77.7% (Llama 3.2*) | 84.5% (Llama 3.1) | 95.1% (Llama 3.1) |
| **ARC-C** | 46.2% (Qwen 2.5) | 69.1% (Llama 3.2) | 79.8% (OLMo 2) | 93.7% (Llama 3.1) |
| **Code Completion** | >15% (Signal) | >30% (Adv) | >60% (SOTA) | >80% (Frontier) |

*\*Denotes Instruct variant where base model scores were unavailable.*

---

## 📖 Detailed Results & References

### 1. Meta Llama 3.1 / 3.2 Series
Meta benchmarks are the industry most-cited baseline. Llama 3.1 established the 8B and 70B standards, while 3.2 introduced the mobile-friendly 1B/3B scales.
- **Reference**: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (2024)
- **Reference**: [Llama 3.2: Revolutionizing edge AI and vision](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)

| Model | MMLU (5-shot) | GSM8K (8-shot) |
| :--- | :---: | :---: |
| Llama 3.2 1B | 32.2 | 30.6 |
| Llama 3.2 3B | 58.0 | 77.7* |
| Llama 3.1 8B | 69.4 | 84.5 |
| Llama 3.1 70B | 83.6 | 95.1 |

---

### 3. Technical Benchmark Suites (The OLMES Implementation)
The following are the exact task identifiers configured in `industry-benchmarks.yaml`, aligned with their high-scale academic sources and exhaustive underlying datasets.

| Tier | Category | Technical Tasks | Benchmarks Included | Academic Reference |
| :--- | :--- | :--- | :--- | :--- |
| **🥉 Tier 1 (1B-3B)** | **Academic Core** | `core_9mcqa::olmes` | ARC-C, ARC-E, BoolQ, CSQA, HellaSwag, OBQA, PIQA, SIQA, Winogrande | [arXiv:2406.08446](https://arxiv.org/abs/2406.08446) |
| | **World Knowledge** | `mmlu::olmes` | 57 subjects (STEM, Humanities, Social Sciences, etc.) | [arXiv:2009.03300](https://arxiv.org/abs/2009.03300) |
| | **Multilingual** | `indic_glue`, `indic_qa` | Indic NLU (GLUE-style) and Multilingual QA | [arXiv:2212.05409](https://arxiv.org/abs/2212.05409) |
| **🥈 Tier 2 (8B)** | **Reasoning & Math** | `gsm8k::olmes` | GSM8K (Manual Few-shot) | [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) |
| | **Code Completion** | `olmo3:base:code_fim` | Fill-in-the-middle code completion (likelihood) | [arXiv:2512.13961](https://arxiv.org/abs/2512.13961) |
| **🥇 Tier 3 (70B)** | **Expert Reasoning** | `bbh:cot::olmes`, `gpqa:0shot_cot_v1` | 23 tasks (Boolean, Causal, Spatial, Logic, etc.), Graduate-Level QA | [arXiv:2501.00656](https://arxiv.org/abs/2501.00656) |
| **🏅 Tier 4 (SFT)** | **Alignment Gate** | `olmo3:adapt` | IFEval, AlpacaEval 2, WildBench, PopQA, SimpleQA, MMLU:CoT, BBH:CoT | [arXiv:2311.07911](https://arxiv.org/abs/2311.07911) |
| | **Safety/Bias** | `indic-bias` | Hindi Social Bias probes | [arXiv:2403.20147](https://arxiv.org/abs/2403.20147) |

> [!NOTE]
> **OLMES Core (`core_9mcqa`)** is the primary "signal of life" suite, maintaining a balanced evaluation of world knowledge, logic, and physical intuition.

### 🇮🇳 4. Indic Multi-Lingual Standards
For models targeting low-resource Indic languages, these are the primary academic suites.

| Benchmark | Academic Reference | Description |
| :--- | :--- | :--- |
| **IndicGLUE** | [arXiv:2212.05409](https://arxiv.org/abs/2212.05409) | General NLU (Natural Language Understanding) for Indic languages. |
| **IndicQA** | [arXiv:2407.13522](https://arxiv.org/abs/2407.13522) | Question Answering across 11 Indian languages. |
| **IndiBias** | [arXiv:2403.20147](https://arxiv.org/abs/2403.20147) | Hindi Social Bias evaluation (essential for safety reporting). |

---

## 📉 Quantization Sensitivity Index
When moving from `float16/bfloat16` to quantized formats (INT8/FP4), benchmarks are impacted differently.

| Sensitivity | Category | Benchmarks | Impact Profile |
| :--- | :--- | :--- | :--- |
| **High** | **Perplexity/BPB** | `lambada`, `base_easy:math_bpb` | PPL is extremely sensitive to weight precision; noise increases significantly. |
| **High** | **Instruction** | `ifeval`, `olmo3:adapt` | Strict constraint adherence is often the first thing to "break" during quantization. |
| **Medium-High** | **Math/CoT** | `gsm8k`, `bbh:cot`, `minerva` | Quantization logic shifts can break the delicate chain-of-thought steps. |
| **Medium** | **World Knowledge** | `mmlu`, `arc-c` | Robust at 8-bit; starts degrading at 4-bit as "nuance" is lost in hard subjects. |
| **Low** | **Code Completion** | `code_fim`, `code_bpb` | Log-likelihood based metrics are extremely robust to quantization. |

> [!TIP]
> **Pipeline vs. Industry**: Your **Developer Pipeline** will show the impact of quantization *much sooner* because it relies on BPB/RC metrics. The **Industry Pipeline** (Multiple Choice) is more robust and may mask minor degradations until they become critical.

---

## 🛠 Strategic Recommendations for your OLMES Runs
Based on these industry standards, you should categorize your milestones as:

1.  **"Signal of Life" (1B Stage)**: Target >30% MMLU (surpass Llama 3.2 1B).
2.  **"Reasoning Milestone" (8B Stage)**: Target >70% GSM8K and >65% MMLU.
3.  **"Frontier Release" (70B Stage)**: Target >80% across all capability pillars.
