# Post-Training SFT Dataset Selection Guide

## Model Context

| Parameter | Value |
|---|---|
| **Model Size** | 70B parameters |
| **Pre-training Pool** | 1.15T tokens (available in S3) |
| **Actual Training Tokens** | 209B tokens (OPUS selected ~40%) |
| **OPUS Efficiency** | ~6x (equivalent to ~1.25T tokens) |
| **Target Languages** | English + 11 Indic languages |
| **Training Pipeline** | 1B → 3B → 8B → 70B (staged) |

---

## 1. Pre-Training Data Summary

### Domains Covered in Pre-Training

| Domain | Key Datasets | Est. Tokens | Languages |
|---|---|---|---|
| Web Crawl | cc_head, cc_tail, cc_middle, RefinedWeb, C4 | ~689B | English + Multi |
| Wikipedia | megawika, ai-bharath-wiki | ~168B | Multilingual |
| Books | books | ~197B | English + Multi |
| Indic Languages | sangraha (11 langs), ai-bharath-*, erav4_lang_*, samvaad_hi, NCERT | ~5-10B | As, Bn, Gu, Hi, Kn, Ml, Mr, Or, Pa, Ta, Te |
| Code | StarCoder, code_tab, code_crlf, SWE-smith-trajectories, OpenCodeReasoning-2 | ~33B | 80+ code languages |
| Math | open_web_math, OpenMathReasoning, OpenMathInstruct-2, NuminaMath-1.5, OpenR1-Math-220k, proof_pile_2 | ~50B | English |
| Scientific | pes2o, redpajama-arxiv | ~38B+ | English |
| Instruction/NLP | flan, flan_v2, glaive-function-calling-v2, mmlu_auxiliary_train | ~3-5B | English |
| News/Social | cc_news, reddit, stackexchange | ~10B+ | English + Multi |

### Key Observations

1. **English dominates** — Web crawl, books, Wikipedia = hundreds of billions of tokens. The model has deep English knowledge.
2. **Indic languages are thin** — 11 languages combined = ~5-10B tokens. This is a small fraction of total training. Some languages (Assamese, Odia) have very little data.
3. **Math and code are strong** — Significant investment in reasoning traces and code corpora. SFT can unlock these effectively.
4. **Some instruction data exists** — FLAN, glaive in pre-training gives a base instruction-following foundation.

### Critical Insight: What SFT Can and Cannot Do

SFT is most effective at teaching the model how to *format and present* knowledge it already has from pre-training. SFT is NOT effective at teaching the model knowledge it never learned. This has direct implications:

- For **English, math, and code** — the model has deep pre-training knowledge. SFT will be highly effective here. A small number of excellent examples will produce large capability gains.
- For **Hindi and Bengali** — moderate pre-training coverage exists. SFT can meaningfully improve conversational ability.
- For **Assamese, Odia, Punjabi** — pre-training coverage is very thin. SFT alone will produce a model that *looks* fluent but generates shallow or hallucinated content. This is dangerous because users will trust fluent-sounding but incorrect answers.

**Recommendation:** Before SFT, evaluate the 70B base model on simple tasks in each Indic language. If a language can't handle basic Q&A coherently, that language needs continued pre-training (mid-training) first, not just SFT.

---

## 2. SFT Capability Buckets

Based on pre-training analysis, SFT should target these 7 categories:

| # | Bucket | Purpose | Pre-training Support | SFT Effectiveness |
|---|---|---|---|---|
| 1 | English General | Instruction following, chat, Q&A | Strong (100B+ tokens) | High — format existing knowledge |
| 2 | Indic Languages | Instruction following in Indian languages | Weak (~5-10B tokens) | Variable — depends on language |
| 3 | Math Reasoning | Step-by-step mathematical problem solving | Strong (~50B tokens) | High — teach reasoning format |
| 4 | Code | Code generation, debugging, explanation | Strong (~33B tokens) | High — teach assistant format |
| 5 | Cross-lingual | Translation, cross-lingual Q&A | Moderate (parallel corpora) | Moderate — some foundation exists |
| 6 | Safety & Refusals | Harmful request refusal, guardrails | None (new behavior) | High — entirely new, critical behavior |
| 7 | Function Calling | Tool use, API calling, structured output | Some (glaive in PT) | Moderate — reinforcement of PT |

---

## 3. Recommended Datasets

### 3.1 English General Instruction Following

| Dataset | Size | HuggingFace Path | License | Role |
|---|---|---|---|---|
| **Tulu 3 SFT Mixture** ⭐ | 939K examples | `allenai/tulu-3-sft-mixture` | ODC-BY-1.0 | **PRIMARY** — Proven SOTA open SFT mix. Contains CoCoNot, FLAN v2, No Robots, OpenAssistant Guanaco, persona-based synthetic data, math, code, and safety subsets. |
| No Robots | 9,500 examples | `HuggingFaceH4/no_robots` | CC-BY-NC-4.0 | Human-written quality anchor. Pure human demonstrations. |
| OpenAssistant Guanaco | 7,132 examples | `timdettmers/openassistant-guanaco` | Apache 2.0 | Real human multi-turn conversations. |
| FLAN v2 | ~90K examples | `ai2-adapt-dev/flan_v2_converted` | Apache 2.0 | NLP task diversity. Already in pre-training — SFT version reformats as instruction pairs. |

**Recommendation:** Tulu 3 SFT Mixture is the most validated open SFT recipe. Subsample to ~50K for your mix. This is where the model gets the most return per SFT example because pre-training support is deep.

---

### 3.2 Indic Language Instruction Following

| Dataset | Size | HuggingFace Path | License | Role |
|---|---|---|---|---|
| **UPDESH** ⭐ | 9.5M data points | `microsoft/Updesh_beta` | Research only | **PRIMARY** — 13 Indic languages. Reasoning (translated OrcaAgent-Instruct/OrcaMath) + generative (Wikipedia-grounded). Multi-turn, long-context. |
| IndicAlign | 74.7M pairs | `ai4bharat/IndicAlign` | Various | Supplement — Human crowd-sourced prompts via Anudesh, translated English SFT data, WordNet QA. |
| Aya Dataset | 204K examples | `CohereLabs/aya_dataset` | Apache 2.0 | Quality anchor — Human-curated by native speakers across 65 languages. |
| Aya Collection | 513M instances | `CohereLabs/aya_collection` | Apache 2.0 | Scale backup — Massive templated + translated collection. Needs aggressive filtering. |

**Honest assessment of UPDESH vs IndicAlign:**

| Dimension | UPDESH | IndicAlign |
|---|---|---|
| Context length | Multi-turn, long-context | Mostly single-turn, short |
| Cultural grounding | Wikipedia-grounded, culturally native | Mostly translated from English |
| Reasoning | Includes OrcaAgent-Instruct subsets | Limited reasoning data |
| Human data | None (fully synthetic) | Some (Anudesh crowd-sourced) |
| Scale | 9.5M | 74.7M |
| Recency | Sep 2025 | 2024 |

**Recommended approach:** Use UPDESH as primary for languages with viable pre-training. Focus SFT effort on Hindi, Bengali, Tamil, Telugu, and Marathi where the model has enough pre-training to benefit. For Assamese, Odia, and Punjabi — consider whether continued pre-training is needed before SFT.

**Languages covered:**

| Language | Script | UPDESH | IndicAlign | Aya | PT Coverage | SFT Viable? |
|---|---|---|---|---|---|---|
| Hindi | Devanagari | ✅ | ✅ | ✅ | Highest | Yes |
| Bengali | Bengali | ✅ | ✅ | ✅ | High | Yes |
| Tamil | Tamil | ✅ | ✅ | ✅ | Medium | Yes |
| Telugu | Telugu | ✅ | ✅ | ✅ | Medium | Yes |
| Marathi | Devanagari | ✅ | ✅ | ✅ | Medium | Yes |
| Kannada | Kannada | ✅ | ✅ | ✅ | Medium | Likely |
| Gujarati | Gujarati | ✅ | ✅ | ✅ | Low-Med | Likely |
| Malayalam | Malayalam | ✅ | ✅ | ✅ | Low-Med | Evaluate first |
| Punjabi | Gurmukhi | ✅ | ✅ | ✅ | Low | Evaluate first |
| Odia | Odia | ✅ | ✅ | ✅ | Low | Evaluate first |
| Assamese | Assamese | ✅ | ✅ | ✅ | Lowest | Needs mid-training |

---

### 3.3 Math Reasoning

| Dataset | Size | HuggingFace Path | License | Role |
|---|---|---|---|---|
| **OpenMathInstruct-2** ⭐ | 14M Q-solution pairs (~600K unique Q) | `nvidia/OpenMathInstruct-2` | Permissive | **PRIMARY** — Generated by Llama 3.1 405B. Commercially permissive. SFT on this outperforms Llama3.1-8B-Instruct on MATH by +15.9%. |
| OpenR1-Math-220k | 220K problems | `open-r1/OpenR1-Math-220k` | Apache 2.0 | Long chain-of-thought reasoning traces from DeepSeek R1. Good for extended reasoning. |
| NuminaMath-CoT | ~860K problems | `AI-MO/NuminaMath-CoT` | Apache 2.0 | Competition math. Already in pre-training — SFT version adds CoT format. |
| NuminaMath-TIR | 64K examples | Part of Tulu 3 mix | Apache 2.0 | Tool-integrated reasoning (math + code execution). |

**Recommendation:** OpenMathInstruct-2 as primary. The model has strong math pre-training, so SFT here is highly effective — you're teaching format, not knowledge. Subsample to ~50K across difficulty levels.

---

### 3.4 Code

| Dataset | Size | HuggingFace Path | License | Role |
|---|---|---|---|---|
| **OpenCodeInstruct** ⭐ | 5M samples | `nvidia/OpenCodeInstruct` | Permissive | **PRIMARY** — Largest open code SFT dataset. Includes test cases, execution feedback, quality scores. |
| Ling-Coder-SFT | 4.48M | `InclusionAI/Ling-Coder-SFT` | Various | EN+CN, 20 programming languages. |
| rStar-Coder | 1M | Microsoft | Research | Competitive coding problems. |

**Recommendation:** OpenCodeInstruct as primary. Same logic as math — strong pre-training base means SFT is highly effective. Subsample to ~45K covering diverse languages and difficulty.

---

### 3.5 Cross-lingual Tasks

| Dataset | Size | Source | License | Role |
|---|---|---|---|---|
| UPDESH cross-lingual subsets | Subset of UPDESH | `microsoft/Updesh_beta` | Research | Cross-lingual reasoning tasks |
| Aya Collection (translation) | Subset | `CohereLabs/aya_collection` | Apache 2.0 | Templated translation instruction pairs |
| FLORES (reformatted) | ~10K pairs | `facebook/flores` | CC-BY-SA | Translation instruction pairs |
| IndicAlign Wiki-Conv | Subset | Part of IndicAlign | Various | India-centric cross-lingual QA |

**Recommendation:** Pull subsets from UPDESH and Aya. Reformat FLORES as instruction pairs. Target ~15K examples. Lower priority bucket since parallel corpora in pre-training provide foundation.

---

### 3.6 Safety & Refusals

| Dataset | Size | HuggingFace Path | License | Role |
|---|---|---|---|---|
| **WildGuardMix** ⭐ | 92K total (50K in Tulu 3) | `allenai/wildguardmix` | Apache 2.0 | **PRIMARY** — 13 risk categories, vanilla + adversarial prompts, refusal + compliance responses. |
| WildJailbreak | 50K | Part of Tulu 3 datasets | ODC-BY-1.0 | Adversarial jailbreak refusals. |
| **CoCoNot** ⭐ | ~11K | `allenai/coconot` | ODC-BY-1.0 | **CRITICAL** — Prevents over-refusal. Tulu 3 found this essential. Always include. |
| IndicAlign-Toxic | Subset | Part of IndicAlign | Various | Indic language safety responses. |
| Aya Red-teaming | Varies | `CohereLabs/aya-redteaming` | Apache 2.0 | Multilingual harmful prompts, 9 harm categories. |

**Recommendation:** WildGuardMix + CoCoNot as primary. This is entirely new behavior — the model has zero safety training from pre-training. Under-investing here causes real harm. Always include CoCoNot to prevent over-refusal. Add IndicAlign-Toxic for multilingual safety.

---

### 3.7 Function Calling

| Dataset | Size | HuggingFace Path | License | Role |
|---|---|---|---|---|
| **glaive-function-calling-v2** ⭐ | ~113K | `glaiveai/glaive-function-calling-v2` | Apache 2.0 | **PRIMARY** — Already in pre-training. SFT reinforces it. |
| xlam-function-calling-60k | 60K | `Salesforce/xlam-function-calling-60k` | Apache 2.0 | Structured function calling. |
| ToolACE | ~8.5K | `Team-ACE/ToolACE` | Apache 2.0 | Multi-turn, parallel + dependent function calls. |
| When2Call | Varies | Available on HF | Research | Teaches when NOT to use tools. Reduces hallucination. |

**Recommendation:** glaive-v2 as primary. Already in pre-training, SFT reinforces. Add ToolACE for multi-turn scenarios.

---

## 4. Dataset Sizing & Weighted Contributions

### Guiding Principle

**Invest SFT budget where the model has strong pre-training AND needs behavioral formatting.** SFT is not effective at teaching knowledge the model doesn't have. It is highly effective at teaching the model how to present and format knowledge it already possesses.

### Total SFT Budget

| Metric | Value |
|---|---|
| **Total examples** | ~250,000 |
| **Total tokens** | ~850M - 1B |
| **% of effective pre-training** | ~0.4% of 209B actual tokens |

### Weighted Distribution

| Bucket | Weight | Examples | Est. Tokens | Primary Dataset | Rationale |
|---|---|---|---|---|---|
| English General | **20%** | ~50K | ~160M | Tulu 3 SFT Mix (subsample) | Deep PT knowledge. SFT unlocks great assistant behavior. Highest return per example. |
| Math Reasoning | **20%** | ~50K | ~200M | OpenMathInstruct-2 (subsample) | Deep PT knowledge. Teaching step-by-step format is highly effective here. |
| Code | **18%** | ~45K | ~180M | OpenCodeInstruct (subsample) | Strong PT base. Teaching assistant-style coding is high value. |
| Indic Languages | **18%** | ~45K | ~150M | UPDESH (filtered) | Important but realistic. Focus on languages with viable PT coverage. Don't spread thin. |
| Safety & Refusals | **12%** | ~30K | ~75M | WildGuardMix + CoCoNot | Entirely new behavior. Must be robust. Under-investing causes real harm to users. |
| Cross-lingual | **6%** | ~15K | ~45M | UPDESH + FLORES subsets | Parallel corpora in PT help. Lower priority. |
| Function Calling | **6%** | ~15K | ~40M | glaive-v2 + ToolACE | Already partially in PT. Small reinforcement sufficient. |
| **TOTAL** | **100%** | **~250K** | **~850M** | | |

### Why These Weights?

- **English at 20%** — The model's strongest area. SFT here converts deep knowledge into excellent assistant behavior. This is where you get the most capability per SFT example. Not over-investing, but giving it enough to be genuinely good.

- **Math at 20%** — Same logic. Deep pre-training in math reasoning traces. SFT teaches clean step-by-step output format. High return on investment.

- **Code at 18%** — Strong pre-training base from StarCoder and SWE-smith. SFT teaches the model to be a useful coding assistant. High value.

- **Indic at 18%** — This is a realistic allocation, not an aspirational one. SFT works well for Hindi and Bengali where pre-training coverage is decent. It works less well for Assamese and Odia where coverage is thin. Rather than spreading 30% across 11 languages and getting mediocre results everywhere, focus 18% on the 5-7 languages where the model can actually deliver quality. Consider continued pre-training for the rest.

- **Safety at 12%** — This is entirely new behavior. The model has zero safety training from pre-training. This must be robust because safety failures cause real harm to real users. CoCoNot is essential to prevent over-refusal. 12% is the minimum to establish reliable safety patterns.

- **Cross-lingual at 6%** — Parallel corpora in pre-training (samanantar, NLLB, ILCI) provide a foundation. SFT activates this. Lower priority.

- **Function calling at 6%** — Already partially learned from glaive in pre-training. Small SFT reinforcement is sufficient.

### Indic Language Sub-Allocation (Honest Assessment)

Within the 18% Indic allocation (~45K examples), allocate based on pre-training viability:

**Tier 1: Full SFT investment (languages with decent PT coverage)**

| Language | Script | SFT Share | ~Examples | Notes |
|---|---|---|---|---|
| Hindi | Devanagari | 25% | ~11,250 | Best Indic PT coverage. Highest confidence. |
| Bengali | Bengali | 15% | ~6,750 | Good PT coverage. |
| Tamil | Tamil | 12% | ~5,400 | Moderate PT coverage. |
| Telugu | Telugu | 12% | ~5,400 | Moderate PT coverage. |
| Marathi | Devanagari | 10% | ~4,500 | Moderate PT coverage. Shares Devanagari script with Hindi (helpful). |

**Tier 2: Conservative SFT (evaluate base model first)**

| Language | Script | SFT Share | ~Examples | Notes |
|---|---|---|---|---|
| Kannada | Kannada | 7% | ~3,150 | Evaluate base model capability before committing. |
| Gujarati | Gujarati | 7% | ~3,150 | Evaluate base model capability before committing. |
| Malayalam | Malayalam | 5% | ~2,250 | Lower PT coverage. Results may be limited. |

**Tier 3: SFT may not be sufficient (consider continued pre-training first)**

| Language | Script | SFT Share | ~Examples | Notes |
|---|---|---|---|---|
| Punjabi | Gurmukhi | 3% | ~1,350 | Very low PT coverage. SFT alone likely insufficient. |
| Odia | Odia | 2% | ~900 | Very low PT coverage. Risk of fluent-looking hallucination. |
| Assamese | Assamese | 2% | ~900 | Lowest PT coverage. Strongly recommend mid-training first. |

**Important:** For Tier 3 languages, the honest recommendation is to either (a) do continued pre-training before SFT, or (b) scope them out of v1 and handle properly in v2. Pushing SFT on languages with insufficient pre-training produces a model that sounds fluent but gives unreliable answers — which is worse than a model that honestly admits its limitations.

---

## 5. Pre-SFT Checklist

Before starting SFT, complete these steps:

1. **Audit OPUS selection logs** — Confirm the actual domain/language distribution in the 209B tokens that were consumed. The pool distribution and the actual training distribution may differ.

2. **Evaluate base model per-language** — Run the 70B base model on simple tasks in each Indic language. If it can't handle basic Q&A in a language, SFT won't fix that.

3. **Decide on continued pre-training** — For languages that fail base evaluation, decide whether to do a mid-training phase before SFT or scope them out of v1.

4. **Download and filter datasets** — Apply quality filtering: deduplicate, remove short/long outliers, score with LLM-as-judge, decontaminate against eval benchmarks.

5. **Decontaminate against ALL eval benchmarks** — MMLU, GSM8K, HumanEval, MATH, IndicMMLU-Pro, IndicGenBench, BeleBele, etc.

---

## 6. Quick Reference: Where to Find Everything

| Dataset | URL |
|---|---|
| Tulu 3 SFT Mixture | https://huggingface.co/datasets/allenai/tulu-3-sft-mixture |
| UPDESH | https://huggingface.co/datasets/microsoft/Updesh_beta |
| IndicAlign | https://github.com/AI4Bharat/IndicLLMSuite |
| Aya Dataset | https://huggingface.co/datasets/CohereLabs/aya_dataset |
| Aya Collection | https://huggingface.co/datasets/CohereLabs/aya_collection |
| OpenMathInstruct-2 | https://huggingface.co/collections/nvidia/openmath-2 |
| OpenR1-Math-220k | https://huggingface.co/datasets/open-r1/OpenR1-Math-220k |
| NuminaMath-CoT | https://huggingface.co/datasets/AI-MO/NuminaMath-CoT |
| NuminaMath-1.5 | https://huggingface.co/datasets/AI-MO/NuminaMath-1.5 |
| OpenCodeInstruct | https://huggingface.co/datasets/nvidia/OpenCodeInstruct |
| WildGuardMix | https://huggingface.co/datasets/allenai/wildguardmix |
| CoCoNot | https://huggingface.co/datasets/allenai/coconot |
| glaive-function-calling-v2 | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 |
| ToolACE | https://huggingface.co/datasets/Team-ACE/ToolACE |
| xlam-function-calling-60k | https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k |
| Aya Red-teaming | https://huggingface.co/datasets/CohereLabs/aya-redteaming |
| FLORES | https://huggingface.co/datasets/facebook/flores |

---

## 7. Evaluation Benchmarks per Bucket

| Bucket | Benchmarks |
|---|---|
| English General | AlpacaEval v2, MT-Bench, MMLU, TruthfulQA |
| Indic Languages | IndicMMLU-Pro, IndicGenBench, BeleBele, MILU, Global MMLU |
| Math | GSM8K, MATH-500, AIME 2024, OmniMATH |
| Code | HumanEval, MBPP, LiveCodeBench, BigCodeBench |
| Cross-lingual | FLORES (BLEU/ChrF), IN22-Conv, XSum |
| Safety | WildGuardTest, XSTest, HarmBench, SorryBench |
| Function Calling | BFCL v4, Tau-bench |

---

## 8. Implementation Roadmap

**Phase 1: Base model evaluation.** Evaluate the 70B base on each target language and capability. Identify where it's strong (SFT will help) vs where it's fundamentally weak (SFT won't help, needs mid-training).

**Phase 2: Data collection and filtering.** Download primary datasets. Filter aggressively for quality. Decontaminate against eval benchmarks. One bad example undoes the effect of many good ones.

**Phase 3: Initial SFT run.** Train with the weighted distribution above. Standard hyperparameters: LR ~5e-6, cosine schedule, batch size 512, 1-2 epochs.

**Phase 4: Per-bucket evaluation.** Evaluate each capability separately. Don't look at just the average — a model that scores 80% average but fails at safety is not a good model.

**Phase 5: Mixture tuning.** Based on per-bucket results, adjust weights. If English underperforms, increase its share. If math plateaus, try harder problems from OpenR1-Math. If safety over-refuses, increase CoCoNot weight. Expect 2-3 iterations.

**Phase 6: Final SFT checkpoint.** Lock the mixture. Train the production checkpoint. Hand off to DPO/RLHF.

---

## 9. Key Warnings

1. **SFT cannot fix weak pre-training.** Don't expect SFT to make the model fluent in languages it barely saw during pre-training. For low-resource Indic languages, consider continued pre-training first.

2. **Quality over quantity.** A 70B model with OPUS-efficient pre-training needs fewer, better examples. 250K excellent examples beats 1M mediocre ones.

3. **These weights are starting points, not final answers.** The final weights should come from empirical evaluation, not from theory. Train, evaluate, adjust, repeat.

4. **Invest where the model is strong.** SFT amplifies existing capability. The biggest gains come from English, math, and code where pre-training is deep.

5. **Don't create a model that confidently speaks languages it doesn't understand.** A model that honestly says "I'm not confident in Assamese" is safer than one that generates fluent-sounding but incorrect Assamese.

6. **Safety is non-negotiable.** 12% minimum. The model has zero safety behavior from pre-training. Every unsafe response is a failure that affects real users.