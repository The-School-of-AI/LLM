
# 70B LLM Pre-training Data Strategy

This repository contains the data curation strategy and sourcing manifest for the development of our 70B parameter Large Language Model. To achieve "Chinchilla-optimal" performance, we have engineered a hybrid dataset of ~2.42 Trillion tokens, balancing global reasoning foundations with high-fidelity Indic linguistic specialized data.

## 🚀 Strategy Overview

The training follows a **staged pre-training curriculum**:
1.  **Foundation Phase:** Utilizing **DOLMA v1.7** and **IndicCorp v2** to build a broad base of world knowledge, code, and basic multilingual fluency.
2.  **Annealing/Specialization Phase:** Shifting the weight to **Dolma 3 Dolmino** and **Sangraha (Verified)** to sharpen logical reasoning, mathematical proficiency, and native-level Indic script performance.

---

## 📊 Dataset Evaluation & Manifest

| Parameter | Sangraha (AI4Bharat) | IndicCorp v2 (AI4Bharat) | Dolma 3 Dolmino Mix | Dolma v1.7 |
| :--- | :--- | :--- | :--- | :--- |
| **Type/Domain** | Multilingual Indic (Synth+PDF) | Monolingual Indic (Web) | Mid-train Reasoning Mix | General Web + Academic |
| **Compositions** | Web(45%), Synth(40%), PDF(10%) | Web (100%) | Math(30%), Code(25%), PDF(20%) | Web(75%), Code(15%), Acad(5%) |
| **Token Scale** | ~251B | ~20.9B | ~100B (Effective) | ~2T (Effective) |
| **License** | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | [CC-0](https://creativecommons.org/publicdomain/zero/1.0/) | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) |
| **Freshness** | 2024–2025 | 2022–2023 | **Nov 2025** | April 2024 |
| **Training Phase** | Pretrain / Mid-train | Pretrain (Base) | **Mid-train / Annealing** | Pretrain (Base) |
| **Dedup Method** | MinHash + Two-level | Sentence-level shuffle | **Duplodocus** (Fuzzy) | Bloom Filter |
| **Junk %** | < 1% (Verified subset) | Moderate | **~0% (olmOCR)** | Low |

---

## 🔬 Data Sources & Governance

### 1. Global Reasoning Foundation
* **Dolma v1.7 & Dolmino Mix**: Sourced from the Allen Institute for AI (Ai2). These datasets provide the core "intelligence" signal. 
* **Key Innovation**: Dolmino utilizes vision-based **olmOCR** to extract clean data from technical PDFs, virtually eliminating the "junk" typical of standard web scrapes.
* **Papers**: [Dolma (2402.00159)](https://arxiv.org/abs/2402.00159) | [OLMo 3 (2512.13961)](https://arxiv.org/abs/2512.13961)

### 2. Indic Sovereignty
* **Sangraha & IndicCorp v2**: Sourced from AI4Bharat. These are critical for our competitive advantage in the Indian market.
* **Key Innovation**: Sangraha includes **40% Synthetic data**, where high-quality English textbooks and Wiki articles were translated into Indic languages to provide high-token-quality depth for low-resource languages.
* **Papers**: [IndicLLMSuite (2403.06350)](https://arxiv.org/abs/2403.06350) | [IndicBERTv2 (2305.10672)](https://arxiv.org/abs/2305.10672)

---

## 🛠️ Engineering Requirements

### Filters & Pre-processing
* **Sangraha**: Utilize the `verified` and `synthetic` slices. Apply olmOCR for any additional PDF ingestion.
* **IndicCorp**: High HTML/Boilerplate cleaning required for legacy web scrapes.
* **Dolma**: Already heavily filtered; use the official **Dolma Toolkit** for PII masking.

### Download & Access
* **Sangraha**: `ai4bharat/sangraha` [HF Link](https://huggingface.co/datasets/ai4bharat/sangraha)
* **IndicCorp v2**: `ai4bharat/IndicCorpV2` [HF Link](https://huggingface.co/datasets/ai4bharat/IndicCorpV2)
* **Dolmino**: `allenai/dolma3_dolmino_mix-100B-1125` [HF Link](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1125)
* **Dolma**: `allenai/dolma` [HF Link](https://huggingface.co/datasets/allenai/dolma)

---

## 📜 Ethical & Legal Compliance
All datasets selected carry permissive licenses (CC-0, CC-BY, ODC-By) allowing for commercial use. 
* **Attribution**: Final model weights must include attribution to AI4Bharat and the Allen Institute for AI.
* **Safety**: All sources have undergone model-based safety filtering to remove toxicity and PII.
