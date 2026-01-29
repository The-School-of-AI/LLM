### Benchmarking in Multilingual/Indic

Multilingual capabilities at the pre-training stage refers to the foundation, ability of the model to train on the large and diverse corpus of data containing text in multiple languages ( in our case: English & Indic Languages )
#### What these Benchmarks actually measures
- Multilingual Language Modelling: The model should be able to continue the text in the same language with Fluency in both English and Indic Languages.
- Sentence-level Multilingual Understanding: Should be able to Identify sentence level semantics and classification of different languages
- Script & Morphology Robustness: Should be robust with scripting and morphology

#### Benchmarks
1) To evaluate Multilingual Language Modelling (Fluency):
 - **IndicQA Benchmark** : This benchmark assesses generation capabilities across 11 Indic languages for tasks abstractive tasks and question answering capabilities in low resource languages.
	- **Reference paper:** https://arxiv.org/abs/2407.13522 ( Used by Gemini )
1) For semantic identification and classification across different languages:
- **MILU (Multi-task Indic Language Understanding):** A benchmark for multi-task understanding across several Indian languages, similar to the global MMLU standard.
    - **Reference Paper:** [MILU: A Multi-task Indic Language Understanding Benchmark](https://huggingface.co/papers/2411.02538).
- **IndicMMLU-Pro:** An advanced version of the MMLU framework tailored for the linguistic complexities and cultural nuances of the Indian subcontinent ( across 9 languages namely: (Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Urdu, Kannada, Punjabi))
    - **Reference Paper:** [IndicMMLU-Pro: Benchmarking the Indic Large Language Model Landscape](https://arxiv.org/html/2501.15747v1).
1) To test how well the model handles different scripts and complex word structures
  - **IndicGLUE:** Provides a standardized framework to evaluate models on morphological and syntactic tasks for South Asian languages.
    - **Reference Paper:** [IndicGLUE: A Natural Language Understanding Benchmark for Indian Languages](https://arxiv.org/abs/2210.04782).


#### Evaluated Properties

- **Valid Training Phase:** The specific stage (e.g., [Base Model Pre-training](https://arxiv.org/html/2410.10739v1) or [Instruction Fine-Tuning](https://arxiv.org/pdf/2508.17184)) where the benchmark provides the most meaningful insights.
- **Assumes Instruction Tuning:** Whether the benchmark requires the model to follow specific [prompt formats](https://arxiv.org/html/2308.10792v5) (like "Question: ... Answer:") to function correctly.
- **Quantization Tolerance:** The model's ability to maintain accuracy when converted to lower precision (e.g., [INT8 or 4-bit](https://raw.githubusercontent.com/mlresearch/v235/main/assets/li24bb/li24bb.pdf)), which is critical for [deployment efficiency](https://arxiv.org/html/2503.07103v2).
- **Known Pitfalls / Misuse Risks:** Documented limitations such as [data leakage](https://pmc.ncbi.nlm.nih.gov/articles/PMC11976015/), [catastrophic forgetting](https://arxiv.org/abs/2403.08350), or [translation artifacts](https://aclanthology.org/2024.emnlp-main.542.pdf) that can skew results.

#### Benchmarking

| **Benchmark**     | **Valid Training Phase**     | **Instruction Tuning** | **Quantization Tolerance** | **Known Pitfalls**                                                                                                                                                                                |
| ----------------- | ---------------------------- | ---------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Indic QA**      | Post Training                | Yes                    | Moderate                   | One significant drawback is the potential for cascading errors. Translation errors occurring early in the pipeline can propagate through subsequent stages, adversely affecting the final output. |
| **MILU**          | Pre training and Fine tuning | Optional               | High (at 8 bit)            | Like MMLU, it measures broad world knowledge; if the pre-training corpus lacked specific Indic cultural data, the model may fail regardless of its linguistic fluency.                            |
| **IndicMMLU-Pro** | Instruction Tuning           | Highly Recommend       | None                       | Similar to general MMLU-Pro findings, models evaluated on IndicMMLU-Pro often show significant performance drops in lower-resource or more complex languages compared to higher-resource ones.    |
| **IndicGLUE**     | Pretraining                  | No                     | High                       | Several datasets within IndicGLUE were created by machine-translating existing English datasets, which may fail to capture the cultural nuances and linguistic subtleties of Indian languages.    |
#### Explicitly classify benchmarks as:
- **RUN NOW (pretraining-valid)** : **IndicGLUE**
- **DEFERRED (Post-SFT / Alignment)** : **IndicQA**
- **RESEARCH-ONLY / UNSAFE** : **Standard MMLU (Translated to Indic)**
