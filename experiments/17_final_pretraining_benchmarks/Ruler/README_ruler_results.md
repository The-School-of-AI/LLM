
# RULER Benchmark Results — Llama‑3.1‑8B (4096 Context)

This repository contains evaluation results for **meta‑llama/Llama‑3.1‑8B** on the **RULER long‑context benchmark** at **4K (4096 tokens)** context length.

The results were validated against the methodology described in the paper:

**RULER: What’s the Real Context Size of Your Long‑Context Language Models?** (2024).

---

# Model
- **Model:** meta-llama/Llama-3.1-8B  
- **Context length tested:** 4096 tokens  
- **Generation settings:**  
  - temperature: 0.0  
  - top_p: 1.0  
  - max_gen_tokens: 50  
  - greedy decoding

All evaluations were run on the **validation split with 100 samples per task**.

---

# Results

| Task | Score |
|-----|------|
| ruler_niah_s_1 | 1.00 |
| ruler_niah_s_2 | 1.00 |
| ruler_niah_s_3 | 1.00 |
| ruler_niah_mk_1 | 1.00 |
| ruler_niah_mk_2 | 1.00 |
| ruler_niah_mk_3 | 1.00 |
| ruler_niah_mv | 1.00 |
| ruler_niah_mq | 0.9925 |
| ruler_vt | 1.00 |
| ruler_cwe | 1.00 |
| ruler_fwe | 0.9633 |
| ruler_qa_1 | 0.84 |
| ruler_qa_2 | 0.63 |

Primary metric: **RULER Recall**.

---

# Validation Against the Paper

The paper reports that **Llama‑3.1‑8B achieves near‑perfect performance on NIAH‑style retrieval tasks at short context lengths (≤4K)**.

Your results match this observation:

### Needle‑in‑a‑Haystack (NIAH)

All NIAH variants achieve **≈100% recall**, which is consistent with the paper’s finding that:

> Modern models easily solve vanilla passkey retrieval at short context lengths.

Tasks matching this category:

- niah_s_*  
- niah_mk_*  
- niah_mv  

These results confirm that the model correctly retrieves inserted keys within the context.

---

### Multi‑Query NIAH

`ruler_niah_mq` achieves **0.9925**, which is also consistent with expectations.  
This task is slightly harder because it requires retrieving multiple keys.

---

### Synthetic Token Tracking Tasks

- `ruler_vt`
- `ruler_cwe`
- `ruler_fwe`

These test word or token tracking across long contexts.

Performance remains high but begins to degrade slightly (`fwe ≈ 0.96`).  
This behavior is also consistent with observations in the paper where non‑retrieval reasoning tasks show small accuracy drops.

---

### Question Answering Tasks

| Task | Score |
|-----|------|
| qa_1 | 0.84 |
| qa_2 | 0.63 |

These tasks are **significantly harder** than simple retrieval because they require:

- multi‑hop reasoning
- aggregating multiple context spans
- semantic understanding rather than token matching

The drop in accuracy aligns with the paper’s conclusion:

> High NIAH scores do not imply strong long‑context reasoning ability.

---

# Summary

Key observations:

- **Perfect retrieval performance** on most NIAH tasks.
- **Minor degradation** on synthetic tracking tasks.
- **Substantial degradation** on QA tasks requiring reasoning.

This pattern **matches the findings reported in the RULER paper**:  
simple retrieval is easy for modern LLMs, while long‑context reasoning remains challenging.

---

# Reproducing

Example command used for evaluation:

```
lm_eval   --model hf   --model_args pretrained=meta-llama/Llama-3.1-8B   --tasks ruler   --batch_size 1
```

---
