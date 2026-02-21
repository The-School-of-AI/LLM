# Industry Benchmark Standards — `industry-benchmarks.yaml`

Tiered evaluation strategy based on industry gold standards (Llama 3.1/3.2, Qwen 2.5, OLMo 2).

---

## 🥉 Tier 1 — `pretrain_small` (1B–3B)
*Essential signal for small / mobile-scale models. Backend: `hf`.*

| Group | Tasks |
| :--- | :--- |
| `world_knowledge` | `mmlu::olmes`, `triviaqa::olmes`, `arc_challenge::olmes` |
| `linguistic_diagnostics` | `blimp` |
| `multilingual_indic` | `indic_glue` |
| `context_window` | `niah_4k`, `niah_8k` |

---

## 🥈 Tier 2 — `pretrain_8b` (8B)
*Industry-standard competitive milestone. Backend: `hf`.*

| Group | Tasks |
| :--- | :--- |
| `world_knowledge` | `mmlu::olmes`, `mmlu_pro:mc::none`, `triviaqa::olmes`, `arc_challenge::olmes` |
| `mathematical_reasoning` | `gsm8k::olmes`, `minerva_math::olmes` |
| `linguistic_diagnostics` | `blimp` |
| `factuality` | `truthfulqa:::olmo1` |
| `multilingual_indic` | `indic_glue`, `indic_qa` |
| `context_window` | `niah_8k`, `niah_16k`, `niah_multikey_1`, `ruler_vt`, `ruler_cwe` |

---

## 🥇 Tier 3 — `pretrain_70b` (70B)
*Frontier / publishable sweep. Backend: `vllm`, `max_new_tokens: 1024`.*

| Group | Tasks | Notes |
| :--- | :--- | :--- |
| `world_knowledge` | `mmlu::olmes`, `mmlu_pro:mc::none`, `mmlu_pro:cot::none`, `triviaqa::olmes`, `arc_challenge::olmes` | |
| `mathematical_reasoning` | `gsm8k::olmes` | |
| `expert_reasoning` | `bbh:cot::olmes`, `gpqa_diamond::olmes` | |
| `linguistic_diagnostics` | `blimp` | |
| `factuality` | `truthfulqa::olmes` | |
| `multilingual_indic` | `indic_glue`, `indic_qa` | |
| `context_window` | `ruler_vt`, `ruler_cwe` | |
| `l_eval` | `l_eval` | ⚠️ `enabled: false` – needs registration |
| `aime_2025` | `aime_2025` | ⚠️ `enabled: false` – needs registration |
| `apps` | `apps` | ⚠️ `enabled: false` – needs custom script |
| `simple_qa` | `simpleqa` | ⚠️ `enabled: false` – needs registration |

---

## � Tier 4 — `sft` (Instruct / Alignment)
*Post-SFT evaluation. Backend: `vllm`, `max_new_tokens: 2048`, `temperature: 0.7`.*

| Group | Tasks | Notes |
| :--- | :--- | :--- |
| `alignment_utility` | `olmo3:adapt` | IFEval, AlpacaEval, WildBench |
| `ifeval` | `ifeval` | |
| `factuality_safety` | `truthfulqa::olmes` | |
| `indic_bias` | `indic_bias` | ⚠️ `enabled: false` – gated dataset |
| `multilingual_indic_instruct` | `indic_glue`, `indic_qa` | |

---

## 📊 SOTA Reference Targets

| Benchmark | 1B–3B | 8B | 70B+ |
| :--- | :---: | :---: | :---: |
| MMLU | 32–60% | 69% (Llama 3.1) | 84% (Llama 3.1) |
| MMLU-Pro (MC) | — | ~35% | ~55% |
| TriviaQA | — | ~75% | ~87% |
| ARC-Challenge | 46–69% | 80% (OLMo 2) | 94% (Llama 3.1) |
| GSM8K | 30–78% | 85% (Llama 3.1) | 95% (Llama 3.1) |
| MATH (Minerva) | — | ~20% | ~50% |
| BBH (CoT) | — | ~67% | ~87% |
| GPQA Diamond | — | ~30% | ~50% |
| TruthfulQA | — | ~50% | ~60% |
| BLiMP | ~80% | ~86% | ~90% |
