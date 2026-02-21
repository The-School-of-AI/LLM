Execution - ./.venv/bin/python3 src/pipeline_runner.py --config configs/benchmark-config.yaml --stage ci_breadth --model_args "pretrained=HuggingFaceTB/SmolLM2-135M" --limit 16 --batch_size 16

{
    "metadata": {
        "stage": "ci_breadth",
        "timestamp": "2026-02-17T04:29:40.085189",
        "model_args": "pretrained=HuggingFaceTB/SmolLM2-135M",
        "device": "cuda",
        "limit": 16,
        "batch_size": "16"
    },
    "aggregates": {
        "breadth_sample:olmo3:base_easy:qa_rc": 0.0,
        "breadth_sample:bbh:cot::olmes": 0.0,
        "breadth_sample:olmo3:base_easy:code_bpb": 0.5190538560858554,
        "IndicGLUE": 0.1875,
        "niah_4096": 1.0
    },
    "granular": {
        "primary_score_micro": 0.5138650368952734,
        "primary_score_macro": 0.5190538560858554,
        "wnli.hi": 0.0625,
        "wnli.mr": 0.375,
        "wnli.gu": 0.0,
        "copa.hi": 0.5625,
        "copa.mr": 0.4375,
        "copa.gu": 0.625,
        "csqa.hi": 0.0,
        "csqa.te": 0.0,
        "csqa.ta": 0.0,
        "csqa.kn": 0.0,
        "csqa.as": 0.0,
        "csqa.ml": 0.0,
        "csqa.or": 0.0,
        "actsa-sc.te": 0.5625
    }
}

# Evaluation Report: HuggingFaceTB/SmolLM2-135M

## 1. Executive Summary

| **Metric** | **Value** |
| :--- | :--- |
| **Status** | ✅ PASS |
| **Completion** | 5/5 (100.0%) |
| **Timestamp** | 2026-02-17T04:29:40.085189 |
| **Run Directory** | `benchmark-results/ci_breadth/20260217_042935` |

## 2. Capability Benchmarks

### english

| Benchmark | Engine | Status | Score | Details |
| :--- | :--- | :--- | :--- | :--- |
| breadth_sample:olmo3:base_easy:qa_rc | olmes | ✅ success | **0.0** | - |

### cot

| Benchmark | Engine | Status | Score | Details |
| :--- | :--- | :--- | :--- | :--- |
| breadth_sample:bbh:cot::olmes | olmes | ✅ success | **0.0** | - |

### coding

| Benchmark | Engine | Status | Score | Details |
| :--- | :--- | :--- | :--- | :--- |
| breadth_sample:olmo3:base_easy:code_bpb | olmes | ✅ success | **0.5190538560858554** | <details><summary>2 subtasks in 1 groups</summary><ul><li><b>primary</b>: 0.5165 <details><summary>2 variants</summary><ul><li>primary_score_macro: 0.5190538560858554</li><li>primary_score_micro: 0.5138650368952734</li></ul></details></li></ul></details> |

### Other / Uncategorized

| Benchmark | Engine | Status | Score | Details |
| :--- | :--- | :--- | :--- | :--- |
| breadth_sample:olmo3:base_easy:qa_rc | olmes | success | **0.0** | - |
| breadth_sample:bbh:cot::olmes | olmes | success | **0.0** | - |
| breadth_sample:olmo3:base_easy:code_bpb | olmes | success | **0.5190538560858554** | <details><summary>2 subtasks in 1 groups</summary><ul><li><b>primary</b>: 0.5165 <details><summary>2 variants</summary><ul><li>primary_score_macro: 0.5190538560858554</li><li>primary_score_micro: 0.5138650368952734</li></ul></details></li></ul></details> |
| IndicGLUE | N/A | success | **0.1875** | <details><summary>14 subtasks in 8 groups</summary><ul><li>actsa-sc.te: 0.5625</li><li>copa.gu: 0.625</li><li>copa.hi: 0.5625</li><li>copa.mr: 0.4375</li><li><b>csqa</b>: 0.0000 <details><summary>7 variants</summary><ul><li>csqa.as: 0.0</li><li>csqa.hi: 0.0</li><li>csqa.kn: 0.0</li><li>csqa.ml: 0.0</li><li>csqa.or: 0.0</li><li>csqa.ta: 0.0</li><li>csqa.te: 0.0</li></ul></details></li><li>wnli.gu: 0.0</li><li>wnli.hi: 0.0625</li><li>wnli.mr: 0.375</li></ul></details> |
| niah_4096 | N/A | success | **1.0** | - |

## 4. Environment & Traceability

- **Model Args**: `pretrained=HuggingFaceTB/SmolLM2-135M`
- **Device**: `cuda`
- **Batch Size**: `16`
- **Limit**: 16
- **Log File**: `execution.log`

## 5. Raw Data (CSV)

```csv
Benchmark,Status,Score,Engine
breadth_sample:olmo3:base_easy:qa_rc,success,0.0,olmes
breadth_sample:bbh:cot::olmes,success,0.0,olmes
breadth_sample:olmo3:base_easy:code_bpb,success,0.5190538560858554,olmes
IndicGLUE,success,0.1875,N/A
niah_4096,success,1.0,N/A
```
