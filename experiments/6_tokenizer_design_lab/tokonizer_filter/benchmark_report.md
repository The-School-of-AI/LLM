# Tokenizer Benchmark Report
*Generated: 2026-02-03 21:43:05*

## Compression Metrics
| Tokenizer | tokens_per_byte | compression_ratio |
| --- | --- | --- |
| tokenizers | 0.4754 | 2.1505 |
| gpt4o | 0.2918 | 3.9545 |
| qwen | 0.3348 | 3.6977 |
| deepseek | 0.3035 | 3.8480 |

## Fertility Metrics
| Tokenizer | fertility | num_words | num_tokens |
| --- | --- | --- | --- |
| tokenizers | 2.9792 | 147006.0000 | 437960.0000 |
| gpt4o | 1.4835 | 147006.0000 | 218077.0000 |
| qwen | 1.5936 | 147006.0000 | 234276.0000 |
| deepseek | 1.5229 | 147006.0000 | 223873.0000 |

## Speed Metrics
| Tokenizer | encode_tokens_per_sec | decode_tokens_per_sec |
| --- | --- | --- |
| tokenizers | 654325.9387 | 4133679.4096 |
| gpt4o | 1205691.1078 | 8990170.3370 |
| qwen | 324828.9635 | 1962790.4471 |
| deepseek | 228067.3405 | 1626745.4665 |

## Fallback Metrics (Safety)
| Tokenizer | unk_rate | byte_fallback_rate |
| --- | --- | --- |
| tokenizers | 0.0000 | 0.0000 |
| gpt4o | 0.0000 | 0.0000 |
| qwen | 0.0000 | 0.0000 |
| deepseek | 0.0000 | 0.0000 |

## Category Performance Breakdown
### Category: Code

| Tokenizer | tokens_per_byte | compression_ratio |
| --- | --- | --- |
| tokenizers | 0.4568 | 2.1988 |
| gpt4o | 0.2560 | 4.1124 |
| qwen | 0.2662 | 4.0545 |
| deepseek | 0.2611 | 4.0173 |

### Category: Dataset

| Tokenizer | tokens_per_byte | compression_ratio |
| --- | --- | --- |
| tokenizers | 0.4236 | 2.3638 |
| gpt4o | 0.1909 | 5.2938 |
| qwen | 0.1946 | 5.2003 |
| deepseek | 0.1888 | 5.3484 |

### Category: Indic

| Tokenizer | tokens_per_byte | compression_ratio |
| --- | --- | --- |
| tokenizers | 0.4512 | 2.2533 |
| gpt4o | 0.3181 | 3.2385 |
| qwen | 0.4360 | 2.3909 |
| deepseek | 0.3978 | 2.6057 |

### Category: Instructions

| Tokenizer | tokens_per_byte | compression_ratio |
| --- | --- | --- |
| tokenizers | 0.4447 | 2.2718 |
| gpt4o | 0.2130 | 4.7817 |
| qwen | 0.2096 | 4.8604 |
| deepseek | 0.2154 | 4.7405 |

### Category: Math

| Tokenizer | tokens_per_byte | compression_ratio |
| --- | --- | --- |
| tokenizers | 0.6007 | 1.6966 |
| gpt4o | 0.5393 | 1.9456 |
| qwen | 0.6589 | 1.6013 |
| deepseek | 0.5239 | 1.9975 |

### Category: Mcq

| Tokenizer | tokens_per_byte | compression_ratio |
| --- | --- | --- |
| tokenizers | 0.5115 | 1.9661 |
| gpt4o | 0.2952 | 3.4296 |
| qwen | 0.3212 | 3.1574 |
| deepseek | 0.3046 | 3.3247 |

## Validation Results

- **probe_neutrality**: ❌ FAIL
  - Probe 548: Matches 'mmlu_specific' pattern
  - Probe 814: Matches 'mmlu_specific' pattern
- **tokenizers_curriculum**: ✅ PASS
- **gpt4o_curriculum**: ✅ PASS
- **qwen_curriculum**: ✅ PASS
- **deepseek_curriculum**: ✅ PASS
- **tokenizers_routing**: ✅ PASS
- **gpt4o_routing**: ✅ PASS
- **qwen_routing**: ✅ PASS
- **deepseek_routing**: ✅ PASS