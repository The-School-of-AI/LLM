# Tokenizer Benchmark Report
*Generated: 2026-02-03 11:42:30*

## Compression Metrics
| Tokenizer | tokens_per_byte | compression_ratio |
| --- | --- | --- |
| mock | 0.2340 | 4.6690 |

## Fertility Metrics
| Tokenizer | fertility | num_words | num_tokens |
| --- | --- | --- | --- |
| mock | 1.8176 | 30557.0000 | 55539.0000 |

## Speed Metrics
| Tokenizer | encode_tokens_per_sec | decode_tokens_per_sec |
| --- | --- | --- |
| mock | 29663243.3562 | 54819915.1638 |

## Validation Results

- **probe_neutrality**: ❌ FAIL
  - Probe 510: Matches 'mmlu_pattern' pattern
  - Probe 512: Matches 'mmlu_pattern' pattern
- **mock_curriculum**: ✅ PASS
- **mock_routing**: ✅ PASS