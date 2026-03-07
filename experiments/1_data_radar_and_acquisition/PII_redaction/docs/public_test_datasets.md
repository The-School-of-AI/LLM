# Public Evaluation Dataset Leads

This project now includes a local synthetic evaluation corpus and a small pulled public metadata sample.

## Pulled metadata

- AI4Privacy Open PII Masking 500K dataset card README:
  - local copy: `datasets/public/ai4privacy_open_pii_500k_README.md`
  - upstream: `https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy`

## Recommended external benchmarks to review before full corpus rollout

- AI4Privacy Open PII Masking 500K
  - multilingual PII masking dataset; dataset card indicates support that includes Hindi and Telugu.
  - source: `https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy`

- Microsoft Presidio Research
  - useful as a synthetic-data generation reference and recognizer benchmark baseline.
  - source: `https://github.com/microsoft/presidio-research`

- Official India Eighth Schedule language list
  - use this as the minimum planning set for Indic language coverage in production evaluation.
  - source: `https://legislative.gov.in/constitution-of-india/`

## Practical note

Public datasets should be used for recognizer evaluation, not as a direct proxy for the true pretraining distribution. The synthetic corpora in `datasets/synthetic/` are intentionally LLM-corpus-shaped: web text, support chats, logs, markdown, JSON snippets, and mixed-language records.
