# PII Redaction Pipeline Runbook

## Scope

This pipeline is intended for pretraining-corpus sanitization before downstream tokenization, packing, or training. It focuses on high-confidence structured PII detection and deterministic redaction, with document dropping only for heavy contamination.

## Default assumptions

- Input format: JSONL with one document per line.
- Default text field: `text`.
- Records can optionally include `id`, `lang`, and `source`.
- Redaction uses typed placeholders such as `<EMAIL_ADDRESS>`.
- Unanchored person-name redaction is disabled by default to avoid destroying public-figure references and general named entities.

## Recommended stage placement

1. Source ingest
2. License validation
3. Content safety filtering
4. Language filtering
5. Dedup on raw content
6. PII redaction and contamination drop
7. Final dedup on sanitized text if placeholder inflation is material
8. Tokenization and packing

## Run locally

```powershell
python run_pii_redaction.py --input examples/sample_input.jsonl --output-dir out --config configs/default_config.json
```

## Generate and evaluate synthetic corpora

```powershell
python tools/generate_eval_corpora.py
python tools/run_eval_suite.py --config configs/default_config.json
python run_pii_redaction.py --input datasets/synthetic/llm_multilingual_structured.jsonl datasets/synthetic/llm_nested_records.jsonl --output-dir out_eval2 --config configs/synthetic_eval_config.json
```

## Produced artifacts

- `run_manifest.json`: aggregate run-level metrics
- `<shard>/redacted.jsonl`: sanitized corpus shard
- `<shard>/dropped.jsonl`: document ids and drop reasons only
- `<shard>/metrics.json`: per-file metrics
- `<shard>/_SUCCESS.json`: resumability marker

## Policy notes

- Structured PII is redacted in-place with typed placeholders.
- URLs are only rewritten when sensitive query keys are present.
- Entire documents are dropped when contamination is dense or repeatedly high-risk.
- Names are only redacted when explicitly anchored by labels like `name:` or `contact:`.
- Synthetic evaluation corpora cover LLM-style chat, web, log, markdown, JSON, and nested-schema records with multilingual Indic coverage.
