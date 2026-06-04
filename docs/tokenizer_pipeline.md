# Tokenizer Pipeline

The default training configs use the tokenizer artifacts in `tokenizer/`.

## Included Artifacts

```text
tokenizer/tokenizer.json
tokenizer/tokenizer_reordered.json
tokenizer/tokenizer_config.json
tokenizer/special_tokens_map.json
tokenizer/token_permutation.npy
tokenizer/token_inv_permutation.npy
```

## Analyze The Included Tokenizer

```bash
python3 tokenizer/byte_analysis/analyze.py --tokenizer-dir tokenizer
```

## Build A Tokenizer

Prepare tokenizer training samples, then run:

```bash
python3 scripts/tokenizer/build_tokenizer.py \
  --data-dir /path/to/tokenizer_samples \
  --output-dir tokenizer_out \
  --work-dir tokenizer_work
```

For the hybrid tokenizer path:

```bash
python3 scripts/tokenizer/build_hybrid_tokenizer.py
```

## Audit

```bash
python3 tokenizer/paper_artifacts/verify_max_byte_length.py tokenizer/tokenizer.json
python3 tokenizer/paper_artifacts/verify_no_cross_script_merges.py tokenizer/tokenizer.json
```

Round-trip checking uses your own Indic audit text files:

```bash
python3 tokenizer/paper_artifacts/roundtrip_check.py \
  --tokenizer tokenizer/tokenizer.json \
  --sft-dir /path/to/indic_audit_texts
```
