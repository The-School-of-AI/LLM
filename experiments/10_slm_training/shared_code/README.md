# Model-1B WikiText-2 Training Guide

This folder contains:
- `model_1b.py`: 1B-style dense hybrid architecture (DeltaNet + GSA + MTP path).
- `train_wikitext2_model_1b_gpt2.py`: training/eval script for WikiText-2 using GPT-2 tokenizer.
- `test_wikitext2_reference_gpt2.py`: reference runner used as template.

## 1) Architecture Summary

The configured model target is:
- Total params: ~1B dense
- Hidden size: 4096
- Backbone layers: 8
- MTP layers: 1 (via `mtp_num_predictions=2`)
- Delta/GSA split: 6/2 (75% / 25%)
- FFN intermediate size: 2048
- Delta gate dim: 384
- Experts: 0 (dense mode)

Core modules in `model_1b.py`:
- Token embedding:
  - `standard`: `nn.Embedding(vocab_size, hidden_size)`
  - `kronecker`: `PureHybridEmbeddingTorch` + `pf_to_model`
- Backbone:
  - `LightningDecoderLayer` repeated `num_layers`
  - First 6 layers: `GatedDeltaNet`
  - Last 2 layers: `GatedSparseAttention`
- Integration:
  - `ReversibleMidpointStack` (or fallback sequential stack if module is missing)
- Prediction heads:
  - NTP: `lm_head(h_t)` for next-token prediction
  - MTP: `MTPTransformerBlock([h_t ; emb_{t+1}])` then `lm_head`

## 2) Forward and Loss Flow

High-level flow:
1. Text -> GPT-2 tokenizer -> token IDs
2. Token IDs -> fixed-length blocks (`seq_length`, `stride`)
3. `input_ids` -> model backbone -> `logits_ntp`, `logits_mtp`, `aux_loss`
4. Losses in `train_wikitext2_model_1b_gpt2.py`:
   - `main_loss` (NTP): CE between `logits_ntp[:, :-1]` and labels shifted by +1
   - `mtp_loss` (t+2): CE between `logits_mtp` and labels shifted by +2
   - `aux_loss`: included only if finite and experts are enabled
   - `total = main + mtp_loss_weight * mtp + aux_loss_weight * aux`
5. Optimizer step:
   - AdamW + cosine decay with warmup
   - gradient accumulation
   - gradient clipping
   - optional AMP (`bfloat16`/`float16`, device dependent)

## 3) Install Dependencies

```bash
pip install torch datasets transformers
```

If you run on CUDA, use your CUDA-compatible PyTorch build.

## 4) Training Commands

### A) Quick smoke test

```bash
python train_wikitext2_model_1b_gpt2.py \
  --tokenizer gpt2 \
  --seq-length 128 \
  --max-train-tokens 50000 \
  --max-steps 20 \
  --batch-size 1 \
  --gradient-accumulation 1 \
  --eval-batches 10
```

### B) Longer training run

```bash
python train_wikitext2_model_1b_gpt2.py \
  --tokenizer gpt2 \
  --seq-length 256 \
  --max-steps 1000 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --learning-rate 3e-4 \
  --min-learning-rate 1e-5 \
  --warmup-steps 100 \
  --weight-decay 0.1 \
  --log-interval 10 \
  --eval-batches 50
```

### C) Kronecker embedding mode

```bash
python train_wikitext2_model_1b_gpt2.py \
  --tokenizer gpt2 \
  --embedding-type kronecker \
  --seq-length 128 \
  --max-steps 100 \
  --batch-size 1
```

### D) Strict architecture check

```bash
python train_wikitext2_model_1b_gpt2.py \
  --tokenizer gpt2 \
  --strict-arch \
  --max-steps 1
```

## 5) Useful Runtime Flags

- `--device {auto,cuda,mps,cpu}`: force device choice.
- `--no-amp`: disable mixed precision.
- `--amp-dtype {bfloat16,float16}`: mixed precision dtype.
- `--num-workers`, `--prefetch-factor`, `--persistent-workers`: DataLoader tuning.
- `--eval-batches N`: run evaluation after training (`0` disables eval).
- `--mtp-loss-weight`, `--aux-loss-weight`: tune combined loss.
- `--max-train-tokens`, `--max-eval-tokens`: reduce dataset size for testing.

## 6) Notes

- `seq_length` must be `>= 3` for NTP + MTP loss computation.
- If `reversible_ops_midpoint` is unavailable, the script auto-injects a compatible sequential fallback.
- Dense configuration (`num_real_experts=0`) is handled safely in the trainer (aux loss ignored when not valid).

