# Test 14 — Pre-flight checklist (last check before long run)

## Verified

- **No fused CE**  
  - `liger_ops.py`: No `LigerFusedLinearCrossEntropyLoss`.  
  - `train.py`: No `use_fused_ce`; always calls model with `return_loss=True`, `return_memory=False`; unpacks `logits_ntp, logits_mtp, aux_loss` and computes loss with `F.cross_entropy` on logits.  
  - `main.py`: No `use_fused_ce` passed to `train_epoch`.

- **Model forward contract**  
  - With `return_loss=True`, `return_memory=False`: returns `(logits_ntp, logits_mtp, total_aux_loss)` (3 values).  
  - Train and evaluate both unpack these 3 values correctly.

- **Recurrence ("different" style)**  
  - Inject: before stream expansion into `x`; then `x_stream[:,:,0,:] = x`.  
  - Readout: `memory_stream_out = h_main[:, -1, :].detach()`.  
  - Training currently uses `return_memory=False`, `prev_memory_stream=None` (single-chunk; no cross-chunk recurrence in loop).

- **Architecture**  
  - DDDGDDDG (6 DeltaNet, 2 GSA).  
  - ModelConfig aligned with Test 5 (including MoE-placeholder attrs).  
  - DeltaNet uses fla `chunk_gated_delta_rule`; GSA uses Triton sparse attn + indexer.

- **Compile / import (Mac)**  
  - `main.py`, `src/models/recurrence_model_1b.py`, `src/train.py`, `src/kernels/__init__.py`, `src/kernels/fla_deltanet.py` compile.  
  - `src.kernels`, `src.models.recurrence_model_1b`, `src.train`, `src.data` import successfully.

---

## Before you run (GPU run)

1. **Environment**  
   - CUDA available.  
   - `pip install fla` (required for DeltaNet when `require_fused_deltanet_kernel=True`).  
   - Triton available (optional but recommended for GSA/sinkhorn).  
   - `deepspeed` and deps installed for `main.py`.

2. **Config**  
   - `model_variant=reversible` (main enforces this).  
   - Config YAML or CLI args point to the right data, max_length, batch size, and DeepSpeed config.

3. **Data**  
   - `get_dataloaders` gets dataset_name/dataset_config (or tokenized path) so train/eval loaders are non-empty.

4. **Optional**  
   - If you use chunked long-doc training later, wire `prev_memory_stream` / `return_memory=True` and pass memory between chunks in the train loop.

---

## Quick re-check (from repo root)

```bash
cd experiments/tests/Test_14_gsa_only_liger_kernels_1000steps/code
python3 -m py_compile main.py src/models/recurrence_model_1b.py src/train.py
python3 -c "import sys; sys.path.insert(0,'.'); from src.train import train_epoch, evaluate; from src.models.recurrence_model_1b import Model1B, ModelConfig, create_model_1b; print('OK')"
```
