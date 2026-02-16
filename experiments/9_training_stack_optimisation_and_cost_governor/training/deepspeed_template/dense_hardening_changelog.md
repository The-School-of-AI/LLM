# Dense Hardening Changelog

Changes adopted from Rohan's `dense_hardened` branch into `p9/feat/dense-hardening`.

---

## Change 1: Non-blocking H2D (Host-to-Device) Copies

**File:** `src/train.py`  
**What:** Changed all `.to(model_engine.device)` → `.to(model_engine.device, non_blocking=True)`  
**Where:** `train_epoch()` (3 calls), `evaluate()` (3 calls), `generate_text()` (2 calls)  

**Why:** H2D = Host-to-Device = copying tensors from CPU RAM to GPU. With blocking calls, the CPU waits for each tensor to fully copy before continuing. With `non_blocking=True`, the CPU kicks off the DMA transfer and immediately moves on, allowing overlap with preceding GPU computation. This is free performance because PyTorch's DataLoader already uses `pin_memory=True`.

```diff
- input_ids = batch["input_ids"].to(model_engine.device)
+ input_ids = batch["input_ids"].to(model_engine.device, non_blocking=True)
```

---

## Change 2: Kernel Fail-Fast

**Files:** `src/models/recurrence_model_1b.py`, `main.py`, `config.yaml`

**What:** 
- Added `require_fused_deltanet_kernel = True` to `ModelConfig`
- Added `require_fused_kernel` parameter to `GatedDeltaNet.__init__()`
- DeltaNet now **raises RuntimeError** if fla kernel is unavailable (instead of silently falling back to Python loop)
- Added `validate_kernel_policy()` in `main.py` that checks `HAS_TRITON` and `HAS_FLA` at startup

**Why:** The Python fallback for DeltaNet's gated delta rule is ~500x slower than the fused Triton kernel. Silently falling back means you could accidentally run a multi-day training job at 0.1% of expected speed without realizing it.

**Config:**
```yaml
training:
  require_fused_kernels: false   # Global: fail if ANY Triton/FLA kernels missing
model:
  require_fused_deltanet_kernel: true  # DeltaNet-specific: crash on missing fla
```

---

## Change 3: Precision Validation at Startup

**File:** `main.py`

**What:** Added `validate_precision_policy(deepspeed_config, model_dtype)` that cross-checks:
- DS bf16 and fp16 can't both be enabled
- Model dtype (bf16) must match DS precision flags
- Called after model is cast to bf16, before DeepSpeed init

**Why:** Model is pre-cast to bf16 for reversible training (DS autocast is OFF). If someone accidentally enables fp16 in the DS config, the model would get mixed fp16/bf16 dtypes causing silent NaN losses hundreds of steps in. This catches the misconfiguration at startup.

---

## Change 4: JSONL Structured Metrics

**Files:** `src/train.py`, `main.py`, `config.yaml`

**What:**
- Added `_append_jsonl(path, record)` helper (rank-0 only, auto-creates dirs)
- Added `metrics_jsonl_path` parameter to `train_epoch()` and `evaluate()`
- Logs training metrics (loss, tokens/s, global_step) at optimizer-step boundaries
- Logs eval metrics (avg_loss, avg_perplexity) after evaluation completes

**Why:** Console logs are hard to parse programmatically. JSONL gives a structured format that downstream tools (W&B, custom dashboards, pandas analysis) can read directly:
```json
{"phase": "train", "epoch": 0, "global_step": 10, "loss": 4.32, "tokens_per_sec": 15234.5, "tokens": 40960}
{"phase": "validation", "avg_loss": 4.18, "avg_perplexity": 65.37, "steps": 102}
```

**Config:**
```yaml
training:
  metrics_jsonl_path: "./logs/metrics.jsonl"  # null to disable
```

---

## Changes NOT Adopted (and Why)

| Rohan's Change | Why We Skipped It |
|---|---|
| Removed variance EMA snapshot from GSA | Breaks reversibility safety for gradient computation |
| Added `torch.cuda.synchronize()` in reversible backward | Performance cost; our code argues it's unnecessary due to intra-stream serialization |
| Lost gradient accumulation boundary tracking | Our loop correctly averages loss across micro-batches and logs at optimizer-step boundaries |
| Re-reads DS config from file for `deepspeed.initialize()` | Loses the `train_batch_size` auto-compute logic we implemented |
