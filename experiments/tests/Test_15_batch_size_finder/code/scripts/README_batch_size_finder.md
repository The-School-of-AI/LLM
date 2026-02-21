# Batch size finder (Test 14)

Finds a safe **micro-batch size** and **effective batch size** for the 1.5B recurrence model (single-GPU, no ZeRO). Use the output in your DeepSpeed config as `train_micro_batch_size_per_gpu` and `gradient_accumulation_steps`.

---

## How to run

From the **code** directory (parent of `scripts/`):

```bash
cd /path/to/Test_14_gsa_only_liger_kernels_1000steps/code
python scripts/batch_size_finder.py
```

**Optional environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKENIZER_PATH` | `code/src/tokenizer/` | Directory containing `tokenizer.json` |
| `REAL_DATA_DATASET` | (none) | e.g. `wikitext` — use real data for Phase 2 loss curves |
| `REAL_DATA_CONFIG` | (none) | e.g. `wikitext-103-raw-v1` |
| `WARMUP_STEPS` | 5 | Steps before timing (not counted in tok/s) |
| `REPEAT_EACH_E` | 1 | Repeat each effective-batch run and average (2 = more stable) |
| `DETERMINISTIC` | 0 | Set to `1` for reproducible runs (seeds + same data per repeat) |
| `RUN_CAPACITY_AGAIN` | 0 | Set to `1` to re-run 1 step at M at the end (fragmentation check) |
| `EFFECTIVE_BATCH_SWEEP` | 256,512,1024,2048,4096 | Comma-separated list to sweep |

**Example with real data and fragmentation check:**

```bash
REAL_DATA_DATASET=wikitext REAL_DATA_CONFIG=wikitext-103-raw-v1 RUN_CAPACITY_AGAIN=1 python scripts/batch_size_finder.py
```

---

## Where logs are saved

- **Path:** `code/batch_size_finder_YYYYMMDD_HHMMSS.log`  
  (same directory as `main.py`, i.e. the **code** root; timestamp is run start.)
- Logging goes to **both** this file and stdout.
- At the end the script prints: `Full log saved to: <path>`.

---

## How long it takes (rough)

| Phase | What it does | Typical time (single GPU, e.g. A100) |
|-------|----------------|--------------------------------------|
| **Phase 1** | Find capacity M (bracket + binary search, 3 steps per try) | ~5–15 min |
| **Phase 2** | Sweep 5 effective batch sizes (256→4096), 200 steps each | ~20–50 min |
| **Total** | | **~30–70 min** |

- With `REPEAT_EACH_E=2` or real data loading (wikitext), add time.
- With `RUN_CAPACITY_AGAIN=1`, add ~1–2 min at the end.
- CPU-only or smaller GPUs will be slower; large GPUs may be at the lower end of the range.

---

## Output you care about

At the end of the log you’ll see something like:

```
RECOMMENDED FOR YOUR RUN (best tok/s within 0.5% of best loss)
  Effective batch size (samples per optimizer step):  E  (exact)
  Micro batch (max per step, capacity M=...):         ...
  Grad accumulation steps:                            ...
  DeepSpeed config: train_micro_batch_size_per_gpu=..., gradient_accumulation_steps=...
```

Copy `train_micro_batch_size_per_gpu` and `gradient_accumulation_steps` into your DeepSpeed JSON. Then do a short real training run with the top 2 configs to confirm.
