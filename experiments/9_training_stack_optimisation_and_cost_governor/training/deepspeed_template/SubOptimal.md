**Dense Pipeline Review (with fixes)**
Scope clarified: this repository is currently a **Dense** pipeline (`ModelConfig` has zero active MoE experts), so MoE references should be removed from this repo’s config/docs.

**Current status vs your objectives**
1. Kronecker embeddings + optimized data: `Partial`  
Kronecker is implemented well; data path is not production-optimized for distributed/offline execution.
2. Offline CPU loading + no GPU starvation: `Not met`  
No offline pretokenized dataset path; no distributed sampler; H2D copies are blocking.
3. Triton kernels (Dense context): `Partial`  
GSA has fused path; DeltaNet can silently fall back to Python loop; no fail-fast if kernels are missing.
4. Proper logging of everything: `Partial`  
Console/NVML logs exist; no structured experiment logging/alerting pipeline.

**Suggested fixes (direct instructions for student)**
1. Add `DistributedSampler` for train/val/test dataloaders and call `set_epoch(epoch)` each epoch.
2. Add offline dataset support (`load_from_disk`) with config key like `data.tokenized_dataset_path`.
3. Keep online tokenization as fallback only; default to offline tokenized shards for AWS runs.
4. Remove per-step `torch.cuda.empty_cache()` from training/eval hot path.
5. Use non-blocking device copies: `.to(model_engine.device, non_blocking=True)` with pinned memory.
6. Add precision guard at startup: validate DeepSpeed bf16/fp16 config against model dtype; hard-fail on mismatch.
7. Add kernel fail-fast mode (`training.require_fused_kernels: true`) and abort if required fused kernels unavailable.
8. Remove Dense-pipeline MoE references from default configs/docs (`zero-2-moe*.json`, `model_type`, MoE wording).
9. Add structured metrics output (JSONL/W&B/TensorBoard): loss components, toks/s, grad norm, GPU mem/util, data wait time, NaN/inf counters.
10. Make a dense preflight test gate mandatory before expensive jobs.

**Test file added**
I added a production-readiness gate test file here:  
`/Users/rohanshravan/TSAI/ERAV4/LLM/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/test/test_dense_pipeline_production_readiness.py`

What it checks:
- Dense config has no active MoE experts.
- DistributedSampler + `set_epoch`.
- Offline dataset loading support and config key.
- No per-step `empty_cache`.
- Non-blocking H2D copies.
- Precision-policy validation.
- Kernel fail-fast flag/checks.
- Structured logging presence.
- Dense default config not using MoE profile.
- Dense configs not exposing stale `model_type` switches.

**Current run result**
I ran only this new test file.  
Result: `12 failed, 2 passed` (expected at this stage; these failures are the implementation backlog).

Run command:
```bash
~/.pyenv/versions/3.11.5/bin/python -m pytest -q /Users/rohanshravan/TSAI/ERAV4/LLM/experiments/9_training_stack_optimisation_and_cost_governor/training/deepspeed_template/test/test_dense_pipeline_production_readiness.py
```

If you want, I can now implement these fixes one by one and make this test file go green.