# Test_9_extend_test8_to_3000steps_save_state

## Objective
Continuation run: resume Test 8 checkpoint and extend to global step ~3000 while saving state.

## Run
```bash
cd "/Users/rohanshravan/Downloads/LLM-code-20260219-1351_rohan_patch_v3/experiments/tests/Test_9_extend_test8_to_3000steps_save_state"
./run.sh
```

Optional:
```bash
NUM_GPUS=4 ./run.sh
```

Force regenerate init model:
```bash
FORCE_REWRITE_INIT=1 ./run.sh
```

## Fixed controls
- model_variant: `reversible`
- embedding_type: `kronecker`
- max_train_steps: `2999` (chosen to reach global step ~3000 when resuming from step 1000)
- log_interval: `1` (every step)
- seed: `42`
- dataset: `wikitext-103-raw-v1`
- seq length: `512`
- global batch size: `32`

## Outputs
- Init model: `results/init/model_init.pt`
- Init metadata: `results/init/model_init_meta.json`
- Train log: `results/run/train.log`
- Metrics: `results/run/metrics.jsonl`
- Notes: `notes/reference_run.md`

## Self-sufficient
This folder includes its own runnable snapshot in `code/` with the reversible model only.

Kronecker assertion: init script verifies `model.use_kronecker=True` and `model.token_embed is None`.


Reversibility assertion: init script verifies reversible stack is active.


Triton GSA assertion: init script verifies `HAS_TRITON`, `fused_indexer_topk`, and `triton_sparse_attention` availability.


DeltaNet fused assertion: init script verifies `HAS_FLA` and `fla_gated_delta_rule` availability.


Additional fused assertion: init script verifies fused CE module and fused SwiGLU usage are present; training uses fused CE path (`use_fused_ce: true`).


## Continuation Inputs
- default source checkpoint dir:
  `/Users/rohanshravan/Downloads/LLM-code-20260219-1351_rohan_patch_v3/experiments/tests/Test_8_additional_fused_kernels_1000steps/results/run/checkpoints`
- default resume tag: `epoch0_step1000`

Override if needed:
```bash
SOURCE_TEST8_CKPT_DIR="/path/to/test8/checkpoints" RESUME_TAG="epoch0_step1000" ./run.sh
```

## Save-state outputs
- resumed training checkpoints: `results/run/checkpoints/*`
- final checkpoint tag expected: `results/run/checkpoints/final`
