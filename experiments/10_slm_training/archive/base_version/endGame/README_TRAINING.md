# Training Recurrence Model 1B with Kronecker Embeddings

## Quick Start

```bash
cd /Users/rohanshravan/TSAI/ValidationCheck/endGame
python train_recurrence_1b.py
```

## What This Does

1. **Loads tokenizer.json** (131k tokens) from the endGame directory
2. **Creates Kronecker embeddings** for all 131k tokens using byte-level encoding (256×32=8192 dims)
3. **Initializes Model1B** (1.513B parameters) with:
   - 8 layers (6 DeltaNet + 2 GSA)
   - 4096 hidden size
   - Multi-Token Prediction (NTP + MTP)
   - Memory stream recurrence for infinite context
4. **Loads synth_local_en dataset** from parent directory
5. **Runs 100 training steps** as a smoke test

## Expected Behavior

On Apple Silicon Mac (MPS):
- First step will be slow (~5-10 seconds) as Metal shaders compile
- Subsequent steps should be ~100-200ms per step at batch_size=4, seq_len=512
- Memory usage: ~8-12GB
- Throughput: ~2,000-4,000 tok/sec (not optimized for Mac, just for testing)

## Files Structure

```
endGame/
├── tokenizer.json              # 131k token vocabulary
├── recurrence_model_1b.py      # 1B recurrence model (from Recurrance Code/)
├── reversible_ops_midpoint.py  # Reversible integration (from Recurrance Code/)
├── data_utils.py               # Dataset utilities (copied from base_null_reversal/)
├── train_recurrence_1b.py      # Training script (NEW)
└── README_TRAINING.md          # This file
```

## What Gets Tested

✅ Tokenizer loading from JSON
✅ Kronecker embeddings creation (131k tokens → 8192-dim byte encodings)
✅ Model initialization (1.513B params)
✅ Forward pass with multi-token prediction
✅ Backward pass and gradient flow
✅ Memory stream recurrence setup (not used in this test)
✅ Dataset loading from synth_local_en

## Next Steps

If the test runs successfully:

1. **Extend to full training:** Increase `num_steps` in `simple_training_loop()`
2. **Add checkpointing:** Save model state periodically
3. **Add generation:** Test model outputs during training
4. **Monitor metrics:** Track loss curves, lambda_e, GSA stats, etc.
5. **Integrate with existing pipeline:** Adapt main.py/training.py to use recurrence_model_1b

## Troubleshooting

### "Dataset not found"
Make sure `synth_local_en/` exists in the parent directory:
```bash
ls -la /Users/rohanshravan/TSAI/ValidationCheck/synth_local_en
```

### "ImportError: No module named 'recurrence_model_1b'"
Make sure you're running from the endGame directory:
```bash
cd /Users/rohanshravan/TSAI/ValidationCheck/endGame
```

### "MPS out of memory"
Reduce batch_size in the script (currently 4):
```python
dataset = SYNTHStream(..., batch_size=2, ...)
train_loader = DataLoader(dataset, batch_size=2, ...)
```

### "Slow first step"
This is normal - Metal shaders compile on first run. Subsequent steps are much faster.

## Model Configuration

The Model1B uses:
- **Vocab:** 131,072 tokens (from tokenizer.json)
- **Embedding:** Kronecker product (byte-level: 256×32=8192 dims)
- **Hidden:** 4096 dimensions
- **Layers:** 8 (6 DeltaNet + 2 GSA)
- **Parameters:** 1.513B total, 1.513B active (no MoE, fully dense)
- **Context:** Targets 256k (currently testing at 512)
- **MTP:** Predicts t+1 (NTP) and t+2 (MTP) simultaneously

## What's Different from Base Model

1. **No MoE:** This model is fully dense (no expert routing)
2. **Kronecker embeddings:** Byte-level encoding instead of learned embeddings
3. **Memory recurrence:** Can process infinite-length documents via chunking
4. **Reversible integration:** Memory-efficient backprop through 8 layers
5. **Multi-token prediction:** Learns to predict 2 tokens ahead

## Performance Notes

This is **NOT** optimized for Mac training. It's a smoke test to verify:
- Model loads correctly
- Forward/backward pass works
- No crashes or errors
- Gradients flow properly

For actual training, you'll want to use:
- CUDA GPUs (much faster)
- Larger batch sizes (16-32)
- Longer sequences (2k-8k)
- Full training loop with checkpoints

## Credits

- Model architecture: From "Recurrance Code" directory (Fixes #1-43 applied)
- Tokenizer: 131k byte-pair encoding vocabulary
- Dataset utilities: From base_null_reversal directory
- Training script: New minimal test harness
