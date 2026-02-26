# 70B Model Training Setup (Modified to 10 Experts)

## Overview

The 70B recurrence model has been adapted for Mac testing with:
- **10 experts** (user-modified from 254)
- **Kronecker embeddings** (131k tokenizer)
- **20 layers** (15 DeltaNet + 5 GSA)
- **4096 hidden size**
- **MoE with null expert routing**

## Key Differences from 1B Model

| Feature | 1B Model | 70B Model (10 experts) |
|---------|----------|------------------------|
| **Experts** | 0 (Dense) | 10 (Sparse MoE) |
| **Layers** | 8 | 20 |
| **FFN Type** | Pure dense FFN | MoE + Shared Expert |
| **Aux Loss** | 0.0 (no routing) | > 0.0 (routing losses) |
| **Active Params** | 100% | ~50% (via top-k routing) |
| **Routing** | None | Dynamic top-k (0-10) |

## Fixes Applied

✅ **Fix #42**: RoPE dtype casting (already in recurrence_model_70b.py)
✅ **Fix #43**: RMSNorm fp32 statistics (already in recurrence_model_70b.py)

**NOT NEEDED**:
- Dense model initialization fixes (lines 1187-1211 in 1B model) - Only needed for num_experts=0
- Dense gradient flow fixes (x.sum() * 0.0) - Only needed for num_experts=0

The 70B model's aux_loss comes from MoEGate routing (L_bal, L_z, L_null), which always has grad_fn.

## Configuration

### Model Architecture
```python
vocab_size = 131072
hidden_size = 4096
num_layers = 20

# Hybrid: 75% DeltaNet + 25% GSA
num_deltanet_layers = 15
num_gsa_layers = 5

# MoE (user-modified)
num_real_experts = 10       # User changed from 254
num_null_experts = 10       # Should match for ρ=0.5
total_expert_slots = 20     # real + null
top_k = 10                  # Dynamic routing
data_sparsity = 0.5         # Target 50% null selections
```

### MoE Routing Losses

Unlike the 1B dense model, the 70B model has non-zero auxiliary losses:

```python
aux_loss = 2e-2 * L_bal + 1e-3 * L_z + 1e-2 * L_null

Where:
- L_bal: Load balancing (encourages uniform expert usage)
- L_z: Z-loss (prevents logit growth)
- L_null: Null-rate regularizer (targets ρ=0.5)
```

**Expected behavior:**
- `aux_loss` should be in range [0.01, 0.1]
- NOT 0.0 like the 1B model
- If aux_loss > 0.5, routing may be unstable

## Running the Training Script

```bash
cd /Users/rohanshravan/TSAI/ValidationCheck/endGame
python train_recurrence_70b.py
```

## Expected Output

### Step 0 (First Step)
```
DEBUG - Model output:
  logits_mtp is None: False
  logits_mtp contains NaN: False
  logits_mtp mean: -0.0006
  logits_mtp std: 1.2798

DEBUG - Step 0:
  x_input shape: torch.Size([2, 62])
  logits_ntp shape: torch.Size([2, 62, 131072])
  logits_mtp shape: torch.Size([2, 62, 131072])
  NTP perplexity: ~300000
  MTP perplexity: ~300000
  aux_loss: 0.025634  ← NON-ZERO for MoE routing
  NTP accuracy: 0.0000
  MTP accuracy: 0.0000
```

### Training Progress
```
step   0 | loss_ntp: 12.60 | loss_mtp: 12.65 | aux: 0.0256 | dt: 8000ms | tok/sec: 31.0
step   1 | loss_ntp: 12.45 | loss_mtp: 12.23 | aux: 0.0234 | dt: 4500ms | tok/sec: 55.1
step   2 | loss_ntp: 12.24 | loss_mtp: 11.54 | aux: 0.0198 | dt: 4200ms | tok/sec: 59.0
...
```

**Key differences from 1B:**
- ✅ `aux > 0.0` (MoE routing losses)
- ⚠️  Slower throughput (20 layers vs 8, MoE overhead)
- ⚠️  Higher memory usage (10 expert weight tensors)

## Performance Expectations (Mac M1/M2)

| Metric | 1B Dense | 70B (10 experts) |
|--------|----------|------------------|
| **First step** | ~5-10 sec | ~10-20 sec |
| **Subsequent steps** | ~100-200ms | ~200-500ms |
| **Throughput** | 10k-20k tok/sec | 3k-8k tok/sec |
| **Memory** | ~8-12GB | ~15-20GB |
| **aux_loss** | 0.0000 | 0.01-0.10 |

**Note:** These are rough estimates for smoke testing. Production training requires CUDA GPUs.

## Monitoring MoE Routing

### Healthy Routing Patterns

1. **aux_loss magnitude**: 0.01 - 0.10
   - Too low (< 0.005): Router may be collapsing
   - Too high (> 0.5): Router unstable or misconfigured

2. **Null selection rate**: ~50% (with ρ=0.5)
   - Track during training to verify data sparsity is maintained

3. **Expert utilization**: Should be roughly uniform
   - Each of 10 experts should get ~10% of tokens (on average)
   - Check for expert collapse (some experts never selected)

### Debugging Commands

```python
# Get routing statistics from a layer
layer = model.layers[0]
moe_ffn = layer.mlp_block.sublayer.moe
last_indices = moe_ffn.last_indices  # [B, T, top_k]

# Check expert distribution
expert_counts = torch.bincount(last_indices.flatten())
print(f"Expert usage: {expert_counts}")
```

## Known Limitations for Mac Testing

⚠️  **This is NOT production training!** It's a smoke test to verify:
- Model initializes correctly
- Forward/backward pass works
- MoE routing functions
- No gradient flow issues
- No crashes

**For production:**
- Use CUDA GPUs (10-50× faster)
- Larger batch sizes (16-32)
- Longer sequences (2k-8k)
- Full 254 experts (not 10)
- Multi-GPU training

## Troubleshooting

### "aux_loss is 0.0000"
**Problem:** MoE routing not active (model behaving like dense)
**Check:**
- Verify num_experts = 10 in ModelConfig
- Ensure MoEGate is being called
- Check if all tokens selecting null experts

### "aux_loss > 1.0"
**Problem:** Routing losses too high, dominating gradients
**Fix:**
- Reduce aux loss coefficients in MoEGate (line 1252)
- Check if router logits exploding (Z-loss should prevent this)

### "OOM (Out of Memory)"
**Problem:** 70B model too large for Mac
**Fix:**
- Reduce batch_size to 1
- Reduce seq_len to 32
- Use CPU instead of MPS (slower but more memory)
- Consider gradient checkpointing

### "Slow throughput (< 10 tok/sec)"
**Expected:** 70B model is MUCH slower than 1B on Mac
- 20 layers vs 8 layers = 2.5× more compute
- MoE routing overhead
- 10 expert weight tensors

**This is normal for Mac testing!**

## Next Steps After Successful Run

1. **Verify routing statistics**
   - Check aux_loss is in expected range
   - Monitor expert utilization
   - Track null selection rate

2. **Compare with 1B model**
   - Loss convergence speed
   - Final loss values
   - Memory/throughput tradeoffs

3. **Scale to GPU**
   - Restore full 254 experts
   - Increase batch size to 16-32
   - Increase seq_len to 2k-8k
   - Monitor MoE routing quality at scale

4. **Production monitoring**
   - Track routing entropy
   - Expert load balancing
   - Null expert selection rates
   - Loss component breakdown

---

**Ready to test!** 🚀

Run `python train_recurrence_70b.py` to verify everything works with your 10-expert configuration.
