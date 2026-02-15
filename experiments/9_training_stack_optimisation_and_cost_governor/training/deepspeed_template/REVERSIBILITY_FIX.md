# Reversibility Fix - DeepSpeed Autocast Bypass

## Problem

Despite implementing reversibility correctly via `MidpointFunction` and `ReversibleMidpointStack`, the model was experiencing OOM errors on 8×A100 (40GB) with:
- Sequence length: 4096
- Global batch size: 36
- Gradient accumulation: 2

**Root Cause**: PyTorch was allocating ~34 GB per GPU, indicating it was caching the entire 25GB forward-pass computational graph, completely defeating the custom reversible logic.

## Why Reversibility Was Bypassed

1. **DeepSpeed's BF16 Autocast**: The DeepSpeed config has `"bf16": {"enabled": true}`, which applies `torch.autocast` to every forward pass
2. **Graph Caching**: PyTorch's autocast natively records and caches tensor operations in the C++ backend for backward passes
3. **Override**: Even though the custom `torch.autograd.Function` is designed to discard activations, DeepSpeed's wrapper overrides it and forces PyTorch to cache the entire computational graph

## The Fix

Applied in `/src/train.py` for both `train_epoch()` and `evaluate()` functions:

### Before (Broken)
```python
logits_ntp, logits_mtp, aux_loss = model_engine(
    x_input, 
    next_token_ids=y_ntp,
    ...
)
```

### After (Fixed)
```python
# CRITICAL FIX: Bypass DeepSpeed's autocast wrapper to enable true reversibility
# - torch.autocast(enabled=False) prevents PyTorch from caching the 25GB forward graph
# - model_engine.module bypasses DeepSpeedEngine wrapper (safe for ZeRO Stage 2)
with torch.autocast(device_type="cuda", enabled=False):
    logits_ntp, logits_mtp, aux_loss = model_engine.module(
        x_input, 
        next_token_ids=y_ntp,
        ...
    )

# Compute losses sequentially and free BF16 tensors instantly
vocab_size = logits_ntp.size(-1)

loss_ntp = torch.nn.functional.cross_entropy(logits_ntp.float().view(-1, vocab_size), y_ntp.view(-1))
del logits_ntp  # Free BF16 tensor immediately

loss_mtp = torch.nn.functional.cross_entropy(logits_mtp.float().view(-1, vocab_size), y_mtp.view(-1))
del logits_mtp  # Free BF16 tensor immediately
```

## Key Changes

1. **Disable Autocast**: `torch.autocast(device_type="cuda", enabled=False)` prevents PyTorch from caching the forward graph
2. **Bypass DeepSpeed Wrapper**: `model_engine.module` instead of `model_engine` calls the raw model directly
3. **Immediate Cleanup**: Explicitly `del` logits after loss computation to free BF16 tensors instantly

## Why This Works

- **Graph Drops**: By disabling autocast and calling `model_engine.module`, PyTorch will no longer cache the 25GB forward pass
- **Memory Recovers**: Expected memory usage drops from ~34GB to ~12GB (3GB weights + 1.5GB optimizer + 8GB FP32 logits)
- **4GB Allocation Succeeds**: With 25GB freed, PyTorch has plenty of room for the 4GB logits backward pass without OOM

## Safety Note

Bypassing the DeepSpeed wrapper with `model_engine.module` is **safe for ZeRO Stage 2** because:
- Parameters are fully resident on the GPU (not partitioned)
- ZeRO-2 only partitions optimizer states and gradients
- **NOT safe for ZeRO Stage 3** (where parameters are partitioned)

## Expected Results

After this fix:
- Memory usage per GPU: ~12-15GB (down from ~34GB)
- No more OOM errors at seq_len=4096, batch_size=36
- Reversibility working as designed
- Training can proceed with full reversible integration benefits

## Files Modified

- `/src/train.py`: 
  - `train_epoch()` function (lines 86-126)
  - `evaluate()` function (lines 329-354)

## Testing

To verify the fix is working:
1. Monitor GPU memory usage: should drop from ~34GB to ~12-15GB
2. Training should complete without OOM errors
3. Check loss convergence is stable (no NaN/Inf)

## Credits

Analysis and solution provided by Gemini AI, identifying the DeepSpeed autocast trap as the root cause of reversibility bypass.
