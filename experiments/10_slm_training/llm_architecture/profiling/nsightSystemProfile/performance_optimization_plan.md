# Performance Optimization Plan for train_nsys.py

## Profiling Analysis Summary

### Current Performance Bottlenecks (from nsys report)

| Issue | Time/Impact | Severity |
|-------|------------|----------|
| `cudaStreamSynchronize` calls | 60.7% of CUDA API time (1.99s) | **CRITICAL** |
| Device-to-Device memcpy | 90.7% of mem ops (8.6ms, 3088 MB) | **HIGH** |
| Deprecated `torch.cuda.amp.autocast` | FutureWarning | **MEDIUM** |
| Loss accumulation bug | Incorrect calculation | **HIGH** |
| Synchronous `.to(device)` calls | Hidden sync points | **MEDIUM** |
| `zero_grad()` with memset | 565 memset calls (636μs) | **LOW** |

---

## Proposed Optimizations

### 1. **Fix Deprecated autocast API** (MEDIUM priority)
**Before:**
```python
from torch.cuda.amp import autocast, GradScaler
with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
```

**After:**
```python
from torch.amp import autocast, GradScaler
with autocast(device_type='cuda', enabled=self.use_amp, dtype=self.amp_dtype):
```

**Why:** 
- Removes FutureWarning
- Uses device-agnostic API (PyTorch 2.0+)
- Better forward compatibility

---

### 2. **Reduce Synchronization Points** (CRITICAL priority)

**Before:**
```python
accumulation_loss += loss.item()  # Calls .item() every micro-batch → sync!
```

**After:**
```python
# Accumulate on GPU, sync only at logging time
accumulation_loss += loss.detach()  # Keep on GPU
# Later, when logging:
loss_value = accumulation_loss.item() / accumulation_steps
```

**Why:**
- `.item()` forces CPU-GPU synchronization
- Called once per micro-batch → many sync points
- 60.7% of CUDA API time is synchronization
- Moving sync to logging interval reduces overhead by ~4x

---

### 3. **Non-blocking Data Transfers** (MEDIUM priority)

**Before:**
```python
input_ids = batch['input_ids'].to(self.device)
labels = batch['labels'].to(self.device)
```

**After:**
```python
input_ids = batch['input_ids'].to(self.device, non_blocking=True)
labels = batch['labels'].to(self.device, non_blocking=True)
```

**Why:**
- Asynchronous H2D transfers overlap with compute
- Requires `pin_memory=True` in DataLoader (already set)
- Reduces idle time waiting for transfers

---

### 4. **Optimize zero_grad() with set_to_none** (LOW priority)

**Before:**
```python
self.optimizer.zero_grad()  # Uses memset (565 calls, 636μs)
```

**After:**
```python
self.optimizer.zero_grad(set_to_none=True)  # Avoids memset
```

**Why:**
- Avoids 565 CUDA memset operations
- Slightly faster (saves ~600μs per epoch)
- More memory efficient (releases tensors instead of zeroing)

---

### 5. **Fix Loss Accumulation Bug** (HIGH priority)

**Before:**
```python
loss = outputs.loss / self.config.gradient_accumulation_steps
accumulation_loss += loss.item()
# ... later:
loss=accumulation_loss * self.config.gradient_accumulation_steps  # WRONG!
```

**Issue:** Divides by accumulation steps, then multiplies back - cancels out but confusing

**After:**
```python
loss = outputs.loss / self.config.gradient_accumulation_steps
accumulation_loss += loss.detach()
# ... later:
loss_value = (accumulation_loss / accumulation_steps).item()
```

**Why:**
- Correct averaging over micro-batches
- Clear and maintainable
- Reduces sync points (detach vs item)

---

### 6. **Enable Gradient Checkpointing** (OPTIONAL)

**Current:** Removed in user's version
**Recommendation:** Re-enable for memory efficiency

```python
self.model.gradient_checkpointing_enable()
```

**Trade-off:**
- Memory: Reduces peak by ~30-40%
- Speed: Adds ~20% overhead (recompute on backward)
- **Recommendation:** Enable for larger batch sizes

---

### 7. **Compile Model with torch.compile()** (ADVANCED - Optional)

```python
# In Trainer.__init__
if torch.__version__ >= '2.0.0':
    self.model = torch.compile(self.model, mode='reduce-overhead')
```

**Why:**
- Can improve throughput by 20-40%
- Reduces Python overhead
- Better kernel fusion
- **Caution:** First iteration will be slow (compilation time)

---

## Expected Performance Improvements

| Optimization | Expected Speedup | Confidence |
|--------------|-----------------|------------|
| Fix synchronization (detach vs item) | **15-25%** | High |
| Non-blocking transfers | **5-10%** | Medium |
| Fix deprecated autocast | **0-2%** | High (removes warning) |
| set_to_none for zero_grad | **0-1%** | Low |
| torch.compile (if used) | **20-40%** | Medium |
| **Combined (without compile)** | **~20-35%** | **High** |
| **Combined (with compile)** | **~40-80%** | **Medium** |

---

## Implementation Priority

1. **CRITICAL** - Fix synchronization (accumulation_loss detach)  
2. **HIGH** - Fix loss accumulation bug  
3. **MEDIUM** - Fix deprecated autocast API  
4. **MEDIUM** - Non-blocking data transfers  
5. **LOW** - zero_grad(set_to_none=True)  
6. **OPTIONAL** - torch.compile() (if PyTorch ≥ 2.0)

---

## Verification Steps

After implementing optimizations:

1. **Re-run nsys profiling:**
   ```bash
   nsys profile --capture-range=cudaProfilerApi \
       python train_nsys.py --profile-steps 10-20 --max-steps 100
   ```

2. **Check metrics:**
   - `cudaStreamSynchronize` time should **drop significantly** (from 60.7%)
   - Tokens/sec should **increase by ~20-35%**
   - No FutureWarning messages

3. **Validate correctness:**
   - Loss values should be similar to before
   - Training should converge normally
   - No NaN/Inf values

---

## Next Steps

Implement these optimizations in order of priority, test after each change, and measure the cumulative performance improvement.
