# Performance Optimization Plan — Model1B Training

## Profiling Summary

Profiled with `batch_size=8, seq_length=2048` on NVIDIA RTX PRO 6000 Blackwell (102GB).

### Top GPU Time Consumers

| Rank | Component | Self CUDA % | Time | Root Cause |
|------|-----------|-------------|------|------------|
| 1 | `aten::mul` (elementwise) | 18.14% | 2.211s | Unfused mul ops across MHC, gating, alpha/beta |
| 2 | `_sparse_attn_fwd_kernel` | 12.76% | 1.556s | GSA attention (2 layers) — already Triton-fused |
| 3 | `elementwise_kernel` variants | ~10% | ~1.2s | PyTorch fallback for unfused pow/rsqrt/add chains |
| 4 | `cutlass` GEMM kernels | ~10% | ~1.2s | matmul — expected, not a target |
| 5 | `aten::copy_` (D2D) | 7.60% | 927ms | `.contiguous()`, stream reshapes, dtype casting |
| 6 | `MidpointFunctionBackward` | 5.13% | 625ms | Reversible recompute + `torch.autograd.grad` |
| 7 | `aten::add_` / `aten::add` | ~9% | 1.1s | Accumulations across MHC, residuals |
| 8 | `Command Buffer Full` stalls | 2.68% | 327ms | Too many small kernel launches |
| 9 | `aten::sum` | 3.07% | 374ms | MHC coefficient reductions |
| 10 | `aten::div` / `aten::pow` | ~3% | ~400ms | RMSNorm PyTorch fallback (pow+mean+rsqrt) |

### Nsys Headline Stats

- **87.7% of CPU time** in `cudaLaunchKernel` → massive kernel launch overhead
- **99.4% of memcpy** is Device-to-Device (188ms, 345 ops)
- `Command Buffer Full` events → GPU stalling on enqueue backlog

---

## Optimization Targets (Priority Order)

### OPT-1: Fused RMSNorm with Autograd Support
**Impact: ~5% total CUDA time (pow+mean+rsqrt+mul chains)**

> [!IMPORTANT]
> RMSNorm Triton kernel exists but is **disabled during training** because it doesn't support autograd, breaking the reversible midpoint backward recompute.

#### Current Code ([RMSNorm.forward](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/recurrence_model_1b.py#L534-L548))
```python
if self._use_triton and x.is_cuda and not torch.is_grad_enabled():  # <-- NEVER runs in training!
    return triton_rmsnorm(x, self.weight, self.eps)
# PyTorch fallback: 4 kernel launches (pow → mean → rsqrt → mul)
x_f = x.float()
norm = x_f.pow(2).mean(dim=-1, keepdim=True)
x = x * torch.rsqrt(norm.to(x.dtype) + self.eps)
return self.weight * x
```

#### Plan
Wrap the Triton forward kernel in a `torch.autograd.Function` with a backward kernel:
- **Forward**: fused `pow² → mean → rsqrt → scale` (1 kernel, already exists)
- **Backward**: fused `dx = w * rsqrt * (dout - x_normed * mean(dout * x_normed))` (1 new kernel)
- Remove the `not torch.is_grad_enabled()` guard

#### Files
- [MODIFY] [triton_rmsnorm.py](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/kernels/triton_rmsnorm.py) — add backward kernel + autograd wrapper
- [MODIFY] [recurrence_model_1b.py](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/recurrence_model_1b.py) — remove `not torch.is_grad_enabled()` guard

---

### OPT-2: Fused SiLU×Gate Kernel
**Impact: ~3-4% total CUDA time**

#### Current Code ([FusedRMSNormSwishGate.forward](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/recurrence_model_1b.py#L701-L704))
```python
x_norm = self.norm(x)      # kernel 1-4: RMSNorm
return g * F.silu(x_norm)   # kernel 5: silu, kernel 6: mul
```
Also in [MoEFFN.forward](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/recurrence_model_1b.py#L1320):
```python
shared_h = F.silu(self.shared_gate(x)) * self.shared_up(x)  # silu + mul = 2 kernels
```

#### Plan
Create a single Triton kernel: `output = gate * silu(x)` (fuses sigmoid + mul + mul):
- Eliminates 2 kernel launches per call (called 33× per forward: once per layer attention output + once per layer FFN)
- Also fuse the `SiLU×gate` pattern in FFN

#### Files
- [NEW] [triton_fused_ops.py](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/kernels/triton_fused_ops.py) — `silu_mul_kernel`, `sigmoid_mul_kernel`
- [MODIFY] [recurrence_model_1b.py](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/recurrence_model_1b.py) — use fused ops in `FusedRMSNormSwishGate`, `MoEFFN`, `GatedDeltaNet`, `GatedSparseAttention`

---

### OPT-3: Reduce Device-to-Device Copies
**Impact: ~7.6% total CUDA time (927ms)**

#### Sources Identified
1. **Stream reshape in `Model1B.forward`**: `x_stream = torch.zeros(B, T, n_streams, D)` then `x_stream[:,:,0,:] = x` → D2D copy
2. **`o_sparse.contiguous().view(...)` in GSA**: forces a copy because stride is non-contiguous after sparse attention
3. **MHC coefficient `torch.einsum("btij,btjd->btid", H_res, x_stream)`**: may trigger copies
4. **`x_flat.to(self.phi_pre.weight.dtype)` in MHCCoeffs**: dtype cast = D2D copy

#### Plan
- Replace `torch.zeros + indexed assign` with `x.unsqueeze(2).expand()` + `F.pad` where possible
- Pre-allocate contiguous buffers for sparse attention output
- Ensure consistent dtypes to avoid casting copies

#### Files
- [MODIFY] [recurrence_model_1b.py](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/recurrence_model_1b.py) — optimize stream creation, remove unnecessary `.contiguous()` calls, fix dtype mismatches

---

### OPT-4: Fused MHC Coefficient Computation
**Impact: ~5% total CUDA time (mul+sigmoid+sum chains)**

#### Current Code ([MHCCoeffs.forward](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/recurrence_model_1b.py#L1457-L1475))
```python
pre_logits = self.alpha_pre * self.phi_pre(x_flat) + self.b_pre      # mul + add = 2 kernels
post_logits = self.alpha_post * self.phi_post(x_flat) + self.b_post  # mul + add = 2 kernels
res_logits = self.alpha_res * self.phi_res(x_flat)                   # mul = 1 kernel
res_logits = res_logits.view(B, T, n, n) + self.b_res               # add = 1 kernel
H_pre = torch.sigmoid(pre_logits)                                    # sigmoid = 1 kernel
H_post = 2.0 * torch.sigmoid(post_logits)                           # sigmoid + mul = 2 kernels
```
And [MHCSublayer.forward](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/recurrence_model_1b.py#L1488-L1508):
```python
x_in = (x_stream * H_pre.unsqueeze(-1)).sum(dim=2)  # mul + sum = 2 kernels (× 16 calls)
y_stream = y.unsqueeze(2) * H_post.unsqueeze(-1)    # mul = 1 kernel (× 16 calls)
```

#### Plan
- Fuse `alpha * linear(x) + bias → sigmoid` into a single Triton kernel (`fused_linear_sigmoid`)
- Fuse `x_stream * H.unsqueeze(-1)).sum(dim=2)` into a single kernel (weighted stream collapse)
- These are called 16× per forward pass (8 layers × 2 sublayers), so per-call savings multiply fast

#### Files
- [NEW] [triton_fused_ops.py](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/kernels/triton_fused_ops.py) — `fused_weighted_stream_collapse`, `fused_sigmoid_scale`
- [MODIFY] [recurrence_model_1b.py](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/recurrence_model_1b.py) — use in `MHCCoeffs`, `MHCSublayer`

---

### OPT-5: Fused Alpha/Beta Computation in DeltaNet
**Impact: ~2-3% total CUDA time**

#### Current Code ([GatedDeltaNet.forward](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/recurrence_model_1b.py#L896-L907))
```python
beta = torch.sigmoid(self.b_proj(x)).unsqueeze(-1)                                    # sigmoid + unsqueeze
gk = self.gk_proj(x)
A = torch.exp(self.A_log)
alpha = -A.view(1,1,H,1) * F.softplus(gk + self.dt_bias).unsqueeze(-1)              # add + softplus + mul + neg
alpha = torch.exp(alpha)                                                              # exp
```
That's ~7 elementwise kernel launches for alpha/beta.

#### Plan
Fuse into 2 kernels:
- `beta = sigmoid(linear(x)).unsqueeze(-1)` → `fused_sigmoid_unsqueeze`
- `alpha = exp(-A * softplus(gk + bias))` → single `fused_alpha_kernel`

#### Files
- [NEW] [triton_fused_ops.py](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/kernels/triton_fused_ops.py) — `fused_alpha_kernel`
- [MODIFY] [recurrence_model_1b.py](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/recurrence_model_1b.py)

---

### OPT-6: Fused Output Gate in GSA
**Impact: ~1-2% total CUDA time**

#### Current Code ([GatedSparseAttention.forward](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/recurrence_model_1b.py#L1122-L1126))
```python
g_o = torch.sigmoid(self.W_go(x))           # sigmoid = 1 kernel
return self.o_proj(o_sparse * g_o)           # mul = 1 kernel
```
And input gating (`v = v * g_v` where `g_v = torch.sigmoid(...)`) — another sigmoid + mul.

#### Plan
- Fuse `sigmoid(linear) * x` into single kernel — used in both GSA input and output gating
- Same kernel reusable for `GatedDeltaNet` output gate

#### Files
- Covered by `triton_fused_ops.py` from OPT-2/OPT-4

---

### OPT-7: Reduce Kernel Launch Overhead (torch.compile)
**Impact: ~3-5% from Command Buffer Full stalls + launch overhead**

#### Plan
Apply `torch.compile` selectively to the hottest non-Triton code paths:
- `MHCCoeffs.forward` (many small elementwise ops)
- `MHCSublayer.forward` (stream ops)
- `sinkhorn_knopp` PyTorch fallback when `torch.is_grad_enabled()`

This lets the compiler automatically fuse elementwise chains without manual Triton kernels, and is the safest optimization.

#### Files
- [MODIFY] [recurrence_model_1b.py](file:///root/LLM/experiments/10_slm_training/optimized_version/endGame/recurrence_model_1b.py) — add `@torch.compile` decorators

---

## Implementation Order

| Phase | Optimization | Est. Impact | Risk |
|-------|-------------|-------------|------|
| **1** | OPT-7: `torch.compile` on MHC/Sinkhorn | 3-5% | Low — no kernel changes |
| **2** | OPT-2: Fused SiLU×Gate kernel | 3-4% | Low — simple pointwise |
| **3** | OPT-3: D2D copy elimination | 5-7% | Low — pure Python changes |
| **4** | OPT-1: RMSNorm with autograd | 5% | Medium — backward kernel |
| **5** | OPT-5: Fused alpha/beta in DeltaNet | 2-3% | Low — simple pointwise |
| **6** | OPT-4: Fused MHC coefficients | 3-5% | Medium — stream reduction |
| **7** | OPT-6: Fused output gates | 1-2% | Low — covered by OPT-2 |

**Total estimated improvement: 22-31% reduction in CUDA time**

---

## Verification Plan

### Unit Tests (BEFORE any changes)
Create `tests/test_model_correctness.py` that:
1. Creates Model1B with a fixed seed
2. Runs forward + backward on a small batch
3. Saves reference outputs (logits, loss, grad norms)
4. After optimization, verifies outputs match within tolerance (atol=1e-3 for bf16)

### Performance Verification
After each optimization:
1. Run `./run_profiling.sh --mode pytorch` to get new profiling data
2. Compare `summary.txt` tables to see per-op time reduction
3. Verify tok/s improvement in training loop
