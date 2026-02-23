import torch
import itertools
from triton_sparse_attn_tunable import triton_sparse_attention, KNOBS

# Hyperparameters matching the actual Test_14 environment
B = 4
H = 4
T = 4096
D = 128
k_sel = 64
scale = D**-0.5

# Create synthetic data
torch.manual_seed(42)
q = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)

# Generate synthetic top-k indices and mask
scores = torch.randn(B, H, T, T, device='cuda')
topk = torch.topk(scores, k_sel, dim=-1)
indices = topk.indices.to(torch.int64)
mask = torch.ones(B, H, T, k_sel, device='cuda', dtype=torch.float32)

do = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16)

# Lock forward + dQ to the Golden Recipe from the broad sweep
KNOBS["fwd_BLOCK_Q"] = 4
KNOBS["fwd_num_warps"] = 2
KNOBS["fwd_num_stages"] = 2
KNOBS["bwd_dq_BLOCK_K"] = 64
KNOBS["bwd_dq_num_warps"] = 2
KNOBS["bwd_dq_num_stages"] = 2

# The focused dK/dV grid: 2 x 2 x 3 = 12 runs
dkdv_grid = {
    "bwd_dkdv_BLOCK_K": [16, 32],
    "bwd_dkdv_num_warps": [4, 8],
    "bwd_dkdv_num_stages": [1, 2, 3],
}

keys, values = zip(*dkdv_grid.items())
all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Focused dK/dV sweep: {len(all_combos)} combinations")
print(f"Shape: B={B}, H={H}, T={T}, D={D}, k={k_sel}")
print(f"Locked: fwd_BQ=4, fwd_warps=2 | dQ_BK=64, dQ_warps=2")
print("-" * 70)

def benchmark(setup, num_iters=10):
    for k_param, val in setup.items():
        KNOBS[k_param] = val
        
    # Warmup
    try:
        out = triton_sparse_attention(q, k, v, indices, mask, scale)
        out.backward(do, retain_graph=True)
        q.grad = None
        k.grad = None
        v.grad = None
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  FAILED: {e}")
        return float('inf')
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iters):
        out = triton_sparse_attention(q, k, v, indices, mask, scale)
        out.backward(do, retain_graph=True)
        q.grad = None
        k.grad = None
        v.grad = None
    end_event.record()
    torch.cuda.synchronize()
    
    ms_time = start_event.elapsed_time(end_event) / num_iters
    return ms_time

results = []
for idx, combo in enumerate(all_combos):
    desc = ", ".join([f"{k}={v}" for k, v in combo.items()])
    time_ms = benchmark(combo)
    results.append((time_ms, desc, combo))
    print(f"[{idx+1}/{len(all_combos)}] {desc}: {time_ms:.2f} ms")

print("\n" + "=" * 70)
print("RESULTS RANKED (fastest first)")
print("=" * 70)
results.sort(key=lambda x: x[0])
best_time = results[0][0]
for rank, (t, desc, combo) in enumerate(results, 1):
    delta = ((t - best_time) / best_time) * 100
    marker = " ⭐ WINNER" if rank == 1 else ""
    print(f"  #{rank:2d}  {t:7.2f} ms  (+{delta:5.1f}%)  {desc}{marker}")

print(f"\n>>> Best config: {results[0][2]}")
print(f">>> Best time:   {results[0][0]:.2f} ms")
