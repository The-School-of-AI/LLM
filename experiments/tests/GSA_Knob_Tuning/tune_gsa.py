import torch
import time
import itertools
from triton_sparse_attn_tunable import triton_sparse_attention, KNOBS

# Hyperparameters matching the actual Test_14 environment
B = 4
H = 4
T = 4096
D = 128
k_sel = 64

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

# The combinations requested by ChatGPT/User
combinations = {
    "fwd_BLOCK_Q": [1, 2, 4],
    "fwd_num_warps": [2, 4, 8],
    "fwd_num_stages": [1, 2, 3],
    
    "bwd_dq_BLOCK_K": [32, 64],
    "bwd_dq_num_warps": [2, 4, 8],
    "bwd_dq_num_stages": [1, 2, 3],
    
    "bwd_dkdv_BLOCK_K": [16, 32],
    "bwd_dkdv_num_warps": [2, 4, 8],
    "bwd_dkdv_num_stages": [1, 2, 3],
}

# Generate all permutations
keys, values = zip(*combinations.items())
all_setups = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Total combinations to test: {len(all_setups)}")
print(f"Shape: B={B}, H={H}, T={T}, D={D}, k={k_sel}")
print("-" * 80)

# Filter criteria to reduce combinatorial explosion (e.g., test one axis at a time)
# As requested, here is one script to rule them all. If testing *all* 8748 is too long,
# we test modifying one knob at a time from a baseline to find optimal settings fast.

baseline = {
    "fwd_BLOCK_Q": 2, "fwd_num_warps": 4, "fwd_num_stages": 2,
    "bwd_dq_BLOCK_K": 64, "bwd_dq_num_warps": 4, "bwd_dq_num_stages": 2,
    "bwd_dkdv_BLOCK_K": 32, "bwd_dkdv_num_warps": 4, "bwd_dkdv_num_stages": 2,
}

# Build a sweep test list that varies one knob at a time from the baseline
sweep_setups = [baseline]
for k_param in combinations:
    for val in combinations[k_param]:
        if val != baseline[k_param]:
            new_setup = baseline.copy()
            new_setup[k_param] = val
            sweep_setups.append(new_setup)

def benchmark(setup, num_iters=10):
    for k_param, val in setup.items():
        KNOBS[k_param] = val
        
    # Warmup
    try:
        out = triton_sparse_attention(q, k, v, indices, mask)
        out.backward(do, retain_graph=True)
        q.grad = None
        k.grad = None
        v.grad = None
        torch.cuda.synchronize()
    except Exception as e:
        return float('inf')  # Failed to run
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iters):
        out = triton_sparse_attention(q, k, v, indices, mask)
        out.backward(do, retain_graph=True)
        q.grad = None
        k.grad = None
        v.grad = None
    end_event.record()
    torch.cuda.synchronize()
    
    # Calculate avg time
    ms_time = start_event.elapsed_time(end_event) / num_iters
    return ms_time

print("Running single-axis sweep around baseline to find independent optimal knobs...")
print("Baseline: ", baseline)
baseline_time = benchmark(baseline)
print(f"Baseline Time: {baseline_time:.2f} ms")

results = []
for idx, setup in enumerate(sweep_setups):
    diff = {k_p: setup[k_p] for k_p in setup if setup[k_p] != baseline[k_p]}
    if not diff:
        continue # Baseline
    
    time_ms = benchmark(setup)
    desc = ", ".join([f"{k_p}={v}" for k_p, v in diff.items()])
    results.append((time_ms, desc))
    print(f"[{idx}/{len(sweep_setups)-1}] {desc}: {time_ms:.2f} ms")

print("\n--- SWEEP RESULTS RANKED ---")
results.sort(key=lambda x: x[0])
for t, desc in results:
    if t < baseline_time:
        print(f"FAST: {desc:<40} {t:.2f} ms ({(baseline_time-t)/baseline_time*100:.1f}% faster)")
    else:
        print(f"SLOW: {desc:<40} {t:.2f} ms")

print("\n(To run the full grid search of all 8000+ combinations, you can modify the script'sweep_setups' variable to use 'all_setups')")
