import torch
import time
import itertools
import json
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

# Free the huge scores tensor
del scores, topk
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════
# All knob values to search
# ═══════════════════════════════════════════════════════════════════════
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

# Generate ALL permutations — full exhaustive grid
keys, values = zip(*combinations.items())
all_setups = [dict(zip(keys, v)) for v in itertools.product(*values)]

total = len(all_setups)
print(f"═══════════════════════════════════════════════════════════════════")
print(f"  EXHAUSTIVE GRID SEARCH: {total} combinations")
print(f"  Shape: B={B}, H={H}, T={T}, D={D}, k={k_sel}")
print(f"═══════════════════════════════════════════════════════════════════")

# ═══════════════════════════════════════════════════════════════════════
# Benchmark function
# ═══════════════════════════════════════════════════════════════════════
def benchmark(setup, num_iters=4):
    for kp, val in setup.items():
        KNOBS[kp] = val

    # Warmup
    try:
        out = triton_sparse_attention(q, k, v, indices, mask, scale)
        out.backward(do, retain_graph=True)
        q.grad = None
        k.grad = None
        v.grad = None
        torch.cuda.synchronize()
    except Exception as e:
        return float('inf'), str(e)

    # Timed iterations
    try:
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
        return ms_time, None
    except Exception as e:
        return float('inf'), str(e)

# ═══════════════════════════════════════════════════════════════════════
# Run full grid search
# ═══════════════════════════════════════════════════════════════════════
results = []
errors = 0
wall_start = time.time()

for idx, setup in enumerate(all_setups):
    t_ms, err = benchmark(setup)

    if err:
        errors += 1
        status = "ERR"
    else:
        status = f"{t_ms:7.2f} ms"

    results.append((t_ms, setup))

    # Progress every 50 combos
    if (idx + 1) % 50 == 0 or idx == 0 or (idx + 1) == total:
        elapsed = time.time() - wall_start
        rate = (idx + 1) / elapsed
        eta_s = (total - idx - 1) / rate if rate > 0 else 0
        eta_m = eta_s / 60

        # Find current best
        best_t, best_s = min(results, key=lambda x: x[0])
        best_desc = ", ".join(f"{kp}={v}" for kp, v in best_s.items())

        print(f"[{idx+1:5d}/{total}] {status}  |  "
              f"Best so far: {best_t:.2f} ms  |  "
              f"ETA: {eta_m:.1f} min  |  Errors: {errors}")

# ═══════════════════════════════════════════════════════════════════════
# Sort and display top 25 results
# ═══════════════════════════════════════════════════════════════════════
results.sort(key=lambda x: x[0])

print("\n" + "═" * 80)
print("  TOP 25 BEST CONFIGURATIONS (out of {})".format(total))
print("═" * 80)

for rank, (t_ms, setup) in enumerate(results[:25], 1):
    desc = ", ".join(f"{kp}={v}" for kp, v in setup.items())
    print(f"  #{rank:2d}  {t_ms:7.2f} ms  |  {desc}")

# Show worst for reference
print("\n" + "─" * 80)
print("  WORST 5 (for reference):")
print("─" * 80)
for rank, (t_ms, setup) in enumerate(results[-5:], total - 4):
    if t_ms == float('inf'):
        desc = "FAILED"
    else:
        desc = ", ".join(f"{kp}={v}" for kp, v in setup.items())
    print(f"  #{rank}  {t_ms:7.2f} ms  |  {desc}")

# ═══════════════════════════════════════════════════════════════════════
# Save all results to JSON for later analysis
# ═══════════════════════════════════════════════════════════════════════
best_time, best_setup = results[0]
print("\n" + "═" * 80)
print(f"  🏆 BEST: {best_time:.2f} ms")
for kp, v in best_setup.items():
    print(f"     {kp}: {v}")
print("═" * 80)

# Save to file
output = {
    "best_time_ms": best_time,
    "best_setup": best_setup,
    "total_tested": total,
    "total_errors": errors,
    "all_results": [(t, s) for t, s in results if t != float('inf')]
}
with open("tune_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nFull results saved to tune_results.json")

# Print the KNOBS dict you can paste directly into triton_sparse_attn.py
print("\n# ── Paste this into triton_sparse_attn.py ──")
print("KNOBS = {")
for kp, v in best_setup.items():
    print(f'    "{kp}": {v},')
print("}")
