# test_memory_safe.py
import gc

import torch
from src.model import get_qwen2_moe_model


def print_memory(stage):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(
        f"{stage:30s} | Allocated: {allocated:5.2f} GB | Reserved: {reserved:5.2f} GB"
    )


# Clear memory
torch.cuda.empty_cache()
gc.collect()

print("\n" + "=" * 70)
print("Memory Test for Ultra-Safe MoE on T4 (16GB)")
print("=" * 70 + "\n")

print_memory("Initial")

# Create model using the MoE model from src/model.py
print("\nCreating model...")
model = get_qwen2_moe_model(device="cuda", print_info=True)
print_memory("After model load (with gradient checkpointing)")

# Create input
print("\nTesting forward pass...")
x = torch.randint(0, 151936, (1, 1024)).cuda()
print_memory("After input creation")

# Forward pass
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    out = model(x, labels=x)
print_memory("After forward pass")

# Backward pass
print("\nTesting backward pass...")
out.loss.backward()
torch.cuda.synchronize()
print_memory("After backward pass")

# Peak memory
peak = torch.cuda.max_memory_allocated() / 1e9
print("\n" + "=" * 70)
print(f"PEAK GPU MEMORY: {peak:.2f} GB")
print("AVAILABLE ON T4: 16.00 GB")
print(f"HEADROOM: {16.0 - peak:.2f} GB")

if peak < 12.0:
    print("✅ VERY SAFE - Huge headroom!")
elif peak < 14.0:
    print("✅ SAFE - Good headroom")
elif peak < 15.0:
    print("⚠️  TIGHT - May work but risky")
else:
    print("❌ UNSAFE - Will likely OOM")

print("=" * 70 + "\n")
